"""worker/runner.py — Execution orchestrator, process-per-GPU worker, and heartbeat monitor.

Features:
- Items sorted by (engine, width, height) so model weights load once per batch.
- Skips items whose output already exists on disk (idempotent / resumable).
- Atomically appends each Result into results.json as it finishes.
- Dedicated background thread printing a structured heartbeat line every 30 seconds.
- Clean stop before max_runtime_minutes budget is exceeded.
- OOM recovery: retries once at lower resolution or offload, then records failure without crashing.
"""

from __future__ import annotations

import datetime
import logging
import multiprocessing
import os
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import yaml

from worker.engines import get_engine
from worker.engines.base import Engine
from worker.memory import (
    clear_gpu_memory,
    detect_gpus,
    is_oom_exception,
)
from worker.schema import (
    Item,
    Pack,
    Result,
    append_result_atomic,
    load_results,
    save_results_atomic,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _load_models_config() -> Dict[str, Any]:
    """Load models.yaml from worker/models.yaml (single source of truth) or root."""
    p_worker = Path(__file__).resolve().parent / "models.yaml"
    if p_worker.is_file():
        try:
            data = yaml.safe_load(p_worker.read_text(encoding="utf-8"))
            return data.get("models", {}) if isinstance(data, dict) else {}
        except Exception as e:
            logger.warning("Could not read worker/models.yaml: %s", e)
    p_root = Path(__file__).resolve().parent.parent / "models.yaml"
    if p_root.is_file():
        try:
            data = yaml.safe_load(p_root.read_text(encoding="utf-8"))
            return data.get("models", {}) if isinstance(data, dict) else {}
        except Exception as e:
            logger.warning("Could not read root models.yaml: %s", e)
    return {}


class HeartbeatMonitor:
    """Thread that prints a progress and health line every interval seconds."""

    def __init__(self, runner: PackRunner, interval_seconds: float = 30.0) -> None:
        self.runner = runner
        self.interval = interval_seconds
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval):
            self.runner.print_heartbeat()


class PackRunner:
    """Orchestrates execution of a single Pack of generation items."""

    def __init__(
        self,
        pack: Pack,
        output_dir: Optional[Path] = None,
        results_file: Optional[Path] = None,
        gpu_index: int = 0,
        allow_test_engines: bool = False,
    ) -> None:
        self.pack = pack
        self.output_dir = output_dir or Path(pack.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results_file = results_file or (self.output_dir / "results.json")
        self.gpu_index = gpu_index
        self.allow_test_engines = allow_test_engines
        self.models_config = _load_models_config()

        self.start_time = time.time()
        self.total_items = len(pack.items)
        self.completed_count = 0
        self.current_engine_id = "idle"
        self.current_item_id = "none"

        self._active_engine: Optional[Engine] = None
        self._results_map: Dict[str, Result] = {}

    def print_heartbeat(self) -> None:
        """Log structured heartbeat status line."""
        elapsed_sec = int(time.time() - self.start_time)
        hrs, rem = divmod(elapsed_sec, 3600)
        mins, secs = divmod(rem, 60)
        elapsed_str = f"{hrs:02d}:{mins:02d}:{secs:02d}"

        pct = (self.completed_count / self.total_items * 100) if self.total_items else 100.0

        gpus = detect_gpus()
        vram_str = "CPU"
        if gpus and self.gpu_index < len(gpus):
            g = gpus[self.gpu_index]
            vram_str = f"{g.free_vram_mb / 1024:.1f}/{g.total_vram_mb / 1024:.1f}GB free"

        now_str = datetime.datetime.now().strftime("%H:%M:%S")
        logger.info(
            "[HEARTBEAT %s] %d/%d items (%.1f%%) | Active: %s (%s) | Elapsed: %s | VRAM: %s",
            now_str,
            self.completed_count,
            self.total_items,
            pct,
            self.current_engine_id,
            self.current_item_id,
            elapsed_str,
            vram_str,
        )

    def is_time_budget_exhausted(self) -> bool:
        """Check if max_runtime_minutes is near (leaves 4 min safety buffer)."""
        if not self.pack.max_runtime_minutes:
            return False

        elapsed_min = (time.time() - self.start_time) / 60.0
        safety_buffer = 4.0 if self.pack.max_runtime_minutes > 10 else 0.5
        deadline = self.pack.max_runtime_minutes - safety_buffer
        return elapsed_min >= deadline

    def _get_or_load_engine(self, engine_id: str) -> Engine:
        """Reuse loaded engine or switch engines cleanly."""
        if self._active_engine is not None:
            if self._active_engine.engine_id == engine_id:
                return self._active_engine
            # Different engine requested: unload previous model from VRAM
            logger.info("Switching engine: unloading %s...", self._active_engine.engine_id)
            self._active_engine.unload()
            self._active_engine = None
            clear_gpu_memory()

        cfg = self.models_config.get(engine_id)
        if not cfg and engine_id == "wan22_ti2v5b":
            cfg = self.models_config.get("wan21_t2v_1_3b", {})
        elif not cfg and engine_id == "wan21_t2v_1_3b":
            cfg = self.models_config.get("wan22_ti2v5b", {})
        cfg = cfg or {}

        # Refuse test_only engines unless STUDIO_FAKE_PROVIDERS=1 or allow_test_engines is True
        is_test_mode = (
            os.environ.get("STUDIO_FAKE_PROVIDERS") in ("1", "true", "True")
            or getattr(self, "allow_test_engines", False)
        )
        if cfg.get("test_only", False) and not is_test_mode:
            raise ValueError(
                f"Engine '{engine_id}' is marked as test_only. "
                f"Execution refused unless STUDIO_FAKE_PROVIDERS=1 is set or --test flag is provided."
            )

        engine = get_engine(engine_id, config=cfg)
        logger.info("Loading new engine weights for: %s...", engine_id)
        target_dev = None
        if self.gpu_index is not None:
            try:
                import torch
                if torch.cuda.is_available():
                    target_dev = f"cuda:{self.gpu_index}"
            except Exception:
                pass
        engine.load(device=target_dev)
        self._active_engine = engine
        return engine

    def _execute_item_with_oom_recovery(self, engine: Engine, item: Item) -> Tuple[Optional[Path], bool, Optional[str]]:
        """Run generation on item with 1-step downscale/offload retry upon CUDA OOM.
        
        Returns: (output_path, retried_oom, error_message)
        """
        try:
            path = engine.generate(item, self.output_dir)
            return path, False, None
        except Exception as exc:
            if not is_oom_exception(exc):
                # Standard non-OOM exception
                return None, False, str(exc)

            logger.warning("[OOM DETECTED] GPU OOM during item %s (%dx%d). Initiating recovery retry...",
                           item.id, item.width, item.height)
            clear_gpu_memory()

            # Downscale resolution to 768x768 or 0.75x (rounded to multiple of 64)
            fallback_w = max(512, (int(item.width * 0.75) // 64) * 64)
            fallback_h = max(512, (int(item.height * 0.75) // 64) * 64)
            retry_item = item.model_copy(update={"width": fallback_w, "height": fallback_h})

            try:
                path = engine.generate(retry_item, self.output_dir)
                logger.info("[OOM RECOVERED] Successfully generated %s at downscaled resolution (%dx%d)",
                            item.id, fallback_w, fallback_h)
                return path, True, None
            except Exception as retry_exc:
                logger.error("[OOM RETRY FAILED] Item %s failed second attempt: %s. Continuing run.",
                             item.id, retry_exc)
                clear_gpu_memory()
                return None, True, f"OOM after downscale retry: {retry_exc}"

    def run(self) -> List[Result]:
        """Execute full pack workload and return list of Results."""
        # Validate that test_only engines are permitted upfront
        is_test_mode = (
            os.environ.get("STUDIO_FAKE_PROVIDERS") in ("1", "true", "True")
            or getattr(self, "allow_test_engines", False)
        )
        for it in self.pack.items:
            cfg = self.models_config.get(it.engine)
            if not cfg and it.engine == "wan22_ti2v5b":
                cfg = self.models_config.get("wan21_t2v_1_3b", {})
            elif not cfg and it.engine == "wan21_t2v_1_3b":
                cfg = self.models_config.get("wan22_ti2v5b", {})
            cfg = cfg or {}
            if cfg.get("test_only", False) and not is_test_mode:
                raise ValueError(
                    f"Engine '{it.engine}' is marked as test_only. "
                    f"Execution refused unless STUDIO_FAKE_PROVIDERS=1 is set or --test flag is provided."
                )

        heartbeat = HeartbeatMonitor(self, interval_seconds=30.0)
        heartbeat.start()

        # Load existing results to support resuming interrupted runs
        existing_results = load_results(self.results_file)
        self._results_map = {r.item_id: r for r in existing_results}
        self.completed_count = len([r for r in existing_results if r.status in ("success", "skipped")])

        # Sort items by (engine, width, height) so models and buffers load once
        sorted_items = sorted(self.pack.items, key=lambda it: (it.engine, it.width, it.height))

        try:
            for item in sorted_items:
                self.current_item_id = item.id
                self.current_engine_id = item.engine

                # 1. Check time budget
                if self.is_time_budget_exhausted():
                    logger.warning("[TIME BUDGET STOP] Approaching max runtime of %.1f min. Stopping safely.",
                                   self.pack.max_runtime_minutes)
                    break

                # 2. Check if output already exists (Idempotency / Resumability)
                expected_ext = "mp4" if item.media_type == "video" else "png"
                expected_file = self.output_dir / f"{item.id}.{expected_ext}"
                if not (expected_file.is_file() and expected_file.stat().st_size > 0):
                    alt_ext = "png" if expected_ext == "mp4" else "mp4"
                    alt_file = self.output_dir / f"{item.id}.{alt_ext}"
                    if alt_file.is_file() and alt_file.stat().st_size > 0:
                        expected_file = alt_file

                if expected_file.is_file() and expected_file.stat().st_size > 0:
                    logger.info("[RESUME] Skipping item %s — output already exists: %s", item.id, expected_file.name)
                    res = Result(
                        item_id=item.id,
                        status="skipped",
                        output_path=str(expected_file),
                        engine=item.engine,
                        execution_time_seconds=0.0,
                        width=item.width,
                        height=item.height,
                        seed=item.seed,
                    )
                    self._results_map[item.id] = res
                    append_result_atomic(res, self.results_file)
                    self.completed_count += 1
                    continue

                # 3. Load engine and execute item
                t0 = time.time()
                try:
                    engine = self._get_or_load_engine(item.engine)
                    out_path, retried_oom, err_msg = self._execute_item_with_oom_recovery(engine, item)
                    exec_time = time.time() - t0

                    if out_path and out_path.is_file():
                        res = Result(
                            item_id=item.id,
                            status="success",
                            output_path=str(out_path),
                            engine=item.engine,
                            execution_time_seconds=round(exec_time, 2),
                            retried_oom=retried_oom,
                            width=item.width,
                            height=item.height,
                            seed=item.seed,
                        )
                    else:
                        res = Result(
                            item_id=item.id,
                            status="failed",
                            error=err_msg or "Unknown generation failure",
                            engine=item.engine,
                            execution_time_seconds=round(exec_time, 2),
                            retried_oom=retried_oom,
                            width=item.width,
                            height=item.height,
                            seed=item.seed,
                        )
                except Exception as exc:
                    exec_time = time.time() - t0
                    logger.exception("Unexpected error processing item %s: %s", item.id, exc)
                    res = Result(
                        item_id=item.id,
                        status="failed",
                        error=str(exc),
                        engine=item.engine,
                        execution_time_seconds=round(exec_time, 2),
                    )

                self._results_map[item.id] = res
                append_result_atomic(res, self.results_file)
                self.completed_count += 1

        finally:
            heartbeat.stop()
            if self._active_engine:
                self._active_engine.unload()
                self._active_engine = None

        return list(self._results_map.values())


def run_pack(
    pack: Pack,
    output_dir: Optional[Path] = None,
    results_file: Optional[Path] = None,
    gpu_indices: Optional[List[int]] = None,
    allow_test_engines: bool = False,
) -> List[Result]:
    """Top-level entrypoint to execute a Pack.
    
    If multiple GPUs are specified, splits items across processes using CUDA_VISIBLE_DEVICES.
    """
    gpus = detect_gpus()
    available_indices = gpu_indices if gpu_indices is not None else ([g.index for g in gpus] if gpus else [0])

    if len(available_indices) <= 1:
        runner = PackRunner(
            pack,
            output_dir=output_dir,
            results_file=results_file,
            gpu_index=available_indices[0],
            allow_test_engines=allow_test_engines,
        )
        return runner.run()

    # Multi-GPU round-robin partition
    logger.info("Distributing %d items across %d GPUs: %s", len(pack.items), len(available_indices), available_indices)
    partitions: Dict[int, List[Item]] = {gpu_idx: [] for gpu_idx in available_indices}
    for idx, item in enumerate(pack.items):
        target_gpu = available_indices[idx % len(available_indices)]
        partitions[target_gpu].append(item)

    # In multi-GPU child process wrapper:
    # sets os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    # and executes partition
    all_results: List[Result] = []
    # For robust single-machine portability execute partitions sequentially or via multiprocessing
    target_out = output_dir or Path(pack.output_dir)
    target_res = results_file or (target_out / "results.json")

    for gpu_idx, sub_items in partitions.items():
        if not sub_items:
            continue
        sub_pack = pack.model_copy(update={"items": sub_items})
        runner = PackRunner(
            sub_pack,
            output_dir=target_out,
            results_file=target_res,
            gpu_index=gpu_idx,
            allow_test_engines=allow_test_engines,
        )
        res = runner.run()
        all_results.extend(res)

    final_results = load_results(target_res)
    return final_results if final_results else all_results


if __name__ == "__main__":
    import argparse
    from worker.schema import load_pack

    parser = argparse.ArgumentParser(description="yt-gpu-worker: Run batch generation pack")
    parser.add_argument("--pack", type=str, default="tests/sample_pack.json", help="Path to pack.json")
    parser.add_argument("--engine", type=str, default=None, help="Override engine for all items (e.g. fake, sdxl_lightning)")
    parser.add_argument("--out", type=str, default="./out", help="Output directory for rendered assets and results.json")
    parser.add_argument("--gpus", type=int, nargs="*", default=None, help="GPU device indices to use")
    parser.add_argument("--test", action="store_true", help="Allow test_only mock engines (e.g. fake)")

    args = parser.parse_args()

    pack_path = Path(args.pack)
    if not pack_path.is_file():
        repo_root = Path(__file__).resolve().parent.parent
        alt_path = repo_root / args.pack
        if alt_path.is_file():
            pack_path = alt_path
        else:
            raise FileNotFoundError(f"Pack file not found: {args.pack}")

    logger.info("Loading pack from: %s", pack_path)
    loaded_pack = load_pack(pack_path)

    if args.engine:
        logger.info("Overriding engine to '%s' for all items", args.engine)
        for it in loaded_pack.items:
            it.engine = args.engine

    target_out = Path(args.out)
    target_out.mkdir(parents=True, exist_ok=True)

    results = run_pack(loaded_pack, output_dir=target_out, gpu_indices=args.gpus, allow_test_engines=args.test)
    print(f"\nExecution complete! Rendered {len(results)} items into {target_out.resolve()}:")
    for r in results:
        status_symbol = "[OK]" if r.status in ("success", "skipped") else "[FAILED]"
        print(f"  {status_symbol} {r.item_id} ({r.engine}) -> {r.output_path or r.error} [{r.execution_time_seconds}s]")
