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
import sys
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
    """Load models.yaml strictly from worker/models.yaml (single source of truth)."""
    p_worker = Path(__file__).resolve().parent / "models.yaml"
    if p_worker.is_file():
        try:
            data = yaml.safe_load(p_worker.read_text(encoding="utf-8"))
            return data.get("models", {}) if isinstance(data, dict) else {}
        except Exception as e:
            logger.warning("Could not read worker/models.yaml: %s", e)
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
        allow_noncommercial: bool = False,
        allow_slow_bf16: bool = False,
        offload: Optional[str] = None,
        allow_resume: bool = True,
    ) -> None:
        self.pack = pack
        self.output_dir = output_dir or Path(pack.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results_file = results_file or (self.output_dir / "results.json")
        self.gpu_index = gpu_index
        self.allow_test_engines = allow_test_engines
        self.allow_noncommercial = allow_noncommercial
        self.allow_slow_bf16 = allow_slow_bf16
        self.offload = (offload.lower() if offload else None)
        self.allow_resume = allow_resume
        self.video_clips_rendered = 0
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
        if gpus and self.gpu_index is not None and self.gpu_index < len(gpus):
            g = gpus[self.gpu_index]
            vram_str = f"{g.free_vram_mb / 1024:.1f}/{g.total_vram_mb / 1024:.1f}GB free (GPU {self.gpu_index})"
        elif gpus:
            total_free = sum(g.free_vram_mb for g in gpus) / 1024.0
            total_vram = sum(g.total_vram_mb for g in gpus) / 1024.0
            vram_str = f"{total_free:.1f}/{total_vram:.1f}GB free across {len(gpus)} GPUs"

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
        cfg = dict(cfg or {})
        if self.allow_slow_bf16:
            cfg["allow_slow_bf16"] = True

        # Refuse test_only and commercial_ok=false engines unless explicitly permitted
        is_test_mode = (
            os.environ.get("STUDIO_FAKE_PROVIDERS") in ("1", "true", "True")
            or getattr(self, "allow_test_engines", False)
        )
        is_noncommercial_mode = (
            os.environ.get("STUDIO_ALLOW_NONCOMMERCIAL") in ("1", "true", "True")
            or getattr(self, "allow_noncommercial", False)
        )
        if cfg.get("test_only", False) and not is_test_mode:
            raise ValueError(
                f"Engine '{engine_id}' is marked as test_only. "
                f"Execution refused unless STUDIO_FAKE_PROVIDERS=1 is set or --test flag is provided."
            )
        if cfg.get("commercial_ok") is False and not is_noncommercial_mode:
            raise ValueError(
                f"Engine '{engine_id}' has commercial_ok=false (license: {cfg.get('license', 'unknown')}). "
                f"Execution refused unless --allow-noncommercial flag or STUDIO_ALLOW_NONCOMMERCIAL=1 is passed."
            )

        if self.offload:
            cfg["offload"] = self.offload
        engine = get_engine(engine_id, config=cfg)
        if self.offload:
            engine.offload = self.offload
        logger.info("Loading new engine weights for: %s (offload=%s)...", engine_id, getattr(engine, 'offload', 'auto'))
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

    def _execute_item_with_oom_recovery(
        self, engine: Engine, item: Item
    ) -> Tuple[Optional[Path], bool, Optional[str], Optional[str]]:
        """Run generation on item with a 2-stage OOM retry fallback chain:
        1. Retry once with model CPU offload
        2. Retry once at a smaller downscaled resolution (0.75x)

        Returns: (output_path, retried_oom, fallback_applied, error_message)
        """
        # Attempt 1: Normal execution
        try:
            path = engine.generate(item, self.output_dir)
            return path, False, None, None
        except Exception as exc:
            if not is_oom_exception(exc):
                return None, False, None, str(exc)

            logger.warning(
                "[OOM DETECTED] GPU OOM during item %s (%dx%d). Fallback Stage 1: Retrying with model CPU offload...",
                item.id, item.width, item.height
            )
            clear_gpu_memory()

        # Fallback Stage 1: Retry once with model CPU offload
        try:
            if hasattr(engine, "pipe") and engine.pipe is not None:
                if hasattr(engine.pipe, "enable_model_cpu_offload") and callable(getattr(engine.pipe, "enable_model_cpu_offload")):
                    target_gpu = self.gpu_index if self.gpu_index is not None else 0
                    try:
                        engine.pipe.enable_model_cpu_offload(gpu_id=target_gpu)
                    except TypeError:
                        engine.pipe.enable_model_cpu_offload()
                    engine.offload = "model"
                    logger.info("[OOM FALLBACK 1] Enabled model CPU offload on engine %s", engine.engine_id)

            path = engine.generate(item, self.output_dir)
            logger.info("[OOM RECOVERED] Successfully generated %s via model CPU offload fallback", item.id)
            return path, True, "cpu_offload", None
        except Exception as offload_exc:
            if not is_oom_exception(offload_exc):
                return None, True, "cpu_offload", str(offload_exc)

            logger.warning(
                "[OOM DETECTED] GPU OOM persisted with model CPU offload during item %s. Fallback Stage 2: Retrying at smaller resolution...",
                item.id
            )
            clear_gpu_memory()

        # Fallback Stage 2: Retry once at smaller resolution (0.75x)
        fallback_w = max(512, (int(item.width * 0.75) // 64) * 64)
        fallback_h = max(512, (int(item.height * 0.75) // 64) * 64)
        retry_item = item.model_copy(update={"width": fallback_w, "height": fallback_h})

        try:
            path = engine.generate(retry_item, self.output_dir)
            logger.info(
                "[OOM RECOVERED] Successfully generated %s at downscaled resolution (%dx%d)",
                item.id, fallback_w, fallback_h
            )
            return path, True, "downscale", None
        except Exception as downscale_exc:
            logger.error(
                "[OOM RETRY FAILED] Item %s failed both CPU offload and downscale fallbacks: %s",
                item.id, downscale_exc
            )
            clear_gpu_memory()
            return None, True, "downscale", f"OOM after cpu_offload and downscale retry: {downscale_exc}"

    def run(self) -> List[Result]:
        """Execute full pack workload and return list of Results."""
        # Validate that test_only and commercial_ok constraints are met upfront
        is_test_mode = (
            os.environ.get("STUDIO_FAKE_PROVIDERS") in ("1", "true", "True")
            or getattr(self, "allow_test_engines", False)
        )
        is_noncommercial_mode = (
            os.environ.get("STUDIO_ALLOW_NONCOMMERCIAL") in ("1", "true", "True")
            or getattr(self, "allow_noncommercial", False)
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
            if cfg.get("commercial_ok") is False and not is_noncommercial_mode:
                raise ValueError(
                    f"Engine '{it.engine}' has commercial_ok=false (license: {cfg.get('license', 'unknown')}). "
                    f"Execution refused unless --allow-noncommercial flag or STUDIO_ALLOW_NONCOMMERCIAL=1 is passed."
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

                # Video guard: check max_clips_per_run
                is_video = (item.media_type == "video" or item.engine in ("wan21_t2v_1_3b", "wan22_ti2v5b"))
                if is_video:
                    cfg = self.models_config.get(item.engine, {})
                    max_clips = int(cfg.get("max_clips_per_run", 2))
                    if self.video_clips_rendered >= max_clips:
                        logger.warning(
                            "[VIDEO GUARD] Skipping video item '%s': exceeds max_clips_per_run (%d). One 33-frame clip takes ~6-22 min on T4.",
                            item.id,
                            max_clips,
                        )
                        skip_res = Result(
                            item_id=item.id,
                            status="skipped",
                            engine=item.engine,
                            error=f"Skipped: exceeds max_clips_per_run ({max_clips}). Video generation takes ~6-22 min per clip on T4.",
                            execution_time_seconds=0.0,
                            width=item.width,
                            height=item.height,
                            seed=item.seed,
                        )
                        self._results_map[item.id] = skip_res
                        append_result_atomic(skip_res, self.results_file)
                        self.completed_count += 1
                        continue
                    self.video_clips_rendered += 1

                # 2. Check if output already exists (Idempotency / Resumability)
                expected_ext = "mp4" if item.media_type == "video" else "png"
                expected_file = self.output_dir / f"{item.id}.{expected_ext}"
                if not (expected_file.is_file() and expected_file.stat().st_size > 0):
                    alt_ext = "png" if expected_ext == "mp4" else "mp4"
                    alt_file = self.output_dir / f"{item.id}.{alt_ext}"
                    if alt_file.is_file() and alt_file.stat().st_size > 0:
                        expected_file = alt_file

                if self.allow_resume and expected_file.is_file() and expected_file.stat().st_size > 0:
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
                    out_path, retried_oom, fallback_applied, err_msg = self._execute_item_with_oom_recovery(engine, item)
                    exec_time = time.time() - t0

                    # Sample peak VRAM allocated during item generation
                    peak_vram_mb = None
                    try:
                        import torch
                        if torch.cuda.is_available():
                            target_gpu = self.gpu_index if self.gpu_index is not None else 0
                            peak_vram_mb = round(torch.cuda.max_memory_allocated(target_gpu) / (1024.0 * 1024.0), 2)
                    except Exception:
                        pass

                    if out_path and out_path.is_file():
                        res = Result(
                            item_id=item.id,
                            status="success",
                            output_path=str(out_path),
                            engine=item.engine,
                            execution_time_seconds=round(exec_time, 2),
                            peak_vram_mb=peak_vram_mb,
                            retried_oom=retried_oom,
                            fallback_applied=fallback_applied,
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
                            peak_vram_mb=peak_vram_mb,
                            retried_oom=retried_oom,
                            fallback_applied=fallback_applied,
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
    allow_noncommercial: bool = False,
    allow_slow_bf16: bool = False,
    parallel_mode: Optional[str] = None,
    offload: Optional[str] = None,
    allow_resume: bool = True,
) -> List[Result]:
    """Top-level entrypoint to execute a Pack.
    
    Supports two multi-GPU parallel execution modes:
    - 'single_process_multi_gpu': Single process with access to all GPUs (e.g. device_map='balanced')
    - 'process_per_gpu': Multiple worker processes, one isolated per GPU via CUDA_VISIBLE_DEVICES
    """
    gpus = detect_gpus()
    available_indices = gpu_indices if gpu_indices is not None else ([g.index for g in gpus] if gpus else [0])
    models_cfg = _load_models_config()

    primary_engine = pack.items[0].engine if pack.items else "sdxl_lightning"
    engine_cfg = models_cfg.get(primary_engine, {})

    effective_parallel_mode = parallel_mode or engine_cfg.get("parallel_mode")
    if not effective_parallel_mode:
        logger.warning(
            "No parallel_mode configured in models.yaml for engine '%s'; defaulting to 'process_per_gpu'.",
            primary_engine,
        )
        effective_parallel_mode = "process_per_gpu"

    target_out = output_dir or Path(pack.output_dir)
    target_out.mkdir(parents=True, exist_ok=True)
    target_res = results_file or (target_out / "results.json")

    # Mode A: Single process with all GPUs visible
    if len(available_indices) <= 1 or effective_parallel_mode == "single_process_multi_gpu":
        logger.info(
            "Executing pack in single-process mode on %d GPU(s): %s (parallel_mode=%s)",
            len(available_indices),
            available_indices,
            effective_parallel_mode,
        )
        runner = PackRunner(
            pack,
            output_dir=target_out,
            results_file=target_res,
            gpu_index=available_indices[0] if len(available_indices) == 1 else None,
            allow_test_engines=allow_test_engines,
            allow_noncommercial=allow_noncommercial,
            allow_slow_bf16=allow_slow_bf16,
            offload=offload,
            allow_resume=allow_resume,
        )
        return runner.run()

    # Mode B: process_per_gpu (enforce host RAM guard & partition)
    from worker.memory import get_host_ram_info
    ram_info = get_host_ram_info()
    max_engine_ram = 12.0
    for it in pack.items:
        cfg = models_cfg.get(it.engine, {})
        est = float(cfg.get("ram_estimate_gb", 12.0))
        if est > max_engine_ram:
            max_engine_ram = est

    safe_host_ram = ram_info.available_ram_gb * 0.85
    max_workers_for_ram = max(1, int(round((safe_host_ram + 0.5) / max_engine_ram))) if max_engine_ram > 0 else len(available_indices)
    allowed_workers = min(len(available_indices), max_workers_for_ram)

    if allowed_workers < len(available_indices):
        logger.info(
            "[HOST RAM WORKER GUARD] Limiting concurrent worker processes from %d to %d "
            "(available RAM: %.1f GB, safe threshold: %.1f GB, max engine RAM estimate: %.1f GB)",
            len(available_indices),
            allowed_workers,
            ram_info.available_ram_gb,
            safe_host_ram,
            max_engine_ram,
        )

    active_indices = available_indices[:allowed_workers]
    logger.info("Distributing %d items across %d active GPUs: %s (mode=process_per_gpu)", len(pack.items), len(active_indices), active_indices)
    partitions: Dict[int, List[Item]] = {gpu_idx: [] for gpu_idx in active_indices}
    for idx, item in enumerate(pack.items):
        target_gpu = active_indices[idx % len(active_indices)]
        partitions[target_gpu].append(item)

    all_results: List[Result] = []

    # In mock test environments or sequential execution fallback:
    is_mock_env = (os.environ.get("STUDIO_FAKE_PROVIDERS") in ("1", "true", "True") or not gpus)

    if is_mock_env:
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
                allow_noncommercial=allow_noncommercial,
                allow_slow_bf16=allow_slow_bf16,
                offload=offload,
                allow_resume=allow_resume,
            )
            res = runner.run()
            all_results.extend(res)
    else:
        # Multi-process execution via subprocess with isolated CUDA_VISIBLE_DEVICES
        import subprocess
        from concurrent.futures import ThreadPoolExecutor

        def _execute_worker_process(gpu_idx: int, sub_items: List[Item]) -> List[Result]:
            sub_pack = pack.model_copy(update={"items": sub_items, "pack_id": f"{pack.pack_id}_gpu{gpu_idx}"})
            sub_pack_path = target_out / f".temp_pack_gpu{gpu_idx}.json"
            sub_pack_path.write_text(sub_pack.model_dump_json(indent=2), encoding="utf-8")
            part_res_path = target_out / f".temp_results_gpu{gpu_idx}.json"

            cmd = [
                sys.executable,
                "-m", "worker.runner",
                "--pack", str(sub_pack_path),
                "--out", str(target_out),
                "--results", str(part_res_path),
                "--gpus", "0",
            ]
            if allow_test_engines:
                cmd.append("--test")
            if allow_noncommercial:
                cmd.append("--allow-noncommercial")
            if allow_slow_bf16:
                cmd.append("--allow-slow-bf16")
            if offload:
                cmd.extend(["--offload", str(offload)])
            if not allow_resume:
                cmd.append("--no-resume")

            child_env = os.environ.copy()
            child_env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
            try:
                subprocess.run(cmd, env=child_env, check=True)
                part_results: List[Result] = []
                if part_res_path.is_file():
                    part_results = load_results(part_res_path)
                    part_res_path.unlink()
                return part_results
            finally:
                if sub_pack_path.is_file():
                    sub_pack_path.unlink()

        with ThreadPoolExecutor(max_workers=len(partitions)) as executor:
            futures = [executor.submit(_execute_worker_process, g_idx, items) for g_idx, items in partitions.items() if items]
            for f in futures:
                worker_results = f.result()
                for r in worker_results:
                    append_result_atomic(r, target_res)

    final_results = load_results(target_res)
    return final_results if final_results else all_results


if __name__ == "__main__":
    import argparse
    from worker.schema import load_pack

    parser = argparse.ArgumentParser(description="yt-gpu-worker: Run batch generation pack")
    parser.add_argument("--pack", type=str, default="tests/sample_pack.json", help="Path to pack.json")
    parser.add_argument("--engine", type=str, default=None, help="Override engine for all items (e.g. fake, sdxl_lightning)")
    parser.add_argument("--out", type=str, default="./out", help="Output directory for rendered assets and results.json")
    parser.add_argument("--results", type=str, default=None, help="Explicit path for results JSON output file")
    parser.add_argument("--gpus", type=int, nargs="*", default=None, help="GPU device indices to use")
    parser.add_argument("--test", action="store_true", help="Allow test_only mock engines (e.g. fake)")
    parser.add_argument("--allow-noncommercial", action="store_true", help="Allow engines with commercial_ok: false (e.g. sdxl_turbo)")
    parser.add_argument("--allow-slow-bf16", action="store_true", help="Allow software bf16 emulation on compute capability < 8.0 GPUs")
    parser.add_argument("--parallel-mode", type=str, choices=["single_process_multi_gpu", "process_per_gpu"], default=None, help="Execution mode for multi-GPU setups")
    parser.add_argument("--offload", type=str, choices=["auto", "none", "model"], default=None, help="Offload policy override (auto | none | model)")
    parser.add_argument("--no-resume", action="store_true", help="Disable resuming from existing rendered assets")

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

    results = run_pack(
        loaded_pack,
        output_dir=target_out,
        results_file=Path(args.results) if args.results else None,
        gpu_indices=args.gpus,
        allow_test_engines=args.test,
        allow_noncommercial=args.allow_noncommercial,
        allow_slow_bf16=args.allow_slow_bf16,
        parallel_mode=args.parallel_mode,
        offload=args.offload,
        allow_resume=not args.no_resume,
    )
    print(f"\nExecution complete! Rendered {len(results)} items into {target_out.resolve()}:")
    for r in results:
        status_symbol = "[OK]" if r.status in ("success", "skipped") else "[FAILED]"
        print(f"  {status_symbol} {r.item_id} ({r.engine}) -> {r.output_path or r.error} [{r.execution_time_seconds}s]")
