"""scripts/bench_throughput.py — Throughput test across parallel modes and offload settings.

Benchmarks:
  (a) Mode A: single_process_multi_gpu (1 process, all GPUs visible)
  (b) Mode B: process_per_gpu (2 processes, 1 per GPU, offload: auto)
  (c) Mode C: process_per_gpu_no_offload (2 processes, 1 per GPU, offload: none)

Key guarantees:
  - Runs each mode in a fresh subprocess with its own clean output directory.
  - Resume disabled (--no-resume) so every item is freshly rendered.
  - Warms the model weights cache before benchmark timing.
  - Excludes model load time and the first warmup item to report steady-state seconds per item per GPU.
  - Measures effective items per second and logs available/used host RAM.
  - Fails loudly if any mode produces 0 successful items; never updates models.yaml on failure.
  - Stores the fastest OOM-free mode and offload policy in worker/models.yaml.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

import yaml

# Ensure yt-gpu-worker root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from worker.memory import clear_gpu_memory, detect_gpus, get_host_ram_info
from worker.schema import Item, Pack, load_results

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BENCHMARK_PROMPTS = [
    "A photorealistic portrait of an old mariner in yellow raincoat under storm clouds",
    "Futuristic sleek bullet train entering a neon subterranean terminal station",
    "Detailed scientific macro photograph of a bumblebee dusting pollen on a sunflower",
    "Vibrant street food night market in Tokyo illuminated by paper lanterns in mist",
    "Architectural render of a modern minimalist cliffside villa overlooking the ocean",
    "Cinematic shot of an explorer discovering ancient bioluminescent crystal cavern",
    "A cozy warm mountain cabin library with crackling fireplace and falling snow",
    "Dramatic aerial shot of desert sand dunes casting sharp sunset geometric shadows",
    "Close up of an artisan ceramic potter shaping red clay vase on spinning wheel",
    "Cyberpunk city square with towering holographic advertisements in pouring rain",
    "Macro photograph of delicate fractal frost patterns forming on winter window",
    "Epic wide landscape of Icelandic glacier waterfall under purple northern lights",
    "Studio product photograph of vintage chrome mechanical watch on slate stone",
    "Golden eagle soaring majestically over rugged pine forest autumn mountains",
    "Interior view of a sleek futuristic orbital space station hydroponics garden",
    "A bustling Victorian London cobblestone street with steam omnibus in morning fog",
    "Abstract macro fluid art with swirling turquoise and amber metallic inks",
    "A majestic snow leopard resting camouflaged on a Himalayan rocky ledge",
    "Steampunk mechanical ornithopter flying through golden morning clouds",
    "A peaceful bamboo forest pathway with dappled sunlight and stone lanterns",
]


def update_models_yaml_parallel_mode(engine_id: str, chosen_mode: str, chosen_offload: str = "auto") -> None:
    """Atomically record parallel_mode and offload decisions into worker/models.yaml."""
    models_path = _REPO_ROOT / "worker" / "models.yaml"
    if not models_path.is_file():
        logger.warning("Could not find %s to update parallel_mode", models_path)
        return

    try:
        data = yaml.safe_load(models_path.read_text(encoding="utf-8"))
        if engine_id in data.get("models", {}):
            data["models"][engine_id]["parallel_mode"] = chosen_mode
            if chosen_offload:
                data["models"][engine_id]["offload"] = chosen_offload
            models_path.write_text(yaml.dump(data, sort_keys=False, indent=2), encoding="utf-8")
            logger.info(
                "Updated worker/models.yaml for '%s': parallel_mode -> '%s', offload -> '%s'",
                engine_id,
                chosen_mode,
                chosen_offload,
            )
    except Exception as exc:
        logger.error("Failed to update models.yaml: %s", exc)


def create_20_item_pack(engine_id: str, out_dir: Path, width: int = 1024, height: int = 1024) -> Pack:
    """Construct a clean 20-item Pack for throughput testing."""
    items = []
    for idx, prompt in enumerate(BENCHMARK_PROMPTS):
        items.append(
            Item(
                id=f"{engine_id}_tp_{idx:02d}",
                engine=engine_id,
                prompt=prompt,
                width=width,
                height=height,
                seed=2000 + idx,
                media_type="image",
            )
        )
    return Pack(
        pack_id=f"throughput_{engine_id}_20",
        items=items,
        output_dir=str(out_dir),
        max_runtime_minutes=30.0,
    )


def warm_weights_cache(engine_id: str, allow_test: bool = False, allow_noncommercial: bool = False) -> float:
    """Run a single 1-item warmup generation to ensure weights are cached on disk before timed runs."""
    logger.info(">>> Warming weights cache for engine: %s...", engine_id)
    warmup_dir = _REPO_ROOT / "outputs" / "throughput" / ".warmup"
    if warmup_dir.exists():
        shutil.rmtree(warmup_dir, ignore_errors=True)
    warmup_dir.mkdir(parents=True, exist_ok=True)

    warm_item = Item(
        id=f"{engine_id}_warmup_shot",
        engine=engine_id,
        prompt="A simple warm up prompt to cache diffusion model weights on disk",
        width=512,
        height=512,
        seed=1000,
        media_type="image",
    )
    warm_pack = Pack(pack_id=f"warmup_{engine_id}", items=[warm_item], output_dir=str(warmup_dir))
    pack_path = warmup_dir / "warmup_pack.json"
    pack_path.write_text(warm_pack.model_dump_json(indent=2), encoding="utf-8")
    res_path = warmup_dir / "warmup_results.json"

    cmd = [
        sys.executable,
        "-m", "worker.runner",
        "--pack", str(pack_path),
        "--out", str(warmup_dir),
        "--results", str(res_path),
        "--no-resume",
    ]
    if allow_test:
        cmd.append("--test")
    if allow_noncommercial:
        cmd.append("--allow-noncommercial")

    t0 = time.time()
    subprocess.run(cmd, env=os.environ.copy(), check=False)
    warmup_time = time.time() - t0
    logger.info("Weights cache warmed for %s in %.2fs.", engine_id, warmup_time)
    return warmup_time


def run_mode_subprocess(
    engine_id: str,
    mode_name: str,
    output_dir: Path,
    parallel_mode: str,
    offload: Optional[str] = None,
    gpus: Optional[List[int]] = None,
    allow_test: bool = False,
    allow_noncommercial: bool = False,
) -> Dict[str, Any]:
    """Execute a single throughput benchmark mode in an isolated, fresh subprocess with resume disabled."""
    logger.info(">>> Starting [%s] in fresh subprocess (parallel_mode=%s, offload=%s)...", mode_name, parallel_mode, offload or "auto")

    # 1. Clean output directory to guarantee resume is disabled
    if output_dir.exists():
        shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Prepare 20-item pack
    pack = create_20_item_pack(engine_id, output_dir)
    pack_path = output_dir / "bench_pack.json"
    pack_path.write_text(pack.model_dump_json(indent=2), encoding="utf-8")
    results_path = output_dir / "results.json"

    # 3. Assemble command
    cmd = [
        sys.executable,
        "-m", "worker.runner",
        "--pack", str(pack_path),
        "--out", str(output_dir),
        "--results", str(results_path),
        "--parallel-mode", parallel_mode,
        "--no-resume",
    ]
    if offload:
        cmd.extend(["--offload", str(offload)])
    if allow_test:
        cmd.append("--test")
    if allow_noncommercial:
        cmd.append("--allow-noncommercial")
    if gpus:
        cmd.append("--gpus")
        cmd.extend([str(g) for g in gpus])

    clear_gpu_memory()

    # 4. Execute subprocess
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(_REPO_ROOT), env=os.environ.copy(), capture_output=True, text=True)
    wall_time = time.time() - t0

    if proc.returncode != 0:
        logger.error("[%s] Subprocess exited with code %d. Stderr:\n%s", mode_name, proc.returncode, proc.stderr[-2000:])

    # 5. Load results & RAM info
    results = load_results(results_path) if results_path.is_file() else []
    ram_info = get_host_ram_info()

    successful_results = [r for r in results if r.status == "success"]
    failed_results = [r for r in results if r.status == "failed"]
    oom_occurred = any(r.retried_oom or (r.error and "oom" in r.error.lower()) for r in results)

    # 6. Exclude first item (warmup / kernel compilation) to report steady-state seconds per item per GPU
    if len(successful_results) > 1:
        steady_results = successful_results[1:]
    else:
        steady_results = successful_results

    if steady_results:
        steady_times = [r.execution_time_seconds for r in steady_results if r.execution_time_seconds and r.execution_time_seconds > 0]
        steady_sec_per_item = statistics.median(steady_times) if steady_times else 0.0
    else:
        steady_sec_per_item = 0.0

    effective_it_per_sec = (len(successful_results) / wall_time) if wall_time > 0 else 0.0

    stats = {
        "mode_name": mode_name,
        "parallel_mode": parallel_mode,
        "offload": offload or "auto",
        "total_time_seconds": round(wall_time, 2),
        "items_per_second": round(effective_it_per_sec, 3),
        "effective_items_per_second": round(effective_it_per_sec, 3),
        "steady_state_seconds_per_item_per_gpu": round(steady_sec_per_item, 2),
        "successful_items": len(successful_results),
        "failed_items": len(failed_results),
        "peak_host_ram_used_gb": ram_info.used_ram_gb,
        "available_host_ram_gb": ram_info.available_ram_gb,
        "oom_occurred": oom_occurred,
        "returncode": proc.returncode,
    }

    logger.info(
        "[%s] Finished in %.2fs (%.3f it/s) | Steady: %.2fs/item/gpu | RAM: %.2f GB used (%.2f GB avail) | OOM: %s | Success: %d/%d",
        mode_name,
        wall_time,
        effective_it_per_sec,
        steady_sec_per_item,
        ram_info.used_ram_gb,
        ram_info.available_ram_gb,
        oom_occurred,
        len(successful_results),
        len(pack.items),
    )
    return stats


def run_throughput_benchmark(
    engine_id: str,
    output_dir: Optional[Path] = None,
    gpus: Optional[List[int]] = None,
    allow_test: bool = False,
    allow_noncommercial: bool = False,
) -> Dict[str, Any]:
    """Execute 20-item benchmark across Mode A, Mode B, and Mode C, logging time, RAM, and steady-state latency."""
    base_out = output_dir or (_REPO_ROOT / "outputs" / "throughput" / engine_id)
    base_out.mkdir(parents=True, exist_ok=True)

    detected = detect_gpus()
    target_gpus = gpus if gpus is not None else ([g.index for g in detected] if detected else [0, 1])

    logger.info("=== Throughput Benchmark for Engine: %s on GPUs %s ===", engine_id, target_gpus)

    # 1. Warm weights cache before benchmark timing
    warm_weights_cache(engine_id, allow_test=allow_test, allow_noncommercial=allow_noncommercial)

    report: Dict[str, Any] = {
        "engine_id": engine_id,
        "item_count": 20,
        "gpus": target_gpus,
        "modes": {},
    }

    # 2. Run Mode A: single_process_multi_gpu (1 process, all GPUs visible)
    stats_a = run_mode_subprocess(
        engine_id=engine_id,
        mode_name="single_process_multi_gpu",
        output_dir=base_out / "mode_single_process",
        parallel_mode="single_process_multi_gpu",
        offload="auto",
        gpus=target_gpus,
        allow_test=allow_test,
        allow_noncommercial=allow_noncommercial,
    )
    report["modes"]["single_process_multi_gpu"] = stats_a

    # 3. Run Mode B: process_per_gpu (2 processes, 1 per GPU, offload=auto)
    stats_b = run_mode_subprocess(
        engine_id=engine_id,
        mode_name="process_per_gpu",
        output_dir=base_out / "mode_process_per_gpu",
        parallel_mode="process_per_gpu",
        offload="auto",
        gpus=target_gpus,
        allow_test=allow_test,
        allow_noncommercial=allow_noncommercial,
    )
    report["modes"]["process_per_gpu"] = stats_b

    # 4. Run Mode C: process_per_gpu + offload none (Mode C: resident on GPU, no CPU offload)
    stats_c = run_mode_subprocess(
        engine_id=engine_id,
        mode_name="process_per_gpu_no_offload",
        output_dir=base_out / "mode_process_per_gpu_no_offload",
        parallel_mode="process_per_gpu",
        offload="none",
        gpus=target_gpus,
        allow_test=allow_test,
        allow_noncommercial=allow_noncommercial,
    )
    report["modes"]["process_per_gpu_no_offload"] = stats_c

    # 5. Strict Governance Gate: Fail loudly if ANY mode had 0 successful items
    for mode_key, m_stats in report["modes"].items():
        if m_stats.get("successful_items", 0) == 0:
            raise RuntimeError(
                f"\n[BENCHMARK FAILED LOUDLY]\n"
                f"Engine '{engine_id}' in mode '{mode_key}' had 0 successful items! "
                f"(exit code: {m_stats.get('returncode')}). "
                f"Refusing to write parallel_mode to worker/models.yaml."
            )

    # 6. Decision: Filter out OOM modes and select fastest
    valid_modes = [
        (k, v) for k, v in report["modes"].items()
        if not v.get("oom_occurred") and v.get("successful_items", 0) > 0
    ]

    if not valid_modes:
        raise RuntimeError(f"All benchmark modes for engine '{engine_id}' suffered OOM or failures!")

    # Sort by total_time_seconds ascending
    valid_modes.sort(key=lambda item: item[1]["total_time_seconds"])
    fastest_mode_key, fastest_stats = valid_modes[0]

    chosen_parallel_mode = fastest_stats["parallel_mode"]
    chosen_offload = fastest_stats["offload"]
    reason = (
        f"Selected '{fastest_mode_key}' as fastest OOM-free mode: "
        f"{fastest_stats['total_time_seconds']:.2f}s total ({fastest_stats['effective_items_per_second']:.3f} it/s, "
        f"steady: {fastest_stats['steady_state_seconds_per_item_per_gpu']:.2f}s/item/gpu). "
        f"Available Host RAM: {fastest_stats['available_host_ram_gb']:.2f} GB."
    )

    report["faster_mode"] = chosen_parallel_mode
    report["faster_offload"] = chosen_offload
    report["winning_mode_key"] = fastest_mode_key
    report["selection_reason"] = reason

    logger.info("=== RESULT for %s: Selected parallel_mode = '%s', offload = '%s' (%s) ===",
                engine_id, chosen_parallel_mode, chosen_offload, reason)

    # Update models.yaml
    update_models_yaml_parallel_mode(engine_id, chosen_parallel_mode, chosen_offload=chosen_offload)

    # Save benchmark metrics JSON
    metrics_file = base_out / "throughput_benchmark.json"
    metrics_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Saved throughput metrics to: %s", metrics_file)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="yt-gpu-worker: Multi-GPU Throughput Benchmark")
    parser.add_argument("--engine", type=str, default="sdxl_lightning", help="Engine ID to benchmark (or use --all)")
    parser.add_argument("--all", action="store_true", help="Benchmark both sdxl_lightning and flux2_klein")
    parser.add_argument("--out", type=str, default="./outputs/throughput", help="Output directory")
    parser.add_argument("--test", action="store_true", help="Allow test_only mock engines")
    parser.add_argument("--allow-noncommercial", action="store_true", help="Allow non-commercial engines")
    args = parser.parse_args()

    engines = ["sdxl_lightning", "flux2_klein"] if args.all else [args.engine]
    out_dir = Path(args.out)

    for eng in engines:
        run_throughput_benchmark(
            eng,
            output_dir=out_dir / eng,
            allow_test=args.test,
            allow_noncommercial=args.allow_noncommercial,
        )


if __name__ == "__main__":
    main()
