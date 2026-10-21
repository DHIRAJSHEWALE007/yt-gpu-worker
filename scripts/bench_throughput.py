"""scripts/bench_throughput.py — Throughput test across parallel modes (single_process vs process_per_gpu).

Runs a 20-item pack for each engine in:
  (a) One process with both GPUs (single_process_multi_gpu)
  (b) Two processes, one per GPU via CUDA_VISIBLE_DEVICES (process_per_gpu)

Logs:
  - Total elapsed time
  - Items per second (throughput)
  - Per-process RAM consumption (peak host RAM)
  - Any OOM occurrences

Determines the faster, OOM-free mode per engine and updates worker/models.yaml with 'parallel_mode'.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

import yaml

# Ensure yt-gpu-worker root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from worker.memory import clear_gpu_memory, detect_gpus, get_host_ram_info
from worker.runner import run_pack
from worker.schema import Item, Pack

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
    "A bustling Victorian London cobblestone street with steam omnibuses in morning fog",
    "Abstract macro fluid art with swirling turquoise and amber metallic inks",
    "A majestic snow leopard resting camouflaged on a Himalayan rocky ledge",
    "Steampunk mechanical ornithopter flying through golden morning clouds",
    "A peaceful bamboo forest pathway with dappled sunlight and stone lanterns",
]


def update_models_yaml_parallel_mode(engine_id: str, chosen_mode: str) -> None:
    """Atomically record parallel_mode decision into worker/models.yaml."""
    models_path = _REPO_ROOT / "worker" / "models.yaml"
    if not models_path.is_file():
        logger.warning("Could not find %s to update parallel_mode", models_path)
        return

    try:
        data = yaml.safe_load(models_path.read_text(encoding="utf-8"))
        if engine_id in data.get("models", {}):
            data["models"][engine_id]["parallel_mode"] = chosen_mode
            models_path.write_text(yaml.dump(data, sort_keys=False, indent=2), encoding="utf-8")
            logger.info("Updated worker/models.yaml for '%s': parallel_mode -> '%s'", engine_id, chosen_mode)
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


def run_throughput_benchmark(
    engine_id: str,
    output_dir: Optional[Path] = None,
    gpus: Optional[List[int]] = None,
    allow_test: bool = False,
    allow_noncommercial: bool = False,
) -> Dict[str, Any]:
    """Execute 20-item benchmark in mode (a) and mode (b), logging time, RAM, and OOM."""
    base_out = output_dir or (_REPO_ROOT / "outputs" / "throughput" / engine_id)
    base_out.mkdir(parents=True, exist_ok=True)

    detected = detect_gpus()
    target_gpus = gpus if gpus is not None else ([g.index for g in detected] if detected else [0, 1])

    logger.info("=== Throughput Benchmark for Engine: %s on GPUs %s ===", engine_id, target_gpus)

    report: Dict[str, Any] = {
        "engine_id": engine_id,
        "item_count": 20,
        "gpus": target_gpus,
        "modes": {},
    }

    # -------------------------------------------------------------
    # (a) Run Mode A: single_process_multi_gpu (1 process, both GPUs)
    # -------------------------------------------------------------
    logger.info(">>> [MODE A: single_process_multi_gpu] Starting 20 items...")
    clear_gpu_memory()
    pack_a = create_20_item_pack(engine_id, base_out / "mode_single_process")

    t0_a = time.time()
    results_a = run_pack(
        pack_a,
        output_dir=base_out / "mode_single_process",
        gpu_indices=target_gpus,
        allow_test_engines=allow_test,
        allow_noncommercial=allow_noncommercial,
        parallel_mode="single_process_multi_gpu",
    )
    time_a = time.time() - t0_a
    ram_a = get_host_ram_info()

    oom_a = any(r.retried_oom or (r.error and "oom" in r.error.lower()) for r in results_a)
    failed_a = len([r for r in results_a if r.status == "failed"])
    success_a = len([r for r in results_a if r.status == "success"])

    report["modes"]["single_process_multi_gpu"] = {
        "total_time_seconds": round(time_a, 2),
        "items_per_second": round(len(results_a) / time_a, 3) if time_a > 0 else 0.0,
        "successful_items": success_a,
        "failed_items": failed_a,
        "peak_host_ram_used_gb": ram_a.used_ram_gb,
        "available_host_ram_gb": ram_a.available_ram_gb,
        "oom_occurred": oom_a,
    }
    logger.info(
        "Mode A Done: %.2fs (%.3f it/s) | RAM Used: %.2f GB | OOM: %s",
        time_a,
        len(results_a) / time_a if time_a > 0 else 0,
        ram_a.used_ram_gb,
        oom_a,
    )

    # -------------------------------------------------------------
    # (b) Run Mode B: process_per_gpu (2 processes, 1 per GPU)
    # -------------------------------------------------------------
    logger.info(">>> [MODE B: process_per_gpu] Starting 20 items partitioned across GPUs...")
    clear_gpu_memory()
    pack_b = create_20_item_pack(engine_id, base_out / "mode_process_per_gpu")

    t0_b = time.time()
    results_b = run_pack(
        pack_b,
        output_dir=base_out / "mode_process_per_gpu",
        gpu_indices=target_gpus,
        allow_test_engines=allow_test,
        allow_noncommercial=allow_noncommercial,
        parallel_mode="process_per_gpu",
    )
    time_b = time.time() - t0_b
    ram_b = get_host_ram_info()

    oom_b = any(r.retried_oom or (r.error and "oom" in r.error.lower()) for r in results_b)
    failed_b = len([r for r in results_b if r.status == "failed"])
    success_b = len([r for r in results_b if r.status == "success"])

    report["modes"]["process_per_gpu"] = {
        "total_time_seconds": round(time_b, 2),
        "items_per_second": round(len(results_b) / time_b, 3) if time_b > 0 else 0.0,
        "successful_items": success_b,
        "failed_items": failed_b,
        "peak_host_ram_used_gb": ram_b.used_ram_gb,
        "available_host_ram_gb": ram_b.available_ram_gb,
        "oom_occurred": oom_b,
    }
    logger.info(
        "Mode B Done: %.2fs (%.3f it/s) | RAM Used: %.2f GB | OOM: %s",
        time_b,
        len(results_b) / time_b if time_b > 0 else 0,
        ram_b.used_ram_gb,
        oom_b,
    )

    # -------------------------------------------------------------
    # Decision: Choose faster mode without OOM
    # -------------------------------------------------------------
    if oom_b and not oom_a:
        chosen_mode = "single_process_multi_gpu"
        reason = "Mode B suffered OOM; selected Mode A for stability."
    elif oom_a and not oom_b:
        chosen_mode = "process_per_gpu"
        reason = "Mode A suffered OOM; selected Mode B for stability."
    elif time_b < time_a:
        speedup = ((time_a - time_b) / time_a) * 100.0 if time_a > 0 else 0.0
        chosen_mode = "process_per_gpu"
        reason = f"Mode B is {speedup:.1f}% faster ({time_b:.2f}s vs {time_a:.2f}s)."
    else:
        speedup = ((time_b - time_a) / time_b) * 100.0 if time_b > 0 else 0.0
        chosen_mode = "single_process_multi_gpu"
        reason = f"Mode A is {speedup:.1f}% faster ({time_a:.2f}s vs {time_b:.2f}s)."

    report["faster_mode"] = chosen_mode
    report["selection_reason"] = reason

    logger.info("=== RESULT for %s: Selected parallel_mode = '%s' (%s) ===", engine_id, chosen_mode, reason)

    # Update models.yaml
    update_models_yaml_parallel_mode(engine_id, chosen_mode)

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
