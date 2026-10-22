"""scripts/bench_sizes.py — Benchmark engines across landscape (1344x768) and portrait (768x1344).

Runs 5 items each at:
  - 1344x768 (16:9 cinematic landscape)
  - 768x1344 (9:16 vertical / YouTube Shorts format)
using each engine's chosen parallel mode and offload policy from worker/models.yaml.

Reports:
  - Steady-state seconds per item per GPU (excluding the first warmup item).
  - Peak VRAM allocated per GPU.
  - Saves report to outputs/bench_sizes/sizes_benchmark.json.
  - Atomically updates worker/models.yaml seconds_per_resolution with a backup.
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

BENCH_SIZES = [
    (1344, 768, "1344x768"),
    (768, 1344, "768x1344"),
]

LANDSCAPE_PROMPTS = [
    "Cinematic 16:9 wide landscape of a misty Scandinavian fjord with snow-capped peaks in morning light",
    "Wide angle establishing shot of a futuristic neon cyber-city skyline at twilight with flying transports",
    "Panoramic view of an ancient desert temple surrounded by vast golden dunes at sunset",
    "Dramatic wide shot of an astronaut looking over an alien crimson planetary valley with twin moons",
    "Atmospheric horizontal composition of a cozy wooden tavern interior filled with warm firelight",
]

PORTRAIT_PROMPTS = [
    "Vertical 9:16 portrait of a cybernetic warrior in dark armor with neon blue glowing visor",
    "Towering vertical view of an ancient redwood tree stretching toward sunlit mist in forest canopy",
    "Full-length street fashion portrait of a model in oversized futuristic techwear in Tokyo rain",
    "Vertical shot of a majestic cascading waterfall falling into deep emerald mountain lagoon",
    "Hero vertical composition of a mythical sorceress in flowing robes casting glowing golden runes",
]


def load_models_yaml() -> Dict[str, Any]:
    models_path = _REPO_ROOT / "worker" / "models.yaml"
    if not models_path.is_file():
        raise FileNotFoundError(f"Missing {models_path}")
    return yaml.safe_load(models_path.read_text(encoding="utf-8"))


def update_models_yaml_resolutions(engine_id: str, per_size_seconds: Dict[str, float]) -> None:
    """Atomically update seconds_per_resolution in worker/models.yaml with backup."""
    models_path = _REPO_ROOT / "worker" / "models.yaml"
    bak_path = _REPO_ROOT / "worker" / "models.yaml.bak"

    if not models_path.is_file():
        logger.warning("Could not find %s to update resolutions", models_path)
        return

    try:
        shutil.copy2(models_path, bak_path)
        data = yaml.safe_load(models_path.read_text(encoding="utf-8"))

        if engine_id in data.get("models", {}):
            model_entry = data["models"][engine_id]
            if "seconds_per_resolution" not in model_entry or not isinstance(model_entry["seconds_per_resolution"], dict):
                model_entry["seconds_per_resolution"] = {}

            for size_str, sec in per_size_seconds.items():
                model_entry["seconds_per_resolution"][size_str] = round(sec, 1)

            # Validate YAML dump
            content = yaml.dump(data, sort_keys=False, indent=2)
            yaml.safe_load(content)  # test parse
            models_path.write_text(content, encoding="utf-8")
            logger.info("Atomically updated worker/models.yaml for '%s' seconds_per_resolution: %s",
                        engine_id, per_size_seconds)
    except Exception as exc:
        logger.error("Failed updating models.yaml for %s: %s", engine_id, exc)
        if bak_path.is_file():
            shutil.copy2(bak_path, models_path)


def create_size_pack(engine_id: str, width: int, height: int, size_label: str, out_dir: Path) -> Pack:
    prompts = LANDSCAPE_PROMPTS if width >= height else PORTRAIT_PROMPTS
    items = []
    for idx, prompt in enumerate(prompts):
        items.append(
            Item(
                id=f"{engine_id}_{size_label}_{idx:02d}",
                engine=engine_id,
                prompt=prompt,
                width=width,
                height=height,
                seed=3000 + idx,
                media_type="image",
            )
        )
    return Pack(
        pack_id=f"bench_sizes_{engine_id}_{size_label}",
        items=items,
        output_dir=str(out_dir),
        max_runtime_minutes=15.0,
    )


def run_size_benchmark_for_engine(
    engine_id: str,
    output_dir: Path,
    gpus: Optional[List[int]] = None,
    allow_test: bool = False,
    allow_noncommercial: bool = False,
) -> Dict[str, Any]:
    """Run 5 items each at 1344x768 and 768x1344 with engine's chosen mode and offload."""
    models_cfg = load_models_yaml().get("models", {})
    engine_meta = models_cfg.get(engine_id, {})
    if not engine_meta and not allow_test:
        raise ValueError(f"Engine '{engine_id}' not found in models.yaml")

    parallel_mode = engine_meta.get("parallel_mode", "process_per_gpu")
    offload = engine_meta.get("offload", "none")

    detected = detect_gpus()
    target_gpus = gpus if gpus is not None else ([g.index for g in detected] if detected else [0])

    logger.info("=== Benchmarking Sizes for Engine '%s' [mode=%s, offload=%s] on GPUs %s ===",
                engine_id, parallel_mode, offload, target_gpus)

    engine_out = output_dir / engine_id
    engine_out.mkdir(parents=True, exist_ok=True)

    results_by_size: Dict[str, Any] = {}
    per_size_seconds: Dict[str, float] = {}

    for width, height, size_label in BENCH_SIZES:
        logger.info(">>> Running 5 items for %s at %s (%dx%d)...", engine_id, size_label, width, height)
        size_out = engine_out / size_label
        if size_out.exists():
            shutil.rmtree(size_out)
        size_out.mkdir(parents=True, exist_ok=True)

        pack = create_size_pack(engine_id, width, height, size_label, size_out)
        pack_path = size_out / f"pack_{size_label}.json"
        pack_path.write_text(pack.model_dump_json(indent=2), encoding="utf-8")
        results_path = size_out / "results.json"

        cmd = [
            sys.executable,
            "-m", "worker.runner",
            "--pack", str(pack_path),
            "--out", str(size_out),
            "--results", str(results_path),
            "--parallel-mode", parallel_mode,
            "--offload", offload,
            "--no-resume",
        ]
        if allow_test:
            cmd.append("--test")
        if allow_noncommercial:
            cmd.append("--allow-noncommercial")
        if target_gpus:
            cmd.append("--gpus")
            cmd.extend([str(g) for g in target_gpus])

        clear_gpu_memory()
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=str(_REPO_ROOT), env=os.environ.copy(), capture_output=True, text=True)
        wall_time = time.time() - t0

        if proc.returncode != 0:
            logger.error("[%s @ %s] Subprocess failed with exit code %d. Stderr:\n%s",
                         engine_id, size_label, proc.returncode, proc.stderr[-2000:])
            raise RuntimeError(f"Size benchmark failed for {engine_id} @ {size_label}: {proc.stderr[-1000:]}")

        results = load_results(results_path) if results_path.is_file() else []
        successful = [r for r in results if r.status == "success"]

        if len(successful) == 0:
            raise RuntimeError(f"0 successful items produced for {engine_id} @ {size_label}")

        # Exclude first item (warmup / compilation) to obtain steady seconds per item
        steady_items = successful[1:] if len(successful) > 1 else successful
        steady_times = [r.execution_time_seconds for r in steady_items if r.execution_time_seconds > 0]
        steady_sec = statistics.median(steady_times) if steady_times else 0.0

        # Peak VRAM per GPU
        peak_vram_per_gpu: Dict[str, float] = {}
        for g_idx in target_gpus:
            gpu_res = [r.peak_vram_mb for r in successful if getattr(r, "peak_vram_mb", None) is not None]
            if gpu_res:
                peak_vram_per_gpu[f"gpu_{g_idx}"] = round(max(gpu_res) / 1024.0, 2)
            else:
                peak_vram_per_gpu[f"gpu_{g_idx}"] = 0.0

        logger.info("[RESULT] %s @ %s: steady %.2f s/item | wall: %.1fs | peak VRAM: %s",
                    engine_id, size_label, steady_sec, wall_time, peak_vram_per_gpu)

        results_by_size[size_label] = {
            "width": width,
            "height": height,
            "total_items": len(results),
            "successful_items": len(successful),
            "steady_seconds_per_item": round(steady_sec, 2),
            "wall_time_seconds": round(wall_time, 2),
            "peak_vram_gb_per_gpu": peak_vram_per_gpu,
        }
        per_size_seconds[size_label] = steady_sec

    # Update models.yaml
    update_models_yaml_resolutions(engine_id, per_size_seconds)

    return {
        "engine_id": engine_id,
        "parallel_mode": parallel_mode,
        "offload": offload,
        "gpus": target_gpus,
        "sizes": results_by_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark diffusion engines across 1344x768 and 768x1344.")
    parser.add_argument("--engines", nargs="+", default=["sdxl_lightning", "flux2_klein"],
                        help="Engine IDs to benchmark (default: sdxl_lightning flux2_klein)")
    parser.add_argument("--out", type=Path, default=_REPO_ROOT / "outputs" / "bench_sizes",
                        help="Base output directory for benchmarks")
    parser.add_argument("--gpus", type=int, nargs="*", default=None,
                        help="GPU device IDs to use")
    parser.add_argument("--test", dest="allow_test", action="store_true",
                        help="Allow test_only engines (e.g. fake)")
    parser.add_argument("--allow-noncommercial", action="store_true",
                        help="Allow non-commercial models")

    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {"benchmarks": {}}
    for eng in args.engines:
        try:
            report = run_size_benchmark_for_engine(
                engine_id=eng,
                output_dir=args.out,
                gpus=args.gpus,
                allow_test=args.allow_test,
                allow_noncommercial=args.allow_noncommercial,
            )
            summary["benchmarks"][eng] = report
        except Exception as exc:
            logger.exception("Benchmark failed for engine %s: %s", eng, exc)
            summary["benchmarks"][eng] = {"error": str(exc)}

    out_file = args.out / "sizes_benchmark.json"
    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Saved size benchmark summary to %s", out_file)


if __name__ == "__main__":
    main()
