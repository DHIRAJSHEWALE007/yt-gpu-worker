"""scripts/soak.py — Memory soak stability test for diffusion engines.

Architecture:
  - One worker process per GPU.
  - 100 images per engine at 1344x768.
  - Explicit torch.cuda.empty_cache() and gc.collect() between items.
  - Every 10 images: logs VRAM allocated, VRAM reserved, and host RAM available.
  - Post warm-up drift gate: fails if VRAM allocated or RAM trends upward by >5% after image 10.
  - Writes consolidated soak report to outputs/soak/soak_{engine}.json.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

try:
    import psutil
except ImportError:
    psutil = None

# Ensure yt-gpu-worker root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from worker.memory import clear_gpu_memory, detect_gpus, get_host_ram_info
from worker.schema import Item, Pack, Result, load_results, save_results_atomic

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SOAK_WIDTH = 1344
SOAK_HEIGHT = 768

SOAK_BASE_PROMPTS = [
    "Cinematic establishing wide shot of a futuristic metropolis at sunset with flying vehicles, 8k",
    "Detailed architectural photograph of an ancient stone cathedral interior with light streaming through stained glass",
    "A majestic snow leopard perched atop a misty Himalayan cliffside at dawn, national geographic style",
    "Macro photograph of vibrant turquoise and gold liquid droplets suspended in crystal clear mineral oil",
    "Steampunk mechanical airship soaring over lush green floating archipelago islands in golden morning light",
    "Cyberpunk back-alley ramen bar in neon rain with reflections on wet asphalt and steam rising",
    "Serene Japanese Zen garden with raked white gravel, mossy stones, and scarlet maple leaves",
    "Cosmic nebula vortex with swirling star dust, vibrant violet flares, and newborn celestial bodies",
    "Portrait of an Arctic explorer with frosted beard and vintage fur-lined parka in blizzard",
    "Intricate golden clockwork mechanism with polished brass gears and glowing gemstone pendulum",
]


def create_soak_items(engine_id: str, count: int = 100, seed_offset: int = 7000) -> List[Item]:
    """Generate count unique Item specifications at 1344x768."""
    items = []
    num_templates = len(SOAK_BASE_PROMPTS)
    for i in range(count):
        template = SOAK_BASE_PROMPTS[i % num_templates]
        prompt = f"{template} (soak cycle {i + 1:03d})"
        items.append(
            Item(
                id=f"soak_{engine_id}_{i:03d}",
                engine=engine_id,
                prompt=prompt,
                width=SOAK_WIDTH,
                height=SOAK_HEIGHT,
                seed=seed_offset + i,
                media_type="image",
            )
        )
    return items


def sample_process_and_gpu_memory(gpu_id: int) -> Dict[str, float]:
    """Sample current VRAM allocated/reserved and RAM available."""
    vram_alloc_mb = 0.0
    vram_res_mb = 0.0
    try:
        import torch
        if torch.cuda.is_available():
            # When CUDA_VISIBLE_DEVICES isolates this process to a single GPU, PyTorch device index is 0
            dev_idx = 0 if os.environ.get("CUDA_VISIBLE_DEVICES") is not None else gpu_id
            vram_alloc_mb = round(torch.cuda.memory_allocated(dev_idx) / (1024.0 * 1024.0), 2)
            vram_res_mb = round(torch.cuda.memory_reserved(dev_idx) / (1024.0 * 1024.0), 2)
    except Exception:
        pass

    ram = get_host_ram_info()
    ram_avail_gb = ram.available_ram_gb

    proc_rss_mb = 0.0
    if psutil is not None:
        try:
            proc_rss_mb = round(psutil.Process().memory_info().rss / (1024.0 * 1024.0), 2)
        except Exception:
            pass

    return {
        "vram_allocated_mb": vram_alloc_mb,
        "vram_reserved_mb": vram_res_mb,
        "ram_available_gb": ram_avail_gb,
        "proc_rss_mb": proc_rss_mb,
    }


def analyze_drift(
    checkpoints: List[Dict[str, Any]],
    threshold_percent: float = 5.0,
) -> Dict[str, Any]:
    """Verify that VRAM and host RAM do not drift upward by > threshold_percent after warmup.
    
    Baseline is checkpoint 0 (after image 10 warmup).
    Fails if:
      - VRAM allocated grows by > threshold_percent relative to baseline.
      - Process RSS grows by > threshold_percent relative to baseline.
      - Host available RAM drops by > threshold_percent relative to baseline.
    """
    if len(checkpoints) <= 1:
        return {"passed": True, "baseline": checkpoints[0] if checkpoints else {}, "drift_failures": []}

    baseline = checkpoints[0]
    failures = []

    base_vram = baseline.get("vram_allocated_mb", 0.0)
    base_rss = baseline.get("proc_rss_mb", 0.0)
    base_ram_avail = baseline.get("ram_available_gb", 0.0)

    for cp in checkpoints[1:]:
        img_idx = cp.get("image_count", 0)

        # 1. VRAM allocated drift
        if base_vram > 0.0:
            curr_vram = cp.get("vram_allocated_mb", 0.0)
            growth_pct = ((curr_vram - base_vram) / base_vram) * 100.0
            cp["vram_growth_pct"] = round(growth_pct, 2)
            if growth_pct > threshold_percent:
                failures.append(
                    f"Image {img_idx}: VRAM allocated grew by {growth_pct:.2f}% "
                    f"({base_vram:.1f} MB -> {curr_vram:.1f} MB, threshold: {threshold_percent}%)"
                )

        # 2. Process RSS drift
        if base_rss > 0.0:
            curr_rss = cp.get("proc_rss_mb", 0.0)
            rss_growth_pct = ((curr_rss - base_rss) / base_rss) * 100.0
            cp["rss_growth_pct"] = round(rss_growth_pct, 2)
            if rss_growth_pct > threshold_percent:
                failures.append(
                    f"Image {img_idx}: Process RSS memory grew by {rss_growth_pct:.2f}% "
                    f"({base_rss:.1f} MB -> {curr_rss:.1f} MB, threshold: {threshold_percent}%)"
                )

        # 3. Available Host RAM drop drift
        if base_ram_avail > 0.0:
            curr_avail = cp.get("ram_available_gb", 0.0)
            avail_drop_pct = ((base_ram_avail - curr_avail) / base_ram_avail) * 100.0
            cp["ram_drop_pct"] = round(avail_drop_pct, 2)
            if avail_drop_pct > threshold_percent:
                failures.append(
                    f"Image {img_idx}: Host available RAM dropped by {avail_drop_pct:.2f}% "
                    f"({base_ram_avail:.2f} GB -> {curr_avail:.2f} GB, threshold: {threshold_percent}%)"
                )

    return {
        "passed": len(failures) == 0,
        "baseline": baseline,
        "drift_failures": failures,
    }


def run_worker_process(
    engine_id: str,
    gpu_id: int,
    pack_file: Path,
    out_dir: Path,
    report_file: Path,
    allow_test: bool = False,
    allow_noncommercial: bool = False,
    threshold_percent: float = 5.0,
) -> None:
    """Worker process logic: loads engine on assigned GPU and executes items with memory checkpoints."""
    from worker.engines import get_engine
    from worker.schema import load_pack

    pack = load_pack(pack_file)
    logger.info("[SOAK WORKER GPU %d] Starting soak on engine %s with %d items...",
                gpu_id, engine_id, len(pack.items))

    # Initialize engine
    # When CUDA_VISIBLE_DEVICES isolates this process, active PyTorch device is cuda:0
    target_dev = "cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES") is not None else f"cuda:{gpu_id}"
    try:
        import torch
        if not torch.cuda.is_available() and allow_test:
            target_dev = "cpu"
    except Exception:
        target_dev = "cpu"

    engine = get_engine(engine_id)
    engine.load(device=target_dev)

    results: List[Result] = []
    checkpoints: List[Dict[str, Any]] = []

    for idx, item in enumerate(pack.items):
        t0 = time.time()
        try:
            out_path = engine.generate(item, out_dir)
            exec_time = time.time() - t0
            results.append(
                Result(
                    item_id=item.id,
                    status="success",
                    output_path=str(out_path),
                    engine=item.engine,
                    execution_time_seconds=round(exec_time, 2),
                    width=item.width,
                    height=item.height,
                    seed=item.seed,
                )
            )
        except Exception as exc:
            exec_time = time.time() - t0
            logger.error("[SOAK ERROR] Item %s failed: %s", item.id, exc)
            results.append(
                Result(
                    item_id=item.id,
                    status="failed",
                    error=str(exc),
                    engine=item.engine,
                    execution_time_seconds=round(exec_time, 2),
                )
            )

        # Cleanup between items
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

        # Checkpoint every 10 images
        image_num = idx + 1
        if image_num % 10 == 0:
            sample = sample_process_and_gpu_memory(gpu_id)
            sample["image_count"] = image_num
            checkpoints.append(sample)
            logger.info(
                "[SOAK CHECKPOINT] Engine: %s | GPU: %d | Image: %d/%d | "
                "VRAM Alloc: %.1f MB | VRAM Res: %.1f MB | RAM Avail: %.2f GB",
                engine_id,
                gpu_id,
                image_num,
                len(pack.items),
                sample["vram_allocated_mb"],
                sample["vram_reserved_mb"],
                sample["ram_available_gb"],
            )

    # Post-run drift analysis
    drift_eval = analyze_drift(checkpoints, threshold_percent=threshold_percent)
    worker_report = {
        "engine_id": engine_id,
        "gpu_id": gpu_id,
        "total_items": len(pack.items),
        "successful_items": len([r for r in results if r.status == "success"]),
        "checkpoints": checkpoints,
        "drift_evaluation": drift_eval,
    }

    report_file.write_text(json.dumps(worker_report, indent=2), encoding="utf-8")
    save_results_atomic(results, out_dir / f"results_gpu_{gpu_id}.json")

    if not drift_eval["passed"]:
        logger.error("[SOAK DRIFT FAILURE GPU %d] Failures:\n%s",
                     gpu_id, "\n".join(drift_eval["drift_failures"]))
        sys.exit(2)

    logger.info("[SOAK WORKER GPU %d] Completed cleanly without memory drift.", gpu_id)


def run_soak_test(
    engine_id: str,
    output_dir: Path,
    total_images: int = 100,
    gpus: Optional[List[int]] = None,
    allow_test: bool = False,
    allow_noncommercial: bool = False,
    threshold_percent: float = 5.0,
) -> Dict[str, Any]:
    """Coordinator: spawns one process per GPU, splits 100 images, and aggregates reports."""
    detected = detect_gpus()
    target_gpus = gpus if gpus is not None else ([g.index for g in detected] if detected else [0])
    num_gpus = len(target_gpus)

    logger.info("=== Starting Soak Test for Engine '%s' [Total Images: %d] across %d GPU(s): %s ===",
                engine_id, total_images, num_gpus, target_gpus)

    engine_out = output_dir / engine_id
    if engine_out.exists():
        shutil.rmtree(engine_out)
    engine_out.mkdir(parents=True, exist_ok=True)

    all_items = create_soak_items(engine_id, count=total_images)

    # Partition items across GPUs
    partitions: Dict[int, List[Item]] = {g: [] for g in target_gpus}
    for idx, it in enumerate(all_items):
        assigned_gpu = target_gpus[idx % num_gpus]
        partitions[assigned_gpu].append(it)

    processes = []
    worker_reports: Dict[int, Path] = {}

    t0 = time.time()
    for g_idx in target_gpus:
        gpu_items = partitions[g_idx]
        gpu_out = engine_out / f"gpu_{g_idx}"
        gpu_out.mkdir(parents=True, exist_ok=True)

        pack = Pack(
            pack_id=f"soak_{engine_id}_gpu_{g_idx}",
            items=gpu_items,
            output_dir=str(gpu_out),
            max_runtime_minutes=60.0,
        )
        pack_file = gpu_out / "pack.json"
        pack_file.write_text(pack.model_dump_json(indent=2), encoding="utf-8")

        rep_file = engine_out / f"report_gpu_{g_idx}.json"
        worker_reports[g_idx] = rep_file

        cmd = [
            sys.executable,
            "-m", "scripts.soak",
            "--worker",
            "--engine", engine_id,
            "--gpu", str(g_idx),
            "--pack", str(pack_file),
            "--out", str(gpu_out),
            "--report", str(rep_file),
            "--threshold", str(threshold_percent),
        ]
        if allow_test:
            cmd.append("--test")
        if allow_noncommercial:
            cmd.append("--allow-noncommercial")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(g_idx)
        if allow_test:
            env["STUDIO_FAKE_PROVIDERS"] = "1"

        logger.info("[COORDINATOR] Launching worker process for GPU %d (%d items)...", g_idx, len(gpu_items))
        proc = subprocess.Popen(cmd, cwd=str(_REPO_ROOT), env=env)
        processes.append((g_idx, proc))

    # Wait for all workers to finish
    all_passed = True
    aggregated_results: Dict[str, Any] = {
        "engine_id": engine_id,
        "total_images": total_images,
        "gpus": target_gpus,
        "workers": {},
    }

    for g_idx, proc in processes:
        ret = proc.wait()
        rep_file = worker_reports[g_idx]
        worker_data = {}
        if rep_file.is_file():
            try:
                worker_data = json.loads(rep_file.read_text(encoding="utf-8"))
            except Exception:
                pass

        if ret != 0 or not worker_data.get("drift_evaluation", {}).get("passed", False):
            all_passed = False
            logger.error("[COORDINATOR] Worker for GPU %d reported failure (exit code: %d)", g_idx, ret)

        aggregated_results["workers"][f"gpu_{g_idx}"] = {
            "exit_code": ret,
            "report": worker_data,
        }

    wall_sec = time.time() - t0
    aggregated_results["wall_time_seconds"] = round(wall_sec, 2)
    aggregated_results["overall_passed"] = all_passed

    summary_file = engine_out / f"soak_summary_{engine_id}.json"
    summary_file.write_text(json.dumps(aggregated_results, indent=2), encoding="utf-8")
    logger.info("Saved soak summary to %s", summary_file)

    if not all_passed:
        raise RuntimeError(f"Soak test failed for engine '{engine_id}': memory drift exceeded {threshold_percent}%")

    return aggregated_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Memory soak test: 100 images at 1344x768 per engine.")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--engine", type=str, default="sdxl_lightning", help="Engine to test")
    parser.add_argument("--engines", nargs="+", default=None, help="Multiple engines to test")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index for worker")
    parser.add_argument("--gpus", type=int, nargs="*", default=None, help="Target GPUs for coordinator")
    parser.add_argument("--pack", type=Path, default=None, help="Pack file for worker")
    parser.add_argument("--out", type=Path, default=_REPO_ROOT / "outputs" / "soak", help="Output directory")
    parser.add_argument("--report", type=Path, default=None, help="Report file for worker")
    parser.add_argument("--images", type=int, default=100, help="Total images per engine")
    parser.add_argument("--threshold", type=float, default=5.0, help="Maximum allowed drift percent after warmup")
    parser.add_argument("--test", dest="allow_test", action="store_true", help="Allow test/mock engines")
    parser.add_argument("--allow-noncommercial", action="store_true", help="Allow non-commercial models")

    args = parser.parse_args()

    # Worker dispatch
    if args.worker:
        if not args.pack or not args.report:
            parser.error("--worker requires --pack and --report")
        run_worker_process(
            engine_id=args.engine,
            gpu_id=args.gpu,
            pack_file=args.pack,
            out_dir=args.out,
            report_file=args.report,
            allow_test=args.allow_test,
            allow_noncommercial=args.allow_noncommercial,
            threshold_percent=args.threshold,
        )
        return

    # Coordinator dispatch
    engines = args.engines if args.engines else [args.engine]
    overall_ok = True
    for eng in engines:
        try:
            run_soak_test(
                engine_id=eng,
                output_dir=args.out,
                total_images=args.images,
                gpus=args.gpus,
                allow_test=args.allow_test,
                allow_noncommercial=args.allow_noncommercial,
                threshold_percent=args.threshold,
            )
        except Exception as exc:
            logger.error("Soak test coordinator error for %s: %s", eng, exc)
            overall_ok = False

    if not overall_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
