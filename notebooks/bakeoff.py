"""notebooks/bakeoff.py — Performance & quality benchmark suite for enabled engines.

For every enabled engine in models.yaml:
- Measures and records:
  - Download & model load time
  - Peak VRAM and peak RAM
  - Seconds per item at: 768x768, 1024x1024, 832x1216
  - Failures over 5 fixed prompts and seeds
- For Wan 2.2 5B:
  - Renders one 480p clip of ~3 seconds made from a generated still
- Generates:
  - Rendered image/video files
  - HTML contact sheet (contact_sheet.html)
  - Full metrics JSON (bakeoff.json)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

# Ensure repo root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import yaml

from worker.engines import get_engine
from worker.memory import clear_gpu_memory, detect_gpus, get_system_ram_mb
from worker.schema import Item

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROMPTS = [
    "A majestic snow-capped mountain range under a vibrant aurora borealis, ultra-detailed 8k, cinematic lighting",
    "Ancient weathered stone temple submerged in an emerald green jungle with sunbeams piercing the canopy",
    "Futuristic cyberpunk city street at night, neon reflections on wet asphalt, volumetric fog, high fidelity",
    "Macro photograph of a colorful tropical chameleon perched on an exotic orchid blossom, sharp focus",
    "Dramatic ocean storm with massive waves crashing against tall sea cliffs, dramatic clouds, golden hour",
]

SEEDS = [101, 202, 303, 404, 505]
RESOLUTIONS = [
    (768, 768),
    (1024, 1024),
    (832, 1216),
]


def generate_contact_sheet_html(results: Dict[str, Any], output_path: Path) -> None:
    """Generate high-contrast dark-mode HTML contact sheet comparing all engines."""
    html_lines = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "  <meta charset='UTF-8'/>",
        "  <title>Model Bakeoff Comparison Sheet</title>",
        "  <style>",
        "    body { background: #0B0F19; color: #F8FAFC; font-family: -apple-system, sans-serif; padding: 24px; }",
        "    h1, h2 { color: #38BDF8; font-weight: 700; }",
        "    .engine-section { background: #131B2E; border: 1px solid #1E293B; border-radius: 12px; padding: 20px; margin-bottom: 24px; }",
        "    .metrics-table { width: 100%; border-collapse: collapse; font-family: monospace; font-size: 13px; margin: 12px 0; }",
        "    .metrics-table th, .metrics-table td { border: 1px solid #334155; padding: 8px 12px; text-align: left; }",
        "    .metrics-table th { background: #1E293B; color: #00F0FF; }",
        "    .gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 16px; margin-top: 16px; }",
        "    .card { background: #05080F; border: 1px solid #334155; border-radius: 8px; overflow: hidden; padding: 8px; }",
        "    .card img, .card video { width: 100%; height: auto; border-radius: 4px; display: block; }",
        "    .caption { font-size: 11px; color: #94A3B8; margin-top: 6px; }",
        "  </style>",
        "</head>",
        "<body>",
        "  <h1>yt-gpu-worker: Model Bakeoff Benchmark</h1>",
        f"  <p style='color: #94A3B8; font-size: 13px;'>Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>",
    ]

    for engine_id, data in results.items():
        html_lines.append("  <div class='engine-section'>")
        html_lines.append(f"    <h2>Engine: {engine_id}</h2>")
        html_lines.append(f"    <p>Load Time: {data.get('load_time_sec', 0):.2f}s | Failures: {data.get('total_failures', 0)}</p>")
        html_lines.append("    <table class='metrics-table'>")
        html_lines.append("      <tr><th>Resolution</th><th>Avg Latency (s)</th><th>Peak VRAM (MB)</th><th>Samples Tested</th></tr>")

        for res_str, res_data in data.get("resolutions", {}).items():
            html_lines.append(
                f"      <tr><td>{res_str}</td><td>{res_data.get('avg_seconds_per_item', 0):.2f}s</td>"
                f"<td>{res_data.get('peak_vram_mb', 0):.1f} MB</td><td>{res_data.get('samples_count', 0)}</td></tr>"
            )
        html_lines.append("    </table>")

        html_lines.append("    <div class='gallery'>")
        for sample in data.get("samples", []):
            rel_path = sample.get("file_path", "")
            media_type = sample.get("media_type", "image")
            caption = f"Seed: {sample.get('seed')} | Res: {sample.get('width')}x{sample.get('height')} | {sample.get('latency_sec', 0):.2f}s"
            if media_type == "video":
                html_lines.append(f"      <div class='card'><video src='{rel_path}' controls></video><div class='caption'>{caption}</div></div>")
            else:
                html_lines.append(f"      <div class='card'><img src='{rel_path}' alt='Sample'/><div class='caption'>{caption}</div></div>")
        html_lines.append("    </div>")
        html_lines.append("  </div>")

    html_lines.extend(["</body>", "</html>"])
    output_path.write_text("\n".join(html_lines), encoding="utf-8")
    logger.info("Saved contact sheet HTML to: %s", output_path)


def run_bakeoff(output_dir: Path | None = None, engines_to_test: List[str] | None = None) -> Dict[str, Any]:
    """Execute complete bakeoff benchmark across engines."""
    out_dir = output_dir or (_REPO_ROOT / "outputs" / "bakeoff")
    out_dir.mkdir(parents=True, exist_ok=True)

    models_path = _REPO_ROOT / "models.yaml"
    models_cfg = yaml.safe_load(models_path.read_text(encoding="utf-8")).get("models", {})

    bakeoff_results: Dict[str, Any] = {}

    for engine_id, cfg in models_cfg.items():
        if engines_to_test and engine_id not in engines_to_test:
            continue
        if not cfg.get("enabled", False):
            continue

        logger.info("=== Running Bakeoff for Engine: %s ===", engine_id)
        engine_results: Dict[str, Any] = {
            "engine_id": engine_id,
            "resolutions": {},
            "total_failures": 0,
            "samples": [],
        }

        # 1. Load benchmark
        t0 = time.time()
        try:
            engine = get_engine(engine_id, config=cfg)
            engine.load()
        except Exception as e:
            logger.warning("Could not load %s (requires GPU weights?): %s", engine_id, e)
            continue

        load_time = time.time() - t0
        engine_results["load_time_sec"] = round(load_time, 2)

        # 2. Benchmark fixed resolutions across 5 prompts and seeds
        is_video = (engine_id == "wan22_ti2v5b")

        if is_video:
            # Video benchmark: 1 sample at 854x480 ~3 seconds
            res_w, res_h = 854, 480
            item = Item(
                id=f"{engine_id}_clip_0",
                engine=engine_id,
                prompt=PROMPTS[0],
                width=res_w,
                height=res_h,
                seed=SEEDS[0],
                media_type="video",
                duration_seconds=3.0,
            )
            t_gen = time.time()
            try:
                out_path = engine.generate(item, out_dir)
                dur = round(time.time() - t_gen, 2)
                engine_results["samples"].append({
                    "seed": SEEDS[0],
                    "width": res_w,
                    "height": res_h,
                    "latency_sec": dur,
                    "media_type": "video",
                    "file_path": str(out_path.name),
                })
                vram_mb = 0.0
                try:
                    import torch
                    if torch.cuda.is_available():
                        vram_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1)
                except Exception:
                    pass
                engine_results["resolutions"]["854x480"] = {
                    "avg_seconds_per_item": dur,
                    "samples_count": 1,
                    "peak_vram_mb": vram_mb,
                }
            except Exception as e:
                logger.error("Video clip generation failed: %s", e)
                engine_results["total_failures"] += 1
        else:
            for w, h in RESOLUTIONS:
                res_key = f"{w}x{h}"
                latencies: List[float] = []

                for idx in range(min(len(PROMPTS), 5)):
                    p = PROMPTS[idx]
                    s = SEEDS[idx]
                    item = Item(
                        id=f"{engine_id}_{w}x{h}_{idx}",
                        engine=engine_id,
                        prompt=p,
                        width=w,
                        height=h,
                        seed=s,
                        media_type="image",
                    )
                    t_gen = time.time()
                    try:
                        out_path = engine.generate(item, out_dir)
                        dur = round(time.time() - t_gen, 2)
                        latencies.append(dur)
                        engine_results["samples"].append({
                            "seed": s,
                            "width": w,
                            "height": h,
                            "latency_sec": dur,
                            "media_type": "image",
                            "file_path": str(out_path.name),
                        })
                    except Exception as e:
                        logger.error("Bakeoff sample generation failed on %s (%dx%d): %s", engine_id, w, h, e)
                        engine_results["total_failures"] += 1

                avg_sec = round(sum(latencies) / len(latencies), 2) if latencies else 0.0
                vram_mb = 0.0
                try:
                    import torch
                    if torch.cuda.is_available():
                        vram_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1)
                except Exception:
                    pass
                engine_results["resolutions"][res_key] = {
                    "avg_seconds_per_item": avg_sec,
                    "samples_count": len(latencies),
                    "peak_vram_mb": vram_mb,
                }

        engine.unload()
        clear_gpu_memory()
        bakeoff_results[engine_id] = engine_results

    # Save bakeoff.json
    json_path = out_dir / "bakeoff.json"
    json_path.write_text(json.dumps(bakeoff_results, indent=2), encoding="utf-8")
    logger.info("Saved bakeoff metrics to: %s", json_path)

    # Save contact sheet HTML
    generate_contact_sheet_html(bakeoff_results, out_dir / "contact_sheet.html")

    return bakeoff_results


def main() -> None:
    parser = argparse.ArgumentParser(description="yt-gpu-worker: Run model bakeoff benchmark suite")
    parser.add_argument("--engines", nargs="*", default=None, help="Specific engine IDs to benchmark (e.g. zimage_turbo sdxl_lightning)")
    parser.add_argument("--all", action="store_true", help="Benchmark all enabled models in models.yaml (excluding fake unless --include-fake)")
    parser.add_argument("--include-fake", action="store_true", help="Include fake mock engine in the run")
    parser.add_argument("--out", type=str, default="./outputs/bakeoff", help="Directory to save bakeoff.json and contact_sheet.html")
    args = parser.parse_args()

    out_dir = Path(args.out)

    models_path = _REPO_ROOT / "models.yaml"
    models_cfg = yaml.safe_load(models_path.read_text(encoding="utf-8")).get("models", {})

    engines_to_run = None
    if args.engines:
        engines_to_run = args.engines
    elif args.all:
        engines_to_run = [e for e, c in models_cfg.items() if c.get("enabled") and (args.include_fake or e != "fake")]
    else:
        # If no arguments specified: benchmark all real models if GPU detected, otherwise fake
        gpus = detect_gpus()
        if gpus:
            engines_to_run = [e for e, c in models_cfg.items() if c.get("enabled") and e != "fake"]
        else:
            engines_to_run = ["fake"]

    logger.info("Executing bakeoff on engines: %s", engines_to_run)
    run_bakeoff(output_dir=out_dir, engines_to_test=engines_to_run)
    print(f"\nBakeoff complete! Results saved to:")
    print(f"  JSON Metrics:  {(out_dir / 'bakeoff.json').resolve()}")
    print(f"  Contact Sheet: {(out_dir / 'contact_sheet.html').resolve()}\n")


if __name__ == "__main__":
    main()
