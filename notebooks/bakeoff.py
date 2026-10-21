"""notebooks/bakeoff.py — Performance & quality benchmark suite for enabled engines.

For every enabled engine in models.yaml:
- Measures and records:
  - Download & model load time
  - Per-item isolated Peak VRAM (reset per item)
  - Median latency per resolution (excluding initial warm-up item)
  - Full prompt text for prompt adherence evaluation
- For video engines:
  - Renders 480p preview clip with latency & isolated VRAM
- Generates:
  - Rendered image/video files
  - Interactive HTML contact sheet (contact_sheet.html) with full prompts
  - Metrics JSON (bakeoff.json) without stub entries for skipped engines
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import statistics
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
    """Generate high-contrast dark-mode HTML contact sheet comparing all engines with prompt text."""
    html_lines = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "  <meta charset='UTF-8'/>",
        "  <meta name='viewport' content='width=device-width, initial-scale=1.0'/>",
        "  <title>Model Bakeoff Comparison Sheet</title>",
        "  <style>",
        "    body { background: #0B0F19; color: #F8FAFC; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; padding: 24px; margin: 0; }",
        "    h1 { color: #38BDF8; font-size: 26px; margin: 0 0 6px 0; font-weight: 700; }",
        "    .timestamp { color: #94A3B8; font-size: 13px; font-family: monospace; margin-bottom: 28px; }",
        "    .engine-section { background: #131B2E; border: 1px solid #1E293B; border-radius: 12px; padding: 24px; margin-bottom: 32px; }",
        "    .engine-header { display: flex; justify-content: space-between; align-items: baseline; border-bottom: 1px solid #1E293B; padding-bottom: 12px; margin-bottom: 16px; flex-wrap: wrap; gap: 8px; }",
        "    .engine-title { color: #38BDF8; font-size: 20px; font-weight: 700; margin: 0; }",
        "    .engine-meta { color: #94A3B8; font-size: 13px; font-family: monospace; }",
        "    .metrics-table { width: 100%; border-collapse: collapse; font-family: monospace; font-size: 13px; margin: 16px 0; }",
        "    .metrics-table th, .metrics-table td { border: 1px solid #334155; padding: 10px 14px; text-align: left; }",
        "    .metrics-table th { background: #1E293B; color: #00F0FF; }",
        "    .metrics-table tr:hover { background: #1E293B40; }",
        "    .gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 20px; margin-top: 20px; }",
        "    .card { background: #070B14; border: 1px solid #1E293B; border-radius: 10px; overflow: hidden; display: flex; flex-direction: column; }",
        "    .card img, .card video { width: 100%; height: auto; display: block; border-bottom: 1px solid #1E293B; background: #000; }",
        "    .card-info { padding: 12px; display: flex; flex-direction: column; gap: 10px; flex-grow: 1; justify-content: space-between; }",
        "    .prompt-box { font-size: 12px; color: #E2E8F0; line-height: 1.45; background: #0F172A; padding: 8px 10px; border-radius: 6px; border-left: 3px solid #38BDF8; }",
        "    .meta-badges { display: flex; flex-wrap: wrap; gap: 6px; font-size: 11px; font-family: monospace; }",
        "    .badge { background: #1E293B; color: #CBD5E1; padding: 3px 8px; border-radius: 4px; border: 1px solid #334155; }",
        "    .badge-warmup { background: #854D0E30; color: #FACC15; border-color: #A16207; }",
        "    .badge-time { background: #065F4630; color: #34D399; border-color: #059669; }",
        "    .badge-vram { background: #312E8130; color: #A5B4FC; border-color: #4338CA; }",
        "  </style>",
        "</head>",
        "<body>",
        "  <h1>yt-gpu-worker: Model Bakeoff Benchmark</h1>",
        f"  <div class='timestamp'>Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC')}</div>",
    ]

    for engine_id, data in results.items():
        html_lines.append("  <div class='engine-section'>")
        html_lines.append("    <div class='engine-header'>")
        html_lines.append(f"      <h2 class='engine-title'>Engine: {engine_id}</h2>")
        html_lines.append(f"      <div class='engine-meta'>Load Time: {data.get('load_time_sec', 0):.2f}s | Failures: {data.get('total_failures', 0)}</div>")
        html_lines.append("    </div>")

        html_lines.append("    <table class='metrics-table'>")
        html_lines.append("      <tr><th>Resolution</th><th>Median Latency (s)</th><th>Warm-up (s)</th><th>Peak VRAM (MB)</th><th>Samples Tested</th></tr>")

        for res_str, res_data in data.get("resolutions", {}).items():
            median_val = res_data.get("median_seconds_per_item", res_data.get("avg_seconds_per_item", 0))
            warmup_val = res_data.get("warmup_seconds", median_val)
            vram_val = res_data.get("peak_vram_mb", 0.0)
            count_val = res_data.get("samples_count", 0)

            html_lines.append(
                f"      <tr><td>{res_str}</td><td><strong>{median_val:.2f}s</strong></td>"
                f"<td>{warmup_val:.2f}s</td><td>{vram_val:.1f} MB</td><td>{count_val}</td></tr>"
            )
        html_lines.append("    </table>")

        html_lines.append("    <div class='gallery'>")
        for sample in data.get("samples", []):
            rel_path = sample.get("file_path", "")
            media_type = sample.get("media_type", "image")
            prompt_text = sample.get("prompt", "Prompt text unavailable")
            is_warmup = sample.get("is_warmup", False)
            lat_sec = sample.get("latency_sec", 0)
            vram_mb = sample.get("peak_vram_mb", 0)
            w = sample.get("width", 0)
            h = sample.get("height", 0)
            seed = sample.get("seed", 0)

            media_tag = (
                f"<video src='{rel_path}' controls></video>"
                if media_type == "video"
                else f"<img src='{rel_path}' alt='Sample' loading='lazy'/>"
            )

            vram_badge = f"<span class='badge badge-vram'>{vram_mb:.0f} MB VRAM</span>" if vram_mb > 0 else ""
            time_badge_class = "badge-warmup" if is_warmup else "badge-time"
            time_label = f"{lat_sec:.2f}s (warm-up)" if is_warmup else f"{lat_sec:.2f}s"

            html_lines.append(f"""      <div class='card'>
        {media_tag}
        <div class='card-info'>
          <div class='prompt-box'>"{prompt_text}"</div>
          <div class='meta-badges'>
            <span class='badge'>{w}x{h}</span>
            <span class='badge'>Seed: {seed}</span>
            <span class='badge {time_badge_class}'>{time_label}</span>
            {vram_badge}
          </div>
        </div>
      </div>""")

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
            # Skip engine entirely if it fails to load; do not write a stub entry
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

            # Reset peak VRAM counter before item generation
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

            t_gen = time.time()
            try:
                out_path = engine.generate(item, out_dir)
                dur = round(time.time() - t_gen, 2)

                item_vram_mb = 0.0
                try:
                    import torch
                    if torch.cuda.is_available():
                        item_vram_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1)
                except Exception:
                    pass

                engine_results["samples"].append({
                    "seed": SEEDS[0],
                    "prompt": PROMPTS[0],
                    "width": res_w,
                    "height": res_h,
                    "latency_sec": dur,
                    "is_warmup": False,
                    "peak_vram_mb": item_vram_mb,
                    "media_type": "video",
                    "file_path": str(out_path.name),
                })
                engine_results["resolutions"]["854x480"] = {
                    "median_seconds_per_item": dur,
                    "warmup_seconds": dur,
                    "samples_count": 1,
                    "peak_vram_mb": item_vram_mb,
                }
            except Exception as e:
                logger.error("Video clip generation failed on %s: %s", engine_id, e)
                engine_results["total_failures"] += 1
        else:
            for w, h in RESOLUTIONS:
                res_key = f"{w}x{h}"
                latencies: List[float] = []
                vram_measurements: List[float] = []

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

                    # Reset peak VRAM counter before each individual item
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.reset_peak_memory_stats()
                    except Exception:
                        pass

                    t_gen = time.time()
                    try:
                        out_path = engine.generate(item, out_dir)
                        dur = round(time.time() - t_gen, 2)

                        item_vram_mb = 0.0
                        try:
                            import torch
                            if torch.cuda.is_available():
                                item_vram_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1)
                        except Exception:
                            pass

                        latencies.append(dur)
                        vram_measurements.append(item_vram_mb)
                        is_warmup = (idx == 0)

                        engine_results["samples"].append({
                            "seed": s,
                            "prompt": p,
                            "width": w,
                            "height": h,
                            "latency_sec": dur,
                            "is_warmup": is_warmup,
                            "peak_vram_mb": item_vram_mb,
                            "media_type": "image",
                            "file_path": str(out_path.name),
                        })
                    except Exception as e:
                        logger.error("Bakeoff sample generation failed on %s (%dx%d): %s", engine_id, w, h, e)
                        engine_results["total_failures"] += 1

                # Calculate median latency excluding the first warm-up item
                if latencies:
                    warmup_val = latencies[0]
                    steady_latencies = latencies[1:] if len(latencies) > 1 else latencies
                    median_sec = round(statistics.median(steady_latencies), 2)
                    peak_vram = max(vram_measurements) if vram_measurements else 0.0

                    engine_results["resolutions"][res_key] = {
                        "median_seconds_per_item": median_sec,
                        "warmup_seconds": warmup_val,
                        "samples_count": len(latencies),
                        "peak_vram_mb": peak_vram,
                    }

        engine.unload()
        del engine
        clear_gpu_memory()

        # Requirement 4: Do not write a stub entry for an engine that produced 0 samples
        if engine_results["samples"]:
            bakeoff_results[engine_id] = engine_results
        else:
            logger.info("Excluding engine '%s' from results: 0 successful samples were rendered.", engine_id)

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
