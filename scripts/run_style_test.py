"""scripts/run_style_test.py — Execute style test pack and generate side-by-side comparison contact sheet.

Executes 8 test scenes (5 channel style suffix + 3 explainer diagrams) across:
  - SDXL-Lightning (role: volume_image)
  - FLUX.2-klein (role: default_image)

Generates:
  - Rendered image outputs
  - High-contrast interactive side-by-side comparison contact sheet (style_contact_sheet.html)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

# Ensure yt-gpu-worker root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from worker.runner import run_pack
from worker.schema import Item, Pack, Result, load_pack, load_results

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_style_contact_sheet_html(pack: Pack, results: List[Result], output_path: Path) -> None:
    """Generate high-contrast dark-mode HTML contact sheet comparing SDXL-Lightning vs FLUX.2 Klein."""
    results_map = {r.item_id: r for r in results}

    # Group items by scene index (01 to 08)
    scenes: Dict[int, Dict[str, Any]] = {}
    for it in pack.items:
        # Extract scene index from id e.g. style_sdxl_01 -> 1
        parts = it.id.split("_")
        idx = int(parts[-1])
        if idx not in scenes:
            scenes[idx] = {
                "idx": idx,
                "prompt": it.prompt,
                "category": it.extra_params.get("category", "channel_style") if it.extra_params else "channel_style",
                "title": it.extra_params.get("title", f"Scene {idx}") if it.extra_params else f"Scene {idx}",
                "sdxl_item": None,
                "sdxl_result": None,
                "flux_item": None,
                "flux_result": None,
            }
        if "sdxl" in it.id:
            scenes[idx]["sdxl_item"] = it
            scenes[idx]["sdxl_result"] = results_map.get(it.id)
        elif "flux" in it.id:
            scenes[idx]["flux_item"] = it
            scenes[idx]["flux_result"] = results_map.get(it.id)

    html_lines = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "  <meta charset='UTF-8'/>",
        "  <meta name='viewport' content='width=device-width, initial-scale=1.0'/>",
        "  <title>Style & Explainer Benchmark: SDXL-Lightning vs FLUX.2 Klein</title>",
        "  <style>",
        "    :root { --bg: #090D16; --card-bg: #111827; --border: #1F2937; --text: #F3F4F6; --accent-blue: #38BDF8; --accent-orange: #FB923C; --accent-teal: #2DD4BF; }",
        "    body { background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; padding: 28px; margin: 0; line-height: 1.5; }",
        "    .container { max-width: 1400px; margin: 0 auto; }",
        "    h1 { color: #F8FAFC; font-size: 28px; margin: 0 0 6px 0; font-weight: 800; letter-spacing: -0.02em; }",
        "    .subtitle { color: #94A3B8; font-size: 14px; margin-bottom: 24px; font-family: monospace; }",
        "    .legend { display: flex; gap: 16px; margin-bottom: 28px; flex-wrap: wrap; }",
        "    .legend-tag { padding: 6px 14px; border-radius: 6px; font-size: 13px; font-weight: 600; display: flex; align-items: center; gap: 8px; }",
        "    .tag-channel { background: #0D3331; color: #2DD4BF; border: 1px solid #14B8A6; }",
        "    .tag-explainer { background: #311B0B; color: #FB923C; border: 1px solid #F97316; }",
        "    .scene-card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 14px; margin-bottom: 32px; overflow: hidden; }",
        "    .scene-header { padding: 18px 24px; background: #162032; border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; flex-wrap: wrap; }",
        "    .scene-title-group { display: flex; align-items: center; gap: 12px; }",
        "    .scene-num { background: #1E293B; color: #38BDF8; font-family: monospace; font-weight: 700; padding: 4px 10px; border-radius: 6px; font-size: 14px; border: 1px solid #334155; }",
        "    .scene-title { color: #F8FAFC; font-size: 18px; font-weight: 700; margin: 0; }",
        "    .scene-prompt { width: 100%; margin-top: 10px; font-size: 13px; color: #CBD5E1; background: #0B0F19; padding: 12px 14px; border-radius: 8px; border-left: 4px solid var(--accent-blue); }",
        "    .comparison-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; padding: 24px; }",
        "    @media (max-width: 900px) { .comparison-grid { grid-template-columns: 1fr; } }",
        "    .model-col { background: #0A0E17; border: 1px solid #1E293B; border-radius: 10px; overflow: hidden; display: flex; flex-direction: column; }",
        "    .model-header { padding: 10px 16px; background: #131C2E; border-bottom: 1px solid #1E293B; display: flex; justify-content: space-between; align-items: center; font-size: 13px; font-family: monospace; }",
        "    .model-name-sdxl { color: #F59E0B; font-weight: 700; }",
        "    .model-name-flux { color: #38BDF8; font-weight: 700; }",
        "    .model-media-wrap { width: 100%; background: #000; display: flex; align-items: center; justify-content: center; min-height: 380px; }",
        "    .model-media-wrap img { width: 100%; height: auto; display: block; }",
        "    .model-footer { padding: 12px 16px; font-size: 12px; font-family: monospace; color: #94A3B8; display: flex; justify-content: space-between; background: #0D1424; }",
        "    .badge-ok { color: #34D399; font-weight: 600; }",
        "    .badge-err { color: #F87171; font-weight: 600; }",
        "  </style>",
        "</head>",
        "<body>",
        "  <div class='container'>",
        "    <h1>yt-gpu-worker: Style & Explainer Quality Benchmark</h1>",
        f"    <div class='subtitle'>Evaluated: {time.strftime('%Y-%m-%d %H:%M:%S UTC')} | SDXL-Lightning (Volume Image) vs FLUX.2 Klein (Default Image)</div>",
        "    <div class='legend'>",
        "      <div class='legend-tag tag-channel'>Channel Style: Flat vector illustration, teal & orange palette, no text</div>",
        "      <div class='legend-tag tag-explainer'>Explainer Diagram: Technical infographics & cross-sections</div>",
        "    </div>",
    ]

    for idx in sorted(scenes.keys()):
        sc = scenes[idx]
        is_explainer = sc["category"] == "explainer_diagram"
        tag_class = "tag-explainer" if is_explainer else "tag-channel"
        tag_label = "Explainer Technical Diagram" if is_explainer else "Channel Style (Teal & Orange)"

        html_lines.append("    <div class='scene-card'>")
        html_lines.append("      <div class='scene-header'>")
        html_lines.append("        <div class='scene-title-group'>")
        html_lines.append(f"          <span class='scene-num'>#{idx:02d}</span>")
        html_lines.append(f"          <h3 class='scene-title'>{sc['title']}</h3>")
        html_lines.append(f"          <span class='legend-tag {tag_class}' style='padding: 2px 8px; font-size: 11px;'>{tag_label}</span>")
        html_lines.append("        </div>")
        html_lines.append(f"        <div class='scene-prompt'><strong>Prompt:</strong> {sc['prompt']}</div>")
        html_lines.append("      </div>")

        html_lines.append("      <div class='comparison-grid'>")

        # Column 1: SDXL-Lightning
        sdxl_res: Optional[Result] = sc["sdxl_result"]
        html_lines.append("        <div class='model-col'>")
        html_lines.append("          <div class='model-header'>")
        html_lines.append("            <span class='model-name-sdxl'>SDXL-Lightning (volume_image)</span>")
        dur_sdxl = f"{sdxl_res.execution_time_seconds:.2f}s" if (sdxl_res and sdxl_res.execution_time_seconds) else "—"
        html_lines.append(f"            <span>Latency: {dur_sdxl}</span>")
        html_lines.append("          </div>")
        html_lines.append("          <div class='model-media-wrap'>")
        if sdxl_res and sdxl_res.output_path and Path(sdxl_res.output_path).is_file():
            img_rel = Path(sdxl_res.output_path).name
            html_lines.append(f"            <img src='{img_rel}' alt='SDXL {sc['title']}' loading='lazy'/>")
        else:
            err_msg = sdxl_res.error if sdxl_res else "Pending / Missing"
            html_lines.append(f"            <div style='color: #EF4444; padding: 24px; font-size: 13px;'>Generation Failed: {err_msg}</div>")
        html_lines.append("          </div>")
        status_sdxl = "<span class='badge-ok'>[SUCCESS]</span>" if (sdxl_res and sdxl_res.status == "success") else "<span class='badge-err'>[FAILED]</span>"
        seed_sdxl = sdxl_res.seed if sdxl_res else "—"
        html_lines.append(f"          <div class='model-footer'><span>Status: {status_sdxl}</span><span>Seed: {seed_sdxl}</span><span>Steps: 4</span></div>")
        html_lines.append("        </div>")

        # Column 2: FLUX.2 Klein
        flux_res: Optional[Result] = sc["flux_result"]
        html_lines.append("        <div class='model-col'>")
        html_lines.append("          <div class='model-header'>")
        html_lines.append("            <span class='model-name-flux'>FLUX.2-klein-4B (default_image)</span>")
        dur_flux = f"{flux_res.execution_time_seconds:.2f}s" if (flux_res and flux_res.execution_time_seconds) else "—"
        html_lines.append(f"            <span>Latency: {dur_flux}</span>")
        html_lines.append("          </div>")
        html_lines.append("          <div class='model-media-wrap'>")
        if flux_res and flux_res.output_path and Path(flux_res.output_path).is_file():
            img_rel = Path(flux_res.output_path).name
            html_lines.append(f"            <img src='{img_rel}' alt='FLUX {sc['title']}' loading='lazy'/>")
        else:
            err_msg = flux_res.error if flux_res else "Pending / Missing"
            html_lines.append(f"            <div style='color: #EF4444; padding: 24px; font-size: 13px;'>Generation Failed: {err_msg}</div>")
        html_lines.append("          </div>")
        status_flux = "<span class='badge-ok'>[SUCCESS]</span>" if (flux_res and flux_res.status == "success") else "<span class='badge-err'>[FAILED]</span>"
        seed_flux = flux_res.seed if flux_res else "—"
        html_lines.append(f"          <div class='model-footer'><span>Status: {status_flux}</span><span>Seed: {seed_flux}</span><span>Steps: 4</span></div>")
        html_lines.append("        </div>")

        html_lines.append("      </div>")
        html_lines.append("    </div>")

    html_lines.extend([
        "  </div>",
        "</body>",
        "</html>",
    ])

    output_path.write_text("\n".join(html_lines), encoding="utf-8")
    logger.info("Saved style comparison contact sheet to: %s", output_path.resolve())


def run_style_benchmark(
    pack_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    allow_test: bool = False,
    allow_noncommercial: bool = False,
    gpus: Optional[List[int]] = None,
) -> None:
    """Execute style test pack across both engines and output contact sheet."""
    default_pack = _REPO_ROOT / "packs" / "style_test_pack.json"
    p_path = pack_path or default_pack
    if not p_path.is_file():
        raise FileNotFoundError(f"Style test pack not found at {p_path}")

    pack = load_pack(p_path)
    out_dir = output_dir or (_REPO_ROOT / "outputs" / "style_test")
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Executing style test pack (%d items) to %s...", len(pack.items), out_dir)

    results = run_pack(
        pack,
        output_dir=out_dir,
        gpu_indices=gpus,
        allow_test_engines=allow_test,
        allow_noncommercial=allow_noncommercial,
    )

    contact_sheet_path = out_dir / "style_contact_sheet.html"
    generate_style_contact_sheet_html(pack, results, contact_sheet_path)

    print(f"\nStyle test complete! Results saved to:")
    print(f"  Output folder:  {out_dir.resolve()}")
    print(f"  Contact Sheet:  {contact_sheet_path.resolve()}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="yt-gpu-worker: Run Style & Explainer Quality Benchmark")
    parser.add_argument("--pack", type=str, default=None, help="Path to style_test_pack.json")
    parser.add_argument("--out", type=str, default="./outputs/style_test", help="Output directory")
    parser.add_argument("--gpus", type=int, nargs="*", default=None, help="GPU device indices")
    parser.add_argument("--test", action="store_true", help="Allow test_only mock engines")
    parser.add_argument("--allow-noncommercial", action="store_true", help="Allow non-commercial engines")
    args = parser.parse_args()

    pack_file = Path(args.pack) if args.pack else None
    out_dir = Path(args.out)

    run_style_benchmark(
        pack_path=pack_file,
        output_dir=out_dir,
        allow_test=args.test,
        allow_noncommercial=args.allow_noncommercial,
        gpus=args.gpus,
    )


if __name__ == "__main__":
    main()
