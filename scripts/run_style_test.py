"""scripts/run_style_test.py — Execute style test pack and generate side-by-side comparison contact sheet.

Executes 8 test scenes (5 channel style suffix + 3 explainer diagrams + no-text variants) across:
  - SDXL-Lightning (role: volume_image)
  - FLUX.2-klein (role: default_image)

Features:
  - No-text variants for technical diagram prompts.
  - Pillow overlay helper that renders clean typographic labels and leader lines on generated backgrounds.
  - Interactive dark-mode contact sheet (style_contact_sheet.html) comparing native text vs Pillow overlay.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

# Ensure yt-gpu-worker root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from worker.runner import run_pack
from worker.schema import Item, Pack, Result, load_pack, load_results

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Predefined typographic callout labels for explainer diagram scenes
DIAGRAM_LABELS_MAP: Dict[int, List[Dict[str, Any]]] = {
    6: [
        {"text": "Volcanic Arc (Magma Eruptions)", "x": 0.22, "y": 0.18, "target": (0.30, 0.35)},
        {"text": "Continental Lithosphere Crust", "x": 0.10, "y": 0.50, "target": (0.22, 0.52)},
        {"text": "Magma Chamber Plume", "x": 0.20, "y": 0.72, "target": (0.32, 0.68)},
        {"text": "Deep Oceanic Subduction Trench", "x": 0.52, "y": 0.28, "target": (0.48, 0.44)},
        {"text": "Descending Oceanic Plate", "x": 0.62, "y": 0.68, "target": (0.55, 0.60)},
    ],
    7: [
        {"text": "Leader Node (Consensus Proposer)", "x": 0.36, "y": 0.16, "target": (0.50, 0.28)},
        {"text": "Raft Quorum Validator A", "x": 0.10, "y": 0.54, "target": (0.25, 0.50)},
        {"text": "Raft Quorum Validator B", "x": 0.65, "y": 0.54, "target": (0.75, 0.50)},
        {"text": "Replicated State Machine Log", "x": 0.32, "y": 0.82, "target": (0.50, 0.74)},
    ],
    8: [
        {"text": "Cornea (Refractive Surface)", "x": 0.10, "y": 0.30, "target": (0.22, 0.45)},
        {"text": "Crystalline Lens (Focusing)", "x": 0.22, "y": 0.16, "target": (0.34, 0.48)},
        {"text": "Focal Optical Rays", "x": 0.44, "y": 0.36, "target": (0.50, 0.50)},
        {"text": "Retinal Fovea Photoreceptors", "x": 0.68, "y": 0.46, "target": (0.76, 0.50)},
        {"text": "Optic Nerve Signal Bundle", "x": 0.68, "y": 0.72, "target": (0.80, 0.62)},
    ],
}


def overlay_diagram_labels(
    image_path: Path,
    output_path: Path,
    labels: List[Dict[str, Any]],
) -> Path:
    """Draw high-contrast typography labels, semi-transparent pill badges, and leader lines on a diagram background using Pillow."""
    if not image_path.is_file():
        raise FileNotFoundError(f"Source image not found: {image_path}")

    img = Image.open(image_path).convert("RGBA")
    width, height = img.size

    # Create transparent overlay layer
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Attempt to load a clean proportional truetype font, falling back to default
    font_size = max(15, int(width * 0.018))
    font = None
    for font_name in ["arial.ttf", "DejaVuSans.ttf", "Helvetica.ttf", "calibri.ttf", "tahoma.ttf"]:
        try:
            font = ImageFont.truetype(font_name, font_size)
            break
        except Exception:
            continue
    if font is None:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

    for label in labels:
        text = str(label.get("text", "")).strip()
        if not text:
            continue

        pos_x = label.get("x", 0.5)
        pos_y = label.get("y", 0.5)
        px = int(pos_x * width) if isinstance(pos_x, float) and 0.0 <= pos_x <= 1.0 else int(pos_x)
        py = int(pos_y * height) if isinstance(pos_y, float) and 0.0 <= pos_y <= 1.0 else int(pos_y)

        # Leader line to target point
        target = label.get("target")
        if target:
            tx_raw, ty_raw = target
            tx = int(tx_raw * width) if isinstance(tx_raw, float) and 0.0 <= tx_raw <= 1.0 else int(tx_raw)
            ty = int(ty_raw * height) if isinstance(ty_raw, float) and 0.0 <= ty_raw <= 1.0 else int(ty_raw)

            line_color = label.get("line_color", (56, 189, 248, 230))  # accent cyan/sky
            draw.line([(px, py), (tx, ty)], fill=line_color, width=3)

            # Draw target locator dot
            dot_r = 5
            draw.ellipse(
                [(tx - dot_r, ty - dot_r), (tx + dot_r, ty + dot_r)],
                fill=(255, 255, 255, 255),
                outline=line_color,
                width=2,
            )

        # Measure text box
        pad_x = 12
        pad_y = 6
        if hasattr(draw, "textbbox") and font:
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
        else:
            tw, th = len(text) * 9, font_size + 4

        box_x0 = px - pad_x
        box_y0 = py - pad_y
        box_x1 = px + tw + pad_x
        box_y1 = py + th + pad_y

        bg_color = label.get("bg_color", (11, 19, 38, 235))      # dark slate pill
        border_color = label.get("border_color", (56, 189, 248, 255)) # neon sky border
        text_color = label.get("text_color", (248, 250, 252, 255))   # pure white text

        # Draw rounded badge
        if hasattr(draw, "rounded_rectangle"):
            draw.rounded_rectangle([box_x0, box_y0, box_x1, box_y1], radius=8, fill=bg_color, outline=border_color, width=2)
        else:
            draw.rectangle([box_x0, box_y0, box_x1, box_y1], fill=bg_color, outline=border_color, width=2)

        # Draw text
        draw.text((px, py), text, font=font, fill=text_color)

    # Composite layers
    final_img = Image.alpha_composite(img, overlay).convert("RGB")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_img.save(output_path, quality=95)
    logger.info("Saved Pillow-annotated diagram to: %s", output_path)
    return output_path


def generate_style_contact_sheet_html(pack: Pack, results: List[Result], output_path: Path) -> None:
    """Generate high-contrast dark-mode HTML contact sheet comparing SDXL-Lightning vs FLUX.2 Klein, including Pillow overlays."""
    results_map = {r.item_id: r for r in results}

    # Group items by base scene number
    scenes: Dict[int, Dict[str, Any]] = {}
    for it in pack.items:
        scene_idx = it.extra_params.get("base_scene") if it.extra_params and "base_scene" in it.extra_params else None
        if scene_idx is None:
            parts = it.id.split("_")
            try:
                scene_idx = int(parts[-1])
            except ValueError:
                scene_idx = 1

        if scene_idx not in scenes:
            scenes[scene_idx] = {
                "idx": scene_idx,
                "prompt": it.prompt,
                "category": it.extra_params.get("category", "channel_style") if it.extra_params else "channel_style",
                "title": it.extra_params.get("title", f"Scene {scene_idx}") if it.extra_params else f"Scene {scene_idx}",
                "sdxl_item": None,
                "sdxl_result": None,
                "flux_item": None,
                "flux_result": None,
                "sdxl_notext_result": None,
                "flux_notext_result": None,
            }

        is_notext = "notext" in it.id
        if "sdxl" in it.id:
            if is_notext:
                scenes[scene_idx]["sdxl_notext_result"] = results_map.get(it.id)
            else:
                scenes[scene_idx]["sdxl_item"] = it
                scenes[scene_idx]["sdxl_result"] = results_map.get(it.id)
        elif "flux" in it.id:
            if is_notext:
                scenes[scene_idx]["flux_notext_result"] = results_map.get(it.id)
            else:
                scenes[scene_idx]["flux_item"] = it
                scenes[scene_idx]["flux_result"] = results_map.get(it.id)

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
        "    .overlay-strip { padding: 16px 24px; background: #0E1626; border-top: 1px solid #1E293B; }",
        "    .overlay-title { font-size: 14px; font-weight: 700; color: #38BDF8; margin-bottom: 12px; display: flex; align-items: center; gap: 8px; }",
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
        "      <div class='legend-tag tag-explainer'>Explainer Diagram: Technical infographics with native AI text vs Pillow overlays</div>",
        "    </div>",
    ]

    for idx in sorted(scenes.keys()):
        sc = scenes[idx]
        is_explainer = (sc["category"] == "explainer_diagram" or idx in (6, 7, 8))
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

        # Explainer Diagram Extension: Show Pillow Typography Overlay Strip
        sdxl_annotated_name = f"style_sdxl_{idx:02d}_notext_annotated.png"
        flux_annotated_name = f"style_flux_{idx:02d}_notext_annotated.png"
        parent_dir = output_path.parent
        has_sdxl_ann = (parent_dir / sdxl_annotated_name).is_file()
        has_flux_ann = (parent_dir / flux_annotated_name).is_file()

        if is_explainer and (has_sdxl_ann or has_flux_ann):
            html_lines.append("      <div class='overlay-strip'>")
            html_lines.append("        <div class='overlay-title'>⚡ Clean No-Text Base + Pillow Typography Overlay (Guaranteed Legibility)</div>")
            html_lines.append("        <div class='comparison-grid' style='padding: 0;'>")

            # SDXL Annotated
            html_lines.append("          <div class='model-col'>")
            html_lines.append("            <div class='model-header'><span class='model-name-sdxl'>SDXL-Lightning + Pillow Callouts</span><span>Overlay: Active</span></div>")
            html_lines.append("            <div class='model-media-wrap'>")
            if has_sdxl_ann:
                html_lines.append(f"              <img src='{sdxl_annotated_name}' alt='SDXL Annotated' loading='lazy'/>")
            else:
                html_lines.append("              <div style='color: #94A3B8; padding: 20px;'>No-text base pending</div>")
            html_lines.append("            </div>")
            html_lines.append("          </div>")

            # FLUX Annotated
            html_lines.append("          <div class='model-col'>")
            html_lines.append("            <div class='model-header'><span class='model-name-flux'>FLUX.2-klein + Pillow Callouts</span><span>Overlay: Active</span></div>")
            html_lines.append("            <div class='model-media-wrap'>")
            if has_flux_ann:
                html_lines.append(f"              <img src='{flux_annotated_name}' alt='FLUX Annotated' loading='lazy'/>")
            else:
                html_lines.append("              <div style='color: #94A3B8; padding: 20px;'>No-text base pending</div>")
            html_lines.append("            </div>")
            html_lines.append("          </div>")

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
    engines: Optional[List[str]] = None,
) -> None:
    """Execute style test pack across both engines, apply Pillow typography overlays, and output contact sheet."""
    default_pack = _REPO_ROOT / "packs" / "style_test_pack.json"
    p_path = pack_path or default_pack
    if not p_path.is_file():
        raise FileNotFoundError(f"Style test pack not found at {p_path}")

    pack = load_pack(p_path)
    if engines:
        pack.items = [
            it for it in pack.items
            if it.engine in engines or any(eng in it.id for eng in engines)
        ]
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

    # Automatically generate Pillow typographic overlays for no-text diagram items
    results_map = {r.item_id: r for r in results}
    for item in pack.items:
        if "notext" in item.id and item.id in results_map:
            res = results_map[item.id]
            if res.status == "success" and res.output_path and Path(res.output_path).is_file():
                scene_num = item.extra_params.get("base_scene", 6)
                labels = DIAGRAM_LABELS_MAP.get(scene_num, [])
                if labels:
                    annotated_path = out_dir / f"{item.id}_annotated.png"
                    try:
                        overlay_diagram_labels(Path(res.output_path), annotated_path, labels)
                    except Exception as err:
                        logger.warning("Could not apply diagram overlay for %s: %s", item.id, err)

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
    parser.add_argument("--engines", nargs="*", default=None, help="Filter engines (e.g. sdxl_lightning flux2_klein)")
    args = parser.parse_args()

    pack_p = Path(args.pack) if args.pack else None
    out_p = Path(args.out) if args.out else None

    run_style_benchmark(
        pack_path=pack_p,
        output_dir=out_p,
        allow_test=args.test,
        allow_noncommercial=args.allow_noncommercial,
        gpus=args.gpus,
        engines=args.engines,
    )


if __name__ == "__main__":
    main()