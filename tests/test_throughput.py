"""tests/test_throughput.py — Verification of throughput harness and style test pack."""

import json
from pathlib import Path
import pytest

from scripts.bench_throughput import (
    BENCHMARK_PROMPTS,
    create_20_item_pack,
    run_throughput_benchmark,
    update_models_yaml_parallel_mode,
)
from scripts.run_style_test import generate_style_contact_sheet_html
from worker.runner import run_pack
from worker.schema import Item, Pack, Result, load_pack


def test_create_20_item_pack(tmp_path: Path) -> None:
    """Verify 20-item pack is properly structured with unique prompts, seeds, and dimensions."""
    pack = create_20_item_pack("sdxl_lightning", tmp_path, width=1024, height=1024)
    assert len(pack.items) == 20
    assert len(BENCHMARK_PROMPTS) == 20
    assert pack.items[0].id == "sdxl_lightning_tp_00"
    assert pack.items[19].id == "sdxl_lightning_tp_19"
    assert pack.items[0].width == 1024
    assert pack.items[0].height == 1024
    assert pack.items[0].seed == 2000
    assert pack.items[19].seed == 2019


def test_throughput_benchmark_execution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify throughput harness runs Mode A & Mode B, logs RAM/time/OOM, and picks faster mode."""
    monkeypatch.setenv("STUDIO_FAKE_PROVIDERS", "1")
    recorded_updates = {}
    monkeypatch.setattr(
        "scripts.bench_throughput.update_models_yaml_parallel_mode",
        lambda e, m, *args, **kwargs: recorded_updates.update({e: (m, kwargs.get("chosen_offload", "auto"))}),
    )

    out_dir = tmp_path / "throughput_fake"
    report = run_throughput_benchmark(
        "fake",
        output_dir=out_dir,
        gpus=[0, 1],
        allow_test=True,
    )

    assert report["engine_id"] == "fake"
    assert report["item_count"] == 20
    assert "single_process_multi_gpu" in report["modes"]
    assert "process_per_gpu" in report["modes"]
    assert "process_per_gpu_no_offload" in report["modes"]

    mode_a = report["modes"]["single_process_multi_gpu"]
    mode_b = report["modes"]["process_per_gpu"]
    mode_c = report["modes"]["process_per_gpu_no_offload"]

    assert mode_a["successful_items"] == 20
    assert mode_b["successful_items"] == 20
    assert mode_c["successful_items"] == 20
    assert mode_a["failed_items"] == 0
    assert mode_b["failed_items"] == 0
    assert mode_c["failed_items"] == 0
    assert mode_a["oom_occurred"] is False
    assert mode_b["oom_occurred"] is False
    assert mode_c["oom_occurred"] is False
    assert mode_a["total_time_seconds"] > 0
    assert mode_b["total_time_seconds"] > 0
    assert mode_c["total_time_seconds"] > 0
    assert mode_a["items_per_second"] > 0
    assert mode_b["items_per_second"] > 0
    assert mode_c["items_per_second"] > 0
    assert "available_host_ram_gb" in mode_c
    assert "steady_state_seconds_per_item_per_gpu" in mode_c

    assert report["faster_mode"] in ("single_process_multi_gpu", "process_per_gpu")
    assert "faster_offload" in report
    assert "selection_reason" in report

    # Verify JSON report was written to disk
    json_path = out_dir / "throughput_benchmark.json"
    assert json_path.is_file()
    saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved["engine_id"] == "fake"
    assert "fake" in recorded_updates


def test_runner_parallel_mode_selection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify run_pack explicitly executes via specified parallel_mode."""
    monkeypatch.setenv("STUDIO_FAKE_PROVIDERS", "1")

    items = [
        Item(id=f"pm_test_{i}", engine="fake", prompt=f"Test prompt {i}", width=64, height=64)
        for i in range(4)
    ]
    pack = Pack(pack_id="pack_parallel_mode_test", items=items, output_dir=str(tmp_path / "pm_out"))

    # 1. Test single_process_multi_gpu mode
    res_a = run_pack(
        pack,
        output_dir=tmp_path / "out_single",
        gpu_indices=[0, 1],
        allow_test_engines=True,
        parallel_mode="single_process_multi_gpu",
    )
    assert len(res_a) == 4
    assert all(r.status == "success" for r in res_a)

    # 2. Test process_per_gpu mode
    res_b = run_pack(
        pack,
        output_dir=tmp_path / "out_multi",
        gpu_indices=[0, 1],
        allow_test_engines=True,
        parallel_mode="process_per_gpu",
    )
    assert len(res_b) == 4
    assert all(r.status == "success" for r in res_b)


def test_style_test_pack_and_contact_sheet(tmp_path: Path) -> None:
    """Verify style_test_pack.json structure and HTML contact sheet rendering."""
    repo_root = Path(__file__).resolve().parent.parent
    pack_path = repo_root / "packs" / "style_test_pack.json"
    assert pack_path.is_file()

    pack = load_pack(pack_path)
    assert len(pack.items) == 22  # 11 SDXL-Lightning + 11 FLUX.2-klein (including no-text diagram variants)

    sdxl_items = [it for it in pack.items if it.engine == "sdxl_lightning"]
    flux_items = [it for it in pack.items if it.engine == "flux2_klein"]
    assert len(sdxl_items) == 11
    assert len(flux_items) == 11

    # Verify 5 channel-style, 3 explainer prompts, and 3 no-text explainer variants
    channel_items = [it for it in sdxl_items if it.extra_params.get("category") == "channel_style"]
    explainer_items = [it for it in sdxl_items if it.extra_params.get("category") in ("explainer_visual", "explainer_diagram")]
    notext_items = [it for it in sdxl_items if it.extra_params.get("category") in ("explainer_visual_notext", "explainer_diagram_notext")]
    assert len(channel_items) == 5
    assert len(explainer_items) == 3
    assert len(notext_items) == 3

    # Verify channel style suffix is present in all channel style prompts
    suffix = "flat vector illustration, consistent teal and orange palette, no text, no logos"
    for it in channel_items:
        assert suffix in it.prompt

    # Mock results to test HTML contact sheet generation
    mock_results = []
    for it in pack.items:
        dummy_img = tmp_path / f"{it.id}.png"
        dummy_img.write_bytes(b"\x89PNG\r\n\x1a\nFakeImage")
        mock_results.append(
            Result(
                item_id=it.id,
                status="success",
                output_path=str(dummy_img),
                engine=it.engine,
                execution_time_seconds=8.5 if it.engine == "sdxl_lightning" else 15.2,
                width=1024,
                height=1024,
                seed=it.seed,
            )
        )

    out_html = tmp_path / "style_contact_sheet.html"
    generate_style_contact_sheet_html(pack, mock_results, out_html)

    assert out_html.is_file()
    html_content = out_html.read_text(encoding="utf-8")
    assert "Style &amp; Explainer Benchmark" in html_content or "Style & Explainer Benchmark" in html_content
    assert "SDXL-Lightning (volume_image)" in html_content
    assert "FLUX.2-klein-4B (default_image)" in html_content
    assert "Channel Style (Teal &amp; Orange)" in html_content or "Channel Style (Teal & Orange)" in html_content
    assert "Explainer" in html_content
    assert "Decoration Note" in html_content


def test_pillow_diagram_overlay(tmp_path: Path) -> None:
    """Verify overlay_diagram_labels helper draws pills, leader lines, and points cleanly."""
    from PIL import Image
    from scripts.run_style_test import overlay_diagram_labels, DIAGRAM_LABELS_MAP

    bg_img = tmp_path / "test_diagram.png"
    Image.new("RGB", (512, 512), color=(20, 30, 45)).save(bg_img)

    out_annotated = tmp_path / "test_annotated.png"
    result_path = overlay_diagram_labels(bg_img, out_annotated, DIAGRAM_LABELS_MAP[6])

    assert result_path.is_file()
    with Image.open(result_path) as out:
        assert out.size == (512, 512)
        assert out.mode == "RGB"


def test_models_yaml_matches_throughput_benchmark() -> None:
    """Verify worker/models.yaml parallel_mode and offload match the latest throughput benchmark results.
    
    Fails if either engine's parallel_mode or offload differs from the recorded benchmark winning configuration.
    """
    from worker.engines import load_models_config

    repo_root = Path(__file__).resolve().parent.parent
    models = load_models_config()

    for engine_id in ("sdxl_lightning", "flux2_klein"):
        bench_file = repo_root / "outputs" / "throughput" / engine_id / "throughput_benchmark.json"
        assert bench_file.is_file(), f"Throughput benchmark result not found at {bench_file}"

        bench_data = json.loads(bench_file.read_text(encoding="utf-8"))
        expected_mode = bench_data.get("faster_mode")
        expected_offload = bench_data.get("faster_offload")

        actual_mode = models[engine_id].get("parallel_mode")
        actual_offload = models[engine_id].get("offload")

        assert actual_mode == expected_mode, (
            f"Engine '{engine_id}' parallel_mode in models.yaml ('{actual_mode}') "
            f"differs from latest benchmark result ('{expected_mode}')"
        )
        assert actual_offload == expected_offload, (
            f"Engine '{engine_id}' offload in models.yaml ('{actual_offload}') "
            f"differs from latest benchmark result ('{expected_offload}')"
        )


def test_explainer_prompts_avoid_forbidden_words() -> None:
    """Verify explainer visuals strictly avoid diagram, infographic, labeled, chart, and annotated."""
    repo_root = Path(__file__).resolve().parent.parent
    pack = load_pack(repo_root / "packs" / "style_test_pack.json")

    forbidden_words = ["diagram", "infographic", "labeled", "chart", "annotated"]
    explainer_items = [it for it in pack.items if "06" in it.id or "07" in it.id or "08" in it.id]
    assert len(explainer_items) >= 6

    for it in explainer_items:
        prompt_lower = it.prompt.lower()
        for forbidden in forbidden_words:
            assert forbidden not in prompt_lower, (
                f"Item '{it.id}' contains forbidden word '{forbidden}' in prompt: '{it.prompt}'"
            )
        assert "clean illustration of" in prompt_lower
        assert "no text" in prompt_lower
        assert "no letters" in prompt_lower
        assert "no labels" in prompt_lower
        assert "no captions" in prompt_lower
        assert "no ui elements" in prompt_lower


def test_pillow_overlay_refuses_labels_without_explicit_anchors(tmp_path: Path) -> None:
    """Verify overlay_diagram_labels scales font to >=3% height and refuses labels without explicit anchors."""
    from PIL import Image
    from scripts.run_style_test import overlay_diagram_labels

    bg_img = tmp_path / "test_refusal_bg.png"
    Image.new("RGB", (800, 600), color=(15, 23, 42)).save(bg_img)
    out_img = tmp_path / "test_refusal_out.png"

    # Label 1 has explicit normalized anchor -> should render
    # Label 2 has NO anchor -> must be refused/skipped
    # Label 3 has out-of-range anchor -> must be refused/skipped
    test_labels = [
        {"text": "Valid Anchor Label", "x": 0.2, "y": 0.2, "anchor": (0.35, 0.45)},
        {"text": "Missing Anchor Label", "x": 0.5, "y": 0.5},
        {"text": "Invalid Anchor Label", "x": 0.7, "y": 0.7, "anchor": (1.5, -0.2)},
    ]

    res = overlay_diagram_labels(bg_img, out_img, test_labels)
    assert res.is_file()
