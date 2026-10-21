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
        lambda e, m: recorded_updates.update({e: m}),
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

    mode_a = report["modes"]["single_process_multi_gpu"]
    mode_b = report["modes"]["process_per_gpu"]

    assert mode_a["successful_items"] == 20
    assert mode_b["successful_items"] == 20
    assert mode_a["failed_items"] == 0
    assert mode_b["failed_items"] == 0
    assert mode_a["oom_occurred"] is False
    assert mode_b["oom_occurred"] is False
    assert mode_a["total_time_seconds"] > 0
    assert mode_b["total_time_seconds"] > 0
    assert mode_a["items_per_second"] > 0
    assert mode_b["items_per_second"] > 0

    assert report["faster_mode"] in ("single_process_multi_gpu", "process_per_gpu")
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
    assert len(pack.items) == 16  # 8 SDXL-Lightning + 8 FLUX.2-klein

    sdxl_items = [it for it in pack.items if it.engine == "sdxl_lightning"]
    flux_items = [it for it in pack.items if it.engine == "flux2_klein"]
    assert len(sdxl_items) == 8
    assert len(flux_items) == 8

    # Verify 5 channel-style and 3 explainer prompts
    channel_items = [it for it in sdxl_items if it.extra_params.get("category") == "channel_style"]
    explainer_items = [it for it in sdxl_items if it.extra_params.get("category") == "explainer_diagram"]
    assert len(channel_items) == 5
    assert len(explainer_items) == 3

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
    assert "Explainer Technical Diagram" in html_content
