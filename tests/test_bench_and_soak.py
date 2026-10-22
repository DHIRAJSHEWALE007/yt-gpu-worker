"""tests/test_bench_and_soak.py — Unit tests for scripts/bench_sizes.py and scripts/soak.py."""

from pathlib import Path
import pytest

from scripts.bench_sizes import BENCH_SIZES, create_size_pack, run_size_benchmark_for_engine
from scripts.soak import analyze_drift, create_soak_items, run_soak_test


def test_create_size_pack() -> None:
    pack_land = create_size_pack("fake", 1344, 768, "1344x768", Path("/tmp"))
    assert len(pack_land.items) == 5
    for it in pack_land.items:
        assert it.width == 1344
        assert it.height == 768
        assert it.engine == "fake"

    pack_port = create_size_pack("fake", 768, 1344, "768x1344", Path("/tmp"))
    assert len(pack_port.items) == 5
    for it in pack_port.items:
        assert it.width == 768
        assert it.height == 1344
        assert it.engine == "fake"


def test_bench_sizes_mock_execution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify run_size_benchmark_for_engine executes 5 items at each size and calculates steady metrics."""
    monkeypatch.setenv("STUDIO_FAKE_PROVIDERS", "1")
    monkeypatch.setattr("scripts.bench_sizes.update_models_yaml_resolutions", lambda e, r: None)

    out_dir = tmp_path / "bench_sizes_fake"
    report = run_size_benchmark_for_engine(
        engine_id="fake",
        output_dir=out_dir,
        gpus=[0],
        allow_test=True,
    )

    assert report["engine_id"] == "fake"
    assert "1344x768" in report["sizes"]
    assert "768x1344" in report["sizes"]

    land = report["sizes"]["1344x768"]
    assert land["total_items"] == 5
    assert land["successful_items"] == 5
    assert land["steady_seconds_per_item"] >= 0.0

    port = report["sizes"]["768x1344"]
    assert port["total_items"] == 5
    assert port["successful_items"] == 5
    assert port["steady_seconds_per_item"] >= 0.0


def test_soak_create_items() -> None:
    items = create_soak_items("fake", count=100)
    assert len(items) == 100
    for idx, it in enumerate(items):
        assert it.width == 1344
        assert it.height == 768
        assert it.engine == "fake"
        assert it.id == f"soak_fake_{idx:03d}"


def test_soak_drift_analysis_logic() -> None:
    # Case 1: Normal stable execution (<= 5% variation)
    cps_stable = [
        {"image_count": 10, "vram_allocated_mb": 1000.0, "proc_rss_mb": 500.0, "ram_available_gb": 25.0},
        {"image_count": 20, "vram_allocated_mb": 1010.0, "proc_rss_mb": 510.0, "ram_available_gb": 24.8},
        {"image_count": 30, "vram_allocated_mb": 1020.0, "proc_rss_mb": 515.0, "ram_available_gb": 24.6},
    ]
    eval_stable = analyze_drift(cps_stable, threshold_percent=5.0)
    assert eval_stable["passed"] is True
    assert len(eval_stable["drift_failures"]) == 0

    # Case 2: VRAM upward drift (> 5%)
    cps_vram_leak = [
        {"image_count": 10, "vram_allocated_mb": 1000.0, "proc_rss_mb": 500.0, "ram_available_gb": 25.0},
        {"image_count": 20, "vram_allocated_mb": 1080.0, "proc_rss_mb": 505.0, "ram_available_gb": 24.8},  # +8%
    ]
    eval_vram_leak = analyze_drift(cps_vram_leak, threshold_percent=5.0)
    assert eval_vram_leak["passed"] is False
    assert any("VRAM allocated grew by 8.00%" in f for f in eval_vram_leak["drift_failures"])

    # Case 3: Process RSS upward drift (> 5%)
    cps_rss_leak = [
        {"image_count": 10, "vram_allocated_mb": 1000.0, "proc_rss_mb": 500.0, "ram_available_gb": 25.0},
        {"image_count": 20, "vram_allocated_mb": 1005.0, "proc_rss_mb": 540.0, "ram_available_gb": 24.8},  # +8%
    ]
    eval_rss_leak = analyze_drift(cps_rss_leak, threshold_percent=5.0)
    assert eval_rss_leak["passed"] is False
    assert any("Process RSS memory grew by 8.00%" in f for f in eval_rss_leak["drift_failures"])

    # Case 4: Host available RAM drop (> 5%)
    cps_ram_drop = [
        {"image_count": 10, "vram_allocated_mb": 1000.0, "proc_rss_mb": 500.0, "ram_available_gb": 25.0},
        {"image_count": 20, "vram_allocated_mb": 1005.0, "proc_rss_mb": 500.0, "ram_available_gb": 23.0},  # -8%
    ]
    eval_ram_drop = analyze_drift(cps_ram_drop, threshold_percent=5.0)
    assert eval_ram_drop["passed"] is False
    assert any("Host available RAM dropped by 8.00%" in f for f in eval_ram_drop["drift_failures"])


def test_soak_test_coordinator_mock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify run_soak_test spawns worker process, completes 20 images at 1344x768, and records summary."""
    monkeypatch.setenv("STUDIO_FAKE_PROVIDERS", "1")

    out_dir = tmp_path / "soak_fake"
    report = run_soak_test(
        engine_id="fake",
        output_dir=out_dir,
        total_images=20,
        gpus=[0],
        allow_test=True,
    )

    assert report["engine_id"] == "fake"
    assert report["total_images"] == 20
    assert report["overall_passed"] is True
    assert "gpu_0" in report["workers"]
    worker_rep = report["workers"]["gpu_0"]["report"]
    assert worker_rep["total_items"] == 20
    assert worker_rep["successful_items"] == 20
    assert len(worker_rep["checkpoints"]) == 2  # image 10 and image 20
    assert worker_rep["drift_evaluation"]["passed"] is True
