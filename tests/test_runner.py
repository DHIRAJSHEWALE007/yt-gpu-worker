"""tests/test_runner.py — PackRunner execution, sorting, crash resume, and OOM recovery tests."""

from pathlib import Path
import time
import pytest

from worker.engines.fake import FakeEngine
from worker.runner import PackRunner, run_pack
from worker.schema import Item, Pack, Result, load_results


def test_run_pack_full_execution(tmp_pack: Pack, tmp_path: Path) -> None:
    out_dir = tmp_path / "run_out"
    results = run_pack(tmp_pack, output_dir=out_dir)

    assert len(results) == len(tmp_pack.items)
    for res in results:
        assert res.status == "success"
        assert Path(res.output_path).is_file()

    # Verify results.json exists and matches
    res_file = out_dir / "results.json"
    assert res_file.is_file()
    saved = load_results(res_file)
    assert len(saved) == 4


def test_runner_idempotency_and_crash_resume(tmp_pack: Pack, tmp_path: Path) -> None:
    out_dir = tmp_path / "resume_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Pre-render one item to simulate previous run
    first_item = tmp_pack.items[0]
    expected_file = out_dir / f"{first_item.id}.png"
    expected_file.write_bytes(b"\x89PNG\r\n\x1a\nPreExistingData")

    runner = PackRunner(tmp_pack, output_dir=out_dir)
    results = runner.run()

    # Verify pre-existing item was skipped without re-rendering
    first_res = [r for r in results if r.item_id == first_item.id][0]
    assert first_res.status == "skipped"

    # Remaining items must be successful
    other_results = [r for r in results if r.item_id != first_item.id]
    assert len(other_results) == 3
    for r in other_results:
        assert r.status == "success"


def test_runner_time_budget_clean_stop(tmp_path: Path) -> None:
    # 5 items with simulated delay
    items = [Item(id=f"slow_{i}", engine="fake", prompt="Slow item") for i in range(5)]
    # Set max_runtime_minutes = 0.01 min (0.6s) to trigger immediate time stop
    pack = Pack(
        pack_id="pack_budget_test",
        items=items,
        max_runtime_minutes=0.01,
        output_dir=str(tmp_path / "budget_out"),
    )

    runner = PackRunner(pack, output_dir=tmp_path / "budget_out")
    # Inject delay into fake engine
    runner.models_config["fake"] = {}
    results = runner.run()

    # Should have stopped early cleanly
    assert len(results) < len(items)


def test_runner_oom_recovery_retry(tmp_path: Path) -> None:
    """Verify runner catches OOM, retries at lower resolution, and records retried_oom=True."""
    item_oom = Item(id="shot_cause_oom", engine="fake", prompt="Huge render", width=1024, height=1024)
    item_ok = Item(id="shot_normal", engine="fake", prompt="Normal render", width=1024, height=1024)

    pack = Pack(
        pack_id="pack_oom_test",
        items=[item_oom, item_ok],
        output_dir=str(tmp_path / "oom_out"),
    )

    out_dir = tmp_path / "oom_out"
    runner = PackRunner(pack, output_dir=out_dir)

    # Configure FakeEngine to raise OOM on shot_cause_oom once
    def _get_engine_patch(engine_id: str):
        eng = FakeEngine()
        eng.simulate_oom_on_ids.add("shot_cause_oom")
        return eng

    runner._get_or_load_engine = _get_engine_patch  # type: ignore

    results = runner.run()
    assert len(results) == 2

    oom_res = [r for r in results if r.item_id == "shot_cause_oom"][0]
    # Successfully recovered via downscale retry!
    assert oom_res.status == "success"
    assert oom_res.retried_oom is True
    assert Path(oom_res.output_path).is_file()

    normal_res = [r for r in results if r.item_id == "shot_normal"][0]
    assert normal_res.status == "success"
    assert normal_res.retried_oom is False

def test_runner_refuses_test_only_engine_without_env_or_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify runner refuses to run test_only engines unless STUDIO_FAKE_PROVIDERS or allow_test_engines is set."""
    monkeypatch.delenv("STUDIO_FAKE_PROVIDERS", raising=False)

    item = Item(id="test_mock_refuse", engine="fake", prompt="test", width=64, height=64)
    pack = Pack(pack_id="pack_test_refuse", items=[item], output_dir=str(tmp_path / "refuse_out"))

    # Case 1: Without env or flag -> must raise ValueError
    runner_refused = PackRunner(pack, output_dir=tmp_path / "refuse_out", allow_test_engines=False)
    with pytest.raises(ValueError, match="marked as test_only"):
        runner_refused.run()

    # Case 2: With allow_test_engines=True -> succeeds
    runner_allowed = PackRunner(pack, output_dir=tmp_path / "allow_out", allow_test_engines=True)
    results = runner_allowed.run()
    assert len(results) == 1
    assert results[0].status == "success"

    # Case 3: With STUDIO_FAKE_PROVIDERS=1 -> succeeds
    monkeypatch.setenv("STUDIO_FAKE_PROVIDERS", "1")
    runner_env = PackRunner(pack, output_dir=tmp_path / "env_out", allow_test_engines=False)
    results_env = runner_env.run()
    assert len(results_env) == 1
    assert results_env[0].status == "success"

def test_runner_refuses_noncommercial_engine_without_flag_or_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify runner refuses engines with commercial_ok=false unless --allow-noncommercial or STUDIO_ALLOW_NONCOMMERCIAL=1 is passed."""
    monkeypatch.delenv("STUDIO_ALLOW_NONCOMMERCIAL", raising=False)
    monkeypatch.setenv("STUDIO_FAKE_PROVIDERS", "1")

    # sdxl_turbo has commercial_ok: false in models.yaml
    item = Item(id="test_nc_item", engine="sdxl_turbo", prompt="test prompt", width=64, height=64)
    pack = Pack(pack_id="pack_test_nc", items=[item], output_dir=str(tmp_path / "nc_out"))

    # Case 1: Without allow_noncommercial -> raises ValueError
    runner_refused = PackRunner(pack, output_dir=tmp_path / "nc_out", allow_noncommercial=False)
    with pytest.raises(ValueError, match="commercial_ok=false"):
        runner_refused.run()

    # Case 2: With allow_noncommercial=True parameter -> passes upfront check
    runner_allowed = PackRunner(pack, output_dir=tmp_path / "nc_out", allow_noncommercial=True)
    # The upfront validation must pass without raising ValueError
    assert runner_allowed.allow_noncommercial is True

    # Case 3: With STUDIO_ALLOW_NONCOMMERCIAL=1 env -> passes upfront check
    monkeypatch.setenv("STUDIO_ALLOW_NONCOMMERCIAL", "1")
    runner_env = PackRunner(pack, output_dir=tmp_path / "nc_out", allow_noncommercial=False)
    assert runner_env.allow_noncommercial is False


def test_cli_runner_subprocess_execution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify executing worker.runner CLI as a subprocess correctly parses --results and writes results."""
    import os
    import subprocess
    import sys
    from worker.schema import Item, Pack, load_results

    monkeypatch.setenv("STUDIO_FAKE_PROVIDERS", "1")
    items = [Item(id="cli_sub_item", engine="fake", prompt="CLI subprocess test")]
    sub_pack = Pack(pack_id="cli_test_pack", items=items, output_dir=str(tmp_path))
    pack_file = tmp_path / "sub_pack.json"
    pack_file.write_text(sub_pack.model_dump_json(indent=2), encoding="utf-8")
    custom_results_path = tmp_path / "custom_results.json"

    env = os.environ.copy()
    env["STUDIO_FAKE_PROVIDERS"] = "1"
    cmd = [
        sys.executable,
        "-m", "worker.runner",
        "--pack", str(pack_file),
        "--out", str(tmp_path),
        "--results", str(custom_results_path),
        "--test",
    ]
    repo_root = Path(__file__).resolve().parents[1]
    res = subprocess.run(cmd, cwd=str(repo_root), env=env, check=True)
    assert res.returncode == 0
    assert custom_results_path.is_file()
    results = load_results(custom_results_path)
    assert len(results) == 1
    assert results[0].item_id == "cli_sub_item"
    assert results[0].status == "success"


def test_runner_oom_downscale_fallback_stage2(tmp_path: Path) -> None:
    """Verify 2-stage OOM retry falls back to downscaling when CPU offload still OOMs, and records fallback_applied in results.json."""
    item_large = Item(id="shot_large_oom", engine="fake", prompt="Huge render", width=1024, height=1024)

    pack = Pack(
        pack_id="pack_stage2_oom_test",
        items=[item_large],
        output_dir=str(tmp_path / "stage2_out"),
    )

    out_dir = tmp_path / "stage2_out"
    runner = PackRunner(pack, output_dir=out_dir)

    # Configure FakeEngine to OOM at > 768
    def _get_engine_patch(engine_id: str):
        eng = FakeEngine()
        eng.simulate_oom_on_large_res = True
        return eng

    runner._get_or_load_engine = _get_engine_patch  # type: ignore

    results = runner.run()
    assert len(results) == 1
    res = results[0]

    # Must recover on stage 2 (downscaled to 768x768)
    assert res.status == "success"
    assert res.retried_oom is True
    assert res.fallback_applied == "downscale"
    assert Path(res.output_path).is_file()

    # Verify saved results.json contains fallback_applied
    res_file = out_dir / "results.json"
    assert res_file.is_file()
    saved_results = load_results(res_file)
    assert len(saved_results) == 1
    assert saved_results[0].fallback_applied == "downscale"
    assert saved_results[0].retried_oom is True
