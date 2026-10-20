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
