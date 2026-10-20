"""tests/test_schema.py — Pydantic schema validation and atomic IO tests."""

import json
from pathlib import Path
import pytest

from worker.schema import (
    Item,
    Pack,
    Result,
    SCHEMA_VERSION,
    append_result_atomic,
    load_pack,
    load_results,
    save_pack,
    save_results_atomic,
)


def test_item_defaults_and_validation() -> None:
    item = Item(id="test_01", engine="sdxl_lightning", prompt="Sunset over Alps")
    assert item.id == "test_01"
    assert item.engine == "sdxl_lightning"
    assert item.width == 1024
    assert item.height == 1024
    assert item.media_type == "image"
    assert item.negative_prompt == ""


def test_pack_schema_version_and_serialization(tmp_path: Path) -> None:
    pack = Pack(
        pack_id="pack_unit_test",
        items=[Item(id="item_1", engine="fake", prompt="Test Prompt")],
    )
    assert pack.schema_version == SCHEMA_VERSION

    target = tmp_path / "test_pack.json"
    saved_path = save_pack(pack, target)
    assert saved_path.is_file()

    loaded = load_pack(saved_path)
    assert loaded.pack_id == "pack_unit_test"
    assert len(loaded.items) == 1
    assert loaded.items[0].id == "item_1"


def test_results_atomic_io(tmp_path: Path) -> None:
    res_file = tmp_path / "results.json"
    assert load_results(res_file) == []

    r1 = Result(item_id="it_1", status="success", engine="fake", execution_time_seconds=0.12)
    append_result_atomic(r1, res_file)

    loaded = load_results(res_file)
    assert len(loaded) == 1
    assert loaded[0].item_id == "it_1"
    assert loaded[0].status == "success"

    # Append second result
    r2 = Result(item_id="it_2", status="failed", engine="fake", error="Mock failure", execution_time_seconds=0.05)
    append_result_atomic(r2, res_file)

    loaded2 = load_results(res_file)
    assert len(loaded2) == 2
    assert loaded2[1].status == "failed"
