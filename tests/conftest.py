"""tests/conftest.py — Test fixtures and shared helpers."""

import os
from pathlib import Path
import pytest

from worker.schema import Item, Pack


@pytest.fixture
def tmp_pack(tmp_path: Path) -> Pack:
    """Fixture providing a standard 4-item Pack."""
    items = [
        Item(id="shot_001", engine="fake", prompt="Volcano erupting at night", width=1024, height=1024, seed=42),
        Item(id="shot_002", engine="fake", prompt="Lava flow entering the sea", width=768, height=768, seed=43),
        Item(id="shot_003", engine="fake", prompt="Volcanic ash cloud in sunlight", width=1024, height=1024, seed=44),
        Item(id="clip_004", engine="fake", prompt="Panning shot of caldera", width=854, height=480, media_type="video", duration_seconds=2.0),
    ]
    return Pack(
        pack_id="pack_test_volcano",
        job_id="job_20261020_volcano",
        items=items,
        output_dir=str(tmp_path / "outputs"),
    )
