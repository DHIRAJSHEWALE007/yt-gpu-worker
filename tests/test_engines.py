"""tests/test_engines.py — Fake engine tests and registry verification."""

from pathlib import Path
import pytest
from PIL import Image

from worker.engines import get_engine
from worker.engines.fake import FakeEngine
from worker.schema import Item


def test_get_engine_factory() -> None:
    engine = get_engine("fake")
    assert isinstance(engine, FakeEngine)
    assert engine.engine_id == "fake"

    with pytest.raises(ValueError):
        get_engine("unsupported_nonexistent_engine")


def test_fake_engine_image_generation(tmp_path: Path) -> None:
    engine = FakeEngine()
    engine.load()

    item = Item(id="mock_img", engine="fake", prompt="A blue ocean", width=128, height=128, seed=99)
    out_file = engine.generate(item, tmp_path)

    assert out_file.is_file()
    assert out_file.suffix == ".png"

    # Verify PIL can open and read valid dimensions
    with Image.open(out_file) as im:
        assert im.size == (128, 128)
        assert im.format == "PNG"

    engine.unload()
    assert not engine.is_loaded


def test_fake_engine_video_generation(tmp_path: Path) -> None:
    engine = FakeEngine()
    engine.load()

    item = Item(id="mock_clip", engine="fake", prompt="Panning drone shot", width=854, height=480, media_type="video", duration_seconds=2.0)
    out_file = engine.generate(item, tmp_path)

    assert out_file.is_file()
    assert out_file.suffix == ".mp4"
    assert out_file.stat().st_size > 0
