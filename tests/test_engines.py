"""tests/test_engines.py — Fake engine tests and registry verification."""

from pathlib import Path
import pytest
from PIL import Image

from worker.engines import get_engine
from worker.engines.base import safe_apply_gpu_optimizations, safe_enable_vae_slicing, safe_enable_vae_tiling
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


def test_safe_vae_tiling_and_slicing_fallbacks() -> None:
    # Case 1: Pipeline has enable_vae_tiling and enable_vae_slicing directly
    class PipeDirect:
        tiled = False
        sliced = False

        def enable_vae_tiling(self) -> None:
            self.tiled = True

        def enable_vae_slicing(self) -> None:
            self.sliced = True

    p1 = PipeDirect()
    assert safe_enable_vae_tiling(p1) is True
    assert p1.tiled is True
    assert safe_enable_vae_slicing(p1) is True
    assert p1.sliced is True

    # Case 2: Pipeline has no direct method, but vae component has enable_tiling and enable_slicing
    class VaeComponent:
        tiled = False
        sliced = False

        def enable_tiling(self) -> None:
            self.tiled = True

        def enable_slicing(self) -> None:
            self.sliced = True

    class PipeWithVae:
        def __init__(self) -> None:
            self.vae = VaeComponent()

    p2 = PipeWithVae()
    assert safe_enable_vae_tiling(p2) is True
    assert p2.vae.tiled is True
    assert safe_enable_vae_slicing(p2) is True
    assert p2.vae.sliced is True

    # Case 3: Pipeline has neither (e.g. object without tiling support) -> returns False without error
    class PipeEmpty:
        pass

    p3 = PipeEmpty()
    assert safe_enable_vae_tiling(p3) is False
    assert safe_enable_vae_slicing(p3) is False

    # Case 4: safe_apply_gpu_optimizations works with mock cuda and cpu
    class PipeMock:
        offloaded = False
        to_dev = None

        def enable_model_cpu_offload(self, gpu_id: int = 0) -> None:
            self.offloaded = True

        def to(self, dev: str) -> None:
            self.to_dev = dev

    pm = PipeMock()
    safe_apply_gpu_optimizations(pm, target_device="cuda:0", gpu_id=0)
    assert pm.offloaded is True

    pm_cpu = PipeMock()
    safe_apply_gpu_optimizations(pm_cpu, target_device="cpu")
    assert pm_cpu.to_dev == "cpu"
