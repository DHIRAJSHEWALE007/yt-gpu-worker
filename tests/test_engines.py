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


def test_check_vram_guard_behavior(monkeypatch: pytest.MonkeyPatch) -> None:
    from worker.engines.base import InsufficientVRAMError
    from worker.engines.fake import FakeEngine
    from worker.memory import GPUInfo
    from worker.schema import Item

    engine = FakeEngine()
    engine.min_vram_gb = 10.0
    item = Item(id="test_it", engine="fake", prompt="test", width=512, height=512)

    # Mock a GPU with 15GB total, but only 3.7GB free (because model is loaded)
    loaded_gpu = GPUInfo(index=0, name="Tesla T4", total_vram_mb=15000, free_vram_mb=3700, compute_capability=(7, 5))
    monkeypatch.setattr("worker.engines.base.detect_gpus", lambda: [loaded_gpu])

    # Case 1: Model is NOT loaded -> 3.7GB free is less than 10GB req -> raises InsufficientVRAMError
    engine.is_loaded = False
    with pytest.raises(InsufficientVRAMError):
        engine.check_vram_guard(item)

    # Case 2: Model IS loaded -> 3.7GB free is plenty of headroom for activations -> does NOT raise
    engine.is_loaded = True
    engine.check_vram_guard(item)

    # Case 3: GPU total capacity is too small (6GB card for 10GB model) -> raises InsufficientVRAMError
    tiny_gpu = GPUInfo(index=0, name="GTX 1060", total_vram_mb=6000, free_vram_mb=5000, compute_capability=(6, 1))
    monkeypatch.setattr("worker.engines.base.detect_gpus", lambda: [tiny_gpu])
    with pytest.raises(InsufficientVRAMError):
        engine.check_vram_guard(item)

def test_models_yaml_single_source_of_truth() -> None:
    """Validate worker/models.yaml schema and required governance fields."""
    import yaml
    from worker.engines import load_models_config

    models = load_models_config()
    assert "zimage_turbo" in models
    assert "sdxl_turbo" in models
    assert "flux2_klein" in models
    assert "wan21_t2v_1_3b" in models
    assert "wan22_ti2v5b" in models
    assert "sdxl_lightning" in models
    assert "fake" in models

    # 1. zimage_turbo checks (Tongyi-MAI Apache-2.0)
    z = models["zimage_turbo"]
    assert z["role"] == "default_image"
    assert z["default_steps"] == 9
    assert z["default_guidance_scale"] == 0.0
    assert z["seconds_per_item"] == 8.5
    assert z["load_seconds"] == 38.0
    assert z["repo_id"] == "Tongyi-MAI/Z-Image-Turbo"
    assert z["revision"] == "f332072aa78be7aecdf3ee76d5c247082da564a6"
    assert z["license"] == "Apache-2.0"
    assert z["commercial_ok"] is True
    assert z["hero_only"] is False
    assert z["test_only"] is False

    # 1b. sdxl_turbo checks (Stability AI non-commercial)
    st = models["sdxl_turbo"]
    assert st["role"] == "test_only_noncommercial"
    assert st["commercial_ok"] is False
    assert st["enabled"] is False
    assert st["test_only"] is True
    assert st["repo_id"] == "stabilityai/sdxl-turbo"
    assert st["revision"] == "71153311d3dbb46851df1931d3ca6e939de83304"
    assert st["license"] == "other" 

    # 2. flux2_klein checks (confirmed 4B Apache variant)
    f = models["flux2_klein"]
    assert f["role"] == "hero_image"
    assert f["repo_id"] == "black-forest-labs/FLUX.2-klein-4B"
    assert f["revision"] == "e7b7dc27f91deacad38e78976d1f2b499d76a294"
    assert f["license"] == "Apache-2.0"
    assert f["commercial_ok"] is True
    assert f["gated"] is False
    assert f["seconds_per_item"] == 40.0
    assert f["load_seconds"] == 321.0
    assert f["peak_vram_gb"] == 11.5

    # 3. Wan 2.1 1.3B checks
    w = models["wan21_t2v_1_3b"]
    assert w["role"] == "hero_clip"
    assert w["hero_only"] is True
    assert w["test_only"] is False
    assert w["repo_id"] == "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    assert w["revision"] == "0fad780a534b6463e45facd96134c9f345acfa5b"
    assert w["license"] == "Apache-2.0"
    assert w["commercial_ok"] is True
    assert w["seconds_per_item"] == 390.0
    assert w["load_seconds"] == 261.0
    assert w["peak_vram_gb"] == 13.0

    # 3b. Wan 2.2 5B placeholder checks
    w5 = models["wan22_ti2v5b"]
    assert w5["enabled"] is False
    assert w5["min_vram_gb"] == 20.0

    # 4. sdxl_lightning checks
    s = models["sdxl_lightning"]
    assert s["role"] == "fallback"
    assert s["enabled"] is False
    assert "smeared" in s["notes"].lower()
    assert s["repo_id"] == "ByteDance/SDXL-Lightning"
    assert s["revision"] == "c9a24f48e1c025556787b0c58dd67a091ece2e44"
    assert s["license"] == "openrail++"

    # 5. fake mock checks
    fk = models["fake"]
    assert fk["role"] == "test"
    assert fk["test_only"] is True
    assert fk["enabled"] is False


def test_no_hardcoded_repo_ids_in_engine_files() -> None:
    """Ensure no engine file in worker/engines/*.py hardcodes Hugging Face repo IDs.
    
    All repo IDs, revisions, files, dtypes, steps, and guidance must come from models.yaml.
    """
    import ast
    import re

    engines_dir = Path(__file__).resolve().parent.parent / "worker" / "engines"
    py_files = list(engines_dir.glob("*.py"))
    assert len(py_files) >= 5, f"Expected at least 5 engine files, found {len(py_files)}"

    # Known HF organization prefixes or repo format
    hf_org_prefixes = (
        "stabilityai/",
        "black-forest-labs/",
        "ByteDance/",
        "Wan-AI/",
        "Tongyi-MAI/",
        "CompVis/",
        "runwayml/",
    )

    for py_file in py_files:
        if py_file.name.startswith("__"):
            continue
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=py_file.name)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                val = node.value.strip()
                for prefix in hf_org_prefixes:
                    if val.startswith(prefix) or val.lower().startswith(prefix.lower()):
                        pytest.fail(
                            f"Engine file '{py_file.name}' hardcodes Hugging Face repo ID '{val}'! "
                            f"All repo IDs must be configured exclusively in worker/models.yaml."
                        )

def test_bakeoff_records_pipeline_class_and_revision(tmp_path: Path) -> None:
    """Verify bakeoff records pipeline_class and revision in bakeoff.json."""
    import json
    from unittest.mock import MagicMock, patch
    from notebooks.bakeoff import run_bakeoff

    mock_engine = MagicMock()
    mock_pipe = MagicMock()
    mock_pipe.__class__.__name__ = "ZImagePipeline"
    mock_engine.pipe = mock_pipe
    mock_engine.generate.side_effect = lambda it, out_dir: (out_dir / f"{it.id}.png")

    out_dir = tmp_path / "bakeoff_verify"
    with patch("notebooks.bakeoff.get_engine", return_value=mock_engine):
        run_bakeoff(output_dir=out_dir, engines_to_test=["zimage_turbo"])
        bakeoff_json = out_dir / "bakeoff.json"
        assert bakeoff_json.is_file()
        data = json.loads(bakeoff_json.read_text(encoding="utf-8"))

        assert "zimage_turbo" in data
        zt = data["zimage_turbo"]
        assert zt["pipeline_class"] == "ZImagePipeline"
        assert zt["repo_id"] == "Tongyi-MAI/Z-Image-Turbo"
        assert zt["revision"] == "f332072aa78be7aecdf3ee76d5c247082da564a6"
        assert len(zt["samples"]) > 0
        assert "prompt" in zt["samples"][0]
        assert "peak_vram_mb" in zt["samples"][0]
        assert "median_seconds_per_item" in zt["resolutions"]["768x768"]
