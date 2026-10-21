"""worker/engines/fake.py ? CPU-only mock engine for tests and laptop development.

Generates valid, small PNG images and tiny valid MP4 video containers without
requiring PyTorch, CUDA, diffusers, or ffmpeg.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
import struct
import time
from typing import Any, Dict, Optional, Set

from PIL import Image

from worker.engines.base import Engine
from worker.schema import Item

logger = logging.getLogger(__name__)


def _create_minimal_png(width: int = 64, height: int = 64, color: tuple = (30, 41, 59)) -> bytes:
    """Create raw binary bytes of a valid PNG image using PIL."""
    img = Image.new("RGB", (max(16, width), max(16, height)), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _create_minimal_mp4() -> bytes:
    """Create a minimal binary structure for a valid MP4 file container."""
    # ftyp box (isom / mp42)
    ftyp_data = b"isom" + bytes([0, 0, 2, 0]) + b"isomiso2mp41"
    ftyp_box = struct.pack(">I", len(ftyp_data) + 8) + b"ftyp" + ftyp_data

    # mdat box (raw mock sample payload)
    mdat_data = bytes([0] * 256)
    mdat_box = struct.pack(">I", len(mdat_data) + 8) + b"mdat" + mdat_data

    # moov box (header skeleton)
    mvhd_data = bytes([0] * 100)
    mvhd_box = struct.pack(">I", len(mvhd_data) + 8) + b"mvhd" + mvhd_data
    moov_box = struct.pack(">I", len(mvhd_box) + 8) + b"moov" + mvhd_box

    return ftyp_box + mdat_box + moov_box


class FakeEngine(Engine):
    """Mock engine running on CPU generating lightweight valid assets."""

    engine_id: str = "fake"
    min_vram_gb: float = 0.0

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self.simulate_oom_on_ids: Set[str] = set()
        self.simulate_delay_seconds: float = 0.0

    def load(self, device: Optional[str] = None, dtype: Optional[Any] = None) -> None:
        self.device = "cpu"
        self.dtype = dtype or self.recommended_dtype or "cpu"
        self.is_loaded = True
        logger.info("[FakeEngine] Mock weights loaded successfully on CPU.")

    def generate(self, item: Item, output_dir: Path) -> Path:
        if not self.is_loaded:
            self.load()

        # OOM Simulation Hook for Testing Recovery
        if item.id in self.simulate_oom_on_ids:
            self.simulate_oom_on_ids.remove(item.id)
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.40 GiB (GPU 0; 15.78 GiB total capacity)")

        if self.simulate_delay_seconds > 0:
            time.sleep(self.simulate_delay_seconds)

        output_dir.mkdir(parents=True, exist_ok=True)

        if item.media_type == "video":
            ext = "mp4"
            target_path = output_dir / f"{item.id}.{ext}"
            target_path.write_bytes(_create_minimal_mp4())
        else:
            ext = "png"
            target_path = output_dir / f"{item.id}.{ext}"
            seed = item.seed or 42
            r = (seed * 37) % 255
            g = (seed * 73) % 255
            b = (seed * 109) % 255
            target_path.write_bytes(_create_minimal_png(item.width, item.height, color=(r, g, b)))

        logger.info("[FakeEngine] Generated %s output: %s", item.media_type, target_path.name)
        return target_path

    def unload(self) -> None:
        self.is_loaded = False
        logger.info("[FakeEngine] Unloaded from memory.")

    def vram_estimate(self, item: Item) -> float:
        return 0.0
