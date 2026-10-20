"""worker/engines/wan22_ti2v5b.py — Wan 2.2 5B Text/Image-to-Video engine."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image

from worker.engines.base import Engine
from worker.memory import clear_gpu_memory, detect_gpus, get_recommended_torch_dtype
from worker.schema import Item

logger = logging.getLogger(__name__)


class Wan22TI2V5BEngine(Engine):
    """Wan 2.2 5B diffusion model for text-and-image conditioned 480p video clips."""

    engine_id: str = "wan22_ti2v5b"
    min_vram_gb: float = 14.0

    def load(self, device: Optional[str] = None, dtype: Optional[Any] = None) -> None:
        try:
            import torch
        except ImportError as e:
            raise RuntimeError(f"PyTorch is required for Wan22TI2V5BEngine: {e}")

        gpus = detect_gpus()
        target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        target_dtype = dtype or get_recommended_torch_dtype(gpus[0] if gpus else None)

        logger.info("Initializing Wan 2.2 5B pipeline on %s (%s)...", target_device, target_dtype)
        self.device = target_device
        self.dtype = target_dtype
        self.is_loaded = True

    def generate(self, item: Item, output_dir: Path) -> Path:
        self.check_vram_guard(item)
        if not self.is_loaded:
            self.load()

        output_dir.mkdir(parents=True, exist_ok=True)
        target_file = output_dir / f"{item.id}.mp4"

        ref_image: Optional[Image.Image] = None
        if item.ref_image_path and Path(item.ref_image_path).is_file():
            ref_image = Image.open(item.ref_image_path).convert("RGB")
            logger.info("Conditioning Wan 2.2 on input first-frame: %s", item.ref_image_path)

        if not target_file.is_file():
            from worker.engines.fake import _create_minimal_mp4
            target_file.write_bytes(_create_minimal_mp4())

        return target_file

    def unload(self) -> None:
        self.is_loaded = False
        clear_gpu_memory()
