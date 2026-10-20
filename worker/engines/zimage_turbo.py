"""worker/engines/zimage_turbo.py — Fast distilled 1-step diffusion image engine."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from worker.engines.base import Engine
from worker.memory import clear_gpu_memory, detect_gpus, get_recommended_torch_dtype
from worker.schema import Item

logger = logging.getLogger(__name__)


class ZImageTurboEngine(Engine):
    """SDXL-Turbo single-step real-time generation."""

    engine_id: str = "zimage_turbo"
    min_vram_gb: float = 7.0

    def load(self, device: Optional[str] = None, dtype: Optional[Any] = None) -> None:
        try:
            import torch
            from diffusers import AutoPipelineForText2Image
        except ImportError as e:
            raise RuntimeError(f"diffusers and torch required for ZImageTurboEngine: {e}")

        gpus = detect_gpus()
        target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        target_dtype = dtype or get_recommended_torch_dtype(gpus[0] if gpus else None)

        repo_id = self.config.get("repo_id", "stabilityai/sdxl-turbo")
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            repo_id,
            torch_dtype=target_dtype,
            variant="fp16" if target_dtype == torch.float16 else None,
        )

        if target_device == "cuda":
            self.pipe.enable_vae_tiling()
            if hasattr(self.pipe, "enable_model_cpu_offload"):
                self.pipe.enable_model_cpu_offload()
            else:
                self.pipe.to(target_device)
        else:
            self.pipe.to("cpu")

        self.device = target_device
        self.dtype = target_dtype
        self.is_loaded = True

    def generate(self, item: Item, output_dir: Path) -> Path:
        self.check_vram_guard(item)
        if not self.is_loaded:
            self.load()

        import torch

        steps = item.steps or 1
        guidance = item.guidance_scale or 0.0
        generator = torch.Generator(device=self.device).manual_seed(item.seed) if item.seed is not None else None

        output_dir.mkdir(parents=True, exist_ok=True)
        target_file = output_dir / f"{item.id}.png"

        result = self.pipe(
            prompt=item.prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
        )

        image = result.images[0]
        image.save(target_file, format="PNG")
        return target_file

    def unload(self) -> None:
        del self.pipe
        self.pipe = None
        self.is_loaded = False
        clear_gpu_memory()
