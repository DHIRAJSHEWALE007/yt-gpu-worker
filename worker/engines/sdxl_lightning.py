"""worker/engines/sdxl_lightning.py — SDXL Lightning 4-step / 8-step image engine."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from worker.engines.base import Engine
from worker.memory import clear_gpu_memory, detect_gpus, get_recommended_torch_dtype
from worker.schema import Item

logger = logging.getLogger(__name__)


class SDXLLightningEngine(Engine):
    """ByteDance/SDXL-Lightning high-speed photorealistic image generation."""

    engine_id: str = "sdxl_lightning"
    min_vram_gb: float = 8.0

    def load(self, device: Optional[str] = None, dtype: Optional[Any] = None) -> None:
        try:
            import torch
            from diffusers import EulerDiscreteScheduler, StableDiffusionXLPipeline
        except ImportError as e:
            raise RuntimeError(f"diffusers and torch are required for SDXLLightningEngine: {e}")

        gpus = detect_gpus()
        target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        target_dtype = dtype or get_recommended_torch_dtype(gpus[0] if gpus else None)

        base_repo = self.config.get("base_model", "stabilityai/stable-diffusion-xl-base-1.0")
        logger.info("Loading SDXL-Lightning pipeline from %s (%s)...", base_repo, target_dtype)

        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            base_repo,
            torch_dtype=target_dtype,
            variant="fp16" if target_dtype == torch.float16 else None,
        )

        self.pipe.scheduler = EulerDiscreteScheduler.from_config(
            self.pipe.scheduler.config,
            timestep_spacing="trailing",
        )

        if target_device == "cuda":
            self.pipe.enable_vae_tiling()
            self.pipe.enable_vae_slicing()
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

        steps = item.steps or self.config.get("default_steps", 4)
        guidance = item.guidance_scale or self.config.get("default_guidance_scale", 0.0)
        generator = torch.Generator(device=self.device).manual_seed(item.seed) if item.seed is not None else None

        output_dir.mkdir(parents=True, exist_ok=True)
        target_file = output_dir / f"{item.id}.png"

        result = self.pipe(
            prompt=item.prompt,
            negative_prompt=item.negative_prompt or None,
            width=item.width,
            height=item.height,
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
