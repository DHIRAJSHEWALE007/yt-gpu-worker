"""worker/engines/zimage_turbo.py ? Fast distilled diffusion image engine driven by models.yaml."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from worker.engines.base import Engine
from worker.memory import clear_gpu_memory, detect_gpus, get_recommended_torch_dtype
from worker.schema import Item

logger = logging.getLogger(__name__)


class ZImageTurboEngine(Engine):
    """Fast distilled diffusion image generation engine driven by models.yaml."""

    engine_id: str = "zimage_turbo"
    min_vram_gb: float = 7.0

    def load(self, device: Optional[str] = None, dtype: Optional[Any] = None) -> None:
        try:
            import torch
            from diffusers import AutoPipelineForText2Image
        except ImportError as e:
            raise RuntimeError(f"diffusers and torch required for ZImageTurboEngine: {e}")

        repo_id = self.repo_id or self.config.get("repo_id")
        if not repo_id:
            raise ValueError(f"Engine '{self.engine_id}' has no repo_id configured in models.yaml.")
        revision = self.revision or self.config.get("revision")

        gpus = detect_gpus()
        target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        target_dtype = dtype or get_recommended_torch_dtype(gpus[0] if gpus else None)

        self.pipe = AutoPipelineForText2Image.from_pretrained(
            repo_id,
            revision=revision,
            torch_dtype=target_dtype,
            variant="fp16" if target_dtype == torch.float16 else None,
        )

        gpu_id = 0
        if target_device.startswith("cuda:") and target_device.split(":")[1].isdigit():
            gpu_id = int(target_device.split(":")[1])

        self.apply_pipeline_optimizations(target_device, gpu_id=gpu_id)

        self.device = target_device
        self.dtype = target_dtype
        self.is_loaded = True

    def generate(self, item: Item, output_dir: Path) -> Path:
        self.check_vram_guard(item)
        if not self.is_loaded:
            self.load()

        import torch

        steps = item.steps if item.steps is not None else self.default_steps
        guidance = item.guidance_scale if item.guidance_scale is not None else self.default_guidance_scale

        generator = None
        if item.seed is not None:
            try:
                generator = torch.Generator(device=self.device).manual_seed(item.seed)
            except Exception:
                generator = torch.Generator(device="cpu").manual_seed(item.seed)

        output_dir.mkdir(parents=True, exist_ok=True)
        target_file = output_dir / f"{item.id}.png"

        # Multiples of 8 check for diffusers VAE compatibility
        w = ((item.width // 8) * 8) if item.width else 512
        h = ((item.height // 8) * 8) if item.height else 512

        result = self.pipe(
            prompt=item.prompt,
            width=w,
            height=h,
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
