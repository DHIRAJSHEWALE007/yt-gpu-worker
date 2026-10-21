"""worker/engines/wan22_ti2v5b.py ? Wan Video diffusion engine driven by models.yaml."""

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
    """Wan Text-to-Video diffusion engine driven by models.yaml."""

    engine_id: str = "wan21_t2v_1_3b"
    min_vram_gb: float = 12.0

    def load(self, device: Optional[str] = None, dtype: Optional[Any] = None) -> None:
        try:
            import torch
        except ImportError as e:
            raise RuntimeError(f"PyTorch is required for Wan22TI2V5BEngine: {e}")

        try:
            from diffusers import WanPipeline
        except ImportError as e:
            raise RuntimeError(
                f"WanPipeline is not available in the installed diffusers package: {e}.\n"
                f"Wan video requires diffusers >= 0.33.0.\n"
                f"Please run: !pip install --upgrade diffusers"
            )

        repo_id = self.repo_id or self.config.get("repo_id")
        if not repo_id:
            raise ValueError(f"Engine '{self.engine_id}' requires 'repo_id' in models.yaml configuration.")
        revision = self.revision or self.config.get("revision")

        gpus = detect_gpus()
        target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        target_dtype = dtype or get_recommended_torch_dtype(gpus[0] if gpus else None)

        logger.info("Loading Wan video pipeline from %s on %s (%s)...", repo_id, target_device, target_dtype)

        self.pipe = WanPipeline.from_pretrained(
            repo_id,
            revision=revision,
            torch_dtype=target_dtype,
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
        from diffusers.utils import export_to_video

        output_dir.mkdir(parents=True, exist_ok=True)
        target_file = output_dir / f"{item.id}.mp4"

        w = ((item.width // 16) * 16) if item.width else 832
        h = ((item.height // 16) * 16) if item.height else 480

        frames_count = 17
        if item.duration_seconds and item.duration_seconds > 2.0:
            frames_count = 33

        steps = item.steps if item.steps is not None else self.default_steps
        guidance = item.guidance_scale if item.guidance_scale is not None else self.default_guidance_scale

        generator = None
        if item.seed is not None:
            try:
                generator = torch.Generator(device=self.device).manual_seed(item.seed)
            except Exception:
                generator = torch.Generator(device="cpu").manual_seed(item.seed)

        logger.info("Synthesizing video with Wan (%dx%d, %d frames)...", w, h, frames_count)

        output = self.pipe(
            prompt=item.prompt,
            negative_prompt=item.negative_prompt or "Bright tones, overexposed, static, blurred details, low quality",
            height=h,
            width=w,
            num_frames=frames_count,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
        ).frames[0]

        export_to_video(output, str(target_file), fps=16)
        logger.info("Saved video clip to %s", target_file)
        return target_file

    def unload(self) -> None:
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        self.is_loaded = False
        clear_gpu_memory()
