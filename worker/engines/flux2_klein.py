"""worker/engines/flux2_klein.py — FLUX.1 Schnell quantized 4-step image engine."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from worker.engines.base import Engine
from worker.memory import clear_gpu_memory, detect_gpus, get_recommended_torch_dtype
from worker.schema import Item

logger = logging.getLogger(__name__)


class Flux2KleinEngine(Engine):
    """FLUX.1 Schnell quantized rectified flow pipeline."""

    engine_id: str = "flux2_klein"
    min_vram_gb: float = 12.0

    def load(self, device: Optional[str] = None, dtype: Optional[Any] = None) -> None:
        try:
            import torch
            from diffusers import FluxPipeline
        except ImportError as e:
            raise RuntimeError(f"diffusers>=0.30.0 and torch required for Flux2KleinEngine: {e}")

        gpus = detect_gpus()
        target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        target_dtype = dtype or get_recommended_torch_dtype(gpus[0] if gpus else None)

        repo_id = self.config.get("repo_id", "black-forest-labs/FLUX.1-schnell")
        self.pipe = FluxPipeline.from_pretrained(
            repo_id,
            torch_dtype=target_dtype,
        )

        if target_device == "cuda":
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

        steps = item.steps or 4
        generator = torch.Generator(device=self.device).manual_seed(item.seed) if item.seed is not None else None

        output_dir.mkdir(parents=True, exist_ok=True)
        target_file = output_dir / f"{item.id}.png"

        result = self.pipe(
            prompt=item.prompt,
            width=item.width,
            height=item.height,
            num_inference_steps=steps,
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
