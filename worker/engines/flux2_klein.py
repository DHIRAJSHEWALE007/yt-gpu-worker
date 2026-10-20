"""worker/engines/flux2_klein.py — FLUX.1 Schnell quantized 4-step image engine."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from worker.engines.base import Engine
from worker.memory import clear_gpu_memory, detect_gpus, get_recommended_torch_dtype
from worker.schema import Item

logger = logging.getLogger(__name__)


def _get_hf_token() -> Optional[str]:
    """Retrieve Hugging Face token from environment or Kaggle Secrets."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if token:
        return token.strip()
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore
        token = UserSecretsClient().get_secret("HF_TOKEN")
        if token:
            return token.strip()
    except Exception:
        pass
    return None


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
        token = _get_hf_token()

        logger.info("Loading FLUX.1 pipeline from %s (authenticated: %s)...", repo_id, bool(token))

        try:
            self.pipe = FluxPipeline.from_pretrained(
                repo_id,
                torch_dtype=target_dtype,
                token=token,
            )
        except Exception as exc:
            err_str = str(exc).lower()
            if "401" in err_str or "unauthorized" in err_str or "gated" in err_str or "restricted" in err_str:
                raise RuntimeError(
                    f"\n[FLUX AUTHENTICATION REQUIRED]\n"
                    f"Model '{repo_id}' is a gated Hugging Face repository.\n"
                    f"To enable it:\n"
                    f"  1. Accept the license at: https://huggingface.co/{repo_id}\n"
                    f"  2. In your Kaggle notebook, set your HF access token before running:\n"
                    f"     import os; os.environ['HF_TOKEN'] = 'hf_your_token_here'\n"
                    f"     (or add 'HF_TOKEN' to Kaggle Secrets -> Add-ons -> Secrets)\n"
                ) from exc
            raise

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

        steps = item.steps or 4
        generator = None
        if item.seed is not None:
            try:
                generator = torch.Generator(device=self.device).manual_seed(item.seed)
            except Exception:
                generator = torch.Generator(device="cpu").manual_seed(item.seed)

        output_dir.mkdir(parents=True, exist_ok=True)
        target_file = output_dir / f"{item.id}.png"

        w = ((item.width // 8) * 8) if item.width else 1024
        h = ((item.height // 8) * 8) if item.height else 1024

        result = self.pipe(
            prompt=item.prompt,
            width=w,
            height=h,
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
