"""worker/engines/zimage_turbo.py ? Tongyi-MAI Z-Image-Turbo 9-step DiT engine driven by models.yaml.

Features:
- Reads repo_id, revision, dtype, steps, and guidance from models.yaml.
- fp16 on Tesla T4 and P100; bf16 only where supported by GPU hardware.
- Guard against NaN or black output with automatic VAE float32 upcast and offload fallback.
- No hardcoded Hugging Face repo IDs in code.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from worker.engines.base import Engine
from worker.memory import clear_gpu_memory, detect_gpus, get_recommended_torch_dtype
from worker.schema import Item

logger = logging.getLogger(__name__)


def _is_black_or_degenerate_image(image: Any) -> bool:
    """Check if PIL Image is pure black, contains all zeros, or degenerate pixels."""
    try:
        extrema = image.getextrema()
        # For RGB: extrema is ((minR, maxR), (minG, maxG), (minB, maxB))
        if isinstance(extrema, tuple):
            if all(ch[1] == 0 for ch in extrema):
                return True
            # Also check if image has 0 dynamic range across all channels
            if all(ch[0] == ch[1] for ch in extrema):
                return True
    except Exception as exc:
        logger.debug("Could not inspect image extrema: %s", exc)
    return False


class ZImageTurboEngine(Engine):
    """Z-Image-Turbo 9-step distilled DiT pipeline driven by models.yaml."""

    engine_id: str = "zimage_turbo"
    min_vram_gb: float = 8.0

    def load(self, device: Optional[str] = None, dtype: Optional[Any] = None) -> None:
        try:
            import torch
        except ImportError as e:
            raise RuntimeError(f"PyTorch is required for ZImageTurboEngine: {e}")

        # Dynamically import ZImagePipeline or fall back to DiffusionPipeline
        pipe_cls = None
        try:
            from diffusers import ZImagePipeline
            pipe_cls = ZImagePipeline
        except ImportError:
            try:
                from diffusers import DiffusionPipeline
                pipe_cls = DiffusionPipeline
            except ImportError as exc:
                raise RuntimeError(
                    f"diffusers with ZImagePipeline support required: {exc}. "
                    f"Please run: !pip install git+https://github.com/huggingface/diffusers"
                )

        repo_id = self.repo_id or self.config.get("repo_id")
        if not repo_id:
            raise ValueError(f"Engine '{self.engine_id}' has no repo_id configured in models.yaml.")
        revision = self.revision or self.config.get("revision")

        gpus = detect_gpus()
        target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Determine target dtype: fp16 on T4/P100, bf16 only where hardware supports it
        if dtype is not None:
            target_dtype = dtype
        else:
            rec = get_recommended_torch_dtype(gpus[0] if gpus else None)
            # If recommended is bf16 and hardware supports it, use bfloat16; otherwise fp16
            if rec == torch.bfloat16:
                target_dtype = torch.bfloat16
            elif target_device.startswith("cuda"):
                target_dtype = torch.float16
            else:
                target_dtype = torch.float32

        logger.info(
            "Loading Z-Image-Turbo from %s (rev: %s, dtype: %s, device: %s)...",
            repo_id,
            revision,
            target_dtype,
            target_device,
        )

        clear_gpu_memory()

        try:
            self.pipe = pipe_cls.from_pretrained(
                repo_id,
                revision=revision,
                torch_dtype=target_dtype,
                low_cpu_mem_usage=True,
            )
        except Exception as load_err:
            logger.warning(
                "Direct loading failed (%s). Attempting fallback with trust_remote_code=True...",
                load_err,
            )
            self.pipe = pipe_cls.from_pretrained(
                repo_id,
                revision=revision,
                torch_dtype=target_dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )

        gpu_id = 0
        if target_device.startswith("cuda:") and target_device.split(":")[1].isdigit():
            gpu_id = int(target_device.split(":")[1])

        self.apply_pipeline_optimizations(target_device, gpu_id=gpu_id)

        self.device = target_device
        self.dtype = target_dtype
        self.is_loaded = True

    def _execute_inference(self, prompt: str, width: int, height: int, steps: int, guidance: float, generator: Any) -> Any:
        """Execute pipeline forward call."""
        try:
            return self.pipe(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
            ).images[0]
        except TypeError:
            # Some diffusers pipeline variants take prompt without guidance_scale if distilled
            return self.pipe(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                generator=generator,
            ).images[0]

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

        w = ((item.width // 16) * 16) if item.width else 1024
        h = ((item.height // 16) * 16) if item.height else 1024

        # Primary inference attempt
        image = self._execute_inference(item.prompt, w, h, steps, guidance, generator)

        # Guard against NaN or all-black output (common in float16 numerical instability)
        if _is_black_or_degenerate_image(image):
            logger.warning(
                "[DEGENERATE OUTPUT GUARD] Detected NaN or black image with %s in fp16. "
                "Initiating recovery: upcasting VAE to float32...",
                self.engine_id,
            )
            try:
                if hasattr(self.pipe, "vae") and self.pipe.vae is not None:
                    self.pipe.vae.to(dtype=torch.float32)

                # Reset seed and regenerate
                if item.seed is not None:
                    try:
                        generator = torch.Generator(device=self.device).manual_seed(item.seed)
                    except Exception:
                        generator = torch.Generator(device="cpu").manual_seed(item.seed)

                image = self._execute_inference(item.prompt, w, h, steps, guidance, generator)
                if not _is_black_or_degenerate_image(image):
                    logger.info("[RECOVERY SUCCESS] Regenerated clean image after VAE float32 upcast.")
                else:
                    logger.warning("[RECOVERY PARTIAL] Output still degenerate. Enabling CPU offload fallback...")
                    if hasattr(self.pipe, "enable_model_cpu_offload"):
                        self.pipe.enable_model_cpu_offload()
                    image = self._execute_inference(item.prompt, w, h, steps, guidance, generator)
            except Exception as rec_err:
                logger.error("Black/NaN image recovery attempt failed: %s", rec_err)

        image.save(target_file, format="PNG")
        return target_file

    def unload(self) -> None:
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        self.is_loaded = False
        clear_gpu_memory()
