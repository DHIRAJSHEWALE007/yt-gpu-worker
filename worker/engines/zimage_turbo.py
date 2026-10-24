"""worker/engines/zimage_turbo.py ? Tongyi-MAI Z-Image-Turbo 9-step DiT engine driven by models.yaml.

Rules:
- Requires native bfloat16 (GPU compute capability >= 8.0).
- Raises UnsupportedGPUError before loading weights if CC < 8.0 unless --allow-slow-bf16 is passed.
- Enforces host-RAM guard before loading.
- Raises DegenerateOutputError on NaN / black output without running memory-exploding recovery loops.
- No hardcoded Hugging Face repo IDs in code.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from worker.engines.base import DegenerateOutputError, Engine, UnsupportedGPUError
from worker.memory import (
    clear_gpu_memory,
    detect_gpus,
    get_recommended_torch_dtype,
    log_host_ram_stage,
)
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
            # Check if all channels have 0 dynamic range
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
        gpus = detect_gpus()
        target_device = device or ("cuda" if (gpus and os.environ.get("CUDA_VISIBLE_DEVICES") != "-1") else "cpu")

        # 1. Compute capability gating: CC >= 8.0 required for native bf16 before loading
        allow_slow_bf16 = (
            self.config.get("allow_slow_bf16", False)
            or os.environ.get("STUDIO_ALLOW_SLOW_BF16") in ("1", "true", "True")
        )

        if gpus and target_device.startswith("cuda"):
            gpu = gpus[0]
            if gpu.compute_capability < self.min_compute_capability:
                if not allow_slow_bf16:
                    raise UnsupportedGPUError(
                        f"Engine '{self.engine_id}' requires GPU compute capability >= "
                        f"{self.min_compute_capability[0]}.{self.min_compute_capability[1]} with native bfloat16 support "
                        f"(detected '{gpu.name}' with compute capability {gpu.compute_capability[0]}.{gpu.compute_capability[1]}). "
                        f"Running in fp16 produces NaN/black output. "
                        f"Pass --allow-slow-bf16 or set STUDIO_ALLOW_SLOW_BF16=1 to force software bf16 emulation with offload and RAM guard."
                    )
                else:
                    logger.warning(
                        "[SLOW BF16 OVERRIDE] Running %s on %s (CC %d.%d) with software bfloat16 emulation and offload.",
                        self.engine_id, gpu.name, gpu.compute_major, gpu.compute_minor,
                    )

        # 2. Enforce Host RAM guard before allocating any weights
        self.check_ram_guard()

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

        # Determine target dtype: native bf16 on CC>=8.0, software bf16 if overridden, else recommended
        if dtype is not None:
            target_dtype = dtype
        elif allow_slow_bf16:
            target_dtype = torch.bfloat16
        else:
            rec = get_recommended_torch_dtype(gpus[0] if gpus else None)
            target_dtype = rec if rec == torch.bfloat16 else (torch.float16 if target_device.startswith("cuda") else torch.float32)

        logger.info(
            "Loading Z-Image-Turbo from %s (rev: %s, dtype: %s, device: %s)...",
            repo_id,
            revision,
            target_dtype,
            target_device,
        )

        clear_gpu_memory()
        log_host_ram_stage(f"pre-load of {self.engine_id}")

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
        log_host_ram_stage(f"post-load of {self.engine_id}")

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
        fmt = (getattr(item, "output_format", None) or (item.extra_params.get("output_format") if hasattr(item, "extra_params") else None) or "png").lower()
        ext = "webp" if fmt == "webp" else "png"
        target_file = output_dir / f"{item.id}.{ext}"

        w = ((item.width // 16) * 16) if item.width else 1024
        h = ((item.height // 16) * 16) if item.height else 1024

        # Primary inference execution
        image = self._execute_inference(item.prompt, w, h, steps, guidance, generator)

        # NaN / Black output check: raise DegenerateOutputError without host-RAM bloating recovery loops
        if _is_black_or_degenerate_image(image):
            raise DegenerateOutputError(
                f"Engine '{self.engine_id}' produced NaN or pure black output. "
                f"This indicates numerical instability in fp16 on compute capability < 8.0 hardware."
            )

        if fmt == "webp":
            image.save(target_file, format="WEBP", quality=95)
        else:
            image.save(target_file, format="PNG")
        return target_file

    def unload(self) -> None:
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        self.is_loaded = False
        clear_gpu_memory()
