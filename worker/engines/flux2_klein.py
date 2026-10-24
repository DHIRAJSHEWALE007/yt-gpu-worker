"""worker/engines/flux2_klein.py ? FLUX rectified flow image engine driven by models.yaml.

Features:
- Dynamically inspects installed diffusers version.
- Resolves pipeline class matching model_index.json (Flux2KleinPipeline with fallback to FluxPipeline / DiffusionPipeline).
- Quantizes only components that actually exist in the model architecture (e.g. transformer, text_encoder).
- Records diffusers_version and pipeline class metadata.
- Zero hardcoded Hugging Face repo IDs in code.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from worker.engines.base import Engine
from worker.memory import (
    clear_gpu_memory,
    detect_gpus,
    get_recommended_torch_dtype,
    log_host_ram_stage,
)
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
    """FLUX rectified flow pipeline driven by models.yaml."""

    engine_id: str = "flux2_klein"
    min_vram_gb: float = 10.0

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self.diffusers_version: str = "unknown"
        self.pipeline_class_name: str = "unknown"

    def load(self, device: Optional[str] = None, dtype: Optional[Any] = None) -> None:
        self.check_ram_guard()

        try:
            import torch
            import diffusers
            self.diffusers_version = getattr(diffusers, "__version__", "unknown")
        except ImportError as e:
            raise RuntimeError(f"diffusers and torch required for Flux2KleinEngine: {e}")

        repo_id = self.repo_id or self.config.get("repo_id")
        if not repo_id:
            raise ValueError(f"Engine '{self.engine_id}' has no repo_id configured in models.yaml.")
        revision = self.revision or self.config.get("revision")

        # Determine pipeline class matching model_index.json (Flux2KleinPipeline / FluxPipeline / DiffusionPipeline)
        pipe_cls = None
        for candidate_name in ("Flux2KleinPipeline", "FluxPipeline", "DiffusionPipeline"):
            if hasattr(diffusers, candidate_name):
                pipe_cls = getattr(diffusers, candidate_name)
                break

        if pipe_cls is None:
            raise RuntimeError(f"Could not find a compatible diffusers pipeline class for {repo_id}")

        clear_gpu_memory()

        gpus = detect_gpus()
        target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        target_dtype = dtype or get_recommended_torch_dtype(gpus[0] if gpus else None)

        token = _get_hf_token()
        logger.info(
            "Loading FLUX pipeline using %s from %s (diffusers: %s, rev: %s)...",
            pipe_cls.__name__,
            repo_id,
            self.diffusers_version,
            revision,
        )
        log_host_ram_stage(f"pre-load of {self.engine_id}")

        # Quantize ONLY components that exist in the architecture (transformer, text_encoder)
        pipeline_quant_config = None
        if target_device.startswith("cuda"):
            try:
                from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
                from diffusers.quantizers import PipelineQuantizationConfig
                from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig

                quant_map: Dict[str, Any] = {
                    "transformer": DiffusersBitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=target_dtype,
                    ),
                    "text_encoder": TransformersBitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=target_dtype,
                    ),
                }

                pipeline_quant_config = PipelineQuantizationConfig(quant_mapping=quant_map)
                logger.info("Enabled 4-bit NF4 quantization for existing components: %s", list(quant_map.keys()))
            except Exception as quant_err:
                logger.debug("Could not initialize bitsandbytes 4-bit config (%s), trying direct loading.", quant_err)

        loaded = False
        if pipeline_quant_config is not None:
            try:
                self.pipe = pipe_cls.from_pretrained(
                    repo_id,
                    revision=revision,
                    quantization_config=pipeline_quant_config,
                    torch_dtype=target_dtype,
                    token=token,
                    low_cpu_mem_usage=True,
                )
                loaded = True
            except Exception as e:
                logger.warning("PipelineQuantizationConfig loading failed (%s), falling back to standard offload...", e)

        if not loaded and target_device.startswith("cuda") and len(gpus) >= 2:
            try:
                max_mem = {g.index: f"{int(g.free_vram_mb * 0.9)}MB" for g in gpus}
                max_mem["cpu"] = "18GiB"
                self.pipe = pipe_cls.from_pretrained(
                    repo_id,
                    revision=revision,
                    torch_dtype=target_dtype,
                    device_map="balanced",
                    max_memory=max_mem,
                    token=token,
                    low_cpu_mem_usage=True,
                )
                loaded = True
            except Exception as e:
                logger.warning("device_map='balanced' loading failed (%s), trying standard loading...", e)

        if not loaded:
            try:
                self.pipe = pipe_cls.from_pretrained(
                    repo_id,
                    revision=revision,
                    torch_dtype=target_dtype,
                    token=token,
                    low_cpu_mem_usage=True,
                )
                loaded = True
            except Exception as exc:
                err_str = str(exc).lower()
                if "401" in err_str or "unauthorized" in err_str or "gated" in err_str:
                    raise RuntimeError(
                        f"\n[FLUX AUTHENTICATION REQUIRED]\n"
                        f"Model '{repo_id}' is restricted or gated. Verify HF_TOKEN."
                    ) from exc
                raise

        self.pipeline_class_name = self.pipe.__class__.__name__

        gpu_id = 0
        if target_device.startswith("cuda:") and target_device.split(":")[1].isdigit():
            gpu_id = int(target_device.split(":")[1])

        if not hasattr(self.pipe, "hf_device_map") or not self.pipe.hf_device_map:
            self.apply_pipeline_optimizations(target_device, gpu_id=gpu_id)

        log_host_ram_stage(f"post-load of {self.engine_id}")

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
        fmt = (getattr(item, "output_format", None) or (item.extra_params.get("output_format") if hasattr(item, "extra_params") else None) or "png").lower()
        ext = "webp" if fmt == "webp" else "png"
        target_file = output_dir / f"{item.id}.{ext}"

        w = ((item.width // 16) * 16) if item.width else 1024
        h = ((item.height // 16) * 16) if item.height else 1024

        kwargs = {
            "prompt": item.prompt,
            "width": w,
            "height": h,
            "num_inference_steps": steps,
            "generator": generator,
        }
        # Only pass guidance_scale if pipeline supports it
        try:
            result = self.pipe(**kwargs, guidance_scale=guidance)
        except TypeError:
            result = self.pipe(**kwargs)

        image = result.images[0]
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
