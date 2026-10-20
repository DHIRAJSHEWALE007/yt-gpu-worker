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
    min_vram_gb: float = 10.0

    def load(self, device: Optional[str] = None, dtype: Optional[Any] = None) -> None:
        try:
            import torch
            from diffusers import FluxPipeline
        except ImportError as e:
            raise RuntimeError(f"diffusers>=0.30.0 and torch required for Flux2KleinEngine: {e}")

        # Ensure any residual host and GPU memory from prior models is released
        clear_gpu_memory()

        gpus = detect_gpus()
        target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        target_dtype = dtype or get_recommended_torch_dtype(gpus[0] if gpus else None)

        repo_id = self.config.get("repo_id", "black-forest-labs/FLUX.1-schnell")
        token = _get_hf_token()

        logger.info("Loading FLUX.1 pipeline from %s (authenticated: %s)...", repo_id, bool(token))

        # Check if 4-bit NF4 quantization config can be built to prevent host RAM OOM
        pipeline_quant_config = None
        if target_device.startswith("cuda"):
            try:
                from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
                from diffusers.quantizers import PipelineQuantizationConfig
                from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig

                pipeline_quant_config = PipelineQuantizationConfig(
                    quant_mapping={
                        "transformer": DiffusersBitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=target_dtype,
                        ),
                        "text_encoder_2": TransformersBitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=target_dtype,
                        ),
                    }
                )
                logger.info("Enabled 4-bit NF4 quantization for FLUX transformer & T5-XXL to fit in RAM/VRAM.")
            except Exception as quant_err:
                logger.debug("Could not initialize bitsandbytes 4-bit config (%s), trying balanced loading.", quant_err)

        loaded = False
        # Strategy 1: Load with 4-bit PipelineQuantizationConfig (avoids Host RAM OOM by quantizing weights on the fly)
        if pipeline_quant_config is not None:
            try:
                self.pipe = FluxPipeline.from_pretrained(
                    repo_id,
                    quantization_config=pipeline_quant_config,
                    torch_dtype=target_dtype,
                    token=token,
                    low_cpu_mem_usage=True,
                )
                loaded = True
            except Exception as e:
                logger.warning("PipelineQuantizationConfig loading failed (%s), falling back...", e)

        # Strategy 2: Multi-GPU / Balanced loading if 2+ GPUs are detected (e.g. Kaggle dual-T4)
        if not loaded and target_device.startswith("cuda") and len(gpus) >= 2:
            try:
                max_mem = {g.index: f"{int(g.free_vram_mb * 0.9)}MB" for g in gpus}
                max_mem["cpu"] = "18GiB"
                logger.info("Attempting multi-GPU balanced loading across %d GPUs with max_memory: %s", len(gpus), max_mem)
                self.pipe = FluxPipeline.from_pretrained(
                    repo_id,
                    torch_dtype=target_dtype,
                    device_map="balanced",
                    max_memory=max_mem,
                    token=token,
                    low_cpu_mem_usage=True,
                )
                loaded = True
            except Exception as e:
                logger.warning("device_map='balanced' loading failed (%s), trying standard offload...", e)

        # Strategy 3: Standard low_cpu_mem_usage loading
        if not loaded:
            try:
                self.pipe = FluxPipeline.from_pretrained(
                    repo_id,
                    torch_dtype=target_dtype,
                    token=token,
                    low_cpu_mem_usage=True,
                )
                loaded = True
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

        # If pipe was loaded via device_map, Accelerate already manages device placement
        if not hasattr(self.pipe, "hf_device_map") or not self.pipe.hf_device_map:
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

        w = ((item.width // 16) * 16) if item.width else 1024
        h = ((item.height // 16) * 16) if item.height else 1024

        result = self.pipe(
            prompt=item.prompt,
            width=w,
            height=h,
            num_inference_steps=steps,
            max_sequence_length=256,
            generator=generator,
        )

        image = result.images[0]
        image.save(target_file, format="PNG")
        return target_file

    def unload(self) -> None:
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        self.is_loaded = False
        clear_gpu_memory()
