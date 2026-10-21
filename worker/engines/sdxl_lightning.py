"""worker/engines/sdxl_lightning.py — SDXL Lightning image engine driven by models.yaml.

Architecture:
- Builds UNet2DConditionModel from base repo config (stabilityai/stable-diffusion-xl-base-1.0).
- Loads ByteDance/SDXL-Lightning sdxl_lightning_4step_unet.safetensors into UNet in fp16.
- Loads madebyollin/sdxl-vae-fp16-fix (MIT license, pinned revision) once at load to prevent NaN overflows
  and stop diffusers per-image upcast_vae warning spam.
- Passes patched UNet and fp16-safe VAE to StableDiffusionXLPipeline.
- Uses EulerDiscreteScheduler with timestep_spacing='trailing'.
- Enforces guidance_scale 0.0 and exactly 4 steps.
- Raises DegenerateOutputError if output is black or degenerate.
- Zero hardcoded Hugging Face repo IDs in code.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from worker.engines.base import DegenerateOutputError, Engine
from worker.memory import (
    clear_gpu_memory,
    detect_gpus,
    get_recommended_torch_dtype,
    log_host_ram_stage,
)
from worker.schema import Item

logger = logging.getLogger(__name__)


def _is_black_image(image: Any) -> bool:
    """Check if PIL Image is pure black."""
    try:
        extrema = image.getextrema()
        if isinstance(extrema, tuple) and all(ch[1] == 0 for ch in extrema):
            return True
    except Exception:
        pass
    return False


class SDXLLightningEngine(Engine):
    """SDXL-Lightning high-speed image generation driven by models.yaml."""

    engine_id: str = "sdxl_lightning"
    min_vram_gb: float = 8.0

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self.lightning_weights_loaded: bool = False

    def load(self, device: Optional[str] = None, dtype: Optional[Any] = None) -> None:
        self.check_ram_guard()

        try:
            import torch
            from diffusers import (
                AutoencoderKL,
                EulerDiscreteScheduler,
                StableDiffusionXLPipeline,
                UNet2DConditionModel,
            )
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file
        except ImportError as e:
            raise RuntimeError(f"diffusers, safetensors, huggingface_hub, and torch are required: {e}")

        base_repo = self.config.get("base_model")
        repo_id = self.repo_id or self.config.get("repo_id")
        if not base_repo or not repo_id:
            raise ValueError(f"Engine '{self.engine_id}' requires 'base_model' and 'repo_id' in models.yaml.")

        base_rev = self.config.get("base_revision")
        lightning_rev = self.revision or self.config.get("revision")
        weight_file = self.files[0] if self.files else "sdxl_lightning_4step_unet.safetensors"

        gpus = detect_gpus()
        target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        target_dtype = dtype or get_recommended_torch_dtype(gpus[0] if gpus else None)

        clear_gpu_memory()
        log_host_ram_stage(f"pre-load of {self.engine_id}")

        logger.info(
            "Building UNet from %s config and loading %s from %s in %s...",
            base_repo,
            weight_file,
            repo_id,
            target_dtype,
        )

        # 1. Build UNet2DConditionModel from base model configuration
        if hasattr(UNet2DConditionModel, "load_config") and callable(getattr(UNet2DConditionModel, "load_config")):
            unet_config = UNet2DConditionModel.load_config(
                base_repo,
                subfolder="unet",
                revision=base_rev,
            )
            unet = UNet2DConditionModel.from_config(unet_config).to(target_device, dtype=target_dtype)
        else:
            unet = UNet2DConditionModel.from_config(
                base_repo,
                subfolder="unet",
                revision=base_rev,
            ).to(target_device, dtype=target_dtype)

        # 2. Download and load ByteDance SDXL-Lightning UNet safetensors weights
        unet_weight_path = hf_hub_download(
            repo_id=repo_id,
            filename=weight_file,
            revision=lightning_rev,
        )
        state_dict = load_file(unet_weight_path)
        unet.load_state_dict(state_dict)
        self.lightning_weights_loaded = True
        log_host_ram_stage("after UNet weights loaded")

        # 3. Load fp16-safe VAE once at load time to prevent NaNs and stop upcast_vae warning spam
        vae = None
        vae_fix_model = self.config.get("vae_fix_model")
        vae_fix_rev = self.config.get("vae_fix_revision", "207b116dae70ace3637169f1ddd2434b91b3a8cd")
        if vae_fix_model:
            try:
                logger.info("Loading fixed VAE from %s (rev: %s) in %s...", vae_fix_model, vae_fix_rev, target_dtype)
                vae = AutoencoderKL.from_pretrained(
                    vae_fix_model,
                    revision=vae_fix_rev,
                    torch_dtype=target_dtype,
                ).to(target_device)
                log_host_ram_stage("after fixed VAE loaded")
            except Exception as vae_err:
                logger.warning("Could not load fixed VAE from %s (%s); falling back to base model VAE", vae_fix_model, vae_err)

        # 4. Instantiate pipeline with the patched UNet and fixed VAE
        pipe_kwargs: Dict[str, Any] = {
            "unet": unet,
            "revision": base_rev,
            "torch_dtype": target_dtype,
            "variant": "fp16" if target_dtype == torch.float16 else None,
        }
        if vae is not None:
            pipe_kwargs["vae"] = vae

        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            base_repo,
            **pipe_kwargs,
        )

        # If custom VAE wasn't loaded, keep the base VAE in float32 once at load to prevent per-image warning spam
        if vae is None and hasattr(self.pipe, "upcast_vae"):
            try:
                self.pipe.upcast_vae()
                logger.info("Base VAE upcast to float32 once at load time.")
            except Exception:
                pass

        # 5. Use EulerDiscreteScheduler with timestep_spacing='trailing'
        self.pipe.scheduler = EulerDiscreteScheduler.from_config(
            self.pipe.scheduler.config,
            timestep_spacing="trailing",
        )

        gpu_id = 0
        if target_device.startswith("cuda:") and target_device.split(":")[1].isdigit():
            gpu_id = int(target_device.split(":")[1])

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

        # SDXL-Lightning 4-step requires exactly 4 steps and guidance_scale 0.0
        steps = 4
        guidance = 0.0

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
            negative_prompt=item.negative_prompt or None,
            width=w,
            height=h,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
        )

        image = result.images[0]

        if _is_black_image(image):
            raise DegenerateOutputError(f"Engine '{self.engine_id}' produced black image output.")

        image.save(target_file, format="PNG")
        return target_file

    def unload(self) -> None:
        del self.pipe
        self.pipe = None
        self.is_loaded = False
        clear_gpu_memory()
