"""worker/engines/base.py — Abstract Base Class for diffusers inference engines.

Enforces:
- diffusers pipeline optimizations (VAE tiling, text-encoder CPU offload).
- fp16 on Tesla T4 and P100 (never bf16).
- Clear error when GPU VRAM is insufficient.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from worker.memory import clear_gpu_memory, detect_gpus, get_recommended_torch_dtype
from worker.schema import Item

logger = logging.getLogger(__name__)


def safe_enable_vae_tiling(pipe: Any) -> bool:
    """Safely enable VAE tiling across various diffusers versions and architectures."""
    try:
        if hasattr(pipe, "enable_vae_tiling") and callable(getattr(pipe, "enable_vae_tiling")):
            pipe.enable_vae_tiling()
            return True
        if hasattr(pipe, "vae") and pipe.vae is not None:
            if hasattr(pipe.vae, "enable_tiling") and callable(getattr(pipe.vae, "enable_tiling")):
                pipe.vae.enable_tiling()
                return True
            if hasattr(pipe.vae, "enable_vae_tiling") and callable(getattr(pipe.vae, "enable_vae_tiling")):
                pipe.vae.enable_vae_tiling()
                return True
    except Exception as exc:
        logger.warning("Could not enable VAE tiling: %s", exc)
    return False


def safe_enable_vae_slicing(pipe: Any) -> bool:
    """Safely enable VAE slicing across various diffusers versions and architectures."""
    try:
        if hasattr(pipe, "enable_vae_slicing") and callable(getattr(pipe, "enable_vae_slicing")):
            pipe.enable_vae_slicing()
            return True
        if hasattr(pipe, "vae") and pipe.vae is not None:
            if hasattr(pipe.vae, "enable_slicing") and callable(getattr(pipe.vae, "enable_slicing")):
                pipe.vae.enable_slicing()
                return True
            if hasattr(pipe.vae, "enable_vae_slicing") and callable(getattr(pipe.vae, "enable_vae_slicing")):
                pipe.vae.enable_vae_slicing()
                return True
    except Exception as exc:
        logger.warning("Could not enable VAE slicing: %s", exc)
    return False


def safe_apply_gpu_optimizations(pipe: Any, target_device: str = "cuda", gpu_id: int = 0) -> None:
    """Apply VAE tiling, slicing, and CPU offload or to(device) safely without crashing."""
    if target_device.startswith("cuda"):
        safe_enable_vae_tiling(pipe)
        safe_enable_vae_slicing(pipe)
        try:
            if hasattr(pipe, "enable_model_cpu_offload") and callable(getattr(pipe, "enable_model_cpu_offload")):
                try:
                    pipe.enable_model_cpu_offload(gpu_id=gpu_id)
                except TypeError:
                    pipe.enable_model_cpu_offload()
            else:
                pipe.to(target_device)
        except Exception as exc:
            logger.warning("enable_model_cpu_offload failed (%s); moving directly to %s", exc, target_device)
            try:
                pipe.to(target_device)
            except Exception as to_exc:
                logger.warning("pipe.to(%s) failed: %s", target_device, to_exc)
    else:
        try:
            pipe.to("cpu")
        except Exception as exc:
            logger.warning("pipe.to('cpu') failed: %s", exc)


class InsufficientVRAMError(RuntimeError):
    """Raised when GPU available VRAM is below the model minimum requirements."""

    def __init__(self, engine_id: str, required_gb: float, available_gb: float) -> None:
        super().__init__(
            f"Engine '{engine_id}' requires at least {required_gb:.1f} GB VRAM, "
            f"but currently only {available_gb:.1f} GB is free. Cannot allocate model."
        )
        self.engine_id = engine_id
        self.required_gb = required_gb
        self.available_gb = available_gb


class Engine(ABC):
    """Abstract interface for all image and video synthesis backends."""

    engine_id: str = "base"
    min_vram_gb: float = 8.0

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.is_loaded = False
        self.pipe: Any = None
        self.device = "cpu"
        self.dtype: Any = None

    def check_vram_guard(self, item: Item) -> None:
        """Verify GPU has enough free VRAM before attempting allocation."""
        gpus = detect_gpus()
        if not gpus:
            return

        free_gb = gpus[0].free_vram_mb / 1024.0
        req_gb = max(self.min_vram_gb, self.vram_estimate(item))
        if free_gb < req_gb * 0.85:
            raise InsufficientVRAMError(self.engine_id, req_gb, free_gb)

    def apply_pipeline_optimizations(self, target_device: str, gpu_id: int = 0) -> None:
        """Safely apply diffusers optimizations without crashing if methods are missing."""
        if self.pipe is not None:
            safe_apply_gpu_optimizations(self.pipe, target_device=target_device, gpu_id=gpu_id)

    @abstractmethod
    def load(self, device: Optional[str] = None, dtype: Optional[Any] = None) -> None:
        """Load model weights into GPU memory with required optimizations."""

    @abstractmethod
    def generate(self, item: Item, output_dir: Path) -> Path:
        """Synthesize image or video asset and return written file path."""

    @abstractmethod
    def unload(self) -> None:
        """Release pipeline from memory and clear GPU cache."""

    def vram_estimate(self, item: Item) -> float:
        """Estimated peak VRAM in GB for item parameters."""
        pixels = item.width * item.height
        base = self.min_vram_gb
        if pixels > 1024 * 1024:
            base += 2.0
        return base
