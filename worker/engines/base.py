"""worker/engines/base.py ? Abstract Base Class for diffusers inference engines.

Enforces:
- diffusers pipeline optimizations (VAE tiling, text-encoder CPU offload) idempotently.
- fp16 on Tesla T4 and P100 (never bf16).
- Clear error when GPU VRAM or host RAM is insufficient.
- Configuration loaded from worker/models.yaml as single source of truth.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from worker.memory import (
    check_host_ram_guard,
    clear_gpu_memory,
    detect_gpus,
    get_recommended_torch_dtype,
    log_host_ram_stage,
)
from worker.schema import Item

logger = logging.getLogger(__name__)


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


class UnsupportedGPUError(RuntimeError):
    """Raised when GPU hardware does not satisfy model architecture requirements (e.g. CC < 8.0)."""


class DegenerateOutputError(RuntimeError):
    """Raised when image or video asset generation produces NaN, Inf, or pure black output."""


def safe_enable_vae_tiling(pipe: Any) -> bool:
    """Safely enable VAE tiling across various diffusers versions and architectures."""
    try:
        if getattr(pipe, "_vae_tiling_enabled", False):
            return True
        if hasattr(pipe, "enable_vae_tiling") and callable(getattr(pipe, "enable_vae_tiling")):
            pipe.enable_vae_tiling()
            pipe._vae_tiling_enabled = True
            return True
        if hasattr(pipe, "vae") and pipe.vae is not None:
            if hasattr(pipe.vae, "enable_tiling") and callable(getattr(pipe.vae, "enable_tiling")):
                pipe.vae.enable_tiling()
                pipe._vae_tiling_enabled = True
                return True
            if hasattr(pipe.vae, "enable_vae_tiling") and callable(getattr(pipe.vae, "enable_vae_tiling")):
                pipe.vae.enable_vae_tiling()
                pipe._vae_tiling_enabled = True
                return True
    except Exception as exc:
        logger.warning("Could not enable VAE tiling: %s", exc)
    return False


def safe_enable_vae_slicing(pipe: Any) -> bool:
    """Safely enable VAE slicing across various diffusers versions and architectures."""
    try:
        if getattr(pipe, "_vae_slicing_enabled", False):
            return True
        if hasattr(pipe, "enable_vae_slicing") and callable(getattr(pipe, "enable_vae_slicing")):
            pipe.enable_vae_slicing()
            pipe._vae_slicing_enabled = True
            return True
        if hasattr(pipe, "vae") and pipe.vae is not None:
            if hasattr(pipe.vae, "enable_slicing") and callable(getattr(pipe.vae, "enable_slicing")):
                pipe.vae.enable_slicing()
                pipe._vae_slicing_enabled = True
                return True
            if hasattr(pipe.vae, "enable_vae_slicing") and callable(getattr(pipe.vae, "enable_vae_slicing")):
                pipe.vae.enable_vae_slicing()
                pipe._vae_slicing_enabled = True
                return True
    except Exception as exc:
        logger.warning("Could not enable VAE slicing: %s", exc)
    return False


def safe_apply_gpu_optimizations(pipe: Any, target_device: str = "cuda", gpu_id: int = 0) -> None:
    """Apply VAE tiling, slicing, and CPU offload idempotently without duplicating hooks."""
    if pipe is None:
        return

    # Idempotency guard: never apply twice to avoid Accelerate hook duplication & RAM explosion
    if getattr(pipe, "_gpu_optimizations_applied", False):
        logger.debug("safe_apply_gpu_optimizations already applied to pipeline; skipping duplicate application.")
        return

    if target_device.startswith("cuda"):
        safe_enable_vae_tiling(pipe)
        safe_enable_vae_slicing(pipe)

        # Check if offload hooks already installed or device map managed
        already_offloaded = (
            getattr(pipe, "_offload_hook_installed", False) is True
            or getattr(pipe, "is_model_cpu_offload", False) is True
            or (isinstance(getattr(pipe, "hf_device_map", None), dict) and bool(pipe.hf_device_map))
        )

        if not already_offloaded:
            try:
                if hasattr(pipe, "enable_model_cpu_offload") and callable(getattr(pipe, "enable_model_cpu_offload")):
                    try:
                        pipe.enable_model_cpu_offload(gpu_id=gpu_id)
                    except TypeError:
                        pipe.enable_model_cpu_offload()
                    pipe._offload_hook_installed = True
                    log_host_ram_stage(f"after enable_model_cpu_offload(gpu_id={gpu_id})")
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

    pipe._gpu_optimizations_applied = True


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

        # Configuration read from worker/models.yaml (single source of truth)
        self.repo_id: Optional[str] = self.config.get("repo_id")
        self.revision: Optional[str] = self.config.get("revision")
        self.files: List[str] = list(self.config.get("files") or [])
        self.recommended_dtype: Optional[str] = self.config.get("recommended_dtype")
        self.default_steps: int = int(self.config.get("default_steps", 4))
        self.default_guidance_scale: float = float(self.config.get("default_guidance_scale", 0.0))
        self.ram_estimate_gb: float = float(self.config.get("ram_estimate_gb", 12.0))
        self.requires_native_bf16: bool = bool(self.config.get("requires_native_bf16", False))

        min_cc = self.config.get("min_compute_capability")
        if isinstance(min_cc, (list, tuple)) and len(min_cc) >= 2:
            self.min_compute_capability: Tuple[int, int] = (int(min_cc[0]), int(min_cc[1]))
        elif isinstance(min_cc, (int, float)):
            major = int(min_cc)
            minor = int(round((min_cc - major) * 10))
            self.min_compute_capability = (major, minor)
        else:
            self.min_compute_capability = (0, 0)

        if "min_vram_gb" in self.config:
            try:
                self.min_vram_gb = float(self.config["min_vram_gb"])
            except (ValueError, TypeError):
                pass

    def check_vram_guard(self, item: Item) -> None:
        """Verify GPU has enough VRAM capacity before loading or generating."""
        gpus = detect_gpus()
        if not gpus:
            return

        total_gb = gpus[0].total_vram_mb / 1024.0
        free_gb = gpus[0].free_vram_mb / 1024.0

        # 1. Total hardware capacity check: verify GPU has enough total capacity for model
        if total_gb < self.min_vram_gb * 0.85:
            raise InsufficientVRAMError(self.engine_id, self.min_vram_gb, total_gb)

        # 2. Before model is loaded: verify enough free VRAM is available to allocate weights
        if not self.is_loaded:
            req_gb = max(self.min_vram_gb, self.vram_estimate(item))
            if free_gb < req_gb * 0.85:
                raise InsufficientVRAMError(self.engine_id, req_gb, free_gb)
        else:
            # 3. Model is ALREADY loaded: verify minimal working headroom (0.5 GB) remains for activations
            min_working_headroom_gb = 0.5
            if free_gb < min_working_headroom_gb:
                raise InsufficientVRAMError(self.engine_id, min_working_headroom_gb, free_gb)

    def check_ram_guard(self) -> None:
        """Verify host RAM capacity is sufficient before loading weights."""
        check_host_ram_guard(self.engine_id, self.ram_estimate_gb)

    def apply_pipeline_optimizations(self, target_device: str, gpu_id: int = 0) -> None:
        """Safely apply diffusers optimizations idempotently without crashing if methods are missing."""
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
