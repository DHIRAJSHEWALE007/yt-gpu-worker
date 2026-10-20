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
            # Running on CPU or mock environment
            return

        free_gb = gpus[0].free_vram_mb / 1024.0
        req_gb = max(self.min_vram_gb, self.vram_estimate(item))
        if free_gb < req_gb * 0.85:  # Allow 15% tolerance buffer
            raise InsufficientVRAMError(self.engine_id, req_gb, free_gb)

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
