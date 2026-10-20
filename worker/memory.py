"""worker/memory.py — Hardware detection, VRAM budgeting, and OOM recovery logic.

Rules:
- Tesla T4 (CC 7.5) and Tesla P100 (CC 6.0) must use fp16 ONLY (never bf16).
- Ampere / Ada / Hopper (CC >= 8.0) support native bfloat16.
- Graceful OOM retry downscales resolution or enables offloading; never crashes the run.
- Aggressively clears GPU and CPU RAM (via malloc_trim) between models.
"""

from __future__ import annotations

import gc
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import psutil
except ImportError:
    psutil = None

logger = logging.getLogger(__name__)


class GPUInfo:
    """Hardware descriptor for a detected GPU."""

    def __init__(
        self,
        index: int,
        name: str,
        total_vram_mb: float,
        free_vram_mb: float,
        compute_capability: Tuple[int, int],
    ) -> None:
        self.index = index
        self.name = name
        self.total_vram_mb = total_vram_mb
        self.free_vram_mb = free_vram_mb
        self.compute_capability = compute_capability

    @property
    def compute_major(self) -> int:
        return self.compute_capability[0]

    @property
    def compute_minor(self) -> int:
        return self.compute_capability[1]

    @property
    def supports_bf16(self) -> bool:
        """True only on Ampere (8.0+) or newer. T4 (7.5) and P100 (6.0) are False."""
        return self.compute_major >= 8

    @property
    def recommended_dtype(self) -> str:
        """Enforces fp16 on T4/P100 and older architectures."""
        if self.supports_bf16:
            return "bf16"
        return "fp16"

    def __repr__(self) -> str:
        return (
            f"GPUInfo(idx={self.index}, name={self.name!r}, "
            f"vram={self.free_vram_mb:.0f}/{self.total_vram_mb:.0f}MB, "
            f"cc={self.compute_capability}, bf16={self.supports_bf16})"
        )


def detect_gpus() -> List[GPUInfo]:
    """Detect available CUDA GPUs and compute capabilities."""
    gpus: List[GPUInfo] = []
    try:
        import torch

        if not torch.cuda.is_available():
            return []

        count = torch.cuda.device_count()
        for idx in range(count):
            name = torch.cuda.get_device_name(idx)
            cc = torch.cuda.get_device_capability(idx)
            free_bytes, total_bytes = torch.cuda.mem_get_info(idx)
            gpus.append(
                GPUInfo(
                    index=idx,
                    name=name,
                    total_vram_mb=total_bytes / (1024 * 1024),
                    free_vram_mb=free_bytes / (1024 * 1024),
                    compute_capability=cc,
                )
            )
    except Exception as e:
        logger.debug("GPU detection fallback: %s", e)

    return gpus


def get_system_ram_mb() -> Tuple[float, float]:
    """Return (total_ram_mb, available_ram_mb) for the host machine."""
    if psutil is not None:
        try:
            mem = psutil.virtual_memory()
            return mem.total / (1024 * 1024), mem.available / (1024 * 1024)
        except Exception:
            pass
    # Fallback default values
    return 16384.0, 8192.0


def clear_gpu_memory() -> None:
    """Aggressively collect Python garbage, clear PyTorch CUDA caches, and release system RAM."""
    gc.collect()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass

    # Free dirty memory pages back to the Linux host OS
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass


def is_oom_exception(exc: BaseException) -> bool:
    """Return True if exception indicates a CUDA Out Of Memory condition."""
    msg = str(exc).lower()
    if "out of memory" in msg or "cuda oom" in msg or "cudaoom" in msg:
        return True
    try:
        import torch
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except Exception:
        pass
    return False


def get_recommended_torch_dtype(gpu: Optional[GPUInfo] = None) -> Any:
    """Return appropriate torch.dtype based on GPU architecture rules."""
    try:
        import torch
    except ImportError:
        return "fp16"

    if gpu is None:
        gpus = detect_gpus()
        gpu = gpus[0] if gpus else None

    if gpu is None:
        return torch.float32

    # Enforce fp16 on T4 / P100
    if not gpu.supports_bf16:
        return torch.float16

    return torch.bfloat16
