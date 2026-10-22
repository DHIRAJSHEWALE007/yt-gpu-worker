"""worker/memory.py ? Hardware detection, VRAM/RAM budgeting, and OOM recovery logic.

Rules:
- Tesla T4 (CC 7.5) and Tesla P100 (CC 6.0) must use fp16 ONLY (never bf16).
- Ampere / Ada / Hopper (CC >= 8.0) support native bfloat16.
- Host RAM Guard: Reads cgroups limit or /proc/meminfo; refuses loading when
  engine ram_estimate_gb exceeds 85% of available host RAM.
- Graceful OOM retry downscales resolution or enables offloading; never crashes the run.
- Aggressively clears GPU and CPU RAM (via malloc_trim) between models.
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
import logging
import os
from pathlib import Path
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import psutil
except ImportError:
    psutil = None

logger = logging.getLogger(__name__)


class InsufficientRAMError(RuntimeError):
    """Raised when estimated model RAM exceeds 85% of available host RAM."""

    def __init__(self, engine_id: str, required_gb: float, available_gb: float, threshold_gb: float) -> None:
        super().__init__(
            f"Engine '{engine_id}' estimated RAM requirement ({required_gb:.1f} GB) "
            f"exceeds 85% of available host RAM ({threshold_gb:.1f} GB of {available_gb:.1f} GB free). "
            f"Refusing to load to prevent host-RAM exhaustion and kernel OOM kills."
        )
        self.engine_id = engine_id
        self.required_gb = required_gb
        self.available_gb = available_gb
        self.threshold_gb = threshold_gb


@dataclass
class HostRAMInfo:
    """Host RAM status in Gigabytes based on MemAvailable."""
    total_ram_gb: float
    available_ram_gb: float
    used_ram_gb: float = 0.0

    def __post_init__(self) -> None:
        if not self.used_ram_gb:
            self.used_ram_gb = max(0.0, round(self.total_ram_gb - self.available_ram_gb, 2))

    def __repr__(self) -> str:
        return f"HostRAM(avail={self.available_ram_gb:.2f}GB / total={self.total_ram_gb:.2f}GB)"


def parse_proc_meminfo(meminfo_text: str) -> Dict[str, int]:
    """Parse /proc/meminfo text into a dictionary of metric keys in bytes."""
    data: Dict[str, int] = {}
    for line in meminfo_text.splitlines():
        if ":" in line:
            parts = line.split(":", 1)
            k = parts[0].strip()
            v_str = parts[1].strip().split()[0]
            if v_str.isdigit():
                # Values in /proc/meminfo are in kB
                data[k] = int(v_str) * 1024
    return data


def get_host_ram_info(
    meminfo_path: Optional[Path] = None,
    cgroup_limit_path: Optional[Path] = None,
    cgroup_usage_path: Optional[Path] = None,
) -> HostRAMInfo:
    """Read host RAM available and total capacity from /proc/meminfo and cgroups.

    Guarantees:
      - Total RAM is based on MemTotal or the cgroup limit when lower.
      - Available RAM is based on MemAvailable (not cache-inflated used), bounded by cgroup limit.
      - When cgroup limits are active, available = min(MemAvailable, cgroup_limit - cgroup_usage).
      - Logs and guards rely on available and total only.
    """
    total_bytes: Optional[int] = None
    available_bytes: Optional[int] = None
    cgroup_limit_bytes: Optional[int] = None
    cgroup_usage_bytes: Optional[int] = None

    # 1. Read /proc/meminfo (system-wide physical metrics)
    m_path = meminfo_path or Path("/proc/meminfo")
    if m_path.is_file():
        try:
            mem_data = parse_proc_meminfo(m_path.read_text(encoding="utf-8"))
            if "MemTotal" in mem_data:
                total_bytes = mem_data["MemTotal"]
            if "MemAvailable" in mem_data:
                available_bytes = mem_data["MemAvailable"]
        except Exception as e:
            logger.debug("Could not read %s: %s", m_path, e)

    # 2. Check cgroups v2 / v1 limits
    lim_path = cgroup_limit_path
    if lim_path is None:
        cg2 = Path("/sys/fs/cgroup/memory.max")
        cg1 = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
        if cg2.is_file():
            lim_path = cg2
        elif cg1.is_file():
            lim_path = cg1

    if lim_path and lim_path.is_file():
        try:
            val_str = lim_path.read_text(encoding="utf-8").strip()
            if val_str != "max" and val_str.isdigit():
                val = int(val_str)
                # Ignore unconstrained default (~2^63 - 4096)
                if val < (1 << 60):
                    cgroup_limit_bytes = val
        except Exception:
            pass

    # Check cgroups usage
    use_path = cgroup_usage_path
    if use_path is None:
        cg2_use = Path("/sys/fs/cgroup/memory.current")
        cg1_use = Path("/sys/fs/cgroup/memory/memory.usage_in_bytes")
        if cg2_use.is_file():
            use_path = cg2_use
        elif cg1_use.is_file():
            use_path = cg1_use

    if use_path and use_path.is_file():
        try:
            val_str = use_path.read_text(encoding="utf-8").strip()
            if val_str.isdigit():
                cgroup_usage_bytes = int(val_str)
        except Exception:
            pass

    # 3. Apply cgroup constraint if cgroup limit is lower than MemTotal
    if cgroup_limit_bytes is not None:
        if total_bytes is None or cgroup_limit_bytes < total_bytes:
            total_bytes = cgroup_limit_bytes

        if cgroup_usage_bytes is not None:
            cgroup_avail = max(0, cgroup_limit_bytes - cgroup_usage_bytes)
            if available_bytes is not None:
                available_bytes = min(available_bytes, cgroup_avail)
            else:
                available_bytes = cgroup_avail

    # 4. Fallback to psutil (Windows / macOS or environments without /proc/meminfo)
    if total_bytes is None or available_bytes is None:
        if psutil is not None:
            try:
                vm = psutil.virtual_memory()
                if total_bytes is None:
                    total_bytes = vm.total
                if available_bytes is None:
                    available_bytes = vm.available
            except Exception:
                pass

    # Defaults if completely unavailable
    if total_bytes is None:
        total_bytes = 16 * 1024 * 1024 * 1024
    if available_bytes is None:
        available_bytes = total_bytes // 2

    gb = 1024.0 * 1024.0 * 1024.0
    return HostRAMInfo(
        total_ram_gb=round(total_bytes / gb, 2),
        available_ram_gb=round(available_bytes / gb, 2),
    )


def log_host_ram_stage(stage_name: str) -> None:
    """Log available and total host RAM at given load/execution stage (never cache-inflated used)."""
    ram = get_host_ram_info()
    logger.info(
        "[HOST RAM %s] Available: %.2f GB / Total: %.2f GB",
        stage_name,
        ram.available_ram_gb,
        ram.total_ram_gb,
    )


def check_host_ram_guard(engine_id: str, ram_estimate_gb: float, safety_ratio: float = 0.85) -> None:
    """Verify available host RAM is sufficient before allocating or offloading model weights."""
    ram = get_host_ram_info()
    threshold = ram.available_ram_gb * safety_ratio
    log_host_ram_stage(f"pre-load guard check for '{engine_id}' (requires ~{ram_estimate_gb:.1f} GB)")

    if ram_estimate_gb > threshold:
        raise InsufficientRAMError(engine_id, ram_estimate_gb, ram.available_ram_gb, threshold)


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
    ram = get_host_ram_info()
    return ram.total_ram_gb * 1024.0, ram.available_ram_gb * 1024.0


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
