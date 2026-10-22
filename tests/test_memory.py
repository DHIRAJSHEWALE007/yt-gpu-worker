from pathlib import Path
import pytest
"""tests/test_memory.py — Memory rules, GPU detection, and OOM exception detection."""

from worker.memory import (
    GPUInfo,
    clear_gpu_memory,
    detect_gpus,
    get_system_ram_mb,
    is_oom_exception,
)


def test_hardware_detection_runs() -> None:
    # Runs safely on any machine (CPU or GPU)
    gpus = detect_gpus()
    assert isinstance(gpus, list)

    total_ram, avail_ram = get_system_ram_mb()
    assert total_ram > 0
    assert avail_ram > 0


def test_gpu_info_dtype_rules() -> None:
    # 1. Tesla T4 (compute 7.5): MUST be fp16, never bf16
    t4 = GPUInfo(index=0, name="Tesla T4", total_vram_mb=15109, free_vram_mb=14000, compute_capability=(7, 5))
    assert t4.supports_bf16 is False
    assert t4.recommended_dtype == "fp16"

    # 2. Tesla P100 (compute 6.0): MUST be fp16, never bf16
    p100 = GPUInfo(index=0, name="Tesla P100-PCIE-16GB", total_vram_mb=16280, free_vram_mb=15000, compute_capability=(6, 0))
    assert p100.supports_bf16 is False
    assert p100.recommended_dtype == "fp16"

    # 3. Ampere A100 (compute 8.0): Supports bf16
    a100 = GPUInfo(index=0, name="NVIDIA A100-SXM4-40GB", total_vram_mb=40960, free_vram_mb=39000, compute_capability=(8, 0))
    assert a100.supports_bf16 is True
    assert a100.recommended_dtype == "bf16"


def test_oom_exception_detection() -> None:
    assert is_oom_exception(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"))
    assert is_oom_exception(Exception("cuda oom encountered during forward pass"))
    assert not is_oom_exception(ValueError("Invalid shape"))
    assert not is_oom_exception(FileNotFoundError("Model weights not found"))

def test_ram_guard_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify check_host_ram_guard raises InsufficientRAMError when estimate > 85% of available RAM."""
    import pytest
    from worker.memory import HostRAMInfo, InsufficientRAMError, check_host_ram_guard

    # Mock available RAM to 10.0 GB (threshold at 85% = 8.5 GB)
    mock_ram = HostRAMInfo(total_ram_gb=16.0, available_ram_gb=10.0, used_ram_gb=6.0)
    monkeypatch.setattr("worker.memory.get_host_ram_info", lambda: mock_ram)

    # 1. Estimate of 9.0 GB exceeds 8.5 GB -> Refused
    with pytest.raises(InsufficientRAMError, match="exceeds 85% of available host RAM"):
        check_host_ram_guard("large_engine", ram_estimate_gb=9.0)

    # 2. Estimate of 5.0 GB is within limits -> Allowed
    check_host_ram_guard("small_engine", ram_estimate_gb=5.0)


def test_synthetic_meminfo_and_cgroup_limits(tmp_path: Path) -> None:
    """Verify get_host_ram_info accurately computes MemAvailable and respects cgroup constraints."""
    from worker.memory import get_host_ram_info, parse_proc_meminfo

    # 1. Synthetic /proc/meminfo with large cache (cache-inflated used)
    # MemTotal: ~31.36 GB, MemFree: ~1.15 GB, MemAvailable: ~27.14 GB, Cached: ~22.88 GB
    synthetic_meminfo = (
        "MemTotal:       32882200 kB\n"
        "MemFree:         1204800 kB\n"
        "MemAvailable:   28456000 kB\n"
        "Buffers:          512000 kB\n"
        "Cached:         24000000 kB\n"
        "SwapTotal:             0 kB\n"
        "SwapFree:              0 kB\n"
    )
    mem_file = tmp_path / "proc_meminfo"
    mem_file.write_text(synthetic_meminfo, encoding="utf-8")

    parsed = parse_proc_meminfo(synthetic_meminfo)
    assert parsed["MemTotal"] == 32882200 * 1024
    assert parsed["MemAvailable"] == 28456000 * 1024

    # Test baseline without cgroup constraints
    ram_unconstrained = get_host_ram_info(meminfo_path=mem_file)
    assert ram_unconstrained.total_ram_gb == round(32882200 * 1024 / (1024**3), 2)  # ~30.62 GB
    assert ram_unconstrained.available_ram_gb == round(28456000 * 1024 / (1024**3), 2)  # ~26.50 GB

    # 2. Synthetic cgroup v2 constraint (e.g. 16 GB container limit with 4 GB used)
    # Limit: 16 GiB = 17179869184 bytes, Usage: 4 GiB = 4294967296 bytes
    # Available inside container = 16 - 4 = 12 GiB, which is lower than MemAvailable (26.5 GB)
    cg_limit = tmp_path / "cgroup_max"
    cg_usage = tmp_path / "cgroup_current"
    cg_limit.write_text("17179869184\n", encoding="utf-8")
    cg_usage.write_text("4294967296\n", encoding="utf-8")

    ram_cgroup = get_host_ram_info(
        meminfo_path=mem_file,
        cgroup_limit_path=cg_limit,
        cgroup_usage_path=cg_usage,
    )
    assert ram_cgroup.total_ram_gb == 16.0  # Clamped to cgroup limit
    assert ram_cgroup.available_ram_gb == 12.0  # Clamped to cgroup quota (16 - 4)

    # 3. Verify string representation logs available and total only
    ram_repr = repr(ram_cgroup)
    assert "avail=12.00GB / total=16.00GB" in ram_repr
    assert "used=" not in ram_repr
