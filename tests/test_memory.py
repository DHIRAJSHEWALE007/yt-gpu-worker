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
