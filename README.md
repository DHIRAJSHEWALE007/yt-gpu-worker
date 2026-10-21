# yt-gpu-worker

Decoupled headless GPU worker designed for automated video production pipelines ([JackTheCreator](https://github.com/DHIRAJSHEWALE007/JackTheCreator)).

Runs autonomously on **Kaggle (dual T4)**, **Google Colab**, or self-hosted GPU server nodes to render images and video clips from structured batch packs.

---

## Features

- **Zero Studio Dependencies**: Interacts purely via Pydantic `Pack`, `Item`, and `Result` schemas (`v1.0.0`) or HTTP conforming to `REMOTE_CONTRACT.md`.
- **Hardware-Aware Memory & Compute Gating**:
  - Automatically detects GPU capabilities, VRAM, and system RAM.
  - Strictly enforces `fp16` on **Tesla T4** and **Tesla P100** (never native `bf16`).
  - Gated native `bf16` models (e.g. `zimage_turbo`) reject compute capability < 8.0 GPUs before loading weights to prevent NaNs and host-RAM exhaustion.
  - Host RAM guard reads cgroups and `/proc/meminfo`; refuses loading when estimated RAM exceeds 85% of available RAM.
  - Automatic 1-step downscale/offload retry on CUDA OOM; never crashes the batch.
- **Engine Tiers Governance**:
  - `hero_scenes` & `text_scenes` -> `flux2_klein` (role: `default_image`, Apache-2.0 4B variant in `fp16`).
  - `filler_scenes` -> `sdxl_lightning` (role: `volume_image`, 4-step fused UNet + fixed `madebyollin` VAE in `fp16`).
  - Escalation order: `sdxl_lightning` -> `flux2_klein`.
- **Parallel Modes & Multi-GPU Execution**:
  - Supports `single_process_multi_gpu` (balanced device map across GPUs for large models) and `process_per_gpu` (isolated worker per GPU via `CUDA_VISIBLE_DEVICES` for high volume throughput).
- **Subprocess Isolation**:
  - Bakeoff and runner execute engines in fresh child processes so 100% of host RAM and GPU VRAM are reclaimed by the OS between engines.
- **Atomic Persistence & Idempotency**: Writes incremental results atomically to `results.json`; skips existing files.
- **Heartbeat & Time Budget**: Logs progress every 30 seconds and stops cleanly before notebook timeouts.
- **Bakeoff Benchmark & Contact Sheet**: Benchmarks models across resolutions, records per-GPU peak VRAM, inspects component device placements, and renders dark-mode HTML contact sheets.
- **100% CPU Test Coverage**: Includes a pure-CPU fake engine so the entire suite runs on any laptop without a GPU.

---

## Directory Layout

```text
yt-gpu-worker/
├── worker/
│   ├── models.yaml          # Model governance, roles, VRAM/RAM specs & tiers
│   ├── schema.py            # Pydantic Pack, Item, Result schemas
│   ├── memory.py            # Hardware detection, host-RAM guard, OOM recovery
│   ├── runner.py            # Process-per-GPU execution loop & heartbeat
│   ├── engines/             # diffusers & fake engines (SDXL, FLUX.2, Wan, etc.)
│   └── transports/          # Kaggle, Drive, and HTTP server transports
├── notebooks/
│   ├── kaggle_runner.py     # Standalone Kaggle execution script
│   ├── colab_runner.ipynb   # Google Colab notebook
│   └── bakeoff.py           # Benchmark & contact sheet tool with per-GPU VRAM
├── scripts/
│   ├── bench_throughput.py  # Throughput harness (single_process vs process_per_gpu)
│   └── run_style_test.py    # Style & explainer diagram benchmark runner
├── packs/
│   └── style_test_pack.json # 16-item style & explainer test pack
└── tests/                   # CPU-safe unit & integration tests
```

---

## Quick Start (Local Testing)

```bash
# Clone and install dependencies
git clone https://github.com/DHIRAJSHEWALE007/yt-gpu-worker.git
cd yt-gpu-worker
pip install -r requirements.txt

# Run full test suite on CPU with fake engine
pytest -v
```

---

## Running on Kaggle (T4 x2)

In a Kaggle Notebook (GPU T4x2 enabled):
```bash
!git clone -b v1.0.0 https://github.com/DHIRAJSHEWALE007/yt-gpu-worker.git
%cd yt-gpu-worker
!pip install -r requirements.txt -r requirements-gpu.txt
```

### 1. Run Complete Bakeoff Benchmark
```bash
!python notebooks/bakeoff.py --all --out ./outputs/bakeoff
```
Outputs `bakeoff.json` (with per-GPU peak VRAM and component device placements) and interactive `contact_sheet.html`.

### 2. Run Throughput Benchmark (Mode Comparison)
```bash
!python scripts/bench_throughput.py --all --out ./outputs/throughput
```
Tests 20 items in both `single_process_multi_gpu` and `process_per_gpu` modes, logs times, RAM, and OOMs, and selects the optimal `parallel_mode` in `models.yaml`.

### 3. Run Style & Explainer Quality Benchmark
```bash
!python scripts/run_style_test.py --out ./outputs/style_test
```
Renders 5 channel-style prompts (teal & orange vector palette) and 3 technical explainer diagrams side-by-side for `sdxl_lightning` and `flux2_klein`, outputting `style_contact_sheet.html`.

### 4. Execute a Custom Pack
```bash
!python -m worker.runner --pack tests/sample_pack.json --engine sdxl_lightning --out ./outputs
```
