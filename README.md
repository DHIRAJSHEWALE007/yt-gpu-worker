# yt-gpu-worker

Decoupled headless GPU worker designed for automated video production pipelines ([JackTheCreator](https://github.com/DHIRAJSHEWALE007/JackTheCreator)).

Runs autonomously on **Kaggle**, **Google Colab**, or self-hosted GPU server nodes to render images and video clips from structured batch packs.

---

## Features

- **Zero Studio Dependencies**: Interacts purely via Pydantic `Pack`, `Item`, and `Result` schemas (`v1.0.0`) or HTTP conforming to `REMOTE_CONTRACT.md`.
- **Hardware-Aware Memory Management**:
  - Automatically detects GPU capabilities, VRAM, and system RAM.
  - Strictly enforces `fp16` on **Tesla T4** and **Tesla P100** (never `bf16`).
  - Automatic 1-step downscale/offload retry on CUDA OOM; never crashes the batch.
- **Engine Sorting**: Sorts pack items by `(engine, width, height)` to load heavy model weights only once.
- **Atomic Persistence & Idempotency**: Writes incremental results atomically to `results.json`; skips existing files.
- **Heartbeat & Time Budget**: Logs progress every 30 seconds and stops cleanly before notebook timeouts.
- **Bakeoff Benchmark**: Includes benchmark tool measuring download, load, peak VRAM/RAM, and seconds per item across sizes.
- **100% CPU Test Coverage**: Includes a pure-CPU fake engine so the entire suite runs on any laptop without a GPU.

---

## Directory Layout

```text
yt-gpu-worker/
├── models.yaml              # Model registry with licensing & VRAM specs
├── worker/
│   ├── schema.py            # Pydantic Pack, Item, Result schemas
│   ├── memory.py            # Hardware detection & OOM recovery
│   ├── runner.py            # Process-per-GPU execution loop & heartbeat
│   ├── engines/             # diffusers & fake engines
│   └── transports/          # Kaggle, Drive, and HTTP server transports
├── notebooks/
│   ├── kaggle_runner.py     # Standalone Kaggle execution script
│   ├── colab_runner.ipynb   # Google Colab notebook
│   └── bakeoff.py           # Benchmark & contact sheet tool
└── tests/                   # CPU-safe unit & integration tests
```

---

## Quick Start (Local Testing)

```bash
# Clone and install dependencies
git clone https://github.com/DHIRAJSHEWALE007/yt-gpu-worker.git
cd yt-gpu-worker
pip install -r requirements.txt

# Run test suite on CPU with fake engine
pytest -v
```

## Running on Kaggle

In a Kaggle Notebook (GPU T4x2 enabled):
```python
!git clone https://github.com/DHIRAJSHEWALE007/yt-gpu-worker.git
%cd yt-gpu-worker
!python notebooks/kaggle_runner.py --pack /kaggle/input/my-video-pack/pack.json
```
