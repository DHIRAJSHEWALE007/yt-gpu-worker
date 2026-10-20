"""notebooks/kaggle_runner.py — Standalone Kaggle runner script.

Key rules:
- Clones this repo at pinned tag (or uses current local checkout).
- Prints system environment information first: nvidia-smi, free -h, df -h, python --version.
- Zero secrets in script.
- NEVER reinstalls PyTorch (preserves pre-installed CUDA drivers).
- Reads embedded or linked pack.json and writes outputs to /kaggle/working/.
"""

import argparse
import json
import logging
import os
from pathlib import Path
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO_URL = "https://github.com/DHIRAJSHEWALE007/yt-gpu-worker.git"
PINNED_TAG = "v1.0.0"


def print_system_diagnostics() -> None:
    """Print system environment: nvidia-smi, free -h, df -h, python --version."""
    print("=" * 80)
    print("yt-gpu-worker: Kaggle Environment Diagnostics")
    print("=" * 80)

    commands = [
        ["python", "--version"],
        ["nvidia-smi"],
        ["free", "-h"],
        ["df", "-h"],
    ]

    for cmd in commands:
        cmd_str = " ".join(cmd)
        print(f"\n--- Running: {cmd_str} ---")
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
            print(res.stdout.strip())
        except Exception as e:
            print(f"Command '{cmd_str}' failed or not available on host: {e}")

    print("=" * 80)


def setup_repository(target_dir: Path) -> Path:
    """Ensure yt-gpu-worker is available on sys.path."""
    if (target_dir / "worker" / "runner.py").is_file():
        logger.info("Found worker locally at %s", target_dir)
        return target_dir

    logger.info("Cloning repository at pinned tag %s...", PINNED_TAG)
    subprocess.run(["git", "clone", "--depth", "1", "--branch", PINNED_TAG, REPO_URL, str(target_dir)], check=True)
    return target_dir


def resolve_pack_path(user_input: str | None) -> Path:
    """Resolve pack file path with smart fallbacks for Linux bash backslash escaping."""
    from worker.transports.kaggle_pack import find_default_pack_file

    if not user_input:
        return find_default_pack_file()

    raw = str(user_input).strip()
    norm = raw.replace("\\", "/")
    repo_root = Path(__file__).resolve().parent.parent

    candidates = [
        Path(raw),
        Path(norm),
        repo_root / norm,
        repo_root / raw,
    ]

    if "testssample" in norm:
        fixed = norm.replace("testssample", "tests/sample")
        candidates.extend([Path(fixed), repo_root / fixed])

    for c in candidates:
        if c.is_file():
            return c.resolve()

    return find_default_pack_file()


def main() -> None:
    parser = argparse.ArgumentParser(description="Kaggle Runner for yt-gpu-worker")
    parser.add_argument("--pack", type=str, default=None, help="Path to input pack.json")
    parser.add_argument("--output-dir", type=str, default="/kaggle/working/outputs", help="Output directory")
    args = parser.parse_args()

    # 1. Print environment diagnostic logs
    print_system_diagnostics()

    # 2. Add repo to python path
    current_dir = Path(__file__).resolve().parent.parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))

    from worker.schema import load_pack
    from worker.runner import run_pack
    from worker.transports.kaggle_pack import find_default_pack_file

    pack_path = resolve_pack_path(args.pack)
    logger.info("Loading pack from: %s", pack_path)
    pack = load_pack(pack_path)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 3. Execute generation workload
    logger.info("Starting PackRunner on %d items...", len(pack.items))
    results = run_pack(pack, output_dir=out_dir)
    logger.info("Completed %d items. Exiting cleanly.", len(results))


if __name__ == "__main__":
    main()
