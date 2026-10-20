"""worker/transports/kaggle_pack.py — Kaggle environment pack adapter.

Finds pack.json in /kaggle/input, runs PackRunner, and packages results into /kaggle/working.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import shutil
import zipfile

from worker.runner import run_pack
from worker.schema import load_pack

logger = logging.getLogger(__name__)


def find_default_pack_file() -> Path:
    """Scan standard Kaggle input directories for pack.json."""
    candidates = [
        Path("/kaggle/input/pack.json"),
        Path("pack.json"),
        Path("inputs/pack.json"),
    ]
    # Check any subfolder in /kaggle/input/
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        for entry in kaggle_input.iterdir():
            if (entry / "pack.json").is_file():
                return entry / "pack.json"

    for cand in candidates:
        if cand.is_file():
            return cand

    raise FileNotFoundError("Could not find pack.json in /kaggle/input/ or current directory.")


def execute_kaggle_pack(pack_path: Path | None = None) -> Path:
    """Execute pack and zip outputs into /kaggle/working/pack_results.zip."""
    target_pack = pack_path or find_default_pack_file()
    pack = load_pack(target_pack)

    working_dir = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path("outputs")
    working_dir.mkdir(parents=True, exist_ok=True)

    out_dir = working_dir / pack.pack_id
    out_dir.mkdir(parents=True, exist_ok=True)

    results = run_pack(pack, output_dir=out_dir)
    logger.info("Kaggle execution finished. Total items evaluated: %d", len(results))

    # Compress into downloadable zip bundle
    zip_path = working_dir / f"{pack.pack_id}_bundle.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in out_dir.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(out_dir)
                zf.write(file_path, arcname)

    logger.info("Created output archive at: %s (%d bytes)", zip_path, zip_path.stat().st_size)
    return zip_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run yt-gpu-worker on Kaggle")
    parser.add_argument("--pack", type=str, default=None, help="Path to pack.json")
    args = parser.parse_args()
    p = Path(args.pack) if args.pack else None
    execute_kaggle_pack(p)
