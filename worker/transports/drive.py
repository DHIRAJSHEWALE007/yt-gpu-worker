"""worker/transports/drive.py — Google Colab Google Drive synchronization transport."""

from __future__ import annotations

import logging
from pathlib import Path
import shutil

logger = logging.getLogger(__name__)


def mount_google_drive(mount_point: str = "/content/drive") -> Path:
    """Mount Google Drive inside Google Colab."""
    try:
        from google.colab import drive  # type: ignore
        drive.mount(mount_point)
        logger.info("Google Drive mounted at %s", mount_point)
        return Path(mount_point)
    except Exception as e:
        logger.warning("Could not mount Google Drive (not running in Colab?): %s", e)
        return Path(mount_point)


def sync_results_to_drive(local_dir: Path, drive_target_dir: Path) -> None:
    """Copy rendered outputs and results.json to Google Drive folder."""
    drive_target_dir.mkdir(parents=True, exist_ok=True)
    for item in local_dir.iterdir():
        if item.is_file():
            shutil.copy2(item, drive_target_dir / item.name)
    logger.info("Synced %s outputs to Google Drive: %s", local_dir, drive_target_dir)
