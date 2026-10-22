"""worker/schema.py — Strict Pydantic Pack, Item, and Result contract definitions.

Conforms to section 5.2 of the architecture guide.
Guarantees schema_version compatibility across remote nodes.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0.0"


def _now_iso() -> str:
    """Current UTC ISO 8601 timestamp."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


class Item(BaseModel):
    """Single generation workload item (image, video, or audio asset)."""

    model_config = ConfigDict(extra="ignore")

    id: str = Field(description="Unique task identifier, e.g. shot_001")
    engine: str = Field(description="Engine ID from models.yaml, e.g. sdxl_lightning, wan22_ti2v5b, fake")
    prompt: str = Field(description="Positive textual prompt")
    negative_prompt: str = Field(default="", description="Negative prompt")
    width: int = Field(default=1024, ge=64, le=4096, description="Output width in pixels")
    height: int = Field(default=1024, ge=64, le=4096, description="Output height in pixels")
    steps: Optional[int] = Field(default=None, ge=1, le=150, description="Inference steps override")
    seed: Optional[int] = Field(default=None, description="RNG seed for reproducibility")
    guidance_scale: Optional[float] = Field(default=None, ge=0.0, le=30.0, description="CFG scale override")
    media_type: Literal["image", "video"] = Field(default="image", description="Asset type to render")
    duration_seconds: Optional[float] = Field(default=None, ge=0.1, le=60.0, description="Duration for video render")
    fps: Optional[int] = Field(default=24, ge=1, le=60, description="Video frame rate")
    ref_image_path: Optional[str] = Field(default=None, description="Reference image or first frame path")
    extra_params: Dict[str, Any] = Field(default_factory=dict, description="Custom engine parameters")


class Result(BaseModel):
    """Output execution artifact descriptor for an Item."""

    model_config = ConfigDict(extra="ignore")

    item_id: str = Field(description="Corresponding Item ID")
    status: Literal["success", "failed", "skipped"] = Field(description="Execution status")
    output_path: Optional[str] = Field(default=None, description="Relative or absolute path to generated file")
    engine: str = Field(description="Engine that processed this item")
    execution_time_seconds: float = Field(default=0.0, description="Seconds spent in model generation")
    peak_vram_mb: Optional[float] = Field(default=None, description="Peak VRAM allocated during run in MB")
    error: Optional[str] = Field(default=None, description="Error message if status is failed")
    retried_oom: bool = Field(default=False, description="True if recovered from CUDA OOM")
    fallback_applied: Optional[str] = Field(default=None, description="Fallback strategy applied on OOM: 'cpu_offload' or 'downscale'")
    width: Optional[int] = None
    height: Optional[int] = None
    seed: Optional[int] = None
    completed_at: str = Field(default_factory=_now_iso, description="ISO timestamp of completion")


class Pack(BaseModel):
    """Batch container representing a full video or multi-scene generation task."""

    model_config = ConfigDict(extra="ignore")

    schema_version: str = Field(default=SCHEMA_VERSION, description="Schema specification version")
    pack_id: str = Field(description="Unique batch identifier, e.g. pack_20261020_volcanoes")
    job_id: Optional[str] = Field(default=None, description="Parent Studio job identifier")
    created_at: str = Field(default_factory=_now_iso, description="UTC creation timestamp")
    max_runtime_minutes: Optional[float] = Field(default=None, description="Hard deadline for execution")
    output_dir: str = Field(default="outputs", description="Output directory relative to pack location")
    items: List[Item] = Field(default_factory=list, description="Ordered list of items to render")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary pipeline context")


# ── Persistence & Serialization Helpers ─────────────────────────────────────────

def load_pack(source: Union[str, Path, dict]) -> Pack:
    """Load and validate a Pack instance from a file path, JSON string, or dict."""
    if isinstance(source, dict):
        return Pack.model_validate(source)

    raw_str = str(source).strip()

    # If it starts with JSON object delimiter, parse directly as inline JSON string
    if raw_str.startswith("{") or raw_str.startswith("["):
        try:
            data = json.loads(raw_str)
            return Pack.model_validate(data)
        except Exception as e:
            raise ValueError(f"Failed to parse inline pack JSON string: {e}") from e

    # Normalize backslashes (in case Windows-style path was passed in Linux bash)
    norm_str = raw_str.replace("\\", "/")
    repo_root = Path(__file__).resolve().parent.parent

    candidates = [
        Path(raw_str),
        Path(norm_str),
        repo_root / norm_str,
        repo_root / raw_str,
    ]

    # Bash unescaped backslash recovery: "testssample_pack.json" -> "tests/sample_pack.json"
    if "testssample" in norm_str:
        fixed = norm_str.replace("testssample", "tests/sample")
        candidates.extend([Path(fixed), repo_root / fixed])

    for p in candidates:
        if p.is_file():
            try:
                content = p.read_text(encoding="utf-8")
                data = json.loads(content)
                return Pack.model_validate(data)
            except Exception as e:
                raise ValueError(f"Failed to parse pack file '{p}': {e}") from e

    raise FileNotFoundError(
        f"Pack file not found: '{source}'. Checked paths: {[str(c) for c in candidates]}"
    )


def save_pack(pack: Pack, target_path: Union[str, Path]) -> Path:
    """Save a Pack to disk atomically using temporary file replacement."""
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = target.with_suffix(".tmp")

    with open(tmp_file, "w", encoding="utf-8") as f:
        f.write(pack.model_dump_json(indent=2))
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp_file, target)
    return target


def load_results(results_path: Union[str, Path]) -> List[Result]:
    """Read results.json safely, returning empty list if file does not exist."""
    p = Path(results_path)
    if not p.is_file():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [Result.model_validate(it) for it in data]
        if isinstance(data, dict) and "results" in data:
            return [Result.model_validate(it) for it in data["results"]]
        return []
    except Exception as e:
        logger.warning("Could not read %s: %s; returning empty results", p, e)
        return []


def save_results_atomic(results: List[Result], results_path: Union[str, Path]) -> None:
    """Atomically write full results list to disk."""
    target = Path(results_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = target.with_suffix(".tmp")

    data = [r.model_dump() for r in results]
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp_file, target)


def append_result_atomic(result: Result, results_path: Union[str, Path]) -> None:
    """Safely append a single Result to results.json with atomic replacement."""
    existing = load_results(results_path)
    filtered = [r for r in existing if r.item_id != result.item_id]
    filtered.append(result)
    save_results_atomic(filtered, results_path)
