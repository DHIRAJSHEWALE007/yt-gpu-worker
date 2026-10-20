"""worker/transports/http_server.py — FastAPI server conforming to REMOTE_CONTRACT.md.

Implements:
- GET /health
- POST /v1/image
- POST /v1/video
- GET /v1/jobs/{job_id}
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
import time
from typing import Any, Dict, Optional
import uuid

from fastapi import FastAPI, Header, HTTPException, Response, status
from pydantic import BaseModel, Field

from worker.engines import get_engine
from worker.schema import Item

logger = logging.getLogger(__name__)

app = FastAPI(title="yt-gpu-worker Remote Model Server", version="1.0.0")

# In-memory video jobs tracker
_VIDEO_JOBS: Dict[str, Dict[str, Any]] = {}
_AUTH_TOKEN: Optional[str] = None


def set_auth_token(token: Optional[str]) -> None:
    global _AUTH_TOKEN
    _AUTH_TOKEN = token


def _verify_auth(authorization: Optional[str] = Header(None)) -> None:
    if _AUTH_TOKEN is None:
        return
    if not authorization or authorization != f"Bearer {_AUTH_TOKEN}":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


class ImageRequest(BaseModel):
    prompt: str
    width: int = 1024
    height: int = 1024
    seed: Optional[int] = None
    negative_prompt: str = ""
    engine: str = "sdxl_lightning"
    ref_image_b64: Optional[str] = None


class VideoRequest(BaseModel):
    prompt: str
    seconds: float = 3.0
    width: int = 854
    height: int = 480
    first_frame_b64: Optional[str] = None
    last_frame_b64: Optional[str] = None
    seed: Optional[int] = None
    engine: str = "wan22_ti2v5b"


@app.get("/health")
async def health(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    _verify_auth(authorization)
    return {
        "status": "healthy",
        "version": "1.0.0",
        "capabilities": ["image", "video"],
    }


@app.post("/v1/image")
async def generate_image(
    req: ImageRequest,
    authorization: Optional[str] = Header(None),
) -> Response:
    _verify_auth(authorization)
    item_id = f"img_{uuid.uuid4().hex[:10]}"
    output_dir = Path("outputs") / "http"
    output_dir.mkdir(parents=True, exist_ok=True)

    item = Item(
        id=item_id,
        engine=req.engine,
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        width=req.width,
        height=req.height,
        seed=req.seed,
        media_type="image",
    )

    engine = get_engine(req.engine)
    engine.load()
    try:
        out_path = engine.generate(item, output_dir)
        content = out_path.read_bytes()
        return Response(content=content, media_type="image/png")
    except Exception as e:
        logger.exception("HTTP image generation error: %s", e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/v1/video")
async def submit_video(
    req: VideoRequest,
    authorization: Optional[str] = Header(None),
) -> Dict[str, Any]:
    _verify_auth(authorization)
    job_id = f"vjob_{uuid.uuid4().hex[:12]}"
    output_dir = Path("outputs") / "http"
    output_dir.mkdir(parents=True, exist_ok=True)

    ref_path = None
    if req.first_frame_b64:
        ref_path = str(output_dir / f"{job_id}_first_frame.png")
        Path(ref_path).write_bytes(base64.b64decode(req.first_frame_b64))

    item = Item(
        id=job_id,
        engine=req.engine,
        prompt=req.prompt,
        width=req.width,
        height=req.height,
        seed=req.seed,
        media_type="video",
        duration_seconds=req.seconds,
        ref_image_path=ref_path,
    )

    # Record job
    _VIDEO_JOBS[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0.0,
        "created_at": time.time(),
        "item": item,
        "output_dir": output_dir,
    }

    # For synchronous/fake generation execution:
    try:
        engine = get_engine(req.engine)
        engine.load()
        out_path = engine.generate(item, output_dir)
        _VIDEO_JOBS[job_id].update({
            "status": "done",
            "progress": 1.0,
            "url": str(out_path),
        })
    except Exception as e:
        _VIDEO_JOBS[job_id].update({
            "status": "failed",
            "error": str(e),
        })

    return {
        "job_id": job_id,
        "status": _VIDEO_JOBS[job_id]["status"],
        "estimated_seconds": int(req.seconds * 4),
    }


@app.get("/v1/jobs/{job_id}")
async def get_job_status(
    job_id: str,
    authorization: Optional[str] = Header(None),
) -> Dict[str, Any]:
    _verify_auth(authorization)
    job = _VIDEO_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "progress": job.get("progress", 0.0),
        "url": job.get("url"),
        "error": job.get("error"),
    }
