from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel, Field


class Role(BaseModel):
    gpu: Optional[int] = None  # None = no GPU (fake/CPU engines)
    engine: str


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    roles: List[Role] = Field(default_factory=lambda: [Role(gpu=0, engine="flux2_klein")])
    default_engine: str = "flux2_klein"
    output_format: str = "webp"
    image_quality: int = 92
    output_dir: str = "/tmp/http_worker_out"
    max_session_minutes: float = 360
    idle_timeout_minutes: float = 0
    heartbeat_seconds: float = 60
    request_timeout_s: float = 300
    max_queue: int = 200
    max_side: int = 2048
    job_ttl_minutes: float = 120
    allow_test_engines: bool = False
    allow_noncommercial: bool = False
    tunnel: str = "none"
    selftest_on_start: bool = True


def load_config(path: Optional[str] = None) -> ServerConfig:
    p = Path(path) if path else Path(__file__).with_name("config.yaml")
    data = yaml.safe_load(p.read_text(encoding="utf-8")) if p.is_file() else {}
    cfg = ServerConfig(**(data or {}))
    # Minimal env overrides so a notebook can change settings without editing files.
    if os.environ.get("HTTP_WORKER_PORT"):
        cfg.port = int(os.environ["HTTP_WORKER_PORT"])
    if os.environ.get("HTTP_WORKER_TUNNEL"):
        cfg.tunnel = os.environ["HTTP_WORKER_TUNNEL"]
    return cfg
