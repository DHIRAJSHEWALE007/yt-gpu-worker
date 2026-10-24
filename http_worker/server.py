"""Warm HTTP image server: OpenAI-compatible image API plus async jobs and packs.

Run:  WORKER_TOKEN=... python -m http_worker.server --config http_worker/config.yaml
"""
from __future__ import annotations

import argparse
import base64
import hmac
import io
import json
import logging
import os
import sys
import threading
import time
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field

from http_worker import __version__
from http_worker.config import ServerConfig, load_config
from http_worker.pool import EnginePool
from http_worker.tunnel import start_tunnel

logger = logging.getLogger("http_worker")
_MIME = {"webp": "image/webp", "jpeg": "image/jpeg", "png": "image/png"}


# ----------------------------------------------------------------------------- helpers
def _parse_size(size: str, max_side: int) -> tuple[int, int]:
    if not size or size == "auto":
        w, h = 1024, 1024
    else:
        try:
            w_s, h_s = size.lower().split("x")
            w, h = int(w_s), int(h_s)
        except Exception:
            raise HTTPException(status_code=400, detail="size must be 'auto' or 'WIDTHxHEIGHT'")
    w, h = max(64, round(w / 16) * 16), max(64, round(h / 16) * 16)
    if w > max_side or h > max_side:
        raise HTTPException(status_code=400, detail=f"max side is {max_side}px")
    return w, h


def _encode(path: str, fmt: str, quality: int) -> bytes:
    fmt = fmt.lower()
    if fmt not in _MIME:
        raise HTTPException(status_code=400, detail="output_format must be webp, jpeg or png")
    img = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    if fmt == "png":
        img.save(buf, format="PNG")
    elif fmt == "jpeg":
        img.save(buf, format="JPEG", quality=quality, optimize=True)
    else:
        img.save(buf, format="WEBP", quality=quality, method=4)
    return buf.getvalue()


class ImageGenRequest(BaseModel):  # OpenAI-compatible (+ a few optional extras)
    model: Optional[str] = None
    prompt: str = Field(min_length=1, max_length=4000)
    n: int = Field(default=1, ge=1, le=4)
    size: str = "auto"
    response_format: str = "b64_json"
    output_format: Optional[str] = None
    seed: Optional[int] = None
    steps: Optional[int] = Field(default=None, ge=1, le=150)


class JobRequest(BaseModel):
    model: Optional[str] = None
    prompt: str = Field(min_length=1, max_length=4000)
    size: str = "auto"
    seed: Optional[int] = None
    steps: Optional[int] = Field(default=None, ge=1, le=150)
    output_format: Optional[str] = None


class PackItem(BaseModel):
    id: str = Field(min_length=1, max_length=120)
    model: Optional[str] = None
    prompt: str = Field(min_length=1, max_length=4000)
    size: str = "auto"
    seed: Optional[int] = None
    steps: Optional[int] = Field(default=None, ge=1, le=150)


class PackRequest(BaseModel):
    items: List[PackItem] = Field(min_length=1, max_length=500)
    output_format: Optional[str] = None


def request_shutdown(app: FastAPI, reason: str) -> None:
    if app.state.shutdown_requested:
        return
    app.state.shutdown_requested = True
    logger.info("[SHUTDOWN] %s", reason)
    srv = getattr(app.state, "server", None)
    if srv is not None:
        srv.should_exit = True


# ----------------------------------------------------------------------------- app factory
def create_app(cfg: ServerConfig, token: Optional[str]) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pool = EnginePool(cfg)
        app.state.pool.start()
        threading.Thread(target=_guard_loop, args=(app,), daemon=True).start()
        if cfg.selftest_on_start:
            threading.Thread(target=_selftest, args=(app,), daemon=True).start()
        yield
        app.state.pool.stop()

    app = FastAPI(title="yt-gpu-worker http_worker", version=__version__, lifespan=lifespan)
    app.state.cfg = cfg
    app.state.token = token
    app.state.started = time.time()
    app.state.last_activity = time.time()
    app.state.requests = 0
    app.state.shutdown_requested = False
    app.state.server = None
    app.state.packs = {}

    def auth(authorization: Optional[str]) -> None:
        app.state.last_activity = time.time()
        if app.state.token is None:  # only possible with ALLOW_NO_AUTH=1
            return
        supplied = (authorization or "").removeprefix("Bearer ").strip()
        if not hmac.compare_digest(supplied.encode(), app.state.token.encode()):
            raise HTTPException(status_code=401, detail="invalid or missing bearer token")

    def resolve_engine(model: Optional[str]) -> str:
        pool: EnginePool = app.state.pool
        name = model or cfg.default_engine
        if name not in pool.engines_loaded():
            raise HTTPException(status_code=404, detail=f"model '{name}' is not loaded; loaded: {pool.engines_loaded()}")
        return name

    def make_item(prompt: str, size: str, seed: Optional[int], steps: Optional[int]) -> Dict[str, Any]:
        w, h = _parse_size(size, cfg.max_side)
        item: Dict[str, Any] = {"prompt": prompt, "width": w, "height": h, "media_type": "image"}
        if seed is not None:
            item["seed"] = seed
        if steps is not None:
            item["steps"] = steps
        return item

    def submit(engine: str, item: Dict[str, Any]) -> str:
        try:
            return app.state.pool.submit(engine, item)
        except OverflowError:
            raise HTTPException(status_code=429, detail="queue is full, retry later")

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        t0 = time.time()
        app.state.requests += 1
        response = await call_next(request)
        size = response.headers.get("content-length", "-")
        logger.info("[REQ] %s %s -> %s  %.2fs  %sB", request.method, request.url.path, response.status_code,
                    time.time() - t0, size)
        return response

    # ------------------------------------------------------------------ endpoints
    @app.get("/health")
    async def health(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
        auth(authorization)
        pool: EnginePool = app.state.pool
        left = None
        if cfg.max_session_minutes:
            left = max(0.0, cfg.max_session_minutes - (time.time() - app.state.started) / 60)
        return {
            "status": "ready" if pool.ready() else ("failed" if pool.any_failed() else "loading"),
            "ready": pool.ready(),
            "error": pool.any_failed(),
            "version": __version__,
            "engines_loaded": pool.engines_loaded(),
            "workers": pool.snapshot(),
            "queue_depth": pool.queue_depth(),
            "requests_total": app.state.requests,
            "uptime_s": round(time.time() - app.state.started, 1),
            "session_minutes_left": None if left is None else round(left, 1),
            "output_format": cfg.output_format,
        }

    @app.get("/v1/models")
    async def models(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
        auth(authorization)
        return {"object": "list", "data": [{"id": e, "object": "model", "owned_by": "yt-gpu-worker"}
                                           for e in app.state.pool.engines_loaded()]}

    @app.post("/v1/images/generations")
    async def generations(req: ImageGenRequest, request: Request, authorization: Optional[str] = Header(None)):
        auth(authorization)
        if req.response_format not in ("b64_json", "url"):
            raise HTTPException(status_code=400, detail="response_format must be b64_json or url")
        engine = resolve_engine(req.model)
        fmt = (req.output_format or cfg.output_format).lower()
        ids = [submit(engine, make_item(req.prompt, req.size, None if req.seed is None else req.seed + i, req.steps))
               for i in range(req.n)]
        data = []
        for jid in ids:
            job = await _await_job(app, jid, cfg.request_timeout_s)
            if job["status"] == "failed":
                raise HTTPException(status_code=500, detail=f"generation failed: {job['error']}")
            if job["status"] != "done":
                raise HTTPException(status_code=504, detail=f"still running; poll /v1/jobs/{jid}")
            blob = _encode(job["path"], fmt, cfg.image_quality)
            entry: Dict[str, Any] = {"model": engine, "size": f"{job['item']['width']}x{job['item']['height']}",
                                     "seed": job["item"].get("seed"), "job_id": jid}
            if req.response_format == "b64_json":
                entry["b64_json"] = base64.b64encode(blob).decode()
            else:
                entry["url"] = f"{str(request.base_url).rstrip('/')}/v1/jobs/{jid}/file?format={fmt}"
            data.append(entry)
        return {"created": int(time.time()), "data": data}

    @app.post("/v1/jobs")
    async def create_job(req: JobRequest, authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
        auth(authorization)
        engine = resolve_engine(req.model)
        jid = submit(engine, make_item(req.prompt, req.size, req.seed, req.steps))
        return {"id": jid, "status": "queued", "model": engine}

    @app.get("/v1/jobs/{job_id}")
    async def job_status(job_id: str, authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
        auth(authorization)
        job = app.state.pool.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="unknown job")
        return {"id": job_id, "status": job["status"], "model": job["engine"], "seconds": job["seconds"],
                "error": job["error"], "file": f"/v1/jobs/{job_id}/file" if job["status"] == "done" else None}

    @app.get("/v1/jobs/{job_id}/file")
    async def job_file(job_id: str, format: Optional[str] = None, authorization: Optional[str] = Header(None)):
        auth(authorization)
        job = app.state.pool.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="unknown job")
        if job["status"] != "done":
            raise HTTPException(status_code=409, detail=f"job is {job['status']}")
        fmt = (format or cfg.output_format).lower()
        return Response(content=_encode(job["path"], fmt, cfg.image_quality), media_type=_MIME.get(fmt, "image/png"))

    @app.post("/v1/packs")
    async def create_pack(req: PackRequest, authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
        auth(authorization)
        if len({i.id for i in req.items}) != len(req.items):
            raise HTTPException(status_code=400, detail="duplicate item ids")
        pack_id = f"pack_{int(time.time())}_{len(app.state.packs)}"
        entries = []
        for it in req.items:
            engine = resolve_engine(it.model)
            entries.append({"id": it.id, "job": submit(engine, make_item(it.prompt, it.size, it.seed, it.steps))})
        app.state.packs[pack_id] = {"items": entries, "format": (req.output_format or cfg.output_format).lower()}
        return {"pack_id": pack_id, "total": len(entries)}

    def pack_state(pack_id: str) -> Dict[str, Any]:
        pack = app.state.packs.get(pack_id)
        if not pack:
            raise HTTPException(status_code=404, detail="unknown pack")
        rows = []
        for e in pack["items"]:
            job = app.state.pool.get(e["job"]) or {"status": "expired", "error": "job expired", "seconds": None,
                                                    "engine": None, "item": {}}
            rows.append({"id": e["id"], "status": job["status"], "seconds": job["seconds"], "error": job["error"],
                         "engine": job["engine"]})
        done = sum(r["status"] == "done" for r in rows)
        failed = sum(r["status"] in ("failed", "expired") for r in rows)
        finished = done + failed == len(rows)
        status = "running" if not finished else ("done" if failed == 0 else ("failed" if done == 0 else "partial"))
        return {"pack_id": pack_id, "status": status, "total": len(rows), "done": done, "failed": failed,
                "items": rows}

    @app.get("/v1/packs/{pack_id}")
    async def pack_status(pack_id: str, authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
        auth(authorization)
        return pack_state(pack_id)

    @app.get("/v1/packs/{pack_id}/download")
    async def pack_download(pack_id: str, authorization: Optional[str] = Header(None)):
        auth(authorization)
        st = pack_state(pack_id)
        if st["status"] == "running":
            raise HTTPException(status_code=409, detail="pack still running")
        pack = app.state.packs[pack_id]
        fmt = pack["format"]
        buf = io.BytesIO()
        results = []
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:  # images are already compressed
            for e in pack["items"]:
                job = app.state.pool.get(e["job"])
                row: Dict[str, Any] = {"item_id": e["id"], "status": "failed", "file": None, "engine": None,
                                       "seconds": None, "error": "job expired", "width": None, "height": None,
                                       "seed": None}
                if job:
                    row.update(engine=job["engine"], seconds=job["seconds"], error=job["error"],
                               width=job["item"].get("width"), height=job["item"].get("height"),
                               seed=job["item"].get("seed"))
                    if job["status"] == "done":
                        name = f"{e['id']}.{fmt if fmt != 'jpeg' else 'jpg'}"
                        zf.writestr(name, _encode(job["path"], fmt, cfg.image_quality))
                        row.update(status="success", file=name, error=None)
                results.append(row)
            zf.writestr("results.json", json.dumps(results, indent=2))
        return Response(content=buf.getvalue(), media_type="application/zip",
                        headers={"Content-Disposition": f'attachment; filename="{pack_id}.zip"'})

    @app.post("/v1/shutdown")
    async def shutdown(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
        auth(authorization)
        threading.Timer(0.5, request_shutdown, args=(app, "shutdown requested via API")).start()
        return {"status": "shutting down"}

    return app


async def _await_job(app: FastAPI, job_id: str, timeout: float) -> Dict[str, Any]:
    import asyncio
    end = time.time() + timeout
    while time.time() < end:
        job = app.state.pool.get(job_id)
        if job and job["status"] in ("done", "failed"):
            return job
        await asyncio.sleep(0.05)
    return app.state.pool.get(job_id)


def _guard_loop(app: FastAPI) -> None:
    cfg: ServerConfig = app.state.cfg
    last_hb = 0.0
    while not app.state.shutdown_requested:
        time.sleep(2)
        now = time.time()
        if cfg.max_session_minutes and now - app.state.started > cfg.max_session_minutes * 60:
            request_shutdown(app, f"max_session_minutes={cfg.max_session_minutes} reached")
            return
        if cfg.idle_timeout_minutes and now - app.state.last_activity > cfg.idle_timeout_minutes * 60:
            request_shutdown(app, f"no authenticated request for {cfg.idle_timeout_minutes} min (idle timeout)")
            return
        if now - last_hb >= cfg.heartbeat_seconds:
            last_hb = now
            pool: EnginePool = app.state.pool
            pool.cleanup()
            workers = " | ".join(f"gpu{r['gpu']} {r['engine']} {r['state']} done={r['jobs_done']}"
                                 for r in pool.snapshot())
            logger.info("[HEARTBEAT] up %s | requests %d | queue %d | %s",
                        time.strftime("%H:%M:%S", time.gmtime(now - app.state.started)), app.state.requests,
                        pool.queue_depth(), workers)


def _selftest(app: FastAPI) -> None:
    pool: EnginePool = app.state.pool
    for _ in range(600):  # wait up to ~10 min for models to load
        if pool.ready() or pool.any_failed():
            break
        time.sleep(1)
    if not pool.ready():
        logger.error("[SELFTEST] skipped, workers not ready: %s", pool.any_failed() or "timeout")
        return
    for engine in pool.engines_loaded():
        t0 = time.time()
        jid = pool.submit(engine, {"prompt": "a small red apple on a table, flat vector illustration",
                                   "width": 512, "height": 512, "media_type": "image", "seed": 1})
        job = pool.wait(jid, 300)
        ok = job and job["status"] == "done"
        logger.info("[SELFTEST] %s %s in %.1fs%s", engine, "OK" if ok else "FAILED", time.time() - t0,
                    "" if ok else f" ({job and job['error']})")
    logger.info("[READY] server is ready for requests")


# ----------------------------------------------------------------------------- entry point
def main() -> None:
    parser = argparse.ArgumentParser(description="yt-gpu-worker warm HTTP server")
    parser.add_argument("--config", default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(args.config)
    if args.host:
        cfg.host = args.host
    if args.port:
        cfg.port = args.port

    token = os.environ.get("WORKER_TOKEN") or None
    if token is None and os.environ.get("ALLOW_NO_AUTH") != "1":
        logger.error("WORKER_TOKEN is not set. Set it (Kaggle Secret) or ALLOW_NO_AUTH=1 for local tests only.")
        sys.exit(2)
    logger.info("[START] http_worker %s | roles=%s | tunnel=%s | max_session=%s min | idle_timeout=%s min",
                __version__, [(r.gpu, r.engine) for r in cfg.roles], cfg.tunnel, cfg.max_session_minutes,
                cfg.idle_timeout_minutes)

    import uvicorn
    app = create_app(cfg, token)
    server = uvicorn.Server(uvicorn.Config(app, host=cfg.host, port=cfg.port, log_level="warning"))
    app.state.server = server
    stop_tunnel = start_tunnel(cfg.tunnel, cfg.port)
    try:
        server.run()
    finally:
        if stop_tunnel:
            try:
                stop_tunnel()
            except Exception:
                pass
        logger.info("[STOPPED] server has exited")


if __name__ == "__main__":
    main()
