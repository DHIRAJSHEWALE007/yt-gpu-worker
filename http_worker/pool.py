"""Warm engine pool: one long-lived process per role (GPU), models loaded once."""
from __future__ import annotations

import multiprocessing as mp
import queue
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from http_worker.config import ServerConfig


def _check_policy(engine_id: str, cfg: Dict[str, Any], allow_test: bool, allow_nc: bool) -> None:
    if not cfg:
        raise ValueError(f"Engine '{engine_id}' not found in worker/models.yaml")
    if cfg.get("enabled") is False and not (cfg.get("test_only") and allow_test):
        raise ValueError(f"Engine '{engine_id}' is disabled in models.yaml")
    if cfg.get("test_only") and not allow_test:
        raise ValueError(f"Engine '{engine_id}' is test_only (set allow_test_engines: true to use it)")
    if cfg.get("commercial_ok") is False and not allow_nc:
        raise ValueError(f"Engine '{engine_id}' has commercial_ok=false (set allow_noncommercial: true to override)")


def _worker_main(role_idx: int, gpu: Optional[int], engine_id: str, allow_test: bool, allow_nc: bool,
                 in_q: Any, out_q: Any) -> None:
    import os
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    try:
        from worker.engines import get_engine, load_models_config
        from worker.schema import Item
        cfg = dict(load_models_config().get(engine_id, {}))
        _check_policy(engine_id, cfg, allow_test, allow_nc)
        engine = get_engine(engine_id, config=cfg)
        device = None
        if gpu is not None:
            try:
                import torch
                if torch.cuda.is_available():
                    device = "cuda:0"  # CUDA_VISIBLE_DEVICES maps the chosen GPU to index 0
            except Exception:
                pass
        engine.load(device=device)
        out_q.put(("ready", role_idx, engine_id, None))
    except BaseException as exc:  # noqa: BLE001
        out_q.put(("load_failed", role_idx, engine_id, f"{type(exc).__name__}: {exc}"))
        return

    while True:
        job = in_q.get()
        if job is None:
            break
        out_q.put(("started", job["id"], role_idx, None))
        t0 = time.time()
        try:
            item = Item(**job["item"])
            path = engine.generate(item, Path(job["out_dir"]))
            out_q.put(("done", job["id"], str(path), time.time() - t0))
        except BaseException as exc:  # noqa: BLE001
            try:
                from worker.memory import clear_gpu_memory
                clear_gpu_memory()
            except Exception:
                pass
            out_q.put(("failed", job["id"], f"{type(exc).__name__}: {exc}", time.time() - t0))
    try:
        engine.unload()
    except Exception:
        pass


class EnginePool:
    def __init__(self, cfg: ServerConfig):
        self.cfg = cfg
        self.ctx = mp.get_context("spawn")
        self.out_q = self.ctx.Queue()
        self.in_qs: Dict[str, Any] = {}
        self.procs: List[Any] = []
        self.roles: List[Dict[str, Any]] = []
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
        self._listener: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.started_at = time.time()
        self.out_dir = Path(cfg.output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- lifecycle
    def start(self) -> None:
        for idx, role in enumerate(self.cfg.roles):
            self.in_qs.setdefault(role.engine, self.ctx.Queue())
            proc = self.ctx.Process(
                target=_worker_main,
                args=(idx, role.gpu, role.engine, self.cfg.allow_test_engines, self.cfg.allow_noncommercial,
                      self.in_qs[role.engine], self.out_q),
                daemon=True,
            )
            self.roles.append({"index": idx, "gpu": role.gpu, "engine": role.engine, "state": "loading",
                               "jobs_done": 0, "error": None})
            proc.start()
            self.procs.append(proc)
        self._listener = threading.Thread(target=self._listen, daemon=True)
        self._listener.start()

    def stop(self) -> None:
        self._stop.set()
        for q in self.in_qs.values():
            for _ in self.cfg.roles:
                try:
                    q.put(None)
                except Exception:
                    pass
        for p in self.procs:
            p.join(timeout=15)
            if p.is_alive():
                p.terminate()

    def _listen(self) -> None:
        while not self._stop.is_set():
            try:
                msg = self.out_q.get(timeout=0.5)
            except queue.Empty:
                continue
            kind = msg[0]
            with self.lock:
                if kind == "ready":
                    self.roles[msg[1]]["state"] = "idle"
                elif kind == "load_failed":
                    self.roles[msg[1]].update(state="failed", error=msg[3])
                elif kind == "started":
                    job = self.jobs.get(msg[1])
                    if job:
                        job.update(status="running", started=time.time())
                        self.roles[msg[2]]["state"] = "busy"
                        job["role"] = msg[2]
                elif kind in ("done", "failed"):
                    job = self.jobs.get(msg[1])
                    if job:
                        role = job.get("role")
                        if role is not None:
                            self.roles[role]["state"] = "idle"
                            self.roles[role]["jobs_done"] += 1
                        job["finished"] = time.time()
                        job["seconds"] = msg[3]
                        if kind == "done":
                            job.update(status="done", path=msg[2])
                        else:
                            job.update(status="failed", error=msg[2])

    # ---- state
    def engines_loaded(self) -> List[str]:
        with self.lock:
            return sorted({r["engine"] for r in self.roles if r["state"] in ("idle", "busy")})

    def ready(self) -> bool:
        with self.lock:
            return bool(self.roles) and all(r["state"] in ("idle", "busy") for r in self.roles)

    def any_failed(self) -> Optional[str]:
        with self.lock:
            for r in self.roles:
                if r["state"] == "failed":
                    return f"{r['engine']}: {r['error']}"
        return None

    def queue_depth(self) -> int:
        with self.lock:
            return sum(1 for j in self.jobs.values() if j["status"] == "queued")

    def snapshot(self) -> List[Dict[str, Any]]:
        with self.lock:
            return [dict(r) for r in self.roles]

    # ---- jobs
    def submit(self, engine_id: str, item: Dict[str, Any]) -> str:
        if engine_id not in self.engines_loaded():
            raise KeyError(engine_id)
        if self.queue_depth() >= self.cfg.max_queue:
            raise OverflowError("queue full")
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        item = dict(item, id=job_id, engine=engine_id)
        with self.lock:
            self.jobs[job_id] = {"id": job_id, "engine": engine_id, "status": "queued", "created": time.time(),
                                 "item": item, "path": None, "error": None, "seconds": None}
        self.in_qs[engine_id].put({"id": job_id, "item": item, "out_dir": str(self.out_dir)})
        return job_id

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            job = self.jobs.get(job_id)
            return dict(job) if job else None

    def wait(self, job_id: str, timeout: float) -> Optional[Dict[str, Any]]:
        end = time.time() + timeout
        while time.time() < end:
            job = self.get(job_id)
            if job and job["status"] in ("done", "failed"):
                return job
            time.sleep(0.05)
        return self.get(job_id)

    def cleanup(self) -> None:
        cutoff = time.time() - self.cfg.job_ttl_minutes * 60
        with self.lock:
            old = [k for k, j in self.jobs.items() if j["status"] in ("done", "failed") and j["created"] < cutoff]
            for k in old:
                p = self.jobs[k].get("path")
                if p:
                    try:
                        Path(p).unlink(missing_ok=True)
                    except Exception:
                        pass
                del self.jobs[k]
