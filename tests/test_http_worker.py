"""CPU-only tests for the optional http_worker package (fake engine, no GPU, no network)."""
import base64
import io
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from http_worker.config import Role, ServerConfig
from http_worker.pool import _check_policy
from http_worker.server import create_app

TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


def _cfg(tmp: Path, **kw) -> ServerConfig:
    base = dict(roles=[Role(gpu=None, engine="fake")], default_engine="fake", allow_test_engines=True,
                selftest_on_start=False, heartbeat_seconds=3600, output_dir=str(tmp), max_session_minutes=0)
    base.update(kw)
    return ServerConfig(**base)


def _wait_ready(client: TestClient, timeout: float = 60.0) -> None:
    end = time.time() + timeout
    while time.time() < end:
        if client.get("/health", headers=AUTH).json()["ready"]:
            return
        time.sleep(0.2)
    pytest.fail("worker never became ready")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    app = create_app(_cfg(tmp_path_factory.mktemp("hw_out")), TOKEN)
    with TestClient(app) as c:
        c.app = app
        _wait_ready(c)
        yield c


def test_auth_required(client):
    assert client.get("/health").status_code == 401
    assert client.get("/health", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert client.post("/v1/images/generations", json={"prompt": "x"}).status_code == 401


def test_health_and_models(client):
    h = client.get("/health", headers=AUTH).json()
    assert h["ready"] and h["status"] == "ready" and h["engines_loaded"] == ["fake"]
    m = client.get("/v1/models", headers=AUTH).json()
    assert m["object"] == "list" and m["data"][0]["id"] == "fake"


def test_openai_images_b64(client):
    r = client.post("/v1/images/generations", headers=AUTH,
                    json={"model": "fake", "prompt": "a cat", "size": "512x768", "n": 2, "seed": 5})
    assert r.status_code == 200, r.text
    data = r.json()["data"]
    assert len(data) == 2 and [d["seed"] for d in data] == [5, 6]
    img = Image.open(io.BytesIO(base64.b64decode(data[0]["b64_json"])))
    assert img.format == "WEBP"


def test_openai_images_url_and_png(client):
    r = client.post("/v1/images/generations", headers=AUTH,
                    json={"prompt": "a dog", "response_format": "url", "output_format": "png"})
    assert r.status_code == 200, r.text
    url = r.json()["data"][0]["url"]
    path = url[url.index("/v1/"):]
    f = client.get(path, headers=AUTH)
    assert f.status_code == 200 and Image.open(io.BytesIO(f.content)).format == "PNG"


def test_validation_errors(client):
    assert client.post("/v1/images/generations", headers=AUTH, json={"prompt": "x", "model": "nope"}).status_code == 404
    assert client.post("/v1/images/generations", headers=AUTH, json={"prompt": "x", "size": "huge"}).status_code == 400
    assert client.post("/v1/images/generations", headers=AUTH, json={"prompt": "x", "size": "4096x4096"}).status_code == 400
    assert client.post("/v1/images/generations", headers=AUTH, json={"prompt": ""}).status_code == 422
    assert client.post("/v1/images/generations", headers=AUTH,
                       json={"prompt": "x", "response_format": "html"}).status_code == 400


def test_async_job_flow(client):
    jid = client.post("/v1/jobs", headers=AUTH, json={"prompt": "tree", "size": "256x256"}).json()["id"]
    for _ in range(100):
        st = client.get(f"/v1/jobs/{jid}", headers=AUTH).json()
        if st["status"] == "done":
            break
        time.sleep(0.1)
    assert st["status"] == "done" and st["file"]
    f = client.get(f"/v1/jobs/{jid}/file?format=jpeg", headers=AUTH)
    assert f.headers["content-type"] == "image/jpeg"
    assert client.get("/v1/jobs/unknown", headers=AUTH).status_code == 404


def test_pack_flow_and_zip(client):
    items = [{"id": f"scene_{i:02d}", "prompt": f"scene {i}", "size": "768x432", "seed": i} for i in range(3)]
    pack = client.post("/v1/packs", headers=AUTH, json={"items": items, "output_format": "webp"}).json()
    assert pack["total"] == 3
    for _ in range(200):
        st = client.get(f"/v1/packs/{pack['pack_id']}", headers=AUTH).json()
        if st["status"] != "running":
            break
        time.sleep(0.1)
    assert st["status"] == "done" and st["done"] == 3
    z = client.get(f"/v1/packs/{pack['pack_id']}/download", headers=AUTH)
    zf = zipfile.ZipFile(io.BytesIO(z.content))
    results = json.loads(zf.read("results.json"))
    assert [r["item_id"] for r in results] == ["scene_00", "scene_01", "scene_02"]
    assert all(r["status"] == "success" and r["file"] in zf.namelist() for r in results)
    dup = client.post("/v1/packs", headers=AUTH, json={"items": [items[0], items[0]]})
    assert dup.status_code == 400


def test_requests_are_logged(client, caplog):
    with caplog.at_level("INFO", logger="http_worker"):
        client.get("/health", headers=AUTH)
    assert any("[REQ] GET /health" in r.getMessage() for r in caplog.records)


def test_shutdown_endpoint(tmp_path):
    app = create_app(_cfg(tmp_path), TOKEN)
    with TestClient(app) as c:
        _wait_ready(c)
        assert c.post("/v1/shutdown", headers=AUTH).json()["status"] == "shutting down"
        time.sleep(1.2)
        assert app.state.shutdown_requested is True


def test_idle_timeout_dead_man_switch(tmp_path):
    app = create_app(_cfg(tmp_path, idle_timeout_minutes=0.01), TOKEN)  # 0.6 s
    with TestClient(app) as c:
        _wait_ready(c)
        for _ in range(100):
            if app.state.shutdown_requested:
                break
            time.sleep(0.1)
        assert app.state.shutdown_requested is True


def test_max_session_cap(tmp_path):
    app = create_app(_cfg(tmp_path, max_session_minutes=0.01), TOKEN)
    with TestClient(app) as c:
        for _ in range(100):
            if app.state.shutdown_requested:
                break
            time.sleep(0.1)
        assert app.state.shutdown_requested is True


def test_policy_refuses_unsafe_engines():
    with pytest.raises(ValueError):
        _check_policy("x", {"test_only": True}, allow_test=False, allow_nc=False)
    with pytest.raises(ValueError):
        _check_policy("x", {"commercial_ok": False}, allow_test=False, allow_nc=False)
    with pytest.raises(ValueError):
        _check_policy("x", {"enabled": False}, allow_test=True, allow_nc=True)
    _check_policy("x", {"enabled": False, "test_only": True}, allow_test=True, allow_nc=False)
    _check_policy("x", {"enabled": True, "commercial_ok": True}, allow_test=False, allow_nc=False)


def test_refuses_to_start_without_token():
    env = {k: v for k, v in os.environ.items() if k not in ("WORKER_TOKEN", "ALLOW_NO_AUTH")}
    r = subprocess.run([sys.executable, "-m", "http_worker.server"], env=env, capture_output=True, text=True,
                       timeout=60)
    assert r.returncode == 2 and "WORKER_TOKEN" in (r.stdout + r.stderr)


def test_failed_engine_load_is_reported(tmp_path):
    cfg = _cfg(tmp_path, roles=[Role(gpu=None, engine="sdxl_turbo")], default_engine="sdxl_turbo")
    app = create_app(cfg, TOKEN)
    with TestClient(app) as c:
        for _ in range(100):
            h = c.get("/health", headers=AUTH).json()
            if h["status"] == "failed":
                break
            time.sleep(0.2)
        assert h["status"] == "failed" and "sdxl_turbo" in h["error"] and not h["ready"]
