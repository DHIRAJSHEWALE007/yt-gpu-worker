"""tests/test_transports.py — Kaggle pack adapter and HTTP server tests."""

from pathlib import Path
from fastapi.testclient import TestClient
import pytest

from worker.schema import Item, Pack, save_pack
from worker.transports.http_server import app, set_auth_token
from worker.transports.kaggle_pack import execute_kaggle_pack


def test_kaggle_pack_execution(tmp_path: Path) -> None:
    pack = Pack(
        pack_id="kaggle_test_pack",
        items=[Item(id="kshot_1", engine="fake", prompt="Kaggle test")],
        output_dir=str(tmp_path / "k_out"),
    )
    pack_file = tmp_path / "pack.json"
    save_pack(pack, pack_file)

    zip_bundle = execute_kaggle_pack(pack_file)
    assert zip_bundle.is_file()
    assert zip_bundle.suffix == ".zip"


def test_http_server_endpoints() -> None:
    set_auth_token("test_secret_token")
    client = TestClient(app)
    auth = {"Authorization": "Bearer test_secret_token"}

    # 1. Health
    h_resp = client.get("/health", headers=auth)
    assert h_resp.status_code == 200
    assert h_resp.json()["status"] == "healthy"

    # 2. Unauthorized without token
    bad_resp = client.get("/health")
    assert bad_resp.status_code == 401

    # 3. Synchronous Image Generation
    img_resp = client.post(
        "/v1/image",
        headers=auth,
        json={"prompt": "A green valley", "width": 64, "height": 64, "engine": "fake"},
    )
    assert img_resp.status_code == 200
    assert img_resp.headers["content-type"] == "image/png"
    assert len(img_resp.content) > 0

    # 4. Asynchronous Video Generation
    vid_resp = client.post(
        "/v1/video",
        headers=auth,
        json={"prompt": "Camera sweeping over cliffs", "seconds": 2.0, "engine": "fake"},
    )
    assert vid_resp.status_code == 200
    job_data = vid_resp.json()
    assert "job_id" in job_data
    assert job_data["status"] in ("queued", "done")

    # 5. Poll Video Status
    job_id = job_data["job_id"]
    poll_resp = client.get(f"/v1/jobs/{job_id}", headers=auth)
    assert poll_resp.status_code == 200
    assert poll_resp.json()["job_id"] == job_id

    # Reset token
    set_auth_token(None)
