"""tests/test_transports.py — Kaggle pack adapter, HTTP server, and Kaggle runner tests."""

import json
import logging
from pathlib import Path
import subprocess
import sys

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


def test_missing_tag_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that failure to clone the specified tag raises RuntimeError with a clear message."""
    from notebooks.kaggle_runner import setup_repository

    def mock_run(cmd, *args, **kwargs):
        if "clone" in cmd:
            tag = cmd[cmd.index("--branch") + 1]
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=128,
                stdout="",
                stderr=f"fatal: Remote branch '{tag}' not found in upstream origin",
            )
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", mock_run)

    target_dir = tmp_path / "worker_clone"
    bad_tag = "nonexistent-tag-v9.9.9"

    with pytest.raises(RuntimeError) as excinfo:
        setup_repository(target_dir, tag=bad_tag, install_reqs=False)

    msg = str(excinfo.value)
    assert f"Failed to clone yt-gpu-worker tag '{bad_tag}'" in msg
    assert "fatal: Remote branch 'nonexistent-tag-v9.9.9' not found" in msg

    # Also test environment variable fallback WORKER_TAG
    monkeypatch.setenv("WORKER_TAG", "env-tag-v8.8.8")
    with pytest.raises(RuntimeError) as excinfo_env:
        setup_repository(target_dir, tag=None, install_reqs=False)

    msg_env = str(excinfo_env.value)
    assert "Failed to clone yt-gpu-worker tag 'env-tag-v8.8.8'" in msg_env


def test_pack_json_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that --pack-json writes the raw JSON string to pack.json and loads correctly."""
    from worker.schema import Pack, Item
    from notebooks.kaggle_runner import main

    monkeypatch.setenv("STUDIO_FAKE_PROVIDERS", "1")

    pack_data = {
        "schema_version": "1.0.0",
        "pack_id": "test_pack_roundtrip",
        "output_dir": str(tmp_path / "out"),
        "items": [
            {
                "id": "item_1",
                "engine": "fake",
                "prompt": "Test embedded JSON pack prompt",
                "width": 64,
                "height": 64,
            }
        ],
    }
    json_str = json.dumps(pack_data)

    target_pack_file = tmp_path / "pack.json"
    out_dir = tmp_path / "results_out"

    test_args = [
        "kaggle_runner.py",
        "--pack",
        str(target_pack_file),
        "--pack-json",
        json_str,
        "--output-dir",
        str(out_dir),
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    captured_pack: list[Pack] = []

    def mock_run_pack(pack: Pack, output_dir=None, **kwargs):
        captured_pack.append(pack)
        return []

    monkeypatch.setattr("worker.runner.run_pack", mock_run_pack)

    main()

    # 1. pack.json was created on disk with exact json_str
    assert target_pack_file.is_file()
    written_data = json.loads(target_pack_file.read_text(encoding="utf-8"))
    assert written_data["pack_id"] == "test_pack_roundtrip"
    assert len(written_data["items"]) == 1
    assert written_data["items"][0]["prompt"] == "Test embedded JSON pack prompt"

    # 2. Pack was loaded and passed to run_pack
    assert len(captured_pack) == 1
    loaded = captured_pack[0]
    assert loaded.pack_id == "test_pack_roundtrip"
    assert loaded.items[0].id == "item_1"
    assert loaded.items[0].prompt == "Test embedded JSON pack prompt"


def test_requirements_gpu_pins() -> None:
    """Verify requirements-gpu.txt contains only == pins and no >= constraints."""
    repo_root = Path(__file__).resolve().parent.parent
    req_file = repo_root / "requirements-gpu.txt"
    assert req_file.is_file(), f"requirements-gpu.txt not found at {req_file}"

    lines = req_file.read_text(encoding="utf-8").splitlines()
    non_empty = [l.strip() for l in lines if l.strip()]

    # Verify header comment
    assert any("Tested on Kaggle T4 x2, Python 3.13" in l for l in non_empty)

    package_lines = [l for l in non_empty if not l.startswith("#")]
    assert len(package_lines) >= 10, "Expected at least 10 package pins"

    for line in package_lines:
        assert ">=" not in line, f"Found '>=' constraint in {line}, all pins must be '=='"
        assert "<=" not in line, f"Found '<=' constraint in {line}, all pins must be '=='"
        assert "==" in line, f"Missing '==' pin in {line}"

    # Verify key packages are present
    pins = {l.split("==")[0]: l.split("==")[1] for l in package_lines}
    assert pins["accelerate"] == "1.14.0"
    assert pins["bitsandbytes"] == "0.50.2"
    assert pins["diffusers"] == "0.40.0"
    assert pins["huggingface_hub"] == "1.29.0"
    assert pins["numpy"] == "2.1.3"
    assert pins["pillow"] == "12.3.0"
    assert pins["psutil"] == "5.9.5"
    assert pins["pydantic"] == "2.13.5"
    assert pins["PyYAML"] == "6.0.3"
    assert pins["safetensors"] == "0.8.0"
    assert pins["transformers"] == "5.16.1"
    assert "torch" not in pins, "torch pin must be removed from requirements-gpu.txt"
    assert "torchvision" not in pins, "torchvision pin must be removed from requirements-gpu.txt"

    # Verify full freeze reference exists in requirements-tested-env.txt
    freeze_file = repo_root / "requirements-tested-env.txt"
    assert freeze_file.is_file(), "requirements-tested-env.txt must exist"
    freeze_text = freeze_file.read_text(encoding="utf-8")
    assert "torch==" in freeze_text
    assert "torchvision==" in freeze_text


def test_runner_parallel_mode_yaml_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """Verify that if models.yaml has no parallel_mode, runner logs a warning and uses process_per_gpu."""
    from worker.schema import Item, Pack
    import worker.runner as wr

    monkeypatch.setenv("STUDIO_FAKE_PROVIDERS", "1")

    original_load = wr._load_models_config

    def mock_load_models():
        cfg = original_load()
        cfg["fake"] = dict(cfg["fake"])
        cfg["fake"].pop("parallel_mode", None)
        return cfg

    monkeypatch.setattr(wr, "_load_models_config", mock_load_models)

    from worker.memory import GPUInfo
    fake_gpus = [
        GPUInfo(total_vram_mb=15000, free_vram_mb=14000, index=0, name="GPU 0", compute_capability=(7, 5)),
        GPUInfo(total_vram_mb=15000, free_vram_mb=14000, index=1, name="GPU 1", compute_capability=(7, 5)),
    ]
    monkeypatch.setattr("worker.runner.detect_gpus", lambda: fake_gpus)

    pack = Pack(
        pack_id="test_no_pm_pack",
        items=[Item(id="item1", engine="fake", prompt="test", width=64, height=64)],
        output_dir=str(tmp_path / "out"),
    )

    with caplog.at_level(logging.INFO):
        results = wr.run_pack(pack, output_dir=tmp_path / "out", gpu_indices=[0, 1], allow_test_engines=True)

    assert any("No parallel_mode configured in models.yaml for engine 'fake'" in rec.message for rec in caplog.records)
    assert any("mode=process_per_gpu" in rec.message for rec in caplog.records)
    assert len(results) == 1
    assert results[0].status == "success"
