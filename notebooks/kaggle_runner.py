"""notebooks/kaggle_runner.py — Standalone Kaggle runner script for yt-gpu-worker.

Key rules:
- Clones this repo at specified tag (--tag or WORKER_TAG env var, default v0.1.0).
- Fails with a clear message if the tag cannot be cloned.
- Accepts --pack-json string and writes it to pack.json before loading.
- After cloning, installs requirements-gpu.txt (never touches PyTorch).
- Prints installed versions of torch, diffusers, transformers, accelerate, bitsandbytes, safetensors, huggingface_hub.
- Calls run_pack with default flags so models.yaml decides parallel_mode and offload.
- Zero secrets in script.
"""

import argparse
import json
import logging
import os
from pathlib import Path
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO_URL = "https://github.com/DHIRAJSHEWALE007/yt-gpu-worker.git"
DEFAULT_TAG = "v0.1.0"


def print_system_diagnostics() -> None:
    """Print system environment: nvidia-smi, free -h, df -h, python --version."""
    print("=" * 80)
    print("yt-gpu-worker: Kaggle Environment Diagnostics")
    print("=" * 80)

    commands = [
        ["python", "--version"],
        ["nvidia-smi"],
        ["free", "-h"],
        ["df", "-h"],
    ]

    for cmd in commands:
        cmd_str = " ".join(cmd)
        print(f"\n--- Running: {cmd_str} ---")
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
            print(res.stdout.strip())
        except Exception as e:
            print(f"Command '{cmd_str}' failed or not available on host: {e}")

    print("=" * 80)


def print_installed_package_versions() -> None:
    """Print installed versions of key worker libraries."""
    packages = [
        "torch",
        "diffusers",
        "transformers",
        "accelerate",
        "bitsandbytes",
        "safetensors",
        "huggingface_hub",
    ]
    print("=" * 80)
    print("yt-gpu-worker: Installed Library Versions")
    print("=" * 80)
    for pkg in packages:
        ver = "not installed"
        try:
            import importlib.metadata
            ver = importlib.metadata.version(pkg)
        except Exception:
            try:
                mod = __import__(pkg)
                ver = getattr(mod, "__version__", "unknown")
            except Exception:
                ver = "not installed"
        print(f"{pkg}: {ver}")
    print("=" * 80)


def install_gpu_requirements(req_path: Path) -> None:
    """Install requirements-gpu.txt without touching torch or torchvision."""
    if not req_path.is_file():
        logger.warning("Requirements file not found at %s; skipping pip install.", req_path)
        return

    logger.info("Installing GPU requirements from %s (preserving torch)...", req_path)
    lines = req_path.read_text(encoding="utf-8").splitlines()
    filtered_lines: list[str] = []
    for line in lines:
        cleaned = line.strip()
        if not cleaned or cleaned.startswith("#"):
            continue
        pkg_name = cleaned.split("==")[0].split(">=")[0].split("<=")[0].strip().lower()
        if pkg_name in ("torch", "torchvision", "torchaudio"):
            logger.info("Preserving pre-installed %s (skipping pin: %s)", pkg_name, cleaned)
            continue
        filtered_lines.append(cleaned)

    if not filtered_lines:
        logger.info("No non-torch requirements to install.")
        return

    temp_req = req_path.parent / "_reqs_tmp_no_torch.txt"
    try:
        temp_req.write_text("\n".join(filtered_lines) + "\n", encoding="utf-8")
        cmd = [sys.executable, "-m", "pip", "install", "-r", str(temp_req)]
        subprocess.run(cmd, check=True)
    finally:
        if temp_req.exists():
            try:
                temp_req.unlink()
            except OSError:
                pass


def setup_repository(
    target_dir: Path,
    tag: str | None = None,
    repo_url: str = REPO_URL,
    install_reqs: bool = True,
    force_clone: bool = False,
) -> Path:
    """Ensure yt-gpu-worker is cloned at the specified tag and available.
    
    Fails with a clear message if the tag cannot be cloned.
    """
    worker_tag = tag or os.environ.get("WORKER_TAG", DEFAULT_TAG)

    did_clone = False
    if force_clone or not (target_dir / "worker" / "runner.py").is_file():
        logger.info("Cloning repository at tag '%s' from %s to %s...", worker_tag, repo_url, target_dir)
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        res = subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", worker_tag, repo_url, str(target_dir)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if res.returncode != 0:
            err_msg = (res.stderr or res.stdout or "").strip()
            raise RuntimeError(
                f"Failed to clone yt-gpu-worker tag '{worker_tag}' from {repo_url}: {err_msg}"
            )
        did_clone = True
    else:
        logger.info("Found worker locally at %s", target_dir)

    if did_clone and install_reqs:
        req_file = target_dir / "requirements-gpu.txt"
        install_gpu_requirements(req_file)
        print_installed_package_versions()

    return target_dir


def resolve_pack_path(user_input: str | None) -> Path:
    """Resolve pack file path with smart fallbacks for Linux bash backslash escaping."""
    from worker.transports.kaggle_pack import find_default_pack_file

    if not user_input:
        return find_default_pack_file()

    raw = str(user_input).strip()
    norm = raw.replace(chr(92), "/")
    repo_root = Path(__file__).resolve().parent.parent

    candidates = [
        Path(raw),
        Path(norm),
        repo_root / norm,
        repo_root / raw,
    ]

    if "testssample" in norm:
        fixed = norm.replace("testssample", "tests/sample")
        candidates.extend([Path(fixed), repo_root / fixed])

    for c in candidates:
        if c.is_file():
            return c.resolve()

    return find_default_pack_file()


def main() -> None:
    parser = argparse.ArgumentParser(description="Kaggle Runner for yt-gpu-worker")
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Worker git tag or branch to clone/checkout (or WORKER_TAG env var, default v0.1.0)",
    )
    parser.add_argument("--pack", type=str, default=None, help="Path to input pack.json")
    parser.add_argument(
        "--pack-json",
        type=str,
        default=None,
        help="JSON string of the pack; written to pack.json before loading",
    )
    parser.add_argument("--output-dir", type=str, default="/kaggle/working/outputs", help="Output directory")
    parser.add_argument("--repo-dir", type=str, default=None, help="Optional path for worker repository")
    args = parser.parse_args()

    # 1. Print environment diagnostic logs
    print_system_diagnostics()

    # 2. Setup repository if needed
    tag = args.tag or os.environ.get("WORKER_TAG", DEFAULT_TAG)
    current_dir = Path(__file__).resolve().parent.parent
    if (current_dir / "worker" / "runner.py").is_file() and not args.repo_dir and not args.tag:
        repo_dir = current_dir
        logger.info("Using local worker repository at %s", repo_dir)
    else:
        target_dir = Path(args.repo_dir) if args.repo_dir else (
            Path("/kaggle/working/yt-gpu-worker") if Path("/kaggle/working").is_dir() else current_dir / "yt-gpu-worker"
        )
        repo_dir = setup_repository(target_dir, tag=tag, install_reqs=True)

    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))

    from worker.schema import load_pack
    from worker.runner import run_pack

    # 3. Handle pack input: --pack-json writes to pack.json before loading
    if args.pack_json:
        target_pack = Path(args.pack) if args.pack else Path("pack.json")
        target_pack.parent.mkdir(parents=True, exist_ok=True)
        target_pack.write_text(args.pack_json, encoding="utf-8")
        pack_path = target_pack
        logger.info("Wrote --pack-json to %s before loading", pack_path.resolve())
    elif args.pack:
        pack_path = resolve_pack_path(args.pack)
    else:
        pack_path = resolve_pack_path(None)

    logger.info("Loading pack from: %s", pack_path)
    pack = load_pack(pack_path)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 4. Execute generation workload with default flags so models.yaml decides parallel_mode and offload
    logger.info("Starting PackRunner on %d items...", len(pack.items))
    results = run_pack(pack, output_dir=out_dir)
    logger.info("Completed %d items. Exiting cleanly.", len(results))


if __name__ == "__main__":
    main()
