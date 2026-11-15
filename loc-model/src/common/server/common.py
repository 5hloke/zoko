"""Shared helpers for serve.py, server_vllm.py and server_mlx.py."""
import os
import platform
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# Which model to serve (override with environment variables).
MODEL_CARD = os.environ.get("MODEL_CARD", "Qwen/Qwen3.5-9B")                 # HF bf16 weights (vLLM)
MLX_REPO = os.environ.get("MLX_REPO", "mlx-community/Qwen3.5-9B-OptiQ-4bit")  # pre-converted weights (MLX)

# Weights live in <MODELS_DIR>/<repo_id>, e.g. ./models/Qwen/Qwen3.5-9B
MODELS_DIR = Path(os.environ.get("LOCAL_MODEL_ROOT", "./models")).expanduser().resolve()

HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8000"))


def ensure_weights(repo_id: str) -> Path:
    """Download repo_id into MODELS_DIR once; reuse it on later runs."""
    target = MODELS_DIR / repo_id
    complete = (target / "config.json").exists() and any(target.glob("*.safetensors"))
    if complete:
        print(f"Using local weights at {target}", flush=True)
        return target
    from huggingface_hub import snapshot_download
    target.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {repo_id} -> {target}", flush=True)
    snapshot_download(repo_id=repo_id, local_dir=target)  # resumes if interrupted
    return target


def detect_backend() -> str:
    """Return 'vllm', 'mlx' or 'transformers'. Override with BACKEND=..."""
    forced = os.environ.get("BACKEND")
    if forced:
        return forced
    try:
        import torch
        if torch.cuda.is_available():
            return "vllm"
    except ImportError:
        pass
    if sys.platform == "darwin" and platform.machine() == "arm64":
        return "mlx"
    return "transformers"


def wait_ready(proc: subprocess.Popen, timeout: int = 900) -> bool:
    """Poll /v1/models until the server answers, or the process dies."""
    url = f"http://{HOST}:{PORT}/v1/models"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(2)
    return False


def serve_forever(cmd: list[str]) -> int:
    """Launch cmd, report when ready, stop it cleanly on Ctrl-C."""
    print("Launching:", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd)
    try:
        if wait_ready(proc):
            print(f"Server ready at http://{HOST}:{PORT}/v1", flush=True)
        return proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        try:
            return proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            return 1
