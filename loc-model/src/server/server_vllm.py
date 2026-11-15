"""Serve Qwen3.5-9B with vLLM (Linux + NVIDIA GPU) from local weights.

Usage:  python server_vllm.py [extra vllm args...]
"""
import os
import shutil
import sys

from common import HOST, MODEL_CARD, PORT, ensure_weights, serve_forever


def build_cmd(extra: list[str] | None = None) -> list[str]:
    if shutil.which("vllm") is None:
        sys.exit("vllm not found. Install it with: uv pip install -U vllm --torch-backend auto")
    path = ensure_weights(MODEL_CARD)
    cmd = [
        "vllm", "serve", str(path),
        "--served-model-name", MODEL_CARD,          # keeps opencode.json model id stable
        "--host", HOST, "--port", str(PORT),
        "--max-model-len", os.environ.get("MAX_MODEL_LEN", "65536"),
        "--enable-auto-tool-choice",
        "--tool-call-parser", "qwen3_coder",        # Qwen3.5's XML-style tool calls
        "--reasoning-parser", "qwen3",              # separates thinking from the answer
        "--language-model-only",                    # skip the vision tower
        "--enable-prefix-caching",                  # reuse OpenCode's long system prompt
    ]
    return cmd + (extra or [])


def main() -> int:
    return serve_forever(build_cmd(sys.argv[1:]))


if __name__ == "__main__":
    sys.exit(main())
