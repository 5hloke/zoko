"""Serve Qwen3.5-9B with mlx-lm (Apple Silicon) from local weights.

Usage:  python server_mlx.py [extra mlx_lm.server args...]
"""
import sys

from common.server.common import HOST, MLX_REPO, PORT, ensure_weights, serve_forever


def build_cmd(extra: list[str] | None = None) -> list[str]:
    try:
        import mlx_lm  # noqa: F401
    except ImportError:
        sys.exit("mlx-lm not found. Install it with: pip install -U mlx-lm")
    path = ensure_weights(MLX_REPO)
    cmd = [
        sys.executable, "-m", "mlx_lm.server",
        "--model", str(path),
        "--host", HOST,
        "--port", str(PORT),
    ]
    return cmd + (extra or [])


def main() -> int:
    return serve_forever(build_cmd(sys.argv[1:]))


if __name__ == "__main__":
    sys.exit(main())
