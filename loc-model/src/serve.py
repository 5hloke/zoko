"""Pick the right backend for this machine and serve the model.

NVIDIA GPU        -> server_vllm.py
Apple Silicon Mac -> server_mlx.py
anything else     -> server.py (your transformers + FastAPI server)

Force a backend with BACKEND=vllm|mlx|transformers.
"""
import sys

from common.server.common import HOST, PORT, detect_backend, serve_forever


def main() -> int:
    backend = detect_backend()
    print(f"Backend: {backend}", flush=True)
    if backend == "vllm":
        import server.server_vllm as server_vllm
        return server_vllm.main()
    if backend == "mlx":
        import server.server_mlx as server_mlx
        return server_mlx.main()
    if backend == "transformers":
        cmd = [sys.executable, "-m", "uvicorn", "server:app", "--host", HOST, "--port", str(PORT)]
        return serve_forever(cmd)
    sys.exit(f"Unknown BACKEND '{backend}'")


if __name__ == "__main__":
    sys.exit(main())
