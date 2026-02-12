#!/usr/bin/env bash
set -euxo pipefail

echo "▶ Activating py310 virtual environment..."
source /data/ephemeral/home/py310/bin/activate

echo "▶ Installing uv (if not installed)..."
pip install -U uv

echo "▶ Installing Python requirements..."
uv pip install -r requirements-py310.txt

echo "▶ Installing / Updating vLLM with torch backend auto..."
uv pip install -U vllm --torch-backend=auto

echo "py310 environment setup complete."
