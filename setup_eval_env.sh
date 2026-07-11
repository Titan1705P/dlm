#!/usr/bin/env bash
set -euo pipefail

# Assumes conda env (Python 3.12) is already created and active.
# Create it with: conda create -n eval python=3.12 -y && conda activate eval

echo "=== Installing PyTorch ==="
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo "=== Installing vLLM (>=0.24.0 for DiffusionGemma) ==="
pip install "vllm>=0.24.0"

echo "=== Installing flash-attn ==="
# Try pre-built wheel first, fall back to source build
pip install flash-attn 2>/dev/null || pip install flash-attn --no-build-isolation 2>/dev/null || echo "WARNING: flash-attn install failed, will use triton attention backend"

echo "=== Installing LiveCodeBench ==="
pip install -e ./LiveCodeBench
pip install "datasets<3.0"

echo "=== Installing lm-evaluation-harness ==="
pip install lm-eval accelerate

echo "=== Installing additional dependencies ==="
pip install transformers tokenizers sentencepiece protobuf huggingface_hub

echo "=== Done ==="
python -c 'import vllm; print(vllm.__version__)'

