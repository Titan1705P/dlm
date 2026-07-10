#!/usr/bin/env bash
set -euo pipefail

# Assumes conda env 'dlm-eval' (Python 3.12) is already created and active.
# Create it with: conda create -n dlm-eval python=3.12 -y

echo "=== Installing LiveCodeBench ==="
pip install -e ./LiveCodeBench
pip install "datasets<3.0"

echo "=== Installing lm-evaluation-harness ==="
pip install lm-eval

echo "=== Done ==="

