#!/usr/bin/env bash
set -euo pipefail

# Patch vLLM's FlashInfer backend to handle tensor-valued `causal` flag
# (needed for DiffusionGemma which uses mixed causal/bidirectional attention)

FLASHINFER_FILE=$(python3 -c "import vllm; import os; print(os.path.join(os.path.dirname(vllm.__file__), 'v1/attention/backends/flashinfer.py'))")

if [ ! -f "$FLASHINFER_FILE" ]; then
    echo "ERROR: Cannot find flashinfer.py at $FLASHINFER_FILE"
    exit 1
fi

# Check if already patched
if grep -q "# PATCHED for DiffusionGemma" "$FLASHINFER_FILE"; then
    echo "Already patched."
    exit 0
fi

# The bug is: `if causal:` where causal can be a tensor
# Fix: convert to bool if it's a tensor
python3 -c "
import re

with open('$FLASHINFER_FILE', 'r') as f:
    content = f.read()

# Find the line 'if causal:' around line 951 in the build() method
# Replace with a safe check that handles tensor values
old = '        if causal:'
new = '        # PATCHED for DiffusionGemma: causal can be a tensor for mixed attention
        import torch
        if isinstance(causal, torch.Tensor):
            causal = causal.all().item()
        if causal:'

if old in content:
    content = content.replace(old, new, 1)
    with open('$FLASHINFER_FILE', 'w') as f:
        f.write(content)
    print(f'Patched {\"$FLASHINFER_FILE\"} successfully.')
else:
    print('WARNING: Could not find target line to patch. File may have different structure.')
    exit(1)
"
