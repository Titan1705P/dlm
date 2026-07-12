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
if grep -q "PATCHED for DiffusionGemma" "$FLASHINFER_FILE"; then
    echo "Already patched."
    exit 0
fi

# Patch: find `if causal:` at line ~951 and add tensor handling
python3 << 'EOF'
import vllm, os

flashinfer_path = os.path.join(os.path.dirname(vllm.__file__), 'v1/attention/backends/flashinfer.py')

with open(flashinfer_path, 'r') as f:
    lines = f.readlines()

patched = False
for i, line in enumerate(lines):
    if line.strip() == 'if causal:' and 'causal = common_attn_metadata.causal' in lines[i-1]:
        indent = line[:len(line) - len(line.lstrip())]
        patch_lines = [
            f"{indent}# PATCHED for DiffusionGemma: causal can be a tensor for mixed attention\n",
            f"{indent}import torch as _torch\n",
            f"{indent}if isinstance(causal, _torch.Tensor):\n",
            f"{indent}    causal = causal.all().item()\n",
        ]
        lines[i:i] = patch_lines
        patched = True
        break

if patched:
    with open(flashinfer_path, 'w') as f:
        f.writelines(lines)
    print(f"Patched {flashinfer_path} successfully.")
else:
    print("WARNING: Could not find target line to patch.")
    exit(1)
EOF
