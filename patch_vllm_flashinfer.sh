#!/usr/bin/env bash
set -euo pipefail

# Patch vLLM for DiffusionGemma compatibility:
# 1. FlashInfer: handle tensor-valued `causal` flag
# 2. DiffusionGemma: remove @torch.compile decorators (broken with dynamic shapes on B200)

echo "=== Patching vLLM for DiffusionGemma ==="

python3 << 'EOF'
import vllm, os

vllm_dir = os.path.dirname(vllm.__file__)

# Patch 1: FlashInfer tensor causal bug
flashinfer_path = os.path.join(vllm_dir, 'v1/attention/backends/flashinfer.py')
if os.path.exists(flashinfer_path):
    with open(flashinfer_path, 'r') as f:
        lines = f.readlines()
    if 'PATCHED for DiffusionGemma' not in ''.join(lines):
        patched = False
        for i, line in enumerate(lines):
            if line.strip() == 'if causal:' and i > 0 and 'causal = common_attn_metadata.causal' in lines[i-1]:
                indent = line[:len(line) - len(line.lstrip())]
                patch_lines = [
                    f"{indent}# PATCHED for DiffusionGemma: causal can be a tensor\n",
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
            print(f"  [1/2] Patched FlashInfer causal tensor bug")
        else:
            print(f"  [1/2] WARNING: Could not find FlashInfer target line")
    else:
        print(f"  [1/2] FlashInfer already patched")
else:
    print(f"  [1/2] FlashInfer file not found, skipping")

# Patch 2: Remove @torch.compile from diffusion_gemma.py
dgemma_path = os.path.join(vllm_dir, 'model_executor/models/diffusion_gemma.py')
if os.path.exists(dgemma_path):
    with open(dgemma_path, 'r') as f:
        content = f.read()
    if '# PATCHED: torch.compile removed' not in content:
        original = content
        content = content.replace(
            '@torch.compile(dynamic=True)',
            '# PATCHED: torch.compile removed for eager compatibility\n# @torch.compile(dynamic=True)'
        )
        if content != original:
            with open(dgemma_path, 'w') as f:
                f.write(content)
            print(f"  [2/2] Removed @torch.compile decorators from diffusion_gemma.py")
        else:
            print(f"  [2/2] WARNING: Could not find @torch.compile in diffusion_gemma.py")
    else:
        print(f"  [2/2] diffusion_gemma.py already patched")
else:
    print(f"  [2/2] diffusion_gemma.py not found, skipping")

print("=== Patching complete ===")
EOF
