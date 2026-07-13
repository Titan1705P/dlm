#!/usr/bin/env bash
set -euo pipefail

# Patch vLLM for DiffusionGemma compatibility:
# 1. FlashInfer: handle tensor-valued `causal` flag
# 2. DiffusionGemma: remove @torch.compile decorators
# 3. DiffusionGemma: fix dtype mismatch in self-conditioning step

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
            print("  [1/3] Patched FlashInfer causal tensor bug")
        else:
            print("  [1/3] WARNING: Could not find FlashInfer target line")
    else:
        print("  [1/3] FlashInfer already patched")
else:
    print("  [1/3] FlashInfer file not found, skipping")

# Patch 2 & 3: DiffusionGemma fixes
dgemma_path = os.path.join(vllm_dir, 'model_executor/models/diffusion_gemma.py')
if os.path.exists(dgemma_path):
    with open(dgemma_path, 'r') as f:
        content = f.read()

    modified = False

    # Patch 2: Remove @torch.compile
    if '# PATCHED: torch.compile removed' not in content:
        new_content = content.replace(
            '@torch.compile(dynamic=True)',
            '# PATCHED: torch.compile removed for eager compatibility\n# @torch.compile(dynamic=True)'
        )
        if new_content != content:
            content = new_content
            modified = True
            print("  [2/3] Removed @torch.compile decorators")
        else:
            print("  [2/3] WARNING: Could not find @torch.compile")
    else:
        print("  [2/3] @torch.compile already patched")

    # Patch 3: Fix dtype mismatch in sc_embeds assignment
    old_line = '    sc_embeds[decode_slots] = soft_embeds * sc_keep'
    new_line = '    sc_embeds[decode_slots] = (soft_embeds * sc_keep).to(sc_embeds.dtype)  # PATCHED: dtype fix'
    if '# PATCHED: dtype fix' not in content:
        if old_line in content:
            content = content.replace(old_line, new_line)
            modified = True
            print("  [3/3] Fixed dtype mismatch in sc_embeds assignment")
        else:
            print("  [3/3] WARNING: Could not find sc_embeds target line")
    else:
        print("  [3/3] dtype fix already patched")

    if modified:
        with open(dgemma_path, 'w') as f:
            f.write(content)
else:
    print("  [2/3] diffusion_gemma.py not found, skipping")
    print("  [3/3] skipped")

print("=== Patching complete ===")
EOF
