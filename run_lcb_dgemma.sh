#!/usr/bin/env bash
set -euo pipefail

# ── Config ───────────────────────────────────────────────────────────────────
MODEL="google/diffusiongemma-26B-A4B-it"
RELEASE="release_v6"
SCENARIO="codegeneration"
N_SAMPLES=1
TEMPERATURE=0.0
TOP_P=0.95
MAX_TOKENS=2048
GPU="0"
TP_SIZE=1
DP_SIZE=1
DTYPE="bfloat16"
MAX_MODEL_LEN=8192
GPU_MEM_UTIL=0.7
NUM_EVAL_PROCS=12
TIMEOUT=10
# ─────────────────────────────────────────────────────────────────────────────

export CUDA_VISIBLE_DEVICES="${GPU}"
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1

echo "=== Running LiveCodeBench ${RELEASE} ==="
echo "  Model:       ${MODEL}"
echo "  Scenario:    ${SCENARIO}"
echo "  Samples:     ${N_SAMPLES}"
echo "  Temperature: ${TEMPERATURE}"
echo "  GPU:         ${GPU}"
echo "  TP:          ${TP_SIZE}"
echo ""

cd "$(dirname "$0")/LiveCodeBench"

python -m lcb_runner.runner.main \
    --model "${MODEL}" \
    --scenario "${SCENARIO}" \
    --release_version "${RELEASE}" \
    --n "${N_SAMPLES}" \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    --max_tokens "${MAX_TOKENS}" \
    --tensor_parallel_size "${TP_SIZE}" \
    --data_parallel_size "${DP_SIZE}" \
    --dtype "${DTYPE}" \
    --max_model_len "${MAX_MODEL_LEN}" \
    --gpu_memory_utilization "${GPU_MEM_UTIL}" \
    --attention_backend "TRITON_ATTN" \
    --num_process_evaluate "${NUM_EVAL_PROCS}" \
    --timeout "${TIMEOUT}" \
    --evaluate

echo ""
echo "=== Done. Results in LiveCodeBench/output/ ==="
echo ""

# Display score and sample trajectories
python3 -c "
import json, glob, os

pattern = 'output/**/Scenario.codegeneration_*_eval_all.json'
files = glob.glob(pattern, recursive=True)
if not files:
    print('No eval results found.')
    exit()

eval_file = sorted(files)[-1]
with open(eval_file) as f:
    data = json.load(f)

total = len(data)
passed_count = sum(1 for item in data if item.get('pass@1', 0) > 0)
score = passed_count / total if total > 0 else 0

print('=' * 60)
print(f'  SCORE: pass@1 = {score:.4f} ({passed_count}/{total})')
print('=' * 60)
print()

passed = [item for item in data if item.get('pass@1', 0) > 0]
failed = [item for item in data if item.get('pass@1', 0) == 0]

def show(item, label):
    print(f'-- {label} --')
    print(f'  Title:      {item[\"question_title\"]}')
    print(f'  Difficulty: {item[\"difficulty\"]}')
    print(f'  Platform:   {item[\"platform\"]}')
    code = (item.get('code_list') or [''])[0]
    lines = code.strip().split('\n')
    preview = '\n'.join(lines[:12])
    if len(lines) > 12:
        preview += f'\n    ... ({len(lines)-12} more lines)'
    print(f'  Code:')
    for l in preview.split('\n'):
        print(f'    {l}')
    print()

print('-- PASSING SAMPLES --')
print()
for item in passed[:2]:
    show(item, 'PASS')

print('-- FAILING SAMPLES --')
print()
for item in failed[:2]:
    meta = (item.get('metadata') or [''])[0]
    if 'timeout' in meta.lower() or 'time limit' in meta.lower():
        show(item, 'TIMEOUT')
    else:
        show(item, 'WRONG ANSWER')
"
