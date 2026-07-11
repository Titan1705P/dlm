#!/usr/bin/env bash
set -euo pipefail

# ── Config (edit these defaults) ─────────────────────────────────────────────
MODEL="Qwen/Qwen3-4B-Instruct-2507"
TASKS="gpqa_diamond_zeroshot,aime25"
GPU="0"
BATCH_SIZE="auto"
NUM_FEWSHOT=5
# ─────────────────────────────────────────────────────────────────────────────

export CUDA_VISIBLE_DEVICES="${GPU}"

echo "=== Running lm-evaluation-harness ==="
echo "  Model: ${MODEL}"
echo "  Tasks: ${TASKS}"
echo "  GPU:   ${GPU}"
echo ""

lm_eval \
    --model vllm \
    --model_args "pretrained=${MODEL},dtype=bfloat16,trust_remote_code=True,max_model_len=16384,gpu_memory_utilization=0.8" \
    --tasks "${TASKS}" \
    --batch_size "${BATCH_SIZE}" \
    --num_fewshot "${NUM_FEWSHOT}" \
    --gen_kwargs "max_gen_toks=12000,temperature=0.0" \
    --output_path "$(dirname "$0")/eval_results/lm_eval"

echo ""
echo "============================================================"
echo "  RESULTS"
echo "============================================================"

# Display results from the latest output
python3 -c "
import json, glob, os

result_dir = '$(dirname \"\$0\")/eval_results/lm_eval'
files = glob.glob(os.path.join(result_dir, '**', 'results.json'), recursive=True)
if not files:
    print('  No results found.')
    exit()

latest = sorted(files, key=os.path.getmtime)[-1]
with open(latest) as f:
    data = json.load(f)

results = data.get('results', {})
print()
for task, metrics in results.items():
    acc = metrics.get('acc,none', metrics.get('exact_match,none', None))
    stderr = metrics.get('acc_stderr,none', metrics.get('exact_match_stderr,none', ''))
    if acc is not None:
        stderr_str = f' +/- {stderr:.4f}' if isinstance(stderr, float) else ''
        print(f'  {task:30s} {acc:.4f}{stderr_str}')
print()
"
echo "============================================================"
echo ""
echo "=== Done. Results in eval_results/lm_eval/ ==="
