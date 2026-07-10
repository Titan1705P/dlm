#!/usr/bin/env bash
set -euo pipefail

# ── Config (edit these defaults) ─────────────────────────────────────────────
MODEL="Qwen/Qwen3-4B-Instruct-2507"
TASKS="mmlu_pro,gpqa_diamond"
GPU="1"
BATCH_SIZE="auto"
NUM_FEWSHOT=5
# ─────────────────────────────────────────────────────────────────────────────

eval "$(conda shell.bash hook)"
conda activate dlm-eval

export CUDA_VISIBLE_DEVICES="${GPU}"

echo "=== Running lm-evaluation-harness ==="
echo "  Model: ${MODEL}"
echo "  Tasks: ${TASKS}"
echo "  GPU:   ${GPU}"
echo ""

lm_eval \
    --model hf \
    --model_args "pretrained=${MODEL},dtype=bfloat16,trust_remote_code=True" \
    --tasks "${TASKS}" \
    --batch_size "${BATCH_SIZE}" \
    --num_fewshot "${NUM_FEWSHOT}" \
    --output_path "$(dirname "$0")/eval_results/lm_eval"

echo ""
echo "=== Done. Results in eval_results/lm_eval/ ==="
