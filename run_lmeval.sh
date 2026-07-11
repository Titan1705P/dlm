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
    --model_args "pretrained=${MODEL},dtype=bfloat16,trust_remote_code=True,max_model_len=4096,gpu_memory_utilization=0.8" \
    --tasks "${TASKS}" \
    --batch_size "${BATCH_SIZE}" \
    --num_fewshot "${NUM_FEWSHOT}" \
    --gen_kwargs "max_gen_toks=256,temperature=0.0" \
    --output_path "$(dirname "$0")/eval_results/lm_eval"

echo ""
echo "=== Done. Results in eval_results/lm_eval/ ==="
