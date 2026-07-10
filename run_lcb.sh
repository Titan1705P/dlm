#!/usr/bin/env bash
set -euo pipefail

# ── Config (edit these defaults) ─────────────────────────────────────────────
MODEL="Qwen/Qwen3-4B-Instruct-2507"
RELEASE="release_v6"
SCENARIO="codegeneration"
N_SAMPLES=10
TEMPERATURE=0.2
TOP_P=0.95
MAX_TOKENS=2048
GPU="0,1,2,3,4,5,6,7"    # all 8 B200s
TP_SIZE=1                 # no tensor parallelism (model fits on 1 GPU)
DP_SIZE=8                 # data parallelism: 8 copies, 8× throughput
DTYPE="bfloat16"
MAX_MODEL_LEN=8192
NUM_EVAL_PROCS=12
TIMEOUT=10
# ─────────────────────────────────────────────────────────────────────────────

export CUDA_VISIBLE_DEVICES="${GPU}"

echo "=== Running LiveCodeBench ${RELEASE} ==="
echo "  Model:       ${MODEL}"
echo "  Scenario:    ${SCENARIO}"
echo "  Samples:     ${N_SAMPLES}"
echo "  Temperature: ${TEMPERATURE}"
echo "  GPU:         ${GPU}"
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
    --num_process_evaluate "${NUM_EVAL_PROCS}" \
    --timeout "${TIMEOUT}" \
    --evaluate

echo ""
echo "=== Done. Results in LiveCodeBench/output/${MODEL}/ ==="
