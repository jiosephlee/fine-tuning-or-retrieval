#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs

GPU_MEMORY_THRESHOLD_MIB="${GPU_MEMORY_THRESHOLD_MIB:-2000}"
POLL_SECONDS="${POLL_SECONDS:-300}"

echo "$(date -Is) Waiting for all 8 GPUs to have <= ${GPU_MEMORY_THRESHOLD_MIB} MiB used before starting E39-E41"

while true; do
    free_enough_gpus="$(
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
            | awk -v threshold="$GPU_MEMORY_THRESHOLD_MIB" '{ if (($1 + 0) <= threshold) count++ } END { print count + 0 }'
    )"
    memory_used="$(
        nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
            | tr '\n' '; '
    )"

    echo "$(date -Is) free_enough_gpus=${free_enough_gpus} memory_used=(${memory_used})"
    if [[ "$free_enough_gpus" -eq 8 ]]; then
        break
    fi

    sleep "$POLL_SECONDS"
done

export CONDA_ENV="${CONDA_ENV:-tuning}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-8}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/data1/joseph/huggingface/hub}"
export HF_HOME="${HF_HOME:-/data1/joseph/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/data1/joseph/huggingface/hub}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export USE_PARCC="${USE_PARCC:-0}"
export INFERENCE_MCQA_PROMPT_COLUMN="${INFERENCE_MCQA_PROMPT_COLUMN:-formatted_question_5shot}"

echo "$(date -Is) Starting E39"
PUSH_TO_HUB_CPT_ID="${E39_PUSH_TO_HUB_CPT_ID:-e39-qwen2-5-7b-source-docmatch-full-20260713}" \
    bash E39_qwen2_5_source_training_doc_match_full.sh

echo "$(date -Is) Starting E40"
PUSH_TO_HUB_CPT_ID="${E40_PUSH_TO_HUB_CPT_ID:-e40-qwen2-5-7b-para9-docmatch-full-20260713}" \
    bash E40_qwen2_5_paraphrase_training_doc_match_full.sh

echo "$(date -Is) Starting E41"
PUSH_TO_HUB_CPT_ID="${E41_PUSH_TO_HUB_CPT_ID:-e41-qwen2-5-7b-para9-expl-full-20260713}" \
    bash E41_qwen2_5_granular_explanations_full.sh

echo "$(date -Is) Completed E39-E41"
