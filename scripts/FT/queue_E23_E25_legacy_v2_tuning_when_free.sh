#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs

GPU_MEMORY_THRESHOLD_MIB="${GPU_MEMORY_THRESHOLD_MIB:-2000}"
POLL_SECONDS="${POLL_SECONDS:-300}"

echo "$(date -Is) Waiting for all 8 GPUs to have <= ${GPU_MEMORY_THRESHOLD_MIB} MiB used before starting E23-E25"

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
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/data1/joseph/huggingface/hub}"
export HF_HOME="${HF_HOME:-/data1/joseph/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/data1/joseph/huggingface/hub}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

echo "$(date -Is) Starting E23"
bash E23_source_training_legacy_v2_local.sh

echo "$(date -Is) Starting E24"
bash E24_paraphrase_training_legacy_v2_local_no_explanation_match.sh

echo "$(date -Is) Starting E25"
bash E25_explanations_legacy_v2_local.sh

echo "$(date -Is) Completed E23-E25"
