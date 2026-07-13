#!/bin/bash
# Launch Gemma 4 12B and NVIDIA Gemma 4 31B NVFP4 over all multiview domains.
# Six one-GPU jobs × 90 minutes gives a hard submitted ceiling of 9 GPU-hours.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT="$SCRIPT_DIR/submit_vllm_multiview.sh"

DRY_RUN=""
MODELS=(12B 31B-NVFP4)
DOMAINS=(arxiv medical legal)
MAX_WORKERS=16
TIME_LIMIT="01:30:00"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN="--dry-run"; shift ;;
        --models) shift; MODELS=(); while [[ $# -gt 0 && "$1" != --* ]]; do MODELS+=("$1"); shift; done ;;
        --domains) shift; DOMAINS=(); while [[ $# -gt 0 && "$1" != --* ]]; do DOMAINS+=("$1"); shift; done ;;
        --max-workers) MAX_WORKERS="$2"; shift 2 ;;
        --time) TIME_LIMIT="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

config_for_model() {
    EXTRA=()
    case "$1" in
        12B)
            MODEL_ID="google/gemma-4-12B-it"
            TAG="gemma_4_12b_it"
            ;;
        31B-NVFP4)
            MODEL_ID="nvidia/Gemma-4-31B-IT-NVFP4"
            TAG="gemma_4_31b_it_nvfp4"
            # FlashInfer's optional FP4 autotuner can remain at 0/21 indefinitely
            # on this cluster.  The default NVFP4 kernel is functional and this
            # toggle does not alter sampling or model weights.
            EXTRA=(--quantization modelopt --disable-flashinfer-autotune 1)
            ;;
        *) echo "Unknown Gemma model: $1" >&2; exit 2 ;;
    esac
}

for size in "${MODELS[@]}"; do
    config_for_model "$size"
    for domain in "${DOMAINS[@]}"; do
        SLUG="${TAG}_${domain}_w${MAX_WORKERS}"
        echo ">>> ${size} (${MODEL_ID}) x ${domain} slug=${SLUG}"
        "$SUBMIT" ${DRY_RUN} \
            --partition dgx-b200 --gpus 1 --tensor-parallel-size 1 \
            --cpus 28 --memory 224G --time "$TIME_LIMIT" \
            --model "$MODEL_ID" --domain "$domain" --parts all \
            --model-slug "$SLUG" --max-workers "$MAX_WORKERS" \
            --reasoning-parser gemma4 --max-model-len 131072 --max-num-seqs 64 \
            --max-tokens 65000 --temperature 1.0 --top-p 0.95 --top-k 64 \
            --limit-mm-per-prompt '{"image":0,"audio":0}' \
            --trust-remote-code 1 --gpu-memory-utilization 0.95 "${EXTRA[@]}"
    done
done
