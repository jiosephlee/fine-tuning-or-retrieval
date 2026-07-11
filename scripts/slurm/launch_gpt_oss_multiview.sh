#!/bin/bash
# Launch the gpt-oss multiview sweep at MAXIMUM length with HIGH reasoning:
#   {20b, 120b} x {arxiv, medical, legal}.
#
# Differences vs the historical gpt_oss_*_low runs:
#   - Served context window at the model max: --max-model-len 131072 (128k).
#   - Per-generation output budget requested at the full window (--max-tokens 131072);
#     utils clamps it down to the room left by each prompt, so every call gets the
#     maximum completion that fits.
#   - Uniform HIGH reasoning effort.
#   - New slugs (gpt_oss_<size>_high) so the existing *_low outputs are preserved.
#
# Sizing (dgx-b200, ~180GB B200 GPUs; the 45/90GB MIG slices are too small for a
# 128k KV cache, especially for 120b):
#   20b  (~16GB weights) -> 1 GPU  (tp=1)
#   120b (~63GB weights) -> 2 GPUs (tp=2); bump to 4 if KV-cache limited at 128k
#
# Usage:
#   ./launch_gpt_oss_multiview.sh                       # submit all 6
#   ./launch_gpt_oss_multiview.sh --dry-run             # print sbatch commands only
#   ./launch_gpt_oss_multiview.sh --models 20b --domains arxiv
#   ./launch_gpt_oss_multiview.sh --max-workers 4 --time 24:00:00

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT="$SCRIPT_DIR/submit_vllm_multiview.sh"

DRY_RUN=""
MODELS=(20b 120b)
DOMAINS=(arxiv medical legal)
REASONING_EFFORT="high"
TIME_LIMIT="48:00:00"
MAX_MODEL_LEN=131072
MAX_TOKENS=131072
PARTITION="dgx-b200"
MAX_WORKERS_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN="--dry-run"; shift ;;
        --models) shift; MODELS=(); while [[ $# -gt 0 && "$1" != --* ]]; do MODELS+=("$1"); shift; done ;;
        --domains) shift; DOMAINS=(); while [[ $# -gt 0 && "$1" != --* ]]; do DOMAINS+=("$1"); shift; done ;;
        --max-workers) MAX_WORKERS_OVERRIDE="$2"; shift 2 ;;
        --reasoning-effort) REASONING_EFFORT="$2"; shift 2 ;;
        --time) TIME_LIMIT="$2"; shift 2 ;;
        --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
        --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
        --partition) PARTITION="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

# Per-size right-sizing. Workers are kept modest because each concurrent request can
# hold a large 128k KV allocation; raise with --max-workers if the GPUs have headroom.
config_for_size() {
    CPUS=28; MEM="224G"
    case "$1" in
        20b)  MODEL_ID="openai/gpt-oss-20b";  TAG="20b";  GPUS=1; TP=1; WORKERS=8 ;;
        120b) MODEL_ID="openai/gpt-oss-120b"; TAG="120b"; GPUS=2; TP=2; WORKERS=4 ;;
        *) echo "Unknown model size: $1 (expected 20b or 120b)" >&2; exit 2 ;;
    esac
    if [[ -n "$MAX_WORKERS_OVERRIDE" ]]; then
        WORKERS="$MAX_WORKERS_OVERRIDE"
    fi
}

for size in "${MODELS[@]}"; do
    config_for_size "$size"
    for domain in "${DOMAINS[@]}"; do
        SLUG="gpt_oss_${TAG}_high"
        echo ">>> ${size} (${MODEL_ID}) x ${domain}  [${PARTITION}, ${GPUS}gpu/tp${TP}, ${CPUS}cpu/${MEM}]  slug=${SLUG}"
        "$SUBMIT" ${DRY_RUN} \
            --partition "$PARTITION" --gpus "$GPUS" --tensor-parallel-size "$TP" \
            --cpus "$CPUS" --memory "$MEM" --time "$TIME_LIMIT" \
            --model "$MODEL_ID" --domain "$domain" --parts all \
            --model-slug "$SLUG" --reasoning-effort "$REASONING_EFFORT" \
            --max-workers "$WORKERS" --max-model-len "$MAX_MODEL_LEN" --max-tokens "$MAX_TOKENS"
    done
done
