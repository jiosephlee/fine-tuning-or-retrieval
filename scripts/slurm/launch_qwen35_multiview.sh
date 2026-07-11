#!/bin/bash
# Launch the Qwen3.5 multiview sweep: {4B, 9B, 27B} x {arxiv, medical, legal}.
#
# Per-size right-sizing (matches the ParcC partition CPU/GPU ratios enforced Aug 3, 2026):
#   4B  (~8GB weights)  -> b200-mig45 : 6 CPU / 48G  (fits the 45GB slice)
#   9B  (~18GB weights) -> b200-mig45 : 6 CPU / 48G  (fits the 45GB slice)
#   27B (~52GB weights) -> dgx-b200   : 28 CPU / 224G (needs >45GB; full B200 = plentiful nodes)
#
# reasoning-effort mirrors the gpt-oss runs (low). Slugs mirror the gpt-oss naming:
#   qwen3_5_<size>_<domain>_<part>_w<workers>
#
# Usage:
#   ./launch_qwen35_multiview.sh                 # submit all 9
#   ./launch_qwen35_multiview.sh --dry-run       # print the sbatch commands only
#   ./launch_qwen35_multiview.sh --models 4B 27B --domains arxiv   # subset

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT="$SCRIPT_DIR/submit_vllm_multiview.sh"

DRY_RUN=""
MODELS=(4B 9B 27B)
DOMAINS=(arxiv medical legal)
MAX_WORKERS=16
REASONING_EFFORT="low"
TIME_LIMIT="08:00:00"
ENABLE_THINKING=""   # "1"/"0"; empty leaves the pipeline default (thinking on)
TEMPERATURE=""       # e.g. 0.7 for sampling; empty leaves the default (greedy, temp 0)
TOP_P=""             # e.g. 0.8; empty leaves the default
REPETITION_PENALTY=""  # e.g. 1.1 to break LaTeX-structure loops; empty leaves it off

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN="--dry-run"; shift ;;
        --models) shift; MODELS=(); while [[ $# -gt 0 && "$1" != --* ]]; do MODELS+=("$1"); shift; done ;;
        --domains) shift; DOMAINS=(); while [[ $# -gt 0 && "$1" != --* ]]; do DOMAINS+=("$1"); shift; done ;;
        --max-workers) MAX_WORKERS="$2"; shift 2 ;;
        --reasoning-effort) REASONING_EFFORT="$2"; shift 2 ;;
        --enable-thinking) ENABLE_THINKING="$2"; shift 2 ;;
        --temperature) TEMPERATURE="$2"; shift 2 ;;
        --top-p) TOP_P="$2"; shift 2 ;;
        --repetition-penalty) REPETITION_PENALTY="$2"; shift 2 ;;
        --time) TIME_LIMIT="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

# All sizes run on full dgx-b200 (180GB) GPUs: the 64k context window (--max-model-len
# below, to fit the largest ~22k-token paper plus a full 32k-token generation) needs far
# more KV cache than a 45GB MIG slice holds, and it re-triggers the 9B Mamba-block limit.
# max_num_seqs is capped for the 9B Mamba-hybrid regardless (we only issue max_workers
# concurrent requests). "" leaves the vLLM default.
MAX_MODEL_LEN=65536
config_for_size() {
    MAX_NUM_SEQS=""
    CPUS=28; MEM="224G"; PART="dgx-b200"
    case "$1" in
        4B)  MODEL_ID="Qwen/Qwen3.5-4B";  TAG="4b" ;;
        9B)  MODEL_ID="Qwen/Qwen3.5-9B";  TAG="9b"; MAX_NUM_SEQS=256 ;;
        27B) MODEL_ID="Qwen/Qwen3.5-27B"; TAG="27b" ;;
        *) echo "Unknown model size: $1" >&2; exit 2 ;;
    esac
}

for size in "${MODELS[@]}"; do
    config_for_size "$size"
    for domain in "${DOMAINS[@]}"; do
        SLUG="qwen3_5_${TAG}_${domain}_w${MAX_WORKERS}"
        echo ">>> ${size} (${MODEL_ID}) x ${domain}  [${PART}, ${CPUS}cpu/${MEM}]  slug=${SLUG}"
        MNS_ARG=()
        [[ -n "$MAX_NUM_SEQS" ]] && MNS_ARG=(--max-num-seqs "$MAX_NUM_SEQS")
        THINK_ARG=()
        [[ -n "$ENABLE_THINKING" ]] && THINK_ARG=(--enable-thinking "$ENABLE_THINKING")
        SAMPLE_ARG=()
        [[ -n "$TEMPERATURE" ]] && SAMPLE_ARG+=(--temperature "$TEMPERATURE")
        [[ -n "$TOP_P" ]] && SAMPLE_ARG+=(--top-p "$TOP_P")
        [[ -n "$REPETITION_PENALTY" ]] && SAMPLE_ARG+=(--repetition-penalty "$REPETITION_PENALTY")
        "$SUBMIT" ${DRY_RUN} \
            --partition "$PART" --gpus 1 --cpus "$CPUS" --memory "$MEM" --time "$TIME_LIMIT" \
            --model "$MODEL_ID" --domain "$domain" --parts all \
            --model-slug "$SLUG" --reasoning-effort "$REASONING_EFFORT" --max-workers "$MAX_WORKERS" \
            --reasoning-parser qwen3 --max-model-len "$MAX_MODEL_LEN" "${MNS_ARG[@]}" "${THINK_ARG[@]}" "${SAMPLE_ARG[@]}"
    done
done
