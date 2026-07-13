#!/bin/bash
# Resume GLM-5-NVFP4 across all three domains from manifest-valid outputs.
# Job 7072802 used 4 GPUs for 4,679 seconds (5.199 GPU-hours). The
# continuation remains strictly below the 20 GPU-hour resume allowance:
# 5.199 + (4 GPUs * 03:41:00) = 19.932 GPU-hours.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT="$SCRIPT_DIR/submit_vllm_multiview.sh"

DRY_RUN=""
TIME_LIMIT="03:41:00"
MAX_WORKERS=16
PARTS="all"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN="--dry-run"; shift ;;
        --time) TIME_LIMIT="$2"; shift 2 ;;
        --max-workers) MAX_WORKERS="$2"; shift 2 ;;
        --parts) PARTS="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

if [[ "$TIME_LIMIT" != "03:41:00" ]]; then
    echo "Refusing --time $TIME_LIMIT: this continuation is capped at 03:41:00 on 4 GPUs (19.932 resume GPU-hours cumulative)." >&2
    exit 2
fi

"$SUBMIT" ${DRY_RUN} \
    --partition dgx-b200 --gpus 4 --tensor-parallel-size 4 \
    --cpus 112 --memory 896G --time "$TIME_LIMIT" \
    --model nvidia/GLM-5-NVFP4 --domain all --parts "$PARTS" \
    --model-slug glm_5_nvfp4 --max-workers "$MAX_WORKERS" \
    --max-model-len auto --gpu-memory-utilization 0.9 \
    --reasoning-parser glm45 --tool-call-parser glm47 \
    --enable-auto-tool-choice 1 --enable-chunked-prefill 1 \
    --enable-expert-parallel 1 --trust-remote-code 1 \
    --max-num-batched-tokens 8192 --max-num-seqs 1024 \
    --disable-flashinfer-autotune 1 \
    --ready-timeout-seconds 1500 --smoke-test-mode quick
