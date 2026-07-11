#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUNNER="$SCRIPT_DIR/run_vllm_multiview.sbatch"

PARTITION="b200-mig45"
GPUS=1
TIME_LIMIT="04:00:00"
CPUS=16
MEMORY="128G"
TENSOR_PARALLEL_SIZE=1
DRY_RUN=0
RUNNER_ARGS=()

usage() {
    echo "Usage: $0 [submission options] [runner options]"
    echo "Submission options:"
    echo "  --partition NAME       Default: b200-mig45"
    echo "  --gpus N               Default: 1"
    echo "  --time HH:MM:SS        Default: 04:00:00"
    echo "  --cpus N               Default: 16"
    echo "  --memory SIZE          Default: 128G"
    echo "  --dry-run              Print the sbatch command without submitting"
    echo "Runner options include --model, --domain, --item, --parts,"
    echo "--tensor-parallel-size, --max-model-len, --max-workers, and --model-slug."
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --partition) PARTITION="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --time) TIME_LIMIT="$2"; shift 2 ;;
        --cpus) CPUS="$2"; shift 2 ;;
        --memory) MEMORY="$2"; shift 2 ;;
        --tensor-parallel-size)
            TENSOR_PARALLEL_SIZE="$2"
            RUNNER_ARGS+=("$1" "$2")
            shift 2
            ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *)
            if [[ $# -ge 2 && "$2" != --* ]]; then
                RUNNER_ARGS+=("$1" "$2")
                shift 2
            else
                echo "Unknown or incomplete option: $1" >&2
                usage >&2
                exit 2
            fi
            ;;
    esac
done

for value in "$GPUS" "$CPUS" "$TENSOR_PARALLEL_SIZE"; do
    if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "Expected a positive integer, got: $value" >&2
        exit 2
    fi
done
if (( TENSOR_PARALLEL_SIZE > GPUS )); then
    echo "tensor parallel size cannot exceed requested GPUs" >&2
    exit 2
fi

mkdir -p "$PROJECT_ROOT/logs/vllm"
SBATCH_CMD=(
    sbatch
    --partition "$PARTITION"
    --gpus "$GPUS"
    --time "$TIME_LIMIT"
    --cpus-per-task "$CPUS"
    --mem "$MEMORY"
    --job-name vllm-multiview
    --output "$PROJECT_ROOT/logs/vllm/slurm-%j.out"
    --error "$PROJECT_ROOT/logs/vllm/slurm-%j.err"
    "$RUNNER"
    "${RUNNER_ARGS[@]}"
)

if (( DRY_RUN )); then
    printf '%q ' "${SBATCH_CMD[@]}"
    printf '\n'
else
    "${SBATCH_CMD[@]}"
fi
