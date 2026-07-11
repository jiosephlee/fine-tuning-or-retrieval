#!/bin/bash
#SBATCH --job-name=E24_para_legacy_v2
#SBATCH --output=logs/E24_para_legacy_v2-%j.out
#SBATCH --error=logs/E24_para_legacy_v2-%j.err
#SBATCH --partition=dgx-b200
#SBATCH --gpus=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=1024G
#SBATCH --time=0-8:00:00

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "$SLURM_SUBMIT_DIR/scripts/FT" ]]; then
    cd "$SLURM_SUBMIT_DIR/scripts/FT"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/finetuning_knowledge_v9.py" ]]; then
    cd "$SLURM_SUBMIT_DIR"
else
    cd "$(dirname "$0")"
fi
mkdir -p logs

export CONDA_ENV="${CONDA_ENV:-openrlhf}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-32}"
export SAVE_LOCAL_MODEL="${SAVE_LOCAL_MODEL:-0}"
export CUSTOM_SUFFIX="${CUSTOM_SUFFIX:-E24_paraphrase_legacy_v2_all_domains_slurm_no_explanation_match}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

if command -v module >/dev/null 2>&1; then
    module load anaconda3 || true
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

echo "Running E24 on ${SLURM_JOB_GPUS:-unknown GPUs}: NPROC_PER_NODE=$NPROC_PER_NODE DEVICE_BATCH_SIZE=$DEVICE_BATCH_SIZE"
bash E24_paraphrase_training_legacy_v2_local_no_explanation_match.sh
