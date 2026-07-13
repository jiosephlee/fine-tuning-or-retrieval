#!/bin/bash
#SBATCH --job-name=E39_qwen_source_docmatch_full
#SBATCH --output=logs/E39_qwen_source_docmatch_full-%j.out
#SBATCH --error=logs/E39_qwen_source_docmatch_full-%j.err
#SBATCH --partition=dgx-b200
#SBATCH --gpus=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=0-8:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B}" \
PRETRAINING_DATA_PATH="${PRETRAINING_DATA_PATH:-../../data/olmo/dclm_qwen2.5_7b_100M_tokens.npy}" \
CUSTOM_SUFFIX="${CUSTOM_SUFFIX:-E39_qwen2_5_7b_source_docmatch_expl_full}" \
NUM_PARAPHRASED="${NUM_PARAPHRASED:-0}" \
EXPLANATION_TRACK_SCALE="${EXPLANATION_TRACK_SCALE:-1.0}" \
EXPLANATION_MATCH_SCALE="${EXPLANATION_MATCH_SCALE:-1.0}" \
bash "$SCRIPT_DIR/E36_source_training_doc_match_full.sh"
