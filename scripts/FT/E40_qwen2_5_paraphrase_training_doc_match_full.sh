#!/bin/bash
#SBATCH --job-name=E40_qwen_para_docmatch_full
#SBATCH --output=logs/E40_qwen_para_docmatch_full-%j.out
#SBATCH --error=logs/E40_qwen_para_docmatch_full-%j.err
#SBATCH --partition=dgx-b200
#SBATCH --gpus=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=0-8:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CUSTOM_SUFFIX="${CUSTOM_SUFFIX:-E40_qwen2_5_7b_paraphrase_docmatch_expl_full}" \
NUM_PARAPHRASED="${NUM_PARAPHRASED:-9}" \
bash "$SCRIPT_DIR/E39_qwen2_5_source_training_doc_match_full.sh"
