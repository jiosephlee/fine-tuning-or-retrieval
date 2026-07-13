#!/bin/bash
#SBATCH --job-name=E41_para_docmatch_scale0_5
#SBATCH --output=logs/E41_para_docmatch_scale0_5-%j.out
#SBATCH --error=logs/E41_para_docmatch_scale0_5-%j.err
#SBATCH --partition=dgx-b200
#SBATCH --gpus=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=0-8:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CUSTOM_SUFFIX="${CUSTOM_SUFFIX:-E41_paraphrase_7b_docmatch_expl_scale0_5}" \
NUM_PARAPHRASED="${NUM_PARAPHRASED:-9}" \
EXPLANATION_TRACK_SCALE="${EXPLANATION_TRACK_SCALE:-0.5}" \
EXPLANATION_MATCH_SCALE="${EXPLANATION_MATCH_SCALE:-0.5}" \
bash "$SCRIPT_DIR/E40_source_training_doc_match_scale0_5.sh"
