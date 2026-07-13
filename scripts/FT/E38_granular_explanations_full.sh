#!/bin/bash
#SBATCH --job-name=E38_granular_full
#SBATCH --output=logs/E38_granular_full-%j.out
#SBATCH --error=logs/E38_granular_full-%j.err
#SBATCH --partition=dgx-b200
#SBATCH --gpus=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=0-8:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CUSTOM_SUFFIX="${CUSTOM_SUFFIX:-E38_granular_explanations_7b_full}" \
EXPLANATION_TRACK_SCALE="${EXPLANATION_TRACK_SCALE:-1.0}" \
EXPLANATION_MATCH_SCALE="${EXPLANATION_MATCH_SCALE:-1.0}" \
KNOWLEDGE_PROBES_VERSION="${KNOWLEDGE_PROBES_VERSION:-v14}" \
PARAPHRASED_KNOWLEDGE_PROBES_VERSION="${PARAPHRASED_KNOWLEDGE_PROBES_VERSION:-v14}" \
MCQA_PROBES_VERSION="${MCQA_PROBES_VERSION:-v15}" \
INFERENCE_MCQA_PROBES_VERSION="${INFERENCE_MCQA_PROBES_VERSION:-v14}" \
bash "$SCRIPT_DIR/E3_granular_explanations.sh"
