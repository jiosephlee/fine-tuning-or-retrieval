#!/bin/bash
#SBATCH --job-name=E42_granular_scale0_5
#SBATCH --output=logs/E42_granular_scale0_5-%j.out
#SBATCH --error=logs/E42_granular_scale0_5-%j.err
#SBATCH --partition=dgx-b200
#SBATCH --gpus=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=0-8:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CUSTOM_SUFFIX="${CUSTOM_SUFFIX:-E42_granular_explanations_7b_scale0_5}" \
EXPLANATION_TRACK_SCALE="${EXPLANATION_TRACK_SCALE:-0.5}" \
EXPLANATION_MATCH_SCALE="${EXPLANATION_MATCH_SCALE:-0.5}" \
KNOWLEDGE_PROBES_VERSION="${KNOWLEDGE_PROBES_VERSION:-v14}" \
PARAPHRASED_KNOWLEDGE_PROBES_VERSION="${PARAPHRASED_KNOWLEDGE_PROBES_VERSION:-v14}" \
MCQA_PROBES_VERSION="${MCQA_PROBES_VERSION:-v15}" \
INFERENCE_MCQA_PROBES_VERSION="${INFERENCE_MCQA_PROBES_VERSION:-v14}" \
bash "$SCRIPT_DIR/E3_granular_explanations.sh"
