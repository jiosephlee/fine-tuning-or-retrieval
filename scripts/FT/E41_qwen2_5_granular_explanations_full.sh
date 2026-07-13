#!/bin/bash
#SBATCH --job-name=E41_qwen_granular_full
#SBATCH --output=logs/E41_qwen_granular_full-%j.out
#SBATCH --error=logs/E41_qwen_granular_full-%j.err
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
CUSTOM_SUFFIX="${CUSTOM_SUFFIX:-E41_qwen2_5_7b_granular_explanations_full}" \
EXPLANATION_TRACK_SCALE="${EXPLANATION_TRACK_SCALE:-1.0}" \
EXPLANATION_MATCH_SCALE="${EXPLANATION_MATCH_SCALE:-1.0}" \
KNOWLEDGE_PROBES_VERSION="${KNOWLEDGE_PROBES_VERSION:-v14}" \
PARAPHRASED_KNOWLEDGE_PROBES_VERSION="${PARAPHRASED_KNOWLEDGE_PROBES_VERSION:-v14}" \
MCQA_PROBES_VERSION="${MCQA_PROBES_VERSION:-v15}" \
INFERENCE_MCQA_PROBES_VERSION="${INFERENCE_MCQA_PROBES_VERSION:-v14}" \
bash "$SCRIPT_DIR/E3_granular_explanations.sh"
