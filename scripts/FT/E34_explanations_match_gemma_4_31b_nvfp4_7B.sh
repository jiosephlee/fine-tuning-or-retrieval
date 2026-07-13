#!/bin/bash
#SBATCH --job-name=E34_expl_match_gemma4_31b
#SBATCH --output=logs/E34_expl_match_gemma4_31b-%j.out
#SBATCH --error=logs/E34_expl_match_gemma4_31b-%j.err
#SBATCH --partition=dgx-b200
#SBATCH --gpus=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --mem=896G
#SBATCH --time=0-8:00:00

# Use E30's matched-explanation training configuration with the audited
# NVIDIA Gemma 4 31B IT NVFP4 corpus (108/108 assembled views passed binary
# review). The {source} placeholder selects the source-specific W16 slug.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUN_ID="${RUN_ID:-${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"

export RUN_ID
export CUSTOM_SUFFIX="${CUSTOM_SUFFIX:-E34_explanations_match_gemma_4_31b_nvfp4_all_domains}"
export PUSH_TO_HUB_CPT_ID="${PUSH_TO_HUB_CPT_ID:-e34-olmo2-7b-para9-expl-match-gemma-4-31b-nvfp4-20260712}"
if [[ -z "${DOCUMENT_MATCH_INSERT_EXPLANATION_MODEL:-}" ]]; then
    DOCUMENT_MATCH_INSERT_EXPLANATION_MODEL='gemma_4_31b_it_nvfp4_{source}_w16'
fi
export DOCUMENT_MATCH_INSERT_EXPLANATION_MODEL
export TORCHRUN_LOG_DIR="${TORCHRUN_LOG_DIR:-logs/E34_expl_match_gemma4_31b_${RUN_ID}}"

for source in arxiv legal medical; do
    resolved_slug="${DOCUMENT_MATCH_INSERT_EXPLANATION_MODEL//\{source\}/$source}"
    explanation_root="$PROJECT_ROOT/data/$source/explanations/$resolved_slug"
    if [[ ! -d "$explanation_root" ]]; then
        echo "Missing E34 explanation corpus: $explanation_root" >&2
        exit 1
    fi
done

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
    echo "E34 preflight passed: $DOCUMENT_MATCH_INSERT_EXPLANATION_MODEL"
    exit 0
fi

exec bash "$SCRIPT_DIR/E30_explanations_match_gpt_5_4_mini_low_7B.sh"
