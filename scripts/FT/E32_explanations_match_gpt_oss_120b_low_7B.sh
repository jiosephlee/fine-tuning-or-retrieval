#!/bin/bash
#SBATCH --job-name=E32_expl_match_gpt_oss_120b_low
#SBATCH --output=logs/E32_expl_match_gpt_oss_120b_low-%j.out
#SBATCH --error=logs/E32_expl_match_gpt_oss_120b_low-%j.err
#SBATCH --partition=dgx-b200
#SBATCH --gpus=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --mem=896G
#SBATCH --time=0-8:00:00

# Use E30's matched-explanation training configuration with the validated
# GPT-OSS-120B low-reasoning recovery corpus. The recovery suffix is intentional:
# it is the audited 108/108 corpus; the older canonical slug is not used.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ID="${RUN_ID:-${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"

export RUN_ID
export CUSTOM_SUFFIX="${CUSTOM_SUFFIX:-E32_explanations_match_gpt_oss_120b_low_all_domains}"
export PUSH_TO_HUB_CPT_ID="${PUSH_TO_HUB_CPT_ID:-e32-olmo2-7b-para9-expl-match-gpt-oss-120b-low-20260712}"
export DOCUMENT_MATCH_INSERT_EXPLANATION_MODEL="${DOCUMENT_MATCH_INSERT_EXPLANATION_MODEL:-gpt_oss_120b_low_recovery}"
export TORCHRUN_LOG_DIR="${TORCHRUN_LOG_DIR:-logs/E32_expl_match_gpt_oss_120b_low_${RUN_ID}}"

exec bash "$SCRIPT_DIR/E30_explanations_match_gpt_5_4_mini_low_7B.sh"
