#!/bin/bash
#SBATCH --job-name=E10_source_32b
#SBATCH --output=logs/E10_source_32b-%j.out
#SBATCH --error=logs/E10_source_32b-%j.err
#SBATCH --partition=dgx-b200
#SBATCH --gpus=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=1-0:00:00

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "$SLURM_SUBMIT_DIR/scripts/FT" ]]; then
    cd "$SLURM_SUBMIT_DIR/scripts/FT"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/finetuning_knowledge_v9.py" ]]; then
    cd "$SLURM_SUBMIT_DIR"
else
    cd "$(dirname "$0")"
fi
mkdir -p logs

CONDA_ENV="${CONDA_ENV:-openrlhf}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
CUSTOM_SUFFIX="${CUSTOM_SUFFIX:-E10_source_32b_all_domains_fa2_packing_chunked_nll}"
PUSH_TO_HUB_CPT_ID="${PUSH_TO_HUB_CPT_ID:-e10-olmo2-32b-source-only-chunked-nll-20260517}"
DOCUMENT_TRACK_BASELINE="${DOCUMENT_TRACK_BASELINE:-1}"
DOCUMENT_MATCH_EXPLANATION_TYPES="${DOCUMENT_MATCH_EXPLANATION_TYPES:-textbooks stackexchange blogs}"
DOCUMENT_MATCH_EXPLANATIONS_CYCLE="${DOCUMENT_MATCH_EXPLANATIONS_CYCLE:-full}"
PROBE_EVERY_N_STEPS="${PROBE_EVERY_N_STEPS:-2}"
MCQA_PROBE_EVERY_N_STEPS="${MCQA_PROBE_EVERY_N_STEPS:-4}"
MCQA_PROMPT_COLUMN="${MCQA_PROMPT_COLUMN:-formatted_question_5shot}"
MCQA_PROBE_BATCH_SIZE="${MCQA_PROBE_BATCH_SIZE:-32}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-1}"
EFFECTIVE_BATCH_SIZE="${EFFECTIVE_BATCH_SIZE:-256}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-4096}"
LOSS_TYPE="${LOSS_TYPE:-chunked_nll}"
OFFLOAD_TO_CPU="${OFFLOAD_TO_CPU:-0}"
DEVICE_MAP_AUTO="${DEVICE_MAP_AUTO:-$OFFLOAD_TO_CPU}"
ACTIVATION_OFFLOADING="${ACTIVATION_OFFLOADING:-1}"
PARAMETER_DELTA_TRACKING="${PARAMETER_DELTA_TRACKING:-1}"
PARAMETER_DELTA_EVERY_N_STEPS="${PARAMETER_DELTA_EVERY_N_STEPS:-5}"
INFERENCE_MCQA_PROBES="${INFERENCE_MCQA_PROBES:-1}"
INFERENCE_MCQA_PROBES_VERSION="${INFERENCE_MCQA_PROBES_VERSION:-v12_reviewed v12}"
INFERENCE_MCQA_PROMPT_COLUMN="${INFERENCE_MCQA_PROMPT_COLUMN:-formatted_question}"

DOC_MATCH_ARGS=()
if [[ "$DOCUMENT_TRACK_BASELINE" == "1" ]]; then
    read -r -a DOCUMENT_MATCH_EXPLANATION_TYPE_ARGS <<< "$DOCUMENT_MATCH_EXPLANATION_TYPES"
    DOC_MATCH_ARGS+=(
        --document_track_baseline
        --document_match_specific_explanation "${DOCUMENT_MATCH_EXPLANATION_TYPE_ARGS[@]}"
        --explanations_insertion_strategy granular
        --granular_explanations_cycle "$DOCUMENT_MATCH_EXPLANATIONS_CYCLE"
    )
fi
INFERENCE_MCQA_ARGS=()
if [[ "$INFERENCE_MCQA_PROBES" == "1" ]]; then
    read -r -a INFERENCE_MCQA_PROBE_VERSION_ARGS <<< "$INFERENCE_MCQA_PROBES_VERSION"
    INFERENCE_MCQA_ARGS+=(
        --inference_mcqa_probes
        --inference_mcqa_probes_version "${INFERENCE_MCQA_PROBE_VERSION_ARGS[@]}"
        --inference_mcqa_prompt_column "$INFERENCE_MCQA_PROMPT_COLUMN"
    )
fi
OFFLOAD_ARGS=()
if [[ "$DEVICE_MAP_AUTO" == "1" ]]; then
    OFFLOAD_ARGS+=(--offload_to_cpu)
elif [[ "$ACTIVATION_OFFLOADING" == "1" ]]; then
    OFFLOAD_ARGS+=(--activation_offloading)
fi
PARAMETER_DELTA_ARGS=()
if [[ "$PARAMETER_DELTA_TRACKING" == "1" ]]; then
    PARAMETER_DELTA_ARGS+=(
        --enable_parameter_delta_tracking
        --parameter_delta_every_n_steps "$PARAMETER_DELTA_EVERY_N_STEPS"
    )
fi

FT_TMPDIR="${FT_TMPDIR:-/tmp/${USER:-ft}/ft_${SLURM_JOB_ID:-$$}}"
mkdir -p "$FT_TMPDIR"
export TMPDIR="$FT_TMPDIR"
export TMP="$FT_TMPDIR"
export TEMP="$FT_TMPDIR"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$FT_TMPDIR/torch_extensions}"
mkdir -p "$TORCH_EXTENSIONS_DIR"
trap 'rm -rf "$FT_TMPDIR"' EXIT
MASTER_PORT="${MASTER_PORT:-$((29000 + (${SLURM_JOB_ID:-0} % 1000)))}"

if command -v module >/dev/null 2>&1; then
    module load anaconda3 || true
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

if [[ "$DEVICE_MAP_AUTO" == "1" ]]; then
    LAUNCH=(python -s finetuning_knowledge_v9.py)
else
    LAUNCH=(torchrun --standalone --nproc_per_node "$NPROC_PER_NODE" finetuning_knowledge_v9.py)
fi

"${LAUNCH[@]}" \
    --custom_suffix "$CUSTOM_SUFFIX" \
    --wandb_group finetuning_official \
    --model_id allenai/OLMo-2-0325-32B \
    --knowledge_probes_version v13 \
    --paraphrased_knowledge_probes \
    --paraphrased_knowledge_probes_version v13 \
    --paraphrased_knowledge_probe_filename_suffix _paraphrased \
    --mcqa_probes \
    --mcqa_probes_version v14 \
    --mcqa_prompt_column "$MCQA_PROMPT_COLUMN" \
    --mcqa_probe_batch_size "$MCQA_PROBE_BATCH_SIZE" \
    "${INFERENCE_MCQA_ARGS[@]}" \
    --num_train_epochs 100 \
    --learning_rate 4e-5 \
    --lr_scheduler_min_lr_ratio 0.1 \
    --num_paraphrased_texts 0 \
    --overlap_sections \
    --overlap_ratio 1_16 \
    "${DOC_MATCH_ARGS[@]}" \
    --device_batch_size "$DEVICE_BATCH_SIZE" \
    --sft_loss_type "$LOSS_TYPE" \
    --effective_batch_size_for_cpt "$EFFECTIVE_BATCH_SIZE" \
    --context_length_for_cpt "$CONTEXT_LENGTH" \
    --fill_batches_with_pretraining \
    --attn_implementation flash_attention_2 \
    "${OFFLOAD_ARGS[@]}" \
    --gradient_checkpointing \
    --full_finetuning \
    --probe_every_n_steps "$PROBE_EVERY_N_STEPS" \
    --mcqa_probe_every_n_steps "$MCQA_PROBE_EVERY_N_STEPS" \
    "${PARAMETER_DELTA_ARGS[@]}" \
    --push_to_hub_cpt_id "$PUSH_TO_HUB_CPT_ID" \
    --no-save_local_model
