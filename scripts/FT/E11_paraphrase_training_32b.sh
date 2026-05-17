#!/bin/bash
#SBATCH --job-name=E11_paraphrase_32b
#SBATCH --output=logs/E11_paraphrase_32b-%j.out
#SBATCH --error=logs/E11_paraphrase_32b-%j.err
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

MODEL_ID="${MODEL_ID:-allenai/OLMo-2-0325-32B}"
CONDA_ENV="${CONDA_ENV:-openrlhf}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-deepspeed_zero2_4gpus.yaml}"
# In this CPT pipeline, NUM_EPOCHS is used as the target number of
# knowledge-injection batches when >1. The default 100 matches E1/E3 local.
NUM_EPOCHS="${NUM_EPOCHS:-100}"
NUM_PARAPHRASED="${NUM_PARAPHRASED:-9}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-1}"
EFFECTIVE_BATCH_SIZE="${EFFECTIVE_BATCH_SIZE:-256}"
LOSS_TYPE="${LOSS_TYPE:-chunked_nll}"
LEARNING_RATE="${LEARNING_RATE:-4e-5}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-4096}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
CUSTOM_SUFFIX="${CUSTOM_SUFFIX:-E11_paraphrase_32b_all_domains_chunked_nll}"
PUSH_TO_HUB_CPT_ID="${PUSH_TO_HUB_CPT_ID:-e11-olmo2-32b-para9-chunked-nll-20260517}"
MCQA_PROBES_VERSION="${MCQA_PROBES_VERSION:-v14}"
MCQA_PROMPT_COLUMN="${MCQA_PROMPT_COLUMN:-formatted_question_5shot}"
MCQA_PROBE_BATCH_SIZE="${MCQA_PROBE_BATCH_SIZE:-32}"
INFERENCE_MCQA_PROBES="${INFERENCE_MCQA_PROBES:-1}"
INFERENCE_MCQA_PROBES_VERSION="${INFERENCE_MCQA_PROBES_VERSION:-v12_reviewed}"
INFERENCE_MCQA_PROMPT_COLUMN="${INFERENCE_MCQA_PROMPT_COLUMN:-formatted_question}"
USE_PARCC="${USE_PARCC:-0}"
SAVE_LOCAL_MODEL="${SAVE_LOCAL_MODEL:-0}"
SPARSE_CALLBACKS="${SPARSE_CALLBACKS:-0}"
PARAMETER_DELTA_EVERY_N_STEPS="${PARAMETER_DELTA_EVERY_N_STEPS:-5}"
PROBE_EVERY_N_STEPS="${PROBE_EVERY_N_STEPS:-2}"
MCQA_PROBE_EVERY_N_STEPS="${MCQA_PROBE_EVERY_N_STEPS:-4}"
DOCUMENT_TRACK_BASELINE="${DOCUMENT_TRACK_BASELINE:-1}"
DOCUMENT_MATCH_EXPLANATION_TYPES="${DOCUMENT_MATCH_EXPLANATION_TYPES:-textbooks stackexchange blogs}"
DOCUMENT_MATCH_EXPLANATIONS_CYCLE="${DOCUMENT_MATCH_EXPLANATIONS_CYCLE:-full}"
EXTRA_ARGS=()
if [[ "$USE_PARCC" == "1" ]]; then
    EXTRA_ARGS+=(--parcc)
fi
if [[ "$SAVE_LOCAL_MODEL" == "1" ]]; then
    EXTRA_ARGS+=(--save_local_model)
else
    EXTRA_ARGS+=(--no-save_local_model)
fi
if [[ "$SPARSE_CALLBACKS" == "1" ]]; then
    EXTRA_ARGS+=(--no_callback_every_step)
fi
if [[ "$INFERENCE_MCQA_PROBES" == "1" ]]; then
    EXTRA_ARGS+=(
        --inference_mcqa_probes
        --inference_mcqa_probes_version "$INFERENCE_MCQA_PROBES_VERSION"
        --inference_mcqa_prompt_column "$INFERENCE_MCQA_PROMPT_COLUMN"
    )
fi
if [[ "$DOCUMENT_TRACK_BASELINE" == "1" ]]; then
    read -r -a DOCUMENT_MATCH_EXPLANATION_TYPE_ARGS <<< "$DOCUMENT_MATCH_EXPLANATION_TYPES"
    EXTRA_ARGS+=(
        --document_track_baseline
        --document_match_specific_explanation "${DOCUMENT_MATCH_EXPLANATION_TYPE_ARGS[@]}"
        --explanations_insertion_strategy granular
        --granular_explanations_cycle "$DOCUMENT_MATCH_EXPLANATIONS_CYCLE"
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

accelerate launch \
    --config_file "$DEEPSPEED_CONFIG" \
    --num_processes "$NPROC_PER_NODE" \
    --main_process_port "$MASTER_PORT" \
    finetuning_knowledge_v9.py \
    --custom_suffix "$CUSTOM_SUFFIX" \
    --wandb_group finetuning_official \
    --model_id "$MODEL_ID" \
    --knowledge_probes_version v13 \
    --paraphrased_knowledge_probes \
    --paraphrased_knowledge_probes_version v13 \
    --paraphrased_knowledge_probe_filename_suffix _paraphrased \
    --mcqa_probes \
    --mcqa_probes_version "$MCQA_PROBES_VERSION" \
    --mcqa_prompt_column "$MCQA_PROMPT_COLUMN" \
    --mcqa_probe_batch_size "$MCQA_PROBE_BATCH_SIZE" \
    --num_train_epochs "$NUM_EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    --lr_scheduler_min_lr_ratio 0.1 \
    --overlap_sections \
    --overlap_ratio 1_16 \
    --num_paraphrased_texts "$NUM_PARAPHRASED" \
    --device_batch_size "$DEVICE_BATCH_SIZE" \
    --sft_loss_type "$LOSS_TYPE" \
    --effective_batch_size_for_cpt "$EFFECTIVE_BATCH_SIZE" \
    --context_length_for_cpt "$CONTEXT_LENGTH" \
    --fill_batches_with_pretraining \
    --attn_implementation "$ATTN_IMPLEMENTATION" \
    --gradient_checkpointing \
    --full_finetuning \
    --probe_every_n_steps "$PROBE_EVERY_N_STEPS" \
    --mcqa_probe_every_n_steps "$MCQA_PROBE_EVERY_N_STEPS" \
    --enable_parameter_delta_tracking \
    --parameter_delta_every_n_steps "$PARAMETER_DELTA_EVERY_N_STEPS" \
    --push_to_hub_cpt_id "$PUSH_TO_HUB_CPT_ID" \
    "${EXTRA_ARGS[@]}"
