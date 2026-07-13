#!/bin/bash
#SBATCH --job-name=E28_expl_match_gpt_5_mini_high
#SBATCH --output=logs/E28_expl_match_gpt_5_mini_high-%j.out
#SBATCH --error=logs/E28_expl_match_gpt_5_mini_high-%j.err
#SBATCH --partition=dgx-b200
#SBATCH --gpus=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --mem=896G
#SBATCH --time=0-8:00:00

# Train with gpt_5_mini_high-generated explanations inserted into a matched auxiliary track
# shaped against the ordinary gpt_5_mini_custom textbooks/stackexchange/blogs
# granular explanation schedule. The gpt_5_mini_custom corpus only defines the
# track's per-step chunk counts (token budget); the content trained on comes
# from data/{source}/explanations/gpt_5_mini_high/{domain}/ (flat textbook/blogs/stackexchange
# files, cycled to fill the schedule) — same insert pattern as E13-15 (prior
# knowledge) and E19 (cited works).

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "$SLURM_SUBMIT_DIR/scripts/FT" ]]; then
    cd "$SLURM_SUBMIT_DIR/scripts/FT"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/finetuning_knowledge_v9.py" ]]; then
    cd "$SLURM_SUBMIT_DIR"
else
    cd "$(dirname "$0")"
fi
mkdir -p logs

MODEL_ID="${MODEL_ID:-allenai/OLMo-2-1124-7B}"
CONDA_ENV="${CONDA_ENV:-openrlhf}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
NUM_EPOCHS="${NUM_EPOCHS:-100}"
NUM_PARAPHRASED="${NUM_PARAPHRASED:-9}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-32}"
EFFECTIVE_BATCH_SIZE="${EFFECTIVE_BATCH_SIZE:-256}"
LEARNING_RATE="${LEARNING_RATE:-4e-5}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-4096}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
CUSTOM_SUFFIX="${CUSTOM_SUFFIX:-E28_explanations_match_gpt_5_mini_high_all_domains}"
PUSH_TO_HUB_CPT_ID="${PUSH_TO_HUB_CPT_ID:-e28-olmo2-7b-para9-expl-match-gpt-5-mini-high-20260710}"
MCQA_PROBES_VERSION="${MCQA_PROBES_VERSION:-v15}"
MCQA_PROMPT_COLUMN="${MCQA_PROMPT_COLUMN:-formatted_question_5shot}"
MCQA_PROBE_BATCH_SIZE="${MCQA_PROBE_BATCH_SIZE:-32}"
INFERENCE_PROBES_VERSION="${INFERENCE_PROBES_VERSION:-v11}"
INFERENCE_MCQA_PROBES="${INFERENCE_MCQA_PROBES:-1}"
INFERENCE_MCQA_PROBES_VERSION="${INFERENCE_MCQA_PROBES_VERSION:-v14}"
INFERENCE_MCQA_PROMPT_COLUMN="${INFERENCE_MCQA_PROMPT_COLUMN:-formatted_question_5shot}"
USE_PARCC="${USE_PARCC:-1}"
SAVE_LOCAL_MODEL="${SAVE_LOCAL_MODEL:-0}"
SPARSE_CALLBACKS="${SPARSE_CALLBACKS:-0}"
PARAMETER_DELTA_EVERY_N_STEPS="${PARAMETER_DELTA_EVERY_N_STEPS:-5}"
PROBE_EVERY_N_STEPS="${PROBE_EVERY_N_STEPS:-2}"
MCQA_PROBE_EVERY_N_STEPS="${MCQA_PROBE_EVERY_N_STEPS:-4}"
DOCUMENT_TRACK_BASELINE="${DOCUMENT_TRACK_BASELINE:-1}"
DOCUMENT_MATCH_EXPLANATION_TYPES="${DOCUMENT_MATCH_EXPLANATION_TYPES:-textbooks stackexchange blogs}"
DOCUMENT_MATCH_EXPLANATIONS_CYCLE="${DOCUMENT_MATCH_EXPLANATIONS_CYCLE:-full}"
DOCUMENT_MATCH_INSERT_CONTENT="${DOCUMENT_MATCH_INSERT_CONTENT:-explanations}"
DOCUMENT_MATCH_INSERT_EXPLANATION_MODEL="${DOCUMENT_MATCH_INSERT_EXPLANATION_MODEL:-gpt_5_mini_high}"

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
    read -r -a INFERENCE_MCQA_PROBE_VERSION_ARGS <<< "$INFERENCE_MCQA_PROBES_VERSION"
    EXTRA_ARGS+=(
        --inference_mcqa_probes
        --inference_mcqa_probes_version "${INFERENCE_MCQA_PROBE_VERSION_ARGS[@]}"
        --inference_mcqa_prompt_column "$INFERENCE_MCQA_PROMPT_COLUMN"
    )
fi
if [[ "$DOCUMENT_TRACK_BASELINE" == "1" ]]; then
    read -r -a DOCUMENT_MATCH_EXPLANATION_TYPE_ARGS <<< "$DOCUMENT_MATCH_EXPLANATION_TYPES"
    EXTRA_ARGS+=(
        --document_track_baseline
        --document_match_specific_explanation "${DOCUMENT_MATCH_EXPLANATION_TYPE_ARGS[@]}"
        --document_match_insert_content "$DOCUMENT_MATCH_INSERT_CONTENT"
        --explanations_insertion_strategy granular
        --granular_explanations_cycle "$DOCUMENT_MATCH_EXPLANATIONS_CYCLE"
    )
    if [[ -n "$DOCUMENT_MATCH_INSERT_EXPLANATION_MODEL" ]]; then
        EXTRA_ARGS+=(--document_match_insert_explanation_model "$DOCUMENT_MATCH_INSERT_EXPLANATION_MODEL")
    fi
fi

if command -v module >/dev/null 2>&1; then
    module load anaconda3 || true
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

RUN_ID="${RUN_ID:-${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"
TORCHRUN_LOG_DIR="${TORCHRUN_LOG_DIR:-logs/E28_expl_match_gpt_5_mini_high_${RUN_ID}}"
mkdir -p "$TORCHRUN_LOG_DIR"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

echo "torchrun logs: $TORCHRUN_LOG_DIR"

torchrun --standalone --nproc_per_node "$NPROC_PER_NODE" --log-dir "$TORCHRUN_LOG_DIR" --tee 3 finetuning_knowledge_v9.py \
    --custom_suffix "$CUSTOM_SUFFIX" \
    --wandb_group finetuning_official \
    --wandb_panel_sources arxiv legal medical \
    --model_id "$MODEL_ID" \
    --push_to_hub_cpt_id "$PUSH_TO_HUB_CPT_ID" \
    --include_sources arxiv legal medical \
    --knowledge_probes_version v14 \
    --inference_probes_version "$INFERENCE_PROBES_VERSION" \
    --paraphrased_knowledge_probes \
    --paraphrased_knowledge_probes_version v14 \
    --paraphrased_knowledge_probe_filename_suffix _paraphrased \
    --mcqa_probes \
    --mcqa_probes_version "$MCQA_PROBES_VERSION" \
    --mcqa_prompt_column "$MCQA_PROMPT_COLUMN" \
    --mcqa_probe_batch_size "$MCQA_PROBE_BATCH_SIZE" \
    --num_train_epochs "$NUM_EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    --lr_scheduler_min_lr_ratio 0.1 \
    --num_paraphrased_texts "$NUM_PARAPHRASED" \
    --overlap_sections \
    --overlap_ratio 1_16 \
    --device_batch_size "$DEVICE_BATCH_SIZE" \
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
    "${EXTRA_ARGS[@]}"
