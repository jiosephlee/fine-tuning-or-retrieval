#!/bin/bash
# Train with gpt_5_mini_low-generated explanations inserted into a matched auxiliary track
# shaped against the ordinary gpt_5_mini_custom textbooks/stackexchange/blogs
# granular explanation schedule. The gpt_5_mini_custom corpus only defines the
# track's per-step chunk counts (token budget); the content trained on comes
# from data/{source}/explanations/gpt_5_mini_low/{domain}/ (flat textbook/blogs/stackexchange
# files, cycled to fill the schedule) — same insert pattern as E13-15 (prior
# knowledge) and E19 (cited works).
#
# Local run: set NPROC_PER_NODE to the number of GPUs on this machine.

set -euo pipefail

# --- Locate the directory containing finetuning_knowledge_v9.py ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/finetuning_knowledge_v9.py" ]]; then
    cd "$SCRIPT_DIR"
elif [[ -f "$SCRIPT_DIR/scripts/FT/finetuning_knowledge_v9.py" ]]; then
    cd "$SCRIPT_DIR/scripts/FT"
else
    cd "$SCRIPT_DIR"
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
CUSTOM_SUFFIX="${CUSTOM_SUFFIX:-E27_explanations_match_gpt_5_mini_low_all_domains}"
PUSH_TO_HUB_CPT_ID="${PUSH_TO_HUB_CPT_ID:-e27-olmo2-7b-para9-expl-match-gpt-5-mini-low-20260710}"
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
DOCUMENT_MATCH_INSERT_EXPLANATION_MODEL="${DOCUMENT_MATCH_INSERT_EXPLANATION_MODEL:-gpt_5_mini_low}"

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

# --- Activate conda env ---
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate "$CONDA_ENV"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
TORCHRUN_LOG_DIR="${TORCHRUN_LOG_DIR:-logs/E27_expl_match_gpt_5_mini_low_${RUN_ID}}"
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
