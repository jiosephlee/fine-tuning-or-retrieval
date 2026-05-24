#!/bin/bash

set -euo pipefail

# Local E2 paraphrase run without the matched explanation/document auxiliary track.

cd "$(dirname "$0")"
mkdir -p logs

MODEL_ID="${MODEL_ID:-allenai/OLMo-2-1124-7B}"
CONDA_ENV="${CONDA_ENV:-openrlhf}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
# In this CPT pipeline, NUM_EPOCHS is used as the target number of
# knowledge-injection batches when >1. The default 100 matches E1/E3 local.
NUM_EPOCHS="${NUM_EPOCHS:-50}"
NUM_PARAPHRASED="${NUM_PARAPHRASED:-49}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-8}"
EFFECTIVE_BATCH_SIZE="${EFFECTIVE_BATCH_SIZE:-256}"
LEARNING_RATE="${LEARNING_RATE:-4e-5}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-4096}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
CUSTOM_SUFFIX="${CUSTOM_SUFFIX:-E2_paraphrase_all_domains_local_no_explanation_match}"
MCQA_PROBES_VERSION="${MCQA_PROBES_VERSION:-v15}"
MCQA_PROMPT_COLUMN="${MCQA_PROMPT_COLUMN:-formatted_question_5shot}"
MCQA_PROBE_BATCH_SIZE="${MCQA_PROBE_BATCH_SIZE:-8}"
INFERENCE_MCQA_PROBES="${INFERENCE_MCQA_PROBES:-1}"
INFERENCE_MCQA_PROBES_VERSION="${INFERENCE_MCQA_PROBES_VERSION:-v14}"
INFERENCE_MCQA_PROMPT_COLUMN="${INFERENCE_MCQA_PROMPT_COLUMN:-formatted_question_5shot}"
USE_PARCC="${USE_PARCC:-0}"
SAVE_LOCAL_MODEL="${SAVE_LOCAL_MODEL:-1}"
SPARSE_CALLBACKS="${SPARSE_CALLBACKS:-0}"
PARAMETER_DELTA_EVERY_N_STEPS="${PARAMETER_DELTA_EVERY_N_STEPS:-10}"
PROBE_EVERY_N_STEPS="${PROBE_EVERY_N_STEPS:-2}"
MCQA_PROBE_EVERY_N_STEPS="${MCQA_PROBE_EVERY_N_STEPS:-4}"

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
else
    EXTRA_ARGS+=(--disable_inference_mcqa_probes)
fi

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
TORCHRUN_LOG_DIR="${TORCHRUN_LOG_DIR:-logs/E20_para49_no_explanation_match_${RUN_ID}}"
mkdir -p "$TORCHRUN_LOG_DIR"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

if [[ "${CONDA_DEFAULT_ENV:-}" == "$CONDA_ENV" ]]; then
    LAUNCH=(torchrun --standalone --nproc_per_node "$NPROC_PER_NODE" --log-dir "$TORCHRUN_LOG_DIR" --tee 3)
else
    LAUNCH=(conda run --no-capture-output -n "$CONDA_ENV" torchrun --standalone --nproc_per_node "$NPROC_PER_NODE" --log-dir "$TORCHRUN_LOG_DIR" --tee 3)
fi

echo "torchrun logs: $TORCHRUN_LOG_DIR"

"${LAUNCH[@]}" finetuning_knowledge_v9.py \
    --custom_suffix "$CUSTOM_SUFFIX" \
    --model_id "$MODEL_ID" \
    --wandb_group finetuning_official \
    --knowledge_probes_version v14 \
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
