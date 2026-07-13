#!/bin/bash

set -euo pipefail

# Explanation run using legacy_v2 paraphrase-tail insertion from modern
# explanation subfolders.

cd "$(dirname "$0")"
mkdir -p logs

MODEL_ID="${MODEL_ID:-allenai/OLMo-2-1124-7B}"
CONDA_ENV="${CONDA_ENV:-openrlhf}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NUM_EPOCHS="${NUM_EPOCHS:-100}"
NUM_PARAPHRASED="${NUM_PARAPHRASED:-9}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-8}"
EFFECTIVE_BATCH_SIZE="${EFFECTIVE_BATCH_SIZE:-256}"
LEARNING_RATE="${LEARNING_RATE:-4e-5}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-4096}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
CUSTOM_SUFFIX="${CUSTOM_SUFFIX:-E25_explanations_legacy_v2_all_domains_local}"
EXPLANATION_TYPES="${EXPLANATION_TYPES:-textbooks stackexchange blogs}"
TIMES_EXPLANATIONS="${TIMES_EXPLANATIONS:-1}"
PUSH_TO_HUB_CPT_ID="${PUSH_TO_HUB_CPT_ID:-}"
KNOWLEDGE_PROBES_VERSION="${KNOWLEDGE_PROBES_VERSION:-v14}"
KNOWLEDGE_PROBE_VARIANT="${KNOWLEDGE_PROBE_VARIANT:-short_targets}"
PARAPHRASED_KNOWLEDGE_PROBES_VERSION="${PARAPHRASED_KNOWLEDGE_PROBES_VERSION:-v14}"
MCQA_PROBES_VERSION="${MCQA_PROBES_VERSION:-v15}"
MCQA_PROMPT_COLUMN="${MCQA_PROMPT_COLUMN:-formatted_question_5shot}"
MCQA_PROBE_BATCH_SIZE="${MCQA_PROBE_BATCH_SIZE:-32}"
INFERENCE_PROBES_VERSION="${INFERENCE_PROBES_VERSION:-v11}"
INFERENCE_MCQA_PROBES="${INFERENCE_MCQA_PROBES:-1}"
INFERENCE_MCQA_PROBES_VERSION="${INFERENCE_MCQA_PROBES_VERSION:-v14}"
INFERENCE_MCQA_PROMPT_COLUMN="${INFERENCE_MCQA_PROMPT_COLUMN:-formatted_question_5shot}"
USE_PARCC="${USE_PARCC:-0}"
SAVE_LOCAL_MODEL="${SAVE_LOCAL_MODEL:-1}"
SPARSE_CALLBACKS="${SPARSE_CALLBACKS:-0}"
PARAMETER_DELTA_EVERY_N_STEPS="${PARAMETER_DELTA_EVERY_N_STEPS:-5}"
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
if [[ -n "$PUSH_TO_HUB_CPT_ID" ]]; then
    EXTRA_ARGS+=(--push_to_hub_cpt_id "$PUSH_TO_HUB_CPT_ID")
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

read -r -a EXPLANATION_TYPE_ARGS <<< "$EXPLANATION_TYPES"

if [[ "${CONDA_DEFAULT_ENV:-}" == "$CONDA_ENV" ]]; then
    LAUNCH=(torchrun --standalone --nproc_per_node "$NPROC_PER_NODE")
else
    LAUNCH=(conda run --no-capture-output -n "$CONDA_ENV" torchrun --standalone --nproc_per_node "$NPROC_PER_NODE")
fi

"${LAUNCH[@]}" finetuning_knowledge_v9.py \
    --custom_suffix "$CUSTOM_SUFFIX" \
    --wandb_group finetuning_official \
    --wandb_panel_sources arxiv legal medical \
    --model_id "$MODEL_ID" \
    --include_sources arxiv legal medical \
    --knowledge_probes_version "$KNOWLEDGE_PROBES_VERSION" \
    --knowledge_probe_variant "$KNOWLEDGE_PROBE_VARIANT" \
    --paraphrased_knowledge_probes \
    --paraphrased_knowledge_probes_version "$PARAPHRASED_KNOWLEDGE_PROBES_VERSION" \
    --paraphrased_knowledge_probe_filename_suffix _paraphrased \
    --mcqa_probes \
    --mcqa_probes_version "$MCQA_PROBES_VERSION" \
    --mcqa_prompt_column "$MCQA_PROMPT_COLUMN" \
    --mcqa_probe_batch_size "$MCQA_PROBE_BATCH_SIZE" \
    --inference_probes_version "$INFERENCE_PROBES_VERSION" \
    --num_train_epochs "$NUM_EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    --lr_scheduler_min_lr_ratio 0.1 \
    --num_paraphrased_texts "$NUM_PARAPHRASED" \
    --with_specific_explanation "${EXPLANATION_TYPE_ARGS[@]}" \
    --times_explanations "$TIMES_EXPLANATIONS" \
    --explanations_insertion_strategy legacy_v2 \
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
