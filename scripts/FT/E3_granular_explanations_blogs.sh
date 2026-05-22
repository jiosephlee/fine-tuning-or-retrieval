#!/bin/bash

set -euo pipefail

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
CUSTOM_SUFFIX="${CUSTOM_SUFFIX:-E3_granular_explanations_blogs_match_local}"
PUSH_TO_HUB_CPT_ID="${PUSH_TO_HUB_CPT_ID:-e3-olmo2-7b-para9-expl-blogs-match-20260522}"
# Schedule shape matches the full textbooks+blogs+stackexchange explanation cadence,
# but only the chosen subfolder is actually inserted into the matched track.
DOCUMENT_MATCH_SIZE_TYPES="${DOCUMENT_MATCH_SIZE_TYPES:-textbooks blogs stackexchange}"
DOCUMENT_MATCH_INSERT_CONTENT="${DOCUMENT_MATCH_INSERT_CONTENT:-blogs}"
EXPLANATIONS_INSERTION_STRATEGY="${EXPLANATIONS_INSERTION_STRATEGY:-granular}"
EXPLANATIONS_CYCLE="${EXPLANATIONS_CYCLE:-full}"
EXPLANATIONS_NUM_TRACKS="${EXPLANATIONS_NUM_TRACKS:-1}"
EXPLANATION_GRANULARITY="${EXPLANATION_GRANULARITY:-file}"
EXPLANATION_TRACK_SIZE_BY_CHUNK="${EXPLANATION_TRACK_SIZE_BY_CHUNK:-4}"
MCQA_PROBES_VERSION="${MCQA_PROBES_VERSION:-v14}"
MCQA_PROMPT_COLUMN="${MCQA_PROMPT_COLUMN:-formatted_question_5shot}"
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
if [[ "$SPARSE_CALLBACKS" == "1" ]]; then
    EXTRA_ARGS+=(--no_callback_every_step)
fi

read -r -a DOCUMENT_MATCH_SIZE_TYPE_ARGS <<< "$DOCUMENT_MATCH_SIZE_TYPES"
EXPLANATION_SCHEDULE_ARGS=(
    --document_track_baseline
    --document_match_specific_explanation "${DOCUMENT_MATCH_SIZE_TYPE_ARGS[@]}"
    --document_match_insert_content "$DOCUMENT_MATCH_INSERT_CONTENT"
    --explanations_insertion_strategy "$EXPLANATIONS_INSERTION_STRATEGY"
    --granular_explanations_num_tracks "$EXPLANATIONS_NUM_TRACKS"
    --granular_explanations_cycle "$EXPLANATIONS_CYCLE"
    --explanation_granularity "$EXPLANATION_GRANULARITY"
    --explanation_track_size_by_chunk "$EXPLANATION_TRACK_SIZE_BY_CHUNK"
)

if [[ "${CONDA_DEFAULT_ENV:-}" == "$CONDA_ENV" ]]; then
    LAUNCH=(torchrun --standalone --nproc_per_node "$NPROC_PER_NODE")
else
    LAUNCH=(conda run --no-capture-output -n "$CONDA_ENV" torchrun --standalone --nproc_per_node "$NPROC_PER_NODE")
fi

"${LAUNCH[@]}" finetuning_knowledge_v9.py \
    --custom_suffix "$CUSTOM_SUFFIX" \
    --model_id "$MODEL_ID" \
    --wandb_group finetuning_official \
    --push_to_hub_cpt_id "$PUSH_TO_HUB_CPT_ID" \
    --knowledge_probes_version v13 \
    --paraphrased_knowledge_probes \
    --paraphrased_knowledge_probes_version v13 \
    --paraphrased_knowledge_probe_filename_suffix _paraphrased \
    --mcqa_probes \
    --mcqa_probes_version "$MCQA_PROBES_VERSION" \
    --mcqa_prompt_column "$MCQA_PROMPT_COLUMN" \
    --mcqa_probe_batch_size 32 \
    --num_train_epochs "$NUM_EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    --lr_scheduler_min_lr_ratio 0.1 \
    --num_paraphrased_texts "$NUM_PARAPHRASED" \
    "${EXPLANATION_SCHEDULE_ARGS[@]}" \
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
