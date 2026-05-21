#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs

PROBE_EVERY_N_STEPS="${PROBE_EVERY_N_STEPS:-2}"
MCQA_PROBE_EVERY_N_STEPS="${MCQA_PROBE_EVERY_N_STEPS:-4}"
MCQA_PROMPT_COLUMN="${MCQA_PROMPT_COLUMN:-formatted_question_5shot}"
INFERENCE_MCQA_PROBES="${INFERENCE_MCQA_PROBES:-1}"
INFERENCE_MCQA_PROBES_VERSION="${INFERENCE_MCQA_PROBES_VERSION:-v12_reviewed v12}"
INFERENCE_MCQA_PROMPT_COLUMN="${INFERENCE_MCQA_PROMPT_COLUMN:-formatted_question_5shot}"

EXTRA_ARGS=()
if [[ "$INFERENCE_MCQA_PROBES" == "1" ]]; then
    read -r -a INFERENCE_MCQA_PROBE_VERSION_ARGS <<< "$INFERENCE_MCQA_PROBES_VERSION"
    EXTRA_ARGS+=(
        --inference_mcqa_probes
        --inference_mcqa_probes_version "${INFERENCE_MCQA_PROBE_VERSION_ARGS[@]}"
        --inference_mcqa_prompt_column "$INFERENCE_MCQA_PROMPT_COLUMN"
    )
fi

conda run --no-capture-output -n openrlhf torchrun --standalone --nproc_per_node 8 finetuning_knowledge_v9.py \
    --custom_suffix E17_source_arxiv_legal_local \
    --model_id allenai/OLMo-2-1124-7B \
    --include_sources arxiv legal \
    --wandb_panel_sources arxiv legal \
    --knowledge_probes_version v13 \
    --mcqa_probes \
    --mcqa_probes_version v14 \
    --mcqa_prompt_column "$MCQA_PROMPT_COLUMN" \
    --num_train_epochs 100 \
    --learning_rate 4e-5 \
    --num_paraphrased_texts 0 \
    --overlap_sections \
    --overlap_ratio 1_8 \
    --device_batch_size 8 \
    --effective_batch_size_for_cpt 256 \
    --context_length_for_cpt 4096 \
    --fill_batches_with_pretraining \
    --attn_implementation flash_attention_2 \
    --gradient_checkpointing \
    --full_finetuning \
    --probe_every_n_steps "$PROBE_EVERY_N_STEPS" \
    --mcqa_probe_every_n_steps "$MCQA_PROBE_EVERY_N_STEPS" \
    --enable_parameter_delta_tracking \
    --parameter_delta_every_n_steps 5 \
    --no-save_local_model \
    "${EXTRA_ARGS[@]}"
