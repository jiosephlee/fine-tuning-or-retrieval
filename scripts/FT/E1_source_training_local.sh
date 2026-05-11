#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs

conda run --no-capture-output -n openrlhf torchrun --standalone --nproc_per_node 8 finetuning_knowledge_v9.py \
    --custom_suffix E1_source_all_domains_fa2_packing \
    --model_id allenai/OLMo-2-1124-7B \
    --knowledge_probes_version v13 \
    --disable_mcqa_probes \
    --num_train_epochs 100 \
    --learning_rate 1e-5 \
    --num_paraphrased_texts 0 \
    --chunk_by_section \
    --overlap_sections \
    --overlap_ratio 1_8 \
    --device_batch_size 16 \
    --effective_batch_size_for_cpt 128 \
    --context_length_for_cpt 4096 \
    --fill_batches_with_pretraining \
    --attn_implementation flash_attention_2 \
    --gradient_checkpointing \
    --full_finetuning \
    --enable_parameter_delta_tracking \
    --no-save_local_model
