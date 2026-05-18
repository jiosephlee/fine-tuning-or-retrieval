#!/bin/bash
#SBATCH --job-name=E1_source_training
#SBATCH --output=logs/E1_source_training-%j.out
#SBATCH --error=logs/E1_source_training-%j.err
#SBATCH --time=10:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=128GB

set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs

conda run --no-capture-output -n openrlhf torchrun --standalone --nproc_per_node 8 finetuning_knowledge_v9.py \
    --custom_suffix E1_source_all_domains_fa2_packing \
    --model_id allenai/OLMo-2-1124-7B \
    --knowledge_probes_version v13 \
    --mcqa_probes \
    --mcqa_probes_version v14 \
    --mcqa_prompt_column formatted_question_5shot \
    --num_train_epochs 10 \
    --learning_rate 1e-5 \
    --num_paraphrased_texts 0 \
    --chunk_by_section \
    --overlap_sections \
    --overlap_ratio 1_8 \
    --device_batch_size 8 \
    --effective_batch_size_for_cpt 256 \
    --context_length_for_cpt 4096 \
    --fill_batches_with_pretraining \
    --attn_implementation flash_attention_2 \
    --gradient_checkpointing \
    --compile \
    --full_finetuning \
    --enable_parameter_delta_tracking \
    --no-save_local_model
