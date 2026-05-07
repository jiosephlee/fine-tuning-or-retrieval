#!/bin/bash
#SBATCH --job-name=E1_source_training
#SBATCH --output=logs/E1_source_training-%j.out
#SBATCH --error=logs/E1_source_training-%j.err
#SBATCH --time=10:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB

set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs

MODEL_ID="${MODEL_ID:-allenai/OLMo-2-0425-1B}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-2}"
EFFECTIVE_BATCH_SIZE="${EFFECTIVE_BATCH_SIZE:-8}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-3072}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
CUSTOM_SUFFIX="${CUSTOM_SUFFIX:-E1_source_all_domains}"

python -s finetuning_knowledge_v9.py \
    --custom_suffix "$CUSTOM_SUFFIX" \
    --model_id "$MODEL_ID" \
    --knowledge_probes_version v13 \
    --disable_mcqa_probes \
    --num_train_epochs "$NUM_EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    --num_paraphrased_texts 0 \
    --device_batch_size "$DEVICE_BATCH_SIZE" \
    --effective_batch_size_for_cpt "$EFFECTIVE_BATCH_SIZE" \
    --context_length_for_cpt "$CONTEXT_LENGTH" \
    --fill_batches_with_pretraining \
    --attn_implementation "$ATTN_IMPLEMENTATION" \
    --gradient_checkpointing \
    --full_finetuning \
    --enable_parameter_delta_tracking \
    --parcc
