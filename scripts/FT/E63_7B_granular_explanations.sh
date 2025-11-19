#!/bin/bash
# Granular Explanation Analysis: Stack Exchange and Blogs
# Tests different numbers of explanation documents distributed across tail batches

set -e  # Exit on any error

# Source environment
source ../../env.sh

# Common settings
MODEL="allenai/OLMo-2-1124-7B"
EPOCHS=200
LR=1e-5
BATCH_SIZE=64
DEVICE_BS=1
CONTEXT_LENGTH=3072
DOMAINS="DPO"

# Training settings
FULL_FT="--full_finetuning"
CONSTANT_LR="--constant_lr"
FILL_PRETRAINING="--fill_batches_with_pretraining"
GRANULAR="--granular_explanation_analysis"

# Base command function
run_experiment() {
    local explanation_type=$1
    local num_docs=$2
    local custom_suffix=$3
    
    echo "=========================================="
    echo "Running: ${explanation_type} with ${num_docs} docs across last ${num_docs} paraphrases"
    echo "=========================================="
    
    python finetuning_knowledge_v8.py \
        --model_id ${MODEL} \
        --num_train_epochs ${EPOCHS} \
        --learning_rate ${LR} \
        --effective_batch_size_for_cpt ${BATCH_SIZE} \
        --device_batch_size ${DEVICE_BS} \
        --context_length_for_cpt ${CONTEXT_LENGTH} \
        --num_paraphrased_texts 9 \
        --override_domains ${DOMAINS} \
        --with_specific_explanation ${explanation_type} \
        --explanation_tail_docs ${num_docs} \
        ${FULL_FT} \
        ${CONSTANT_LR} \
        ${FILL_PRETRAINING} \
        ${GRANULAR} \
        --custom_suffix "${custom_suffix}"
}

# ============================================
# Stack Exchange Experiments
# ============================================

# 8 Stack Exchange posts distributed across last 8 paraphrases
run_experiment "stackexchange" 8 "stack_8"

# 9 Stack Exchange posts distributed across last 9 paraphrases
run_experiment "stackexchange" 9 "stack_9"

# 15 Stack Exchange posts distributed across last 9 paraphrases (should load first 15 files)
# Note: This will load 15 files but we only have 9 slots, so first 9 will be used
run_experiment "stackexchange" 15 "stack_15"

# ============================================
# Blogs Experiments
# ============================================

# 3 Blogs distributed across last 3 paraphrases
run_experiment "blogs" 3 "blogs_3"

# 6 Blogs distributed across last 6 paraphrases
run_experiment "blogs" 6 "blogs_6"

# 9 Blogs distributed across last 9 paraphrases
run_experiment "blogs" 9 "blogs_9"

echo "=========================================="
echo "All experiments completed successfully!"
echo "=========================================="

