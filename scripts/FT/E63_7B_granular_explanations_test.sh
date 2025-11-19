#!/bin/bash
# Test version of granular explanation analysis
# Uses smaller model and fewer epochs for quick validation

set -e  # Exit on any error

# Source environment
source ../../env.sh

# Common settings (TEST VERSION)
MODEL="allenai/OLMo-2-0425-1B"  # Smaller model for testing
EPOCHS=1  # Just 1 epoch
LR=1e-5
BATCH_SIZE=64
DEVICE_BS=2  # Larger device batch since smaller model
CONTEXT_LENGTH=3072
DOMAINS="DPO"

# Training settings
FULL_FT="--full_finetuning"
CONSTANT_LR="--constant_lr"
FILL_PRETRAINING="--fill_batches_with_pretraining"
GRANULAR="--granular_explanation_analysis"
TEST="--test_script"

# Base command function
run_experiment() {
    local explanation_type=$1
    local num_docs=$2
    local custom_suffix=$3
    
    echo "=========================================="
    echo "TEST: ${explanation_type} with ${num_docs} docs"
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
        ${TEST} \
        --custom_suffix "${custom_suffix}"
}

# ============================================
# Test with just one configuration each
# ============================================

# Test Stack Exchange with 8 posts
run_experiment "stackexchange" 8 "stack_8_TEST"

# Test Blogs with 3 posts
run_experiment "blogs" 3 "blogs_3_TEST"

echo "=========================================="
echo "Test experiments completed successfully!"
echo "=========================================="

