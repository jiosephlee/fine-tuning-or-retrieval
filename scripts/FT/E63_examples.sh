#!/bin/bash
# E63 Examples: Demonstrating various explanations_cycle modes

source ../../env.sh

MODEL="allenai/OLMo-2-1124-7B"
EPOCHS=100
LR=2e-5
BATCH_SIZE=64
DEVICE_BS=32
CONTEXT_LENGTH=3072

# Example 1: Cycle through 3 stack exchange posts
python finetuning_knowledge_v8.py \
    --model_id ${MODEL} \
    --num_train_epochs ${EPOCHS} \
    --learning_rate ${LR} \
    --effective_batch_size_for_cpt ${BATCH_SIZE} \
    --device_batch_size ${DEVICE_BS} \
    --context_length_for_cpt ${CONTEXT_LENGTH} \
    --num_paraphrased_texts 19 \
    --override_domains DPO \
    --with_specific_explanation stackexchange \
    --explanations_cycle 3 \
    --full_finetuning \
    --fill_batches_with_pretraining \
    --granular_explanation_analysis \
    --attn_implementation sdpa \
    --gradient_checkpointing

# Example 2: Use ALL available textbook chapters (dynamic per domain)
python finetuning_knowledge_v8.py \
    --model_id ${MODEL} \
    --num_train_epochs ${EPOCHS} \
    --learning_rate ${LR} \
    --effective_batch_size_for_cpt ${BATCH_SIZE} \
    --device_batch_size ${DEVICE_BS} \
    --context_length_for_cpt ${CONTEXT_LENGTH} \
    --num_paraphrased_texts 19 \
    --override_domains DPO 1_58 BOFT \
    --with_specific_explanation textbooks \
    --explanations_cycle full \
    --full_finetuning \
    --fill_batches_with_pretraining \
    --granular_explanation_analysis \
    --attn_implementation sdpa \
    --gradient_checkpointing

# Example 3: Stack multiple cycles - blogs + stackexchange
# This loads files from both subfolders and cycles through all of them
python finetuning_knowledge_v8.py \
    --model_id ${MODEL} \
    --num_train_epochs ${EPOCHS} \
    --learning_rate ${LR} \
    --effective_batch_size_for_cpt ${BATCH_SIZE} \
    --device_batch_size ${DEVICE_BS} \
    --context_length_for_cpt ${CONTEXT_LENGTH} \
    --num_paraphrased_texts 19 \
    --override_domains DPO \
    --with_specific_explanation blogs stackexchange \
    --explanations_cycle 6 \
    --full_finetuning \
    --fill_batches_with_pretraining \
    --granular_explanation_analysis \
    --attn_implementation sdpa \
    --gradient_checkpointing

# Example 4: Stack multiple cycles - load ALL from both subfolders
python finetuning_knowledge_v8.py \
    --model_id ${MODEL} \
    --num_train_epochs ${EPOCHS} \
    --learning_rate ${LR} \
    --effective_batch_size_for_cpt ${BATCH_SIZE} \
    --device_batch_size ${DEVICE_BS} \
    --context_length_for_cpt ${CONTEXT_LENGTH} \
    --num_paraphrased_texts 19 \
    --override_domains DPO 1_58 \
    --with_specific_explanation blogs stackexchange textbooks \
    --explanations_cycle full \
    --full_finetuning \
    --fill_batches_with_pretraining \
    --granular_explanation_analysis \
    --attn_implementation sdpa \
    --gradient_checkpointing

