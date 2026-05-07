#!/bin/bash
#SBATCH --job-name=E63_cycle
#SBATCH --output=slurm-%j.out
#SBATCH --time=12:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB

# Granular Stack Exchange Analysis with Cycling
# Tests different numbers of explanation files that cycle through ALL document batches

num_epochs=100
num_paraphrased=19
lr=2e-5
batch_size=64
device_bs=32
context_length=3072

# Run with 3 stack exchange posts cycling through all 20 document types
python -s finetuning_knowledge_v8.py \
    --model_id allenai/OLMo-2-1124-7B \
    --num_train_epochs $num_epochs \
    --learning_rate $lr \
    --effective_batch_size_for_cpt $batch_size \
    --device_batch_size $device_bs \
    --context_length_for_cpt $context_length \
    --num_paraphrased_texts $num_paraphrased \
    --override_domains DPO \
    --with_specific_explanation stackexchange \
    --explanations_cycle 3 \
    --full_finetuning \
    --fill_batches_with_pretraining \
    --granular_explanation_analysis \
    --attn_implementation sdpa \
    --gradient_checkpointing \
    --compile_model \
    --custom_suffix "stack_cycle3"

# Run with 6 stack exchange posts cycling
python -s finetuning_knowledge_v8.py \
    --model_id allenai/OLMo-2-1124-7B \
    --num_train_epochs $num_epochs \
    --learning_rate $lr \
    --effective_batch_size_for_cpt $batch_size \
    --device_batch_size $device_bs \
    --context_length_for_cpt $context_length \
    --num_paraphrased_texts $num_paraphrased \
    --override_domains DPO \
    --with_specific_explanation stackexchange \
    --explanations_cycle 6 \
    --full_finetuning \
    --fill_batches_with_pretraining \
    --granular_explanation_analysis \
    --attn_implementation sdpa \
    --gradient_checkpointing \
    --compile_model \
    --custom_suffix "stack_cycle6"

# Run with 9 stack exchange posts cycling
python -s finetuning_knowledge_v8.py \
    --model_id allenai/OLMo-2-1124-7B \
    --num_train_epochs $num_epochs \
    --learning_rate $lr \
    --effective_batch_size_for_cpt $batch_size \
    --device_batch_size $device_bs \
    --context_length_for_cpt $context_length \
    --num_paraphrased_texts $num_paraphrased \
    --override_domains DPO \
    --with_specific_explanation stackexchange \
    --explanations_cycle 9 \
    --full_finetuning \
    --fill_batches_with_pretraining \
    --granular_explanation_analysis \
    --attn_implementation sdpa \
    --gradient_checkpointing \
    --compile_model \
    --custom_suffix "stack_cycle9"

# Run with 12 stack exchange posts cycling
python -s finetuning_knowledge_v8.py \
    --model_id allenai/OLMo-2-1124-7B \
    --num_train_epochs $num_epochs \
    --learning_rate $lr \
    --effective_batch_size_for_cpt $batch_size \
    --device_batch_size $device_bs \
    --context_length_for_cpt $context_length \
    --num_paraphrased_texts $num_paraphrased \
    --override_domains DPO \
    --with_specific_explanation stackexchange \
    --explanations_cycle 12 \
    --full_finetuning \
    --fill_batches_with_pretraining \
    --granular_explanation_analysis \
    --attn_implementation sdpa \
    --gradient_checkpointing \
    --compile_model \
    --custom_suffix "stack_cycle12"

