#!/bin/bash
#SBATCH --job-name=run
#SBATCH --output=%j.out
#SBATCH --time=2:00:00
#SBATCH --partition=dgx-b200          # example GPU partition on Betty
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB

num_epochs=100
num_paraphrased=9

python -s finetuning_knowledge_v8.py \
    --model_id allenai/OLMo-2-1124-7B \
    --device_batch_size 64 \
    --override_domains DPO 1_58 GRPO BOFT OFT QLoRA \
    --effective_batch_size_for_cpt 64 \
    --fill_batches_with_pretraining \
    --with_specific_explanation blogs \
    --explanations_cycle full \
    --granular_explanation_analysis \
    --num_train_epochs $num_epochs \
    --learning_rate 2e-5 \
    --num_paraphrased_texts $num_paraphrased \
    --overlap_sections \
    --overlap_ratio 1_4 \
    --attn_implementation sdpa \
    --gradient_checkpointing \
    --full_finetuning \
    --parcc \