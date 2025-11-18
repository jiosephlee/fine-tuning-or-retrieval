#!/bin/bash
#SBATCH --job-name=run
#SBATCH --output=slurm-%j.out
#SBATCH --time=12:00:00
#SBATCH --partition=dgx-b200          # example GPU partition on Betty
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB

num_epochs=200
num_paraphrased=9

python -s finetuning_knowledge_v8.py \
    --model_id allenai/OLMo-2-1124-7B \
    --device_batch_size 32 \
    --override_domains DPO 1_58 GRPO BOFT OFT QLoRA \
    --effective_batch_size_for_cpt 32 \
    --separate_batches_with_pretraining 1 \
    --fill_batches_with_pretraining \
    --num_train_epochs $num_epochs \
    --learning_rate 4e-5 \
    --constant_lr \
    --num_paraphrased_texts $num_paraphrased \
    --overlap_sections \
    --overlap_ratio 1_4 \
    --attn_implementation sdpa \
    --gradient_checkpointing \
    --full_finetuning

# 20 hours