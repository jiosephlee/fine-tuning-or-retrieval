#!/bin/bash
#SBATCH --job-name=run
#SBATCH --output=bs-%j.out
#SBATCH --time=14:00:00
#SBATCH --partition=dgx-b200          # example GPU partition on Betty
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB

module load anaconda3
module load cuda/12.8.1
module load gcc/13.3.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/.conda/envs/finetuning"

num_epochs=100
num_paraphrased=0

python -s finetuning_knowledge_v8.py \
    --device_batch_size 32 \
    --override_domains DPO 1_58 GRPO BOFT OFT QLoRA \
    --effective_batch_size_for_cpt 32 \
    --fill_batches_with_pretraining \
    --num_train_epochs $num_epochs \
    --learning_rate 2e-5 \
    --num_paraphrased_texts $num_paraphrased \
    --overlap_sections \
    --overlap_ratio 1_4 \
    --attn_implementation sdpa \
    --gradient_checkpointing \
    --full_finetuning \
    --parcc \

python -s finetuning_knowledge_v8.py \
    --device_batch_size 128 \
    --override_domains DPO 1_58 GRPO BOFT OFT QLoRA \
    --effective_batch_size_for_cpt 128 \
    --fill_batches_with_pretraining \
    --num_train_epochs $num_epochs \
    --learning_rate 2e-5 \
    --num_paraphrased_texts $num_paraphrased \
    --overlap_sections \
    --overlap_ratio 1_4 \
    --attn_implementation sdpa \
    --gradient_checkpointing \
    --full_finetuning \
    --parcc \

python -s finetuning_knowledge_v8.py \
    --device_batch_size 128 \
    --override_domains DPO 1_58 GRPO BOFT OFT QLoRA \
    --effective_batch_size_for_cpt 256 \
    --fill_batches_with_pretraining \
    --num_train_epochs $num_epochs \
    --learning_rate 2e-5 \
    --num_paraphrased_texts $num_paraphrased \
    --overlap_sections \
    --overlap_ratio 1_4 \
    --attn_implementation sdpa \
    --gradient_checkpointing \
    --full_finetuning \
    --parcc \

num_epochs=100
num_paraphrased=9

python -s finetuning_knowledge_v8.py \
    --device_batch_size 32 \
    --override_domains DPO 1_58 GRPO BOFT OFT QLoRA \
    --effective_batch_size_for_cpt 32 \
    --fill_batches_with_pretraining \
    --num_train_epochs $num_epochs \
    --learning_rate 2e-5 \
    --num_paraphrased_texts $num_paraphrased \
    --overlap_sections \
    --overlap_ratio 1_4 \
    --attn_implementation sdpa \
    --gradient_checkpointing \
    --full_finetuning \
    --parcc \

python -s finetuning_knowledge_v8.py \
    --device_batch_size 128 \
    --override_domains DPO 1_58 GRPO BOFT OFT QLoRA \
    --effective_batch_size_for_cpt 128 \
    --fill_batches_with_pretraining \
    --num_train_epochs $num_epochs \
    --learning_rate 2e-5 \
    --num_paraphrased_texts $num_paraphrased \
    --overlap_sections \
    --overlap_ratio 1_4 \
    --attn_implementation sdpa \
    --gradient_checkpointing \
    --full_finetuning \
    --parcc \

python -s finetuning_knowledge_v8.py \
    --device_batch_size 128 \
    --override_domains DPO 1_58 GRPO BOFT OFT QLoRA \
    --effective_batch_size_for_cpt 256 \
    --fill_batches_with_pretraining \
    --num_train_epochs $num_epochs \
    --learning_rate 2e-5 \
    --num_paraphrased_texts $num_paraphrased \
    --overlap_sections \
    --overlap_ratio 1_4 \
    --attn_implementation sdpa \
    --gradient_checkpointing \
    --full_finetuning \
    --parcc \