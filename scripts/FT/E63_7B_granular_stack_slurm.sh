#!/bin/bash
#SBATCH --job-name=E63_stack
#SBATCH --output=slurm-%j.out
#SBATCH --time=12:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB

module load anaconda3
module load cuda/12.8.1
module load gcc/13.3.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/.conda/envs/finetuning"

num_epochs=100
num_paraphrased=19
lr=2e-5

# Run 8 stack exchange posts across last 8 paraphrases
python -s finetuning_knowledge_v8.py \
    --model_id allenai/OLMo-2-1124-7B \
    --num_train_epochs $num_epochs \
    --num_paraphrased_texts $num_paraphrased \
    --learning_rate $lr \
    --effective_batch_size_for_cpt 64 \
    --device_batch_size 64 \
    --override_domains DPO \
    --with_specific_explanation stackexchange \
    --explanation_tail_docs 6 \
    --full_finetuning \
    --fill_batches_with_pretraining \
    --granular_explanation_analysis \
    --attn_implementation sdpa \
    --gradient_checkpointing \

# Run 9 stack exchange posts across last 9 paraphrases
python -s finetuning_knowledge_v8.py \
    --model_id allenai/OLMo-2-1124-7B \
    --num_train_epochs $num_epochs \
    --learning_rate $lr \
    --effective_batch_size_for_cpt 64 \
    --device_batch_size 64 \
    --num_paraphrased_texts $num_paraphrased \
    --override_domains DPO \
    --with_specific_explanation stackexchange \
    --explanation_tail_docs 12 \
    --full_finetuning \
    --fill_batches_with_pretraining \
    --granular_explanation_analysis \
    --attn_implementation sdpa \
    --gradient_checkpointing \

# Run 15 stack exchange posts (will use first 9 due to slot limit)
python -s finetuning_knowledge_v8.py \
    --model_id allenai/OLMo-2-1124-7B \
    --num_train_epochs $num_epochs \
    --learning_rate $lr \
    --effective_batch_size_for_cpt 64 \
    --device_batch_size 64 \
    --num_paraphrased_texts $num_paraphrased \
    --override_domains DPO \
    --with_specific_explanation stackexchange \
    --explanation_tail_docs 18 \
    --full_finetuning \
    --fill_batches_with_pretraining \
    --granular_explanation_analysis \
    --attn_implementation sdpa \
    --gradient_checkpointing \

