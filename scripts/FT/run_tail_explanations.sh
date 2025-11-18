#!/bin/bash
#SBATCH --job-name=tail_expl
#SBATCH --output=slurm-%j.out
#SBATCH --time=12:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB

# Usage: sbatch run_tail_explanations.sh <stack|blog> <count>
# Example: sbatch run_tail_explanations.sh stack 8

TYPE=$1
COUNT=$2

if [ -z "$TYPE" ] || [ -z "$COUNT" ]; then
    echo "Usage: $0 <stack|blog> <count>"
    exit 1
fi

# Paths relative to scripts/FT/ where this script resides
BASE_DIR="../../data/arxiv/explanations/DPO"

if [ "$TYPE" == "stack" ]; then
    EXPLANATION_DIR="$BASE_DIR/stackexchange"
    # e.g. 8, 16, 24
elif [ "$TYPE" == "blog" ]; then
    EXPLANATION_DIR="$BASE_DIR/blogs"
    # e.g. 3, 6, 9
else
    echo "Unknown type: $TYPE. Use 'stack' or 'blog'."
    exit 1
fi

echo "Running with $COUNT $TYPE explanations from $EXPLANATION_DIR"
echo "Using explanation_tail_docs=$COUNT, max_explanation_files=$COUNT"

# Using standard settings but with new strategy args
python -s finetuning_knowledge_v8.py \
    --model_id allenai/OLMo-2-1124-7B \
    --device_batch_size 32 \
    --override_domains DPO 1_58 GRPO BOFT OFT QLoRA \
    --effective_batch_size_for_cpt 64 \
    --separate_batches_with_pretraining 0 \
    --fill_batches_with_pretraining \
    --num_train_epochs 200 \
    --learning_rate 4e-5 \
    --constant_lr \
    --num_paraphrased_texts 9 \
    --overlap_sections \
    --overlap_ratio 1_4 \
    --attn_implementation sdpa \
    --gradient_checkpointing \
    --full_finetuning \
    --with_explanations \
    --explanation_tail_docs $COUNT \
    --max_explanation_files $COUNT \
    --override_explanation_dir $EXPLANATION_DIR \
    --custom_suffix "tail_${TYPE}_${COUNT}"

