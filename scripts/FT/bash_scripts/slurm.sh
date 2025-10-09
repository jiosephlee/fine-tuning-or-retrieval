#!/bin/sh

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=ai-short
#SBATCH --mem-per-gpu=160GB
#SBATCH --job-name=fine_tuning_study_v8
#SBATCH --output=logs/output.out
#SBATCH --error=logs/error.err

echo "➤ START"
source /cbica/software/external/python/anaconda/3/etc/profile.d/conda.sh
echo "➤ ACTIVATING CONDA"
conda activate trl
echo "➤ SET UP CUDA"
module unload cuda
module load cuda/12.4

echo "➤ RUN SCRIPT"
num_epochs=1
num_paraphrased=0
python finetuning_knowledge_v8.py \
    --model_id Qwen/Qwen2.5-7B\
    --device_batch_size 1 \
    --override_domains DPO 1_58 GRPO BOFT OFT QLoRA \
    --effective_batch_size_for_cpt 32 \
    --separate_batches_with_pretraining 1 \
    --fill_batches_with_pretraining \
    --num_train_epochs $num_epochs \
    --learning_rate 2e-5 \
    --num_paraphrased_texts $num_paraphrased \
    --overlap_sections \
    --overlap_ratio 1_4 \
    --lima_afterwards \
    --context_length_for_lima 2560 \
    --full_finetuning > output_1.log 
echo "➤ DONE"