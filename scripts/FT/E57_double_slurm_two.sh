#!/bin/sh

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=ai
#SBATCH --mem-per-gpu=160GB
#SBATCH --job-name=E57_double_explanations_two
#SBATCH --output=logs/E57_double_explanations_two.out
#SBATCH --error=logs/E57_double_explanations_two.err

echo "➤ START"

echo "➤ SETTING UP HOST CUDA"
module unload cuda
module load cuda/12.4

# Define the path to your SIF file
YOUR_SIF_FILE="/gpfs/fs001/cbica/home/leejose/joseph/pytorch-2.4.0-cuda12.4-cudnn9-devel.sif"

echo "➤ RUNNING SCRIPT INSIDE APPTAINER: ${YOUR_SIF_FILE}"

# Execute the python script INSIDE the container
# --nv: Mounts the host NVIDIA drivers
# --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES: Passes the GPU assignment from SLURM into the container
apptainer exec --cleanenv --nv \
    --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    ${YOUR_SIF_FILE} \
    python finetuning_knowledge_v8.py      \
    --model_id allenai/OLMo-2-1124-7B \
    --device_batch_size 1 \
    --override_domains DPO 1_58 GRPO BOFT OFT QLoRA \
    --effective_batch_size_for_cpt 32 \
    --separate_batches_with_pretraining 1 \
    --fill_batches_with_pretraining \
    --num_train_epochs 100 \
    --learning_rate 2e-5 \
    --with_explanations \
    --constant_lr \
    --explanation_every_round \
    --num_paraphrased_texts 9 \
    --overlap_sections \
    --overlap_ratio 1_4 \
    --gradient_checkpointing \
    --full_finetuning

echo "➤ DONE"
