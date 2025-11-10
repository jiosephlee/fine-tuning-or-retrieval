#!/bin/sh

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=1:00:00
#SBATCH --gpus-per-node=a100:2
#SBATCH --partition=ai
#SBATCH --mem-per-gpu=160GB
#SBATCH --job-name=E47_32B_explanations_if_it_fails
#SBATCH --output=logs/E47_32B_explanations_if_it_fails.out
#SBATCH --error=logs/E47_32B_explanations_if_it_fails.err

echo "➤ START"

echo "➤ SETTING UP HOST CUDA"
module unload cuda
module load cuda/12.4

# Define the path to your SIF file
YOUR_SIF_FILE="/gpfs/fs001/cbica/home/leejose/joseph/pytorch-2.4.0-cuda12.4-cudnn9-devel.sif"
CUDA_VISIBLE_DEVICES=$(get_CUDA_VISIBLE_DEVICES)
echo "➤ RUNNING SCRIPT INSIDE APPTAINER: ${YOUR_SIF_FILE}"

# Execute the python script INSIDE the container
# --nv: Mounts the host NVIDIA drivers
# --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES: Passes the GPU assignment from SLURM into the container
apptainer exec --cleanenv --nv \
    --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    ${YOUR_SIF_FILE} \
    /cbica/home/leejose/.local/bin/accelerate launch --config_file deepspeed_4gpus.yaml finetuning_knowledge_v8.py \
    --model_id allenai/OLMo-2-0325-32B \
    --device_batch_size 1 \
    --with_explanations \
    --override_domains DPO 1_58 GRPO BOFT OFT QLoRA \
    --effective_batch_size_for_cpt 8 \
    --separate_batches_with_pretraining 1 \
    --fill_batches_with_pretraining \
    --num_train_epochs 100 \
    --learning_rate 2e-5 \
    --num_paraphrased_texts 9 \
    --overlap_sections \
    --overlap_ratio 1_4 \
    --gradient_checkpointing \
    --full_finetuning

echo "➤ DONE"
