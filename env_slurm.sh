#!/bin/bash
#SBATCH --job-name=flash_attention_install
#SBATCH --output=slurm-%j.out
#SBATCH --time=00:30:00
#SBATCH --partition=dgx-b200         
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB

module load anaconda3
module load cuda/12.8.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/.conda/envs/finetuning"
"$HOME/.conda/envs/finetuning/bin/pip" install flash-attn --no-build-isolation
nvidia-smi || true
