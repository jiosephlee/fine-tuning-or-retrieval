#!/bin/bash
#SBATCH --job-name=flash_attention_install
#SBATCH --output=slurm-%j.out
#SBATCH --time=00:30:00
#SBATCH --partition=dgx-b200          # example GPU partition on Betty
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/envs/finetuning"
"$HOME/envs/finetuning/bin/pip" install flash-attn --no-build-isolation
nvidia-smi || true
