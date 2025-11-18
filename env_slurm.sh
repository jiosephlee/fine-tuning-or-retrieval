#!/bin/bash
#SBATCH --job-name=flash_attention_install
#SBATCH --output=slurm-%j.out
#SBATCH --time=02:00:00
#SBATCH --partition=dgx-b200         
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96GB

module load anaconda3
module load cuda/12.8.1
module load gcc/13.3.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/.conda/envs/finetuning"
cd /vast/home/j/jojolee/flash-attention/hopper
python setup.py install
# "$HOME/.conda/envs/finetuning/bin/pip" install flash-attn --no-build-isolation
