#!/bin/sh

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=07:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=ai
#SBATCH --mem-per-gpu=160GB
#SBATCH --job-name=fine_tuning_study_v5
#SBATCH --output=logs/output.out
#SBATCH --error=logs/error.err

echo "➤ START"
source /cbica/software/external/python/anaconda/3/etc/profile.d/conda.sh
echo "➤ ACTIVATING CONDA"
conda activate finetune_env
echo "➤ SET UP CUDA"
module unload cuda
module load cuda/12.4

experiment_name="SingleArxivPaper_1B_Test_Run"

echo "➤ RUN SCRIPT"
python finetuning_knowledge_v5.py --experiment_name "$experiment_name" --num_train_epochs 100 --learning_rate 2e-5 --full_finetuning
echo "➤ DONE"