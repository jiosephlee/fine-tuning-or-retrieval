#! /bin/bash 
#$ -l gpu 
#$ -l h_vmem=36G 
module load deepmedic/2019-01/14 
CUDA_VISIBLE_DEVICES=$(get_CUDA_VISIBLE_DEVICES) || exit export CUDA_VISIBLE_DEVICES deepMedicRun -model /path/to/modelConfig.cfg -train /path/to/train/trainConfigWithValidation.cfg -dev cuda${CUDA_VISIBLE_DEVICES}