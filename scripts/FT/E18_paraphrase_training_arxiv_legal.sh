#!/bin/bash
#SBATCH --job-name=E18_para_arxiv_legal
#SBATCH --output=logs/E18_para_arxiv_legal-%j.out
#SBATCH --error=logs/E18_para_arxiv_legal-%j.err
#SBATCH --time=10:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=128GB

set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs

MODEL_ID="${MODEL_ID:-allenai/OLMo-2-1124-7B}"
CONDA_ENV="${CONDA_ENV:-openrlhf}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
# In this CPT pipeline, NUM_EPOCHS is used as the target number of
# knowledge-injection batches when >1. The default 10 is one full
# source+9-paraphrase cycle, matching E1/E3 compute.
NUM_EPOCHS="${NUM_EPOCHS:-10}"
NUM_PARAPHRASED="${NUM_PARAPHRASED:-9}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-16}"
EFFECTIVE_BATCH_SIZE="${EFFECTIVE_BATCH_SIZE:-128}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-4096}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
CUSTOM_SUFFIX="${CUSTOM_SUFFIX:-E18_paraphrase_arxiv_legal}"
MCQA_PROBES_VERSION="${MCQA_PROBES_VERSION:-v14}"
MCQA_PROMPT_COLUMN="${MCQA_PROMPT_COLUMN:-formatted_question_5shot}"
USE_PARCC="${USE_PARCC:-0}"
SAVE_LOCAL_MODEL="${SAVE_LOCAL_MODEL:-0}"
SPARSE_CALLBACKS="${SPARSE_CALLBACKS:-0}"
PARAMETER_DELTA_EVERY_N_STEPS="${PARAMETER_DELTA_EVERY_N_STEPS:-5}"
EXTRA_ARGS=()
if [[ "$USE_PARCC" == "1" ]]; then
    EXTRA_ARGS+=(--parcc)
fi
if [[ "$SAVE_LOCAL_MODEL" == "1" ]]; then
    EXTRA_ARGS+=(--save_local_model)
else
    EXTRA_ARGS+=(--no-save_local_model)
fi
if [[ "$SPARSE_CALLBACKS" == "1" ]]; then
    EXTRA_ARGS+=(--no_callback_every_step)
fi

if [[ "${CONDA_DEFAULT_ENV:-}" == "$CONDA_ENV" ]]; then
    LAUNCH=(torchrun --standalone --nproc_per_node "$NPROC_PER_NODE")
else
    LAUNCH=(conda run --no-capture-output -n "$CONDA_ENV" torchrun --standalone --nproc_per_node "$NPROC_PER_NODE")
fi

"${LAUNCH[@]}" finetuning_knowledge_v9.py \
    --custom_suffix "$CUSTOM_SUFFIX" \
    --model_id "$MODEL_ID" \
    --include_sources arxiv legal \
    --wandb_panel_sources arxiv legal \
    --knowledge_probes_version v13 \
    --mcqa_probes \
    --mcqa_probes_version "$MCQA_PROBES_VERSION" \
    --mcqa_prompt_column "$MCQA_PROMPT_COLUMN" \
    --num_train_epochs "$NUM_EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    --num_paraphrased_texts "$NUM_PARAPHRASED" \
    --device_batch_size "$DEVICE_BATCH_SIZE" \
    --effective_batch_size_for_cpt "$EFFECTIVE_BATCH_SIZE" \
    --context_length_for_cpt "$CONTEXT_LENGTH" \
    --fill_batches_with_pretraining \
    --attn_implementation "$ATTN_IMPLEMENTATION" \
    --gradient_checkpointing \
    --full_finetuning \
    --enable_parameter_delta_tracking \
    --parameter_delta_every_n_steps "$PARAMETER_DELTA_EVERY_N_STEPS" \
    "${EXTRA_ARGS[@]}"
