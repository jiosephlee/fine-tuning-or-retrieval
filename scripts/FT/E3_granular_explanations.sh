#!/bin/bash
#SBATCH --job-name=E3_granular
#SBATCH --output=logs/E3_granular-%j.out
#SBATCH --error=logs/E3_granular-%j.err
#SBATCH --partition=dgx-b200
#SBATCH --gpus=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=0-8:00:00

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "$SLURM_SUBMIT_DIR/scripts/FT" ]]; then
    cd "$SLURM_SUBMIT_DIR/scripts/FT"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/finetuning_knowledge_v9.py" ]]; then
    cd "$SLURM_SUBMIT_DIR"
else
    cd "$(dirname "$0")"
fi
mkdir -p logs

MODEL_ID="${MODEL_ID:-allenai/OLMo-2-1124-7B}"
CONDA_ENV="${CONDA_ENV:-openrlhf}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
# In this CPT pipeline, NUM_EPOCHS is used as the target number of
# knowledge-injection batches when >1. The default 100 matches E1/E2 local.
NUM_EPOCHS="${NUM_EPOCHS:-100}"
NUM_PARAPHRASED="${NUM_PARAPHRASED:-9}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-64}"
EFFECTIVE_BATCH_SIZE="${EFFECTIVE_BATCH_SIZE:-256}"
LEARNING_RATE="${LEARNING_RATE:-4e-5}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-4096}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
CUSTOM_SUFFIX="${CUSTOM_SUFFIX:-E3_granular_explanations_all_domains}"
PUSH_TO_HUB_CPT_ID="${PUSH_TO_HUB_CPT_ID:-e3-olmo2-7b-para9-expl-20260516}"
EXPLANATION_TYPES="${EXPLANATION_TYPES:-textbooks stackexchange blogs}"
EXPLANATIONS_INSERTION_STRATEGY="${EXPLANATIONS_INSERTION_STRATEGY:-granular}"
EXPLANATIONS_CYCLE="${EXPLANATIONS_CYCLE:-full}"
EXPLANATIONS_NUM_TRACKS="${EXPLANATIONS_NUM_TRACKS:-1}"
EXPLANATION_GRANULARITY="${EXPLANATION_GRANULARITY:-file}"
EXPLANATION_TRACK_SIZE_BY_CHUNK="${EXPLANATION_TRACK_SIZE_BY_CHUNK:-4}"
EXPLANATION_TRACK_SCALE="${EXPLANATION_TRACK_SCALE:-1.0}"
EXPLANATION_MATCH_SCALE="${EXPLANATION_MATCH_SCALE:-$EXPLANATION_TRACK_SCALE}"
PRETRAINING_DATA_PATH="${PRETRAINING_DATA_PATH:-}"
KNOWLEDGE_PROBES_VERSION="${KNOWLEDGE_PROBES_VERSION:-v13}"
PARAPHRASED_KNOWLEDGE_PROBES_VERSION="${PARAPHRASED_KNOWLEDGE_PROBES_VERSION:-v13}"
MCQA_PROBES_VERSION="${MCQA_PROBES_VERSION:-v14}"
MCQA_PROMPT_COLUMN="${MCQA_PROMPT_COLUMN:-formatted_question_5shot}"
MCQA_PROBE_BATCH_SIZE="${MCQA_PROBE_BATCH_SIZE:-32}"
INFERENCE_MCQA_PROBES="${INFERENCE_MCQA_PROBES:-1}"
INFERENCE_MCQA_PROBES_VERSION="${INFERENCE_MCQA_PROBES_VERSION:-v12}"
INFERENCE_MCQA_PROMPT_COLUMN="${INFERENCE_MCQA_PROMPT_COLUMN:-formatted_question}"
USE_PARCC="${USE_PARCC:-0}"
SAVE_LOCAL_MODEL="${SAVE_LOCAL_MODEL:-0}"
SPARSE_CALLBACKS="${SPARSE_CALLBACKS:-0}"
PARAMETER_DELTA_EVERY_N_STEPS="${PARAMETER_DELTA_EVERY_N_STEPS:-5}"
PROBE_EVERY_N_STEPS="${PROBE_EVERY_N_STEPS:-2}"
MCQA_PROBE_EVERY_N_STEPS="${MCQA_PROBE_EVERY_N_STEPS:-4}"
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
if [[ "$INFERENCE_MCQA_PROBES" == "1" ]]; then
    EXTRA_ARGS+=(
        --inference_mcqa_probes
        --inference_mcqa_probes_version "$INFERENCE_MCQA_PROBES_VERSION"
        --inference_mcqa_prompt_column "$INFERENCE_MCQA_PROMPT_COLUMN"
    )
fi
if [[ -n "$PRETRAINING_DATA_PATH" ]]; then
    EXTRA_ARGS+=(--pretraining_data_path "$PRETRAINING_DATA_PATH")
fi
read -r -a EXPLANATION_TYPE_ARGS <<< "$EXPLANATION_TYPES"
EXPLANATION_SCHEDULE_ARGS=(
    --explanations_insertion_strategy "$EXPLANATIONS_INSERTION_STRATEGY"
    --granular_explanations_num_tracks "$EXPLANATIONS_NUM_TRACKS"
    --explanation_granularity "$EXPLANATION_GRANULARITY"
    --explanation_track_size_by_chunk "$EXPLANATION_TRACK_SIZE_BY_CHUNK"
    --explanation_track_scale "$EXPLANATION_TRACK_SCALE"
    --explanation_match_scale "$EXPLANATION_MATCH_SCALE"
)
if [[ "$EXPLANATIONS_INSERTION_STRATEGY" == "granular" ]]; then
    EXPLANATION_SCHEDULE_ARGS+=(--granular_explanations_cycle "$EXPLANATIONS_CYCLE")
fi

if command -v module >/dev/null 2>&1; then
    module load anaconda3 || true
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

torchrun --standalone --nproc_per_node "$NPROC_PER_NODE" finetuning_knowledge_v9.py \
    --custom_suffix "$CUSTOM_SUFFIX" \
    --wandb_group finetuning_official \
    --push_to_hub_cpt_id "$PUSH_TO_HUB_CPT_ID" \
    --model_id "$MODEL_ID" \
    --knowledge_probes_version "$KNOWLEDGE_PROBES_VERSION" \
    --paraphrased_knowledge_probes \
    --paraphrased_knowledge_probes_version "$PARAPHRASED_KNOWLEDGE_PROBES_VERSION" \
    --paraphrased_knowledge_probe_filename_suffix _paraphrased \
    --mcqa_probes \
    --mcqa_probes_version "$MCQA_PROBES_VERSION" \
    --mcqa_prompt_column "$MCQA_PROMPT_COLUMN" \
    --mcqa_probe_batch_size "$MCQA_PROBE_BATCH_SIZE" \
    --num_train_epochs "$NUM_EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    --lr_scheduler_min_lr_ratio 0.1 \
    --num_paraphrased_texts "$NUM_PARAPHRASED" \
    --with_specific_explanation "${EXPLANATION_TYPE_ARGS[@]}" \
    "${EXPLANATION_SCHEDULE_ARGS[@]}" \
    --overlap_sections \
    --overlap_ratio 1_16 \
    --device_batch_size "$DEVICE_BATCH_SIZE" \
    --effective_batch_size_for_cpt "$EFFECTIVE_BATCH_SIZE" \
    --context_length_for_cpt "$CONTEXT_LENGTH" \
    --fill_batches_with_pretraining \
    --attn_implementation "$ATTN_IMPLEMENTATION" \
    --gradient_checkpointing \
    --full_finetuning \
    --probe_every_n_steps "$PROBE_EVERY_N_STEPS" \
    --mcqa_probe_every_n_steps "$MCQA_PROBE_EVERY_N_STEPS" \
    --enable_parameter_delta_tracking \
    --parameter_delta_every_n_steps "$PARAMETER_DELTA_EVERY_N_STEPS" \
    "${EXTRA_ARGS[@]}"
