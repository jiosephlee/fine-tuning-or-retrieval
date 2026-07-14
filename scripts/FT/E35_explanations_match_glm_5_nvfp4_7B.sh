#!/bin/bash
#SBATCH --job-name=E35_expl_match_glm_5_nvfp4
#SBATCH --output=logs/E35_expl_match_glm_5_nvfp4-%j.out
#SBATCH --error=logs/E35_expl_match_glm_5_nvfp4-%j.err
#SBATCH --partition=dgx-b200
#SBATCH --gpus=2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=448G
#SBATCH --time=0-8:00:00

# Train with the audited GLM-5-NVFP4 explanations inserted into a matched
# auxiliary track shaped against the ordinary gpt_5_mini_custom granular
# explanation schedule. The reference corpus defines the per-step chunk counts
# (token budget); training content comes from the 36-item, 108-view
# data/{source}/explanations/glm_5_nvfp4 corpus.

set -euo pipefail

# Direct sbatch execution runs a spool copy, so prefer the original submit path
# over BASH_SOURCE when locating the project.
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "$SLURM_SUBMIT_DIR/scripts/FT" ]]; then
    SCRIPT_DIR="$(cd "$SLURM_SUBMIT_DIR/scripts/FT" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/finetuning_knowledge_v9.py" ]]; then
    SCRIPT_DIR="$(cd "$SLURM_SUBMIT_DIR" && pwd)"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$SCRIPT_DIR"
mkdir -p logs

MODEL_ID="${MODEL_ID:-allenai/OLMo-2-1124-7B}"
E35_ENV_ROOT="${E35_ENV_ROOT:-/vast/projects/myatskar/design-documents/conda_env}"
CONDA_ENV="${CONDA_ENV:-$E35_ENV_ROOT/tuning}"
PYTHON_PACKAGE_OVERLAY="${PYTHON_PACKAGE_OVERLAY:-$E35_ENV_ROOT/e35_e28_overlay}"
E35_REQUIRE_REPRO_ENV="${E35_REQUIRE_REPRO_ENV:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
NUM_EPOCHS="${NUM_EPOCHS:-100}"
NUM_PARAPHRASED="${NUM_PARAPHRASED:-9}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-32}"
EFFECTIVE_BATCH_SIZE="${EFFECTIVE_BATCH_SIZE:-256}"
LEARNING_RATE="${LEARNING_RATE:-4e-5}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-4096}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
CUSTOM_SUFFIX="${CUSTOM_SUFFIX:-E35_explanations_match_glm_5_nvfp4_all_domains}"
PUSH_TO_HUB_CPT_ID="${PUSH_TO_HUB_CPT_ID:-e35-olmo2-7b-para9-expl-match-glm-5-nvfp4-20260713}"
MCQA_PROBES_VERSION="${MCQA_PROBES_VERSION:-v15}"
MCQA_PROMPT_COLUMN="${MCQA_PROMPT_COLUMN:-formatted_question_5shot}"
MCQA_PROBE_BATCH_SIZE="${MCQA_PROBE_BATCH_SIZE:-32}"
INFERENCE_PROBES_VERSION="${INFERENCE_PROBES_VERSION:-v11}"
INFERENCE_MCQA_PROBES="${INFERENCE_MCQA_PROBES:-1}"
INFERENCE_MCQA_PROBES_VERSION="${INFERENCE_MCQA_PROBES_VERSION:-v14}"
INFERENCE_MCQA_PROMPT_COLUMN="${INFERENCE_MCQA_PROMPT_COLUMN:-formatted_question_5shot}"
USE_PARCC="${USE_PARCC:-1}"
SAVE_LOCAL_MODEL="${SAVE_LOCAL_MODEL:-0}"
SPARSE_CALLBACKS="${SPARSE_CALLBACKS:-0}"
PARAMETER_DELTA_EVERY_N_STEPS="${PARAMETER_DELTA_EVERY_N_STEPS:-5}"
PROBE_EVERY_N_STEPS="${PROBE_EVERY_N_STEPS:-2}"
MCQA_PROBE_EVERY_N_STEPS="${MCQA_PROBE_EVERY_N_STEPS:-4}"
DOCUMENT_TRACK_BASELINE="${DOCUMENT_TRACK_BASELINE:-1}"
DOCUMENT_MATCH_EXPLANATION_TYPES="${DOCUMENT_MATCH_EXPLANATION_TYPES:-textbooks stackexchange blogs}"
DOCUMENT_MATCH_EXPLANATIONS_CYCLE="${DOCUMENT_MATCH_EXPLANATIONS_CYCLE:-full}"
DOCUMENT_MATCH_INSERT_CONTENT="${DOCUMENT_MATCH_INSERT_CONTENT:-explanations}"
DOCUMENT_MATCH_INSERT_EXPLANATION_MODEL="${DOCUMENT_MATCH_INSERT_EXPLANATION_MODEL:-glm_5_nvfp4}"

for source in arxiv legal medical; do
    explanation_root="$PROJECT_ROOT/data/$source/explanations/$DOCUMENT_MATCH_INSERT_EXPLANATION_MODEL"
    if [[ ! -d "$explanation_root" ]]; then
        echo "Missing E35 explanation corpus: $explanation_root" >&2
        exit 1
    fi
    item_count="$(find "$explanation_root" -mindepth 1 -maxdepth 1 -type d | wc -l)"
    if [[ "$item_count" -ne 12 ]]; then
        echo "Incomplete E35 explanation corpus: $explanation_root has $item_count items; expected 12" >&2
        exit 1
    fi
done

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
    echo "E35 preflight passed: $DOCUMENT_MATCH_INSERT_EXPLANATION_MODEL (12 items in each domain)"
    exit 0
fi

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
    read -r -a INFERENCE_MCQA_PROBE_VERSION_ARGS <<< "$INFERENCE_MCQA_PROBES_VERSION"
    EXTRA_ARGS+=(
        --inference_mcqa_probes
        --inference_mcqa_probes_version "${INFERENCE_MCQA_PROBE_VERSION_ARGS[@]}"
        --inference_mcqa_prompt_column "$INFERENCE_MCQA_PROMPT_COLUMN"
    )
fi
if [[ "$DOCUMENT_TRACK_BASELINE" == "1" ]]; then
    read -r -a DOCUMENT_MATCH_EXPLANATION_TYPE_ARGS <<< "$DOCUMENT_MATCH_EXPLANATION_TYPES"
    EXTRA_ARGS+=(
        --document_track_baseline
        --document_match_specific_explanation "${DOCUMENT_MATCH_EXPLANATION_TYPE_ARGS[@]}"
        --document_match_insert_content "$DOCUMENT_MATCH_INSERT_CONTENT"
        --explanations_insertion_strategy granular
        --granular_explanations_cycle "$DOCUMENT_MATCH_EXPLANATIONS_CYCLE"
    )
    if [[ -n "$DOCUMENT_MATCH_INSERT_EXPLANATION_MODEL" ]]; then
        EXTRA_ARGS+=(--document_match_insert_explanation_model "$DOCUMENT_MATCH_INSERT_EXPLANATION_MODEL")
    fi
fi

if command -v module >/dev/null 2>&1; then
    module load anaconda3 || true
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

if [[ -n "$PYTHON_PACKAGE_OVERLAY" ]]; then
    if [[ ! -d "$PYTHON_PACKAGE_OVERLAY" ]]; then
        echo "Missing Python package overlay: $PYTHON_PACKAGE_OVERLAY" >&2
        exit 1
    fi
    export PYTHONPATH="$PYTHON_PACKAGE_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
fi

# Record the numerical stack in the Slurm log and optionally fail fast unless
# it matches the E28 environment whose base-model probe metrics were validated.
python -u - "$E35_REQUIRE_REPRO_ENV" <<'PY'
import sys

import accelerate
import flash_attn
import torch
import transformers
import trl

versions = {
    "python": ".".join(map(str, sys.version_info[:3])),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "transformers": transformers.__version__,
    "trl": trl.__version__,
    "accelerate": accelerate.__version__,
    "flash_attn": flash_attn.__version__,
}
print("E35 runtime: " + ", ".join(f"{key}={value}" for key, value in versions.items()))

if sys.argv[1] == "1":
    expected = {
        "python": "3.12.13",
        "torch": "2.10.0+cu130",
        "cuda": "13.0",
        "transformers": "5.13.1",
        "trl": "1.8.0",
        "accelerate": "1.14.0",
        "flash_attn": "2.8.3",
    }
    mismatches = [
        f"{key}: got {versions[key]!r}, expected {value!r}"
        for key, value in expected.items()
        if versions[key] != value
    ]
    if mismatches:
        raise SystemExit("E35 reproducibility environment mismatch:\n" + "\n".join(mismatches))
PY

RUN_ID="${RUN_ID:-${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"
TORCHRUN_LOG_DIR="${TORCHRUN_LOG_DIR:-logs/E35_expl_match_glm_5_nvfp4_${RUN_ID}}"
mkdir -p "$TORCHRUN_LOG_DIR"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

echo "torchrun logs: $TORCHRUN_LOG_DIR"

torchrun --standalone --nproc_per_node "$NPROC_PER_NODE" --log-dir "$TORCHRUN_LOG_DIR" --tee 3 finetuning_knowledge_v9.py \
    --custom_suffix "$CUSTOM_SUFFIX" \
    --wandb_group finetuning_official \
    --wandb_panel_sources arxiv legal medical \
    --model_id "$MODEL_ID" \
    --push_to_hub_cpt_id "$PUSH_TO_HUB_CPT_ID" \
    --include_sources arxiv legal medical \
    --knowledge_probes_version v14 \
    --inference_probes_version "$INFERENCE_PROBES_VERSION" \
    --paraphrased_knowledge_probes \
    --paraphrased_knowledge_probes_version v14 \
    --paraphrased_knowledge_probe_filename_suffix _paraphrased \
    --mcqa_probes \
    --mcqa_probes_version "$MCQA_PROBES_VERSION" \
    --mcqa_prompt_column "$MCQA_PROMPT_COLUMN" \
    --mcqa_probe_batch_size "$MCQA_PROBE_BATCH_SIZE" \
    --num_train_epochs "$NUM_EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    --lr_scheduler_min_lr_ratio 0.1 \
    --num_paraphrased_texts "$NUM_PARAPHRASED" \
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
