#!/bin/bash
#
# reeval_v4 on the four OLMo-2 BASE (un-fine-tuned) models, factual MCQA
# (5-shot) at both v15 and v16. Serialized on 1 local GPU.
#
# Writes into:
#   results/FT/full/<model_sanitized>/eval_bundles/reeval_v4_mcqa_v15/
#   results/FT/full/<model_sanitized>/eval_bundles/reeval_v4_mcqa_v16/

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"
BATCH_SIZE="${BATCH_SIZE:-32}"
ONLY="${ONLY:-}"   # optional substring filter on the model tag
CONDA_ENV="${CONDA_ENV:-tuning}"
GPU_ID="${GPU_ID:-0}"

if [[ "${CONDA_DEFAULT_ENV:-}" == "$CONDA_ENV" ]]; then
    PY=(python)
else
    PY=(conda run --no-capture-output -n "$CONDA_ENV" python)
fi

# (tag  model_id)
MODELS=(
    "olmo2_32b_base|allenai/OLMo-2-0325-32B"
    "olmo2_13b_base|allenai/OLMo-2-1124-13B"
    "olmo2_7b_base|allenai/OLMo-2-1124-7B"
    "olmo2_1b_base|allenai/OLMo-2-0425-1B"
)

run_pass () {
    local tag="$1" model_id="$2" version="$3"
    local model_sanitized="${model_id//\//_}"
    local experiment_dir="../../results/FT/full/${model_sanitized}"
    local reeval_subdir="eval_bundles/reeval_v4_mcqa_${version}"
    mkdir -p "$experiment_dir"
    echo "--- $tag mcqa $version ---"
    echo "    model: $model_id"
    echo "    out:   $experiment_dir/$reeval_subdir"

    CUDA_VISIBLE_DEVICES="$GPU_ID" \
    "${PY[@]}" evaluate_probes.py \
        --model_path "$model_id" \
        --experiment_dir "$experiment_dir" \
        --reeval_subdir "$reeval_subdir" \
        --mcqa_probes \
        --mcqa_probes_version "$version" \
        --mcqa_prompt_column formatted_question_5shot \
        --mcqa_probe_batch_size "$BATCH_SIZE" \
        --attn_implementation "$ATTN_IMPL" \
        2>&1 | tee "logs/reeval_v4_base_${tag}_mcqa_${version}.log"
}

for entry in "${MODELS[@]}"; do
    IFS='|' read -r tag model_id <<< "$entry"
    if [[ -n "$ONLY" && "$tag" != *"$ONLY"* ]]; then
        echo "[skip] $tag (filter ONLY=$ONLY)"
        continue
    fi
    echo "=== $tag ==="
    run_pass "$tag" "$model_id" v15
    run_pass "$tag" "$model_id" v16
done

echo "All reeval_v4 base-model runs done."
