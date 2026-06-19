#!/bin/bash
#
# reeval_v5: fill-in probe reeval for the 3 single-explanation "match" 7B runs
# (textbooks-only, stackexchange-only, blogs-only), serialized on 1 local GPU.
# Writes into the new results system at:
#   <run_dir>/eval_bundles/reeval_v5/
#
# Probes evaluated per model (same set as reeval_v3):
#   - factual knowledge probes (short targets)   v14   probes_v14_short_targets.csv
#   - factual paraphrased probes                 v14   probes_v14_paraphrased.csv
#   - factual MCQA (5-shot)                       v15
#   - inference probes                            v11_reviewed
#   - inference MCQA (5-shot)                     v14

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"
BATCH_SIZE="${BATCH_SIZE:-32}"
ONLY="${ONLY:-}"   # optional substring filter on the experiment tag
CONDA_ENV="${CONDA_ENV:-tuning}"
GPU_ID="${GPU_ID:-0}"
REEVAL_SUBDIR="${REEVAL_SUBDIR:-eval_bundles/reeval_v5}"

if [[ "${CONDA_DEFAULT_ENV:-}" == "$CONDA_ENV" ]]; then
    PY=(python)
else
    PY=(conda run --no-capture-output -n "$CONDA_ENV" python)
fi

SUFFIX="fill_dclm/domains_arxiv_all-legal_all-medical_all/e100/bs256_lr4e-05/overlap_1_16"
BASE_7B="../../results/FT/full/7b"

# (tag  model_id  experiment_dir)
JOBS=(
    "E3_7b_textbooks|jiosephlee/e3-olmo2-7b-para9-expl-textbooks-match-20260522|$BASE_7B/para9_docmatch_expl_inserttextbooks/$SUFFIX/E3_granular_explanations_textbooks_match_local"
    "E3_7b_stackexchange|jiosephlee/e3-olmo2-7b-para9-expl-stackexchange-match-20260522|$BASE_7B/para9_docmatch_expl_insertstackexchange/$SUFFIX/E3_granular_explanations_stackexchange_match_local"
    "E3_7b_blogs|jiosephlee/e3-olmo2-7b-para9-expl-blogs-match-20260522|$BASE_7B/para9_docmatch_expl_insertblogs/$SUFFIX/E3_granular_explanations_blogs_match_local"
)

run_one () {
    local tag="$1" model_id="$2" experiment_dir="$3"
    mkdir -p "$experiment_dir"
    echo "=== $tag ==="
    echo "    model: $model_id"
    echo "    out:   $experiment_dir/$REEVAL_SUBDIR"

    CUDA_VISIBLE_DEVICES="$GPU_ID" \
    "${PY[@]}" evaluate_probes.py \
        --model_path "$model_id" \
        --experiment_dir "$experiment_dir" \
        --reeval_subdir "$REEVAL_SUBDIR" \
        --knowledge_probes \
        --knowledge_probes_version v14 \
        --knowledge_probe_filename_suffix _short_targets \
        --paraphrased_knowledge_probes \
        --paraphrased_knowledge_probes_version v14 \
        --paraphrased_knowledge_probe_filename_suffix _paraphrased \
        --mcqa_probes \
        --mcqa_probes_version v15 \
        --mcqa_prompt_column formatted_question_5shot \
        --inference_probes \
        --inference_probes_version v11_reviewed \
        --inference_mcqa_probes \
        --inference_mcqa_probes_version v14 \
        --inference_mcqa_prompt_column formatted_question_5shot \
        --mcqa_probe_batch_size "$BATCH_SIZE" \
        --probe_batch_size "$BATCH_SIZE" \
        --attn_implementation "$ATTN_IMPL" \
        2>&1 | tee "logs/reeval_v5_${tag}.log"
}

for entry in "${JOBS[@]}"; do
    IFS='|' read -r tag model_id experiment_dir <<< "$entry"
    if [[ -n "$ONLY" && "$tag" != *"$ONLY"* ]]; then
        echo "[skip] $tag (filter ONLY=$ONLY)"
        continue
    fi
    run_one "$tag" "$model_id" "$experiment_dir"
done

echo "All reeval_v5 runs done."
