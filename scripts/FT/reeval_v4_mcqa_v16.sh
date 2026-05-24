#!/bin/bash
#
# reeval_v4: factual MCQA v16 only (5-shot), for all 12 OLMo-2 runs
# (1B/7B/13B/32B), serialized on 1 local GPU. Writes into:
#   <run_dir>/eval_bundles/reeval_v4/
#
# Probes evaluated per model:
#   - factual MCQA (5-shot)   v16   probes_v16_mcqa.csv

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"
BATCH_SIZE="${BATCH_SIZE:-32}"
ONLY="${ONLY:-}"   # optional substring filter on the experiment tag
CONDA_ENV="${CONDA_ENV:-tuning}"
GPU_ID="${GPU_ID:-0}"
REEVAL_SUBDIR="${REEVAL_SUBDIR:-eval_bundles/reeval_v4}"

if [[ "${CONDA_DEFAULT_ENV:-}" == "$CONDA_ENV" ]]; then
    PY=(python)
else
    PY=(conda run --no-capture-output -n "$CONDA_ENV" python)
fi

SUFFIX="fill_dclm/domains_arxiv_all-legal_all-medical_all/e100/bs256_lr4e-05/overlap_1_16"

BASE_1B="../../results/FT/full/1b"
BASE_7B="../../results/FT/full/7b"
BASE_13B="../../results/FT/full/13b"
BASE_32B="../../results/FT/full/32b"

JOBS=(
    "E10_32b|jiosephlee/e10-olmo2-32b-source-only-chunked-nll-1gpu-device-map-auto-36h-nodelta-20260518|$BASE_32B/source_only_docmatch_expl/$SUFFIX/E10_source_32b_all_domains_fa2_packing_chunked_nll_1gpu_device_map_auto_36h_nodelta"
    "E11_32b|jiosephlee/e11-olmo2-32b-para9-chunked-nll-1gpu-device-map-auto-36h-nodelta-20260518|$BASE_32B/para9_docmatch_expl/$SUFFIX/E11_paraphrase_32b_all_domains_chunked_nll_1gpu_device_map_auto_36h_nodelta"
    "E12_32b|jiosephlee/e12-olmo2-32b-para9-expl-1gpu-device-map-auto-36h-nodelta-20260518|$BASE_32B/para9_expl_textbooks+stackexchange+blogs_cyclefull/$SUFFIX/E12_granular_explanations_32b_all_domains_1gpu_device_map_auto_36h_nodelta"
    "E7_13b|jiosephlee/e7-olmo2-13b-source-only-20260516|$BASE_13B/source_only_docmatch_expl/$SUFFIX/E7_source_13b_all_domains_fa2_packing"
    "E8_13b|jiosephlee/e8-olmo2-13b-para9-20260516|$BASE_13B/para9_docmatch_expl/$SUFFIX/E8_paraphrase_13b_all_domains"
    "E9_13b|jiosephlee/e9-olmo2-13b-para9-expl-20260516|$BASE_13B/para9_expl_textbooks+stackexchange+blogs_cyclefull/$SUFFIX/E9_granular_explanations_13b_all_domains"
    "E1_7b|jiosephlee/e1-olmo2-7b-source-only-20260516|$BASE_7B/source_only_docmatch_expl/$SUFFIX/E1_source_all_domains_fa2_packing"
    "E2_7b|jiosephlee/e2-olmo2-7b-para9-20260516|$BASE_7B/para9_docmatch_expl/$SUFFIX/E2_paraphrase_all_domains"
    "E3_7b|jiosephlee/e3-olmo2-7b-para9-expl-20260516|$BASE_7B/para9_expl_textbooks+stackexchange+blogs_cyclefull/$SUFFIX/E3_granular_explanations_all_domains"
    "E4_1b|jiosephlee/e4-olmo2-1b-source-only-20260516|$BASE_1B/source_only_docmatch_expl/$SUFFIX/E4_source_1b_all_domains_fa2_packing"
    "E5_1b|jiosephlee/e5-olmo2-1b-para9-20260516|$BASE_1B/para9_docmatch_expl/$SUFFIX/E5_paraphrase_1b_all_domains"
    "E6_1b|jiosephlee/e6-olmo2-1b-para9-expl-20260516|$BASE_1B/para9_expl_textbooks+stackexchange+blogs_cyclefull/$SUFFIX/E6_granular_explanations_1b_all_domains"
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
        --mcqa_probes \
        --mcqa_probes_version v16 \
        --mcqa_prompt_column formatted_question_5shot \
        --mcqa_probe_batch_size "$BATCH_SIZE" \
        --attn_implementation "$ATTN_IMPL" \
        2>&1 | tee "logs/reeval_v4_${tag}.log"
}

for entry in "${JOBS[@]}"; do
    IFS='|' read -r tag model_id experiment_dir <<< "$entry"
    if [[ -n "$ONLY" && "$tag" != *"$ONLY"* ]]; then
        echo "[skip] $tag (filter ONLY=$ONLY)"
        continue
    fi
    run_one "$tag" "$model_id" "$experiment_dir"
done

echo "All reeval_v4 runs done."
