#!/bin/bash
#
# Reevaluate v12_reviewed inference MCQA only, for 1B (E4/E5/E6),
# 7B (E1/E2/E3), and 32B (E10/E11/E12) runs that already have a v12-only
# reeval (or, for 32B, an empty v12_reviewed+v12 reeval).
#
# For 1B/7B we create new sibling experiment dirs with `_inf_mcqa_v12_reviewed`
# (mirroring the 13B convention). For 32B we reuse the existing
# `_inf_mcqa_v12_reviewed+v12` path (currently empty).

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"
BATCH_SIZE="${BATCH_SIZE:-32}"
ONLY="${ONLY:-}"   # optional substring filter on the experiment tag
CONDA_ENV="${CONDA_ENV:-openrlhf}"

if [[ "${CONDA_DEFAULT_ENV:-}" == "$CONDA_ENV" ]]; then
    PY=(python)
else
    PY=(conda run --no-capture-output -n "$CONDA_ENV" python)
fi

run_one () {
    local tag="$1"
    local model_id="$2"
    local experiment_dir="$3"

    if [[ -n "$ONLY" && "$tag" != *"$ONLY"* ]]; then
        echo "[skip] $tag (filter ONLY=$ONLY)"
        return 0
    fi

    mkdir -p "$experiment_dir"
    echo "=== $tag ==="
    echo "    model: $model_id"
    echo "    out:   $experiment_dir/reeval"

    "${PY[@]}" evaluate_probes.py \
        --model_path "$model_id" \
        --experiment_dir "$experiment_dir" \
        --inference_mcqa_probes \
        --inference_mcqa_probes_version v12_reviewed \
        --inference_mcqa_prompt_column formatted_question_5shot \
        --mcqa_probe_batch_size "$BATCH_SIZE" \
        --attn_implementation "$ATTN_IMPL" \
        2>&1 | tee "logs/reeval_${tag}.log"
}

# 1B  (path: ..._inf_mcqa_v12 -> ..._inf_mcqa_v12_reviewed)
BASE_1B="../../results/FT/full/1b/probes_v13_para_v13_paraphrased_inf_v11_reviewed_mcqa_v14_prompt_formatted_question_5shot_inf_mcqa_v12_reviewed/newline2"
SUFFIX="fill_dclm/domains_arxiv_all-legal_all-medical_all/e100/bs256_lr4e-05/overlap_1_16"

run_one E4_1b jiosephlee/e4-olmo2-1b-source-only-20260516 \
    "$BASE_1B/source_only_docmatch_expl/$SUFFIX/E4_source_1b_all_domains_fa2_packing"
run_one E5_1b jiosephlee/e5-olmo2-1b-para9-20260516 \
    "$BASE_1B/para9_docmatch_expl/$SUFFIX/E5_paraphrase_1b_all_domains"
run_one E6_1b jiosephlee/e6-olmo2-1b-para9-expl-20260516 \
    "$BASE_1B/para9_expl_textbooks+stackexchange+blogs_cyclefull/$SUFFIX/E6_granular_explanations_1b_all_domains"

# 7B
BASE_7B="../../results/FT/full/7b/probes_v13_para_v13_paraphrased_inf_v11_reviewed_mcqa_v14_prompt_formatted_question_5shot_inf_mcqa_v12_reviewed/newline2"

run_one E1_7b jiosephlee/e1-olmo2-7b-source-only-20260516 \
    "$BASE_7B/source_only_docmatch_expl/$SUFFIX/E1_source_all_domains_fa2_packing"
run_one E2_7b jiosephlee/e2-olmo2-7b-para9-20260516 \
    "$BASE_7B/para9_docmatch_expl/$SUFFIX/E2_paraphrase_all_domains"
run_one E3_7b jiosephlee/e3-olmo2-7b-para9-expl-20260516 \
    "$BASE_7B/para9_expl_textbooks+stackexchange+blogs_cyclefull/$SUFFIX/E3_granular_explanations_all_domains"

# 32B (existing v12_reviewed+v12 path, currently empty)
BASE_32B="../../results/FT/full/allenai_OLMo-2-0325-32B/probes_v13_para_v13_paraphrased_inf_v11_reviewed_mcqa_v14_prompt_formatted_question_5shot_inf_mcqa_v12_reviewed+v12/newline2"

run_one E10_32b jiosephlee/e10-olmo2-32b-source-only-chunked-nll-1gpu-device-map-auto-36h-nodelta-20260518 \
    "$BASE_32B/source_only_docmatch_expl/$SUFFIX/E10_source_32b_all_domains_fa2_packing_chunked_nll_1gpu_device_map_auto_36h_nodelta"
run_one E11_32b jiosephlee/e11-olmo2-32b-para9-chunked-nll-1gpu-device-map-auto-36h-nodelta-20260518 \
    "$BASE_32B/para9_docmatch_expl/$SUFFIX/E11_paraphrase_32b_all_domains_chunked_nll_1gpu_device_map_auto_36h_nodelta"
run_one E12_32b jiosephlee/e12-olmo2-32b-para9-expl-1gpu-device-map-auto-36h-nodelta-20260518 \
    "$BASE_32B/para9_expl_textbooks+stackexchange+blogs_cyclefull/$SUFFIX/E12_granular_explanations_32b_all_domains_1gpu_device_map_auto_36h_nodelta"

echo "All reevals done."
