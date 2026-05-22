#!/bin/bash
#
# Reevaluate inference MCQA v13 on LEGAL domains only, across the 1B/7B/13B/32B
# OLMo-2 runs. Writes into new sibling experiment dirs with the
# `_inf_mcqa_v13_legal` tag (mirroring the v12_reviewed convention).
#
# Schedules jobs across 2 local GPUs: one job per GPU, dispatched as slots free.
# Largest models first so the 32B runs finish around when 1B/7B fill the tail.

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"
BATCH_SIZE="${BATCH_SIZE:-32}"
ONLY="${ONLY:-}"   # optional substring filter on the experiment tag
CONDA_ENV="${CONDA_ENV:-openrlhf}"
NUM_GPUS="${NUM_GPUS:-2}"

if [[ "${CONDA_DEFAULT_ENV:-}" == "$CONDA_ENV" ]]; then
    PY=(python)
else
    PY=(conda run --no-capture-output -n "$CONDA_ENV" python)
fi

# 12 legal domains under probes/legal/
LEGAL_DOMAINS=(
    America_First_Legal_Foundation_v_Jamieson_Greer
    Apex_Bank_v_Cc_Serve_Corp
    Bruce_Cohen_v_Consilio_LLC
    Finesse_Wireless_LLC_v_Att_Mobility_LLC
    Foad_Farahi_v_FBI
    Jimenez_v_Bondi
    Pacito_v_Trump
    Santos_v_Kimmel
    United_States_v_Constantinescu
    United_States_v_Jaison_Coleman
    United_States_v_Justin_Cutbank
    Williams_v_GoAuto_Insurance
)

SUFFIX="fill_dclm/domains_arxiv_all-legal_all-medical_all/e100/bs256_lr4e-05/overlap_1_16"
PROBES_TAG="probes_v13_para_v13_paraphrased_inf_v11_reviewed_mcqa_v14_prompt_formatted_question_5shot_inf_mcqa_v13_legal"

BASE_1B="../../results/FT/full/1b/${PROBES_TAG}/newline2"
BASE_7B="../../results/FT/full/7b/${PROBES_TAG}/newline2"
BASE_13B="../../results/FT/full/allenai_OLMo-2-1124-13B/${PROBES_TAG}/newline2"
BASE_32B="../../results/FT/full/allenai_OLMo-2-0325-32B/${PROBES_TAG}/newline2"

# (tag  model_id  experiment_dir) — largest first
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
    local gpu="$1" tag="$2" model_id="$3" experiment_dir="$4"
    mkdir -p "$experiment_dir"
    echo "[gpu $gpu] === $tag ==="
    echo "[gpu $gpu]     model: $model_id"
    echo "[gpu $gpu]     out:   $experiment_dir"

    CUDA_VISIBLE_DEVICES="$gpu" \
    "${PY[@]}" evaluate_probes.py \
        --model_path "$model_id" \
        --experiment_dir "$experiment_dir" \
        --override_domains "${LEGAL_DOMAINS[@]}" \
        --inference_mcqa_probes \
        --inference_mcqa_probes_version v13 \
        --inference_mcqa_prompt_column formatted_question_5shot \
        --mcqa_probe_batch_size "$BATCH_SIZE" \
        --attn_implementation "$ATTN_IMPL" \
        2>&1 | tee "logs/reeval_v13_legal_${tag}.log"
}

# Track which PID is using which GPU
declare -A PID_GPU=()
free_gpus=()
for ((g=0; g<NUM_GPUS; g++)); do free_gpus+=("$g"); done

wait_for_slot () {
    # Block until any child finishes, then mark its GPU free.
    local finished_pid
    wait -n -p finished_pid || true
    if [[ -n "${finished_pid:-}" && -n "${PID_GPU[$finished_pid]:-}" ]]; then
        free_gpus+=("${PID_GPU[$finished_pid]}")
        unset 'PID_GPU[$finished_pid]'
    fi
}

for entry in "${JOBS[@]}"; do
    IFS='|' read -r tag model_id experiment_dir <<< "$entry"
    if [[ -n "$ONLY" && "$tag" != *"$ONLY"* ]]; then
        echo "[skip] $tag (filter ONLY=$ONLY)"
        continue
    fi

    while [[ ${#free_gpus[@]} -eq 0 ]]; do
        wait_for_slot
    done

    gpu="${free_gpus[0]}"
    free_gpus=("${free_gpus[@]:1}")

    run_one "$gpu" "$tag" "$model_id" "$experiment_dir" &
    PID_GPU[$!]="$gpu"
done

# Drain remaining jobs
while [[ ${#PID_GPU[@]} -gt 0 ]]; do
    wait_for_slot
done

echo "All v13 legal reevals done."
