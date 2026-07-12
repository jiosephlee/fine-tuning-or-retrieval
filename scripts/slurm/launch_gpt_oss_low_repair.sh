#!/bin/bash
# Targeted low-reasoning GPT-OSS repair sweep.
#
# The five one-GPU allocations have a hard aggregate ceiling of 20 GPU-hours.
# Pipelines write transactionally into the recovery slugs and skip only views whose
# generation manifests still validate against their current file hashes.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT="$SCRIPT_DIR/submit_vllm_multiview.sh"
DRY_RUN=0
MAX_GPU_MINUTES=1200

if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
    shift
fi
if (( $# )); then
    echo "Usage: $0 [--dry-run]" >&2
    exit 2
fi

# size|domain|wall minutes|comma-separated items with at least one missing view
JOBS=(
  "20b|arxiv|420|ByteLatent,DPO,FeatLLM,GRPO,GSPO,LongRoPE,OFT,QLoRA,fa3,xLSTM"
  "20b|medical|60|Nontype_1_and_nontype_2_diabetes_in_a_young_man_due_to_novel"
  "20b|legal|120|Foad_Farahi_v_FBI,United_States_v_Constantinescu,Williams_v_GoAuto_Insurance"
  "120b|arxiv|420|1_58,BOFT,ByteLatent,DPO,FeatLLM,GRPO,GSPO,LongRoPE,OFT,QLoRA,fa3,xLSTM"
  "120b|medical|180|Dermatomyositis_masquerading_as_angioedema_a_crucial_differe,Monoallelic_PARN_mutation_presenting_as_pancytopenia_hepatic,TAVinTAVinTAV_after_treated_endocarditis_procedural_strategy"
)

total=0
for spec in "${JOBS[@]}"; do
    IFS='|' read -r size domain minutes items <<< "$spec"
    total=$((total + minutes))
done
if (( total > MAX_GPU_MINUTES )); then
    echo "Refusing submission: ${total} GPU-minutes exceeds ${MAX_GPU_MINUTES}" >&2
    exit 1
fi
echo "Hard allocation ceiling: ${total} GPU-minutes ($((total / 60)) GPU-hours)"

for spec in "${JOBS[@]}"; do
    IFS='|' read -r size domain minutes items <<< "$spec"
    hours=$((minutes / 60))
    mins=$((minutes % 60))
    printf -v wall '%02d:%02d:00' "$hours" "$mins"
    model="openai/gpt-oss-${size}"
    slug="gpt_oss_${size}_low_recovery"
    cmd=(
      "$SUBMIT" --partition dgx-b200 --gpus 1 --tensor-parallel-size 1
      --cpus 28 --memory 224G --time "$wall"
      --model "$model" --domain "$domain" --item "$items" --parts all
      --model-slug "$slug" --reasoning-effort low --max-workers 4
      --max-model-len auto --max-tokens 24576
      --gpu-memory-utilization 0.95 --max-num-batched-tokens 8192 --max-num-seqs 256
      --tool-call-parser openai --enable-auto-tool-choice 1
    )
    if (( DRY_RUN )); then cmd+=(--dry-run); fi
    echo ">>> ${size} ${domain}: ${wall}, slug=${slug}"
    "${cmd[@]}"
done
