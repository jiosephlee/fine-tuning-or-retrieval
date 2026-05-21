#!/bin/bash
#SBATCH --job-name=E17_source_arxiv_legal
#SBATCH --output=logs/E17_source_arxiv_legal-%j.out
#SBATCH --error=logs/E17_source_arxiv_legal-%j.err
#SBATCH --time=10:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=128GB

set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs

INFERENCE_MCQA_PROBES="${INFERENCE_MCQA_PROBES:-1}"
INFERENCE_MCQA_PROBES_VERSION="${INFERENCE_MCQA_PROBES_VERSION:-v12_reviewed v12}"
INFERENCE_MCQA_PROMPT_COLUMN="${INFERENCE_MCQA_PROMPT_COLUMN:-formatted_question}"

EXTRA_ARGS=()
if [[ "$INFERENCE_MCQA_PROBES" == "1" ]]; then
    read -r -a INFERENCE_MCQA_PROBE_VERSION_ARGS <<< "$INFERENCE_MCQA_PROBES_VERSION"
    EXTRA_ARGS+=(
        --inference_mcqa_probes
        --inference_mcqa_probes_version "${INFERENCE_MCQA_PROBE_VERSION_ARGS[@]}"
        --inference_mcqa_prompt_column "$INFERENCE_MCQA_PROMPT_COLUMN"
    )
fi

conda run --no-capture-output -n openrlhf torchrun --standalone --nproc_per_node 8 finetuning_knowledge_v9.py \
    --custom_suffix E17_source_arxiv_legal \
    --wandb_panel_sources arxiv legal \
    --model_id allenai/OLMo-2-1124-7B \
    --include_sources arxiv legal \
    --knowledge_probes_version v13 \
    --mcqa_probes \
    --mcqa_probes_version v14 \
    --mcqa_prompt_column formatted_question_5shot \
    --num_train_epochs 10 \
    --learning_rate 1e-5 \
    --num_paraphrased_texts 0 \
    --chunk_by_section \
    --overlap_sections \
    --overlap_ratio 1_8 \
    --device_batch_size 8 \
    --effective_batch_size_for_cpt 256 \
    --context_length_for_cpt 4096 \
    --fill_batches_with_pretraining \
    --attn_implementation flash_attention_2 \
    --gradient_checkpointing \
    --compile \
    --full_finetuning \
    --enable_parameter_delta_tracking \
    --no-save_local_model \
    "${EXTRA_ARGS[@]}"
