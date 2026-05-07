#!/bin/bash

# Quick end-to-end FFN channel heatmap pipeline:
# 1) train source-only (singular) model
# 2) train paraphrase model
# 3) train auxiliary-views model
# 4) run per-channel FFN heatmap analysis

set -euo pipefail

# ---------------------------
# Shared config
# ---------------------------
MODEL_ID="allenai/OLMo-2-1124-7B"
DOMAINS=(DPO 1_58 GRPO BOFT OFT QLoRA)

NUM_EPOCHS=50
LR="2e-5"
DEVICE_BS=4
EFFECTIVE_BS=64
NUM_PARAPHRASED=9

# Auxiliary-view setup (use uncorrupted aux views)
AUX_EXPLANATIONS=(textbooks blogs stackexchange)

# Optional distributed launch:
#   USE_TORCHRUN=1 NPROC_PER_NODE=2 bash E99_7B_ffn_channel_heatmaps_quick.sh
USE_TORCHRUN="${USE_TORCHRUN:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

if [[ "${USE_TORCHRUN}" == "1" ]]; then
  PYTHON_LAUNCH=(torchrun --nproc_per_node "${NPROC_PER_NODE}")
else
  PYTHON_LAUNCH=(python -s)
fi

# Fixed experiment names so downstream paths are deterministic.
EXP_ROOT="quick_ffn_heatmaps/7b/e50"
EXP_SOURCE="${EXP_ROOT}/source"
EXP_PARA="${EXP_ROOT}/para9"
EXP_AUX="${EXP_ROOT}/aux_views"

# Where finetuning_knowledge_v8.py saves local checkpoints by default.
SOURCE_MODEL_PATH="../../results/FT/${EXP_SOURCE}/model_cpt"
PARA_MODEL_PATH="../../results/FT/${EXP_PARA}/model_cpt"
AUX_MODEL_PATH="../../results/FT/${EXP_AUX}/model_cpt"

ANALYSIS_DIR="../../plots/ffn_channel_heatmaps/quick_7b_e50"
MANIFEST_PATH="${ANALYSIS_DIR}/manifest_quick_7b_e50.json"

mkdir -p "${ANALYSIS_DIR}"

# echo "=== [1/4] Training source-only checkpoint ==="
# "${PYTHON_LAUNCH[@]}" finetuning_knowledge_v8.py \
#   --model_id "${MODEL_ID}" \
#   --override_domains "${DOMAINS[@]}" \
#   --num_train_epochs "${NUM_EPOCHS}" \
#   --learning_rate "${LR}" \
#   --device_batch_size "${DEVICE_BS}" \
#   --effective_batch_size_for_cpt "${EFFECTIVE_BS}" \
#   --fill_batches_with_pretraining \
#   --num_paraphrased_texts 0 \
#   --full_finetuning \
#   --override_experiment_name "${EXP_SOURCE}" \
#   --save_local_model

echo "=== [2/4] Training paraphrase checkpoint ==="
"${PYTHON_LAUNCH[@]}" finetuning_knowledge_v8.py \
  --model_id "${MODEL_ID}" \
  --override_domains "${DOMAINS[@]}" \
  --num_train_epochs "${NUM_EPOCHS}" \
  --learning_rate "${LR}" \
  --device_batch_size "${DEVICE_BS}" \
  --effective_batch_size_for_cpt "${EFFECTIVE_BS}" \
  --fill_batches_with_pretraining \
  --num_paraphrased_texts "${NUM_PARAPHRASED}" \
  --full_finetuning \
  --override_experiment_name "${EXP_PARA}" \
  --save_local_model

echo "=== [3/4] Training auxiliary-views checkpoint ==="
"${PYTHON_LAUNCH[@]}" finetuning_knowledge_v8.py \
  --model_id "${MODEL_ID}" \
  --override_domains "${DOMAINS[@]}" \
  --num_train_epochs "${NUM_EPOCHS}" \
  --learning_rate "${LR}" \
  --device_batch_size "${DEVICE_BS}" \
  --effective_batch_size_for_cpt "${EFFECTIVE_BS}" \
  --fill_batches_with_pretraining \
  --num_paraphrased_texts "${NUM_PARAPHRASED}" \
  --with_specific_explanation "${AUX_EXPLANATIONS[@]}" \
  --explanations_cycle full \
  --granular_explanation_analysis \
  --full_finetuning \
  --override_experiment_name "${EXP_AUX}" \
  --save_local_model

echo "=== [4/4] Building manifest + running FFN heatmap analysis ==="
cat > "${MANIFEST_PATH}" <<EOF
{
  "base": "${MODEL_ID}",
  "source": "${SOURCE_MODEL_PATH}",
  "para": "${PARA_MODEL_PATH}",
  "aux": "${AUX_MODEL_PATH}"
}
EOF

ANALYSIS_DIR="../../plots/ffn_channel_heatmaps/quick_7b_e50"
MANIFEST_PATH="${ANALYSIS_DIR}/manifest_quick_7b_e50.json"
python -s ../plotting/plot_ffn_channel_heatmaps.py \
  --manifest "${MANIFEST_PATH}" \
  --output_dir "${ANALYSIS_DIR}" \
  --device cpu \
  --clip_percentile 99 \
  --fig_width 22 \
  --fig_height 12 \
  --dpi 240

echo "Done. Outputs:"
echo "  Manifest: ${MANIFEST_PATH}"
echo "  Figures/arrays: ${ANALYSIS_DIR}"
