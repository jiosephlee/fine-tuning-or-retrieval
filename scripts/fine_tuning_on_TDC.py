# add .. path 
import os
import sys
sys.path.append('..')
import utils.llm_training as llm_training
import utils.llm_configs as llm_configs

import logging
import re
from tqdm import tqdm
import numpy as np
from datasets import Dataset
import pandas as pd
import argparse

# --- Basic Configuration ---
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="AMES")
parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B")
# Model names: jiosephlee/therapeutic_fine_tuning_1M_v2, jiosephlee/therapeutic_fine_tuning_10M, jiosephlee/therapeutic_fine_tuning_36M
args = parser.parse_args()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

os.environ["WANDB_PROJECT"]="medex_fine_tuning"

# --- Load Data and Preprocess---


training_ds = Dataset.from_pandas(train_df[["text"]], preserve_index=False)
test_ds = Dataset.from_pandas(test_df[["text"]], preserve_index=False).select_columns(["text"])

log.info(f"Training dataset example: {training_ds[0]}")
log.info(f"Test dataset example: {test_ds[0]}")

# --- Load Model ---
model_config = llm_configs.ModelConfig(
    id=args.model,
    peft=llm_configs.PeftConfig(
        enabled=False,
        add_eot_token=False,  # No longer doing EOT token for LIMA
    ),
    quantization=llm_configs.QuantizationConfig(mode=None), # Use QLoRA
)

log.info("--- Model Configuration ---")
log.info(model_config.model_dump_json(indent=2))

log.info("\n--- Loading Model for Training ---\n")
model, tokenizer = llm_training.load_model_for_training(model_config, log)

lima_training_config = llm_configs.TrainingConfig(
    run_name = f"{args.dataset} fine-tuning",
    num_train_epochs = 1,
    learning_rate  = 4e-5,
    logging_strategy = "steps", 
    logging_steps = 1,
    gradient_checkpointing=False,
    context_length = 512,
    use_liger_kernel=True,
    per_device_train_batch_size = 16,
    gradient_accumulation_steps= 16,
    # warmup_steps  = 0, # LIMA specifies no warmup, so we set this explicitly
    warmup_ratio = 0.3, # Use our default warmup ratio instead
    packing=True,
    padding_free = True,
    sequential_sampling = False,
    reverse_ffd_packing= False,
    remove_unused_columns=False,
)

log.info(f"\n--- Starting {args.dataset} Fine-Tuning ---")
trainer = llm_training.sft_train_on_dataset(
    model=model,
    tokenizer=tokenizer,
    log=log,
    train_dataset=training_ds,
    train_cfg=lima_training_config,
    train=True,
    use_liger_loss = True
)

log.info("\n\n--- Fine-Tuning Complete ---\n\n")
log.info(f"Training arguments: {trainer.args}")

# --- Evaluate ---
log.info("\n\n--- Evaluating ---\n\n")

inference_cfg = llm_configs.InferenceConfig(
    temperature=0,
    repetition_penalty=1.0,
    max_new_tokens=64,   # 32 is plenty for a single number
)

# regular expressions
row_pat   = re.compile(
    r"\[Drug SMILE]\s+(.*?)\s+\[Target]\s+(.*?)\s+\[Binding Affinity]\s+([-+]?\d*\.?\d+)"
)
num_pat   = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")    # catch first float in the generation

targets, preds = [], []

for row in tqdm(test_ds["text"], desc="Inference on test set"):
    m = row_pat.match(row)
    if m is None:
        # skip badly-formatted rows
        continue

    drug_smiles, target_id, gt_aff_str = m.groups()
    gt_aff = float(gt_aff_str)

    prompt = f"[Drug SMILE] {drug_smiles} [Target] {target_id} [Binding Affinity] "

    gen_text = llm_training.generate_text(model, tokenizer, prompt, inference_cfg)

    num_match = num_pat.search(gen_text)
    if num_match is None:
        # model didn’t output a float we can parse → skip
        continue

    pred_aff = float(num_match.group())

    targets.append(gt_aff)
    preds.append(pred_aff)

# ------------------
# 2. compute MSE
# ------------------
targets = np.array(targets, dtype=np.float32)
preds   = np.array(preds,   dtype=np.float32)

mse = np.mean((preds - targets) ** 2)
print(f"\nMSE on {len(targets)} examples: {mse:.4f}")

# Save model before we LIMA tune
#model.push_to_hub('jiosephlee/therapeutic_fine_tuning_36M')
#tokenizer.push_to_hub('jiosephlee/therapeutic_fine_tuning_36M')