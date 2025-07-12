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
from sklearn.metrics import roc_auc_score

# --- Basic Configuration ---
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="AMES")
parser.add_argument("--metric", type=str, default="auroc")
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
train_df = pd.read_csv(f'./../data/TDC/{args.dataset}/train_df.csv')
val_df = pd.read_csv(f'./../data/TDC/{args.dataset}/val_df.csv')
test_df = pd.read_csv(f'./../data/TDC/{args.dataset}/test_df.csv')

def row_to_text( row, split='train', dataset='AMES'):
    if dataset == 'AMES':
        text = f"Q: This is the SMILES string of the drug: {row['Drug']}. Is this drug mutagenic?\nA: "
    elif dataset == 'Skin Reaction':
        text = f"Q: This is the SMILES string of the drug: {row['Drug']}. Can this drug cause skin reaction?\nA: "
    if split == 'train':
        text += f"{'Yes' if row['Y']==1 else 'No'}"
    return text

def row_to_prompt( row, split='train', dataset='AMES'):
    if dataset == 'AMES':
        text = f"Q: This is the SMILES string of the drug: {row['Drug']}. Is this drug mutagenic?\nA: "
    elif dataset == 'Skin Reaction':
        text = f"Q: This is the SMILES string of the drug: {row['Drug']}. Can this drug cause skin reaction?\nA: "
    if split == 'train':
        text += f"{'Yes' if row['Y']==1 else 'No'}"
    return text

def row_to_completion( row, split='train', dataset='AMES'):
    if dataset == 'AMES':
        text = f"Q: This is the SMILES string of the drug: {row['Drug']}. Is this drug mutagenic?\nA: "
    elif dataset == 'Skin Reaction':
        text = f"Q: This is the SMILES string of the drug: {row['Drug']}. Can this drug cause skin reaction?\nA: "
    if split == 'train':
        text += f"{'Yes' if row['Y']==1 else 'No'}"
    return text

train_df["text"] = train_df.apply(row_to_text, axis=1, split = 'train', dataset = 'AMES')
val_df["text"] = val_df.apply(row_to_text, axis=1, split = 'val', dataset = 'AMES')
test_df["text"] = test_df.apply(row_to_text, axis=1, split = 'test', dataset = 'AMES')

training_ds = Dataset.from_pandas(train_df, preserve_index=False)
training_ds = training_ds.select_columns(
                    {"text", "Y", "prompt", "completion"}.intersection(training_ds.column_names)
                )
val_ds = Dataset.from_pandas(val_df, preserve_index=False)
val_ds = val_ds.select_columns(
                    {"text", "Y", "prompt", "completion"}.intersection(val_ds.column_names)
                )
test_ds = Dataset.from_pandas(test_df, preserve_index=False)
test_ds = test_ds.select_columns(
                    {"text", "Y", "prompt", "completion"}.intersection(test_ds.column_names)
                )

log.info(f"Training dataset example: {training_ds[0]}")
log.info(f"Validation dataset example: {val_ds[0]}")
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
    run_name = f"{args.dataset} fine-tuning with {args.model}",
    num_train_epochs = 100,
    learning_rate  = 4e-5,
    logging_strategy = "steps", 
    logging_steps = 1,
    gradient_checkpointing=False,
    context_length = 512,
    use_liger_kernel=True,
    per_device_train_batch_size = 16,
    gradient_accumulation_steps=16,
    warmup_steps  = 0, # If 0, it does not override warmup ratio
    warmup_ratio = 0.1, # Use our default warmup ratio instead
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
    do_sample=False,
    repetition_penalty=1.0,
    max_new_tokens=64,
)

targets, preds = [], []

for i in tqdm(range(len(test_ds)), desc="Inference on test set"):
    row = test_ds[i]
    prompt = row["text"]
    gt_answer = "yes" if row["Y"] == 1 else "no"
    
    gen_text = llm_training.generate_text(model, tokenizer, prompt, inference_cfg)
    
    # Extract generated text (remove the prompt part)
    generated_response = gen_text[len(prompt):].strip().lower()

    if i < 10:
        print(f"Prompt: {prompt}")
        print(f"Generated response: {gen_text}")
        print(f"GT answer: {gt_answer}")
        print("-"*100)
        
    # Simple matching - check if "yes" or "no" appears in the response
    if "yes" in generated_response:
        pred_answer = "yes"
    elif "no" in generated_response:
        pred_answer = "no"
    else:
        # If neither yes nor no is found, skip this example
        continue

    
    targets.append(gt_answer)
    preds.append(pred_answer)

# ------------------
# Compute Accuracy
# ------------------
targets = np.array(targets)
preds = np.array(preds)

if args.metric == "accuracy":
    accuracy = np.mean(targets == preds)
    print(f"\nAccuracy on {len(targets)} examples: {accuracy:.4f}")
elif args.metric == "auroc":
    auroc = roc_auc_score(targets, preds)
    print(f"\nAUROC on {len(targets)} examples: {auroc:.4f}")

# Save model before we LIMA tune
#model.push_to_hub('jiosephlee/therapeutic_fine_tuning_36M')
#tokenizer.push_to_hub('jiosephlee/therapeutic_fine_tuning_36M')