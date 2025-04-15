from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments, is_bfloat16_supported
import torch
import re
from tqdm import tqdm
from datasets import load_dataset, Dataset
import numpy as np
import pandas as pd
import copy # Needed for deep copying state dict

import sys
import os
sys.path.append(os.path.abspath(".."))
from importlib import reload
import utils.utils as utils
import utils.prompts as prompts
reload(utils)
reload(prompts)

# Track experiment
import wandb
wandb.login(key='d385cbc08ef0c734e84aff78ce2bb293b07f34e0')
os.environ["WANDB_PROJECT"]="Fine-Tuning-or-Retrieval"

# --- Model Configuration ---
max_seq_length = 2048
dtype = None
load_in_4bit = True
model_name = "unsloth/Meta-Llama-3.1-8B" # Or your preferred model
original_seed = 3407 # Define the base seed

print("Loading base model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
print("Base model loaded.")

EOS_TOKEN = tokenizer.eos_token
if tokenizer.pad_token is None:
    print("Setting pad token to EOS token.")
    tokenizer.pad_token = tokenizer.eos_token

print("Adding LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj", "embed_tokens", "lm_head",],
    lora_alpha=128,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=original_seed, # Use base seed here
    use_rslora=True,
    loftq_config=None,
)
print("LoRA adapters added.")

# --- Dataset Loading ---
print("Loading dataset...")
dataset = utils.load_dataset('PubMedQA', split='train', start_index=0, end_index=10) # Use 'train', smaller subset first
print(f"Dataset loaded with {len(dataset)} examples.")

# --- Helper Functions ---

def format_pretraining_text(context_list):
    """Formats context list into a single string for pre-training."""
    return "\n".join(context_list) + EOS_TOKEN

def format_qa_prompt(background, question):
    """Formats background and question into the Yes/No prompt."""
    return f"Background: {background}\n\nQuestion: {question}\n\nPlease answer with Yes or No." # EOS is handled by generation

def parse_yes_no(text):
    """Parses generated text to extract 'yes' or 'no'."""
    text_lower = text.lower().strip()
    # More robust parsing
    if re.search(r"^\s*yes", text_lower):
        return "yes"
    elif re.search(r"^\s*no", text_lower):
        return "no"
    # Fallback if not at the beginning
    elif "yes" in text_lower:
        return "yes"
    elif "no" in text_lower:
        return "no"
    return "unknown"

def evaluate_question(model, tokenizer, qa_prompt_text):
    """Generates an answer for the QA prompt and parses Yes/No."""
    FastLanguageModel.for_inference(model) # <<< Enable fast inference
    # model.eval()
    # Note: No EOS token added in format_qa_prompt now, let generation handle it
    inputs = tokenizer(qa_prompt_text, 
                       return_tensors="pt", 
    ).to(model.device) # Leave space

    outputs = model.generate(
        **inputs,
        max_new_tokens=50, 
        use_cache=True
    )
    prediction_text = tokenizer.decode(outputs)
    parsed_answer = parse_yes_no(prediction_text)
    # Optional: Put model back into training mode if needed outside this function
    # model.train() # Typically trainer handles this before training step
    return parsed_answer

# --- Experiment Setup ---
num_finetune_epochs_per_question = 5
results = []
# Store the initial state of the adapters
print("Saving initial model state...")
initial_model_state_dict = copy.deepcopy(model.state_dict())
print("Initial model state saved.")

# --- Main Experiment Loop ---
print("Starting experiment loop...")
for idx, example in enumerate(tqdm(dataset, desc="Processing Questions")):
    question_id = example.get('id', f'idx_{idx}')
    true_answer = example['final_decision'].lower()
    question_text = example['question']
    background_text = "\n".join(example['context']['contexts'])

    current_results = {
        'id': question_id,
        'question': question_text,
        'true_answer': true_answer,
        'predictions': {}
    }

    qa_prompt_text = format_qa_prompt(background_text, question_text)

    # 1. Pre-Tune Evaluation
    model.load_state_dict(initial_model_state_dict) # Reset state
    pre_train_pred = evaluate_question(model, tokenizer, qa_prompt_text)
    current_results['predictions']['pre_train'] = pre_train_pred

    # 2. Prepare Context Data
    context_for_tuning = format_pretraining_text(example['context']['contexts'])
    tuning_data = Dataset.from_dict({"text": [context_for_tuning]})

    # 3. Iterative Fine-Tuning & Evaluation Loop
    for epoch in range(1, num_finetune_epochs_per_question + 1):
        # Configure trainer - Use original args where possible
        temp_output_dir = f"./outputs_temp_{question_id}_epoch{epoch}"

        # Define arguments, keeping originals where feasible
        args = UnslothTrainingArguments(
            # --- Args to keep from original (potentially) ---
            warmup_ratio = 0.1,           # Original: 0.1
            learning_rate = 5e-5,         # Original: 5e-5
            # embedding_learning_rate = 5e-6, # Original: 5e-6 (can include if needed)
            optim = "adamw_8bit",         # Original: adamw_8bit
            weight_decay = 0.00,          # Original: 0.00
            lr_scheduler_type = "cosine", # Original: cosine (though effect minimal for 1 step)
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            # --- Args specific to this step ---
            per_device_train_batch_size = 1, # MUST be 1 for single example
            gradient_accumulation_steps = 1, # MUST be 1 for single step update
            num_train_epochs = 1,          # MUST be 1 for single step update
            logging_steps = 10,            # Adjust logging frequency if desired (original was 1)
            seed = original_seed + epoch,  # Vary seed per step
            output_dir = temp_output_dir,  # Temporary output
            report_to = "none",            # Disable reporting
            save_strategy = "no",          # Disable saving checkpoints
        )

        trainer = UnslothTrainer(
            model=model, # Pass current model state
            tokenizer=tokenizer,
            train_dataset=tuning_data,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            dataset_num_proc=1,
            args=args, # Use the defined args
        )

        # Fine-tune for one step
        # Trainer should handle model.train() / model.eval() transitions
        train_result = trainer.train()

        # Evaluate on the question *after* this epoch
        epoch_pred = evaluate_question(model, tokenizer, qa_prompt_text)
        current_results['predictions'][f'epoch_{epoch}'] = epoch_pred

        # Optional cleanup
        import shutil
        if os.path.exists(temp_output_dir): shutil.rmtree(temp_output_dir)

    results.append(current_results)

print("Experiment loop finished.")

# --- Analysis ---
print("\n--- Analyzing Results ---")
if not results:
    print("No results collected.")
else:
    df = pd.DataFrame(results)
    predictions_df = pd.json_normalize(df['predictions'])
    analysis_df = pd.concat([df[['id', 'question', 'true_answer']], predictions_df], axis=1)

    from sklearn.metrics import accuracy_score
    accuracies = {}
    stages = ['pre_train'] + [f'epoch_{e}' for e in range(1, num_finetune_epochs_per_question + 1)]

    for stage in stages:
        if stage in analysis_df.columns:
            valid_preds_mask = analysis_df[stage] != 'unknown'
            accuracy = accuracy_score(
                analysis_df.loc[valid_preds_mask, 'true_answer'],
                analysis_df.loc[valid_preds_mask, stage]
            ) if valid_preds_mask.sum() > 0 else 0.0 # Handle case with zero valid preds
            num_unknown = len(analysis_df) - valid_preds_mask.sum()
            accuracies[stage] = (accuracy, num_unknown)
        else:
            accuracies[stage] = (0.0, len(analysis_df))

    print(f"\nProcessed {len(df)} questions.")
    print("\nAccuracies (Ignoring 'unknown' predictions):")
    for stage, (acc, unknown_count) in accuracies.items():
        total_count = len(analysis_df)
        valid_count = total_count - unknown_count
        print(f"- {stage}: {acc:.4f} ({valid_count}/{total_count} valid predictions, {unknown_count} unknown)")

    output_filename = "rq1_experiment_results.csv"
    analysis_df.to_csv(output_filename, index=False)
    print(f"\nDetailed results saved to {output_filename}")