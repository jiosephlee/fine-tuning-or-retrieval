import os
import torch
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer
from tqdm.auto import tqdm
import logging
import sys
from dataclasses import dataclass, field
from typing import Optional, List
import math # For math.ceil

sys.path.append('..')
from utils.utils import query_llm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M",
)
logger = logging.getLogger(__name__)

# --- Configuration Dataclasses (ModelArguments remains the same) ---
@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"})
    torch_dtype: Optional[str] = field(default="auto", metadata={"help": "Override the default `torch.dtype`. Examples: 'float16', 'bfloat16', 'auto'."})
    attn_implementation: Optional[str] = field(default=None, metadata={"help": "Attention implementation (e.g., 'flash_attention_2')."})
    use_peft: bool = field(default=False, metadata={"help": "Whether to use PEFT (LoRA)."})
    lora_r: int = field(default=8, metadata={"help": "LoRA r."})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA alpha."})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout."})
    lora_target_modules: Optional[List[str]] = field(default_factory=lambda: ["Wqkv", "out_proj", "fc1", "fc2"], metadata={"help": "LoRA target modules."})
    quantization: Optional[str] = field(default=None, metadata={"help": "Quantization type ('4bit', '8bit')."})

# --- Global Script Configuration (remains the same) ---
GLOBAL_MODEL_ID = "allenai/OLMo-2-1124-7B-Instruct"
DATASET_ID = "princeton-nlp/TutorEval"
MAX_EPOCHS_PER_CHAPTER = 10
CONTEXT_LENGTH = 4096 # This is our max_seq_length for SFTTrainer
OUTPUT_DIR_BASE = "./olmo2_tutor_eval_finetune_sft_independent_dyn_grad_accum" # Updated dir name
MAX_GENERATION_LENGTH = 1024

# --- Helper for Quantization Config (remains the same) ---
def get_quantization_config_obj(model_args: ModelArguments) -> Optional[BitsAndBytesConfig]:
    if model_args.quantization == "4bit":
        compute_dtype = torch.float16
        if model_args.torch_dtype == "bfloat16" and torch.cuda.is_bf16_supported():
            compute_dtype = torch.bfloat16
        elif model_args.torch_dtype == "float16":
            compute_dtype = torch.float16
        return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=True)
    elif model_args.quantization == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    return None

# --- LLM Evaluation Function (remains the same) ---
def llm_output_is_correct_on_tutor_eval(llm_output: str, key_points: str, question: str) -> bool:
    prompt = {"system": "You are an expert evaluator...", "user": f"Question: {question}\n\nStudent Answer: {llm_output}\n\nReference Solution Key Points: {key_points}\n\n...Return only \"True\" or \"False\"."}
    try:
        result = query_llm(prompt)
        return result.strip().lower() == "true"
    except Exception as e:
        logger.error(f"Error in query_llm: {e}")
        return False

# --- Model and Tokenizer Loading (remains the same) ---
def load_fresh_model_and_tokenizer(model_args: ModelArguments, training_device: torch.device):
    logger.info(f"Loading FRESH base model and tokenizer for {model_args.model_name_or_path}...")
    quant_config = get_quantization_config_obj(model_args)
    actual_torch_dtype = getattr(torch, model_args.torch_dtype, None) if model_args.torch_dtype != "auto" else None
    model_load_kwargs = {"trust_remote_code": True, "attn_implementation": model_args.attn_implementation, "torch_dtype": actual_torch_dtype, "quantization_config": quant_config}
    if quant_config: model_load_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    if model.config.pad_token_id is None: model.config.pad_token_id = tokenizer.pad_token_id
    if not quant_config and "device_map" not in model_load_kwargs:
        try: model.to(training_device)
        except Exception as e: logger.error(f"Could not move model to {training_device}: {e}.")
    logger.info("FRESH Model and tokenizer loaded.")
    return model, tokenizer

    # Define the system prompt

system_prompt_content = "You are a helpful AI Assistant. Help me answer these textbook questions."
# --- Model Evaluation (remains the same) ---
@torch.no_grad()
def evaluate_model(model_to_eval, tokenizer, eval_device, questions: list[str], key_points_list: list[str], chapter_questions: list[str]):
    logger.info(f"Evaluating model on {len(questions)} questions...")
    model_to_eval.eval()
    correct_predictions = 0
    for i, question_text in tqdm(enumerate(questions), total=len(questions), desc="Evaluating Questions"):
        # print(f"Evaluating question: {question_text[:30]}...")
        print(f"Asking OLMO the following question: {question_text}")
                # Construct the chat messages
        messages = [
            {"role": "system", "content": system_prompt_content},
            {"role": "user", "content": question_text}
        ]
        inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True, # This is key for generation
                return_tensors="pt",
                truncation=True,
                # Max length for the tokenized prompt, leaving room for MAX_GENERATION_LENGTH
                max_length=CONTEXT_LENGTH - MAX_GENERATION_LENGTH
            )       
        try:
            if not (hasattr(model_to_eval, 'hf_device_map') and model_to_eval.hf_device_map): model_to_eval.to(eval_device)
            outputs = model_to_eval.generate(**inputs, max_new_tokens=MAX_GENERATION_LENGTH, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, do_sample=False)
            generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            print(f"The model is generating: {generated_text}")
        except Exception as e:
            logger.error(f"Error during generation for question '{question_text}': {e}")
            generated_text = ""
        if llm_output_is_correct_on_tutor_eval(generated_text, key_points_list[i], chapter_questions[i]):
            correct_predictions += 1
    accuracy = correct_predictions / len(questions) if questions else 0
    logger.info(f"Evaluation complete. Accuracy: {accuracy:.4f}")
    return accuracy

# --- Fine-tuning with SFTTrainer (MODIFIED for dynamic gradient_accumulation_steps) ---
def fine_tune_model_sft(
    model_to_fine_tune,
    tokenizer, # Pass tokenizer for calculating num_chunks
    chapter_full_text: str,
    chapter_name: str,
    epoch_num: int,
    model_args: ModelArguments,
    sft_config_base_overrides: dict, # Base overrides (like LR, batch_size)
    base_sft_output_dir: str,
    max_seq_length: int # This is CONTEXT_LENGTH
):
    logger.info(f"Starting SFT fine-tuning for chapter '{chapter_name}', epoch {epoch_num}...")
    dataset_text_column = "text_for_sft"
    dataset = Dataset.from_dict({dataset_text_column: [chapter_full_text]})
    sft_epoch_output_dir = os.path.join(base_sft_output_dir, f"sft_chapter_{chapter_name.replace(' ', '_').replace('/', '_')}_epoch_{epoch_num}")
    os.makedirs(sft_epoch_output_dir, exist_ok=True)

    # **MODIFICATION: Calculate number of effective chunks for gradient_accumulation_steps**
    if not chapter_full_text.strip():
        logger.warning(f"Chapter '{chapter_name}' text is empty. Skipping fine-tuning for this chapter/epoch.")
        return

    tokens = tokenizer(chapter_full_text, truncation=False, add_special_tokens=False)["input_ids"]
    num_tokens = len(tokens)
    if num_tokens == 0:
        logger.warning(f"Chapter '{chapter_name}' tokenized to 0 tokens. Skipping fine-tuning.")
        return

    # Calculate how many max_seq_length segments the chapter text will be broken into
    # This assumes SFTTrainer with packing=True will create this many effective sequences.
    num_effective_chunks = math.ceil(num_tokens / max_seq_length)
    num_effective_chunks = max(1, num_effective_chunks) # Ensure at least 1

    logger.info(f"Chapter '{chapter_name}': {num_tokens} tokens, {max_seq_length} max_seq_length -> {num_effective_chunks} effective chunks for grad_accum.")

    sft_args = {
        "output_dir": sft_epoch_output_dir,
        "overwrite_output_dir": True,
        "num_train_epochs": 1,
        "max_seq_length": max_seq_length,
        "dataset_text_field": dataset_text_column,
        "packing": True,
        "report_to": "none",
        "save_strategy": "no", # Model object is updated in-place
        # logging_steps is in sft_config_base_overrides
    }
    # Apply base overrides (e.g., batch size, lr, device)
    sft_args.update(sft_config_base_overrides) # Apply base overrides first

    # **MODIFICATION: Set gradient_accumulation_steps dynamically**
    # This will override any gradient_accumulation_steps in sft_config_base_overrides
    sft_args["gradient_accumulation_steps"] = num_effective_chunks

    if model_args.torch_dtype == "float16":
        sft_args["fp16"] = torch.cuda.is_available()
    elif model_args.torch_dtype == "bfloat16":
        sft_args["bf16"] = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    sft_training_config = SFTConfig(**sft_args)

    trainer = SFTTrainer(
        model=model_to_fine_tune,
        tokenizer=tokenizer, # Pass tokenizer to SFTTrainer
        args=sft_training_config,
        train_dataset=dataset,
    )
    logger.info(f"Calling SFTTrainer.train() for chapter '{chapter_name}', epoch {epoch_num}. Effective batch size: {sft_training_config.train_batch_size * num_effective_chunks}. Grad_accum_steps: {num_effective_chunks}")
    try:
        trainer.train()
        logger.info(f"SFT fine-tuning for chapter '{chapter_name}', epoch {epoch_num} complete.")
    except Exception as e:
        logger.error(f"Error during SFTTrainer.train() for chapter '{chapter_name}', epoch {epoch_num}: {e}")
        raise

# --- Main Experiment (Adjusted sft_config_step_overrides handling) ---
def run_experiment():
    logger.info("Starting OLMo2-1B TutorEval Fine-tuning Experiment (INDEPENDENT Training, DYNAMIC Grad Accum).")

    primary_device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Primary device selected: {primary_device}")

    model_args_config = ModelArguments(
        model_name_or_path=GLOBAL_MODEL_ID,
        torch_dtype="bfloat16" if primary_device.type == 'cuda' and torch.cuda.is_bf16_supported() else "float16",
        attn_implementation="flash_attention_2" if primary_device.type == 'cuda' else None,
        use_peft=False, # Set to True to enable LoRA
        quantization="8bit", # "4bit" or "8bit"
    )
    logger.info(f"Model Arguments: {model_args_config}")

    # Load tokenizer ONCE
    _, tokenizer_global = load_fresh_model_and_tokenizer(model_args_config, primary_device)
    logger.info("Global tokenizer loaded.")

    logger.info(f"Loading TutorEval dataset from {DATASET_ID}...")
    try:
        df = load_dataset(DATASET_ID, split="train").to_pandas()
    except Exception as e:
        logger.error(f"Failed to load TutorEval dataset: {e}"); return
    grouped_by_chapter = df.groupby("chapter")
    all_results = {}

    # Base SFT config overrides (gradient_accumulation_steps will be set dynamically per chapter)
    sft_config_base_overrides = {
        "per_device_train_batch_size": 1, # Keep this at 1 if grad_accum is num_chunks for 1 update per chapter
        "learning_rate": 5e-5,
        "logging_strategy": "steps",
        "logging_steps": 1, # Log after each effective step (which is per chapter if grad_accum=num_chunks)
        "device": primary_device, # Pass the determined device to SFTConfig
        "weight_decay": 0.01, # Example: add other args here
    }
    # Adjust base batch size if not aiming for 1 update per chapter with dynamic grad_accum
    # if model_args_config.use_peft or model_args_config.quantization:
    #     sft_config_base_overrides["per_device_train_batch_size"] = 2 # Example

    for chapter_idx, (chapter_full_text, chapter_data) in enumerate(grouped_by_chapter):
        # print(chapter_full_text)
        # print(chapter_data)
        chapter_name_for_log = f"Chapter{chapter_idx+1}_" + chapter_full_text[:30].replace("\n", " ").replace("/", "_").strip() + "..."
        logger.info(f"\n--- Processing Chapter: {chapter_name_for_log} (Independent Training) ---")

        logger.info(f"Loading FRESH model for chapter: {chapter_name_for_log}")
        model_for_this_chapter, _ = load_fresh_model_and_tokenizer(model_args_config, primary_device)

        if model_args_config.use_peft:
            logger.info(f"Applying PEFT (LoRA) to fresh model for chapter: {chapter_name_for_log}")
            # (PEFT application logic - ensure lora_target_modules are correct)
            if not model_args_config.lora_target_modules: model_args_config.lora_target_modules = ["Wqkv", "out_proj", "fc1", "fc2"]
            peft_config = LoraConfig(r=model_args_config.lora_r, lora_alpha=model_args_config.lora_alpha, lora_dropout=model_args_config.lora_dropout, target_modules=model_args_config.lora_target_modules, bias="none", task_type="CAUSAL_LM")
            model_for_this_chapter = get_peft_model(model_for_this_chapter, peft_config)
            model_for_this_chapter.print_trainable_parameters()

        if not (hasattr(model_for_this_chapter, 'hf_device_map') and model_for_this_chapter.hf_device_map):
            model_for_this_chapter.to(primary_device)
        logger.info(f"Model for chapter {chapter_name_for_log} is on device: {next(model_for_this_chapter.parameters()).device}")

        questions_for_chapter = chapter_data["question"].tolist()
        key_points_for_chapter = chapter_data["key_points"].tolist()

        if not chapter_full_text.strip() or not questions_for_chapter: # Check if chapter_full_text is not just whitespace
            logger.warning(f"Skipping chapter '{chapter_name_for_log}' due to empty text or no questions.")
            if hasattr(model_for_this_chapter, 'delete'): model_for_this_chapter.delete() # If using accelerate's dispatch
            del model_for_this_chapter
            if primary_device.type == 'cuda': torch.cuda.empty_cache()
            continue

        chapter_results_log = {"chapter_name": chapter_name_for_log, "epochs": []}
        logger.info(f"Evaluating UNTRAINED model state for chapter: {chapter_name_for_log}")
        accuracy_pristine_state = evaluate_model(model_for_this_chapter, tokenizer_global, primary_device, questions_for_chapter, key_points_for_chapter, questions_for_chapter)
        logger.info(f"Accuracy for '{chapter_name_for_log}' (Pristine Model State): {accuracy_pristine_state:.4f}")
        accuracy_after_previous_epoch_ft = accuracy_pristine_state
        for epoch in range(MAX_EPOCHS_PER_CHAPTER):
            current_epoch_num = epoch + 1
            logger.info(f"--- Chapter: {chapter_name_for_log}, Epoch: {current_epoch_num}/{MAX_EPOCHS_PER_CHAPTER} ---")
            fine_tune_model_sft(
                model_for_this_chapter,
                tokenizer_global, # Pass tokenizer here
                chapter_full_text,
                chapter_name_for_log,
                current_epoch_num,
                model_args_config,
                sft_config_base_overrides, # Pass the base dictionary
                OUTPUT_DIR_BASE,
                CONTEXT_LENGTH # This is max_seq_length
            )
            accuracy_after_current_epoch_ft = evaluate_model(model_for_this_chapter, tokenizer_global, primary_device, questions_for_chapter, key_points_for_chapter, questions_for_chapter)
            logger.info(f"Accuracy for '{chapter_name_for_log}' (Epoch {current_epoch_num} - After FT): {accuracy_after_current_epoch_ft:.4f}")
            chapter_results_log["epochs"].append({"epoch_num": current_epoch_num, "accuracy_before_ft_this_epoch": accuracy_after_previous_epoch_ft, "accuracy_after_ft_this_epoch": accuracy_after_current_epoch_ft})
            accuracy_after_previous_epoch_ft = accuracy_after_current_epoch_ft

        all_results[chapter_name_for_log] = chapter_results_log
        logger.info(f"Finished processing chapter: {chapter_name_for_log}. Results: {chapter_results_log['epochs']}")
        del model_for_this_chapter
        if primary_device.type == 'cuda': torch.cuda.empty_cache()
        elif primary_device.type == 'mps': import gc; gc.collect()
        logger.info(f"Cleaned up model for chapter: {chapter_name_for_log}")
        break
    # --- Report Overall Results (remains the same) ---
    logger.info("\n\n--- Overall Experiment Results (Independent Training, Dynamic Grad Accum) ---")
    results_list_for_df = []
    for chapter_name_key, chapter_res_data in all_results.items():
        logger.info(f"Chapter: {chapter_res_data['chapter_name']}")
        for epoch_data in chapter_res_data["epochs"]:
            logger.info(f"  Epoch {epoch_data['epoch_num']}: Before FT: {epoch_data['accuracy_before_ft_this_epoch']:.4f}, After FT: {epoch_data['accuracy_after_ft_this_epoch']:.4f}")
            results_list_for_df.append({"chapter": chapter_res_data['chapter_name'], "epoch": epoch_data["epoch_num"], "accuracy_before_ft_this_epoch": epoch_data['accuracy_before_ft_this_epoch'], "accuracy_after_ft_this_epoch": epoch_data['accuracy_after_ft_this_epoch']})
    if results_list_for_df:
        results_df = pd.DataFrame(results_list_for_df)
        results_path = os.path.join(OUTPUT_DIR_BASE, "experiment_results_sft_independent_dyn_grad_accum.csv")
        os.makedirs(OUTPUT_DIR_BASE, exist_ok=True)
        results_df.to_csv(results_path, index=False)
        logger.info(f"Detailed results saved to {results_path}")
    else: logger.info("No results to save.")

if __name__ == "__main__":
    run_experiment()
    logger.info("Script generated. Uncomment 'run_experiment()' to execute.")