import os
import torch
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig, # For quantization
)
from peft import LoraConfig, get_peft_model, PeftModel # For PEFT (LoRA)
from trl import SFTConfig, SFTTrainer # The new trainer
from tqdm.auto import tqdm
import logging
import sys
from dataclasses import dataclass, field
from typing import Optional, List

sys.path.append('..') 
from utils.utils import query_llm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M",
)
logger = logging.getLogger(__name__)

# --- Configuration Dataclasses ---
@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"})
    torch_dtype: Optional[str] = field(
        default="auto",
        metadata={"help": "Override the default `torch.dtype` and load the model under this dtype. Examples: 'float16', 'bfloat16', 'auto'."}
    )
    attn_implementation: Optional[str] = field(
        default="flash_attention_2", # Recommended: "flash_attention_2" if available and supported
        metadata={"help": "Attention implementation to use (e.g., 'flash_attention_2', None for default)."}
    )
    use_peft: bool = field(default=False, metadata={"help": "Whether to use PEFT (LoRA) for fine-tuning."})
    lora_r: int = field(default=8, metadata={"help": "LoRA r parameter."})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA alpha parameter."})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout."})
    lora_target_modules: Optional[List[str]] = field(
        default_factory=lambda: ["Wqkv", "out_proj", "fc1", "fc2"], # Default for OLMo-1B, VERIFY for OLMo-2
        metadata={"help": "List of module names to apply LoRA to. If None, defaults are used."}
    )
    quantization: Optional[str] = field(
        default=None, # Examples: "4bit", "8bit"
        metadata={"help": "Quantization type (e.g., '4bit', '8bit'). Requires bitsandbytes."}
    )

# --- Global Script Configuration ---
GLOBAL_MODEL_ID = "allenai/OLMo-2-0425-1B"
DATASET_ID = "princeton-nlp/TutorEval"
MAX_EPOCHS_PER_CHAPTER = 3
CONTEXT_LENGTH = 4096
OUTPUT_DIR_BASE = "./olmo2_tutor_eval_finetune_sft_independent" # Changed dir name
MAX_GENERATION_LENGTH = 256

# --- Helper for Quantization Config ---
def get_quantization_config_obj(model_args: ModelArguments) -> Optional[BitsAndBytesConfig]:
    if model_args.quantization == "4bit":
        compute_dtype = torch.float16
        if model_args.torch_dtype == "bfloat16" and torch.cuda.is_bf16_supported():
            compute_dtype = torch.bfloat16
        elif model_args.torch_dtype == "float16":
            compute_dtype = torch.float16
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
    elif model_args.quantization == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    return None

# --- LLM Evaluation Function (User Provided) ---
def llm_output_is_correct_on_tutor_eval(llm_output: str, key_points: str, question: str) -> bool:
    prompt = {}
    prompt["system"] = """You are an expert evaluator of textbook question answers."""
    prompt["user"] = f"""Here is the given textbook question, student answer, and reference solution.

Question: {question}

Student Answer: {llm_output}

Reference Solution Key Points: {key_points}

Your task is to determine if the student answer correctly addresses the question and contains the key points from the reference solution.
The student answer doesn't need to match the reference exactly, but it should demonstrate understanding of the core concepts and provide a correct solution.

Evaluate only the correctness of the answer, not writing style or grammar.
Return only "True" if the answer is correct or "False" if it is incorrect.
"""
    try:
        result = query_llm(prompt)
        return result.strip().lower() == "true"
    except Exception as e:
        logger.error(f"Error in query_llm during evaluation: {e}")
        return False


# --- Model and Tokenizer Loading ---
def load_fresh_model_and_tokenizer(model_args: ModelArguments, training_device: torch.device): # Renamed for clarity
    logger.info(f"Loading FRESH base model and tokenizer for {model_args.model_name_or_path}...")
    quant_config = get_quantization_config_obj(model_args)
    if model_args.torch_dtype == "auto":
        actual_torch_dtype = None
    else:
        actual_torch_dtype = getattr(torch, model_args.torch_dtype, None)
        if actual_torch_dtype is None:
            logger.warning(f"Invalid torch_dtype '{model_args.torch_dtype}'. Defaulting to auto.")

    model_load_kwargs = {
        "trust_remote_code": True,
        "attn_implementation": model_args.attn_implementation,
        "torch_dtype": actual_torch_dtype,
        "quantization_config": quant_config,
    }
    if quant_config:
        model_load_kwargs["device_map"] = "auto"
    
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    if not quant_config and "device_map" not in model_load_kwargs:
        try:
            model.to(training_device)
        except Exception as e:
            logger.error(f"Could not move model to {training_device}: {e}. Model might remain on CPU or meta device.")
    
    logger.info("FRESH Model and tokenizer loaded.")
    return model, tokenizer


# --- Text Chunking (No changes needed) ---
def chunk_text(text: str, tokenizer: AutoTokenizer, chunk_size: int = CONTEXT_LENGTH):
    logger.debug(f"Chunking text of length {len(text)} characters.")
    tokens = tokenizer(text, truncation=False, return_attention_mask=False, add_special_tokens=False)["input_ids"]
    text_chunks = []
    for i in range(0, len(tokens), chunk_size):
        chunk_token_ids = tokens[i:i + chunk_size]
        text_chunks.append(tokenizer.decode(chunk_token_ids))
    if not text_chunks:
        text_chunks.append(text)
    logger.debug(f"Text yielded {len(text_chunks)} potential text segments for SFTTrainer (pre-chunking for dataset).")
    return text_chunks

# --- Model Evaluation (No changes needed in logic, but device handling is key) ---
@torch.no_grad()
def evaluate_model(model_to_eval, tokenizer, eval_device, questions: list[str], key_points_list: list[str], chapter_questions: list[str]):
    logger.info(f"Evaluating model on {len(questions)} questions...")
    model_to_eval.eval()
    correct_predictions = 0
    
    # Determine device for generation inputs
    # If model is PeftModel, model.device might point to the device of the first parameter.
    # If quantized and device_mapped, generation should handle device placement.
    # Forcing inputs to eval_device which should be consistent with SFTConfig.device.
    
    for i, question_text in tqdm(enumerate(questions), total=len(questions), desc="Evaluating Questions"):
        inputs = tokenizer(
            question_text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=CONTEXT_LENGTH - MAX_GENERATION_LENGTH
        ).to(eval_device) # Ensure inputs are on the evaluation device
        
        try:
            # Ensure model is also on the eval_device if not device_mapped
            if not (hasattr(model_to_eval, 'hf_device_map') and model_to_eval.hf_device_map):
                 model_to_eval.to(eval_device)

            outputs = model_to_eval.generate(
                **inputs,
                max_new_tokens=MAX_GENERATION_LENGTH,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False
            )
            generated_ids = outputs[0][inputs.input_ids.shape[1]:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Error during generation for question '{question_text}': {e}")
            generated_text = "" 

        key_points = key_points_list[i]
        current_question_for_eval = chapter_questions[i]
        
        if llm_output_is_correct_on_tutor_eval(generated_text, key_points, current_question_for_eval):
            correct_predictions += 1
    
    accuracy = correct_predictions / len(questions) if questions else 0
    logger.info(f"Evaluation complete. Accuracy: {accuracy:.4f}")
    return accuracy

# --- Fine-tuning with SFTTrainer (No changes needed in logic) ---
def fine_tune_model_sft(
    model_to_fine_tune,
    tokenizer,
    chapter_full_text: str,
    chapter_name: str,
    epoch_num: int,
    model_args: ModelArguments,
    sft_config_overrides: dict,
    base_sft_output_dir: str,
    max_seq_length: int
):
    logger.info(f"Starting SFT fine-tuning for chapter '{chapter_name}', epoch {epoch_num}...")
    dataset_text_column = "text_for_sft" 
    dataset = Dataset.from_dict({dataset_text_column: [chapter_full_text]})
    sft_epoch_output_dir = os.path.join(base_sft_output_dir, f"sft_chapter_{chapter_name.replace(' ', '_').replace('/', '_')}_epoch_{epoch_num}") # Sanitize chapter name
    os.makedirs(sft_epoch_output_dir, exist_ok=True)

    sft_args = {
        "output_dir": sft_epoch_output_dir,
        "overwrite_output_dir": True,
        "num_train_epochs": 1,
        "max_seq_length": max_seq_length,
        "dataset_text_field": dataset_text_column,
        "packing": True,
        "report_to": "none",
        "save_strategy": "no",
        "logging_steps": 10,
    }
    sft_args.update(sft_config_overrides)
    if model_args.torch_dtype == "float16":
        sft_args["fp16"] = torch.cuda.is_available()
    elif model_args.torch_dtype == "bfloat16":
        sft_args["bf16"] = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    
    sft_training_config = SFTConfig(**sft_args)
    
    trainer = SFTTrainer(
        model=model_to_fine_tune, # This is the fresh model for the current chapter
        tokenizer=tokenizer,
        args=sft_training_config,
        train_dataset=dataset,
    )
    logger.info(f"Calling SFTTrainer.train() for chapter '{chapter_name}', epoch {epoch_num}. Output dir: {sft_epoch_output_dir}")
    try:
        trainer.train()
        logger.info(f"SFT fine-tuning for chapter '{chapter_name}', epoch {epoch_num} complete.")
    except Exception as e:
        logger.error(f"Error during SFTTrainer.train() for chapter '{chapter_name}', epoch {epoch_num}: {e}")
        raise # Re-raise to stop if a chapter fails, or handle more gracefully

# --- Main Experiment ---
def run_experiment():
    logger.info("Starting OLMo2-1B TutorEval Fine-tuning Experiment (INDEPENDENT Training per Chapter).")

    if torch.backends.mps.is_available():
        primary_device = torch.device("mps")
    elif torch.cuda.is_available():
        primary_device = torch.device("cuda")
    else:
        primary_device = torch.device("cpu")
    logger.info(f"Primary device selected: {primary_device}")

    model_args_config = ModelArguments(
        model_name_or_path=GLOBAL_MODEL_ID,
        torch_dtype="bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16",
        attn_implementation="flash_attention_2" if (torch.cuda.is_available() and primary_device.type == 'cuda') else None,
        use_peft=False, # Set to True to enable LoRA for each chapter independently
        # lora_target_modules=["Wqkv", "out_proj", "fc1", "fc2"], # VERIFY for OLMo-2
        quantization=None, # "4bit" or "8bit"
    )
    logger.info(f"Model Arguments: {model_args_config}")

    # Load tokenizer ONCE outside the loop
    # We need a temporary model_args for tokenizer loading if it's part of the same function.
    # Or, just load tokenizer separately. For simplicity with current `load_fresh_model_and_tokenizer`:
    _, tokenizer_global = load_fresh_model_and_tokenizer(model_args_config, primary_device)
    # Clean up the temporary model loaded just for the tokenizer if it was substantial
    # This is a bit of a hack; ideally, tokenizer loading is separate or very lightweight.
    # For now, we assume load_fresh_model_and_tokenizer is efficient enough or we accept the overhead.
    # Or, better:
    # tokenizer_global = AutoTokenizer.from_pretrained(model_args_config.model_name_or_path, trust_remote_code=True)
    # if tokenizer_global.pad_token is None:
    #    tokenizer_global.pad_token = tokenizer_global.eos_token
    logger.info("Global tokenizer loaded.")


    logger.info(f"Loading TutorEval dataset from {DATASET_ID}...")
    try:
        tutor_eval_dataset = load_dataset(DATASET_ID, split="train")
        df = tutor_eval_dataset.to_pandas()
    except Exception as e:
        logger.error(f"Failed to load or process TutorEval dataset: {e}")
        return
        
    grouped_by_chapter = df.groupby("chapter")
    all_results = {}

    sft_config_step_overrides = {
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 5e-5,
        "logging_strategy": "steps", # Added for clarity
        "logging_steps": 10, # Ensure this is part of the overrides
    }
    if model_args_config.use_peft or model_args_config.quantization:
        sft_config_step_overrides["per_device_train_batch_size"] = 2 
        sft_config_step_overrides["gradient_accumulation_steps"] = 4
    
    # Ensure SFTConfig gets the device from primary_device
    sft_config_step_overrides["device"] = primary_device # Pass the determined device to SFTConfig

    for chapter_idx, (chapter_full_text, chapter_data) in enumerate(grouped_by_chapter):
        chapter_name_for_log = f"Chapter{chapter_idx+1}_" + chapter_full_text[:30].replace("\n", " ").replace("/", "_").strip() + "..."
        logger.info(f"\n--- Processing Chapter: {chapter_name_for_log} (Independent Training) ---")

        # **MODIFICATION: Load a FRESH model for each chapter**
        logger.info(f"Loading FRESH model for chapter: {chapter_name_for_log}")
        model_for_this_chapter, _ = load_fresh_model_and_tokenizer(model_args_config, primary_device)
        # Tokenizer is already loaded globally (tokenizer_global)

        # **MODIFICATION: Apply PEFT to the FRESH model if enabled**
        if model_args_config.use_peft:
            logger.info(f"Applying PEFT (LoRA) to the fresh model for chapter: {chapter_name_for_log}")
            if not model_args_config.lora_target_modules:
                 model_args_config.lora_target_modules = ["Wqkv", "out_proj", "fc1", "fc2"]
                 logger.info(f"Using default LoRA target modules: {model_args_config.lora_target_modules}")
            
            peft_config = LoraConfig(
                r=model_args_config.lora_r,
                lora_alpha=model_args_config.lora_alpha,
                lora_dropout=model_args_config.lora_dropout,
                target_modules=model_args_config.lora_target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model_for_this_chapter = get_peft_model(model_for_this_chapter, peft_config)
            logger.info("PEFT model created for this chapter. Trainable parameters:")
            model_for_this_chapter.print_trainable_parameters()
        
        # Ensure the model (fresh or PEFT-wrapped fresh) is on the primary_device for evaluation and training
        # SFTTrainer will also handle device placement based on its args.device.
        # If not device_mapped by quantization, explicitly move.
        if not (hasattr(model_for_this_chapter, 'hf_device_map') and model_for_this_chapter.hf_device_map):
            model_for_this_chapter.to(primary_device)
        logger.info(f"Model for chapter {chapter_name_for_log} is on device: {next(model_for_this_chapter.parameters()).device}")


        questions_for_chapter = chapter_data["question"].tolist()
        key_points_for_chapter = chapter_data["key_points"].tolist()

        if not chapter_full_text or not questions_for_chapter:
            logger.warning(f"Skipping chapter '{chapter_name_for_log}' due to missing text or questions.")
            if model_args_config.use_peft or model_args_config.quantization:
                del model_for_this_chapter # Clean up to free VRAM
                torch.cuda.empty_cache() if primary_device.type == 'cuda' else None
            continue
            
        chapter_results_log = {"chapter_name": chapter_name_for_log, "epochs": []}

        # The "before fine-tuning" accuracy for each chapter will now always be on the pristine base model
        # (or pristine base model with fresh LoRA adapters if PEFT is used but adapters are not yet trained).
        # For PEFT, this means evaluating the base model's performance through the randomly initialized adapters.
        logger.info(f"Evaluating PRISTINE model state before any fine-tuning for chapter: {chapter_name_for_log}")
        accuracy_pristine_state = evaluate_model(
            model_for_this_chapter, # This is the fresh (potentially PEFT-wrapped) model
            tokenizer_global, 
            primary_device, 
            questions_for_chapter, 
            key_points_for_chapter, 
            questions_for_chapter
        )
        logger.info(f"Accuracy for '{chapter_name_for_log}' (Pristine Model State): {accuracy_pristine_state:.4f}")

        # Store the model state *after* the previous epoch's fine-tuning for this chapter
        accuracy_after_previous_epoch_ft = accuracy_pristine_state

        for epoch in range(MAX_EPOCHS_PER_CHAPTER):
            current_epoch_num = epoch + 1
            logger.info(f"--- Chapter: {chapter_name_for_log}, Epoch: {current_epoch_num}/{MAX_EPOCHS_PER_CHAPTER} ---")

            # The "accuracy_before_ft" for this epoch is the accuracy after the previous epoch's FT
            # (or pristine state if it's the first epoch for this chapter).
            # This was captured as accuracy_after_previous_epoch_ft.

            fine_tune_model_sft(
                model_for_this_chapter, # Pass the chapter-specific model
                tokenizer_global, 
                chapter_full_text,
                chapter_name_for_log, 
                current_epoch_num,
                model_args_config,
                sft_config_step_overrides,
                OUTPUT_DIR_BASE,
                CONTEXT_LENGTH
            )
            
            accuracy_after_current_epoch_ft = evaluate_model(
                model_for_this_chapter, # Evaluate the model fine-tuned on this chapter's text up to current epoch
                tokenizer_global, 
                primary_device, 
                questions_for_chapter, 
                key_points_for_chapter, 
                questions_for_chapter
            )
            logger.info(f"Accuracy for '{chapter_name_for_log}' (Epoch {current_epoch_num} - After FT): {accuracy_after_current_epoch_ft:.4f}")
            
            chapter_results_log["epochs"].append({
                "epoch_num": current_epoch_num,
                "accuracy_before_ft_this_epoch": accuracy_after_previous_epoch_ft, 
                "accuracy_after_ft_this_epoch": accuracy_after_current_epoch_ft,
            })
            
            accuracy_after_previous_epoch_ft = accuracy_after_current_epoch_ft # Update for the next iteration

        all_results[chapter_name_for_log] = chapter_results_log
        logger.info(f"Finished processing chapter: {chapter_name_for_log}")
        logger.info(f"Results for {chapter_name_for_log}: {chapter_results_log['epochs']}")

        # **MODIFICATION: Clean up the model for this chapter to free VRAM**
        # Especially important if loading large models repeatedly.
        del model_for_this_chapter
        if primary_device.type == 'cuda':
            torch.cuda.empty_cache()
        elif primary_device.type == 'mps':
            try:
                # For MPS, there isn't a direct empty_cache equivalent that's as impactful.
                # Re-importing torch and gc can sometimes help, but it's less certain.
                import gc
                gc.collect()
            except ImportError:
                pass
        logger.info(f"Cleaned up model for chapter: {chapter_name_for_log}")


    # --- Report Overall Results ---
    logger.info("\n\n--- Overall Experiment Results (Independent Training per Chapter) ---")
    # (Reporting logic remains the same)
    results_list_for_df = []
    for chapter_name_key, chapter_res_data in all_results.items():
        logger.info(f"Chapter: {chapter_res_data['chapter_name']}")
        for epoch_data in chapter_res_data["epochs"]:
            logger.info(
                f"  Epoch {epoch_data['epoch_num']}: "
                f"Accuracy Before FT this Epoch (or Pristine): {epoch_data['accuracy_before_ft_this_epoch']:.4f}, "
                f"Accuracy After FT this Epoch: {epoch_data['accuracy_after_ft_this_epoch']:.4f}"
            )
            results_list_for_df.append({
                "chapter": chapter_res_data['chapter_name'],
                "epoch": epoch_data["epoch_num"],
                "accuracy_before_ft_this_epoch": epoch_data['accuracy_before_ft_this_epoch'],
                "accuracy_after_ft_this_epoch": epoch_data['accuracy_after_ft_this_epoch']
            })
    
    if results_list_for_df:
        results_df = pd.DataFrame(results_list_for_df)
        results_path = os.path.join(OUTPUT_DIR_BASE, "experiment_results_sft_independent.csv")
        os.makedirs(OUTPUT_DIR_BASE, exist_ok=True)
        results_df.to_csv(results_path, index=False)
        logger.info(f"Detailed results saved to {results_path}")
    else:
        logger.info("No results to save.")


if __name__ == "__main__":
    # run_experiment() 
    logger.info("Python script for OLMo2-1B TutorEval SFT INDEPENDENT fine-tuning experiment generated.")
    logger.info("To run: uncomment 'run_experiment()', configure ModelArguments, ensure utils.query_llm.")