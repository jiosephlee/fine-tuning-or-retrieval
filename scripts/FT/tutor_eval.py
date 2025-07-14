import os
import torch
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from tqdm.auto import tqdm
import logging
import sys 
sys.path.append('..')
from utils.utils import query_llm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
MODEL_ID = "allenai/OLMo-2-0425-1B"
DATASET_ID = "princeton-nlp/TutorEval"
MAX_EPOCHS_PER_CHAPTER = 5  # Max epochs to fine-tune on each chapter's text
CONTEXT_LENGTH = 4096  # Context length for OLMo-2-1B
OUTPUT_DIR_BASE = "./olmo2_tutor_eval_finetune"
MAX_GENERATION_LENGTH = 256 # Max length for generated answers

# --- Placeholder for GPT evaluation ---
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
    result = query_llm(prompt)
    return result.strip().lower() == "true"

# --- Helper Functions ---
def load_model_and_tokenizer(model_id: str):
    logger.info(f"Loading model and tokenizer for {model_id}...")
    # Check for MPS availability for Mac M1/M2/M3 chips
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    logger.info(f"Using device: {device}")

    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    model.to(device)
    logger.info("Model and tokenizer loaded.")
    return model, tokenizer, device

def chunk_text(text: str, tokenizer: AutoTokenizer, chunk_size: int = CONTEXT_LENGTH):
    logger.debug(f"Chunking text of length {len(text)} characters.")
    tokens = tokenizer(text, truncation=False, return_attention_mask=False)["input_ids"]
    chunks = []
    for i in range(0, len(tokens), chunk_size):
        chunk = tokens[i:i + chunk_size]
        chunks.append(tokenizer.decode(chunk))
    logger.debug(f"Text chunked into {len(chunks)} chunks.")
    return chunks

def evaluate_model(model, tokenizer, device, questions: list[str], key_points_list: list[str]):
    logger.info(f"Evaluating model on {len(questions)} questions...")
    model.eval()
    correct_predictions = 0
    
    # It's good practice to disable gradients during evaluation
    with torch.no_grad():
        for i, question in tqdm(enumerate(questions), total=len(questions), desc="Evaluating Questions"):
            inputs = tokenizer(question, return_tensors="pt", truncation=True, max_length=CONTEXT_LENGTH - MAX_GENERATION_LENGTH).to(device) # Reserve space for generation
            
            # Generate answer
            # Using try-except for potential generation errors
            try:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_GENERATION_LENGTH,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=False # For deterministic output during eval
                )
                generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            except Exception as e:
                logger.error(f"Error during generation for question '{question}': {e}")
                generated_text = "" # Assign empty string if generation fails

            key_points = key_points_list[i]
            
            if llm_output_is_correct_on_tutor_eval( generated_text, key_points, question):
                correct_predictions += 1
    
    accuracy = correct_predictions / len(questions) if questions else 0
    logger.info(f"Evaluation complete. Accuracy: {accuracy:.4f}")
    return accuracy

def fine_tune_model(model, tokenizer, chapter_text_chunks: list[str], chapter_name: str, epoch: int):
    logger.info(f"Starting fine-tuning for chapter '{chapter_name}', epoch {epoch + 1}...")
    
    # Prepare dataset for fine-tuning
    # Each chunk is a separate "document"
    dataset = Dataset.from_pandas(pd.DataFrame({"text": chapter_text_chunks}))

    def tokenize_function(examples):
        # Tokenize texts and prepare them for the model.
        # We are doing Causal LM, so we don't need explicit labels if using DataCollatorForLanguageModeling.
        # The model will learn to predict the next token.
        tokenized_output = tokenizer(examples["text"], truncation=True, max_length=CONTEXT_LENGTH, padding="max_length")
        tokenized_output["labels"] = tokenized_output["input_ids"].copy() # For Causal LM, labels are the same as input_ids
        return tokenized_output

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Define unique output directory for this chapter and epoch to avoid conflicts
    # and allow resuming or inspecting specific checkpoints if needed.
    # However, for this experiment, we re-initialize or continue training on the same model instance.
    # So, the output_dir for Trainer might be more for logs/checkpoints of that specific run.
    # We will save the model explicitly after each epoch.
    training_output_dir = os.path.join(OUTPUT_DIR_BASE, f"chapter_{chapter_name.replace(' ', '_')}_epoch_{epoch+1}")
    os.makedirs(training_output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=training_output_dir,
        overwrite_output_dir=True,
        num_train_epochs=1,  # We control epochs externally, so 1 epoch per call
        per_device_train_batch_size=1,  # Adjust based on GPU memory
        gradient_accumulation_steps=4, # Adjust based on GPU memory
        save_steps=999999, # Don't save checkpoints automatically by Trainer, we save manually
        save_total_limit=1,
        logging_steps=50,
        learning_rate=5e-5, # A common starting learning rate
        weight_decay=0.01,
        fp16=torch.cuda.is_available(), # Use mixed precision if CUDA is available
        # mps_bf16=torch.backends.mps.is_available(), # For MPS, bf16 might be better if supported
        report_to="none", # Disable wandb/tensorboard for simplicity here
        remove_unused_columns=False, # Important if dataset has extra columns
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    model.train() # Ensure model is in training mode
    trainer.train()
    logger.info(f"Fine-tuning for chapter '{chapter_name}', epoch {epoch + 1} complete.")
    # The model object is updated in-place by the Trainer.

# --- Main Experiment ---
def run_experiment():
    logger.info("Starting OLMo2-1B TutorEval Fine-tuning Experiment.")

    # 1. Load Model and Tokenizer
    # We load the base model once at the beginning.
    # For each chapter, we will fine-tune this model sequentially.
    # If you want to start fresh for each chapter, move this inside the chapter loop.
    # However, the RQ implies continuous learning.
    base_model, tokenizer, device = load_model_and_tokenizer(MODEL_ID)

    # 2. Load TutorEval Dataset
    logger.info(f"Loading TutorEval dataset from {DATASET_ID}...")
    tutor_eval_dataset = load_dataset(DATASET_ID, split="train")
    
    # Group by chapter
    # Convert to pandas for easier grouping if not already
    try:
        df = tutor_eval_dataset.to_pandas()
    except AttributeError: # If it's already a pandas DataFrame or similar
        df = pd.DataFrame(tutor_eval_dataset)
        
    grouped_by_chapter = df.groupby("chapter")
    
    all_results = {}

    # Iterate through each chapter
    for chapter_name, chapter_data in grouped_by_chapter:
        logger.info(f"\n--- Processing Chapter: {chapter_name} ---")
        
        # Ensure chapter_text is a single string
        # Some chapters might be split into multiple rows if the dataset structure is complex.
        # Assuming 'chapter' column in TutorEval contains the actual text content of the chapter.
        # If 'chapter' is just an identifier and text is elsewhere (e.g. from 'path_to_chapter'),
        # this needs adjustment. Based on TutorEval viewer, 'chapter' seems to be the text.
        if chapter_data["chapter"].nunique() > 1:
            logger.warning(f"Chapter '{chapter_name}' has multiple unique text entries. Concatenating them.")
            # This might happen if 'chapter_name' is a title and 'chapter' column has parts.
            # For TutorEval, 'chapter' column IS the text, and 'path_to_chapter' might be a filename.
            # The groupby('chapter') should group by the actual text content if 'chapter' is the text.
            # If 'chapter' is a name/ID, and text is in another column, group by that ID and concat text.
            # Let's assume chapter_data['chapter'].iloc[0] is the full text for that group.
            # If not, and multiple rows form one chapter, they need to be concatenated.
            # For TutorEval, each row seems to be a Q&A pair with its associated chapter text.
            # So, the chapter text might be repeated. We take the first instance.
            chapter_full_text = chapter_data["chapter"].iloc[0]
        else:
            chapter_full_text = chapter_data["chapter"].iloc[0]

        questions = chapter_data["question"].tolist()
        key_points_list = chapter_data["key_points"].tolist()

        if not chapter_full_text or not questions:
            logger.warning(f"Skipping chapter '{chapter_name}' due to missing text or questions.")
            continue

        # Chunk chapter text
        chapter_text_chunks = chunk_text(chapter_full_text, tokenizer)
        if not chapter_text_chunks:
            logger.warning(f"Skipping chapter '{chapter_name}' as text chunking resulted in no chunks.")
            continue
            
        chapter_results = {"chapter_name": chapter_name, "epochs": []}

        # Create a fresh copy of the base model for each chapter if isolated training is desired
        # Or, continue fine-tuning the same model instance if cumulative learning is desired.
        # The problem description implies cumulative learning on the *same* model instance.
        # If you want isolated training per chapter:
        # current_model, _, _ = load_model_and_tokenizer(MODEL_ID) # Reloads base model
        # For cumulative, we use 'base_model' which gets updated.
        current_model = base_model 

        for epoch in range(MAX_EPOCHS_PER_CHAPTER):
            logger.info(f"--- Chapter: {chapter_name}, Epoch: {epoch + 1}/{MAX_EPOCHS_PER_CHAPTER} ---")

            # a) Evaluate LLM (before fine-tuning for this epoch, or initial state for epoch 0)
            # For the very first epoch (epoch 0), this evaluates the model *before* any fine-tuning on this chapter.
            # For subsequent epochs, it evaluates the model fine-tuned up to epoch-1 on this chapter.
            accuracy_before_ft = evaluate_model(current_model, tokenizer, device, questions, key_points_list)
            logger.info(f"Accuracy for '{chapter_name}' (Epoch {epoch + 1} - Before FT): {accuracy_before_ft:.4f}")

            # b) Fine-tune an epoch
            fine_tune_model(current_model, tokenizer, chapter_text_chunks, chapter_name, epoch)
            
            # c) Evaluate LLM (after fine-tuning for this epoch)
            accuracy_after_ft = evaluate_model(current_model, tokenizer, device, questions, key_points_list)
            logger.info(f"Accuracy for '{chapter_name}' (Epoch {epoch + 1} - After FT): {accuracy_after_ft:.4f}")
            
            chapter_results["epochs"].append({
                "epoch_num": epoch + 1,
                "accuracy_before_ft_on_chapter": accuracy_before_ft if epoch == 0 else chapter_results["epochs"][-1]["accuracy_after_ft_on_chapter"], # Use previous after_ft for subsequent epochs
                "accuracy_after_ft_on_chapter": accuracy_after_ft,
            })
            
            # Optional: Save model checkpoint after each epoch of fine-tuning on a chapter
            # checkpoint_dir = os.path.join(OUTPUT_DIR_BASE, f"model_chapter_{chapter_name.replace(' ', '_')}_epoch_{epoch+1}")
            # current_model.save_pretrained(checkpoint_dir)
            # tokenizer.save_pretrained(checkpoint_dir)
            # logger.info(f"Model saved to {checkpoint_dir}")

        all_results[chapter_name] = chapter_results
        logger.info(f"Finished processing chapter: {chapter_name}")
        logger.info(f"Results for {chapter_name}: {chapter_results}")

    # --- Report Results ---
    logger.info("\n\n--- Overall Experiment Results ---")
    for chapter_name, results in all_results.items():
        logger.info(f"Chapter: {results['chapter_name']}")
        for epoch_data in results["epochs"]:
            logger.info(
                f"  Epoch {epoch_data['epoch_num']}: "
                f"Accuracy Before FT on this chapter: {epoch_data['accuracy_before_ft_on_chapter']:.4f}, "
                f"Accuracy After FT on this chapter: {epoch_data['accuracy_after_ft_on_chapter']:.4f}"
            )
    
    # Save results to a file
    results_df = pd.DataFrame()
    for chapter_name, data in all_results.items():
        for epoch_res in data['epochs']:
            row = {
                "chapter": chapter_name,
                "epoch": epoch_res["epoch_num"],
                "accuracy_before_ft": epoch_res["accuracy_before_ft_on_chapter"],
                "accuracy_after_ft": epoch_res["accuracy_after_ft_on_chapter"]
            }
            results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)
    
    results_path = os.path.join(OUTPUT_DIR_BASE, "experiment_results.csv")
    os.makedirs(OUTPUT_DIR_BASE, exist_ok=True)
    results_df.to_csv(results_path, index=False)
    logger.info(f"Detailed results saved to {results_path}")

if __name__ == "__main__":
    # Note: This script requires significant computational resources (GPU with enough VRAM)
    # and time to run. Ensure your environment is set up with PyTorch, Transformers, Datasets, etc.
    
    # Example of how to run (actual execution is not done here):
    # Create a virtual environment, install dependencies:
    # pip install torch transformers datasets pandas tqdm scikit-learn accelerate
    # Then run: python your_script_name.py
    
    # For demonstration, we will just print that the script would run.
    # To actually run, you would call run_experiment()
    # run_experiment() 
    
    print("Python script for OLMo2-1B TutorEval fine-tuning experiment generated.")
    print("To run the experiment, uncomment 'run_experiment()' and execute the script.")
    print("Ensure you have a suitable environment with necessary libraries and hardware (GPU recommended).")