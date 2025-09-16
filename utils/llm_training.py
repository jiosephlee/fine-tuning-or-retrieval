import math
import os
import wandb
import torch
from typing import Optional, List
from datasets import Dataset
from transformers import (
    TrainerCallback,
)
from trl import SFTTrainer

from utils.llm_configs import TrainingConfig
from utils.chunking import chunk_text, chunk_text_by_sections, chunk_texts
from utils.data_preparation import prepare_training_mix

# --------------------------------------------------------------------------
# SECTION 2: CORE LLM OPERATIONS
# --------------------------------------------------------------------------

# **IMPORTANT** Custom trainer to use 'sum' loss, a best practice for chat models.
# No longer does this but we leave it here in case we want to go back to it. This is functionally just a SFTTrainer
class CustomSFTTrainer(SFTTrainer):
    def __init__(self, use_liger_loss, *args, **kwargs):
        super().__init__(*args, **kwargs) # Pass all remaining args/kwargs to parent
        self.use_liger_loss = use_liger_loss
    
def fine_tune_on_text(
    model, tokenizer, log, text_content: str, train_cfg: TrainingConfig, *, train=True, tag: str = "finetuning on text...", callbacks: Optional[List[TrainerCallback]] = None, chunk_by_section: bool = False, overlap_sections = False, overlap_ratio = "1_4", add_title_prefix: bool = True
):
    """
    Fine-tunes a model on a given string of text by chunking it properly.
    
    Args:
        model: The model to fine-tune
        tokenizer: The tokenizer
        text_content: The text to fine-tune on
        train_cfg: Training configuration
        tag: Tag for logging
        callbacks: Optional list of TrainerCallbacks to add to the trainer.
        chunk_by_section: If True, use section-based chunking instead of token-based chunking
    """
    log.info(f"Starting SFT for '{tag}'...")
    
    text_content = text_content + tokenizer.eos_token
    
    if chunk_by_section:
        log.info(f"[{tag}] Using section-based chunking...")
        overlap_numer, overlap_denom = overlap_ratio.split("_")
        overlap_denom = int(overlap_denom)
        overlap_numer = int(overlap_numer)
        text_chunks, num_tokens = chunk_text_by_sections(text_content, tokenizer, train_cfg.context_length, overlap_sections, overlap_denom, overlap_numer, log=log, add_title_prefix=add_title_prefix)
        log.info(f"[{tag}] Section-based chunking: Total tokens: {num_tokens}, Overlapping: {overlap_sections}, Prefix added: {add_title_prefix}, Context: {train_cfg.context_length} -> {len(text_chunks)} total chunks")
    else:
        log.info(f"[{tag}] Using token-based chunking...")
        text_chunks, num_tokens = chunk_text(text_content, tokenizer, train_cfg.context_length)
        log.info(f"[{tag}] Token-based chunking: Tokens: {num_tokens}, Context: {train_cfg.context_length} -> {len(text_chunks)} chunks")
    
    dataset = Dataset.from_dict({"text": text_chunks})
    log.info(f"[{tag}] Created dataset with {len(text_chunks)} chunks (including the eos token)")
    
    assert(len(text_chunks) <= (train_cfg.per_device_train_batch_size * train_cfg.gradient_accumulation_steps))
    log.info(f"[{tag}] Gradient_accumulation_steps ({train_cfg.gradient_accumulation_steps}) | {len(text_chunks)} chunks")
    
    training_args = train_cfg.to_sft_training_args() 
    
    trainer = CustomSFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        processing_class=tokenizer,
        use_liger_loss=True,
        callbacks=callbacks
    )
    
    if train:
        trainer.train()
        log.info(f"SFT complete for '{tag}'.")
        wandb.finish()
    return trainer

def fine_tune_on_texts(
    model, tokenizer, log, texts: List[str], train_cfg: TrainingConfig, *, train=True, tag: str = "finetuning on texts...", callbacks: Optional[List[TrainerCallback]] = None, chunk_by_section: bool = False, overlap_sections = False, overlap_ratio = "1_4", add_title_prefix: bool = True
):
    """
    Fine-tunes a model on a given list of texts by chunking them and training on all chunks together.
    
    Args:
        model: The model to fine-tune.
        tokenizer: The tokenizer.
        texts: The list of text strings to fine-tune on.
        train_cfg: Training configuration.
        tag: Tag for logging.
        callbacks: Optional list of TrainerCallbacks to add to the trainer.
        chunk_by_section: If True, use section-based chunking instead of token-based chunking.
        add_title_prefix: If True, prefixes each new chunk with the document's title.
    """
    log.info(f"Starting SFT for '{tag}' on {len(texts)} documents...")

    # Add EOS token to each text
    for i in range(len(texts)):
        texts[i] = texts[i] + tokenizer.eos_token 
    
    # Chunk the texts into smaller pieces based on context length
    if chunk_by_section:
        log.info(f"[{tag}] Using section-based chunking...")
        all_text_chunks = []
        total_tokens = 0
        for i,text in enumerate(texts):
            if "\\section" in text.lower():
                overlap_numer, overlap_denom = overlap_ratio.split("_")
                overlap_denom = int(overlap_denom)
                overlap_numer = int(overlap_numer)
                chunks, num_tokens = chunk_text_by_sections(text, tokenizer, train_cfg.context_length, overlap_sections, overlap_denom, overlap_numer, log=log, add_title_prefix=add_title_prefix)
                log.info(f"[{tag}] Section-based chunking: Total tokens: {total_tokens}, Overlapping: {overlap_sections}, Title added: {add_title_prefix}, Context: {train_cfg.context_length} -> {len(chunks)} total chunks")
            else:
                chunks, num_tokens = chunk_text(text, tokenizer, train_cfg.context_length, delimiter="\n\n")
                log.info(f"[{tag}] Chunking text {i}: Context: {train_cfg.context_length} -> {len(chunks)} chunks")
            all_text_chunks.extend(chunks)
            total_tokens += num_tokens
            
        log.info(f"[{tag}] Total tokens: {total_tokens}, Context: {train_cfg.context_length} -> {len(all_text_chunks)} total chunks")

    else:
        log.info(f"[{tag}] Using token-based chunking...")
        all_text_chunks, total_tokens = chunk_texts(texts, tokenizer, train_cfg.context_length)
        log.info(f"[{tag}] Token-based chunking: Total tokens: {total_tokens}, Context: {train_cfg.context_length} -> {len(all_text_chunks)} total chunks")

    # Create dataset with all chunked texts
    dataset = Dataset.from_dict({"text": all_text_chunks})
    
    assert(len(all_text_chunks) % train_cfg.gradient_accumulation_steps * train_cfg.per_device_train_batch_size == 0)
    log.info(f"[{tag}] Gradient_accumulation_steps ({train_cfg.gradient_accumulation_steps}) | {len(all_text_chunks)} chunks")

    training_args = train_cfg.to_sft_training_args() # Packing is False to avoid document re-ordering and padding free is false to avoid any unexpected bugs

    trainer = CustomSFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        processing_class=tokenizer,
        use_liger_loss=True,
        callbacks=callbacks
    )
    if train:
        trainer.train()
        log.info(f"SFT complete for '{tag}'.")
        wandb.finish()
    return trainer

def sft_train_on_dataset(
    model,  tokenizer, log, train_dataset: Dataset, train_cfg: TrainingConfig, train=True, use_liger_loss =False, callbacks: Optional[List[TrainerCallback]] = None
):
    """
    A generalized function to run SFT on a prepared dataset. Effective batch size is batch_size (2) * gradient_accumulation_steps (16) = 32 as per LIMA
    """
    log.info("Starting SFT training run...")
    training_args = train_cfg.to_sft_training_args()

    trainer = CustomSFTTrainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        processing_class=tokenizer,
        use_liger_loss = use_liger_loss,
        callbacks=callbacks
    )

    if train:
        trainer.train()
        log.info("SFT training complete.")
        wandb.finish()
    return trainer

def save_model(model, tokenizer, log, save_path: str):
    """
    Saves the model and tokenizer. 
    """ # If LoRA was used, it merges the adapters into the base model for easy deployment.
    os.makedirs(save_path, exist_ok=True)
    # if hasattr(model, "merge_and_unload"):
    #     log.info("Merging LoRA adapters and saving full model...")
    #     model = model.merge_and_unload()
    # else:
    log.info("Saving full model...")

    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    log.info(f"Model saved to {save_path}")

"""Used in v8 and onwards"""

def fine_tune(
    model,
    tokenizer,
    log,
    train_cfg: TrainingConfig,
    strategy_name: str,
    strategy_args: dict,
    output_dir_for_debug: str,
    callbacks: Optional[List[TrainerCallback]] = None,
    train: bool = True,
    **chunking_args,
):
    """
    A universal fine-tuning function that prepares data based on a strategy and runs the training.
    Includes a debugging mechanism to verify sequential data loading.
    """
    test_script = strategy_args.get("test_script", False)
    dataset, updated_train_cfg = prepare_training_mix(
        strategy_name=strategy_name,
        tokenizer=tokenizer,
        log=log,
        train_cfg=train_cfg,
        **strategy_args,
        **chunking_args,
    )
    
    for i in dataset:
        print(i['raw_text'][:50])
    
    if test_script:
        log.info(f"Dataset has {len(dataset)} chunks...")
    training_args = updated_train_cfg.to_sft_training_args()
    
    trainer = CustomSFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        processing_class=tokenizer,
        use_liger_loss=True, # Assuming this is standard
        callbacks=callbacks,
    )
    
    # --- Debugging Sequential Sampling ---
    os.makedirs(output_dir_for_debug, exist_ok=True)
    
    def get_dataloader_content(dataloader, debug=False):
        content = []
        eos_token_id = tokenizer.eos_token_id
        for i, batch in enumerate(dataloader):
            # Assuming 'input_ids' is the key for tokenized text
            # and we decode it back to string for inspection.
            # Process each sequence in the batch
            for j, (input_ids, attention_mask) in enumerate(zip(batch['input_ids'], batch['attention_mask'])):
                # Find the last non-padded token using attention mask
                last_valid_idx = (attention_mask == 1).nonzero(as_tuple=True)[0][-1]
                last_token = input_ids[last_valid_idx]
                
                # Verify that the last token of the sequence is NOT an EOS token
                if last_token == eos_token_id and debug:
                    #log.warning(f"CPT batch {i}, sequence {j} ends with an EOS token, which is expected for the last batch of the unique document.")
                    pass
                first_50 = tokenizer.decode(input_ids[:50])
                last_50 = tokenizer.decode(input_ids[-50:])
                text_sample = first_50+ "..." + last_50
                content.append(text_sample)
        return content

    log.info("Running first dataloader pass for debugging and verification...")
    dataloader1 = trainer.get_train_dataloader()
    log.info(f"Dataloader has a batch size of {len(dataloader1)} chunks...")
    content1 = get_dataloader_content(dataloader1, debug=True)
    with open(os.path.join(output_dir_for_debug, "debug_run_1.txt"), "w") as f:
        f.write("\n------\n".join(content1))
        
    log.info("Running second dataloader pass for debugging...")
    dataloader2 = trainer.get_train_dataloader()
    content2 = get_dataloader_content(dataloader2)
    with open(os.path.join(output_dir_for_debug, "debug_run_2.txt"), "w") as f:
        f.write("\n------\n".join(content2))

    assert content1 == content2, "Sequential sampling is not deterministic!"
    log.info("Sequential sampling verification successful.")
    
    if train:
        trainer.train()
        log.info("Fine-tuning complete.")
        if train_cfg.report_to == "wandb":
            wandb.finish()
        
    return trainer