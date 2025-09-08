import math
import os
import wandb
import torch
from typing import Optional, List
from datasets import Dataset, load_dataset
from peft import LoraConfig as PeftLoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
#from liger_kernel.transformers import LigerCrossEntropyLoss
from trl import SFTTrainer
from utils.llm_configs import PeftConfig, ModelConfig, TrainingConfig, InferenceConfig
from utils.chunking import chunk, chunk_text, chunk_text_by_sections, chunk_texts

def get_pretraining_data(num_batches: int, type: str = 'wiki') -> List[List[str]]:
    """
    Returns N batches of pretraining data.
    NOTE: This is a placeholder and will be replaced with a proper implementation.
    """
    raise NotImplementedError("get_pretraining_data is not yet implemented.")

def fill_up_batch_with_pretraining_data(batch: List[str], type: str = 'wiki') -> List[str]:
    """
    Fills up a batch with pretraining data to the desired size.
    NOTE: This is a placeholder and will be replaced with a proper implementation.
    """
    raise NotImplementedError("fill_up_batch_with_pretraining_data is not yet implemented.")


# --------------------------------------------------------------------------
# SECTION 2: CORE LLM OPERATIONS
# --------------------------------------------------------------------------

def create_peft_model_for_training(model, log, config: PeftConfig):

    model = prepare_model_for_kbit_training(model)
    log.info("Applying PEFT (LoRA)...")
    # Prepare modules_to_save for instruction tuning
    modules_to_save = ["lm_head", "embed_tokens"] if config.add_eot_token else None
                
    peft_config = PeftLoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=modules_to_save,  # Add this line
    )
    model = get_peft_model(model, peft_config)
    log.info("LoRA applied. Trainable parameters:")
    model.print_trainable_parameters()

    log.info("PEFT Model Created successfully.")
    return model
    
def load_model_for_training(config: ModelConfig, log, use_cpu_and_gpu = False, add_special_token = None, use_existing_lima_tokenizer=False, use_existing_lima_model = False):
    """
    Loads a model and tokenizer for training, applying quantization and PEFT.
    **ENHANCED** with robust QLoRA setup from open-instruct.
    """
    log.info(f"Loading model '{config.id}' for training...")

    # Determine torch dtype
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    quant_config = None
    if config.quantization.mode == "4bit":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype, # Use bfloat16 for compute
            bnb_4bit_use_double_quant=True,
        )
    elif config.quantization.mode == "8bit":
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        if quant_config is None:
            model = AutoModelForCausalLM.from_pretrained(
                config.id,
                trust_remote_code=True,
                torch_dtype=dtype,
                device_map='auto' if use_cpu_and_gpu else "cuda",
                attn_implementation=config.attn_implementation,
            )
        else:
            print("...Quantizing...")
            model = AutoModelForCausalLM.from_pretrained(
            config.id,
            trust_remote_code=True,
            torch_dtype=dtype,
            quantization_config=quant_config,
            device_map='auto' if use_cpu_and_gpu else "cuda", #Assume we're operating in a low VRAM environment since we're quantizing
            attn_implementation=config.attn_implementation,
        )
        if use_existing_lima_tokenizer:
            tokenizer = AutoTokenizer.from_pretrained("jiosephlee/olmo2-lima", trust_remote_code=True)
            model.resize_token_embeddings(len(tokenizer))
        else:
            tokenizer = AutoTokenizer.from_pretrained(config.id, trust_remote_code=True, use_fast=True)
            # Add special tokens before doing PEFT
            if add_special_token is not None:
                log.info(f"Adding special token: {add_special_token}")
                special_tokens_dict = {'additional_special_tokens': [add_special_token]}
                tokenizer.add_special_tokens(special_tokens_dict)  
                model.resize_token_embeddings(len(tokenizer))
            
        # Crucial step for preparing a quantized model for PEFT training.
        if config.quantization.mode:
            model = prepare_model_for_kbit_training(model)

        if config.peft.enabled:
            if use_existing_lima_model:
                model.load_adapter("jiosephlee/olmo2-lima", adapter_name="lima")
            else:
                log.info("Applying PEFT (LoRA)...")
                # Prepare modules_to_save for instruction tuning
                modules_to_save = ["lm_head", "embed_tokens"] if config.peft.add_eot_token else None
                
                peft_config = PeftLoraConfig(
                    r=config.peft.lora_r,
                    lora_alpha=config.peft.lora_alpha,
                    lora_dropout=config.peft.lora_dropout,
                    target_modules=config.peft.target_modules,
                    bias="none",
                    task_type="CAUSAL_LM",
                    modules_to_save=modules_to_save,  # Add this line
                )
                model = get_peft_model(model, peft_config)
                log.info("LoRA applied. Trainable parameters:")
                model.print_trainable_parameters()

    log.info("Model and tokenizer loaded successfully.")
    return model, tokenizer

def prepare_lima_dataset(tokenizer: AutoTokenizer, log, use_eot_token=False, sort=False):
    """
    Loads the GAIR/lima dataset, and formats
    the conversations into a text format suitable for SFTTrainer.

    Args:
        tokenizer: The tokenizer to modify.
        model: The model to resize embeddings for.

    Returns:
        A tuple of (train_dataset, eval_dataset).
    """
    log.info("Preparing GAIR/lima dataset...")
    EOT_TOKEN = "<|EOT|>"
    if not use_eot_token:
        EOT_TOKEN = "\nResponse:"
    # 2. Load the dataset
    dataset = load_dataset("GAIR/lima")
    # The paper uses 1000 for training, 50 for dev. The HF dataset has 1030 train examples.
    # We'll split it accordingly.
    train_dataset = dataset["train"].shuffle(seed=42)
    # train_dataset = full_train_dataset
    log.info(f"{len(train_dataset)} training examples.")

    # 3. Define the formatting function
    def format_lima_conversation(example):
        conversation = example['conversations']
        # Join turns with the EOT token. Add one at the very end.
        formatted_text = f"{EOT_TOKEN}".join(conversation) + tokenizer.eos_token
        return {"text": formatted_text}

        
    # 4. Apply the formatting
    train_dataset = train_dataset.map(format_lima_conversation, remove_columns=['conversations', 'source'])

    if sort:
        # add a temporary 'length' column
        train_dataset = train_dataset.map(
            lambda x: {"_len": len(x["text"])},
            desc="Computing lengths for sort",
        )
        # Dataset.sort always sorts ascending, so we reverse afterwards
        train_dataset = (
            train_dataset
            .sort("_len")                                      # shortest → longest
            .select(list(range(len(train_dataset) - 1, -1, -1)))  # flip order
            .remove_columns("_len")                            # clean-up
        )
        log.info("Training set sorted by descending length.")
        
    return train_dataset

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
            if "section" in text.lower():
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


def prepare_training_mix(
    strategy_name: str,
    tokenizer,
    log,
    train_cfg: TrainingConfig,
    chunk_by_section: bool = False,
    overlap_sections: bool = False,
    overlap_ratio: str = "1_4",
    add_title_prefix: bool = True,
    **strategy_args,
):
    """
    Prepares a training dataset based on a specified strategy.
    Handles complex mixing of source, paraphrased, and explanation texts
    while controlling for the total number of training steps. This function
    is domain-agnostic and handles single or multiple domains naturally.
    """
    log.info(f"Preparing training mix for strategy: {strategy_name}")
    
    # Helper to chunk a single text
    def _chunk(text: str) -> List[str]:
        text_with_eos = text + tokenizer.eos_token
        chunks, _ = chunk(
            text_with_eos,
            tokenizer,
            train_cfg.context_length,
            chunk_by_section,
            overlap_sections,
            overlap_ratio,
            add_title_prefix,
            log=log
        )
        return chunks

    # Determine domains: use override if provided, otherwise scan directory
    domains = strategy_args.get("override_domains", None)
    if domains is None:
        cleaned_dir = 'data/arxiv/cleaned'
        domains = [f.replace('.txt', '') for f in os.listdir(cleaned_dir) if f.endswith('.txt')]
    log.info(f"Processing domains: {domains}")

    num_paraphrased_texts = strategy_args.get("num_paraphrased_texts", 0)
    with_explanations = "WithExplanations" in strategy_name

    # Each inner list holds chunks for a "unique document type" across all domains
    # e.g., unique_document_batches[0] = all source chunks
    #       unique_document_batches[1] = all paraphrase-1 chunks
    unique_document_batches = [[] for _ in range(1 + num_paraphrased_texts)]

    for domain in domains:
        log.info(f"Loading data for domain: {domain}")
        
        # 1. Load source text
        source_path = f'data/arxiv/cleaned/{domain}.txt'
        try:
            with open(source_path, 'r', encoding='utf-8') as f:
                source_text = f.read()
        except FileNotFoundError:
            log.error(f"Source file not found for domain {domain} at {source_path}. Skipping.")
            continue
            
        source_chunks = _chunk(source_text)
        

        # 2. Load paraphrased texts
        paraphrased_texts = []
        paraphrased_chunks_by_doc = []
        if num_paraphrased_texts > 0:
            paraphrased_dir = f'data/arxiv/paraphrased/{domain}/'
            if os.path.isdir(paraphrased_dir):
                for i in range(num_paraphrased_texts):
                    para_path = os.path.join(paraphrased_dir, f'{i}.txt')
                    if os.path.exists(para_path):
                        with open(para_path, 'r', encoding='utf-8') as f:
                            paraphrased_texts.append(f.read())
                    else:
                        log.warning(f"Paraphrased text not found: {para_path}")
            paraphrased_chunks_by_doc = [_chunk(text) for text in paraphrased_texts]

        # 3. Load explanation texts
        explanation_chunks = []
        if with_explanations:
            explanation_dir = f'data/arxiv/explanations/{domain}/'
            if os.path.isdir(explanation_dir):
                for filename in sorted(os.listdir(explanation_dir)):
                    if filename.endswith('.txt'):
                        file_path = os.path.join(explanation_dir, filename)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            explanation_chunks.extend(_chunk(f.read()))
            log.info(f"Domain {domain}: Found {len(explanation_chunks)} explanation chunks.")
        
        # This part ensures that chunks from each document type of a domain are added to the correct overall batch
        
        # Document Chunks for the current domain
        domain_doc_chunks = [source_chunks] + paraphrased_chunks_by_doc

        # 4. Handle explanation replacement logic
        num_chunks_per_source_doc = len(source_chunks)
        if with_explanations and explanation_chunks:
            paraphrased_units_for_explanations = math.ceil(len(explanation_chunks) / num_chunks_per_source_doc)
            
            # Start replacing from the last paraphrased doc towards the first
            if paraphrased_units_for_explanations > 0:
                # Distribute explanation chunks among the slots of the documents they replace
                num_to_replace = min(paraphrased_units_for_explanations, len(paraphrased_chunks_by_doc))
                
                # Split explanations into `num_to_replace` parts
                split_explanations = [explanation_chunks[i::num_to_replace] for i in range(num_to_replace)]

                for i in range(num_to_replace):
                    # Index of paraphrased doc to replace (from the end)
                    replace_idx = len(paraphrased_chunks_by_doc) - 1 - i
                    # Corresponding index in domain_doc_chunks (source is at 0)
                    doc_chunk_idx = replace_idx + 1
                    
                    domain_doc_chunks[doc_chunk_idx] = split_explanations[i]
                log.info(f"Domain {domain}: Replaced {num_to_replace} paraphrased documents with explanation chunks.")
        
        # Add the processed chunks for this domain to the main batches
        for i, chunks in enumerate(domain_doc_chunks):
            if i < len(unique_document_batches):
                unique_document_batches[i].extend(chunks)

    # 5. Assemble final chunk list with optional pretraining data replay
    final_chunks = []
    pretraining_separators = strategy_args.get("pretraining_batches_separating_docs", 0)

    # Assert that batches expected to have content are not empty
    for i, batch in enumerate(unique_document_batches):
        assert batch, f"Batch for unique document type {i} is empty. Check data and strategy arguments."

    for i, batch in enumerate(unique_document_batches):
        if strategy_args.get("fill_batches_with_pretraining", False):
            batch = fill_up_batch_with_pretraining_data(batch, type=strategy_args.get('pretraining_data_type', 'wiki'))
        
        effective_batch_size = train_cfg.per_device_train_batch_size * train_cfg.gradient_accumulation_steps
        assert len(batch) == effective_batch_size, \
            f"Batch {i} has size {len(batch)}, which is not equal to the effective batch size {effective_batch_size}."

        final_chunks.extend(batch)
        
        if i < len(unique_document_batches) - 1 and pretraining_separators > 0:
            pretraining_fill = get_pretraining_data(pretraining_separators, type=strategy_args.get('pretraining_data_type', 'wiki'))
            flat_fill = [item for sublist in pretraining_fill for item in sublist] # Flatten the list of lists
            final_chunks.extend(flat_fill)
    
    # 6. Duplicate the dataset to match the desired number of training steps
    original_epochs = train_cfg.num_train_epochs
    chunks_in_mix = len(final_chunks)
    
    if chunks_in_mix > 0:
        # We assume 1 epoch over the constructed mix. To simulate more epochs, we replicate data.
        if original_epochs > 1:
            replication_factor = original_epochs / len(unique_document_batches)
            log.info(f"Replicating dataset {replication_factor} times to simulate {original_epochs} epochs.")
            final_chunks = final_chunks * replication_factor
        else: # original_epochs is 1, no replication needed
             pass
    else:
        log.warning("No chunks in training mix.")

    total_tokens = sum(len(tokenizer(c, add_special_tokens=False)["input_ids"]) for c in final_chunks)
    log.info(f"Final training mix: Total chunks: {len(final_chunks)}, Total tokens: {total_tokens}")
    
    # The trainer will run for 1 epoch on the fully constructed dataset
    train_cfg.num_train_epochs = 1
    dataset = Dataset.from_dict({"text": final_chunks})
    return dataset, train_cfg


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
    dataset, updated_train_cfg = prepare_training_mix(
        strategy_name=strategy_name,
        tokenizer=tokenizer,
        log=log,
        train_cfg=train_cfg,
        **strategy_args,
        **chunking_args,
    )
    
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
    
    def get_dataloader_content(dataloader):
        content = []
        for batch in dataloader:
            # Assuming 'input_ids' is the key for tokenized text
            # and we decode it back to string for inspection.
            text_sample = tokenizer.decode(batch['input_ids'][0][:100])
            content.append(text_sample)
        return content

    log.info("Running first dataloader pass for debugging...")
    dataloader1 = trainer.get_train_dataloader()
    content1 = get_dataloader_content(dataloader1)
    with open(os.path.join(output_dir_for_debug, "debug_run_1.txt"), "w") as f:
        f.write("##\n\n".join(content1))
        
    log.info("Running second dataloader pass for debugging...")
    dataloader2 = trainer.get_train_dataloader()
    content2 = get_dataloader_content(dataloader2)
    with open(os.path.join(output_dir_for_debug, "debug_run_2.txt"), "w") as f:
        f.write("##\n\n".join(content2))

    assert content1 == content2, "Sequential sampling is not deterministic!"
    log.info("Sequential sampling verification successful.")
    
    if train:
        trainer.train()
        log.info("Fine-tuning complete.")
        wandb.finish()
        
    return trainer