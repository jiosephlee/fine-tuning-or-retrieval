import sys 
#sys.path.append('../../trl')

import math
import os
import wandb
import torch
from typing import Optional, List, Literal
import pandas as pd

# Third-party imports
from datasets import Dataset, load_dataset
from peft import LoraConfig as PeftLoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
#from liger_kernel.transformers import LigerCrossEntropyLoss
from trl import SFTConfig, SFTTrainer
from utils.llm_configs import PeftConfig, ModelConfig, TrainingConfig, InferenceConfig

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
    model, tokenizer, log, text_content: str, train_cfg: TrainingConfig, *, train=True, tag: str = "finetuning on text...", callbacks: Optional[List[TrainerCallback]] = None, chunk_by_section: bool = False
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
        text_chunks, num_tokens = chunk_text_by_sections(text_content, tokenizer, train_cfg.context_length)
        log.info(f"[{tag}] Section-based chunking: Tokens: {num_tokens}, Context: {train_cfg.context_length} -> {len(text_chunks)} chunks")
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
    model, tokenizer, log, texts: List[str], train_cfg: TrainingConfig, *, train=True, tag: str = "finetuning on texts...", callbacks: Optional[List[TrainerCallback]] = None, chunk_by_section: bool = False
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
            chunks, num_tokens = chunk_text_by_sections(text, tokenizer, train_cfg.context_length)
            log.info(f"[{tag}] Chunking text {i}: Context: {train_cfg.context_length} -> {len(chunks)} chunks")
            all_text_chunks.extend(chunks)
            total_tokens += num_tokens
        log.info(f"[{tag}] Section-based chunking: Total tokens: {total_tokens}, Context: {train_cfg.context_length} -> {len(all_text_chunks)} total chunks")
    else:
        log.info(f"[{tag}] Using token-based chunking...")
        all_text_chunks, total_tokens = chunk_texts(texts, tokenizer, train_cfg.context_length)
        log.info(f"[{tag}] Token-based chunking: Total tokens: {total_tokens}, Context: {train_cfg.context_length} -> {len(all_text_chunks)} total chunks")

    # Create dataset with all chunked texts
    dataset = Dataset.from_dict({"text": all_text_chunks})
    
    assert(train_cfg.gradient_accumulation_steps * train_cfg.per_device_train_batch_size >= len(all_text_chunks))
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

def chunk_texts(texts: List[str], tokenizer, context_length: int) -> tuple[List[str], int]:
    """
    Chunks a list of texts into smaller pieces based on context length.
    
    Args:
        texts: List of text strings to chunk
        tokenizer: The tokenizer to use
        context_length: Maximum tokens per chunk
        
    Returns:
        Tuple of (all_text_chunks, total_tokens)
    """
    all_text_chunks = []
    total_tokens = 0
    
    for text_content in texts:
        tokens = tokenizer(text_content, add_special_tokens=False, truncation=False)["input_ids"]
        num_tokens = len(tokens)
        total_tokens += num_tokens
        num_chunks = math.ceil(num_tokens / context_length)
        
        for i in range(num_chunks):
            start_idx = i * context_length
            end_idx = min((i + 1) * context_length, num_tokens)
            chunk_tokens = tokens[start_idx:end_idx]
            # print(chunk_tokens[end_idx-1-start_idx])
            chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=False)
            all_text_chunks.append(chunk_text)
    
    return all_text_chunks, total_tokens

def chunk_text(text_content: str, tokenizer, context_length: int) -> tuple[List[str], int]:
    """
    Chunks a single text into smaller pieces based on context length.
    
    Args:
        text_content: Text string to chunk
        tokenizer: The tokenizer to use
        context_length: Maximum tokens per chunk
        
    Returns:
        Tuple of (text_chunks, num_tokens)
    """
    tokens = tokenizer(text_content, add_special_tokens=False, truncation=False)["input_ids"]
    num_tokens = len(tokens)
    num_chunks = math.ceil(num_tokens / context_length)
    
    text_chunks = []
    for i in range(num_chunks):
        start_idx = i * context_length
        end_idx = min((i + 1) * context_length, num_tokens)
        chunk_tokens = tokens[start_idx:end_idx]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=False)
        text_chunks.append(chunk_text)
    
    return text_chunks, num_tokens

def chunk_text_by_subsections(text_content: str, tokenizer, max_tokens: int = 2048):
    """
    Helper function to chunk a section by subsections when possible.
    
    Args:
        text_content: The section text to chunk
        tokenizer: The tokenizer to use for counting tokens
        max_tokens: Maximum tokens per chunk
        
    Returns:
        List of text chunks and total token count
    """
    import re
    subsection_pattern = r'(\\subsection\{[^}]+\})'
    parts = re.split(subsection_pattern, text_content)
    
    subsections = []
    
    # Handle content before first subsection
    if parts[0].strip():
        subsections.append(parts[0])
    
    # Group subsection headers with their content
    i = 1
    while i < len(parts):
        if re.match(subsection_pattern, parts[i]):  # This is a subsection header
            subsection_content = parts[i]  # Start with the subsection header
            if i + 1 < len(parts):
                subsection_content += parts[i + 1]  # Add the content after the header
            subsections.append(subsection_content)
            i += 2
        else:
            i += 1
    
    # If we only have one subsection (no actual subsection splits), return original
    if len(subsections) <= 1:
        return [text_content], len(tokenizer(text_content, add_special_tokens=False, truncation=False)["input_ids"])
    
    chunks = []
    total_tokens = 0
    current_chunk = ""
    
    for subsection in subsections:
        if not subsection.strip():
            continue
            
        # Check if this would exceed max_tokens when added to current chunk
        test_chunk = current_chunk + subsection
        tokens = tokenizer(test_chunk, add_special_tokens=False, truncation=False)["input_ids"]
        token_count = len(tokens)
        
        if token_count <= max_tokens:
            # Add to current chunk
            current_chunk = test_chunk
        else:
            # Save current chunk if it has content
            if current_chunk.strip():
                chunks.append(current_chunk)
                chunk_tokens = tokenizer(current_chunk, add_special_tokens=False, truncation=False)["input_ids"]
                total_tokens += len(chunk_tokens)
            
            # Check if this subsection alone exceeds max_tokens
            subsection_tokens = tokenizer(subsection, add_special_tokens=False, truncation=False)["input_ids"]
            if len(subsection_tokens) > max_tokens:
                # Fall back to token-based chunking for this subsection
                subsection_chunks, subsection_token_count = chunk_text(subsection, tokenizer, max_tokens)
                chunks.extend(subsection_chunks)
                total_tokens += subsection_token_count
                current_chunk = ""
            else:
                # Start new chunk with this subsection
                current_chunk = subsection
    
    # Add final chunk if it has content
    if current_chunk.strip():
        chunks.append(current_chunk)
        chunk_tokens = tokenizer(current_chunk, add_special_tokens=False, truncation=False)["input_ids"]
        total_tokens += len(chunk_tokens)
    
    return chunks, total_tokens

def chunk_text_by_sections(text_content: str, tokenizer, max_tokens: int = 2048):
    """
    Chunks text by sections, ensuring each chunk doesn't exceed max_tokens.
    If a section is too large, it tries to split by subsections first.
    If no subsections exist or they're still too large, it falls back to token-based chunking.
    Each new chunk is prefixed with the document's title.

    Args:
        text_content: The full text to chunk
        tokenizer: The tokenizer to use for counting tokens
        max_tokens: Maximum tokens per chunk

    Returns:
        List of text chunks and total token count
    """
    import re
    # Extract title
    title_match = re.search(r'\\title\{([^}]+)\}', text_content)
    title = ""
    if title_match:
        title = f"\\title{{{title_match.group(1)}}}\n\n"
    else:
        raise ValueError("No title found in the text")

    # Find all section boundaries and split into sections
    section_pattern = r'(\\section\{[^}]+\})'
    parts = re.split(section_pattern, text_content)
    
    sections = []
    # Handle content before first section (e.g., abstract)
    if parts[0].strip():
        sections.append(parts[0])
    
    # Group section headers with their content
    i = 1
    while i < len(parts):
        if re.match(section_pattern, parts[i]):  # This is a section header
            section_content = parts[i]  # Start with the section header
            if i + 1 < len(parts):
                section_content += parts[i + 1]  # Add the content after the header
            sections.append(section_content)
            i += 2
        else:
            i += 1

    chunks = []
    current_chunk = ""
    
    for section in sections:
        if not section.strip():
            continue

        # Handle title for the very first chunk or any new, empty chunk.
        is_title_block = section.strip().startswith('\\title')
        if not current_chunk and not is_title_block:
            current_chunk = title

        # First, try to fit the whole section.
        candidate_chunk = current_chunk + section
        candidate_tokens = tokenizer(candidate_chunk, add_special_tokens=False, truncation=False)["input_ids"]

        if len(candidate_tokens) <= max_tokens:
            # Section fits, add it to the current chunk.
            current_chunk = candidate_chunk
        else:
            # Section does not fit. Break it down into pieces and add them individually.
            title_token_len = len(tokenizer(title, add_special_tokens=False)["input_ids"])
            # Max size for a piece that might start a new chunk (which would include the title).
            subsection_max_size = max_tokens - title_token_len
            
            pieces, _ = chunk_text_by_subsections(section, tokenizer, subsection_max_size)
                
            # Try to add pieces to the current chunk until it's full.
            for piece in pieces:
                piece_candidate = current_chunk + piece
                piece_candidate_tokens = tokenizer(piece_candidate, add_special_tokens=False, truncation=False)["input_ids"]

                if len(piece_candidate_tokens) <= max_tokens:
                    # Piece fits, add it.
                    current_chunk = piece_candidate
                else:
                    # Piece does not fit. Finalize the current chunk.
                    if current_chunk.strip():
                        chunks.append(current_chunk)
                    
                    # Start a new chunk with the title and this piece.
                    current_chunk = title + piece

    # Add the final chunk if it has content.
    if current_chunk.strip():
        chunks.append(current_chunk)
    
    total_tokens = sum(len(tokenizer(c, add_special_tokens=False)["input_ids"]) for c in chunks)
    return chunks, total_tokens


@torch.inference_mode()
def generate_text(model, tokenizer, prompt: str, config: InferenceConfig) -> str:
    """Simple inference function using Hugging Face transformers.generate."""
    inputs = tokenizer(prompt , return_tensors="pt").to(model.device)
    if config.do_sample:
        outputs = model.generate(
            **inputs,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            do_sample=config.do_sample,
            repetition_penalty=config.repetition_penalty,
            no_repeat_ngram_size = config.no_repeat_ngram_size
        )
    else:
        outputs = model.generate(
        **inputs,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=config.max_new_tokens,
        do_sample=config.do_sample,
        repetition_penalty=config.repetition_penalty,
        no_repeat_ngram_size = config.no_repeat_ngram_size
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=False)

@torch.inference_mode()
def analyze_text_generation(model, tokenizer, prompt, device, max_new_tokens=1024):
    """
    Generates text from a prompt and analyzes the top 5 token choices at each step.

    Args:
        model_name (str): The name of the pretrained model to use (e.g., "gpt2").
        prompt (str): The input text to generate from.
        max_new_tokens (int): The maximum number of new tokens to generate.

    Returns:
        str: A formatted string detailing the generation process.
    """
    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate text and get scores
    outputs = model.generate(
        **inputs,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True
    )
    print(f"Output: {tokenizer.decode(outputs.sequences[0], skip_special_tokens=False)}\n")
          
    # Get the generated token IDs, excluding the input prompt's tokens
    generated_token_ids = outputs.sequences[0, inputs.input_ids.shape[-1]:]
    # Get the scores for each generation step
    token_scores = outputs.scores

    # --- Formatting the Output ---
    output_string = ""
    # Iterate through each generated token and its corresponding scores
    for i, generated_token_id in enumerate(generated_token_ids):
        # Get the scores for the current step
        step_scores = token_scores[i][0]

        # Apply softmax to convert logits to probabilities
        step_probs = torch.nn.functional.softmax(step_scores, dim=0)
        
        # Get the top 5 tokens and their probabilities
        top_5_probs, top_5_indices = torch.topk(step_probs, 5)

        # Decode the generated token and the top 5 tokens
        generated_token = tokenizer.decode(generated_token_id)
        
        # Get the probability of the actual chosen token
        chosen_token_prob = step_probs[generated_token_id].item()

        output_string += f'➡️ Generated Token #{i+1}: "{generated_token.strip()}" (Probability: {chosen_token_prob:.2%})\n'
        output_string += "   Top 5 candidates for this position:\n"
        
        for j, (prob, index) in enumerate(zip(top_5_probs, top_5_indices)):
            decoded_token = tokenizer.decode(index)
            output_string += f"      {j+1}. \"{decoded_token.strip()}\" ({prob:.2%})\n"
        
        output_string += "\n"
        
    return output_string.strip()


@torch.inference_mode()
def extract_logits_first_step(
    model,
    tokenizer,
    prompt: str,
    target_tokens: List[str],
    device = 'cuda',
):
    """
    Greedily generates ONE token after *prompt* and returns the raw logits
    assigned to each token in *target_tokens* at that first generation step.

    Returns
    -------
    dict {token: logit}
    """
    # Encode prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Map each candidate token to a single ID
    token_id_map = {}
    for tok in target_tokens:
        ids = tokenizer(tok, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            raise ValueError(f"'{tok}' is not a single-token string.")
        token_id_map[tok] = ids[0]

    # Generate exactly one new token (greedy)
    gen_out = model.generate(
        **inputs,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=1,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True,
    )
    first_step_logits = gen_out.scores[0][0]        # shape [vocab_size]

    # Extract logits for requested tokens
    return {tok: first_step_logits[tid].item() for tok, tid in token_id_map.items()}


# ---------- usage ----------
# prompt = "Answer with yes or no: Is acetaminophen mutagenic?\nA: "
# logits = extract_logits_first_step(model, tokenizer, prompt, [" yes", " no"])
# print(logits)          # {' yes': -3.21, ' no': -1.05}
# prediction = int(logits[" yes"] > logits[" no"])   # 1 = yes, 0 = no