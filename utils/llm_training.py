
import math
import os
import liger_kernel.transformers
import torch
from typing import Optional, List, Literal
from liger_kernel.transformers import AutoLigerKernelForCausalLM
import liger_kernel 

# from unsloth import FastLanguageModel

# Third-party imports
from datasets import Dataset, load_dataset
from peft import LoraConfig as PeftLoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer
from utils.llm_configs import PeftConfig, ModelConfig, TrainingConfig, InferenceConfig

import os
import warnings
from typing import Any, Callable, Optional, TypeVar, Union

import torch
import torch.nn as nn
from accelerate import PartialState
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BaseImageProcessor,
    DataCollator,
    FeatureExtractionMixin,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainingArguments,
)
from trl import is_conversational, pack_dataset, truncate_dataset

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
    
def load_model_for_training(config: ModelConfig, log, liger_kernel=True, unsloth=False, add_special_token = None, use_existing_lima_tokenizer=False, use_existing_lima_model = False):
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
            if liger_kernel:
                model = AutoLigerKernelForCausalLM.from_pretrained(
                    config.id,
                    trust_remote_code=True,
                    torch_dtype=dtype,
                    device_map="cuda",
                    attn_implementation=config.attn_implementation,
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    config.id,
                    trust_remote_code=True,
                    torch_dtype=dtype,
                    device_map="cuda",
                    attn_implementation=config.attn_implementation,
                )
        else:
            print("...Quantizing...")
            model = AutoModelForCausalLM.from_pretrained(
            config.id,
            trust_remote_code=True,
            torch_dtype=dtype,
            quantization_config=quant_config,
            device_map="auto", #Assume we're operating in a low VRAM environment since we're quantizing
            attn_implementation=config.attn_implementation,
        )
        if use_existing_lima_tokenizer:
            tokenizer = AutoTokenizer.from_pretrained("jiosephlee/olmo2-lima", trust_remote_code=True)
            model.resize_token_embeddings(len(tokenizer))
        else:
            tokenizer = AutoTokenizer.from_pretrained(config.id, trust_remote_code=True)
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

def prepare_lima_dataset(tokenizer: AutoTokenizer, log, use_eot_token=False):
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

    return train_dataset

# **IMPORTANT** Custom trainer to use 'sum' loss, a best practice for chat models.
class SumLossSFTTrainer(SFTTrainer):
    def __init__(self, use_liger_loss, *args, **kwargs):
        super().__init__(*args, **kwargs) # Pass all remaining args/kwargs to parent
        self.use_liger_loss = use_liger_loss

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Computes loss by summing over the sequence dimension, which weights all
        tokens equally. This can improve performance on instruction-following tasks.
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs, use_cache=False)
        logits = outputs.get("logits")

        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        if self.use_liger_loss:
            loss_fct = liger_kernel.transformers.LigerCrossEntropyLoss(reduction="sum")
        else: 
            loss_fct = torch.nn.CrossEntropyLoss(reduction='sum')
        loss = loss_fct(shift_logits.view(-1, self.model.config.vocab_size), shift_labels.view(-1))

        # Normalize by the number of examples and gradient accumulation steps
        loss = loss / self.args.per_device_train_batch_size / self.args.gradient_accumulation_steps

        return (loss, outputs) if return_outputs else loss
    
    def _prepare_dataset(
        self,
        dataset: Union[Dataset, IterableDataset],
        processing_class: Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin],
        args: SFTConfig,
        packing: bool,
        formatting_func: Optional[Callable[[dict], str]],
        dataset_name: str,
    ) -> Union[Dataset, IterableDataset]:
        # Convert the dataset to an IterableDataset if it is a ConstantLengthDataset
        # if isinstance(dataset, ConstantLengthDataset):
        #     return dataset

        # Tabular backends like Arrow/Parquet insert `None` for mismatched keys in nested structures. Clean them from
        # sampled data.
        if isinstance(dataset, Dataset):  # IterableDataset does not support `with_transform`
            dataset = dataset.with_transform(remove_none_values)

        # If the dataset is already preprocessed (tokenized), skip the processing steps.
        column_names = list(next(iter(dataset)).keys())
        is_processed = "input_ids" in column_names

        # Build the kwargs for the `map` function
        map_kwargs = {}
        if isinstance(dataset, Dataset):  # IterableDataset does not support num_proc
            map_kwargs["num_proc"] = args.dataset_num_proc

        with PartialState().main_process_first():
            # Apply the formatting function if any
            if formatting_func is not None and is_processed:
                warnings.warn(
                    "You passed a dataset that is already processed (contains an `input_ids` field) together with a "
                    "formatting function. Therefore `formatting_func` will be ignored. Either remove the "
                    "`formatting_func` or pass a dataset that is not already processed.",
                    UserWarning,
                )

            if formatting_func is not None and not is_processed:
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Applying formatting function to {dataset_name} dataset"

                def _func(example):
                    return {"text": formatting_func(example)}

                try:
                    dataset = dataset.map(_func, batched=False, **map_kwargs)
                except Exception as e:
                    warnings.warn(
                        f"Failed to apply the formatting function due to the following error: {e}. This may be "
                        "because the function is designed for batched input. Please update it to process one example "
                        "at a time (i.e., accept and return a single example). For now, we will attempt to apply the "
                        "function in batched mode, but note that batched formatting is deprecated and will be removed "
                        "in version 0.21.",
                        DeprecationWarning,
                    )
                    dataset = dataset.map(_func, batched=True, **map_kwargs)

            if not is_processed:
                # Removed the chat template conversion and eos token addition

                # Tokenize the dataset
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Tokenizing {dataset_name} dataset"

                def tokenize(example, processing_class, dataset_text_field, assistant_only_loss):
                    if "prompt" in example:  # prompt-completion case
                        if is_conversational(example):
                            prompt_ids = processing_class.apply_chat_template(
                                example["prompt"],
                                tools=example.get("tools"),
                                **example.get("chat_template_kwargs", {}),
                            )
                            prompt_completion_ids = processing_class.apply_chat_template(
                                example["prompt"] + example["completion"],
                                tools=example.get("tools"),
                                **example.get("chat_template_kwargs", {}),
                            )
                        else:
                            prompt_ids = processing_class(text=example["prompt"]).input_ids
                            prompt_completion_ids = processing_class(
                                text=example["prompt"] + example["completion"]
                            ).input_ids

                        # Check if the tokenized prompt starts with the tokenized prompt+completion
                        if not prompt_completion_ids[: len(prompt_ids)] == prompt_ids:
                            warnings.warn(
                                "Mismatch between tokenized prompt and the start of tokenized prompt+completion. "
                                "This may be due to unexpected tokenizer behavior, whitespace issues, or special "
                                "token handling. Verify that the tokenizer is processing text consistently."
                            )

                        # Create a completion mask
                        completion_mask = [0] * len(prompt_ids) + [1] * (len(prompt_completion_ids) - len(prompt_ids))
                        processed = {"input_ids": prompt_completion_ids, "completion_mask": completion_mask}

                    else:  # language modeling case
                        if is_conversational(example):
                            processed = processing_class.apply_chat_template(
                                example["messages"],
                                return_dict=True,
                                return_assistant_tokens_mask=assistant_only_loss,
                                tools=example.get("tools"),
                                **example.get("chat_template_kwargs", {}),
                            )
                            if "assistant_masks" in processed and 1 not in processed["assistant_masks"]:
                                raise RuntimeError(
                                    "You're using `assistant_only_loss=True`, but at least one example has no "
                                    "assistant tokens. This usually means the tokenizer's chat template doesn't "
                                    "generate assistant masks — it may be missing the `{% generation %}` keyword. Please "
                                    "check the template and ensure it's correctly configured to support assistant "
                                    "masking."
                                )
                            processed = {k: processed[k] for k in ("input_ids", "assistant_masks") if k in processed}
                        else:
                            processed = {"input_ids": processing_class(text=example[dataset_text_field]).input_ids}
                    return processed

                dataset = dataset.map(
                    tokenize,
                    fn_kwargs={
                        "processing_class": processing_class,
                        "dataset_text_field": args.dataset_text_field,
                        "assistant_only_loss": args.assistant_only_loss,
                    },
                    **map_kwargs,
                )

            # Pack or truncate
            if packing:
                if args.max_length is None:
                    raise ValueError("When packing is enabled, `max_length` can't be `None`.")
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Packing {dataset_name} dataset"
                dataset = dataset.select_columns("input_ids")
                # Packing adds new column "position_ids" needed for document aware flash attention
                dataset = pack_dataset(dataset, args.max_length, args.packing_strategy, map_kwargs)
            elif args.max_length is not None:
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Truncating {dataset_name} dataset"
                dataset = truncate_dataset(dataset, args.max_length, map_kwargs)
            # For Liger kernel, ensure only input_ids is present
            if args.use_liger_kernel:
                dataset = dataset.select_columns(
                    {"input_ids", "position_ids", "completion_mask"}.intersection(dataset.column_names)
                )

        return dataset
    
def fine_tune_on_text(
    model, tokenizer, log, text_content: str, train_cfg: TrainingConfig, *, tag: str = "finetuning on text..."
):
    """
    Fine-tunes a model on a given string of text by chunking it properly.
    
    Args:
        model: The model to fine-tune
        tokenizer: The tokenizer
        text_content: The text to fine-tune on
        train_cfg: Training configuration
        tag: Tag for logging
    """
    if not text_content or not text_content.strip():
        log.warning(f"[{tag}] Text content is empty. Skipping fine-tuning.")
        return

    log.info(f"Starting SFT for '{tag}'...")
    
    text_content = text_content + tokenizer.eos_token
    
    text_chunks, num_tokens = chunk_text(text_content, tokenizer, train_cfg.context_length)
    
    log.info(f"[{tag}] Tokens: {num_tokens}, Context: {train_cfg.context_length} -> {len(text_chunks)} chunks")
    
    dataset = Dataset.from_dict({"text": text_chunks})
    log.info(f"[{tag}] Created dataset with {len(text_chunks)} chunks (including the eos token)")
    
    train_cfg.gradient_accumulation_steps = len(text_chunks)
    log.info(f"[{tag}] Setting gradient_accumulation_steps to {len(text_chunks)} (one optimizer step per document)")
    
    training_args = train_cfg.to_sft_training_args(packing=False, padding_free=False)
    
    trainer = SumLossSFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        processing_class=tokenizer
    )
    
    trainer.train()
    log.info(f"SFT complete for '{tag}'.")

def sft_train_on_dataset(
    model,  tokenizer, log, train_dataset: Dataset, train_cfg: TrainingConfig, batch_size = 2, grad_accum = 16, use_liger_loss =False
):
    """
    A generalized function to run SFT on a prepared dataset. Effective batch size is batch_size (2) * gradient_accumulation_steps (16) = 32 as per LIMA
    """
    log.info("Starting SFT training run...")
    train_cfg.per_device_train_batch_size = batch_size
    train_cfg.gradient_accumulation_steps = grad_accum
    training_args = train_cfg.to_sft_training_args()

    if grad_accum > 1:
        trainer = SumLossSFTTrainer(
            model=model,
            train_dataset=train_dataset,
            args=training_args,
            processing_class=tokenizer,
            use_liger_loss = use_liger_loss,
        )
    else:
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            args=training_args,
            processing_class=tokenizer
        )

    trainer.train()
    log.info("SFT training complete.")

def fine_tune_on_texts(
    model, tokenizer, log, texts: List[str], train_cfg: TrainingConfig, *, override_grad_steps = 1, tag: str = "finetuning on texts..."
):
    """
    Fine-tunes a model on a given list of texts by chunking them and training on all chunks together.
    
    Args:
        model: The model to fine-tune.
        tokenizer: The tokenizer.
        texts: The list of text strings to fine-tune on.
        train_cfg: Training configuration.
        tag: Tag for logging.
    """
    log.info(f"Starting SFT for '{tag}' on {len(texts)} documents...")

    # Add EOS token to each text
    for i in range(len(texts)):
        texts[i] = texts[i] + tokenizer.eos_token 
    
    # Chunk the texts into smaller pieces based on context length
    all_text_chunks, total_tokens = chunk_texts(texts, tokenizer, train_cfg.context_length)

    log.info(f"[{tag}] Total tokens: {total_tokens}, Context: {train_cfg.context_length} -> {len(all_text_chunks)} total chunks")
    
    # Create dataset with all chunked texts
    dataset = Dataset.from_dict({"text": all_text_chunks})
    
    # This would backprop only once per epoch
    train_cfg.gradient_accumulation_steps = len(all_text_chunks)

    if override_grad_steps > 0:
        train_cfg.gradient_accumulation_steps = override_grad_steps
    training_args = train_cfg.to_sft_training_args(packing=False, padding_free=False) # Packing is False to avoid document re-ordering and padding free is false to avoid OOM issues
    if override_grad_steps == 1:
        # Regular mean reduction is fine for no gradient accumulation; this also technically should enable liger...? since liger requires the labels as a column
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            args=training_args,
            processing_class=tokenizer
        )
    else:
        trainer = SumLossSFTTrainer(
            model=model,
            train_dataset=dataset,
            args=training_args,
            processing_class=tokenizer
        )
    
    trainer.train()
    log.info(f"SFT complete for '{tag}'.")

@torch.inference_mode()
def generate_text(model, tokenizer, prompt: str, config: InferenceConfig) -> str:
    """Simple inference function using Hugging Face transformers.generate."""
    inputs = tokenizer(prompt , return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=config.max_new_tokens,
        temperature=max(config.temperature, 1e-3),
        top_p=config.top_p,
        do_sample=True,
        repetition_penalty=config.repetition_penalty,
        no_repeat_ngram_size = config.no_repeat_ngram_size
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=False)

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

from collections.abc import Mapping

TListOrMapping = TypeVar("TListOrMapping", list, Mapping)


def remove_none_values(example: TListOrMapping) -> TListOrMapping:
    """
    Recursively removes entries with `None` values from a nested structure (list or dictionary).

    Args:
        example (`list` or `Mapping`):
            Input nested structure (list or dictionary) from which to remove `None`.

    Example:
    ```python
    >>> [{
    ...     "a": {"aa": None,
    ...           "ab": 1},
    ...     "b": "my_string",
    ... }]
    >>> remove_none_values(example)
    [{'a': {'ab': 1}, 'b': 'my_string'}]
    ```
    """
    if isinstance(example, list):
        return [remove_none_values(value) if isinstance(value, (dict, list)) else value for value in example]
    elif isinstance(example, Mapping):
        return {
            key: remove_none_values(value) if isinstance(value, (dict, list)) else value
            for key, value in example.items()
            if value is not None
        }
    else:
        raise TypeError("Input must be a list or a dictionary.")