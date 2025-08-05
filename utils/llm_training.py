import sys 
#sys.path.append('../../trl')

import math
import os
import wandb
import torch
import itertools
import ast
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
from liger_kernel.transformers import LigerCrossEntropyLoss
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
    model, tokenizer, log, text_content: str, train_cfg: TrainingConfig, *, train=True, tag: str = "finetuning on text...", callbacks: Optional[List[TrainerCallback]] = None
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
    """
    log.info(f"Starting SFT for '{tag}'...")
    
    text_content = text_content + tokenizer.eos_token
    text_chunks, num_tokens = chunk_text(text_content, tokenizer, train_cfg.context_length)
    
    log.info(f"[{tag}] Tokens: {num_tokens}, Context: {train_cfg.context_length} -> {len(text_chunks)} chunks")
    
    dataset = Dataset.from_dict({"text": text_chunks})
    log.info(f"[{tag}] Created dataset with {len(text_chunks)} chunks (including the eos token)")
    
    assert((train_cfg.gradient_accumulation_steps <= len(text_chunks)) and (len(text_chunks)%(train_cfg.per_device_train_batch_size * train_cfg.gradient_accumulation_steps) == 0))
    log.info(f"[{tag}] Gradient_accumulation_steps ({train_cfg.gradient_accumulation_steps}) is less than {len(text_chunks)}")
    
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
    model, tokenizer, log, texts: List[str], train_cfg: TrainingConfig, *, train=True, tag: str = "finetuning on texts...", callbacks: Optional[List[TrainerCallback]] = None
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
    
    assert(train_cfg.gradient_accumulation_steps <= len(all_text_chunks))
    log.info(f"[{tag}] Gradient_accumulation_steps ({train_cfg.gradient_accumulation_steps}) is less than {len(all_text_chunks)}")

    training_args = train_cfg.to_sft_training_args() # Packing is False to avoid document re-ordering and padding free is false to avoid OOM issues

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
    
    
class KnowledgeProbeCallback(TrainerCallback):
    """
    A unified callback that evaluates model performance on a custom set of knowledge probes.
    It tracks multiple key metrics in an efficient way:
    1. Perplexity for a "raw" knowledge statement.
    2. For atomic probes (context + target), it calculates in a single pass:
        - Perplexity of the whole statement (context + target).
        - Perplexity of just the target.
        - Log probability of the whole statement.
        - Log probability of the target.
    3. It also tracks the DELTA of each metric relative to its value before training (at step 0).
    """
    def __init__(self, tokenizer: AutoTokenizer, probe_dataset_path: str, max_length: int, batch_size: int = 8, log_prefix="probe_eval"):
        print("Initializing KnowledegProbeCallback...")
        self.tokenizer = tokenizer
        self.log_prefix = log_prefix
        self.max_length = max_length
        self.batch_size = batch_size
        self.initial_metrics = {}

        # User-facing configuration to easily enable/disable metrics for probe groups.
        self.PROBE_CONFIG = {
            'raw_knowledge': {
                'perplexity': {'track_delta': True, 'track_paraphrased': False}
            },
            'atomic_whole': {
                'perplexity': {'track_delta': True, 'track_paraphrased': True},
                'log_prob': {'track_delta': True, 'track_paraphrased': True}
            },
            'atomic_target': {
                'perplexity': {'track_delta': True, 'track_paraphrased': True},
                'log_prob': {'track_delta': True, 'track_paraphrased': True},
                'hit_at_5': {'track_delta': False, 'track_paraphrased': False},
                'hit_at_50': {'track_delta': False, 'track_paraphrased': False},
                'hit_at_100': {'track_delta': False, 'track_paraphrased': False},
            }
        }

        # --- Internal Configuration (Generated from PROBE_CONFIG) ---
        self.METRICS_CONFIG = {}
        for group, metrics in self.PROBE_CONFIG.items():
            for metric_type, config in metrics.items():
                full_name = f"{group}_{metric_type}"
                self.METRICS_CONFIG[full_name] = {
                    'value_col_name': metric_type,
                    **config
                }

        # Names of metrics that are calculated together in _calculate_atomic_metrics
        self.atomic_metric_names = [k for k in self.METRICS_CONFIG if k.startswith('atomic_')]

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load and store all necessary data from the CSV if the path is provided
        if probe_dataset_path and os.path.exists(probe_dataset_path):
            df = pd.read_csv(probe_dataset_path)
            self.probe_indices = df.index.tolist()
            self.sections = df["section"].tolist()
            self.raw_knowledge_statements = df["raw_knowledge_statement"].tolist()
            self.atomic_probes = df["atomic_knowledge_probe"].tolist()
            self.atomic_targets = df["atomic_target_span"].tolist()
            if "paraphrased_atomic_knowledge_probes" in df.columns:
                paraphrased_probes_raw = df["paraphrased_atomic_knowledge_probes"].tolist()
                print(paraphrased_probes_raw)
                # Safely evaluate the string representation of lists
                self.paraphrased_atomic_probes = [ast.literal_eval(s) for s in paraphrased_probes_raw]
                # Check for data integrity before transposing.
                if self.paraphrased_atomic_probes:
                    first_len = len(self.paraphrased_atomic_probes[0])
                    assert all(len(p) == first_len for p in self.paraphrased_atomic_probes), \
                        "All probes must have the same number of paraphrased variants."
                # Transpose to get lists of probes per variant
                print("Atomic Probes: " + self.paraphrased_atomic_probes)
                self.paraphrased_atomic_probes_by_variant = list(zip(*self.paraphrased_atomic_probes))
                print(self.paraphrased_atomic_probes_by_variant)
                self.num_paraphrase_variants = len(self.paraphrased_atomic_probes_by_variant)
                assert(self.num_paraphrase_variants == 10)
                # Check that there are 10 every row
                assert(all(len(probes) == 10 for probes in self.paraphrased_atomic_probes))
            else:
                self.paraphrased_atomic_probes_by_variant = []
                self.num_paraphrase_variants = 0
        else: # For testing purposes
            raise ValueError("Probe dataset path is not provided")

        # Programmatically initialize history dictionaries based on config
        self.history = {name: [] for name in self.METRICS_CONFIG}
        self.delta_history = {name: [] for name, cfg in self.METRICS_CONFIG.items() if cfg['track_delta']}

        paraphrased_metrics = {name for name, cfg in self.METRICS_CONFIG.items() if cfg['track_paraphrased']}
        self.paraphrased_history = {name: [[] for _ in range(self.num_paraphrase_variants)] for name in paraphrased_metrics}
        self.paraphrased_delta_history = {name: [[] for _ in range(self.num_paraphrase_variants)] for name in paraphrased_metrics if self.METRICS_CONFIG[name]['track_delta']}

    def on_train_begin(self, args, state, control, model, **kwargs):
        """
        Calculate initial perplexity/log-prob values before training starts.
        """
        print("KnowledgeProbeCallback: Calculating initial metrics before training...")
        model.eval()
        device = model.device
        print(str(device))
        # --- Initial Raw Knowledge Perplexity ---
        if 'raw_knowledge_perplexity' in self.METRICS_CONFIG:
            self.initial_metrics['raw_knowledge_perplexity'] = self._calculate_perplexity(model, self.raw_knowledge_statements, device)

        # --- Initial Atomic Metrics ---
        if self.atomic_metric_names:
            atomic_metrics = self._calculate_atomic_metrics(model, self.atomic_probes, self.atomic_targets, device)
            self.initial_metrics.update(atomic_metrics)

        # --- Initial Paraphrased Atomic Metrics ---
        self.initial_metrics['paraphrased'] = [{} for _ in range(self.num_paraphrase_variants)]
        if self.num_paraphrase_variants > 0 and any(cfg['track_paraphrased'] for cfg in self.METRICS_CONFIG.values()):
            print(f"KnowledgeProbeCallback: Calculating initial metrics for {self.num_paraphrase_variants} paraphrase variants...")
            for i in range(self.num_paraphrase_variants):
                paraphrased_probes = self.paraphrased_atomic_probes_by_variant[i]
                paraphrased_metrics = self._calculate_atomic_metrics(model, paraphrased_probes, self.atomic_targets, device)
                for name, values in paraphrased_metrics.items():
                    if self.METRICS_CONFIG.get(name, {}).get('track_paraphrased', False):
                        self.initial_metrics['paraphrased'][i][name] = values

        print("KnowledgeProbeCallback: Initial metrics calculated and stored. Training begins...")
        model.train()

    def _calculate_perplexity(self, model, statements: List[str], device):
        """
        Calculates perplexity for a list of statements.
        """
        all_perplexities = []
        for i in range(0, len(statements), self.batch_size):
            batch_statements = statements[i:i + self.batch_size]
            if not batch_statements:
                continue

            inputs = self.tokenizer(
                batch_statements,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            ).to(device)

            input_ids = inputs["input_ids"]
            
            with torch.no_grad():
                outputs = model(input_ids)
                logits = outputs.logits

            # Prepare for loss calculation
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()

            # Set padding token labels to -100 to be ignored by loss
            shift_labels[shift_labels == self.tokenizer.pad_token_id] = -100
            
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            # loss tensor has shape (batch_size, seq_len-1). Values are 0 where label was -100.
            loss = loss_fct(shift_logits.permute(0, 2, 1), shift_labels)
            
            sum_loss = loss.sum(dim=1)
            num_tokens = (shift_labels != -100).sum(dim=1).float()
            
            mean_nll = sum_loss / num_tokens
            perplexities = torch.exp(mean_nll)

            all_perplexities.append(perplexities)

        if not all_perplexities:
            return torch.tensor([])
        return torch.cat(all_perplexities)

    def _calculate_atomic_metrics(self, model, contexts: List[str], targets: List[str], device):
        """
        In a single forward pass, calculates perplexity, log probability, and hit rate
        for atomic probes based on the current PROBE_CONFIG settings.
        This version uses the tokenizer's standard padding and is more efficient.
        """
        all_metrics = {name: [] for name in self.atomic_metric_names}

        # Determine which metric types we need to compute at all.
        metrics_to_calc = set()
        for group in ['atomic_whole', 'atomic_target']:
            if group in self.PROBE_CONFIG:
                metrics_to_calc.update(self.PROBE_CONFIG[group].keys())

        needs_loss_calc = 'perplexity' in metrics_to_calc or 'log_prob' in metrics_to_calc
        needs_hit_rate_calc = any(k.startswith('hit_at_') for k in metrics_to_calc)

        for i in range(0, len(contexts), self.batch_size):
            batch_contexts = contexts[i:i + self.batch_size]
            batch_targets = targets[i:i + self.batch_size]
            if not batch_contexts:
                continue

            # --- Tokenization ---
            batch_full_text = [c + t for c, t in zip(batch_contexts, batch_targets)]
            inputs = self.tokenizer(
                batch_full_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=False,
            ).to(device)
            input_ids = inputs["input_ids"]

            context_tokenized = self.tokenizer(batch_contexts, add_special_tokens=False, padding="longest", return_tensors="pt")
            context_lengths = context_tokenized.attention_mask.sum(dim=1).to(device)

            # --- Model Forward Pass ---
            with torch.no_grad():
                outputs = model(input_ids)
                logits = outputs.logits

            # --- Metric Calculations ---
            if needs_hit_rate_calc:
                hit_rate_results = self._calculate_hit_rate_metrics_for_batch(logits, context_lengths, batch_targets, metrics_to_calc)
                for name, values in hit_rate_results.items():
                    all_metrics[name].append(values)
            
            if needs_loss_calc:
                nll_results = self._calculate_nll_metrics_for_batch(logits, input_ids, context_lengths, batch_targets)
                for name, values in nll_results.items():
                    all_metrics[name].append(values)

        return {k: torch.cat(v) for k, v in all_metrics.items() if v}

    def _calculate_hit_rate_metrics_for_batch(self, logits, context_lengths, batch_targets, metrics_to_calc):
        """Calculates top-K hit rate metrics for a batch."""
        results = {}
        target_tokenized = self.tokenizer(batch_targets, add_special_tokens=False, padding="longest", return_tensors="pt")
        batch_indices = torch.arange(logits.shape[0], device=logits.device)
        next_token_logits = logits[batch_indices, context_lengths - 1, :]
        first_target_token_ids = target_tokenized.input_ids[:, 0].to(logits.device)
        top_100_indices = torch.topk(next_token_logits, 100, dim=1).indices
        target_ids_expanded = first_target_token_ids.unsqueeze(1)

        if 'hit_at_5' in metrics_to_calc:
            results["atomic_target_hit_at_5"] = (top_100_indices[:, :5] == target_ids_expanded).any(dim=1).float()
        if 'hit_at_50' in metrics_to_calc:
            results["atomic_target_hit_at_50"] = (top_100_indices[:, :50] == target_ids_expanded).any(dim=1).float()
        if 'hit_at_100' in metrics_to_calc:
            results["atomic_target_hit_at_100"] = (top_100_indices[:, :100] == target_ids_expanded).any(dim=1).float()
        return results

    def _calculate_nll_metrics_for_batch(self, logits, input_ids, context_lengths, batch_targets):
        """Calculates NLL-based metrics (perplexity, log-prob) for a batch."""
        results = {}
        shift_logits = logits[..., :-1, :].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')

        # Calculate for whole statement if needed
        if 'atomic_whole_perplexity' in self.atomic_metric_names or 'atomic_whole_log_prob' in self.atomic_metric_names:
            shift_labels_whole = input_ids[..., 1:].contiguous().clone()
            shift_labels_whole[shift_labels_whole == self.tokenizer.pad_token_id] = -100
            loss_whole = loss_fct(shift_logits.permute(0, 2, 1), shift_labels_whole)
            sum_loss_whole = loss_whole.sum(dim=1)

            if 'atomic_whole_log_prob' in self.atomic_metric_names:
                results["atomic_whole_log_prob"] = -sum_loss_whole
            if 'atomic_whole_perplexity' in self.atomic_metric_names:
                num_tokens_whole = (shift_labels_whole != -100).sum(dim=1).float()
                mean_nll_whole = sum_loss_whole / num_tokens_whole
                results["atomic_whole_perplexity"] = torch.exp(mean_nll_whole)

        # Calculate for target statement if needed
        if 'atomic_target_perplexity' in self.atomic_metric_names or 'atomic_target_log_prob' in self.atomic_metric_names:
            target_tokenized = self.tokenizer(batch_targets, add_special_tokens=False, padding="longest", return_tensors="pt")
            shift_labels_target = input_ids[..., 1:].contiguous().clone()
            for j, length in enumerate(context_lengths):
                if length > 0: shift_labels_target[j, :length-1] = -100
            shift_labels_target[shift_labels_target == self.tokenizer.pad_token_id] = -100
            loss_target = loss_fct(shift_logits.permute(0, 2, 1), shift_labels_target)
            sum_loss_target = loss_target.sum(dim=1)

            if 'atomic_target_log_prob' in self.atomic_metric_names:
                results["atomic_target_log_prob"] = -sum_loss_target
            if 'atomic_target_perplexity' in self.atomic_metric_names:
                target_lengths = target_tokenized.attention_mask.sum(dim=1).to(logits.device)
                num_tokens_target = (shift_labels_target != -100).sum(dim=1)
                assert torch.equal(num_tokens_target, target_lengths), \
                    "Number of target tokens after masking does not match expected target lengths."
                mean_nll_target = sum_loss_target / num_tokens_target.float()
                results["atomic_target_perplexity"] = torch.exp(mean_nll_target)
        
        return results

    def on_step_end(self, args, state, control, model, **kwargs):
        if not self.initial_metrics:
            print("KnowledgeProbeCallback: Initial metrics not found. Calculating now...")
            self.on_train_begin(args, state, control, model, **kwargs)

        model.eval()
        device = model.device
        step = state.global_step
        log_data = {}

        def log_metric_wandb(name, tensor):
            valid_mask = ~torch.isinf(tensor) & ~torch.isnan(tensor)
            if valid_mask.any():
                log_data[f"{self.log_prefix}/{name}_avg"] = tensor[valid_mask].mean().item()

        # --- Original Probes ---
        current_metrics = {}
        if 'raw_knowledge_perplexity' in self.METRICS_CONFIG:
            current_metrics['raw_knowledge_perplexity'] = self._calculate_perplexity(model, self.raw_knowledge_statements, device)
        if self.atomic_metric_names:
            current_metrics.update(self._calculate_atomic_metrics(model, self.atomic_probes, self.atomic_targets, device))

        for name, values in current_metrics.items():
            self.history[name].append({'step': step, 'values': values.cpu().tolist()})
            log_metric_wandb(name, values)
            if self.METRICS_CONFIG[name]['track_delta']:
                delta = values - self.initial_metrics[name]
                self.delta_history[name].append({'step': step, 'values': delta.cpu().tolist()})
                log_metric_wandb(f"{name}_delta", delta)

        # --- Paraphrased Probes ---
        if self.num_paraphrase_variants > 0:
            paraphrased_metrics_to_track = {name for name, cfg in self.METRICS_CONFIG.items() if cfg['track_paraphrased']}
            if paraphrased_metrics_to_track:
                # These will store tensors of shape [num_variants, num_probes] for mean calculation
                paraphrased_tensors = {name: [] for name in self.paraphrased_history}
                paraphrased_delta_tensors = {name: [] for name in self.paraphrased_delta_history}

                for i in range(self.num_paraphrase_variants):
                    paraphrased_probes = self.paraphrased_atomic_probes_by_variant[i]
                    variant_metrics = self._calculate_atomic_metrics(model, paraphrased_probes, self.atomic_targets, device)

                    for name, values in variant_metrics.items():
                        if name in paraphrased_metrics_to_track:
                            self.paraphrased_history[name][i].append({'step': step, 'values': values.cpu().tolist()})
                            paraphrased_tensors[name].append(values)

                            if self.METRICS_CONFIG[name]['track_delta']:
                                initial_vals = self.initial_metrics['paraphrased'][i][name]
                                delta = values - initial_vals
                                self.paraphrased_delta_history[name][i].append({'step': step, 'values': delta.cpu().tolist()})
                                paraphrased_delta_tensors[name].append(delta)

                # Log the mean across variants to W&B
                for name, tensors in paraphrased_tensors.items():
                    if tensors: log_metric_wandb(f"paraphrased_{name}", torch.stack(tensors).mean(dim=0))
                for name, tensors in paraphrased_delta_tensors.items():
                    if tensors: log_metric_wandb(f"paraphrased_{name}_delta", torch.stack(tensors).mean(dim=0))

        # --- Final Logging ---
        if state.is_world_process_zero and log_data:
            wandb.log(log_data, step=step)

        model.train()

    def _build_dataframe_from_history(self, history_data, value_col_name, paraphrase_variant_index=None):
        """Helper to construct a DataFrame from a list of history entries."""
        if not history_data:
            return pd.DataFrame()
        records = []
        for entry in history_data:
            step, values = entry['step'], entry['values']
            for i, value in enumerate(values):
                record = {
                    'step': step,
                    'probe_index': self.probe_indices[i],
                    'section': self.sections[i],
                    value_col_name: value
                }
                if paraphrase_variant_index is not None:
                    record['paraphrase_variant'] = paraphrase_variant_index
                records.append(record)
        return pd.DataFrame(records)

    # --- DataFrame Getter Methods (Public API) ---

    def _get_metric_df(self, metric_name, is_delta=False):
        """Generic method to build a DataFrame for a given metric."""
        if metric_name not in self.METRICS_CONFIG: return pd.DataFrame()
        config = self.METRICS_CONFIG[metric_name]
        history_data = self.delta_history.get(metric_name) if is_delta else self.history.get(metric_name)
        value_col_name = f"{config['value_col_name']}_delta" if is_delta else config['value_col_name']
        return self._build_dataframe_from_history(history_data, value_col_name)

    def _get_paraphrased_metric_df(self, metric_name, is_delta=False):
        """Generic method for paraphrased metrics."""
        if metric_name not in self.METRICS_CONFIG or not self.METRICS_CONFIG[metric_name]['track_paraphrased']:
            return pd.DataFrame()
        config = self.METRICS_CONFIG[metric_name]
        history_list = self.paraphrased_delta_history.get(metric_name) if is_delta else self.paraphrased_history.get(metric_name)
        value_col_name = f"{config['value_col_name']}_delta" if is_delta else config['value_col_name']
        
        all_variants_df = []
        if history_list:
            for i, history_data in enumerate(history_list):
                df = self._build_dataframe_from_history(history_data, value_col_name, paraphrase_variant_index=i)
                all_variants_df.append(df)
        return pd.concat(all_variants_df, ignore_index=True) if all_variants_df else pd.DataFrame()

    def save_results(self, output_dir: str):
        """Saves all collected raw and delta metrics to a single CSV file in the specified directory."""
        os.makedirs(output_dir, exist_ok=True)
        print(f"KnowledgeProbeCallback: Saving all probe metrics to {output_dir}")

        # --- Build Non-Paraphrased DataFrames ---
        non_paraphrased_dfs = []
        for name, config in self.METRICS_CONFIG.items():
            if not config['track_paraphrased']:
                # Raw values
                df = self._get_metric_df(name)
                if not df.empty:
                    non_paraphrased_dfs.append(df.rename(columns={config['value_col_name']: name}))
                # Deltas
                if config['track_delta']:
                    delta_df = self._get_metric_df(name, is_delta=True)
                    if not delta_df.empty:
                        delta_col = f"{config['value_col_name']}_delta"
                        non_paraphrased_dfs.append(delta_df.rename(columns={delta_col: f"{name}_delta"}))

        # Merge all non-paraphrased DataFrames
        merged_df = None
        if non_paraphrased_dfs:
            merged_df = non_paraphrased_dfs[0]
            for df_to_merge in non_paraphrased_dfs[1:]:
                cols_to_drop = ['section'] if 'section' in df_to_merge.columns else []
                merged_df = pd.merge(merged_df, df_to_merge.drop(columns=cols_to_drop, errors='ignore'), on=['step', 'probe_index'], how='outer')

        # --- Build and Merge Paraphrased DataFrame ---
        paraphrased_dfs = []
        paraphrased_metric_names = [name for name, cfg in self.METRICS_CONFIG.items() if cfg['track_paraphrased']]
        for name in paraphrased_metric_names:
            config = self.METRICS_CONFIG[name]
            # Raw values
            df = self._get_paraphrased_metric_df(name)
            if not df.empty:
                paraphrased_dfs.append(df.rename(columns={config['value_col_name']: name}))
            # Deltas
            if config['track_delta']:
                delta_df = self._get_paraphrased_metric_df(name, is_delta=True)
                if not delta_df.empty:
                    delta_col = f"{config['value_col_name']}_delta"
                    paraphrased_dfs.append(delta_df.rename(columns={delta_col: f"{name}_delta"}))

        if paraphrased_dfs:
            merged_paraphrased_df = paraphrased_dfs[0]
            for df_to_merge in paraphrased_dfs[1:]:
                cols_to_drop = ['section'] if 'section' in df_to_merge.columns else []
                merged_paraphrased_df = pd.merge(merged_paraphrased_df, df_to_merge.drop(columns=cols_to_drop, errors='ignore'), on=['step', 'probe_index', 'paraphrase_variant'], how='outer')
            
            # --- Combine with main DataFrame ---
            if merged_df is not None:
                merged_df = pd.merge(merged_df, merged_paraphrased_df, on=['step', 'probe_index'], how='outer')
            else:
                merged_df = merged_paraphrased_df

        # --- Save the final consolidated dataframe ---
        if merged_df is None or merged_df.empty:
            print(" > No metrics to save.")
            return

        output_path = os.path.join(output_dir, 'knowledge_probe_metrics.csv')
        merged_df.to_csv(output_path, index=False)
        print(f" > Saved consolidated metrics to 'knowledge_probe_metrics.csv' with {len(merged_df)} rows.")


class CorpusPerplexityCallback(TrainerCallback):
    """
    Calculates the perplexity of an entire text corpus at the end of each
    training step using a strided sliding window approach. This provides a
    more accurate perplexity measure for long documents than naive chunking.
    Based on the Hugging Face documentation for PPL with fixed-length models.
    """
    def __init__(self, text_content: str, tokenizer: AutoTokenizer, max_length: int, stride: int = 512, log_prefix="corpus_perplexity"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.log_prefix = log_prefix
        self.encodings = self.tokenizer(text_content, return_tensors="pt")
        self.history = []

    def on_step_end(self, args, state, control, model, **kwargs):
        model.eval()
        device = model.device

        seq_len = self.encodings.input_ids.size(1)
        nll_sum = 0.0
        n_tokens = 0
        prev_end_loc = 0

        for begin_loc in range(0, seq_len, self.stride):
            end_loc = min(begin_loc + self.max_length, seq_len)
            trg_len = end_loc - prev_end_loc
            input_ids = self.encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            
            # Mask out tokens that are only used for context. The model will not
            # calculate loss for these tokens (label = -100).
            target_ids[:, :-trg_len] = -100

            if torch.all(target_ids == -100):
                prev_end_loc = end_loc
                if end_loc == seq_len:
                    break
                continue

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                # outputs.loss is the *average* negative log-likelihood for the window.
                neg_log_likelihood = outputs.loss

            # To get the total NLL for the window, we multiply the average by the
            # number of tokens the loss was calculated over.
            num_valid_tokens = (target_ids != -100).sum().item()
            # The model internally shifts labels, so loss is on one less token per sequence.
            # Our batch size is 1 here.
            num_loss_tokens = num_valid_tokens - 1
            if num_loss_tokens > 0:
                nll_sum += neg_log_likelihood.item() * num_loss_tokens
                n_tokens += num_loss_tokens

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break
        
        if n_tokens > 0:
            avg_nll = nll_sum / n_tokens
            perplexity = torch.exp(torch.tensor(avg_nll))
        else:
            perplexity = torch.tensor(float('inf'))

        perplexity_item = perplexity.item()
        if state.is_world_process_zero:
            wandb.log({f"{self.log_prefix}/full_paper": perplexity_item}, step=state.global_step)
        
        self.history.append({'step': state.global_step, 'corpus_perplexity': perplexity_item})

        model.train()

    def get_results_as_dataframe(self):
        """
        Returns the collected corpus perplexity data as a pandas DataFrame.
        """
        return pd.DataFrame(self.history)


class TrainingLossPerplexityCallback(TrainerCallback):
    """
    A callback that captures the training loss at each logging step,
    calculates perplexity from it, logs it to Weights & Biases,
    and stores it for external analysis.
    This represents the perplexity of the specific data chunk seen in that step.
    """
    def __init__(self):
        self.history = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        # The 'loss' key is only present during training steps.
        if logs is not None and 'loss' in logs:
            if state.is_world_process_zero:
                # The 'loss' is the average cross-entropy loss for the batch.
                # Perplexity is the exponentiation of this loss.
                chunk_perplexity = math.exp(logs['loss'])
                self.history.append({'step': state.global_step, 'chunked_perplexity': chunk_perplexity})
                wandb.log({"chunked_perplexity/full_paper": chunk_perplexity}, step=state.global_step+1)
    
    def get_results_as_dataframe(self):
        """
        Returns the collected training loss perplexity data as a pandas DataFrame.
        """
        return pd.DataFrame(self.history)


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