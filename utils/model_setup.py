import torch
from peft import LoraConfig as PeftLoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from utils.llm_configs import PeftConfig, ModelConfig


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
    
def load_model_for_training(config: ModelConfig, log, use_cpu_and_gpu = False, add_special_token = None, use_existing_lima_tokenizer=False, use_existing_lima_model = False, cache_dir=None):
    """
    Loads a model and tokenizer for training, applying quantization and PEFT.
    **ENHANCED** with robust QLoRA setup from open-instruct.
    """
    log.info(f"Loading model '{config.id}' for training...")

    # Determine torch dtype and device map.
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        # CPU execution should use float32 for broad operator compatibility.
        dtype = torch.float32
    device_map = 'auto' if use_cpu_and_gpu else ("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Model load config: dtype={dtype}, device_map={device_map}")

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
    
    if quant_config is None:
        model = AutoModelForCausalLM.from_pretrained(
            config.id,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map=device_map,
            attn_implementation=config.attn_implementation,
            cache_dir=cache_dir,
        )
    else:
        if not torch.cuda.is_available():
            raise ValueError("4bit/8bit quantization requires CUDA. Disable quantization for CPU-only runs.")
        print("...Quantizing...")
        model = AutoModelForCausalLM.from_pretrained(
        config.id,
        trust_remote_code=True,
        torch_dtype=dtype,
        quantization_config=quant_config,
        device_map=device_map, # Assume low VRAM when quantizing
        attn_implementation=config.attn_implementation,
        cache_dir=cache_dir,
    )
    if use_existing_lima_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained("jiosephlee/olmo2-lima", trust_remote_code=True, cache_dir=cache_dir)
        model.resize_token_embeddings(len(tokenizer))
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.id, trust_remote_code=True, use_fast=True, cache_dir=cache_dir)
        # Add special tokens before doing PEFT
        if add_special_token is not None:
            log.info(f"Adding special token: {add_special_token}")
            special_tokens_dict = {'additional_special_tokens': [add_special_token]}
            tokenizer.add_special_tokens(special_tokens_dict)  
            model.resize_token_embeddings(len(tokenizer))
    
    # Assert that the EOT token is a single special token
    if add_special_token is not None:
        eot_token_str = add_special_token
        eot_token_ids = tokenizer.encode(eot_token_str, add_special_tokens=False)
        assert len(eot_token_ids) == 1, f"EOT token '{eot_token_str}' is not tokenized as a single token. It is tokenized as {eot_token_ids}"
        log.info(f"EOT token '{eot_token_str}' correctly tokenized as a single token ID: {eot_token_ids[0]}")
            
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
