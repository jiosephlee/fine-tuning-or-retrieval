# add .. path 

# pip install flash-attn --no-build-isolation
# pip install git+https://github.com/huggingface/trl
# pip install pydantic datasets==3.6.0 peft bitsandbytes wandb matplotlib seaborn liger-kernel scikit-learn openai
# git clone https://github.com/jiosephlee/transformers; pip install .[torch]

import os
import sys
import json
from datetime import datetime
sys.path.append('../..')
import utils.llm_training as llm_training
import utils.data_preparation as data_preparation
import utils.model_setup as model_setup
import utils.llm_configs as llm_configs
from utils import experiment_utils
import argparse
import wandb
import logging
# Local callback types are no longer used directly; delegated to utils.experiment_utils


def construct_experiment_name(args):
    """Construct experiment path as a nested directory structure."""
    
    # 1. Training Type: e.g., 'peft', 'full'
    training_type = "full" if args.full_finetuning else "peft"
    
    # 2. Model Size: e.g., '1b', '7b'
    model_id_lower = args.model_id.lower()
    if "1b" in model_id_lower:
        model_size = "1b"
    elif "7b" in model_id_lower:
        model_size = "7b"
    else:
        model_size = args.model_id.replace('/', '_')
    
    # 3. Probes Version: e.g., 'probes_v7'
    probes_version = f"probes_{args.knowledge_probes_version}"

    # 4. Domains: e.g., 'd_DPO', 'd_DPO-CoT', 'd_all'
    if args.override_domains:
        if len(args.override_domains) == 1:
            domains = f"domains_{args.override_domains[0]}"
        else:
            domains = f"domains_{'-'.join(args.override_domains)}"
    else:
        domains = "domains_all"

    # 5. Epochs: e.g., 'e1'
    epochs = f"e{args.num_train_epochs}"

    # 6. Batch size and learning rate
    training_params = f"bs{args.effective_batch_size_for_cpt}_lr{args.learning_rate:g}"

    path_parts = [
        training_type,
        model_size,
        probes_version,
    ]

    # Add pretraining strategy if applicable
    if args.fill_batches_with_pretraining:
        pretrain_info = f"fill_{args.pretraining_data_type}"
        path_parts.append(pretrain_info)
    elif args.separate_batches_with_pretraining > 0:
        pretrain_info = f"sep_{args.separate_batches_with_pretraining}_{args.pretraining_data_type}"
        path_parts.append(pretrain_info)

    path_parts.extend([
        domains,
        epochs,
        training_params,
    ])
    
    # Suffix becomes the final leaf directory name for the run
    run_name = args.custom_suffix if args.custom_suffix else datetime.now().strftime('%m_%d_%H_%M')
    path_parts.append(run_name)
    
    return os.path.join(*path_parts)

def get_all_domains():
    return experiment_utils.get_all_domains()

def setup_callbacks(domains, tokenizer, log, args, is_lima=False):
    return experiment_utils.setup_callbacks(domains, tokenizer, log, args, is_lima=is_lima)

def save_probe_results(callbacks, log, args):
    return experiment_utils.save_probe_results(callbacks, log, args)

def load_prompts(prompt_files, append_eot=False):
    return experiment_utils.load_prompts(prompt_files, append_eot=append_eot)

def prior_knowledge_training(model, tokenizer, log, args):
    assert args.effective_batch_size_for_cpt % args.device_batch_size == 0, \
        "Effective batch size for CPT must be divisible by device batch size."
    grad_accum_steps = args.effective_batch_size_for_cpt // args.device_batch_size

    # --- Continued Pretraining Configuration ---
    training_config = llm_configs.TrainingConfig(
        run_name = args.experiment_name,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=1,
        gradient_checkpointing=False,
        per_device_train_batch_size=args.device_batch_size,
        context_length = args.context_length_for_cpt,
        weight_decay=0.1,
        gradient_accumulation_steps=grad_accum_steps,
        warmup_ratio = 0.1, 
        sequential_sampling = True,
        reverse_ffd_packing= False,
        remove_unused_columns=False,
        packing = False,
        padding_free = False,
        report_to="wandb" if not args.test_script else "none",
        activation_offloading=args.offload_to_cpu,
    )
    # --- Load Probe Data ---
    callbacks_to_use = setup_callbacks(
        domains=args.override_domains, 
        tokenizer=tokenizer, 
        log=log, 
        args=args, 
        is_lima=False,
    )
    
    # --- Load the Texts and Fine-Tune ---
    strategy_args = {
        "override_domains": args.override_domains,
        "fill_batches_with_pretraining": args.fill_batches_with_pretraining,
        "separate_batches_with_pretraining": args.separate_batches_with_pretraining,
        "pretraining_data_type": args.pretraining_data_type,
        "add_title_prefix": False,
        "test_script": args.test_script,
    }

    strategy_name = "PriorKnowledge"
    
    output_dir_for_debug = os.path.join(args.base_results_dir, args.experiment_name, "debug")

    trainer = llm_training.fine_tune(
            model=model,
            tokenizer=tokenizer,
            log=log,
            train_cfg=training_config,
            strategy_name=strategy_name,
            strategy_args=strategy_args,
            output_dir_for_debug=output_dir_for_debug,
            callbacks=callbacks_to_use,
            train=True
        )
    log.info("Finished training.")

    if args.push_to_hub_cpt_id:
        log.info(f"Pushing model to hub: {args.push_to_hub_cpt_id}")
        model.push_to_hub(args.push_to_hub_cpt_id)
        #trainer.push_to_hub(args.push_to_hub_cpt_id)
        tokenizer.push_to_hub(args.push_to_hub_cpt_id)
        
    # --- Save Metrics and Generate Plots ---
    save_probe_results(callbacks_to_use, log, args)
    log.info("Finished training and saving all probe results.")
    return trainer.model


def lima_training(model, tokenizer, log, args):
    # --- Prepare LIMA Training Dataset ---
    log.info("\n--- Starting LIMA-based Instruction Tuning ---")
    lima_train_ds = data_preparation.prepare_lima_dataset(tokenizer, log, use_eot_token=True, cache_dir=args.cache_dir)
    log.info(f"Sample formatted training example:\\n{lima_train_ds}")

    assert args.effective_batch_size_for_lima % args.device_batch_size == 0, \
        "Effective batch size for LIMA must be divisible by device batch size."
    grad_accum_steps = args.effective_batch_size_for_lima // args.device_batch_size
    
    # --- LIMA Training Configuration ---
    lima_training_config = llm_configs.TrainingConfig(
        run_name = args.experiment_name + "_LIMA",
        num_train_epochs=args.num_lima_epochs,
        learning_rate=2e-5,
        logging_strategy = "steps",
        logging_steps = 1,
        gradient_checkpointing=False,
        context_length = args.context_length_for_lima,
        gradient_accumulation_steps=grad_accum_steps,
        warmup_ratio = 0.1,
        per_device_train_batch_size=args.device_batch_size,
        weight_decay=0.1,
        use_liger_kernel=True,
        sequential_sampling = False,
        reverse_ffd_packing= False,
        remove_unused_columns=False,
        packing = True,
        padding_free = True,
        dataset_text_field="text",
        report_to="wandb" if not args.test_script else "none",
        activation_offloading=args.offload_to_cpu,
    )
    
    # --- Load Probes ---
    callbacks = setup_callbacks(
        domains=args.override_domains, 
        tokenizer=tokenizer, 
        log=log, 
        args=args, 
        is_lima=True,
    )

    trainer = llm_training.sft_train_on_dataset(
        model=model,
        tokenizer=tokenizer,
        log=log,
        train_dataset=lima_train_ds,
        train_cfg=lima_training_config,
        use_liger_loss=True, 
        train=True,
        callbacks=callbacks
    )
    
    # --- QUALITY CONTROL: Check and assert that seq lengths is properly working ---
    seq_counts = []
    found_multi_seq_batch = False
    
    log.info("Verifying LIMA dataloader integrity...")
    eos_token_id = tokenizer.eos_token_id

    for i, batch in enumerate(trainer.get_train_dataloader()):
        # Check 1: Verify sequence packing
        seq_count = 0
        for j in batch['position_ids'][0]:
            if j == 0:
                seq_count += 1
        seq_counts.append(seq_count)
        if seq_count >= 2:
            found_multi_seq_batch = True

        # Check 2: Verify last token of the batch is EOS
        input_ids = batch['input_ids'][-1]
        last_token_id = input_ids[-1]
        if last_token_id != eos_token_id:
            last_token_decoded = tokenizer.decode([last_token_id])
            log.warning(f"Batch {i} does not end with an EOS token. Instead, it ends with token ID {last_token_id} ('{last_token_decoded}').")
            
    avg_seqs = sum(seq_counts) / len(seq_counts)
    min_seqs = min(seq_counts)
    max_seqs = max(seq_counts)
    
    log.info(f"Sequence stats - Avg: {avg_seqs:.2f}, Min: {min_seqs}, Max: {max_seqs}")
    assert found_multi_seq_batch, "No batch found with at least 2 sequences"

    # --- Train the model ---
    trainer.train()


    if args.push_to_hub_lima_id:
        log.info(f"Pushing model to hub: {args.push_to_hub_lima_id}")
        #trainer.push_to_hub(args.push_to_hub_lima_id)
        model.push_to_hub(args.push_to_hub_lima_id)
        tokenizer.push_to_hub(args.push_to_hub_lima_id)
        
    # --- Save results ---
    save_probe_results(callbacks, log, args)
    
    log.info("LIMA-based instruction tuning complete.")
    
    if not args.test_script:
        wandb.finish()

if __name__ == "__main__":
    # --- Parser ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--custom_suffix", type=str, default="", help="Custom text to append to experiment name")
    parser.add_argument("--override_experiment_name", type=str, default="", help="Override experiment name")
    parser.add_argument("--model_id", type=str, default="allenai/OLMo-2-0425-1B") # allenai/OLMo-2-1124-7B
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--num_lima_epochs", type=int, default=10)
    parser.add_argument("--full_finetuning", default=False, action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--knowledge_probes_version", type=str, default="v9", help="Version of the knowledge probes to use.")
    parser.add_argument("--inference_probes_version", type=str, default="v6", help="Version of the inference probes to use.")
    parser.add_argument("--lima_afterwards", default=False, action="store_true", help="LIMA-based instruction tuning after continued pretraining")
    parser.add_argument("--do_eval", default=False, action="store_true", help="Enable evaluation of generations using an LLM judge.")
    parser.add_argument("--test_script", action="store_true", help="Run in test mode with a small model and minimal epochs.")
    parser.add_argument("--override_domains", type=str, nargs='+', default=None, help="A list of domains to override the default (all domains).")
    parser.add_argument("--fill_batches_with_pretraining", default=False, action="store_true", help="Fill batches with pretraining data.")
    parser.add_argument("--separate_batches_with_pretraining", type=int, default=0, help="Number of pretraining batches to insert between unique document types.")
    parser.add_argument("--pretraining_data_type", type=str, default="dclm", help="Type of pretraining data for filling batches.")
    parser.add_argument("--effective_batch_size_for_cpt", type=int, default=8, help="The effective batch size for continued pretraining.")
    parser.add_argument("--effective_batch_size_for_lima", type=int, default=32, help="The effective batch size for LIMA training.")
    parser.add_argument("--device_batch_size", type=int, default=2, help="The batch size per device.")
    parser.add_argument("--context_length_for_cpt", type=int, default=3072, help="Context length for continued pretraining.")
    parser.add_argument("--context_length_for_lima", type=int, default=3072, help="Context length for LIMA training.")
    parser.add_argument("--push_to_hub_cpt_id", type=str, default="prior", help="Hub model ID to push CPT model to.")
    parser.add_argument("--push_to_hub_lima_id", type=str, default="prior_with_lima", help="Hub model ID to push LIMA model to.")
    parser.add_argument("--offload_to_cpu", action="store_true", help="Enable activation offloading to CPU.")
    parser.add_argument(
        "--no_callback_every_step",
        action="store_true",
        help="If set, run heavy callbacks only at 25%, 50%, and 75% of training instead of every step.",
    )
    parser.add_argument("--parcc", action="store_true", help="Use /vast/projects/myatskar/design-documents as cache directory for model and dataset loading operations")


    args = parser.parse_args()

    # Set cache_dir based on --parcc flag
    if args.parcc:
        args.cache_dir = "/vast/projects/myatskar/design-documents"
    else:
        args.cache_dir = None

    # --- Setup Logging & Wandb ---
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log = logging.getLogger(__name__)

    if args.test_script:
        log.info("--- RUNNING IN TEST SCRIPT MODE ---")
        args.num_train_epochs = 4
        args.num_lima_epochs = 1
        args.base_results_dir = os.path.join("../../results", "tests", "prior_knowledge")
    else: 
        os.environ["WANDB_PROJECT"]="fine_tuning_study"
        args.base_results_dir = os.path.join("../../results", "prior_knowledge")

    if args.override_experiment_name:
        args.experiment_name = args.override_experiment_name
    else:
        args.experiment_name = construct_experiment_name(args)

    # --- Save Hyperparameters ---
    experiment_dir = os.path.join(args.base_results_dir, args.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    hyperparameters_path = os.path.join(experiment_dir, 'hyperparameters.json')
    with open(hyperparameters_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
    log.info(f"Hyperparameters saved to {hyperparameters_path}")

    # --- Load the model ---
    model_config = llm_configs.ModelConfig(
        id= args.model_id, #"allenai/OLMo-2-0425-1B", #"allenai/OLMo-2-1124-7B",
        peft=llm_configs.PeftConfig(
            enabled=(not args.full_finetuning),
            instruction_tuning=False,
        ),
        quantization=llm_configs.QuantizationConfig(mode=None),
    )

    log.info("\n--- Loading Model for Training ---")
    model, tokenizer = model_setup.load_model_for_training(model_config, log, add_special_token="<|EOT|>", use_existing_lima_tokenizer =False, use_existing_lima_model=False, cache_dir=args.cache_dir)

    # --- Prior Knowledge Pretraining (we also evaluate our probes during this) ---
    if args.num_train_epochs > 0:
        model = prior_knowledge_training(model, tokenizer, log, args)
        log.info("Finished prior knowledge training.")
    # -- LIMA-based instruction tuning ---
    if args.lima_afterwards:
        lima_training(model, tokenizer, log, args)
    elif not args.test_script:
        wandb.finish()
    
