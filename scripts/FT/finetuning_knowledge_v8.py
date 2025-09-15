# add .. path 

# pip install flash-attn --no-build-isolation
# pip install git+https://github.com/huggingface/trl
# pip install pydantic datasets==3.6.0 peft bitsandbytes wandb matplotlib seaborn liger-kernel scikit-learn openai
# git clone https://github.com/jiosephlee/transformers; pip install .[torch]

import os
import sys
import json
sys.path.append('../..')
import pandas as pd
import utils.llm_training as llm_training
import utils.data_preparation as data_preparation
import utils.model_setup as model_setup
import utils.llm_callbacks as llm_callbacks
import utils.llm_configs as llm_configs
import argparse
import wandb
import logging
from utils.old.llm_callbacks_old import TrainingLossPerplexityCallback
from utils.llm_callbacks import CorpusPerplexityCallback


def construct_experiment_name(args):
    """Construct experiment path as a nested directory structure."""
    
    # 1. Training Type: e.g., 'peft', 'full'
    training_type = "full" if args.full_finetuning else "peft"
    
    # 2. Model Size: e.g., '1b', '7b'
    model_size = "1b" if "1B" in args.model_id else "7b"
    
    # 3. Probes Version: e.g., 'probes_v7'
    probes_version = f"probes_{args.knowledge_probes_version}"

    # 4. Chunking Style: e.g., 'sec_no-ovp', 'sec_ovp_1_4', 'tok'
    if args.chunk_by_section:
        chunking_style = "section"
        if args.overlap_sections:
            chunking_style += f"_overlap_{args.overlap_ratio}"
        else:
            chunking_style += "_no-overlap"
    else:
        chunking_style = "newline2"

    # 5. Data Mix: e.g., 'source_only', 'para9', 'para9_expl'
    if args.num_paraphrased_texts > 0:
        if args.with_explanations:
            data_mix = f"para{args.num_paraphrased_texts}_expl"
        else:
            data_mix = f"para{args.num_paraphrased_texts}"
    else:
        data_mix = "source_only"

    # 6. Domains: e.g., 'd_DPO', 'd_DPO-CoT', 'd_all'
    if args.override_domains:
        if len(args.override_domains) == 1:
            domains = f"domains_{args.override_domains[0]}"
        else:
            domains = f"domains_{'-'.join(args.override_domains)}"
    else:
        domains = "domains_all"

    # 7. Epochs: e.g., 'e1'
    epochs = f"e{args.num_train_epochs}"

    path_parts = [
        training_type,
        model_size,
        probes_version,
        chunking_style,
        data_mix,
        domains,
        epochs,
    ]
    
    # Suffix becomes the final leaf directory name for the run
    run_name = args.custom_suffix if args.custom_suffix else "run"
    path_parts.append(run_name)
    
    return os.path.join(*path_parts)

def get_all_domains():
    """Scans the data directory to find all available domains for probes."""
    facts_dir = '../../data/probes/facts'
    if not os.path.isdir(facts_dir):
        return []
    return [name for name in os.listdir(facts_dir) if os.path.isdir(os.path.join(facts_dir, name))]

def setup_callbacks(domains, tokenizer, log, args, is_lima=False):
    """Sets up probe callbacks for all specified domains."""
    callbacks = []
    
    report_to_wandb = not args.test_script

    if not domains:
        domains = get_all_domains()
        log.info(f"No domains specified, found and using: {domains}")

    all_generation_prompts = {}

    for domain in domains:
        log.info(f"--- Setting up probes for domain: {domain} ---")
        
        # --- Output Directories ---
        suffix = "_lima" if is_lima else ""
        output_dir_knowledge_probe = os.path.join(args.base_results_dir, args.experiment_name, f"{domain}{suffix}_knowledge_probe")
        output_dir_inference_probe = os.path.join(args.base_results_dir, args.experiment_name, f"{domain}{suffix}_inference_probe")
        os.makedirs(output_dir_knowledge_probe, exist_ok=True)
        os.makedirs(output_dir_inference_probe, exist_ok=True)

        # --- Knowledge Probes ---
        knowledge_probes_version = args.knowledge_probes_version
        if knowledge_probes_version == 'v8':
            knowledge_probe_path = f'../../data/probes/facts/{domain}/probes_{knowledge_probes_version}.csv'
        else:
            knowledge_probe_path = f'../../data/probes/facts/{domain}/{domain}_knowledge_probes_{knowledge_probes_version}.csv'

        if os.path.exists(knowledge_probe_path):
            knowledge_probe_df = pd.read_csv(knowledge_probe_path)
            facts = knowledge_probe_df['fact'].tolist()
            probes = knowledge_probe_df['probe'].tolist()
            targets = knowledge_probe_df['target'].tolist()
            
            knowledge_probe_callback = llm_callbacks.BaseKnowledgeProbeCallBack(
                tokenizer=tokenizer,
                facts=facts,
                probes=probes,
                targets=targets,
                probes_df=knowledge_probe_df,
                batch_size=8,
                logger=log,
                output_dir=output_dir_knowledge_probe,
                log_prefix=f"{domain}_knowledge_probe",
                report_to_wandb=report_to_wandb,
            )
            callbacks.append(knowledge_probe_callback)
            log.info(f"Loaded {len(knowledge_probe_df)} knowledge probes from {knowledge_probe_path}")
        else:
            log.warning(f"Knowledge probe file not found for domain {domain} at {knowledge_probe_path}")

        # --- Inference Probes ---
        inference_probes_version = args.inference_probes_version
        path1 = f'../../data/probes/inference/{domain}/probes_{inference_probes_version}.csv'
        path2 = f'../../data/probes/inference/{domain}/{domain.lower()}_high_level_probes_{inference_probes_version}.csv'
        
        if os.path.exists(path1):
            inference_probe_path = path1
        elif os.path.exists(path2):
            inference_probe_path = path2
        else:
            inference_probe_path = None
            log.warning(f"Inference probe file not found for domain {domain} with version {inference_probes_version}")

        if inference_probe_path:
            inference_probe_df = pd.read_csv(inference_probe_path)
            facts = inference_probe_df['fact'].tolist()
            probes = inference_probe_df['probe'].tolist()
            targets = inference_probe_df['target'].tolist()

            inference_probe_callback = llm_callbacks.BaseKnowledgeProbeCallBack(
                tokenizer=tokenizer,
                facts=facts,
                probes=probes,
                targets=targets,
                probes_df=inference_probe_df,
                batch_size=8,
                logger=log,
                output_dir=output_dir_inference_probe,
                log_prefix=f"{domain}_inference_probe",
                report_to_wandb=report_to_wandb,
            )
            callbacks.append(inference_probe_callback)
            log.info(f"Loaded {len(inference_probe_df)} inference probes from {inference_probe_path}")
        
        # --- Corpus Perplexity Callback ---
        corpus_path = f'../../data/arxiv/cleaned/{domain}.tex'
        if os.path.exists(corpus_path):
            with open(corpus_path, 'r', encoding='utf-8') as f:
                text_content = f.read()
            
            context_length = args.context_length_for_lima if is_lima else args.context_length_for_cpt
            
            output_dir_corpus_ppl = os.path.join(args.base_results_dir, args.experiment_name, f"{domain}{suffix}_corpus_perplexity")
            os.makedirs(output_dir_corpus_ppl, exist_ok=True)

            corpus_perplexity_callback = CorpusPerplexityCallback(
                text_content=text_content,
                tokenizer=tokenizer,
                max_length=context_length,
                stride=512,
                output_dir=output_dir_corpus_ppl,
                log_prefix=f"{domain}_corpus_perplexity",
                report_to_wandb=report_to_wandb,
            )
            callbacks.append(corpus_perplexity_callback)
            log.info(f"Added CorpusPerplexityCallback for domain {domain} from {corpus_path}")
        else:
            log.warning(f"Corpus file not found for domain {domain} at {corpus_path}")

        # --- Generation Probes (collect for single callback) ---
        if is_lima:
            prompt_files = {
                f'recall_{domain}_QA': f'../../data/probes/generation/{domain}/recall_{domain}_QA.json',
                f'yourbench_{domain}': f'../../data/probes/generation/{domain}/yourbench_{domain}.json'
            }
        else:
            prompt_files = {
                f'recall_{domain}': f'../../data/probes/generation/{domain}/recall_{domain}.json'
            }
        
        domain_prompts = load_prompts(prompt_files, append_eot=is_lima)
        all_generation_prompts.update(domain_prompts)

    if is_lima:
        background_prompt_path = '../../data/probes/generation/DPO/recall_background_QA.json'
        if os.path.exists(background_prompt_path):
            background_prompts = load_prompts({'recall_background_QA': background_prompt_path}, append_eot=is_lima)
            all_generation_prompts.update(background_prompts)

    if all_generation_prompts:
        suffix = "_lima" if is_lima else ""
        output_dir_generation = os.path.join(args.base_results_dir, args.experiment_name, f"generation{suffix}")
        os.makedirs(output_dir_generation, exist_ok=True)
        
        inference_config = llm_configs.InferenceConfig(no_repeat_ngram_size=6)
        
        generation_probe_callback = llm_callbacks.GenerationProbeCallback(
            prompts=all_generation_prompts,
            tokenizer=tokenizer,
            inference_config=inference_config,
            eval_every_n_steps=6 if is_lima else 10,
            logger=log,
            output_dir=output_dir_generation,
            do_eval=args.do_eval,
            report_to_wandb=report_to_wandb,
        )
        callbacks.append(generation_probe_callback)
        log.info(f"Loaded generation probes for domains: {list(all_generation_prompts.keys())}")

    callbacks.append(TrainingLossPerplexityCallback())
    return callbacks

def save_probe_results(callbacks, log):
    """Saves results for all probe callbacks."""
    for callback in callbacks:
        if isinstance(callback, llm_callbacks.BaseKnowledgeProbeCallBack):
            callback.save_results(output_dir=callback.output_dir)
            log.info(f"Probe metrics for {callback.log_prefix} saved to {callback.output_dir}")
        elif isinstance(callback, CorpusPerplexityCallback):
            callback.save_results(output_dir=callback.output_dir)
            log.info(f"Corpus perplexity metrics for {callback.log_prefix} saved to {callback.output_dir}")

def load_prompts(prompt_files, append_eot=False):
    prompts = {}
    for name, path in prompt_files.items():
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            prompt_list = []
            for item in data:
                question = item.get('question', '')
                if append_eot:
                    question += "<|EOT|>"
                prompt_list.append({
                    "prompt_name": item.get('id', 'unknown'),
                    "question": question,
                    "reference_answer": item.get('reference_answer', '')
                })
            prompts[name] = prompt_list
    return prompts

def continue_pretraining(model, tokenizer, log, args):
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
    # --- Determine Training Strategy ---
    strategy_args = {
        "num_paraphrased_texts": args.num_paraphrased_texts,
        "override_domains": args.override_domains,
        "fill_batches_with_pretraining": args.fill_batches_with_pretraining,
        "pretraining_batches_separating_docs": args.separate_batches_with_pretraining,
        "pretraining_data_type": args.pretraining_data_type,
        "test_script": args.test_script,
    }

    if args.with_explanations:
        strategy_name = "ParaphrasedArxivPaperWithExplanations"
    elif args.num_paraphrased_texts > 0:
        strategy_name = "ParaphrasedArxivPaper"
    else:
        strategy_name = "SingleArxivPaper" # Or a more generic name like "Source"

    chunking_args = {
        "chunk_by_section": args.chunk_by_section,
        "overlap_sections": args.overlap_sections,
        "overlap_ratio": args.overlap_ratio,
        "add_title_prefix": not args.no_title_prefix
    }
    
    output_dir_for_debug = os.path.join(args.base_results_dir, args.experiment_name, "debug")

    llm_training.fine_tune(
            model=model,
            tokenizer=tokenizer,
            log=log,
            train_cfg=training_config,
        strategy_name=strategy_name,
        strategy_args=strategy_args,
        output_dir_for_debug=output_dir_for_debug,
        callbacks=callbacks_to_use,
            train=True,
        **chunking_args
        )

    # --- Save Metrics and Generate Plots ---
    save_probe_results(callbacks_to_use, log)

    # --- Generate Plots ---
    # Note: Plotting logic is removed as it's complex with multiple domains. 
    # Please use regenerate_plots.py script or add custom plotting logic.
    log.info("Finished training and saving all probe results.")


def lima_training(model, tokenizer, log, args, num_train_epochs=15):
    # --- Prepare LIMA Training Dataset ---
    log.info("\n--- Starting LIMA-based Instruction Tuning ---")
    lima_train_ds = data_preparation.prepare_lima_dataset(tokenizer, log, use_eot_token=True)
    log.info(f"Sample formatted training example:\\n{lima_train_ds}")

    assert args.effective_batch_size_for_lima % args.device_batch_size == 0, \
        "Effective batch size for LIMA must be divisible by device batch size."
    grad_accum_steps = args.effective_batch_size_for_lima // args.device_batch_size
    
    # --- LIMA Training Configuration ---
    lima_training_config = llm_configs.TrainingConfig(
        run_name = args.experiment_name + "_LIMA",
        num_train_epochs=num_train_epochs,
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
        report_to="wandb" if not args.test_script else "none",
    )
    
    # --- Load Probes ---
    # Note 1: We track the DPO knowledge probes, DPO inference probes in OG & Q&A format, and generative recall in Q&A format
    # NOte 2: Since we are retracking some of the same probes, we need to make sure they are in separate folders and different prefixes for WandDB
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
        train=False,
        callbacks=callbacks
    )
    
    # --- QUALITY CONTROL: Check and assert that seq lengths is properly working ---
    seq_counts = []
    found_multi_seq_batch = False
    
    for i, batch in enumerate(trainer.get_train_dataloader()):
        seq_count = 0
        for j in batch['position_ids'][0]:
            if j == 0:
                seq_count += 1
        seq_counts.append(seq_count)
        
        if seq_count >= 2:
            found_multi_seq_batch = True
            
    avg_seqs = sum(seq_counts) / len(seq_counts)
    min_seqs = min(seq_counts)
    max_seqs = max(seq_counts)
    
    log.info(f"Sequence stats - Avg: {avg_seqs:.2f}, Min: {min_seqs}, Max: {max_seqs}")
    assert found_multi_seq_batch, "No batch found with at least 2 sequences"

    # --- Train the model ---
    trainer.train()

    # --- Save results ---
    save_probe_results(callbacks, log)
    
    # --- Generate plots ---
    # llm_plotting.generate_new_plots_for_knowledge_probes(args.knowledge_probes_version,output_dir_knowledge_probe, logger=log)
    # llm_plotting.generate_new_plots_for_inference_probes("v2",output_dir_inference_probe, logger=log)
    # log.info("Finished generating all plots.")

    log.info("LIMA-based instruction tuning complete.")
    wandb.finish()

if __name__ == "__main__":
    # --- Parser ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--custom_suffix", type=str, default="", help="Custom text to append to experiment name")
    parser.add_argument("--override_experiment_name", type=str, default="", help="Override experiment name")
    parser.add_argument("--model_id", type=str, default="allenai/OLMo-2-0425-1B") # allenai/OLMo-2-1124-7B
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--full_finetuning", default=False, action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--knowledge_probes_version", type=str, default="v8", help="Version of the knowledge probes to use.")
    parser.add_argument("--inference_probes_version", type=str, default="v2", help="Version of the inference probes to use.")
    parser.add_argument("--num_paraphrased_texts", type=int, default=9, help="Number of paraphrased texts to use for training (0-9)")
    parser.add_argument("--lima_afterwards", default=False, action="store_true", help="LIMA-based instruction tuning after continued pretraining")

    # Defaults, do not change unless for ablations
    parser.add_argument("--chunk_by_section", action="store_true", help="Use section-based chunking instead of token-based chunking")
    parser.add_argument("--no_title_prefix", action="store_false", help="Add title prefix to chunks when chunking")
    parser.add_argument("--overlap_sections", default=False, action="store_true", help="Overlap sections when chunking")
    parser.add_argument("--overlap_ratio", type=str, default="1_4", help="Ratio of overlap when chunking")
    parser.add_argument("--with_explanations", default=False, action="store_true", help="Use explanations when fine-tuning on paraphrased texts")
    parser.add_argument("--with_prior_knowledge", default=False, action="store_true", help="Use prior knowledge when fine-tuning on paraphrased texts")
    parser.add_argument("--do_eval", default=False, action="store_true", help="Enable evaluation of generations using an LLM judge.")
    parser.add_argument("--test_script", action="store_true", help="Run in test mode with a small model and minimal epochs.")

    # New arguments for multi-domain training
    parser.add_argument("--override_domains", type=str, nargs='+', default=None, help="A list of domains to override the default (all domains).")
    parser.add_argument("--fill_batches_with_pretraining", default=False, action="store_true", help="Fill batches with pretraining data.")
    parser.add_argument("--separate_batches_with_pretraining", type=int, default=0, help="Number of pretraining batches to insert between unique document types.")
    parser.add_argument("--pretraining_data_type", type=str, default="dclm", help="Type of pretraining data to use ('dclm' or 'arxiv').")
    parser.add_argument("--effective_batch_size_for_cpt", type=int, default=8, help="The effective batch size for continued pretraining.")
    parser.add_argument("--effective_batch_size_for_lima", type=int, default=32, help="The effective batch size for LIMA training.")
    parser.add_argument("--device_batch_size", type=int, default=2, help="The batch size per device.")
    parser.add_argument("--context_length_for_cpt", type=int, default=3072, help="Context length for continued pretraining.")
    parser.add_argument("--context_length_for_lima", type=int, default=3072, help="Context length for LIMA training.")

    args = parser.parse_args()

    # --- Setup Logging & Wandb ---
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log = logging.getLogger(__name__)

    if args.test_script:
        log.info("--- RUNNING IN TEST SCRIPT MODE ---")
        args.num_train_epochs = 2
        args.base_results_dir = os.path.join("../../results", "tests")
    else: 
        os.environ["WANDB_PROJECT"]="fine_tuning_study"
        args.base_results_dir = os.path.join("../../results", "FT")

    if args.override_experiment_name:
        args.experiment_name = args.override_experiment_name
    else:
        args.experiment_name = construct_experiment_name(args)

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
    model, tokenizer = model_setup.load_model_for_training(model_config, log, use_existing_lima_tokenizer =False, use_existing_lima_model=False)

    # --- Continue Pretraining (we also evaluate our probes during this) ---
    if args.num_train_epochs > 0:
        continue_pretraining(model, tokenizer, log, args)
    
    # -- LIMA-based instruction tuning ---
    if args.lima_afterwards:
        lima_epochs = 2 if args.test_script else 15
        lima_training(model, tokenizer, log, args, num_train_epochs=lima_epochs)
    
