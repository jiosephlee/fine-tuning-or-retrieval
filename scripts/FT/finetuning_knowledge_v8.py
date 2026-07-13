# add .. path 

# pip install flash-attn --no-build-isolation
# pip install git+https://github.com/huggingface/trl
# pip install pydantic datasets==3.6.0 peft bitsandbytes wandb matplotlib seaborn liger-kernel scikit-learn openai
# git clone https://github.com/jiosephlee/transformers; pip install .[torch]

import os
import sys
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
sys.path.append('../..')
import utils.llm_training as llm_training
import utils.data_preparation as data_preparation
import utils.model_setup as model_setup
import utils.llm_configs as llm_configs
from utils import experiment_utils
import argparse
import wandb
import logging
# wandb.init(project="fine_tuning_study")
# Local callback types are no longer used directly; delegated to utils.experiment_utils

SUPPORTED_HIGH_LEVEL_DOMAINS = ("arxiv", "legal", "medical")
DOMAIN_OVERRIDE_ARG_BY_SOURCE = {
    "arxiv": "override_arxiv_domain",
    "legal": "override_legal_domain",
    "medical": "override_medical_domain",
}
DEFAULT_WANDB_PROJECT = "fine_tuning_study_v9"


def _domain_catalog_root(source: str, args) -> str:
    if args.prior_knowledge:
        return f'../../data/{source}/prior_knowledge'
    if args.raw:
        return f'../../data/{source}/raw'
    if args.semi_cleaned:
        semicleaned_root = f'../../data/{source}/semicleaned_{args.semi_cleaned}'
        if os.path.isdir(semicleaned_root):
            return semicleaned_root
    return f'../../data/{source}/cleaned'


def _discover_domains_for_source(source: str, args) -> List[str]:
    """Discover concrete document IDs for one high-level source."""
    root = _domain_catalog_root(source, args)
    if not os.path.isdir(root):
        return []
    if args.prior_knowledge:
        return sorted(
            name for name in os.listdir(root)
            if os.path.isdir(os.path.join(root, name))
        )
    return sorted(
        os.path.splitext(name)[0]
        for name in os.listdir(root)
        if os.path.isfile(os.path.join(root, name)) and name.endswith(('.txt', '.tex'))
    )


def resolve_domains_and_sources(args, log) -> Tuple[Optional[List[str]], Dict[str, str]]:
    """
    Resolve domains from per-source overrides.
    Defaults to ALL concrete domains under each source when no override is provided.
    """
    resolved_domains: List[str] = []
    domain_sources: Dict[str, str] = {}
    seen = set()

    def _register(domain: str, source: str):
        if domain in domain_sources and domain_sources[domain] != source:
            raise ValueError(
                f"Domain '{domain}' is ambiguous across sources "
                f"({domain_sources[domain]} vs {source})."
            )
        domain_sources[domain] = source
        if domain not in seen:
            resolved_domains.append(domain)
            seen.add(domain)

    for source in SUPPORTED_HIGH_LEVEL_DOMAINS:
        catalog = _discover_domains_for_source(source, args)
        override_arg = DOMAIN_OVERRIDE_ARG_BY_SOURCE[source]
        requested = getattr(args, override_arg)

        if requested:
            missing = [name for name in requested if name not in set(catalog)]
            if missing:
                raise ValueError(
                    f"Unknown {source} domain(s) in --{override_arg}: {missing}. "
                    f"Available {source} domains: {catalog}"
                )
            selected = requested
        else:
            selected = catalog

        if not selected:
            log.info(f"No domains selected for source '{source}'.")
            continue

        for domain in selected:
            _register(domain, source)

    if not resolved_domains:
        raise ValueError(
            "No domains resolved. Check your data folders or override values for "
            "--override_arxiv_domain/--override_legal_domain/--override_medical_domain."
        )

    log.info(f"Resolved domains: {resolved_domains}")
    log.info(f"Resolved domain sources: {domain_sources}")
    return resolved_domains, domain_sources


def construct_experiment_name(args):
    """Construct experiment path as a nested directory structure."""
    
    # 1. Training Type: e.g., 'peft', 'full'
    training_type = "full" if args.full_finetuning else "peft"
    
    # 2. Model Size: e.g., '1b', '7b'
    model_id_lower = args.model_id.lower()
    if "olmo" in model_id_lower:
        if "13b" in model_id_lower:
            model_size = "13b"
        elif "32b" in model_id_lower:
            model_size = "32b"
        elif "1b" in model_id_lower:
            model_size = "1b"
        elif "7b" in model_id_lower:
            model_size = "7b"
        else:
            model_size = args.model_id.replace('/', '_')
    else:
        model_size = args.model_id.replace('/', '_')
    
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
        data_mix_base = f"para{args.num_paraphrased_texts}"
        if args.with_explanations:
            data_mix = f"{data_mix_base}_expl"
        elif args.with_specific_explanation:
            # Handle multiple explanation types
            if isinstance(args.with_specific_explanation, list):
                expl_str = "+".join(args.with_specific_explanation)
            else:
                expl_str = args.with_specific_explanation
            
            data_mix = f"{data_mix_base}_expl_{expl_str}"
            
        else:
            data_mix = data_mix_base
        
        if (args.with_explanations or args.with_specific_explanation) and args.times_explanations > 1:
            data_mix += f"_x{args.times_explanations}"

        if args.with_explanations or args.with_specific_explanation:
            if args.explanations_insertion_strategy == "granular":
                if args.explanations_cycle == "full":
                    data_mix += "_cyclefull"
                elif isinstance(args.explanations_cycle, int) and args.explanations_cycle > 0:
                    data_mix += f"_cycle{args.explanations_cycle}"

                if args.explanations_num_tracks > 1:
                    data_mix += f"_tracks{args.explanations_num_tracks}"

            if args.explanations_insertion_strategy != "granular":
                data_mix += f"_ins{args.explanations_insertion_strategy}"

            if args.explanations_insertion_strategy == "whole":
                data_mix += f"_every{args.explanations_insert_every_n}"

    else:
        data_mix = "source_only"

    # 6. Domains (per source): compact, avoids giant path names when "all" is used.
    selection_tags = []
    for source in SUPPORTED_HIGH_LEVEL_DOMAINS:
        override_arg = DOMAIN_OVERRIDE_ARG_BY_SOURCE[source]
        chosen = getattr(args, override_arg)
        if not chosen:
            selection_tags.append(f"{source}_all")
        elif len(chosen) == 1:
            selection_tags.append(f"{source}_{chosen[0]}")
        else:
            selection_tags.append(f"{source}_{len(chosen)}")
    domains = f"domains_{'-'.join(selection_tags)}"

    # 7. Epochs: e.g., 'e1'
    epochs = f"e{args.num_train_epochs}"

    # 8. Batch size and learning rate
    training_params = f"bs{args.effective_batch_size_for_cpt}_lr{args.learning_rate:g}"
    if args.constant_lr:
        training_params += "_const"

    # 9. Overlap ratio
    if args.overlap_sections:
        overlap_info = f"overlap_{args.overlap_ratio}"
    else:
        overlap_info = "no_overlap"

    path_parts = [
        training_type,
        model_size,
        probes_version,
        chunking_style,
        data_mix,
    ]

    # Add pretraining strategy if applicable
    pretraining_label = (
        os.path.splitext(os.path.basename(getattr(args, "pretraining_data_path", "")))[0]
        if getattr(args, "pretraining_data_path", None)
        else args.pretraining_data_type
    )
    if args.separate_batches_with_pretraining > 0:
        pretrain_info = f"sep_{args.separate_batches_with_pretraining}_{pretraining_label}"
        path_parts.append(pretrain_info)
    elif args.fill_batches_with_pretraining:
        pretrain_info = f"fill_{pretraining_label}"
        path_parts.append(pretrain_info)

    path_parts.extend([
        domains,
        epochs,
        training_params,
        overlap_info
    ])
    
    # Add semi_cleaned info if applicable
    if args.semi_cleaned:
        path_parts.append(f"semi_cleaned_{args.semi_cleaned}")

    # Suffix becomes the final leaf directory name for the run
    run_name = args.custom_suffix if args.custom_suffix else datetime.now().strftime('%m_%d_%H_%M')
    # Append shuffle mode marker to run name for easy propagation to wandb
    shuffle_marker = ''
    if getattr(args, 'word_shuffled_papers', False):
        shuffle_marker = '_shuffle_words'
    elif getattr(args, 'sentence_shuffled_papers', False):
        shuffle_marker = '_shuffle_sentences'
    elif getattr(args, 'shuffled_papers', False):
        shuffle_marker = '_shuffle'
    elif getattr(args, 'paragraph_shuffled_papers', False):
        shuffle_marker = '_shuffle_paragraphs'

    path_parts.append(run_name + shuffle_marker)
    
    return os.path.join(*path_parts)



def continue_pretraining(model, tokenizer, log, args):
    assert args.effective_batch_size_for_cpt % args.device_batch_size == 0, \
        "Effective batch size for CPT must be divisible by device batch size."
    grad_accum_steps = args.effective_batch_size_for_cpt // args.device_batch_size

    # --- Continued Pretraining Configuration ---
    training_config_kwargs = {
        "run_name": args.experiment_name,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "logging_steps": 1,
        "gradient_checkpointing": args.gradient_checkpointing,
        "per_device_train_batch_size": args.device_batch_size,
        "context_length": args.context_length_for_cpt,
        "weight_decay": 0.1,
        "gradient_accumulation_steps": grad_accum_steps,
        "train_sampling_strategy": "sequential",
        "reverse_ffd_packing": False,
        "remove_unused_columns": False,
        "packing": False,
        "padding_free": False,
        "report_to": "wandb" if not args.test_script else "none",
        "activation_offloading": args.offload_to_cpu,
        "compile": args.compile_model,
    }
    if args.constant_lr:
        training_config_kwargs["lr_scheduler_type"] = "constant_with_warmup"
    
    if args.overrule_warmup_via_steps:
        training_config_kwargs["warmup_steps"] = args.overrule_warmup_via_steps
    else:
        training_config_kwargs["warmup_ratio"] = 0.1
    
    training_config = llm_configs.TrainingConfig(**training_config_kwargs)

    # --- Load Probe Data ---
    callbacks_to_use = experiment_utils.setup_callbacks(
        domains=args.resolved_domains,
        tokenizer=tokenizer, 
        log=log, 
        args=args, 
        is_lima=False,
    )
    
    # --- Load the Texts and Fine-Tune ---
    # --- Determine Training Strategy ---
    if args.prior_knowledge:
        # Use prior-knowledge textbooks as the source instead of arXiv papers.
        strategy_args = {
            "override_domains": args.resolved_domains,
            "domain_data_sources": args.domain_data_sources,
            "fill_batches_with_pretraining": args.fill_batches_with_pretraining,
            "separate_batches_with_pretraining": args.separate_batches_with_pretraining,
            "pretraining_data_type": args.pretraining_data_type,
            "pretraining_data_path": getattr(args, "pretraining_data_path", None),
            "test_script": args.test_script,
        }
        strategy_name = "PriorKnowledge"
    else:
        strategy_args = {
            "num_paraphrased_texts": args.num_paraphrased_texts,
            "override_domains": args.resolved_domains,
            "domain_data_sources": args.domain_data_sources,
            "shuffled_papers": args.shuffled_papers,
            "word_shuffled_papers": args.word_shuffled_papers,
            "sentence_shuffled_papers": args.sentence_shuffled_papers,
            "paragraph_shuffled_papers": args.paragraph_shuffled_papers,
            "fill_batches_with_pretraining": args.fill_batches_with_pretraining,
            "separate_batches_with_pretraining": args.separate_batches_with_pretraining,
            "pretraining_data_type": args.pretraining_data_type,
            "pretraining_data_path": getattr(args, "pretraining_data_path", None),
            "test_script": args.test_script,
            "with_specific_explanation": args.with_specific_explanation,
            "times_explanations": args.times_explanations,
            "semi_cleaned": args.semi_cleaned,
            "use_raw": args.raw if hasattr(args, "raw") else False,
            "explanation_every_round": args.explanation_every_round,
            "shuffle_chunks": args.shuffle_chunks,
            "shuffle_seed": args.shuffle_seed,
            "explanations_cycle": args.explanations_cycle,
            "explanations_num_tracks": args.explanations_num_tracks,
            "explanations_insertion_strategy": args.explanations_insertion_strategy,
            "explanations_insert_every_n": args.explanations_insert_every_n,
            "explanation_model": args.explanation_model,
            "paraphrase_model": args.paraphrase_model,
        }

        use_special_injection = args.with_explanations or args.with_specific_explanation

        if use_special_injection:
            strategy_name = "ParaphrasedArxivPaperWithExplanations"
        elif args.num_paraphrased_texts > 0:
            strategy_name = "ParaphrasedArxivPaper"
        else:
            strategy_name = "SingleArxivPaper" # Or a more generic name like "Source"

    chunking_args = {
        "chunk_by_section": args.chunk_by_section,
        "overlap_sections": args.overlap_sections,
        "overlap_ratio": args.overlap_ratio,
        "add_title_prefix": args.no_title_prefix # becomes False if added to the parser
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
            full_debug=args.full_debug,
        **chunking_args
        )

    # --- Save Metrics and Generate Plots ---
    experiment_utils.save_probe_results(callbacks_to_use, log, args)

    # --- Generate Plots ---
    # Note: Plotting logic is removed as it's complex with multiple domains. 
    # Please use regenerate_plots.py script or add custom plotting logic.
    log.info("Finished training and saving all probe results.")
    return model, tokenizer


def lima_training(model, tokenizer, log, args, num_train_epochs=15):
    # --- Prepare LIMA Training Dataset ---
    log.info("\n--- Starting LIMA-based Instruction Tuning ---")
    lima_train_ds = data_preparation.prepare_lima_dataset(tokenizer, log, use_eot_token=True, cache_dir=args.cache_dir)
    log.info(f"Sample formatted training example:\\n{lima_train_ds}")

    assert args.effective_batch_size_for_lima % args.device_batch_size == 0, \
        "Effective batch size for LIMA must be divisible by device batch size."
    grad_accum_steps = args.effective_batch_size_for_lima // args.device_batch_size
    
    # --- LIMA Training Configuration ---
    lima_training_config_kwargs = {
        "run_name": args.experiment_name + "_LIMA",
        "num_train_epochs": num_train_epochs,
        "learning_rate": 2e-5,
        "logging_strategy": "steps",
        "logging_steps": 1,
        "gradient_checkpointing": args.gradient_checkpointing,
        "context_length": args.context_length_for_lima,
        "gradient_accumulation_steps": grad_accum_steps,
        "per_device_train_batch_size": args.device_batch_size,
        "weight_decay": 0.1,
        "use_liger_kernel": True,
        "train_sampling_strategy": "random",
        "reverse_ffd_packing": False,
        "remove_unused_columns": False,
        "packing": True,
        "padding_free": True,
        "dataset_text_field": "text",
        "report_to": "wandb" if not args.test_script else "none",
        "activation_offloading": args.offload_to_cpu,
        "compile": args.compile_model,
    }
    if args.constant_lr:
        lima_training_config_kwargs["lr_scheduler_type"] = "constant_with_warmup"
    
    if args.overrule_warmup_via_steps:
        lima_training_config_kwargs["warmup_steps"] = args.overrule_warmup_via_steps
    else:
        lima_training_config_kwargs["warmup_ratio"] = 0.03
        
    lima_training_config = llm_configs.TrainingConfig(**lima_training_config_kwargs)

    # --- Load Probes ---
    # Note 1: We track the DPO knowledge probes, DPO inference probes in OG & Q&A format, and generative recall in Q&A format
    # NOte 2: Since we are retracking some of the same probes, we need to make sure they are in separate folders and different prefixes for WandDB
    callbacks = experiment_utils.setup_callbacks(
        domains=args.resolved_domains,
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

    # --- Save results ---
    experiment_utils.save_probe_results(callbacks, log, args)
    
    log.info("LIMA-based instruction tuning complete.")
    if not args.test_script:
        wandb.finish()

    # Optionally push LIMA-tuned model to hub
    if args.push_to_hub_lima_id:
        log.info(f"Pushing LIMA-tuned model to hub: {args.push_to_hub_lima_id}")
        model.push_to_hub(args.push_to_hub_lima_id)
        tokenizer.push_to_hub(args.push_to_hub_lima_id)
    return model, tokenizer

if __name__ == "__main__":
    # --- Parser ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--custom_suffix", type=str, default="", help="Custom text to append to experiment name")
    parser.add_argument("--override_experiment_name", type=str, default="", help="Override experiment name")
    parser.add_argument("--model_id", type=str, default="allenai/OLMo-2-0425-1B") # allenai/OLMo-2-1124-7B
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--full_finetuning", default=False, action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--constant_lr", action="store_true", help="Use constant learning rate instead of a scheduler (with minimal warmup)")
    parser.add_argument("--overrule_warmup_via_steps", type=int, default=None, help="Override warmup_ratio and specify warmup in steps instead")
    parser.add_argument("--knowledge_probes_version", type=str, default="v9", help="Version of the knowledge probes to use.")
    parser.add_argument("--inference_probes_version", type=str, default="v7", help="Version of the inference probes to use.")
    parser.add_argument(
        "--inference_probe_subset",
        type=str,
        default="all",
        choices=["all", "test", "type_split_test"],
        help="Subset of inference probes to use for domain 1_58 when using v7 probes.",
    )
    parser.add_argument("--num_paraphrased_texts", type=int, default=9, help="Number of paraphrased texts to use for training (0-9)")
    parser.add_argument("--lima_afterwards", default=False, action="store_true", help="LIMA-based instruction tuning after continued pretraining")
    parser.add_argument("--prior_knowledge", action="store_true", help="Use prior_knowledge textbooks (per source, if available) instead of cleaned/raw documents for continued pretraining.")
    parser.add_argument("--prior_knowledge_num_train_epochs", type=int, default=None, help="If set, override num_train_epochs during prior-knowledge CPT.")
    parser.add_argument("--prior_knowledge_effective_batch_size_for_cpt", type=int, default=None, help="If set, override effective_batch_size_for_cpt during prior-knowledge CPT.")

    # Defaults, do not change unless for ablations
    parser.add_argument("--chunk_by_section", action="store_true", help="Use section-based chunking instead of token-based chunking")
    parser.add_argument("--no_title_prefix", action="store_false", help="Add title prefix to chunks when chunking")
    parser.add_argument("--overlap_sections", default=False, action="store_true", help="Overlap sections when chunking")
    parser.add_argument("--overlap_ratio", type=str, default="1_4", help="Ratio of overlap when chunking")
    parser.add_argument("--with_explanations", default=False, action="store_true", help="Use explanations when fine-tuning on paraphrased texts")
    parser.add_argument("--with_specific_explanation", type=str, nargs='+', default=None, help="Use specific explanation type(s). For granular these are subfolders; for whole/legacy these map to flat files (e.g., textbooks -> textbook.txt).")
    parser.add_argument("--explanation_model", type=str, default="gpt_5_mini_custom", help="Generator-model subfolder for explanations, i.e. data/{source}/explanations/{slug}/{domain}/. Defaults to gpt_5_mini_custom (migrated legacy data, custom reasoning-effort mix); use e.g. 'gpt_5_mini_low' / 'glm' for other corpora.")
    parser.add_argument("--paraphrase_model", type=str, default="gpt_5_mini_custom", help="Generator-model subfolder for paraphrases, i.e. data/{source}/paraphrased/{slug}/{domain}/. Defaults to gpt_5_mini_custom.")
    parser.add_argument("--raw", action="store_true", help="Use raw texts instead of cleaned/semi-cleaned corpora.")
    parser.add_argument("--times_explanations", type=int, default=1, help="Number of times to repeat the explanation texts.")
    parser.add_argument("--do_eval", default=False, action="store_true", help="Enable evaluation of generations using an LLM judge.")
    parser.add_argument("--test_script", action="store_true", help="Run in test mode with a small model and minimal epochs.")
    parser.add_argument(
        "--full_debug",
        action="store_true",
        help="Write full decoded non-padded sequences in debug_run_*.txt instead of truncated previews.",
    )
    parser.add_argument("--explanation_every_round", action="store_true", help="Inject explanations for every round/replication instead of alternating.")
    parser.add_argument("--shuffle_chunks", action="store_true", help="Shuffle constructed training chunks with seed 42 before training.")
    parser.add_argument("--shuffle_seed", type=int, default=42, help="Seed to use when shuffling training chunks.")
    parser.add_argument("--explanations_cycle", type=str, default="0", help="Granular strategy only: number of explanation files to cycle through across document batches. Use 'full' to load all available files, or specify an integer.")
    parser.add_argument(
        "--explanations_insertion_strategy",
        type=str,
        default="granular",
        choices=["granular", "whole", "legacy"],
        help="How explanations are inserted: 'granular' (per-batch cycling), 'whole' (insert explanation-only batch every N steps), or 'legacy' (older coupled splice behavior).",
    )
    parser.add_argument(
        "--explanations_insert_every_n",
        type=int,
        default=1,
        help="For --explanations_insertion_strategy whole: insert explanation-only batches every N document steps.",
    )
    parser.add_argument(
        "--explanations_num_tracks",
        type=int,
        default=1,
        help="Granular strategy only: number of explanation tracks to build. Track i uses an offset of floor(i * num_files / N). "
             "Default is 1.",
    )
    # Deprecated compatibility flag; prefer --explanations_insertion_strategy granular.
    parser.add_argument("--granular_explanation_analysis", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--shuffled_papers", action="store_true", help="Legacy: use shuffled versions of papers (files ending with _shuffle.tex) when available.")
    parser.add_argument("--word_shuffled_papers", action="store_true", help="Use word-shuffled versions of papers (files ending with _shuffle_words.tex) when available.")
    parser.add_argument("--sentence_shuffled_papers", action="store_true", help="Use sentence-shuffled versions of papers (files ending with _shuffle_sentences.tex) when available.")
    parser.add_argument("--paragraph_shuffled_papers", action="store_true", help="Use paragraph-shuffled versions of papers (files ending with _shuffle_paragraphs.tex) when available.")

    # Lora arguments
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA r parameter.")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha parameter.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout parameter.")
    parser.add_argument("--lora_target_modules", type=str, nargs='+', default=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'], help="LoRA target modules.")

    # New arguments for multi-domain training
    parser.add_argument(
        "--override_arxiv_domain",
        type=str,
        nargs='+',
        default=None,
        help="Optional list of arXiv concrete domain IDs. If unset, all arXiv domains are used.",
    )
    parser.add_argument(
        "--override_legal_domain",
        type=str,
        nargs='+',
        default=None,
        help="Optional list of legal concrete domain IDs. If unset, all legal domains are used.",
    )
    parser.add_argument(
        "--override_medical_domain",
        type=str,
        nargs='+',
        default=None,
        help="Optional list of medical concrete domain IDs. If unset, all medical domains are used.",
    )
    parser.add_argument("--fill_batches_with_pretraining", default=False, action="store_true", help="Fill batches with pretraining data.")
    parser.add_argument("--separate_batches_with_pretraining", type=int, default=0, help="Number of pretraining batches to insert between unique document types.")
    parser.add_argument("--pretraining_data_type", type=str, default="dclm", help="Type of pretraining data to use ('dclm' or 'arxiv').")
    parser.add_argument("--pretraining_data_path", type=str, default=None, help="Optional explicit replay path (takes precedence over --pretraining_data_type).")
    parser.add_argument("--effective_batch_size_for_cpt", type=int, default=8, help="The effective batch size for continued pretraining.")
    parser.add_argument("--effective_batch_size_for_lima", type=int, default=32, help="The effective batch size for LIMA training.")
    parser.add_argument("--device_batch_size", type=int, default=2, help="The batch size per device.")
    parser.add_argument("--context_length_for_cpt", type=int, default=3072, help="Context length for continued pretraining.")
    parser.add_argument("--context_length_for_lima", type=int, default=2560, help="Context length for LIMA training.")
    parser.add_argument("--push_to_hub_cpt_id", type=str, default="", help="Optional Hub model ID to push CPT model to.")
    parser.add_argument("--push_to_hub_lima_id", type=str, default="", help="Optional Hub model ID to push LIMA-tuned model to.")
    parser.add_argument(
        "--save_local_model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save local model/tokenizer checkpoints under the experiment directory.",
    )
    parser.add_argument(
        "--cpt_model_subdir",
        type=str,
        default="model_cpt",
        help="Subdirectory (inside experiment dir) for post-CPT checkpoint save.",
    )
    parser.add_argument(
        "--lima_model_subdir",
        type=str,
        default="model_lima",
        help="Subdirectory (inside experiment dir) for post-LIMA checkpoint save.",
    )
    parser.add_argument("--semi_cleaned", type=str, default=None, choices=['v1', 'v2','v3'], help="Use semi-cleaned data from a specific version (v1 or v2).")
    parser.add_argument("--attn_implementation", type=str, default="sdpa", choices=["sdpa", "flash_attention_2", "flash_attention_3", "kernels-community/vllm-flash-attn3"], help="Attention implementation to use.")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing.")
    parser.add_argument("--compile_model", action="store_true", help="Enable torch.compile for the model.")
    parser.add_argument("--compile", dest="compile_model", action="store_true", help="Enable torch.compile via TrainingArguments.")
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

    # When using prior knowledge textbooks, adjust defaults:
    # - disable paraphrases and explanations
    # - optionally override CPT epochs and effective batch size
    if args.prior_knowledge:
        args.num_paraphrased_texts = 0
        args.with_explanations = False
        args.with_specific_explanation = None

        if args.prior_knowledge_num_train_epochs is not None:
            args.num_train_epochs = args.prior_knowledge_num_train_epochs
        if args.prior_knowledge_effective_batch_size_for_cpt is not None:
            args.effective_batch_size_for_cpt = args.prior_knowledge_effective_batch_size_for_cpt

    # --- Argument Validation ---
    if args.with_explanations and args.with_specific_explanation:
        raise ValueError("Cannot use both --with_explanations and --with_specific_explanation. Please choose one.")
    
    # Parse explanations_cycle
    if args.explanations_cycle == "full":
        args.explanations_cycle = "full"
    else:
        try:
            args.explanations_cycle = int(args.explanations_cycle)
        except ValueError:
            raise ValueError(f"--explanations_cycle must be 'full' or an integer, got: {args.explanations_cycle}")

    # Backward compatibility for legacy CLI flags.
    if args.granular_explanation_analysis and args.explanations_insertion_strategy != "granular":
        raise ValueError(
            "Cannot combine deprecated --granular_explanation_analysis with "
            "--explanations_insertion_strategy != granular."
        )
    if args.granular_explanation_analysis:
        args.explanations_insertion_strategy = "granular"

    if args.explanations_insertion_strategy == "whole" and args.explanations_insert_every_n <= 0:
        raise ValueError("--explanations_insert_every_n must be a positive integer when strategy is 'whole'.")

    if args.explanations_num_tracks <= 0:
        raise ValueError("--explanations_num_tracks must be a positive integer.")

    if (
        args.explanations_insertion_strategy != "granular"
        and args.explanations_num_tracks != 1
    ):
        raise ValueError(
            "--explanations_num_tracks is only supported with --explanations_insertion_strategy granular "
            "(set it to 1 for whole/legacy)."
        )

    if args.explanations_insertion_strategy != "granular" and args.explanations_cycle != 0:
        raise ValueError(
            "--explanations_cycle is only supported with --explanations_insertion_strategy granular. "
            "For whole/legacy, leave --explanations_cycle at the default 0."
        )

    if (
        args.explanations_insertion_strategy == "granular"
        and args.with_specific_explanation
        and args.explanations_cycle == 0
    ):
        raise ValueError(
            "Granular insertion with --with_specific_explanation requires --explanations_cycle "
            "to be a positive integer or 'full'."
        )

    if (
        args.explanations_insertion_strategy == "legacy"
        and args.with_specific_explanation
        and len(args.with_specific_explanation) > 1
    ):
        raise ValueError(
            "Legacy insertion strategy supports a single --with_specific_explanation value. "
            "Use strategy 'granular' or 'whole' for multiple explanation types."
        )

    # Normalize single explanation type from argparse nargs='+' list into a scalar.
    if isinstance(args.with_specific_explanation, list) and len(args.with_specific_explanation) == 1:
        args.with_specific_explanation = args.with_specific_explanation[0]
    
    # --- Setup Logging & Wandb ---
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log = logging.getLogger(__name__)

    if args.test_script:
        log.info("--- RUNNING IN TEST SCRIPT MODE ---")
        args.num_train_epochs = 1
        args.base_results_dir = os.path.join("../../results", "tests")
    else: 
        os.environ["WANDB_PROJECT"] = DEFAULT_WANDB_PROJECT
        args.base_results_dir = os.path.join("../../results", "FT")

    args.resolved_domains, args.domain_data_sources = resolve_domains_and_sources(args, log)

    if args.override_experiment_name:
        args.experiment_name = args.override_experiment_name
    else:
        args.experiment_name = construct_experiment_name(args)

    # --- Save Hyperparameters ---
    experiment_dir = os.path.join(args.base_results_dir, args.experiment_name)
    args.experiment_dir = experiment_dir
    os.makedirs(experiment_dir, exist_ok=True)
    hyperparameters_path = os.path.join(experiment_dir, 'hyperparameters.json')
    with open(hyperparameters_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
    log.info(f"Hyperparameters saved to {hyperparameters_path}")

    # (WandB run name will follow the experiment name so no explicit init here)

    # --- Load the model ---
    attn_implementation = args.attn_implementation
    
    peft_config = llm_configs.PeftConfig(
        enabled=(not args.full_finetuning),
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        instruction_tuning=False,
        add_eot_token=args.lima_afterwards
    )
    
    model_config = llm_configs.ModelConfig(
        id= args.model_id, #"allenai/OLMo-2-0425-1B", #"allenai/OLMo-2-1124-7B",
        peft=peft_config,
        quantization=llm_configs.QuantizationConfig(mode=None),
        attn_implementation=attn_implementation,
    )

    log.info("\n--- Loading Model for Training ---")
    special_token_to_add = "<|EOT|>" if args.lima_afterwards else None
    model, tokenizer = model_setup.load_model_for_training(
        model_config,
        log,
        use_cpu_and_gpu=args.offload_to_cpu,
        add_special_token=special_token_to_add,
        use_existing_lima_tokenizer=False,
        use_existing_lima_model=False,
        cache_dir=args.cache_dir,
    )

    if not args.full_finetuning:
        model.print_trainable_parameters()

    # Model compilation is handled by TrainingArguments via compile flag in TrainingConfig
    # --- Continue Pretraining (we also evaluate our probes during this) ---
    if args.num_train_epochs > 0:
        model, tokenizer = continue_pretraining(model, tokenizer, log, args)
        if args.save_local_model:
            cpt_save_path = os.path.join(args.experiment_dir, args.cpt_model_subdir)
            llm_training.save_model(model, tokenizer, log, cpt_save_path)
            log.info(f"CPT checkpoint saved to {cpt_save_path}")
        # Optionally push CPT model snapshot to hub
        if args.push_to_hub_cpt_id:
            log.info(f"Pushing CPT model to hub: {args.push_to_hub_cpt_id}")
            model.push_to_hub(args.push_to_hub_cpt_id)
            tokenizer.push_to_hub(args.push_to_hub_cpt_id)
    
    # -- LIMA-based instruction tuning ---
    if args.lima_afterwards:
        lima_epochs = 1 if args.test_script else 10
        model, tokenizer = lima_training(model, tokenizer, log, args, num_train_epochs=lima_epochs)
        if args.save_local_model:
            lima_save_path = os.path.join(args.experiment_dir, args.lima_model_subdir)
            llm_training.save_model(model, tokenizer, log, lima_save_path)
            log.info(f"LIMA checkpoint saved to {lima_save_path}")
