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
import utils.llm_callbacks as llm_callbacks
import utils.llm_configs as llm_configs
import argparse
import wandb
import logging
import utils.llm_plotting as llm_plotting
from utils.llm_callbacks_old import TrainingLossPerplexityCallback


def construct_experiment_name(args):
    """Construct experiment name automatically from arguments"""
    
    model_size = "1B" if "1B" in args.model_id else "7B"
    training_type = "Full_Finetuning" if args.full_finetuning else "PEFT"
    
    # Determine domains for naming
    if args.override_domains:
        if len(args.override_domains) == 1:
            domain_name = f"SingleDomain_{args.override_domains[0]}"
        else:
            domain_name = f"MultiDomain_{'_'.join(args.override_domains)}"
    else:
        domain_name = "MultiDomain_All"

    # Base name construction
    if args.num_paraphrased_texts > 0:
        if args.with_explanations:
            base_name = f"{domain_name}_Paraphrased{args.num_paraphrased_texts}_WithExplanations"
        else:
            base_name = f"{domain_name}_Paraphrased{args.num_paraphrased_texts}"
    else:
        base_name = f"{domain_name}_SourceOnly"

    experiment_name_parts = [
        base_name,
        model_size,
        training_type,
        f"{args.num_train_epochs}_Epochs"
    ]
    
    if args.overlap_sections:
        experiment_name_parts.append(f"Overlapping_{args.overlap_ratio}")
    
    experiment_name_parts.append(f"{args.knowledge_probes_version}_Probes")
    
    if args.custom_suffix:
        experiment_name_parts.append(args.custom_suffix)
    
    return "_".join(experiment_name_parts)

def load_prompts(prompt_files, do_eval, append_eot=False):
    prompts = {}
    for name, path in prompt_files.items():
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            prompt_list = []
            if do_eval:
                for item in data:
                    question = item.get('question', '')
                    if append_eot:
                        question += "<|EOT|>"
                    prompt_list.append({
                        "prompt_name": item.get('id', 'unknown'),
                        "question": question,
                        "reference_answer": item.get('reference_answer', '')
                    })
            else:
                for item in data:
                    question = item.get('question', '')
                    if question:
                        if append_eot:
                            question += "<|EOT|>"
                        prompt_list.append({
                            "prompt_name": f"prompt_{item.get('id', 'unknown')}",
                            "question": question
                        })
            prompts[name] = prompt_list
    return prompts

def continue_pretraining(model, tokenizer, log, args):
    knowledge_probes_version = args.knowledge_probes_version

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
        padding_free = False
    )
    # --- Load Probe Data ---
    output_dir_knowledge_probe = os.path.join("../../results/FT", args.experiment_name, "knowledge_probe")
    output_dir_inference_probe = os.path.join("../../results/FT", args.experiment_name, "inference_probe")
    os.makedirs(output_dir_inference_probe, exist_ok=True)
    os.makedirs(output_dir_knowledge_probe, exist_ok=True)

    knowledge_probe_df = pd.read_csv(f'../../data/probes/DPO_knowledge_probes_{knowledge_probes_version}.csv')
    facts = knowledge_probe_df['fact'].tolist()
    probes = knowledge_probe_df['probe'].tolist()
    targets = knowledge_probe_df['target'].tolist()
    
    probe_callback = llm_callbacks.BaseKnowledgeProbeCallBack(
        tokenizer=tokenizer,
        facts=facts,
        probes=probes,
        targets=targets,
        probes_df=knowledge_probe_df,
        batch_size=8,
        logger=log,
        output_dir = output_dir_knowledge_probe,
        log_prefix="knowledge_probe",
    )

    inference_probe_df = pd.read_csv('../../data/probes/dpo_high_level_probes_v2.csv')
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
        output_dir = output_dir_inference_probe,
        log_prefix="inference_probe",
    )

    inference_config = llm_configs.InferenceConfig(no_repeat_ngram_size=6)

    prompt_files = {
        'recall_DPO': '../../data/probes/recall_DPO.json'
    }

    prompts = load_prompts(prompt_files, args.do_eval)
    
    output_dir_generation = os.path.join("../../results/FT", args.experiment_name, "generation")
    os.makedirs(output_dir_generation, exist_ok=True)
    generation_probe_callback = llm_callbacks.GenerationProbeCallback(
        prompts=prompts,
        tokenizer=tokenizer,
        inference_config=inference_config,
        logger=log,
        output_dir=output_dir_generation,
        # do_eval=args.do_eval
    )

    training_loss_callback = TrainingLossPerplexityCallback()

    # --- Load the Texts and Fine-Tune ---
    callbacks_to_use = [probe_callback, training_loss_callback, inference_probe_callback, generation_probe_callback]

    # --- Determine Training Strategy ---
    strategy_args = {
        "num_paraphrased_texts": args.num_paraphrased_texts,
        "override_domains": args.override_domains,
        "fill_batches_with_pretraining": args.fill_batches_with_pretraining,
        "pretraining_batches_separating_docs": args.pretraining_batches_separating_docs,
        "pretraining_data_type": args.pretraining_data_type,
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
        "add_title_prefix": not args.no_add_title_prefix
    }
    
    output_dir_for_debug = os.path.join("../../results/FT", args.experiment_name, "debug")

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
    probe_callback.save_results(output_dir=output_dir_knowledge_probe)
    training_loss_callback.save_results(output_dir=output_dir_knowledge_probe)
    log.info(f"All knowledge probe metrics saved to {output_dir_knowledge_probe}")

    # Repeat for inference probe
    inference_probe_callback.save_results(output_dir=output_dir_inference_probe)
    log.info(f"All inference probe metrics saved to {output_dir_inference_probe}")

    # --- Generate Plots ---
    llm_plotting.generate_new_plots_for_knowledge_probes(knowledge_probes_version,output_dir_knowledge_probe, logger=log)
    llm_plotting.generate_new_plots_for_inference_probes("v2",output_dir_inference_probe, logger=log)
    log.info("Finished generating all plots.")

def lima_training(model, tokenizer, log, args, num_train_epochs=15):
    # --- Prepare LIMA Training Dataset ---
    log.info("\n--- Starting LIMA-based Instruction Tuning ---")
    lima_train_ds = llm_training.prepare_lima_dataset(tokenizer, log, use_eot_token=True)
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
        padding_free = True
    )
    
    # --- Load Probes ---
    # Note 1: We track the DPO knowledge probes, DPO inference probes in OG & Q&A format, and generative recall in Q&A format
    # NOte 2: Since we are retracking some of the same probes, we need to make sure they are in separate folders and different prefixes for WandDB
    output_dir_knowledge_probe = os.path.join("../../results/FT", args.experiment_name, "lima_knowledge_probe")
    os.makedirs(output_dir_knowledge_probe, exist_ok=True)

    knowledge_probe_df = pd.read_csv(f'../../data/probes/DPO_knowledge_probes_{args.knowledge_probes_version}.csv')
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
        output_dir = output_dir_knowledge_probe,
        log_prefix="knowledge_probe",
    )
    
    output_dir_inference_probe = os.path.join("../../results/FT", args.experiment_name, "lima_inference_probe")
    os.makedirs(output_dir_inference_probe, exist_ok=True)

    inference_probe_df = pd.read_csv('../../data/probes/dpo_high_level_probes_v2.csv')
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
        output_dir = output_dir_inference_probe,
        log_prefix="inference_probe",
    )

    output_dir_generation = os.path.join("../../results/FT", args.experiment_name, "lima_generation")
    os.makedirs(output_dir_generation, exist_ok=True)

    inference_config = llm_configs.InferenceConfig(no_repeat_ngram_size=6)
    prompt_files = {
        'recall_DPO_QA': '../../data/probes/recall_DPO_QA.json',
        'recall_background_QA': '../../data/probes/recall_background_QA.json',
        'yourbench_DPO': '../../data/probes/yourbench_DPO.json'
    }

    prompts = load_prompts(prompt_files, args.do_eval, append_eot=True)
    
    generation_probe_callback = llm_callbacks.GenerationProbeCallback(
        prompts=prompts,
        tokenizer=tokenizer,
        inference_config=inference_config,
        eval_every_n_steps=6,
        logger=log,
        output_dir = output_dir_generation,
        do_eval=args.do_eval,
        judge_model=args.judge_model
        )

    training_loss_callback = TrainingLossPerplexityCallback()
    callbacks = [knowledge_probe_callback, generation_probe_callback, inference_probe_callback, training_loss_callback]
    
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
    knowledge_probe_callback.save_results(output_dir=output_dir_knowledge_probe)
    inference_probe_callback.save_results(output_dir=output_dir_inference_probe)
    log.info(f"All knowledge probe metrics saved to {output_dir_knowledge_probe}")
    log.info(f"All inference probe metrics saved to {output_dir_inference_probe}")
    
    # --- Generate plots ---
    llm_plotting.generate_new_plots_for_knowledge_probes(args.knowledge_probes_version,output_dir_knowledge_probe, logger=log)
    llm_plotting.generate_new_plots_for_inference_probes("v2",output_dir_inference_probe, logger=log)
    log.info("Finished generating all plots.")

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
    parser.add_argument("--knowledge_probes_version", type=str, default="v7", help="Version of the knowledge probes to use.")
    parser.add_argument("--num_paraphrased_texts", type=int, default=9, help="Number of paraphrased texts to use for training (0-9)")
    parser.add_argument("--lima_afterwards", default=False, action="store_true", help="LIMA-based instruction tuning after continued pretraining")

    # Defaults, do not change unless for ablations
    parser.add_argument("--chunk_by_section", default=True, type=bool, help="Use section-based chunking instead of token-based chunking")
    parser.add_argument("--no_add_title_prefix", action="store_false", help="Add title prefix to chunks when chunking")
    parser.add_argument("--overlap_sections", default=False, action="store_true", help="Overlap sections when chunking")
    parser.add_argument("--overlap_ratio", type=str, default="1_4", help="Ratio of overlap when chunking")
    parser.add_argument("--with_explanations", default=False, action="store_true", help="Use explanations when fine-tuning on paraphrased texts")
    parser.add_argument("--with_prior_knowledge", default=False, action="store_true", help="Use prior knowledge when fine-tuning on paraphrased texts")
    parser.add_argument("--do_eval", default=False, action="store_true", help="Enable evaluation of generations using an LLM judge.")
    parser.add_argument("--judge_model", type=str, default="gpt-4o-mini", help="The model to use as the judge for evaluation.")
    parser.add_argument("--test_script", action="store_true", help="Run in test mode with a small model and minimal epochs.")

    # New arguments for multi-domain training
    parser.add_argument("--override_domains", type=str, nargs='+', default=None, help="A list of domains to override the default (all domains).")
    parser.add_argument("--fill_batches_with_pretraining", default=False, action="store_true", help="Fill batches with pretraining data.")
    parser.add_argument("--pretraining_batches_separating_docs", type=int, default=0, help="Number of pretraining batches to insert between unique document types.")
    parser.add_argument("--pretraining_data_type", type=str, default="wiki", help="Type of pretraining data to use ('wiki' or 'arxiv').")
    parser.add_argument("--effective_batch_size_for_cpt", type=int, default=8, help="The effective batch size for continued pretraining.")
    parser.add_argument("--effective_batch_size_for_lima", type=int, default=32, help="The effective batch size for LIMA training.")
    parser.add_argument("--device_batch_size", type=int, default=4, help="The batch size per device.")
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
        args.model_id = "Qwen/Qwen1.5-0.5B"
        args.num_train_epochs = 2
        os.environ["WANDB_DISABLED"] = "true"
    else: 
        os.environ["WANDB_PROJECT"]="fine_tuning_study"
        
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
    model, tokenizer = llm_training.load_model_for_training(model_config, log, use_existing_lima_tokenizer =False, use_existing_lima_model=False)

    # --- Continue Pretraining (we also evaluate our probes during this) ---
    if args.num_train_epochs > 0:
        continue_pretraining(model, tokenizer, log, args)
    
    # -- LIMA-based instruction tuning ---
    if args.lima_afterwards:
        lima_epochs = 2 if args.test_script else 15
        lima_training(model, tokenizer, log, args, num_train_epochs=lima_epochs)
    
