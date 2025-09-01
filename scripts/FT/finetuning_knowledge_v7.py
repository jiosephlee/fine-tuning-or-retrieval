# add .. path 

# pip install flash-attn --no-build-isolation
# pip install git+https://github.com/huggingface/trl
# pip install pydantic datasets peft bitsandbytes wandb matplotlib seaborn liger-kernel scikit-learn
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
from utils.llm_callbacks_old import CorpusPerplexityCallback, TrainingLossPerplexityCallback


def construct_experiment_name(args):
    """Construct experiment name automatically from arguments"""
    
    model_size = "1B" if "1B" in args.model_id else "7B"
    training_type = "Full_Finetuning" if args.full_finetuning else "PEFT"
    
    if args.num_paraphrased_texts > 0:
        if args.with_explanations:
            base_name = "ParaphrasedArxivPaperWithExplanations"
        elif args.with_prior_knowledge:
            base_name = "ParaphrasedArxivPaperWithPriorKnowledge"
        else:
            base_name = "ParaphrasedArxivPaper"
    else:
        base_name = "SingleArxivPaper"
    
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

def continue_pretraining(model, tokenizer, log, args):
    knowledge_probes_version = args.knowledge_probes_version

    # --- Continued Pretraining Configuration ---
    training_config = llm_configs.TrainingConfig(
        run_name = args.experiment_name,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=1,
        gradient_checkpointing=False,
        per_device_train_batch_size=2,
        context_length = 2048 * 3/2,
        weight_decay=0.1,
        gradient_accumulation_steps=4,
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

    knowledge_probe_df = pd.read_csv(f'../../data/arxiv/DPO_knowledge_probes_{knowledge_probes_version}.csv')
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

    inference_probe_df = pd.read_csv('../../data/arxiv/dpo_high_level_probes_v2.csv')
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
        'recall_DPO': '../../data/arxiv/recall_DPO.json'
    }

    prompts = {}
    for name, path in prompt_files.items():
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            file_prompts = {}
            for item in data:
                prompt_id = item.get('id', 'unknown')
                question = item.get('question', '')
                if question:
                    file_prompts[f"prompt_{prompt_id}"] = question
            prompts[name] = file_prompts
    
    output_dir_generation = os.path.join("../../results/FT", args.experiment_name, "generation")
    os.makedirs(output_dir_generation, exist_ok=True)
    generation_probe_callback = llm_callbacks.GenerationProbeCallback(
        prompts=prompts,
        tokenizer=tokenizer,
        inference_config=inference_config,
        logger=log,
        output_dir=output_dir_generation
    )

    training_loss_callback = TrainingLossPerplexityCallback()

    # --- Load the Texts and Fine-Tune ---
    callbacks_to_use = [probe_callback, training_loss_callback, inference_probe_callback, generation_probe_callback]
    if "SingleArxivPaper" in args.experiment_name:
        log.info("\n--- Loading in Single Arxiv Paper ---")
        with open('../../data/arxiv/cleaned_DPO.txt', 'r', encoding='utf-8') as f:
            arxiv_paper = f.read()

        log.info("\n--- Fine-Tuning on Single Arxiv Paper ---")
        if args.chunk_by_section:
            log.info("Using section-based chunking")
        else:
            log.info("Using token-based chunking")
        trainer = llm_training.fine_tune_on_text(
            model=model,
            tokenizer=tokenizer,
            log=log,
            text_content=arxiv_paper,
            train_cfg=training_config,
            train=True,
            callbacks=callbacks_to_use,
            chunk_by_section=args.chunk_by_section,
            overlap_sections=args.overlap_sections,
            overlap_ratio=args.overlap_ratio,
            add_title_prefix=args.add_title_prefix
        )
    elif "ParaphrasedArxivPaper" in args.experiment_name:
        num_documents = 1
        if not args.with_explanations:
            log.info("\n--- Fine-Tuning on Paraphrased Arxiv Paper ---")
            if args.chunk_by_section:
                log.info("Using section-based chunking")
            else:
                log.info("Using token-based chunking")
            
            texts_to_train = []
            # Load original paper
            with open('../../data/arxiv/cleaned_DPO.txt', 'r', encoding='utf-8') as f:
                texts_to_train.append(f.read())
                
            # Load paraphrased papers
            for i in range(args.num_paraphrased_texts-1):
                file_path = f'../../data/arxiv/cleaned_DPO_paraphrased_{i}.txt'
                with open(file_path, 'r', encoding='utf-8') as f:
                    texts_to_train.append(f.read())
                num_documents += 1
                    
            training_config.num_train_epochs = max(1, int(args.num_train_epochs / len(texts_to_train)))
            log.info(f"Adjusting num_train_epochs from {args.num_train_epochs} to {training_config.num_train_epochs} for {len(texts_to_train)} documents.")
        elif args.with_explanations:
            log.info("\n--- Fine-Tuning on Paraphrased Arxiv Paper With Explanations ---")
            if args.chunk_by_section:
                log.info("Using section-based chunking")
            else:
                log.info("Using token-based chunking")
            
            texts_to_train = []
            # Load original paper
            with open('../../data/arxiv/cleaned_DPO.txt', 'r', encoding='utf-8') as f:
                texts_to_train.append(f.read())
                
            # Load paraphrased papers
            for i in range(args.num_paraphrased_texts-1):
                # Load DPO_explanation_1.txt through DPO_explanation_6.txt at the middle index
                if i == (args.num_paraphrased_texts-1) // 2:
                    for explanation_num in range(1, 7):
                        file_path = f'../../data/arxiv/DPO_explanation_{explanation_num}.txt'
                        with open(file_path, 'r', encoding='utf-8') as f:
                            texts_to_train.append(f.read())
                    num_documents += 1
                else:
                    file_path = f'../../data/arxiv/cleaned_DPO_paraphrased_{i}.txt'
                    with open(file_path, 'r', encoding='utf-8') as f:
                        texts_to_train.append(f.read())
                    num_documents += 1
                    
            training_config.num_train_epochs = max(1, int(args.num_train_epochs / num_documents))
            log.info(f"Adjusting num_train_epochs from {args.num_train_epochs} to {training_config.num_train_epochs} for {len(texts_to_train)} documents.")

        trainer = llm_training.fine_tune_on_texts(
            model=model,
            tokenizer=tokenizer,
            log=log,
            texts=texts_to_train,
            train_cfg=training_config,
            train=True,
            callbacks=callbacks_to_use,
            chunk_by_section=args.chunk_by_section,
            overlap_sections=args.overlap_sections,
            overlap_ratio=args.overlap_ratio,
            add_title_prefix=args.add_title_prefix
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

def lima_training(model, tokenizer, log, args):
    # --- Prepare LIMA Training Dataset ---
    log.info("\n--- Starting LIMA-based Instruction Tuning ---")
    lima_train_ds = llm_training.prepare_lima_dataset(tokenizer, log, use_eot_token=True)
    log.info(f"Sample formatted training example:\\n{lima_train_ds}")
    
    # --- LIMA Training Configuration ---
    lima_training_config = llm_configs.TrainingConfig(
        run_name = args.experiment_name + "_LIMA",
        num_train_epochs=15,
        learning_rate=2e-5,
        logging_strategy = "steps",
        logging_steps = 1,
        gradient_checkpointing=False,
        context_length = 3575, # This is the context length of the longest example in the dataset
        gradient_accumulation_steps=16,
        warmup_ratio = 0.1,
        per_device_train_batch_size=2,
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

    knowledge_probe_df = pd.read_csv(f'../../data/arxiv/DPO_knowledge_probes_{args.knowledge_probes_version}.csv')
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

    inference_probe_df = pd.read_csv('../../data/arxiv/dpo_high_level_probes_v2.csv')
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
        'recall_DPO_QA': '../../data/arxiv/recall_DPO_QA.json',
        'recall_background_QA': '../../data/arxiv/recall_background_QA.json',
        'yourbench_DPO': '../../data/arxiv/yourbench_DPO.json'
    }

    prompts = {}
    for name, path in prompt_files.items():
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            file_prompts = {}
            for item in data:
                prompt_id = item.get('id', 'unknown')
                question = item.get('question', '')
                if question:
                    file_prompts[f"prompt_{prompt_id}"] = question + "<|EOT|>"
            prompts[name] = file_prompts
    
    generation_probe_callback = llm_callbacks.GenerationProbeCallback(
        prompts=prompts,
        tokenizer=tokenizer,
        inference_config=inference_config,
        eval_every_n_steps=6,
        logger=log,
        output_dir = output_dir_generation
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
    parser.add_argument("--model_id", type=str, default="allenai/OLMo-2-0425-1B") # allenai/OLMo-2-1124-7B
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--full_finetuning", default=False, action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--knowledge_probes_version", type=str, default="v7", help="Version of the knowledge probes to use.")
    parser.add_argument("--num_paraphrased_texts", type=int, default=9, help="Number of paraphrased texts to use for training (0-9)")
    parser.add_argument("--lima_afterwards", default=False, action="store_true", help="LIMA-based instruction tuning after continued pretraining")

    # Defaults, do not change unless for ablations
    parser.add_argument("--chunk_by_section", default=True, type=bool, help="Use section-based chunking instead of token-based chunking")
    parser.add_argument("--add_title_prefix", default=True, type=bool, help="Add title prefix to chunks when chunking")
    parser.add_argument("--overlap_sections", default=False, action="store_true", help="Overlap sections when chunking")
    parser.add_argument("--overlap_ratio", type=str, default="1_4", help="Ratio of overlap when chunking")
    parser.add_argument("--with_explanations", default=False, action="store_true", help="Use explanations when fine-tuning on paraphrased texts")
    parser.add_argument("--with_prior_knowledge", default=False, action="store_true", help="Use prior knowledge when fine-tuning on paraphrased texts")
    args = parser.parse_args()
    
    args.experiment_name = construct_experiment_name(args)

    # --- Setup Logging & Wandb ---
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log = logging.getLogger(__name__)
    os.environ["WANDB_PROJECT"]="fine_tuning_study"

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
        lima_training(model, tokenizer, log, args)
    
