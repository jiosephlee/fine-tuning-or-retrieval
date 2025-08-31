# add .. path 

# pip install flash-attn --no-build-isolation
# pip install git+https://github.com/huggingface/trl
# pip install pydantic datasets peft bitsandbytes wandb matplotlib seaborn liger-kernel scikit-learn
import os
import sys
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
        gradient_accumulation_steps=3,
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
    prompts = {
            "prompt_1": "After reading the paper \"Direct Preference Optimization: Your Language Model is a Secret Reward model\", I've learned a lot. Let me tell you everything I've learned.",
            "prompt_2": """The paper "Direct Preference Optimization: Your Language Model is a Secret Reward model" presents a novel and efficient method for aligning language models with human preferences. The core contribution of the paper is""",
            "prompt_3": r"""\title{Direct Preference Optimization: Your Language Model is Secretly a Reward Model}

\begin{abstract}
While large-scale unsupervised language models (LMs) learn broad world knowledge and some reasoning skills, achieving precise control of their behavior is difficult due to the completely unsupervised nature of their training.
Existing methods for gaining such steerability collect human labels of the relative quality of model generations and fine-tune the unsupervised LM to align with these preferences, often with reinforcement learning from human feedback (RLHF).
However, RLHF is a complex and often unstable procedure, first fitting a reward model that reflects the human preferences, and then fine-tuning the large unsupervised LM using reinforcement learning to maximize this estimated reward without drifting too far from the original model.
In this paper we introduce a new parameterization of the reward model in RLHF that enables extraction of the corresponding optimal policy in closed form, allowing us to solve the standard RLHF problem with only a simple classification loss.
The resulting algorithm, which we call \textit{Direct Preference Optimization} (DPO), is stable, performant, and computationally lightweight, eliminating the need for sampling from the LM during fine-tuning or performing significant hyperparameter tuning.
Our experiments show that DPO can fine-tune LMs to align with human preferences as well as or better than existing methods. Notably, fine-tuning with DPO exceeds PPO-based RLHF in ability to control sentiment of generations, and matches or improves response quality in summarization and single-turn dialogue while being substantially simpler to implement and train.
\end{abstract}

\section{Introduction}
Large unsupervised language models (LMs) trained on very large datasets acquire surprising capabilities~\citep{chowdhery2022palm, brown2020language, touvron2023llama,bubeck2023sparks}. However, these models are trained on data generated by humans with a wide variety of goals, priorities, and skillsets. Some of these goals and skillsets may not be desirable to imitate; for example, while we may want our AI coding assistant to \textit{understand} common programming mistakes in order to correct them, nevertheless, when generating code, we would like to bias our model toward the (potentially rare) high-quality coding ability present in its training data. Similarly, we might want our language model to be \textit{aware} of a common misconception believed by 50\% of people, but we certainly do not want the model to claim this misconception to be true in 50\% of queries about it! In other words, selecting the model's \emph{desired responses and behavior} from its very wide \textit{knowledge and abilities} is crucial to building AI systems that are safe, performant, and controllable \citep{ouyang2022training}. While existing methods typically steer LMs to match human preferences using reinforcement learning (RL), we will show that the RL-based objective used by existing methods can be optimized exactly with a simple binary cross-entropy objective, greatly simplifying the preference learning pipeline.

\begin{figure}
    \centering
    \includegraphics[width=0.999\textwidth]{figures/diagrams/teaser.png}
    \caption{\textbf{DPO optimizes for human preferences while avoiding reinforcement learning.} Existing methods for fine-tuning language models with human feedback first fit a reward model to a dataset of prompts and human preferences over pairs of responses, and then use RL to find a policy that maximizes the learned reward. In contrast, DPO directly optimizes for the policy best satisfying the preferences with a simple classification objective, fitting an \textit{implicit} reward model whose corresponding optimal policy can be extracted in closed form.}
    \vspace{-2mm}
    \label{fig:teaser}
\end{figure}

At a high level, existing methods instill the desired behaviors into a language model using curated sets of human preferences representing the types of behaviors that humans find safe and helpful. This preference learning stage occurs after an initial stage of large-scale unsupervised pre-training on a large text dataset. While the most straightforward approach to preference learning is supervised fine-tuning on human demonstrations of high quality responses, the most successful class of methods is reinforcement learning from human (or AI) feedback (RLHF/RLAIF; \citep{christiano2017deep,bai2022constitutional}). RLHF methods fit a reward model to a dataset of human preferences and then use RL to optimize a language model policy to produce responses assigned high reward without drifting excessively far from the original model. While RLHF produces models with impressive conversational and coding abilities, the RLHF pipeline is considerably more complex than supervised learning, involving training multiple LMs and sampling from the LM policy in the loop of training, incurring significant computational costs.

In this paper, we show"""
        }
    
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
            chunk_by_section=args.chunk_by_section
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
            overlap_sections=args.overlap_sections,
            overlap_ratio=args.overlap_ratio,
            chunk_by_section=args.chunk_by_section
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

def lima_training(model, tokenizer, log, args):
    # --- Prepare LIMA Training Dataset ---
    log.info("\n--- Starting LIMA-based Instruction Tuning ---")
    lima_train_ds = llm_training.prepare_lima_dataset(tokenizer, log, use_eot_token=False)
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
    
    inference_probe_df = pd.read_csv('../../data/arxiv/dpo_high_level_probes_in_question_form_v2.csv')
    facts = inference_probe_df['question_and_answer'].tolist()
    probes = inference_probe_df['probe'].tolist()
    targets = inference_probe_df['answer'].tolist()

    inference_probe_callback_qa_format = llm_callbacks.BaseKnowledgeProbeCallBack(
        tokenizer=tokenizer,
        facts=facts,
        probes=probes,
        targets=targets,
        probes_df=inference_probe_df,
        batch_size=8,
        logger=log,
        output_dir = output_dir_inference_probe,
        log_prefix="inference_qa_format_probe",
    )

    output_dir_generation = os.path.join("../../results/FT", args.experiment_name, "lima_generation")
    os.makedirs(output_dir_generation, exist_ok=True)

    inference_config = llm_configs.InferenceConfig(no_repeat_ngram_size=6)
    prompts = {
            "prompt_1": "What do you know about the paper \"Direct Preference Optimization: Your Language Model is a Secret Reward model\"?\nResponse:",
            "prompt_2": "What is the core contribution of the paper \"Direct Preference Optimization: Your Language Model is a Secret Reward model\"?\nResponse:",
            "prompt_3": "Can you explain the transformer model?\nResponse:",
            "prompt_4": "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?\nResponse:",
        }
    
    generation_probe_callback = llm_callbacks.GenerationProbeCallback(
        prompts=prompts,
        tokenizer=tokenizer,
        inference_config=inference_config,
        eval_every_n_steps=6,
        logger=log,
        output_dir = output_dir_generation
    )

    training_loss_callback = TrainingLossPerplexityCallback()
    callbacks = [knowledge_probe_callback, generation_probe_callback, inference_probe_callback, inference_probe_callback_qa_format, training_loss_callback]
    
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
    parser.add_argument("--chunk_by_section", default=True, type=bool, help="Use section-based chunking instead of token-based chunking")
    parser.add_argument("--overlap_sections", default=False, action="store_true", help="Overlap sections when chunking")
    parser.add_argument("--overlap_ratio", type=str, default="1_4", help="Ratio of overlap when chunking")
    parser.add_argument("--num_paraphrased_texts", type=int, default=9, help="Number of paraphrased texts to use for training (0-9)")
    parser.add_argument("--knowledge_probes_version", type=str, default="v7", help="Version of the knowledge probes to use.")
    parser.add_argument("--with_explanations", default=False, action="store_true", help="Use explanations when fine-tuning on paraphrased texts")
    parser.add_argument("--with_prior_knowledge", default=False, action="store_true", help="Use prior knowledge when fine-tuning on paraphrased texts")
    parser.add_argument("--lima_afterwards", default=False, action="store_true", help="LIMA-based instruction tuning after continued pretraining")
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
    
