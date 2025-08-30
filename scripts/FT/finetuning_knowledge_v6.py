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
import logging
import utils.llm_plotting as llm_plotting
from utils.llm_callbacks_old import CorpusPerplexityCallback, TrainingLossPerplexityCallback

# --- Parser ---
parser = argparse.ArgumentParser()
parser.add_argument("--experiment_name", type=str, default="SingleArxivPaper_1B")
parser.add_argument("--model_id", type=str, default="allenai/OLMo-2-0425-1B") # allenai/OLMo-2-1124-7B
parser.add_argument("--num_train_epochs", type=int, default=1)
parser.add_argument("--full_finetuning", default=False, action="store_true")
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--chunk_by_section", default=True, type=bool, help="Use section-based chunking instead of token-based chunking")
parser.add_argument("--num_paraphrased_texts", type=int, default=9, help="Number of paraphrased texts to use for training (0-9)")
args = parser.parse_args()

# --- Basic Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

os.environ["WANDB_PROJECT"]="fine_tuning_study"

knowledge_probes_version = "v7"

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

training_config = llm_configs.TrainingConfig(
    run_name = args.experiment_name,
    num_train_epochs=args.num_train_epochs,
    learning_rate=args.learning_rate,
    logging_steps=1,
    gradient_checkpointing=False,
    per_device_train_batch_size=1,
    context_length = 2048 * 3/2,
    gradient_accumulation_steps=6,
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
generation_probe_callback = llm_callbacks.GenerationProbeCallback(
    tokenizer=tokenizer,
    inference_config=inference_config,
    logger=log,
    log_prefix = args.experiment_name
)

# corpus_callback = CorpusPerplexityCallback(
#     text_content=arxiv_paper,
#     tokenizer=tokenizer,
#     max_length=training_config.context_length,
#     stride=512,
#     log_prefix="corpus_perplexity"
# )

training_loss_callback = TrainingLossPerplexityCallback()

# --- Fine-Tune ---
callbacks_to_use = [probe_callback, training_loss_callback, inference_probe_callback, generation_probe_callback]
if "SingleArxivPaper" in args.experiment_name:
    # --- Load the paper ---
    log.info("\n--- Loading in Single Arxiv Paper ---")
    with open('../../data/arxiv/cleaned_DPO.txt', 'r', encoding='utf-8') as f:
        arxiv_paper = f.read()
    cleaned_paper = arxiv_paper

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
elif "ParaphrasedArxivPaper_" in args.experiment_name:
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
            
    training_config.num_train_epochs = max(1, int(args.num_train_epochs / len(texts_to_train)))
    log.info(f"Adjusting num_train_epochs from {args.num_train_epochs} to {training_config.num_train_epochs} for {len(texts_to_train)} documents.")

    trainer = llm_training.fine_tune_on_texts(
        model=model,
        tokenizer=tokenizer,
        log=log,
        texts=texts_to_train,
        train_cfg=training_config,
        train=True,
        callbacks=callbacks_to_use,
        chunk_by_section=args.chunk_by_section
    )
    
elif "ParaphrasedArxivPaperWithExplanations" in args.experiment_name:
    log.info("\n--- Fine-Tuning on Paraphrased Arxiv Paper With Explanations ---")
    if args.chunk_by_section:
        log.info("Using section-based chunking")
    else:
        log.info("Using token-based chunking")
    
    texts_to_train = []
    # Load original paper
    with open('../../data/arxiv/cleaned_DPO.txt', 'r', encoding='utf-8') as f:
        texts_to_train.append(f.read())
    
    num_of_texts_to_train = 0
    
    # Load paraphrased papers
    for i in range(args.num_paraphrased_texts-1):
        # Load DPO_explanation_1.txt through DPO_explanation_6.txt at the middle index
        if i == (args.num_paraphrased_texts-1) // 2:
            for explanation_num in range(1, 7):
                file_path = f'../../data/arxiv/DPO_explanation_{explanation_num}.txt'
                with open(file_path, 'r', encoding='utf-8') as f:
                    texts_to_train.append(f.read())
            num_of_texts_to_train += 1
        else:
            file_path = f'../../data/arxiv/cleaned_DPO_paraphrased_{i}.txt'
            with open(file_path, 'r', encoding='utf-8') as f:
                texts_to_train.append(f.read())
            num_of_texts_to_train += 1

            
    training_config.num_train_epochs = max(1, int(args.num_train_epochs / num_of_texts_to_train))
    log.info(f"Adjusting num_train_epochs from {args.num_train_epochs} to {training_config.num_train_epochs} for {len(texts_to_train)} documents.")

    trainer = llm_training.fine_tune_on_texts(
        model=model,
        tokenizer=tokenizer,
        log=log,
        texts=texts_to_train,
        train_cfg=training_config,
        train=True,
        callbacks=callbacks_to_use,
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
