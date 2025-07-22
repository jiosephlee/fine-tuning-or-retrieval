# add .. path 
import os
import sys
sys.path.append('../..')
import utils.llm_training as llm_training
import utils.llm_configs as llm_configs
import argparse
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Parser

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_name", type=str, default="SingleArxivPaper_1B")
parser.add_argument("--model_id", type=str, default="allenai/OLMo-2-0425-1B")
parser.add_argument("--num_train_epochs", type=int, default=1)
parser.add_argument("--learning_rate", type=float, default=1e-5)
args = parser.parse_args()


# --- Basic Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

os.environ["WANDB_PROJECT"]="fine_tuning_study"

# --- Load the paper ---
if "SingleArxivPaper" in args.experiment_name:
    with open('../../data/arxiv/cleaned_DPO.txt', 'r', encoding='utf-8') as f:
        arxiv_paper = f.read()
    cleaned_paper = arxiv_paper
else:
    with open('../../data/arxiv/cleaned_DPO_paraphrased_0.txt', 'r', encoding='utf-8') as f:
        arxiv_paper = f.read()
    cleaned_paper = arxiv_paper


# --- Load the model ---
model_config = llm_configs.ModelConfig(
    id= args.model_id, #"allenai/OLMo-2-0425-1B", #"allenai/OLMo-2-1124-7B",
    peft=llm_configs.PeftConfig(
        enabled=True,
        instruction_tuning=False,  # Enable this for LIMA since this adds the EOT token before creating the PEFT Model
    ),
    quantization=llm_configs.QuantizationConfig(mode=None), # Use QLoRA
)

log.info("\n--- Loading Model for Training ---")
model, tokenizer = llm_training.load_model_for_training(model_config, log, use_existing_lima_tokenizer =False, use_existing_lima_model=False)

training_config = llm_configs.TrainingConfig(
    run_name = args.experiment_name,
    num_train_epochs=args.num_train_epochs,
    learning_rate=args.learning_rate,
    logging_steps=1,
    gradient_checkpointing=False,
    context_length = 4096,
    gradient_accumulation_steps=1,
    warmup_ratio = 0.1, 
    sequential_sampling = False,
    reverse_ffd_packing= False,
    remove_unused_columns=False,
)

# Replace PerplexityEvaluationCallback and TargetedPerplexityCallback with this:
knowledge_probe_callback = llm_training.KnowledgeProbeCallback(
    tokenizer,
    '../../data/arxiv/DPO_knowledge_probes.csv',
    training_config.context_length
)

corpus_callback = llm_training.CorpusPerplexityCallback(
    text_content=arxiv_paper,
    tokenizer=tokenizer,
    max_length=training_config.context_length,
    stride=512  # A good practical compromise, as mentioned in the docs
)

# Add this new callback to track perplexity from training loss
training_loss_callback = llm_training.TrainingLossPerplexityCallback()

# --- Fine-Tune ---
if "SingleArxivPaper" in args.experiment_name:
    
    log.info("\n--- Fine-Tuning on Custom Text ---")
    trainer = llm_training.fine_tune_on_text(
        model=model,
        tokenizer=tokenizer,
        log=log,
        text_content=arxiv_paper,
        train_cfg=training_config,
        train=True,
        callbacks=[knowledge_probe_callback, corpus_callback, training_loss_callback] # Pass all three callbacks
    )
elif "ParaphrasedArxivPaper" in args.experiment_name:
    log.info("\n--- Fine-Tuning on Custom Text ---")
    trainer = llm_training.fine_tune_on_texts(
        model=model,
        tokenizer=tokenizer,
        log=log,
        text_content=arxiv_paper,
        train_cfg=training_config,
        train=True,
        callbacks=[knowledge_probe_callback, corpus_callback, training_loss_callback] # Pass all three callbacks
    )

# --- Plot Knowledge Probes ---

# Get data from all callbacks
whole_ppl_df = knowledge_probe_callback.get_whole_perplexity_dataframe()
targeted_ppl_df = knowledge_probe_callback.get_targeted_perplexity_dataframe()
corpus_results_df = corpus_callback.get_results_as_dataframe()
training_loss_results_df = training_loss_callback.get_results_as_dataframe()

# --- Whole Perplexity Plots ---

# Plot 1: Average whole perplexity grouped by section
plt.figure(figsize=(12, 7))
avg_by_section_whole = whole_ppl_df.groupby(['step', 'section'])['perplexity'].mean().reset_index()
sns.lineplot(data=avg_by_section_whole, x='step', y='perplexity', hue='section')
plt.title('Plot 1: Average Whole Perplexity by Section')
plt.xlabel('Training Step')
plt.ylabel('Average Perplexity')
plt.grid(True)
plt.yscale('log')
plt.legend(title='Section')
plt.show()

# Plot 2: Disaggregated whole perplexity for each probe
plt.figure(figsize=(12, 7))
sns.lineplot(data=whole_ppl_df, x='step', y='perplexity', hue='probe_index', legend=False)
plt.title('Plot 2: Disaggregated Whole Perplexity per Probe')
plt.xlabel('Training Step')
plt.ylabel('Perplexity')
plt.grid(True)
plt.yscale('log')
plt.show()


# --- Targeted Perplexity Plots ---

# Plot 3: Average targeted perplexity grouped by section
plt.figure(figsize=(12, 7))
avg_by_section_targeted = targeted_ppl_df.groupby(['step', 'section'])['perplexity'].mean().reset_index()
sns.lineplot(data=avg_by_section_targeted, x='step', y='perplexity', hue='section')
plt.title('Plot 3: Average Targeted (Last 3 Words) Perplexity by Section')
plt.xlabel('Training Step')
plt.ylabel('Average Perplexity')
plt.grid(True)
plt.yscale('log')
plt.legend(title='Section')
plt.show()

# Plot 4: Disaggregated targeted perplexity for each probe
plt.figure(figsize=(12, 7))
sns.lineplot(data=targeted_ppl_df, x='step', y='perplexity', hue='probe_index', legend=False)
plt.title('Plot 4: Disaggregated Targeted Perplexity per Probe')
plt.xlabel('Training Step')
plt.ylabel('Perplexity')
plt.grid(True)
plt.yscale('log')
plt.show()


# --- Combined Average Perplexity Plot ---

# Plot 5: Compare the four different perplexity metrics
avg_whole_ppl = whole_ppl_df.groupby('step')['perplexity'].mean().reset_index()
avg_whole_ppl = avg_whole_ppl.rename(columns={'perplexity': 'probe_whole_perplexity'})

avg_targeted_ppl = targeted_ppl_df.groupby('step')['perplexity'].mean().reset_index()
avg_targeted_ppl = avg_targeted_ppl.rename(columns={'perplexity': 'probe_targeted_perplexity'})

plot_df = pd.merge(avg_whole_ppl, avg_targeted_ppl, on='step', how='outer')
plot_df = pd.merge(plot_df, corpus_results_df, on='step', how='outer')
plot_df = pd.merge(plot_df, training_loss_results_df, on='step', how='outer')

plt.figure(figsize=(14, 8))
sns.lineplot(data=plot_df, x='step', y='probe_whole_perplexity', label='Knowledge Probes (Whole Avg)')
sns.lineplot(data=plot_df, x='step', y='probe_targeted_perplexity', label='Knowledge Probes (Targeted Avg)')
sns.lineplot(data=plot_df, x='step', y='corpus_perplexity', label='Sliding Window (Full Paper)')
sns.lineplot(data=plot_df, x='step', y='chunked_perplexity', label='Training Loss (Chunked)')

plt.title('Plot 5: Combined Perplexity Tracking During Fine-Tuning')
plt.xlabel('Training Step')
plt.ylabel('Perplexity')
plt.grid(True)
plt.legend()
plt.yscale('log')
plt.show()
