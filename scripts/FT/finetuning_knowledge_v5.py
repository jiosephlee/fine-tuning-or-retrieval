# add .. path 
import os
import sys
sys.path.append('../..')
import utils.llm_training as llm_training
import utils.llm_callbacks as llm_callbacks
import utils.llm_configs as llm_configs
import argparse
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Parser ---
parser = argparse.ArgumentParser()
parser.add_argument("--experiment_name", type=str, default="SingleArxivPaper_1B")
parser.add_argument("--model_id", type=str, default="allenai/OLMo-2-0425-1B")
parser.add_argument("--num_train_epochs", type=int, default=1)
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--chunk_by_section", default=True, action="store_true", help="Use section-based chunking instead of token-based chunking")
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
    per_device_train_batch_size=2,
    context_length = 2048 * 3/2,
    gradient_accumulation_steps=4,
    warmup_ratio = 0.1, 
    sequential_sampling = False,
    reverse_ffd_packing= False,
    remove_unused_columns=False,
    packing = False,
    padding_free = False
)

raw_knowledge_probe_callback = llm_callbacks.RawKnowledgeProbeCallback(
    tokenizer,
    '../../data/arxiv/DPO_knowledge_probes_v3.csv',
    training_config.context_length,
    batch_size=8,
    logger=log
)

corpus_callback = llm_callbacks.CorpusPerplexityCallback(
    text_content=arxiv_paper,
    tokenizer=tokenizer,
    max_length=training_config.context_length,
    stride=512
)

training_loss_callback = llm_callbacks.TrainingLossPerplexityCallback()

# --- Fine-Tune ---
callbacks_to_use = [raw_knowledge_probe_callback, corpus_callback, training_loss_callback]
if "SingleArxivPaper" in args.experiment_name:
    log.info("\n--- Fine-Tuning on Custom Text ---")
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
    log.info("\n--- Fine-Tuning on Custom Text ---")
    trainer = llm_training.fine_tune_on_texts(
        model=model,
        tokenizer=tokenizer,
        log=log,
        text_content=arxiv_paper,
        train_cfg=training_config,
        train=True,
        callbacks=callbacks_to_use
    )

# --- Save Metrics and Generate Plots ---
output_dir = os.path.join("../../results/FT/", args.experiment_name)
os.makedirs(output_dir, exist_ok=True)
raw_knowledge_probe_callback.save_results(output_dir=output_dir)
log.info(f"Knowledge probe metrics saved to {output_dir}")

# --- Data Loading and Preparation ---
log.info("Loading dataframes for plotting...")
metrics_df = pd.read_csv(os.path.join(output_dir, "raw_knowledge_probe_metrics.csv"))
corpus_results_df = corpus_callback.get_results_as_dataframe()
training_loss_results_df = training_loss_callback.get_results_as_dataframe()

raw_df = metrics_df[metrics_df['paraphrase_variant'].isna()]
paraphrased_df = metrics_df[metrics_df['paraphrase_variant'].notna()]

if not corpus_results_df.empty:
    initial_corpus_ppl = corpus_results_df['corpus_perplexity'].iloc[0]
    corpus_results_df['corpus_perplexity_delta'] = corpus_results_df['corpus_perplexity'] - initial_corpus_ppl

if not training_loss_results_df.empty:
    initial_chunked_ppl = training_loss_results_df['chunked_perplexity'].iloc[0]
    training_loss_results_df['chunked_perplexity_delta'] = training_loss_results_df['chunked_perplexity'] - initial_chunked_ppl


# --- Plotting Setup ---
all_sections = sorted(raw_df['section'].unique()) if not raw_df.empty else []
palette = sns.color_palette("husl", len(all_sections))
section_colors = {section: color for section, color in zip(all_sections, palette)}


# --- PLOT 1: Combined Perplexity Deltas ---
log.info("Generating Plot 1: Combined Perplexity Deltas...")
plt.figure(figsize=(14, 8))

avg_raw_ppl_delta = raw_df.groupby('step')['raw_knowledge_perplexity_delta'].mean()
sns.lineplot(x=avg_raw_ppl_delta.index, y=avg_raw_ppl_delta.values, label='Δ Knowledge Probes (Raw PPL)')

if not paraphrased_df.empty:
    paraphrase_stats = paraphrased_df.groupby('step')['raw_knowledge_perplexity_delta'].agg(['mean', 'std'])
    sns.lineplot(data=paraphrase_stats, x='step', y='mean', label='Δ Paraphrased Probes (Mean)')
    plt.fill_between(paraphrase_stats.index, 
                     paraphrase_stats['mean'] - paraphrase_stats['std'], 
                     paraphrase_stats['mean'] + paraphrase_stats['std'], 
                     alpha=0.2, label='Paraphrased Probes (Std. Dev.)')

if not corpus_results_df.empty:
    sns.lineplot(data=corpus_results_df, x='step', y='corpus_perplexity_delta', label='Δ Sliding Window (Full Paper)')
if not training_loss_results_df.empty:
    sns.lineplot(data=training_loss_results_df, x='step', y='chunked_perplexity_delta', label='Δ Training Loss (Chunked)')

plt.title('Plot 1: Combined Average Perplexity Deltas During Fine-Tuning')
plt.xlabel('Training Step')
plt.ylabel('Perplexity Delta')
plt.grid(True, which="both", ls="--")
plt.legend()
plt.savefig(os.path.join(output_dir, "plot1_combined_avg_ppl_deltas.png"))
plt.close()


# --- PLOT 2: Raw vs. Paraphrased Perplexities ---
log.info("Generating Plot 2: Raw vs. Paraphrased Perplexities...")
plt.figure(figsize=(14, 8))

avg_raw_ppl = raw_df.groupby('step')['raw_knowledge_perplexity'].mean()
sns.lineplot(x=avg_raw_ppl.index, y=avg_raw_ppl.values, label='Raw Knowledge Probes', color='black', linewidth=2.5)

if not paraphrased_df.empty:
    avg_paraphrased_ppl = paraphrased_df.groupby(['step', 'paraphrase_variant'])['raw_knowledge_perplexity'].mean().reset_index()
    sns.lineplot(data=avg_paraphrased_ppl, x='step', y='raw_knowledge_perplexity', hue='paraphrase_variant', palette='viridis', alpha=0.7)

plt.title('Plot 2: Raw vs. Paraphrased Knowledge Statement Perplexity')
plt.xlabel('Training Step')
plt.ylabel('Perplexity')
plt.grid(True, which="both", ls="--")
plt.legend()
plt.savefig(os.path.join(output_dir, "plot2_raw_vs_paraphrased_ppl.png"))
plt.close()


# --- PLOT 3: Perplexity Deltas by Section vs. Paraphrased Average ---
log.info("Generating Plot 3: Perplexity Deltas by Section...")
plt.figure(figsize=(12, 7))

if not raw_df.empty:
    avg_by_section = raw_df.groupby(['step', 'section'])['raw_knowledge_perplexity_delta'].mean().reset_index()
    sns.lineplot(data=avg_by_section, x='step', y='raw_knowledge_perplexity_delta', hue='section', palette=section_colors, hue_order=all_sections)

if not paraphrased_df.empty:
    avg_paraphrased_ppl_delta = paraphrased_df.groupby('step')['raw_knowledge_perplexity_delta'].mean()
    sns.lineplot(x=avg_paraphrased_ppl_delta.index, y=avg_paraphrased_ppl_delta.values, color='gray', linestyle='--', label='Avg. Paraphrased')
    
plt.title('Plot 3: Raw Knowledge Perplexity Delta by Section vs. Avg. Paraphrased Delta')
plt.xlabel('Training Step')
plt.ylabel('Perplexity Delta')
plt.grid(True)
plt.legend(title='Section')
plt.savefig(os.path.join(output_dir, "plot3_ppl_delta_by_section.png"))
plt.close()
