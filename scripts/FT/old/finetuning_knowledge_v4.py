# add .. path 
import os
import sys
sys.path.append('../..')
import utils.llm_training as llm_training
import utils.llm_callbacks_old as llm_callbacks_old
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
    per_device_train_batch_size=3,
    context_length = 2048,
    gradient_accumulation_steps=2,
    warmup_ratio = 0.1, 
    sequential_sampling = False,
    reverse_ffd_packing= False,
    remove_unused_columns=False,
)

# Replace PerplexityEvaluationCallback and TargetedPerplexityCallback with this:
knowledge_probe_callback = llm_callbacks_old.KnowledgeProbeCallback(
    tokenizer,
    '../../data/arxiv/DPO_knowledge_probes_v2.csv',
    training_config.context_length,
    batch_size=8,
)

corpus_callback = llm_callbacks_old.CorpusPerplexityCallback(
    text_content=arxiv_paper,
    tokenizer=tokenizer,
    max_length=training_config.context_length,
    stride=512  # A good practical compromise, as mentioned in the docs
)

# Add this new callback to track perplexity from training loss
training_loss_callback = llm_callbacks_old.TrainingLossPerplexityCallback()

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

# --- Save Metrics and Generate Plots ---
output_dir = os.path.join("../../results/FT/", args.experiment_name)
os.makedirs(output_dir, exist_ok=True)
knowledge_probe_callback.save_results(output_dir=output_dir)
log.info(f"Knowledge probe metrics saved to {output_dir}")


# --- Data Loading and Preparation ---
log.info("Loading dataframes for plotting...")
# Perplexity Deltas
raw_ppl_delta_df = knowledge_probe_callback.get_raw_knowledge_perplexity_delta_dataframe()
atomic_whole_ppl_delta_df = knowledge_probe_callback.get_atomic_whole_perplexity_delta_dataframe()
atomic_target_ppl_delta_df = knowledge_probe_callback.get_atomic_target_perplexity_delta_dataframe()
corpus_results_df = corpus_callback.get_results_as_dataframe()
training_loss_results_df = training_loss_callback.get_results_as_dataframe()

# Load paraphrased dataframes
paraphrased_atomic_whole_ppl_delta_df = knowledge_probe_callback.get_paraphrased_atomic_whole_perplexity_delta_dataframe()
paraphrased_atomic_target_ppl_delta_df = knowledge_probe_callback.get_paraphrased_atomic_target_perplexity_delta_dataframe()

# Log Probability Deltas
atomic_whole_log_prob_delta_df = knowledge_probe_callback.get_atomic_whole_log_prob_delta_dataframe()
atomic_target_log_prob_delta_df = knowledge_probe_callback.get_atomic_target_log_prob_delta_dataframe()

# Load paraphrased log prob dataframes
paraphrased_atomic_whole_log_prob_delta_df = knowledge_probe_callback.get_paraphrased_atomic_whole_log_prob_delta_dataframe()
paraphrased_atomic_target_log_prob_delta_df = knowledge_probe_callback.get_paraphrased_atomic_target_log_prob_delta_dataframe()

# Hit Rate Data
atomic_target_hit_at_5_df = knowledge_probe_callback.get_atomic_target_hit_at_5_dataframe()
atomic_target_hit_at_50_df = knowledge_probe_callback.get_atomic_target_hit_at_50_dataframe()
atomic_target_hit_at_100_df = knowledge_probe_callback.get_atomic_target_hit_at_100_dataframe()

# Manually calculate deltas for corpus and training loss perplexity
if not corpus_results_df.empty:
    initial_corpus_ppl = corpus_results_df['corpus_perplexity'].iloc[0]
    corpus_results_df['corpus_perplexity_delta'] = corpus_results_df['corpus_perplexity'] - initial_corpus_ppl

if not training_loss_results_df.empty:
    initial_chunked_ppl = training_loss_results_df['chunked_perplexity'].iloc[0]
    training_loss_results_df['chunked_perplexity_delta'] = training_loss_results_df['chunked_perplexity'] - initial_chunked_ppl

# --- Plotting Setup ---
all_sections = sorted(raw_ppl_delta_df['section'].unique()) if not raw_ppl_delta_df.empty else []
palette = sns.color_palette("husl", len(all_sections))
section_colors = {section: color for section, color in zip(all_sections, palette)}


# --- GROUP 1: PERPLEXITY DELTA PLOTS ---
log.info("Generating Perplexity Delta plots...")

# Plot 1: Combined Average Perplexity Deltas
plt.figure(figsize=(14, 8))
avg_raw_ppl_delta = raw_ppl_delta_df.groupby('step')['perplexity_delta'].mean().reset_index()
avg_atomic_whole_ppl_delta = atomic_whole_ppl_delta_df.groupby('step')['perplexity_delta'].mean().reset_index()
avg_atomic_target_ppl_delta = atomic_target_ppl_delta_df.groupby('step')['perplexity_delta'].mean().reset_index()
avg_paraphrased_atomic_whole_ppl_delta = paraphrased_atomic_whole_ppl_delta_df.groupby('step')['perplexity_delta'].mean().reset_index()
avg_paraphrased_atomic_target_ppl_delta = paraphrased_atomic_target_ppl_delta_df.groupby('step')['perplexity_delta'].mean().reset_index()

sns.lineplot(data=avg_raw_ppl_delta, x='step', y='perplexity_delta', label='Δ Knowledge Probes (Raw PPL)', linestyle='-')
sns.lineplot(data=avg_atomic_whole_ppl_delta, x='step', y='perplexity_delta', label='Δ Knowledge Probes (Atomic PPL)', linestyle='--')
sns.lineplot(data=avg_atomic_target_ppl_delta, x='step', y='perplexity_delta', label='Δ Knowledge Probes (Target PPL)', linestyle=':')
sns.lineplot(data=avg_paraphrased_atomic_whole_ppl_delta, x='step', y='perplexity_delta', label='Δ Paraphrased Knowledge Probes (Atomic PPL)', linestyle='--')
sns.lineplot(data=avg_paraphrased_atomic_target_ppl_delta, x='step', y='perplexity_delta', label='Δ Paraphrased Knowledge Probes (Target PPL)', linestyle=':')
if not corpus_results_df.empty:
    sns.lineplot(data=corpus_results_df, x='step', y='corpus_perplexity_delta', label='Δ Sliding Window (Full Paper)', linestyle='-')
if not training_loss_results_df.empty:
    sns.lineplot(data=training_loss_results_df, x='step', y='chunked_perplexity_delta', label='Δ Training Loss (Chunked)', linestyle='--')

plt.title('Plot 1: Combined Average Perplexity Deltas During Fine-Tuning')
plt.xlabel('Training Step')
plt.ylabel('Perplexity Delta')
plt.grid(True, which="both", ls="--")
plt.legend()
plt.savefig(os.path.join(output_dir, "plot1_combined_avg_ppl_deltas.png"))
plt.close()


def plot_by_section(df, y_col, title, y_label, output_dir, filename):
    if df.empty:
        log.warning(f"Skipping plot '{title}' due to empty dataframe.")
        return
    plt.figure(figsize=(12, 7))
    avg_by_section = df.groupby(['step', 'section'])[y_col].mean().reset_index()
    overall_avg = df.groupby('step')[y_col].mean().reset_index()
    
    sns.lineplot(data=overall_avg, x='step', y=y_col, color='gray', linestyle=':', label='Overall Avg.', alpha=0.8)
    sns.lineplot(data=avg_by_section, x='step', y=y_col, hue='section', palette=section_colors, hue_order=all_sections)
    
    plt.title(title)
    plt.xlabel('Training Step')
    plt.ylabel(y_label)
    plt.grid(True)
    plt.legend(title='Section')
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path)
    log.info(f"Plot saved to {save_path}")
    plt.close()

# Plot 2, 3, 4: Perplexity Deltas by Section
plot_by_section(raw_ppl_delta_df, 'perplexity_delta', 'Plot 2: Raw Knowledge Perplexity Delta by Section', 'Perplexity Delta', output_dir, "plot2_raw_ppl_delta_by_section.png")
plot_by_section(atomic_whole_ppl_delta_df, 'perplexity_delta', 'Plot 3: Atomic Knowledge (Whole) Perplexity Delta by Section', 'Perplexity Delta', output_dir, "plot3_atomic_whole_ppl_delta_by_section.png")
plot_by_section(atomic_target_ppl_delta_df, 'perplexity_delta', 'Plot 4: Atomic Knowledge (Target) Perplexity Delta by Section', 'Perplexity Delta', output_dir, "plot4_atomic_target_ppl_delta_by_section.png")

# Plot 5: Mean Atomic Probe (Target) Perplexity Delta
plt.figure(figsize=(12, 7))

# Original Probes
avg_atomic_target_ppl_delta = atomic_target_ppl_delta_df.groupby('step')['perplexity_delta'].mean().reset_index()
sns.lineplot(data=avg_atomic_target_ppl_delta, x='step', y='perplexity_delta', label='Original Probes')

# Paraphrased Probes with Std Dev Shadow
if not paraphrased_atomic_target_ppl_delta_df.empty:
    # Group by step and calculate mean and std dev across all paraphrases and probes
    paraphrase_stats = paraphrased_atomic_target_ppl_delta_df.groupby('step')['perplexity_delta'].agg(['mean', 'std']).reset_index()
    
    # Plot the mean line
    sns.lineplot(data=paraphrase_stats, x='step', y='mean', label='Paraphrased Probes (Mean)')
    
    # Add the standard deviation shadow
    plt.fill_between(paraphrase_stats['step'], 
                     paraphrase_stats['mean'] - paraphrase_stats['std'], 
                     paraphrase_stats['mean'] + paraphrase_stats['std'], 
                     alpha=0.2, label='Paraphrased Probes (Std. Dev.)')

plt.title('Plot 5: Mean Atomic Knowledge (Target) Perplexity Delta Comparison')
plt.xlabel('Training Step')
plt.ylabel('Average Perplexity Delta')
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(output_dir, "plot5_mean_atomic_target_ppl_delta.png"))
plt.close()


# --- GROUP 2: LOG PROBABILITY DELTA PLOTS ---
log.info("Generating Log Probability Delta plots...")

# Plot 7: Combined Average Log-Prob Deltas
plt.figure(figsize=(14, 8))
avg_atomic_whole_log_prob_delta = atomic_whole_log_prob_delta_df.groupby('step')['log_prob_delta'].mean().reset_index()
avg_atomic_target_log_prob_delta = atomic_target_log_prob_delta_df.groupby('step')['log_prob_delta'].mean().reset_index()
avg_paraphrased_atomic_whole_log_prob_delta = paraphrased_atomic_whole_log_prob_delta_df.groupby('step')['log_prob_delta'].mean().reset_index()
avg_paraphrased_atomic_target_log_prob_delta = paraphrased_atomic_target_log_prob_delta_df.groupby('step')['log_prob_delta'].mean().reset_index()

sns.lineplot(data=avg_atomic_whole_log_prob_delta, x='step', y='log_prob_delta', label='Δ Atomic Knowledge (Whole Log-Prob)', linestyle='--')
sns.lineplot(data=avg_atomic_target_log_prob_delta, x='step', y='log_prob_delta', label='Δ Atomic Knowledge (Target Log-Prob)', linestyle=':')
sns.lineplot(data=avg_paraphrased_atomic_whole_log_prob_delta, x='step', y='log_prob_delta', label='Δ Paraphrased Knowledge (Whole Log-Prob)', linestyle='--')
sns.lineplot(data=avg_paraphrased_atomic_target_log_prob_delta, x='step', y='log_prob_delta', label='Δ Paraphrased Knowledge (Target Log-Prob)', linestyle=':')

plt.title('Plot 7: Combined Average Log-Probability Deltas')
plt.xlabel('Training Step')
plt.ylabel('Log-Probability Delta')
plt.grid(True, which="both", ls="--")
plt.legend()
plt.savefig(os.path.join(output_dir, "plot7_combined_avg_log_prob_deltas.png"))
plt.close()

# Plot 8, 9: Log-Prob Deltas by Section
plot_by_section(atomic_whole_log_prob_delta_df, 'log_prob_delta', 'Plot 8: Atomic Knowledge (Whole) Log-Prob Delta by Section', 'Log-Prob Delta', output_dir, "plot8_atomic_whole_log_prob_delta_by_section.png")
plot_by_section(atomic_target_log_prob_delta_df, 'log_prob_delta', 'Plot 9: Atomic Knowledge (Target) Log-Prob Delta by Section', 'Log-Prob Delta', output_dir, "plot9_atomic_target_log_prob_delta_by_section.png")

# Plot 11: Mean Atomic Probe (Target) Log-Prob Delta
plt.figure(figsize=(12, 7))

# Original Probes
avg_atomic_target_log_prob_delta = atomic_target_log_prob_delta_df.groupby('step')['log_prob_delta'].mean().reset_index()
sns.lineplot(data=avg_atomic_target_log_prob_delta, x='step', y='log_prob_delta', label='Original Probes')

# Paraphrased Probes with Std Dev Shadow
if not paraphrased_atomic_target_log_prob_delta_df.empty:
    paraphrase_stats = paraphrased_atomic_target_log_prob_delta_df.groupby('step')['log_prob_delta'].agg(['mean', 'std']).reset_index()
    sns.lineplot(data=paraphrase_stats, x='step', y='mean', label='Paraphrased Probes (Mean)')
    plt.fill_between(paraphrase_stats['step'],
                     paraphrase_stats['mean'] - paraphrase_stats['std'],
                     paraphrase_stats['mean'] + paraphrase_stats['std'],
                     alpha=0.2, label='Paraphrased Probes (Std. Dev.)')

plt.title('Plot 11: Mean Atomic Knowledge (Target) Log-Prob Delta Comparison')
plt.xlabel('Training Step')
plt.ylabel('Average Log-Prob Delta')
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(output_dir, "plot11_mean_atomic_target_log_prob_delta.png"))
plt.close()


# --- Plot 12: Atomic Target First Token Hit Rate ---
log.info("Generating First Token Hit Rate plot...")
plt.figure(figsize=(12, 7))

if not atomic_target_hit_at_5_df.empty:
    avg_hit_at_5 = atomic_target_hit_at_5_df.groupby('step')['hit_at_5'].mean().reset_index()
    sns.lineplot(data=avg_hit_at_5, x='step', y='hit_at_5', label='Hit@5')

if not atomic_target_hit_at_50_df.empty:
    avg_hit_at_50 = atomic_target_hit_at_50_df.groupby('step')['hit_at_50'].mean().reset_index()
    sns.lineplot(data=avg_hit_at_50, x='step', y='hit_at_50', label='Hit@50')

if not atomic_target_hit_at_100_df.empty:
    avg_hit_at_100 = atomic_target_hit_at_100_df.groupby('step')['hit_at_100'].mean().reset_index()
    sns.lineplot(data=avg_hit_at_100, x='step', y='hit_at_100', label='Hit@100')
    
plt.title('Plot 12: Atomic Target First Token Hit Rate')
plt.xlabel('Training Step')
plt.ylabel('Average Hit Rate')
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(output_dir, "plot12_atomic_target_hit_rate.png"))
plt.close()
