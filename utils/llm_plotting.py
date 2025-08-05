import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_plots_from_files(output_dir: str, logger=None):
    """
    Generates all plots from the saved CSV files in wide format.
    """
    def log_info(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)

    # --- Data Loading ---
    log_info("Loading dataframes for plotting...")
    knowledge_probe_path = os.path.join(output_dir, "raw_knowledge_probe_metrics.csv")
    corpus_ppl_path = os.path.join(output_dir, "corpus_perplexity_metrics.csv")
    training_loss_path = os.path.join(output_dir, "training_loss_perplexity_metrics.csv")

    if not os.path.exists(knowledge_probe_path):
        log_info(f"Skipping plotting: '{knowledge_probe_path}' not found.")
        return

    metrics_df = pd.read_csv(knowledge_probe_path)
    corpus_results_df = pd.read_csv(corpus_ppl_path) if os.path.exists(corpus_ppl_path) else pd.DataFrame()
    training_loss_results_df = pd.read_csv(training_loss_path) if os.path.exists(training_loss_path) else pd.DataFrame()

    # --- Data Preparation for Deltas ---
    if not corpus_results_df.empty:
        initial_corpus_ppl = corpus_results_df['corpus_perplexity'].iloc[0]
        corpus_results_df['corpus_perplexity_delta'] = corpus_results_df['corpus_perplexity'] - initial_corpus_ppl

    if not training_loss_results_df.empty:
        initial_chunked_ppl = training_loss_results_df['chunked_perplexity'].iloc[0]
        training_loss_results_df['chunked_perplexity_delta'] = training_loss_results_df['chunked_perplexity'] - initial_chunked_ppl

    # --- Plotting Setup ---
    all_sections = sorted(metrics_df['section'].unique()) if not metrics_df.empty else []
    palette = sns.color_palette("husl", len(all_sections))
    section_colors = {section: color for section, color in zip(all_sections, palette)}

    # Identify paraphrase columns
    paraphrase_ppl_cols = [col for col in metrics_df.columns if 'raw_knowledge_perplexity_paraphrase_' in col]
    paraphrase_delta_cols = [col for col in metrics_df.columns if 'raw_knowledge_perplexity_delta_paraphrase_' in col]

    # --- PLOT 1: Combined Perplexity Deltas ---
    log_info("Generating Plot 1: Combined Perplexity Deltas...")
    plt.figure(figsize=(14, 8))

    if 'raw_knowledge_perplexity_delta' in metrics_df.columns:
        avg_raw_ppl_delta = metrics_df.groupby('step')['raw_knowledge_perplexity_delta'].mean()
        sns.lineplot(x=avg_raw_ppl_delta.index, y=avg_raw_ppl_delta.values, label='Δ Knowledge Probes (Raw PPL)')

    if paraphrase_delta_cols:
        metrics_df['paraphrase_delta_mean'] = metrics_df[paraphrase_delta_cols].mean(axis=1)
        metrics_df['paraphrase_delta_std'] = metrics_df[paraphrase_delta_cols].std(axis=1)
        
        paraphrase_stats = metrics_df.groupby('step').agg(
            mean=('paraphrase_delta_mean', 'mean'),
            std=('paraphrase_delta_std', 'mean') # Approx std of means
        )
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
    log_info("Generating Plot 2: Raw vs. Paraphrased Perplexities...")
    plt.figure(figsize=(14, 8))

    if 'raw_knowledge_perplexity' in metrics_df.columns:
        avg_raw_ppl = metrics_df.groupby('step')['raw_knowledge_perplexity'].mean()
        sns.lineplot(x=avg_raw_ppl.index, y=avg_raw_ppl.values, label='Raw Knowledge Probes', color='black', linewidth=2.5, zorder=5)

    if paraphrase_ppl_cols:
        for i, col in enumerate(paraphrase_ppl_cols):
            avg_paraphrased_ppl = metrics_df.groupby('step')[col].mean()
            sns.lineplot(x=avg_paraphrased_ppl.index, y=avg_paraphrased_ppl.values, label=f'Paraphrase Variant {i}', alpha=0.7)

    plt.title('Plot 2: Raw vs. Paraphrased Knowledge Statement Perplexity')
    plt.xlabel('Training Step')
    plt.ylabel('Perplexity')
    plt.grid(True, which="both", ls="--")
    plt.legend(title='Probe Type')
    plt.savefig(os.path.join(output_dir, "plot2_raw_vs_paraphrased_ppl.png"))
    plt.close()

    # --- PLOT 3: Perplexity Deltas by Section vs. Paraphrased Average ---
    log_info("Generating Plot 3: Perplexity Deltas by Section...")
    plt.figure(figsize=(12, 7))

    if 'raw_knowledge_perplexity_delta' in metrics_df.columns:
        avg_by_section = metrics_df.groupby(['step', 'section'])['raw_knowledge_perplexity_delta'].mean().reset_index()
        sns.lineplot(data=avg_by_section, x='step', y='raw_knowledge_perplexity_delta', hue='section', palette=section_colors, hue_order=all_sections)

    if paraphrase_delta_cols:
        # Calculate the mean of the paraphrase deltas for each row
        avg_paraphrased_ppl_delta = metrics_df.groupby('step')[paraphrase_delta_cols].mean().mean(axis=1)
        sns.lineplot(x=avg_paraphrased_ppl_delta.index, y=avg_paraphrased_ppl_delta.values, color='gray', linestyle='--', label='Avg. Paraphrased Delta')

    plt.title('Plot 3: Raw Knowledge Perplexity Delta by Section vs. Avg. Paraphrased Delta')
    plt.xlabel('Training Step')
    plt.ylabel('Perplexity Delta')
    plt.grid(True)
    plt.legend(title='Section')
    plt.savefig(os.path.join(output_dir, "plot3_ppl_delta_by_section.png"))
    plt.close()
