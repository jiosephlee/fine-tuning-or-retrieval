import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

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
    atomic_probe_path = os.path.join(output_dir, "simple_atomic_knowledge_probe_metrics.csv")
    corpus_ppl_path = os.path.join(output_dir, "corpus_perplexity_metrics.csv")
    training_loss_path = os.path.join(output_dir, "training_loss_perplexity_metrics.csv")

    if not os.path.exists(knowledge_probe_path):
        log_info(f"Skipping plotting: '{knowledge_probe_path}' not found.")
        return

    metrics_df = pd.read_csv(knowledge_probe_path)
    atomic_metrics_df = pd.read_csv(atomic_probe_path) if os.path.exists(atomic_probe_path) else pd.DataFrame()
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

    # --- PLOT 0: Raw Knowledge Perplexity Distribution ---
    log_info("Generating Plot 0: Raw Knowledge Perplexity Distribution...")
    plt.figure(figsize=(14, 8))
    
    if 'raw_knowledge_perplexity' in metrics_df.columns:
        stats = metrics_df.groupby('step')['raw_knowledge_perplexity'].agg(['mean', 'std'])
        mean = stats['mean']
        std = stats['std']

        # Plot the mean line
        sns.lineplot(x=mean.index, y=mean.values, label='Mean Perplexity', color='blue')

        # Add shadows for 1 and 2 standard deviations
        plt.fill_between(mean.index, mean - 2 * std, mean + 2 * std, alpha=0.1, color='blue', label='2 Std. Dev.')
        plt.fill_between(mean.index, mean - 1 * std, mean + 1 * std, alpha=0.2, color='blue', label='1 Std. Dev.')
        
        # Calculate fractions within std deviations at the last step
        last_step_df = metrics_df[metrics_df['step'] == metrics_df['step'].max()]
        last_perplexities = last_step_df['raw_knowledge_perplexity']
        mean_last = last_perplexities.mean()
        std_last = last_perplexities.std()
        
        within_1_std = ((last_perplexities >= mean_last - std_last) & (last_perplexities <= mean_last + std_last)).mean()
        within_2_std = ((last_perplexities >= mean_last - 2*std_last) & (last_perplexities <= mean_last + 2*std_last)).mean()

        text_str = (f"At final step:\n"
                    f"Fraction within 1 Std. Dev.: {within_1_std:.2%}\n"
                    f"Fraction within 2 Std. Dev.: {within_2_std:.2%}")
        
        plt.text(0.95, 0.95, text_str, transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

    plt.title('Plot 0: Raw Knowledge Perplexity Distribution')
    plt.xlabel('Training Step')
    plt.ylabel('Perplexity')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "plot0_raw_ppl_distribution.png"))
    plt.close()

    # --- PLOT 0b: Paraphrased Knowledge Perplexity Distribution ---
    log_info("Generating Plot 0b: Paraphrased Knowledge Perplexity Distribution...")
    plt.figure(figsize=(14, 8))

    if paraphrase_ppl_cols:
        metrics_df['paraphrase_ppl_mean'] = metrics_df[paraphrase_ppl_cols].mean(axis=1)
        
        stats = metrics_df.groupby('step')['paraphrase_ppl_mean'].agg(['mean', 'std'])
        mean = stats['mean']
        std = stats['std']

        # Plot the mean line
        sns.lineplot(x=mean.index, y=mean.values, label='Mean Paraphrased Perplexity', color='green')

        # Add shadows for 1 and 2 standard deviations
        plt.fill_between(mean.index, mean - 2 * std, mean + 2 * std, alpha=0.1, color='green', label='2 Std. Dev.')
        plt.fill_between(mean.index, mean - 1 * std, mean + 1 * std, alpha=0.2, color='green', label='1 Std. Dev.')

        # Calculate fractions within std deviations at the last step
        last_step_df = metrics_df[metrics_df['step'] == metrics_df['step'].max()]
        last_perplexities = last_step_df['paraphrase_ppl_mean']
        mean_last = last_perplexities.mean()
        std_last = last_perplexities.std()
        
        within_1_std = ((last_perplexities >= mean_last - std_last) & (last_perplexities <= mean_last + std_last)).mean()
        within_2_std = ((last_perplexities >= mean_last - 2*std_last) & (last_perplexities <= mean_last + 2*std_last)).mean()

        text_str = (f"At final step:\n"
                    f"Fraction within 1 Std. Dev.: {within_1_std:.2%}\n"
                    f"Fraction within 2 Std. Dev.: {within_2_std:.2%}")
        
        plt.text(0.95, 0.95, text_str, transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

    plt.title('Plot 0b: Paraphrased Knowledge Perplexity Distribution')
    plt.xlabel('Training Step')
    plt.ylabel('Perplexity')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "plot0b_paraphrased_ppl_distribution.png"))
    plt.close()


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

    # --- PLOT 1b: Combined Perplexities ---
    log_info("Generating Plot 1b: Combined Perplexities...")
    plt.figure(figsize=(14, 8))

    if 'raw_knowledge_perplexity' in metrics_df.columns:
        avg_raw_ppl = metrics_df.groupby('step')['raw_knowledge_perplexity'].mean()
        sns.lineplot(x=avg_raw_ppl.index, y=avg_raw_ppl.values, label='Knowledge Probes (Raw PPL)')

    if paraphrase_ppl_cols:
        if 'paraphrase_ppl_mean' not in metrics_df.columns:
            metrics_df['paraphrase_ppl_mean'] = metrics_df[paraphrase_ppl_cols].mean(axis=1)
        if 'paraphrase_ppl_std' not in metrics_df.columns:
            metrics_df['paraphrase_ppl_std'] = metrics_df[paraphrase_ppl_cols].std(axis=1)
        
        paraphrase_stats = metrics_df.groupby('step').agg(
            mean=('paraphrase_ppl_mean', 'mean'),
            std=('paraphrase_ppl_std', 'mean') # Approx std of means
        )
        sns.lineplot(data=paraphrase_stats, x='step', y='mean', label='Paraphrased Probes (Mean)')
        plt.fill_between(paraphrase_stats.index, 
                         paraphrase_stats['mean'] - paraphrase_stats['std'], 
                         paraphrase_stats['mean'] + paraphrase_stats['std'], 
                         alpha=0.2, label='Paraphrased Probes (Std. Dev.)')

    if not corpus_results_df.empty and 'corpus_perplexity' in corpus_results_df.columns:
        sns.lineplot(data=corpus_results_df, x='step', y='corpus_perplexity', label='Sliding Window (Full Paper)')
    if not training_loss_results_df.empty and 'chunked_perplexity' in training_loss_results_df.columns:
        sns.lineplot(data=training_loss_results_df, x='step', y='chunked_perplexity', label='Training Loss (Chunked)')

    plt.title('Plot 1b: Combined Average Perplexities During Fine-Tuning')
    plt.xlabel('Training Step')
    plt.ylabel('Perplexity')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "plot1b_combined_avg_ppl.png"))
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

    # --- PLOT 3: Perplexity Deltas by Section (Raw vs. Paraphrased Avg) ---
    log_info("Generating Plot 3: Perplexity Deltas by Section...")
    plt.figure(figsize=(12, 7))

    plot_data = []

    # Prepare raw data
    if 'raw_knowledge_perplexity_delta' in metrics_df.columns:
        raw_df = metrics_df.groupby(['step', 'section'])['raw_knowledge_perplexity_delta'].mean().reset_index()
        raw_df.rename(columns={'raw_knowledge_perplexity_delta': 'delta'}, inplace=True)
        raw_df['type'] = 'Raw'
        plot_data.append(raw_df)

    # Prepare paraphrased data
    if paraphrase_delta_cols:
        metrics_df['paraphrase_delta_mean'] = metrics_df[paraphrase_delta_cols].mean(axis=1)
        paraphrased_df = metrics_df.groupby(['step', 'section'])['paraphrase_delta_mean'].mean().reset_index()
        paraphrased_df.rename(columns={'paraphrase_delta_mean': 'delta'}, inplace=True)
        paraphrased_df['type'] = 'Paraphrased (Avg)'
        plot_data.append(paraphrased_df)
    
    if plot_data:
        combined_df = pd.concat(plot_data, ignore_index=True)
        sns.lineplot(data=combined_df, x='step', y='delta', hue='section', style='type',
                     palette=section_colors, hue_order=all_sections)

    plt.title('Plot 3: Raw vs. Avg. Paraphrased Perplexity Delta by Section')
    plt.xlabel('Training Step')
    plt.ylabel('Perplexity Delta')
    plt.grid(True)
    plt.legend(title='Section / Type')
    plt.savefig(os.path.join(output_dir, "plot3_ppl_delta_by_section.png"))
    plt.close()

    # --- PLOT 3b: Perplexity by Section (Raw vs. Paraphrased Avg) ---
    log_info("Generating Plot 3b: Perplexity by Section...")
    plt.figure(figsize=(12, 7))

    plot_data_3b = []

    # Prepare raw data
    if 'raw_knowledge_perplexity' in metrics_df.columns:
        raw_df_3b = metrics_df.groupby(['step', 'section'])['raw_knowledge_perplexity'].mean().reset_index()
        raw_df_3b.rename(columns={'raw_knowledge_perplexity': 'perplexity'}, inplace=True)
        raw_df_3b['type'] = 'Raw'
        plot_data_3b.append(raw_df_3b)

    # Prepare paraphrased data
    if paraphrase_ppl_cols:
        if 'paraphrase_ppl_mean' not in metrics_df.columns:
            metrics_df['paraphrase_ppl_mean'] = metrics_df[paraphrase_ppl_cols].mean(axis=1)
        paraphrased_df_3b = metrics_df.groupby(['step', 'section'])['paraphrase_ppl_mean'].mean().reset_index()
        paraphrased_df_3b.rename(columns={'paraphrase_ppl_mean': 'perplexity'}, inplace=True)
        paraphrased_df_3b['type'] = 'Paraphrased (Avg)'
        plot_data_3b.append(paraphrased_df_3b)
    
    if plot_data_3b:
        combined_df_3b = pd.concat(plot_data_3b, ignore_index=True)
        sns.lineplot(data=combined_df_3b, x='step', y='perplexity', hue='section', style='type',
                     palette=section_colors, hue_order=all_sections)

    plt.title('Plot 3b: Raw vs. Avg. Paraphrased Perplexity by Section')
    plt.xlabel('Training Step')
    plt.ylabel('Perplexity')
    plt.grid(True)
    plt.legend(title='Section / Type')
    plt.savefig(os.path.join(output_dir, "plot3b_ppl_by_section.png"))
    plt.close()

    # --- PLOT 4: Raw vs. Atomic Knowledge Perplexity by Section ---
    log_info("Generating Plot 4: Raw vs. Atomic Knowledge Perplexity by Section...")
    plt.figure(figsize=(12, 7))

    plot_data_4 = []

    if 'raw_knowledge_perplexity' in metrics_df.columns:
        raw_df_4 = metrics_df.groupby(['step', 'section'])['raw_knowledge_perplexity'].mean().reset_index()
        raw_df_4.rename(columns={'raw_knowledge_perplexity': 'perplexity'}, inplace=True)
        raw_df_4['type'] = 'Raw'
        plot_data_4.append(raw_df_4)
        
    if not atomic_metrics_df.empty and 'atomic_knowledge_statement_perplexity' in atomic_metrics_df.columns:
        atomic_df_4 = atomic_metrics_df.groupby(['step', 'section'])['atomic_knowledge_statement_perplexity'].mean().reset_index()
        atomic_df_4.rename(columns={'atomic_knowledge_statement_perplexity': 'perplexity'}, inplace=True)
        atomic_df_4['type'] = 'Atomic'
        plot_data_4.append(atomic_df_4)

    if plot_data_4:
        combined_df_4 = pd.concat(plot_data_4, ignore_index=True)
        sns.lineplot(data=combined_df_4, x='step', y='perplexity', hue='section', style='type',
                     palette=section_colors, hue_order=all_sections, style_order=['Raw', 'Atomic'], dashes={'Raw': '', 'Atomic': (4, 4)})

    plt.title('Plot 4: Raw vs. Atomic Knowledge Perplexity by Section')
    plt.xlabel('Training Step')
    plt.ylabel('Perplexity')
    plt.grid(True)
    plt.legend(title='Section / Type')
    plt.savefig(os.path.join(output_dir, "plot4_raw_vs_atomic_by_section.png"))
    plt.close()

    # --- PLOT 4b: Disaggregated Raw vs. Atomic Perplexity (10 Random Probes) ---
    log_info("Generating Plot 4b: Disaggregated Raw vs. Atomic Perplexity (10 Random Probes)...")
    
    # Ensure there's data to plot
    if not metrics_df.empty and not atomic_metrics_df.empty:
        # Get common probe indices
        common_probe_indices = sorted(list(set(metrics_df['probe_index'].unique()) & set(atomic_metrics_df['probe_index'].unique())))
        
        if common_probe_indices:
            # Select 10 random probes
            num_probes_to_plot = min(10, len(common_probe_indices))
            random_probes = np.random.choice(common_probe_indices, num_probes_to_plot, replace=False)
            
            # Filter data for these probes
            raw_sample_df = metrics_df[metrics_df['probe_index'].isin(random_probes)][['step', 'probe_index', 'raw_knowledge_perplexity']]
            raw_sample_df.rename(columns={'raw_knowledge_perplexity': 'perplexity'}, inplace=True)
            raw_sample_df['type'] = 'Raw'
            
            atomic_sample_df = atomic_metrics_df[atomic_metrics_df['probe_index'].isin(random_probes)][['step', 'probe_index', 'atomic_knowledge_statement_perplexity']]
            atomic_sample_df.rename(columns={'atomic_knowledge_statement_perplexity': 'perplexity'}, inplace=True)
            atomic_sample_df['type'] = 'Atomic'
            
            plot_data_4b = pd.concat([raw_sample_df, atomic_sample_df], ignore_index=True)
            
            if not plot_data_4b.empty:
                plt.figure(figsize=(15, 10))
                g = sns.FacetGrid(plot_data_4b, col="probe_index", col_wrap=5, hue="type", sharey=False)
                g.map(sns.lineplot, "step", "perplexity")
                g.add_legend()
                g.fig.suptitle('Plot 4b: Perplexity for 10 Random Probes (Raw vs. Atomic)', y=1.03)
                g.set_axis_labels("Training Step", "Perplexity")
                g.set_titles("Probe {col_name}")
                plt.savefig(os.path.join(output_dir, "plot4b_disaggregated_comparison.png"))
                plt.close()
            else:
                log_info("Could not generate Plot 4b: No overlapping probe data to plot.")
        else:
            log_info("Could not generate Plot 4b: No common probe indices between raw and atomic metrics.")
    else:
        log_info("Could not generate Plot 4b: Missing raw or atomic metrics data.")

