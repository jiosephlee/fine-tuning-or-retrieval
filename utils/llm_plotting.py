import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def generate_new_plots(output_dir: str, logger=None):
    """
    Generates a new set of plots from the saved CSV files.
    """
    def log_info(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)

    # --- Data Loading ---
    log_info("Loading dataframes for new plotting...")
    probe_metrics_path = os.path.join(output_dir, "probe_eval_metrics.csv")
    training_loss_path = os.path.join(output_dir, "training_loss_perplexity_metrics.csv")

    if not os.path.exists(probe_metrics_path):
        log_info(f"Skipping plotting: '{probe_metrics_path}' not found.")
        return
    
    probe_df = pd.read_csv(probe_metrics_path)
    loss_df = pd.read_csv(training_loss_path) if os.path.exists(training_loss_path) else pd.DataFrame()

    # --- PLOT 1: Normalized Loss vs. Hit Accuracy ---
    log_info("Generating Plot 1: Normalized Loss vs. Hit Accuracy...")
    if not loss_df.empty and 'hit_accuracy_at_50' in probe_df.columns:
        plt.figure(figsize=(14, 8))
        
        # Aggregate by step
        avg_hits = probe_df.groupby('step')['hit_accuracy_at_50'].mean().reset_index()
        
        # Merge dataframes on 'step'
        merged_df = pd.merge(loss_df, avg_hits, on='step', how='inner')

        if not merged_df.empty:
            scaler = MinMaxScaler()
            merged_df[['chunked_perplexity', 'hit_accuracy_at_50']] = scaler.fit_transform(merged_df[['chunked_perplexity', 'hit_accuracy_at_50']])
            
            sns.lineplot(data=merged_df, x='step', y='chunked_perplexity', label='Normalized Training Perplexity')
            sns.lineplot(data=merged_df, x='step', y='hit_accuracy_at_50', label='Normalized Hit Accuracy @ 50')

            plt.title('Plot 1: Normalized Training Perplexity vs. Hit Accuracy')
            plt.xlabel('Training Step')
            plt.ylabel('Normalized Value (0 to 1)')
            plt.grid(True, which="both", ls="--")
            plt.legend()
            plt.savefig(os.path.join(output_dir, "plot_new_1_loss_vs_hits.png"))
            plt.close()

    # --- PLOT 2: Hit Accuracy by Section ---
    log_info("Generating Plot 2: Hit Accuracy by Section...")
    if 'hit_accuracy_at_50' in probe_df.columns and 'section' in probe_df.columns:
        plt.figure(figsize=(14, 8))
        
        avg_section_hits = probe_df.groupby(['step', 'section'])['hit_accuracy_at_50'].mean().reset_index()
        
        all_sections = sorted(probe_df['section'].unique())
        palette = sns.color_palette("husl", len(all_sections))

        sns.lineplot(data=avg_section_hits, x='step', y='hit_accuracy_at_50', hue='section', hue_order=all_sections, palette=palette)
        
        plt.title('Plot 2: Hit Accuracy @ 50 by Section')
        plt.xlabel('Training Step')
        plt.ylabel('Hit Accuracy @ 50')
        plt.grid(True, which="both", ls="--")
        plt.legend(title='Section')
        plt.savefig(os.path.join(output_dir, "plot_new_2_hits_by_section.png"))
        plt.close()

    # --- PLOT 3: Disaggregated Hit Accuracy for 10 Random Probes ---
    log_info("Generating Plot 3: Disaggregated Hit Accuracy (10 Random Probes)...")
    if 'hit_accuracy_at_50' in probe_df.columns:
        unique_probes = probe_df['probe_index'].unique()
        if len(unique_probes) > 0:
            num_probes_to_plot = min(10, len(unique_probes))
            random_probes = np.random.choice(unique_probes, num_probes_to_plot, replace=False)
            
            sample_df = probe_df[probe_df['probe_index'].isin(random_probes)]
            
            g = sns.FacetGrid(sample_df, col="probe_index", col_wrap=5, hue="probe_index", sharey=False)
            g.map(sns.lineplot, "step", "hit_accuracy_at_50")
            g.add_legend()
            g.fig.suptitle('Plot 3: Hit Accuracy @ 50 for 10 Random Probes', y=1.03)
            g.set_axis_labels("Training Step", "Hit Accuracy @ 50")
            g.set_titles("Probe {col_name}")
            plt.savefig(os.path.join(output_dir, "plot_new_3_disaggregated_hits.png"))
            plt.close()
