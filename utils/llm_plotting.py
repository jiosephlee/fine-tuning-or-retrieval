import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import textwrap

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
    probes_path = "data/arxiv/DPO_knowledge_probes_v4.csv" # Hardcoded for now

    if not os.path.exists(probe_metrics_path):
        log_info(f"Skipping plotting: '{probe_metrics_path}' not found.")
        return
    
    probe_df = pd.read_csv(probe_metrics_path)
    loss_df = pd.read_csv(training_loss_path) if os.path.exists(training_loss_path) else pd.DataFrame()
    probes_csv = pd.read_csv(probes_path) if os.path.exists(probes_path) else pd.DataFrame()

    # --- PLOT 1: Normalized Loss vs. Hit Accuracy ---
    log_info("Generating Plot 1: Normalized Loss vs. Hit Accuracy...")
    hit_cols = [f'hit_accuracy_at_{k}' for k in [1, 5, 10] if f'hit_accuracy_at_{k}' in probe_df.columns]
    if not loss_df.empty and hit_cols:
        plt.figure(figsize=(14, 8))
        
        # Aggregate by step
        avg_hits = probe_df.groupby('step')[hit_cols].mean().reset_index()
        
        # Merge dataframes on 'step'
        merged_df = pd.merge(loss_df, avg_hits, on='step', how='inner')

        if not merged_df.empty:
            scaler = MinMaxScaler()
            cols_to_scale = ['chunked_perplexity'] + hit_cols
            merged_df[cols_to_scale] = scaler.fit_transform(merged_df[cols_to_scale])
            
            sns.lineplot(data=merged_df, x='step', y='chunked_perplexity', label='Normalized Training Perplexity')
            for col in hit_cols:
                sns.lineplot(data=merged_df, x='step', y=col, label=f'Normalized {col.replace("_", " ").title()}')

            plt.title('Plot 1: Normalized Training Perplexity vs. Hit Accuracy')
            plt.xlabel('Training Step')
            plt.ylabel('Normalized Value (0 to 1)')
            plt.grid(True, which="both", ls="--")
            plt.legend()
            plt.savefig(os.path.join(output_dir, "plot_new_1_loss_vs_hits.png"))
            plt.close()

    # --- PLOT 2: Hit Accuracy by Section ---
    log_info("Generating Plot 2: Hit Accuracy by Section...")
    plot_2_cols = ['hit_accuracy_at_10', 'perplexity', 'log_probs']
    if all(c in probe_df.columns for c in plot_2_cols) and 'section' in probe_df.columns:
        plt.figure(figsize=(14, 8))
        
        avg_section_hits = probe_df.groupby(['step', 'section'])[plot_2_cols].mean().reset_index()
        melted_df = avg_section_hits.melt(id_vars=['step', 'section'], value_vars=plot_2_cols, var_name='metric', value_name='value')
        
        all_sections = sorted(probe_df['section'].unique())
        palette = sns.color_palette("husl", len(all_sections))

        sns.lineplot(data=melted_df, x='step', y='value', hue='section', style='metric', hue_order=all_sections, palette=palette)
        
        plt.title('Plot 2: Metrics by Section')
        plt.xlabel('Training Step')
        plt.ylabel('Value')
        plt.grid(True, which="both", ls="--")
        plt.legend(title='Section & Metric')
        plt.savefig(os.path.join(output_dir, "plot_new_2_metrics_by_section.png"))
        plt.close()

    # --- PLOT 3: Disaggregated Hit Accuracy for 10 Random Probes ---
    log_info("Generating Plot 3: Disaggregated Hit Accuracy (10 Random Probes)...")
    if hit_cols:
        unique_probes = probe_df['probe_index'].unique()
        if len(unique_probes) > 0:
            num_probes_to_plot = min(10, len(unique_probes))
            random_probes = np.random.choice(unique_probes, num_probes_to_plot, replace=False)
            
            sample_df = probe_df[probe_df['probe_index'].isin(random_probes)]
            melted_sample_df = sample_df.melt(id_vars=['step', 'probe_index'], value_vars=hit_cols, var_name='k', value_name='accuracy')
            
            g = sns.FacetGrid(melted_sample_df, col="probe_index", col_wrap=5, hue="k", sharey=False)
            g.map(sns.lineplot, "step", "accuracy")
            g.add_legend(title='Hit Accuracy @')
            g.fig.suptitle('Plot 3: Hit Accuracy for 10 Random Probes', y=1.03)
            g.set_axis_labels("Training Step", "Hit Accuracy")
            
            if not probes_csv.empty and 'fact' in probes_csv.columns:
                for ax, probe_idx in zip(g.axes.flat, g.col_names):
                    probe_idx = int(probe_idx)
                    if probe_idx in probes_csv.index:
                        fact = probes_csv.loc[probe_idx, 'fact']
                        wrapped_title = textwrap.fill(f"Probe {probe_idx}: {fact}", 60)
                        ax.set_title(wrapped_title, fontsize=8)
            else:
                 g.set_titles("Probe {col_name}")

            plt.tight_layout(rect=[0, 0, 1, 0.97])
            plt.savefig(os.path.join(output_dir, "plot_new_3_disaggregated_hits.png"))
            plt.close()

    # --- PLOT 4: Disaggregated Normalized Perplexity vs. Log Probs for 10 Random Probes ---
    log_info("Generating Plot 4: Disaggregated Normalized Perplexity vs. Log Probs...")
    plot_4_cols = ['perplexity', 'log_probs']
    if 'random_probes' in locals() and all(c in probe_df.columns for c in plot_4_cols):
        sample_df = probe_df[probe_df['probe_index'].isin(random_probes)]
        
        # Normalize perplexity and log_probs for each probe
        scaled_df_parts = []
        scaler = MinMaxScaler()
        for probe_idx in random_probes:
            probe_data = sample_df[sample_df['probe_index'] == probe_idx].copy()
            if not probe_data.empty and len(probe_data) > 1: # Scaler needs at least 2 points
                probe_data[plot_4_cols] = scaler.fit_transform(probe_data[plot_4_cols])
                scaled_df_parts.append(probe_data)

        if scaled_df_parts:
            scaled_df = pd.concat(scaled_df_parts)
            melted_scaled_df = scaled_df.melt(id_vars=['step', 'probe_index'], value_vars=plot_4_cols, var_name='metric', value_name='normalized_value')

            g = sns.FacetGrid(melted_scaled_df, col="probe_index", col_wrap=5, hue="metric", sharey=True)
            g.map(sns.lineplot, "step", "normalized_value")
            g.add_legend(title='Metric')
            g.fig.suptitle('Plot 4: Normalized Perplexity vs. Log Probs for 10 Random Probes', y=1.03)
            g.set_axis_labels("Training Step", "Normalized Value (0 to 1)")

            if not probes_csv.empty and 'fact' in probes_csv.columns:
                for ax, probe_idx in zip(g.axes.flat, g.col_names):
                    probe_idx = int(probe_idx)
                    if probe_idx in probes_csv.index:
                        fact = probes_csv.loc[probe_idx, 'fact']
                        wrapped_title = textwrap.fill(f"Probe {probe_idx}: {fact}", 60)
                        ax.set_title(wrapped_title, fontsize=8)
            else:
                g.set_titles("Probe {col_name}")

            plt.tight_layout(rect=[0, 0, 1, 0.97])
            plt.savefig(os.path.join(output_dir, "plot_new_4_disaggregated_normalized_ppl_logprobs.png"))
            plt.close()
