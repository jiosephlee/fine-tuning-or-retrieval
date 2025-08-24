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
    if hit_cols and 'section' in probe_df.columns:
        plt.figure(figsize=(14, 8))
        
        avg_section_hits = probe_df.groupby(['step', 'section'])[hit_cols].mean().reset_index()
        melted_df = avg_section_hits.melt(id_vars=['step', 'section'], value_vars=hit_cols, var_name='k', value_name='accuracy')
        
        all_sections = sorted(probe_df['section'].unique())
        palette = sns.color_palette("husl", len(all_sections))

        sns.lineplot(data=melted_df, x='step', y='accuracy', hue='section', style='k', hue_order=all_sections, palette=palette)
        
        plt.title('Plot 2: Hit Accuracy by Section')
        plt.xlabel('Training Step')
        plt.ylabel('Hit Accuracy')
        plt.grid(True, which="both", ls="--")
        plt.legend(title='Section')
        plt.savefig(os.path.join(output_dir, "plot_new_2_hits_by_section.png"))
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
                    if probe_idx in probes_csv.index:
                        fact = probes_csv.loc[probe_idx, 'fact']
                        wrapped_title = textwrap.fill(f"Probe {probe_idx}: {fact}", 60)
                        ax.set_title(wrapped_title, fontsize=8)
            else:
                 g.set_titles("Probe {col_name}")

            plt.tight_layout(rect=[0, 0, 1, 0.97])
            plt.savefig(os.path.join(output_dir, "plot_new_3_disaggregated_hits.png"))
            plt.close()
