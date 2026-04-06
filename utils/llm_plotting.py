import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import textwrap
import re
from utils import probe_paths


def set_plot_style():
    """
    Sets a consistent academic-style plotting theme using matplotlib's classic style.
    """
    plt.style.use('classic')
    plt.rcParams.update({
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.titlesize": 18,
        "axes.titlesize": 16,
        "savefig.dpi": 300,
    })

def plot_textbook_insertion_lines(fig, max_step, vline_interval):
    """
    Plots vertical lines for textbook insertion points.
    """
    for i in range(1, max_step // vline_interval + 1):
        plt.axvline(x=i * vline_interval, color='grey', linestyle='--', alpha=0.5)


def generate_revamped_plots(domain: str, knowledge_probes_version: str, inference_probes_version: str, experiment_dir: str, logger=None):
    """
    Generates a new, simplified set of plots for knowledge and inference probes.
    """
    def log_info(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)

    # --- Directory for new plots ---
    plot_output_dir = os.path.join(experiment_dir, f"{domain}_plots")
    os.makedirs(plot_output_dir, exist_ok=True)
    log_info(f"Saving new plots to {plot_output_dir}")

    # --- Data Loading ---
    log_info("Loading dataframes for revamped plotting...")
    
    # Probe metrics
    knowledge_probe_metrics_path = os.path.join(experiment_dir, f"{domain}_knowledge_probe", f"{domain}_knowledge_probe_metrics.csv")
    inference_probe_metrics_path = os.path.join(experiment_dir, f"{domain}_inference_probe", f"{domain}_inference_probe_metrics.csv")
    
    knowledge_probe_df = pd.read_csv(knowledge_probe_metrics_path) if os.path.exists(knowledge_probe_metrics_path) else pd.DataFrame()
    inference_probe_df = pd.read_csv(inference_probe_metrics_path) if os.path.exists(inference_probe_metrics_path) else pd.DataFrame()

    if knowledge_probe_df.empty and inference_probe_df.empty:
        log_info("No probe metrics found. Skipping plotting.")
        return

    # Training loss
    training_loss_path = os.path.join(experiment_dir, "training_loss_perplexity_metrics.csv")
    loss_df = pd.read_csv(training_loss_path) if os.path.exists(training_loss_path) else pd.DataFrame()

    # Knowledge Probes CSV for facts and sections
    knowledge_probes_path = str(probe_paths.resolve_knowledge_probe_path(domain, knowledge_probes_version))
    knowledge_probes_csv = pd.read_csv(knowledge_probes_path) if os.path.exists(knowledge_probes_path) else pd.DataFrame()
    
    # Inference Probes CSV for facts
    path1, path2 = [
        str(path) for path in probe_paths.resolve_inference_probe_candidates(domain, inference_probes_version)
    ]
    if os.path.exists(path1):
        inference_probes_path = path1
    elif os.path.exists(path2):
        inference_probes_path = path2
    else:
        inference_probes_path = None
    inference_probes_csv = pd.read_csv(inference_probes_path) if inference_probes_path and os.path.exists(inference_probes_path) else pd.DataFrame()

    # Map sections to knowledge_probe_df
    if not knowledge_probe_df.empty and not knowledge_probes_csv.empty and 'section' in knowledge_probes_csv.columns:
        section_map = knowledge_probes_csv['section'].to_dict()
        knowledge_probe_df['section'] = knowledge_probe_df['probe_index'].map(section_map)

    # --- Plot 1: Combined Probes vs. Training Loss ---
    log_info("Generating Plot 1: Combined Probes vs. Training Loss...")
    
    # Prepare data
    mean_knowledge_log_probs = knowledge_probe_df.groupby('step')['log_prob'].mean().reset_index() if not knowledge_probe_df.empty else pd.DataFrame()
    mean_inference_log_probs = inference_probe_df.groupby('step')['log_prob'].mean().reset_index() if not inference_probe_df.empty else pd.DataFrame()
    mean_knowledge_hits10 = knowledge_probe_df.groupby('step')['hit_accuracy_at_10'].mean().reset_index() if not knowledge_probe_df.empty and 'hit_accuracy_at_10' in knowledge_probe_df.columns else pd.DataFrame()
    mean_inference_hits10 = inference_probe_df.groupby('step')['hit_accuracy_at_10'].mean().reset_index() if not inference_probe_df.empty and 'hit_accuracy_at_10' in inference_probe_df.columns else pd.DataFrame()

    # --- Plot 1a ---
    log_info("Generating Plot 1a: Mean Log Probs vs. Training Loss...")
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Absolute Mean Log Probs')
    
    if not mean_knowledge_log_probs.empty:
        sns.lineplot(data=mean_knowledge_log_probs, x='step', y='log_prob', ax=ax1, label='Knowledge Probes Mean Log Probs', color='blue')
    if not mean_inference_log_probs.empty:
        sns.lineplot(data=mean_inference_log_probs, x='step', y='log_prob', ax=ax1, label='Inference Probes Mean Log Probs', color='green')
    ax1.legend(loc='upper left')

    if not loss_df.empty and 'chunked_perplexity' in loss_df.columns:
        ax2 = ax1.twinx()
        ax2.set_ylabel('Normalized Training Loss (Perplexity)', color='grey')
        scaler = MinMaxScaler()
        loss_df['normalized_loss'] = scaler.fit_transform(loss_df[['chunked_perplexity']])
        sns.lineplot(data=loss_df, x='step', y='normalized_loss', ax=ax2, label='Normalized Training Loss', color='grey', linestyle='--')
        ax2.tick_params(axis='y', labelcolor='grey')
        ax2.legend(loc='upper right')

    plt.title('Plot 1a: Mean Log Probs vs. Training Loss')
    plt.grid(True, which="both", ls="--")
    plt.savefig(os.path.join(plot_output_dir, "plot_1a_mean_log_probs_vs_loss.png"))
    plt.close()

    # --- Plot 1b ---
    log_info("Generating Plot 1b: Mean Hit Accuracy @ 10 vs. Training Loss...")
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Absolute Mean Hit Accuracy @ 10')
    ax1.set_ylim(0, 1)

    if not mean_knowledge_hits10.empty:
        sns.lineplot(data=mean_knowledge_hits10, x='step', y='hit_accuracy_at_10', ax=ax1, label='Knowledge Probes Mean Hits@10', color='blue')
    if not mean_inference_hits10.empty:
        sns.lineplot(data=mean_inference_hits10, x='step', y='hit_accuracy_at_10', ax=ax1, label='Inference Probes Mean Hits@10', color='green')
    ax1.legend(loc='upper left')

    if not loss_df.empty and 'chunked_perplexity' in loss_df.columns:
        ax2 = ax1.twinx()
        ax2.set_ylabel('Normalized Training Loss (Perplexity)', color='grey')
        if 'normalized_loss' not in loss_df.columns: # reuse if already computed
            scaler = MinMaxScaler()
            loss_df['normalized_loss'] = scaler.fit_transform(loss_df[['chunked_perplexity']])
        sns.lineplot(data=loss_df, x='step', y='normalized_loss', ax=ax2, label='Normalized Training Loss', color='grey', linestyle='--')
        ax2.tick_params(axis='y', labelcolor='grey')
        ax2.legend(loc='upper right')

    plt.title('Plot 1b: Mean Hit Accuracy @ 10 vs. Training Loss')
    plt.grid(True, which="both", ls="--")
    plt.savefig(os.path.join(plot_output_dir, "plot_1b_mean_hits_at_10_vs_loss.png"))
    plt.close()
    
    # --- PLOT 2: Knowledge Probes by Section ---
    if not knowledge_probe_df.empty and 'section' in knowledge_probe_df.columns:
        log_info("Generating Plot 2: Knowledge Probe Mean Log Probs by Section...")
        
        plt.figure(figsize=(14, 8))
        
        avg_section_log_probs = knowledge_probe_df.groupby(['step', 'section'])['log_prob'].mean().reset_index()
        
        all_sections = sorted(knowledge_probe_df['section'].unique())
        palette = sns.color_palette("husl", len(all_sections))
        
        sns.lineplot(data=avg_section_log_probs, x='step', y='log_prob', hue='section', hue_order=all_sections, palette=palette)
        
        plt.title('Plot 2: Knowledge Probe Mean Log Probs by Section')
        plt.xlabel('Training Step')
        plt.ylabel('Mean Log Prob')
        plt.grid(True, which="both", ls="--")
        plt.legend(title='Section')
        plt.savefig(os.path.join(plot_output_dir, "plot_2_knowledge_log_probs_by_section.png"))
        plt.close()

    # --- PLOT 3: Disaggregated Knowledge Probes ---
    if not knowledge_probe_df.empty:
        log_info("Generating Plot 3: Disaggregated Knowledge Probes...")
        
        unique_probes = knowledge_probe_df['probe_index'].unique()
        if len(unique_probes) > 0:
            num_probes_to_plot = min(10, len(unique_probes))
            random_probes = np.random.choice(unique_probes, num_probes_to_plot, replace=False)
            
            sample_df = knowledge_probe_df[knowledge_probe_df['probe_index'].isin(random_probes)].copy()

            # Normalize log_prob for each probe
            scaler = MinMaxScaler()
            sample_df['normalized_log_prob'] = sample_df.groupby('probe_index')['log_prob'].transform(
                lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).flatten() if len(x.values) > 1 else x
            )

            g = sns.FacetGrid(sample_df, col="probe_index", col_wrap=2, height=4, aspect=1.5, sharex=True, sharey=False)

            def plot_dual_axis(data, **kwargs):
                ax1 = plt.gca()
                ax2 = ax1.twinx()
                
                hit_cols = [c for c in ['hit_accuracy_at_1', 'hit_accuracy_at_10', 'hit_accuracy_at_100'] if c in data.columns]
                
                # Plot hits on ax1
                for col in hit_cols:
                    sns.lineplot(data=data, x='step', y=col, ax=ax1, label=col.replace('_', ' ').title(), linestyle='--', **kwargs)
                ax1.set_ylabel('Hit Accuracy')
                ax1.set_ylim(0, 1)
                ax1.legend(loc='upper left')

                # Plot normalized log probs on ax2
                sns.lineplot(data=data, x='step', y='normalized_log_prob', ax=ax2, label='Normalized Log Prob', color='purple', **kwargs)
                ax2.set_ylabel('Normalized Log Prob', color='purple')
                ax2.tick_params(axis='y', labelcolor='purple')
                ax2.set_ylim(0, 1)
                ax2.legend(loc='upper right')

            g.map_dataframe(plot_dual_axis)

            # Set titles with facts
            if not knowledge_probes_csv.empty and 'fact' in knowledge_probes_csv.columns:
                for ax, probe_idx in zip(g.axes.flat, g.col_names):
                    probe_idx = int(probe_idx)
                    if probe_idx in knowledge_probes_csv.index:
                        fact = knowledge_probes_csv.loc[probe_idx, 'fact']
                        fact = re.sub(r'\\bm\{([^}]+)\}', r'\1', fact)
                        fact = re.sub(r'\\mathbf\{([^}]+)\}', r'\1', fact)
                        wrapped_title = textwrap.fill(f"Probe {probe_idx}: {fact}", 60)
                        ax.set_title(wrapped_title, fontsize=10)
                    else:
                        ax.set_title(f"Probe {probe_idx}")
            else:
                g.set_titles("Probe {col_name}")

            g.fig.suptitle('Plot 3: Disaggregated Knowledge Probes', y=1.03, fontsize=16)
            g.set_axis_labels("Training Step", "Value")
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            plt.savefig(os.path.join(plot_output_dir, "plot_3_disaggregated_knowledge_probes.png"))
            plt.close()

    # --- PLOT 4: Disaggregated Inference Probes ---
    if not inference_probe_df.empty:
        log_info("Generating Plot 4: Disaggregated Inference Probes...")
        
        unique_probes = inference_probe_df['probe_index'].unique()
        if len(unique_probes) > 0:
            num_probes_to_plot = min(10, len(unique_probes))
            random_probes = np.random.choice(unique_probes, num_probes_to_plot, replace=False)
            
            sample_df = inference_probe_df[inference_probe_df['probe_index'].isin(random_probes)].copy()

            # Normalize log_prob for each probe
            scaler = MinMaxScaler()
            sample_df['normalized_log_prob'] = sample_df.groupby('probe_index')['log_prob'].transform(
                lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).flatten() if len(x.values) > 1 else x
            )

            g = sns.FacetGrid(sample_df, col="probe_index", col_wrap=2, height=4, aspect=1.5, sharex=True, sharey=False)

            def plot_dual_axis_inference(data, **kwargs):
                ax1 = plt.gca()
                ax2 = ax1.twinx()
                
                hit_cols = [c for c in ['hit_accuracy_at_1', 'hit_accuracy_at_10', 'hit_accuracy_at_100'] if c in data.columns]
                
                # Plot hits on ax1
                for col in hit_cols:
                    sns.lineplot(data=data, x='step', y=col, ax=ax1, label=col.replace('_', ' ').title(), linestyle='--', **kwargs)
                ax1.set_ylabel('Hit Accuracy')
                ax1.set_ylim(0, 1)
                ax1.legend(loc='upper left')

                # Plot normalized log probs on ax2
                sns.lineplot(data=data, x='step', y='normalized_log_prob', ax=ax2, label='Normalized Log Prob', color='purple', **kwargs)
                ax2.set_ylabel('Normalized Log Prob', color='purple')
                ax2.tick_params(axis='y', labelcolor='purple')
                ax2.set_ylim(0, 1)
                ax2.legend(loc='upper right')

            g.map_dataframe(plot_dual_axis_inference)

            # Set titles with facts
            if not inference_probes_csv.empty and 'fact' in inference_probes_csv.columns:
                for ax, probe_idx in zip(g.axes.flat, g.col_names):
                    probe_idx = int(probe_idx)
                    if probe_idx in inference_probes_csv.index:
                        fact = inference_probes_csv.loc[probe_idx, 'fact']
                        fact = re.sub(r'\\bm\{([^}]+)\}', r'\1', fact)
                        fact = re.sub(r'\\mathbf\{([^}]+)\}', r'\1', fact)
                        wrapped_title = textwrap.fill(f"Probe {probe_idx}: {fact}", 60)
                        ax.set_title(wrapped_title, fontsize=10)
                    else:
                        ax.set_title(f"Probe {probe_idx}")
            else:
                g.set_titles("Probe {col_name}")

            g.fig.suptitle('Plot 4: Disaggregated Inference Probes', y=1.03, fontsize=16)
            g.set_axis_labels("Training Step", "Value")
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            plt.savefig(os.path.join(plot_output_dir, "plot_4_disaggregated_inference_probes.png"))
            plt.close()


def generate_averaged_plots(experiment_dir: str, logger=None):
    """
    Generates plots averaged across all domains for an experiment.
    """
    def log_info(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)

    # --- Directory for new plots ---
    plot_output_dir = os.path.join(experiment_dir, "_averaged_plots")
    os.makedirs(plot_output_dir, exist_ok=True)
    log_info(f"Saving averaged plots to {plot_output_dir}")

    # --- Data Loading and Aggregation ---
    log_info("Loading and aggregating dataframes for averaged plotting...")
    
    domains = set()
    for subdir in os.listdir(experiment_dir):
        if subdir.endswith("_knowledge_probe"):
            domains.add(subdir.replace("_knowledge_probe", ""))
        elif subdir.endswith("_inference_probe"):
            domains.add(subdir.replace("_inference_probe", ""))

    if not domains:
        log_info("No domains found. Skipping averaged plotting.")
        return

    all_knowledge_dfs = []
    all_inference_dfs = []

    for domain in domains:
        knowledge_path = os.path.join(experiment_dir, f"{domain}_knowledge_probe", f"{domain}_knowledge_probe_metrics.csv")
        if os.path.exists(knowledge_path):
            all_knowledge_dfs.append(pd.read_csv(knowledge_path))
        
        inference_path = os.path.join(experiment_dir, f"{domain}_inference_probe", f"{domain}_inference_probe_metrics.csv")
        if os.path.exists(inference_path):
            all_inference_dfs.append(pd.read_csv(inference_path))

    knowledge_probe_df = pd.concat(all_knowledge_dfs) if all_knowledge_dfs else pd.DataFrame()
    inference_probe_df = pd.concat(all_inference_dfs) if all_inference_dfs else pd.DataFrame()

    if knowledge_probe_df.empty and inference_probe_df.empty:
        log_info("No probe metrics found across any domain. Skipping plotting.")
        return

    # Training loss
    training_loss_path = os.path.join(experiment_dir, "training_loss_perplexity_metrics.csv")
    loss_df = pd.read_csv(training_loss_path) if os.path.exists(training_loss_path) else pd.DataFrame()

    # --- Plot 1: Combined Probes vs. Training Loss (Averaged) ---
    log_info("Generating Averaged Plot 1: Combined Probes vs. Training Loss...")
    
    # Prepare data
    mean_knowledge_log_probs = knowledge_probe_df.groupby('step')['log_prob'].mean().reset_index() if not knowledge_probe_df.empty else pd.DataFrame()
    mean_inference_log_probs = inference_probe_df.groupby('step')['log_prob'].mean().reset_index() if not inference_probe_df.empty else pd.DataFrame()
    mean_knowledge_hits10 = knowledge_probe_df.groupby('step')['hit_accuracy_at_10'].mean().reset_index() if not knowledge_probe_df.empty and 'hit_accuracy_at_10' in knowledge_probe_df.columns else pd.DataFrame()
    mean_inference_hits10 = inference_probe_df.groupby('step')['hit_accuracy_at_10'].mean().reset_index() if not inference_probe_df.empty and 'hit_accuracy_at_10' in inference_probe_df.columns else pd.DataFrame()

    # --- Plot 1a (Averaged) ---
    log_info("Generating Averaged Plot 1a: Mean Log Probs vs. Training Loss...")
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Absolute Mean Log Probs (Averaged Across Domains)')
    
    if not mean_knowledge_log_probs.empty:
        sns.lineplot(data=mean_knowledge_log_probs, x='step', y='log_prob', ax=ax1, label='Knowledge Probes Mean Log Probs', color='blue')
    if not mean_inference_log_probs.empty:
        sns.lineplot(data=mean_inference_log_probs, x='step', y='log_prob', ax=ax1, label='Inference Probes Mean Log Probs', color='green')
    ax1.legend(loc='upper left')

    if not loss_df.empty and 'chunked_perplexity' in loss_df.columns:
        ax2 = ax1.twinx()
        ax2.set_ylabel('Normalized Training Loss (Perplexity)', color='grey')
        scaler = MinMaxScaler()
        loss_df['normalized_loss'] = scaler.fit_transform(loss_df[['chunked_perplexity']])
        sns.lineplot(data=loss_df, x='step', y='normalized_loss', ax=ax2, label='Normalized Training Loss', color='grey', linestyle='--')
        ax2.tick_params(axis='y', labelcolor='grey')
        ax2.legend(loc='upper right')

    plt.title('Plot 1a (Averaged): Mean Log Probs vs. Training Loss')
    plt.grid(True, which="both", ls="--")
    plt.savefig(os.path.join(plot_output_dir, "plot_1a_avg_mean_log_probs_vs_loss.png"))
    plt.close()

    # --- Plot 1b (Averaged) ---
    log_info("Generating Averaged Plot 1b: Mean Hit Accuracy @ 10 vs. Training Loss...")
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Absolute Mean Hit Accuracy @ 10 (Averaged Across Domains)')
    ax1.set_ylim(0, 1)

    if not mean_knowledge_hits10.empty:
        sns.lineplot(data=mean_knowledge_hits10, x='step', y='hit_accuracy_at_10', ax=ax1, label='Knowledge Probes Mean Hits@10', color='blue')
    if not mean_inference_hits10.empty:
        sns.lineplot(data=mean_inference_hits10, x='step', y='hit_accuracy_at_10', ax=ax1, label='Inference Probes Mean Hits@10', color='green')
    ax1.legend(loc='upper left')

    if not loss_df.empty and 'chunked_perplexity' in loss_df.columns:
        ax2 = ax1.twinx()
        ax2.set_ylabel('Normalized Training Loss (Perplexity)', color='grey')
        if 'normalized_loss' not in loss_df.columns: # reuse if already computed
            scaler = MinMaxScaler()
            loss_df['normalized_loss'] = scaler.fit_transform(loss_df[['chunked_perplexity']])
        sns.lineplot(data=loss_df, x='step', y='normalized_loss', ax=ax2, label='Normalized Training Loss', color='grey', linestyle='--')
        ax2.tick_params(axis='y', labelcolor='grey')
        ax2.legend(loc='upper right')

    plt.title('Plot 1b (Averaged): Mean Hit Accuracy @ 10 vs. Training Loss')
    plt.grid(True, which="both", ls="--")
    plt.savefig(os.path.join(plot_output_dir, "plot_1b_avg_mean_hits_at_10_vs_loss.png"))
    plt.close()


    
