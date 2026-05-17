import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import textwrap
import re
from utils import probe_paths


def _read_csv_if_exists(path: str) -> pd.DataFrame:
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


def _mean_by_step(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if df.empty or metric not in df.columns:
        return pd.DataFrame()
    return df.groupby('step')[metric].mean().reset_index()


def _load_probe_metric_csvs(probe_dir: str, file_suffix: str) -> pd.DataFrame:
    """Load one or more metric CSVs from a probe result directory."""
    if not os.path.isdir(probe_dir):
        return pd.DataFrame()

    metric_paths = sorted(
        os.path.join(probe_dir, name)
        for name in os.listdir(probe_dir)
        if name.endswith(file_suffix)
    )
    if not metric_paths:
        return pd.DataFrame()

    dfs = []
    for path in metric_paths:
        df = pd.read_csv(path)
        if not df.empty:
            df = df.copy()
            df['metric_file'] = os.path.basename(path)
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def _load_inference_probe_metadata(domain: str, inference_probes_version: str) -> pd.DataFrame:
    domain_source = probe_paths.infer_domain_source(domain)
    path1, path2 = [
        str(path)
        for path in probe_paths.resolve_inference_probe_candidates(
            domain,
            inference_probes_version,
            domain_source=domain_source,
        )
    ]
    if os.path.exists(path1):
        return pd.read_csv(path1)
    if os.path.exists(path2):
        return pd.read_csv(path2)
    return pd.DataFrame()


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


def generate_revamped_plots(
    domain: str,
    knowledge_probes_version: str,
    inference_probes_version: str,
    experiment_dir: str,
    logger=None,
    knowledge_probe_filename_suffix: str = "",
):
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
    knowledge_probe_dir = os.path.join(experiment_dir, f"{domain}_knowledge_probe")
    inference_probe_dir = os.path.join(experiment_dir, f"{domain}_inference_probe")
    knowledge_probe_metrics_path = os.path.join(knowledge_probe_dir, f"{domain}_knowledge_probe_metrics.csv")
    paraphrased_knowledge_probe_metrics_path = os.path.join(
        knowledge_probe_dir, f"{domain}_knowledge_probe_paraphrased_metrics.csv"
    )
    
    knowledge_probe_df = _read_csv_if_exists(knowledge_probe_metrics_path)
    paraphrased_knowledge_probe_df = _read_csv_if_exists(paraphrased_knowledge_probe_metrics_path)
    inference_probe_df = _load_probe_metric_csvs(inference_probe_dir, "_inference_probe_metrics.csv")

    if knowledge_probe_df.empty and paraphrased_knowledge_probe_df.empty and inference_probe_df.empty:
        log_info("No probe metrics found. Skipping plotting.")
        return

    # Training loss
    training_loss_path = os.path.join(experiment_dir, "training_loss_perplexity_metrics.csv")
    loss_df = _read_csv_if_exists(training_loss_path)

    # Knowledge Probes CSV for facts and sections
    domain_source = probe_paths.infer_domain_source(domain)
    knowledge_probes_path = str(
        probe_paths.resolve_knowledge_probe_path(
            domain,
            knowledge_probes_version,
            domain_source=domain_source,
            filename_suffix=knowledge_probe_filename_suffix,
        )
    )
    knowledge_probes_csv = _read_csv_if_exists(knowledge_probes_path)
    
    # Inference Probes CSV for facts
    inference_probes_csv = _load_inference_probe_metadata(domain, inference_probes_version)

    # Map sections to knowledge_probe_df
    if not knowledge_probe_df.empty and not knowledge_probes_csv.empty and 'section' in knowledge_probes_csv.columns:
        section_map = knowledge_probes_csv['section'].to_dict()
        knowledge_probe_df['section'] = knowledge_probe_df['probe_index'].map(section_map)

    # --- Plot 1: Combined Probes vs. Training Loss ---
    log_info("Generating Plot 1: Combined Probes vs. Training Loss...")
    
    # Prepare data
    mean_knowledge_log_probs = _mean_by_step(knowledge_probe_df, 'log_prob')
    mean_paraphrased_knowledge_log_probs = _mean_by_step(paraphrased_knowledge_probe_df, 'log_prob')
    mean_inference_log_probs = _mean_by_step(inference_probe_df, 'log_prob')
    mean_knowledge_rank = _mean_by_step(knowledge_probe_df, 'target_rank')
    mean_paraphrased_knowledge_rank = _mean_by_step(paraphrased_knowledge_probe_df, 'target_rank')
    mean_inference_rank = _mean_by_step(inference_probe_df, 'target_rank')

    # --- Plot 1a ---
    log_info("Generating Plot 1a: Mean Log Probs vs. Training Loss...")
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Absolute Mean Log Probs')
    
    if not mean_knowledge_log_probs.empty:
        sns.lineplot(data=mean_knowledge_log_probs, x='step', y='log_prob', ax=ax1, label='Knowledge Probes Mean Log Probs', color='blue')
    if not mean_paraphrased_knowledge_log_probs.empty:
        sns.lineplot(data=mean_paraphrased_knowledge_log_probs, x='step', y='log_prob', ax=ax1, label='Paraphrased Knowledge Probes Mean Log Probs', color='orange')
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
    log_info("Generating Plot 1b: Mean Target Rank vs. Training Loss...")
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Mean Target Rank (lower is better)')

    if not mean_knowledge_rank.empty:
        sns.lineplot(data=mean_knowledge_rank, x='step', y='target_rank', ax=ax1, label='Knowledge Probes Mean Target Rank', color='blue')
    if not mean_paraphrased_knowledge_rank.empty:
        sns.lineplot(data=mean_paraphrased_knowledge_rank, x='step', y='target_rank', ax=ax1, label='Paraphrased Knowledge Probes Mean Target Rank', color='orange')
    if not mean_inference_rank.empty:
        sns.lineplot(data=mean_inference_rank, x='step', y='target_rank', ax=ax1, label='Inference Probes Mean Target Rank', color='green')
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

    plt.title('Plot 1b: Mean Target Rank vs. Training Loss')
    plt.grid(True, which="both", ls="--")
    plt.savefig(os.path.join(plot_output_dir, "plot_1b_mean_target_rank_vs_loss.png"))
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
                
                if 'target_rank' in data.columns:
                    sns.lineplot(data=data, x='step', y='target_rank', ax=ax1, label='Target Rank', linestyle='--', **kwargs)
                ax1.set_ylabel('Target Rank (lower is better)')
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
    if not inference_probe_df.empty and 'log_prob' in inference_probe_df.columns:
        log_info("Generating Plot 4: Disaggregated Inference Probes...")
        
        split_metric_files = (
            'metric_file' in inference_probe_df.columns
            and inference_probe_df['metric_file'].nunique() > 1
        )
        facet_col = 'plot_probe_id' if split_metric_files else 'probe_index'
        plot_df = inference_probe_df.copy()
        if split_metric_files:
            plot_df[facet_col] = (
                plot_df['metric_file'].str.replace('_metrics.csv', '', regex=False)
                + '#'
                + plot_df['probe_index'].astype(str)
            )

        unique_probes = plot_df[facet_col].unique()
        if len(unique_probes) > 0:
            num_probes_to_plot = min(10, len(unique_probes))
            random_probes = np.random.choice(unique_probes, num_probes_to_plot, replace=False)
            
            sample_df = plot_df[plot_df[facet_col].isin(random_probes)].copy()

            # Normalize log_prob for each probe
            scaler = MinMaxScaler()
            sample_df['normalized_log_prob'] = sample_df.groupby(facet_col)['log_prob'].transform(
                lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).flatten() if len(x.values) > 1 else x
            )

            g = sns.FacetGrid(sample_df, col=facet_col, col_wrap=2, height=4, aspect=1.5, sharex=True, sharey=False)

            def plot_dual_axis_inference(data, **kwargs):
                ax1 = plt.gca()
                ax2 = ax1.twinx()
                
                if 'target_rank' in data.columns:
                    sns.lineplot(data=data, x='step', y='target_rank', ax=ax1, label='Target Rank', linestyle='--', **kwargs)
                ax1.set_ylabel('Target Rank (lower is better)')
                ax1.legend(loc='upper left')

                # Plot normalized log probs on ax2
                sns.lineplot(data=data, x='step', y='normalized_log_prob', ax=ax2, label='Normalized Log Prob', color='purple', **kwargs)
                ax2.set_ylabel('Normalized Log Prob', color='purple')
                ax2.tick_params(axis='y', labelcolor='purple')
                ax2.set_ylim(0, 1)
                ax2.legend(loc='upper right')

            g.map_dataframe(plot_dual_axis_inference)

            # Set titles with facts
            if not split_metric_files and not inference_probes_csv.empty and 'fact' in inference_probes_csv.columns:
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
    all_paraphrased_knowledge_dfs = []
    all_inference_dfs = []

    for domain in domains:
        knowledge_path = os.path.join(experiment_dir, f"{domain}_knowledge_probe", f"{domain}_knowledge_probe_metrics.csv")
        knowledge_df = _read_csv_if_exists(knowledge_path)
        if not knowledge_df.empty:
            all_knowledge_dfs.append(knowledge_df)

        paraphrased_knowledge_path = os.path.join(
            experiment_dir,
            f"{domain}_knowledge_probe",
            f"{domain}_knowledge_probe_paraphrased_metrics.csv",
        )
        paraphrased_knowledge_df = _read_csv_if_exists(paraphrased_knowledge_path)
        if not paraphrased_knowledge_df.empty:
            all_paraphrased_knowledge_dfs.append(paraphrased_knowledge_df)
        
        inference_dir = os.path.join(experiment_dir, f"{domain}_inference_probe")
        inference_df = _load_probe_metric_csvs(inference_dir, "_inference_probe_metrics.csv")
        if not inference_df.empty:
            all_inference_dfs.append(inference_df)

    knowledge_probe_df = pd.concat(all_knowledge_dfs) if all_knowledge_dfs else pd.DataFrame()
    paraphrased_knowledge_probe_df = (
        pd.concat(all_paraphrased_knowledge_dfs)
        if all_paraphrased_knowledge_dfs
        else pd.DataFrame()
    )
    inference_probe_df = pd.concat(all_inference_dfs) if all_inference_dfs else pd.DataFrame()

    if knowledge_probe_df.empty and paraphrased_knowledge_probe_df.empty and inference_probe_df.empty:
        log_info("No probe metrics found across any domain. Skipping plotting.")
        return

    # Training loss
    training_loss_path = os.path.join(experiment_dir, "training_loss_perplexity_metrics.csv")
    loss_df = _read_csv_if_exists(training_loss_path)

    # --- Plot 1: Combined Probes vs. Training Loss (Averaged) ---
    log_info("Generating Averaged Plot 1: Combined Probes vs. Training Loss...")
    
    # Prepare data
    mean_knowledge_log_probs = _mean_by_step(knowledge_probe_df, 'log_prob')
    mean_paraphrased_knowledge_log_probs = _mean_by_step(paraphrased_knowledge_probe_df, 'log_prob')
    mean_inference_log_probs = _mean_by_step(inference_probe_df, 'log_prob')
    mean_knowledge_rank = _mean_by_step(knowledge_probe_df, 'target_rank')
    mean_paraphrased_knowledge_rank = _mean_by_step(paraphrased_knowledge_probe_df, 'target_rank')
    mean_inference_rank = _mean_by_step(inference_probe_df, 'target_rank')

    # --- Plot 1a (Averaged) ---
    log_info("Generating Averaged Plot 1a: Mean Log Probs vs. Training Loss...")
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Absolute Mean Log Probs (Averaged Across Domains)')
    
    if not mean_knowledge_log_probs.empty:
        sns.lineplot(data=mean_knowledge_log_probs, x='step', y='log_prob', ax=ax1, label='Knowledge Probes Mean Log Probs', color='blue')
    if not mean_paraphrased_knowledge_log_probs.empty:
        sns.lineplot(data=mean_paraphrased_knowledge_log_probs, x='step', y='log_prob', ax=ax1, label='Paraphrased Knowledge Probes Mean Log Probs', color='orange')
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
    log_info("Generating Averaged Plot 1b: Mean Target Rank vs. Training Loss...")
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Mean Target Rank (lower is better; averaged across domains)')

    if not mean_knowledge_rank.empty:
        sns.lineplot(data=mean_knowledge_rank, x='step', y='target_rank', ax=ax1, label='Knowledge Probes Mean Target Rank', color='blue')
    if not mean_paraphrased_knowledge_rank.empty:
        sns.lineplot(data=mean_paraphrased_knowledge_rank, x='step', y='target_rank', ax=ax1, label='Paraphrased Knowledge Probes Mean Target Rank', color='orange')
    if not mean_inference_rank.empty:
        sns.lineplot(data=mean_inference_rank, x='step', y='target_rank', ax=ax1, label='Inference Probes Mean Target Rank', color='green')
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

    plt.title('Plot 1b (Averaged): Mean Target Rank vs. Training Loss')
    plt.grid(True, which="both", ls="--")
    plt.savefig(os.path.join(plot_output_dir, "plot_1b_avg_mean_target_rank_vs_loss.png"))
    plt.close()


    
