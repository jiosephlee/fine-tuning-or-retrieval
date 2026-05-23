import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# Ensure repo root on path for imports (adjust if needed)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(REPO_ROOT)

# Reuse styling and helpers
from utils.llm_plotting import set_plot_style
from scripts.plotting.plot_utils import (
    discover_domains,
    find_latest_run,
    compute_unified_ylim,
    apply_ylim,
)

def get_trajectory_data(run_path, probe_type, domains, project_root):
    """
    Returns aggregated mean log_prob and target_rank per step.
    """
    if not run_path:
        return None
    
    resolved = find_latest_run(run_path)
    if not resolved or not os.path.isdir(resolved):
        return None

    all_domain_dfs = []
    
    # Map probe_type to folder/file names
    if probe_type == "knowledge":
        probe_dir_suffix = "_knowledge_probe"
        file_suffix = "_knowledge_probe_metrics.csv"
    else: # inference
        probe_dir_suffix = "_inference_probe"
        file_suffix = "_inference_probe_metrics.csv"

    for domain in domains:
        probe_dir = f"{domain}{probe_dir_suffix}"
        file_name = f"{domain}{file_suffix}"
        metrics_path = os.path.join(resolved, probe_dir, file_name)
        
        if os.path.exists(metrics_path) and os.path.getsize(metrics_path) > 0:
            try:
                df = pd.read_csv(metrics_path)
            except Exception:
                continue

            # Check for essential columns
            if 'step' not in df.columns or 'log_prob' not in df.columns:
                continue

            # Normalize numeric columns
            df['step'] = pd.to_numeric(df['step'], errors='coerce')
            df['log_prob'] = pd.to_numeric(df['log_prob'], errors='coerce')
            
            if 'target_rank' in df.columns:
                df['target_rank'] = pd.to_numeric(df['target_rank'], errors='coerce')
            else:
                df['target_rank'] = np.nan

            df.dropna(subset=['step', 'log_prob'], inplace=True)
            
            if not df.empty:
                df['step'] = df['step'].astype(int)
                df['domain'] = domain
                all_domain_dfs.append(df)

    if not all_domain_dfs:
        return None

    combined_df = pd.concat(all_domain_dfs, ignore_index=True)
    
    metric_cols = ['log_prob']
    if 'target_rank' in combined_df.columns:
        metric_cols.append('target_rank')

    agg = combined_df.groupby('step')[metric_cols].mean().reset_index()
    agg = agg.sort_values('step')
    
    return agg

def plot_2panel_factual(p1_data, p2_data, model_colors, lp_ylim, rank_ylim):
    print("Plotting 2-Panel Factual Figure...")
    
    # Layout: [P1] [P2]
    # Ratios:  1    1
    # Width: 10 (approx half of the 4-panel which was 20 for 4 panels + gaps)
    
    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.15)
    
    axes = []
    axes.append(fig.add_subplot(gs[0, 0])) # Panel 1
    axes.append(fig.add_subplot(gs[0, 1])) # Panel 2
    
    # --- Panel 1: Source Factual ---
    ax1 = axes[0]
    ax1_right = ax1.twinx()
    
    for model_size, df in p1_data:
        ax1.plot(df['Exposure Steps'], df['log_prob'], color=model_colors[model_size], linestyle='-', linewidth=2, label=f"{model_size} LogProb")
        if 'target_rank' in df.columns:
            ax1_right.plot(df['Exposure Steps'], df['target_rank'], color=model_colors[model_size], linestyle=':', linewidth=2, label=f"{model_size} Target Rank")
            
    ax1.set_title("Source")
    ax1.set_xlabel("Exposure #")
    ax1.set_ylabel("Log Prob.")
    ax1_right.set_yticklabels([]) # Hide right ticks
    
    apply_ylim([ax1], lp_ylim)
    if rank_ylim:
        apply_ylim([ax1_right], rank_ylim)
    ax1.grid(True, alpha=0.1)

    # --- Panel 2: Para 9 Factual ---
    ax2 = axes[1]
    ax2_right = ax2.twinx()
    
    for model_size, df in p2_data:
        ax2.plot(df['Exposure Steps'], df['log_prob'], color=model_colors[model_size], linestyle='-', linewidth=2)
        if 'target_rank' in df.columns:
            ax2_right.plot(df['Exposure Steps'], df['target_rank'], color=model_colors[model_size], linestyle=':', linewidth=2)

    ax2.set_title("Para. 9")
    ax2.set_xlabel("Exposure #")
    ax2.set_yticklabels([]) # Hide left ticks
    ax2_right.set_ylabel("Target Rank") # Show right label
    
    apply_ylim([ax2], lp_ylim)
    if rank_ylim:
        apply_ylim([ax2_right], rank_ylim)
    ax2.grid(True, alpha=0.1)

    # --- Legends ---
    p1_legend_elements = [
        Line2D([0], [0], color='#ffd700', lw=2, linestyle='-', label='1B'),
        Line2D([0], [0], color='#ff7f0e', lw=2, linestyle='-', label='7B'),
        Line2D([0], [0], color='#d62728', lw=2, linestyle='-', label='13B'),
        Line2D([0], [0], color='#9467bd', lw=2, linestyle='-', label='32B'),
        Line2D([0], [0], color='gray', lw=2, linestyle='-', label='Log Prob'),
        Line2D([0], [0], color='gray', lw=2, linestyle=':', label='Target Rank'),
    ]
    axes[0].legend(handles=p1_legend_elements, loc='lower right', fontsize='small', frameon=True)

    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    out_path = os.path.join('plots', 'factual_scaling.pdf')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved factual scaling plot to {out_path}")

def main():
    set_plot_style()
    # Override with larger fonts (same as plot_para_fake.py)
    plt.rcParams.update({
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "figure.titlesize": 22,
        "axes.titlesize": 20,
    })
    project_root = REPO_ROOT
    domains = discover_domains(project_root)
    
    # --- Data Configuration (Same as plot_para_fake.py) ---
    trajectory_config = {
        "Source": {
            "1B": "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/1b/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs32_lr2e-05/overlap_1_4/11_23_05_24",
            "7B": "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs32_lr2e-05/overlap_1_4/11_23_05_24",
            "13B": "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/allenai_OLMo-2-1124-13B/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs32_lr2e-05",
        },
        "Para 9": {
            "1B": "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/1b/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs32_lr2e-05/overlap_1_4/11_23_07_13",
            "7B": "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs32_lr2e-05/overlap_1_4/11_23_10_36",
            "13B": "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/allenai_OLMo-2-1124-13B/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs32_lr2e-05/overlap_1_4/11_23_07_02",
        },
        "32B": {
            "Para 9": {
                "32B": "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/allenai_OLMo-2-0325-32B/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs32_lr2e-05/overlap_1_4/11_27_05_20",
            },
            "Source": {
                "32B": "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/allenai_OLMo-2-0325-32B/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs32_lr2e-05/overlap_1_4/11_27_05_20",
            },
        },
    }

    # Styles
    model_colors = {
        "1B": "#ffd700",   # Yellow
        "7B": "#ff7f0e",   # Orange
        "13B": "#d62728",  # Red
        "32B": "#9467bd",  # Purple
    }

    # --- Data Collection (FACTUAL) ---
    p1_data = []
    p2_data = []
    
    # Panel 1: Source (Factual)
    # Panel 1: Source (Factual)
    for model_size in ["1B", "7B", "13B", "32B"]:
        path = trajectory_config["Source"].get(model_size)
        if not path and model_size == "32B":
             path = trajectory_config["32B"]["Source"].get("32B")
        
        if path:
            # CHANGED: 'inference' -> 'knowledge'
            df = get_trajectory_data(path, 'knowledge', domains, project_root)
            if df is not None:
                df['Exposure Steps'] = df['step']
                p1_data.append((model_size, df))
                
    # Panel 2: Para 9 (Factual)
    # Panel 2: Para 9 (Factual)
    for model_size in ["1B", "7B", "13B", "32B"]:
        path = trajectory_config["Para 9"].get(model_size)
        if not path and model_size == "32B":
             path = trajectory_config["32B"]["Para 9"].get("32B")

        if path:
            # CHANGED: 'inference' -> 'knowledge'
            df = get_trajectory_data(path, 'knowledge', domains, project_root)
            if df is not None:
                df['Exposure Steps'] = df['step']
                p2_data.append((model_size, df))

    # Compute Limits
    all_logprobs = []
    all_ranks = []
    for _, df in p1_data + p2_data:
        all_logprobs.extend(df['log_prob'].tolist())
        if 'target_rank' in df.columns:
            all_ranks.extend(df['target_rank'].dropna().tolist())
            
    lp_ylim = compute_unified_ylim(all_logprobs)
    rank_ylim = compute_unified_ylim(all_ranks) if all_ranks else None
    
    # --- Generate Plot ---
    plot_2panel_factual(p1_data, p2_data, model_colors, lp_ylim, rank_ylim)

if __name__ == "__main__":
    main()
