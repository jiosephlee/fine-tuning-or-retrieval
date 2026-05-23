import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# Ensure repo root on path for imports (adjust if needed)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(REPO_ROOT)

# Reuse styling and helpers
from utils.llm_plotting import set_plot_style
from scripts.plotting.legacy.plot_comparison import aggregate_across_domains
from scripts.plotting.plot_utils import (
    discover_domains,
    find_latest_run,
    make_subplots,
    compute_unified_ylim,
    apply_ylim,
    add_legend,
    get_final_step_value,
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

def filter_steps(df):
    if df is None: return None
    # Logic: Exposure at step 0, then every 2 steps starting from 1 (0, 1, 3, 5...)
    filtered = df[(df['step'] == 0) | ((df['step'] > 0) & ((df['step'] - 1) % 2 == 0))].copy()
    filtered['Exposure Steps'] = (filtered['step'] + 1) // 2
    return filtered

def plot_4panel(p1_data, p2_data, df_bs, model_colors, bs_styles, lp_ylim, rank_ylim):
    print("Plotting 4-Panel Figure...")
    # Layout: [P1] [P2] [Space] [P3] [P4]
    # Ratios:  1    1    0.2     1    1
    # Total width ratio units: 4.2. Previous was 5.4.
    # Keeping width=20 makes them wider (20/4.2 > 20/5.4)
    
    fig = plt.figure(figsize=(20, 6))
    gs = gridspec.GridSpec(1, 5, width_ratios=[1, 1, 0.2, 1, 1], wspace=0.15)
    
    axes = []
    axes.append(fig.add_subplot(gs[0, 0])) # Panel 1
    axes.append(fig.add_subplot(gs[0, 1])) # Panel 2
    axes.append(fig.add_subplot(gs[0, 3])) # Panel 3
    axes.append(fig.add_subplot(gs[0, 4])) # Panel 4
    
    # --- Panel 1: Source Compositional ---
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

    # --- Panel 2: Para 9 Compositional ---
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
    # ax2_right.set_yticklabels([]) # Show right ticks (Default)
    
    apply_ylim([ax2], lp_ylim)
    if rank_ylim:
        apply_ylim([ax2_right], rank_ylim)
    ax2.grid(True, alpha=0.1)

    # --- Panels 3 & 4: Batch Size Scaling ---
    probe_types = ["Factual", "Compositional"]
    ax_bs_list = [axes[2], axes[3]]
    
    for i, (ax, p_type) in enumerate(zip(ax_bs_list, probe_types)):
        if df_bs.empty:
            continue
            
        subset = df_bs[df_bs["Type"] == p_type]
        pairs = subset[['Model', 'Strategy']].drop_duplicates()
        
        for _, row in pairs.iterrows():
            m = row['Model']
            s = row['Strategy']
            
            series = subset[(subset['Model'] == m) & (subset['Strategy'] == s)].sort_values("BatchSize")
            if series.empty:
                continue
                
            color = model_colors.get(m, 'black')
            ls = bs_styles.get(s, {}).get("linestyle", "-")
            marker = bs_styles.get(s, {}).get("marker", "o")
            
            ax.plot(series['BatchSize'], series['Value'], color=color, linestyle=ls, marker=marker, linewidth=1.5)

        ax.set_title(p_type)
        ax.set_xscale('log', base=2)
        ax.set_xticks([32, 64, 128, 256])
        ax.set_xticklabels(['32', '64', '128', '256'])
        ax.set_xlabel("Batch Size")
        
        if i == 0:
            pass
        else:
            ax.set_yticklabels([])
            
        ax.grid(True, which="major", ls="-", alpha=0.1)

    if not df_bs.empty:
        ylim = compute_unified_ylim(values=df_bs["Value"].tolist())
        if ylim:
            apply_ylim(ax_bs_list, ylim)

    # --- Legends ---
    p1_legend_elements = [
        Line2D([0], [0], color='#ffd700', lw=2, linestyle='-', label='1B'),
        Line2D([0], [0], color='#ff7f0e', lw=2, linestyle='-', label='7B'),
        Line2D([0], [0], color='#d62728', lw=2, linestyle='-', label='13B'),
        Line2D([0], [0], color='#9467bd', lw=2, linestyle='-', label='32B'),
        Line2D([0], [0], color='gray', lw=2, linestyle='-', label='Log Prob'),
        Line2D([0], [0], color='gray', lw=2, linestyle=':', label='Target Rank'),
    ]
    leg1 = axes[0].legend(handles=p1_legend_elements, loc='lower right', fontsize=14, frameon=True)
    leg1.get_frame().set_facecolor('none')

    bs_legend_elements = [
        Line2D([0], [0], color='gray', lw=2, linestyle='-', label='Para 9'),
        Line2D([0], [0], color='gray', lw=2, linestyle='--', label='Source'),
        Line2D([0], [0], color='#ffd700', lw=2, linestyle='-', label='1B'),
        Line2D([0], [0], color='#ff7f0e', lw=2, linestyle='-', label='7B'),
        Line2D([0], [0], color='#d62728', lw=2, linestyle='-', label='13B'),
        Line2D([0], [0], color='#9467bd', lw=2, linestyle='-', label='32B'),
    ]
    leg2 = axes[2].legend(handles=bs_legend_elements, loc='lower left', fontsize=14, frameon=True)
    leg2.get_frame().set_facecolor('none')

    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    out_path = os.path.join('plots', 'combined_4panel.pdf')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved 4-panel plot to {out_path}")

def plot_grokking(df_k_src, df_i_src, df_k_para, df_i_para, color_factual, color_comp):
    print("Plotting Grokking Figure...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    
    # Helper to plot one panel
    def plot_panel(ax, df_k, df_i, title):
        ax_right = ax.twinx()
        
        # Knowledge (Factual)
        if df_k is not None:
            ax.plot(df_k['Exposure Steps'], df_k['log_prob'], color=color_factual, linestyle='-', linewidth=2.5, label="Factual LogProb")
            if 'target_rank' in df_k.columns:
                ax_right.plot(df_k['Exposure Steps'], df_k['target_rank'], color=color_factual, linestyle=':', linewidth=2.0, label="Factual Target Rank")

        # Inference (Compositional)
        if df_i is not None:
            ax.plot(df_i['Exposure Steps'], df_i['log_prob'], color=color_comp, linestyle='-', linewidth=2.5, label="Comp. LogProb")
            if 'target_rank' in df_i.columns:
                ax_right.plot(df_i['Exposure Steps'], df_i['target_rank'], color=color_comp, linestyle=':', linewidth=2.0, label="Comp. Target Rank")

        ax.set_title(title)
        ax.set_xlabel("Exposure #")
        ax.set_ylabel("Log Prob.")
        ax_right.set_ylabel("Target Rank")
        ax.grid(True, alpha=0.1)
        
        # Keep the rank axis away from zero when rank values are available.
        current_ylim = ax_right.get_ylim()
        ax_right.set_ylim(0.5, current_ylim[1] + 0.08)
        
        return ax_right

    # Plot Source
    plot_panel(axes[0], df_k_src, df_i_src, "Source")
    
    # Plot Para
    plot_panel(axes[1], df_k_para, df_i_para, "Para 9")

    # Legend (on the last plot)
    traj_legend_elements = [
        Line2D([0], [0], color=color_factual, lw=2, linestyle='-', label='Factual'),
        Line2D([0], [0], color=color_comp, lw=2, linestyle='-', label='Compositional'),
        Line2D([0], [0], color='gray', lw=2, linestyle='-', label='Log Prob'),
        Line2D([0], [0], color='gray', lw=2, linestyle=':', label='Target Rank'),
    ]
    axes[1].legend(handles=traj_legend_elements, loc='lower right', fontsize='small', frameon=True)

    plt.tight_layout()
    out_path = os.path.join('plots', 'grokking.pdf')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)

def main():
    set_plot_style()
    # Override with larger fonts
    plt.rcParams.update({
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 14,
        "figure.titlesize": 22,
        "axes.titlesize": 20,
    })
    project_root = REPO_ROOT
    domains = discover_domains(project_root)
    
    # --- Data Configuration ---
    # Grokking Paths
    traj_path_7b_500_source = "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/source_only/sep_1_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e500/bs32_lr2e-05_const/overlap_1_4/11_22_16_47"
    traj_path_7b_500_para = "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/para9/sep_1_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e500/bs32_lr2e-05_const/overlap_1_4/11_19_20_24/"
    
    # BS32 Trajectories (Derived from bs_run_config where possible)
    trajectory_config = {
        "Source": {
            "1B": "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/1b/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs32_lr2e-05/overlap_1_4/11_23_05_24",
            "7B": "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs32_lr2e-05/overlap_1_4/11_23_05_24",
            "13B": "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/allenai_OLMo-2-1124-13B/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs32_lr2e-05",
            "32B": "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/allenai_OLMo-2-0325-32B/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs32_lr2e-05/overlap_1_4/11_27_05_20",
        },
        "Para 9": {
            "1B": "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/1b/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs32_lr2e-05/overlap_1_4/11_23_07_13",
            "7B": "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs32_lr2e-05/overlap_1_4/11_23_10_36",
            "13B": "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/allenai_OLMo-2-1124-13B/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs32_lr2e-05/overlap_1_4/11_23_07_02",
            "32B": "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/allenai_OLMo-2-0325-32B/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs32_lr2e-05/overlap_1_4/11_27_05_20",
        }
    }

    bs_run_config = {
        "7B": {
            "Para 9": {
                32: "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs32_lr2e-05/overlap_1_4/11_23_10_36",
                64: "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_20_22_51",
                128: "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs128_lr2e-05/overlap_1_4/11_23_11_36",
                256: "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs256_lr2e-05/overlap_1_4/11_23_13_18",
            },
            "Source": {
                32: "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs32_lr2e-05/overlap_1_4/11_23_05_24",
                64: "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_20_22_51",
                128: "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs128_lr2e-05/overlap_1_4/11_23_06_24",
                256: "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs256_lr2e-05/overlap_1_4/11_23_08_04",
            },
        },
        "13B": {
            "Para 9": {
                32: "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/allenai_OLMo-2-1124-13B/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs32_lr2e-05/overlap_1_4/11_23_07_02",
                64: "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/allenai_OLMo-2-1124-13B/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05",
                128: "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/allenai_OLMo-2-1124-13B/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs128_lr2e-05/overlap_1_4/11_24_05_16",
                256: "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/allenai_OLMo-2-1124-13B/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs256_lr2e-05/overlap_1_4/11_24_07_46",
            },
            "Source": {
                32: "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/allenai_OLMo-2-1124-13B/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs32_lr2e-05",
                64: "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/allenai_OLMo-2-1124-13B/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_21_11_30",
                128: "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/allenai_OLMo-2-1124-13B/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs128_lr2e-05/overlap_1_4/11_23_22_57",
                256: "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/allenai_OLMo-2-1124-13B/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs256_lr2e-05/overlap_1_4/11_24_01_25",
            },
        },
        "32B": {
            "Para 9": {
                32: "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/allenai_OLMo-2-0325-32B/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs32_lr2e-05/overlap_1_4/11_27_05_20",
                64: "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/allenai_OLMo-2-0325-32B/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_21_06_44",
                128: "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/allenai_OLMo-2-0325-32B/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs128_lr2e-05/overlap_1_4/11_27_05_26",
                256: "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/allenai_OLMo-2-0325-32B/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs256_lr2e-05/overlap_1_4/11_27_05_26",
            },
            "Source": {
                32: "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/allenai_OLMo-2-0325-32B/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs32_lr2e-05/overlap_1_4/11_27_05_20",
                64: "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/allenai_OLMo-2-0325-32B/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_21_06_14",
                128: "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/allenai_OLMo-2-0325-32B/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs128_lr2e-05/overlap_1_4/11_27_05_20",
                256: "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/allenai_OLMo-2-0325-32B/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs256_lr2e-05/overlap_1_4/11_27_05_20",
            },
        },
        "1B": {
            "Source": {
                32: "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/1b/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs32_lr2e-05/overlap_1_4/11_23_05_24",
                64: "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/1b/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_23_22_57",
                128: "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/1b/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs128_lr2e-05/overlap_1_4/11_23_05_47",
                256: "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/1b/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs256_lr2e-05/overlap_1_4/11_23_06_23",
            },
            "Para 9": {
                32: "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/1b/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs32_lr2e-05/overlap_1_4/11_23_07_13",
                64: "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/1b/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_23_23_24",
                128: "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/1b/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs128_lr2e-05/overlap_1_4/11_23_07_37",
                256: "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/1b/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs256_lr2e-05/overlap_1_4/11_23_08_13"
            }
        },
    }

    # Styles
    model_colors = {
        "1B": "#ffd700",   # Yellow
        "7B": "#ff7f0e",   # Orange
        "13B": "#d62728",  # Red
        "32B": "#9467bd",  # Purple
    }
    bs_styles = {
        "Source": {"linestyle": "--", "marker": "s"},
        "Para 9": {"linestyle": "-", "marker": "o"},
    }
    color_factual = "#00008B" # Dark Blue
    color_comp = "#87CEEB"    # Light Blue

    # --- Data Collection ---
    # Panels 1 & 2
    p1_data = []
    p2_data = []
    for model_size in ["1B", "7B", "13B", "32B"]:
        path = trajectory_config["Source"].get(model_size)
        if path:
            df = get_trajectory_data(path, 'inference', domains, project_root)
            if df is not None:
                df['Exposure Steps'] = df['step']
                p1_data.append((model_size, df))
                
    for model_size in ["1B", "7B", "13B", "32B"]:
        path = trajectory_config["Para 9"].get(model_size)
        if path:
            df = get_trajectory_data(path, 'inference', domains, project_root)
            if df is not None:
                df['Exposure Steps'] = df['step']
                p2_data.append((model_size, df))

    # Compute Limits for Panels 1 & 2
    all_logprobs = []
    all_ranks = []
    for _, df in p1_data + p2_data:
        all_logprobs.extend(df['log_prob'].tolist())
        if 'target_rank' in df.columns:
            all_ranks.extend(df['target_rank'].dropna().tolist())
            
    lp_ylim = compute_unified_ylim(all_logprobs)
    if lp_ylim:
        lp_ylim = (lp_ylim[0] - 1, lp_ylim[1])
    rank_ylim = compute_unified_ylim(all_ranks) if all_ranks else None
    
    # Panels 3 & 4
    bs_data = []
    for model, strategies in bs_run_config.items():
        for strategy, batches in strategies.items():
            for bs, path in batches.items():
                resolved = find_latest_run(path)
                if not resolved:
                    continue
                df_k = aggregate_across_domains(resolved, 'knowledge', domains, split_probes=False, project_root=project_root)
                df_i = aggregate_across_domains(resolved, 'inference', domains, split_probes=False, project_root=project_root)
                val_k = get_final_step_value(df_k)
                val_i = get_final_step_value(df_i)
                if not np.isnan(val_k):
                    bs_data.append({"Model": model, "Strategy": strategy, "BatchSize": bs, "Type": "Factual", "Value": val_k})
                if not np.isnan(val_i):
                    bs_data.append({"Model": model, "Strategy": strategy, "BatchSize": bs, "Type": "Compositional", "Value": val_i})
    df_bs = pd.DataFrame(bs_data)

    # Panel 5 (Grokking)
    # Source
    df_k_grok_src = get_trajectory_data(traj_path_7b_500_source, 'knowledge', domains, project_root)
    df_k_grok_src = filter_steps(df_k_grok_src)
    df_i_grok_src = get_trajectory_data(traj_path_7b_500_source, 'inference', domains, project_root)
    df_i_grok_src = filter_steps(df_i_grok_src)
    
    # Para
    df_k_grok_para = get_trajectory_data(traj_path_7b_500_para, 'knowledge', domains, project_root)
    df_k_grok_para = filter_steps(df_k_grok_para)
    df_i_grok_para = get_trajectory_data(traj_path_7b_500_para, 'inference', domains, project_root)
    df_i_grok_para = filter_steps(df_i_grok_para)

    # --- Generate Plots ---
    plot_4panel(p1_data, p2_data, df_bs, model_colors, bs_styles, lp_ylim, rank_ylim)
    plot_grokking(df_k_grok_src, df_i_grok_src, df_k_grok_para, df_i_grok_para, color_factual, color_comp)

if __name__ == "__main__":
    main()
