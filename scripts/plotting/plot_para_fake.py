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
from scripts.plotting.plot_comparison import aggregate_across_domains
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
    Returns aggregated mean log_prob and hits@10 per step.
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
            
            if 'hit_accuracy_at_10' in df.columns:
                df['hit_accuracy_at_10'] = pd.to_numeric(df['hit_accuracy_at_10'], errors='coerce')
            else:
                df['hit_accuracy_at_10'] = np.nan

            df.dropna(subset=['step', 'log_prob'], inplace=True)
            
            if not df.empty:
                df['step'] = df['step'].astype(int)
                df['domain'] = domain
                all_domain_dfs.append(df)

    if not all_domain_dfs:
        return None

    combined_df = pd.concat(all_domain_dfs, ignore_index=True)
    
    metric_cols = ['log_prob']
    if 'hit_accuracy_at_10' in combined_df.columns:
        metric_cols.append('hit_accuracy_at_10')

    agg = combined_df.groupby('step')[metric_cols].mean().reset_index()
    agg = agg.sort_values('step')
    
    return agg

def filter_steps(df):
    if df is None: return None
    # Logic: Exposure at step 0, then every 2 steps starting from 1 (0, 1, 3, 5...)
    filtered = df[(df['step'] == 0) | ((df['step'] > 0) & ((df['step'] - 1) % 2 == 0))].copy()
    filtered['Exposure Steps'] = (filtered['step'] + 1) // 2
    return filtered

def main():
    set_plot_style()
    project_root = REPO_ROOT
    domains = discover_domains(project_root)
    
    # --- Data Configuration ---
    
    # 1. Trajectory Data for Panel 5 (7B 500 epochs) - Source Only
    traj_path_7b_500 = "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/source_only/sep_1_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e500/bs32_lr2e-05_const/overlap_1_4/11_22_16_47"
    
    # 2. Trajectory Data for Panels 1 & 2 (Compositional: Source & Para 9)
    # Paths from plot_hitsat10_and_logprobs.py
    trajectory_config = {
        "Source": {
            "1B": "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/1b/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4",
            "7B": "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_20_22_51",
            "13B": "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/allenai_OLMo-2-1124-13B/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_21_11_30",
        },
        "Para 9": {
            "1B": "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/1b/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4",
            "7B": "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_20_22_51",
            "13B": "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/allenai_OLMo-2-1124-13B/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_21_13_18",
        }
    }

    # 3. Batch Size Scaling Data
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
                64: "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/allenai_OLMo-2-1124-13B/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_23_22_57",
                128: "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/allenai_OLMo-2-1124-13B/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs128_lr2e-05/overlap_1_4/11_23_22_57",
                256: "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/allenai_OLMo-2-1124-13B/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs256_lr2e-05/overlap_1_4/11_24_01_25",
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
    }
    bs_styles = {
        "Source": {"linestyle": "--", "marker": "s"},
        "Para 9": {"linestyle": "-", "marker": "o"},
    }
    
    # Trajectory Colors (Panel 5)
    color_factual = "#00008B" # Dark Blue
    color_comp = "#87CEEB"    # Light Blue

    # --- Plotting Setup ---
    # 5 panels, equal width, with spacers between (2,3) and (4,5)
    # Layout: [P1] [P2] [Space] [P3] [P4] [Space] [P5]
    # Ratios:  1    1    0.2     1    1    0.2     1
    
    fig = plt.figure(figsize=(20, 5))
    gs = gridspec.GridSpec(1, 7, width_ratios=[1, 1, 0.2, 1, 1, 0.2, 1], wspace=0.1)
    
    axes = []
    axes.append(fig.add_subplot(gs[0, 0])) # Panel 1
    axes.append(fig.add_subplot(gs[0, 1])) # Panel 2
    axes.append(fig.add_subplot(gs[0, 3])) # Panel 3
    axes.append(fig.add_subplot(gs[0, 4])) # Panel 4
    axes.append(fig.add_subplot(gs[0, 6])) # Panel 5
    
    # --- Collect Data for Panels 1 & 2 to Compute Shared Limits ---
    p1_data = [] # List of (model, df)
    p2_data = [] # List of (model, df)
    
    # Panel 1 Data
    for model_size in ["1B", "7B", "13B"]:
        path = trajectory_config["Source"].get(model_size)
        if path:
            df = get_trajectory_data(path, 'inference', domains, project_root)
            # No filtering for Panels 1 & 2
            if df is not None:
                df['Exposure Steps'] = df['step'] # Every step is exposure
                p1_data.append((model_size, df))
                
    # Panel 2 Data
    for model_size in ["1B", "7B", "13B"]:
        path = trajectory_config["Para 9"].get(model_size)
        if path:
            df = get_trajectory_data(path, 'inference', domains, project_root)
            # No filtering for Panels 1 & 2
            if df is not None:
                df['Exposure Steps'] = df['step'] # Every step is exposure
                p2_data.append((model_size, df))

    # Compute Limits for Panels 1 & 2
    all_logprobs = []
    all_hits = []
    for _, df in p1_data + p2_data:
        all_logprobs.extend(df['log_prob'].tolist())
        if 'hit_accuracy_at_10' in df.columns:
            all_hits.extend(df['hit_accuracy_at_10'].dropna().tolist())
            
    lp_ylim = compute_unified_ylim(all_logprobs)
    hits_ylim = compute_unified_ylim(all_hits) if all_hits else None
    
    # Scale down Hits@10 by increasing ymax by 0.5
    if hits_ylim:
        hits_ylim = (hits_ylim[0], hits_ylim[1])

    # --- Panel 1: Source Compositional (1B, 7B, 13B) ---
    print("Plotting Panel 1: Source Compositional...")
    ax1 = axes[0]
    ax1_right = ax1.twinx()
    
    for model_size, df in p1_data:
        # LogProb (Solid)
        ax1.plot(df['Exposure Steps'], df['log_prob'], color=model_colors[model_size], linestyle='-', linewidth=2, label=f"{model_size} LogProb")
        # Hits@10 (Dotted)
        if 'hit_accuracy_at_10' in df.columns:
            ax1_right.plot(df['Exposure Steps'], df['hit_accuracy_at_10'], color=model_colors[model_size], linestyle=':', linewidth=2, label=f"{model_size} Hits@10")
            
    ax1.set_title("Source")
    ax1.set_xlabel("Exposure #")
    ax1.set_ylabel("Log Prob.") # Only leftmost label
    
    # Hide right tick labels for Panel 1
    ax1_right.set_yticklabels([])
    
    apply_ylim([ax1], lp_ylim)
    if hits_ylim:
        apply_ylim([ax1_right], hits_ylim)
        
    ax1.grid(True, alpha=0.1)

    # --- Panel 2: Para 9 Compositional (1B, 7B, 13B) ---
    print("Plotting Panel 2: Para 9 Compositional...")
    ax2 = axes[1]
    ax2_right = ax2.twinx()
    
    for model_size, df in p2_data:
        # LogProb (Solid)
        ax2.plot(df['Exposure Steps'], df['log_prob'], color=model_colors[model_size], linestyle='-', linewidth=2)
        # Hits@10 (Dotted)
        if 'hit_accuracy_at_10' in df.columns:
            ax2_right.plot(df['Exposure Steps'], df['hit_accuracy_at_10'], color=model_colors[model_size], linestyle=':', linewidth=2)

    ax2.set_title("Para. 9")
    ax2.set_xlabel("Exposure #")
    
    # Hide left tick labels for Panel 2
    ax2.set_yticklabels([])
    
    apply_ylim([ax2], lp_ylim)
    if hits_ylim:
        apply_ylim([ax2_right], hits_ylim)
    
    ax2.grid(True, alpha=0.1)

    # --- Panels 3 & 4: Batch Size Scaling ---
    print("Plotting Panels 3 & 4: Batch Size Scaling...")
    
    # Collect BS data
    bs_data = []
    for model, strategies in bs_run_config.items():
        for strategy, batches in strategies.items():
            for bs, path in batches.items():
                resolved = find_latest_run(path)
                if not resolved:
                    continue
                
                # We need final step values
                df_k = aggregate_across_domains(resolved, 'knowledge', domains, split_probes=False, project_root=project_root)
                df_i = aggregate_across_domains(resolved, 'inference', domains, split_probes=False, project_root=project_root)
                
                val_k = get_final_step_value(df_k)
                val_i = get_final_step_value(df_i)
                
                if not np.isnan(val_k):
                    bs_data.append({"Model": model, "Strategy": strategy, "BatchSize": bs, "Type": "Factual", "Value": val_k})
                if not np.isnan(val_i):
                    bs_data.append({"Model": model, "Strategy": strategy, "BatchSize": bs, "Type": "Compositional", "Value": val_i})

    df_bs = pd.DataFrame(bs_data)
    
    # Panel 3: Factual, Panel 4: Compositional
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
        
        # Add Y-label for Panel 3 (index 0 in this loop)
        if i == 0:
            ax.set_ylabel("Final Log Prob.")
        
        # Hide left tick labels for Panel 4 (which is index 1 in this loop)
        if i == 1:
            ax.set_yticklabels([])
            
        ax.grid(True, which="major", ls="-", alpha=0.1)

    # Unified Y-limit for Panels 3 & 4
    if not df_bs.empty:
        ylim = compute_unified_ylim(values=df_bs["Value"].tolist())
        if ylim:
            apply_ylim(ax_bs_list, ylim)

    # --- Panel 5: Training Dynamics (Source Only, Dual Axis) ---
    print("Plotting Panel 5: Training Dynamics...")
    ax5 = axes[4]
    ax5_right = ax5.twinx()
    
    # Knowledge (Factual) - Dark Blue
    df_k = get_trajectory_data(traj_path_7b_500, 'knowledge', domains, project_root)
    df_k = filter_steps(df_k)
    
    if df_k is not None:
        # Log Prob (Solid)
        ax5.plot(df_k['Exposure Steps'], df_k['log_prob'], color=color_factual, linestyle='-', linewidth=2, label="Factual LogProb")
        # Hits@10 (Dotted)
        if 'hit_accuracy_at_10' in df_k.columns:
            ax5_right.plot(df_k['Exposure Steps'], df_k['hit_accuracy_at_10'], color=color_factual, linestyle=':', linewidth=2, label="Factual Hits@10")

    # Inference (Compositional) - Light Blue
    df_i = get_trajectory_data(traj_path_7b_500, 'inference', domains, project_root)
    df_i = filter_steps(df_i)
    
    if df_i is not None:
        # Log Prob (Solid)
        ax5.plot(df_i['Exposure Steps'], df_i['log_prob'], color=color_comp, linestyle='-', linewidth=2, label="Comp. LogProb")
        # Hits@10 (Dotted)
        if 'hit_accuracy_at_10' in df_i.columns:
            ax5_right.plot(df_i['Exposure Steps'], df_i['hit_accuracy_at_10'], color=color_comp, linestyle=':', linewidth=2, label="Comp. Hits@10")

    ax5.set_title("Source")
    ax5.set_xlabel("Exposure #")
    ax5.set_ylabel("Log Prob.")
    ax5_right.set_ylabel("Hits@10")
    ax5.grid(True, alpha=0.1)
    
    # Scale down Hits@10 for Panel 5 as well
    # Get current ylim and add 0.5
    current_ylim = ax5_right.get_ylim()
    ax5_right.set_ylim(0.5, current_ylim[1] + 0.08)

    # --- Legends ---
    
    # Legend for Panels 1 & 2 (Models + Metrics)
    p1_legend_elements = [
        Line2D([0], [0], color='#ffd700', lw=2, linestyle='-', label='1B'),
        Line2D([0], [0], color='#ff7f0e', lw=2, linestyle='-', label='7B'),
        Line2D([0], [0], color='#d62728', lw=2, linestyle='-', label='13B'),
        Line2D([0], [0], color='gray', lw=2, linestyle='-', label='Log Prob'),
        Line2D([0], [0], color='gray', lw=2, linestyle=':', label='Hits@10'),
    ]
    axes[0].legend(handles=p1_legend_elements, loc='lower right', fontsize='x-small', frameon=True)

    # Legend for Panels 3 & 4 (Batch Size)
    bs_legend_elements = [
        Line2D([0], [0], color='gray', lw=2, linestyle='-', label='Para 9'),
        Line2D([0], [0], color='gray', lw=2, linestyle='--', label='Source'),
        Line2D([0], [0], color='#ffd700', lw=2, linestyle='-', label='1B'),
        Line2D([0], [0], color='#ff7f0e', lw=2, linestyle='-', label='7B'),
        Line2D([0], [0], color='#d62728', lw=2, linestyle='-', label='13B'),
    ]
    axes[2].legend(handles=bs_legend_elements, loc='lower left', fontsize='x-small', frameon=True)

    # Legend for Panel 5 (Trajectory)
    traj_legend_elements = [
        Line2D([0], [0], color=color_factual, lw=2, linestyle='-', label='Factual'),
        Line2D([0], [0], color=color_comp, lw=2, linestyle='-', label='Compositional'),
        Line2D([0], [0], color='gray', lw=2, linestyle='-', label='Log Prob'),
        Line2D([0], [0], color='gray', lw=2, linestyle=':', label='Hits@10'),
    ]
    axes[4].legend(handles=traj_legend_elements, loc='lower right', fontsize='x-small', frameon=True)

    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    out_path = os.path.join('plots', 'combined_5panel.pdf')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    main()