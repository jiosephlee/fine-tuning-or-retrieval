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
    compute_unified_ylim,
    apply_ylim,
    get_final_step_value,
)

def plot_batch_scaling_hits(df_bs, model_colors, bs_styles):
    print("Plotting Batch Scaling Target Rank Figure...")
    
    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.15)
    
    metrics = ["Target Rank"]
    probe_types = ["Factual", "Compositional"]
    
    axes_grid = []
    for r in range(1):
        row_axes = []
        for c in range(2):
            row_axes.append(fig.add_subplot(gs[r, c]))
        axes_grid.append(row_axes)
        
    for r, metric in enumerate(metrics):
        for c, p_type in enumerate(probe_types):
            ax = axes_grid[r][c]
            
            if df_bs.empty:
                continue
                
            subset = df_bs[(df_bs["Type"] == p_type) & (df_bs["Metric"] == metric)]
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

            # Titles only on top row
            if r == 0:
                ax.set_title(p_type)
            
            ax.set_xscale('log', base=2)
            ax.set_xticks([32, 64, 128, 256])
            ax.set_xticklabels(['32', '64', '128', '256'])
            
            ax.set_xlabel("Batch Size")
            
            # Y-label on left column
            if c == 0:
                ax.set_ylabel(metric)
            else:
                ax.set_yticklabels([])
                
            ax.grid(True, which="major", ls="-", alpha=0.1)

    if not df_bs.empty:
        for r, metric in enumerate(metrics):
            row_vals = df_bs[df_bs["Metric"] == metric]["Value"].tolist()
            ylim = compute_unified_ylim(values=row_vals)
            if ylim:
                apply_ylim(axes_grid[r], ylim)

    # --- Legend (Bottom Left Panel) ---
    bs_legend_elements = [
        Line2D([0], [0], color='gray', lw=2, linestyle='-', label='Para 9'),
        Line2D([0], [0], color='gray', lw=2, linestyle='--', label='Source'),
        Line2D([0], [0], color='#ffd700', lw=2, linestyle='-', label='1B'),
        Line2D([0], [0], color='#ff7f0e', lw=2, linestyle='-', label='7B'),
        Line2D([0], [0], color='#d62728', lw=2, linestyle='-', label='13B'),
        Line2D([0], [0], color='#9467bd', lw=2, linestyle='-', label='32B'),
    ]
    # Place legend in the first subplot (Top Left) or maybe outside?
    # Previous was lower left of first panel.
    axes_grid[0][0].legend(handles=bs_legend_elements, loc='lower left', fontsize='small', frameon=True)

    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    out_path = os.path.join('plots', 'batch_scaling_target_rank.pdf')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved batch scaling target rank plot to {out_path}")

def main():
    set_plot_style()
    # Override with larger fonts
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

    # --- Data Collection (Target Rank) ---
    bs_data = []
    for model, strategies in bs_run_config.items():
        for strategy, batches in strategies.items():
            for bs, path in batches.items():
                resolved = find_latest_run(path)
                if not resolved:
                    continue
                df_k = aggregate_across_domains(resolved, 'knowledge', domains, split_probes=False, project_root=project_root)
                df_i = aggregate_across_domains(resolved, 'inference', domains, split_probes=False, project_root=project_root)
                
                # Fetch Metrics
                metrics_map = {
                    "Target Rank": 'target_rank',
                }
                
                for m_name, col in metrics_map.items():
                    val_k = get_final_step_value(df_k, value_col=col)
                    val_i = get_final_step_value(df_i, value_col=col)
                    
                    if not np.isnan(val_k):
                        bs_data.append({"Model": model, "Strategy": strategy, "BatchSize": bs, "Type": "Factual", "Metric": m_name, "Value": val_k})
                    if not np.isnan(val_i):
                        bs_data.append({"Model": model, "Strategy": strategy, "BatchSize": bs, "Type": "Compositional", "Metric": m_name, "Value": val_i})

    df_bs = pd.DataFrame(bs_data)

    # --- Generate Plot ---
    plot_batch_scaling_hits(df_bs, model_colors, bs_styles)

if __name__ == "__main__":
    main()
