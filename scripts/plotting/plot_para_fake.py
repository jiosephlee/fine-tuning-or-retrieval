import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

def main():
    set_plot_style()
    project_root = REPO_ROOT
    domains = discover_domains(project_root)
    
    # Configuration of runs
    # Structure: Model -> Strategy -> Batch Size -> Path
    run_config = {
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
                128: "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/allenai_OLMo-2-1124-13B/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs128_lr2e-05/overlap_1_4/11_23_08_30",
                256: "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/allenai_OLMo-2-1124-13B/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs256_lr2e-05/overlap_1_4/11_23_08_35",
            },
            "Source": {
                32: "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/allenai_OLMo-2-1124-13B/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs32_lr2e-05",
                64: "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/allenai_OLMo-2-1124-13B/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4",
                128: "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/allenai_OLMo-2-1124-13B/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs128_lr2e-05/overlap_1_4/11_23_06_50",
                256: "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/allenai_OLMo-2-1124-13B/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs256_lr2e-05/overlap_1_4/11_23_06_54",
            },
        },
        "1B": {
            # Note: Paths containing 'source_only' were listed under Para 9 in prompt, corrected here based on path string.
            "Source": {
                32: "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/1b/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs32_lr2e-05/overlap_1_4/11_23_05_24",
                # 64 missing
                128: "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/1b/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs128_lr2e-05/overlap_1_4/11_23_05_47",
                256: "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/1b/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs256_lr2e-05/overlap_1_4/11_23_06_23",
            },
            # Note: Paths containing 'para9' were listed under Source in prompt, corrected here.
            "Para 9": {
                32: "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/1b/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs32_lr2e-05/overlap_1_4/11_23_07_13",
                # 64 is missing
                128: "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/1b/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs128_lr2e-05/overlap_1_4/11_23_07_37",
                256: "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/1b/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs256_lr2e-05/overlap_1_4/11_23_08_13"
            }
        },
    }

    style_map = {
        "1B": {"color": "#1f77b4"},   # Blue
        "7B": {"color": "#d62728"},   # Red
        "13B": {"color": "#2ca02c"},  # Green
        "Source": {"linestyle": ":", "marker": "s"},   # Dashed
        "Para 9": {"linestyle": "-", "marker": "o"},   # Solid
    }

    # Collect data
    data = []
    
    print("Collecting data...")
    for model, strategies in run_config.items():
        for strategy, batches in strategies.items():
            for bs, path in batches.items():
                resolved = find_latest_run(path)
                if not resolved:
                    print(f"  Skipping (not found): {model} {strategy} BS={bs}")
                    continue
                
                # Fetch probe data
                # We simply want the final step average log_prob
                df_k, df_i = None, None
                try:
                    df_k = aggregate_across_domains(resolved, 'knowledge', domains, split_probes=False, project_root=project_root)
                    df_i = aggregate_across_domains(resolved, 'inference', domains, split_probes=False, project_root=project_root)
                except Exception as e:
                    print(f"  Error loading {resolved}: {e}")
                    continue

                val_k = get_final_step_value(df_k)
                val_i = get_final_step_value(df_i)
                
                if not np.isnan(val_k):
                    data.append({"Model": model, "Strategy": strategy, "BatchSize": bs, "Type": "Factual", "Value": val_k})
                if not np.isnan(val_i):
                    data.append({"Model": model, "Strategy": strategy, "BatchSize": bs, "Type": "Compositional", "Value": val_i})

    df = pd.DataFrame(data)
    if df.empty:
        print("No data collected.")
        return

    # Plotting
    fig, axes = make_subplots(1, 2, figsize=(10, 4), sharey=True)
    
    probe_types = ["Factual", "Compositional"]
    
    for ax, p_type in zip(axes, probe_types):
        subset = df[df["Type"] == p_type]
        
        # Unique Model/Strategy pairs
        pairs = subset[['Model', 'Strategy']].drop_duplicates()
        
        for _, row in pairs.iterrows():
            m = row['Model']
            s = row['Strategy']
            
            series = subset[(subset['Model'] == m) & (subset['Strategy'] == s)].sort_values("BatchSize")
            if series.empty:
                continue
                
            color = style_map[m]["color"]
            ls = style_map[s]["linestyle"]
            marker = style_map[s]["marker"]
            
            # Label only if it's the first time to help with custom legend later (or we can just do custom legend)
            ax.plot(series['BatchSize'], series['Value'], color=color, linestyle=ls, marker=marker, linewidth=1.5, label=f"{m} {s}")

        ax.set_title(p_type)
        ax.set_xscale('log', base=2)
        ax.set_xticks([32, 64, 128, 256])
        ax.set_xticklabels(['32', '64', '128', '256'])
        ax.set_xlabel("Batch Size")
        if p_type == "Factual":
            ax.set_ylabel("Final Step Log Prob")
        ax.grid(True, which="major", ls="-", alpha=0.1)

    # Apply unified Y-limit
    ylim = compute_unified_ylim(values=df["Value"].tolist())
    if ylim:
        apply_ylim(axes, ylim)

    # Custom Legend
    legend_elements = [
        Line2D([0], [0], color='gray', lw=2, linestyle='-', label='Para 9'),
        Line2D([0], [0], color='gray', lw=2, linestyle=':', label='Source'),
        Line2D([0], [0], color='#1f77b4', lw=2, linestyle='-', label='1B'),
        Line2D([0], [0], color='#d62728', lw=2, linestyle='-', label='7B'),
        Line2D([0], [0], color='#2ca02c', lw=2, linestyle='-', label='13B'),
    ]
    # Add legend to the right of the last subplot
    axes[1].legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=True)

    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    out_path = os.path.join('plots', 'batch_size_scaling.pdf')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    main()