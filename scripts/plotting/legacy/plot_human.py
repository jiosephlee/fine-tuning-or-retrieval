import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

# Ensure repo root on path for imports
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(REPO_ROOT)

from utils.llm_plotting import set_plot_style
from scripts.plotting.legacy.plot_comparison import aggregate_across_domains
from scripts.plotting.plot_utils import (
    discover_domains,
    find_latest_run,
    get_final_step_value,
)

def main():
    set_plot_style()
    project_root = REPO_ROOT
    # Restrict to DPO and GRPO as requested
    domains = ["DPO", "GRPO"]

    # --- Data Configuration ---
    # Structure: Strategy -> Model Size -> Path
    data_config = {
        "Para. 9": {
            "7B": os.path.join(project_root, "results/FT/full/7b/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_20_22_51"),
            "13B": os.path.join(project_root, "results/FT/full/allenai_OLMo-2-1124-13B/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_21_13_18"),
            "32B": os.path.join(project_root, "results/FT/full/allenai_OLMo-2-0325-32B/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_21_06_44"),
        },
        "Para. 9 + Humans": {
            "7B": os.path.join(project_root, "results/FT/full/7b/probes_v9/newline2/para9_expl_human_cyclefull/fill_dclm/domains_DPO-GRPO/e100/bs64_lr2e-05/overlap_1_4/11_24_05_15"),
            "13B": os.path.join(project_root, "results/FT/full/allenai_OLMo-2-1124-13B/probes_v9/newline2/para9_expl_human_cyclefull/fill_dclm/domains_DPO-GRPO/e100/bs64_lr2e-05/overlap_1_4/11_24_06_09"),
            "32B": os.path.join(project_root, "results/FT/full/allenai_OLMo-2-0325-32B/probes_v9/newline2/para9_expl_human_cyclefull/fill_dclm/domains_DPO-GRPO/e100/bs64_lr2e-05/overlap_1_4/11_26_12_57"),
        }
    }

    # --- Calculate Deltas ---
    # Delta = (Para. 9 + Humans) - (Para. 9)
    model_sizes = ["7B", "13B", "32B"]
    deltas = {
        "labels": model_sizes,
        "f": [], # Factual Deltas
        "c": []  # Compositional Deltas
    }

    for size in model_sizes:
        # Get Baseline (Para 9)
        base_path = data_config["Para. 9"].get(size)
        base_resolved = find_latest_run(base_path)
        
        base_k = get_final_step_value(aggregate_across_domains(base_resolved, 'knowledge', domains, split_probes=False, project_root=project_root))
        base_i = get_final_step_value(aggregate_across_domains(base_resolved, 'inference', domains, project_root))

        # Get Target (Para 9 + Humans)
        target_path = data_config["Para. 9 + Humans"].get(size)
        target_resolved = find_latest_run(target_path)
        
        target_k = get_final_step_value(aggregate_across_domains(target_resolved, 'knowledge', domains, split_probes=False, project_root=project_root))
        target_i = get_final_step_value(aggregate_across_domains(target_resolved, 'inference', domains, project_root))

        # Calculate Delta
        if not np.isnan(target_k) and not np.isnan(base_k):
            deltas['f'].append(target_k - base_k)
        else:
            deltas['f'].append(0) # Or None

        if not np.isnan(target_i) and not np.isnan(base_i):
            deltas['c'].append(target_i - base_i)
        else:
            deltas['c'].append(0) # Or None

    # --- Plotting ---
    # Make plot thinner (width 4 instead of 8)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_box_aspect(2/3)
    
    x = np.arange(len(deltas['labels']))
    # Thinner bars
    width = 0.25
    
    # Colors (using Orange for Para 9 + Humans theme)
    bar_color = '#ff7f0e' 
    
    # Factual Bars (Solid)
    ax.bar(x - width/2, deltas['f'], width, label='Factual', color=bar_color, alpha=0.8)
    
    # Compositional Bars (Hatched)
    ax.bar(x + width/2, deltas['c'], width, label='Inference', color=bar_color, hatch='//', alpha=0.5, edgecolor='black')
    
    ax.set_xticks(x)
    ax.set_xticklabels(deltas['labels'], fontsize=12)
    ax.set_ylabel(r"$\Delta$ Final Log Prob.", fontsize=12)
    ax.set_xlabel("Model Size", fontsize=12)
    
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(0, color='black', linewidth=0.8)

    # Custom Legend
    legend_elements = [
        mpatches.Patch(facecolor=bar_color, alpha=0.8, label='Factual'),
        mpatches.Patch(facecolor=bar_color, alpha=0.5, hatch='//', edgecolor='black', label='Inference')
    ]
    ax.legend(handles=legend_elements, loc='best', fontsize=10)

    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    out_path = os.path.join('plots', 'human.pdf')
    plt.savefig(out_path, bbox_inches='tight')
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    main()
