import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Ensure repo root on path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from utils.llm_plotting import set_plot_style

# --- Helper Functions ---

def aggregate_across_domains(run_path, probe_type, domains):
    """
    Aggregates probe data across multiple domains from a specific run path.
    Adapted to capture both log_prob and hit_accuracy_at_10.
    """
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
        metrics_path = os.path.join(run_path, probe_dir, file_name)
        
        if os.path.exists(metrics_path) and os.path.getsize(metrics_path) > 0:
            try:
                df = pd.read_csv(metrics_path)
            except Exception as e:
                print(f"Warning: Could not read {metrics_path}: {e}")
                continue

            # Check for essential columns
            if 'step' not in df.columns or 'log_prob' not in df.columns:
                continue

            # Normalize numeric columns
            df['step'] = pd.to_numeric(df['step'], errors='coerce')
            df['log_prob'] = pd.to_numeric(df['log_prob'], errors='coerce')
            
            # check for hit_accuracy_at_10 (might be missing in some CSVs)
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
        return pd.DataFrame()

    combined_df = pd.concat(all_domain_dfs, ignore_index=True)
    return combined_df

def get_trajectory_data(run_path, probe_type, domains):
    """
    Returns aggregated mean log_prob and hits@10 per step.
    """
    if not run_path or not os.path.isdir(run_path):
        return None

    df = aggregate_across_domains(run_path, probe_type, domains)
    if df.empty:
        return None

    # Group by step. 
    # Crucially: Do NOT include 'step' in the value columns list, as it's the grouper.
    metric_cols = ['log_prob']
    if 'hit_accuracy_at_10' in df.columns:
        metric_cols.append('hit_accuracy_at_10')

    agg = df.groupby('step')[metric_cols].mean().reset_index()
    agg = agg.sort_values('step')
    
    return agg

def discover_domains_local(project_root):
    domains_path = os.path.join(project_root, 'data/arxiv/cleaned')
    if not os.path.exists(domains_path):
        return []
    return sorted([
        os.path.splitext(f)[0] 
        for f in os.listdir(domains_path) 
        if f.endswith('.tex') and os.path.isfile(os.path.join(domains_path, f))
    ])

# --- Main Plotting Logic ---

def main():
    set_plot_style()
    project_root = REPO_ROOT
    domains = discover_domains_local(project_root)
    if not domains:
        print("No domains found. Aborting.")
        return

    # Configuration: Method -> Model -> Path
    # Using the paths provided in previous turns
    runs_config = {
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

    model_colors = {
        "1B": "#1f77b4",  # Blue
        "7B": "#d62728",  # Red
        "13B": "#2ca02c", # Green
    }
    
    probe_types = [
        ("knowledge", "Factual"),
        ("inference", "Compositional")
    ]

    os.makedirs('plots', exist_ok=True)

    # We create one figure per Method (Source, Para 9)
    for method_name, models_dict in runs_config.items():
        print(f"Generating plot for {method_name}...")
        
        # 1x2 Subplots: Left=Factual, Right=Compositional
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        for idx, (probe_key, probe_title) in enumerate(probe_types):
            ax = axes[idx]
            ax_right = ax.twinx() # For hits@10
            
            # Plot data for each model size
            for model_size in ["1B", "7B", "13B"]:
                path = models_dict.get(model_size)
                if not path: 
                    continue
                
                print(f"  Processing {model_size} {probe_key}...")
                df = get_trajectory_data(path, probe_key, domains)
                
                if df is not None and not df.empty:
                    color = model_colors[model_size]
                    
                    # Log Prob (Solid, Left Axis)
                    ax.plot(df['step'], df['log_prob'], color=color, linestyle='-', linewidth=1.5, alpha=0.9, label=f"{model_size} LogProb")
                    
                    # Hits@10 (Dashed, Right Axis)
                    if 'hit_accuracy_at_10' in df.columns:
                        # Filter out NaNs to avoid plotting gaps if some steps miss metrics
                        valid_hits = df.dropna(subset=['hit_accuracy_at_10'])
                        if not valid_hits.empty:
                            ax_right.plot(valid_hits['step'], valid_hits['hit_accuracy_at_10'], color=color, linestyle='--', linewidth=1.5, alpha=0.9, label=f"{model_size} Hits@10")

            ax.set_title(f"{probe_title} Probes ({method_name})")
            ax.set_xlabel("Steps")
            ax.set_ylabel("Log Probability (Solid)")
            ax_right.set_ylabel("Hits@10 (Dashed)")
            
            # Align grids roughly if possible, or just use default
            ax.grid(True, linestyle='-', alpha=0.1)

        # Custom Legend (One shared legend for the whole figure usually works best, or one per subplot)
        # We'll put a consolidated legend on the right of the figure
        legend_elements = [
            Line2D([0], [0], color=model_colors['1B'], lw=2, label='1B'),
            Line2D([0], [0], color=model_colors['7B'], lw=2, label='7B'),
            Line2D([0], [0], color=model_colors['13B'], lw=2, label='13B'),
            Line2D([0], [0], color='gray', linestyle='-', lw=2, label='Log Prob'),
            Line2D([0], [0], color='gray', linestyle='--', lw=2, label='Hits@10'),
        ]
        
        fig.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1.08, 0.5))
        
        plt.tight_layout()
        safe_name = method_name.lower().replace(" ", "_")
        out_path = os.path.join('plots', f"trajectory_combined_{safe_name}.pdf")
        plt.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {out_path}")

if __name__ == "__main__":
    main()