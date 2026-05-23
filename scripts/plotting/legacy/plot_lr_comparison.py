import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from datetime import datetime
import json
import argparse
from cycler import cycler

# Adjust the path to include the utils directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils.llm_plotting import set_plot_style


def aggregate_across_domains(run_path, probe_type, domains, project_root='.'):
    """
    Aggregates probe data across multiple domains from a specific run path.
    """
    all_domain_dfs = []
    for domain in domains:
        if probe_type == "knowledge":
            probe_dir = f"{domain}_knowledge_probe"
            file_name = f"{domain}_knowledge_probe_metrics.csv"
        else: # inference
            probe_dir = f"{domain}_inference_probe"
            file_name = f"{domain}_inference_probe_metrics.csv"
            
        metrics_path = os.path.join(run_path, probe_dir, file_name)
        
        if os.path.exists(metrics_path) and os.path.getsize(metrics_path) > 0:
            df = pd.read_csv(metrics_path)
            if 'step' in df.columns and 'log_prob' in df.columns:
                df['step'] = pd.to_numeric(df['step'], errors='coerce')
                df['log_prob'] = pd.to_numeric(df['log_prob'], errors='coerce')
                df.dropna(subset=['step', 'log_prob'], inplace=True)
                if not df.empty:
                    df['step'] = df['step'].astype(int)
                else:
                    continue
            else:
                print(f"Warning: 'step' or 'log_prob' column not found in {metrics_path}. Skipping.")
                continue

            df['domain'] = domain
            all_domain_dfs.append(df)
        else:
            if not os.path.exists(metrics_path):
                print(f"Warning: File not found at {metrics_path}")
            else:
                print(f"Warning: File is empty at {metrics_path}")

    if not all_domain_dfs:
        return pd.DataFrame()

    combined_df = pd.concat(all_domain_dfs, ignore_index=True)
    
    return combined_df


def main():
    """
    Main function to generate the comparison plot.
    """
    parser = argparse.ArgumentParser(description="Generate comparison plots for knowledge and inference probes based on learning rates.")
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
    
    # Define paths for experiments
    experiments = {
        '1B': {
            '1e-5': 'results/FT/full/7b/probes_v9/newline2/source_only/sep_1_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e50/bs32_lr1e-05/overlap_1_4/09_24_12_52',
            '2e-5': 'results/FT/full/7b/probes_v9/newline2/source_only/sep_1_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e50/bs32_lr2e-05/run',
            '4e-5': 'results/FT/full/7b/probes_v9/newline2/source_only/sep_1_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e50/bs32_lr4e-05/overlap_1_4/09_24_18_48'
        },
        '7B': {
            '1e-5': 'results/FT/full/7b/probes_v9/newline2/source_only/sep_1_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e50/bs32_lr1e-05/overlap_1_4/09_24_12_52',
            '2e-5': 'results/FT/full/7b/probes_v9/newline2/source_only/sep_1_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e50/bs32_lr2e-05/run',
            '4e-5': 'results/FT/full/7b/probes_v9/newline2/source_only/sep_1_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e50/bs32_lr4e-05/overlap_1_4/09_24_18_48'
        }
    }

    # --- Configuration ---
    color_palette = ['#2ca02c', '#ff7f0e', '#1f77b4']  # Green, Orange, Blue

    domains_path = os.path.join(project_root, 'data/arxiv/cleaned')
    try:
        domains = sorted([os.path.splitext(f)[0] for f in os.listdir(domains_path) if f.endswith('.tex') and os.path.isfile(os.path.join(domains_path, f))])
        print(f"Dynamically found domains: {domains}")
    except FileNotFoundError:
        print(f"Error: Domains path '{domains_path}' not found. Cannot determine domains.")
        domains = []
        
    output_dir = 'plots'
    os.makedirs(output_dir, exist_ok=True)

    if not domains:
        print("No domains found. Exiting.")
        return

    # --- Data Aggregation ---
    all_data = {
        '1B': {'knowledge': pd.DataFrame(), 'inference': pd.DataFrame()},
        '7B': {'knowledge': pd.DataFrame(), 'inference': pd.DataFrame()}
    }

    for model_id, lrs in experiments.items():
        print(f"Processing model: {model_id}")
        all_knowledge_data = []
        all_inference_data = []
        for lr, path in lrs.items():
            print(f"  Learning Rate: {lr}")
            if not os.path.isdir(path):
                print(f"    Warning: Path not found {path}")
                continue
            
            knowledge_df = aggregate_across_domains(path, 'knowledge', domains, project_root=project_root)
            if not knowledge_df.empty:
                knowledge_df['lr'] = lr
                all_knowledge_data.append(knowledge_df)

            inference_df = aggregate_across_domains(path, 'inference', domains, project_root=project_root)
            if not inference_df.empty:
                inference_df['lr'] = lr
                all_inference_data.append(inference_df)
        
        if all_knowledge_data:
            all_data[model_id]['knowledge'] = pd.concat(all_knowledge_data, ignore_index=True)
        if all_inference_data:
            all_data[model_id]['inference'] = pd.concat(all_inference_data, ignore_index=True)


    # --- Plotting ---
    print("Generating comparison plot...")
    set_plot_style()

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
    
    plot_titles = ['1B: Factual Probes', '1B: Compositional Probes', '7B: Factual Probes', '7B: Compositional Probes']
    plot_data = [
        all_data['1B']['knowledge'],
        all_data['1B']['inference'],
        all_data['7B']['knowledge'],
        all_data['7B']['inference']
    ]
    
    learning_rates = ['1e-5', '2e-5', '4e-5']

    for i, (df, title) in enumerate(zip(plot_data, plot_titles)):
        ax = axes[i]
        ax.set_prop_cycle(color=color_palette)
        if not df.empty:
            for lr in learning_rates:
                lr_df = df[df['lr'] == lr]
                if not lr_df.empty:
                    plot_df = lr_df.groupby('step')['log_prob'].mean().reset_index()
                    ax.plot(plot_df['step'], plot_df['log_prob'], label=lr, lw=1.6)
        
        ax.set_title(title)
        ax.set_xlabel('Training Steps')
        if i == 0:
            ax.set_ylabel('Mean Log Probability')
        
        ax.grid(False)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, loc='lower right', fontsize='medium', title="Learning Rate")

    fig.tight_layout()
    
    output_filename = 'lr_comparison_1B_7B.pdf'
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
