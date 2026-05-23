import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from cycler import cycler

# Adjust the path to include the utils directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
try:
    from utils.llm_plotting import set_plot_style
except ImportError:
    set_plot_style = None


def aggregate_across_domains(run_path, probe_type, domains):
    """
    Aggregates probe data across multiple domains from a specific run path.
    """
    all_domain_dfs = []
    for domain in domains:
        if probe_type == "knowledge":
            probe_dir = f"{domain}_knowledge_probe"
            file_name = f"{domain}_knowledge_probe_metrics.csv"
        else:  # inference
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
                    df['domain'] = domain
                    all_domain_dfs.append(df)
            else:
                print(f"Warning: 'step' or 'log_prob' column not found in {metrics_path}. Skipping.")
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
    Generate comparison plot for Paraphrase vs Textbook vs Corrupted Textbook.
    """
    parser = argparse.ArgumentParser(description="Compare Paraphrase, Textbook, and Corrupted Textbook runs.")
    parser.add_argument(
        "--model_id",
        type=str,
        default="7b",
        help="The model ID to plot results for (e.g., '7b')."
    )
    parser.add_argument(
        "--project_root",
        type=str,
        default="/Users/jlee0/Desktop/research/fine-tuning-or-retrieval",
        help="Project root directory."
    )
    args = parser.parse_args()

    project_root = os.path.abspath(args.project_root)
    
    # Get domains from cleaned directory
    domains_path = os.path.join(project_root, 'data/arxiv/cleaned')
    try:
        domains = sorted([os.path.splitext(f)[0] for f in os.listdir(domains_path) 
                         if f.endswith('.tex') and os.path.isfile(os.path.join(domains_path, f))])
        print(f"Found domains: {domains}")
    except FileNotFoundError:
        print(f"Error: Domains path '{domains_path}' not found.")
        return

    if not domains:
        print("No domains found. Exiting.")
        return

    # Define the three runs to compare
    runs = {
        "Paraphrase": "results/FT/full/7b/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_22_05_50",
        "Textbook": "results/FT/full/7b/probes_v9/newline2/para9_expl_textbooks_cyclefull/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_21_02_11",
        "Corrupted Textbook": "results/FT/full/7b/probes_v9/newline2/para9_expl_fruit_textbooks_cyclefull/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_22_03_57",
    }

    # Colors for each method (keeping the same colors)
    color_palette = ['#1f77b4', '#2ca02c', '#8c564b']  # Blue, Green, Brown
    method_names_in_order = ["Paraphrase", "Textbook", "Corrupted Textbook"]

    # Aggregate data for each run
    all_data = {}
    for method_name, rel_path in runs.items():
        run_path = os.path.join(project_root, rel_path)
        
        if not os.path.exists(run_path):
            print(f"Warning: Run path not found for {method_name}: {run_path}")
            continue
            
        print(f"Processing {method_name}...")
        
        knowledge_df = aggregate_across_domains(run_path, 'knowledge', domains)
        inference_df = aggregate_across_domains(run_path, 'inference', domains)
        
        if not knowledge_df.empty:
            knowledge_df['method'] = method_name
        if not inference_df.empty:
            inference_df['method'] = method_name
            
        all_data[method_name] = {
            'knowledge': knowledge_df,
            'inference': inference_df
        }

    # Set plot style
    if set_plot_style:
        set_plot_style()

    # Create 1x2 subplot (Factual and Compositional)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot Factual Probes
    ax_knowledge = axes[0]
    ax_knowledge.set_prop_cycle(cycler(color=color_palette))
    
    for method in method_names_in_order:
        if method in all_data and not all_data[method]['knowledge'].empty:
            df = all_data[method]['knowledge']
            plot_df = df.groupby('step')['log_prob'].mean().reset_index()
            ax_knowledge.plot(plot_df['step'], plot_df['log_prob'], label=method, lw=1.6)
    
    ax_knowledge.set_title('Factual Probes')
    ax_knowledge.set_xlabel('Training Steps')
    ax_knowledge.set_ylabel('Mean Log Probability')
    ax_knowledge.legend(loc='lower right', fontsize='medium')
    ax_knowledge.grid(False)

    # Plot Compositional Probes
    ax_compositional = axes[1]
    ax_compositional.set_prop_cycle(cycler(color=color_palette))
    
    for method in method_names_in_order:
        if method in all_data and not all_data[method]['inference'].empty:
            df = all_data[method]['inference']
            plot_df = df.groupby('step')['log_prob'].mean().reset_index()
            ax_compositional.plot(plot_df['step'], plot_df['log_prob'], label=method, lw=1.6)
    
    ax_compositional.set_title('Compositional Probes')
    ax_compositional.set_xlabel('Training Steps')
    ax_compositional.set_ylabel('Mean Log Probability')
    ax_compositional.legend(loc='lower right', fontsize='medium')
    ax_compositional.grid(False)

    # Save plot
    output_dir = os.path.join(project_root, 'plots')
    os.makedirs(output_dir, exist_ok=True)
    output_filename = 'paraphrase_vs_textbook_vs_corrupted_comparison.pdf'
    output_path = os.path.join(output_dir, output_filename)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    main()
