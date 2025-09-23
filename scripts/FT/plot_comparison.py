import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
from datetime import datetime
import json
import argparse

# Adjust the path to include the utils directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

def find_latest_run_path(base_path):
    """
    Finds the path to the latest run data within a given base experiment directory,
    following a specific priority for subdirectories.
    """
    try:
        # Find domains_* directory (assuming one)
        domain_dirs = [d for d in os.listdir(base_path) if d.startswith('domains_') and os.path.isdir(os.path.join(base_path, d))]
        if not domain_dirs: return None
        # Let's assume we take the first one if multiple exist, or sort by modification time
        domain_dir = domain_dirs[0]
        
        path = os.path.join(base_path, domain_dir, 'e100')

        # Check for learning rate / batch size folder
        sub_dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        if not sub_dirs: return None
        # Assuming one bs/lr directory
        path = os.path.join(path, sub_dirs[0])
        
        run_dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

        # Priority 1: Check for 'overlap_1_4'
        if 'overlap_1_4' in run_dirs:
            return os.path.join(path, 'overlap_1_4')

        # Priority 2: Check for latest date-stamped directory
        date_dirs = [d for d in run_dirs if re.match(r'\d{2}_\d{2}_\d{2}_\d{2}', d)]
        if date_dirs:
            latest_date = max(date_dirs, key=lambda d: datetime.strptime(d, '%m_%d_%H_%M'))
            return os.path.join(path, latest_date)

        # Priority 3: Check for 'run'
        if 'run' in run_dirs:
            return os.path.join(path, 'run')
            
        # Priority 4: Check for 'no_overlap' as a fallback from previous attempt
        if 'no_overlap' in run_dirs:
            return os.path.join(path, 'no_overlap')

    except FileNotFoundError:
        return None

    return None


def aggregate_across_domains(run_path, probe_type, domains, split_probes=False, project_root='.'):
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
            df['domain'] = domain
            
            if probe_type == 'inference' and split_probes:
                filter_path = os.path.join(project_root, 'data/probes/inference', domain, 'filter.json')
                if os.path.exists(filter_path):
                    with open(filter_path, 'r') as f:
                        filter_data = json.load(f)
                    
                    explanations_only_indices = filter_data.get('in_explanations_only', [])
                    source_only_indices = filter_data.get('in_source_only', [])
                    
                    df['origin'] = 'Other'
                    df.loc[df['probe_index'].isin(explanations_only_indices), 'origin'] = 'Explanations Only'
                    df.loc[df['probe_index'].isin(source_only_indices), 'origin'] = 'Source Only'
                else:
                    print(f"Warning: filter.json not found for domain {domain}")
                    df['origin'] = 'Unknown'

            all_domain_dfs.append(df)
        else:
            if not os.path.exists(metrics_path):
                print(f"Warning: File not found at {metrics_path}")
            else:
                print(f"Warning: File is empty at {metrics_path}")

    if not all_domain_dfs:
        return pd.DataFrame()

    combined_df = pd.concat(all_domain_dfs, ignore_index=True)
    
    # Average across domains for each step
    if probe_type == 'inference' and split_probes:
        grouping_cols = ['step', 'origin']
    else:
        grouping_cols = ['step']
    
    averaged_df = combined_df.groupby(grouping_cols)['log_prob'].mean().reset_index()
    
    return averaged_df

def check_step_consistency(df: pd.DataFrame, probe_type: str):
    """
    Checks if all methods have the same number of training steps and warns if they do not.
    """
    if df.empty or 'method' not in df.columns or 'step' not in df.columns:
        return

    step_counts = df.groupby('method')['step'].nunique()
    
    if step_counts.nunique() > 1:
        print(f"Warning for {probe_type} probes: Not all methods have the same number of training steps.")
        print(step_counts)

def main():
    """
    Main function to generate the comparison plot.
    """
    parser = argparse.ArgumentParser(description="Generate comparison plots for knowledge and inference probes.")
    parser.add_argument(
        "--model_id", 
        type=str, 
        default="1b", 
        help="The model ID to plot results for (e.g., '1b', '7b')."
    )
    parser.add_argument(
        "--split_probes", 
        action='store_true', 
        help="If set, splits the inference probes into separate plots based on their origin."
    )
    args = parser.parse_args()

    # --- Configuration ---
    split_probes = args.split_probes
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    experiment_base_path = f'results/FT/full/{args.model_id}/probes_v9/newline2'
    methods = [
        ('source_only', 'Source Only', 'sep_1_dclm'),
        ('para9', 'Para 9', 'sep_1_dclm'),
        ('para9_expl', 'Para 9 Exp', 'sep_1_dclm'),
    ]
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
    all_knowledge_data = []
    all_inference_data = []

    for method_key, method_name, sub_dir in methods:
        print(f"Processing method: {method_key}")
        method_base_path = os.path.join(experiment_base_path, method_key, sub_dir)
        run_path = find_latest_run_path(method_base_path)

        if run_path:
            # Knowledge Probes
            knowledge_df = aggregate_across_domains(run_path, 'knowledge', domains, project_root=project_root)
            if not knowledge_df.empty:
                knowledge_df['method'] = method_name
                all_knowledge_data.append(knowledge_df)
            
            # Inference Probes
            inference_df = aggregate_across_domains(run_path, 'inference', domains, split_probes=split_probes, project_root=project_root)
            if not inference_df.empty:
                inference_df['method'] = method_name
                all_inference_data.append(inference_df)
        else:
            print(f"Could not find a valid run path for {method_key}")


    if not all_knowledge_data or not all_inference_data:
        print("Error: No data was aggregated. Check paths and file availability.")
        return

    final_knowledge_df = pd.concat(all_knowledge_data, ignore_index=True)
    final_inference_df = pd.concat(all_inference_data, ignore_index=True)
    
    # Check for step consistency
    check_step_consistency(final_knowledge_df, "Knowledge")
    check_step_consistency(final_inference_df, "Inference")

    # --- Plotting ---
    print("Generating comparison plot...")

    if split_probes:
        fig, axes = plt.subplots(2, 2, figsize=(22, 18), sharey=True)
        fig.suptitle('Comparison of 1B Models: Averaged Log Probs Across Domains', fontsize=20)
        
        # Subplot 1 (Top-Left): Knowledge Probes
        ax_knowledge = axes[0, 0]
        sns.lineplot(data=final_knowledge_df, x='step', y='log_prob', hue='method', ax=ax_knowledge, marker='o')
        ax_knowledge.set_title('Knowledge Probes (All)', fontsize=16)
        ax_knowledge.set_xlabel('Training Step', fontsize=12)
        ax_knowledge.set_ylabel('Mean Log Probability', fontsize=12)
        ax_knowledge.grid(True, which="both", ls="--")
        ax_knowledge.legend(title='Method')

        # Subplots for Inference Probes
        origins = ['Explanations Only', 'Source Only', 'Other']
        plot_positions = [(0, 1), (1, 0), (1, 1)]

        for origin, pos in zip(origins, plot_positions):
            ax = axes[pos[0], pos[1]]
            origin_df = final_inference_df[final_inference_df['origin'] == origin]
            
            if not origin_df.empty:
                sns.lineplot(data=origin_df, x='step', y='log_prob', hue='method', ax=ax, marker='o')
                ax.set_title(f'Inference Probes ({origin})', fontsize=16)
                ax.legend(title='Method')
            else:
                ax.set_title(f'Inference Probes ({origin}) - No Data', fontsize=16)
                if ax.get_legend() is not None:
                    ax.get_legend().remove()

            ax.set_xlabel('Training Step', fontsize=12)
            ax.set_ylabel('') # Y-axis is shared
            ax.grid(True, which="both", ls="--")
            
    else:
        fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
        fig.suptitle('Comparison of 1B Models: Averaged Log Probs Across 6 Domains', fontsize=16)

        # Subplot 1: Knowledge Probes
        sns.lineplot(data=final_knowledge_df, x='step', y='log_prob', hue='method', ax=axes[0], marker='o')
        axes[0].set_title('Knowledge Probes')
        axes[0].set_xlabel('Training Step')
        axes[0].set_ylabel('Mean Log Probability (Averaged Across Domains)')
        axes[0].grid(True, which="both", ls="--")
        axes[0].legend(title='Method')

        # Subplot 2: Inference Probes
        sns.lineplot(data=final_inference_df, x='step', y='log_prob', hue='method', ax=axes[1], marker='o')
        axes[1].set_title('Inference Probes')
        axes[1].set_xlabel('Training Step')
        axes[1].set_ylabel('') # Y-axis is shared
        axes[1].grid(True, which="both", ls="--")
        axes[1].legend(title='Method')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_path = os.path.join(output_dir, 'knowledge_inference_log_prob_comparison.png')
    plt.savefig(output_path)
    plt.close()

    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
