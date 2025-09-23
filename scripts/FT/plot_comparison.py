import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from datetime import datetime
import json
import argparse

# Adjust the path to include the utils directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils.llm_plotting import set_plot_style


def find_latest_run_path(base_path):
    """
    Finds the path to the latest run data within a given base experiment directory,
    following a specific priority for subdirectories.
    """
    try:
        # Find domains_* directory (assuming one)
        domain_dirs = [d for d in os.listdir(base_path) if d.startswith('domains_') and os.path.isdir(os.path.join(base_path, d))]
        if not domain_dirs:
            return None
        # Let's assume we take the first one if multiple exist, or sort by modification time
        domain_dir = domain_dirs[0]
        
        path = os.path.join(base_path, domain_dir, 'e100')

        # Check for learning rate / batch size folder
        sub_dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        if not sub_dirs:
            return None
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
    
    return combined_df

def aggregate_lima_data(model_id, probe_type, domains, split_probes=False, project_root='.'):
    """
    Aggregates LIMA probe data across multiple domains.
    """
    all_lima_dfs = []
    # This path is based on the user's example. It might need to be adjusted if the structure varies.
    lima_run_path = os.path.join(project_root, f'results/prior_knowledge/full/{model_id.lower()}/probes_v9/domains_all/e50/run')

    if not os.path.exists(lima_run_path):
        print(f"Warning: LIMA run path not found at {lima_run_path}")
        return pd.DataFrame()

    for domain in domains:
        if probe_type == "knowledge":
            # Assuming a naming convention for LIMA probes, e.g., 'domain_lima_knowledge_probe'
            probe_dir = f"{domain}_lima_knowledge_probe"
            file_name = f"{domain}_lima_knowledge_probe_metrics.csv"
        else: # inference
            probe_dir = f"{domain}_lima_inference_probe"
            file_name = f"{domain}_lima_inference_probe_metrics.csv"
            
        metrics_path = os.path.join(lima_run_path, probe_dir, file_name)
        
        if os.path.exists(metrics_path) and os.path.getsize(metrics_path) > 0:
            df = pd.read_csv(metrics_path)
            df['domain'] = domain
            
            # Note: LIMA data might not have the same 'origin' split as FT data.
            # This implementation assumes no split for LIMA data for simplicity.
            if probe_type == 'inference' and split_probes:
                 df['origin'] = 'Other' # Default origin

            all_lima_dfs.append(df)
        # Quietly skip if files don't exist, as not all domains might have LIMA probes.

    if not all_lima_dfs:
        return pd.DataFrame()

    combined_df = pd.concat(all_lima_dfs, ignore_index=True)
    return combined_df

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
    parser.add_argument(
        "--with_LIMA",
        action='store_true',
        help="If set, includes LIMA experiment results in the plot."
    )
    args = parser.parse_args()

    # --- Configuration ---
    show_std_dev_shadows = False # New internal parameter
    errorbar_setting = 'sd' if show_std_dev_shadows else None
    
    split_probes = args.split_probes
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    experiment_base_path = f'results/FT/full/{args.model_id}/probes_v9/newline2'
    methods = [
        ('para9_expl', 'Para. + Multiview', 'sep_1_dclm'),
        ('para9', 'Para.', 'sep_1_dclm'),
        ('source_only', 'Source', 'sep_1_dclm'),
    ]
    method_names_in_order = [m[1] for m in methods]
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

    final_knowledge_df = pd.concat(all_knowledge_data, ignore_index=True) if all_knowledge_data else pd.DataFrame()
    final_inference_df = pd.concat(all_inference_data, ignore_index=True) if all_inference_data else pd.DataFrame()
    
    # Check for step consistency
    check_step_consistency(final_knowledge_df, "Knowledge")
    check_step_consistency(final_inference_df, "Inference")

    # --- LIMA Data Appending ---
    if args.with_LIMA:
        print("Aggregating and appending LIMA data...")
        method_names_in_order = [m[1] for m in methods]

        # --- LIMA Knowledge Data ---
        if not final_knowledge_df.empty:
            max_knowledge_step = final_knowledge_df['step'].max()
            lima_knowledge_df = aggregate_lima_data(args.model_id, 'knowledge', domains, project_root=project_root)
            if not lima_knowledge_df.empty:
                lima_knowledge_df['step'] += max_knowledge_step
                appended_k_dfs = [final_knowledge_df]
                for method_name in method_names_in_order:
                    method_lima_df = lima_knowledge_df.copy()
                    method_lima_df['method'] = method_name
                    appended_k_dfs.append(method_lima_df)
                final_knowledge_df = pd.concat(appended_k_dfs, ignore_index=True)

        # --- LIMA Inference Data ---
        if not final_inference_df.empty:
            max_inference_step = final_inference_df['step'].max()
            lima_inference_df = aggregate_lima_data(args.model_id, 'inference', domains, split_probes=split_probes, project_root=project_root)
            if not lima_inference_df.empty:
                lima_inference_df['step'] += max_inference_step
                appended_i_dfs = [final_inference_df]
                for method_name in method_names_in_order:
                    method_lima_df = lima_inference_df.copy()
                    method_lima_df['method'] = method_name
                    appended_i_dfs.append(method_lima_df)
                final_inference_df = pd.concat(appended_i_dfs, ignore_index=True)

    # --- Plotting ---
    print("Generating comparison plot...")

    # Set academic plot style
    set_plot_style()

    if split_probes:
        # --- PLOTTING FOR SPLIT PROBES (SINGLE MODEL) ---
        model_name = args.model_id.upper()
        title = f'Comparison of {model_name} Models: Averaged Log Probs'
        output_filename = f'knowledge_inference_log_prob_{args.model_id}_split_probes.pdf'
        output_path = os.path.join(output_dir, output_filename)

        final_knowledge_df = pd.concat(all_knowledge_data, ignore_index=True) if all_knowledge_data else pd.DataFrame()
        final_inference_df = pd.concat(all_inference_data, ignore_index=True) if all_inference_data else pd.DataFrame()

        fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
        
        # Subplot 1: Knowledge Probes
        ax_knowledge = axes[0]
        for method in method_names_in_order:
            method_df = final_knowledge_df[final_knowledge_df['method'] == method]
            if not method_df.empty:
                plot_df = method_df.groupby('step')['log_prob'].mean().reset_index()
                ax_knowledge.plot(plot_df['step'], plot_df['log_prob'], label=method)
        ax_knowledge.set_title(f'{model_name}: Factual Probes')
        ax_knowledge.set_xlabel('Training Step')
        ax_knowledge.set_ylabel('Mean Log Probability')
        ax_knowledge.legend(loc='lower right', fontsize='small', title_fontsize='small')

        # Subplots for Inference Probes
        origins = ['Explanations Only', 'Source Only', 'Other']
        for i, origin in enumerate(origins):
            ax = axes[i + 1]
            origin_df = final_inference_df[final_inference_df['origin'] == origin]
            
            if not origin_df.empty:
                for method in method_names_in_order:
                    method_df = origin_df[origin_df['method'] == method]
                    if not method_df.empty:
                        plot_df = method_df.groupby('step')['log_prob'].mean().reset_index()
                        ax.plot(plot_df['step'], plot_df['log_prob'], label=method)
                ax.set_title(f'{model_name}: Compositional ({origin})')
            else:
                ax.set_title(f'{model_name}: Compositional ({origin}) - No Data')

            ax.set_xlabel('Training Step')
            ax.set_ylabel('')
        
    else:
        # --- PLOTTING FOR UNIFIED VIEW (1B vs 7B) ---
        output_filename = 'knowledge_inference_log_prob_1B_7B_unified.pdf'
        output_path = os.path.join(output_dir, output_filename)

        all_data = {'1B': {'knowledge': [], 'inference': []}, '7B': {'knowledge': [], 'inference': []}}
        models_to_plot = ['1B', '7B']
        for model_id_str in models_to_plot:
            experiment_base_path = f'results/FT/full/{model_id_str.lower()}/probes_v9/newline2'
            for method_key, method_name, sub_dir in methods:
                method_base_path = os.path.join(experiment_base_path, method_key, sub_dir)
                run_path = find_latest_run_path(method_base_path)
                if run_path:
                    knowledge_df = aggregate_across_domains(run_path, 'knowledge', domains, project_root=project_root)
                    if not knowledge_df.empty:
                        knowledge_df['method'] = method_name
                        all_data[model_id_str]['knowledge'].append(knowledge_df)
                    
                    inference_df = aggregate_across_domains(run_path, 'inference', domains, project_root=project_root)
                    if not inference_df.empty:
                        inference_df['method'] = method_name
                        all_data[model_id_str]['inference'].append(inference_df)

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        # Manually share y-axes
        axes[1].sharey(axes[0])
        axes[3].sharey(axes[2])
        plt.setp(axes[1].get_yticklabels(), visible=False)
        plt.setp(axes[3].get_yticklabels(), visible=False)

        for i, model_id in enumerate(models_to_plot):
            # Knowledge Plot
            ax_knowledge = axes[i*2]
            if all_data[model_id]['knowledge']:
                df = pd.concat(all_data[model_id]['knowledge'], ignore_index=True)
                for method in method_names_in_order:
                    method_df = df[df['method'] == method]
                    if not method_df.empty:
                        plot_df = method_df.groupby('step')['log_prob'].mean().reset_index()
                        ax_knowledge.plot(plot_df['step'], plot_df['log_prob'], label=method)
            ax_knowledge.set_title(f'{model_id}: Factual Probes')
            ax_knowledge.set_xlabel('Training Step')
            if i == 0:
                ax_knowledge.set_ylabel('Mean Log Probability')

            # Compositional Plot
            ax_compositional = axes[i*2 + 1]
            if all_data[model_id]['inference']:
                df = pd.concat(all_data[model_id]['inference'], ignore_index=True)
                for method in method_names_in_order:
                    method_df = df[df['method'] == method]
                    if not method_df.empty:
                        plot_df = method_df.groupby('step')['log_prob'].mean().reset_index()
                        ax_compositional.plot(plot_df['step'], plot_df['log_prob'], label=method)
            ax_compositional.set_title(f'{model_id}: Compositional Probes')
            ax_compositional.set_xlabel('Training Step')
        
        axes[0].legend(loc='lower right', fontsize='small', title_fontsize='small')

    # --- FINAL STYLING ---
    if 'final_knowledge_df' in locals() and not final_knowledge_df.empty:
        max_step = final_knowledge_df['step'].max()
        vline_steps = np.arange(30, max_step + 1, 40)
        for ax in fig.get_axes():
            for step in vline_steps:
                ax.axvline(x=step, color='grey', linestyle='--', linewidth=1, alpha=0.7)

    for ax in fig.get_axes():
        ax.grid(False)

    fig.subplots_adjust(wspace=0.25, top=0.9)
    
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
