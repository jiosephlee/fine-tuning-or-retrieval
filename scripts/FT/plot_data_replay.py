import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from scripts.FT.plot_comparison import aggregate_across_domains, find_latest_run_path
from utils.llm_plotting import set_plot_style

def transform_to_exposure_steps(df, strategy_name):
    """
    Transforms the 'step' column to 'Exposure Steps' based on the data replay strategy.
    """
    if df.empty:
        return df
    
    df = df.copy()
    if 'With No Data Replay' in strategy_name:
        # Each step is an exposure
        df['Exposure Steps'] = df['step']
    elif 'With Data Replay (1:1)' in strategy_name:
        # Exposure at step 0, then every 2 steps starting from 1 (0, 1, 3, 5...)
        df = df[(df['step'] == 0) | ((df['step'] > 0) & ((df['step'] - 1) % 2 == 0))].copy()
        df['Exposure Steps'] = (df['step'] + 1) // 2
    elif 'With Data Replay (1:5)' in strategy_name:
        # Exposure at step 0, then every 6 steps starting from 1 (0, 1, 7, 13...)
        df = df[(df['step'] == 0) | ((df['step'] > 0) & ((df['step'] - 1) % 6 == 0))].copy()
        df['Exposure Steps'] = (df['step'] - 1) // 6 + 1
    else:
        df['Exposure Steps'] = df['step']
        
    return df

def check_exposure_step_consistency(df: pd.DataFrame, probe_type: str):
    """
    Checks if all methods have the same number of exposure steps and warns if they do not.
    """
    if df.empty or 'method' not in df.columns or 'Exposure Steps' not in df.columns:
        return

    step_counts = df.groupby('method')['Exposure Steps'].nunique()
    
    if step_counts.nunique() > 1:
        print(f"Warning for {probe_type} probes: Not all methods have the same number of exposure steps.")
        print(step_counts)

def main():
    """
    Generates a comparison plot for different data replay strategies.
    """
    # --- Configuration ---
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    
    runs_to_compare = {
        '1B': {
            'With No Data Replay': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/1b/probes_v9/newline2/source_only/fill_dclm',
            'With Data Replay (1:1)': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/1b/probes_v9/newline2/source_only/sep_1_dclm',
            'With Data Replay (1:5)': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/1b/probes_v9/newline2/source_only/sep_5_dclm'
        },
        '7B': {
            'With No Data Replay': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/source_only/fill_dclm',
            'With Data Replay (1:1)': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/source_only/sep_1_dclm',
            'With Data Replay (1:5)': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/source_only/sep_5_dclm'
        }
    }

    domains_path = os.path.join(project_root, 'data/arxiv/cleaned')
    try:
        domains = sorted([os.path.splitext(f)[0] for f in os.listdir(domains_path) if f.endswith('.tex') and os.path.isfile(os.path.join(domains_path, f))])
    except FileNotFoundError:
        print(f"Error: Domains path '{domains_path}' not found.")
        return
        
    output_dir = 'plots'
    os.makedirs(output_dir, exist_ok=True)

    # --- Data Aggregation ---
    all_data = {'1B': {'knowledge': [], 'inference': []}, '7B': {'knowledge': [], 'inference': []}}

    for model_id, runs in runs_to_compare.items():
        for run_name, base_path in runs.items():
            print(f"Processing {model_id} run: {run_name}")
            run_path = find_latest_run_path(base_path)

            if not run_path or not os.path.isdir(run_path):
                print(f"  - Warning: Run path not found for {run_name} at {base_path}")
                continue

            # Knowledge Probes
            knowledge_df = aggregate_across_domains(run_path, 'knowledge', domains, project_root=project_root)
            if not knowledge_df.empty:
                knowledge_df = transform_to_exposure_steps(knowledge_df, run_name)
                knowledge_df['method'] = run_name
                all_data[model_id]['knowledge'].append(knowledge_df)
            
            # Inference Probes
            inference_df = aggregate_across_domains(run_path, 'inference', domains, project_root=project_root)
            if not inference_df.empty:
                inference_df = transform_to_exposure_steps(inference_df, run_name)
                inference_df['method'] = run_name
                all_data[model_id]['inference'].append(inference_df)

    # --- Plotting ---
    print("Generating comparison plot...")

    set_plot_style()

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Manually share y-axes for 1B and 7B plots separately
    axes[1].sharey(axes[0])
    axes[3].sharey(axes[2])
    # Hide y-tick labels on shared axes
    plt.setp(axes[1].get_yticklabels(), visible=False)
    plt.setp(axes[3].get_yticklabels(), visible=False)
    
    for i, model_id in enumerate(['1B', '7B']):
        # --- Knowledge Plot ---
        ax_knowledge = axes[i*2]
        if all_data[model_id]['knowledge']:
            final_knowledge_df = pd.concat(all_data[model_id]['knowledge'], ignore_index=True)
            check_exposure_step_consistency(final_knowledge_df, f"{model_id} Knowledge")
            for method in sorted(final_knowledge_df['method'].unique()):
                method_df = final_knowledge_df[final_knowledge_df['method'] == method]
                plot_df = method_df.groupby('Exposure Steps')['log_prob'].mean().reset_index()
                ax_knowledge.plot(plot_df['Exposure Steps'], plot_df['log_prob'], label=method)
            ax_knowledge.set_title(f'{model_id}: Factual Probes')
        else:
            ax_knowledge.set_title(f'{model_id}: Factual Probes (No Data)')
        ax_knowledge.set_xlabel('Exposure Steps')
        if i == 0:
            ax_knowledge.set_ylabel('Mean Log Probability')

        # --- Inference Plot ---
        ax_inference = axes[i*2 + 1]
        if all_data[model_id]['inference']:
            final_inference_df = pd.concat(all_data[model_id]['inference'], ignore_index=True)
            check_exposure_step_consistency(final_inference_df, f"{model_id} Inference")
            for method in sorted(final_inference_df['method'].unique()):
                method_df = final_inference_df[final_inference_df['method'] == method]
                plot_df = method_df.groupby('Exposure Steps')['log_prob'].mean().reset_index()
                ax_inference.plot(plot_df['Exposure Steps'], plot_df['log_prob'], label=method)
            ax_inference.set_title(f'{model_id}: Compositional Probes')
        else:
            ax_inference.set_title(f'{model_id}: Compositional Probes (No Data)')
        ax_inference.set_xlabel('Exposure Steps')
        ax_inference.set_ylabel('')

    axes[3].legend(loc='upper right', fontsize='small', title_fontsize='small')

    for ax in fig.get_axes():
        ax.grid(True)

    fig.subplots_adjust(wspace=0.25, top=0.925)
    
    output_path = os.path.join(output_dir, 'data_replay_1B_7B_comparison_exposure_steps.pdf')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
