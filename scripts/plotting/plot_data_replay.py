import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import re
import argparse
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from scripts.plotting.plot_comparison import aggregate_across_domains, find_latest_run_path
from utils.llm_plotting import set_plot_style

def transform_to_exposure_steps(df, strategy_name):
    """
    Transforms the 'step' column to 'Exposure Steps' based on the data replay strategy.
    """
    if df.empty:
        return df
    
    df = df.copy()
    if strategy_name == 'With No Data Replay' or 'fill' in strategy_name:
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
    Returns the minimum of the maximum steps if they are inconsistent, otherwise None.
    """
    if df.empty or 'method' not in df.columns or 'Exposure Steps' not in df.columns:
        return None

    step_maxes = df.groupby('method')['Exposure Steps'].max()
    
    if step_maxes.nunique() > 1:
        print(f"Warning for {probe_type} probes: Not all methods have the same number of exposure steps.")
        print(step_maxes)
        return int(step_maxes.min())
        
    return None

def main():
    """
    Generates a comparison plot for different data replay strategies.
    """
    # --- Configuration ---
    parser = argparse.ArgumentParser(description="Generates a comparison plot for different data replay strategies.")
    parser.add_argument(
        "--with_LIMA",
        action='store_true',
        help="If set, includes LIMA experiment results in the plot."
    )
    parser.add_argument(
        "--cut_off_at_minimal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If there are inconsistencies in the number of steps, cut off plots at the minimal number of steps."
    )
    args = parser.parse_args()
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    
    runs_to_compare = {
        '1B': {
            'With No Data Replay': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/1b/probes_v9/newline2/source_only/fill_dclm',
            'With Data Replay (1:1)': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/1b/probes_v9/newline2/source_only/sep_1_dclm',
            'With Data Replay (1:5)': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/1b/probes_v9/newline2/source_only/sep_5_dclm',
            'Data replay (1:1) via fill': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/1b/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/09_25_02_08'
        },
        '7B': {
            'With No Data Replay': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/source_only/fill_dclm',
            'With Data Replay (1:1)': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/source_only/sep_1_dclm',
            'Data replay (1:1) via fill': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e50/bs64_lr2e-05/overlap_1_4/09_24_23_35',
            'With Data Replay (1:5)': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/source_only/sep_5_dclm'
        }
    }
    method_order = ['With No Data Replay', 'With Data Replay (1:1)', 'With Data Replay (1:5)', 'Data replay (1:1) via fill']
    xlabel = 'Exposure/Training Steps' if args.with_LIMA else 'Exposure Steps'

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
    all_lima_data = {'1B': {'knowledge': [], 'inference': []}, '7B': {'knowledge': [], 'inference': []}}
    max_exposure_steps = {'1B': {'knowledge': 0, 'inference': 0}, '7B': {'knowledge': 0, 'inference': 0}}

    for model_id, runs in runs_to_compare.items():
        for run_name, base_path in runs.items():
            print(f"Processing {model_id} run: {run_name}")
            run_path = find_latest_run_path(base_path, model_id)

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
                
            if args.with_LIMA:
                # LIMA Knowledge Probes
                lima_knowledge_df = aggregate_across_domains(run_path, 'knowledge', domains, project_root=project_root, lima=True)
                if not lima_knowledge_df.empty:
                    lima_knowledge_df['method'] = run_name
                    all_lima_data[model_id]['knowledge'].append(lima_knowledge_df)

                # LIMA Inference Probes
                lima_inference_df = aggregate_across_domains(run_path, 'inference', domains, project_root=project_root, lima=True)
                if not lima_inference_df.empty:
                    lima_inference_df['method'] = run_name
                    all_lima_data[model_id]['inference'].append(lima_inference_df)

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
            
            # Separate LIMA and non-LIMA data to calculate offset
            if args.with_LIMA and all_lima_data[model_id]['knowledge']:
                final_lima_knowledge_df = pd.concat(all_lima_data[model_id]['knowledge'], ignore_index=True)
                final_lima_knowledge_df['Exposure Steps'] = final_lima_knowledge_df['step'] # LIMA doesn't have exposure steps

                min_lima_knowledge_step = check_exposure_step_consistency(final_lima_knowledge_df, f"{model_id} LIMA Knowledge")
                if args.cut_off_at_minimal and min_lima_knowledge_step is not None:
                    final_lima_knowledge_df = final_lima_knowledge_df[final_lima_knowledge_df['Exposure Steps'] <= min_lima_knowledge_step]

                if not final_knowledge_df.empty:
                    max_exp_k = final_knowledge_df['Exposure Steps'].max()
                    max_exposure_steps[model_id]['knowledge'] = max_exp_k
                    if not final_lima_knowledge_df.empty:
                        final_lima_knowledge_df['Exposure Steps'] += max_exp_k
                        final_knowledge_df = pd.concat([final_knowledge_df, final_lima_knowledge_df], ignore_index=True)
            
            check_exposure_step_consistency(final_knowledge_df, f"{model_id} Knowledge")
            for method in method_order:
                if method not in final_knowledge_df['method'].unique():
                    continue
                method_df = final_knowledge_df[final_knowledge_df['method'] == method]
                plot_df = method_df.groupby('Exposure Steps')['log_prob'].mean().reset_index()
                ax_knowledge.plot(plot_df['Exposure Steps'], plot_df['log_prob'], label=method, lw=1.6)
            ax_knowledge.set_title(f'{model_id}: Factual Probes')
            if not final_knowledge_df.empty:
                max_x = final_knowledge_df['Exposure Steps'].max()
                ax_knowledge.set_xticks(np.arange(0, max_x + 1, 30))
        else:
            ax_knowledge.set_title(f'{model_id}: Factual Probes (No Data)')
        ax_knowledge.set_xlabel(xlabel)
        if i == 0:
            ax_knowledge.set_ylabel('Mean Log Probability')

        # --- Inference Plot ---
        ax_inference = axes[i*2 + 1]
        if all_data[model_id]['inference']:
            final_inference_df = pd.concat(all_data[model_id]['inference'], ignore_index=True)

            if args.with_LIMA and all_lima_data[model_id]['inference']:
                final_lima_inference_df = pd.concat(all_lima_data[model_id]['inference'], ignore_index=True)
                final_lima_inference_df['Exposure Steps'] = final_lima_inference_df['step']

                min_lima_inference_step = check_exposure_step_consistency(final_lima_inference_df, f"{model_id} LIMA Inference")
                if args.cut_off_at_minimal and min_lima_inference_step is not None:
                    final_lima_inference_df = final_lima_inference_df[final_lima_inference_df['Exposure Steps'] <= min_lima_inference_step]

                if not final_inference_df.empty:
                    max_exp_i = final_inference_df['Exposure Steps'].max()
                    max_exposure_steps[model_id]['inference'] = max_exp_i
                    if not final_lima_inference_df.empty:
                        final_lima_inference_df['Exposure Steps'] += max_exp_i
                        final_inference_df = pd.concat([final_inference_df, final_lima_inference_df], ignore_index=True)

            check_exposure_step_consistency(final_inference_df, f"{model_id} Inference")
            for method in method_order:
                if method not in final_inference_df['method'].unique():
                    continue
                method_df = final_inference_df[final_inference_df['method'] == method]
                plot_df = method_df.groupby('Exposure Steps')['log_prob'].mean().reset_index()
                ax_inference.plot(plot_df['Exposure Steps'], plot_df['log_prob'], label=method, lw=1.6)
            ax_inference.set_title(f'{model_id}: Compositional Probes')
            if not final_inference_df.empty:
                max_x = final_inference_df['Exposure Steps'].max()
                ax_inference.set_xticks(np.arange(0, max_x + 1, 30))
        else:
            ax_inference.set_title(f'{model_id}: Compositional Probes (No Data)')
        ax_inference.set_xlabel(xlabel)
        ax_inference.set_ylabel('')

        # Add vertical lines for this model's plots if LIMA is enabled
        if args.with_LIMA:
            max_step_k = max_exposure_steps[model_id]['knowledge']
            max_step_i = max_exposure_steps[model_id]['inference']
            if max_step_k > 0:
                ax_knowledge.axvline(x=max_step_k, color='red', linestyle='--', linewidth=1.5, alpha=0.9)
            if max_step_i > 0:
                ax_inference.axvline(x=max_step_i, color='red', linestyle='--', linewidth=1.5, alpha=0.9)

    axes[3].legend(loc='upper right', fontsize='small', title_fontsize='small')

    for ax in fig.get_axes():
        ax.grid(True)

    fig.subplots_adjust(wspace=0.2 if args.with_LIMA else 0.25, top=0.925)
    
    output_filename = 'data_replay_1B_7B_comparison_exposure_steps.pdf'
    if args.with_LIMA:
        output_filename = output_filename.replace('.pdf', '_with_LIMA.pdf')
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
