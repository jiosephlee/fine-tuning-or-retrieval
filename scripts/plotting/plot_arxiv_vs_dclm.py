import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from scripts.FT.plot_comparison import aggregate_across_domains, check_step_consistency, find_latest_run_path
from utils.llm_plotting import set_plot_style

def main():
    """
    Generates a side-by-side comparison plot for two specific experimental runs across 1B and 7B models.
    """
    # --- Configuration ---
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    
    runs_to_compare = {
        '1B': {
            'ArXiv': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/1b/probes_v9/newline2/para9/sep_1_arxiv/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs32_lr2e-05/run',
            'DCLM': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/1b/probes_v9/newline2/para9/sep_1_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs32_lr2e-05/run'
        },
        '7B': {
            'ArXiv': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/source_only/sep_1_arxiv',
            'DCLM': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/source_only/sep_1_dclm'
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
            
            run_path = base_path if base_path.endswith('/run') else find_latest_run_path(base_path)

            if not run_path or not os.path.isdir(run_path):
                print(f"  - Warning: Run path not found or invalid for {model_id} {run_name} at {base_path}")
                continue

            # Knowledge Probes
            knowledge_df = aggregate_across_domains(run_path, 'knowledge', domains, project_root=project_root)
            if not knowledge_df.empty:
                knowledge_df['method'] = run_name
                all_data[model_id]['knowledge'].append(knowledge_df)
            
            # Inference Probes
            inference_df = aggregate_across_domains(run_path, 'inference', domains, project_root=project_root)
            if not inference_df.empty:
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
            knowledge_df = pd.concat(all_data[model_id]['knowledge'], ignore_index=True)
            check_step_consistency(knowledge_df, f"{model_id} Knowledge")
            for method in sorted(knowledge_df['method'].unique()):
                method_df = knowledge_df[knowledge_df['method'] == method]
                plot_df = method_df.groupby('step')['log_prob'].mean().reset_index()
                ax_knowledge.plot(plot_df['step'], plot_df['log_prob'], label=method, lw=2.5)
            ax_knowledge.set_title(f'{model_id}: Factual Probes')
        else:
            ax_knowledge.set_title(f'{model_id}: Factual Probes (No Data)')
        
        ax_knowledge.set_xlabel('Training Step')
        if i == 0:
            ax_knowledge.set_ylabel('Mean Log Probability')

        # --- Inference Plot ---
        ax_inference = axes[i*2 + 1]
        if all_data[model_id]['inference']:
            inference_df = pd.concat(all_data[model_id]['inference'], ignore_index=True)
            check_step_consistency(inference_df, f"{model_id} Inference")
            for method in sorted(inference_df['method'].unique()):
                method_df = inference_df[inference_df['method'] == method]
                plot_df = method_df.groupby('step')['log_prob'].mean().reset_index()
                ax_inference.plot(plot_df['step'], plot_df['log_prob'], label=method, lw=2.5)
            ax_inference.set_title(f'{model_id}: Compositional Probes')
        else:
            ax_inference.set_title(f'{model_id}: Compositional Probes (No Data)')

        ax_inference.set_xlabel('Training Step')
        ax_inference.set_ylabel('')

    axes[3].legend(title='Data Replay\nSource', loc='upper right', fontsize='small', title_fontsize='small')

    for ax in fig.get_axes():
        ax.grid(True)

    fig.subplots_adjust(wspace=0.25, top=0.9)
    
    output_path = os.path.join(output_dir, 'arxiv_vs_dclm_1B_7B_comparison.pdf')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
