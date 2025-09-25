import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from scripts.plotting.plot_comparison import aggregate_across_domains, check_step_consistency, find_latest_run_path
from utils.llm_plotting import set_plot_style

def main():
    """
    Generates a comparison plot for different overlap ratios across 1B and 7B models,
    separating 'source_only' and 'para9' fine-tuning data.
    """
    # --- Configuration ---
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    
    runs_to_compare = {
        '1B': {
            'Source': {
                'base_path': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/1b/probes_v9/newline2/source_only/sep_1_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e50/bs32_lr2e-05/',
                'overlaps': ['no_overlap', 'overlap_1_10', 'overlap_2_10']
            },
            'Para': {
                'base_path': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/1b/probes_v9/newline2/para9/sep_1_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e50/bs32_lr2e-05/',
                'overlaps': ['no_overlap', 'overlap_1_10', 'overlap_2_10']
            }
        },
        '7B': {
            'Source': {
                'base_path': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/source_only/sep_1_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e50/bs32_lr2e-05/',
                'overlaps': ['no_overlap', 'overlap_1_10', 'overlap_2_10']
            },
            'Para': {
                'base_path': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/para9/sep_1_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e50/bs32_lr2e-05/',
                'overlaps': ['no_overlap', 'overlap_1_10', 'overlap_2_10']
            }
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
    all_data = {
        '1B': {'Source': {'knowledge': [], 'inference': []}, 'Para': {'knowledge': [], 'inference': []}},
        '7B': {'Source': {'knowledge': [], 'inference': []}, 'Para': {'knowledge': [], 'inference': []}}
    }

    for model_id, data_types in runs_to_compare.items():
        for data_type, config in data_types.items():
            for overlap in config['overlaps']:
                print(f"Processing {model_id} {data_type} with overlap {overlap}")
                
                base_path = os.path.join(config['base_path'], overlap)
                run_path = find_latest_run_path(base_path, model_id)

                if not run_path or not os.path.isdir(run_path):
                    print(f"  - Warning: Run path not found or invalid for {model_id} {data_type} {overlap} at {base_path}")
                    continue

                # Knowledge Probes
                knowledge_df = aggregate_across_domains(run_path, 'knowledge', domains, project_root=project_root)
                if not knowledge_df.empty:
                    knowledge_df['overlap'] = overlap
                    all_data[model_id][data_type]['knowledge'].append(knowledge_df)
                
                # Inference Probes
                inference_df = aggregate_across_domains(run_path, 'inference', domains, project_root=project_root)
                if not inference_df.empty:
                    inference_df['overlap'] = overlap
                    all_data[model_id][data_type]['inference'].append(inference_df)

    # --- Plotting ---
    print("Generating comparison plot...")

    set_plot_style()
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5), sharey=False)
    
    # Manually share y-axes for 1B and 7B plots separately
    axes[1].sharey(axes[0])
    axes[3].sharey(axes[2])
    plt.setp(axes[1].get_yticklabels(), visible=False)
    plt.setp(axes[3].get_yticklabels(), visible=False)

    plot_configs = [('1B', 'Source'), ('1B', 'Para'), ('7B', 'Source'), ('7B', 'Para')]
    overlap_labels = {
        'no_overlap': 'No Overlap',
        'overlap_1_10': '1/10 Overlap',
        'overlap_2_10': '2/10 Overlap'
    }

    for i, (model_id, data_type) in enumerate(plot_configs):
        ax = axes[i]
        
        # Define colors and legend order
        colors = plt.cm.Blues(np.linspace(0.4, 1, 3)) if data_type == 'Source' else plt.cm.Oranges(np.linspace(0.4, 1, 3))
        overlap_order = ['no_overlap', 'overlap_1_10', 'overlap_2_10']
        color_map = dict(zip(overlap_order, colors))
        legend_order = sorted(overlap_order, reverse=True)

        # --- Factual (Knowledge) Probes ---
        if all_data[model_id][data_type]['knowledge']:
            knowledge_df = pd.concat(all_data[model_id][data_type]['knowledge'], ignore_index=True)
            check_step_consistency(knowledge_df, f"{model_id} {data_type} Knowledge")
            
            for overlap in legend_order:
                if overlap not in knowledge_df['overlap'].unique(): continue
                method_df = knowledge_df[knowledge_df['overlap'] == overlap]
                plot_df = method_df.groupby('step')['log_prob'].mean().reset_index()
                ax.plot(plot_df['step'], plot_df['log_prob'], linestyle='-', color=color_map[overlap], label=overlap_labels.get(overlap, overlap), lw=1.75)
        
        # --- Compositional (Inference) Probes ---
        if all_data[model_id][data_type]['inference']:
            inference_df = pd.concat(all_data[model_id][data_type]['inference'], ignore_index=True)
            check_step_consistency(inference_df, f"{model_id} {data_type} Inference")
            
            for overlap in legend_order:
                if overlap not in inference_df['overlap'].unique(): continue
                method_df = inference_df[inference_df['overlap'] == overlap]
                plot_df = method_df.groupby('step')['log_prob'].mean().reset_index()
                ax.plot(plot_df['step'], plot_df['log_prob'], linestyle='--', color=color_map[overlap], lw=1.75)
        
        ax.set_title(f'{model_id}: {data_type}')
        ax.set_xlabel('Training Step')
        if i == 0:
            ax.set_ylabel('Mean Log Probability')
        
        ax.legend(title='Overlap Ratio', loc='lower right')
        ax.grid(True)

    # Custom legend for line styles
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='black', lw=1.75, linestyle='-', label='Factual'),
                       Line2D([0], [0], color='black', lw=1.75, linestyle='--', label='Compositional')]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0), ncol=2, title="Probe Type")

    fig.tight_layout(rect=[0, 0.05, 1, 1]) # Adjust layout to make space for custom legend
    
    output_path = os.path.join(output_dir, 'overlap_ratios_comparison.pdf')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
