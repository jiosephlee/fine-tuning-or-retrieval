import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from scripts.plotting.plot_comparison import aggregate_across_domains, find_latest_run_path
from utils.llm_plotting import set_plot_style


def check_step_consistency(df: pd.DataFrame, probe_type: str):
    """
    Checks if all methods have the same number of training steps and warns if they do not.
    Returns the minimum of the maximum steps if they are inconsistent, otherwise None.
    """
    if df.empty or 'method' not in df.columns or 'step' not in df.columns:
        return None

    step_maxes = df.groupby('method')['step'].max()
    
    if step_maxes.nunique() > 1:
        print(f"Warning for {probe_type} probes: Not all methods have the same number of training steps.")
        print(step_maxes)
        return int(step_maxes.min())

    return None


def get_model_data(model_id, prior_knowledge_path, ft_methods, domains, project_root, with_LIMA=False, cut_off_at_minimal=True):
    """
    Aggregates data for a specific model, combining a prior knowledge stage with fine-tuning stages.
    """
    all_methods_knowledge_data = []
    all_methods_inference_data = []
    all_methods_lima_knowledge_data = []
    all_methods_lima_inference_data = []

    # 1. Load Prior Knowledge Data
    print(f"Processing Prior Knowledge for model: {model_id}")
    prior_knowledge_df_k = aggregate_across_domains(prior_knowledge_path, 'knowledge', domains, project_root=project_root)
    prior_knowledge_df_i = aggregate_across_domains(prior_knowledge_path, 'inference', domains, project_root=project_root)
    
    max_step_prior_k = 0
    if not prior_knowledge_df_k.empty:
        max_step_prior_k = prior_knowledge_df_k['step'].max()

    max_step_prior_i = 0
    if not prior_knowledge_df_i.empty:
        max_step_prior_i = prior_knowledge_df_i['step'].max()

    # 2. Load Fine-tuning Data for each method
    for method_key, method_name, base_path_template in ft_methods:
        print(f"Processing method: {method_key} for model: {model_id}")
        
        # Handle cases where a method is not applicable for a model
        if base_path_template is None:
            print(f"  - Skipping {method_key} for {model_id} as it is not configured.")
            continue
        
        if model_id == '1B':
            run_path = base_path_template.format(method_key=method_key)
        else: # 7B
            run_path = base_path_template

        if run_path and os.path.isdir(run_path):
            # Knowledge Probes
            knowledge_df = aggregate_across_domains(run_path, 'knowledge', domains, project_root=project_root)
            if not knowledge_df.empty:
                knowledge_df['step'] += max_step_prior_k
                knowledge_df['method'] = method_name
                all_methods_knowledge_data.append(knowledge_df)
            
            # Inference Probes
            inference_df = aggregate_across_domains(run_path, 'inference', domains, project_root=project_root)
            if not inference_df.empty:
                inference_df['step'] += max_step_prior_i
                inference_df['method'] = method_name
                all_methods_inference_data.append(inference_df)

            if with_LIMA:
                lima_knowledge_df = aggregate_across_domains(run_path, 'knowledge', domains, project_root=project_root, lima=True)
                if not lima_knowledge_df.empty:
                    lima_knowledge_df['method'] = method_name
                    all_methods_lima_knowledge_data.append(lima_knowledge_df)

                lima_inference_df = aggregate_across_domains(run_path, 'inference', domains, project_root=project_root, lima=True)
                if not lima_inference_df.empty:
                    lima_inference_df['method'] = method_name
                    all_methods_lima_inference_data.append(lima_inference_df)
        else:
            print(f"  - Warning: Run path not found or invalid for {model_id} {method_key} at {run_path}")

    final_knowledge_df = pd.concat(all_methods_knowledge_data, ignore_index=True) if all_methods_knowledge_data else pd.DataFrame()
    final_inference_df = pd.concat(all_methods_inference_data, ignore_index=True) if all_methods_inference_data else pd.DataFrame()

    max_ft_step_k = final_knowledge_df['step'].max() if not final_knowledge_df.empty else 0
    max_ft_step_i = final_inference_df['step'].max() if not final_inference_df.empty else 0

    if with_LIMA:
        if all_methods_lima_knowledge_data:
            final_lima_knowledge_df = pd.concat(all_methods_lima_knowledge_data, ignore_index=True)
            min_lima_knowledge_step = check_step_consistency(final_lima_knowledge_df, "LIMA Knowledge")
            if cut_off_at_minimal and min_lima_knowledge_step is not None:
                final_lima_knowledge_df = final_lima_knowledge_df[final_lima_knowledge_df['step'] <= min_lima_knowledge_step]
            
            if not final_knowledge_df.empty and not final_lima_knowledge_df.empty:
                final_lima_knowledge_df['step'] += max_ft_step_k
                final_knowledge_df = pd.concat([final_knowledge_df, final_lima_knowledge_df], ignore_index=True)

        if all_methods_lima_inference_data:
            final_lima_inference_df = pd.concat(all_methods_lima_inference_data, ignore_index=True)
            min_lima_inference_step = check_step_consistency(final_lima_inference_df, "LIMA Inference")
            if cut_off_at_minimal and min_lima_inference_step is not None:
                final_lima_inference_df = final_lima_inference_df[final_lima_inference_df['step'] <= min_lima_inference_step]

            if not final_inference_df.empty and not final_lima_inference_df.empty:
                final_lima_inference_df['step'] += max_ft_step_i
                final_inference_df = pd.concat([final_inference_df, final_lima_inference_df], ignore_index=True)

    return prior_knowledge_df_k, prior_knowledge_df_i, final_knowledge_df, final_inference_df, max_ft_step_k, max_ft_step_i


def get_ft_only_data(model_id, ft_methods, domains, project_root):
    """
    Aggregates data for fine-tuning only models.
    """
    all_methods_knowledge_data = []
    all_methods_inference_data = []

    for method_key, method_name, base_path_template in ft_methods:
        print(f"Processing FT-only method: {method_key} for model: {model_id}")

        if base_path_template is None:
            print(f"  - Skipping {method_key} for {model_id} as it is not configured.")
            continue

        base_path = base_path_template.format(method_key=method_key)
        
        # We need to find the latest run for these
        run_path = find_latest_run_path(base_path)

        if run_path and os.path.isdir(run_path):
            # Knowledge Probes
            knowledge_df = aggregate_across_domains(run_path, 'knowledge', domains, project_root=project_root)
            if not knowledge_df.empty:
                knowledge_df['method'] = method_name
                all_methods_knowledge_data.append(knowledge_df)
            
            # Inference Probes
            inference_df = aggregate_across_domains(run_path, 'inference', domains, project_root=project_root)
            if not inference_df.empty:
                inference_df['method'] = method_name
                all_methods_inference_data.append(inference_df)
        else:
            print(f"  - Warning: Run path not found or invalid for FT-only {model_id} {method_key} at {base_path}")

    final_knowledge_df = pd.concat(all_methods_knowledge_data, ignore_index=True) if all_methods_knowledge_data else pd.DataFrame()
    final_inference_df = pd.concat(all_methods_inference_data, ignore_index=True) if all_methods_inference_data else pd.DataFrame()

    return final_knowledge_df, final_inference_df


def main():
    """
    Main function to generate the comparison plot for prior knowledge experiments.
    """
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Generate comparison plots for prior knowledge experiments.")
    parser.add_argument(
        "--with_LIMA",
        action='store_true',
        help="If set, includes LIMA experiment results in the plot and excludes FT-only comparison."
    )
    parser.add_argument(
        "--cut_off_at_minimal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If there are inconsistencies in the number of steps, cut off plots at the minimal number of steps."
    )
    args = parser.parse_args()

    # --- Configuration ---
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    
    prior_knowledge_paths = {
        '1B': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/prior_knowledge/full/1b/probes_v9/domains_all/e50/run',
        '7B': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/prior_knowledge/full/7b/probes_v9/fill_dclm/domains_all/e50/bs256_lr4e-05/09_20_20_34'
    }

    ft_base_paths = {
        '1B': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/jiosephlee_6_papers_50_epochs/probes_v9/newline2/{method_key}/sep_1_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs32_lr2e-05/run',
        '7B': {
            'para9': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/jiosephlee_prior_6_papers_50_epochs_7B/probes_v9/newline2/para9/sep_1_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs32_lr2e-05/overlap_1_4/09_22_00_24',
            'source_only': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/jiosephlee_prior_6_papers_50_epochs_7B/probes_v9/newline2/source_only/sep_1_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs32_lr2e-05/overlap_1_4/09_22_00_24'
        }
    }
    
    ft_only_base_paths = {
        '1B': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/1b/probes_v9/newline2/{method_key}/sep_1_dclm',
        '7B': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/{method_key}/sep_1_dclm'
    }

    methods = [
        ('para9_expl', 'Para. + Multiview', ft_base_paths['1B']),
        ('para9', 'Para.', ft_base_paths['1B']),
        ('source_only', 'Source', ft_base_paths['1B']),
    ]
    
    method_names_in_order = [m[1] for m in methods]
    color_palette = ['#1f77b4', '#2ca02c', '#ff7f0e']

    domains_path = os.path.join(project_root, 'data/arxiv/cleaned')
    try:
        domains = sorted([os.path.splitext(f)[0] for f in os.listdir(domains_path) if f.endswith('.tex') and os.path.isfile(os.path.join(domains_path, f))])
    except FileNotFoundError:
        print(f"Error: Domains path '{domains_path}' not found.")
        return
        
    output_dir = 'plots'
    os.makedirs(output_dir, exist_ok=True)
    
    output_filename = 'prior_knowledge_1B_7B_comparison.pdf'
    if args.with_LIMA:
        output_filename = output_filename.replace('.pdf', '_with_LIMA.pdf')
    output_path = os.path.join(output_dir, output_filename)

    # --- Data Aggregation ---
    all_data = {}
    models_to_plot = ['1B', '7B']
    for model_id in models_to_plot:
        if model_id == '1B':
            ft_methods_config = [
                (m[0], m[1], m[2]) for m in methods
            ]
        else: # 7B
            ft_methods_config = [
                (m[0], m[1], ft_base_paths['7B'].get(m[0])) for m in methods
            ]
        
        prior_k, prior_i, ft_k, ft_i, max_ft_step_k, max_ft_step_i = get_model_data(
            model_id, prior_knowledge_paths[model_id], ft_methods_config, domains, project_root, with_LIMA=args.with_LIMA, cut_off_at_minimal=args.cut_off_at_minimal
        )

        all_data[model_id] = {
            'prior_knowledge': prior_k,
            'prior_inference': prior_i,
            'ft_knowledge': ft_k,
            'ft_inference': ft_i,
            'max_ft_step_k': max_ft_step_k,
            'max_ft_step_i': max_ft_step_i
        }

        if not args.with_LIMA:
            ft_only_methods_config = [
                (m[0], m[1], ft_only_base_paths[model_id]) for m in methods
            ]
            ft_only_k, ft_only_i = get_ft_only_data(
                model_id, ft_only_methods_config, domains, project_root
            )
            all_data[model_id]['ft_only_knowledge'] = ft_only_k
            all_data[model_id]['ft_only_inference'] = ft_only_i
    # --- Plotting ---
    print("Generating comparison plot...")
    set_plot_style()
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Manually share y-axes
    axes[1].sharey(axes[0])
    axes[3].sharey(axes[2])
    plt.setp(axes[1].get_yticklabels(), visible=False)
    plt.setp(axes[3].get_yticklabels(), visible=False)

    for i, model_id in enumerate(models_to_plot):
        data = all_data[model_id]
        
        # --- Knowledge Plot ---
        ax_knowledge = axes[i*2]
        ax_knowledge.set_prop_cycle(color=color_palette)

        # Plot prior knowledge stage
        if not data['prior_knowledge'].empty:
            prior_k_plot = data['prior_knowledge'].groupby('step')['log_prob'].mean().reset_index()
            ax_knowledge.plot(prior_k_plot['step'], prior_k_plot['log_prob'], color='grey', linestyle=':', label='Prior Knowledge Stage')
            max_prior_step = prior_k_plot['step'].max()
            last_prior_point = prior_k_plot[prior_k_plot['step'] == max_prior_step]
        else:
            last_prior_point = pd.DataFrame()

        # Plot FT stages
        if not data['ft_knowledge'].empty:
            for method in method_names_in_order:
                method_df = data['ft_knowledge'][data['ft_knowledge']['method'] == method]
                if not method_df.empty:
                    plot_df = method_df.groupby('step')['log_prob'].mean().reset_index()
                    # Combine with last point of prior stage to connect lines
                    combined_plot_df = pd.concat([last_prior_point, plot_df], ignore_index=True)
                    ax_knowledge.plot(combined_plot_df['step'], combined_plot_df['log_prob'], label=method)
        
        if not args.with_LIMA:
            if 'ft_only_knowledge' in data and not data['ft_only_knowledge'].empty:
                for method in method_names_in_order:
                    method_df = data['ft_only_knowledge'][data['ft_only_knowledge']['method'] == method]
                    if not method_df.empty:
                        plot_df = method_df.groupby('step')['log_prob'].mean().reset_index()
                        ax_knowledge.plot(plot_df['step'], plot_df['log_prob'], linestyle='--')

        ax_knowledge.set_title(f'{model_id}: Factual Probes')
        ax_knowledge.set_xlabel('Training Step')
        if i == 0:
            ax_knowledge.set_ylabel('Mean Log Probability')

        # --- Inference Plot ---
        ax_inference = axes[i*2 + 1]
        ax_inference.set_prop_cycle(color=color_palette)

        # Plot prior knowledge stage
        if not data['prior_inference'].empty:
            prior_i_plot = data['prior_inference'].groupby('step')['log_prob'].mean().reset_index()
            ax_inference.plot(prior_i_plot['step'], prior_i_plot['log_prob'], color='grey', linestyle=':')
            max_prior_step = prior_i_plot['step'].max()
            last_prior_point = prior_i_plot[prior_i_plot['step'] == max_prior_step]
        else:
            last_prior_point = pd.DataFrame()

        # Plot FT stages
        if not data['ft_inference'].empty:
            for method in method_names_in_order:
                method_df = data['ft_inference'][data['ft_inference']['method'] == method]
                if not method_df.empty:
                    plot_df = method_df.groupby('step')['log_prob'].mean().reset_index()
                    combined_plot_df = pd.concat([last_prior_point, plot_df], ignore_index=True)
                    ax_inference.plot(combined_plot_df['step'], combined_plot_df['log_prob'], label=method)

        if not args.with_LIMA:
            if 'ft_only_inference' in data and not data['ft_only_inference'].empty:
                for method in method_names_in_order:
                    method_df = data['ft_only_inference'][data['ft_only_inference']['method'] == method]
                    if not method_df.empty:
                        plot_df = method_df.groupby('step')['log_prob'].mean().reset_index()
                        ax_inference.plot(plot_df['step'], plot_df['log_prob'], linestyle='--')

        ax_inference.set_title(f'{model_id}: Compositional Probes')
        ax_inference.set_xlabel('Training Step')

        if args.with_LIMA:
            if data['max_ft_step_k'] > 0:
                axes[i*2].axvline(x=data['max_ft_step_k'], color='red', linestyle='--', linewidth=1.5, alpha=0.9)
            if data['max_ft_step_i'] > 0:
                axes[i*2 + 1].axvline(x=data['max_ft_step_i'], color='red', linestyle='--', linewidth=1.5, alpha=0.9)


    from matplotlib.lines import Line2D
    legend_elements = []
    if not args.with_LIMA:
        legend_elements.extend([
            Line2D([0], [0], color='black', linestyle='-', label='With Prior Knowledge'),
            Line2D([0], [0], color='black', linestyle='--', label='Without Prior Knowledge')
        ])
    legend_elements.append(Line2D([0], [0], color='grey', linestyle=':', label='Prior Knowledge Stage'))
    
    # Add method colors to legend
    for i, method_name in enumerate(method_names_in_order):
        legend_elements.append(Line2D([0], [0], color=color_palette[i], label=method_name))

    axes[1].legend(handles=legend_elements, loc='lower right', fontsize='small', title_fontsize='small')

    for ax in fig.get_axes():
        ax.grid(True)

    fig.subplots_adjust(wspace=0.25, top=0.9)
    
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    main()
