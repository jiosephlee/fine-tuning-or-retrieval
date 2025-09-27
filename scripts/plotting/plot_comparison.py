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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils.llm_plotting import set_plot_style


def find_latest_run_path(base_path, model_id=None, override_dir=None):
    """
    Finds the path to the latest run data within a given base experiment directory,
    accommodating different directory structures.
    """
    basename = os.path.basename(base_path)
    if re.match(r'\d{2}_\d{2}_\d{2}_\d{2}', basename) and os.path.isdir(base_path):
        return base_path

    if not os.path.isdir(base_path):
        return None

    # --- Strategy 1: Check for timestamped directories directly in base_path ---
    # This is for structures where runs are directly inside the path (e.g., overlap ratio experiments)
    potential_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    run_dirs = [d for d in potential_dirs if re.match(r'\d{2}_\d{2}_\d{2}_\d{2}', d)]

    if run_dirs:
        latest_dir = sorted(run_dirs, key=lambda x: datetime.strptime(x, '%m_%d_%H_%M'), reverse=True)[0]
        return os.path.join(base_path, latest_dir)
    elif 'run' in potential_dirs:
        # Fallback to a 'run' directory if no timestamped dirs are found
        return os.path.join(base_path, 'run')

    # --- Strategy 2: Traverse a more complex, nested directory structure ---
    # This is for the original, more deeply nested experiment structure.
    try:
        domain_dirs = [d for d in os.listdir(base_path) if d.startswith('domains_') and os.path.isdir(os.path.join(base_path, d))]
        if not domain_dirs:
            return None
        domain_dir = domain_dirs[0]
        
        path = os.path.join(base_path, domain_dir)
        
        # Prefer 'e100' if it exists, otherwise find the highest epoch directory.
        e100_path = os.path.join(path, 'e100')
        if os.path.isdir(e100_path):
            path = e100_path
        else:
            epoch_dirs = [d for d in os.listdir(path) if d.startswith('e') and os.path.isdir(os.path.join(path, d)) and re.match(r'e\d+', d)]
            if not epoch_dirs:
                return None
            # Sort by epoch number to get the highest
            epoch_dirs.sort(key=lambda x: int(x[1:]), reverse=True)
            path = os.path.join(path, epoch_dirs[0])

        sub_dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        if not sub_dirs:
            return None
        path = os.path.join(path, sub_dirs[0])
        
        if model_id and '13b' in model_id.lower():
            potential_path = os.path.join(path, 'overlap_1_4')
            if os.path.isdir(potential_path):
                path = potential_path
        
        run_dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

        if override_dir and override_dir in run_dirs:
            return os.path.join(path, override_dir)

        if 'overlap_1_4' in run_dirs:
            return os.path.join(path, 'overlap_1_4')

        date_dirs = [d for d in run_dirs if re.match(r'\d{2}_\d{2}_\d{2}_\d{2}', d)]
        if date_dirs:
            latest_date = max(date_dirs, key=lambda d: datetime.strptime(d, '%m_%d_%H_%M'))
            return os.path.join(path, latest_date)

        if 'run' in run_dirs:
            return os.path.join(path, 'run')
            
        if 'no_overlap' in run_dirs:
            return os.path.join(path, 'no_overlap')

    except (FileNotFoundError, IndexError):
        return None

    return None


def aggregate_across_domains(run_path, probe_type, domains, split_probes=False, project_root='.', lima=False):
    """
    Aggregates probe data across multiple domains from a specific run path.
    """
    all_domain_dfs = []
    for domain in domains:
        if probe_type == "knowledge":
            probe_dir = f"{domain}_knowledge_probe"
            file_name = f"{domain}_knowledge_probe_metrics.csv"
            if lima:
                probe_dir = f"{domain}_lima_knowledge_probe"
                file_name = f"{domain}_knowledge_probe_metrics.csv" # The filename is the same, just the directory changes
        else: # inference
            probe_dir = f"{domain}_inference_probe"
            file_name = f"{domain}_inference_probe_metrics.csv"
            if lima:
                probe_dir = f"{domain}_lima_inference_probe"
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
            
            if split_probes:
                probe_folder = 'inference' if probe_type == 'inference' else 'facts'
                filter_path = os.path.join(project_root, 'data/probes', probe_folder, domain, 'filter.json')

                if os.path.exists(filter_path):
                    with open(filter_path, 'r') as f:
                        filter_data = json.load(f)
                    
                    explanations_only_indices = filter_data.get('in_explanations_only', [])
                    source_only_indices = filter_data.get('in_source_only', [])
                    
                    df['origin'] = 'Other'
                    df.loc[df['probe_index'].isin(explanations_only_indices), 'origin'] = 'Explanations Only'
                    df.loc[df['probe_index'].isin(source_only_indices), 'origin'] = 'Source Only'
                else:
                    print(f"Warning: filter.json not found for domain {domain} in {probe_folder}. Probes will not be split by origin.")
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


def get_model_data(model_id, methods, domains, split_probes, path_overrides, project_root, with_LIMA, cut_off_at_minimal):
    """
    Aggregates all probe data (including LIMA if requested) for a specific model.
    """
    experiment_base_path = f'results/FT/full/{model_id.lower()}/probes_v9/newline2'
    
    all_knowledge_data = []
    all_inference_data = []
    all_lima_knowledge_data = []
    all_lima_inference_data = []

    for method_key, method_name, sub_dir in methods:
        print(f"Processing method: {method_key} for model: {model_id}")
        method_base_path = os.path.join(experiment_base_path, method_key, sub_dir)
        override_dir = path_overrides.get(model_id.lower(), {}).get(method_key)
        run_path = find_latest_run_path(method_base_path, model_id, override_dir=override_dir)

        if run_path:
            # Standard Probes
            knowledge_df = aggregate_across_domains(run_path, 'knowledge', domains, split_probes=split_probes, project_root=project_root)
            if not knowledge_df.empty:
                knowledge_df['method'] = method_name
                all_knowledge_data.append(knowledge_df)
            
            inference_df = aggregate_across_domains(run_path, 'inference', domains, split_probes=split_probes, project_root=project_root)
            if not inference_df.empty:
                inference_df['method'] = method_name
                all_inference_data.append(inference_df)

            # LIMA Probes
            if with_LIMA:
                lima_knowledge_df = aggregate_across_domains(run_path, 'knowledge', domains, split_probes=split_probes, project_root=project_root, lima=True)
                if not lima_knowledge_df.empty:
                    lima_knowledge_df['method'] = method_name
                    all_lima_knowledge_data.append(lima_knowledge_df)
                
                lima_inference_df = aggregate_across_domains(run_path, 'inference', domains, split_probes=split_probes, project_root=project_root, lima=True)
                if not lima_inference_df.empty:
                    lima_inference_df['method'] = method_name
                    all_lima_inference_data.append(lima_inference_df)
        else:
            print(f"Could not find a valid run path for {method_key} on model {model_id}")

    # Combine data
    final_knowledge_df = pd.concat(all_knowledge_data, ignore_index=True) if all_knowledge_data else pd.DataFrame()
    final_inference_df = pd.concat(all_inference_data, ignore_index=True) if all_inference_data else pd.DataFrame()
    
    # Check for step consistency on non-LIMA data and cut off if needed
    min_knowledge_step = check_step_consistency(final_knowledge_df, "Knowledge")
    if cut_off_at_minimal and min_knowledge_step is not None:
        final_knowledge_df = final_knowledge_df[final_knowledge_df['step'] <= min_knowledge_step]

    min_inference_step = check_step_consistency(final_inference_df, "Inference")
    if cut_off_at_minimal and min_inference_step is not None:
        final_inference_df = final_inference_df[final_inference_df['step'] <= min_inference_step]

    max_knowledge_step_ft = final_knowledge_df['step'].max() if not final_knowledge_df.empty else 0
    max_inference_step_ft = final_inference_df['step'].max() if not final_inference_df.empty else 0

    # Append LIMA data if available
    if with_LIMA:
        if all_lima_knowledge_data:
            final_lima_knowledge_df = pd.concat(all_lima_knowledge_data, ignore_index=True)
            min_lima_knowledge_step = check_step_consistency(final_lima_knowledge_df, "LIMA Knowledge")
            if cut_off_at_minimal and min_lima_knowledge_step is not None:
                final_lima_knowledge_df = final_lima_knowledge_df[final_lima_knowledge_df['step'] <= min_lima_knowledge_step]
            
            if not final_knowledge_df.empty and not final_lima_knowledge_df.empty:
                final_lima_knowledge_df['step'] += max_knowledge_step_ft
                final_knowledge_df = pd.concat([final_knowledge_df, final_lima_knowledge_df], ignore_index=True)

        if all_lima_inference_data:
            final_lima_inference_df = pd.concat(all_lima_inference_data, ignore_index=True)
            min_lima_inference_step = check_step_consistency(final_lima_inference_df, "LIMA Inference")
            if cut_off_at_minimal and min_lima_inference_step is not None:
                final_lima_inference_df = final_lima_inference_df[final_lima_inference_df['step'] <= min_lima_inference_step]
            
            if not final_inference_df.empty and not final_lima_inference_df.empty:
                final_lima_inference_df['step'] += max_inference_step_ft
                final_inference_df = pd.concat([final_inference_df, final_lima_inference_df], ignore_index=True)

    return final_knowledge_df, final_inference_df, max_knowledge_step_ft, max_inference_step_ft


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
        "--larger",
        action='store_true',
        help="If set, compares 7B and 13B models instead of 1B and 7B."
    )
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
    parser.add_argument(
        "--separate_y_axis",
        action='store_true',
        help="If set, each plot in the unified view will have its own y-axis scale."
    )
    parser.add_argument(
        "--all_three",
        action='store_true',
        help="If set, plots 1B, 7B, and 13B models together in a 1x6 layout."
    )
    args = parser.parse_args()

    model_path_map = {
        '1b': '1b',
        '7b': '7b',
        '13b': 'allenai_OLMo-2-1124-13B'
    }
    model_id_for_path = model_path_map.get(args.model_id.lower(), args.model_id)

    # --- Configuration ---
    path_overrides = {
        '1b': {
            'para9_expl': 'run_2'
        }
    }
    # show_std_dev_shadows = False # New internal parameter
    # errorbar_setting = 'sd' if show_std_dev_shadows else None
    
    split_probes = args.split_probes
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    methods = [
        ('para9_expl', 'Para. + Aux. Views', 'sep_1_dclm'),
        ('para9', 'Para.', 'sep_1_dclm'),
        ('source_only', 'Source', 'sep_1_dclm'),
    ]
    method_names_in_order = [m[1] for m in methods]
    
    # Define a new color palette that doesn't include red
    color_palette = ['#2ca02c', '#ff7f0e', '#1f77b4']  # Green, Orange, Blue

    domains_path = os.path.join(project_root, 'data/arxiv/cleaned')
    try:
        domains = sorted([os.path.splitext(f)[0] for f in os.listdir(domains_path) if f.endswith('.tex') and os.path.isfile(os.path.join(domains_path, f))])
        print(f"Dynamically found domains: {domains}")
    except FileNotFoundError:
        print(f"Error: Domains path '{domains_path}' not found. Cannot determine domains.")
        domains = []
        
    max_step_ft = 0 # Initialize max_step_ft to handle cases where it's not set
    output_dir = 'plots'
    os.makedirs(output_dir, exist_ok=True)

    if not domains:
        print("No domains found. Exiting.")
        return

    # --- Data Aggregation ---
    final_knowledge_df, final_inference_df, max_knowledge_step_ft, max_inference_step_ft = get_model_data(model_id_for_path, methods, domains, split_probes, path_overrides, project_root, args.with_LIMA, args.cut_off_at_minimal)

    # --- Plotting ---
    print("Generating comparison plot...")

    # Set academic plot style
    set_plot_style()

    if split_probes:
        # --- PLOTTING FOR SPLIT PROBES (SINGLE MODEL) ---
        model_name = args.model_id.upper()
        output_filename = f'knowledge_inference_log_prob_{args.model_id}_split_probes.pdf'
        if args.with_LIMA:
            output_filename = output_filename.replace('.pdf', '_with_LIMA.pdf')
        output_path = os.path.join(output_dir, output_filename)

        if final_knowledge_df.empty and final_inference_df.empty:
            print("Error: No data was aggregated. Check paths and file availability.")
            return

        fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharey=False)
        
        max_step_ft = max_knowledge_step_ft

        # Add shading before plotting if LIMA is off
        if not args.with_LIMA and max_step_ft > 0:
            vline_steps = np.arange(30, max_step_ft + 1, 40)
            for step in vline_steps:
                start_shade = step - 5
                if start_shade < 0:
                    continue
                for ax_row in axes:
                    for ax in ax_row:
                        ax.axvline(x=start_shade, color='grey', linestyle='--', linewidth=1, alpha=0.7, zorder=0)
                        ax.axvline(x=step, color='grey', linestyle='--', linewidth=1, alpha=0.7, zorder=0)
                        ax.axvspan(start_shade, step, color='grey', alpha=0.2, hatch='/', zorder=0)

        probe_data_map = {
            'Factual': final_knowledge_df,
            'Compositional': final_inference_df
        }

        origin_title_map = {
            'Explanations Only': '(In Multi-view Only)',
            'Source Only': '(In Source Only)',
            'Other': '(In Both)'
        }

        for row, (probe_name, df) in enumerate(probe_data_map.items()):
            # Plot 1: Aggregated Probes
            ax_agg = axes[row, 0]
            ax_agg.set_prop_cycle(cycler(color=color_palette))
            
            total_probes = 0
            if not df.empty:
                total_probes = df[['domain', 'probe_index']].drop_duplicates().shape[0]
                for method in method_names_in_order:
                    method_df = df[df['method'] == method]
                    if not method_df.empty:
                        plot_df = method_df.groupby('step')['log_prob'].mean().reset_index()
                        ax_agg.plot(plot_df['step'], plot_df['log_prob'], label=method, lw=1.6)
            ax_agg.set_title(f'{probe_name} ({total_probes} Probes)')

            # Subplots for split probes by origin
            origins = ['Explanations Only', 'Source Only', 'Other']
            for i, origin in enumerate(origins):
                ax = axes[row, i + 1]
                ax.set_prop_cycle(cycler(color=color_palette))

                base_title = origin_title_map.get(origin, f'({origin})')
                
                num_probes = 0
                if not df.empty and 'origin' in df.columns:
                    origin_df = df[df['origin'] == origin]
                    if not origin_df.empty:
                        num_probes = origin_df[['domain', 'probe_index']].drop_duplicates().shape[0]
                        for method in method_names_in_order:
                            method_df = origin_df[origin_df['method'] == method]
                            if not method_df.empty:
                                plot_df = method_df.groupby('step')['log_prob'].mean().reset_index()
                                ax.plot(plot_df['step'], plot_df['log_prob'], label=method, lw=1.6)
                
                title = f"{base_title[:-1]}: {num_probes})"
                ax.set_title(title)

        # --- Final styling for the new grid ---
        for i in range(2):
            axes[i, 0].set_ylabel('Mean Log Probability')
            for j in range(4):
                axes[i, j].set_xlabel('Training Steps' if i == 1 else '')
                if i == 0:
                    axes[i,j].set_xticklabels([])


        # Add vertical lines for split-probe plot
        if max_step_ft > 0:
            if args.with_LIMA:
                vline_steps = np.arange(30, max_step_ft + 1, 40)
                for ax in fig.get_axes():
                    ax.axvline(x=max_step_ft, color='red', linestyle='--', linewidth=1.6, alpha=0.9)
                for ax in fig.get_axes():
                    for step in vline_steps:
                        ax.axvline(x=step, color='grey', linestyle='--', linewidth=1, alpha=0.7)
        
    else:
        # --- PLOTTING FOR UNIFIED VIEW ---
        if args.all_three:
            models_map = {'1B': '1b', '7B': '7b', '13B': 'allenai_OLMo-2-1124-13B'}
            output_filename = 'knowledge_inference_log_prob_1B_7B_13B_unified.pdf'
        elif args.larger:
            models_map = {'7B': '7b', '13B': 'allenai_OLMo-2-1124-13B'}
            output_filename = 'knowledge_inference_log_prob_7B_13B_unified.pdf'
        else:
            models_map = {'1B': '1b', '7B': '7b'}
            output_filename = 'knowledge_inference_log_prob_1B_7B_unified.pdf'

        if args.with_LIMA:
            output_filename = output_filename.replace('.pdf', '_with_LIMA.pdf')
        output_path = os.path.join(output_dir, output_filename)

        num_models = len(models_map)
        all_data = {model_name: {'knowledge': None, 'inference': None, 'max_step_ft': 0} for model_name in models_map.keys()}
        models_to_plot = list(models_map.keys())
        
        for model_name, model_path_id in models_map.items():
            knowledge_df, inference_df, max_step_ft_k, _ = get_model_data(
                model_path_id, methods, domains, split_probes, path_overrides, project_root, args.with_LIMA, args.cut_off_at_minimal
            )
            all_data[model_name]['knowledge'] = knowledge_df
            all_data[model_name]['inference'] = inference_df
            all_data[model_name]['max_step_ft'] = max_step_ft_k

        # --- Calculate unified y-axis limits ---
        unified_ylim = None
        if not args.separate_y_axis:
            y_min, y_max = np.inf, -np.inf
            for model_name in models_to_plot:
                df_k = all_data[model_name]['knowledge']
                if df_k is not None and not df_k.empty:
                    means_k = df_k.groupby(['method', 'step'])['log_prob'].mean()
                    if not means_k.empty:
                        y_min = min(y_min, means_k.min())
                        y_max = max(y_max, means_k.max())
                
                df_i = all_data[model_name]['inference']
                if df_i is not None and not df_i.empty:
                    means_i = df_i.groupby(['method', 'step'])['log_prob'].mean()
                    if not means_i.empty:
                        y_min = min(y_min, means_i.min())
                        y_max = max(y_max, means_i.max())

            if np.isfinite(y_min) and np.isfinite(y_max):
                padding = (y_max - y_min) * 0.05
                unified_ylim = (y_min - padding, y_max + padding)

        if args.larger and not args.all_three:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        else:
            fig, axes = plt.subplots(1, num_models * 2, figsize=(8 * num_models, 4))
        
        # Add shading before plotting if LIMA is off
        if not args.with_LIMA:
            for i, model_id in enumerate(models_to_plot):
                max_step_ft_model = all_data[model_id]['max_step_ft']
                if max_step_ft_model > 0:
                    if args.larger and not args.all_three:
                        model_axes = [axes[i, 0], axes[i, 1]]
                    else:
                        model_axes = [axes[i*2], axes[i*2 + 1]]
                    vline_steps = np.arange(30, max_step_ft_model + 1, 40)
                    for step in vline_steps:
                        start_shade = step - 5
                        if start_shade < 0:
                            continue
                        for ax in model_axes:
                            ax.axvline(x=start_shade, color='grey', linestyle='--', linewidth=1, alpha=0.7, zorder=0)
                            ax.axvline(x=step, color='grey', linestyle='--', linewidth=1, alpha=0.7, zorder=0)
                            ax.axvspan(start_shade, step, color='grey', alpha=0.2, hatch='/', zorder=0)

        # Set unified y-axis and hide unnecessary tick labels
        if not args.separate_y_axis:
            if args.larger and not args.all_three:
                plt.setp(axes[0, 1].get_yticklabels(), visible=False)
                plt.setp(axes[1, 1].get_yticklabels(), visible=False)
            else:
                for j in range(1, num_models * 2):
                    plt.setp(axes[j].get_yticklabels(), visible=False)

        for i, model_id in enumerate(models_to_plot):
            # Knowledge Plot
            if args.larger and not args.all_three:
                ax_knowledge = axes[i, 0]
            else:
                ax_knowledge = axes[i*2]
            ax_knowledge.set_prop_cycle(color=color_palette)
            df = all_data[model_id]['knowledge']
            if df is not None and not df.empty:
                for method in method_names_in_order:
                    method_df = df[df['method'] == method]
                    if not method_df.empty:
                        plot_df = method_df.groupby('step')['log_prob'].mean().reset_index()
                        ax_knowledge.plot(plot_df['step'], plot_df['log_prob'], label=method, lw=1.6)
            ax_knowledge.set_title(f'{model_id}: Factual Probes')

            if args.larger and not args.all_three:
                if i == len(models_to_plot) - 1:
                    ax_knowledge.set_xlabel('Training Steps')
                ax_knowledge.set_ylabel('Mean Log Probability')
            else:
                ax_knowledge.set_xlabel('Training Steps')
                if i == 0:
                    ax_knowledge.set_ylabel('Mean Log Probability')
            
            if unified_ylim:
                ax_knowledge.set_ylim(unified_ylim)

            # Compositional Plot
            if args.larger and not args.all_three:
                ax_compositional = axes[i, 1]
            else:
                ax_compositional = axes[i*2 + 1]
            ax_compositional.set_prop_cycle(color=color_palette)
            df = all_data[model_id]['inference']
            if df is not None and not df.empty:
                for method in method_names_in_order:
                    method_df = df[df['method'] == method]
                    if not method_df.empty:
                        plot_df = method_df.groupby('step')['log_prob'].mean().reset_index()
                        ax_compositional.plot(plot_df['step'], plot_df['log_prob'], label=method, lw=1.6)
            ax_compositional.set_title(f'{model_id}: Compositional Probes')
            if args.larger and not args.all_three:
                if i == len(models_to_plot) - 1:
                    ax_compositional.set_xlabel('Training Steps')
            else:
                ax_compositional.set_xlabel('Training Steps')
            
            if unified_ylim:
                ax_compositional.set_ylim(unified_ylim)

            # Add vertical lines for this model's plots
            max_step_ft = all_data[model_id]['max_step_ft']
            if max_step_ft > 0:
                if args.with_LIMA:
                    if args.larger and not args.all_three:
                        model_axes = [ax_knowledge, ax_compositional]
                    else:
                        model_axes = [axes[i*2], axes[i*2 + 1]]
                    vline_steps = np.arange(30, max_step_ft + 1, 40)
                    for ax in model_axes:
                        ax.axvline(x=max_step_ft, color='red', linestyle='--', linewidth=1.6, alpha=0.9)
                    for ax in model_axes:
                        for step in vline_steps:
                            ax.axvline(x=step, color='grey', linestyle='--', linewidth=1, alpha=0.7)
    
    if split_probes:
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles: # Only show legend if there are items to show
            fig.legend(handles, labels, loc='lower center', ncol=len(method_names_in_order), bbox_to_anchor=(0.5, -0.05), fontsize='medium')
    else:
        if args.larger and not args.all_three:
            handles, labels = axes[0, 0].get_legend_handles_labels()
            if handles:
                axes[0, 0].legend(handles, labels, loc='lower right', fontsize='medium', title_fontsize='medium')
        else:
            handles, labels = axes[0].get_legend_handles_labels()
            if handles:
                axes[0].legend(handles, labels, loc='lower right', fontsize='medium', title_fontsize='medium')

    # --- FINAL STYLING ---
    for ax in fig.get_axes():
        ax.grid(False)

    if args.larger or args.all_three:
        fig.tight_layout()
    else:
        fig.subplots_adjust(wspace=0.25, top=0.9)
    
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
