import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import re
from datetime import datetime
import argparse
from matplotlib.lines import Line2D
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from scripts.plotting.plot_comparison import aggregate_across_domains
from utils.llm_plotting import set_plot_style

def find_latest_run_path(base_path):
    """
    Finds the path to the latest run directory based on timestamped directory names.
    Expected format: MM_DD_HH_MM.
    """
    try:
        run_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
        date_dirs = [d for d in run_dirs if re.match(r'\d{2}_\d{2}_\d{2}_\d{2}', d)]
        if not date_dirs:
            if 'run' in run_dirs:
                return os.path.join(base_path, 'run')
            # Check for directories that might contain date dirs
            for r_dir in run_dirs:
                path = os.path.join(base_path, r_dir)
                sub_date_dirs = [d for d in os.listdir(path) if re.match(r'\d{2}_\d{2}_\d{2}_\d{2}', d)]
                if sub_date_dirs:
                    latest_date_dir = max(sub_date_dirs)
                    return os.path.join(path, latest_date_dir)
            return None

        latest_date_dir = max(date_dirs)
        return os.path.join(base_path, latest_date_dir)
    except FileNotFoundError:
        print(f"Directory not found: {base_path}")
        return None

def get_pretraining_length_data(model_size, domains, project_root):
    """
    Gathers data for different pretraining lengths.
    """
    data = {'knowledge': [], 'inference': []}
    
    pretraining_epochs = [30, 50, 100, 150]
    pretraining_types = [('source_only', 'Source'), ('para9', 'Para.')]
    
    model_size_path = '7b' if model_size == '1b' else model_size # Use 7b data for 1b as requested

    for epoch in pretraining_epochs:
        for p_type, p_name in pretraining_types:
            base_path = f"results/FT/full/{model_size_path}/probes_v9/newline2/{p_type}/sep_1_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e{epoch}/bs32_lr2e-05"
            run_path = find_latest_run_path(base_path)
            
            if not run_path:
                print(f"Could not find run path for {model_size}, {p_type}, e{epoch}")
                continue

            for probe_type in ['knowledge', 'inference']:
                df = aggregate_across_domains(run_path, probe_type, domains, project_root=project_root)
                if not df.empty:
                    df = df[np.isfinite(df['log_prob'])]
                    if not df.empty:
                        df['method'] = f"e{epoch} - {p_name}"
                        df['epoch'] = epoch
                        df['pretraining_type'] = p_type
                        data[probe_type].append(df)

    return (pd.concat(data['knowledge'], ignore_index=True) if data['knowledge'] else pd.DataFrame(),
            pd.concat(data['inference'], ignore_index=True) if data['inference'] else pd.DataFrame())

def get_paraphrasing_data(model_size, domains, project_root):
    """
    Gathers data for different paraphrasing levels.
    """
    data = {'knowledge': [], 'inference': []}
    paraphrase_types = [('para4', 'Para. 4'), ('para9', 'Para. 9'), ('para19', 'Para. 19')]
    if model_size == '1b' or model_size == '7b':
        paraphrase_types.append(('para49', 'Para. 49'))

    for p_type, p_name in paraphrase_types:
        base_path = f"results/FT/full/{model_size}/probes_v9/newline2/{p_type}/sep_1_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs32_lr2e-05"
        
        # Conditionally add 'overlap_1_4' if it exists
        if model_size == '1b':
            potential_path = os.path.join(base_path, "overlap_1_4")
            if os.path.isdir(potential_path):
                base_path = potential_path

        run_path = find_latest_run_path(base_path)

        if not run_path:
            print(f"Could not find run path for {model_size}, {p_type}")
            continue

        for probe_type in ['knowledge', 'inference']:
            df = aggregate_across_domains(run_path, probe_type, domains, project_root=project_root)
            if not df.empty:
                df = df[np.isfinite(df['log_prob'])]
                if not df.empty:
                    df['method'] = p_name
                    data[probe_type].append(df)

    return (pd.concat(data['knowledge'], ignore_index=True) if data['knowledge'] else pd.DataFrame(),
            pd.concat(data['inference'], ignore_index=True) if data['inference'] else pd.DataFrame())

def plot_epoch_comparison(ax, k_df, i_df, source_color, para_color):
    """
    Plots a comparison for a single epoch, showing factual vs. compositional probes.
    """
    # Factual - solid
    k_source_df = k_df[k_df['pretraining_type'] == 'source_only']
    if not k_source_df.empty:
        plot_df = k_source_df.groupby('step')['log_prob'].mean().reset_index()
        ax.plot(plot_df['step'], plot_df['log_prob'], color=source_color, linestyle='-', lw=2)

    k_para_df = k_df[k_df['pretraining_type'] == 'para9']
    if not k_para_df.empty:
        plot_df = k_para_df.groupby('step')['log_prob'].mean().reset_index()
        ax.plot(plot_df['step'], plot_df['log_prob'], color=para_color, linestyle='-', lw=2)

    # Compositional - dashed
    i_source_df = i_df[i_df['pretraining_type'] == 'source_only']
    if not i_source_df.empty:
        plot_df = i_source_df.groupby('step')['log_prob'].mean().reset_index()
        ax.plot(plot_df['step'], plot_df['log_prob'], color=source_color, linestyle='--', lw=2)

    i_para_df = i_df[i_df['pretraining_type'] == 'para9']
    if not i_para_df.empty:
        plot_df = i_para_df.groupby('step')['log_prob'].mean().reset_index()
        ax.plot(plot_df['step'], plot_df['log_prob'], color=para_color, linestyle='--', lw=2)

def plot_paraphrasing_data(ax, df, title, ylabel, show_legend=False):
    """
    Plots the paraphrasing data on a given axes.
    """
    if df.empty:
        ax.set_title(f'{title}\\n(No data)')
        return

    paraphrase_methods = ['Para. 49', 'Para. 19', 'Para. 9', 'Para. 4']
    colors = ['#211511', '#795548', '#b56535', '#D2B48C']  # Very dark, dark, lighter medium, light brown
    color_map = {
        'Para. 49': colors[0],
        'Para. 19': colors[1],
        'Para. 9': colors[2],
        'Para. 4': colors[3]
    }

    for method in paraphrase_methods:
        if method in df['method'].unique():
            method_df = df[df['method'] == method]
            plot_df = method_df.groupby('step')['log_prob'].mean().reset_index()
            ax.plot(plot_df['step'], plot_df['log_prob'], label=method, color=color_map.get(method), lw=2)
    
    ax.set_title(title)
    ax.set_xlabel('Training Steps')
    if ylabel:
        ax.set_ylabel('Mean Log Probability')
    if show_legend:
        ax.legend(loc='lower right', fontsize='large')
    ax.grid(True)

def main():
    parser = argparse.ArgumentParser(description="Generate plots for pretraining and paraphrasing effects.")
    parser.add_argument('--left', type=float, default=0.05, help='Left margin.')
    parser.add_argument('--right', type=float, default=0.95, help='Right margin.')
    parser.add_argument('--bottom', type=float, default=0.1, help='Bottom margin.')
    parser.add_argument('--top', type=float, default=0.95, help='Top margin.')
    parser.add_argument('--wspace', type=float, default=0.15, help='Width space between subplots.')
    parser.add_argument('--hspace', type=float, default=0.25, help='Height space between subplots.')
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    domains_path = os.path.join(project_root, 'data/arxiv/cleaned')
    try:
        domains = sorted([os.path.splitext(f)[0] for f in os.listdir(domains_path) if f.endswith('.tex')])
    except FileNotFoundError:
        print(f"Error: Domains path '{domains_path}' not found.")
        return

    set_plot_style()
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # --- Top Row: Pretraining Length Comparison (7B only) ---
    knowledge_df_7b, inference_df_7b = get_pretraining_length_data('7b', domains, project_root)

    if knowledge_df_7b.empty and inference_df_7b.empty:
        print("No data found for 7B model pretraining length comparison. Skipping top row.")
        for ax in axes[0, :]:
            ax.set_visible(False)
    else:
        max_steps = 0
        if not knowledge_df_7b.empty:
            max_steps = max(max_steps, knowledge_df_7b['step'].max())
        if not inference_df_7b.empty:
            max_steps = max(max_steps, inference_df_7b['step'].max())

        # --- Calculate y-axis limits for the top row ---
        y_min, y_max = np.inf, -np.inf
        if not knowledge_df_7b.empty:
            means_k = knowledge_df_7b.groupby(['pretraining_type', 'epoch', 'step'])['log_prob'].mean()
            if not means_k.empty:
                y_min = min(y_min, means_k.min())
                y_max = max(y_max, means_k.max())
        if not inference_df_7b.empty:
            means_i = inference_df_7b.groupby(['pretraining_type', 'epoch', 'step'])['log_prob'].mean()
            if not means_i.empty:
                y_min = min(y_min, means_i.min())
                y_max = max(y_max, means_i.max())

        top_ylim = None
        if np.isfinite(y_min) and np.isfinite(y_max):
            padding = (y_max - y_min) * 0.05
            top_ylim = (y_min - padding, y_max + padding)

        pretraining_epochs = [30, 50, 100, 150]
        source_color = '#1f77b4'
        para_color = '#ff7f0e'
        
        for i, epoch in enumerate(pretraining_epochs):
            ax = axes[0, i]
            
            k_df_epoch = knowledge_df_7b[knowledge_df_7b['epoch'] == epoch]
            i_df_epoch = inference_df_7b[inference_df_7b['epoch'] == epoch]
            
            plot_epoch_comparison(ax, k_df_epoch, i_df_epoch, source_color, para_color)
            
            ax.set_title(f'7B: {epoch} Exposures')
            ax.set_xlabel('Training Steps')
            ax.set_xlim(0, max_steps)
            ax.grid(True)
            
            if i == 0:
                ax.set_ylabel('Mean Log Probability')

        if top_ylim:
            axes[0, 0].set_ylim(top_ylim)

        for i in range(1, 4):
            axes[0, i].sharey(axes[0, 0])
            plt.setp(axes[0, i].get_yticklabels(), visible=False)

        legend_elements = [
            Line2D([0], [0], color=para_color, lw=2, label='Para.'),
            Line2D([0], [0], color=source_color, lw=2, label='Source'),
            Line2D([0], [0], color='black', linestyle='-', lw=2, label='Factual'),
            Line2D([0], [0], color='black', linestyle='--', lw=2, label='Compositional')
        ]
        axes[0, -1].legend(handles=legend_elements, loc='lower right', fontsize='large')

    # --- Bottom Row: Paraphrasing Effects ---
    models = ['1b', '7b']
    for i, model_size in enumerate(models):
        k_df, i_df = get_paraphrasing_data(model_size, domains, project_root)
        
        ax_k = axes[1, i*2]
        plot_paraphrasing_data(ax_k, k_df, f'{model_size.upper()}: Factual Probes', ylabel=(i == 0), show_legend=False)
        
        ax_i = axes[1, i*2 + 1]
        is_last_plot = (i == len(models) - 1)
        plot_paraphrasing_data(ax_i, i_df, f'{model_size.upper()}: Compositional Probes', ylabel=False, show_legend=is_last_plot)

    # --- Calculate and set y-limits for the bottom row ---
    models_data = {'1b': get_paraphrasing_data('1b', domains, project_root),
                   '7b': get_paraphrasing_data('7b', domains, project_root)}

    for i, model_size in enumerate(models):
        k_df, i_df = models_data[model_size]
        y_min, y_max = np.inf, -np.inf
        
        if not k_df.empty:
            means_k = k_df.groupby(['method', 'step'])['log_prob'].mean()
            if not means_k.empty:
                y_min = min(y_min, means_k.min())
                y_max = max(y_max, means_k.max())
        
        if not i_df.empty:
            means_i = i_df.groupby(['method', 'step'])['log_prob'].mean()
            if not means_i.empty:
                y_min = min(y_min, means_i.min())
                y_max = max(y_max, means_i.max())

        if np.isfinite(y_min) and np.isfinite(y_max):
            data_range = y_max - y_min
            padding = data_range * 0.05
            # Raise the bottom limit to zoom in on the differences
            bottom_limit = y_min + data_range * 0.25
            top_limit = y_max + padding
            axes[1, i*2].set_ylim(bottom_limit, top_limit)

    axes[1, 1].sharey(axes[1, 0])
    axes[1, 3].sharey(axes[1, 2])
    plt.setp(axes[1, 1].get_yticklabels(), visible=False)
    plt.setp(axes[1, 3].get_yticklabels(), visible=False)
    
    fig.subplots_adjust(left=args.left, right=args.right, bottom=args.bottom, top=args.top, wspace=args.wspace, hspace=args.hspace)
    
    output_dir = 'plots'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'pretraining_paraphrasing_effects.pdf')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
