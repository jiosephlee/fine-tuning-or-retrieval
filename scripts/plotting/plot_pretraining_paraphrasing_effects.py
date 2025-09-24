import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import re
from datetime import datetime
import argparse
from matplotlib.lines import Line2D

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
        ax.plot(plot_df['step'], plot_df['log_prob'], color=source_color, linestyle='-')

    k_para_df = k_df[k_df['pretraining_type'] == 'para9']
    if not k_para_df.empty:
        plot_df = k_para_df.groupby('step')['log_prob'].mean().reset_index()
        ax.plot(plot_df['step'], plot_df['log_prob'], color=para_color, linestyle='-')

    # Compositional - dashed
    i_source_df = i_df[i_df['pretraining_type'] == 'source_only']
    if not i_source_df.empty:
        plot_df = i_source_df.groupby('step')['log_prob'].mean().reset_index()
        ax.plot(plot_df['step'], plot_df['log_prob'], color=source_color, linestyle='--')

    i_para_df = i_df[i_df['pretraining_type'] == 'para9']
    if not i_para_df.empty:
        plot_df = i_para_df.groupby('step')['log_prob'].mean().reset_index()
        ax.plot(plot_df['step'], plot_df['log_prob'], color=para_color, linestyle='--')

def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    domains_path = os.path.join(project_root, 'data/arxiv/cleaned')
    try:
        domains = sorted([os.path.splitext(f)[0] for f in os.listdir(domains_path) if f.endswith('.tex')])
    except FileNotFoundError:
        print(f"Error: Domains path '{domains_path}' not found.")
        return

    set_plot_style()
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
    
    knowledge_df, inference_df = get_pretraining_length_data('7b', domains, project_root)

    if knowledge_df.empty and inference_df.empty:
        print("No data found for 7B model. Exiting.")
        return

    max_steps = 0
    if not knowledge_df.empty:
        max_steps = max(max_steps, knowledge_df['step'].max())
    if not inference_df.empty:
        max_steps = max(max_steps, inference_df['step'].max())
        
    pretraining_epochs = [30, 50, 100, 150]
    source_color = '#1f77b4'
    para_color = '#ff7f0e'
    
    for i, epoch in enumerate(pretraining_epochs):
        ax = axes[i]
        
        k_df_epoch = knowledge_df[knowledge_df['epoch'] == epoch]
        i_df_epoch = inference_df[inference_df['epoch'] == epoch]
        
        plot_epoch_comparison(ax, k_df_epoch, i_df_epoch, source_color, para_color)
        
        ax.set_title(f'{epoch} Exposures')
        ax.set_xlabel('Exposures')
        ax.set_xlim(0, max_steps)
        ax.grid(True)
        
        if i == 0:
            ax.set_ylabel('Mean Log Probability')

    legend_elements = [
        Line2D([0], [0], color=source_color, lw=2, label='Source'),
        Line2D([0], [0], color=para_color, lw=2, label='Para.'),
        Line2D([0], [0], color='black', linestyle='-', label='Factual'),
        Line2D([0], [0], color='black', linestyle='--', label='Compositional')
    ]
    axes[-1].legend(handles=legend_elements, loc='lower right', fontsize='small')
    
    fig.suptitle('7B Model: Analysis of Pretraining Length', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    output_dir = 'plots'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'pretraining_length_effects_7B.pdf')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
