import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import argparse
from matplotlib.gridspec import GridSpec

# Adjust the path to include the utils directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)

# Import from utils
from utils.llm_plotting import set_plot_style
from scripts.plotting.plot_utils import (
    discover_domains,
    load_probe_series,
    add_legend,
    apply_ylim,
    compute_unified_ylim
)

# === Specialized Data Loaders for this Plot ===

def get_final_val(run_path, probe_type, domains, project_root):
    """
    Helper to load series and extract the final step mean log_prob.
    Returns None if data is missing or path is invalid.
    """
    if not run_path or not os.path.exists(run_path):
        return None
        
    # Use load_probe_series to handle finding the run and aggregating
    df = load_probe_series(run_path, probe_type, domains, project_root)
    
    if df is None or df.empty:
        return None
        
    # Get value at max step
    max_step = df['step'].max()
    final_val = df[df['step'] == max_step]['log_prob'].mean()
    return final_val

def load_special_inference_data(base_path, domain_name='1_58'):
    """
    Specialized loader for Subplot 2 due to idiosyncratic path structure 
    where train and test CSVs are explicitly separated in a specific domain folder.
    """
    if not base_path or not os.path.exists(base_path):
        return None, None

    probe_dir = os.path.join(base_path, f'{domain_name}_inference_probe')
    if not os.path.isdir(probe_dir):
        print(f"Warning: Special probe dir not found: {probe_dir}")
        return None, None

    train_file = os.path.join(probe_dir, f'train_{domain_name}_inference_probe_metrics.csv')
    test_file = os.path.join(probe_dir, f'test_{domain_name}_inference_probe_metrics.csv')

    def load_and_process(fpath):
        if not os.path.exists(fpath) or os.path.getsize(fpath) == 0:
            return pd.DataFrame()
        try:
            df = pd.read_csv(fpath)
            if 'step' in df.columns and 'log_prob' in df.columns:
                df['step'] = pd.to_numeric(df['step'], errors='coerce')
                df['log_prob'] = pd.to_numeric(df['log_prob'], errors='coerce')
                df.dropna(subset=['step', 'log_prob'], inplace=True)
                df['step'] = df['step'].astype(int)
                # Aggregate mean per step immediately
                return df.groupby('step')['log_prob'].mean().reset_index().sort_values('step')
        except Exception as e:
            print(f"Error reading {fpath}: {e}")
        return pd.DataFrame()

    train_df = load_and_process(train_file)
    test_df = load_and_process(test_file)
    
    return train_df, test_df


def main():
    parser = argparse.ArgumentParser(description="Generate intricate multi-panel comparison plot.")
    args = parser.parse_args()

    set_plot_style()
    domains = discover_domains(project_root)
    if not domains:
        print("No domains found in data/arxiv/cleaned. Exiting.")
        return
    print(f"Discovered domains: {domains}")

    # ==========================================
    # Data Gathering: Subplot 1 (Model Scaling)
    # ==========================================
    print("Gathering data for Subplot 1...")
    models = ['1B', '7B', '13B', '32B']
    model_map = {m: i for i, m in enumerate(models)}
    
    s1_paths = {
        'Baseline (Para9)': {
            '1B': None, # Placeholder
            '7B': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_20_22_51',
            '13B': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/allenai_OLMo-2-1124-13B/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_21_13_18',
            '32B': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/allenai_OLMo-2-0325-32B/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_21_06_44',
        },
        'Aux Views': {
             '1B': None, '13B': None, '32B': None,
             '7B': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/para9_expl_stackexchange+blogs+textbooks_cyclefull/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_21_04_43'
        },
        'Corrupted Aux': {'1B': None, '7B': None, '13B': None, '32B': None}
    }

    s1_data = {}
    for method, m_paths in s1_paths.items():
        s1_data[method] = {'factual': [], 'compositional': []}
        for model in models:
            path = m_paths.get(model)
            # Factual
            val_k = get_final_val(path, 'knowledge', domains, project_root)
            s1_data[method]['factual'].append(val_k)
            # Compositional (Inference)
            val_i = get_final_val(path, 'inference', domains, project_root)
            s1_data[method]['compositional'].append(val_i)


    # ==========================================
    # Data Gathering: Subplot 2 (Training Dynamics)
    # ==========================================
    print("Gathering data for Subplot 2...")
    s2_experiments = [
        ('Control (Para9)', '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_22_05_50_test_probes'),
        ('Exp 1 (LLM Probes)', '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/para9_expl_llm_probes_cyclefull/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_22_05_50'),
        ('Exp 2 (w/ Expl)', '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/para9_expl_llm_probes_with_explanations_cyclefull/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_22_05_50'),
        ('Exp 3 (Longer Expl)', '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/para9_expl_llm_probes_with_explanations_longer_cyclefull/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_22_05_50'),
    ]
    
    s2_data = {}
    for label, path in s2_experiments:
        # Use specialized loader for the specific '1_58' domain structure
        train_df, test_df = load_special_inference_data(path, domain_name='1_58')
        s2_data[label] = {'train': train_df, 'test': test_df}


    # ==========================================
    # Data Gathering: Subplot 3 (Shuffling Histograms)
    # ==========================================
    print("Gathering data for Subplot 3...")
    s3_bunch1_paths = {
        'Baseline': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_20_22_51',
        'Word Shuf': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_21_11_30_shuffle_words',
        'Sent Shuf': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_21_11_31_shuffle_sentences',
    }
    s3_bunch2_paths = {
        'Aux TB': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/para9_expl_textbooks_cyclefull/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_21_02_11',
        'TB Word Shuf': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/para9_expl_shuffled_words_textbook_cyclefull/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_21_02_11',
        'TB Sent Shuf': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/para9_expl_shuffled_sentences_textbook_cyclefull/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_21_02_11',
    }

    s3_data = {'bunch1': {'f': [], 'c': [], 'labels': []}, 'bunch2': {'f': [], 'c': [], 'labels': []}}

    for label, path in s3_bunch1_paths.items():
        s3_data['bunch1']['labels'].append(label)
        s3_data['bunch1']['f'].append(get_final_val(path, 'knowledge', domains, project_root))
        s3_data['bunch1']['c'].append(get_final_val(path, 'inference', domains, project_root))
        
    for label, path in s3_bunch2_paths.items():
        s3_data['bunch2']['labels'].append(label)
        s3_data['bunch2']['f'].append(get_final_val(path, 'knowledge', domains, project_root))
        s3_data['bunch2']['c'].append(get_final_val(path, 'inference', domains, project_root))


    # ==========================================
    # PLOTTING
    # ==========================================
    print("Generating plot...")
    # Use GridSpec to create the layout: 1 row, 4 cols, where last two are narrower
    fig = plt.figure(figsize=(22, 5))
    gs = GridSpec(1, 4, width_ratios=[2, 2, 1, 1], figure=fig, wspace=0.25)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    ax4 = fig.add_subplot(gs[3])

    # --- SUBPLOT 1: Scaling Lines ---
    s1_colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # Blue, Orange, Green
    x_vals = np.arange(len(models))
    
    for i, (method, data) in enumerate(s1_data.items()):
        c = s1_colors[i]
        # Factual (solid)
        ax1.plot(x_vals, data['factual'], label=f"{method} (Fact)", color=c, marker='o', lw=2)
        # Compositional (dashed)
        # Filter out None values to avoid plotting errors if a whole series is missing
        comp_data = [d if d is not None else np.nan for d in data['compositional']]
        if not all(np.isnan(comp_data)):
             ax1.plot(x_vals, comp_data, label=f"{method} (Comp)", color=c, linestyle='--', marker='x', lw=2)

    ax1.set_xticks(x_vals)
    ax1.set_xticklabels(models)
    ax1.set_title("Model Scaling (Final Step Log Prob)")
    ax1.set_ylabel("Mean Log Probability")
    ax1.set_xlabel("Model Size")
    add_legend(ax1, loc='lower left', fontsize='small', ncol=2)
    ax1.grid(True, alpha=0.3)


    # --- SUBPLOT 2: Training Dynamics (Compositional) ---
    s2_colors = ['gray', '#d62728', '#9467bd', '#8c564b'] # Gray (control), Red, Purple, Brown
    
    for i, (label, dfs) in enumerate(s2_data.items()):
        c = s2_colors[i]
        train_df = dfs['train']
        test_df = dfs['test']
        
        if not train_df.empty:
            ax2.plot(train_df['step'], train_df['log_prob'], label=f"{label} (Train)", color=c, lw=1.5)
        if not test_df.empty:
            # Ensure test doesn't overlap weirdly if steps are same, though usually they are later
            ax2.plot(test_df['step'], test_df['log_prob'], label=f"{label} (Test)", color=c, linestyle='--', lw=1.5)

    ax2.set_title("Compositional Probes: Training Dynamics")
    ax2.set_xlabel("Training Steps")
    # Share Y label implicitly by proximity, or add if needed.
    add_legend(ax2, loc='lower right', fontsize='small', ncol=1)
    ax2.grid(True, alpha=0.3)


    # --- SUBPLOT 3 & 4: Shuffling Histograms ---
    # Shared settings for histograms
    bar_width = 0.35
    opacity = 0.8
    
    # Helper for plotting bunches
    def plot_bunch(ax, bunch_data, title_suffix):
        labels = bunch_data['labels']
        f_vals = [v if v is not None else 0 for v in bunch_data['f']]
        c_vals = [v if v is not None else 0 for v in bunch_data['c']]
        
        x = np.arange(len(labels))
        
        ax.bar(x - bar_width/2, f_vals, bar_width, alpha=opacity, color='#1f77b4', label='Factual')
        ax.bar(x + bar_width/2, c_vals, bar_width, alpha=opacity, color='#ff7f0e', hatch='//', label='Compositional')
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha='right', fontsize='small')
        ax.set_title(f"Shuffling Effect\n({title_suffix})", fontsize='medium')
        ax.grid(axis='y', alpha=0.3)

    # Plot Bunch 1 (Standard) on AX3
    plot_bunch(ax3, s3_data['bunch1'], "Standard Para9")
    
    # Plot Bunch 2 (Textbook Aux) on AX4
    plot_bunch(ax4, s3_data['bunch2'], "Aux Textbook")
    
    # Shared legend for 3 & 4 placed on 4
    handles, labels = ax4.get_legend_handles_labels()
    ax4.legend(handles, labels, loc='upper center', bbox_to_anchor=(-0.1, -0.35), ncol=2, fontsize='small')

    # Ensure Y-limits match across all plots for easier comparison
    all_axes = [ax1, ax2, ax3, ax4]
    unified_ylim = compute_unified_ylim(axes=all_axes, padding=0.1)
    if unified_ylim:
        apply_ylim(all_axes, unified_ylim)

    # Final adjustments
    # plt.tight_layout() # GridSpec makes tight_layout tricky, relying on wspace/hspace in GridSpec def
    
    output_dir = 'plots'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'complex_comparison_summary.pdf')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
