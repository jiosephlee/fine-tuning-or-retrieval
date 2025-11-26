import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import argparse
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

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
    # Data Definitions
    # ==========================================
    
    # --- 1. Model Scaling Data (Absolute Values) ---
    print("Gathering data for Model Scaling...")
    models = ['1B', '7B', '13B', '32B']
    
    s_scaling_paths = {
        'Baseline (Para9)': {
            '1B': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/1b/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_23_23_24',
            '7B': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_20_22_51',
            '13B': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/allenai_OLMo-2-1124-13B/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_21_13_18',
            '32B': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/allenai_OLMo-2-0325-32B/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_21_06_44',
        },
        'Aux Views': {
             '1B': None, 
             '7B': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/para9_expl_stackexchange+blogs+textbooks_cyclefull/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_21_04_43/',
             '13B': None, 
             '32B': None,
        },
        'Corrupted Aux': {
            '1B': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/1b/probes_v9/newline2/para9_expl_fruit_textbooks_v3+fruit_blogs_v3+fruit_stackexchange_v3_cyclefull/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_24_10_47',
            '7B': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/para9_expl_fruit_textbooks_v3+fruit_blogs_v3+fruit_stackexchange_v3_cyclefull/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_24_02_47/',
            '13B': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/allenai_OLMo-2-1124-13B/probes_v9/newline2/para9_expl_fruit_textbooks_v3+fruit_blogs_v3+fruit_stackexchange_v3_cyclefull/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_24_08_50',
            '32B': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/allenai_OLMo-2-0325-32B/probes_v9/newline2/para9_expl_fruit_textbooks_v3+fruit_blogs_v3+fruit_stackexchange_v3_cyclefull/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_24_05_10/'
        }
    }

    s_scaling_data = {}
    for method, m_paths in s_scaling_paths.items():
        s_scaling_data[method] = {'factual': [], 'compositional': []}
        for model in models:
            path = m_paths.get(model)
            val_k = get_final_val(path, 'knowledge', domains, project_root)
            val_i = get_final_val(path, 'inference', domains, project_root)
            s_scaling_data[method]['factual'].append(val_k)
            s_scaling_data[method]['compositional'].append(val_i)


    # --- 2. Strategies Data (Delta Values) ---
    print("Gathering data for Strategies...")
    s_strat_paths = {
        'Textbooks': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/para9_expl_textbooks_cyclefull/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_21_02_11/',
        'StackExchange': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/para9_expl_stackexchange_cyclefull/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_21_02_11/',
        'Blogs': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/para9_expl_blogs_cyclefull/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_21_02_11/',
        'Aux Views': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/para9_expl_stackexchange+blogs+textbooks_cyclefull/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_21_04_43/'
    }
    
    # Calculate Baseline (Para 9 7B) for Delta
    baseline_path_7b = s_scaling_paths['Baseline (Para9)']['7B']
    base_f = get_final_val(baseline_path_7b, 'knowledge', domains, project_root)
    base_c = get_final_val(baseline_path_7b, 'inference', domains, project_root)
    
    s_strat_data = {'labels': [], 'f': [], 'c': []}
    order_strat = ['Textbooks', 'StackExchange', 'Blogs', 'Aux Views']
    
    for label in order_strat:
        path = s_strat_paths[label]
        val_f = get_final_val(path, 'knowledge', domains, project_root)
        val_c = get_final_val(path, 'inference', domains, project_root)
        
        # Store DELTA
        s_strat_data['labels'].append(label)
        s_strat_data['f'].append(val_f - base_f if val_f is not None and base_f is not None else None)
        s_strat_data['c'].append(val_c - base_c if val_c is not None and base_c is not None else None)


    # --- 3. Shuffling Data (Relative Delta Values) ---
    print("Gathering data for Shuffling...")
    
    # Define Baselines for each group
    path_source_base = '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_20_22_51'
    path_para9_base = s_scaling_paths['Baseline (Para9)']['7B'] # Same as above
    path_tb_base = s_strat_paths['Textbooks']
    
    # Load Baseline Values
    # Source
    base_source_f = get_final_val(path_source_base, 'knowledge', domains, project_root)
    base_source_c = get_final_val(path_source_base, 'inference', domains, project_root)
    # Para 9
    base_para9_f = base_f # Already loaded
    base_para9_c = base_c # Already loaded
    # Textbook
    base_tb_f = get_final_val(path_tb_base, 'knowledge', domains, project_root)
    base_tb_c = get_final_val(path_tb_base, 'inference', domains, project_root)
    
    s_shuf_paths = {
        # Source Group
        'Source Word Shuf': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_21_11_30_shuffle_words',
        'Source Sent Shuf': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_21_11_31_shuffle_sentences',
        
        # Para 9 Group
        'Para 9 Word Shuf': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_21_11_30_shuffle_words',
        'Para 9 Sent Shuf': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_21_11_31_shuffle_sentences',
        
        # Textbook Group
        'TB Word Shuf': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/para9_expl_shuffled_words_textbook_cyclefull/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_21_02_11',
        'TB Sent Shuf': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/para9_expl_shuffled_sentences_textbook_cyclefull/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_21_02_11',
    }
    
    s_shuf_data = {'labels': [], 'f': [], 'c': []}
    order_shuf = [
        'Source Word Shuf', 'Source Sent Shuf',
        'Para 9 Word Shuf', 'Para 9 Sent Shuf',
        'TB Word Shuf', 'TB Sent Shuf'
    ]
    
    for label in order_shuf:
        path = s_shuf_paths[label]
        val_f = get_final_val(path, 'knowledge', domains, project_root)
        val_c = get_final_val(path, 'inference', domains, project_root)
        
        # Select Baseline
        if 'Source' in label:
            curr_base_f, curr_base_c = base_source_f, base_source_c
        elif 'Para 9' in label:
            curr_base_f, curr_base_c = base_para9_f, base_para9_c
        elif 'TB' in label:
            curr_base_f, curr_base_c = base_tb_f, base_tb_c
        else:
            curr_base_f, curr_base_c = None, None
            
        # Store RELATIVE DELTA
        s_shuf_data['labels'].append(label)
        s_shuf_data['f'].append(val_f - curr_base_f if val_f is not None and curr_base_f is not None else None)
        s_shuf_data['c'].append(val_c - curr_base_c if val_c is not None and curr_base_c is not None else None)


    # ==========================================
    # PLOTTING
    # ==========================================
    print("Generating plot...")
    # Layout: Strategies (Hist) | Scaling (Lines) | Shuffling (Hist)
    fig = plt.figure(figsize=(20, 6))
    gs = GridSpec(1, 3, width_ratios=[1.2, 1.5, 1.5], figure=fig, wspace=0.25)
    ax1 = fig.add_subplot(gs[0]) # Strategies
    ax2 = fig.add_subplot(gs[1]) # Scaling
    ax3 = fig.add_subplot(gs[2]) # Shuffling

    # --- Colors ---
    colors = {
        'Textbooks': '#8c564b',      # Brown
        'StackExchange': '#d4ac0d',  # Dark Yellow/Gold
        'Blogs': '#9467bd',          # Purple
        'Aux Views': '#2ca02c',      # Green
        'Baseline (Para9)': '#ff7f0e', # Orange
        'Corrupted Aux': '#d62728',    # Red
        
        'Source Word Shuf': '#1f77b4', # Blue (Source Based)
        'Source Sent Shuf': '#1f77b4', # Blue
        'Para 9 Word Shuf': '#ff7f0e', # Orange (Para 9 Based)
        'Para 9 Sent Shuf': '#ff7f0e', # Orange
        'TB Word Shuf': '#8c564b',     # Brown (TB Based)
        'TB Sent Shuf': '#8c564b',     # Brown
    }

    # --- SUBPLOT 1: Strategies (Histogram) ---
    def plot_delta_hist(ax, data, title, color_map, ylabel, xtick_labels=None):
        x = np.arange(len(data['labels']))
        width = 0.35
        
        # Get colors for each bar
        bar_colors = [color_map.get(l, 'gray') for l in data['labels']]
        
        f_vals = [v if v is not None else 0 for v in data['f']]
        c_vals = [v if v is not None else 0 for v in data['c']]
        
        ax.bar(x - width/2, f_vals, width, label='Factual', color=bar_colors, alpha=0.8)
        ax.bar(x + width/2, c_vals, width, label='Compositional', color=bar_colors, hatch='//', alpha=0.5, edgecolor='black')
        
        ax.set_xticks(x)
        if xtick_labels:
            ax.set_xticklabels(xtick_labels, rotation=25, ha='right')
        else:
            ax.set_xticklabels(data['labels'], rotation=25, ha='right')
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(0, color='black', linewidth=0.8)

    plot_delta_hist(ax1, s_strat_data, "Data Strategies (7B)", colors, r"$\Delta$ Final Log Prob. (Target - Para. 9)")
    
    # --- SUBPLOT 2: Model Scaling (Lines) ---
    x_vals = np.arange(len(models))
    for method, data in s_scaling_data.items():
        c = colors.get(method, 'gray')
        # Factual (solid)
        f_data = [d if d is not None else np.nan for d in data['factual']]
        if not all(np.isnan(f_data)):
            ax2.plot(x_vals, f_data, label=f"{method}", color=c, marker='o', lw=2, linestyle='-')
        
        # Compositional (dotted)
        c_data = [d if d is not None else np.nan for d in data['compositional']]
        if not all(np.isnan(c_data)):
            ax2.plot(x_vals, c_data, color=c, linestyle=':', marker='o', lw=2)

    ax2.set_xticks(x_vals)
    ax2.set_xticklabels(models)
    ax2.set_title("Model Scaling")
    ax2.set_ylabel("Final Log Prob.")
    ax2.set_xlabel("Model Size")
    
    # Legend for Subplot 2
    legend_elements_s2 = [
        Line2D([0], [0], color='#ff7f0e', lw=2, label='Baseline (Para9)'),
        Line2D([0], [0], color='#2ca02c', lw=2, label='Aux Views'),
        Line2D([0], [0], color='#d62728', lw=2, label='Corrupted Aux'),
        Line2D([0], [0], color='gray', lw=2, linestyle='-', label='Factual'),
        Line2D([0], [0], color='gray', lw=2, linestyle=':', label='Compositional'),
    ]
    ax2.legend(handles=legend_elements_s2, loc='lower right', fontsize='small')
    ax2.grid(True, alpha=0.3)


    # --- SUBPLOT 3: Shuffling Effect (Histogram) ---
    # Simplified labels: "Word", "Sentence" repeated
    simplified_labels = ['Word', 'Sentence', 'Word', 'Sentence', 'Word', 'Sentence']
    plot_delta_hist(ax3, s_shuf_data, "Shuffling Effect (7B)", colors, r"$\Delta$ Final Log Prob. (Shuffled - Unshuffled)", xtick_labels=simplified_labels)

    # Legend for Subplot 3 (Colors + Hatching)
    legend_elements_s3 = [
        mpatches.Patch(facecolor='#1f77b4', alpha=0.8, label='Source Based'),
        mpatches.Patch(facecolor='#ff7f0e', alpha=0.8, label='Para 9 Based'),
        mpatches.Patch(facecolor='#8c564b', alpha=0.8, label='Textbook Based'),
        mpatches.Patch(facecolor='gray', alpha=0.8, label='Factual'),
        mpatches.Patch(facecolor='gray', alpha=0.5, hatch='//', edgecolor='black', label='Compositional')
    ]
    ax3.legend(handles=legend_elements_s3, loc='lower right', fontsize='small')

    # Auto-scale Y-limits for Histograms (Delta)
    all_deltas = []
    for d in [s_strat_data, s_shuf_data]:
        all_deltas.extend([x for x in d['f'] if x is not None])
        all_deltas.extend([x for x in d['c'] if x is not None])
    
    # Include 0 in the range
    all_deltas.append(0)
    
    min_y = min(all_deltas)
    max_y = max(all_deltas)
    
    # Add padding
    range_y = max_y - min_y
    padding = range_y * 0.1 if range_y > 0 else 0.1
    
    ylim_delta = (min_y - padding, max_y + padding)
    
    # Set limits
    # Subplot 1: Force bottom to 0
    s1_deltas = [x for x in s_strat_data['f'] if x is not None] + [x for x in s_strat_data['c'] if x is not None] + [0]
    s1_max = max(s1_deltas)
    s1_pad = s1_max * 0.1 if s1_max > 0 else 0.1
    ax1.set_ylim(bottom=0, top=s1_max + s1_pad)
    
    # Subplot 3: Use full range (can be negative)
    ax3.set_ylim(ylim_delta)

    output_dir = 'plots'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'aux_views_comparison.pdf')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
