import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import matplotlib.patches as mpatches

# Adjust the path to include the utils directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(project_root)

# Import from utils
from utils.llm_plotting import set_plot_style
from scripts.plotting.plot_utils import (
    discover_domains,
    load_probe_series,
)

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

def main():
    parser = argparse.ArgumentParser(description="Generate shuffling effect comparison plot.")
    args = parser.parse_args()

    set_plot_style()
    domains = discover_domains(project_root)
    if not domains:
        print("No domains found in data/arxiv/cleaned. Exiting.")
        return
    print(f"Discovered domains: {domains}")

    # --- Shuffling Data (Relative Delta Values) ---
    print("Gathering data for Shuffling...")
    
    # Define Baselines for each group
    # Note: Using paths from plot_aux_views.py
    s_scaling_paths_baseline_7b = '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_20_22_51'
    path_source_base = '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_20_22_51'
    path_para9_base = s_scaling_paths_baseline_7b
    path_tb_base = '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/para9_expl_textbooks_cyclefull/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_21_02_11/'
    
    # Load Baseline Values
    # Source
    base_source_f = get_final_val(path_source_base, 'knowledge', domains, project_root)
    base_source_c = get_final_val(path_source_base, 'inference', domains, project_root)
    # Para 9
    base_para9_f = get_final_val(path_para9_base, 'knowledge', domains, project_root)
    base_para9_c = get_final_val(path_para9_base, 'inference', domains, project_root)
    # Textbook
    base_tb_f = get_final_val(path_tb_base, 'knowledge', domains, project_root)
    base_tb_c = get_final_val(path_tb_base, 'inference', domains, project_root)
    
    s_shuf_paths = {
        # Source Group
        'Source Word Shuf': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_21_11_30_shuffle_words',
        'Source Sent Shuf': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_21_11_31_shuffle_sentences',
        'Source Para Shuf': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_26_12_57_shuffle_paragraphs',
        
        # Para 9 Group
        'Para 9 Word Shuf': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_21_11_30_shuffle_words',
        'Para 9 Sent Shuf': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_21_11_31_shuffle_sentences',
        'Para 9 Para Shuf': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_26_12_57_shuffle_paragraphs',
        
        # Textbook Group
        'TB Word Shuf': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/para9_expl_shuffled_words_textbook_cyclefull/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_21_02_11',
        'TB Sent Shuf': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/para9_expl_shuffled_sentences_textbook_cyclefull/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_21_02_11',
        'TB Para Shuf': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/para9_expl_shuffled_paragraphs_textbook_cyclefull/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_26_12_57/',

        # Corrupt Textbook Group
        'Corrupt TB Word Shuf': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/para9_expl_fruit_shuffled_words_textbook_v3_cyclefull/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_26_04_11/',
        'Corrupt TB Sent Shuf': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/para9_expl_fruit_shuffled_sentences_textbook_v3_cyclefull/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_26_04_11/',
        'Corrupt TB Para Shuf': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/para9_expl_fruit_shuffled_paragraphs_textbook_v3_cyclefull/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_26_04_11/',
    }
    
    s_shuf_data = {'labels': [], 'f': [], 'c': []}
    order_shuf = [
        'Source Word Shuf', 'Source Sent Shuf', 'Source Para Shuf',
        'Para 9 Word Shuf', 'Para 9 Sent Shuf', 'Para 9 Para Shuf',
        'TB Word Shuf', 'TB Sent Shuf', 'TB Para Shuf',
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
        elif 'TB' in label and 'Corrupt' not in label:
            curr_base_f, curr_base_c = base_tb_f, base_tb_c
        elif 'Corrupt TB' in label:
            # Using Textbooks as baseline for Corrupt TB as well, to show degradation from clean
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
    fig, ax = plt.subplots(figsize=(10, 6))

    # --- Colors ---
    colors = {
        'Source Word Shuf': '#1f77b4', # Blue (Source Based)
        'Source Sent Shuf': '#1f77b4', # Blue
        'Source Para Shuf': '#1f77b4', # Blue

        'Para 9 Word Shuf': '#ff7f0e', # Orange (Para 9 Based)
        'Para 9 Sent Shuf': '#ff7f0e', # Orange
        'Para 9 Para Shuf': '#ff7f0e', # Orange

        'TB Word Shuf': '#8c564b',     # Brown (TB Based)
        'TB Sent Shuf': '#8c564b',     # Brown
        'TB Para Shuf': '#8c564b',     # Brown

        'Corrupt TB Word Shuf': '#d62728', # Red (Corrupt Based)
        'Corrupt TB Sent Shuf': '#d62728', # Red
        'Corrupt TB Para Shuf': '#d62728', # Red
    }

    # Simplified labels: "Word", "Sentence", "Paragraph" repeated
    simplified_labels = ['Word', 'Sentence', 'Paragraph'] * 3
    plot_delta_hist(ax, s_shuf_data, "Shuffling Effect (7B)", colors, r"$\Delta$ Final Log Prob. (Shuffled - Unshuffled)", xtick_labels=simplified_labels)

    # Legend (Colors + Hatching)
    legend_elements = [
        mpatches.Patch(facecolor='#1f77b4', alpha=0.8, label='Source'),
        mpatches.Patch(facecolor='#ff7f0e', alpha=0.8, label='Para 9'),
        mpatches.Patch(facecolor='#8c564b', alpha=0.8, label='Textbook'),
        mpatches.Patch(facecolor='gray', alpha=0.8, label='Factual'),
        mpatches.Patch(facecolor='gray', alpha=0.5, hatch='//', edgecolor='black', label='Compositional')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize='small')
    
    # Auto-scale Y-limits
    all_deltas = [x for x in s_shuf_data['f'] if x is not None] + [x for x in s_shuf_data['c'] if x is not None]
    all_deltas.append(0)
    min_y = min(all_deltas)
    max_y = max(all_deltas)
    range_y = max_y - min_y
    padding = range_y * 0.1 if range_y > 0 else 0.1
    ax.set_ylim(min_y - padding, max_y + padding)

    output_dir = 'plots'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'shuffling_effect.pdf')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
