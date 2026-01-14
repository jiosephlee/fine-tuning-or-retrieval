import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import argparse
import json
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

def get_avg_token_counts(json_path):
    """
    Parses reports/token_lengths_summary.json to get average token counts
    for blogs, stackexchange, textbooks, and aux views.
    """
    if not os.path.exists(json_path):
        print(f"Warning: Token summary file not found at {json_path}")
        return None
    
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    papers = data.get('papers', {})
    
    # Accumulate averages
    blogs_avgs = []
    se_avgs = []
    tb_avgs = []
    
    for domain, details in papers.items():
        expl = details.get('explanations', {})
        if 'blogs' in expl:
            blogs_avgs.append(expl['blogs']['avg'])
        if 'stackexchange' in expl:
            se_avgs.append(expl['stackexchange']['avg'])
        if 'textbooks' in expl:
            tb_avgs.append(expl['textbooks']['avg'])
            
    # Compute global averages
    avg_blogs = np.mean(blogs_avgs) if blogs_avgs else 1.0
    avg_se = np.mean(se_avgs) if se_avgs else 1.0
    avg_tb = np.mean(tb_avgs) if tb_avgs else 1.0
    
    # Aux Views is the sum of the three (since the strategy uses all three)
    avg_aux = avg_blogs + avg_se + avg_tb
    
    return {
        'Blogs': avg_blogs,
        'StackExchange': avg_se,
        'Textbooks': avg_tb,
        'Aux Views': avg_aux
    }

def load_filtered_series(run_path, probe_type, domains, project_root):
    """
    Loads probe series and filters out 'Source Only' probes.
    """
    if not run_path:
        return None
        
    all_dfs = []
    # Map probe_type to folder/file names
    if probe_type == "knowledge":
        probe_dir_suffix = "_knowledge_probe"
        file_suffix = "_knowledge_probe_metrics.csv"
        probe_folder = 'facts'
    else: # inference
        probe_dir_suffix = "_inference_probe"
        file_suffix = "_inference_probe_metrics.csv"
        probe_folder = 'inference'

    resolved_path = run_path # Assuming path is already resolved or direct
    
    for domain in domains:
        probe_dir = f"{domain}{probe_dir_suffix}"
        file_name = f"{domain}{file_suffix}"
        metrics_path = os.path.join(resolved_path, probe_dir, file_name)
        
        if os.path.exists(metrics_path) and os.path.getsize(metrics_path) > 0:
            try:
                df = pd.read_csv(metrics_path)
            except Exception:
                continue

            if 'step' not in df.columns or 'log_prob' not in df.columns:
                continue

            # Load Filter
            filter_path = os.path.join(project_root, 'data/probes', probe_folder, domain, 'filter.json')
            if os.path.exists(filter_path):
                with open(filter_path, 'r') as f:
                    filter_data = json.load(f)
                source_only_indices = filter_data.get('in_source_only', [])
                
                # Filter out Source Only
                df = df[~df['probe_index'].isin(source_only_indices)]
            
            df['step'] = pd.to_numeric(df['step'], errors='coerce')
            df['log_prob'] = pd.to_numeric(df['log_prob'], errors='coerce')
            df.dropna(subset=['step', 'log_prob'], inplace=True)
            
            if not df.empty:
                all_dfs.append(df)

    if not all_dfs:
        return None

    combined = pd.concat(all_dfs, ignore_index=True)
    agg = combined.groupby('step')['log_prob'].mean().reset_index().sort_values('step')
    return agg


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
    
    # --- Colors ---
    colors = {
        'Textbooks': '#8c564b',      # Brown
        'StackExchange': '#d4ac0d',  # Dark Yellow/Gold
        'Blogs': '#9467bd',          # Purple
        'Aux Views': '#2ca02c',      # Green
        'Baseline (Para9)': '#ff7f0e', # Orange
        'Corrupted Aux': '#d62728',    # Red
        'Source': '#1f77b4',           # Blue
        'Para 9': '#ff7f0e',           # Orange
    }

    # --- 1. Step-by-Step Comparison (32B) ---
    print("Gathering data for Step-by-Step Comparison...")
    # Paths for 32B BS64
    path_32b_source = '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/allenai_OLMo-2-0325-32B/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_21_06_14'
    path_32b_para9 = '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/allenai_OLMo-2-0325-32B/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_21_06_44'
    path_32b_aux = '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/allenai_OLMo-2-0325-32B/probes_v9/newline2/para9_expl_textbooks+blogs+stackexchange_cyclefull/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_26_04_11/'

    s1_data = {
        'Source': {'path': path_32b_source, 'color': colors['Source']},
        'Para 9': {'path': path_32b_para9, 'color': colors['Para 9']},
        'Aux Views': {'path': path_32b_aux, 'color': colors['Aux Views']}
    }
    
    for label, info in s1_data.items():
        info['f_series'] = load_filtered_series(info['path'], 'knowledge', domains, project_root)
        info['c_series'] = load_filtered_series(info['path'], 'inference', domains, project_root)

    
    # --- 2. Strategies Data (Normalized Delta Values) ---
    print("Gathering data for Strategies...")
    s_strat_paths = {
        'Textbooks': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/para9_expl_textbooks_cyclefull/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_21_02_11/',
        'StackExchange': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/para9_expl_stackexchange_cyclefull/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_21_02_11/',
        'Blogs': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/para9_expl_blogs_cyclefull/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_21_02_11/',
        'Aux Views': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/para9_expl_stackexchange+blogs+textbooks_cyclefull/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_21_04_43/'
    }
    
    # Token Counts for Normalization
    token_counts = get_avg_token_counts(os.path.join(project_root, 'reports/token_lengths_summary.json'))
    
    # Calculate Baseline (Para 9 7B) for Delta
    # Note: Using the same baseline path as before
    baseline_path_7b = '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_20_22_51'
    base_f = get_final_val(baseline_path_7b, 'knowledge', domains, project_root)
    base_c = get_final_val(baseline_path_7b, 'inference', domains, project_root)
    
    s_strat_data = {'labels': [], 'f': [], 'c': []}
    order_strat = ['Textbooks', 'StackExchange', 'Blogs']
    
    for label in order_strat:
        path = s_strat_paths[label]
        val_f = get_final_val(path, 'knowledge', domains, project_root)
        val_c = get_final_val(path, 'inference', domains, project_root)
        
        # Calculate Delta
        delta_f = val_f - base_f if val_f is not None and base_f is not None else 0
        delta_c = val_c - base_c if val_c is not None and base_c is not None else 0
        
        # Normalize (per 1K tokens)
        norm_factor = token_counts.get(label, 1.0)
        # Avoid division by zero
        if norm_factor == 0: norm_factor = 1.0
        
        s_strat_data['labels'].append(label)
        s_strat_data['f'].append((delta_f / norm_factor) * 1000)
        s_strat_data['c'].append((delta_c / norm_factor) * 1000)


    # --- 3. Model Scaling Data (Delta vs Para 9) ---
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
             '1B': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/1b/probes_v9/newline2/para9_expl_textbooks+blogs+stackexchange_cyclefull/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_26_10_05', 
             '7B': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/para9_expl_stackexchange+blogs+textbooks_cyclefull/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_21_04_43/',
             '13B': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/allenai_OLMo-2-1124-13B/probes_v9/newline2/para9_expl_textbooks+blogs+stackexchange_cyclefull/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_26_07_58', 
             '32B': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/allenai_OLMo-2-0325-32B/probes_v9/newline2/para9_expl_textbooks+blogs+stackexchange_cyclefull/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_26_04_11/',
        },
        'Corrupted Aux': {
            '1B': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/1b/probes_v9/newline2/para9_expl_fruit_textbooks_v3+fruit_blogs_v3+fruit_stackexchange_v3_cyclefull/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_24_10_47',
            '7B': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/para9_expl_fruit_textbooks_v3+fruit_blogs_v3+fruit_stackexchange_v3_cyclefull/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_24_02_47/',
            '13B': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/allenai_OLMo-2-1124-13B/probes_v9/newline2/para9_expl_fruit_textbooks_v3+fruit_blogs_v3+fruit_stackexchange_v3_cyclefull/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_24_08_50',
            '32B': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/allenai_OLMo-2-0325-32B/probes_v9/newline2/para9_expl_fruit_textbooks_v3+fruit_blogs_v3+fruit_stackexchange_v3_cyclefull/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_24_05_10/'
        }
    }

    s_scaling_data = {}
    # We only want Aux Views and Corrupted Aux, plotted as delta against Para 9
    target_methods = ['Aux Views', 'Corrupted Aux']
    
    for method in target_methods:
        s_scaling_data[method] = {'factual': [], 'compositional': []}
        for model in models:
            # Get Target Value
            path = s_scaling_paths[method].get(model)
            val_k = get_final_val(path, 'knowledge', domains, project_root)
            val_i = get_final_val(path, 'inference', domains, project_root)
            
            # Get Baseline Value (Para 9)
            base_path = s_scaling_paths['Baseline (Para9)'].get(model)
            base_k = get_final_val(base_path, 'knowledge', domains, project_root)
            base_i = get_final_val(base_path, 'inference', domains, project_root)
            
            # Calculate Delta
            delta_k = val_k - base_k if val_k is not None and base_k is not None else None
            delta_i = val_i - base_i if val_i is not None and base_i is not None else None
            
            s_scaling_data[method]['factual'].append(delta_k)
            s_scaling_data[method]['compositional'].append(delta_i)


    # ==========================================
    # PLOTTING
    # ==========================================
    print("Generating plot...")
    # Layout: Step-by-Step (Lines) | Strategies (Hist) | Scaling (Lines)
    fig = plt.figure(figsize=(20, 6))
    gs = GridSpec(1, 3, width_ratios=[1.2, 1.2, 1.5], figure=fig, wspace=0.25)
    ax1 = fig.add_subplot(gs[0]) # Step-by-Step
    ax2 = fig.add_subplot(gs[1]) # Strategies
    ax3 = fig.add_subplot(gs[2]) # Scaling

    # --- Colors ---
    # Already defined above

    # --- SUBPLOT 1: Step-by-Step Comparison (32B) ---
    for label, info in s1_data.items():
        if info['f_series'] is not None:
            ax1.plot(info['f_series']['step'], info['f_series']['log_prob'], color=info['color'], linestyle='-', lw=2, label=f"{label} (Factual)")
        if info['c_series'] is not None:
            ax1.plot(info['c_series']['step'], info['c_series']['log_prob'], color=info['color'], linestyle=':', lw=2, label=f"{label} (Comp.)")
            
    ax1.set_title("32B: Source-Only Probes")
    ax1.set_xlabel("Exposure #")
    ax1.set_ylabel("Log Prob.")
    ax1.grid(True, alpha=0.3)
    
    # Legend for Subplot 1
    legend_elements_s1 = [
        Line2D([0], [0], color=colors['Source'], lw=2, label='Source'),
        Line2D([0], [0], color=colors['Para 9'], lw=2, label='Para 9'),
        Line2D([0], [0], color=colors['Aux Views'], lw=2, label='Aux Views'),
        Line2D([0], [0], color='gray', lw=2, linestyle='-', label='Factual'),
        Line2D([0], [0], color='gray', lw=2, linestyle=':', label='Compositional'),
    ]
    ax1.legend(handles=legend_elements_s1, loc='lower right', fontsize='small')


    # --- SUBPLOT 2: Strategies (Normalized Histogram) ---
    def plot_delta_hist(ax, data, title, color_map, ylabel, xtick_labels=None):
        x = np.arange(len(data['labels']))
        width = 0.3 # Thinner bars
        
        # Get colors for each bar
        bar_colors = [color_map.get(l, 'gray') for l in data['labels']]
        
        f_vals = [v if v is not None else 0 for v in data['f']]
        c_vals = [v if v is not None else 0 for v in data['c']]
        
        # Lighter colors (alpha=0.6)
        ax.bar(x - width/2, f_vals, width, label='Factual', color=bar_colors, alpha=0.6)
        ax.bar(x + width/2, c_vals, width, label='Compositional', color=bar_colors, hatch='//', alpha=0.4, edgecolor='black')
        
        ax.set_xticks(x)
        if xtick_labels:
            ax.set_xticklabels(xtick_labels, rotation=25, ha='right')
        else:
            ax.set_xticklabels(data['labels'], rotation=25, ha='right')
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(0, color='black', linewidth=0.8)

    plot_delta_hist(ax2, s_strat_data, "Data Strategies (7B)", colors, r"$\Delta$ Final Log Prob / 1K Tokens")
    
    # Legend for Subplot 2
    legend_elements_s2 = [
        mpatches.Patch(facecolor='gray', alpha=0.6, label='Factual'),
        mpatches.Patch(facecolor='gray', alpha=0.4, hatch='//', edgecolor='black', label='Compositional')
    ]
    ax2.legend(handles=legend_elements_s2, loc='upper right', fontsize='small')
    
    
    # --- SUBPLOT 3: Model Scaling (Delta Lines) ---
    x_vals = np.arange(len(models))
    for method, data in s_scaling_data.items():
        c = colors.get(method, 'gray')
        # Factual (solid)
        f_data = [d if d is not None else np.nan for d in data['factual']]
        if not all(np.isnan(f_data)):
            ax3.plot(x_vals, f_data, label=f"{method}", color=c, marker='o', lw=2, linestyle='-')
        
        # Compositional (dotted)
        c_data = [d if d is not None else np.nan for d in data['compositional']]
        if not all(np.isnan(c_data)):
            ax3.plot(x_vals, c_data, color=c, linestyle=':', marker='o', lw=2)

    ax3.set_xticks(x_vals)
    ax3.set_xticklabels(models)
    ax3.set_title("Corrupted Aux. Views")
    ax3.set_ylabel(r"$\Delta$ Final Log Prob. (Target - Para. 9)")
    ax3.set_xlabel("Model Size")
    ax3.axhline(0, color='black', linewidth=0.8, linestyle='--')
    
    # Legend for Subplot 3
    legend_elements_s3 = [
        Line2D([0], [0], color=colors['Aux Views'], lw=2, label='Aux Views'),
        Line2D([0], [0], color=colors['Corrupted Aux'], lw=2, label='Corrupted Aux'),
        Line2D([0], [0], color='gray', lw=2, linestyle='-', label='Factual'),
        Line2D([0], [0], color='gray', lw=2, linestyle=':', label='Compositional'),
    ]
    ax3.legend(handles=legend_elements_s3, loc='lower right', fontsize='small')
    ax3.grid(True, alpha=0.3)

    # Auto-scale Y-limits for Histogram (Subplot 2)
    s2_deltas = [x for x in s_strat_data['f'] if x is not None] + [x for x in s_strat_data['c'] if x is not None] + [0]
    s2_max = max(s2_deltas)
    # Manually increase by 0.1
    ax2.set_ylim(bottom=0, top=s2_max + 0.1)

    output_dir = 'plots'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'aux_views_comparison.pdf')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
