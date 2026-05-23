import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib.lines import Line2D
from transformers import AutoTokenizer

# Adjust the path to include the utils directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils.llm_plotting import set_plot_style

def load_and_aggregate_data(run_path, probe_type, domains):
    """
    Aggregates probe data across multiple domains from a specific run path.
    """
    all_domain_dfs = []
    for domain in domains:
        if probe_type == "knowledge":
            probe_dir = f"{domain}_knowledge_probe"
            file_name = f"{domain}_knowledge_probe_metrics.csv"
        else:  # inference
            probe_dir = f"{domain}_inference_probe"
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
                    all_domain_dfs.append(df)
            else:
                print(f"Warning: 'step' or 'log_prob' column not found in {metrics_path}. Skipping.")
    
    if not all_domain_dfs:
        return pd.DataFrame()

    combined_df = pd.concat(all_domain_dfs, ignore_index=True)
    return combined_df

def get_total_tokens(strategies, domains, project_root, tokenizer):
    """
    Calculates the total number of tokens for each explanation strategy.
    """
    token_counts = {}
    base_path = os.path.join(project_root, 'data/arxiv/explanations')

    for strategy_label, files in strategies.items():
        total_tokens = 0
        for domain in domains:
            for file_name in files:
                file_path = os.path.join(base_path, domain, file_name)
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        total_tokens += len(tokenizer.encode(content))
        token_counts[strategy_label] = total_tokens
    
    # Handle derived strategies
    if 'Blogs' in token_counts and 'Stack Exchange' in token_counts and 'Textbook' in token_counts:
        token_counts['All'] = token_counts['Blogs'] + token_counts['Stack Exchange'] + token_counts['Textbook']
    
    if 'Textbook' in token_counts:
        token_counts['Textbook x2'] = token_counts['Textbook']

    token_counts['Para. Only'] = 0

    return token_counts

def main():
    """
    Main function to generate the auxiliary view ablation plot.
    """
    # --- Configuration ---
    color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    experiments = [
        {
            "path": "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/para9_expl/sep_1_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e60/bs32_lr2e-05/overlap_1_4/09_24_04_18",
            "label": "All"
        },
        {
            "path": "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/para9_expl_blogs/sep_1_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e60/bs32_lr2e-05/overlap_1_4/09_23_22_12",
            "label": "Blogs"
        },
        {
            "path": "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/para9_expl_stackexchange/sep_1_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e60/bs32_lr2e-05/overlap_1_4/09_24_04_11",
            "label": "Stack Exchange"
        },
        {
            "path": "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/para9_expl_textbook/sep_1_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e60/bs32_lr2e-05/overlap_1_4/09_24_00_57",
            "label": "Textbook"
        },
        {
            "path": "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/para9_expl_textbook_x2/sep_1_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e60/bs32_lr2e-05/overlap_1_4/09_23_16_14",
            "label": "Textbook x2"
        },
        {
            "path": "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/para9/sep_1_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e50/bs32_lr2e-05/run",
            "label": "Para. Only"
        }
    ]

    domains = ['1_58', 'DPO', 'GRPO', 'BOFT', 'OFT', 'QLoRA']
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

    print("Loading OLMo tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-1124-7B")
    print("Tokenizer loaded.")

    explanation_strategies = {
        "Blogs": ["blogs.txt"],
        "Stack Exchange": ["stackexchange.txt"],
        "Textbook": ["textbook.txt"],
    }
    
    token_counts = get_total_tokens(explanation_strategies, domains, project_root, tokenizer)
    
    # --- Data Loading ---
    all_data = {
        "Factual": {},
        "Compositional": {}
    }

    for exp in experiments:
        label = exp['label']
        path = exp['path']
        print(f"Loading data for {label}...")
        
        knowledge_df = load_and_aggregate_data(path, 'knowledge', domains)
        if not knowledge_df.empty:
            all_data["Factual"][label] = knowledge_df
        
        inference_df = load_and_aggregate_data(path, 'inference', domains)
        if not inference_df.empty:
            all_data["Compositional"][label] = inference_df
            
    # --- Plotting ---
    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    
    # Add shading for evaluation steps
    max_step = 0
    for data_dict in all_data.values():
        for df in data_dict.values():
            if not df.empty:
                max_step = max(max_step, df['step'].max())
    
    if max_step > 0:
        vline_steps = np.arange(30, max_step + 1, 40)
        for step in vline_steps:
            start_shade = step - 5
            if start_shade < 0:
                continue
            for ax in axes:
                ax.axvline(x=start_shade, color='grey', linestyle='--', linewidth=1, alpha=0.7, zorder=0)
                ax.axvline(x=step, color='grey', linestyle='--', linewidth=1, alpha=0.7, zorder=0)
                ax.axvspan(start_shade, step, color='grey', alpha=0.2, hatch='/', zorder=0)

    # Factual Probes
    ax = axes[0]
    ax.set_prop_cycle(cycler(color=color_palette))
    for label, df in all_data["Factual"].items():
        plot_df = df.groupby('step')['log_prob'].mean().reset_index()
        ax.plot(plot_df['step'], plot_df['log_prob'], label=label, lw=1.65)
    ax.set_title('7B: Factual Probes')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Mean Log Probability')
    ax.grid(False)

    # Compositional Probes
    ax = axes[1]
    ax.set_prop_cycle(cycler(color=color_palette))
    for label, df in all_data["Compositional"].items():
        plot_df = df.groupby('step')['log_prob'].mean().reset_index()
        ax.plot(plot_df['step'], plot_df['log_prob'], label=label, lw=1.65)
    ax.set_title('7B: Compositional Probes')
    ax.set_xlabel('Training Step')
    ax.grid(False)

    # Create a unified legend with token counts
    handles, labels = axes[0].get_legend_handles_labels()
    new_labels = []
    for label in labels:
        if label in token_counts:
            count = token_counts[label]
            if label == 'Textbook x2':
                count *= 2  # This experiment used the textbook explanations twice
            
            if count >= 1_000_000:
                count_str = f"{count/1_000_000:.1f}M"
            else:
                count_str = f"{count/1_000:.1f}K"
            new_labels.append(f"{label} ({count_str} tokens)")
        else:
            new_labels.append(label)

    fig.legend(handles, new_labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.08))

    fig.tight_layout(rect=[0, 0.1, 1, 0.95])

    output_dir = 'plots'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'auxiliary_view_ablation.pdf')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
