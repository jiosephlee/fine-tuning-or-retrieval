import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler

# Adjust the path to include the utils directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils.llm_plotting import set_plot_style


def aggregate_across_domains(run_path, probe_type, domains):
    """
    Aggregates probe data across multiple domains from a specific run path.
    """
    all_domain_dfs = []
    for domain in domains:
        if probe_type == "knowledge":
            probe_dir = f"{domain}_knowledge_probe"
            file_name = f"{domain}_knowledge_probe_metrics.csv"
        else: # inference
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
                else:
                    continue
            else:
                print(f"Warning: 'step' or 'log_prob' column not found in {metrics_path}. Skipping.")
                continue

            df['domain'] = domain
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


def main():
    """
    Main function to generate the comparison plot.
    """
    # --- Configuration ---
    runs_to_plot = [
        ("Semi-cleaned v1", "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/source_only/sep_1_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e50/bs48_lr2e-05/overlap_1_4/semi_cleaned_v1/09_25_02_08"),
        ("Semi-cleaned v2", "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/source_only/sep_1_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e50/bs32_lr2e-05/overlap_1_4/semi_cleaned_v2/09_24_23_29"),
        ("Full cleaning", "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/source_only/sep_1_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e50/bs32_lr2e-05/run"),
    ]
    run_labels_in_order = [r[0] for r in runs_to_plot]

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    domains_path = os.path.join(project_root, 'data/arxiv/cleaned')
    try:
        domains = sorted([os.path.splitext(f)[0] for f in os.listdir(domains_path) if f.endswith('.tex') and os.path.isfile(os.path.join(domains_path, f))])
        print(f"Dynamically found domains: {domains}")
    except FileNotFoundError:
        print(f"Error: Domains path '{domains_path}' not found. Cannot determine domains.")
        domains = []
    
    if not domains:
        print("No domains found. Exiting.")
        return

    # --- Data Aggregation ---
    all_knowledge_data = []
    all_inference_data = []

    for run_label, run_path in runs_to_plot:
        print(f"Processing run: {run_label}")
        
        knowledge_df = aggregate_across_domains(run_path, 'knowledge', domains)
        if not knowledge_df.empty:
            knowledge_df['run_label'] = run_label
            all_knowledge_data.append(knowledge_df)

        inference_df = aggregate_across_domains(run_path, 'inference', domains)
        if not inference_df.empty:
            inference_df['run_label'] = run_label
            all_inference_data.append(inference_df)

    final_knowledge_df = pd.concat(all_knowledge_data, ignore_index=True) if all_knowledge_data else pd.DataFrame()
    final_inference_df = pd.concat(all_inference_data, ignore_index=True) if all_inference_data else pd.DataFrame()

    # --- Plotting ---
    print("Generating comparison plot...")
    set_plot_style()
    color_palette = ['#2ca02c', '#ff7f0e', '#1f77b4']  # Green, Orange, Blue

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Knowledge Plot
    ax_knowledge = axes[0]
    ax_knowledge.set_prop_cycle(color=color_palette)
    if not final_knowledge_df.empty:
        for run_label in run_labels_in_order:
            run_df = final_knowledge_df[final_knowledge_df['run_label'] == run_label]
            if not run_df.empty:
                plot_df = run_df.groupby('step')['log_prob'].mean().reset_index()
                ax_knowledge.plot(plot_df['step'], plot_df['log_prob'], label=run_label, lw=1.6)
    ax_knowledge.set_title(f'Factual Probes')
    ax_knowledge.set_xlabel('Training Steps')
    ax_knowledge.set_ylabel('Mean Log Probability')

    # Inference Plot
    ax_inference = axes[1]
    ax_inference.set_prop_cycle(color=color_palette)
    if not final_inference_df.empty:
        for run_label in run_labels_in_order:
            run_df = final_inference_df[final_inference_df['run_label'] == run_label]
            if not run_df.empty:
                plot_df = run_df.groupby('step')['log_prob'].mean().reset_index()
                ax_inference.plot(plot_df['step'], plot_df['log_prob'], label=run_label, lw=1.6)
    ax_inference.set_title(f'Compositional Probes')
    ax_inference.set_xlabel('Training Steps')
    
    # --- Final styling ---
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='lower center', ncol=len(run_labels_in_order), bbox_to_anchor=(0.5, -0.05), fontsize='medium')

    for ax in fig.get_axes():
        ax.grid(False)

    fig.tight_layout(rect=[0, 0.05, 1, 1])

    output_dir = 'plots'
    os.makedirs(output_dir, exist_ok=True)
    output_filename = 'cleaning_comparison_7B.pdf'
    output_path = os.path.join(output_dir, output_filename)
    
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
