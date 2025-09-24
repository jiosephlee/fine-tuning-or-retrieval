import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import textwrap

# Adjust path to import from other scripts/utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from scripts.plotting.plot_comparison import find_latest_run_path, aggregate_across_domains
from utils.llm_plotting import set_plot_style

def get_final_step_performance(run_path, domains, project_root):
    """
    Aggregates probe data and gets the performance at the last step for each probe.
    """
    # Get both knowledge and inference probes
    knowledge_df = aggregate_across_domains(run_path, 'knowledge', domains, project_root=project_root)
    inference_df = aggregate_across_domains(run_path, 'inference', domains, project_root=project_root)

    # Add probe type identifier
    if not knowledge_df.empty:
        knowledge_df['probe_type'] = 'knowledge'
    if not inference_df.empty:
        inference_df['probe_type'] = 'inference'
    
    combined_df = pd.concat([knowledge_df, inference_df], ignore_index=True)

    if combined_df.empty:
        return pd.DataFrame()

    # Get the last step for each probe
    # Using .loc[...idxmax()] is a reliable way to get the whole row for the max value in a group
    last_step_df = combined_df.loc[combined_df.groupby(['domain', 'probe_index', 'probe_type'])['step'].idxmax()]
    
    return last_step_df

def plot_delta_distribution(ax, data, color):
    """Helper function to plot a delta distribution with box and strip plots."""
    # Standard boxplot where whiskers show 1.5 * IQR
    sns.boxplot(ax=ax, x='log_prob_delta', data=data, color=color, width=0.4)
    sns.stripplot(ax=ax, x='log_prob_delta', data=data, jitter=True, alpha=0.4, color='black', size=3)
    ax.axvline(0, color='red', linestyle='--', linewidth=1.5)
    
    if not data.empty:
        median_val = data['log_prob_delta'].median()
        ax.text(median_val, -0.4, f'Median: {median_val:.2f}', 
                horizontalalignment='center', size='small', color='black', weight='semibold')
    
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_yticks([])
    ax.grid(True, axis='x', linestyle='--', alpha=0.6)

def generate_outlier_report(data, comparison_name, probe_type, output_dir, project_root, top_n=10):
    """Generates a report of the top N positive outliers with probe text."""
    if data.empty:
        return

    Q1 = data['log_prob_delta'].quantile(0.25)
    Q3 = data['log_prob_delta'].quantile(0.75)
    IQR = Q3 - Q1
    outlier_threshold = Q3 + 1.5 * IQR

    # Filter for positive outliers
    positive_outliers = data[data['log_prob_delta'] > outlier_threshold].copy()

    if positive_outliers.empty:
        report_content = "No significant positive outliers found."
    else:
        # --- Load Probe Text ---
        all_probes_list = []
        for domain in positive_outliers['domain'].unique():
            probe_file_path = ''
            if probe_type == 'factual':
                probe_file_path = os.path.join(project_root, 'data', 'probes', 'facts', domain, 'probes_v9.csv')
            elif probe_type == 'compositional':
                probe_file_path = os.path.join(project_root, 'data', 'probes', 'inference', domain, 'probes_v6.csv')
            
            if os.path.exists(probe_file_path):
                domain_probes_df = pd.read_csv(probe_file_path)
                domain_probes_df['domain'] = domain
                domain_probes_df['probe_index'] = domain_probes_df.index
                all_probes_list.append(domain_probes_df)
        
        if all_probes_list:
            all_probes_df = pd.concat(all_probes_list, ignore_index=True)
            positive_outliers = pd.merge(
                positive_outliers,
                all_probes_df[['domain', 'probe_index', 'probe', 'target']],
                on=['domain', 'probe_index'],
                how='left'
            )
        else:
            print(f"Warning: Could not find any probe text files for {probe_type} probes.")
            positive_outliers['probe'] = 'N/A'
            positive_outliers['target'] = 'N/A'
            
        # Sort by the delta to get the top outliers
        positive_outliers.sort_values(by='log_prob_delta', ascending=False, inplace=True)
        
        report_content = f"Top {top_n} Positive Outliers for {probe_type.capitalize()} Probes ({comparison_name})\n"
        report_content += "=" * (len(report_content) - 1) + "\n\n"
        report_content += "An outlier is defined as any probe with a delta > Q3 + 1.5 * IQR.\n"
        report_content += f"Q3: {Q3:.2f}, IQR: {IQR:.2f}, Threshold: > {outlier_threshold:.2f}\n"
        
        report_cols = ['domain', 'probe_index', 'log_prob_delta', 'probe', 'target']
        report_df = positive_outliers[report_cols].head(top_n)
        
        # Manually format the report for legibility
        report_lines = []
        for _, row in report_df.iterrows():
            report_lines.append("-" * 80)
            report_lines.append(f"Domain:         {row['domain']}")
            report_lines.append(f"Probe Index:    {row['probe_index']}")
            report_lines.append(f"Delta:          {row['log_prob_delta']:.4f}")
            
            probe_text = textwrap.fill(str(row['probe']), width=70)
            target_text = textwrap.fill(str(row['target']), width=70)
            
            report_lines.append(f"Probe:\n{textwrap.indent(probe_text, '  ')}")
            report_lines.append(f"Target:\n{textwrap.indent(target_text, '  ')}")
        
        report_content += "\n".join(report_lines)

    report_filename = f"{comparison_name}_{probe_type}_positive_outliers_report.txt"
    report_path = os.path.join(output_dir, report_filename)
    
    with open(report_path, 'w') as f:
        f.write(report_content)
    print(f"Outlier report saved to {report_path}")

def main():
    """
    Generates a plot showing the delta of log_prob between two experiment runs.
    """
    # --- Configuration ---
    parser = argparse.ArgumentParser(description="Generates a plot of log_prob deltas between two runs.")
    args = parser.parse_args()
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    
    # Define the two runs to compare
    runs_to_compare = {
        'source_only': 'results/FT/full/7b/probes_v9/newline2/source_only/sep_1_dclm',
        'para9': 'results/FT/full/7b/probes_v9/newline2/para9/sep_1_dclm',
        'para9_expl': 'results/FT/full/7b/probes_v9/newline2/para9_expl/sep_1_dclm'
    }

    # Get domains
    domains_path = os.path.join(project_root, 'data/arxiv/cleaned')
    try:
        domains = sorted([os.path.splitext(f)[0] for f in os.listdir(domains_path) if f.endswith('.tex') and os.path.isfile(os.path.join(domains_path, f))])
    except FileNotFoundError:
        print(f"Error: Domains path '{domains_path}' not found.")
        return

    # --- Data Aggregation ---
    final_performances = {}
    for run_name, base_path in runs_to_compare.items():
        print(f"Processing run: {run_name}")
        run_path = find_latest_run_path(base_path, model_id='7b')
        
        if not run_path or not os.path.isdir(run_path):
            print(f"  - Warning: Run path not found for {run_name}")
            continue
        
        final_performances[run_name] = get_final_step_performance(run_path, domains, project_root)

    # --- Comparison 1: para9_expl vs para9 ---
    if 'para9' in final_performances and 'para9_expl' in final_performances and not final_performances['para9'].empty and not final_performances['para9_expl'].empty:
        df_para9 = final_performances['para9']
        df_para9_expl = final_performances['para9_expl']
        merged_df1 = pd.merge(
            df_para9, df_para9_expl,
            on=['domain', 'probe_index', 'probe_type'],
            suffixes=('_para9', '_para9_expl')
        )
        merged_df1['log_prob_delta'] = merged_df1['log_prob_para9_expl'] - merged_df1['log_prob_para9']
        knowledge_deltas1 = merged_df1[merged_df1['probe_type'] == 'knowledge']
        inference_deltas1 = merged_df1[merged_df1['probe_type'] == 'inference']
    else:
        print("Missing or empty data for 'para9' or 'para9_expl'. Skipping this comparison.")
        knowledge_deltas1, inference_deltas1 = pd.DataFrame(), pd.DataFrame()

    # --- Comparison 2: para9 vs source_only ---
    if 'para9' in final_performances and 'source_only' in final_performances and not final_performances['source_only'].empty and not final_performances['para9'].empty:
        df_source_only = final_performances['source_only']
        df_para9 = final_performances['para9']
        merged_df2 = pd.merge(
            df_source_only, df_para9,
            on=['domain', 'probe_index', 'probe_type'],
            suffixes=('_source_only', '_para9')
        )
        merged_df2['log_prob_delta'] = merged_df2['log_prob_para9'] - merged_df2['log_prob_source_only']
        knowledge_deltas2 = merged_df2[merged_df2['probe_type'] == 'knowledge']
        inference_deltas2 = merged_df2[merged_df2['probe_type'] == 'inference']
    else:
        print("Missing or empty data for 'source_only' or 'para9'. Skipping this comparison.")
        knowledge_deltas2, inference_deltas2 = pd.DataFrame(), pd.DataFrame()

    # --- Plotting & Report Generation ---
    print("Generating delta plot and outlier reports...")
    set_plot_style()
    
    output_dir = 'plots'
    os.makedirs(output_dir, exist_ok=True)

    # Get domains dynamically from the data
    all_domains_df = pd.concat([knowledge_deltas1, inference_deltas1, knowledge_deltas2, inference_deltas2])
    if all_domains_df.empty or 'domain' not in all_domains_df.columns:
        print("No data available to determine domains. Exiting plot generation.")
        return
    domains = sorted(all_domains_df['domain'].unique())
    num_domains = len(domains)

    fig, axes = plt.subplots(num_domains, 4, figsize=(20, 2 * num_domains), sharex=True)
    plt.setp(axes, yticks=[]) # Hide y-ticks for all plots

    # Method names from plot_comparison.py
    method_names = {
        'para9_expl': 'Para. + Multiview',
        'para9': 'Para.',
        'source_only': 'Source'
    }

    # Column titles
    col_titles = [
        f"Factual\n({method_names['para9_expl']} vs. {method_names['para9']})",
        f"Compositional\n({method_names['para9_expl']} vs. {method_names['para9']})",
        f"Factual\n({method_names['para9']} vs. {method_names['source_only']})",
        f"Compositional\n({method_names['para9']} vs. {method_names['source_only']})"
    ]

    # Data for each column
    col_data = [
        (knowledge_deltas1, 'skyblue'),
        (inference_deltas1, 'skyblue'),
        (knowledge_deltas2, 'lightgreen'),
        (inference_deltas2, 'lightgreen')
    ]

    for i, domain in enumerate(domains):
        # Set domain name as the y-axis label for the first column
        axes[i, 0].set_ylabel(domain, rotation=0, ha='right', va='center', fontsize=10, labelpad=10)
        
        for j, (data, color) in enumerate(col_data):
            domain_data = data[data['domain'] == domain] if not data.empty else pd.DataFrame()
            plot_delta_distribution(axes[i, j], domain_data, color=color)
            
            # Hide x-tick labels for non-bottom rows to keep the plot clean
            if i < num_domains - 1:
                axes[i, j].tick_params(labelbottom=False)

    # Set column titles on the top row of plots
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=12)
    
    # Generate reports (this part doesn't need to be in the loop)
    generate_outlier_report(knowledge_deltas1, 'para9expl_vs_para9', 'factual', output_dir, project_root)
    generate_outlier_report(inference_deltas1, 'para9expl_vs_para9', 'compositional', output_dir, project_root)
    generate_outlier_report(knowledge_deltas2, 'para9_vs_source_only', 'factual', output_dir, project_root)
    generate_outlier_report(inference_deltas2, 'para9_vs_source_only', 'compositional', output_dir, project_root)
    
    # Set a tighter x-axis limit
    plt.setp(axes, xlim=(-5, 13.55))
    
    fig.supxlabel('Log Probability Delta', y=0.03)
    
    fig.tight_layout(rect=[0.05, 0.05, 1, 0.95])
    
    output_filename = 'probe_performance_delta_comparisons_7B_by_domain.pdf'
    output_path = os.path.join(output_dir, output_filename)
    
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    main()
