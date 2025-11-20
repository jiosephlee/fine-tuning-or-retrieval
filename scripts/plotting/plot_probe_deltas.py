import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import textwrap

# Adjust path to import from other scripts/utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from scripts.plotting.plot_comparison import aggregate_across_domains
from utils.llm_plotting import set_plot_style

def get_final_step_performance(run_path, domains, project_root, source_only=False):
    """
    Aggregates probe data and gets the performance at the last step for each probe.
    """
    # Get both knowledge and inference probes
    # Enable split_probes if we need to filter by origin
    knowledge_df = aggregate_across_domains(run_path, 'knowledge', domains, split_probes=source_only, project_root=project_root)
    inference_df = aggregate_across_domains(run_path, 'inference', domains, split_probes=source_only, project_root=project_root)

    # Add probe type identifier
    if not knowledge_df.empty:
        knowledge_df['probe_type'] = 'knowledge'
    if not inference_df.empty:
        inference_df['probe_type'] = 'inference'
    
    combined_df = pd.concat([knowledge_df, inference_df], ignore_index=True)

    if combined_df.empty:
        return pd.DataFrame()

    # Filter for source-only probes if requested
    if source_only and 'origin' in combined_df.columns:
        combined_df = combined_df[combined_df['origin'] == 'Source Only']
        if combined_df.empty:
            print("Warning: No 'Source Only' probes found after filtering.")
            return pd.DataFrame()

    # Get the last step for each probe
    # Using .loc[...idxmax()] is a reliable way to get the whole row for the max value in a group
    last_step_df = combined_df.loc[combined_df.groupby(['domain', 'probe_index', 'probe_type'])['step'].idxmax()]
    
    return last_step_df

def plot_delta_distribution(ax, data, color):
    """Helper function to plot a delta distribution with box and strip plots."""
    # Check if data is empty or missing the required column
    if data.empty or 'log_prob_delta' not in data.columns:
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes, 
                fontsize=10, color='gray')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_yticks([])
        return
    
    # Standard boxplot where whiskers show 1.5 * IQR
    sns.boxplot(ax=ax, x='log_prob_delta', data=data, color=color, width=0.4, linewidth=1.6)
    sns.stripplot(ax=ax, x='log_prob_delta', data=data, jitter=True, alpha=0.4, color='black', size=3)
    ax.axvline(0, color='red', linestyle='--', linewidth=1.6)
    
    median_val = data['log_prob_delta'].median()
    ax.text(median_val, -0.4, f'Median: {median_val:.2f}', 
            horizontalalignment='center', size='small', color='black', weight='semibold')
    
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_yticks([])
    ax.grid(True, axis='x', linestyle='--', alpha=0.6)

def generate_outlier_report(data, comparison_name, probe_type, output_dir, project_root, top_n=10, top_percentile=0.25):
    """Generates a comprehensive report of top performing probes with probe text.
    
    Args:
        data: DataFrame with log_prob_delta values
        comparison_name: Name of the comparison for the report
        probe_type: 'factual' or 'compositional'
        output_dir: Directory to save the report
        project_root: Root directory of the project
        top_n: Number of outliers to show in detail
        top_percentile: Percentile of top probes to include (default 0.25 = top 25%)
    """
    if data.empty:
        return

    # Calculate statistics
    Q1 = data['log_prob_delta'].quantile(0.25)
    Q3 = data['log_prob_delta'].quantile(0.75)
    IQR = Q3 - Q1
    outlier_threshold = Q3 + 1.5 * IQR
    median_delta = data['log_prob_delta'].median()
    mean_delta = data['log_prob_delta'].mean()
    
    # Get top percentile
    percentile_threshold = data['log_prob_delta'].quantile(1 - top_percentile)
    top_percentile_probes = data[data['log_prob_delta'] >= percentile_threshold].copy()
    
    # Filter for positive outliers
    positive_outliers = data[data['log_prob_delta'] > outlier_threshold].copy()

    # --- Load Probe Text for all relevant probes ---
    all_probes_list = []
    relevant_domains = pd.concat([positive_outliers, top_percentile_probes])['domain'].unique() if not top_percentile_probes.empty else []
    
    for domain in relevant_domains:
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
        
        # Merge with outliers and top percentile
        if not positive_outliers.empty:
            positive_outliers = pd.merge(
                positive_outliers,
                all_probes_df[['domain', 'probe_index', 'probe', 'target']],
                on=['domain', 'probe_index'],
                how='left'
            )
        
        if not top_percentile_probes.empty:
            top_percentile_probes = pd.merge(
                top_percentile_probes,
                all_probes_df[['domain', 'probe_index', 'probe', 'target']],
                on=['domain', 'probe_index'],
                how='left'
            )
    else:
        print(f"Warning: Could not find any probe text files for {probe_type} probes.")
        if not positive_outliers.empty:
            positive_outliers['probe'] = 'N/A'
            positive_outliers['target'] = 'N/A'
        if not top_percentile_probes.empty:
            top_percentile_probes['probe'] = 'N/A'
            top_percentile_probes['target'] = 'N/A'
    
    # Sort both dataframes
    positive_outliers.sort_values(by='log_prob_delta', ascending=False, inplace=True)
    top_percentile_probes.sort_values(by='log_prob_delta', ascending=False, inplace=True)
    
    # --- Build Report Content ---
    report_content = f"Performance Analysis for {probe_type.capitalize()} Probes ({comparison_name})\n"
    report_content += "=" * 80 + "\n\n"
    
    # Summary Statistics
    report_content += "SUMMARY STATISTICS\n"
    report_content += "-" * 80 + "\n"
    report_content += f"Total Probes:        {len(data)}\n"
    report_content += f"Mean Delta:          {mean_delta:.4f}\n"
    report_content += f"Median Delta:        {median_delta:.4f}\n"
    report_content += f"Q1:                  {Q1:.4f}\n"
    report_content += f"Q3:                  {Q3:.4f}\n"
    report_content += f"IQR:                 {IQR:.4f}\n"
    report_content += f"Outlier Threshold:   {outlier_threshold:.4f} (Q3 + 1.5 * IQR)\n"
    report_content += f"Outliers Found:      {len(positive_outliers)}\n"
    report_content += f"Top {int(top_percentile*100)}% Threshold:    {percentile_threshold:.4f}\n"
    report_content += f"Top {int(top_percentile*100)}% Count:        {len(top_percentile_probes)}\n\n"
    
    # Section 1: Statistical Outliers
    report_content += "\n" + "=" * 80 + "\n"
    report_content += f"SECTION 1: TOP {top_n} STATISTICAL OUTLIERS\n"
    report_content += "=" * 80 + "\n\n"
    
    if positive_outliers.empty:
        report_content += "No significant positive outliers found (above Q3 + 1.5 * IQR).\n"
    else:
        report_cols = ['domain', 'probe_index', 'log_prob_delta', 'probe', 'target']
        outlier_df = positive_outliers[report_cols].head(top_n)
        
        for rank, (_, row) in enumerate(outlier_df.iterrows(), 1):
            report_content += f"\nRank #{rank}\n"
            report_content += "-" * 80 + "\n"
            report_content += f"Domain:         {row['domain']}\n"
            report_content += f"Probe Index:    {row['probe_index']}\n"
            report_content += f"Delta:          {row['log_prob_delta']:.4f}\n"
            
            probe_text = textwrap.fill(str(row['probe']), width=70)
            target_text = textwrap.fill(str(row['target']), width=70)
            
            report_content += f"Probe:\n{textwrap.indent(probe_text, '  ')}\n"
            report_content += f"Target:\n{textwrap.indent(target_text, '  ')}\n"
    
    # Section 2: Top Percentile
    report_content += "\n\n" + "=" * 80 + "\n"
    report_content += f"SECTION 2: ALL PROBES IN TOP {int(top_percentile*100)}% (RANKED)\n"
    report_content += "=" * 80 + "\n\n"
    
    if top_percentile_probes.empty:
        report_content += f"No probes found in top {int(top_percentile*100)}%.\n"
    else:
        report_cols = ['domain', 'probe_index', 'log_prob_delta', 'probe', 'target']
        percentile_df = top_percentile_probes[report_cols]
        
        for rank, (_, row) in enumerate(percentile_df.iterrows(), 1):
            report_content += f"\nRank #{rank}\n"
            report_content += "-" * 80 + "\n"
            report_content += f"Domain:         {row['domain']}\n"
            report_content += f"Probe Index:    {row['probe_index']}\n"
            report_content += f"Delta:          {row['log_prob_delta']:.4f}\n"
            
            probe_text = textwrap.fill(str(row['probe']), width=70)
            target_text = textwrap.fill(str(row['target']), width=70)
            
            report_content += f"Probe:\n{textwrap.indent(probe_text, '  ')}\n"
            report_content += f"Target:\n{textwrap.indent(target_text, '  ')}\n"

    report_filename = f"{comparison_name}_{probe_type}_performance_report.txt"
    report_path = os.path.join(output_dir, report_filename)
    
    with open(report_path, 'w') as f:
        f.write(report_content)
    print(f"Performance report saved to {report_path}")

def main():
    """
    Generates a plot showing the delta of log_prob between two experiment runs.
    """
    # --- Configuration ---
    parser = argparse.ArgumentParser(description="Generates a plot of log_prob deltas between two runs.")
    parser.add_argument(
        "--source_only",
        action='store_true',
        help="If set, only analyzes probes that appear in the source material only."
    )
    parser.add_argument(
        "--top_percentile",
        type=float,
        default=0.25,
        help="Percentile of top probes to include in detailed report (default: 0.25 for top 25%%)."
    )
    args = parser.parse_args()
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    
    # Define the two runs to compare - using explicit paths
    runs_to_compare = {
        'source_only': os.path.join(project_root, 'results/FT/full/7b/probes_v9/newline2/source_only/sep_1_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs32_lr2e-05/run'),
        'para9': os.path.join(project_root, 'results/FT/full/7b/probes_v9/newline2/para9/sep_1_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs32_lr2e-05/run'),
        'para9_expl': os.path.join(project_root, 'results/FT/full/7b/probes_v9/newline2/para9_expl/sep_1_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs32_lr2e-05/run')
    }

    # Get domains
    domains_path = os.path.join(project_root, 'data/arxiv/cleaned')
    try:
        domains = sorted([os.path.splitext(f)[0] for f in os.listdir(domains_path) if f.endswith('.tex') and os.path.isfile(os.path.join(domains_path, f))])
    except FileNotFoundError:
        print(f"Error: Domains path '{domains_path}' not found.")
        return

    # --- Data Aggregation ---
    if args.source_only:
        print("Filtering for source-only probes")
    
    final_performances = {}
    for run_name, run_path in runs_to_compare.items():
        print(f"Processing run: {run_name}")
        
        if not os.path.isdir(run_path):
            print(f"  - Warning: Run path not found for {run_name}: {run_path}")
            continue
        
        print(f"  - Using path: {run_path}")
        final_performances[run_name] = get_final_step_performance(run_path, domains, project_root, source_only=args.source_only)

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
    print("Generating delta plot and performance reports...")
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
        'para9_expl': 'Para. + Aux. Views',
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
    suffix = '_source_only' if args.source_only else ''
    generate_outlier_report(knowledge_deltas1, f'para9expl_vs_para9{suffix}', 'factual', output_dir, project_root, top_percentile=args.top_percentile)
    generate_outlier_report(inference_deltas1, f'para9expl_vs_para9{suffix}', 'compositional', output_dir, project_root, top_percentile=args.top_percentile)
    generate_outlier_report(knowledge_deltas2, f'para9_vs_source_only{suffix}', 'factual', output_dir, project_root, top_percentile=args.top_percentile)
    generate_outlier_report(inference_deltas2, f'para9_vs_source_only{suffix}', 'compositional', output_dir, project_root, top_percentile=args.top_percentile)
    
    # Set a tighter x-axis limit
    plt.setp(axes, xlim=(-5, 13.55))
    
    fig.supxlabel('Log Probability Delta', y=0.03)
    
    fig.tight_layout(rect=[0.05, 0.05, 1, 0.95])
    
    output_filename = 'probe_performance_delta_comparisons_7B_by_domain'
    if args.source_only:
        output_filename += '_source_only'
    output_filename += '.pdf'
    output_path = os.path.join(output_dir, output_filename)
    
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    main()
