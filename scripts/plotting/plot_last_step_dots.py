import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure repo root on path for imports
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(REPO_ROOT)

# Reuse styling and helpers from existing code
from utils.llm_plotting import set_plot_style  # noqa: E402
from scripts.plotting.plot_comparison import aggregate_across_domains  # noqa: E402
from scripts.plotting.plot_utils import (  # noqa: E402
    discover_domains,
    find_latest_run,
    make_subplots,
    compute_unified_ylim,
    apply_ylim,
    add_legend,
    get_final_step_value,
)
from scripts.plotting.plot_pretraining_paraphrasing_effects import (  # noqa: E402
    get_paraphrasing_data,
)


EPOCH_STEPS = [10, 25, 50, 100, 150, 200]  # Epochs to look for
STEPS = [0] + EPOCH_STEPS  # Include step 0 for baseline


def get_probe_data_for_method(run_path: str, domains, project_root: str):
    # We do not split probes here; simple aggregate across domains
    df_k = aggregate_across_domains(run_path, 'knowledge', domains, split_probes=False, project_root=project_root)
    df_i = aggregate_across_domains(run_path, 'inference', domains, split_probes=False, project_root=project_root)
    return df_k, df_i


def summarize_epoch_final_steps(epoch_paths: dict, domains, project_root: str):
    """
    For each epoch path, get the final step metrics and return as {epoch_num: {knowledge: val, inference: val}}.
    Also include step 0 from the first available epoch.
    """
    results = {}
    
    # Get step 0 from any available epoch (use the lowest epoch number that resolves)
    valid_epochs = [e for e, p in epoch_paths.items() if p]
    if valid_epochs:
        first_epoch = min(valid_epochs)
        first_base = epoch_paths[first_epoch]
        first_resolved = first_base if (first_base and os.path.isdir(first_base)) else find_latest_run(first_base)
        if first_resolved and os.path.isdir(first_resolved):
            df_k, df_i = get_probe_data_for_method(first_resolved, domains, project_root)
        
            # Extract step 0 values
            step_0_k = np.nan
            step_0_i = np.nan
            if df_k is not None and not df_k.empty:
                df_k_clean = df_k.copy()
                df_k_clean['step'] = pd.to_numeric(df_k_clean['step'], errors='coerce')
                df_k_clean['log_prob'] = pd.to_numeric(df_k_clean['log_prob'], errors='coerce')
                df_k_clean = df_k_clean.dropna(subset=['step', 'log_prob'])
                if not df_k_clean.empty:
                    step_0_vals = df_k_clean.loc[df_k_clean['step'] == 0, 'log_prob']
                    step_0_k = float(step_0_vals.mean()) if not step_0_vals.empty else np.nan
            
            if df_i is not None and not df_i.empty:
                df_i_clean = df_i.copy()
                df_i_clean['step'] = pd.to_numeric(df_i_clean['step'], errors='coerce')
                df_i_clean['log_prob'] = pd.to_numeric(df_i_clean['log_prob'], errors='coerce')
                df_i_clean = df_i_clean.dropna(subset=['step', 'log_prob'])
                if not df_i_clean.empty:
                    step_0_vals = df_i_clean.loc[df_i_clean['step'] == 0, 'log_prob']
                    step_0_i = float(step_0_vals.mean()) if not step_0_vals.empty else np.nan
            
            results[0] = {"knowledge": step_0_k, "inference": step_0_i}
    
    # Get final step values for each epoch (resolve given path or fall back to latest)
    for epoch_num, base_path in epoch_paths.items():
        if not base_path:
            continue
        resolved = base_path if os.path.isdir(base_path) else find_latest_run(base_path)
        if not resolved or not os.path.isdir(resolved):
            continue
        df_k, df_i = get_probe_data_for_method(resolved, domains, project_root)
        results[epoch_num] = {
            "knowledge": get_final_step_value(df_k),
            "inference": get_final_step_value(df_i),
        }
    
    return results


def plot_dots(ax, step_to_value: dict, label: str, color: str, marker: str):
    xs = list(step_to_value.keys())
    ys = [step_to_value[s] for s in xs]
    ax.scatter(xs, ys, label=label, color=color, marker=marker, s=50, zorder=3)


def main():
    parser = argparse.ArgumentParser(description="Plot last-step dots at selected steps for Source and Paraphrase.")
    parser.add_argument(
        "--model_id",
        type=str,
        default="7b",
        help="Model ID (e.g., '1b', '7b', 'allenai_OLMo-2-1124-13B').",
    )
    parser.add_argument(
        "--sub_dir",
        type=str,
        default="sep_1_dclm",
        help="Subdirectory under each method path.",
    )
    parser.add_argument(
        "--source_only_full_path",
        type=str,
        default=None,
        help="Absolute path to the source_only run directory for full-step (right subplot).",
    )
    parser.add_argument(
        "--para9_full_path",
        type=str,
        default=None,
        help="Absolute path to the para9 run directory for full-step (right subplot).",
    )
    parser.add_argument(
        "--source_only_dots_path",
        type=str,
        default=None,
        help="Absolute path to the source_only run directory for dots (left subplot).",
    )
    parser.add_argument(
        "--para9_dots_path",
        type=str,
        default=None,
        help="Absolute path to the para9 run directory for dots (left subplot).",
    )
    args = parser.parse_args()

    project_root = REPO_ROOT
    domains = discover_domains(project_root)
    if not domains:
        print("No domains discovered; aborting.")
        return

    # Methods to plot independently
    methods = [
        ("source_only", "Source"),
        ("para9", "Paraphrase"),
    ]

    # Colors/markers per probe type
    # (Not used in combined view; kept minimal styling below)

    # Style
    set_plot_style()

    os.makedirs('plots', exist_ok=True)

    # Collect summaries using hardcoded epoch paths (7B)
    summaries = {}
    source_epoch_paths = {
        10: "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/source_only/sep_1_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e10/bs32_lr2e-05/overlap_1_4/11_07_15_19",
        25: "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/source_only/sep_1_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e25/bs32_lr2e-05/run",
        50: "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/source_only/sep_1_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e50/bs32_lr2e-05/run",
        100: "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/source_only/sep_1_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs32_lr2e-05/run",
        150: "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/source_only/sep_1_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e150/bs32_lr2e-05/09_21_06_34",
        200: "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/source_only/sep_1_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e200/bs32_lr2e-05/overlap_1_4/11_07_15_19",
    }
    para_epoch_paths = {
        10: "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/para9/sep_1_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e10/bs32_lr2e-05/overlap_1_4/11_07_15_52",
        25: "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/para9/sep_1_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e25/bs32_lr2e-05/run",
        50: "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/para9/sep_1_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e50/bs32_lr2e-05/run",
        100: "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/para9/sep_1_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs32_lr2e-05/run",
        150: "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/para9/sep_1_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e150/bs32_lr2e-05/09_21_06_33",
        200: "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/para9/sep_1_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e200/bs32_lr2e-05/overlap_1_4/11_08_00_16",
    }

    for method_key, _ in methods:
        epoch_paths = source_epoch_paths if method_key == "source_only" else para_epoch_paths
        epoch_summaries = summarize_epoch_final_steps(epoch_paths, domains, project_root)
        summaries[method_key] = {
            "knowledge": {step: epoch_summaries.get(step, {}).get("knowledge", np.nan) for step in STEPS},
            "inference": {step: epoch_summaries.get(step, {}).get("inference", np.nan) for step in STEPS},
        }

    # Prepare 1x3 subplot: left = epoch dots, middle = full curves, right = paraphrase-level dots
    fig, axes = make_subplots(1, 3, figsize=(12, 3.6), sharey='col', hide_shared_yticks_on='right')
    ax = axes[0]

    color_map = {
        "source_only": "#1f77b4",  # blue
        "para9": "#ff7f0e",        # orange
    }

    linestyle_map = {
        "knowledge": "-",     # factual: solid
        "inference": ":",     # compositional: dotted
    }
    
    # Collect all y-values to determine unified y-axis limits
    all_y_values = []

    def plot_series(step_means: dict, color: str, linestyle: str, label: str):
        xs = STEPS
        ys = [step_means.get(s, np.nan) for s in xs]
        # Collect y-values for unified axis calculation
        valid_ys = [y for y in ys if not np.isnan(y)]
        all_y_values.extend(valid_ys)
        ax.plot(xs, ys, color=color, linestyle=linestyle, linewidth=1.0, marker='o', markersize=4, label=label)

    # Plot all four series if available
    if "source_only" in summaries:
        plot_series(summaries["source_only"]["knowledge"], color_map["source_only"], linestyle_map["knowledge"], "Source - Factual")
        plot_series(summaries["source_only"]["inference"], color_map["source_only"], linestyle_map["inference"], "Source - Compositional")

    if "para9" in summaries:
        plot_series(summaries["para9"]["knowledge"], color_map["para9"], linestyle_map["knowledge"], "Paraphrase - Factual")
        plot_series(summaries["para9"]["inference"], color_map["para9"], linestyle_map["inference"], "Paraphrase - Compositional")

    ax.set_xlabel("Total # of Exposures")
    ax.set_ylabel("Mean Log Probability")
    ax.set_xticks(STEPS)
    ax.set_xlim(min(STEPS), max(STEPS))
    ax.grid(False)

    # Middle subplot: continuous curves from explicit runs
    ax_r = axes[1]

    # Defaults for 7B if not provided
    if args.model_id.lower() == "7b":
        default_para9 = "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/para9/sep_1_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e500/bs32_lr2e-05_const/overlap_1_4/11_19_20_24"
        default_source = "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/source_only/sep_1_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e500/bs32_lr2e-05_const/overlap_1_4/11_22_16_47"
    else:
        default_para9 = None
        default_source = None

    para9_full_path = args.para9_full_path or default_para9
    source_full_path = args.source_only_full_path or default_source

    def plot_full(run_path: str, color: str, label_prefix: str):
        resolved = find_latest_run(run_path) if run_path else None
        if not resolved or not os.path.isdir(resolved):
            print(f"Full-step path missing or invalid: {run_path}")
            return 0
        df_k, df_i = get_probe_data_for_method(resolved, domains, project_root)
        max_step_local = 0
        if df_k is not None and not df_k.empty:
            gk = df_k.groupby('step')['log_prob'].mean().reset_index()
            ax_r.plot(gk['step'], gk['log_prob'], color=color, linestyle='-', linewidth=1.0, label=f"{label_prefix} - Factual")
            max_step_local = max(max_step_local, int(gk['step'].max()))
            # Collect y-values for unified axis
            all_y_values.extend(gk['log_prob'].tolist())
        if df_i is not None and not df_i.empty:
            gi = df_i.groupby('step')['log_prob'].mean().reset_index()
            ax_r.plot(gi['step'], gi['log_prob'], color=color, linestyle=':', linewidth=1.0, label=f"{label_prefix} - Compositional")
            max_step_local = max(max_step_local, int(gi['step'].max()))
            # Collect y-values for unified axis
            all_y_values.extend(gi['log_prob'].tolist())
        return max_step_local

    max_right = 0
    if source_full_path:
        max_right = max(max_right, plot_full(source_full_path, color_map["source_only"], "Source"))
    if para9_full_path:
        max_right = max(max_right, plot_full(para9_full_path, color_map["para9"], "Paraphrase"))

    ax_r.set_xlabel("Step # during training")
    # Remove y-axis title on the middle subplot
    ax_r.set_ylabel("")
    ax_r.set_yticks([]) # Hide y-tick labels on the right subplot
    ax_r.set_xlim(0, max_right)
    ax_r.grid(False)

    # Calculate unified y-axis limits for left and middle initially
    ylim = compute_unified_ylim(values=all_y_values)
    if ylim:
        apply_ylim([ax, ax_r], ylim)

    # Per-subplot legends (lower right) with simplified style guide:
    # - Orange = Paraphrase, Blue = Source
    # - Gray solid = Factual, Gray dotted = Compositional
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#ff7f0e', lw=2, linestyle='-', label='Paraphrase'),
        Line2D([0], [0], color='#1f77b4', lw=2, linestyle='-', label='Source'),
        Line2D([0], [0], color='gray', lw=2, linestyle='-', label='Factual'),
        Line2D([0], [0], color='gray', lw=2, linestyle=':', label='Compositional'),
    ]
    labels = [le.get_label() for le in legend_elements]
    add_legend(ax, loc='lower right', fontsize='small', frameon=True, handles_labels=(legend_elements, labels))
    add_legend(ax_r, loc='lower right', fontsize='small', frameon=True, handles_labels=(legend_elements, labels))

    # Right subplot: paraphrasing level last-step dots (1B vs 7B)
    ax_rr = axes[2]
    ax_rr.set_xlabel("Paraphrase Level")
    ax_rr.grid(False)

    para_order = ['Para. 4', 'Para. 9', 'Para. 19', 'Para. 49']
    # Equidistant x positions with labels starting at 1 to reserve 0 for baseline
    x_positions = {label: idx for idx, label in enumerate(para_order, start=1)}
    color_1b = '#1f77b4'   # blue
    color_7b = '#d62728'   # bright red
    color_13b = '#2ca02c'  # green

    # 13B paraphrasing level paths
    para_13b_paths = {
        'Para. 4': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/allenai_OLMo-2-1124-13B/probes_v9/newline2/para4/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_22_16_48',
        'Para. 9': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/allenai_OLMo-2-1124-13B/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_21_13_18',
        'Para. 19': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/allenai_OLMo-2-1124-13B/probes_v9/newline2/para19/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_22_18_39',
        'Para. 49': '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/allenai_OLMo-2-1124-13B/probes_v9/newline2/para49/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_22_20_30',
    }

    # Collect data and plot for both models
    for model_size, color in [('1b', color_1b), ('7b', color_7b), ('13b', color_13b)]:
        xs, ys_k, ys_i = [], [], []
        
        # For 13B, use direct paths; for 1B and 7B, use get_paraphrasing_data
        if model_size.lower() == '13b':
            for label in para_order:
                path = para_13b_paths.get(label)
                if path:
                    resolved = path if os.path.isdir(path) else find_latest_run(path)
                    if resolved and os.path.isdir(resolved):
                        df_k, df_i = get_probe_data_for_method(resolved, domains, project_root)
                        val_k = get_final_step_value(df_k)
                        val_i = get_final_step_value(df_i)
                        if not np.isnan(val_k) or not np.isnan(val_i):
                            xs.append(x_positions[label])
                            ys_k.append(val_k)
                            ys_i.append(val_i)
        else:
            k_df, i_df = get_paraphrasing_data(model_size, domains, project_root)
            for label in para_order:
                val_k = np.nan
                val_i = np.nan
                if k_df is not None and not k_df.empty and label in k_df['method'].unique():
                    val_k = get_final_step_value(k_df[k_df['method'] == label])
                if i_df is not None and not i_df.empty and label in i_df['method'].unique():
                    val_i = get_final_step_value(i_df[i_df['method'] == label])
                if not np.isnan(val_k) or not np.isnan(val_i):
                    xs.append(x_positions[label])
                    ys_k.append(val_k)
                    ys_i.append(val_i)
        # Determine baseline (x=0) for both models
        baseline_k = np.nan
        baseline_i = np.nan
        if model_size.lower() == '7b':
            baseline_k = summaries.get("source_only", {}).get("knowledge", {}).get(100, np.nan)
            baseline_i = summaries.get("source_only", {}).get("inference", {}).get(100, np.nan)
        elif model_size.lower() == '1b':
            baseline_1b_path = "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/1b/probes_v9/newline2/source_only/sep_1_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs32_lr2e-05/run"
            resolved_1b = baseline_1b_path if os.path.isdir(baseline_1b_path) else find_latest_run(baseline_1b_path)
            if resolved_1b and os.path.isdir(resolved_1b):
                df_k_base, df_i_base = get_probe_data_for_method(resolved_1b, domains, project_root)
                baseline_k = get_final_step_value(df_k_base)
                baseline_i = get_final_step_value(df_i_base)
        elif model_size.lower() == '13b':
            baseline_13b_path = "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/allenai_OLMo-2-1124-13B/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_21_11_30"
            if baseline_13b_path:
                resolved_13b = baseline_13b_path if os.path.isdir(baseline_13b_path) else find_latest_run(baseline_13b_path)
                if resolved_13b and os.path.isdir(resolved_13b):
                    df_k_base, df_i_base = get_probe_data_for_method(resolved_13b, domains, project_root)
                    baseline_k = get_final_step_value(df_k_base)
                    baseline_i = get_final_step_value(df_i_base)

        # Plot with baseline included to connect lines
        xs_plot = xs.copy()
        ys_k_plot = ys_k.copy()
        ys_i_plot = ys_i.copy()
        if not np.isnan(baseline_k) or not np.isnan(baseline_i):
            xs_plot = [0] + xs_plot
            ys_k_plot = [baseline_k] + ys_k_plot
            ys_i_plot = [baseline_i] + ys_i_plot
            for val in (baseline_k, baseline_i):
                if not np.isnan(val):
                    all_y_values.append(val)

        if xs_plot:
            ax_rr.plot(xs_plot, ys_k_plot, color=color, linestyle='-', marker='o', linewidth=1.0)
            ax_rr.plot(xs_plot, ys_i_plot, color=color, linestyle=':', marker='o', linewidth=1.0)
            all_y_values.extend([y for y in ys_k_plot if not np.isnan(y)])
            all_y_values.extend([y for y in ys_i_plot if not np.isnan(y)])

    if para_order:
        xticks = [0] + [x_positions[label] for label in para_order]
        ax_rr.set_xticks(xticks)
        ax_rr.set_xticklabels(['0'] + [label.split()[-1] for label in para_order])  # ['0','4','9','19','49']
        # Add extra padding at both ends so first/last points aren't on the axes
        ax_rr.set_xlim(-0.5, len(para_order) + 0.5)

    # Legend for rightmost subplot: 7B vs 1B vs 13B colors + styles
    legend_rr = [
        Line2D([0], [0], color=color_7b, lw=2, linestyle='-', label='7B'),
        Line2D([0], [0], color=color_1b, lw=2, linestyle='-', label='1B'),
        Line2D([0], [0], color=color_13b, lw=2, linestyle='-', label='13B'),
        Line2D([0], [0], color='gray', lw=2, linestyle='-', label='Factual'),
        Line2D([0], [0], color='gray', lw=2, linestyle=':', label='Compositional'),
    ]
    labels_rr = [le.get_label() for le in legend_rr]
    # Place legend on right-middle, slightly lower
    add_legend(ax_rr, loc='center right', bbox_to_anchor=(0.98, 0.4), fontsize='x-small', frameon=True, handles_labels=(legend_rr, labels_rr))

    # Recalculate unified y-axis limits across all three subplots now that rightmost dots are added
    ylim = compute_unified_ylim(values=all_y_values)
    if ylim:
        apply_ylim([ax, ax_r, ax_rr], ylim)

    fig.tight_layout()

    out_name = f"last_step_dots_combined_{args.model_id.upper()}.pdf"
    out_path = os.path.join('plots', out_name)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()


