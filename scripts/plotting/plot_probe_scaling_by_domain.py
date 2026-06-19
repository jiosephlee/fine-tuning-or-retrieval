import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.plotting.plot_inference_mcqa_scaling import (  # noqa: E402
    METHODS,
    METHOD_LABELS,
    MODEL_LABELS,
    apply_plot_style,
    save_figure,
)
from scripts.plotting.plot_probe_scaling_by_model import (  # noqa: E402
    INFERENCE_MCQA_REEVAL_CHOICES,
    FACTUAL_PROBE_VARIANT_CHOICES,
    LEGEND_LABELS,
    METHOD_COLORS,
    PANELS,
    PANEL_GROUPS,
    REEVAL_DIR_CHOICES,
    VALUE_MODE_CHOICES,
    iter_run_items,
    load_configured_values,
    panel_title,
    panel_ylabel,
)
from scripts.plotting.plot_utils import compute_unified_ylim  # noqa: E402


DOMAIN_GROUPS = ("arxiv", "legal", "medical")
DOMAIN_LABELS = {
    "arxiv": "Arxiv",
    "legal": "Legal",
    "medical": "Medical",
}


def domains_for_group(group: str) -> List[str]:
    group_path = REPO_ROOT / "probes" / group
    if not group_path.is_dir():
        return []
    return sorted(
        path.name
        for path in group_path.iterdir()
        if path.is_dir()
    )


def discover_domain_groups(groups: Sequence[str]) -> Dict[str, List[str]]:
    return {group: domains_for_group(group) for group in groups}


def load_grouped_values(
    domain_groups: Dict[str, Sequence[str]],
    mcqa_variant: str,
    reviewed_fallback: str,
    use_reeval: bool,
    inference_mcqa_reeval: Optional[str],
    reeval_dir: str,
    value_mode: str,
    factual_probe_variant: str,
) -> Dict[str, dict]:
    grouped = {}
    for group, domains in domain_groups.items():
        if not domains:
            continue
        group_inference_mcqa_reeval = (
            "v13_legal"
            if group == "legal" and inference_mcqa_reeval is None and use_reeval
            else inference_mcqa_reeval
        )
        grouped[group] = load_configured_values(
            domains,
            mcqa_variant=mcqa_variant,
            reviewed_fallback=reviewed_fallback,
            use_reeval=use_reeval,
            inference_mcqa_reeval=group_inference_mcqa_reeval,
            reeval_dir=reeval_dir,
            value_mode=value_mode,
            factual_probe_variant=factual_probe_variant,
        )
    return grouped


def plot_domain_scaling(
    grouped_values: Dict[str, dict],
    output: str,
    value_mode: str = "final",
    factual_probe_variant: str = "canonical",
):
    apply_plot_style()
    plt.rcParams.update(
        {
            "axes.labelsize": 16,
            "axes.titlesize": 18,
            "font.size": 15,
            "legend.fontsize": 14,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
        }
    )
    groups = [group for group in DOMAIN_GROUPS if group in grouped_values]
    if not groups:
        raise RuntimeError("No domain groups to plot")

    n_rows = len(groups)
    fig = plt.figure(figsize=(15, 3.2 * n_rows))
    # 5-column grid with a thin spacer between the two metric groups: panels in
    # each group stay flush, while the middle gap leaves room for the second
    # group's y-axis label and tick labels.
    gs = fig.add_gridspec(n_rows, 5, width_ratios=[1, 1, 0.02, 1, 1], wspace=0.2, hspace=0.22)
    grid_cols = (0, 1, 3, 4)
    axes = np.empty((n_rows, len(PANELS)), dtype=object)
    first_ax = None
    for row in range(n_rows):
        for col, grid_col in enumerate(grid_cols):
            ax = fig.add_subplot(gs[row, grid_col], sharex=first_ax)
            first_ax = first_ax or ax
            axes[row, col] = ax
    group_for_col = {col: panel_group for panel_group in PANEL_GROUPS for col in panel_group}
    # Share the y-axis within each metric group (per row) so one y-label suffices.
    for row in range(n_rows):
        for panel_group in PANEL_GROUPS:
            for col in panel_group[1:]:
                axes[row, col].sharey(axes[row, panel_group[0]])
    x = np.arange(len(MODEL_LABELS))
    model_tick_labels = [MODEL_LABELS[model] for model in MODEL_LABELS]

    for row, group in enumerate(groups):
        values = grouped_values[group]
        group_values: Dict[tuple, List[float]] = {pg: [] for pg in PANEL_GROUPS}
        for col, (probe_type, metric, title, ylabel) in enumerate(PANELS):
            ax = axes[row, col]
            panel_group = group_for_col[col]
            for method in METHODS:
                metric_values = []
                for model in MODEL_LABELS:
                    item = values[(probe_type, method, model)]
                    value = item[metric]
                    metric_values.append(np.nan if value is None else value)

                group_values[panel_group].extend([v for v in metric_values if not np.isnan(v)])
                ax.plot(
                    x,
                    metric_values,
                    color=METHOD_COLORS[method],
                    linewidth=2,
                    marker="o",
                    linestyle="--" if metric == "mcqa_accuracy" else "-",
                )

            if row == 0:
                ax.set_title(panel_title(title, probe_type, metric, factual_probe_variant))
            if row == len(groups) - 1:
                ax.set_xlabel("Model Size")
                ax.set_xticks(x)
                ax.set_xticklabels(model_tick_labels)
            else:
                ax.set_xticks(x)
                ax.set_xticklabels([])
            if col == 0:
                ax.set_ylabel(DOMAIN_LABELS.get(group, group.title()))
            elif col != panel_group[0]:
                plt.setp(ax.get_yticklabels(), visible=False)
            if row == 0 and col == panel_group[0]:
                ax.text(
                    0.02,
                    0.98,
                    panel_ylabel(ylabel, value_mode),
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=10,
                )
            if value_mode == "delta":
                ax.axhline(0, color="black", linewidth=0.8, alpha=0.35)
            ax.grid(True, axis="y", alpha=0.25)

        for panel_group, vals in group_values.items():
            ylim = compute_unified_ylim(vals, padding=0.05)
            if ylim:
                axes[row, panel_group[0]].set_ylim(ylim)

    method_handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_COLORS[method],
            linewidth=2,
            marker="o",
            label=LEGEND_LABELS[method],
        )
        for method in METHODS
    ]
    axes[-1, -1].legend(
        handles=method_handles,
        loc="lower right",
        frameon=True,
        framealpha=0.9,
        borderpad=0.4,
        handlelength=1.8,
        fontsize=12,
    )
    save_figure(fig, output)


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mcqa_variant",
        choices=("regular", "reviewed", "preferred"),
        default="preferred",
    )
    parser.add_argument(
        "--reviewed_fallback",
        choices=("regular", "drop", "error"),
        default="regular",
    )
    parser.add_argument("--output", default="plots/probe_scaling_by_domain")
    parser.add_argument("--groups", nargs="+", choices=DOMAIN_GROUPS, default=list(DOMAIN_GROUPS))
    parser.add_argument(
        "--no_reeval",
        action="store_true",
        help="Use canonical run metrics instead of nearby final-model re-eval metrics.",
    )
    parser.add_argument(
        "--reeval_dir",
        choices=REEVAL_DIR_CHOICES,
        default="reeval",
        help="Re-eval result directory to use when re-eval loading is enabled.",
    )
    parser.add_argument(
        "--inference_mcqa_reeval",
        choices=INFERENCE_MCQA_REEVAL_CHOICES,
        help="Override inference MCQA loading with a specific re-eval result tree.",
    )
    parser.add_argument(
        "--value_mode",
        choices=VALUE_MODE_CHOICES,
        default="final",
        help="Plot final metric values or final-minus-initial deltas.",
    )
    parser.add_argument(
        "--factual_probe_variant",
        choices=FACTUAL_PROBE_VARIANT_CHOICES,
        default="canonical",
        help="Use canonical or paraphrased factual log-prob probe metrics in the factual probe panel.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    domain_groups = discover_domain_groups(args.groups)
    missing_groups = [group for group, domains in domain_groups.items() if not domains]
    for group in missing_groups:
        print(f"Warning: no domains found for {group}")
    for group, domains in domain_groups.items():
        if domains:
            print(f"Using {len(domains)} {group} domains")

    grouped_values = load_grouped_values(
        domain_groups,
        mcqa_variant=args.mcqa_variant,
        reviewed_fallback=args.reviewed_fallback,
        use_reeval=not args.no_reeval,
        inference_mcqa_reeval=args.inference_mcqa_reeval,
        reeval_dir=args.reeval_dir,
        value_mode=args.value_mode,
        factual_probe_variant=args.factual_probe_variant,
    )

    missing = []
    for group, values in grouped_values.items():
        for (probe_type, method, model), item in values.items():
            for metric in ("log_prob", "mcqa_accuracy"):
                if item[metric] is None:
                    missing.append((group, probe_type, METHOD_LABELS[method], MODEL_LABELS[model], metric))
    if missing:
        for group, probe_type, method, model, metric in missing:
            print(f"Warning: missing {group} {probe_type} {metric} for {method} {model}")

    plot_domain_scaling(
        grouped_values,
        args.output,
        value_mode=args.value_mode,
        factual_probe_variant=args.factual_probe_variant,
    )


if __name__ == "__main__":
    main()
