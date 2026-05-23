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
    METHOD_COLORS,
    PANELS,
    REEVAL_DIR_CHOICES,
    iter_run_items,
    load_configured_values,
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
        )
    return grouped


def plot_domain_scaling(grouped_values: Dict[str, dict], output: str):
    apply_plot_style()
    groups = [group for group in DOMAIN_GROUPS if group in grouped_values]
    if not groups:
        raise RuntimeError("No domain groups to plot")

    fig, axes = plt.subplots(
        len(groups),
        len(PANELS),
        figsize=(15, 3.2 * len(groups)),
        sharex=True,
        squeeze=False,
    )
    x = np.arange(len(MODEL_LABELS))
    model_tick_labels = [MODEL_LABELS[model] for model in MODEL_LABELS]

    for row, group in enumerate(groups):
        values = grouped_values[group]
        for col, (probe_type, metric, title, ylabel) in enumerate(PANELS):
            ax = axes[row, col]
            panel_values = []
            for method in METHODS:
                metric_values = []
                for model in MODEL_LABELS:
                    item = values[(probe_type, method, model)]
                    value = item[metric]
                    metric_values.append(np.nan if value is None else value)

                panel_values.extend([v for v in metric_values if not np.isnan(v)])
                ax.plot(
                    x,
                    metric_values,
                    color=METHOD_COLORS[method],
                    linewidth=2,
                    marker="o",
                    linestyle="--" if metric == "mcqa_accuracy" else "-",
                )

            if row == 0:
                ax.set_title(title)
            if row == len(groups) - 1:
                ax.set_xlabel("Model Size")
                ax.set_xticks(x)
                ax.set_xticklabels(model_tick_labels)
            else:
                ax.set_xticks(x)
                ax.set_xticklabels([])
            if col == 0:
                ax.set_ylabel(DOMAIN_LABELS.get(group, group.title()))
            if row == 0:
                ax.text(
                    0.02,
                    0.98,
                    ylabel,
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=10,
                )
            ax.grid(True, axis="y", alpha=0.25)

            ylim = compute_unified_ylim(panel_values, padding=0.05)
            if ylim:
                ax.set_ylim(ylim)

    method_handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_COLORS[method],
            linewidth=2,
            marker="o",
            label=METHOD_LABELS[method],
        )
        for method in METHODS
    ]
    axes[0, 1].legend(
        handles=method_handles,
        loc="lower right",
        frameon=True,
        framealpha=0.9,
        borderpad=0.4,
        handlelength=1.8,
    )
    fig.tight_layout()
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

    plot_domain_scaling(grouped_values, args.output)


if __name__ == "__main__":
    main()
