import argparse
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
    apply_plot_style,
    save_figure,
)
from scripts.plotting.plot_probe_scaling_by_model_v14 import (  # noqa: E402
    FACTUAL_PROBE_VARIANT_CHOICES,
    METHOD_COLORS,
    MODEL_LABELS,
    PANELS,
    VALUE_MODE_CHOICES,
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
    return sorted(path.name for path in group_path.iterdir() if path.is_dir())


def discover_domain_groups(groups: Sequence[str]) -> Dict[str, List[str]]:
    return {group: domains_for_group(group) for group in groups}


def load_grouped_values(
    domain_groups: Dict[str, Sequence[str]],
    value_mode: str,
    factual_probe_variant: str,
) -> Dict[str, dict]:
    grouped = {}
    for group, domains in domain_groups.items():
        if not domains:
            continue
        grouped[group] = load_configured_values(
            domains,
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

                metric_values = np.asarray(metric_values, dtype=float)
                mask = ~np.isnan(metric_values)
                panel_values.extend(metric_values[mask].tolist())
                ax.plot(
                    x[mask],
                    metric_values[mask],
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
            if row == 0:
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
    axes[-1, -1].legend(
        handles=method_handles,
        loc="lower right",
        frameon=True,
        framealpha=0.9,
        borderpad=0.4,
        handlelength=1.8,
        fontsize=12,
    )
    fig.tight_layout()
    save_figure(fig, output)


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="plots/probe_scaling/probe_scaling_by_domain_v14")
    parser.add_argument("--groups", nargs="+", choices=DOMAIN_GROUPS, default=list(DOMAIN_GROUPS))
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
    for group, domains in domain_groups.items():
        if not domains:
            print(f"Warning: no domains found for {group}")
        else:
            print(f"Using {len(domains)} {group} domains")

    grouped_values = load_grouped_values(
        domain_groups,
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
