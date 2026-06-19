"""Grouped-bar ("histogram") comparison of the prior-knowledge / cited-works 7B runs.

Bar counterpart to ``plot_prior_knowledge_match.py`` (which draws trajectory lines).
It compares the same four arxiv+legal runs as grouped bars, with the domain group
(arxiv / legal) on the x-axis and one bar per condition:

  - E17 Source, E18 Para. 9, E16 Para. 9 + Prior Knowledge, E19 Para. 9 + Cited Works.

The run paths, labels, and colors are imported from ``plot_prior_knowledge_match`` so
the two figures stay in sync. All four runs store probe folders at the run root with a
full step 2->100 trajectory, so deltas (final - initial) compute within each run.
Inference MCQA uses the loader default ("preferred"), matching the line script.
"""

import argparse
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.plotting.plot_inference_mcqa_scaling import (  # noqa: E402
    apply_plot_style,
    save_figure,
)
from scripts.plotting.plot_prior_knowledge_match import RUNS  # noqa: E402
from scripts.plotting.plot_probe_scaling_by_domain import (  # noqa: E402
    DOMAIN_GROUPS,
    DOMAIN_LABELS,
    domains_for_group,
)
from scripts.plotting.plot_probe_scaling_by_model import (  # noqa: E402
    FACTUAL_PROBE_VARIANT_CHOICES,
    PANELS,
    PANEL_GROUPS,
    VALUE_MODE_CHOICES,
    metric_value,
    panel_title,
    panel_ylabel,
)
from scripts.plotting.plot_utils import (  # noqa: E402
    COLORS,
    compute_unified_ylim,
    load_metrics,
)


# These runs store probe folders at the run root (no eval_bundles sub-tree) and are
# trained on arxiv+legal only (no medical).
DEFAULT_GROUPS = ("arxiv", "legal")

# Build the condition table from the shared RUNS config in plot_prior_knowledge_match
# so run paths / labels / colors stay identical between the line and bar figures.
CONDITIONS: "OrderedDict[str, dict]" = OrderedDict(
    (
        key,
        {
            "run_path": str(REPO_ROOT / run_path),
            "label": label,
            "color": color,
        },
    )
    for key, label, run_path, color in RUNS
)

# Display-only label overrides for legends / axis text (run config stays intact).
LABEL_OVERRIDES = {
    "source": "Source",
    "para": "Para. 9",
    "prior_match": "Para. 9 + Prerequisite",
    "cited_match": "Para. 9 + Contextual",
}


def display_label(key: str) -> str:
    return LABEL_OVERRIDES.get(key, CONDITIONS[key]["label"])


def load_condition_group_value(
    condition: dict,
    group_domains: Sequence[str],
    probe_type: str,
    metric: str,
    value_mode: str,
    factual_probe_variant: str,
) -> Optional[float]:
    """Compute one bar value: a condition's metric for a domain group at 7B.

    Probes live at the run root with a full trajectory, so ``delta_value`` resolves
    final-minus-initial within ``df`` and no separate step-0 baseline is needed.
    """
    probe_family = "classic" if metric == "log_prob" else "mcqa"
    metric_file_variant = (
        "paraphrased"
        if probe_type == "knowledge"
        and metric == "log_prob"
        and factual_probe_variant == "paraphrased"
        else "default"
    )

    df = load_metrics(
        condition["run_path"],
        probe_type,
        group_domains,
        str(REPO_ROOT),
        metrics=(metric,),
        probe_family=probe_family,
        metric_file_variant=metric_file_variant,
    )

    return metric_value(df, metric, value_mode, baseline_df=None)


def load_all_values(
    cond_keys: Sequence[str],
    groups: Sequence[str],
    group_domains: Dict[str, List[str]],
    value_mode: str,
    factual_probe_variant: str,
    reference_key: Optional[str] = None,
) -> Dict[Tuple[str, str, str, str], Optional[float]]:
    """Return ``{(condition_key, group, probe_type, metric): value}``.

    When ``reference_key`` is given, each value is the displayed condition's *final*
    performance minus the reference condition's *final* performance (so bars show how
    much a condition beats the reference's endpoint, regardless of ``value_mode``).
    """
    def final_for(key: str, domains, probe_type, metric) -> Optional[float]:
        return load_condition_group_value(
            CONDITIONS[key], domains, probe_type, metric, "final", factual_probe_variant
        )

    values: Dict[Tuple[str, str, str, str], Optional[float]] = {}
    for cond_key in cond_keys:
        for group in groups:
            domains = group_domains[group]
            for probe_type, metric, _, _ in PANELS:
                if reference_key is not None:
                    cond_final = final_for(cond_key, domains, probe_type, metric)
                    ref_final = final_for(reference_key, domains, probe_type, metric)
                    val = (
                        None
                        if cond_final is None or ref_final is None
                        else cond_final - ref_final
                    )
                else:
                    val = load_condition_group_value(
                        CONDITIONS[cond_key],
                        domains,
                        probe_type,
                        metric,
                        value_mode,
                        factual_probe_variant,
                    )
                values[(cond_key, group, probe_type, metric)] = val
    return values


def plot_condition_histogram(
    values: Dict[Tuple[str, str, str, str], Optional[float]],
    cond_keys: Sequence[str],
    groups: Sequence[str],
    output: str,
    value_mode: str = "delta",
    factual_probe_variant: str = "canonical",
    reference_label: Optional[str] = None,
    hide_mcqa: bool = False,
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
    # Each panel gets its own independent y-axis (no sharing) and, optionally, the
    # MCQA panels are dropped. Panels are rendered square.
    active_panels = [
        panel for panel in PANELS if not (hide_mcqa and panel[1] == "mcqa_accuracy")
    ]
    n_panels = len(active_panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.8 * n_panels, 5.0))
    axes = np.atleast_1d(axes)

    cond_keys = list(cond_keys)
    n_cond = len(cond_keys)
    bar_width = 0.8 / n_cond
    x = np.arange(len(groups))

    for col, (ax, (probe_type, metric, title, ylabel)) in enumerate(
        zip(axes, active_panels)
    ):
        panel_values: List[float] = []
        for i, cond_key in enumerate(cond_keys):
            condition = CONDITIONS[cond_key]
            offset = (i - (n_cond - 1) / 2) * bar_width
            heights = []
            for group in groups:
                val = values[(cond_key, group, probe_type, metric)]
                heights.append(np.nan if val is None else val)
            panel_values.extend([v for v in heights if not np.isnan(v)])
            ax.bar(
                x + offset,
                heights,
                bar_width,
                color=condition["color"],
                edgecolor="black",
                linewidth=0.4,
            )

        ax.set_title(panel_title(title, probe_type, metric, factual_probe_variant))
        ax.set_xticks(x)
        ax.set_xticklabels([DOMAIN_LABELS.get(g, g.title()) for g in groups])
        if reference_label is not None:
            metric_name = "Log Prob" if "Log Prob" in ylabel else "Accuracy"
            ax.set_ylabel(f"$\\Delta$ {metric_name} (vs. {reference_label})")
        else:
            ax.set_ylabel(panel_ylabel(ylabel, value_mode))
        if value_mode == "delta" or reference_label is not None:
            ax.axhline(0, color="black", linewidth=0.8, alpha=0.35)
        ax.grid(True, axis="y", alpha=0.25)
        # Anchor the y-axis at 0 so bar heights are read honestly from a zero
        # baseline (extend below 0 only if some values are negative).
        if panel_values:
            data_min = min(panel_values)
            data_max = max(panel_values)
            lo = min(0.0, data_min)
            hi = max(0.0, data_max)
            pad = 0.08 * (hi - lo) if hi > lo else max(abs(hi), 1.0) * 0.08
            ax.set_ylim(lo, hi + pad)
        ax.set_box_aspect(1)

    handles = [
        mpatches.Patch(
            facecolor=CONDITIONS[key]["color"],
            edgecolor="black",
            linewidth=0.4,
            label=display_label(key),
        )
        for key in cond_keys
    ]
    # Legend in the top-left of the 1st subplot.
    axes[0].legend(
        handles=handles,
        loc="upper left",
        frameon=True,
        framealpha=0.9,
        borderpad=0.4,
        handlelength=1.4,
        fontsize=12,
    )
    fig.tight_layout()
    save_figure(fig, output)


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", default="plots/prior_knowledge_match_histogram_by_group_7B"
    )
    parser.add_argument(
        "--groups", nargs="+", choices=DOMAIN_GROUPS, default=list(DEFAULT_GROUPS)
    )
    parser.add_argument(
        "--value_mode", choices=VALUE_MODE_CHOICES, default="delta"
    )
    parser.add_argument(
        "--factual_probe_variant",
        choices=FACTUAL_PROBE_VARIANT_CHOICES,
        default="canonical",
    )
    parser.add_argument(
        "--hide_mcqa",
        action="store_true",
        help="Drop the MCQA panels, showing only the log-prob probe panels.",
    )
    parser.add_argument(
        "--reference_condition",
        choices=["none"] + list(CONDITIONS.keys()),
        default="none",
        help=(
            "If set, bars show each condition's FINAL performance minus this "
            "reference condition's final performance (e.g. 'para')."
        ),
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=list(CONDITIONS.keys()),
        help=(
            "Which conditions to draw as bars. Defaults to all; when "
            "--reference_condition is set, defaults to all except source and the "
            "reference."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    group_domains = {group: domains_for_group(group) for group in args.groups}
    groups = [g for g in DOMAIN_GROUPS if g in args.groups and group_domains.get(g)]
    if not groups:
        raise RuntimeError("No domain groups with domains found")
    for group in groups:
        print(f"Using {len(group_domains[group])} {group} domains")

    reference_key = None if args.reference_condition == "none" else args.reference_condition
    reference_label = display_label(reference_key) if reference_key else None

    if args.conditions:
        cond_keys = list(args.conditions)
    elif reference_key is not None:
        # Default to the "treatment" conditions: drop source and the reference itself.
        cond_keys = [
            k for k in CONDITIONS if k not in {"source", reference_key}
        ]
    else:
        cond_keys = list(CONDITIONS.keys())

    values = load_all_values(
        cond_keys,
        groups,
        group_domains,
        args.value_mode,
        args.factual_probe_variant,
        reference_key=reference_key,
    )

    for (cond_key, group, probe_type, metric), val in values.items():
        if val is None:
            print(
                f"Warning: missing {probe_type} {metric} for "
                f"{CONDITIONS[cond_key]['label']} / {group}"
            )

    plot_condition_histogram(
        values,
        cond_keys,
        groups,
        args.output,
        value_mode=args.value_mode,
        factual_probe_variant=args.factual_probe_variant,
        reference_label=reference_label,
        hide_mcqa=args.hide_mcqa,
    )


if __name__ == "__main__":
    main()
