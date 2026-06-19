"""Grouped-bar comparison of prior-knowledge placement log-prob deltas.

The plot aggregates all domains and shows front / middle / end placement as
colors. Solid bars are factual probes; hatched bars are inference probes.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.plotting.plot_inference_mcqa_scaling import apply_plot_style, save_figure  # noqa: E402
from scripts.plotting.plot_probe_scaling_by_model import VALUE_MODE_CHOICES, metric_value  # noqa: E402
from scripts.plotting.plot_utils import COLORS, compute_unified_ylim, find_latest_run, load_metrics  # noqa: E402


RUN_BASE = (
    "results/FT/full/7b/para9/fill_dclm/"
    "domains_arxiv_all-legal_all-medical_all/e100/bs256_lr4e-05/overlap_1_16"
)

POSITIONS: Tuple[Tuple[str, str, str, str], ...] = (
    ("front", "Front", "E13_prior_knowledge_front_local", COLORS["method"]["source"]),
    ("middle", "Middle", "E14_prior_knowledge_middle_local", COLORS["method"]["aux_views"]),
    ("end", "End", "E15_prior_knowledge_end_local", COLORS["method"]["corrupted"]),
)

PROBES: Tuple[Tuple[str, str, str], ...] = (
    ("knowledge", "Factual", ""),
    ("inference", "Inference", "//"),
)


def _abs_path(path: str) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return str(candidate)


def discover_domains(run_paths: Sequence[str]) -> List[str]:
    domain_sets = []
    suffix = "_knowledge_probe"
    for run_path in run_paths:
        resolved = find_latest_run(run_path)
        if not resolved:
            continue
        domains = {
            name[: -len(suffix)]
            for name in os.listdir(resolved)
            if name.endswith(suffix) and os.path.isdir(os.path.join(resolved, name))
        }
        if domains:
            domain_sets.append(domains)
    if not domain_sets:
        return []
    return sorted(set.intersection(*domain_sets))


def load_values(
    run_paths: Dict[str, str],
    domains: Sequence[str],
    value_mode: str,
) -> Dict[Tuple[str, str], Optional[float]]:
    values: Dict[Tuple[str, str], Optional[float]] = {}
    for key, run_path in run_paths.items():
        for probe_type, _label, _hatch in PROBES:
            df = load_metrics(
                run_path,
                probe_type,
                domains,
                str(REPO_ROOT),
                metrics=("log_prob",),
                probe_family="classic",
            )
            values[(key, probe_type)] = metric_value(
                df,
                "log_prob",
                value_mode,
                baseline_df=None,
            )
    return values


def ylabel_for(value_mode: str) -> str:
    if value_mode == "delta":
        return r"$\Delta$ Log Prob"
    return "Final Log Prob"


def plot_histogram(
    values: Dict[Tuple[str, str], Optional[float]],
    output: str,
    value_mode: str,
) -> None:
    apply_plot_style()
    plt.rcParams.update(
        {
            "axes.labelsize": 16,
            "axes.titlesize": 18,
            "font.size": 15,
            "legend.fontsize": 13,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 3.8), sharey=False)
    x = np.arange(len(POSITIONS)) * 0.82
    bar_width = 0.34

    for ax, (probe_type, title, hatch) in zip(axes, PROBES):
        heights = []
        colors = []
        plotted_values: List[float] = []
        for key, _label, _subdir, color in POSITIONS:
            value = values[(key, probe_type)]
            heights.append(np.nan if value is None else value)
            colors.append(color)
            if value is not None:
                plotted_values.append(value)
        ax.bar(
            x,
            heights,
            bar_width,
            color=colors,
            edgecolor="black",
            linewidth=0.8,
            hatch=hatch,
            alpha=0.82,
            zorder=3,
        )

        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([label for _key, label, _subdir, _color in POSITIONS], rotation=20, ha="right")
        ax.set_ylabel(ylabel_for(value_mode))
        ax.grid(True, axis="y", alpha=0.25, zorder=0, linestyle=(0, (2, 4)))
        if value_mode == "delta":
            ax.axhline(0, color="black", linewidth=0.8, alpha=0.35, zorder=1)
        # Anchor the y-axis at 0 so bar heights read from a zero baseline (extend
        # below 0 only if some values are negative).
        if plotted_values:
            lo = min(0.0, min(plotted_values))
            hi = max(0.0, max(plotted_values))
            span = (hi - lo) if hi > lo else max(abs(hi), 1.0)
            # A little top headroom so the legend clears the bars.
            ax.set_ylim(lo, hi + 0.20 * span)
        ax.tick_params(axis="x", top=True, direction="in")

    condition_handles = [
        mpatches.Patch(
            facecolor=color,
            edgecolor="black",
            linewidth=0.5,
            label=label,
        )
        for _key, label, _subdir, color in POSITIONS
    ]
    axes[0].legend(
        handles=condition_handles,
        loc="upper left",
        ncol=3,
        frameon=True,
        framealpha=0.9,
        fancybox=False,
        edgecolor="black",
        borderaxespad=0.25,
    )
    fig.tight_layout(w_pad=2.0)
    save_figure(fig, output)


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="plots/prior_knowledge_position_logprob_histogram_delta_all_domains_7B",
    )
    parser.add_argument("--value_mode", choices=VALUE_MODE_CHOICES, default="delta")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    run_paths = {
        key: _abs_path(str(Path(RUN_BASE) / subdir))
        for key, _label, subdir, _color in POSITIONS
    }
    missing = [key for key, path in run_paths.items() if not find_latest_run(path)]
    if missing:
        formatted = "\n".join(f"{key}: {run_paths[key]}" for key in missing)
        raise FileNotFoundError(f"Missing prior-knowledge runs:\n{formatted}")

    domains = discover_domains(list(run_paths.values()))
    if not domains:
        raise RuntimeError("No shared domains discovered across prior-knowledge runs.")
    print(f"Using {len(domains)} shared domains")

    values = load_values(run_paths, domains, args.value_mode)
    for key, label, _subdir, _color in POSITIONS:
        factual = values[(key, "knowledge")]
        inference = values[(key, "inference")]
        print(
            f"{label}: factual={factual if factual is not None else 'missing'} "
            f"inference={inference if inference is not None else 'missing'}"
        )

    plot_histogram(values, args.output, args.value_mode)


if __name__ == "__main__":
    main()
