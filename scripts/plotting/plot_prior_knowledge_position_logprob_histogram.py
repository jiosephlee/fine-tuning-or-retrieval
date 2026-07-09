"""Grouped-bar comparison of prior-knowledge placement deltas.

The plot aggregates all domains and shows front / middle / end placement as
colors. Solid bars are log-prob probe metrics; diagonally hatched bars are MCQA.
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
from scripts.plotting.plot_utils import COLORS, find_latest_run, load_metrics  # noqa: E402


RUN_BASE = (
    "results/FT/full/7b/para9/fill_dclm/"
    "domains_arxiv_all-legal_all-medical_all/e100/bs256_lr4e-05/overlap_1_16"
)
EVAL_BUNDLE = "eval_bundles/inf_mcqa_v14"

POSITIONS: Tuple[Tuple[str, str, str, str], ...] = (
    ("front", "Front", "E13_prior_knowledge_front_local", COLORS["method"]["source"]),
    ("middle", "Middle", "E14_prior_knowledge_middle_local", COLORS["method"]["aux_views"]),
    ("end", "End", "E15_prior_knowledge_end_local", COLORS["method"]["corrupted"]),
)

PROBES: Tuple[Tuple[str, str, str], ...] = (
    ("knowledge", "Factual", ""),
    ("inference", "Inference", ""),
)

METRICS: Tuple[Tuple[str, str, str], ...] = (
    ("log_prob", "classic", "Log Prob"),
    ("mcqa_accuracy", "mcqa", "MCQA"),
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
) -> Dict[Tuple[str, str, str], Optional[float]]:
    values: Dict[Tuple[str, str, str], Optional[float]] = {}
    for key, run_path in run_paths.items():
        for probe_type, _label, _hatch in PROBES:
            for metric, probe_family, _metric_label in METRICS:
                df = load_metrics(
                    run_path,
                    probe_type,
                    domains,
                    str(REPO_ROOT),
                    metrics=(metric,),
                    probe_family=probe_family,
                )
                values[(key, probe_type, metric)] = metric_value(
                    df,
                    metric,
                    value_mode,
                    baseline_df=None,
                )
    return values


def logprob_ylabel_for(value_mode: str) -> str:
    if value_mode == "delta":
        return r"$\Delta$ Log Prob"
    return "Final Log Prob"


def mcqa_ylabel_for(value_mode: str) -> str:
    if value_mode == "delta":
        return r"$\Delta$ MCQA Accuracy"
    return "Final MCQA Accuracy"


def set_metric_ylim(ax, values: Sequence[float], value_mode: str, is_mcqa: bool = False) -> None:
    finite = [v for v in values if np.isfinite(v)]
    if not finite:
        return
    if is_mcqa and value_mode == "final":
        ax.set_ylim(0.0, 1.0)
        return
    data_min = min(finite)
    data_max = max(finite)
    if value_mode == "delta":
        lo = min(0.0, data_min)
        hi = max(0.0, data_max)
    else:
        lo = min(0.0, data_min) if data_min >= 0 else data_min
        hi = max(0.0, data_max)
    span = (hi - lo) if hi > lo else max(abs(hi), 1.0)
    pad = 0.22 * span
    ax.set_ylim(lo, hi + pad)


def plot_histogram(
    values: Dict[Tuple[str, str, str], Optional[float]],
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

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.2), sharey=False)
    x = np.arange(len(POSITIONS)) * 0.82
    bar_width = 0.26

    for idx, (ax, (probe_type, title, _hatch)) in enumerate(zip(axes, PROBES)):
        log_values = []
        mcqa_values = []
        colors = []
        for key, _label, _subdir, color in POSITIONS:
            log_value = values[(key, probe_type, "log_prob")]
            mcqa_value = values[(key, probe_type, "mcqa_accuracy")]
            log_values.append(np.nan if log_value is None else log_value)
            mcqa_values.append(np.nan if mcqa_value is None else mcqa_value)
            colors.append(color)
        ax2 = ax.twinx()
        ax.bar(
            x - bar_width / 2,
            log_values,
            bar_width,
            color=colors,
            edgecolor="black",
            linewidth=0.8,
            alpha=0.82,
            zorder=3,
        )
        ax2.bar(
            x + bar_width / 2,
            mcqa_values,
            bar_width,
            color=colors,
            edgecolor="black",
            linewidth=0.8,
            hatch="//",
            alpha=0.46,
            zorder=3,
        )

        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([label for _key, label, _subdir, _color in POSITIONS], rotation=20, ha="right")
        ax.set_ylabel(logprob_ylabel_for(value_mode) if idx == 0 else "")
        ax2.set_ylabel(mcqa_ylabel_for(value_mode) if idx == len(PROBES) - 1 else "")
        ax.grid(True, axis="y", alpha=0.25, zorder=0, linestyle=(0, (2, 4)))
        if value_mode == "delta":
            ax.axhline(0, color="black", linewidth=0.8, alpha=0.35, zorder=1)
            ax2.axhline(0, color="black", linewidth=0.8, alpha=0.20, zorder=1)
        set_metric_ylim(ax, log_values, value_mode)
        set_metric_ylim(ax2, mcqa_values, value_mode, is_mcqa=True)
        ax.tick_params(axis="x", top=True, direction="in")

    metric_handles = [
        mpatches.Patch(facecolor="gray", edgecolor="black", linewidth=0.8, alpha=0.82, label="Log Prob"),
        mpatches.Patch(
            facecolor="gray",
            edgecolor="black",
            linewidth=0.8,
            hatch="//",
            alpha=0.46,
            label="MCQA",
        ),
    ]
    axes[0].legend(
        handles=metric_handles,
        loc="upper left",
        ncol=2,
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
        default="plots/prior_knowledge/prior_knowledge_position_logprob_histogram_delta_all_domains_7B",
    )
    parser.add_argument("--value_mode", choices=VALUE_MODE_CHOICES, default="delta")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    run_paths = {
        key: _abs_path(str(Path(RUN_BASE) / subdir / EVAL_BUNDLE))
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
        factual = values[(key, "knowledge", "log_prob")]
        factual_mcqa = values[(key, "knowledge", "mcqa_accuracy")]
        inference = values[(key, "inference", "log_prob")]
        inference_mcqa = values[(key, "inference", "mcqa_accuracy")]
        print(
            f"{label}: factual={factual if factual is not None else 'missing'} "
            f"factual_mcqa={factual_mcqa if factual_mcqa is not None else 'missing'} "
            f"inference={inference if inference is not None else 'missing'} "
            f"inference_mcqa={inference_mcqa if inference_mcqa is not None else 'missing'}"
        )

    plot_histogram(values, args.output, args.value_mode)


if __name__ == "__main__":
    main()
