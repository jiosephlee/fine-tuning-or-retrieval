import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.plotting.plot_inference_mcqa_scaling import apply_plot_style, save_figure  # noqa: E402
from scripts.plotting.plot_probe_scaling_by_model import (  # noqa: E402
    PANELS,
    VALUE_MODE_CHOICES,
    metric_value,
    panel_ylabel,
)
from scripts.plotting.plot_utils import COLORS, compute_unified_ylim, find_latest_run, load_metrics  # noqa: E402


SOURCE_RUNS: Dict[str, str] = {
    "textbooks": (
        "results/FT/full/7b/para9_docmatch_expl_inserttextbooks/fill_dclm/"
        "domains_arxiv_all-legal_all-medical_all/e100/bs256_lr4e-05/overlap_1_16/"
        "E3_granular_explanations_textbooks_match_local"
    ),
    "stackexchange": (
        "results/FT/full/7b/para9_docmatch_expl_insertstackexchange/fill_dclm/"
        "domains_arxiv_all-legal_all-medical_all/e100/bs256_lr4e-05/overlap_1_16/"
        "E3_granular_explanations_stackexchange_match_local"
    ),
    "blogs": (
        "results/FT/full/7b/para9_docmatch_expl_insertblogs/fill_dclm/"
        "domains_arxiv_all-legal_all-medical_all/e100/bs256_lr4e-05/overlap_1_16/"
        "E3_granular_explanations_blogs_match_local"
    ),
    "all_sources": (
        "results/FT/full/7b/para9_expl_textbooks+stackexchange+blogs_cyclefull/fill_dclm/"
        "domains_arxiv_all-legal_all-medical_all/e100/bs256_lr4e-05/overlap_1_16/"
        "E3_granular_explanations_all_domains"
    ),
}
SOURCE_LABELS = {
    "textbooks": "Textbooks",
    "stackexchange": "StackExchange",
    "blogs": "Blogs",
    "all_sources": "All",
}
SOURCE_COLORS = {
    "textbooks": COLORS["method"]["textbook"],
    "stackexchange": COLORS["method"]["stackexchange"],
    "blogs": COLORS["method"]["blogs"],
    "all_sources": COLORS["method"]["aux_views"],
}
DEFAULT_BASELINE_RUN = (
    "results/FT/full/7b/para9_expl_textbooks+stackexchange+blogs_cyclefull/fill_dclm/"
    "domains_arxiv_all-legal_all-medical_all/e100/bs256_lr4e-05/overlap_1_16/"
    "E3_granular_explanations_all_domains"
)
DOMAIN_GROUPS = ("arxiv", "legal", "medical")
DOMAIN_LABELS = {
    "arxiv": "Arxiv",
    "legal": "Legal",
    "medical": "Medical",
    "all": "All",
}
COMBINED_PANELS = (
    ("knowledge", "Factual"),
    ("inference", "Inference"),
)


def _abs_path(path: str) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return str(candidate)


def domains_for_group(group: str) -> List[str]:
    group_path = REPO_ROOT / "probes" / group
    if not group_path.is_dir():
        return []
    return sorted(path.name for path in group_path.iterdir() if path.is_dir())


def discover_domain_groups(groups: Sequence[str]) -> Dict[str, List[str]]:
    return {group: domains_for_group(group) for group in groups}


def discover_aggregate_domain_group(groups: Sequence[str]) -> Dict[str, List[str]]:
    domains: List[str] = []
    for group in groups:
        domains.extend(domains_for_group(group))
    return {"all": sorted(set(domains))}


def reeval_path(run_path: str, reeval_dir: str) -> str:
    return str(Path(_abs_path(run_path)) / "eval_bundles" / reeval_dir)


def source_endpoint_path(source: str, run_path: str, reeval_dir: str, baseline_reeval_dir: str) -> str:
    if source == "all_sources":
        return reeval_path(run_path, baseline_reeval_dir)
    return reeval_path(run_path, reeval_dir)


def load_panel_metric(
    run_path: str,
    probe_type: str,
    metric: str,
    domains: Sequence[str],
):
    probe_family = "mcqa" if metric == "mcqa_accuracy" else "classic"
    return load_metrics(
        run_path,
        probe_type,
        domains,
        str(REPO_ROOT),
        metrics=(metric,),
        probe_family=probe_family,
        mcqa_variant="regular",
    )


def load_source_values(
    domain_groups: Dict[str, Sequence[str]],
    reeval_dir: str,
    value_mode: str,
    baseline_run: str,
    baseline_reeval_dir: str,
) -> Dict[str, Dict[Tuple[str, str, str], Optional[float]]]:
    if value_mode not in VALUE_MODE_CHOICES:
        raise ValueError(f"Unsupported value mode: {value_mode}")

    values: Dict[str, Dict[Tuple[str, str, str], Optional[float]]] = {}
    baseline_path = _abs_path(baseline_run)
    for group, domains in domain_groups.items():
        if not domains:
            continue
        group_values: Dict[Tuple[str, str, str], Optional[float]] = {}
        for source, run_path in SOURCE_RUNS.items():
            endpoint_path = source_endpoint_path(source, run_path, reeval_dir, baseline_reeval_dir)
            for probe_type, metric, _, _ in PANELS:
                endpoint = load_panel_metric(endpoint_path, probe_type, metric, domains)
                baseline = (
                    load_panel_metric(baseline_path, probe_type, metric, domains)
                    if value_mode == "delta"
                    else None
                )
                group_values[(source, probe_type, metric)] = metric_value(
                    endpoint,
                    metric,
                    value_mode,
                    baseline,
                )
        values[group] = group_values
    return values


def panel_metric_label(ylabel: str, value_mode: str) -> str:
    return panel_ylabel(ylabel, value_mode).replace("Delta", r"$\Delta$")


def metric_axis_label(metric: str, value_mode: str) -> str:
    label = "Final Accuracy" if metric == "mcqa_accuracy" else "Final Log Prob"
    return panel_metric_label(label, value_mode)


def apply_metric_ylim(ax, values: Sequence[float], value_mode: str) -> None:
    numeric_values = [value for value in values if not np.isnan(value)]
    ylim = compute_unified_ylim(numeric_values, padding=0.12)
    if not ylim:
        return
    if value_mode == "delta":
        top = max(numeric_values) if numeric_values else ylim[1]
        top_padding = max(abs(top) * 0.30, 0.05)
        ax.set_ylim(min(ylim[0], 0), max(top + top_padding, 0))
    else:
        ax.set_ylim(ylim)


def validate_reeval_paths(reeval_dir: str, baseline_run: str, baseline_reeval_dir: str, value_mode: str) -> None:
    missing = []
    for source, run_path in SOURCE_RUNS.items():
        path = source_endpoint_path(source, run_path, reeval_dir, baseline_reeval_dir)
        if not find_latest_run(path):
            missing.append((source, path))
    if missing:
        formatted = "\n".join(f"{SOURCE_LABELS[source]}: {path}" for source, path in missing)
        raise FileNotFoundError(f"Missing re-eval folders:\n{formatted}")
    if value_mode == "delta":
        baseline_root = _abs_path(baseline_run)
        if not find_latest_run(baseline_root):
            raise FileNotFoundError(f"Missing baseline run folder: {baseline_root}")
        baseline_path = reeval_path(baseline_run, baseline_reeval_dir)
        if not find_latest_run(baseline_path):
            raise FileNotFoundError(f"Missing baseline re-eval folder: {baseline_path}")


def plot_source_comparison(
    grouped_values: Dict[str, Dict[Tuple[str, str, str], Optional[float]]],
    output: str,
    value_mode: str,
):
    apply_plot_style()
    # Styling transferred from plot_32b_conditions_2x2.py: black-edged bars and
    # a light y-grid, with modest fonts and extra breathing room.
    plt.rcParams.update(
        {
            "axes.labelsize": 13,
            "axes.titlesize": 14,
            "font.size": 12,
            "legend.fontsize": 11,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )
    groups = list(grouped_values)
    if not groups:
        raise RuntimeError("No domain groups to plot")

    fig, axes = plt.subplots(
        len(groups),
        len(COMBINED_PANELS),
        figsize=(11, 3.3 * len(groups)),
        sharex=True,
        squeeze=False,
    )
    sources = tuple(SOURCE_RUNS)
    x = np.arange(len(sources))
    labels = [SOURCE_LABELS[source] for source in sources]
    width = 0.34

    for row, group in enumerate(groups):
        values = grouped_values[group]
        for col, (probe_type, title) in enumerate(COMBINED_PANELS):
            ax = axes[row, col]
            ax2 = ax.twinx()
            ax.set_axisbelow(True)
            ax2.set_axisbelow(True)
            log_values = [
                values.get((source, probe_type, "log_prob"))
                for source in sources
            ]
            mcqa_values = [
                values.get((source, probe_type, "mcqa_accuracy"))
                for source in sources
            ]
            log_y = [np.nan if value is None else value for value in log_values]
            mcqa_y = [np.nan if value is None else value for value in mcqa_values]
            bar_colors = [SOURCE_COLORS[source] for source in sources]
            log_bars = ax.bar(
                x - width / 2,
                log_y,
                color=bar_colors,
                width=width,
                edgecolor="black",
                linewidth=0.8,
                alpha=0.85,
                zorder=3,
            )
            mcqa_bars = ax2.bar(
                x + width / 2,
                mcqa_y,
                color=bar_colors,
                width=width,
                edgecolor="black",
                linewidth=0.8,
                hatch="//",
                alpha=0.55,
                zorder=4,
            )
            if row == 0:
                ax.set_title(title)
            if col == 0:
                row_label = "" if group == "all" else DOMAIN_LABELS.get(group, group.title())
                if row_label:
                    ax.text(
                        -0.18,
                        0.5,
                        row_label,
                        transform=ax.transAxes,
                        ha="right",
                        va="center",
                        rotation=90,
                        fontsize=11,
                    )
            if row == len(groups) - 1:
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=20, ha="right")
            else:
                ax.set_xticks(x)
                ax.set_xticklabels([])
            # Only label the outer y-axes: log-prob on the far-left panel and
            # MCQA on the far-right panel; the inner ("middle") axes go unlabeled.
            ax.set_ylabel(
                metric_axis_label("log_prob", value_mode) if col == 0 else ""
            )
            ax2.set_ylabel(
                metric_axis_label("mcqa_accuracy", value_mode)
                if col == len(COMBINED_PANELS) - 1
                else ""
            )
            if value_mode == "delta":
                ax.axhline(0, color="black", linewidth=0.8, alpha=0.35, zorder=1)
                ax2.axhline(0, color="black", linewidth=0.8, alpha=0.2, zorder=1)
            ax.grid(True, axis="y", alpha=0.25, zorder=0)

            apply_metric_ylim(ax, log_y, value_mode)
            apply_metric_ylim(ax2, mcqa_y, value_mode)
            if row == 0 and col == 0:
                legend_elements = [
                    mpatches.Patch(facecolor="0.6", edgecolor="white", label="Log Prob"),
                    mpatches.Patch(
                        facecolor="0.6",
                        edgecolor="black",
                        hatch="//",
                        alpha=0.6,
                        label="MCQA",
                    ),
                ]
                ax.legend(
                    handles=legend_elements,
                    loc="upper right",
                    ncol=2,
                    frameon=True,
                    framealpha=0.9,
                    fontsize=9,
                    borderaxespad=0.35,
                )

    fig.tight_layout()
    save_figure(fig, output)


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="plots/explanation_source_comparison_7b_reeval_v5_delta_by_domain",
    )
    parser.add_argument(
        "--reeval_dir",
        default="reeval_v5",
        help="Re-eval result directory under each single-source explanation run.",
    )
    parser.add_argument(
        "--value_mode",
        choices=VALUE_MODE_CHOICES,
        default="delta",
        help="Plot final metric values or final-minus-baseline deltas.",
    )
    parser.add_argument(
        "--baseline_run",
        default=DEFAULT_BASELINE_RUN,
        help="Run root whose step-0 metrics are used as the baseline for delta mode.",
    )
    parser.add_argument(
        "--baseline_reeval_dir",
        default="reeval_v3",
        help="Adjacent baseline re-eval directory to validate under --baseline_run/eval_bundles for delta mode.",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=DOMAIN_GROUPS,
        default=list(DOMAIN_GROUPS),
    )
    parser.add_argument(
        "--aggregate_domains",
        action="store_true",
        help="Collapse the selected domain groups into one all-domain row.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    validate_reeval_paths(args.reeval_dir, args.baseline_run, args.baseline_reeval_dir, args.value_mode)

    domain_groups = (
        discover_aggregate_domain_group(args.groups)
        if args.aggregate_domains
        else discover_domain_groups(args.groups)
    )
    missing_groups = [group for group, domains in domain_groups.items() if not domains]
    for group in missing_groups:
        print(f"Warning: no domains found for {group}")
    for group, domains in domain_groups.items():
        if domains:
            print(f"Using {len(domains)} {group} domains")

    if args.value_mode == "delta":
        print(f"Using step-0 baseline from {_abs_path(args.baseline_run)}")
        print(f"Validated paired baseline re-eval {reeval_path(args.baseline_run, args.baseline_reeval_dir)}")

    grouped_values = load_source_values(
        domain_groups,
        args.reeval_dir,
        args.value_mode,
        args.baseline_run,
        args.baseline_reeval_dir,
    )
    missing = [
        (group, SOURCE_LABELS[source], title)
        for group, values in grouped_values.items()
        for source in SOURCE_RUNS
        for probe_type, metric, title, _ in PANELS
        if values.get((source, probe_type, metric)) is None
    ]
    if missing:
        for group, source, title in missing:
            print(f"Warning: missing {group} {title} for {source}")

    plot_source_comparison(grouped_values, args.output, args.value_mode)


if __name__ == "__main__":
    main()
