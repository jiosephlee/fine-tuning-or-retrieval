"""2x2 line plots of the three 32B training conditions over training steps.

Conditions (one line per panel):
  source_only       -> Source
  para9             -> Para. 9
  with_explanations -> Para. 9 + Aux

Panels (one per subplot):
  Factual MCQA      (knowledge / mcqa_accuracy)
  Factual Probes    (knowledge / log_prob)
  Inference Probes  (inference / log_prob)
  Inference MCQA    (inference / mcqa_accuracy)

Unlike the bar version (plot_32b_conditions_2x2.py), which reads the single-step
``reeval_v3`` final-model re-eval bundle, this plot needs the *per-step* training
curves, which live in the run root. Endpoints therefore differ slightly from the
reeval_v3 numbers (different probe-eval pass), but the shape of the growth is
what we care about here.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.plotting.plot_inference_mcqa_scaling import (  # noqa: E402
    METHODS,
    apply_plot_style,
    save_figure,
)
from scripts.plotting.plot_utils import (  # noqa: E402
    COLORS,
    compute_unified_ylim,
    find_latest_run,
    load_metrics,
)

CONDITION_LABELS = {
    "source_only": "Source",
    "para9": "Para. 9",
    "with_explanations": "Para. 9 + Aux",
}
CONDITION_COLORS = {
    "source_only": COLORS["method"]["source"],
    "para9": COLORS["paraphrase_level"]["para9"],
    "with_explanations": COLORS["method"]["aux_views"],
}

# Panel definition: (probe_type, metric, probe_family, title, ylabel).
# Ordered row-major so the top row holds both MCQA panels and the bottom row both
# log-prob ("Probes") panels -> y-axis label appears once per row.
PANELS = (
    ("knowledge", "mcqa_accuracy", "mcqa", "Factual MCQA", "Accuracy"),
    ("inference", "mcqa_accuracy", "mcqa", "Inference MCQA", "Accuracy"),
    ("knowledge", "log_prob", "classic", "Factual Probes", "Log Prob"),
    ("inference", "log_prob", "classic", "Inference Probes", "Log Prob"),
)

# 32B run roots (per-step training curves live here, not in the reeval bundle).
_32B_OVERLAP = "fill_dclm/domains_arxiv_all-legal_all-medical_all/e100/bs256_lr4e-05/overlap_1_16"
DEFAULT_RUNS: Dict[str, str] = {
    "source_only": f"results/FT/full/32b/source_only_docmatch_expl/{_32B_OVERLAP}/E10_source_32b_all_domains_fa2_packing_chunked_nll_1gpu_device_map_auto_36h_nodelta",
    "para9": f"results/FT/full/32b/para9_docmatch_expl/{_32B_OVERLAP}/E11_paraphrase_32b_all_domains_chunked_nll_1gpu_device_map_auto_36h_nodelta",
    "with_explanations": f"results/FT/full/32b/para9_expl_textbooks+stackexchange+blogs_cyclefull/{_32B_OVERLAP}/E12_granular_explanations_32b_all_domains_1gpu_device_map_auto_36h_nodelta",
}


def _abs_path(path: str) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return str(candidate)


def discover_domains(run_paths: Sequence[str]) -> list:
    import os

    suffix = "_inference_probe"
    domain_sets = []
    for run_path in run_paths:
        resolved = find_latest_run(run_path)
        if not resolved:
            continue
        names = {
            name[: -len(suffix)]
            for name in os.listdir(resolved)
            if name.endswith(suffix) and os.path.isdir(os.path.join(resolved, name))
        }
        if names:
            domain_sets.append(names)
    if not domain_sets:
        return []
    return sorted(set.intersection(*domain_sets))


def load_series(domains: Sequence[str]):
    """Return {(condition, probe_type, metric): DataFrame(step, metric)}."""
    series: Dict[tuple, Optional["pd.DataFrame"]] = {}
    for condition in METHODS:
        run_path = _abs_path(DEFAULT_RUNS[condition])
        for probe_type, metric, probe_family, _, _ in PANELS:
            df = load_metrics(
                run_path,
                probe_type,
                domains,
                str(REPO_ROOT),
                metrics=(metric,),
                probe_family=probe_family,
                mcqa_variant="preferred",
            )
            series[(condition, probe_type, metric)] = df
    return series


def plot(series, output: str):
    apply_plot_style()
    plt.rcParams.update(
        {
            "axes.labelsize": 16,
            "axes.titlesize": 18,
            "font.size": 15,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.1), sharex=True, sharey="row")
    n_cols = axes.shape[1]
    legend_ax = None
    row_values: Dict[int, list] = {}

    for idx, (ax, (probe_type, metric, _, title, ylabel)) in enumerate(zip(axes.flat, PANELS)):
        row, col = divmod(idx, n_cols)
        panel_values = row_values.setdefault(row, [])
        for condition in METHODS:
            df = series.get((condition, probe_type, metric))
            if df is None or df.empty or metric not in df.columns:
                continue
            ax.plot(
                df["step"],
                df[metric],
                color=CONDITION_COLORS[condition],
                linewidth=2,
                marker="o",
                markersize=3,
                markeredgewidth=0,
                alpha=0.9,
            )
            panel_values.extend(df[metric].dropna().tolist())
        ax.set_title(title)
        if row == axes.shape[0] - 1:
            ax.set_xlabel("Training Step")
        # Rows are metric-homogeneous after the reorder, so only the left subplot
        # of each row carries the (shared) y-axis label.
        if col == 0:
            ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        if title == "Factual Probes":
            legend_ax = ax

    # Apply a row-wide y-limit so the shared axis fits both columns' data.
    for row in range(axes.shape[0]):
        ylim = compute_unified_ylim(row_values.get(row, []), padding=0.05)
        if ylim:
            axes[row, 0].set_ylim(ylim)

    handles = [
        Line2D([0], [0], color=CONDITION_COLORS[c], linewidth=2, marker="o", label=CONDITION_LABELS[c])
        for c in METHODS
    ]
    (legend_ax or axes.flat[0]).legend(
        handles=handles,
        loc="lower right",
        frameon=True,
        framealpha=0.9,
        fontsize=11,
    )
    fig.tight_layout(pad=0.6, h_pad=0.8, w_pad=0.8)
    save_figure(fig, output)


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="plots/probe_scaling/probe_32b_conditions_2x2_lines")
    parser.add_argument("--domains", nargs="+")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    run_paths = [_abs_path(p) for p in DEFAULT_RUNS.values()]
    domains = args.domains or discover_domains(run_paths)
    if not domains:
        raise RuntimeError("No domains found in configured 32B run roots")
    print(f"Using {len(domains)} domains")

    series = load_series(domains)
    for key, df in series.items():
        if df is None or df.empty:
            print(f"Warning: missing series for {key}")
    plot(series, args.output)


if __name__ == "__main__":
    main()
