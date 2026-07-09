import argparse
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.plotting.plot_inference_mcqa_scaling import final_value  # noqa: E402
from scripts.plotting.plot_utils import (  # noqa: E402
    COLORS,
    apply_plot_style,
    compute_unified_ylim,
    load_metrics,
    save_figure,
)


CONDITIONS = (
    (
        "source",
        "Source",
        "results/FT/full/7b/source_only_docmatch_expl/fill_dclm/"
        "domains_arxiv_all-legal_all-medical_all/e100",
        COLORS["method"]["source"],
    ),
    (
        "para49",
        "Para. 49",
        "results/FT/full/7b/para49_docmatch_expl/fill_dclm/"
        "domains_arxiv_all-legal_all-medical_all/e100",
        COLORS["method"]["paraphrase"],
    ),
    (
        "para49_aux",
        "Para. 49 + Aux.",
        "results/FT/full/7b/para49_expl_textbooks+stackexchange+blogs_cyclefull/fill_dclm/"
        "domains_arxiv_all-legal_all-medical_all/e100",
        COLORS["method"]["aux_views"],
    ),
)

PANELS = (
    ("knowledge", "log_prob", "classic", "Factual Probes", "Final Log Prob"),
    ("inference", "log_prob", "classic", "Inference Probes", "Final Log Prob"),
    ("knowledge", "mcqa_accuracy", "mcqa", "Factual MCQA", "Final Accuracy"),
    ("inference", "mcqa_accuracy", "mcqa", "Inference MCQA", "Final Accuracy"),
)
# Columns grouped by metric: cols 0-1 are log-prob probes, cols 2-3 are MCQA.
PANEL_GROUPS = ((0, 1), (2, 3))

LR_PATTERN = re.compile(r"(?:^|_)lr([0-9.eE+-]+)(?:_|$)")
REEVAL_LR = 4e-5
REEVAL_LR_RTOL = 1e-6


def abs_path(path: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate


def parse_lr(path: Path) -> Optional[float]:
    match = LR_PATTERN.search(path.name)
    if not match:
        return None
    return float(match.group(1))


def discover_lr_dirs(root: Path) -> List[Tuple[float, Path]]:
    items = []
    for child in root.iterdir() if root.is_dir() else []:
        if not child.is_dir():
            continue
        lr = parse_lr(child)
        if lr is not None:
            items.append((lr, child))
    return sorted(items, key=lambda item: item[0])


def experiment_dirs(lr_dir: Path) -> List[Path]:
    candidates = list(lr_dir.glob("E*"))
    candidates.extend(lr_dir.glob("*/E*"))
    return sorted([path for path in candidates if path.is_dir()])


def has_probe_dirs(path: Path) -> bool:
    probe_suffixes = (
        "_knowledge_probe",
        "_inference_probe",
        "_mcqa_probe",
        "_inference_mcqa_probe",
    )
    return any(
        child.is_dir() and any(child.name.endswith(suffix) for suffix in probe_suffixes)
        for child in path.iterdir()
    )


def bundle_run_path(lr_dir: Path, eval_bundle: str, reeval_dir: Optional[str] = None) -> Optional[Path]:
    for experiment_dir in experiment_dirs(lr_dir):
        if reeval_dir:
            reeval_candidate = experiment_dir / "eval_bundles" / reeval_dir
            if reeval_candidate.is_dir():
                return reeval_candidate
        candidate = experiment_dir / "eval_bundles" / eval_bundle
        if candidate.is_dir():
            return candidate
        if eval_bundle == "root" or has_probe_dirs(experiment_dir):
            return experiment_dir
    return None


def discover_domains(run_paths: Iterable[Path]) -> List[str]:
    domain_sets = []
    suffix = "_inference_probe"
    for run_path in run_paths:
        domains = {
            child.name[: -len(suffix)]
            for child in run_path.iterdir()
            if child.is_dir() and child.name.endswith(suffix)
        }
        if domains:
            domain_sets.append(domains)
    if not domain_sets:
        return []
    return sorted(set.intersection(*domain_sets))


def load_metric_value(
    run_path: Path,
    domains: Sequence[str],
    probe_type: str,
    metric: str,
    probe_family: str,
) -> Optional[float]:
    df = load_metrics(
        str(run_path),
        probe_type,
        domains,
        str(REPO_ROOT),
        metrics=(metric,),
        probe_family=probe_family,
        mcqa_variant="regular",
    )
    return final_value(df, metric)


def load_values(
    eval_bundle: str,
    domains: Optional[Sequence[str]] = None,
    reeval_lr_dir: Optional[str] = "reeval_v3",
):
    runs: Dict[str, List[Tuple[float, Path]]] = {}
    for key, _, root, _ in CONDITIONS:
        lr_runs = []
        for lr, lr_dir in discover_lr_dirs(abs_path(root)):
            use_reeval = (
                reeval_lr_dir
                if reeval_lr_dir and np.isclose(lr, REEVAL_LR, rtol=REEVAL_LR_RTOL, atol=0.0)
                else None
            )
            run_path = bundle_run_path(lr_dir, eval_bundle, reeval_dir=use_reeval)
            if run_path is None:
                print(f"Warning: no {eval_bundle} run found under {lr_dir}")
                continue
            lr_runs.append((lr, run_path))
        runs[key] = lr_runs

    all_run_paths = [run_path for lr_runs in runs.values() for _, run_path in lr_runs]
    resolved_domains = list(domains) if domains else discover_domains(all_run_paths)
    if not resolved_domains:
        raise RuntimeError("No shared domains found in the configured learning-rate runs")

    values = {}
    for key, _, _, _ in CONDITIONS:
        for lr, run_path in runs[key]:
            for probe_type, metric, probe_family, _, _ in PANELS:
                values[(key, lr, probe_type, metric)] = load_metric_value(
                    run_path,
                    resolved_domains,
                    probe_type,
                    metric,
                    probe_family,
                )
    return runs, resolved_domains, values


def format_lr(lr: float) -> str:
    return f"{lr:.0e}".replace("e-0", "e-").replace("e+0", "e")


def plot_values(runs, values, output: str):
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
    fig = plt.figure(figsize=(20, 4.4))
    grid = fig.add_gridspec(1, 5, width_ratios=[1, 1, 0.14, 1, 1], wspace=0.16)
    axes = [fig.add_subplot(grid[0, col]) for col in (0, 1, 3, 4)]
    all_lrs = sorted({lr for lr_runs in runs.values() for lr, _ in lr_runs})
    x = np.arange(len(all_lrs))
    lr_tick_labels = [format_lr(lr) for lr in all_lrs]
    panel_values_by_col: List[List[float]] = [[] for _ in PANELS]

    for col, (ax, (probe_type, metric, _, title, ylabel)) in enumerate(zip(axes, PANELS)):
        for key, _, _, color in CONDITIONS:
            values_by_lr = {}
            for lr, _ in runs[key]:
                value = values[(key, lr, probe_type, metric)]
                values_by_lr[lr] = np.nan if value is None else value
            metric_values = [values_by_lr.get(lr, np.nan) for lr in all_lrs]
            panel_values_by_col[col].extend([v for v in metric_values if not np.isnan(v)])
            ax.plot(
                x,
                metric_values,
                color=color,
                linewidth=2,
                marker="o",
                linestyle="-",
            )

        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(lr_tick_labels)
        ax.set_xlabel("Learning Rate")
        ax.set_ylabel(ylabel if col in {0, 2} else "")
        ax.grid(True, axis="y", alpha=0.25)

    for group in PANEL_GROUPS:
        grouped_values = [value for col in group for value in panel_values_by_col[col]]
        ylim = compute_unified_ylim(grouped_values, padding=0.05)
        if ylim:
            for col in group:
                axes[col].set_ylim(ylim)

    handles = [
        Line2D([0], [0], color=color, linewidth=2, marker="o", label=label)
        for _, label, _, color in CONDITIONS
    ]
    axes[0].legend(
        handles=handles,
        loc="best",
        frameon=True,
        framealpha=0.9,
        borderpad=0.4,
        handlelength=1.8,
        fontsize=12,
    )
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.16, top=0.88)
    save_figure(fig, output)


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="plots/batch_scaling/7b_lr_sweep_para49_main_metrics")
    parser.add_argument(
        "--eval_bundle",
        default="inf_mcqa_v14",
        help="Evaluation bundle to read under each learning-rate experiment.",
    )
    parser.add_argument(
        "--reeval_lr_dir",
        default="reeval_v3",
        help="Evaluation bundle to prefer for the 4e-5 learning-rate point.",
    )
    parser.add_argument("--domains", nargs="+")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    runs, domains, values = load_values(args.eval_bundle, args.domains, args.reeval_lr_dir)
    print(f"Using {len(domains)} domains")

    missing = [
        (key, lr, probe_type, metric)
        for (key, lr, probe_type, metric), value in values.items()
        if value is None
    ]
    for key, lr, probe_type, metric in missing:
        print(f"Warning: missing {probe_type} {metric} for {key} at lr={lr:g}")

    plot_values(runs, values, args.output)


if __name__ == "__main__":
    main()
