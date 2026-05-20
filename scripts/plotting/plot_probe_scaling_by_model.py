import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.plotting.plot_inference_mcqa_scaling import (  # noqa: E402
    DEFAULT_RUNS,
    METHODS,
    METHOD_LABELS,
    MODEL_LABELS,
    apply_plot_style,
    final_value,
    regular_mcqa_run_path,
    run_path_for_variant,
    save_figure,
)
from scripts.plotting.plot_utils import (  # noqa: E402
    COLORS,
    _is_reviewed_inference_mcqa_root,
    compute_unified_ylim,
    find_latest_run,
    load_metrics,
)


PROBE_TYPES = (
    ("inference", "Inference Probes"),
    ("knowledge", "Factual Probes"),
)
PANELS = (
    ("inference", "log_prob", "Inference Log Prob", "Final Log Prob"),
    ("inference", "mcqa_accuracy", "Inference MCQA", "Final MCQA Accuracy"),
    ("knowledge", "log_prob", "Factual Log Prob", "Final Log Prob"),
    ("knowledge", "mcqa_accuracy", "Factual MCQA", "Final MCQA Accuracy"),
)
METHOD_COLORS = {
    "source_only": COLORS["method"]["source"],
    "para9": COLORS["paraphrase_level"]["para9"],
    "with_explanations": COLORS["method"]["aux_views"],
}


def _abs_path(path: str) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return str(candidate)


def iter_run_items(mcqa_variant: str = "preferred") -> Iterable[Tuple[str, str, str]]:
    for method in METHODS:
        for model in MODEL_LABELS:
            yield method, model, _abs_path(run_path_for_variant(DEFAULT_RUNS[method][model], mcqa_variant))


def discover_domains_from_runs(run_paths: Sequence[str]) -> List[str]:
    domain_sets = []
    suffix = "_inference_probe"
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


def has_reviewed_mcqa_folder(run_path: str, probe_type: str) -> bool:
    resolved = find_latest_run(run_path)
    if not resolved:
        return False
    if probe_type == "inference" and _is_reviewed_inference_mcqa_root(resolved):
        return any(
            "_inference_mcqa_probe" in name and os.path.isdir(os.path.join(resolved, name))
            for name in os.listdir(resolved)
        )
    needle = "_inference_mcqa_probe" if probe_type == "inference" else "_knowledge_mcqa_probe"
    return any(
        "reviewed" in name
        and needle in name
        and os.path.isdir(os.path.join(resolved, name))
        for name in os.listdir(resolved)
    )


def load_probe_pair(
    run_path: str,
    probe_type: str,
    domains: Sequence[str],
    mcqa_variant: str,
    reviewed_fallback: str,
):
    classic = load_metrics(
        run_path,
        probe_type,
        domains,
        str(REPO_ROOT),
        metrics=("log_prob",),
        probe_family="classic",
    )
    actual_variant = mcqa_variant
    if mcqa_variant == "reviewed" and not has_reviewed_mcqa_folder(run_path, probe_type):
        if reviewed_fallback == "regular":
            actual_variant = "regular"
            run_path = regular_mcqa_run_path(run_path)
        elif reviewed_fallback == "drop":
            return classic, None, "dropped"
        else:
            raise FileNotFoundError(f"No reviewed {probe_type} MCQA folders found under {run_path}")

    mcqa = load_metrics(
        run_path,
        probe_type,
        domains,
        str(REPO_ROOT),
        metrics=("mcqa_accuracy",),
        probe_family="mcqa",
        mcqa_variant=actual_variant,
    )
    if mcqa is None and actual_variant == "reviewed":
        if reviewed_fallback == "regular":
            run_path = regular_mcqa_run_path(run_path)
            mcqa = load_metrics(
                run_path,
                probe_type,
                domains,
                str(REPO_ROOT),
                metrics=("mcqa_accuracy",),
                probe_family="mcqa",
                mcqa_variant="regular",
            )
            actual_variant = "regular"
        elif reviewed_fallback == "drop":
            actual_variant = "dropped"
        else:
            raise FileNotFoundError(f"No reviewed {probe_type} MCQA metrics found under {run_path}")
    return classic, mcqa, actual_variant


def load_configured_values(
    domains: Sequence[str],
    mcqa_variant: str,
    reviewed_fallback: str,
):
    values: Dict[Tuple[str, str, str], dict] = {}
    for method, model, run_path in iter_run_items(mcqa_variant):
        for probe_type, _ in PROBE_TYPES:
            classic, mcqa, actual_variant = load_probe_pair(
                run_path,
                probe_type,
                domains,
                mcqa_variant=mcqa_variant,
                reviewed_fallback=reviewed_fallback,
            )
            values[(probe_type, method, model)] = {
                "log_prob": final_value(classic, "log_prob"),
                "mcqa_accuracy": final_value(mcqa, "mcqa_accuracy"),
                "mcqa_variant": actual_variant,
                "run_path": find_latest_run(run_path),
            }
    return values


def plot_probe_scaling(values, output: str):
    apply_plot_style()
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), sharex=True)
    x = np.arange(len(MODEL_LABELS))
    model_tick_labels = [MODEL_LABELS[model] for model in MODEL_LABELS]

    for ax, (probe_type, metric, title, ylabel) in zip(axes, PANELS):
        panel_values = []
        for method in METHODS:
            color = METHOD_COLORS[method]
            metric_values = []
            for model in MODEL_LABELS:
                item = values[(probe_type, method, model)]
                value = item[metric]
                metric_values.append(np.nan if value is None else value)

            panel_values.extend([v for v in metric_values if not np.isnan(v)])
            ax.plot(
                x,
                metric_values,
                color=color,
                linewidth=2,
                marker="o",
                linestyle="--" if metric == "mcqa_accuracy" else "-",
            )

        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(model_tick_labels)
        ax.set_xlabel("Model Size")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.25)

        ylim = compute_unified_ylim(panel_values, padding=0.05)
        if ylim:
            ax.set_ylim(ylim)

    method_handles = [
        Line2D([0], [0], color=METHOD_COLORS[method], linewidth=2, marker="o", label=METHOD_LABELS[method])
        for method in METHODS
    ]
    metric_handles = [
        Line2D([0], [0], color="black", linewidth=2, linestyle="-", label="Log Prob"),
        Line2D([0], [0], color="black", linewidth=2, linestyle="--", label="MCQA Accuracy"),
    ]
    fig.legend(
        handles=method_handles + metric_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=5,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
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
    parser.add_argument("--output", default="plots/probe_scaling_by_model")
    parser.add_argument("--domains", nargs="+")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    run_paths = [path for _, _, path in iter_run_items(args.mcqa_variant)]
    domains = args.domains or discover_domains_from_runs(run_paths)
    if not domains:
        raise RuntimeError("No domains found in configured result folders")

    print(f"Using {len(domains)} domains")
    values = load_configured_values(
        domains,
        mcqa_variant=args.mcqa_variant,
        reviewed_fallback=args.reviewed_fallback,
    )

    missing = [
        (probe_type, METHOD_LABELS[method], MODEL_LABELS[model], metric)
        for (probe_type, method, model), item in values.items()
        for metric in ("log_prob", "mcqa_accuracy")
        if item[metric] is None
    ]
    if missing:
        for probe_type, method, model, metric in missing:
            print(f"Warning: missing {probe_type} {metric} for {method} {model}")

    plot_probe_scaling(values, args.output)


if __name__ == "__main__":
    main()
