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
    METHODS,
    METHOD_LABELS,
    apply_plot_style,
    canonical_run_root,
    final_value,
    save_figure,
)
from scripts.plotting.plot_utils import (  # noqa: E402
    COLORS,
    compute_unified_ylim,
    find_latest_run,
    load_metrics,
)


MODEL_LABELS = {
    "1b": "1B",
    "7b": "7B",
    "13b": "13B",
    "32b": "32B",
}

PROBE_TYPES = (
    ("inference", "Inference Probes"),
    ("knowledge", "Factual Probes"),
)
PANELS = (
    ("knowledge", "log_prob", "Factual Probes", "Final Log Prob"),
    ("knowledge", "mcqa_accuracy", "Factual MCQA", "Final Accuracy"),
    ("inference", "log_prob", "Inference Probes", "Final Log Prob"),
    ("inference", "mcqa_accuracy", "Inference MCQA", "Final Accuracy"),
)
METHOD_COLORS = {
    "source_only": COLORS["method"]["source"],
    "para9": COLORS["paraphrase_level"]["para9"],
    "with_explanations": COLORS["method"]["aux_views"],
}
VALUE_MODE_CHOICES = ("final", "delta")
FACTUAL_PROBE_VARIANT_CHOICES = ("canonical", "paraphrased")

# Newest (v14 probe / inf_mcqa_v14) runs per method x model size.
# Paths point directly at the leaf folder that holds the per-domain probe
# directories (either an ``eval_bundles/inf_mcqa_v14`` bundle or a run root).
# Use ``None`` where a run does not exist yet.
DEFAULT_RUNS: Dict[str, Dict[str, Optional[str]]] = {
    # Source has no inf_mcqa_v14 bundle; use the reeval_v3 final-model re-eval
    # bundle (same source plot_probe_scaling_by_model.py uses via --reeval_dir reeval_v3).
    "source_only": {
        "1b": "results/FT/full/1b/source_only_docmatch_expl/fill_dclm/domains_arxiv_all-legal_all-medical_all/e100/bs256_lr4e-05/overlap_1_16/E4_source_1b_all_domains_fa2_packing/eval_bundles/reeval_v3",
        "7b": "results/FT/full/7b/source_only_docmatch_expl/fill_dclm/domains_arxiv_all-legal_all-medical_all/e100/bs256_lr4e-05/overlap_1_16/E1_source_all_domains_fa2_packing/eval_bundles/reeval_v3",
        "13b": "results/FT/full/13b/source_only_docmatch_expl/fill_dclm/domains_arxiv_all-legal_all-medical_all/e100/bs256_lr4e-05/overlap_1_16/E7_source_13b_all_domains_fa2_packing/eval_bundles/reeval_v3",
        "32b": "results/FT/full/32b/source_only_docmatch_expl/fill_dclm/domains_arxiv_all-legal_all-medical_all/e100/bs256_lr4e-05/overlap_1_16/E10_source_32b_all_domains_fa2_packing_chunked_nll_1gpu_device_map_auto_36h_nodelta/eval_bundles/reeval_v3",
    },
    "para9": {
        "1b": "results/FT/full/1b/para49/fill_dclm/domains_arxiv_all-legal_all-medical_all/e100/bs256_lr4e-05/overlap_1_16/E2_paraphrase_all_domains_local/eval_bundles/inf_mcqa_v14",
        "7b": "results/FT/full/7b/para49/fill_dclm/domains_arxiv_all-legal_all-medical_all/e100/bs256_lr4e-05/overlap_1_16/E2_paraphrase_all_domains_local/eval_bundles/inf_mcqa_v14",
        "13b": "results/FT/full/allenai_OLMo-2-1124-13B/para49/fill_dclm/domains_arxiv_all-legal_all-medical_all/e100/bs256_lr4e-05/overlap_1_16/E2_paraphrase_all_domains/eval_bundles/inf_mcqa_v14",
        # 32B has no pure para49 run; para49_docmatch_expl is the paraphrase-only
        # (doc-count-matched) variant, matching the original "Paraphrased" convention.
        "32b": "results/FT/full/allenai_OLMo-2-0325-32B/para49_docmatch_expl/fill_dclm/domains_arxiv_all-legal_all-medical_all/e100/bs256_lr4e-05/overlap_1_16/E11_paraphrase_32b_all_domains_chunked_nll/eval_bundles/inf_mcqa_v14",
    },
    "with_explanations": {
        "1b": "results/FT/full/1b/para49_expl_textbooks+stackexchange+blogs_cyclefull/fill_dclm/domains_arxiv_all-legal_all-medical_all/e100/bs256_lr4e-05/overlap_1_16/E3_granular_explanations_all_domains_local/eval_bundles/inf_mcqa_v14",
        "7b": "results/FT/full/7b/para49_expl_textbooks+stackexchange+blogs_cyclefull/fill_dclm/domains_arxiv_all-legal_all-medical_all/e100/bs256_lr4e-05/overlap_1_16/E3_granular_explanations_all_domains_local/eval_bundles/inf_mcqa_v14",
        "13b": "results/FT/full/allenai_OLMo-2-1124-13B/para49_expl_textbooks+stackexchange+blogs_cyclefull/fill_dclm/domains_arxiv_all-legal_all-medical_all/e100/bs256_lr4e-05/overlap_1_16/E9_granular_explanations_13b_all_domains/eval_bundles/inf_mcqa_v14",
        "32b": "results/FT/full/allenai_OLMo-2-0325-32B/para49_expl_textbooks+stackexchange+blogs_cyclefull/fill_dclm/domains_arxiv_all-legal_all-medical_all/e100/bs256_lr4e-05/overlap_1_16/E12_granular_explanations_32b_all_domains/eval_bundles/inf_mcqa_v14",
    },
}


def _abs_path(path: str) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return str(candidate)


def iter_run_items() -> Iterable[Tuple[str, str, Optional[str]]]:
    for method in METHODS:
        for model in MODEL_LABELS:
            path = DEFAULT_RUNS[method][model]
            yield method, model, (_abs_path(path) if path else None)


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


def delta_value(df, metric: str) -> Optional[float]:
    if df is None or df.empty or "step" not in df.columns or metric not in df.columns:
        return None
    valid = df[["step", metric]].dropna()
    if valid.empty or valid["step"].nunique() < 2:
        return None
    min_step = valid["step"].min()
    max_step = valid["step"].max()
    initial = valid.loc[valid["step"] == min_step, metric].mean()
    final = valid.loc[valid["step"] == max_step, metric].mean()
    return float(final - initial)


def initial_value(df, metric: str) -> Optional[float]:
    if df is None or df.empty or "step" not in df.columns or metric not in df.columns:
        return None
    valid = df[["step", metric]].dropna()
    if valid.empty:
        return None
    min_step = valid["step"].min()
    return float(valid.loc[valid["step"] == min_step, metric].mean())


def metric_value(df, metric: str, value_mode: str, baseline_df=None) -> Optional[float]:
    if value_mode == "final":
        return final_value(df, metric)
    if value_mode == "delta":
        delta = delta_value(df, metric)
        if delta is not None:
            return delta
        # Single-step (final-model re-eval) source bundles can't yield a delta on
        # their own; fall back to final - initial using a multi-step baseline read
        # from the run root (matches plot_probe_scaling_by_model.py).
        final = final_value(df, metric)
        initial = initial_value(baseline_df, metric)
        if final is None or initial is None:
            return None
        return float(final - initial)
    raise ValueError(f"Unsupported value mode: {value_mode}")


def load_configured_values(
    domains: Sequence[str],
    value_mode: str = "final",
    factual_probe_variant: str = "canonical",
):
    if value_mode not in VALUE_MODE_CHOICES:
        raise ValueError(f"Unsupported value mode: {value_mode}")
    if factual_probe_variant not in FACTUAL_PROBE_VARIANT_CHOICES:
        raise ValueError(f"Unsupported factual probe variant: {factual_probe_variant}")
    values: Dict[Tuple[str, str, str], dict] = {}
    for method, model, run_path in iter_run_items():
        for probe_type, _ in PROBE_TYPES:
            if run_path is None:
                values[(probe_type, method, model)] = {
                    "log_prob": None,
                    "mcqa_accuracy": None,
                    "run_path": None,
                }
                continue
            classic_metric_file_variant = (
                "paraphrased"
                if probe_type == "knowledge" and factual_probe_variant == "paraphrased"
                else "default"
            )
            classic = load_metrics(
                run_path,
                probe_type,
                domains,
                str(REPO_ROOT),
                metrics=("log_prob",),
                probe_family="classic",
                metric_file_variant=classic_metric_file_variant,
            )
            mcqa = load_metrics(
                run_path,
                probe_type,
                domains,
                str(REPO_ROOT),
                metrics=("mcqa_accuracy",),
                probe_family="mcqa",
                mcqa_variant="preferred",
            )
            classic_baseline = None
            mcqa_baseline = None
            if value_mode == "delta":
                # The baseline (initial / base-model) value comes from the run
                # root's multi-step probe outputs. Needed when the configured path
                # is a single-step re-eval bundle (e.g. source's reeval_v3).
                root = canonical_run_root(run_path)
                baseline_path = str(root) if root is not None else run_path
                classic_baseline = load_metrics(
                    baseline_path,
                    probe_type,
                    domains,
                    str(REPO_ROOT),
                    metrics=("log_prob",),
                    probe_family="classic",
                    metric_file_variant=classic_metric_file_variant,
                )
                mcqa_baseline = load_metrics(
                    baseline_path,
                    probe_type,
                    domains,
                    str(REPO_ROOT),
                    metrics=("mcqa_accuracy",),
                    probe_family="mcqa",
                    mcqa_variant="preferred",
                )
            values[(probe_type, method, model)] = {
                "log_prob": metric_value(classic, "log_prob", value_mode, classic_baseline),
                "mcqa_accuracy": metric_value(mcqa, "mcqa_accuracy", value_mode, mcqa_baseline),
                "run_path": find_latest_run(run_path),
            }
    return values


def panel_ylabel(ylabel: str, value_mode: str) -> str:
    if value_mode == "delta":
        return ylabel.replace("Final", "Delta")
    return ylabel


def panel_title(title: str, probe_type: str, metric: str, factual_probe_variant: str) -> str:
    if factual_probe_variant == "paraphrased" and probe_type == "knowledge" and metric == "log_prob":
        return "Paraphrased Factual Probes"
    return title


def plot_probe_scaling(
    values,
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

            metric_values = np.asarray(metric_values, dtype=float)
            mask = ~np.isnan(metric_values)
            panel_values.extend(metric_values[mask].tolist())
            ax.plot(
                x[mask],
                metric_values[mask],
                color=color,
                linewidth=2,
                marker="o",
                linestyle="--" if metric == "mcqa_accuracy" else "-",
            )

        ax.set_title(panel_title(title, probe_type, metric, factual_probe_variant))
        ax.set_xticks(x)
        ax.set_xticklabels(model_tick_labels)
        ax.set_xlabel("Model Size")
        ax.set_ylabel(panel_ylabel(ylabel, value_mode))
        if value_mode == "delta":
            ax.axhline(0, color="black", linewidth=0.8, alpha=0.35)
        ax.grid(True, axis="y", alpha=0.25)

        ylim = compute_unified_ylim(panel_values, padding=0.05)
        if ylim:
            ax.set_ylim(ylim)

    method_handles = [
        Line2D([0], [0], color=METHOD_COLORS[method], linewidth=2, marker="o", label=METHOD_LABELS[method])
        for method in METHODS
    ]
    axes[0].legend(
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
    parser.add_argument("--output", default="plots/probe_scaling/probe_scaling_by_model_v14")
    parser.add_argument("--domains", nargs="+")
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
    run_paths = [path for _, _, path in iter_run_items() if path]
    domains = args.domains or discover_domains_from_runs(run_paths)
    if not domains:
        raise RuntimeError("No domains found in configured result folders")

    print(f"Using {len(domains)} domains")
    values = load_configured_values(
        domains,
        value_mode=args.value_mode,
        factual_probe_variant=args.factual_probe_variant,
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

    plot_probe_scaling(
        values,
        args.output,
        value_mode=args.value_mode,
        factual_probe_variant=args.factual_probe_variant,
    )


if __name__ == "__main__":
    main()
