import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.plotting.plot_inference_mcqa_scaling import (  # noqa: E402
    DEFAULT_RUNS,
    METHODS,
    METHOD_LABELS,
    MODEL_LABELS,
    REEVAL_DIR_CHOICES,
    apply_plot_style,
    eval_bundle_path,
    final_value,
    mcqa_run_path_for_variant,
    regular_mcqa_run_path,
    reeval_mcqa_run_path,
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
V13_LEGAL_INFERENCE_MCQA_TAG = (
    "probes_v13_para_v13_paraphrased_inf_v11_reviewed_mcqa_v14_prompt_formatted_question_5shot_"
    "inf_mcqa_v13_legal"
)
INFERENCE_MCQA_REEVAL_CHOICES = ("v13_legal", "v13_mixed")
VALUE_MODE_CHOICES = ("final", "delta")
FACTUAL_PROBE_VARIANT_CHOICES = ("canonical", "paraphrased")


def _abs_path(path: str) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return str(candidate)


def iter_run_items(mcqa_variant: str = "preferred") -> Iterable[Tuple[str, str, str]]:
    for method in METHODS:
        for model in MODEL_LABELS:
            yield method, model, _abs_path(run_path_for_variant(DEFAULT_RUNS[method][model], mcqa_variant))


def inference_mcqa_reeval_run_path(
    run_path: str,
    inference_mcqa_reeval: Optional[str],
    reeval_dir: str = "reeval",
) -> str:
    if inference_mcqa_reeval is None:
        return run_path
    if inference_mcqa_reeval == "v13_mixed":
        return run_path
    if inference_mcqa_reeval != "v13_legal":
        raise ValueError(f"Unsupported inference MCQA reeval: {inference_mcqa_reeval}")
    canonical = eval_bundle_path(run_path, ("inf_mcqa_v13_legal",))
    if canonical != run_path:
        return canonical
    reeval_canonical = eval_bundle_path(run_path, ("inf_mcqa_v13_legal",), require_leaf=False)
    if reeval_canonical != run_path and find_latest_run(os.path.join(reeval_canonical, reeval_dir)):
        return reeval_canonical
    if reeval_canonical != run_path:
        return run_path
    if (Path(run_path) / "eval_bundles").is_dir():
        return run_path

    parts = Path(run_path).parts
    mapped_parts = []
    replaced = False
    for part in parts:
        if part.startswith("probes_v13_") and ("inf_mcqa_v12" in part or "inf_mcqa_v13" in part):
            mapped_parts.append(V13_LEGAL_INFERENCE_MCQA_TAG)
            replaced = True
        else:
            mapped_parts.append(part)
    if not replaced:
        raise ValueError(f"Could not map run path to v13 legal inference MCQA reeval: {run_path}")
    return str(Path(*mapped_parts))


def legal_domains_for(domains: Sequence[str]) -> List[str]:
    legal_root = REPO_ROOT / "probes" / "legal"
    legal_domains = {path.name for path in legal_root.iterdir() if path.is_dir()} if legal_root.is_dir() else set()
    return [domain for domain in domains if domain in legal_domains]


def aggregate_metric_frames(frames: Sequence[pd.DataFrame], metrics: Sequence[str]) -> Optional[pd.DataFrame]:
    present = [df for df in frames if df is not None and not df.empty]
    if not present:
        return None
    combined = pd.concat(present, ignore_index=True)
    available_metrics = [metric for metric in metrics if metric in combined.columns]
    if not available_metrics or "step" not in combined.columns:
        return None
    combined = combined.copy()
    for metric in available_metrics:
        combined[metric] = pd.to_numeric(combined[metric], errors="coerce")
    combined.dropna(subset=available_metrics, how="all", inplace=True)
    if combined.empty:
        return None
    aggregated = combined.groupby("step")[available_metrics].mean().reset_index()
    aggregated["step"] = aggregated["step"].astype(int)
    aggregated.sort_values("step", inplace=True)
    return aggregated


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
    model: str,
    probe_type: str,
    domains: Sequence[str],
    mcqa_variant: str,
    reviewed_fallback: str,
    use_reeval: bool = True,
    inference_mcqa_reeval: Optional[str] = None,
    reeval_dir: str = "reeval",
    classic_metric_file_variant: str = "default",
):
    classic_run_path = (
        reeval_mcqa_run_path(
            run_path,
            model,
            probe_type,
            reeval_dir=reeval_dir,
            probe_family="classic",
        )
        if use_reeval
        else run_path
    )
    classic = load_metrics(
        classic_run_path,
        probe_type,
        domains,
        str(REPO_ROOT),
        metrics=("log_prob",),
        probe_family="classic",
        metric_file_variant=classic_metric_file_variant,
    )
    if classic is None and classic_run_path != run_path:
        classic = load_metrics(
            run_path,
            probe_type,
            domains,
            str(REPO_ROOT),
            metrics=("log_prob",),
            probe_family="classic",
            metric_file_variant=classic_metric_file_variant,
        )
    actual_variant = mcqa_variant
    mcqa_base_run_path = (
        inference_mcqa_reeval_run_path(run_path, inference_mcqa_reeval, reeval_dir=reeval_dir)
        if probe_type == "inference"
        else run_path
    )
    mcqa_run_path = mcqa_run_path_for_variant(
        mcqa_base_run_path,
        model,
        probe_type,
        mcqa_variant,
        use_reeval=use_reeval,
        reeval_dir=reeval_dir,
    )
    if mcqa_variant == "reviewed" and not has_reviewed_mcqa_folder(mcqa_run_path, probe_type):
        if reviewed_fallback == "regular":
            actual_variant = "regular"
            regular_run_path = regular_mcqa_run_path(mcqa_base_run_path)
            mcqa_run_path = reeval_mcqa_run_path(
                regular_run_path,
                model,
                probe_type,
                reeval_dir=reeval_dir,
                probe_family="mcqa",
            ) if use_reeval else regular_run_path
        elif reviewed_fallback == "drop":
            return classic, None, "dropped", mcqa_run_path
        else:
            raise FileNotFoundError(f"No reviewed {probe_type} MCQA folders found under {run_path}")

    mcqa = load_metrics(
        mcqa_run_path,
        probe_type,
        domains,
        str(REPO_ROOT),
        metrics=("mcqa_accuracy",),
        probe_family="mcqa",
        mcqa_variant=actual_variant,
    )
    if mcqa is None and mcqa_run_path != mcqa_base_run_path:
        fallback_mcqa = load_metrics(
            mcqa_base_run_path,
            probe_type,
            domains,
            str(REPO_ROOT),
            metrics=("mcqa_accuracy",),
            probe_family="mcqa",
            mcqa_variant=actual_variant,
        )
        if fallback_mcqa is not None:
            return classic, fallback_mcqa, actual_variant, mcqa_base_run_path
    if mcqa is None and actual_variant == "reviewed":
        if reviewed_fallback == "regular":
            regular_run_path = regular_mcqa_run_path(mcqa_base_run_path)
            mcqa_run_path = reeval_mcqa_run_path(
                regular_run_path,
                model,
                probe_type,
                reeval_dir=reeval_dir,
                probe_family="mcqa",
            ) if use_reeval else regular_run_path
            mcqa = load_metrics(
                mcqa_run_path,
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
    return classic, mcqa, actual_variant, mcqa_run_path


def load_mixed_v13_inference_mcqa(
    run_path: str,
    model: str,
    domains: Sequence[str],
    mcqa_variant: str,
    use_reeval: bool,
    reeval_dir: str = "reeval",
):
    legal_domains = legal_domains_for(domains)
    generic_domains = [domain for domain in domains if domain not in set(legal_domains)]
    frames = []
    paths = []

    if generic_domains:
        generic_path = mcqa_run_path_for_variant(
            run_path,
            model,
            "inference",
            mcqa_variant,
            use_reeval=use_reeval,
            reeval_dir=reeval_dir,
        )
        generic = load_metrics(
            generic_path,
            "inference",
            generic_domains,
            str(REPO_ROOT),
            metrics=("mcqa_accuracy",),
            aggregate=False,
            probe_family="mcqa",
            mcqa_variant=mcqa_variant,
        )
        if generic is not None:
            frames.append(generic)
            paths.append(find_latest_run(generic_path))

    if legal_domains:
        legal_base_path = inference_mcqa_reeval_run_path(run_path, "v13_legal", reeval_dir=reeval_dir)
        legal_path = mcqa_run_path_for_variant(
            legal_base_path,
            model,
            "inference",
            mcqa_variant,
            use_reeval=use_reeval,
            reeval_dir=reeval_dir,
        )
        legal = load_metrics(
            legal_path,
            "inference",
            legal_domains,
            str(REPO_ROOT),
            metrics=("mcqa_accuracy",),
            aggregate=False,
            probe_family="mcqa",
            mcqa_variant=mcqa_variant,
        )
        if legal is not None:
            frames.append(legal)
            paths.append(find_latest_run(legal_path))

    return aggregate_metric_frames(frames, ("mcqa_accuracy",)), paths


def load_configured_values(
    domains: Sequence[str],
    mcqa_variant: str,
    reviewed_fallback: str,
    use_reeval: bool = True,
    inference_mcqa_reeval: Optional[str] = None,
    reeval_dir: str = "reeval",
    value_mode: str = "final",
    factual_probe_variant: str = "canonical",
):
    if value_mode not in VALUE_MODE_CHOICES:
        raise ValueError(f"Unsupported value mode: {value_mode}")
    if factual_probe_variant not in FACTUAL_PROBE_VARIANT_CHOICES:
        raise ValueError(f"Unsupported factual probe variant: {factual_probe_variant}")
    values: Dict[Tuple[str, str, str], dict] = {}
    for method, model, run_path in iter_run_items(mcqa_variant):
        for probe_type, _ in PROBE_TYPES:
            classic_baseline = None
            mcqa_baseline = None
            classic_metric_file_variant = (
                "paraphrased"
                if probe_type == "knowledge" and factual_probe_variant == "paraphrased"
                else "default"
            )
            if probe_type == "inference" and inference_mcqa_reeval == "v13_mixed":
                classic, _, actual_variant, mcqa_run_path = load_probe_pair(
                    run_path,
                    model,
                    probe_type,
                    domains,
                    mcqa_variant=mcqa_variant,
                    reviewed_fallback=reviewed_fallback,
                    use_reeval=use_reeval,
                    inference_mcqa_reeval=None,
                    reeval_dir=reeval_dir,
                    classic_metric_file_variant=classic_metric_file_variant,
                )
                mcqa, mcqa_run_path = load_mixed_v13_inference_mcqa(
                    run_path,
                    model,
                    domains,
                    mcqa_variant=mcqa_variant,
                    use_reeval=use_reeval,
                    reeval_dir=reeval_dir,
                )
                if value_mode == "delta":
                    classic_baseline = load_metrics(
                        run_path,
                        probe_type,
                        domains,
                        str(REPO_ROOT),
                        metrics=("log_prob",),
                        probe_family="classic",
                        metric_file_variant=classic_metric_file_variant,
                    )
                    mcqa_baseline, _ = load_mixed_v13_inference_mcqa(
                        run_path,
                        model,
                        domains,
                        mcqa_variant=mcqa_variant,
                        use_reeval=False,
                        reeval_dir=reeval_dir,
                    )
            else:
                classic, mcqa, actual_variant, mcqa_run_path = load_probe_pair(
                    run_path,
                    model,
                    probe_type,
                    domains,
                    mcqa_variant=mcqa_variant,
                    reviewed_fallback=reviewed_fallback,
                    use_reeval=use_reeval,
                    inference_mcqa_reeval=inference_mcqa_reeval,
                    reeval_dir=reeval_dir,
                    classic_metric_file_variant=classic_metric_file_variant,
                )
                if value_mode == "delta":
                    classic_baseline = load_metrics(
                        run_path,
                        probe_type,
                        domains,
                        str(REPO_ROOT),
                        metrics=("log_prob",),
                        probe_family="classic",
                        metric_file_variant=classic_metric_file_variant,
                    )
                    mcqa_baseline_path = (
                        inference_mcqa_reeval_run_path(run_path, inference_mcqa_reeval, reeval_dir=reeval_dir)
                        if probe_type == "inference"
                        else run_path
                    )
                    if actual_variant != "dropped":
                        mcqa_baseline = load_metrics(
                            mcqa_baseline_path,
                            probe_type,
                            domains,
                            str(REPO_ROOT),
                            metrics=("mcqa_accuracy",),
                            probe_family="mcqa",
                            mcqa_variant=actual_variant,
                        )
            values[(probe_type, method, model)] = {
                "log_prob": metric_value(classic, "log_prob", value_mode, classic_baseline),
                "mcqa_accuracy": metric_value(mcqa, "mcqa_accuracy", value_mode, mcqa_baseline),
                "mcqa_variant": actual_variant,
                "run_path": find_latest_run(run_path),
                "mcqa_run_path": (
                    [path for path in mcqa_run_path if path]
                    if isinstance(mcqa_run_path, list)
                    else find_latest_run(mcqa_run_path)
                ),
            }
    return values


def delta_value(df, metric: str) -> Optional[float]:
    if df is None or df.empty or "step" not in df.columns or metric not in df.columns:
        return None
    valid = df[["step", metric]].dropna()
    if valid.empty:
        return None
    if valid["step"].nunique() < 2:
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
        final = final_value(df, metric)
        initial = initial_value(baseline_df, metric)
        if final is None or initial is None:
            return None
        return float(final - initial)
    raise ValueError(f"Unsupported value mode: {value_mode}")


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
    axes[1].legend(
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
    parser.add_argument("--output", default="plots/probe_scaling_by_model")
    parser.add_argument("--domains", nargs="+")
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
    run_paths = [path for _, _, path in iter_run_items(args.mcqa_variant)]
    domains = args.domains or discover_domains_from_runs(run_paths)
    if not domains:
        raise RuntimeError("No domains found in configured result folders")

    print(f"Using {len(domains)} domains")
    values = load_configured_values(
        domains,
        mcqa_variant=args.mcqa_variant,
        reviewed_fallback=args.reviewed_fallback,
        use_reeval=not args.no_reeval,
        inference_mcqa_reeval=args.inference_mcqa_reeval,
        reeval_dir=args.reeval_dir,
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
