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

from scripts.plotting.plot_utils import (  # noqa: E402
    COLORS,
    _is_reviewed_inference_mcqa_root,
    apply_ylim,
    compute_unified_ylim,
    find_latest_run,
    load_metrics,
    setup_style,
)


METHODS = ("source_only", "para9", "with_explanations")
METHOD_LABELS = {
    "source_only": "Source Only",
    "para9": "Paraphrased",
    "with_explanations": "With Explanations",
}
MODEL_LABELS = {
    "1b": "1B",
    "7b": "7B",
    "13b": "13B",
    "32b": "32B",
}
REEVAL_DIR_CHOICES = ("reeval", "reeval_v1", "reeval_v2")

DEFAULT_RUNS: Dict[str, Dict[str, str]] = {
    "source_only": {
        "1b": "results/FT/full/1b/source_only_docmatch_expl/fill_dclm/domains_arxiv_all-legal_all-medical_all/e100/bs256_lr4e-05/overlap_1_16/E4_source_1b_all_domains_fa2_packing",
        "7b": "results/FT/full/7b/source_only_docmatch_expl/fill_dclm/domains_arxiv_all-legal_all-medical_all/e100/bs256_lr4e-05/overlap_1_16/E1_source_all_domains_fa2_packing",
        "13b": "results/FT/full/13b/source_only_docmatch_expl/fill_dclm/domains_arxiv_all-legal_all-medical_all/e100/bs256_lr4e-05/overlap_1_16/E7_source_13b_all_domains_fa2_packing",
        "32b": "results/FT/full/32b/source_only_docmatch_expl/fill_dclm/domains_arxiv_all-legal_all-medical_all/e100/bs256_lr4e-05/overlap_1_16/E10_source_32b_all_domains_fa2_packing_chunked_nll_1gpu_device_map_auto_36h_nodelta",
    },
    "para9": {
        "1b": "results/FT/full/1b/para9_docmatch_expl/fill_dclm/domains_arxiv_all-legal_all-medical_all/e100/bs256_lr4e-05/overlap_1_16/E5_paraphrase_1b_all_domains",
        "7b": "results/FT/full/7b/para9_docmatch_expl/fill_dclm/domains_arxiv_all-legal_all-medical_all/e100/bs256_lr4e-05/overlap_1_16/E2_paraphrase_all_domains",
        "13b": "results/FT/full/13b/para9_docmatch_expl/fill_dclm/domains_arxiv_all-legal_all-medical_all/e100/bs256_lr4e-05/overlap_1_16/E8_paraphrase_13b_all_domains",
        "32b": "results/FT/full/32b/para9_docmatch_expl/fill_dclm/domains_arxiv_all-legal_all-medical_all/e100/bs256_lr4e-05/overlap_1_16/E11_paraphrase_32b_all_domains_chunked_nll_1gpu_device_map_auto_36h_nodelta",
    },
    "with_explanations": {
        "1b": "results/FT/full/1b/para9_expl_textbooks+stackexchange+blogs_cyclefull/fill_dclm/domains_arxiv_all-legal_all-medical_all/e100/bs256_lr4e-05/overlap_1_16/E6_granular_explanations_1b_all_domains",
        "7b": "results/FT/full/7b/para9_expl_textbooks+stackexchange+blogs_cyclefull/fill_dclm/domains_arxiv_all-legal_all-medical_all/e100/bs256_lr4e-05/overlap_1_16/E3_granular_explanations_all_domains",
        "13b": "results/FT/full/13b/para9_expl_textbooks+stackexchange+blogs_cyclefull/fill_dclm/domains_arxiv_all-legal_all-medical_all/e100/bs256_lr4e-05/overlap_1_16/E9_granular_explanations_13b_all_domains",
        "32b": "results/FT/full/32b/para9_expl_textbooks+stackexchange+blogs_cyclefull/fill_dclm/domains_arxiv_all-legal_all-medical_all/e100/bs256_lr4e-05/overlap_1_16/E12_granular_explanations_32b_all_domains_1gpu_device_map_auto_36h_nodelta",
    },
}


def _abs_path(path: str) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return str(candidate)


def canonical_run_root(path: str) -> Optional[Path]:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    parts = candidate.parts
    if "eval_bundles" in parts:
        index = parts.index("eval_bundles")
        return Path(*parts[:index])
    if (candidate / "eval_bundles").is_dir():
        return candidate
    return None


def eval_bundle_path(path: str, bundle_names: Sequence[str], require_leaf: bool = True) -> str:
    root = canonical_run_root(path)
    if root is None:
        return path
    for bundle_name in bundle_names:
        candidate = root / "eval_bundles" / bundle_name
        if candidate.is_dir() and (not require_leaf or find_latest_run(str(candidate))):
            return str(candidate)
    return path


def regular_mcqa_run_path(path: str) -> str:
    canonical = eval_bundle_path(path, ("inf_mcqa_v12",))
    if canonical != path:
        return canonical
    return (
        path
        .replace("_inf_mcqa_v13+v12/newline2", "_inf_mcqa_v12/newline2")
        .replace("_inf_mcqa_v13/newline2", "_inf_mcqa_v12/newline2")
        .replace("_inf_mcqa_v12_reviewed+v12/newline2", "_inf_mcqa_v12/newline2")
        .replace("_inf_mcqa_v12_reviewed/newline2", "_inf_mcqa_v12/newline2")
    )


def reviewed_mcqa_run_path(path: str, reeval_dir: Optional[str] = None) -> str:
    reviewed_bundles = (
        "inf_mcqa_v13+v12",
        "inf_mcqa_v13",
        "inf_mcqa_v12_reviewed+v12",
        "inf_mcqa_v12_reviewed",
    )
    root = canonical_run_root(path)
    if root is not None:
        if reeval_dir:
            for bundle_name in reviewed_bundles:
                candidate = root / "eval_bundles" / bundle_name
                if candidate.is_dir() and find_latest_run(str(candidate / reeval_dir)):
                    return str(candidate)
        for bundle_name in reviewed_bundles:
            candidate = root / "eval_bundles" / bundle_name
            if candidate.is_dir() and find_latest_run(str(candidate)):
                return str(candidate)
        for bundle_name in reviewed_bundles:
            candidate = root / "eval_bundles" / bundle_name
            if candidate.is_dir():
                return str(candidate)
        return path
    return (
        path
        .replace("_inf_mcqa_v12_reviewed+v12/newline2", "_inf_mcqa_v13+v12/newline2")
        .replace("_inf_mcqa_v12_reviewed/newline2", "_inf_mcqa_v13/newline2")
        .replace("_inf_mcqa_v12/newline2", "_inf_mcqa_v13/newline2")
    )


def legacy_reviewed_mcqa_run_paths(path: str, model: Optional[str] = None) -> List[str]:
    candidates = [
        path
        .replace("_inf_mcqa_v13_legal/newline2", "_inf_mcqa_v12_reviewed/newline2")
        .replace("_inf_mcqa_v13+v12/newline2", "_inf_mcqa_v12_reviewed+v12/newline2")
        .replace("_inf_mcqa_v13/newline2", "_inf_mcqa_v12_reviewed/newline2")
        .replace("_inf_mcqa_v12/newline2", "_inf_mcqa_v12_reviewed/newline2"),
        path
        .replace("_inf_mcqa_v13_legal/newline2", "_inf_mcqa_v12_reviewed+v12/newline2")
        .replace("_inf_mcqa_v13+v12/newline2", "_inf_mcqa_v12_reviewed+v12/newline2")
        .replace("_inf_mcqa_v13/newline2", "_inf_mcqa_v12_reviewed+v12/newline2")
        .replace("_inf_mcqa_v12/newline2", "_inf_mcqa_v12_reviewed+v12/newline2"),
    ]
    if model == "32b":
        candidates = list(reversed(candidates))
    unique = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def run_path_for_variant(path: str, mcqa_variant: str) -> str:
    if mcqa_variant == "regular":
        return regular_mcqa_run_path(path)
    direct_reviewed = eval_bundle_path(
        path,
        (
            "inf_mcqa_v13+v12",
            "inf_mcqa_v13",
            "inf_mcqa_v12_reviewed+v12",
            "inf_mcqa_v12_reviewed",
        ),
    )
    if direct_reviewed != path:
        return direct_reviewed
    return regular_mcqa_run_path(path)


def reeval_mcqa_run_path(
    run_path: str,
    model: Optional[str],
    probe_type: str = "inference",
    reeval_dir: str = "reeval",
) -> str:
    """Use final-model re-evaluation outputs for inference MCQA when present."""
    if reeval_dir not in REEVAL_DIR_CHOICES:
        raise ValueError(f"Unsupported re-eval directory: {reeval_dir}")
    if probe_type != "inference":
        return run_path
    candidates = [run_path]
    if canonical_run_root(run_path) is None:
        candidates.extend(legacy_reviewed_mcqa_run_paths(run_path, model))
    for candidate in candidates:
        resolved = find_latest_run(candidate) or (candidate if os.path.isdir(candidate) else None)
        if not resolved:
            continue
        reeval_path = os.path.join(resolved, reeval_dir)
        if find_latest_run(reeval_path):
            return reeval_path
    return run_path


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


def _has_reviewed_mcqa(run_path: str) -> bool:
    resolved = find_latest_run(run_path)
    if not resolved:
        return False
    if _is_reviewed_inference_mcqa_root(resolved):
        return any(
            "_inference_mcqa_probe" in name and os.path.isdir(os.path.join(resolved, name))
            for name in os.listdir(resolved)
        )
    return any(
        name.endswith("_reviewed")
        and "_inference_mcqa_probe" in name
        and os.path.isdir(os.path.join(resolved, name))
        for name in os.listdir(resolved)
    )


def mcqa_run_path_for_variant(
    run_path: str,
    model: Optional[str],
    probe_type: str,
    mcqa_variant: str,
    use_reeval: bool = True,
    reeval_dir: str = "reeval",
) -> str:
    base_run_path = run_path
    if mcqa_variant in {"reviewed", "preferred"}:
        reviewed_path = reviewed_mcqa_run_path(run_path, reeval_dir=reeval_dir if use_reeval else None)
        if os.path.isdir(reviewed_path):
            base_run_path = reviewed_path
    mcqa_run_path = (
        reeval_mcqa_run_path(base_run_path, model, probe_type, reeval_dir=reeval_dir)
        if use_reeval
        else base_run_path
    )
    if (
        mcqa_variant == "regular"
        and mcqa_run_path != base_run_path
        and _is_reviewed_inference_mcqa_root(mcqa_run_path)
        and _has_reviewed_mcqa(mcqa_run_path)
    ):
        return base_run_path
    if (
        mcqa_variant in {"reviewed", "preferred"}
        and mcqa_run_path != base_run_path
        and not _has_reviewed_mcqa(mcqa_run_path)
        and _has_reviewed_mcqa(base_run_path)
    ):
        return base_run_path
    return mcqa_run_path


def _load_mcqa(
    run_path: str,
    domains: Sequence[str],
    mcqa_variant: str,
    reviewed_fallback: str,
    model: Optional[str] = None,
    use_reeval: bool = True,
    reeval_dir: str = "reeval",
):
    mcqa_run_path = mcqa_run_path_for_variant(
        run_path,
        model,
        "inference",
        mcqa_variant,
        use_reeval=use_reeval,
        reeval_dir=reeval_dir,
    )
    if mcqa_variant == "reviewed" and not _has_reviewed_mcqa(mcqa_run_path):
        if reviewed_fallback == "regular":
            mcqa_variant = "regular"
            regular_run_path = regular_mcqa_run_path(run_path)
            mcqa_run_path = reeval_mcqa_run_path(
                regular_run_path,
                model,
                "inference",
                reeval_dir=reeval_dir,
            ) if use_reeval else regular_run_path
        elif reviewed_fallback == "drop":
            return None, "dropped", mcqa_run_path
        else:
            raise FileNotFoundError(f"No reviewed MCQA folders found under {run_path}")

    df = load_metrics(
        mcqa_run_path,
        "inference",
        domains,
        str(REPO_ROOT),
        metrics=("mcqa_accuracy",),
        probe_family="mcqa",
        mcqa_variant=mcqa_variant,
    )
    if df is None and mcqa_run_path != run_path:
        fallback_df = load_metrics(
            run_path,
            "inference",
            domains,
            str(REPO_ROOT),
            metrics=("mcqa_accuracy",),
            probe_family="mcqa",
            mcqa_variant=mcqa_variant,
        )
        if fallback_df is not None:
            return fallback_df, mcqa_variant, run_path
    if df is not None or mcqa_variant != "reviewed":
        return df, mcqa_variant, mcqa_run_path
    if reviewed_fallback == "regular":
        regular_run_path = regular_mcqa_run_path(run_path)
        mcqa_run_path = reeval_mcqa_run_path(
            regular_run_path,
            model,
            "inference",
            reeval_dir=reeval_dir,
        ) if use_reeval else regular_run_path
        return load_metrics(
            mcqa_run_path,
            "inference",
            domains,
            str(REPO_ROOT),
            metrics=("mcqa_accuracy",),
            probe_family="mcqa",
            mcqa_variant="regular",
        ), "regular", mcqa_run_path
    if reviewed_fallback == "drop":
        return None, "dropped", mcqa_run_path
    raise FileNotFoundError(f"No reviewed MCQA metrics found under {run_path}")


def load_configured_series(
    domains: Sequence[str],
    mcqa_variant: str = "regular",
    reviewed_fallback: str = "regular",
    use_reeval: bool = True,
    reeval_dir: str = "reeval",
):
    series = {}
    for method, model, run_path in iter_run_items(mcqa_variant):
        classic = load_metrics(
            run_path,
            "inference",
            domains,
            str(REPO_ROOT),
            metrics=("log_prob",),
            probe_family="classic",
        )
        mcqa, actual_variant, mcqa_run_path = _load_mcqa(
            run_path,
            domains,
            mcqa_variant=mcqa_variant,
            reviewed_fallback=reviewed_fallback,
            model=model,
            use_reeval=use_reeval,
            reeval_dir=reeval_dir,
        )
        series[(method, model)] = {
            "classic": classic,
            "mcqa": mcqa,
            "mcqa_variant": actual_variant,
            "run_path": find_latest_run(run_path),
            "mcqa_run_path": find_latest_run(mcqa_run_path),
        }
    return series


def apply_plot_style():
    try:
        setup_style()
    except ModuleNotFoundError as exc:
        if exc.name != "seaborn":
            raise
        plt.rcParams.update({
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "font.size": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        })


def save_figure(fig, output: str):
    output_path = Path(output)
    if output_path.suffix:
        output_path = output_path.with_suffix("")
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        fig.savefig(str(output_path) + suffix, bbox_inches="tight", dpi=300)
        print(f"Saved {output_path}{suffix}")
    plt.close(fig)


def final_value(df, metric: str) -> Optional[float]:
    if df is None or df.empty or "step" not in df.columns or metric not in df.columns:
        return None
    valid = df[["step", metric]].dropna()
    if valid.empty:
        return None
    max_step = valid["step"].max()
    return float(valid.loc[valid["step"] == max_step, metric].mean())


def plot_scaling(series, output: str):
    apply_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=False)
    right_axes = [ax.twinx() for ax in axes]
    left_values = []
    right_values = []

    for col, method in enumerate(METHODS):
        ax = axes[col]
        ax_right = right_axes[col]
        for model, model_label in MODEL_LABELS.items():
            color = COLORS["model"][model]
            item = series[(method, model)]
            classic = item["classic"]
            mcqa = item["mcqa"]
            if classic is not None and not classic.empty:
                ax.plot(
                    classic["step"],
                    classic["log_prob"],
                    color=color,
                    linewidth=1.8,
                    linestyle="-",
                )
                left_values.extend(classic["log_prob"].dropna().tolist())
            if mcqa is not None and not mcqa.empty:
                ax_right.plot(
                    mcqa["step"],
                    mcqa["mcqa_accuracy"],
                    color=color,
                    linewidth=1.8,
                    linestyle="--",
                )
                right_values.extend(mcqa["mcqa_accuracy"].dropna().tolist())

        ax.set_title(METHOD_LABELS[method])
        ax.set_xlabel("Training Step")
        ax.grid(True, alpha=0.25)
        if col == 0:
            ax.set_ylabel("Inference Log Prob")
        else:
            ax.set_ylabel("")
        if col == len(METHODS) - 1:
            ax_right.set_ylabel("MCQA Accuracy")
        else:
            ax_right.set_ylabel("")

    left_ylim = compute_unified_ylim(left_values, padding=0.05)
    right_ylim = compute_unified_ylim(right_values, padding=0.05)
    if left_ylim:
        apply_ylim(axes, left_ylim)
    if right_ylim:
        apply_ylim(right_axes, right_ylim)

    model_handles = [
        Line2D([0], [0], color=COLORS["model"][model], linewidth=2, label=label)
        for model, label in MODEL_LABELS.items()
    ]
    metric_handles = [
        Line2D([0], [0], color="black", linewidth=2, linestyle="-", label="Log Prob"),
        Line2D([0], [0], color="black", linewidth=2, linestyle="--", label="MCQA Accuracy"),
    ]
    fig.legend(
        handles=model_handles + metric_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=6,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_figure(fig, output)


def plot_final_value_only(series, output: str):
    apply_plot_style()
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.8))
    ax_right = ax.twinx()
    x = np.arange(len(METHODS))
    left_values = []
    right_values = []

    for model, model_label in MODEL_LABELS.items():
        color = COLORS["model"][model]
        log_prob_values = []
        mcqa_values = []
        for method in METHODS:
            item = series[(method, model)]
            log_prob = final_value(item["classic"], "log_prob")
            mcqa = final_value(item["mcqa"], "mcqa_accuracy")
            log_prob_values.append(np.nan if log_prob is None else log_prob)
            mcqa_values.append(np.nan if mcqa is None else mcqa)
        left_values.extend([v for v in log_prob_values if not np.isnan(v)])
        right_values.extend([v for v in mcqa_values if not np.isnan(v)])
        ax.plot(
            x,
            log_prob_values,
            color=color,
            linewidth=2,
            marker="o",
            linestyle="-",
        )
        ax_right.plot(
            x,
            mcqa_values,
            color=color,
            linewidth=2,
            marker="o",
            linestyle="--",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[method] for method in METHODS])
    ax.set_ylabel("Final Inference Log Prob")
    ax_right.set_ylabel("Final MCQA Accuracy")
    ax.grid(True, axis="y", alpha=0.25)

    left_ylim = compute_unified_ylim(left_values, padding=0.05)
    right_ylim = compute_unified_ylim(right_values, padding=0.05)
    if left_ylim:
        ax.set_ylim(left_ylim)
    if right_ylim:
        ax_right.set_ylim(right_ylim)

    model_handles = [
        Line2D([0], [0], color=COLORS["model"][model], linewidth=2, marker="o", label=label)
        for model, label in MODEL_LABELS.items()
    ]
    metric_handles = [
        Line2D([0], [0], color="black", linewidth=2, linestyle="-", label="Log Prob"),
        Line2D([0], [0], color="black", linewidth=2, linestyle="--", label="MCQA Accuracy"),
    ]
    fig.legend(
        handles=model_handles + metric_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=6,
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
    parser.add_argument("--output", default="plots/inference_mcqa_scaling")
    parser.add_argument("--domains", nargs="+")
    parser.add_argument(
        "--mode",
        choices=("trajectory", "final_value_only"),
        default="trajectory",
    )
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
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    run_paths = [path for _, _, path in iter_run_items(args.mcqa_variant)]
    domains = args.domains or discover_domains_from_runs(run_paths)
    if not domains:
        raise RuntimeError("No domains found in configured result folders")

    print(f"Using {len(domains)} domains")
    series = load_configured_series(
        domains,
        mcqa_variant=args.mcqa_variant,
        reviewed_fallback=args.reviewed_fallback,
        use_reeval=not args.no_reeval,
        reeval_dir=args.reeval_dir,
    )

    missing = [
        (METHOD_LABELS[method], MODEL_LABELS[model], key)
        for (method, model), item in series.items()
        for key in ("classic", "mcqa")
        if item[key] is None or item[key].empty
    ]
    if missing:
        for method, model, key in missing:
            print(f"Warning: missing {key} data for {method} {model}")
    if args.mode == "final_value_only":
        plot_final_value_only(series, args.output)
    else:
        plot_scaling(series, args.output)


if __name__ == "__main__":
    main()
