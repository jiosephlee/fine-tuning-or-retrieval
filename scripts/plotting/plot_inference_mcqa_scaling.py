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
    "para9": "Para 9",
    "with_explanations": "With Explanations",
}
MODEL_LABELS = {
    "1b": "1B",
    "7b": "7B",
    "13b": "13B",
    "32b": "32B",
}

DEFAULT_RUNS: Dict[str, Dict[str, str]] = {
    "source_only": {
        "1b": "results/FT/full/1b/probes_v13_para_v13_paraphrased_inf_v11_reviewed_mcqa_v14_prompt_formatted_question_5shot_inf_mcqa_v12/newline2/source_only_docmatch_expl/fill_dclm/domains_arxiv_all-legal_all-medical_all/e100/bs256_lr4e-05/overlap_1_16/E4_source_1b_all_domains_fa2_packing",
        "7b": "results/FT/full/7b/probes_v13_para_v13_paraphrased_inf_v11_reviewed_mcqa_v14_prompt_formatted_question_5shot_inf_mcqa_v12/newline2/source_only_docmatch_expl/fill_dclm/domains_arxiv_all-legal_all-medical_all/e100/bs256_lr4e-05/overlap_1_16/E1_source_all_domains_fa2_packing",
        "13b": "results/FT/full/allenai_OLMo-2-1124-13B/probes_v13_para_v13_paraphrased_inf_v11_reviewed_mcqa_v14_prompt_formatted_question_5shot_inf_mcqa_v12_reviewed/newline2/source_only_docmatch_expl/fill_dclm/domains_arxiv_all-legal_all-medical_all/e100/bs256_lr4e-05/overlap_1_16/E7_source_13b_all_domains_fa2_packing",
        "32b": "results/FT/full/allenai_OLMo-2-0325-32B/probes_v13_para_v13_paraphrased_inf_v11_reviewed_mcqa_v14_prompt_formatted_question_5shot_inf_mcqa_v12_reviewed+v12/newline2/source_only_docmatch_expl/fill_dclm/domains_arxiv_all-legal_all-medical_all/e100/bs256_lr4e-05/overlap_1_16/E10_source_32b_all_domains_fa2_packing_chunked_nll_1gpu_device_map_auto_36h_nodelta",
    },
    "para9": {
        "1b": "results/FT/full/1b/probes_v13_para_v13_paraphrased_inf_v11_reviewed_mcqa_v14_prompt_formatted_question_5shot_inf_mcqa_v12/newline2/para9_docmatch_expl/fill_dclm/domains_arxiv_all-legal_all-medical_all/e100/bs256_lr4e-05/overlap_1_16/E5_paraphrase_1b_all_domains",
        "7b": "results/FT/full/7b/probes_v13_para_v13_paraphrased_inf_v11_reviewed_mcqa_v14_prompt_formatted_question_5shot_inf_mcqa_v12/newline2/para9_docmatch_expl/fill_dclm/domains_arxiv_all-legal_all-medical_all/e100/bs256_lr4e-05/overlap_1_16/E2_paraphrase_all_domains",
        "13b": "results/FT/full/allenai_OLMo-2-1124-13B/probes_v13_para_v13_paraphrased_inf_v11_reviewed_mcqa_v14_prompt_formatted_question_5shot_inf_mcqa_v12_reviewed/newline2/para9_docmatch_expl/fill_dclm/domains_arxiv_all-legal_all-medical_all/e100/bs256_lr4e-05/overlap_1_16/E8_paraphrase_13b_all_domains",
        "32b": "results/FT/full/allenai_OLMo-2-0325-32B/probes_v13_para_v13_paraphrased_inf_v11_reviewed_mcqa_v14_prompt_formatted_question_5shot_inf_mcqa_v12_reviewed+v12/newline2/para9_docmatch_expl/fill_dclm/domains_arxiv_all-legal_all-medical_all/e100/bs256_lr4e-05/overlap_1_16/E11_paraphrase_32b_all_domains_chunked_nll_1gpu_device_map_auto_36h_nodelta",
    },
    "with_explanations": {
        "1b": "results/FT/full/1b/probes_v13_para_v13_paraphrased_inf_v11_reviewed_mcqa_v14_prompt_formatted_question_5shot_inf_mcqa_v12/newline2/para9_expl_textbooks+stackexchange+blogs_cyclefull/fill_dclm/domains_arxiv_all-legal_all-medical_all/e100/bs256_lr4e-05/overlap_1_16/E6_granular_explanations_1b_all_domains",
        "7b": "results/FT/full/7b/probes_v13_para_v13_paraphrased_inf_v11_reviewed_mcqa_v14_prompt_formatted_question_5shot_inf_mcqa_v12/newline2/para9_expl_textbooks+stackexchange+blogs_cyclefull/fill_dclm/domains_arxiv_all-legal_all-medical_all/e100/bs256_lr4e-05/overlap_1_16/E3_granular_explanations_all_domains",
        "13b": "results/FT/full/allenai_OLMo-2-1124-13B/probes_v13_para_v13_paraphrased_inf_v11_reviewed_mcqa_v14_prompt_formatted_question_5shot_inf_mcqa_v12_reviewed/newline2/para9_expl_textbooks+stackexchange+blogs_cyclefull/fill_dclm/domains_arxiv_all-legal_all-medical_all/e100/bs256_lr4e-05/overlap_1_16/E9_granular_explanations_13b_all_domains",
        "32b": "results/FT/full/allenai_OLMo-2-0325-32B/probes_v13_para_v13_paraphrased_inf_v11_reviewed_mcqa_v14_prompt_formatted_question_5shot_inf_mcqa_v12_reviewed+v12/newline2/para9_expl_textbooks+stackexchange+blogs_cyclefull/fill_dclm/domains_arxiv_all-legal_all-medical_all/e100/bs256_lr4e-05/overlap_1_16/E12_granular_explanations_32b_all_domains_1gpu_device_map_auto_36h_nodelta",
    },
}


def _abs_path(path: str) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return str(candidate)


def regular_mcqa_run_path(path: str) -> str:
    return path.replace("_inf_mcqa_v12_reviewed/newline2", "_inf_mcqa_v12/newline2")


def run_path_for_variant(path: str, mcqa_variant: str) -> str:
    if mcqa_variant == "regular":
        return regular_mcqa_run_path(path)
    return path


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


def _load_mcqa(
    run_path: str,
    domains: Sequence[str],
    mcqa_variant: str,
    reviewed_fallback: str,
):
    if mcqa_variant == "reviewed" and not _has_reviewed_mcqa(run_path):
        if reviewed_fallback == "regular":
            mcqa_variant = "regular"
            run_path = regular_mcqa_run_path(run_path)
        elif reviewed_fallback == "drop":
            return None, "dropped"
        else:
            raise FileNotFoundError(f"No reviewed MCQA folders found under {run_path}")

    df = load_metrics(
        run_path,
        "inference",
        domains,
        str(REPO_ROOT),
        metrics=("mcqa_accuracy",),
        probe_family="mcqa",
        mcqa_variant=mcqa_variant,
    )
    if df is not None or mcqa_variant != "reviewed":
        return df, mcqa_variant
    if reviewed_fallback == "regular":
        run_path = regular_mcqa_run_path(run_path)
        return load_metrics(
            run_path,
            "inference",
            domains,
            str(REPO_ROOT),
            metrics=("mcqa_accuracy",),
            probe_family="mcqa",
            mcqa_variant="regular",
        ), "regular"
    if reviewed_fallback == "drop":
        return None, "dropped"
    raise FileNotFoundError(f"No reviewed MCQA metrics found under {run_path}")


def load_configured_series(
    domains: Sequence[str],
    mcqa_variant: str = "regular",
    reviewed_fallback: str = "regular",
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
        mcqa, actual_variant = _load_mcqa(
            run_path,
            domains,
            mcqa_variant=mcqa_variant,
            reviewed_fallback=reviewed_fallback,
        )
        series[(method, model)] = {
            "classic": classic,
            "mcqa": mcqa,
            "mcqa_variant": actual_variant,
            "run_path": find_latest_run(run_path),
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
