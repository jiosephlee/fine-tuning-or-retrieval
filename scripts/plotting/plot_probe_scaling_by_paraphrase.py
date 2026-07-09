import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.plotting.plot_inference_mcqa_scaling import (  # noqa: E402
    MODEL_LABELS as BASE_MODEL_LABELS,
    save_figure,
)
from scripts.plotting.plot_probe_scaling_by_model import (  # noqa: E402
    FACTUAL_PROBE_VARIANT_CHOICES,
    PANELS,
    VALUE_MODE_CHOICES,
    discover_domains_from_runs,
    metric_value,
    panel_title,
    panel_ylabel,
)
from scripts.plotting.plot_utils import (  # noqa: E402
    COLORS,
    aggregate_across_domains,
    apply_plot_style as apply_shared_plot_style,
    compute_unified_ylim,
    find_latest_run,
    get_final_step_value,
    load_metrics,
)


MODELS = ("1b", "7b", "13b")
COMBINED_DEFAULT_MODELS = ("7b", "13b")
PARAPHRASE_LEVELS = ("source_only", "para4", "para9", "para24", "para49")
PARAPHRASE_LABELS = {
    "source_only": "0",
    "para4": "4",
    "para9": "9",
    "para24": "24",
    "para49": "49",
}
MODEL_RUNS: Dict[str, Dict[str, str]] = {
    "1b": {
        "para4": "results/FT/full/1b/para4/fill_dclm/domains_arxiv_all-legal_all-medical_all/e50/bs256_lr4e-05/overlap_1_16/E2_paraphrase_all_domains_local_no_explanation_match/eval_bundles/inf_mcqa_v14",
        "para9": "results/FT/full/1b/para9/fill_dclm/domains_arxiv_all-legal_all-medical_all/e50/bs256_lr4e-05/overlap_1_16/E2_paraphrase_all_domains_local_no_explanation_match/eval_bundles/inf_mcqa_v14",
        "para24": "results/FT/full/1b/para24/fill_dclm/domains_arxiv_all-legal_all-medical_all/e50/bs256_lr4e-05/overlap_1_16/E2_paraphrase_all_domains_local_no_explanation_match/eval_bundles/inf_mcqa_v14",
        "para49": "results/FT/full/1b/para49/fill_dclm/domains_arxiv_all-legal_all-medical_all/e50/bs256_lr4e-05/overlap_1_16/E2_paraphrase_all_domains_local_no_explanation_match/eval_bundles/inf_mcqa_v14",
    },
    "7b": {
        "source_only": "results/FT/full/7b/source_only/fill_dclm/domains_arxiv_all-legal_all-medical_all/e50/bs256_lr4e-05/overlap_1_16/E2_paraphrase_all_domains_local_no_explanation_match/eval_bundles/inf_mcqa_v14",
        "para4": "results/FT/full/7b/para4/fill_dclm/domains_arxiv_all-legal_all-medical_all/e50/bs256_lr4e-05/overlap_1_16/E2_paraphrase_all_domains_local_no_explanation_match/eval_bundles/inf_mcqa_v14",
        "para9": "results/FT/full/7b/para9/fill_dclm/domains_arxiv_all-legal_all-medical_all/e50/bs256_lr4e-05/overlap_1_16/E2_paraphrase_all_domains_local_no_explanation_match/eval_bundles/inf_mcqa_v14",
        "para24": "results/FT/full/7b/para24/fill_dclm/domains_arxiv_all-legal_all-medical_all/e50/bs256_lr4e-05/overlap_1_16/E2_paraphrase_all_domains_local_no_explanation_match/eval_bundles/inf_mcqa_v14",
        "para49": "results/FT/full/7b/para49/fill_dclm/domains_arxiv_all-legal_all-medical_all/e50/bs256_lr4e-05/overlap_1_16/E2_paraphrase_all_domains_local_no_explanation_match/eval_bundles/inf_mcqa_v14",
    },
    "13b": {
        "source_only": "results/FT/full/allenai_OLMo-2-1124-13B/source_only/fill_dclm/domains_arxiv_all-legal_all-medical_all/e50/bs256_lr4e-05/overlap_1_16/E20_source_13b_all_domains_no_explanation_match_para0/eval_bundles/inf_mcqa_v14",
        "para4": "results/FT/full/allenai_OLMo-2-1124-13B/para4/fill_dclm/domains_arxiv_all-legal_all-medical_all/e50/bs256_lr4e-05/overlap_1_16/E2_paraphrase_13b_all_domains_no_explanation_match_para9/eval_bundles/inf_mcqa_v14",
        "para9": "results/FT/full/allenai_OLMo-2-1124-13B/para9/fill_dclm/domains_arxiv_all-legal_all-medical_all/e50/bs256_lr4e-05/overlap_1_16/E2_paraphrase_13b_all_domains_no_explanation_match_para9/eval_bundles/inf_mcqa_v14",
        "para24": "results/FT/full/allenai_OLMo-2-1124-13B/para24/fill_dclm/domains_arxiv_all-legal_all-medical_all/e50/bs256_lr4e-05/overlap_1_16/E2_paraphrase_13b_all_domains_no_explanation_match_para24/eval_bundles/inf_mcqa_v14",
        "para49": "results/FT/full/allenai_OLMo-2-1124-13B/para49/fill_dclm/domains_arxiv_all-legal_all-medical_all/e50/bs256_lr4e-05/overlap_1_16/E2_paraphrase_13b_all_domains_no_explanation_match_para49/eval_bundles/inf_mcqa_v14",
    },
}
LEGACY_DOMAINS = ("1_58", "BOFT", "DPO", "GRPO", "OFT", "QLoRA")
LEGACY_PARAPHRASE_LEVELS = ("source_only", "para4", "para9", "para19", "para49")
LEGACY_PARAPHRASE_LABELS = {
    "source_only": "0",
    "para4": "4",
    "para9": "9",
    "para19": "19",
    "para49": "49",
}
LEGACY_MODEL_RUNS: Dict[str, Dict[str, str]] = {
    "1b": {
        "source_only": "results/FT/full/1b/probes_v9/newline2/source_only/sep_1_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs32_lr2e-05/run",
        "para4": "results/FT/full/1b/probes_v9/newline2/para4/sep_1_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs32_lr2e-05/overlap_1_4/09_21_18_38",
        "para9": "results/FT/full/1b/probes_v9/newline2/para9/sep_1_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs32_lr2e-05/run",
        "para19": "results/FT/full/1b/probes_v9/newline2/para19/sep_1_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs32_lr2e-05/overlap_1_4/09_22_12_55",
        "para49": "results/FT/full/1b/probes_v9/newline2/para49/sep_1_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs32_lr2e-05/overlap_1_4/09_24_00_18",
    },
    "7b": {
        "source_only": "results/FT/full/7b/probes_v9/newline2/source_only/sep_1_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs32_lr2e-05/run",
        "para4": "results/FT/full/7b/probes_v9/newline2/para4/sep_1_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs32_lr2e-05/overlap_1_4/09_21_18_26",
        "para9": "results/FT/full/7b/probes_v9/newline2/para9/sep_1_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs32_lr2e-05/run",
        "para19": "results/FT/full/7b/probes_v9/newline2/para19/sep_1_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs32_lr2e-05/09_21_00_54",
        "para49": "results/FT/full/7b/probes_v9/newline2/para49/sep_1_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs32_lr2e-05/overlap_1_4/09_22_22_16",
    },
    "13b": {
        "source_only": "results/FT/full/allenai_OLMo-2-1124-13B/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_21_11_30",
        "para4": "results/FT/full/allenai_OLMo-2-1124-13B/probes_v9/newline2/para4/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_22_16_48",
        "para9": "results/FT/full/allenai_OLMo-2-1124-13B/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_21_13_18",
        "para19": "results/FT/full/allenai_OLMo-2-1124-13B/probes_v9/newline2/para19/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_22_18_39",
        "para49": "results/FT/full/allenai_OLMo-2-1124-13B/probes_v9/newline2/para49/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_22_20_30",
    },
}
LEGACY_MODEL_COLORS = {
    "1b": "#1f77b4",
    "7b": "#d62728",
    "13b": "#2ca02c",
}


def _abs_path(path: str) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return str(candidate)


def iter_run_paths():
    for model in MODELS:
        for paraphrase in PARAPHRASE_LEVELS:
            path = MODEL_RUNS.get(model, {}).get(paraphrase)
            if path:
                yield model, paraphrase, _abs_path(path)


def iter_legacy_run_paths():
    for model in MODELS:
        for paraphrase in LEGACY_PARAPHRASE_LEVELS:
            path = LEGACY_MODEL_RUNS.get(model, {}).get(paraphrase)
            if path:
                yield model, paraphrase, _abs_path(path)


def load_configured_values(
    domains: Sequence[str],
    value_mode: str = "final",
    factual_probe_variant: str = "canonical",
    models: Sequence[str] = MODELS,
):
    if value_mode not in VALUE_MODE_CHOICES:
        raise ValueError(f"Unsupported value mode: {value_mode}")
    if factual_probe_variant not in FACTUAL_PROBE_VARIANT_CHOICES:
        raise ValueError(f"Unsupported factual probe variant: {factual_probe_variant}")

    values: Dict[Tuple[str, str, str], dict] = {}
    for model in models:
        for paraphrase in PARAPHRASE_LEVELS:
            run_path = MODEL_RUNS.get(model, {}).get(paraphrase)
            resolved_run_path = find_latest_run(_abs_path(run_path)) if run_path else None
            for probe_type, _title in (("knowledge", "Factual Probes"), ("inference", "Inference Probes")):
                classic_metric_file_variant = (
                    "paraphrased"
                    if probe_type == "knowledge" and factual_probe_variant == "paraphrased"
                    else "default"
                )
                classic = None
                mcqa = None
                if run_path:
                    classic = load_metrics(
                        _abs_path(run_path),
                        probe_type,
                        domains,
                        str(REPO_ROOT),
                        metrics=("log_prob",),
                        probe_family="classic",
                        metric_file_variant=classic_metric_file_variant,
                    )
                    mcqa = load_metrics(
                        _abs_path(run_path),
                        probe_type,
                        domains,
                        str(REPO_ROOT),
                        metrics=("mcqa_accuracy",),
                        probe_family="mcqa",
                        mcqa_variant="preferred",
                    )

                values[(probe_type, model, paraphrase)] = {
                    "log_prob": metric_value(classic, "log_prob", value_mode),
                    "mcqa_accuracy": metric_value(mcqa, "mcqa_accuracy", value_mode),
                    "run_path": resolved_run_path,
                }
    return values


def plot_probe_scaling(
    values,
    output: str,
    value_mode: str = "final",
    factual_probe_variant: str = "canonical",
    models: Sequence[str] = MODELS,
):
    apply_shared_plot_style("new")
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
    x = np.arange(len(PARAPHRASE_LEVELS))
    paraphrase_tick_labels = [PARAPHRASE_LABELS[level] for level in PARAPHRASE_LEVELS]

    for ax, (probe_type, metric, title, ylabel) in zip(axes, PANELS):
        panel_values = []
        for model in models:
            metric_values = []
            for paraphrase in PARAPHRASE_LEVELS:
                item = values[(probe_type, model, paraphrase)]
                value = item[metric]
                metric_values.append(np.nan if value is None else value)

            panel_values.extend([v for v in metric_values if not np.isnan(v)])
            ax.plot(
                x,
                metric_values,
                color=COLORS["model"][model],
                linewidth=2,
                marker="o",
                linestyle="--" if metric == "mcqa_accuracy" else "-",
            )

        ax.set_title(panel_title(title, probe_type, metric, factual_probe_variant))
        ax.set_xticks(x)
        ax.set_xticklabels(paraphrase_tick_labels)
        ax.set_xlabel("Paraphrases")
        ax.set_ylabel(panel_ylabel(ylabel, value_mode))
        if value_mode == "delta":
            ax.axhline(0, color="black", linewidth=0.8, alpha=0.35)
        ax.grid(True, axis="y", alpha=0.25)

        ylim = compute_unified_ylim(panel_values, padding=0.05)
        if ylim:
            ax.set_ylim(ylim)

    model_handles = [
        Line2D(
            [0],
            [0],
            color=COLORS["model"][model],
            linewidth=2,
            marker="o",
            label=BASE_MODEL_LABELS[model],
        )
        for model in models
    ]
    axes[0].legend(
        handles=model_handles,
        loc="lower right",
        frameon=True,
        framealpha=0.9,
        borderpad=0.4,
        handlelength=1.8,
        fontsize=12,
    )
    fig.tight_layout()
    save_figure(fig, output)


def load_legacy_values(domains: Sequence[str], models: Sequence[str] = MODELS):
    values: Dict[Tuple[str, str, str], float] = {}
    resolved_paths: Dict[Tuple[str, str], Optional[str]] = {}
    for model in models:
        for paraphrase in LEGACY_PARAPHRASE_LEVELS:
            run_path = LEGACY_MODEL_RUNS[model][paraphrase]
            resolved = find_latest_run(_abs_path(run_path))
            resolved_paths[(model, paraphrase)] = resolved
            for probe_type in ("knowledge", "inference"):
                value = np.nan
                if resolved:
                    df = aggregate_across_domains(
                        resolved,
                        probe_type,
                        domains,
                        project_root=str(REPO_ROOT),
                    )
                    value = get_final_step_value(df)
                values[(probe_type, model, paraphrase)] = value
    return values, resolved_paths


def plot_legacy_side_by_side(values, output: str, models: Sequence[str] = MODELS):
    apply_shared_plot_style("legacy")
    plt.rcParams.update(
        {
            "axes.labelsize": 15,
            "axes.titlesize": 17,
            "font.size": 13,
            "legend.fontsize": 12,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)
    x = np.arange(len(LEGACY_PARAPHRASE_LEVELS))
    xticklabels = [LEGACY_PARAPHRASE_LABELS[level] for level in LEGACY_PARAPHRASE_LEVELS]
    panel_specs = (("knowledge", "Factual Probes"), ("inference", "Compositional Probes"))

    all_y_values = []
    for ax, (probe_type, title) in zip(axes, panel_specs):
        for model in models:
            ys = [
                values[(probe_type, model, paraphrase)]
                for paraphrase in LEGACY_PARAPHRASE_LEVELS
            ]
            all_y_values.extend([y for y in ys if not np.isnan(y)])
            ax.plot(
                x,
                ys,
                color=LEGACY_MODEL_COLORS[model],
                linewidth=1.5,
                marker="o",
                markersize=4.5,
                label=BASE_MODEL_LABELS[model],
            )

        ax.set_title(title)
        ax.set_xlabel("# Paraphrases")
        ax.set_xticks(x)
        ax.set_xticklabels(xticklabels)
        ax.set_xlim(-0.35, len(LEGACY_PARAPHRASE_LEVELS) - 0.65)
        ax.grid(False)

    axes[0].set_ylabel("Mean Log Probability")
    ylim = compute_unified_ylim(all_y_values, padding=0.05)
    if ylim:
        for ax in axes:
            ax.set_ylim(ylim)

    handles = [
        Line2D([0], [0], color=LEGACY_MODEL_COLORS[model], lw=2, marker="o", label=BASE_MODEL_LABELS[model])
        for model in models
    ]
    axes[1].legend(handles=handles, loc="lower right", frameon=True, framealpha=0.95)
    fig.tight_layout()
    save_figure(fig, output)


def plot_current_and_legacy(
    current_values,
    legacy_values,
    output: str,
    models: Sequence[str] = COMBINED_DEFAULT_MODELS,
    value_mode: str = "final",
    factual_probe_variant: str = "canonical",
):
    apply_shared_plot_style("new")
    plt.rcParams.update(
        {
            "axes.labelsize": 13,
            "axes.titlesize": 14,
            "font.size": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.7), sharey=False)
    current_x = np.arange(len(PARAPHRASE_LEVELS))
    legacy_x = np.arange(len(LEGACY_PARAPHRASE_LEVELS))
    panel_specs = (
        ("current", "knowledge", "Current Factual", PARAPHRASE_LEVELS, current_x),
        ("current", "inference", "Current Compositional", PARAPHRASE_LEVELS, current_x),
        ("legacy", "knowledge", "Legacy Factual", LEGACY_PARAPHRASE_LEVELS, legacy_x),
        ("legacy", "inference", "Legacy Compositional", LEGACY_PARAPHRASE_LEVELS, legacy_x),
    )

    y_values_by_run_set = {"current": [], "legacy": []}
    for ax, (run_set, probe_type, title, levels, x) in zip(axes, panel_specs):
        for model in models:
            if run_set == "current":
                ys = [
                    np.nan
                    if current_values[(probe_type, model, paraphrase)]["log_prob"] is None
                    else current_values[(probe_type, model, paraphrase)]["log_prob"]
                    for paraphrase in levels
                ]
            else:
                ys = [legacy_values[(probe_type, model, paraphrase)] for paraphrase in levels]

            y_values_by_run_set[run_set].extend([y for y in ys if not np.isnan(y)])
            ax.plot(
                x,
                ys,
                color=COLORS["model"][model],
                linewidth=1.8,
                marker="o",
                markersize=5,
                label=BASE_MODEL_LABELS[model],
            )

        ax.set_title(title)
        ax.set_xlabel("# Paraphrases")
        ax.set_xticks(x)
        labels = PARAPHRASE_LABELS if run_set == "current" else LEGACY_PARAPHRASE_LABELS
        ax.set_xticklabels([labels[level] for level in levels])
        ax.grid(True, axis="y", alpha=0.25)
        if value_mode == "delta":
            ax.axhline(0, color="black", linewidth=0.8, alpha=0.35)

    axes[0].set_ylabel(panel_ylabel("Mean Log Probability", value_mode))
    axes[2].set_ylabel(panel_ylabel("Mean Log Probability", value_mode))
    axes[2].axvline(-0.55, color="black", linewidth=0.8, alpha=0.2)
    for axis_group, run_set in ((axes[:2], "current"), (axes[2:], "legacy")):
        ylim = compute_unified_ylim(y_values_by_run_set[run_set], padding=0.05)
        if ylim:
            for ax in axis_group:
                ax.set_ylim(ylim)

    handles = [
        Line2D([0], [0], color=COLORS["model"][model], lw=2, marker="o", label=BASE_MODEL_LABELS[model])
        for model in models
    ]
    axes[0].legend(handles=handles, loc="lower right", frameon=True, framealpha=0.9)
    fig.tight_layout()
    save_figure(fig, output)


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="plots/probe_scaling/probe_scaling_by_paraphrase")
    parser.add_argument(
        "--run_set",
        choices=("current_v14_e50", "legacy_e100", "current_and_legacy"),
        default="current_v14_e50",
        help="Configured run set to plot.",
    )
    parser.add_argument("--domains", nargs="+")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODELS,
        help=(
            "Model lines to include. Defaults to all models for single-run-set plots "
            "and to 7b 13b for current_and_legacy."
        ),
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
    models = tuple(args.models) if args.models else MODELS
    if args.run_set == "legacy_e100":
        legacy_models = tuple(args.models) if args.models else MODELS
        domains = args.domains or list(LEGACY_DOMAINS)
        print(f"Using {len(domains)} legacy domains: {', '.join(domains)}")
        values, resolved_paths = load_legacy_values(domains, models=legacy_models)
        for model in legacy_models:
            for paraphrase in LEGACY_PARAPHRASE_LEVELS:
                if not resolved_paths[(model, paraphrase)]:
                    print(f"Warning: missing legacy run for {BASE_MODEL_LABELS[model]} {paraphrase}")
        missing = [
            (probe_type, BASE_MODEL_LABELS[model], paraphrase)
            for (probe_type, model, paraphrase), value in values.items()
            if np.isnan(value)
        ]
        for probe_type, model, paraphrase in missing:
            print(f"Warning: missing legacy {probe_type} value for {model} {paraphrase}")
        plot_legacy_side_by_side(values, args.output, models=legacy_models)
        return

    if args.run_set == "current_and_legacy":
        combined_models = tuple(args.models) if args.models else COMBINED_DEFAULT_MODELS
        current_run_paths = [
            _abs_path(MODEL_RUNS[model][paraphrase])
            for model in combined_models
            for paraphrase in PARAPHRASE_LEVELS
            if MODEL_RUNS.get(model, {}).get(paraphrase)
        ]
        current_domains = args.domains or discover_domains_from_runs(current_run_paths)
        if not current_domains:
            raise RuntimeError("No domains found in configured current result folders")
        print(f"Using {len(current_domains)} current domains")
        current_values = load_configured_values(
            current_domains,
            value_mode=args.value_mode,
            factual_probe_variant=args.factual_probe_variant,
            models=combined_models,
        )
        legacy_values, legacy_resolved_paths = load_legacy_values(
            LEGACY_DOMAINS,
            models=combined_models,
        )
        for model in combined_models:
            for paraphrase in LEGACY_PARAPHRASE_LEVELS:
                if not legacy_resolved_paths[(model, paraphrase)]:
                    print(f"Warning: missing legacy run for {BASE_MODEL_LABELS[model]} {paraphrase}")
        plot_current_and_legacy(
            current_values,
            legacy_values,
            args.output,
            models=combined_models,
            value_mode=args.value_mode,
            factual_probe_variant=args.factual_probe_variant,
        )
        return

    run_paths = [
        path
        for model, _, path in iter_run_paths()
        if model in models
    ]
    domains = args.domains or discover_domains_from_runs(run_paths)
    if not domains:
        raise RuntimeError("No domains found in configured result folders")

    print(f"Using {len(domains)} domains")
    values = load_configured_values(
        domains,
        value_mode=args.value_mode,
        factual_probe_variant=args.factual_probe_variant,
        models=models,
    )

    for model in models:
        for paraphrase in PARAPHRASE_LEVELS:
            if not MODEL_RUNS.get(model, {}).get(paraphrase):
                print(f"Warning: missing run for {BASE_MODEL_LABELS[model]} {paraphrase}")

    missing = [
        (probe_type, BASE_MODEL_LABELS[model], paraphrase, metric)
        for (probe_type, model, paraphrase), item in values.items()
        if item["run_path"] is not None
        for metric in ("log_prob", "mcqa_accuracy")
        if item[metric] is None
    ]
    if missing:
        for probe_type, model, paraphrase, metric in missing:
            print(f"Warning: missing {probe_type} {metric} for {model} {paraphrase}")

    plot_probe_scaling(
        values,
        args.output,
        value_mode=args.value_mode,
        factual_probe_variant=args.factual_probe_variant,
        models=models,
    )


if __name__ == "__main__":
    main()
