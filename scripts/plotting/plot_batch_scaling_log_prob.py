import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(REPO_ROOT)

from scripts.plotting.plot_utils import (  # noqa: E402
    COLORS,
    aggregate_across_domains,
    apply_ylim,
    compute_unified_ylim,
    find_latest_run,
    get_final_step_value,
    apply_plot_style,
    save_figure,
)
from scripts.plotting.plot_probe_scaling_by_model import (  # noqa: E402
    DEFAULT_RUNS as NEW_TRAJECTORY_RUNS,
    MODEL_LABELS as NEW_MODEL_LABELS,
    _abs_path as new_abs_path,
    discover_domains_from_runs as discover_new_domains_from_runs,
)


LEGACY_RESULTS_ROOT = os.path.join(REPO_ROOT, "results", "legacy")
LEGACY_DOMAINS = ["DPO", "1_58", "GRPO", "BOFT", "OFT", "QLoRA"]
TRAJECTORY_PATH_CHOICES = ("legacy", "new")


def legacy_path(path):
    return os.path.join(LEGACY_RESULTS_ROOT, path)


def get_batch_scaling_config():
    return {
        "7B": {
            "Para 9": {
                32: legacy_path("FT/full/7b/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs32_lr2e-05/overlap_1_4/11_23_10_36"),
                64: legacy_path("FT/full/7b/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_20_22_51"),
                128: legacy_path("FT/full/7b/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs128_lr2e-05/overlap_1_4/11_23_11_36"),
                256: legacy_path("FT/full/7b/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs256_lr2e-05/overlap_1_4/11_23_13_18"),
            },
            "Source": {
                32: legacy_path("FT/full/7b/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs32_lr2e-05/overlap_1_4/11_23_05_24"),
                64: legacy_path("FT/full/7b/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_20_22_51"),
                128: legacy_path("FT/full/7b/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs128_lr2e-05/overlap_1_4/11_23_06_24"),
                256: legacy_path("FT/full/7b/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs256_lr2e-05/overlap_1_4/11_23_08_04"),
            },
        },
        "13B": {
            "Para 9": {
                32: legacy_path("FT/full/13b/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs32_lr2e-05/overlap_1_4/11_23_07_02"),
                64: legacy_path("FT/full/13b/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05"),
                128: legacy_path("FT/full/13b/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs128_lr2e-05/overlap_1_4/11_24_05_16"),
                256: legacy_path("FT/full/13b/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs256_lr2e-05/overlap_1_4/11_24_07_46"),
            },
            "Source": {
                32: legacy_path("FT/full/13b/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs32_lr2e-05"),
                64: legacy_path("FT/full/13b/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4"),
                128: legacy_path("FT/full/13b/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs128_lr2e-05/overlap_1_4/11_23_22_57"),
                256: legacy_path("FT/full/13b/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs256_lr2e-05/overlap_1_4/11_24_01_25"),
            },
        },
        "1B": {
            "Source": {
                32: legacy_path("FT/full/1b/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs32_lr2e-05/overlap_1_4/11_23_05_24"),
                64: legacy_path("FT/full/1b/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_23_22_57"),
                128: legacy_path("FT/full/1b/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs128_lr2e-05/overlap_1_4/11_23_05_47"),
                256: legacy_path("FT/full/1b/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs256_lr2e-05/overlap_1_4/11_23_06_23"),
            },
            "Para 9": {
                32: legacy_path("FT/full/1b/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs32_lr2e-05/overlap_1_4/11_23_07_13"),
                64: legacy_path("FT/full/1b/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_23_23_24"),
                128: legacy_path("FT/full/1b/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs128_lr2e-05/overlap_1_4/11_23_07_37"),
                256: legacy_path("FT/full/1b/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs256_lr2e-05/overlap_1_4/11_23_08_13"),
            },
        },
        "32B": {
            "Para 9": {
                32: legacy_path("FT/full/32b/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs32_lr2e-05/overlap_1_4/11_27_05_20"),
                64: legacy_path("FT/full/32b/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_21_06_44"),
                128: legacy_path("FT/full/32b/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs128_lr2e-05/overlap_1_4/11_27_05_26"),
                256: legacy_path("FT/full/32b/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs256_lr2e-05/overlap_1_4/11_27_05_26"),
            },
            "Source": {
                32: legacy_path("FT/full/32b/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs32_lr2e-05/overlap_1_4/11_27_05_20"),
                64: legacy_path("FT/full/32b/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_21_06_14"),
                128: legacy_path("FT/full/32b/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs128_lr2e-05/overlap_1_4/11_27_05_20"),
                256: legacy_path("FT/full/32b/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs256_lr2e-05/overlap_1_4/11_27_05_20"),
            },
        },
    }


def collect_batch_scaling_data():
    metrics_map = {
        "Log Prob": "log_prob",
    }
    rows = []

    for model, strategies in get_batch_scaling_config().items():
        for strategy, batches in strategies.items():
            for batch_size, path in batches.items():
                resolved = find_latest_run(path)
                if not resolved:
                    print(f"Warning: could not resolve legacy run path: {path}")
                    continue

                knowledge_df = aggregate_across_domains(
                    resolved,
                    "knowledge",
                    LEGACY_DOMAINS,
                    split_probes=False,
                    project_root=REPO_ROOT,
                )
                inference_df = aggregate_across_domains(
                    resolved,
                    "inference",
                    LEGACY_DOMAINS,
                    split_probes=False,
                    project_root=REPO_ROOT,
                )

                for metric_name, column in metrics_map.items():
                    factual_value = get_final_step_value(knowledge_df, value_col=column)
                    compositional_value = get_final_step_value(inference_df, value_col=column)
                    if not np.isnan(factual_value):
                        rows.append(
                            {
                                "Model": model,
                                "Strategy": strategy,
                                "BatchSize": batch_size,
                                "Type": "Factual",
                                "Metric": metric_name,
                                "Value": factual_value,
                            }
                        )
                    if not np.isnan(compositional_value):
                        rows.append(
                            {
                                "Model": model,
                                "Strategy": strategy,
                                "BatchSize": batch_size,
                                "Type": "Inference",
                                "Metric": metric_name,
                                "Value": compositional_value,
                            }
                        )

    return pd.DataFrame(rows)


def _append_trajectory_rows(rows, strategy, model, run_path, domains, probe_type):
    resolved = find_latest_run(run_path)
    if not resolved:
        print(f"Warning: could not resolve trajectory run path: {run_path}")
        return

    df = aggregate_across_domains(
        resolved,
        probe_type,
        domains,
        split_probes=False,
        project_root=REPO_ROOT,
    )
    if df.empty or "step" not in df.columns or "log_prob" not in df.columns:
        return

    series = (
        df[["step", "log_prob"]]
        .dropna()
        .groupby("step", as_index=False)["log_prob"]
        .mean()
        .sort_values("step")
    )
    for _, row in series.iterrows():
        rows.append(
            {
                "Strategy": strategy,
                "Model": model,
                "Step": int(row["step"]),
                "Value": float(row["log_prob"]),
            }
        )


def collect_legacy_strategy_trajectory_data(batch_size=32, probe_type="inference"):
    rows = []
    config = get_batch_scaling_config()

    for strategy in ("Source", "Para 9"):
        for model in ("1B", "7B", "13B", "32B"):
            path = config.get(model, {}).get(strategy, {}).get(batch_size)
            _append_trajectory_rows(rows, strategy, model, path, LEGACY_DOMAINS, probe_type)

    return pd.DataFrame(rows)


def collect_new_strategy_trajectory_data(probe_type="inference"):
    rows = []
    run_items = []
    strategy_to_method = {
        "Source": "source_only",
        "Para 9": "para9",
    }
    for strategy, method in strategy_to_method.items():
        for model_key, model_label in NEW_MODEL_LABELS.items():
            run_path = new_abs_path(NEW_TRAJECTORY_RUNS[method][model_key])
            run_items.append((strategy, model_label, run_path))

    domains = discover_new_domains_from_runs([run_path for _, _, run_path in run_items])
    if not domains:
        print("Warning: could not discover domains for new trajectory runs")
        return pd.DataFrame(rows)

    for strategy, model, run_path in run_items:
        _append_trajectory_rows(rows, strategy, model, run_path, domains, probe_type)

    return pd.DataFrame(rows)


def collect_strategy_trajectory_data(path_source="legacy", probe_type="inference"):
    if path_source == "legacy":
        return collect_legacy_strategy_trajectory_data(probe_type=probe_type)
    if path_source == "new":
        return collect_new_strategy_trajectory_data(probe_type=probe_type)
    raise ValueError(f"Unsupported trajectory path source: {path_source}")


def plot_combined_log_prob(df_traj, df_bs, model_colors, style, output_name="batch_scaling_log_prob"):
    print("Plotting legacy 4-panel log-prob figure...")

    fig = plt.figure(figsize=(16.8, 4.8))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 0.15, 1, 1], wspace=0.1)
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 3]),
        fig.add_subplot(gs[0, 4]),
    ]

    strategy_styles = style["strategy_styles"]
    trajectory_values = []
    trajectory_steps = []
    trajectory_panels = [
        ("Source", "Source", axes[0]),
        ("Para 9", "Paraphrased", axes[1]),
    ]
    for strategy, title, ax in trajectory_panels:
        subset = df_traj[df_traj["Strategy"] == strategy]
        for model in ("1B", "7B", "13B", "32B"):
            series = subset[subset["Model"] == model].sort_values("Step")
            if series.empty:
                continue
            ax.plot(
                series["Step"],
                series["Value"],
                color=model_colors[model],
                linestyle="-",
                linewidth=style["line_width"],
            )
            trajectory_values.extend(series["Value"].dropna().tolist())
            trajectory_steps.extend(series["Step"].dropna().tolist())

        ax.set_title(title)
        ax.set_xlabel("Exposure #")
        ax.grid(True, alpha=style["grid_alpha"])
        if trajectory_steps:
            ax.set_xlim(0, max(trajectory_steps))

    axes[0].set_ylabel("Log Prob.")
    axes[1].set_yticklabels([])
    axes[0].set_xticks([0, 25, 50, 75, 100])
    axes[1].set_xticks([0, 25, 50, 75, 100])
    axes[1].set_xticklabels(["", "25", "50", "75", "100"])

    trajectory_ylim = compute_unified_ylim(trajectory_values, padding=0.05)
    if trajectory_ylim:
        apply_ylim(axes[:2], trajectory_ylim)

    batch_panels = [
        ("Factual", axes[2]),
        ("Inference", axes[3]),
    ]
    for index, (probe_type, ax) in enumerate(batch_panels):
        subset = df_bs[df_bs["Type"] == probe_type]
        for _, pair in subset[["Model", "Strategy"]].drop_duplicates().iterrows():
            model = pair["Model"]
            strategy = pair["Strategy"]
            series = subset[
                (subset["Model"] == model) & (subset["Strategy"] == strategy)
            ].sort_values("BatchSize")
            if series.empty:
                continue

            series_style = strategy_styles.get(strategy, {})
            ax.plot(
                series["BatchSize"],
                series["Value"],
                color=model_colors[model],
                linestyle=series_style.get("linestyle", "-"),
                marker=series_style.get("marker", "o"),
                linewidth=style["batch_line_width"],
                markersize=style["marker_size"],
                markerfacecolor=model_colors[model],
                markeredgecolor="#444444",
                markeredgewidth=style["marker_edge_width"],
            )

        ax.set_title(probe_type)
        ax.set_xscale("log", base=2)
        ax.set_xticks([32, 64, 128, 256])
        ax.set_xticklabels(["32", "64", "128", "256"])
        ax.set_xlabel("Batch Size")
        ax.grid(True, which="major", ls="-", alpha=style["grid_alpha"])
        if index != 0:
            ax.set_yticklabels([])

    axes[2].set_ylabel("Final Log Prob")

    batch_ylim = compute_unified_ylim(df_bs["Value"].dropna().tolist(), padding=0.05)
    if batch_ylim:
        apply_ylim(axes[2:], batch_ylim)

    trajectory_legend_elements = [
        Line2D([0], [0], color=model_colors["1B"], lw=style["line_width"], linestyle="-", label="1B"),
        Line2D([0], [0], color=model_colors["7B"], lw=style["line_width"], linestyle="-", label="7B"),
        Line2D([0], [0], color=model_colors["13B"], lw=style["line_width"], linestyle="-", label="13B"),
        Line2D([0], [0], color=model_colors["32B"], lw=style["line_width"], linestyle="-", label="32B"),
    ]
    trajectory_legend = axes[0].legend(
        handles=trajectory_legend_elements,
        loc="lower right",
        **style["legend"],
    )
    trajectory_legend.get_frame().set_facecolor("none")

    batch_legend_elements = [
        Line2D(
            [0], [0],
            color="gray",
            lw=style["line_width"],
            linestyle="-",
            marker="o",
            markerfacecolor="gray",
            markeredgecolor="#444444",
            markeredgewidth=style["marker_edge_width"],
            markersize=style["marker_size"],
            label="Paraphrased",
        ),
        Line2D(
            [0], [0],
            color="gray",
            lw=style["line_width"],
            linestyle="--",
            marker="s",
            markerfacecolor="gray",
            markeredgecolor="#444444",
            markeredgewidth=style["marker_edge_width"],
            markersize=style["marker_size"],
            label="Source",
        ),
        Line2D([0], [0], color=model_colors["1B"], lw=style["line_width"], linestyle="-", label="1B"),
        Line2D([0], [0], color=model_colors["7B"], lw=style["line_width"], linestyle="-", label="7B"),
        Line2D([0], [0], color=model_colors["13B"], lw=style["line_width"], linestyle="-", label="13B"),
        Line2D([0], [0], color=model_colors["32B"], lw=style["line_width"], linestyle="-", label="32B"),
    ]
    batch_legend = axes[2].legend(
        handles=batch_legend_elements,
        loc="lower left",
        **style["legend"],
    )
    batch_legend.get_frame().set_facecolor("none")

    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.16, top=0.88, wspace=0.1)
    out_path = os.path.join(REPO_ROOT, "plots", "legacy", output_name)
    save_figure(fig, out_path)


def main():
    parser = argparse.ArgumentParser(
        description="Plot legacy batch scaling with selectable Source/Paraphrased trajectory runs."
    )
    parser.add_argument(
        "--trajectory-paths",
        choices=TRAJECTORY_PATH_CHOICES,
        default="legacy",
        help="Run paths for the left Source/Paraphrased trajectory panels. Batch panels always use legacy paths.",
    )
    args = parser.parse_args()

    style = apply_plot_style("legacy")

    model_colors = {
        "1B": COLORS["model"]["1b"],
        "7B": COLORS["model"]["7b"],
        "13B": COLORS["model"]["13b"],
        "32B": COLORS["model"]["32b"],
    }
    df_bs = collect_batch_scaling_data()
    df_traj = collect_strategy_trajectory_data(args.trajectory_paths)
    output_name = "batch_scaling_log_prob"
    if args.trajectory_paths == "new":
        output_name = "batch_scaling_log_prob_new_trajectories"
    plot_combined_log_prob(df_traj, df_bs, model_colors, style, output_name=output_name)


if __name__ == "__main__":
    main()
