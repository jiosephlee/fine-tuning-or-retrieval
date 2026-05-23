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


LEGACY_RESULTS_ROOT = os.path.join(REPO_ROOT, "results", "legacy")
LEGACY_DOMAINS = ["DPO", "1_58", "GRPO", "BOFT", "OFT", "QLoRA"]


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
                32: legacy_path("FT/full/allenai_OLMo-2-1124-13B/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs32_lr2e-05/overlap_1_4/11_23_07_02"),
                64: legacy_path("FT/full/allenai_OLMo-2-1124-13B/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05"),
                128: legacy_path("FT/full/allenai_OLMo-2-1124-13B/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs128_lr2e-05/overlap_1_4/11_24_05_16"),
                256: legacy_path("FT/full/allenai_OLMo-2-1124-13B/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs256_lr2e-05/overlap_1_4/11_24_07_46"),
            },
            "Source": {
                32: legacy_path("FT/full/allenai_OLMo-2-1124-13B/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs32_lr2e-05"),
                64: legacy_path("FT/full/allenai_OLMo-2-1124-13B/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4"),
                128: legacy_path("FT/full/allenai_OLMo-2-1124-13B/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs128_lr2e-05/overlap_1_4/11_23_22_57"),
                256: legacy_path("FT/full/allenai_OLMo-2-1124-13B/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs256_lr2e-05/overlap_1_4/11_24_01_25"),
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
                32: legacy_path("FT/full/allenai_OLMo-2-0325-32B/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs32_lr2e-05/overlap_1_4/11_27_05_20"),
                64: legacy_path("FT/full/allenai_OLMo-2-0325-32B/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_21_06_44"),
                128: legacy_path("FT/full/allenai_OLMo-2-0325-32B/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs128_lr2e-05/overlap_1_4/11_27_05_26"),
                256: legacy_path("FT/full/allenai_OLMo-2-0325-32B/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs256_lr2e-05/overlap_1_4/11_27_05_26"),
            },
            "Source": {
                32: legacy_path("FT/full/allenai_OLMo-2-0325-32B/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs32_lr2e-05/overlap_1_4/11_27_05_20"),
                64: legacy_path("FT/full/allenai_OLMo-2-0325-32B/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_21_06_14"),
                128: legacy_path("FT/full/allenai_OLMo-2-0325-32B/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs128_lr2e-05/overlap_1_4/11_27_05_20"),
                256: legacy_path("FT/full/allenai_OLMo-2-0325-32B/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs256_lr2e-05/overlap_1_4/11_27_05_20"),
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
                                "Type": "Compositional",
                                "Metric": metric_name,
                                "Value": compositional_value,
                            }
                        )

    return pd.DataFrame(rows)


def collect_strategy_trajectory_data(batch_size=32, probe_type="inference"):
    rows = []
    config = get_batch_scaling_config()

    for strategy in ("Source", "Para 9"):
        for model in ("1B", "7B", "13B", "32B"):
            path = config.get(model, {}).get(strategy, {}).get(batch_size)
            resolved = find_latest_run(path)
            if not resolved:
                print(f"Warning: could not resolve legacy run path: {path}")
                continue

            df = aggregate_across_domains(
                resolved,
                probe_type,
                LEGACY_DOMAINS,
                split_probes=False,
                project_root=REPO_ROOT,
            )
            if df.empty or "step" not in df.columns or "log_prob" not in df.columns:
                continue

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

    return pd.DataFrame(rows)


def plot_combined_log_prob(df_traj, df_bs, model_colors, style):
    print("Plotting combined legacy log-prob figure...")

    fig = plt.figure(figsize=(16, 5))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.1, 0.18, 1, 1], wspace=0.15)
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[0, 3]),
    ]

    strategy_styles = style["strategy_styles"]
    trajectory_styles = {
        "Para 9": {"linestyle": strategy_styles["Para 9"]["linestyle"], "label": "Paraphrased"},
        "Source": {"linestyle": strategy_styles["Source"]["linestyle"], "label": "Source Only"},
    }
    trajectory_values = []
    ax_traj = axes[0]
    for strategy in ("Para 9", "Source"):
        subset = df_traj[df_traj["Strategy"] == strategy]
        for model in ("1B", "7B", "13B", "32B"):
            series = subset[subset["Model"] == model].sort_values("Step")
            if series.empty:
                continue
            ax_traj.plot(
                series["Step"],
                series["Value"],
                color=model_colors[model],
                linestyle=trajectory_styles[strategy]["linestyle"],
                linewidth=style["line_width"],
            )
            trajectory_values.extend(series["Value"].dropna().tolist())

    ax_traj.set_title("Source vs. Paraphrased")
    ax_traj.set_xlabel("Training Step")
    ax_traj.set_ylabel("Inference Log Prob")
    ax_traj.grid(True, alpha=style["grid_alpha"])
    ax_traj.set_xlim(left=0)
    ax_traj.margins(x=0)

    trajectory_ylim = compute_unified_ylim(trajectory_values, padding=0.05)
    if trajectory_ylim:
        apply_ylim([ax_traj], trajectory_ylim)

    batch_panels = [
        ("Factual", axes[1]),
        ("Compositional", axes[2]),
    ]
    for probe_type, ax in batch_panels:
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
                markerfacecolor="white",
                markeredgecolor=model_colors[model],
                markeredgewidth=style["marker_edge_width"],
            )

        ax.set_title(probe_type)
        ax.set_xscale("log", base=2)
        ax.set_xticks([32, 64, 128, 256])
        ax.set_xticklabels(["32", "64", "128", "256"])
        ax.set_xlabel("Batch Size")
        ax.grid(True, which="major", ls="-", alpha=style["grid_alpha"])

    axes[1].set_ylabel("Final Log Prob")
    axes[2].set_ylabel("")
    axes[2].set_yticklabels([])

    batch_ylim = compute_unified_ylim(df_bs["Value"].dropna().tolist(), padding=0.05)
    if batch_ylim:
        apply_ylim(axes[1:], batch_ylim)

    legend_elements = [
        Line2D([0], [0], color=model_colors["1B"], lw=style["line_width"], linestyle="-", label="1B"),
        Line2D([0], [0], color=model_colors["7B"], lw=style["line_width"], linestyle="-", label="7B"),
        Line2D([0], [0], color=model_colors["13B"], lw=style["line_width"], linestyle="-", label="13B"),
        Line2D([0], [0], color=model_colors["32B"], lw=style["line_width"], linestyle="-", label="32B"),
        Line2D(
            [0], [0],
            color="gray",
            lw=style["line_width"],
            linestyle="-",
            marker="o",
            markerfacecolor="white",
            markeredgecolor="gray",
            markersize=style["marker_size"],
            markeredgewidth=style["marker_edge_width"],
            label="Paraphrased",
        ),
        Line2D(
            [0], [0],
            color="gray",
            lw=style["line_width"],
            linestyle="--",
            marker="s",
            markerfacecolor="white",
            markeredgecolor="gray",
            markersize=style["marker_size"],
            markeredgewidth=style["marker_edge_width"],
            label="Source Only",
        ),
    ]
    legend = axes[1].legend(
        handles=legend_elements,
        loc="lower right",
        ncol=1,
        **style["legend"],
    )
    legend.get_frame().set_facecolor("none")

    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.16, top=0.88, wspace=0.15)
    out_path = os.path.join(REPO_ROOT, "plots", "legacy", "batch_scaling_log_prob")
    save_figure(fig, out_path)


def main():
    style = apply_plot_style("legacy")

    model_colors = {
        "1B": COLORS["model"]["1b"],
        "7B": COLORS["model"]["7b"],
        "13B": COLORS["model"]["13b"],
        "32B": COLORS["model"]["32b"],
    }
    df_bs = collect_batch_scaling_data()
    df_traj = collect_strategy_trajectory_data()
    plot_combined_log_prob(df_traj, df_bs, model_colors, style)


if __name__ == "__main__":
    main()
