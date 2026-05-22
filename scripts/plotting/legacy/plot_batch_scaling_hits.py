import os
import sys

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(REPO_ROOT)

from scripts.plotting.plot_utils import (  # noqa: E402
    aggregate_across_domains,
    apply_ylim,
    compute_unified_ylim,
    find_latest_run,
    get_final_step_value,
)


LEGACY_RESULTS_ROOT = os.path.join(REPO_ROOT, "results", "legacy")
LEGACY_DOMAINS = ["DPO", "1_58", "GRPO", "BOFT", "OFT", "QLoRA"]


def legacy_path(path):
    return os.path.join(LEGACY_RESULTS_ROOT, path)


def set_legacy_plot_style():
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "font.size": 14,
        }
    )


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
        "Hits@1": "hit_accuracy_at_1",
        "Hits@10": "hit_accuracy_at_10",
        "Hits@100": "hit_accuracy_at_100",
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


def plot_batch_scaling_hits(df_bs, model_colors, strategy_styles):
    print("Plotting legacy batch scaling Hits@1/10/100 figure...")

    fig = plt.figure(figsize=(10, 15))
    gs = gridspec.GridSpec(3, 2, width_ratios=[1, 1], wspace=0.15, hspace=0.3)

    metrics = ["Hits@1", "Hits@10", "Hits@100"]
    probe_types = ["Factual", "Compositional"]
    axes_grid = [
        [fig.add_subplot(gs[row, column]) for column in range(2)]
        for row in range(3)
    ]

    for row, metric in enumerate(metrics):
        for column, probe_type in enumerate(probe_types):
            ax = axes_grid[row][column]
            subset = df_bs[(df_bs["Type"] == probe_type) & (df_bs["Metric"] == metric)]

            for _, pair in subset[["Model", "Strategy"]].drop_duplicates().iterrows():
                model = pair["Model"]
                strategy = pair["Strategy"]
                series = subset[
                    (subset["Model"] == model) & (subset["Strategy"] == strategy)
                ].sort_values("BatchSize")
                if series.empty:
                    continue

                style = strategy_styles.get(strategy, {})
                ax.plot(
                    series["BatchSize"],
                    series["Value"],
                    color=model_colors.get(model, "black"),
                    linestyle=style.get("linestyle", "-"),
                    marker=style.get("marker", "o"),
                    linewidth=1.5,
                )

            if row == 0:
                ax.set_title(probe_type)
            if column == 0:
                ax.set_ylabel(metric)
            else:
                ax.set_yticklabels([])

            ax.set_xscale("log", base=2)
            ax.set_xticks([32, 64, 128, 256])
            ax.set_xticklabels(["32", "64", "128", "256"])
            if row == 2:
                ax.set_xlabel("Batch Size")
            else:
                ax.set_xticklabels([])
            ax.grid(True, which="major", ls="-", alpha=0.1)

    for row, metric in enumerate(metrics):
        row_values = df_bs[df_bs["Metric"] == metric]["Value"].tolist()
        ylim = compute_unified_ylim(values=row_values)
        if ylim:
            apply_ylim(axes_grid[row], ylim)

    legend_elements = [
        Line2D([0], [0], color="gray", lw=2, linestyle="-", label="Para 9"),
        Line2D([0], [0], color="gray", lw=2, linestyle="--", label="Source"),
        Line2D([0], [0], color="#ffd700", lw=2, linestyle="-", label="1B"),
        Line2D([0], [0], color="#ff7f0e", lw=2, linestyle="-", label="7B"),
        Line2D([0], [0], color="#d62728", lw=2, linestyle="-", label="13B"),
        Line2D([0], [0], color="#9467bd", lw=2, linestyle="-", label="32B"),
    ]
    axes_grid[0][0].legend(
        handles=legend_elements, loc="lower left", fontsize="small", frameon=True
    )

    plt.tight_layout()
    out_dir = os.path.join(REPO_ROOT, "plots", "legacy")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "batch_scaling_hits.pdf")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved legacy batch scaling hits plot to {out_path}")


def main():
    set_legacy_plot_style()
    plt.rcParams.update(
        {
            "axes.labelsize": 18,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16,
            "figure.titlesize": 22,
            "axes.titlesize": 20,
        }
    )

    model_colors = {
        "1B": "#ffd700",
        "7B": "#ff7f0e",
        "13B": "#d62728",
        "32B": "#9467bd",
    }
    strategy_styles = {
        "Source": {"linestyle": "--", "marker": "s"},
        "Para 9": {"linestyle": "-", "marker": "o"},
    }

    df_bs = collect_batch_scaling_data()
    plot_batch_scaling_hits(df_bs, model_colors, strategy_styles)


if __name__ == "__main__":
    main()
