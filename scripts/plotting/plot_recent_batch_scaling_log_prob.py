import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from scripts.plotting.plot_utils import (  # noqa: E402
    COLORS,
    aggregate_across_domains,
    apply_plot_style,
    apply_ylim,
    compute_unified_ylim,
    find_latest_run,
    get_final_step_value,
    save_figure,
)


RUNS = {
    "1B": {
        128: "results/FT/full/1b/para49/fill_dclm/domains_arxiv_all-legal_all-medical_all/e50/bs128_lr4e-05/overlap_1_16/E2_paraphrase_all_domains_local_no_explanation_match/eval_bundles/inf_mcqa_v14",
        256: "results/FT/full/1b/para49/fill_dclm/domains_arxiv_all-legal_all-medical_all/e50/bs256_lr4e-05/overlap_1_16/E2_paraphrase_all_domains_local_no_explanation_match/eval_bundles/inf_mcqa_v14",
        512: "results/FT/full/1b/para49/fill_dclm/domains_arxiv_all-legal_all-medical_all/e50/bs512_lr4e-05/overlap_1_16/E2_paraphrase_all_domains_local_no_explanation_match/eval_bundles/inf_mcqa_v14",
    },
    "7B": {
        128: "results/FT/full/7b/para49/fill_dclm/domains_arxiv_all-legal_all-medical_all/e50/bs128_lr4e-05/overlap_1_16/E2_paraphrase_all_domains_local_no_explanation_match/eval_bundles/inf_mcqa_v14",
        256: "results/FT/full/7b/para49/fill_dclm/domains_arxiv_all-legal_all-medical_all/e50/bs256_lr4e-05/overlap_1_16/E2_paraphrase_all_domains_local_no_explanation_match/eval_bundles/inf_mcqa_v14",
        512: "results/FT/full/7b/para49/fill_dclm/domains_arxiv_all-legal_all-medical_all/e50/bs512_lr4e-05/overlap_1_16/E2_paraphrase_all_domains_local_no_explanation_match/eval_bundles/inf_mcqa_v14",
    },
}

PROBE_LABELS = {
    "knowledge": "Factual",
    "inference": "Compositional",
}


def _abs(path: str) -> str:
    return str(REPO_ROOT / path)


def discover_probe_domains(run_path: str, probe_type: str) -> list[str]:
    suffix = f"_{probe_type}_probe"
    try:
        children = os.listdir(run_path)
    except FileNotFoundError:
        return []
    return sorted(
        name[: -len(suffix)]
        for name in children
        if name.endswith(suffix) and os.path.isdir(os.path.join(run_path, name))
    )


def collect_data() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    trajectory_rows = []
    final_rows = []
    missing = []

    for model, batch_runs in RUNS.items():
        for batch_size, rel_path in batch_runs.items():
            requested_path = _abs(rel_path)
            run_path = find_latest_run(requested_path)
            if not run_path:
                if os.path.isdir(requested_path):
                    run_path = requested_path
                else:
                    missing.append(f"{model} bs{batch_size}: no run path at {rel_path}")
                    continue

            for probe_type, label in PROBE_LABELS.items():
                domains = discover_probe_domains(run_path, probe_type)
                if not domains:
                    missing.append(f"{model} bs{batch_size} {label}: no {probe_type} probe metrics")
                    continue

                df = aggregate_across_domains(
                    run_path,
                    probe_type,
                    domains,
                    split_probes=False,
                    project_root=str(REPO_ROOT),
                )
                if df.empty or "log_prob" not in df.columns:
                    missing.append(f"{model} bs{batch_size} {label}: empty log_prob data")
                    continue

                series = (
                    df[["step", "log_prob"]]
                    .dropna()
                    .groupby("step", as_index=False)["log_prob"]
                    .mean()
                    .sort_values("step")
                )
                for _, row in series.iterrows():
                    trajectory_rows.append(
                        {
                            "Model": model,
                            "BatchSize": batch_size,
                            "Type": label,
                            "Step": int(row["step"]),
                            "Value": float(row["log_prob"]),
                            "RunPath": run_path,
                        }
                    )

                final_value = get_final_step_value(df, value_col="log_prob")
                if not np.isnan(final_value):
                    final_rows.append(
                        {
                            "Model": model,
                            "BatchSize": batch_size,
                            "Type": label,
                            "Value": float(final_value),
                            "FinalStep": int(series["step"].max()),
                            "Domains": len(domains),
                            "Rows": len(df),
                            "RunPath": run_path,
                        }
                    )

    return pd.DataFrame(trajectory_rows), pd.DataFrame(final_rows), missing


def plot(df_traj: pd.DataFrame, df_final: pd.DataFrame) -> None:
    style = apply_plot_style("legacy")
    model_colors = {
        "1B": COLORS["model"]["1b"],
        "7B": COLORS["model"]["7b"],
    }
    batch_styles = {
        128: {"linestyle": "-", "marker": "o"},
        256: {"linestyle": "-.", "marker": "^"},
        512: {"linestyle": "--", "marker": "s"},
    }
    batch_sizes = sorted({batch_size for runs in RUNS.values() for batch_size in runs})

    fig = plt.figure(figsize=(16.8, 4.8))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 0.15, 1, 1], wspace=0.1)
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 3]),
        fig.add_subplot(gs[0, 4]),
    ]

    for index, probe_label in enumerate(("Factual", "Compositional")):
        ax = axes[index]
        subset = df_traj[df_traj["Type"] == probe_label]
        for model in ("1B", "7B"):
            for batch_size in batch_sizes:
                series = subset[
                    (subset["Model"] == model) & (subset["BatchSize"] == batch_size)
                ].sort_values("Step")
                if series.empty:
                    continue
                ax.plot(
                    series["Step"],
                    series["Value"],
                    color=model_colors[model],
                    linestyle=batch_styles[batch_size]["linestyle"],
                    linewidth=style["line_width"],
                )
        ax.set_title(probe_label)
        ax.set_xlabel("Exposure #")
        ax.set_xticks([0, 25, 50])
        ax.grid(True, alpha=style["grid_alpha"])
    axes[0].set_ylabel("Log Prob.")
    axes[1].set_yticklabels([])

    traj_ylim = compute_unified_ylim(df_traj["Value"].dropna().tolist(), padding=0.05)
    if traj_ylim:
        apply_ylim(axes[:2], traj_ylim)

    for index, probe_label in enumerate(("Factual", "Compositional")):
        ax = axes[index + 2]
        subset = df_final[df_final["Type"] == probe_label]
        for model in ("1B", "7B"):
            series = subset[subset["Model"] == model].sort_values("BatchSize")
            if series.empty:
                continue
            ax.plot(
                series["BatchSize"],
                series["Value"],
                color=model_colors[model],
                linestyle="-",
                marker="o",
                linewidth=style["batch_line_width"],
                markersize=style["marker_size"],
                markerfacecolor=model_colors[model],
                markeredgecolor="#444444",
                markeredgewidth=style["marker_edge_width"],
            )
        ax.set_title(probe_label)
        ax.set_xscale("log", base=2)
        ax.set_xticks(batch_sizes)
        ax.set_xticklabels([str(batch_size) for batch_size in batch_sizes])
        ax.set_xlabel("Batch Size")
        ax.grid(True, which="major", ls="-", alpha=style["grid_alpha"])
        if index != 0:
            ax.set_yticklabels([])
    axes[2].set_ylabel("Final Log Prob")

    final_ylim = compute_unified_ylim(df_final["Value"].dropna().tolist(), padding=0.05)
    if final_ylim:
        apply_ylim(axes[2:], final_ylim)

    model_legend = [
        Line2D([0], [0], color=model_colors["1B"], lw=style["line_width"], label="1B"),
        Line2D([0], [0], color=model_colors["7B"], lw=style["line_width"], label="7B"),
    ]
    batch_legend = [
        Line2D([0], [0], color="gray", lw=style["line_width"], linestyle="-", label="bs128"),
        Line2D([0], [0], color="gray", lw=style["line_width"], linestyle="-.", label="bs256"),
        Line2D([0], [0], color="gray", lw=style["line_width"], linestyle="--", label="bs512"),
    ]
    axes[0].legend(handles=model_legend, loc="lower right", **style["legend"])
    axes[1].legend(handles=batch_legend, loc="lower right", **style["legend"])
    axes[2].legend(handles=model_legend, loc="lower left", **style["legend"])

    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.16, top=0.88, wspace=0.1)
    save_figure(fig, str(REPO_ROOT / "plots" / "batch_scaling_log_prob_para49_e50_recent"))


def main() -> None:
    df_traj, df_final, missing = collect_data()
    csv_path = REPO_ROOT / "plots" / "batch_scaling_log_prob_para49_e50_recent.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

    if missing:
        print("Missing data:")
        for item in missing:
            print(f"  - {item}")

    if df_traj.empty or df_final.empty:
        raise SystemExit("No data found to plot.")
    plot(df_traj, df_final)


if __name__ == "__main__":
    main()
