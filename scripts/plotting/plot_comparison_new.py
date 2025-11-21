"""
Lightweight replacement of plot_comparison.py focused on three plotting flows requested by the user:

1) 2x2 grid: factual (left) and compositional (right) plots for three methods (source_only, para9, para9_expl), plotting log_prob and hits@10 -> results in 2x2 (logprob & hits@10) for factual and compositional (so total 2*2 plots saved separately).
2) Repeat using split probes: create two figures (factual and compositional) each with 4 subplots (one per probe split group) showing log_prob only.
3) For each inference category from the original probe CSV, create a figure with four subplots (log_prob, hits@1, hits@10, hits@100) for probes filtered to that inference_type. There will be one figure per inference category.

This script uses helpers from scripts.plotting.plot_utils where possible.
"""

import os
import sys
import argparse
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

# Ensure the project root (containing 'scripts' and 'utils') is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from scripts.plotting.plot_utils import make_subplots, add_legend
from scripts.plotting.plot_comparison import aggregate_across_domains

# Fixed 7B runs requested by the user (paths are relative to PROJECT_ROOT).
RUNS_7B: Dict[str, str] = {
    "para9": "results/FT/full/7b/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/"
    "e100/bs64_lr2e-05/overlap_1_4/11_20_22_51",
    "para9_expl_textbooks_cyclefull": "results/FT/full/7b/probes_v9/newline2/para9_expl_textbooks_cyclefull/"
    "fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_21_02_11",
    "source_only": "results/FT/full/7b/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/"
    "e100/bs64_lr2e-05/overlap_1_4/11_20_22_51",
}

METHODS: List[Dict[str, str]] = [
    {"key": "source_only", "label": "Source", "run_key": "source_only"},
    {"key": "para9", "label": "Paraphrase", "run_key": "para9"},
    {"key": "textbooks", "label": "Paraphrase + Textbooks", "run_key": "para9_expl_textbooks_cyclefull"},
]

# RL-style domains present in the run directory (they correspond to *_knowledge_probe and *_inference_probe dirs).
DOMAINS = ["1_58", "DPO", "GRPO", "BOFT", "OFT", "QLoRA"]

PROBE_MASTER_CSV = os.path.join(
    PROJECT_ROOT,
    "data",
    "probes",
    "inference",
    "BOFT",
    "probes_v7.csv",
)


def _get_run_path(method_run_key: str) -> str:
    rel = RUNS_7B.get(method_run_key)
    return os.path.join(PROJECT_ROOT, rel) if rel is not None else ""


def _load_metric_series(
    run_path: str,
    probe_type: str,
    metric_col: str,
    domains: List[str],
    split_probes: bool = False,
):
    """
    Use aggregate_across_domains from plot_comparison.py to get per-step averages
    of a given metric (e.g., 'log_prob', 'hit_accuracy_at_10').
    """
    if not run_path or not os.path.isdir(run_path):
        return None

    df = aggregate_across_domains(
        run_path,
        probe_type,
        domains,
        split_probes=split_probes,
        project_root=PROJECT_ROOT,
        lima=False,
    )
    if df.empty or "step" not in df.columns or metric_col not in df.columns:
        return None

    out = (
        df.groupby("step")[metric_col]
        .mean()
        .reset_index()
        .sort_values("step")
    )
    return out


def plot_two_by_two(
    model_id: str,
    out_dir: str,
):
    """
    Plot factual (knowledge) vs compositional (inference) for the three 7B runs,
    once for log_prob and once for hits@10 (from llm_callbacks).

    - Figure 1: left=factual log_prob, right=compositional log_prob
    - Figure 2: left=factual hits@10, right=compositional hits@10
    """
    os.makedirs(out_dir, exist_ok=True)

    # --- Figure 1: log_prob ---
    fig_lp, axes_lp = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    for method in METHODS:
        run_path = _get_run_path(method["run_key"])
        # factual = knowledge, compositional = inference
        series_k = _load_metric_series(run_path, "knowledge", "log_prob", DOMAINS)
        series_i = _load_metric_series(run_path, "inference", "log_prob", DOMAINS)

        if series_k is not None and not series_k.empty:
            axes_lp[0].plot(series_k["step"], series_k["log_prob"], label=method["label"])
        if series_i is not None and not series_i.empty:
            axes_lp[1].plot(series_i["step"], series_i["log_prob"], label=method["label"])

    axes_lp[0].set_title("Factual Probes — log_prob")
    axes_lp[0].set_xlabel("step")
    axes_lp[0].set_ylabel("mean log_prob")

    axes_lp[1].set_title("Compositional Probes — log_prob")
    axes_lp[1].set_xlabel("step")
    axes_lp[1].set_ylabel("mean log_prob")

    add_legend(axes_lp[0])
    add_legend(axes_lp[1])

    out_path_lp = os.path.join(out_dir, f"factual_vs_compositional_logprob_7b_{model_id}.pdf")
    plt.savefig(out_path_lp, bbox_inches="tight")
    plt.close(fig_lp)

    # --- Figure 2: hits@10 ---
    fig_h, axes_h = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    for method in METHODS:
        run_path = _get_run_path(method["run_key"])
        series_k = _load_metric_series(run_path, "knowledge", "hit_accuracy_at_10", DOMAINS)
        series_i = _load_metric_series(run_path, "inference", "hit_accuracy_at_10", DOMAINS)

        if series_k is not None and not series_k.empty:
            axes_h[0].plot(series_k["step"], series_k["hit_accuracy_at_10"], label=method["label"])
        if series_i is not None and not series_i.empty:
            axes_h[1].plot(series_i["step"], series_i["hit_accuracy_at_10"], label=method["label"])

    axes_h[0].set_title("Factual Probes — hits@10")
    axes_h[0].set_xlabel("step")
    axes_h[0].set_ylabel("hits@10")

    axes_h[1].set_title("Compositional Probes — hits@10")
    axes_h[1].set_xlabel("step")
    axes_h[1].set_ylabel("hits@10")

    add_legend(axes_h[0])
    add_legend(axes_h[1])

    out_path_h = os.path.join(out_dir, f"factual_vs_compositional_hits10_7b_{model_id}.pdf")
    plt.savefig(out_path_h, bbox_inches="tight")
    plt.close(fig_h)

    print(f"2x2-style plots (log_prob and hits@10) saved to {out_dir}")


def plot_split_probes(
    model_id: str,
    out_dir: str,
):
    """
    Using split probes (origin labels from per-domain filter.json used by aggregate_across_domains),
    create:
      - one figure for factual (knowledge) with 4 subplots: one per origin group, log_prob only
      - one figure for compositional (inference) with the same layout.
    Each subplot overlays all three methods: Source, Paraphrase, Textbooks.
    """
    os.makedirs(out_dir, exist_ok=True)
    origins = ["Explanations Only", "Source Only", "Both", "Neither"]

    # factual (knowledge)
    fig_k, axes_k = make_subplots(2, 2, figsize=(10, 7), sharey=False)
    axes_k_arr = axes_k.flatten()

    for method in METHODS:
        run_path = _get_run_path(method["run_key"])
        df_k = aggregate_across_domains(
            run_path,
            "knowledge",
            DOMAINS,
            split_probes=True,
            project_root=PROJECT_ROOT,
            lima=False,
        )
        if df_k.empty or "origin" not in df_k.columns:
            continue
        for idx, origin in enumerate(origins):
            ax = axes_k_arr[idx]
            sub = df_k[df_k["origin"] == origin]
            if sub.empty:
                continue
            series = (
                sub.groupby("step")["log_prob"]
                .mean()
                .reset_index()
                .sort_values("step")
            )
            ax.plot(series["step"], series["log_prob"], label=method["label"])
            ax.set_title(origin)
            ax.set_xlabel("step")
            ax.set_ylabel("mean log_prob")

    add_legend(axes_k_arr[0])

    out_path_k = os.path.join(out_dir, f"split_probes_factual_logprob_7b_{model_id}.pdf")
    plt.savefig(out_path_k, bbox_inches="tight")
    plt.close(fig_k)

    # compositional (inference)
    fig_i, axes_i = make_subplots(2, 2, figsize=(10, 7), sharey=False)
    axes_i_arr = axes_i.flatten()

    for method in METHODS:
        run_path = _get_run_path(method["run_key"])
        df_i = aggregate_across_domains(
            run_path,
            "inference",
            DOMAINS,
            split_probes=True,
            project_root=PROJECT_ROOT,
            lima=False,
        )
        if df_i.empty or "origin" not in df_i.columns:
            continue
        for idx, origin in enumerate(origins):
            ax = axes_i_arr[idx]
            sub = df_i[df_i["origin"] == origin]
            if sub.empty:
                continue
            series = (
                sub.groupby("step")["log_prob"]
                .mean()
                .reset_index()
                .sort_values("step")
            )
            ax.plot(series["step"], series["log_prob"], label=method["label"])
            ax.set_title(origin)
            ax.set_xlabel("step")
            ax.set_ylabel("mean log_prob")

    add_legend(axes_i_arr[0])

    out_path_i = os.path.join(out_dir, f"split_probes_compositional_logprob_7b_{model_id}.pdf")
    plt.savefig(out_path_i, bbox_inches="tight")
    plt.close(fig_i)

    print(f"Split-probe (origin) plots saved to {out_dir}")


def plot_by_inference_category(
    out_dir: str,
    categories_csv: str = PROBE_MASTER_CSV,
):
    """
    For each inference_type category in the BOFT master probe CSV, use the BOFT
    compositional probe metrics from all three runs, and create a 1x4 subplot
    figure:
      - log_prob
      - hits@1
      - hits@10
      - hits@100
    Each subplot overlays Source, Paraphrase, and Textbooks (from their
    respective BOFT_inference_probe metrics).
    """
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(categories_csv):
        print(f"Probe CSV {categories_csv} not found; skipping per-category plots")
        return

    probes_df = pd.read_csv(categories_csv)
    if "inference_type" not in probes_df.columns:
        print("No inference_type column in master CSV; aborting")
        return

    # Assume probe_index in metrics corresponds to row index in probes_v7.csv
    probes_df = probes_df.reset_index().rename(columns={"index": "probe_index"})

    # Load and join BOFT inference metrics for each method
    method_metrics: Dict[str, pd.DataFrame] = {}
    for method in METHODS:
        run_path = _get_run_path(method["run_key"])
        metrics_path = os.path.join(
            run_path,
            "BOFT_inference_probe",
            "BOFT_inference_probe_metrics.csv",
        )
        if not os.path.exists(metrics_path):
            print(f"No metrics file {metrics_path} for {method['label']}; skipping this method")
            continue
        mdf = pd.read_csv(metrics_path)
        if "probe_index" not in mdf.columns:
            print(f"Expected 'probe_index' in metrics CSV for {method['label']}; skipping")
            continue
        merged = mdf.merge(
            probes_df[["probe_index", "inference_type"]],
            on="probe_index",
            how="left",
        )
        method_metrics[method["label"]] = merged

    if not method_metrics:
        print("No usable BOFT inference metrics found for any method; aborting")
        return

    # Gather all categories present across methods
    cats = set()
    for df in method_metrics.values():
        if "inference_type" in df.columns:
            cats.update(df["inference_type"].dropna().unique().tolist())
    cats = sorted(cats)

    if not cats:
        print("No non-null inference_type values; nothing to plot")
        return

    for cat in cats:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=False)
        has_any = False

        for method_label, df in method_metrics.items():
            sub = df[df["inference_type"] == cat]
            if sub.empty:
                continue

            agg = (
                sub.groupby("step")[
                    [
                        "log_prob",
                        "hit_accuracy_at_1",
                        "hit_accuracy_at_10",
                        "hit_accuracy_at_100",
                    ]
                ]
                .mean()
                .reset_index()
                .sort_values("step")
            )
            if agg.empty:
                continue

            has_any = True
            axes[0].plot(agg["step"], agg["log_prob"], label=method_label)
            axes[1].plot(agg["step"], agg["hit_accuracy_at_1"], label=method_label)
            axes[2].plot(agg["step"], agg["hit_accuracy_at_10"], label=method_label)
            axes[3].plot(agg["step"], agg["hit_accuracy_at_100"], label=method_label)

        if not has_any:
            plt.close(fig)
            continue

        axes[0].set_title("log_prob")
        axes[1].set_title("hits@1")
        axes[2].set_title("hits@10")
        axes[3].set_title("hits@100")

        for a in axes:
            a.set_xlabel("step")

        add_legend(axes[0])

        safe_cat = str(cat).replace(" ", "_")
        out_path = os.path.join(out_dir, f"inference_category_{safe_cat}.pdf")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

    print(f"Per-inference-category compositional plots saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="7b")
    parser.add_argument("--out_dir", type=str, default="plots")
    args = parser.parse_args()

    plot_two_by_two(args.model_id, args.out_dir)
    plot_split_probes(args.model_id, args.out_dir)
    plot_by_inference_category(args.out_dir)


if __name__ == '__main__':
    main()
