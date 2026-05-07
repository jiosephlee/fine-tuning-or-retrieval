import os
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import pandas as pd

from utils.llm_plotting import set_plot_style
from utils.parameter_delta_metrics import METRICS, PROJECTIONS


COMPONENT_ORDER = ["gate", "up", "down", "mlp_all", "embed_tokens"]
COMPONENT_LABELS = {
    "gate": "gate_proj",
    "up": "up_proj",
    "down": "down_proj",
    "mlp_all": "MLP all",
    "embed_tokens": "embed_tokens",
}
METRIC_LABELS = {
    "relative_delta_norm": "Relative delta norm",
    "cosine_distance": "Cosine distance",
}


def _read_csv(path: str) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def _save(fig, path: str, dpi: int = 200) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _ordered_components(values: Iterable[str]):
    seen = set(values)
    ordered = [component for component in COMPONENT_ORDER if component in seen]
    ordered.extend(sorted(seen - set(ordered)))
    return ordered


def plot_aggregate_lines(scalar_df: pd.DataFrame, output_dir: str, value_col: str = "mean") -> None:
    if scalar_df.empty:
        return
    for metric in METRICS:
        df = scalar_df[scalar_df["metric"] == metric]
        if df.empty:
            continue
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        for component in _ordered_components(df["component"].dropna().astype(str)):
            sub = df[df["component"] == component].sort_values("step")
            ax.plot(
                sub["step"],
                sub[value_col],
                marker="o",
                linewidth=2,
                label=COMPONENT_LABELS.get(component, component),
            )
        ax.set_title(f"Parameter Delta: {METRIC_LABELS.get(metric, metric)}")
        ax.set_xlabel("Training step")
        ax.set_ylabel(value_col)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        _save(fig, os.path.join(output_dir, f"aggregate_{metric}_{value_col}.png"))


def plot_embedding_lines(scalar_df: pd.DataFrame, output_dir: str, value_col: str = "mean") -> None:
    if scalar_df.empty:
        return
    embed_df = scalar_df[scalar_df["component"] == "embed_tokens"]
    if embed_df.empty:
        return
    for metric in METRICS:
        df = embed_df[embed_df["metric"] == metric].sort_values("step")
        if df.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        ax.plot(df["step"], df[value_col], marker="o", linewidth=2)
        ax.set_title(f"Embedding Delta: {METRIC_LABELS.get(metric, metric)}")
        ax.set_xlabel("Training step")
        ax.set_ylabel(value_col)
        ax.grid(True, alpha=0.25)
        _save(fig, os.path.join(output_dir, f"embed_tokens_{metric}_{value_col}.png"))


def plot_layer_lines(layer_df: pd.DataFrame, output_dir: str, value_col: str = "mean") -> None:
    if layer_df.empty:
        return
    for metric in METRICS:
        metric_df = layer_df[layer_df["metric"] == metric]
        if metric_df.empty:
            continue
        for projection in PROJECTIONS:
            df = metric_df[metric_df["projection"] == projection]
            if df.empty:
                continue
            fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
            for layer in sorted(df["layer"].dropna().unique()):
                sub = df[df["layer"] == layer].sort_values("step")
                ax.plot(sub["step"], sub[value_col], linewidth=1.4, alpha=0.8, label=f"L{int(layer)}")
            ax.set_title(
                f"{COMPONENT_LABELS.get(projection, projection)} Layer Delta: "
                f"{METRIC_LABELS.get(metric, metric)}"
            )
            ax.set_xlabel("Training step")
            ax.set_ylabel(value_col)
            ax.grid(True, alpha=0.25)
            ax.legend(
                title="Layer",
                frameon=False,
                ncol=4,
                fontsize=7,
                loc="center left",
                bbox_to_anchor=(1.01, 0.5),
            )
            _save(fig, os.path.join(output_dir, f"layers_{projection}_{metric}_{value_col}.png"))


def plot_concentration(concentration_df: pd.DataFrame, output_dir: str) -> None:
    if concentration_df.empty:
        return
    stats = ["top_1pct_share", "top_5pct_share", "gini"]
    for metric in METRICS:
        metric_df = concentration_df[concentration_df["metric"] == metric]
        if metric_df.empty:
            continue
        for stat in stats:
            if stat not in metric_df.columns:
                continue
            fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
            for component in _ordered_components(metric_df["component"].dropna().astype(str)):
                sub = metric_df[metric_df["component"] == component].sort_values("step")
                ax.plot(
                    sub["step"],
                    sub[stat],
                    marker="o",
                    linewidth=2,
                    label=COMPONENT_LABELS.get(component, component),
                )
            ax.set_title(f"Parameter Delta Concentration: {METRIC_LABELS.get(metric, metric)}")
            ax.set_xlabel("Training step")
            ax.set_ylabel(stat)
            ax.grid(True, alpha=0.25)
            ax.legend(frameon=False)
            _save(fig, os.path.join(output_dir, f"concentration_{metric}_{stat}.png"))


def plot_final_alignment(
    aggregate_df: Optional[pd.DataFrame],
    layer_df: Optional[pd.DataFrame],
    output_dir: str,
) -> None:
    if aggregate_df is not None and not aggregate_df.empty:
        fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
        for component in _ordered_components(aggregate_df["component"].dropna().astype(str)):
            sub = aggregate_df[aggregate_df["component"] == component].sort_values("step")
            ax.plot(
                sub["step"],
                sub["alignment"],
                marker="o",
                linewidth=2,
                label=COMPONENT_LABELS.get(component, component),
            )
        ax.set_title("Final-Direction Alignment")
        ax.set_xlabel("Training step")
        ax.set_ylabel("cos(delta_t, delta_final)")
        ax.set_ylim(-1.05, 1.05)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        _save(fig, os.path.join(output_dir, "final_alignment_aggregate.png"))

    if layer_df is None or layer_df.empty:
        return
    for projection in PROJECTIONS:
        df = layer_df[layer_df["projection"] == projection]
        if df.empty:
            continue
        fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
        for layer in sorted(df["layer"].dropna().unique()):
            sub = df[df["layer"] == layer].sort_values("step")
            ax.plot(sub["step"], sub["alignment"], linewidth=1.4, alpha=0.8, label=f"L{int(layer)}")
        ax.set_title(f"{COMPONENT_LABELS.get(projection, projection)} Final-Direction Alignment")
        ax.set_xlabel("Training step")
        ax.set_ylabel("cos(delta_t, delta_final)")
        ax.set_ylim(-1.05, 1.05)
        ax.grid(True, alpha=0.25)
        ax.legend(
            title="Layer",
            frameon=False,
            ncol=4,
            fontsize=7,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
        )
        _save(fig, os.path.join(output_dir, f"final_alignment_layers_{projection}.png"))


def plot_parameter_delta_outputs(output_dir: str, plots_dir: Optional[str] = None) -> None:
    set_plot_style()
    plots_dir = plots_dir or os.path.join(output_dir, "plots")
    scalar_df = _read_csv(os.path.join(output_dir, "parameter_delta_scalar_metrics.csv"))
    layer_df = _read_csv(os.path.join(output_dir, "parameter_delta_layer_metrics.csv"))
    concentration_df = _read_csv(os.path.join(output_dir, "parameter_delta_concentration_metrics.csv"))
    alignment_scalar_df = _read_csv(os.path.join(output_dir, "parameter_delta_final_alignment_scalar.csv"))
    alignment_layer_df = _read_csv(os.path.join(output_dir, "parameter_delta_final_alignment_layer.csv"))

    plot_aggregate_lines(scalar_df, plots_dir)
    plot_embedding_lines(scalar_df, plots_dir)
    plot_layer_lines(layer_df, plots_dir)
    plot_concentration(concentration_df, plots_dir)
    plot_final_alignment(alignment_scalar_df, alignment_layer_df, plots_dir)
