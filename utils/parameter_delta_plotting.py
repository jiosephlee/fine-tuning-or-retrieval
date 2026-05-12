import os
from typing import Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from utils.llm_plotting import set_plot_style
from utils.parameter_delta_metrics import COMPONENTS, LAYER_COMPONENTS, METRICS


PLOT_GROUPS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("mlp_embed", ("embed_tokens", "gate_proj", "up_proj", "down_proj")),
    ("attention", ("q_proj", "k_proj", "v_proj", "o_proj")),
)
GROUP_LABELS = {
    "mlp_embed": "Embeddings + MLP projections",
    "attention": "Attention projections",
}
COMPONENT_LABELS = {
    "embed_tokens": "embed_tokens",
    "gate_proj": "gate_proj",
    "up_proj": "up_proj",
    "down_proj": "down_proj",
    "q_proj": "q_proj",
    "k_proj": "k_proj",
    "v_proj": "v_proj",
    "o_proj": "o_proj",
}
METRIC_LABELS = {
    "relative_delta_norm": "Relative delta norm",
    "cosine_distance": "Cosine distance",
    "relative_delta_gini": "Relative delta Gini",
    "cosine_distance_gini": "Cosine distance Gini",
}


def _read_csv(path: str) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def _save(fig, path: str, dpi: int = 200) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _ordered_components(values: Iterable[str], include_embeddings: bool = True):
    allowed = COMPONENTS if include_embeddings else LAYER_COMPONENTS
    seen = set(values)
    ordered = [component for component in allowed if component in seen]
    ordered.extend(sorted(seen - set(ordered)))
    return ordered


def _clean_old_combined_plots(output_dir: str) -> None:
    for metric in METRICS:
        for filename in (f"time_{metric}.png", f"final_layer_{metric}.png"):
            path = os.path.join(output_dir, filename)
            if os.path.exists(path):
                os.remove(path)


def plot_time_metrics(metrics_df: pd.DataFrame, output_dir: str) -> List[str]:
    saved_paths: List[str] = []
    time_df = metrics_df[metrics_df["view"] == "time"]
    if time_df.empty:
        return saved_paths
    for metric in METRICS:
        metric_df = time_df[time_df["metric"] == metric]
        if metric_df.empty:
            continue
        for group_name, components in PLOT_GROUPS:
            group_df = metric_df[metric_df["component"].isin(components)]
            if group_df.empty:
                continue
            fig, ax = plt.subplots(figsize=(9.5, 5.2), constrained_layout=True)
            for component in _ordered_components(group_df["component"].dropna().astype(str)):
                sub = group_df[group_df["component"] == component].sort_values("step")
                ax.plot(
                    sub["step"],
                    sub["value"],
                    marker="o",
                    linewidth=2,
                    label=COMPONENT_LABELS.get(component, component),
                )
            metric_label = METRIC_LABELS.get(metric, metric)
            ax.set_title(
                f"Parameter Delta Over Training: {metric_label} "
                f"({GROUP_LABELS.get(group_name, group_name)})"
            )
            ax.set_xlabel("Training step")
            ax.set_ylabel(metric_label)
            ax.grid(True, alpha=0.25)
            ax.legend(frameon=False, ncol=2)
            saved_paths.append(
                _save(fig, os.path.join(output_dir, f"time_{group_name}_{metric}.png"))
            )
    return saved_paths


def plot_final_layer_metrics(metrics_df: pd.DataFrame, output_dir: str) -> List[str]:
    saved_paths: List[str] = []
    layer_df = metrics_df[
        (metrics_df["view"] == "final_layer")
        & (metrics_df["component"] != "embed_tokens")
    ]
    if layer_df.empty:
        return saved_paths
    for metric in METRICS:
        metric_df = layer_df[layer_df["metric"] == metric]
        if metric_df.empty:
            continue
        for group_name, components in PLOT_GROUPS:
            group_df = metric_df[metric_df["component"].isin(components)]
            if group_df.empty:
                continue
            fig, ax = plt.subplots(figsize=(9.5, 5.2), constrained_layout=True)
            for component in _ordered_components(
                group_df["component"].dropna().astype(str),
                include_embeddings=False,
            ):
                sub = group_df[group_df["component"] == component].sort_values("layer")
                ax.plot(
                    sub["layer"],
                    sub["value"],
                    marker="o",
                    linewidth=2,
                    label=COMPONENT_LABELS.get(component, component),
                )
            metric_label = METRIC_LABELS.get(metric, metric)
            ax.set_title(
                f"Final Parameter Delta By Layer: {metric_label} "
                f"({GROUP_LABELS.get(group_name, group_name)})"
            )
            ax.set_xlabel("Layer")
            ax.set_ylabel(metric_label)
            ax.grid(True, alpha=0.25)
            ax.legend(frameon=False, ncol=2)
            saved_paths.append(
                _save(fig, os.path.join(output_dir, f"final_layer_{group_name}_{metric}.png"))
            )
    return saved_paths


def expected_split_plot_filenames() -> List[str]:
    filenames = []
    for metric in METRICS:
        for group_name, _ in PLOT_GROUPS:
            filenames.append(f"time_{group_name}_{metric}.png")
            filenames.append(f"final_layer_{group_name}_{metric}.png")
    return filenames


def plot_parameter_delta_outputs(
    output_dir: str,
    plots_dir: Optional[str] = None,
    clean_old_combined: bool = True,
) -> List[str]:
    set_plot_style()
    plots_dir = plots_dir or os.path.join(output_dir, "plots")
    metrics_df = _read_csv(os.path.join(output_dir, "parameter_delta_metrics.csv"))
    if metrics_df.empty:
        return []

    saved_paths = []
    saved_paths.extend(plot_time_metrics(metrics_df, plots_dir))
    saved_paths.extend(plot_final_layer_metrics(metrics_df, plots_dir))
    if saved_paths and clean_old_combined:
        _clean_old_combined_plots(plots_dir)
    return saved_paths
