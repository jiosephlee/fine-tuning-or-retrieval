import hashlib
import os
import re
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scripts.plotting import plot_utils
from scripts.plotting.plot_inference_mcqa_scaling import apply_plot_style

try:
    from utils.parameter_delta_metrics import (
        COMPONENTS,
        GINI_MEAN_ABS_EPS,
        LAYER_COMPONENTS,
        METRICS,
    )
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    COMPONENTS = (
        "embed_tokens",
        "gate_proj",
        "up_proj",
        "down_proj",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    )
    LAYER_COMPONENTS = tuple(component for component in COMPONENTS if component != "embed_tokens")
    METRICS = (
        "relative_delta_norm",
        "cosine_distance",
        "relative_delta_gini",
        "cosine_distance_gini",
    )
    GINI_MEAN_ABS_EPS = 1.5e-9

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots", "parameter_delta")


MLP_COMPONENTS = ("gate_proj", "up_proj", "down_proj")
MLP_COMPARISON_LABELS = {
    "source_only": "Source.",
    "para9": "Para. 9",
    "with_explanations": "Para. 9 + Aux.",
}
PLOT_GROUPS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("mlp", MLP_COMPONENTS),
)
LEGACY_PLOT_GROUP_NAMES = ("mlp_embed", "attention")
GROUP_LABELS = {
    "mlp": "MLP projections",
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
GRID_VIEW_LABELS = {
    "time": "Over training",
    "final_layer": "Final by layer",
}
VIEW_FILENAME_PARTS = {
    "time": "time",
    "final_layer": "layer",
}


def _read_csv(path: str) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def _setup_style() -> None:
    apply_plot_style()


def _save(fig, output_dir: str, filename: str, dpi: int = 200) -> str:
    output_path = os.path.join(output_dir, filename)
    plot_utils.save_figure(fig, output_path, suffixes=(".pdf", ".png"))
    return output_path


def _safe_filename_part(value: str, max_len: int = 80) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    safe = safe or "run"
    return safe[:max_len].rstrip("._-") or "run"


def _parameter_delta_run_slug(output_dir: str) -> str:
    parameter_delta_dir = os.path.abspath(output_dir)
    run_dir = os.path.dirname(parameter_delta_dir)
    try:
        rel_run_dir = os.path.relpath(run_dir, PROJECT_ROOT)
    except ValueError:
        rel_run_dir = run_dir
    run_name = _safe_filename_part(os.path.basename(run_dir))
    digest = hashlib.sha1(rel_run_dir.encode("utf-8")).hexdigest()[:10]
    return f"{run_name}_{digest}"


def _ordered_components(values: Iterable[str], include_embeddings: bool = True):
    allowed = COMPONENTS if include_embeddings else LAYER_COMPONENTS
    seen = set(values)
    ordered = [component for component in allowed if component in seen]
    ordered.extend(sorted(seen - set(ordered)))
    return ordered


def _clean_old_combined_plots(output_dir: str) -> None:
    for metric in METRICS:
        filenames = [f"time_{metric}.png", f"final_layer_{metric}.png"]
        for group_name in (*LEGACY_PLOT_GROUP_NAMES, *(name for name, _ in PLOT_GROUPS)):
            filenames.extend(
                [
                    f"time_{group_name}_{metric}.png",
                    f"final_layer_{group_name}_{metric}.png",
                ]
            )
        for filename in filenames:
            path = os.path.join(output_dir, filename)
            if os.path.exists(path):
                os.remove(path)


def _clean_old_run_plots(output_dir: str, run_slug: str) -> None:
    for group_name in (*LEGACY_PLOT_GROUP_NAMES, *(name for name, _ in PLOT_GROUPS)):
        suffix = f"parameter_delta_{group_name}_grid.png"
        path = os.path.join(output_dir, f"{run_slug}_{suffix}")
        if os.path.exists(path):
            os.remove(path)


def _metric_plot_values(values: pd.Series, metric: str) -> pd.Series:
    if metric != "relative_delta_norm":
        return values
    return np.log10(1.0 + values.clip(lower=0.0))


def _metric_axis_label(metric: str) -> str:
    metric_label = METRIC_LABELS.get(metric, metric)
    if metric == "relative_delta_norm":
        return f"log10(1 + {metric_label})"
    return metric_label


def _zero_tiny_gini_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    df = metrics_df.copy()
    id_cols = [
        col
        for col in df.columns
        if col not in {"metric", "value", "num_values"}
    ]
    key_cols = [f"__key_{col}" for col in id_cols]
    for col, key_col in zip(id_cols, key_cols):
        df[key_col] = df[col].where(df[col].notna(), "__nan__")

    for base_metric, gini_metric in (
        ("relative_delta_norm", "relative_delta_gini"),
        ("cosine_distance", "cosine_distance_gini"),
    ):
        base = df[df["metric"] == base_metric][key_cols + ["value"]].rename(
            columns={"value": "__base_value"}
        )
        if base.empty:
            continue
        gini_mask = df["metric"] == gini_metric
        merged = df.loc[gini_mask, key_cols].merge(base, on=key_cols, how="left")
        tiny = merged["__base_value"].fillna(np.inf).to_numpy(dtype=float) <= GINI_MEAN_ABS_EPS
        df.loc[df.index[gini_mask][tiny], "value"] = 0.0

    return df.drop(columns=key_cols)


def _plot_group_view_grid(
    metrics_df: pd.DataFrame,
    output_dir: str,
    run_slug: str,
    group_name: str,
    components: Tuple[str, ...],
    view: str,
) -> Optional[str]:
    group_df = metrics_df[metrics_df["component"].isin(components)]
    if group_df.empty:
        return None

    fig, axes = plot_utils.make_subplots(
        nrows=1,
        ncols=len(METRICS),
        figsize=(15, 4.2),
        hide_shared_yticks_on=None,
    )
    axes = np.asarray(axes).reshape(-1)
    handles_by_label = {}
    for ax, metric in zip(axes, METRICS):
        view_df = group_df[
            (group_df["view"] == view)
            & (group_df["metric"] == metric)
        ].copy()
        if view == "final_layer":
            view_df = view_df[view_df["component"] != "embed_tokens"]

        if view_df.empty:
            ax.text(
                0.5,
                0.5,
                "No data",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="0.45",
            )
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            include_embeddings = view != "final_layer"
            for component in _ordered_components(
                view_df["component"].dropna().astype(str),
                include_embeddings=include_embeddings,
            ):
                sub = view_df[view_df["component"] == component].sort_values(
                    "step" if view == "time" else "layer"
                )
                x_col = "step" if view == "time" else "layer"
                line, = ax.plot(
                    sub[x_col],
                    _metric_plot_values(sub["value"], metric),
                    marker="o",
                    markersize=3.5,
                    linewidth=2,
                    label=COMPONENT_LABELS.get(component, component),
                )
                handles_by_label[line.get_label()] = line

        ax.set_title(METRIC_LABELS.get(metric, metric))
        ax.set_xlabel("Training step" if view == "time" else "Layer")
        ax.set_ylabel(_metric_axis_label(metric))
        ax.grid(True, axis="y", alpha=0.25)

    if handles_by_label:
        labels = [
            COMPONENT_LABELS.get(component, component)
            for component in _ordered_components(components)
            if COMPONENT_LABELS.get(component, component) in handles_by_label
        ]
        handles = [handles_by_label[label] for label in labels]
        plot_utils.add_legend(
            fig,
            loc="lower center",
            handles_labels=(handles, labels),
            bbox_to_anchor=(0.5, -0.02),
            ncol=min(4, len(labels)),
            frameon=False,
        )

    fig.tight_layout(rect=(0, 0.1, 1, 1))
    view_name = VIEW_FILENAME_PARTS.get(view, view)
    filename = f"{run_slug}_parameter_delta_{group_name}_{view_name}_grid.png"
    return _save(fig, output_dir, filename)


def plot_parameter_delta_mlp_comparison(
    runs: Sequence[Tuple[str, str, str, str]],
    output_dir: Optional[str] = None,
    prefix: str = "7b",
) -> List[str]:
    _setup_style()
    plt.rcParams.update({
        "axes.labelsize": 13,
        "axes.titlesize": 15,
        "font.size": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })
    output_dir = output_dir or DEFAULT_PLOTS_DIR

    frames = []
    for method, label, color, parameter_delta_dir in runs:
        df = _read_csv(os.path.join(parameter_delta_dir, "parameter_delta_metrics.csv"))
        if df.empty:
            continue
        df = df.copy()
        df["method"] = method
        df["label"] = MLP_COMPARISON_LABELS.get(method, label)
        df["color"] = color
        frames.append(df)
    if not frames:
        return []

    metrics_df = _zero_tiny_gini_metrics(pd.concat(frames, ignore_index=True))
    saved_paths = []
    for view, view_name in VIEW_FILENAME_PARTS.items():
        view_df = metrics_df[
            (metrics_df["view"] == view)
            & (metrics_df["component"].isin(MLP_COMPONENTS))
        ].copy()
        if view_df.empty:
            continue

        fig, axes = plot_utils.make_subplots(
            nrows=len(MLP_COMPONENTS),
            ncols=len(METRICS),
            figsize=(15, 8.6),
            hide_shared_yticks_on=None,
        )
        axes = np.atleast_2d(axes)
        handles_by_label = {}

        for row, component in enumerate(MLP_COMPONENTS):
            for col, metric in enumerate(METRICS):
                ax = axes[row, col]
                sub_df = view_df[
                    (view_df["component"] == component)
                    & (view_df["metric"] == metric)
                ]
                x_col = "step" if view == "time" else "layer"
                for label in sub_df["label"].dropna().drop_duplicates():
                    method_df = sub_df[sub_df["label"] == label].sort_values(x_col)
                    if method_df.empty:
                        continue
                    color = method_df["color"].iloc[0]
                    line, = ax.plot(
                        method_df[x_col],
                        _metric_plot_values(method_df["value"], metric),
                        color=color,
                        linewidth=2,
                        marker="o",
                        markersize=3.0,
                        label=label,
                    )
                    handles_by_label[label] = line

                if row == 0:
                    ax.set_title(METRIC_LABELS.get(metric, metric))
                if row == len(MLP_COMPONENTS) - 1:
                    ax.set_xlabel("Training step" if view == "time" else "Layer")
                if col == 0:
                    ax.set_ylabel(COMPONENT_LABELS.get(component, component))
                ax.grid(True, axis="y", alpha=0.25)

        if handles_by_label:
            labels = list(handles_by_label.keys())
            handles = [handles_by_label[label] for label in labels]
            axes[0, 0].legend(
                handles,
                labels,
                loc="lower right",
                frameon=True,
                framealpha=0.9,
                borderpad=0.4,
                handlelength=1.8,
            )

        fig.tight_layout(pad=0.8, w_pad=0.7, h_pad=0.55)
        saved_paths.append(
            _save(
                fig,
                output_dir,
                f"{_safe_filename_part(prefix)}_parameter_delta_mlp_{view_name}_comparison.png",
                dpi=300,
            )
        )
    return saved_paths


def expected_grid_plot_filenames(parameter_delta_dir: Optional[str] = None) -> List[str]:
    if parameter_delta_dir:
        run_slug = _parameter_delta_run_slug(parameter_delta_dir)
        return [
            f"{run_slug}_parameter_delta_{group_name}_{view_name}_grid.png"
            for group_name, _ in PLOT_GROUPS
            for view_name in VIEW_FILENAME_PARTS.values()
        ]
    return [
        f"*_parameter_delta_{group_name}_{view_name}_grid.png"
        for group_name, _ in PLOT_GROUPS
        for view_name in VIEW_FILENAME_PARTS.values()
    ]


def expected_grid_plot_paths(
    parameter_delta_dir: str,
    plots_dir: Optional[str] = None,
) -> List[str]:
    plots_dir = plots_dir or DEFAULT_PLOTS_DIR
    return [
        os.path.join(plots_dir, filename)
        for filename in expected_grid_plot_filenames(parameter_delta_dir)
    ]


def plot_parameter_delta_outputs(
    output_dir: str,
    plots_dir: Optional[str] = None,
    clean_old_combined: bool = True,
) -> List[str]:
    _setup_style()
    plots_dir = plots_dir or DEFAULT_PLOTS_DIR
    metrics_df = _zero_tiny_gini_metrics(
        _read_csv(os.path.join(output_dir, "parameter_delta_metrics.csv"))
    )
    if metrics_df.empty:
        return []

    run_slug = _parameter_delta_run_slug(output_dir)
    saved_paths = []
    for group_name, components in PLOT_GROUPS:
        for view in VIEW_FILENAME_PARTS:
            saved_path = _plot_group_view_grid(
                metrics_df,
                plots_dir,
                run_slug,
                group_name,
                components,
                view,
            )
            if saved_path:
                saved_paths.append(saved_path)
    if saved_paths and clean_old_combined:
        _clean_old_combined_plots(plots_dir)
        _clean_old_run_plots(plots_dir, run_slug)
    return saved_paths
