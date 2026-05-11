"""
Shared plotting utilities for the fine-tuning-or-retrieval project.

Provides:
  - Data loading:  aggregate_across_domains, load_metrics, load_probe_series
  - Final-value helpers:  get_final_step_value, get_final_val
  - Path/domain discovery:  find_latest_run, discover_domains
  - Subplot helpers:  make_subplots, make_probe_grid
  - Axis helpers:  compute_unified_ylim, apply_ylim, unify_ylim
  - Legend builders:  add_legend, make_line_legend, make_bar_legend
  - Bar chart helper:  plot_grouped_bars
  - Color registry:  COLORS
  - Style presets:  setup_style
"""

import json
import os
import re
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from utils import probe_paths


# ================================================================
# COLOR REGISTRY
# ================================================================

COLORS: Dict[str, Dict[str, str]] = {
    # Training data methods
    "method": {
        "source":        "#1f77b4",  # blue
        "paraphrase":    "#ff7f0e",  # orange
        "textbook":      "#8c564b",  # brown
        "stackexchange": "#d4ac0d",  # gold
        "blogs":         "#9467bd",  # purple
        "aux_views":     "#2ca02c",  # green
        "corrupted":     "#d62728",  # red
    },
    # Model sizes
    "model": {
        "1b":  "#ffd700",  # yellow
        "7b":  "#ff7f0e",  # orange
        "13b": "#d62728",  # red
        "32b": "#9467bd",  # purple
    },
    # Paraphrase levels (dark-to-light gradient)
    "paraphrase_level": {
        "para4":  "#D2B48C",  # light brown
        "para9":  "#b56535",  # medium brown
        "para19": "#795548",  # dark brown
        "para49": "#211511",  # very dark
    },
    # Data replay ratios
    "replay": {
        "none":       "gray",
        "1_1":        "deepskyblue",
        "1_3":        "cornflowerblue",
        "1_5":        "darkblue",
    },
    # Cleaning levels
    "cleaning": {
        "v1": "#e41a1c",  # red
        "v2": "#377eb8",  # blue
        "v3": "#4daf4a",  # green
    },
}


# ================================================================
# STYLE PRESETS
# ================================================================

def setup_style(preset: str = "default"):
    """
    Configure matplotlib rcParams.

    Presets:
      - "default": Standard academic style (axes.labelsize=14, etc.)
      - "publication": Larger fonts for camera-ready figures (axes.labelsize=18, etc.)

    Calls set_plot_style() from utils.llm_plotting internally,
    then applies preset-specific overrides.
    """
    # Import here to avoid hard coupling at module level
    import sys
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from utils.llm_plotting import set_plot_style
    set_plot_style()

    if preset == "publication":
        plt.rcParams.update({
            "axes.labelsize": 18,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16,
            "figure.titlesize": 22,
            "axes.titlesize": 20,
        })


# ================================================================
# PATH / DATA DISCOVERY
# ================================================================

def discover_domains(project_root: str) -> List[str]:
    """Return sorted list of domain names discovered under data/arxiv/cleaned."""
    domains_path = os.path.join(project_root, "data", "arxiv", "cleaned")
    if not os.path.isdir(domains_path):
        return []
    return sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(domains_path)
        if f.endswith(".tex") and os.path.isfile(os.path.join(domains_path, f))
    )


def _latest_timestamp_dir(path: str) -> Optional[str]:
    """Find the latest timestamped directory (MM_DD_HH_MM) inside *path*."""
    try:
        candidates = [
            d
            for d in os.listdir(path)
            if os.path.isdir(os.path.join(path, d))
            and re.match(r"\d{2}_\d{2}_\d{2}_\d{2}", d)
        ]
        if not candidates:
            return None
        latest = sorted(
            candidates,
            key=lambda x: datetime.strptime(x, "%m_%d_%H_%M"),
            reverse=True,
        )[0]
        return os.path.join(path, latest)
    except FileNotFoundError:
        return None


def find_latest_run(base_path: str) -> Optional[str]:
    """
    Resolve a run directory from a manually-given base path.

    Rules:
      1. If base_path itself looks like a run leaf (ends with timestamp or ``run``),
         return it.
      2. If a direct ``run`` subdirectory exists, return it.
      3. If timestamped subdirectories exist, return the latest.
      4. If ``overlap_1_4`` or ``no_overlap`` exist, descend one level and retry.
      5. Otherwise return ``None``.
    """
    if not base_path or not os.path.isdir(base_path):
        return None

    tail = os.path.basename(base_path.rstrip(os.sep))
    if tail == "run" or re.match(r"\d{2}_\d{2}_\d{2}_\d{2}", tail):
        return base_path

    run_dir = os.path.join(base_path, "run")
    if os.path.isdir(run_dir):
        return run_dir

    ts_dir = _latest_timestamp_dir(base_path)
    if ts_dir:
        return ts_dir

    for sub in ("overlap_1_4", "no_overlap"):
        candidate = os.path.join(base_path, sub)
        if os.path.isdir(candidate):
            run_dir = os.path.join(candidate, "run")
            if os.path.isdir(run_dir):
                return run_dir
            ts_dir = _latest_timestamp_dir(candidate)
            if ts_dir:
                return ts_dir

    return None


# ================================================================
# DATA LOADING
# ================================================================

def aggregate_across_domains(
    run_path: str,
    probe_type: str,
    domains: Sequence[str],
    split_probes: bool = False,
    project_root: str = ".",
    lima: bool = False,
) -> pd.DataFrame:
    """
    Load and concatenate probe CSVs across all *domains* from a single run.

    Parameters
    ----------
    run_path : str
        Resolved path to a run directory containing per-domain probe folders.
    probe_type : str
        ``"knowledge"`` or ``"inference"``.
    domains : sequence of str
        Domain names (e.g. ``["DPO", "1_58", ...]``).
    split_probes : bool
        If True, annotate each row with its probe origin using ``filter.json``.
    project_root : str
        Repository root.
    lima : bool
        If True, read from the ``*_lima_*`` probe directories.

    Returns
    -------
    pd.DataFrame
        Combined dataframe with columns at minimum: ``step``, ``log_prob``, ``domain``.
        Empty DataFrame if no data found.
    """
    all_domain_dfs: list[pd.DataFrame] = []
    for domain in domains:
        if probe_type == "knowledge":
            probe_dir = f"{domain}_lima_knowledge_probe" if lima else f"{domain}_knowledge_probe"
            file_name = f"{domain}_knowledge_probe_metrics.csv"
        else:
            probe_dir = f"{domain}_lima_inference_probe" if lima else f"{domain}_inference_probe"
            file_name = f"{domain}_inference_probe_metrics.csv"

        metrics_path = os.path.join(run_path, probe_dir, file_name)

        if not os.path.exists(metrics_path) or os.path.getsize(metrics_path) == 0:
            if not os.path.exists(metrics_path):
                print(f"Warning: File not found at {metrics_path}")
            else:
                print(f"Warning: File is empty at {metrics_path}")
            continue

        df = pd.read_csv(metrics_path)
        if "step" not in df.columns or "log_prob" not in df.columns:
            print(f"Warning: 'step' or 'log_prob' column not found in {metrics_path}. Skipping.")
            continue

        df["step"] = pd.to_numeric(df["step"], errors="coerce")
        df["log_prob"] = pd.to_numeric(df["log_prob"], errors="coerce")
        df.dropna(subset=["step", "log_prob"], inplace=True)
        if df.empty:
            continue
        df["step"] = df["step"].astype(int)
        df["domain"] = domain

        if split_probes:
            probe_folder = "inference" if probe_type == "inference" else "facts"
            filter_path = str(probe_paths.resolve_filter_path(probe_folder, domain))
            if os.path.exists(filter_path):
                with open(filter_path, "r") as f:
                    filter_data = json.load(f)
                df["origin"] = "Both"
                for origin_key, origin_label in [
                    ("in_explanations_only", "Explanations Only"),
                    ("in_source_only", "Source Only"),
                    ("in_both", "Both"),
                    ("in_neither", "Neither"),
                ]:
                    indices = filter_data.get(origin_key, [])
                    df.loc[df["probe_index"].isin(indices), "origin"] = origin_label
            else:
                print(
                    f"Warning: filter.json not found for domain {domain} in {probe_folder}. "
                    "Probes will not be split by origin."
                )
                df["origin"] = "Unknown"

        all_domain_dfs.append(df)

    if not all_domain_dfs:
        return pd.DataFrame()
    return pd.concat(all_domain_dfs, ignore_index=True)


def load_metrics(
    run_path: str,
    probe_type: str,
    domains: Sequence[str],
    project_root: str,
    metrics: Sequence[str] = ("log_prob",),
    with_lima: bool = False,
    filter_file: Optional[str] = None,
    exclude_origins: Optional[Sequence[str]] = None,
    aggregate: bool = True,
) -> Optional[pd.DataFrame]:
    """
    High-level loader: resolve path -> load CSVs -> optionally filter probes
    -> aggregate across domains -> optionally append LIMA continuation.

    Parameters
    ----------
    run_path : str
        Base path (will be resolved via ``find_latest_run``).
    probe_type : str
        ``"knowledge"`` or ``"inference"``.
    domains : sequence of str
        Domain names.
    project_root : str
        Repository root.
    metrics : sequence of str
        Metric columns to keep (e.g. ``("log_prob",)`` or
        ``("log_prob", "target_rank")``).
    with_lima : bool
        Append LIMA probe data after the base fine-tuning steps.
    filter_file : str or None
        Name of a filter JSON inside the canonical probe directory
        (e.g. ``"filter.json"`` or ``"filter_em.json"``). When set,
        ``split_probes`` is enabled in ``aggregate_across_domains``.
    exclude_origins : sequence of str or None
        If *filter_file* is set, drop rows whose ``origin`` is in this list
        (e.g. ``["Source Only"]``).
    aggregate : bool
        If True (default), return one row per step with mean metric values
        across domains/probes.  If False, return the raw per-probe DataFrame.

    Returns
    -------
    pd.DataFrame or None
        DataFrame with ``step`` + requested metric columns, or None on failure.
    """
    resolved = find_latest_run(run_path)
    if not resolved or not os.path.isdir(resolved):
        return None

    split = filter_file is not None
    base_df = aggregate_across_domains(
        resolved, probe_type, domains,
        split_probes=split, project_root=project_root,
    )
    if base_df.empty:
        return None

    # Probe filtering
    if exclude_origins and "origin" in base_df.columns:
        base_df = base_df[~base_df["origin"].isin(exclude_origins)]
        if base_df.empty:
            return None

    keep_cols = ["step"] + [m for m in metrics if m in base_df.columns]
    if not aggregate:
        extra = [c for c in ("domain", "probe_index", "origin", "probe", "target")
                 if c in base_df.columns]
        return base_df[list(dict.fromkeys(keep_cols + extra))]

    # Aggregate: mean per step
    base_agg = base_df.groupby("step")[list(metrics)].mean().reset_index()
    base_agg["step"] = base_agg["step"].astype(int)
    base_agg.sort_values("step", inplace=True)

    if not with_lima:
        return base_agg

    # LIMA continuation
    max_step_ft = int(base_agg["step"].max())
    lima_df = aggregate_across_domains(
        resolved, probe_type, domains,
        split_probes=split, project_root=project_root, lima=True,
    )
    if lima_df.empty:
        return base_agg

    if exclude_origins and "origin" in lima_df.columns:
        lima_df = lima_df[~lima_df["origin"].isin(exclude_origins)]
        if lima_df.empty:
            return base_agg

    lima_agg = lima_df.groupby("step")[list(metrics)].mean().reset_index()
    lima_agg["step"] = lima_agg["step"].astype(int)
    if max_step_ft > 0:
        lima_agg["step"] += max_step_ft
    lima_agg.sort_values("step", inplace=True)

    combined = pd.concat([base_agg, lima_agg], ignore_index=True)
    combined.sort_values("step", inplace=True)
    return combined


def load_probe_series(
    run_path: str,
    probe_type: str,
    domains,
    project_root: str,
    with_lima: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Load mean log_prob vs. step for a single run, optionally appending LIMA
    continuation.  Thin wrapper around ``load_metrics`` for backward compat.
    """
    return load_metrics(
        run_path, probe_type, domains, project_root,
        metrics=("log_prob",), with_lima=with_lima,
    )


# ================================================================
# METRIC EXTRACTORS
# ================================================================

def get_final_step_value(df: pd.DataFrame, value_col: str = "log_prob") -> float:
    """
    Return the mean of *value_col* at the maximum ``step`` in *df*.
    Returns ``np.nan`` when not available.
    """
    if df is None or df.empty:
        return np.nan
    if "step" not in df.columns or value_col not in df.columns:
        return np.nan
    s = pd.to_numeric(df["step"], errors="coerce")
    v = pd.to_numeric(df[value_col], errors="coerce")
    mask = s.notna() & v.notna()
    if not mask.any():
        return np.nan
    max_step = int(s[mask].max())
    vals = v[mask & (s == max_step)]
    return float(vals.mean()) if not vals.empty else np.nan


def get_final_val(
    run_path: str,
    probe_type: str,
    domains: Sequence[str],
    project_root: str,
    metric: str = "log_prob",
) -> Optional[float]:
    """
    Shorthand: load a run and return the final-step value for *metric*.
    Returns None if data is missing.
    """
    df = load_metrics(run_path, probe_type, domains, project_root, metrics=(metric,))
    if df is None or df.empty:
        return None
    max_step = df["step"].max()
    return float(df.loc[df["step"] == max_step, metric].mean())


# ================================================================
# SUBPLOT HELPERS
# ================================================================

def make_subplots(
    nrows: int,
    ncols: int,
    figsize: Tuple[float, float],
    sharey: Union[bool, str] = False,
    hide_shared_yticks_on: Optional[str] = "right",
):
    """
    Create subplots with optional y-sharing and convenience tick hiding.

    Parameters
    ----------
    sharey : False | True | 'row' | 'col'
    hide_shared_yticks_on : 'right' | 'all_but_left' | None
    """
    fig, axes = plt.subplots(
        nrows, ncols, figsize=figsize,
        sharey=sharey if isinstance(sharey, (bool, str)) else False,
    )
    axes_arr = np.atleast_2d(axes)
    if hide_shared_yticks_on:
        for r in range(axes_arr.shape[0]):
            for c in range(axes_arr.shape[1]):
                if c != 0:
                    plt.setp(axes_arr[r, c].get_yticklabels(), visible=False)
    return fig, axes


def make_probe_grid(
    probe_types: Sequence[Tuple[str, str]] = (("knowledge", "Factual"), ("inference", "Compositional")),
    model_sizes: Optional[Sequence[str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    sharey: Union[bool, str] = True,
) -> Tuple[plt.Figure, dict]:
    """
    Create a figure with one column per probe type (and optionally one row per model size).

    Returns ``(fig, axes_dict)`` where ``axes_dict`` maps
    ``(model_size, probe_key)`` -> ``Axes`` (or ``(probe_key,)`` -> ``Axes`` when
    *model_sizes* is None).

    Example
    -------
    >>> fig, ax = make_probe_grid()
    >>> ax["knowledge"].plot(...)
    >>> ax["inference"].plot(...)
    """
    n_probes = len(probe_types)
    if model_sizes:
        nrows = len(model_sizes)
        ncols = n_probes
        default_figsize = (5 * ncols, 4 * nrows)
    else:
        nrows = 1
        ncols = n_probes
        default_figsize = (5 * ncols, 4)

    if figsize is None:
        figsize = default_figsize

    fig, raw_axes = make_subplots(nrows, ncols, figsize=figsize, sharey=sharey)
    raw_axes = np.atleast_2d(raw_axes)

    axes_dict: dict = {}
    for r in range(nrows):
        for c, (probe_key, probe_label) in enumerate(probe_types):
            ax = raw_axes[r, c]
            if model_sizes:
                model = model_sizes[r]
                ax.set_title(f"{model.upper()}: {probe_label} Probes")
                axes_dict[(model, probe_key)] = ax
                if r == nrows - 1:
                    ax.set_xlabel("Training Steps")
                if c == 0:
                    ax.set_ylabel("Mean Log Probability")
            else:
                ax.set_title(f"{probe_label} Probes")
                axes_dict[probe_key] = ax
                ax.set_xlabel("Training Steps")
                if c == 0:
                    ax.set_ylabel("Mean Log Probability")

    return fig, axes_dict


# ================================================================
# AXIS HELPERS
# ================================================================

def compute_unified_ylim(
    values: Optional[Iterable[float]] = None,
    axes: Optional[Iterable] = None,
    padding: float = 0.05,
) -> Optional[Tuple[float, float]]:
    """Compute unified y-limits from values and/or axes data limits."""
    y_min = np.inf
    y_max = -np.inf
    if values:
        arr = [v for v in values if v is not None and not np.isnan(v)]
        if arr:
            y_min = min(y_min, min(arr))
            y_max = max(y_max, max(arr))
    if axes:
        for ax in axes:
            d = ax.dataLim
            y_min = min(y_min, d.y0)
            y_max = max(y_max, d.y1)
    if not np.isfinite(y_min) or not np.isfinite(y_max):
        return None
    pad = (y_max - y_min) * padding
    return (y_min - pad, y_max + pad)


def apply_ylim(axes: Iterable, ylim: Tuple[float, float]):
    """Apply a y-limit to all provided axes."""
    for ax in axes:
        ax.set_ylim(ylim)


def unify_ylim(axes: Iterable, padding: float = 0.05):
    """Compute + apply unified y-limits across the given axes in one call."""
    axes_list = list(axes)
    ylim = compute_unified_ylim(axes=axes_list, padding=padding)
    if ylim:
        apply_ylim(axes_list, ylim)


# ================================================================
# LEGEND BUILDERS
# ================================================================

def add_legend(
    target,
    loc: str = "lower right",
    ncol: Optional[int] = None,
    fontsize: str = "medium",
    title_fontsize: str = "medium",
    handles_labels: Optional[Tuple[List, List]] = None,
    bbox_to_anchor: Optional[Tuple[float, float]] = None,
    frameon: bool = False,
):
    """Add a legend to an axes or figure (backward-compatible)."""
    if handles_labels:
        handles, labels = handles_labels
    else:
        if hasattr(target, "get_legend_handles_labels"):
            handles, labels = target.get_legend_handles_labels()
        elif hasattr(target, "axes"):
            handles, labels = (
                target.axes[0].get_legend_handles_labels() if target.axes else ([], [])
            )
        else:
            handles, labels = ([], [])
    if not handles:
        return
    kwargs = {"loc": loc, "fontsize": fontsize, "title_fontsize": title_fontsize, "frameon": frameon}
    if ncol is not None:
        kwargs["ncol"] = ncol
    if bbox_to_anchor is not None:
        kwargs["bbox_to_anchor"] = bbox_to_anchor
    target.legend(handles, labels, **kwargs)


def make_line_legend(
    style_map: Dict[str, Dict[str, str]],
    extra: Optional[List[Tuple[str, dict]]] = None,
) -> List[Line2D]:
    """
    Build ``Line2D`` legend handles from a style map.

    Parameters
    ----------
    style_map : dict
        ``{label: {"color": ..., "linestyle": ..., ...}}``
        Any key accepted by ``Line2D`` works (color, linestyle, linewidth, marker, ...).
    extra : list of (label, kwargs) tuples, optional
        Additional legend entries (e.g. for factual/compositional line-style indicators).

    Returns
    -------
    list of Line2D
    """
    handles = []
    for label, kw in style_map.items():
        props = {"linewidth": 2}
        props.update(kw)
        handles.append(Line2D([0], [0], label=label, **props))
    if extra:
        for label, kw in extra:
            props = {"linewidth": 2}
            props.update(kw)
            handles.append(Line2D([0], [0], label=label, **props))
    return handles


def make_bar_legend(
    color_map: Optional[Dict[str, str]] = None,
    include_factual_compositional: bool = True,
    factual_alpha: float = 0.8,
    compositional_alpha: float = 0.5,
    compositional_hatch: str = "//",
) -> List[mpatches.Patch]:
    """
    Build ``Patch`` legend handles for grouped bar charts.

    Parameters
    ----------
    color_map : dict or None
        ``{label: color}`` for category patches.
    include_factual_compositional : bool
        Add generic Factual (solid) + Compositional (hatched) indicators.
    """
    handles: List[mpatches.Patch] = []
    if color_map:
        for label, color in color_map.items():
            handles.append(mpatches.Patch(facecolor=color, alpha=factual_alpha, label=label))
    if include_factual_compositional:
        handles.append(mpatches.Patch(facecolor="gray", alpha=factual_alpha, label="Factual"))
        handles.append(
            mpatches.Patch(
                facecolor="gray",
                alpha=compositional_alpha,
                hatch=compositional_hatch,
                edgecolor="black",
                label="Compositional",
            )
        )
    return handles


# ================================================================
# BAR CHART HELPER
# ================================================================

def plot_grouped_bars(
    ax,
    labels: Sequence[str],
    factual_values: Sequence[float],
    compositional_values: Sequence[float],
    color_map: Optional[Dict[str, str]] = None,
    default_color: str = "gray",
    bar_width: float = 0.35,
    factual_alpha: float = 0.8,
    compositional_alpha: float = 0.5,
    compositional_hatch: str = "//",
    xtick_labels: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    zero_line: bool = True,
):
    """
    Plot a grouped bar chart (factual solid + compositional hatched).

    Parameters
    ----------
    ax : matplotlib Axes
    labels : sequence of str
        Category labels (used to look up colors in *color_map*).
    factual_values, compositional_values : sequences of float
    color_map : dict mapping label -> color, optional
    xtick_labels : override display labels on x-axis
    """
    x = np.arange(len(labels))
    colors = [color_map.get(l, default_color) if color_map else default_color for l in labels]
    f_vals = [v if v is not None else 0 for v in factual_values]
    c_vals = [v if v is not None else 0 for v in compositional_values]

    ax.bar(x - bar_width / 2, f_vals, bar_width, color=colors, alpha=factual_alpha)
    ax.bar(
        x + bar_width / 2, c_vals, bar_width,
        color=colors, hatch=compositional_hatch,
        alpha=compositional_alpha, edgecolor="black",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels or labels, rotation=25, ha="right")
    if title:
        ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    if zero_line:
        ax.axhline(0, color="black", linewidth=0.8)


# ================================================================
# STEP TRANSFORMATIONS
# ================================================================

def transform_to_exposure_steps(df: pd.DataFrame, strategy_name: str) -> pd.DataFrame:
    """
    Transform the ``step`` column to ``Exposure Steps`` based on the data replay strategy.

    Supported strategy names (substring matching):
      - ``"No Data Replay"`` -> identity
      - ``"fill"`` without interleave -> identity
      - ``"interleave 1 batch"`` -> every 2nd step
      - ``"interleave 2 batches"`` -> every 3rd step
      - ``"(1:1) via interleave"`` -> every 2nd step
      - ``"(1:3) via interleave"`` -> every 4th step
      - ``"(1:5) via interleave"`` -> every 6th step
    """
    if df.empty:
        return df
    df = df.copy()

    if "No Data Replay" in strategy_name:
        df["Exposure Steps"] = df["step"]
    elif "fill" in strategy_name:
        if "interleave 1 batch" in strategy_name:
            df = df[(df["step"] == 0) | ((df["step"] > 0) & ((df["step"] - 1) % 2 == 0))].copy()
            df["Exposure Steps"] = (df["step"] + 1) // 2
        elif "interleave 2 batches" in strategy_name:
            df = df[(df["step"] - 1) % 3 == 0].copy()
            df["Exposure Steps"] = (df["step"] + 2) // 3
        else:
            df["Exposure Steps"] = df["step"]
    elif "(1:1) via interleave" in strategy_name:
        df = df[(df["step"] == 0) | ((df["step"] > 0) & ((df["step"] - 1) % 2 == 0))].copy()
        df["Exposure Steps"] = (df["step"] + 1) // 2
    elif "(1:3) via interleave" in strategy_name:
        df = df[(df["step"] == 0) | ((df["step"] > 0) & ((df["step"] - 1) % 4 == 0))].copy()
        df["Exposure Steps"] = (df["step"] - 1) // 4 + 1
    elif "(1:5) via interleave" in strategy_name:
        df = df[(df["step"] == 0) | ((df["step"] > 0) & ((df["step"] - 1) % 6 == 0))].copy()
        df["Exposure Steps"] = (df["step"] - 1) // 6 + 1
    else:
        df["Exposure Steps"] = df["step"]

    return df


# ================================================================
# CONVENIENCE: SAVE
# ================================================================

def save_plot(filename: str, output_dir: str = "plots", **savefig_kwargs):
    """
    ``os.makedirs`` + ``plt.savefig`` + ``plt.close`` in one call.

    Default kwargs: ``bbox_inches='tight'``.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    kw = {"bbox_inches": "tight"}
    kw.update(savefig_kwargs)
    plt.savefig(path, **kw)
    plt.close()
    print(f"Saved plot to {path}")
    return path
