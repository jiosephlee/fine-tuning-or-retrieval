import os
import re
from datetime import datetime
from typing import Iterable, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.plotting.plot_comparison import aggregate_across_domains  # noqa: E402


# ---------- PATH/DATA DISCOVERY ----------
def discover_domains(project_root: str) -> List[str]:
    """
    Return sorted list of domain names discovered under data/arxiv/cleaned.
    """
    domains_path = os.path.join(project_root, 'data', 'arxiv', 'cleaned')
    if not os.path.isdir(domains_path):
        return []
    return sorted(
        [
            os.path.splitext(f)[0]
            for f in os.listdir(domains_path)
            if f.endswith('.tex') and os.path.isfile(os.path.join(domains_path, f))
        ]
    )


def _latest_timestamp_dir(path: str) -> Optional[str]:
    """
    Find the latest timestamped directory (MM_DD_HH_MM) inside path.
    """
    try:
        candidates = [
            d for d in os.listdir(path)
            if os.path.isdir(os.path.join(path, d)) and re.match(r'\d{2}_\d{2}_\d{2}_\d{2}', d)
        ]
        if not candidates:
            return None
        latest = sorted(
            candidates,
            key=lambda x: datetime.strptime(x, '%m_%d_%H_%M'),
            reverse=True
        )[0]
        return os.path.join(path, latest)
    except FileNotFoundError:
        return None


def find_latest_run(base_path: str) -> Optional[str]:
    """
    Resolve a 'run' directory from a manually-given base path.
    Rules (kept simple and robust):
      - If base_path itself is a valid directory that already looks like a run leaf,
        return it (e.g., .../09_24_11_58 or .../run).
      - Else, if a direct 'run' subdirectory exists, return it.
      - Else, if timestamped subdirectories exist, return the latest.
      - Else, if typical overlap/no_overlap exists, descend one level and search for
        a run or timestamped dir.
      - Otherwise, return None.
    """
    if not base_path or not os.path.isdir(base_path):
        return None

    tail = os.path.basename(base_path.rstrip(os.sep))
    if tail == 'run' or re.match(r'\d{2}_\d{2}_\d{2}_\d{2}', tail):
        return base_path

    # Direct 'run' folder
    run_dir = os.path.join(base_path, 'run')
    if os.path.isdir(run_dir):
        return run_dir

    # Direct timestamped folder
    ts_dir = _latest_timestamp_dir(base_path)
    if ts_dir:
        return ts_dir

    # Overlap structures
    for sub in ('overlap_1_4', 'no_overlap'):
        candidate = os.path.join(base_path, sub)
        if os.path.isdir(candidate):
            # Try run inside
            run_dir = os.path.join(candidate, 'run')
            if os.path.isdir(run_dir):
                return run_dir
            ts_dir = _latest_timestamp_dir(candidate)
            if ts_dir:
                return ts_dir

    return None


# ---------- PLOTTING HELPERS ----------
def make_subplots(
    nrows: int,
    ncols: int,
    figsize: Tuple[float, float],
    sharey: Union[bool, str] = False,
    hide_shared_yticks_on: Optional[str] = 'right',
):
    """
    Create subplots with optional y-sharing and convenience tick hiding.
    - sharey: False | True | 'row' | 'col' (passed through to plt.subplots)
    - hide_shared_yticks_on: 'right' | 'all_but_left' | None
    Returns (fig, axes).
    """
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=sharey if isinstance(sharey, (bool, str)) else False)
    # Normalize axes to a 2D list for uniform handling
    axes_arr = np.atleast_2d(axes)
    if hide_shared_yticks_on:
        if hide_shared_yticks_on == 'right':
            # Hide y tick labels on all but the first column
            for r in range(axes_arr.shape[0]):
                for c in range(1, axes_arr.shape[1]):
                    plt.setp(axes_arr[r, c].get_yticklabels(), visible=False)
        elif hide_shared_yticks_on == 'all_but_left':
            for r in range(axes_arr.shape[0]):
                for c in range(axes_arr.shape[1]):
                    if c != 0:
                        plt.setp(axes_arr[r, c].get_yticklabels(), visible=False)
    return fig, axes


def add_legend(
    target,
    loc: str = 'lower right',
    ncol: Optional[int] = None,
    fontsize: str = 'medium',
    title_fontsize: str = 'medium',
    handles_labels: Optional[Tuple[List, List]] = None,
    bbox_to_anchor: Optional[Tuple[float, float]] = None,
    frameon: bool = False,
):
    """
    Add a legend to either a figure or an axes.
    If handles_labels is provided, it should be (handles, labels).
    """
    if handles_labels:
        handles, labels = handles_labels
    else:
        if hasattr(target, 'get_legend_handles_labels'):
            handles, labels = target.get_legend_handles_labels()
        elif hasattr(target, 'axes'):
            handles, labels = target.axes[0].get_legend_handles_labels() if target.axes else ([], [])
        else:
            handles, labels = ([], [])
    if not handles:
        return
    kwargs = {
        'loc': loc,
        'fontsize': fontsize,
        'title_fontsize': title_fontsize,
        'frameon': frameon,
    }
    if ncol is not None:
        kwargs['ncol'] = ncol
    if bbox_to_anchor is not None:
        kwargs['bbox_to_anchor'] = bbox_to_anchor
    if hasattr(target, 'legend'):  # Axes
        target.legend(handles, labels, **kwargs)
    else:  # Figure
        target.legend(handles, labels, **kwargs)


def compute_unified_ylim(values: Optional[Iterable[float]] = None, axes: Optional[Iterable] = None, padding: float = 0.05):
    """
    Compute unified y-limits either from a collection of values or by inspecting axes data limits.
    """
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
    """
    Apply a y-limit to all provided axes.
    """
    for ax in axes:
        ax.set_ylim(ylim)


def load_probe_series(
    run_path: str,
    probe_type: str,
    domains,
    project_root: str,
    with_lima: bool = False,
):
    """
    Load mean log_prob vs. step for a single run, optionally appending LIMA
    continuation after the base fine-tuning steps.
    """
    resolved = find_latest_run(run_path)
    if not resolved or not os.path.isdir(resolved):
        return None

    series_list = []
    max_step_ft = 0

    base_df = aggregate_across_domains(
        resolved,
        probe_type,
        domains,
        split_probes=False,
        project_root=project_root,
    )
    if not base_df.empty:
        g = base_df.groupby("step")["log_prob"].mean().reset_index()
        g["step"] = pd.to_numeric(g["step"], errors="coerce")
        g = g.dropna(subset=["step"])
        if not g.empty:
            g["step"] = g["step"].astype(int)
            g.sort_values("step", inplace=True)
            series_list.append(g)
            max_step_ft = int(g["step"].max())

    if with_lima:
        lima_df = aggregate_across_domains(
            resolved,
            probe_type,
            domains,
            split_probes=False,
            project_root=project_root,
            lima=True,
        )
        if not lima_df.empty:
            lg = lima_df.groupby("step")["log_prob"].mean().reset_index()
            lg["step"] = pd.to_numeric(lg["step"], errors="coerce")
            lg = lg.dropna(subset=["step"])
            if not lg.empty:
                lg["step"] = lg["step"].astype(int)
                if max_step_ft > 0:
                    lg["step"] += max_step_ft
                lg.sort_values("step", inplace=True)
                series_list.append(lg)

    if not series_list:
        return None

    combined = pd.concat(series_list, ignore_index=True)
    combined.sort_values("step", inplace=True)
    return combined


# ---------- METRIC EXTRACTORS ----------
def get_final_step_value(df: pd.DataFrame, value_col: str = 'log_prob') -> float:
    """
    Return the mean value at the maximum 'step' in df[value_col].
    Returns np.nan when not available.
    """
    if df is None or df.empty:
        return np.nan
    if 'step' not in df.columns or value_col not in df.columns:
        return np.nan
    s = pd.to_numeric(df['step'], errors='coerce')
    v = pd.to_numeric(df[value_col], errors='coerce')
    mask = s.notna() & v.notna()
    if not mask.any():
        return np.nan
    max_step = int(s[mask].max())
    vals = v[mask & (s == max_step)]
    return float(vals.mean()) if not vals.empty else np.nan


