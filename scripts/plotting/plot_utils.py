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
  - Style presets:  setup_style, apply_plot_style, get_style_preset
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path
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

STYLE_PRESETS = {
    "new": {
        "rc": {
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "font.size": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        },
        "line_width": 1.8,
        "batch_line_width": 1.8,
        "marker_size": 4.5,
        "marker_edge_width": 1.2,
        "grid_alpha": 0.25,
        "legend": {"frameon": False},
        "strategy_styles": {
            "Para 9": {"linestyle": "-", "marker": "o"},
            "Source": {"linestyle": "--", "marker": "s"},
        },
    },
    "legacy": {
        "rc": {
            "axes.labelsize": 18,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 14,
            "figure.titlesize": 22,
            "axes.titlesize": 20,
        },
        "line_width": 2,
        "batch_line_width": 1.5,
        "marker_size": 4.5,
        "marker_edge_width": 1.2,
        "grid_alpha": 0.1,
        "legend": {"frameon": True, "fontsize": 14},
        "strategy_styles": {
            "Para 9": {"linestyle": "-", "marker": "o"},
            "Source": {"linestyle": "--", "marker": "s"},
        },
    },
}


def get_style_preset(preset: str = "new") -> dict:
    if preset == "default":
        preset = "new"
    if preset == "publication":
        preset = "legacy"
    if preset not in STYLE_PRESETS:
        raise ValueError(f"Unknown plot style preset: {preset}")
    return STYLE_PRESETS[preset]


def apply_plot_style(preset: str = "new") -> dict:
    style = get_style_preset(preset)
    try:
        setup_style(preset)
    except ModuleNotFoundError as exc:
        if exc.name != "seaborn":
            raise
        plt.rcParams.update(style["rc"])
    return style


def save_figure(fig, output: str, suffixes: Sequence[str] = (".pdf", ".png")):
    output_path = Path(output)
    if output_path.suffix:
        output_path = output_path.with_suffix("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in suffixes:
        fig.savefig(str(output_path) + suffix, bbox_inches="tight", dpi=300)
        print(f"Saved {output_path}{suffix}")
    plt.close(fig)


def setup_style(preset: str = "default"):
    """
    Configure matplotlib rcParams.

    Presets:
      - "default" / "new": Standard academic style.
      - "publication" / "legacy": Larger legacy figure style.

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

    if preset == "default":
        preset = "new"
    if preset == "publication":
        preset = "legacy"
    if preset in STYLE_PRESETS:
        plt.rcParams.update(STYLE_PRESETS[preset]["rc"])


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


def _looks_like_run_leaf(path: str) -> bool:
    """Return True if *path* directly contains per-domain result folders."""
    try:
        children = [
            name
            for name in os.listdir(path)
            if os.path.isdir(os.path.join(path, name))
        ]
    except FileNotFoundError:
        return False

    probe_suffixes = (
        "_knowledge_probe",
        "_inference_probe",
        "_mcqa_probe",
        "_inference_mcqa_probe",
        "_corpus_perplexity",
    )
    return any(
        any(suffix in child for suffix in probe_suffixes)
        for child in children
    )


def _latest_named_dir(path: str, pattern: str) -> Optional[str]:
    """Find the newest child directory whose name matches *pattern*."""
    try:
        candidates = [
            os.path.join(path, name)
            for name in os.listdir(path)
            if os.path.isdir(os.path.join(path, name)) and re.match(pattern, name)
        ]
    except FileNotFoundError:
        return None
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda p: (os.path.getmtime(p), os.path.basename(p)),
        reverse=True,
    )[0]


def find_latest_run(base_path: str) -> Optional[str]:
    """
    Resolve a run directory from a manually-given base path.

    Rules:
      1. If base_path itself looks like a run leaf (ends with timestamp or ``run``),
         return it.
      2. If base_path directly contains per-domain result folders, return it.
      3. If a direct ``run`` subdirectory exists, return it.
      4. If timestamped subdirectories exist, return the latest.
      5. If an ``E*`` experiment directory exists, descend into the newest one.
      6. If ``overlap_*`` or ``no_overlap`` directories exist, descend and retry.
      7. Otherwise return ``None``.
    """
    if not base_path or not os.path.isdir(base_path):
        return None

    tail = os.path.basename(base_path.rstrip(os.sep))
    if tail == "run" or re.match(r"\d{2}_\d{2}_\d{2}_\d{2}", tail):
        return base_path

    if _looks_like_run_leaf(base_path):
        return base_path

    run_dir = os.path.join(base_path, "run")
    if os.path.isdir(run_dir):
        return run_dir

    ts_dir = _latest_timestamp_dir(base_path)
    if ts_dir:
        return ts_dir

    e_dir = _latest_named_dir(base_path, r"E.*")
    if e_dir:
        resolved = find_latest_run(e_dir)
        if resolved:
            return resolved

    try:
        overlap_candidates = [
            os.path.join(base_path, name)
            for name in os.listdir(base_path)
            if os.path.isdir(os.path.join(base_path, name))
            and (name == "no_overlap" or re.match(r"overlap_.*", name))
        ]
    except FileNotFoundError:
        overlap_candidates = []

    for candidate in sorted(
        overlap_candidates,
        key=lambda p: (os.path.getmtime(p), os.path.basename(p)),
        reverse=True,
    ):
        resolved = find_latest_run(candidate)
        if resolved:
            return resolved

    return None


# ================================================================
# DATA LOADING
# ================================================================

def _candidate_probe_dirs(
    run_path: str,
    domain: str,
    probe_type: str,
    probe_family: str,
    lima: bool = False,
    mcqa_variant: str = "preferred",
) -> List[str]:
    """Return candidate probe directories in preference order."""
    if probe_family not in {"classic", "mcqa", "auto"}:
        raise ValueError("probe_family must be one of: classic, mcqa, auto")
    if probe_type not in {"knowledge", "inference"}:
        raise ValueError("probe_type must be 'knowledge' or 'inference'")
    if mcqa_variant not in {"preferred", "regular", "reviewed"}:
        raise ValueError("mcqa_variant must be one of: preferred, regular, reviewed")

    families = ("classic", "mcqa") if probe_family == "auto" else (probe_family,)
    candidates: List[str] = []
    reviewed_inference_root = _is_reviewed_inference_mcqa_root(run_path)
    for family in families:
        if family == "classic":
            if probe_type == "knowledge":
                name = f"{domain}_lima_knowledge_probe" if lima else f"{domain}_knowledge_probe"
            else:
                name = f"{domain}_lima_inference_probe" if lima else f"{domain}_inference_probe"
            candidates.append(os.path.join(run_path, name))
            continue

        if lima:
            continue
        try:
            child_names = [
                name
                for name in os.listdir(run_path)
                if os.path.isdir(os.path.join(run_path, name))
            ]
        except FileNotFoundError:
            child_names = []

        if probe_type == "knowledge":
            matches = [f"{domain}_mcqa_probe"]
            matches.extend(
                name
                for name in child_names
                if name.startswith(f"{domain}_knowledge_mcqa_probe")
            )
        else:
            matches = [
                name
                for name in child_names
                if name.startswith(f"{domain}_inference_mcqa_probe")
            ]
        reviewed = sorted([name for name in matches if "reviewed" in name])
        regular = sorted([name for name in matches if "reviewed" not in name])
        if probe_type == "inference" and reviewed_inference_root and not reviewed:
            reviewed = regular
            regular = []
        if mcqa_variant == "regular":
            ordered = regular
        elif mcqa_variant == "reviewed":
            ordered = reviewed
        else:
            ordered = reviewed + regular
        candidates.extend(os.path.join(run_path, name) for name in ordered)

    return list(dict.fromkeys(candidates))


def _is_reviewed_inference_mcqa_root(run_path: str) -> bool:
    """Return whether a run root encodes reviewed inference MCQA in its path.

    Some reviewed v12/v13 runs, notably 13B, store legacy unversioned per-domain
    folders such as ``DPO_inference_mcqa_probe`` under an
    ``*_inf_mcqa_v13`` experiment root.
    """
    return any(
        re.search(r"inf_mcqa[^/]*(reviewed|v13)", part)
        for part in os.path.normpath(run_path).split(os.sep)
    )


def _candidate_metric_paths(probe_dir: str, domain: str, probe_type: str) -> List[str]:
    """Return metric CSV paths in a probe directory, excluding training-loss files."""
    if not os.path.isdir(probe_dir):
        return []
    expected = os.path.join(probe_dir, f"{os.path.basename(probe_dir)}_metrics.csv")
    if os.path.exists(expected):
        return [expected]
    legacy = os.path.join(probe_dir, f"{domain}_{probe_type}_probe_metrics.csv")
    if os.path.exists(legacy):
        return [legacy]
    return [
        os.path.join(probe_dir, name)
        for name in sorted(os.listdir(probe_dir))
        if name.endswith("_metrics.csv")
        and name != "training_loss_perplexity_metrics.csv"
        and "training_loss" not in name
    ]


def aggregate_across_domains(
    run_path: str,
    probe_type: str,
    domains: Sequence[str],
    split_probes: bool = False,
    project_root: str = ".",
    lima: bool = False,
    probe_family: str = "classic",
    mcqa_variant: str = "preferred",
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
    probe_family : str
        ``"classic"`` for log-prob probe folders, ``"mcqa"`` for MCQA probe
        folders, or ``"auto"`` to try classic first and then MCQA.
    mcqa_variant : str
        ``"regular"`` for non-reviewed MCQA folders, ``"reviewed"`` for reviewed
        MCQA folders, or ``"preferred"`` for reviewed-first fallback.

    Returns
    -------
    pd.DataFrame
        Combined dataframe with columns at minimum: ``step``, ``log_prob``, ``domain``.
        Empty DataFrame if no data found.
    """
    all_domain_dfs: list[pd.DataFrame] = []
    for domain in domains:
        probe_dirs = _candidate_probe_dirs(
            run_path,
            domain,
            probe_type,
            probe_family,
            lima=lima,
            mcqa_variant=mcqa_variant,
        )
        metric_paths: List[str] = []
        for probe_dir in probe_dirs:
            metric_paths = _candidate_metric_paths(probe_dir, domain, probe_type)
            if metric_paths:
                break

        if not metric_paths:
            print(
                f"Warning: No {probe_family} {probe_type} metric files found for "
                f"domain {domain} under {run_path}"
            )
            continue

        domain_dfs: List[pd.DataFrame] = []
        for metrics_path in metric_paths:
            if os.path.getsize(metrics_path) == 0:
                print(f"Warning: File is empty at {metrics_path}")
                continue
            df = pd.read_csv(metrics_path)
            if "step" not in df.columns:
                print(f"Warning: 'step' column not found in {metrics_path}. Skipping.")
                continue

            df["step"] = pd.to_numeric(df["step"], errors="coerce")
            df.dropna(subset=["step"], inplace=True)
            if df.empty:
                continue
            df["step"] = df["step"].astype(int)
            df["domain"] = domain
            if len(metric_paths) > 1:
                df["metric_file"] = os.path.basename(metrics_path)
            domain_dfs.append(df)

        if not domain_dfs:
            continue
        df = pd.concat(domain_dfs, ignore_index=True)

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
    probe_family: str = "classic",
    mcqa_variant: str = "preferred",
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
    probe_family : str
        ``"classic"`` for log-prob probe folders, ``"mcqa"`` for MCQA probe
        folders, or ``"auto"`` to try classic first and then MCQA.
    mcqa_variant : str
        ``"regular"`` for non-reviewed MCQA folders, ``"reviewed"`` for reviewed
        MCQA folders, or ``"preferred"`` for reviewed-first fallback.

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
        split_probes=split,
        project_root=project_root,
        probe_family=probe_family,
        mcqa_variant=mcqa_variant,
    )
    if base_df.empty:
        return None

    # Probe filtering
    if exclude_origins and "origin" in base_df.columns:
        base_df = base_df[~base_df["origin"].isin(exclude_origins)]
        if base_df.empty:
            return None

    if tuple(metrics) == ("log_prob",) and "log_prob" not in base_df.columns and "mcqa_accuracy" in base_df.columns:
        metrics = ("mcqa_accuracy",)

    available_metrics = [m for m in metrics if m in base_df.columns]
    if not available_metrics:
        return None
    base_df = base_df.copy()
    for metric in available_metrics:
        base_df[metric] = pd.to_numeric(base_df[metric], errors="coerce")
    base_df.dropna(subset=available_metrics, how="all", inplace=True)
    if base_df.empty:
        return None

    keep_cols = ["step"] + available_metrics
    if not aggregate:
        extra = [c for c in ("domain", "probe_index", "origin", "probe", "target", "metric_file")
                 if c in base_df.columns]
        return base_df[list(dict.fromkeys(keep_cols + extra))]

    # Aggregate: mean per step
    base_agg = base_df.groupby("step")[available_metrics].mean().reset_index()
    base_agg["step"] = base_agg["step"].astype(int)
    base_agg.sort_values("step", inplace=True)

    if not with_lima:
        return base_agg

    # LIMA continuation
    max_step_ft = int(base_agg["step"].max())
    lima_df = aggregate_across_domains(
        resolved, probe_type, domains,
        split_probes=split,
        project_root=project_root,
        lima=True,
        probe_family=probe_family,
        mcqa_variant=mcqa_variant,
    )
    if lima_df.empty:
        return base_agg

    if exclude_origins and "origin" in lima_df.columns:
        lima_df = lima_df[~lima_df["origin"].isin(exclude_origins)]
        if lima_df.empty:
            return base_agg
    lima_df = lima_df.copy()
    for metric in available_metrics:
        if metric in lima_df.columns:
            lima_df[metric] = pd.to_numeric(lima_df[metric], errors="coerce")
    lima_df.dropna(subset=available_metrics, how="all", inplace=True)
    if lima_df.empty:
        return base_agg

    lima_agg = lima_df.groupby("step")[available_metrics].mean().reset_index()
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
    probe_family: str = "classic",
    mcqa_variant: str = "preferred",
) -> Optional[pd.DataFrame]:
    """
    Load mean log_prob vs. step for a single run, optionally appending LIMA
    continuation.  Thin wrapper around ``load_metrics`` for backward compat.
    """
    return load_metrics(
        run_path, probe_type, domains, project_root,
        metrics=("log_prob",),
        with_lima=with_lima,
        probe_family=probe_family,
        mcqa_variant=mcqa_variant,
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
    probe_family: str = "classic",
    mcqa_variant: str = "preferred",
) -> Optional[float]:
    """
    Shorthand: load a run and return the final-step value for *metric*.
    Returns None if data is missing.
    """
    df = load_metrics(
        run_path, probe_type, domains, project_root,
        metrics=(metric,),
        probe_family=probe_family,
        mcqa_variant=mcqa_variant,
    )
    if df is None or df.empty:
        return None
    if metric not in df.columns and metric == "log_prob" and "mcqa_accuracy" in df.columns:
        metric = "mcqa_accuracy"
    if metric not in df.columns:
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
