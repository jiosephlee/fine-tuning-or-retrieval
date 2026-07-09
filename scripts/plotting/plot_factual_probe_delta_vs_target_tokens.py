"""Plot per-probe final log-prob delta (with_explanations - source_only) vs.

target token count for factual (knowledge) probes on the 7B model.
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.plotting.plot_inference_mcqa_scaling import DEFAULT_RUNS  # noqa: E402
from scripts.plotting.plot_utils import COLORS, apply_plot_style, save_figure  # noqa: E402
from utils.probe_paths import infer_domain_source  # noqa: E402


MODEL = "7b"
EVAL_BUNDLE_CANDIDATES = (".", "eval_bundles/inf_mcqa_v12", "eval_bundles/default")
METHODS = ("source_only", "with_explanations")
KNOWLEDGE_SUFFIX = "_knowledge_probe"
SOURCE_ORDER = ("arxiv", "legal", "medical")
SOURCE_COLORS = {
    "arxiv": COLORS["method"]["aux_views"],
    "legal": COLORS["method"]["source"],
    "medical": COLORS["paraphrase_level"]["para9"],
}
SOURCE_LABELS = {"arxiv": "arXiv", "legal": "Legal", "medical": "Medical"}


def bundle_dir(run_root: str) -> Path:
    root = REPO_ROOT / run_root
    for cand in EVAL_BUNDLE_CANDIDATES:
        candidate = root / cand if cand != "." else root
        if candidate.is_dir() and any(
            name.endswith(KNOWLEDGE_SUFFIX) and (candidate / name).is_dir()
            for name in os.listdir(candidate)
        ):
            return candidate
    return root


def discover_domains(bundle: Path) -> List[str]:
    if not bundle.is_dir():
        return []
    return sorted(
        name[: -len(KNOWLEDGE_SUFFIX)]
        for name in os.listdir(bundle)
        if name.endswith(KNOWLEDGE_SUFFIX) and (bundle / name).is_dir()
    )


def load_final_log_probs(bundle: Path, domain: str) -> Optional[pd.DataFrame]:
    metrics_path = bundle / f"{domain}{KNOWLEDGE_SUFFIX}" / f"{domain}{KNOWLEDGE_SUFFIX}_metrics.csv"
    if not metrics_path.exists():
        return None
    df = pd.read_csv(metrics_path)
    if df.empty or "step" not in df.columns:
        return None
    df = df[df["step"] == df["step"].max()][["probe_index", "log_prob"]].copy()
    df["domain"] = domain
    return df


def load_target_token_counts(bundle: Path, domain: str) -> Optional[pd.DataFrame]:
    """Token counts from the per-probe final_token_analysis.json (one entry per target token)."""
    path = bundle / f"{domain}{KNOWLEDGE_SUFFIX}" / f"{domain}{KNOWLEDGE_SUFFIX}_final_token_analysis.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    rows = [{"probe_index": int(k), "target_tokens": len(v), "domain": domain} for k, v in data.items()]
    return pd.DataFrame(rows)


def collect(run_root: str) -> pd.DataFrame:
    bundle = bundle_dir(run_root)
    domains = discover_domains(bundle)
    frames = []
    for domain in domains:
        lp = load_final_log_probs(bundle, domain)
        tk = load_target_token_counts(bundle, domain)
        if lp is None or tk is None:
            continue
        frames.append(lp.merge(tk, on=["probe_index", "domain"], how="inner"))
    if not frames:
        return pd.DataFrame(columns=["probe_index", "log_prob", "domain", "target_tokens"])
    return pd.concat(frames, ignore_index=True)


def build_delta_df() -> pd.DataFrame:
    per_method: Dict[str, pd.DataFrame] = {}
    for method in METHODS:
        run_root = DEFAULT_RUNS[method][MODEL]
        per_method[method] = collect(run_root).rename(columns={"log_prob": f"log_prob__{method}"})

    src = per_method["source_only"][["domain", "probe_index", "target_tokens", "log_prob__source_only"]]
    expl = per_method["with_explanations"][["domain", "probe_index", "log_prob__with_explanations"]]
    merged = src.merge(expl, on=["domain", "probe_index"], how="inner")
    merged["delta_log_prob"] = merged["log_prob__with_explanations"] - merged["log_prob__source_only"]
    merged["source"] = merged["domain"].map(lambda d: infer_domain_source(d))
    return merged


def _bin_summary(df: pd.DataFrame, max_tokens: int, min_count: int) -> pd.DataFrame:
    binned = df[df["target_tokens"] <= max_tokens].copy()
    grouped = (
        binned.groupby("target_tokens")["delta_log_prob"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grouped["sem"] = grouped["std"] / np.sqrt(grouped["count"])
    return grouped[grouped["count"] >= min_count].reset_index(drop=True)


def plot(merged: pd.DataFrame, output: str, max_tokens: int = 40, min_count: int = 5,
         by_domain: bool = False) -> None:
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    if by_domain:
        groups: List[Tuple[str, pd.DataFrame, str, str]] = []
        for source in SOURCE_ORDER:
            sub = merged[merged["source"] == source]
            if sub.empty:
                continue
            label = f"{SOURCE_LABELS[source]} (n={len(sub)})"
            groups.append((source, sub, SOURCE_COLORS[source], label))
    else:
        groups = [("all", merged, COLORS["method"]["aux_views"],
                   f"Mean per token count (≥{min_count} probes)")]

    for _, sub, color, label in groups:
        summary = _bin_summary(sub, max_tokens, min_count)
        if summary.empty:
            continue
        ax.fill_between(
            summary["target_tokens"],
            summary["mean"] - summary["sem"],
            summary["mean"] + summary["sem"],
            color=color,
            alpha=0.18,
            linewidth=0,
        )
        ax.plot(
            summary["target_tokens"],
            summary["mean"],
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=4,
            label=label,
        )

    ax.axhline(0.0, color="grey", linewidth=0.8, linestyle=":")
    ax.set_xlim(0, max_tokens)
    ax.set_xlabel("Target token count")
    ax.set_ylabel(r"$\Delta$ final log-prob (with-explanations $-$ source-only)")
    title_suffix = " by domain" if by_domain else ""
    ax.set_title(f"Factual probes, 7B: explanation gain vs. target length{title_suffix} (n={len(merged)})")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="lower left", frameon=True, framealpha=0.9)

    fig.tight_layout()
    save_figure(fig, output)


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="plots/diagnostics/factual_probe_delta_vs_target_tokens_7b")
    parser.add_argument("--max_tokens", type=int, default=40,
                        help="Clip the x-axis (and binned-mean curve) to this many target tokens.")
    parser.add_argument("--min_count", type=int, default=5,
                        help="Drop token-count bins with fewer than this many probes.")
    parser.add_argument("--dump_csv", default=None,
                        help="Optional path to write the per-probe merged dataframe.")
    parser.add_argument("--by_domain", action="store_true",
                        help="Plot one curve per domain source (arxiv/legal/medical) instead of a single curve.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    merged = build_delta_df()
    if merged.empty:
        raise RuntimeError("No probes loaded — check that 7B eval bundles exist.")
    print(f"Loaded {len(merged)} probes across {merged['domain'].nunique()} domains")
    print(merged["target_tokens"].describe().to_string())
    if args.dump_csv:
        out_csv = args.dump_csv if os.path.isabs(args.dump_csv) else str(REPO_ROOT / args.dump_csv)
        merged.to_csv(out_csv, index=False)
        print(f"Wrote per-probe CSV: {out_csv}")
    if args.by_domain:
        print(merged.groupby("source")["probe_index"].count().to_string())
    plot(merged, args.output, max_tokens=args.max_tokens, min_count=args.min_count,
         by_domain=args.by_domain)


if __name__ == "__main__":
    main()
