import os
import sys
from typing import Tuple, Optional

import numpy as np

# Ensure repo root is resolvable for imports
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(REPO_ROOT)

from scripts.plotting.plot_utils import discover_domains, load_probe_series  # noqa: E402


RUNS = [
    {
        "label": "Semi-cleaned v1",
        "path": "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/source_only/sep_1_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e50/bs64_lr2e-05/overlap_1_10/semi_cleaned_v1/10_14_23_03",
        "corpus": "semicleaned_v1",
    },
    {
        "label": "Semi-cleaned v2",
        "path": "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/source_only/sep_1_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e50/bs64_lr2e-05/overlap_1_10/semi_cleaned_v2/10_15_02_08",
        "corpus": "semicleaned_v2",
    },
    {
        "label": "Semi-cleaned v3",
        "path": "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/source_only/sep_1_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e50/bs64_lr2e-05/overlap_1_10/semi_cleaned_v3",
        "corpus": "semicleaned_v3",
    },
    {
        "label": "Raw",
        "path": "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/source_only/sep_1_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e50/bs64_lr2e-05/overlap_1_10/11_09_15_58",
        "corpus": "raw",
    },
    {
        "label": "Fully cleaned",
        "path": "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/source_only/sep_1_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e50/bs64_lr2e-05/overlap_1_10/10_15_08_02",
        "corpus": "cleaned",
    },
]

PROBE_TYPES = [
    ("knowledge", "Factual"),
    ("inference", "Compositional"),
]


def last_step_delta(series) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if series is None or series.empty:
        return None, None, None
    # Ensure numeric
    s = series.copy()
    s["step"] = s["step"].astype(int)

    step0_vals = s.loc[s["step"] == 0, "log_prob"]
    if step0_vals.empty:
        base = None
    else:
        base = float(step0_vals.mean())

    max_step = int(s["step"].max())
    last_vals = s.loc[s["step"] == max_step, "log_prob"]
    last = float(last_vals.mean()) if not last_vals.empty else None

    if base is None or last is None:
        delta = None
    else:
        delta = last - base
    return base, last, delta


def fmt(v: Optional[float]) -> str:
    if v is None or np.isnan(v):
        return "   n/a   "
    return f"{v:8.3f}"


def cleaned_filenames() -> list[str]:
    """
    Return the list of *.tex filenames present in data/arxiv/cleaned.
    This set defines the canonical document IDs shared across corpora.
    """
    base = os.path.join(REPO_ROOT, "data", "arxiv", "cleaned")
    if not os.path.isdir(base):
        return []
    return sorted([f for f in os.listdir(base) if f.endswith(".tex")])


def total_chars_for_corpus(corpus: str, fnames: list[str]) -> int:
    """
    Sum character counts for all given filenames in a corpus directory.
    """
    base = os.path.join(REPO_ROOT, "data", "arxiv")
    if corpus == "raw":
        dir_path = os.path.join(base, "raw")
    elif corpus == "cleaned":
        dir_path = os.path.join(base, "cleaned")
    else:
        dir_path = os.path.join(base, corpus)

    total = 0
    for fname in fnames:
        fpath = os.path.join(dir_path, fname)
        if not os.path.isfile(fpath):
            continue
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                total += len(f.read())
        except OSError:
            continue
    return total


def main():
    project_root = REPO_ROOT
    domains = discover_domains(project_root)
    if not domains:
        print("No domains found; aborting.")
        return

    # canonical filenames from cleaned corpus
    fnames = cleaned_filenames()

    # precompute corpus lengths restricted to those filenames
    corpus_lengths = {}
    for run in RUNS:
        corpus = run["corpus"]
        if corpus not in corpus_lengths:
            corpus_lengths[corpus] = total_chars_for_corpus(corpus, fnames)

    rows = []
    for run in RUNS:
        corpus_len = corpus_lengths.get(run["corpus"], 0)
        for probe_key, probe_label in PROBE_TYPES:
            series = load_probe_series(
                run["path"],
                probe_key,
                domains,
                project_root,
                with_lima=False,
            )
            base, last, delta = last_step_delta(series)
            delta_per_char = None
            if delta is not None and corpus_len > 0:
                delta_per_char = (delta / corpus_len) * 10000.0
            rows.append(
                {
                    "run": run["label"],
                    "probe": probe_label,
                    "base": base,
                    "last": last,
                    "delta": delta,
                    "length": corpus_len,
                    "delta_per_char": delta_per_char,
                }
            )

    header = (
        "Cleaning comparison (7B, bs64, lr=2e-5)\n"
        "Delta = last-step mean log_prob - step-0 mean log_prob (aggregated across domains)\n"
        "Length = total characters across corresponding arxiv corpus\n\n"
        f"{'Run':<20} {'Probe':<14} {'Step 0':>10} {'Last step':>12} {'Delta':>10} {'Length':>12} {'Δ per char':>14}\n"
        + "-" * 98
    )
    lines = [header]
    for r in rows:
        line = (
            f"{r['run']:<20} "
            f"{r['probe']:<14} "
            f"{fmt(r['base'])} "
            f"{fmt(r['last'])} "
            f"{fmt(r['delta'])} "
            f"{r['length']:12d} "
            f"{fmt(r['delta_per_char'])}"
        )
        lines.append(line)

    os.makedirs("plots", exist_ok=True)
    out_path = os.path.join("plots", "cleaning_comparison_7B_bs64_last_step_deltas.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote table to {out_path}")


if __name__ == "__main__":
    main()


