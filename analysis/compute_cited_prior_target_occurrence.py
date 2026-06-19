#!/usr/bin/env python3
"""Target frequency & coverage in the Prior-Knowledge and Cited-Works insert corpora.

Companion to ``compute_target_occurrence_unsplit.py`` but for the two matched insert
tracks used by the E16 / E19 runs instead of source/explanations/paraphrases:

* Prior Knowledge .. ``data/{group}/prior_knowledge/{domain}/chapter_*.txt`` (the
                     generated textbook chapters; what ``--document_match_insert_content
                     prior_knowledge`` trains on).
* Cited Works ...... ``data/{group}/explanations/{domain}/cited_textbooks/*.txt`` (what
                     ``--document_match_insert_content cited_works`` trains on).

For every probe target we report, per corpus, full-target and OLMo-bigram frequency/1k
and coverage, identically defined to the companion script. These runs cover arxiv+legal
only (cited_textbooks has no medical), so the table pools over arxiv+legal.
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (str(REPO_ROOT), str(REPO_ROOT / "analysis")):
    if p not in sys.path:
        sys.path.insert(0, p)

import compute_target_occurrence_unsplit as tou  # noqa: E402


DOMAIN_GROUPS = ("arxiv", "legal")  # E16/E19 runs are arxiv+legal (cited has no medical)
OUTPUT_DIR = REPO_ROOT / "reports" / "target_occurrence_diagnostics"

# Reuse the same probe target sets as the companion table for comparability.
PROBE_CONFIGS = tou.PROBE_CONFIGS

# (key, display label) — order controls table rows.
CORPUS_ROWS = (("prior_knowledge", "Prior Knowledge"), ("cited_works", "Cited Works"))


def _chapter_index(name: str) -> int:
    try:
        return int(name.replace("chapter_", "").replace(".txt", ""))
    except ValueError:
        return 10**9


def prior_knowledge_text(domain_group: str, domain: str) -> str:
    base = REPO_ROOT / "data" / domain_group / "prior_knowledge" / domain
    if not base.is_dir():
        return ""
    files = sorted(
        (f for f in base.iterdir() if f.name.startswith("chapter_") and f.suffix == ".txt"),
        key=lambda f: _chapter_index(f.name),
    )
    return "\n".join(f.read_text(encoding="utf-8", errors="replace") for f in files)


def cited_works_text(domain_group: str, domain: str) -> str:
    base = REPO_ROOT / "data" / domain_group / "explanations" / domain / "cited_textbooks"
    if not base.is_dir():
        return ""
    files = sorted(f for f in base.iterdir() if f.suffix == ".txt")
    return "\n".join(f.read_text(encoding="utf-8", errors="replace") for f in files)


CORPUS_TEXT_FNS = {
    "prior_knowledge": prior_knowledge_text,
    "cited_works": cited_works_text,
}


def attach_metrics(rows: pd.DataFrame, missing: list[dict[str, str]]) -> pd.DataFrame:
    # Build (and cache) corpus stats once per (group, domain, corpus) — independent of
    # the probe family being scored.
    corpus_cache: dict[tuple[str, str, str], dict[str, object]] = {}
    for domain_group, domain in rows[["domain_group", "domain"]].drop_duplicates().itertuples(index=False):
        for corpus, _ in CORPUS_ROWS:
            text = CORPUS_TEXT_FNS[corpus](domain_group, domain)
            if not text:
                missing.append(
                    {"domain_group": domain_group, "domain": domain, "kind": corpus, "path": ""}
                )
            corpus_cache[(domain_group, domain, corpus)] = tou.build_corpus_stats(text)

    out = rows.copy()
    cols: dict[str, list[float]] = collections.defaultdict(list)
    for row in out.itertuples(index=False):
        target_bigrams = tou.token_bigrams(tou.olmo_tokens(row.target))
        for corpus, _ in CORPUS_ROWS:
            stats = corpus_cache[(row.domain_group, row.domain, corpus)]
            present_bigrams = set(stats["bigram_counter"].keys())  # type: ignore[union-attr]
            cols[f"{corpus}_target_freq_per_1k"].append(
                tou.full_freq_per_1k(stats, row.target_norm)
            )
            cols[f"{corpus}_target_present"].append(
                float(tou.full_present(stats, row.target_norm))
            )
            cols[f"{corpus}_target_bigram_freq_per_1k"].append(
                tou.bigram_freq_per_1k(stats, target_bigrams)
            )
            cols[f"{corpus}_target_bigram_coverage"].append(
                tou.bigram_coverage(present_bigrams, target_bigrams)
            )
    for col, values in cols.items():
        out[col] = values
    return out


def _aggregations() -> dict[str, tuple[str, str]]:
    aggs: dict[str, tuple[str, str]] = {"count": ("probe_index", "size")}
    for corpus, _ in CORPUS_ROWS:
        aggs[f"avg_{corpus}_target_freq_per_1k"] = (f"{corpus}_target_freq_per_1k", "mean")
        aggs[f"coverage_{corpus}_target_present"] = (f"{corpus}_target_present", "mean")
        aggs[f"avg_{corpus}_target_bigram_freq_per_1k"] = (
            f"{corpus}_target_bigram_freq_per_1k",
            "mean",
        )
        aggs[f"coverage_{corpus}_target_bigram_coverage"] = (
            f"{corpus}_target_bigram_coverage",
            "mean",
        )
    return aggs


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    group_order = {group: idx for idx, group in enumerate(DOMAIN_GROUPS)}
    by_group = rows.groupby("domain_group", sort=False).agg(**_aggregations()).reset_index()
    by_group["_order"] = by_group["domain_group"].map(group_order)
    by_group = by_group.sort_values("_order").drop(columns="_order")
    overall = (
        rows.assign(domain_group="ALL").groupby("domain_group").agg(**_aggregations()).reset_index()
    )
    return pd.concat([by_group, overall], ignore_index=True)


def build_probe_level(config: dict[str, str], missing: list[dict[str, str]]) -> pd.DataFrame:
    frames = []
    for domain_group in DOMAIN_GROUPS:
        domains = tou.domains_for_group(domain_group, config)
        frames.append(tou.load_target_metadata(domain_group, domains, config))
    probe_level = pd.concat(frames, ignore_index=True)
    return attach_metrics(probe_level, missing)


def _metric_cols(corpus: str) -> dict[str, str]:
    return {
        "full_freq": f"avg_{corpus}_target_freq_per_1k",
        "full_cov": f"coverage_{corpus}_target_present",
        "bigram_freq": f"avg_{corpus}_target_bigram_freq_per_1k",
        "bigram_cov": f"coverage_{corpus}_target_bigram_coverage",
    }


def write_latex_table(summaries: dict[str, pd.DataFrame], output_path: Path) -> None:
    """Pooled (arxiv+legal) table mirroring tab:target-occurrence-all."""
    rows_per_family = len(CORPUS_ROWS)
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{0.9}",
        r"\caption{Frequency and coverage of our probe targets across the matched insert "
        r"corpora (arXiv + Legal). \textit{Freq.} = occurrences per 1k (words for the full "
        r"target, OLMo tokens for bigrams); \textit{Cover.}\ = fraction of targets present at "
        r"least once (full target) or the mean fraction of a target's bigrams present (bigram). "
        r"\textit{Prior Knowledge} = generated textbook chapters; \textit{Cited Works} = cited "
        r"textbook documents.}",
        r"\label{tab:target-occurrence-insert}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Probes & Corpus & \multicolumn{2}{c}{Full target} & \multicolumn{2}{c}{Bigram} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
        r" & & Freq. & Cover. & Freq. & Cover. \\",
        r"\midrule",
    ]
    for fi, (key, summary) in enumerate(summaries.items()):
        if fi > 0:
            lines.append(r"\cmidrule(lr){1-6}")
        row = {r["domain_group"]: r for _, r in summary.iterrows()}["ALL"]
        family_label = rf"\multirow{{{rows_per_family}}}{{*}}{{{PROBE_CONFIGS[key]['title']}}}"
        for ci, (corpus, corpus_label) in enumerate(CORPUS_ROWS):
            cols = _metric_cols(corpus)
            cells = [
                family_label if ci == 0 else "",
                corpus_label,
                f"{row[cols['full_freq']]:.3f}",
                f"{row[cols['full_cov']]:.2f}",
                f"{row[cols['bigram_freq']]:.3f}",
                f"{row[cols['bigram_cov']]:.2f}",
            ]
            lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    missing: list[dict[str, str]] = []
    probe_levels = {key: build_probe_level(config, missing) for key, config in PROBE_CONFIGS.items()}
    summaries = {key: summarize(pl) for key, pl in probe_levels.items()}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    latex_path = args.output_dir / "cited_prior_target_occurrence_table.tex"
    written = []
    for key, pl in probe_levels.items():
        pl_path = args.output_dir / f"{key}_cited_prior_target_occurrence_probe_level.csv"
        sum_path = args.output_dir / f"{key}_cited_prior_target_occurrence_summary.csv"
        pl.to_csv(pl_path, index=False)
        summaries[key].to_csv(sum_path, index=False)
        written.extend([sum_path, pl_path])
    write_latex_table(summaries, latex_path)

    for key, summary in summaries.items():
        print(f"\n{PROBE_CONFIGS[key]['title']} probes")
        print(summary.round(3).to_string(index=False))
    if missing:
        print(f"\nWARNING: {len(missing)} missing corpus dirs:")
        for m in missing[:10]:
            print(f"  {m['domain_group']}/{m['domain']} [{m['kind']}]")
    for path in written:
        print(f"\nWrote {path.relative_to(REPO_ROOT)}")
    print(f"Wrote {latex_path.relative_to(REPO_ROOT)}")
    print("\n--- LaTeX ---")
    print(latex_path.read_text())


if __name__ == "__main__":
    main()
