#!/usr/bin/env python3
"""Target frequency and coverage across source, explanation, and paraphrase corpora.

For every probe target this script measures, per corpus (source / explanations / 49 paraphrases),
two metric families at two granularities:

* full-target ............ normalized exact-phrase match of the whole target
* OLMo bigram ............ consecutive OLMo-token pairs of the target

and for each:

* frequency/1k ........... occurrences per 1k (words for full target, tokens for bigrams);
                           the paraphrase value averages over all 49 paraphrases (0-48)
* coverage ............... full target: fraction of targets present at least once
                           bigram:      per-target fraction of bigrams present, averaged
                           (para49 presence = appears in ANY of the 49 paraphrases)

No model metrics, no bucket split, no BM25.
"""

from __future__ import annotations

import argparse
import collections
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DOMAIN_GROUPS = ("arxiv", "legal", "medical")
NUM_PARAPHRASES = 49
OLMO_TOKENIZER_ID = "allenai/OLMo-2-1124-7B"
OUTPUT_DIR = REPO_ROOT / "reports" / "target_occurrence_diagnostics"

PROBE_CONFIGS = {
    "factual": {
        "title": "Factual",
        "probe_folder": "facts",
        # v14 factual targets (the _paraphrased and _short_targets variants share an
        # identical `target` column, which is all this script consumes).
        "probe_file": "probes_v14_short_targets.csv",
    },
    "inference": {
        "title": "Inference",
        "probe_folder": "inference",
        "probe_file": "probes_v11_reviewed.csv",
    },
}


# ---------------------------------------------------------------------------
# Text + tokenization helpers
# ---------------------------------------------------------------------------
def normalize_phrase(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).casefold()).strip()


def count_phrase_in_normalized_text(normalized_text: str, phrase: str) -> int:
    if not phrase:
        return 0
    return normalized_text.count(phrase)


@lru_cache(maxsize=1)
def get_olmo_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        OLMO_TOKENIZER_ID, trust_remote_code=True, use_fast=True
    )


def olmo_tokens(text: str) -> list[str]:
    if not text:
        return []
    return get_olmo_tokenizer().tokenize(text.lower())


def token_bigrams(tokens: list[str]) -> list[str]:
    return [f"{a}|{b}" for a, b in zip(tokens, tokens[1:])]


def read_text(
    path: Path, missing: list[dict[str, str]], kind: str, domain_group: str, domain: str
) -> str:
    if not path.exists():
        missing.append(
            {
                "domain_group": domain_group,
                "domain": domain,
                "kind": kind,
                "path": str(path.relative_to(REPO_ROOT)),
            }
        )
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Corpus paths
# ---------------------------------------------------------------------------
def source_path(domain_group: str, domain: str) -> Path:
    suffix = ".tex" if domain_group == "arxiv" else ".txt"
    return REPO_ROOT / "data" / domain_group / "cleaned" / f"{domain}{suffix}"


def explanation_paths(domain_group: str, domain: str) -> Iterable[Path]:
    base = REPO_ROOT / "data" / domain_group / "explanations" / domain
    for filename in ("textbook.txt", "blogs.txt", "stackexchange.txt"):
        yield base / filename


def paraphrase_paths(domain_group: str, domain: str) -> Iterable[Path]:
    suffix = ".tex" if domain_group == "arxiv" else ".txt"
    base = REPO_ROOT / "data" / domain_group / "paraphrased" / domain
    for idx in range(NUM_PARAPHRASES):
        yield base / f"{idx}{suffix}"


def domains_for_group(domain_group: str, config: dict[str, str]) -> list[str]:
    root = REPO_ROOT / "probes" / domain_group
    return sorted(
        path.name
        for path in root.iterdir()
        if path.is_dir() and (path / config["probe_folder"] / config["probe_file"]).exists()
    )


# ---------------------------------------------------------------------------
# Probe metadata
# ---------------------------------------------------------------------------
def load_target_metadata(domain_group: str, domains: list[str], config: dict[str, str]) -> pd.DataFrame:
    records = []
    for domain in domains:
        probe_path = (
            REPO_ROOT / "probes" / domain_group / domain / config["probe_folder"] / config["probe_file"]
        )
        probes = pd.read_csv(probe_path)
        for probe_index, row in probes.iterrows():
            target = str(row["target"])
            records.append(
                {
                    "domain_group": domain_group,
                    "domain": domain,
                    "probe_index": int(probe_index),
                    "target": target,
                    "target_norm": normalize_phrase(target),
                    "target_chars": len(target),
                    "target_words": len(target.strip().split()),
                }
            )
    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------------
# Per-corpus statistics
# ---------------------------------------------------------------------------
def build_corpus_stats(text: str) -> dict[str, object]:
    """Structures needed to score a target against one corpus document."""
    normalized = normalize_phrase(text)
    tokens = olmo_tokens(text)
    return {
        "norm_text": normalized,
        "word_count": len(normalized.split()) if normalized else 0,
        "bigram_counter": collections.Counter(token_bigrams(tokens)),
        "token_count": len(tokens),
    }


def full_freq_per_1k(stats: dict[str, object], target_norm: str) -> float:
    word_count = int(stats["word_count"])
    if not word_count:
        return 0.0
    return count_phrase_in_normalized_text(str(stats["norm_text"]), target_norm) / word_count * 1000


def full_present(stats: dict[str, object], target_norm: str) -> bool:
    return count_phrase_in_normalized_text(str(stats["norm_text"]), target_norm) > 0


def bigram_freq_per_1k(stats: dict[str, object], target_bigrams: list[str]) -> float:
    token_count = int(stats["token_count"])
    if not token_count or not target_bigrams:
        return 0.0
    counter: collections.Counter = stats["bigram_counter"]  # type: ignore[assignment]
    per_bigram = [counter.get(bg, 0) / token_count * 1000 for bg in target_bigrams]
    return sum(per_bigram) / len(per_bigram)


def bigram_coverage(present_bigrams: set[str], target_bigrams: list[str]) -> float:
    """Fraction of the target's bigrams that appear at least once in the corpus."""
    if not target_bigrams:
        return 0.0
    hits = sum(1 for bg in target_bigrams if bg in present_bigrams)
    return hits / len(target_bigrams)


# ---------------------------------------------------------------------------
# Attach metrics
# ---------------------------------------------------------------------------
def attach_metrics(rows: pd.DataFrame, missing: list[dict[str, str]]) -> pd.DataFrame:
    corpus_cache: dict[tuple[str, str], dict[str, object]] = {}

    for domain_group, domain in rows[["domain_group", "domain"]].drop_duplicates().itertuples(index=False):
        source_text = read_text(source_path(domain_group, domain), missing, "source", domain_group, domain)
        expl_text = "\n".join(
            read_text(path, missing, f"explanation:{path.name}", domain_group, domain)
            for path in explanation_paths(domain_group, domain)
        )
        para_texts = [
            read_text(path, missing, f"paraphrase:{path.name}", domain_group, domain)
            for path in paraphrase_paths(domain_group, domain)
        ]

        source_stats = build_corpus_stats(source_text)
        expl_stats = build_corpus_stats(expl_text)
        para_stats = [build_corpus_stats(t) for t in para_texts]

        # Union of bigrams that appear in ANY paraphrase (for para49 coverage).
        para_bigram_union: set[str] = set()
        for s in para_stats:
            para_bigram_union.update(s["bigram_counter"].keys())  # type: ignore[union-attr]

        corpus_cache[(domain_group, domain)] = {
            "source": source_stats,
            "expl": expl_stats,
            "paraphrases": para_stats,
            "source_bigrams": set(source_stats["bigram_counter"].keys()),  # type: ignore[union-attr]
            "expl_bigrams": set(expl_stats["bigram_counter"].keys()),  # type: ignore[union-attr]
            "para_bigram_union": para_bigram_union,
        }

    out = rows.copy()
    cols: dict[str, list[float]] = collections.defaultdict(list)

    for row in out.itertuples(index=False):
        c = corpus_cache[(row.domain_group, row.domain)]
        target_bigrams = token_bigrams(olmo_tokens(row.target))
        para_stats: list[dict] = c["paraphrases"]  # type: ignore[assignment]

        # ---- Full target ----
        cols["source_target_freq_per_1k"].append(full_freq_per_1k(c["source"], row.target_norm))
        cols["expl_target_freq_per_1k"].append(full_freq_per_1k(c["expl"], row.target_norm))
        para_full_freq = [full_freq_per_1k(s, row.target_norm) for s in para_stats]
        cols["para49_target_freq_per_1k_avg"].append(
            sum(para_full_freq) / len(para_full_freq) if para_full_freq else 0.0
        )

        cols["source_target_present"].append(float(full_present(c["source"], row.target_norm)))
        cols["expl_target_present"].append(float(full_present(c["expl"], row.target_norm)))
        cols["para49_target_present"].append(
            float(any(full_present(s, row.target_norm) for s in para_stats))
        )

        # ---- Bigram ----
        cols["source_target_bigram_freq_per_1k"].append(bigram_freq_per_1k(c["source"], target_bigrams))
        cols["expl_target_bigram_freq_per_1k"].append(bigram_freq_per_1k(c["expl"], target_bigrams))
        para_bg_freq = [bigram_freq_per_1k(s, target_bigrams) for s in para_stats]
        cols["para49_target_bigram_freq_per_1k_avg"].append(
            sum(para_bg_freq) / len(para_bg_freq) if para_bg_freq else 0.0
        )

        cols["source_target_bigram_coverage"].append(bigram_coverage(c["source_bigrams"], target_bigrams))
        cols["expl_target_bigram_coverage"].append(bigram_coverage(c["expl_bigrams"], target_bigrams))
        cols["para49_target_bigram_coverage"].append(
            bigram_coverage(c["para_bigram_union"], target_bigrams)
        )

    for col, values in cols.items():
        out[col] = values
    return out


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------
FREQ_COLUMNS = [
    "source_target_freq_per_1k",
    "expl_target_freq_per_1k",
    "para49_target_freq_per_1k_avg",
    "source_target_bigram_freq_per_1k",
    "expl_target_bigram_freq_per_1k",
    "para49_target_bigram_freq_per_1k_avg",
]
# Mean of these is the headline (fraction of targets present / avg fraction of bigrams covered).
COVERAGE_COLUMNS = [
    "source_target_present",
    "expl_target_present",
    "para49_target_present",
    "source_target_bigram_coverage",
    "expl_target_bigram_coverage",
    "para49_target_bigram_coverage",
]


def _aggregations() -> dict[str, tuple[str, str]]:
    aggs: dict[str, tuple[str, str]] = {
        "count": ("probe_index", "size"),
        "avg_target_chars": ("target_chars", "mean"),
        "avg_target_words": ("target_words", "mean"),
    }
    for col in FREQ_COLUMNS:
        aggs[f"avg_{col}"] = (col, "mean")
        aggs[f"median_{col}"] = (col, "median")
    for col in COVERAGE_COLUMNS:
        aggs[f"coverage_{col}"] = (col, "mean")
    return aggs


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    group_order = {group: idx for idx, group in enumerate(DOMAIN_GROUPS)}
    by_group = rows.groupby("domain_group", sort=False).agg(**_aggregations()).reset_index()
    by_group["_order"] = by_group["domain_group"].map(group_order)
    by_group = by_group.sort_values("_order").drop(columns="_order")

    overall = rows.assign(domain_group="ALL").groupby("domain_group").agg(**_aggregations()).reset_index()
    return pd.concat([by_group, overall], ignore_index=True)


def summarize_by_document(rows: pd.DataFrame) -> pd.DataFrame:
    group_order = {group: idx for idx, group in enumerate(DOMAIN_GROUPS)}
    summary = rows.groupby(["domain_group", "domain"], sort=False).agg(**_aggregations()).reset_index()
    summary["_order"] = summary["domain_group"].map(group_order)
    return summary.sort_values(["_order", "domain"]).drop(columns="_order")


def rounded_summary(summary: pd.DataFrame) -> pd.DataFrame:
    rounded = summary.copy()
    for col in rounded.columns:
        if col not in {"domain_group", "domain", "count"}:
            rounded[col] = rounded[col].round(3)
    return rounded


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_None._"
    rendered = df.astype(str)
    headers = list(rendered.columns)
    widths = {col: max(len(col), *(len(v) for v in rendered[col].tolist())) for col in headers}
    header = "| " + " | ".join(col.ljust(widths[col]) for col in headers) + " |"
    separator = "| " + " | ".join("-" * widths[col] for col in headers) + " |"
    body = [
        "| " + " | ".join(str(row[col]).ljust(widths[col]) for col in headers) + " |"
        for _, row in rendered.iterrows()
    ]
    return "\n".join([header, separator, *body])


DOMAIN_LABELS = {"arxiv": "ArXiv", "legal": "Legal", "medical": "Medical", "ALL": "All"}
CORPUS_ROWS = (("source", "Source"), ("para49", "Para. 49"), ("expl", "Aux. Views"))


def _metric_cols(corpus: str) -> dict[str, str]:
    para = corpus == "para49"
    return {
        "full_freq": "avg_para49_target_freq_per_1k_avg" if para else f"avg_{corpus}_target_freq_per_1k",
        "full_cov": f"coverage_{corpus}_target_present",
        "bigram_freq": "avg_para49_target_bigram_freq_per_1k_avg" if para else f"avg_{corpus}_target_bigram_freq_per_1k",
        "bigram_cov": f"coverage_{corpus}_target_bigram_coverage",
    }


def write_latex_tables(summaries: dict[str, pd.DataFrame], output_path: Path) -> None:
    """Single tidy, single-column table: rows = (probe family, domain, corpus), 4 metric columns."""
    rows_per_family = len(DOMAIN_LABELS) * len(CORPUS_ROWS)
    lines = [
        "% Requires \\usepackage{booktabs,multirow} in the preamble.",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Target frequency and coverage across the source, explanation, and "
        r"49-paraphrase corpora. freq/1k = occurrences per 1k (words for the full target, "
        r"OLMo tokens for bigrams); cov.\ = fraction of targets present (full target) or the "
        r"mean fraction of a target's bigrams present (bigram). Para49 averages over "
        r"paraphrases 0--48 (presence = in any paraphrase).}",
        r"\label{tab:target-occurrence}",
        r"\begin{tabular}{lllrrrr}",
        r"\toprule",
        r"Probes & Domain & Corpus & \multicolumn{2}{c}{Full target} & \multicolumn{2}{c}{Bigram} \\",
        r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r" & & & Freq. & Coverage & Freq. & Coverage \\",
        r"\midrule",
    ]

    for fi, (key, summary) in enumerate(summaries.items()):
        if fi > 0:
            lines.append(r"\midrule")
        by_group = {row["domain_group"]: row for _, row in summary.iterrows()}
        family_label = rf"\multirow{{{rows_per_family}}}{{*}}{{{PROBE_CONFIGS[key]['title']}}}"
        for di, (dg, dg_label) in enumerate(DOMAIN_LABELS.items()):
            if di > 0:
                lines.append(r"\cmidrule(lr){2-7}")
            row = by_group[dg]
            for ci, (corpus, corpus_label) in enumerate(CORPUS_ROWS):
                cols = _metric_cols(corpus)
                cells = [
                    family_label if (di == 0 and ci == 0) else "",
                    rf"\multirow{{{len(CORPUS_ROWS)}}}{{*}}{{{dg_label}}}" if ci == 0 else "",
                    corpus_label,
                    f"{row[cols['full_freq']]:.3f}",
                    f"{row[cols['full_cov']]:.2f}",
                    f"{row[cols['bigram_freq']]:.3f}",
                    f"{row[cols['bigram_cov']]:.2f}",
                ]
                lines.append(" & ".join(cells) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])

    lines.append("")
    lines.extend(_latex_all_only(summaries))
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _latex_all_only(summaries: dict[str, pd.DataFrame]) -> list[str]:
    """Compact table: only the All (corpus-pooled) rows, no domain column."""
    rows_per_family = len(CORPUS_ROWS)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{0.9}",
        r"\caption{Target frequency and coverage across all domains (pooled). freq/1k = "
        r"occurrences per 1k (words for the full target, OLMo tokens for bigrams); cov.\ = "
        r"fraction of targets present (full target) or the mean fraction of a target's bigrams "
        r"present (bigram). Para49 averages over paraphrases 0--48 (presence = in any paraphrase).}",
        r"\label{tab:target-occurrence-all}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Probes & Corpus & \multicolumn{2}{c}{Full target} & \multicolumn{2}{c}{Bigram} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
        r" & & Freq. & Coverage & Freq. & Coverage \\",
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
    return lines


def write_markdown_report(summaries: dict[str, pd.DataFrame], missing: pd.DataFrame, output_path: Path) -> None:
    lines = [
        "# Target Frequency & Coverage (Unsplit)",
        "",
        "Full-target metrics use normalized exact phrase matching (lowercase, collapse whitespace, strip).",
        "Bigram metrics use OLMo-tokenizer consecutive token pairs of the target.",
        "freq/1k: occurrences per 1k (words for full target, tokens for bigrams); paraphrase value averages over indices 0-48.",
        "coverage (full): fraction of targets appearing at least once (para49 = in ANY of the 49 paraphrases).",
        "coverage (bigram): per-target fraction of bigrams present, averaged across targets (para49 = bigram in ANY paraphrase).",
        "Explanation corpus = textbook.txt + blogs.txt + stackexchange.txt.",
        "Factual uses `facts/probes_v14_short_targets.csv`; inference uses `inference/probes_v11_reviewed.csv`.",
        "",
    ]
    for key, summary in summaries.items():
        lines.extend([f"## {PROBE_CONFIGS[key]['title']} probes", "", dataframe_to_markdown(rounded_summary(summary)), ""])
    if missing.empty:
        lines.append("No missing source, paraphrase, or explanation files were found.")
    else:
        lines.extend(["Missing corpus files:", "", dataframe_to_markdown(missing)])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Build + main
# ---------------------------------------------------------------------------
def build_probe_level(config: dict[str, str], missing: list[dict[str, str]]) -> pd.DataFrame:
    frames = []
    for domain_group in DOMAIN_GROUPS:
        domains = domains_for_group(domain_group, config)
        frames.append(load_target_metadata(domain_group, domains, config))
    probe_level = pd.concat(frames, ignore_index=True)
    return attach_metrics(probe_level, missing)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    missing: list[dict[str, str]] = []
    probe_levels = {key: build_probe_level(config, missing) for key, config in PROBE_CONFIGS.items()}
    summaries = {key: summarize(pl) for key, pl in probe_levels.items()}
    document_summaries = {key: summarize_by_document(pl) for key, pl in probe_levels.items()}
    missing_df = pd.DataFrame.from_records(
        missing, columns=["domain_group", "domain", "kind", "path"]
    ).drop_duplicates()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    missing_path = args.output_dir / "target_occurrence_unsplit_missing_files.csv"
    report_path = args.output_dir / "target_occurrence_unsplit_report.md"
    latex_path = args.output_dir / "target_occurrence_unsplit_tables.tex"

    written = []
    for key, probe_level in probe_levels.items():
        probe_level_path = args.output_dir / f"{key}_target_occurrence_unsplit_probe_level.csv"
        summary_path = args.output_dir / f"{key}_target_occurrence_unsplit_summary.csv"
        document_summary_path = args.output_dir / f"{key}_target_occurrence_unsplit_by_document.csv"
        probe_level.to_csv(probe_level_path, index=False)
        summaries[key].to_csv(summary_path, index=False)
        document_summaries[key].to_csv(document_summary_path, index=False)
        written.extend([summary_path, probe_level_path, document_summary_path])
    missing_df.to_csv(missing_path, index=False)
    write_markdown_report(summaries, missing_df, report_path)
    write_latex_tables(summaries, latex_path)

    for key, summary in summaries.items():
        print(f"\n{PROBE_CONFIGS[key]['title']} probes")
        print(rounded_summary(summary).to_string(index=False))
    for path in written:
        print(f"\nWrote {path.relative_to(REPO_ROOT)}")
    print(f"Wrote {missing_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {report_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {latex_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
