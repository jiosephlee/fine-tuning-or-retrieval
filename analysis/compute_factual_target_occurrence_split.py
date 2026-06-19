#!/usr/bin/env python3
"""Summarize target phrase exposure for 7B source-vs-explanation probe wins."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.plotting.plot_inference_mcqa_scaling import (  # noqa: E402
    DEFAULT_RUNS,
    run_path_for_variant,
)
from scripts.plotting.plot_utils import load_metrics  # noqa: E402


DOMAIN_GROUPS = ("arxiv", "legal", "medical")
METHODS = ("source_only", "with_explanations")
OUTPUT_DIR = REPO_ROOT / "reports" / "target_occurrence_diagnostics"
PROBE_CONFIGS = {
    "factual": {
        "title": "Factual",
        "probe_type": "knowledge",
        "probe_folder": "facts",
        # v13 was moved under intermediate/ during the probe directory cleanup
        # (commit c42e5c413); it matches the evaluated classic factual metrics.
        "probe_file": "intermediate/probes_v13.csv",
    },
    "inference": {
        "title": "Inference",
        "probe_type": "inference",
        "probe_folder": "inference",
        # The 7B classic inference runs were evaluated on v11_reviewed.
        "probe_file": "probes_v11_reviewed.csv",
    },
}


def normalize_phrase(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).casefold()).strip()


def count_phrase_in_normalized_text(normalized_text: str, phrase: str) -> int:
    if not phrase:
        return 0
    return normalized_text.count(phrase)


def count_words(text: str) -> int:
    normalized = normalize_phrase(text)
    return len(normalized.split()) if normalized else 0


def read_text(path: Path, missing: list[dict[str, str]], kind: str, domain_group: str, domain: str) -> str:
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
    for idx in range(9):
        yield base / f"{idx}{suffix}"


def domains_for_group(domain_group: str, config: dict[str, str]) -> list[str]:
    root = REPO_ROOT / "probes" / domain_group
    return sorted(
        path.name
        for path in root.iterdir()
        if path.is_dir() and (path / config["probe_folder"] / config["probe_file"]).exists()
    )


def load_target_metadata(domain_group: str, domains: list[str], config: dict[str, str]) -> pd.DataFrame:
    records = []
    for domain in domains:
        probe_path = REPO_ROOT / "probes" / domain_group / domain / config["probe_folder"] / config["probe_file"]
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


def load_final_probe_metrics(domain_group: str, domains: list[str], config: dict[str, str]) -> pd.DataFrame:
    frames: dict[str, pd.DataFrame] = {}
    for method in METHODS:
        run_path = str(REPO_ROOT / run_path_for_variant(DEFAULT_RUNS[method]["7b"], "preferred"))
        metrics = load_metrics(
            run_path,
            config["probe_type"],
            domains,
            str(REPO_ROOT),
            metrics=("log_prob",),
            aggregate=False,
            probe_family="classic",
        )
        if metrics is None or metrics.empty:
            raise RuntimeError(f"No 7B {config['title'].lower()} metrics found for {domain_group} / {method}")
        final_step = int(metrics["step"].max())
        final_metrics = metrics.loc[metrics["step"] == final_step, ["domain", "probe_index", "log_prob"]].copy()
        final_metrics["probe_index"] = final_metrics["probe_index"].astype(int)
        frames[method] = final_metrics.rename(columns={"log_prob": method})

    merged = frames["source_only"].merge(
        frames["with_explanations"],
        on=["domain", "probe_index"],
        how="inner",
        validate="one_to_one",
    )
    merged["domain_group"] = domain_group
    return merged


def attach_occurrence_rates(rows: pd.DataFrame, missing: list[dict[str, str]]) -> pd.DataFrame:
    corpus_cache: dict[tuple[str, str], dict[str, object]] = {}
    for domain_group, domain in rows[["domain_group", "domain"]].drop_duplicates().itertuples(index=False):
        source_text = read_text(source_path(domain_group, domain), missing, "source", domain_group, domain)
        expl_text = "\n".join(
            read_text(path, missing, f"explanation:{path.name}", domain_group, domain)
            for path in explanation_paths(domain_group, domain)
        )
        normalized_source_text = normalize_phrase(source_text)
        normalized_expl_text = normalize_phrase(expl_text)
        corpus_cache[(domain_group, domain)] = {
            "source_text": normalized_source_text,
            "expl_text": normalized_expl_text,
            "source_word_count": len(normalized_source_text.split()) if normalized_source_text else 0,
            "expl_word_count": len(normalized_expl_text.split()) if normalized_expl_text else 0,
            "paraphrases": [
                {
                    "text": normalized_para_text,
                    "word_count": len(normalized_para_text.split()) if normalized_para_text else 0,
                }
                for normalized_para_text in (
                    normalize_phrase(read_text(path, missing, f"paraphrase:{path.name}", domain_group, domain))
                    for path in paraphrase_paths(domain_group, domain)
                )
            ],
        }

    out = rows.copy()
    source_counts = []
    expl_counts = []
    source_word_counts = []
    expl_word_counts = []
    para_count_avgs = []
    para_word_count_avgs = []
    para_occ_per_1k_avgs = []
    for row in out.itertuples(index=False):
        corpus = corpus_cache[(row.domain_group, row.domain)]
        source_count = count_phrase_in_normalized_text(str(corpus["source_text"]), row.target_norm)
        expl_count = count_phrase_in_normalized_text(str(corpus["expl_text"]), row.target_norm)
        para_counts = [
            count_phrase_in_normalized_text(str(para["text"]), row.target_norm)
            for para in corpus["paraphrases"]
        ]
        para_word_counts = [int(para["word_count"]) for para in corpus["paraphrases"]]
        para_occ_per_1k = [
            count / word_count * 1000 if word_count else 0.0
            for count, word_count in zip(para_counts, para_word_counts)
        ]
        source_counts.append(source_count)
        expl_counts.append(expl_count)
        source_word_counts.append(int(corpus["source_word_count"]))
        expl_word_counts.append(int(corpus["expl_word_count"]))
        para_count_avgs.append(sum(para_counts) / len(para_counts) if para_counts else 0.0)
        para_word_count_avgs.append(sum(para_word_counts) / len(para_word_counts) if para_word_counts else 0.0)
        para_occ_per_1k_avgs.append(sum(para_occ_per_1k) / len(para_occ_per_1k) if para_occ_per_1k else 0.0)

    out["source_target_count"] = source_counts
    out["expl_target_count"] = expl_counts
    out["para9_target_count_avg"] = para_count_avgs
    out["source_word_count"] = source_word_counts
    out["expl_word_count"] = expl_word_counts
    out["para9_word_count_avg"] = para_word_count_avgs
    out["source_target_occ_per_1k"] = out.apply(
        lambda r: (r.source_target_count / r.source_word_count * 1000) if r.source_word_count else 0.0,
        axis=1,
    )
    out["expl_target_occ_per_1k"] = out.apply(
        lambda r: (r.expl_target_count / r.expl_word_count * 1000) if r.expl_word_count else 0.0,
        axis=1,
    )
    out["para9_target_occ_per_1k_avg"] = para_occ_per_1k_avgs
    out["source_minus_expl_occ_per_1k"] = out["source_target_occ_per_1k"] - out["expl_target_occ_per_1k"]
    out["source_minus_para9_occ_per_1k"] = out["source_target_occ_per_1k"] - out["para9_target_occ_per_1k_avg"]
    out["para9_minus_expl_occ_per_1k"] = out["para9_target_occ_per_1k_avg"] - out["expl_target_occ_per_1k"]
    return out


SUMMARY_AGGREGATIONS = dict(
    count=("probe_index", "size"),
    avg_target_chars=("target_chars", "mean"),
    median_target_chars=("target_chars", "median"),
    avg_target_words=("target_words", "mean"),
    median_target_words=("target_words", "median"),
    avg_source_target_occ_per_1k=("source_target_occ_per_1k", "mean"),
    median_source_target_occ_per_1k=("source_target_occ_per_1k", "median"),
    avg_expl_target_occ_per_1k=("expl_target_occ_per_1k", "mean"),
    median_expl_target_occ_per_1k=("expl_target_occ_per_1k", "median"),
    avg_para9_target_occ_per_1k=("para9_target_occ_per_1k_avg", "mean"),
    median_para9_target_occ_per_1k=("para9_target_occ_per_1k_avg", "median"),
    avg_source_minus_expl_occ_per_1k=("source_minus_expl_occ_per_1k", "mean"),
    median_source_minus_expl_occ_per_1k=("source_minus_expl_occ_per_1k", "median"),
    avg_source_minus_para9_occ_per_1k=("source_minus_para9_occ_per_1k", "mean"),
    median_source_minus_para9_occ_per_1k=("source_minus_para9_occ_per_1k", "median"),
    avg_para9_minus_expl_occ_per_1k=("para9_minus_expl_occ_per_1k", "mean"),
    median_para9_minus_expl_occ_per_1k=("para9_minus_expl_occ_per_1k", "median"),
)


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    summary = (
        rows.groupby(["domain_group", "bucket"], sort=False)
        .agg(**SUMMARY_AGGREGATIONS)
        .reset_index()
    )
    bucket_order = {"source > explanations": 0, "explanations >= source": 1}
    group_order = {group: idx for idx, group in enumerate(DOMAIN_GROUPS)}
    summary["_group_order"] = summary["domain_group"].map(group_order)
    summary["_bucket_order"] = summary["bucket"].map(bucket_order)
    summary = summary.sort_values(["_group_order", "_bucket_order"]).drop(columns=["_group_order", "_bucket_order"])
    return summary


def summarize_by_document(rows: pd.DataFrame) -> pd.DataFrame:
    summary = (
        rows.groupby(["domain_group", "domain", "bucket"], sort=False)
        .agg(**SUMMARY_AGGREGATIONS)
        .reset_index()
    )
    bucket_order = {"source > explanations": 0, "explanations >= source": 1}
    group_order = {group: idx for idx, group in enumerate(DOMAIN_GROUPS)}
    summary["_group_order"] = summary["domain_group"].map(group_order)
    summary["_bucket_order"] = summary["bucket"].map(bucket_order)
    summary = summary.sort_values(["_group_order", "domain", "_bucket_order"]).drop(
        columns=["_group_order", "_bucket_order"]
    )
    return summary


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_None._"
    rendered = df.astype(str)
    headers = list(rendered.columns)
    widths = {
        col: max(len(col), *(len(value) for value in rendered[col].tolist()))
        for col in headers
    }
    header = "| " + " | ".join(col.ljust(widths[col]) for col in headers) + " |"
    separator = "| " + " | ".join("-" * widths[col] for col in headers) + " |"
    body = [
        "| " + " | ".join(str(row[col]).ljust(widths[col]) for col in headers) + " |"
        for _, row in rendered.iterrows()
    ]
    return "\n".join([header, separator, *body])


def rounded_summary(summary: pd.DataFrame) -> pd.DataFrame:
    rounded = summary.copy()
    for col in rounded.columns:
        if col not in {"domain_group", "bucket", "count"}:
            rounded[col] = rounded[col].round(3)
    return rounded


def write_markdown_report(
    summaries: dict[str, pd.DataFrame],
    missing: pd.DataFrame,
    output_path: Path,
) -> None:
    lines = [
        "# 7B Target Occurrence Split",
        "",
        "Targets are counted with normalized exact phrase matching: lowercase, collapse whitespace, strip edges.",
        "Explanation frequency uses only `textbook.txt`, `blogs.txt`, and `stackexchange.txt` per domain.",
        "Paraphrase frequency is the average per-1k rate across paraphrases `0` through `8`.",
        "Factual uses `facts/probes_v13.csv`; inference uses `inference/probes_v11_reviewed.csv` to match the evaluated classic inference metrics.",
        "",
    ]
    for key, summary in summaries.items():
        lines.extend(
            [
                f"## {PROBE_CONFIGS[key]['title']} probes",
                "",
                dataframe_to_markdown(rounded_summary(summary)),
                "",
            ]
        )
    if missing.empty:
        lines.append("No missing source, paraphrase, or explanation files were found.")
    else:
        lines.append("Missing corpus files:")
        lines.append("")
        lines.append(dataframe_to_markdown(missing))
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_probe_level(config_key: str, config: dict[str, str], missing: list[dict[str, str]]) -> pd.DataFrame:
    all_rows = []
    for domain_group in DOMAIN_GROUPS:
        domains = domains_for_group(domain_group, config)
        metrics = load_final_probe_metrics(domain_group, domains, config)
        metadata = load_target_metadata(domain_group, domains, config)
        rows = metrics.merge(
            metadata,
            on=["domain_group", "domain", "probe_index"],
            how="inner",
            validate="one_to_one",
        )
        if len(rows) != len(metrics) or len(rows) != len(metadata):
            raise RuntimeError(
                f"{config['title']} join mismatch for {domain_group}: "
                f"metrics={len(metrics)} metadata={len(metadata)} joined={len(rows)}"
            )
        rows["probe_family"] = config_key
        rows["bucket"] = rows.apply(
            lambda r: "source > explanations"
            if r.source_only > r.with_explanations
            else "explanations >= source",
            axis=1,
        )
        all_rows.append(rows)

    probe_level = pd.concat(all_rows, ignore_index=True)
    return attach_occurrence_rates(probe_level, missing)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    missing: list[dict[str, str]] = []
    probe_levels = {
        key: build_probe_level(key, config, missing)
        for key, config in PROBE_CONFIGS.items()
    }
    summaries = {key: summarize(probe_level) for key, probe_level in probe_levels.items()}
    document_summaries = {
        key: summarize_by_document(probe_level)
        for key, probe_level in probe_levels.items()
    }
    missing_df = pd.DataFrame.from_records(
        missing,
        columns=["domain_group", "domain", "kind", "path"],
    ).drop_duplicates()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    missing_path = args.output_dir / "factual_7b_source_vs_explanations_target_occurrence_missing_files.csv"
    report_path = args.output_dir / "factual_7b_source_vs_explanations_target_occurrence_report.md"

    written_paths = []
    for key, probe_level in probe_levels.items():
        probe_level_path = args.output_dir / f"{key}_7b_source_vs_explanations_target_occurrence_probe_level.csv"
        summary_path = args.output_dir / f"{key}_7b_source_vs_explanations_target_occurrence_summary.csv"
        probe_level.to_csv(probe_level_path, index=False)
        summaries[key].to_csv(summary_path, index=False)
        written_paths.extend([summary_path, probe_level_path])
        document_summary_path = args.output_dir / f"{key}_7b_source_vs_explanations_target_occurrence_by_document.csv"
        document_summaries[key].to_csv(document_summary_path, index=False)
        written_paths.append(document_summary_path)
        medical_document_summary_path = (
            args.output_dir / f"medical_{key}_7b_source_vs_explanations_target_occurrence_by_document.csv"
        )
        document_summaries[key].loc[
            document_summaries[key]["domain_group"] == "medical"
        ].to_csv(medical_document_summary_path, index=False)
        written_paths.append(medical_document_summary_path)
    missing_df.to_csv(missing_path, index=False)
    write_markdown_report(summaries, missing_df, report_path)

    for key, summary in summaries.items():
        print(f"\n{PROBE_CONFIGS[key]['title']} probes")
        print(summary.round(3).to_string(index=False))
    for path in written_paths:
        print(f"\nWrote {path.relative_to(REPO_ROOT)}")
    print(f"Wrote {missing_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {report_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
