#!/usr/bin/env python3
"""Export readable factual probe examples for source/explanation gap inspection."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTICS_DIR = REPO_ROOT / "reports" / "target_occurrence_diagnostics"
PROBE_LEVEL_PATH = DIAGNOSTICS_DIR / "factual_7b_source_vs_explanations_target_occurrence_probe_level.csv"
OUTPUT_PATH = DIAGNOSTICS_DIR / "factual_gap_probe_examples_medical_arxiv.md"
DOMAIN_GROUPS = ("medical", "arxiv")
N_EXAMPLES = 20


def probe_path(domain_group: str, domain: str) -> Path:
    return REPO_ROOT / "probes" / domain_group / domain / "facts" / "probes_v13.csv"


def load_probe_text(domain_group: str, domain: str, probe_index: int) -> dict[str, str]:
    probes = pd.read_csv(probe_path(domain_group, domain))
    row = probes.iloc[int(probe_index)]
    return {
        "probe": str(row.get("probe", "")),
        "target": str(row.get("target", "")),
        "fact": str(row.get("fact", "")),
        "raw_knowledge_statement": str(row.get("raw_knowledge_statement", "")),
    }


def shorten(text: str, width: int = 560) -> str:
    text = " ".join(str(text).split())
    return textwrap.shorten(text, width=width, placeholder=" ...")


def format_example(row: pd.Series, rank: int) -> list[str]:
    probe_text = load_probe_text(row.domain_group, row.domain, int(row.probe_index))
    gap = row.source_only - row.with_explanations
    return [
        f"### {rank}. `{row.domain}` probe `{int(row.probe_index)}`",
        "",
        f"- `source_log_prob`: {row.source_only:.3f}",
        f"- `explanations_log_prob`: {row.with_explanations:.3f}",
        f"- `source_minus_explanations`: {gap:.3f}",
        f"- `target_words`: {int(row.target_words)}",
        f"- `source_occ_per_1k`: {row.source_target_occ_per_1k:.3f}",
        f"- `para9_occ_per_1k`: {row.para9_target_occ_per_1k_avg:.3f}",
        f"- `expl_occ_per_1k`: {row.expl_target_occ_per_1k:.3f}",
        "",
        "**Cloze**",
        "",
        "```text",
        shorten(probe_text["probe"]),
        "```",
        "",
        "**Target**",
        "",
        "```text",
        shorten(probe_text["target"], width=360),
        "```",
        "",
        "**Fact**",
        "",
        "```text",
        shorten(probe_text["fact"]),
        "```",
        "",
    ]


def section_for_group(df: pd.DataFrame, domain_group: str) -> list[str]:
    group = df.loc[df["domain_group"] == domain_group].copy()
    group["gap"] = group["source_only"] - group["with_explanations"]
    source_wins = group.loc[group["gap"] > 0].sort_values("gap", ascending=False).head(N_EXAMPLES)
    explanation_wins = group.loc[group["gap"] < 0].sort_values("gap", ascending=True).head(N_EXAMPLES)

    lines = [
        f"# {domain_group.title()} factual probe gap examples",
        "",
        f"Selected the largest {N_EXAMPLES} 7B factual cloze gaps in each direction.",
        "`source_minus_explanations` is final-step `source_only` log-prob minus final-step `with_explanations` log-prob.",
        "Occurrence columns are normalized exact full-target phrase appearances per 1,000 words.",
        "",
        "## Source > explanations",
        "",
    ]
    for rank, (_, row) in enumerate(source_wins.iterrows(), start=1):
        lines.extend(format_example(row, rank))

    lines.extend(["## Explanations > source", ""])
    for rank, (_, row) in enumerate(explanation_wins.iterrows(), start=1):
        lines.extend(format_example(row, rank))
    return lines


def main() -> None:
    df = pd.read_csv(PROBE_LEVEL_PATH)
    report_lines = [
        "# Factual Probe Source-vs-Explanation Gap Examples",
        "",
        "This report is generated from `factual_7b_source_vs_explanations_target_occurrence_probe_level.csv`.",
        "",
    ]
    for domain_group in DOMAIN_GROUPS:
        report_lines.extend(section_for_group(df, domain_group))
        report_lines.append("")
    OUTPUT_PATH.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
