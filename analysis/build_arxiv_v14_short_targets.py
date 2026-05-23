#!/usr/bin/env python3
"""Build a conservative short-target variant for arxiv factual probes."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
ARXIV_ROOT = REPO_ROOT / "probes" / "arxiv"
DATA_ROOT = REPO_ROOT / "data" / "arxiv" / "cleaned"
REPORT_DIR = REPO_ROOT / "reports" / "arxiv_factual_v14_short_targets"
REVIEW_INPUT = REPORT_DIR / "review_input_gt8.csv"
DECISIONS_INPUT = REPORT_DIR / "agent_decisions.csv"
OUTPUT_FILENAME = "probes_v14_short_targets.csv"


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def normalize_for_search(text: str) -> str:
    return normalize_space(text).casefold()


def strip_terminal_punctuation(text: str) -> str:
    return str(text).strip().rstrip(" \t\r\n.,;:!?")


def target_words(target: str) -> list[str]:
    return str(target).strip().split()


def discover_domains() -> list[str]:
    return sorted(
        path.name
        for path in ARXIV_ROOT.iterdir()
        if path.is_dir() and (path / "facts" / "probes_v13.csv").exists()
    )


def source_text(domain: str) -> str:
    path = DATA_ROOT / f"{domain}.tex"
    return path.read_text(encoding="utf-8", errors="replace")


def build_review_input() -> pd.DataFrame:
    rows = []
    for domain in discover_domains():
        probe_path = ARXIV_ROOT / domain / "facts" / "probes_v13.csv"
        df = pd.read_csv(probe_path)
        for idx, row in df.iterrows():
            words = target_words(row["target"])
            if len(words) > 8:
                rows.append(
                    {
                        "domain": domain,
                        "probe_index": int(idx),
                        "target_words": len(words),
                        "probe": row["probe"],
                        "target": row["target"],
                        "fact": row["fact"],
                        "raw_knowledge_statement": row["raw_knowledge_statement"],
                        "contextualized_question": row.get("contextualized_question", ""),
                    }
                )
    return pd.DataFrame(rows)


def parse_decisions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Decision CSV not found: {path}")
    decisions = pd.read_csv(path, keep_default_na=False)
    required = {"domain", "probe_index", "decision", "split_word_index", "new_target", "reason"}
    missing = required - set(decisions.columns)
    if missing:
        raise ValueError(f"Decision CSV missing required columns: {sorted(missing)}")
    decisions["probe_index"] = decisions["probe_index"].astype(int)
    decisions["decision"] = decisions["decision"].str.strip().str.upper()
    return decisions


def validate_decision(row: pd.Series, original: pd.Series, doc_text: str) -> tuple[bool, str, str, str]:
    old_target = str(original["target"])
    old_words = target_words(old_target)

    if row["decision"] != "ACCEPT":
        return False, str(original["probe"]), old_target, "rejected"

    try:
        split_idx = int(row["split_word_index"])
    except ValueError:
        return False, str(original["probe"]), old_target, "invalid split_word_index"

    if split_idx <= 0 or split_idx >= len(old_words):
        return False, str(original["probe"]), old_target, "split index is not a strict suffix split"

    moved_prefix = " ".join(old_words[:split_idx])
    proposed_target = " ".join(old_words[split_idx:])
    supplied_target = normalize_space(row["new_target"])
    if normalize_space(proposed_target) != supplied_target:
        return False, str(original["probe"]), old_target, "new_target does not match suffix from split_word_index"

    if len(target_words(proposed_target)) >= len(old_words):
        return False, str(original["probe"]), old_target, "new target is not shorter"

    if len(target_words(proposed_target)) == 0:
        return False, str(original["probe"]), old_target, "new target is empty"

    old_probe = str(original["probe"]).rstrip()
    joiner = "" if moved_prefix.startswith((" ", ",", ".", ";", ":", ")")) else " "
    new_probe = f"{old_probe}{joiner}{moved_prefix}".strip()
    new_target = proposed_target
    if old_target.startswith(" "):
        new_target = " " + new_target

    if normalize_space(new_probe + new_target) != normalize_space(original["fact"]):
        return False, str(original["probe"]), old_target, "new probe + target does not reconstruct fact"

    if new_probe.count("?") > str(original["probe"]).count("?"):
        return False, str(original["probe"]), old_target, "new probe introduces question mark"

    target_for_search = normalize_for_search(strip_terminal_punctuation(new_target).strip())
    if target_for_search not in normalize_for_search(doc_text):
        return False, str(original["probe"]), old_target, "new target not verbatim in source document"

    return True, new_probe, new_target, "accepted"


def apply_decisions(review_input: pd.DataFrame, decisions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = review_input.merge(
        decisions,
        on=["domain", "probe_index"],
        how="left",
        validate="one_to_one",
    )
    if merged["decision"].isna().any():
        missing = merged.loc[merged["decision"].isna(), ["domain", "probe_index"]]
        raise ValueError(f"Missing decisions for {len(missing)} review rows")

    records = []
    for _, row in merged.iterrows():
        valid, new_probe, new_target, validation_status = validate_decision(
            row,
            row,
            source_text(row["domain"]),
        )
        records.append(
            {
                "domain": row["domain"],
                "probe_index": int(row["probe_index"]),
                "old_target_words": len(target_words(row["target"])),
                "new_target_words": len(target_words(new_target)),
                "decision": row["decision"],
                "accepted": bool(valid),
                "validation_status": validation_status,
                "split_word_index": row["split_word_index"],
                "old_probe": row["probe"],
                "old_target": row["target"],
                "new_probe": new_probe,
                "new_target": new_target,
                "fact": row["fact"],
                "reason": row["reason"],
            }
        )
    decisions_full = pd.DataFrame(records)
    accepted = decisions_full[decisions_full["accepted"]].copy()
    return decisions_full, accepted


def write_probe_outputs(accepted: pd.DataFrame) -> None:
    accepted_map = {
        (row.domain, int(row.probe_index)): row
        for row in accepted.itertuples(index=False)
    }
    for domain in discover_domains():
        source_path = ARXIV_ROOT / domain / "facts" / "probes_v13.csv"
        out_path = ARXIV_ROOT / domain / "facts" / OUTPUT_FILENAME
        df = pd.read_csv(source_path)
        before_cols = list(df.columns)
        for idx in range(len(df)):
            key = (domain, idx)
            if key not in accepted_map:
                continue
            decision = accepted_map[key]
            df.loc[idx, "probe"] = decision.new_probe
            df.loc[idx, "target"] = decision.new_target
        if list(df.columns) != before_cols:
            raise AssertionError(f"Column mismatch for {domain}")
        if len(df) != len(pd.read_csv(source_path)):
            raise AssertionError(f"Row count mismatch for {domain}")
        df.to_csv(out_path, index=False)


def write_report(review_input: pd.DataFrame, decisions_full: pd.DataFrame) -> None:
    accepted = decisions_full[decisions_full["accepted"]].copy()
    rejected = decisions_full[~decisions_full["accepted"]].copy()
    before = review_input.groupby("domain").size().rename("reviewed")
    after = accepted.groupby("domain").size().rename("accepted")
    summary = pd.concat([before, after], axis=1).fillna(0).astype(int).reset_index()
    summary["rejected_or_invalid"] = summary["reviewed"] - summary["accepted"]

    lines = [
        "# Arxiv Factual v14 Short Targets",
        "",
        "Conservative suffix-only target shortening for arxiv factual v13 rows with targets longer than 8 words.",
        "",
        "## Summary",
        "",
        summary.to_csv(index=False),
        "",
        "## Accepted Examples",
        "",
    ]
    for _, row in accepted.head(20).iterrows():
        lines.extend(
            [
                f"### {row['domain']} row {row['probe_index']}",
                "",
                f"- old target words: {row['old_target_words']}",
                f"- new target words: {row['new_target_words']}",
                f"- reason: {row['reason']}",
                "",
                "Old probe:",
                "```text",
                f"{row['old_probe']}",
                "```",
                "Old target:",
                "```text",
                f"{row['old_target']}",
                "```",
                "New probe:",
                "```text",
                f"{row['new_probe']}",
                "```",
                "New target:",
                "```text",
                f"{row['new_target']}",
                "```",
                "",
            ]
        )

    lines.extend(["## Rejection Reasons", ""])
    for _, row in rejected.head(40).iterrows():
        lines.append(f"- `{row['domain']}` row `{row['probe_index']}`: {row['reason']} ({row['validation_status']})")

    (REPORT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", type=Path, default=DECISIONS_INPUT)
    parser.add_argument(
        "--review-only",
        action="store_true",
        help="Only write review_input_gt8.csv; do not require decisions or write probe outputs.",
    )
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    review_input = build_review_input()
    review_input.to_csv(REVIEW_INPUT, index=False)

    if args.review_only:
        print(f"Reviewed: {len(review_input)}")
        print(f"Wrote {REVIEW_INPUT.relative_to(REPO_ROOT)}")
        return

    decisions = parse_decisions(args.decisions)
    decisions_full, accepted = apply_decisions(review_input, decisions)
    rejected = decisions_full[~decisions_full["accepted"]].copy()

    decisions_full.to_csv(REPORT_DIR / "decisions.csv", index=False)
    accepted.to_csv(REPORT_DIR / "accepted_rewrites.csv", index=False)
    rejected.to_csv(REPORT_DIR / "rejected_rewrites.csv", index=False)
    write_probe_outputs(accepted)
    write_report(review_input, decisions_full)

    print(f"Reviewed: {len(review_input)}")
    print(f"Accepted: {len(accepted)}")
    print(f"Rejected/invalid: {len(rejected)}")
    print(f"Wrote {REPORT_DIR.relative_to(REPO_ROOT)}")
    print(f"Wrote per-domain facts/{OUTPUT_FILENAME}")


if __name__ == "__main__":
    main()
