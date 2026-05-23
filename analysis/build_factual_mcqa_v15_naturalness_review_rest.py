#!/usr/bin/env python3
"""Build and combine the naturalness review for unsampled factual v15 MCQA rows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd

from build_factual_mcqa_v15_naturalness_review import (
    LABELS,
    OPTION_COLS,
    PROBE_ROOT,
    REPO_ROOT,
    build_mcqa_5shot_prompt,
    discover_v15_paths,
    format_question,
    normalize_answer,
    normalize_label,
    normalize_space,
    read_v14_row,
    validate_decision_row,
)


DEFAULT_REPORT_DIR = REPO_ROOT / "reports" / "factual_mcqa_v15_naturalness_review_rest"
DEFAULT_EXCLUDE_INPUT = (
    REPO_ROOT
    / "reports"
    / "factual_mcqa_v15_naturalness_review"
    / "review_input_sample.csv"
)
REVIEW_INPUT_NAME = "review_input_rest.csv"
ISSUES_GLOB = "agent_issue_rows_*.csv"


def load_excluded_review_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return set(pd.read_csv(path, keep_default_na=False)["review_id"].astype(str))


def build_review_input(excluded_review_ids: set[str]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for path in discover_v15_paths():
        rel = path.relative_to(PROBE_ROOT)
        domain_group, document = rel.parts[0], rel.parts[1]
        v15 = pd.read_csv(path, keep_default_na=False)
        v14 = pd.read_csv(path.with_name("probes_v14_mcqa.csv"), keep_default_na=False)
        for row_index, row in v15.iterrows():
            review_id = f"{domain_group}/{document}/{int(row_index)}"
            if review_id in excluded_review_ids:
                continue
            v14_row = v14.iloc[int(row_index)]
            records.append(
                {
                    "review_id": review_id,
                    "domain_group": domain_group,
                    "document": document,
                    "v15_row_index": int(row_index),
                    "question_stem": row["probe"],
                    "target": row["target"],
                    "correct_label": row["correct_label"],
                    "option_a": row["option_a"],
                    "option_b": row["option_b"],
                    "option_c": row["option_c"],
                    "option_d": row["option_d"],
                    "option_e": row["option_e"],
                    "distractors": row["distractors"],
                    "formatted_question": row["formatted_question"],
                    "fact": row["fact"],
                    "raw_knowledge_statement": row.get("raw_knowledge_statement", ""),
                    "section": row.get("section", ""),
                    "v14_cloze_probe": v14_row["probe"],
                    "v14_formatted_question": v14_row["formatted_question"],
                }
            )
    return pd.DataFrame(records).sort_values(
        ["domain_group", "document", "v15_row_index"]
    )


def write_review_input(report_dir: Path, exclude_input: Path, batch_size: int) -> pd.DataFrame:
    report_dir.mkdir(parents=True, exist_ok=True)
    review_input = build_review_input(load_excluded_review_ids(exclude_input))
    review_input.to_csv(report_dir / REVIEW_INPUT_NAME, index=False)

    batch_rows: list[dict[str, object]] = []
    for domain_group, domain_rows in review_input.groupby("domain_group", sort=True):
        domain_rows = domain_rows.reset_index(drop=True)
        domain_rows.to_csv(report_dir / f"review_input_rest_{domain_group}.csv", index=False)
        for batch_index, start in enumerate(range(0, len(domain_rows), batch_size), start=1):
            batch = domain_rows.iloc[start : start + batch_size]
            batch_path = report_dir / f"review_batch_{domain_group}_{batch_index:02d}.csv"
            batch.to_csv(batch_path, index=False)
            batch_rows.append(
                {
                    "batch_file": batch_path.name,
                    "domain_group": domain_group,
                    "batch_index": batch_index,
                    "rows": len(batch),
                    "first_review_id": batch.iloc[0]["review_id"],
                    "last_review_id": batch.iloc[-1]["review_id"],
                }
            )
    pd.DataFrame(batch_rows).to_csv(report_dir / "batch_manifest.csv", index=False)
    return review_input


def read_issue_rows(report_dir: Path) -> pd.DataFrame:
    paths = sorted(report_dir.glob(ISSUES_GLOB))
    if not paths:
        return pd.DataFrame()
    frames = [pd.read_csv(path, keep_default_na=False) for path in paths]
    issues = pd.concat(frames, ignore_index=True)
    if issues["review_id"].duplicated().any():
        dupes = sorted(issues.loc[issues["review_id"].duplicated(), "review_id"].unique())
        raise ValueError(f"Duplicate issue decisions for review_id values: {dupes}")
    return issues


def combine_decisions(report_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    review_input = pd.read_csv(report_dir / REVIEW_INPUT_NAME, keep_default_na=False)
    issues = read_issue_rows(report_dir)
    issue_map = {
        str(row.review_id): row._asdict()
        for row in issues.itertuples(index=False)
    }

    rows: list[dict[str, object]] = []
    for _, base in review_input.iterrows():
        record = base.to_dict()
        issue = issue_map.get(str(base["review_id"]), {})
        decision = str(issue.get("decision", "accept")).strip().lower() or "accept"
        record.update(
            {
                "decision": decision,
                "issue_categories": issue.get("issue_categories", ""),
                "review_reason": issue.get("review_reason", ""),
                "fixed_question_stem": issue.get("fixed_question_stem", ""),
                "fixed_option_a": issue.get("fixed_option_a", ""),
                "fixed_option_b": issue.get("fixed_option_b", ""),
                "fixed_option_c": issue.get("fixed_option_c", ""),
                "fixed_option_d": issue.get("fixed_option_d", ""),
                "fixed_option_e": issue.get("fixed_option_e", ""),
                "fixed_correct_label": issue.get("fixed_correct_label", ""),
                "fixed_target": issue.get("fixed_target", ""),
            }
        )
        if decision == "fix":
            fixed_correct = normalize_label(record["fixed_correct_label"]) or normalize_label(
                record["correct_label"]
            )
            if fixed_correct in LABELS:
                correct_idx = LABELS.index(fixed_correct)
                fixed_options = [record[f"fixed_{col}"] for col in OPTION_COLS]
                record["fixed_target"] = (
                    normalize_space(record["fixed_target"])
                    or normalize_space(fixed_options[correct_idx])
                )
                record["target_changed_for_fix"] = (
                    normalize_answer(record["fixed_target"]) != normalize_answer(record["target"])
                )
            else:
                record["target_changed_for_fix"] = ""
        else:
            record["fixed_target"] = ""
            record["target_changed_for_fix"] = ""

        valid, validation_error = validate_decision_row(pd.Series(record))
        record["validation_error"] = validation_error
        if decision == "fix" and valid:
            fixed_options = [record[f"fixed_{col}"] for col in OPTION_COLS]
            fixed_formatted = format_question(str(record["fixed_question_stem"]), fixed_options)
            fixed_correct = normalize_label(record["fixed_correct_label"]) or normalize_label(
                record["correct_label"]
            )
            record["fixed_correct_label"] = fixed_correct
            correct_idx = LABELS.index(fixed_correct)
            record["fixed_formatted_question"] = fixed_formatted
            record["fixed_formatted_question_5shot"] = build_mcqa_5shot_prompt(fixed_formatted)
            record["fixed_distractors"] = json.dumps(
                [option for idx, option in enumerate(fixed_options) if idx != correct_idx],
                ensure_ascii=False,
            )
        else:
            record["fixed_formatted_question"] = ""
            record["fixed_formatted_question_5shot"] = ""
            record["fixed_distractors"] = ""
        rows.append(record)

    decisions = pd.DataFrame(rows)
    accepted = decisions[decisions["decision"] == "accept"].copy()
    fixed = decisions[decisions["decision"] == "fix"].copy()
    rejected = decisions[decisions["decision"] == "reject"].copy()
    return decisions, accepted, fixed, rejected


def write_outputs(report_dir: Path) -> None:
    decisions, accepted, fixed, rejected = combine_decisions(report_dir)
    decisions.to_csv(report_dir / "agent_decisions.csv", index=False)
    accepted.to_csv(report_dir / "accepted_rows.csv", index=False)
    fixed.to_csv(report_dir / "fixed_rows.csv", index=False)
    rejected.to_csv(report_dir / "rejected_rows.csv", index=False)

    summary = (
        decisions.groupby(["domain_group", "document", "decision"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for col in ("accept", "fix", "reject"):
        if col not in summary.columns:
            summary[col] = 0
    summary["reviewed_rows"] = summary[["accept", "fix", "reject"]].sum(axis=1)
    summary.to_csv(report_dir / "summary.csv", index=False)

    invalid = decisions[decisions["validation_error"].astype(str) != ""]
    if not invalid.empty:
        raise AssertionError(
            f"{len(invalid)} decisions failed validation; see "
            f"{(report_dir / 'agent_decisions.csv').relative_to(REPO_ROOT)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--exclude-input", type=Path, default=DEFAULT_EXCLUDE_INPUT)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--combine", action="store_true")
    args = parser.parse_args()

    review_input = write_review_input(args.report_dir, args.exclude_input, args.batch_size)
    print(f"Wrote {(args.report_dir / REVIEW_INPUT_NAME).relative_to(REPO_ROOT)} ({len(review_input)} rows)")
    print(
        review_input.groupby("domain_group")
        .size()
        .rename("review_rows")
        .to_string()
    )
    print(f"Wrote {(args.report_dir / 'batch_manifest.csv').relative_to(REPO_ROOT)}")

    if args.combine:
        write_outputs(args.report_dir)
        print(f"Wrote {(args.report_dir / 'agent_decisions.csv').relative_to(REPO_ROOT)}")
        print(f"Wrote {(args.report_dir / 'summary.csv').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    csv.field_size_limit(10_000_000)
    main()
