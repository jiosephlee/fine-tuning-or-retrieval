#!/usr/bin/env python3
"""Build a stratified naturalness review for factual v15 MCQA rows."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from utils.mcqa_prompts import build_mcqa_5shot_prompt  # noqa: E402


PROBE_ROOT = REPO_ROOT / "probes"
REPORT_DIR = REPO_ROOT / "reports" / "factual_mcqa_v15_naturalness_review"
REVIEW_INPUT = REPORT_DIR / "review_input_sample.csv"
DECISIONS = REPORT_DIR / "agent_decisions.csv"
ACCEPTED = REPORT_DIR / "accepted_rows.csv"
FIXED = REPORT_DIR / "fixed_rows.csv"
REJECTED = REPORT_DIR / "rejected_rows.csv"
SUMMARY = REPORT_DIR / "summary.csv"
ISSUES_GLOB = "agent_issue_rows_*.csv"

DOMAIN_GROUPS = ("medical", "legal", "arxiv")
OPTION_COLS = ("option_a", "option_b", "option_c", "option_d", "option_e")
LABELS = ("(A)", "(B)", "(C)", "(D)", "(E)")
DEFAULT_SEED = 20260523
DEFAULT_ROWS_PER_DOCUMENT = 10


def normalize_space(text: object) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def normalize_answer(text: object) -> str:
    return normalize_space(text).casefold()


def format_question(stem: str, options: list[str]) -> str:
    return normalize_space(stem) + "\n" + "\n".join(
        f"{label} {option}" for label, option in zip(LABELS, options)
    )


def parse_json_list(value: object) -> list[str]:
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed]


def normalize_label(label: object) -> str:
    text = str(label).strip().upper()
    if len(text) == 1 and text in "ABCDE":
        return f"({text})"
    return text


def discover_v15_paths() -> list[Path]:
    paths: list[Path] = []
    for group in DOMAIN_GROUPS:
        paths.extend((PROBE_ROOT / group).glob("*/facts/probes_v15_mcqa.csv"))
    return sorted(paths)


def read_v14_row(v15_path: Path, row_index: int) -> pd.Series:
    return pd.read_csv(v15_path.with_name("probes_v14_mcqa.csv"), keep_default_na=False).iloc[
        row_index
    ]


def build_review_input(rows_per_document: int, seed: int) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for path in discover_v15_paths():
        rel = path.relative_to(PROBE_ROOT)
        domain_group, document = rel.parts[0], rel.parts[1]
        v15 = pd.read_csv(path, keep_default_na=False)
        sample = v15.sample(
            n=min(rows_per_document, len(v15)),
            random_state=seed,
        ).sort_index()
        for row_index, row in sample.iterrows():
            v14_row = read_v14_row(path, int(row_index))
            review_id = f"{domain_group}/{document}/{int(row_index)}"
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
    review_input = pd.DataFrame(records).sort_values(
        ["domain_group", "document", "v15_row_index"]
    )
    return review_input


def write_review_input(rows_per_document: int, seed: int) -> pd.DataFrame:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    review_input = build_review_input(rows_per_document, seed)
    review_input.to_csv(REVIEW_INPUT, index=False)
    for domain_group, domain_rows in review_input.groupby("domain_group", sort=True):
        domain_rows.to_csv(REPORT_DIR / f"review_input_{domain_group}.csv", index=False)
    return review_input


def read_issue_rows() -> pd.DataFrame:
    paths = sorted(REPORT_DIR.glob(ISSUES_GLOB))
    if not paths:
        return pd.DataFrame()
    frames = [pd.read_csv(path, keep_default_na=False) for path in paths]
    issues = pd.concat(frames, ignore_index=True)
    if issues["review_id"].duplicated().any():
        dupes = sorted(issues.loc[issues["review_id"].duplicated(), "review_id"].unique())
        raise ValueError(f"Duplicate issue decisions for review_id values: {dupes}")
    return issues


def validate_decision_row(row: pd.Series) -> tuple[bool, str]:
    decision = str(row["decision"]).strip().lower()
    if decision not in {"accept", "fix", "reject"}:
        return False, "invalid decision"
    if decision != "fix":
        return True, ""

    fixed_stem = normalize_space(row.get("fixed_question_stem", ""))
    options = [normalize_space(row.get(f"fixed_{col}", "")) for col in OPTION_COLS]
    correct_label = normalize_label(row.get("fixed_correct_label", "")) or normalize_label(
        row["correct_label"]
    )
    if not fixed_stem:
        return False, "fix missing fixed_question_stem"
    if len(options) != 5 or any(not option for option in options):
        return False, "fix missing one or more fixed options"
    if correct_label not in LABELS:
        return False, "fix invalid fixed_correct_label"
    correct_idx = LABELS.index(correct_label)
    fixed_target = normalize_space(row.get("fixed_target", "")) or options[correct_idx]
    if options[correct_idx] != fixed_target:
        return False, "fix correct option does not equal fixed_target"
    if sum(normalize_answer(option) == normalize_answer(fixed_target) for option in options) != 1:
        return False, "fix fixed_target is not unique among options"
    if normalize_answer(fixed_target) in normalize_answer(fixed_stem):
        return False, "fix leaks fixed_target in stem"
    return True, ""


def combine_decisions() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    review_input = pd.read_csv(REVIEW_INPUT, keep_default_na=False)
    issues = read_issue_rows()
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
                    normalize_space(record["fixed_target"]) or normalize_space(fixed_options[correct_idx])
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
    bad = decisions[decisions["validation_error"].astype(str) != ""].copy()
    accepted = decisions[decisions["decision"] == "accept"].copy()
    fixed = decisions[decisions["decision"] == "fix"].copy()
    rejected = decisions[decisions["decision"] == "reject"].copy()
    return decisions, accepted, fixed, rejected


def write_outputs() -> None:
    decisions, accepted, fixed, rejected = combine_decisions()
    decisions.to_csv(DECISIONS, index=False)
    accepted.to_csv(ACCEPTED, index=False)
    fixed.to_csv(FIXED, index=False)
    rejected.to_csv(REJECTED, index=False)

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
    summary.to_csv(SUMMARY, index=False)

    invalid = decisions[decisions["validation_error"].astype(str) != ""]
    if not invalid.empty:
        raise AssertionError(
            f"{len(invalid)} decisions failed validation; see {DECISIONS.relative_to(REPO_ROOT)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows-per-document", type=int, default=DEFAULT_ROWS_PER_DOCUMENT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--combine",
        action="store_true",
        help="Combine agent_issue_rows_*.csv files into final decision artifacts.",
    )
    args = parser.parse_args()

    review_input = write_review_input(args.rows_per_document, args.seed)
    print(f"Wrote {REVIEW_INPUT.relative_to(REPO_ROOT)} ({len(review_input)} rows)")
    print(
        review_input.groupby("domain_group")
        .size()
        .rename("sampled_rows")
        .to_string()
    )

    if args.combine:
        write_outputs()
        print(f"Wrote {DECISIONS.relative_to(REPO_ROOT)}")
        print(f"Wrote {SUMMARY.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    csv.field_size_limit(10_000_000)
    main()
