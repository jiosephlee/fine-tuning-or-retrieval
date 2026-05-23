#!/usr/bin/env python3
"""Apply accepted factual v15 MCQA naturalness repairs and drop rejects."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import pandas as pd


def find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "utils").is_dir() and (parent / "probes").is_dir():
            return parent
    raise RuntimeError("Could not locate repository root")


REPO_ROOT = find_repo_root()
sys.path.append(str(REPO_ROOT))

from utils.mcqa_prompts import build_mcqa_5shot_prompt  # noqa: E402


PROBE_ROOT = REPO_ROOT / "probes"
DEFAULT_DECISIONS = (
    REPO_ROOT
    / "reports"
    / "factual_mcqa_v15_naturalness_review_rest"
    / "full_agent_decisions_including_sample.csv"
)
DEFAULT_REPORT_DIR = REPO_ROOT / "reports" / "factual_mcqa_v15_repaired"
OPTION_COLS = ("option_a", "option_b", "option_c", "option_d", "option_e")
LABELS = ("(A)", "(B)", "(C)", "(D)", "(E)")
REPAIR_COLS = (
    "probe",
    "target",
    "formatted_question",
    "option_a",
    "option_b",
    "option_c",
    "option_d",
    "option_e",
    "correct_label",
    "distractors",
    "formatted_question_5shot",
)
VALIDATION_COLS = (
    "domain_group",
    "document",
    "row_index",
    "errors",
    "probe",
    "target",
)


def normalize_space(value: object) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_answer(value: object) -> str:
    return normalize_space(value).casefold()


def normalize_label(value: object) -> str:
    text = str(value).strip().upper()
    if len(text) == 1 and text in "ABCDE":
        return f"({text})"
    return text


def format_question(stem: object, options: list[str]) -> str:
    return normalize_space(stem) + "\n" + "\n".join(
        f"{label} {option}" for label, option in zip(LABELS, options)
    )


def option_for_label(row: pd.Series, label: str) -> str:
    return str(row[OPTION_COLS[LABELS.index(label)]])


def decision_key(row: pd.Series) -> tuple[str, str, int]:
    return (
        str(row["domain_group"]),
        str(row["document"]),
        int(row["v15_row_index"]),
    )


def validate_mcqa_row(row: pd.Series) -> list[str]:
    errors: list[str] = []
    label = normalize_label(row.get("correct_label", ""))
    if label not in LABELS:
        errors.append("bad_correct_label")
        return errors
    options = [str(row[col]) for col in OPTION_COLS]
    target = str(row.get("target", ""))
    if option_for_label(row, label) != target:
        errors.append("correct_option_not_target")
    if sum(normalize_answer(option) == normalize_answer(target) for option in options) != 1:
        errors.append("target_not_unique_in_options")
    stem = str(row.get("formatted_question", "")).split("\n(A)", 1)[0]
    if normalize_answer(target) and normalize_answer(target) in normalize_answer(stem):
        errors.append("target_leaks_in_stem")
    if not str(row.get("formatted_question_5shot", "")).strip():
        errors.append("missing_formatted_question_5shot")
    return errors


def apply_fix(base: pd.Series, decision: pd.Series) -> pd.Series:
    fixed = base.copy()
    fixed_options = [normalize_space(decision[f"fixed_{col}"]) for col in OPTION_COLS]
    fixed_label = normalize_label(decision.get("fixed_correct_label", "")) or normalize_label(
        base["correct_label"]
    )
    if fixed_label not in LABELS:
        raise ValueError(f"{decision['review_id']}: invalid fixed_correct_label {fixed_label!r}")
    fixed_target = normalize_space(decision.get("fixed_target", "")) or fixed_options[
        LABELS.index(fixed_label)
    ]
    fixed_stem = normalize_space(decision["fixed_question_stem"])
    fixed_formatted = format_question(fixed_stem, fixed_options)

    fixed["probe"] = fixed_stem
    fixed["target"] = fixed_target
    fixed["formatted_question"] = fixed_formatted
    for col, value in zip(OPTION_COLS, fixed_options):
        fixed[col] = value
    fixed["correct_label"] = fixed_label
    fixed["distractors"] = str(decision.get("fixed_distractors", "")).strip() or json.dumps(
        [option for idx, option in enumerate(fixed_options) if idx != LABELS.index(fixed_label)],
        ensure_ascii=False,
    )
    fixed["formatted_question_5shot"] = str(
        decision.get("fixed_formatted_question_5shot", "")
    ).strip() or build_mcqa_5shot_prompt(fixed_formatted)
    return fixed


def apply_repairs(decisions_path: Path, report_dir: Path, dry_run: bool) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    decisions = pd.read_csv(decisions_path, keep_default_na=False)
    decisions["decision"] = decisions["decision"].astype(str).str.strip().str.lower()
    if decisions["review_id"].duplicated().any():
        dupes = sorted(decisions.loc[decisions["review_id"].duplicated(), "review_id"].unique())
        raise ValueError(f"Duplicate review_id values in decisions: {dupes[:10]}")

    decision_map = {decision_key(row): row for _, row in decisions.iterrows()}
    rejected_records: list[dict[str, object]] = []
    summary_records: list[dict[str, object]] = []
    validation_records: list[dict[str, object]] = []

    for (domain, document), group in decisions.groupby(["domain_group", "document"], sort=True):
        path = PROBE_ROOT / domain / document / "facts" / "probes_v15_mcqa.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        source = pd.read_csv(path, keep_default_na=False)
        out_rows: list[pd.Series] = []
        fixed_count = 0
        rejected_count = 0

        for source_index, source_row in source.iterrows():
            key = (str(domain), str(document), int(source_index))
            decision = decision_map.get(key)
            if decision is None:
                raise ValueError(f"Missing naturalness decision for {domain}/{document}/{source_index}")
            action = str(decision["decision"]).strip().lower()
            if action == "accept":
                out_rows.append(source_row.copy())
            elif action == "fix":
                fixed_count += 1
                out_rows.append(apply_fix(source_row, decision))
            elif action == "reject":
                rejected_count += 1
                rec = source_row.to_dict()
                rec.update(
                    {
                        "domain_group": domain,
                        "document": document,
                        "v15_row_index": int(source_index),
                        "review_id": decision["review_id"],
                        "review_reason": decision.get("review_reason", ""),
                        "issue_categories": decision.get("issue_categories", ""),
                    }
                )
                rejected_records.append(rec)
            else:
                raise ValueError(f"{decision['review_id']}: invalid decision {action!r}")

        repaired = pd.DataFrame(out_rows, columns=source.columns)
        for row_index, row in repaired.iterrows():
            errors = validate_mcqa_row(row)
            if errors:
                validation_records.append(
                    {
                        "domain_group": domain,
                        "document": document,
                        "row_index": int(row_index),
                        "errors": ";".join(errors),
                        "probe": row.get("probe", ""),
                        "target": row.get("target", ""),
                    }
                )

        backup_path = path.with_suffix(".pre_naturalness_repair.csv")
        summary_records.append(
            {
                "domain_group": domain,
                "document": document,
                "source_rows": len(source),
                "accepted_rows": int((group["decision"] == "accept").sum()),
                "fixed_rows": fixed_count,
                "rejected_rows": rejected_count,
                "final_rows": len(repaired),
                "path": str(path.relative_to(REPO_ROOT)),
                "backup_path": str(backup_path.relative_to(REPO_ROOT)),
            }
        )

        if not dry_run:
            if not backup_path.exists():
                shutil.copy2(path, backup_path)
            repaired.to_csv(path, index=False)

    rejected = pd.DataFrame(rejected_records)
    summary = pd.DataFrame(summary_records)
    validation = pd.DataFrame(validation_records, columns=VALIDATION_COLS)
    rejected.to_csv(report_dir / "rejected_rows_applied.csv", index=False)
    summary.to_csv(report_dir / "summary.csv", index=False)
    validation.to_csv(report_dir / "validation.csv", index=False)
    if not validation.empty:
        raise AssertionError(
            f"{len(validation)} repaired factual rows failed validation; see "
            f"{(report_dir / 'validation.csv').relative_to(REPO_ROOT)}"
        )
    print(f"Wrote {(report_dir / 'summary.csv').relative_to(REPO_ROOT)}")
    print(f"Wrote {(report_dir / 'rejected_rows_applied.csv').relative_to(REPO_ROOT)}")
    if dry_run:
        print("Dry run only; probe files were not modified.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    apply_repairs(args.decisions, args.report_dir, args.dry_run)


if __name__ == "__main__":
    main()
