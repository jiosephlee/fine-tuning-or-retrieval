#!/usr/bin/env python3
"""Recover medical inference MCQA rows missing from v13 MCQA."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd


def find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "utils").is_dir() and (parent / "probes").is_dir():
            return parent
    raise RuntimeError("Could not locate repository root")


ROOT = find_repo_root()
sys.path.append(str(ROOT))

from utils import utils  # noqa: E402
from utils.mcqa_prompts import build_mcqa_5shot_prompt  # noqa: E402


MEDICAL_DIR = ROOT / "probes" / "medical"
REPORT_DIR = ROOT / "reports" / "medical_mcqa_recovery_v13"
LABELS = ("(A)", "(B)", "(C)", "(D)", "(E)")
OPTION_COLS = ("option_a", "option_b", "option_c", "option_d", "option_e")
MCQA_COLS = [
    "probe",
    "target",
    "correct_label",
    "formatted_question",
    "option_a",
    "option_b",
    "option_c",
    "option_d",
    "option_e",
    "distractors",
    "fact",
    "raw_knowledge_statement",
    "section",
    "inference_type",
    "source_fact(s)",
    "source_facts",
    "text_sentences",
    "derivation",
    "question",
    "answer",
    "formatted_question_5shot",
]
ISSUE_COLS = [
    "review_id",
    "decision",
    "issue_categories",
    "review_reason",
    "fixed_question_stem",
    "fixed_option_a",
    "fixed_option_b",
    "fixed_option_c",
    "fixed_option_d",
    "fixed_option_e",
    "fixed_correct_label",
    "fixed_target",
]
VALIDATION_COLS = [
    "review_id",
    "document",
    "v13_row_index",
    "errors",
    "probe",
    "target",
]
GENERATION_FAILURE_COLS = ["review_id", "document", "v13_row_index", "error"]


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
        f"{label} {option}" for label, option in zip(LABELS, options, strict=True)
    )


def parse_json_response(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    parsed = json.loads(str(value).strip())
    if not isinstance(parsed, dict):
        raise ValueError("LLM JSON response was not an object")
    return parsed


def review_id(document: str, row_index: int) -> str:
    return f"medical/{document}/{row_index}"


def discover_missing_rows() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for path in sorted(MEDICAL_DIR.glob("*/inference/probes_v13.csv")):
        document = path.parts[-3]
        v13 = pd.read_csv(path, keep_default_na=False)
        mcqa_path = path.with_name("probes_v13_mcqa.csv")
        mcqa = pd.read_csv(mcqa_path, keep_default_na=False)
        covered = set(zip(mcqa["question"].astype(str), mcqa["answer"].astype(str)))
        for row_index, row in v13.iterrows():
            if (str(row["question"]), str(row["answer"])) in covered:
                continue
            record = row.to_dict()
            record.update(
                {
                    "review_id": review_id(document, int(row_index)),
                    "document": document,
                    "v13_row_index": int(row_index),
                    "v13_path": str(path.relative_to(ROOT)),
                    "v13_mcqa_path": str(mcqa_path.relative_to(ROOT)),
                }
            )
            records.append(record)
    columns = [
        "review_id",
        "document",
        "v13_row_index",
        "v13_path",
        "v13_mcqa_path",
        "original_row_index",
        "target",
        "probe",
        "fact",
        "inference_type",
        "source_fact(s)",
        "derivation",
        "question",
        "answer",
        "source_facts",
        "text_sentences",
    ]
    if not records:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(records).sort_values(["document", "v13_row_index"])


def generation_prompt(row: pd.Series) -> dict[str, str]:
    payload = {
        "document": row["document"],
        "question": row.get("question", ""),
        "answer": row.get("answer", ""),
        "target": row.get("target", ""),
        "probe": row.get("probe", ""),
        "fact": row.get("fact", ""),
        "inference_type": row.get("inference_type", ""),
        "source_facts": row.get("source_facts", row.get("source_fact(s)", "")),
        "text_sentences": row.get("text_sentences", ""),
        "derivation": row.get("derivation", ""),
    }
    return {
        "system": (
            "You create medical case-report inference MCQA probes. Return JSON only."
        ),
        "user": (
            "Create one five-option multiple-choice question from the provided inference probe.\n\n"
            "Requirements:\n"
            "- The correct answer must be exactly the provided target/answer text, preserving spelling.\n"
            "- Write a natural standalone question stem. Do not use a cloze fragment.\n"
            "- Do not reveal the answer string in the stem.\n"
            "- All options must be medically plausible and the same semantic type as the answer.\n"
            "- Exactly one option must equal the target. The other four options must be distinct distractors.\n"
            "- Do not invent facts beyond the supplied case-report text, source facts, and derivation.\n"
            "- Avoid near-binary choices and avoid options that are just negations of each other.\n"
            "- Return JSON with keys: question_stem, target, correct_label, options, rationale.\n"
            "- options must be an object with keys A, B, C, D, E.\n\n"
            f"Input JSON:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
        ),
    }


def validate_candidate_record(row: pd.Series) -> list[str]:
    errors: list[str] = []
    label = normalize_label(row.get("correct_label", ""))
    if label not in LABELS:
        return ["bad_correct_label"]
    options = [normalize_space(row.get(col, "")) for col in OPTION_COLS]
    target = normalize_space(row.get("target", ""))
    if any(not option for option in options):
        errors.append("missing_option")
    if options[LABELS.index(label)] != target:
        errors.append("correct_option_not_target")
    if sum(normalize_answer(option) == normalize_answer(target) for option in options) != 1:
        errors.append("target_not_unique")
    stem = normalize_space(row.get("probe", ""))
    if normalize_answer(target) and normalize_answer(target) in normalize_answer(stem):
        errors.append("target_leaks_in_stem")
    if not normalize_space(row.get("formatted_question_5shot", "")):
        errors.append("missing_5shot")
    return errors


def mcqa_from_parts(source: pd.Series, stem: str, target: str, correct_label: str, options: list[str]) -> dict[str, object]:
    correct_label = normalize_label(correct_label)
    formatted = format_question(stem, options)
    correct_index = LABELS.index(correct_label)
    distractors = [option for idx, option in enumerate(options) if idx != correct_index]
    return {
        "review_id": source["review_id"],
        "document": source["document"],
        "v13_row_index": source["v13_row_index"],
        "probe": normalize_space(stem),
        "target": normalize_space(target),
        "correct_label": correct_label,
        "formatted_question": formatted,
        "option_a": options[0],
        "option_b": options[1],
        "option_c": options[2],
        "option_d": options[3],
        "option_e": options[4],
        "distractors": json.dumps(distractors, ensure_ascii=False),
        "fact": source.get("fact", ""),
        "raw_knowledge_statement": "",
        "section": "",
        "inference_type": source.get("inference_type", ""),
        "source_fact(s)": source.get("source_fact(s)", source.get("source_facts", "")),
        "source_facts": source.get("source_facts", source.get("source_fact(s)", "")),
        "text_sentences": source.get("text_sentences", ""),
        "derivation": source.get("derivation", ""),
        "question": source.get("question", ""),
        "answer": source.get("answer", ""),
        "formatted_question_5shot": build_mcqa_5shot_prompt(formatted),
    }


def generate_candidates(args: argparse.Namespace) -> None:
    args.report_dir.mkdir(parents=True, exist_ok=True)
    missing = discover_missing_rows()
    missing.to_csv(args.report_dir / "missing_medical_v13_rows.csv", index=False)
    output_path = args.report_dir / "generated_candidates.csv"
    failure_path = args.report_dir / "generation_failures.csv"
    existing = pd.read_csv(output_path, keep_default_na=False) if output_path.exists() else pd.DataFrame()
    done = set(existing.get("review_id", pd.Series(dtype=str)).astype(str))
    candidates = existing.to_dict("records") if not existing.empty else []
    failures: list[dict[str, object]] = []

    for _, row in missing.iterrows():
        if str(row["review_id"]) in done and not args.force:
            continue
        last_error = ""
        for _ in range(args.max_try_num):
            try:
                raw = utils.query_llm(
                    generation_prompt(row),
                    model=args.model,
                    reasoning_effort=args.reasoning_effort,
                    system_prompt_included=True,
                    return_json=True,
                    max_tokens=args.max_tokens,
                    max_try_num=1,
                )
                parsed = parse_json_response(raw)
                options_obj = parsed.get("options", {})
                options = [normalize_space(options_obj.get(letter, "")) for letter in "ABCDE"]
                target = normalize_space(parsed.get("target", "")) or normalize_space(row.get("target", ""))
                candidate = mcqa_from_parts(
                    row,
                    str(parsed.get("question_stem", "")),
                    target,
                    str(parsed.get("correct_label", "")),
                    options,
                )
                candidate["generation_rationale"] = parsed.get("rationale", "")
                candidate["raw_response"] = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
                errors = validate_candidate_record(pd.Series(candidate))
                if errors:
                    raise ValueError(";".join(errors))
                candidates = [c for c in candidates if c.get("review_id") != row["review_id"]]
                candidates.append(candidate)
                pd.DataFrame(candidates).to_csv(output_path, index=False)
                break
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
        else:
            failures.append(
                {
                    "review_id": row["review_id"],
                    "document": row["document"],
                    "v13_row_index": row["v13_row_index"],
                    "error": last_error,
                }
            )
            pd.DataFrame(failures, columns=GENERATION_FAILURE_COLS).to_csv(failure_path, index=False)
    pd.DataFrame(candidates).to_csv(output_path, index=False)
    pd.DataFrame(failures, columns=GENERATION_FAILURE_COLS).to_csv(failure_path, index=False)
    print(f"Wrote {output_path.relative_to(ROOT)} ({len(candidates)} candidates)")
    if failures:
        raise RuntimeError(f"{len(failures)} candidate generations failed")


def review_prompt(rows: pd.DataFrame) -> dict[str, str]:
    review_rows = []
    for _, row in rows.iterrows():
        review_rows.append(
            {
                "review_id": row["review_id"],
                "question_stem": row["probe"],
                "target": row["target"],
                "correct_label": row["correct_label"],
                "options": {label: row[col] for label, col in zip(LABELS, OPTION_COLS, strict=True)},
                "source_question": row["question"],
                "answer": row["answer"],
                "source_facts": row.get("source_facts", ""),
                "derivation": row.get("derivation", ""),
            }
        )
    return {
        "system": "You audit medical inference MCQA probes. Return JSON only.",
        "user": (
            "Review each candidate. Omit acceptable rows. Flag a row as fix only if a small rewrite can "
            "make it valid and natural. Flag reject if five plausible same-type options cannot be supported.\n\n"
            "A valid row has a natural standalone stem, exactly five options, exactly one correct option, "
            "no answer leakage in the stem, medically plausible same-type distractors, and the same intended answer.\n\n"
            "Return JSON object {\"issues\": [...]}. Each issue must have review_id, decision ('fix' or 'reject'), "
            "issue_categories, review_reason. For fix include fixed_question_stem, fixed_option_a, fixed_option_b, "
            "fixed_option_c, fixed_option_d, fixed_option_e, fixed_correct_label, fixed_target.\n\n"
            f"Rows:\n{json.dumps(review_rows, ensure_ascii=False, indent=2)}"
        ),
    }


def run_review(args: argparse.Namespace) -> None:
    candidates = pd.read_csv(args.report_dir / "generated_candidates.csv", keep_default_na=False)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    raw = utils.query_llm(
        review_prompt(candidates),
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        system_prompt_included=True,
        return_json=True,
        max_tokens=args.review_max_tokens,
        max_try_num=args.max_try_num,
    )
    parsed = parse_json_response(raw)
    issues = parsed.get("issues", [])
    if not isinstance(issues, list):
        raise ValueError("review issues was not a list")
    allowed = set(candidates["review_id"].astype(str))
    records: list[dict[str, object]] = []
    for issue in issues:
        rid = str(issue.get("review_id", ""))
        if rid not in allowed:
            raise ValueError(f"unknown review_id from review: {rid}")
        records.append({col: issue.get(col, "") for col in ISSUE_COLS})
        records[-1]["decision"] = str(records[-1]["decision"]).strip().lower()
    pd.DataFrame(records, columns=ISSUE_COLS).to_csv(args.report_dir / "agent_issue_rows.csv", index=False)
    combine_review(args.report_dir)


def apply_fix(base: pd.Series, issue: pd.Series) -> dict[str, object]:
    options = [normalize_space(issue[f"fixed_{col}"]) for col in OPTION_COLS]
    label = normalize_label(issue["fixed_correct_label"])
    target = normalize_space(issue["fixed_target"]) or options[LABELS.index(label)]
    source = base.copy()
    return mcqa_from_parts(source, str(issue["fixed_question_stem"]), target, label, options)


def combine_review(report_dir: Path) -> None:
    candidates = pd.read_csv(report_dir / "generated_candidates.csv", keep_default_na=False)
    issues_path = report_dir / "agent_issue_rows.csv"
    issues = pd.read_csv(issues_path, keep_default_na=False) if issues_path.exists() else pd.DataFrame(columns=ISSUE_COLS)
    if not issues.empty and issues["review_id"].duplicated().any():
        dupes = sorted(issues.loc[issues["review_id"].duplicated(), "review_id"].unique())
        raise ValueError(f"duplicate review ids: {dupes}")
    issue_map = {str(row.review_id): row._asdict() for row in issues.itertuples(index=False)}
    decisions: list[dict[str, object]] = []
    accepted_rows: list[dict[str, object]] = []
    rejected_rows: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []
    for _, candidate in candidates.iterrows():
        issue = issue_map.get(str(candidate["review_id"]), {})
        decision = str(issue.get("decision", "accept")).strip().lower() or "accept"
        if decision == "accept":
            final = candidate.to_dict()
        elif decision == "fix":
            final = apply_fix(candidate, pd.Series(issue))
        elif decision == "reject":
            final = {}
        else:
            raise ValueError(f"{candidate['review_id']}: invalid decision {decision}")

        record = candidate.to_dict()
        record.update({col: issue.get(col, "") for col in ISSUE_COLS if col != "review_id"})
        record["decision"] = decision
        if final:
            errors = validate_candidate_record(pd.Series(final))
            if errors:
                validation_rows.append(
                    {
                        "review_id": candidate["review_id"],
                        "document": candidate["document"],
                        "v13_row_index": candidate["v13_row_index"],
                        "errors": ";".join(errors),
                        "probe": final.get("probe", ""),
                        "target": final.get("target", ""),
                    }
                )
            accepted_rows.append(final)
        else:
            rejected_rows.append(record)
        decisions.append(record)

    pd.DataFrame(decisions).to_csv(report_dir / "agent_decisions.csv", index=False)
    pd.DataFrame(accepted_rows).to_csv(report_dir / "accepted_recovered_mcqa.csv", index=False)
    pd.DataFrame(rejected_rows, columns=list(candidates.columns) + ISSUE_COLS).to_csv(
        report_dir / "rejected_recovered_mcqa.csv", index=False
    )
    pd.DataFrame(validation_rows, columns=VALIDATION_COLS).to_csv(report_dir / "validation_failures.csv", index=False)
    summary = pd.DataFrame(decisions).groupby(["document", "decision"]).size().unstack(fill_value=0).reset_index()
    for col in ("accept", "fix", "reject"):
        if col not in summary.columns:
            summary[col] = 0
    summary["reviewed_rows"] = summary[["accept", "fix", "reject"]].sum(axis=1)
    summary.to_csv(report_dir / "summary.csv", index=False)
    if validation_rows:
        raise AssertionError(f"{len(validation_rows)} accepted/fixed rows failed validation")
    print(f"Wrote {str((report_dir / 'agent_decisions.csv').relative_to(ROOT))}")


def append_accepted(args: argparse.Namespace) -> None:
    accepted = pd.read_csv(args.report_dir / "accepted_recovered_mcqa.csv", keep_default_na=False)
    if accepted.empty:
        print("No accepted rows to append.")
        return
    backup_dir = args.report_dir / "canonical_backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    append_summary: list[dict[str, object]] = []
    for document, rows in accepted.groupby("document", sort=True):
        path = MEDICAL_DIR / document / "inference" / "probes_v13_mcqa.csv"
        current = pd.read_csv(path, keep_default_na=False)
        backup_path = backup_dir / document / "inference" / "probes_v13_mcqa.csv"
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        if not backup_path.exists():
            shutil.copy2(path, backup_path)
        existing_keys = set(zip(current["question"].astype(str), current["answer"].astype(str)))
        to_append = []
        for _, row in rows.iterrows():
            key = (str(row["question"]), str(row["answer"]))
            if key not in existing_keys:
                to_append.append({col: row.get(col, "") for col in MCQA_COLS})
        combined = pd.concat([current, pd.DataFrame(to_append, columns=MCQA_COLS)], ignore_index=True)
        combined.to_csv(path, index=False)
        append_summary.append(
            {
                "document": document,
                "before_rows": len(current),
                "appended_rows": len(to_append),
                "after_rows": len(combined),
                "path": str(path.relative_to(ROOT)),
                "backup_path": str(backup_path.relative_to(ROOT)),
            }
        )
    pd.DataFrame(append_summary).to_csv(args.report_dir / "append_summary.csv", index=False)
    print(f"Wrote {(args.report_dir / 'append_summary.csv').relative_to(ROOT)}")


def validate_appended(args: argparse.Namespace) -> None:
    missing_after = discover_missing_rows()
    records = []
    total_mcqa = 0
    for path in sorted(MEDICAL_DIR.glob("*/inference/probes_v13_mcqa.csv")):
        df = pd.read_csv(path, keep_default_na=False)
        total_mcqa += len(df)
        for row_index, row in df.iterrows():
            errors = validate_candidate_record(row)
            if errors:
                records.append(
                    {
                        "review_id": f"{path.parts[-3]}/{row_index}",
                        "document": path.parts[-3],
                        "v13_row_index": row_index,
                        "errors": ";".join(errors),
                        "probe": row.get("probe", ""),
                        "target": row.get("target", ""),
                    }
                )
    pd.DataFrame(records, columns=VALIDATION_COLS).to_csv(args.report_dir / "post_append_validation_failures.csv", index=False)
    pd.DataFrame(
        [
            {
                "medical_v13_mcqa_rows": total_mcqa,
                "medical_missing_mcqa_after_append": len(missing_after),
            }
        ]
    ).to_csv(args.report_dir / "post_append_summary.csv", index=False)
    if records:
        raise AssertionError(f"{len(records)} medical v13 MCQA rows failed validation")
    print(f"Medical v13 MCQA rows: {total_mcqa}; missing after append: {len(missing_after)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("generate", "review", "combine-review", "append", "validate", "all"))
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--max-tokens", type=int, default=3000)
    parser.add_argument("--review-max-tokens", type=int, default=12000)
    parser.add_argument("--max-try-num", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command in {"generate", "all"}:
        generate_candidates(args)
    if args.command in {"review", "all"}:
        run_review(args)
    if args.command == "combine-review":
        combine_review(args.report_dir)
    if args.command in {"append", "all"}:
        append_accepted(args)
    if args.command in {"validate", "all"}:
        validate_appended(args)


if __name__ == "__main__":
    csv.field_size_limit(10_000_000)
    main()
