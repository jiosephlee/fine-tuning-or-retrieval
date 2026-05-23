#!/usr/bin/env python3
"""Build inference MCQA v14 from existing v13 MCQA rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
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


REPO_ROOT = find_repo_root()
sys.path.append(str(REPO_ROOT))

from utils import utils  # noqa: E402
from utils.mcqa_prompts import build_mcqa_5shot_prompt  # noqa: E402


PROBE_ROOT = REPO_ROOT / "probes"
CONTEXT_REPORT_DIR = (
    REPO_ROOT / "reports" / "inference_mcqa_v14_contextualized_questions_gpt54mini_20260523"
)
REVIEW_REPORT_DIR = (
    REPO_ROOT / "reports" / "inference_mcqa_v14_naturalness_review_gpt54mini_20260523"
)
PROMPT_VERSION = "fact_probe_v1_style_gpt54mini_20260523"
DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_REASONING_EFFORT = "medium"
DOMAIN_GROUPS = ("arxiv", "legal", "medical")
LABELS = ("(A)", "(B)", "(C)", "(D)", "(E)")
OPTION_COLS = ("option_a", "option_b", "option_c", "option_d", "option_e")
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
ISSUE_COLS = (
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
)
VALIDATION_COLS = (
    "domain_group",
    "document",
    "row_index",
    "errors",
    "probe",
    "target",
)
DROPPED_COLS = (
    "review_id",
    "domain_group",
    "document",
    "mcqa_row_index",
    "v13_source_row_index",
    "v13_original_row_index",
    "review_reason",
    "issue_categories",
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


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def parse_json_response(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    text = str(value).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM did not return valid JSON: {text[:500]}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"LLM JSON was not an object: {type(parsed).__name__}")
    return parsed


def parse_json_list(value: object) -> list[Any]:
    if isinstance(value, list):
        return value
    text = str(value).strip()
    if not text:
        return []
    parsed = json.loads(text)
    if not isinstance(parsed, list):
        raise ValueError("expected JSON list")
    return parsed


def stable_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def discover_v13_mcqa_paths() -> list[Path]:
    paths: list[Path] = []
    for group in DOMAIN_GROUPS:
        paths.extend((PROBE_ROOT / group).glob("*/inference/probes_v13_mcqa.csv"))
    return sorted(paths)


def canonical_probe_dir(domain: str, document: str) -> Path:
    return PROBE_ROOT / domain / document / "inference"


def staged_probe_dir(context_report_dir: Path, domain: str, document: str) -> Path:
    return context_report_dir / "staged_probes" / domain / document / "inference"


def review_id(domain: str, document: str, row_index: int) -> str:
    return f"{domain}/{document}/{row_index}"


def extract_document_label(row: pd.Series) -> str:
    text_parts = [
        str(row.get(col, ""))
        for col in (
            "question",
            "probe",
            "fact",
            "source_facts",
            "source_fact(s)",
            "text_sentences",
        )
    ]
    text = "\n".join(part for part in text_parts if part)
    for pattern in (r'paper\s+"([^"]+)"', r"paper\s+'([^']+)'", r'case report\s+"([^"]+)"', r"case report\s+'([^']+)'"):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return normalize_space(match.group(1))
    return normalize_space(str(row["document"]).replace("_", " "))


def domain_context_name(domain_group: str) -> str:
    if domain_group == "legal":
        return "legal opinion"
    if domain_group == "medical":
        return "medical case report"
    return "academic paper"


def collect_v13_mcqa_rows() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for path in discover_v13_mcqa_paths():
        domain, document = path.relative_to(PROBE_ROOT).parts[:2]
        mcqa = pd.read_csv(path, keep_default_na=False)
        source = pd.read_csv(path.with_name("probes_v13.csv"), keep_default_na=False)
        for row_index, row in mcqa.iterrows():
            mask = (
                (source["question"].astype(str) == str(row["question"]))
                & (source["answer"].astype(str) == str(row["answer"]))
            )
            if int(mask.sum()) != 1:
                raise ValueError(
                    f"{path.relative_to(REPO_ROOT)} row {row_index}: expected one v13 source match, "
                    f"found {int(mask.sum())}"
                )
            source_row = source.loc[mask].iloc[0]
            record = row.to_dict()
            record.update(
                {
                    "review_id": review_id(domain, document, int(row_index)),
                    "domain_group": domain,
                    "document": document,
                    "mcqa_row_index": int(row_index),
                    "v13_original_row_index": source_row.get("original_row_index", ""),
                    "v13_source_row_index": int(source_row.name),
                    "v13_mcqa_path": str(path.relative_to(REPO_ROOT)),
                    "v13_path": str(path.with_name("probes_v13.csv").relative_to(REPO_ROOT)),
                }
            )
            records.append(record)
    return pd.DataFrame(records).sort_values(["domain_group", "document", "mcqa_row_index"])


def contextualization_payload(row: pd.Series) -> dict[str, object]:
    return {
        "prompt_version": PROMPT_VERSION,
        "document": row["document"],
        "document_label": extract_document_label(row),
        "document_type": domain_context_name(str(row["domain_group"])),
        "domain_group": row["domain_group"],
        "question": row.get("question", ""),
        "answer": row.get("answer", ""),
        "target": row.get("target", ""),
        "fact": row.get("fact", ""),
        "source_facts": row.get("source_facts", row.get("source_fact(s)", "")),
        "text_sentences": row.get("text_sentences", ""),
        "derivation": row.get("derivation", ""),
    }


def build_contextualization_prompt(row: pd.Series) -> dict[str, str]:
    payload = contextualization_payload(row)
    document_type = payload["document_type"]
    document_label = payload["document_label"]
    if (
        normalize_answer(payload.get("target", "")) in normalize_answer(document_label)
        or normalize_answer(payload.get("answer", "")) in normalize_answer(document_label)
    ):
        document_reference = f"this {document_type}"
    else:
        document_reference = f'the {document_type} "{document_label}"'
    return {
        "system": (
            "You turn inference probe questions into self-contained, precise MCQA stems. "
            "All information in the rewritten question must be grounded in the provided inputs. "
            "Return JSON only."
        ),
        "user": (
            "Rewrite the existing inference question into one natural, standalone, document-aware "
            "question for a five-choice MCQA probe.\n\n"
            "Use the style of the factual probe contextualizer:\n"
            f"- Start with a clear context phrase such as `In {document_reference}, ...` "
            f"or `According to {document_reference}, ...` when it fits naturally.\n"
            "- If the document title contains the answer phrase, do not quote or paraphrase that part of the title.\n"
            "- Use only the supplied fact, source facts, text sentences, derivation, question, and answer. "
            "Do not add, infer, correct, or update facts from internal knowledge.\n"
            "- Add enough context that the question is self-contained and unambiguous without looking back "
            "at the source material.\n"
            "- Clarify pronouns, referential phrases, unnamed methods, experiment settings, legal/procedural "
            "posture, or clinical context when the inputs identify them.\n"
            "- Preserve the same intended answer exactly. Minor grammatical adjustments are allowed only if "
            "the answer itself already appears that way in the MCQA target.\n"
            "- Do not reveal or hint at the answer phrase in the question. The answer string and target string "
            "must not appear in the rewritten question.\n"
            "- Avoid quoting the source sentence directly; write a natural question instead.\n"
            "- Preserve LaTeX formatting exactly when mathematical notation appears.\n"
            "- Do not add answer choices.\n\n"
            "Return JSON with keys:\n"
            "- contextualized_question: the final rewritten question only\n"
            "- notes: short explanation of what context was added or why no extra context was needed\n\n"
            f"Input JSON:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
        ),
    }


def load_context_cache(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    df = pd.read_csv(path, keep_default_na=False)
    cache: dict[str, dict[str, str]] = {}
    for _, row in df.iterrows():
        key = str(row.get("input_hash", ""))
        if key:
            cache[key] = {
                "contextualized_question": str(row.get("contextualized_question", "")),
                "raw_response": str(row.get("raw_response", "")),
            }
    return cache


def contextualized_question_errors(question: str, answer: object, target: object) -> list[str]:
    errors: list[str] = []
    normalized_question = normalize_answer(question)
    for label, value in (("answer", answer), ("target", target)):
        normalized_value = normalize_answer(value)
        if normalized_value and normalized_value in normalized_question:
            errors.append(f"{label}_leaks_in_contextualized_question")
    if not normalize_space(question):
        errors.append("missing_contextualized_question")
    return errors


def generate_contextualized_questions(args: argparse.Namespace) -> None:
    args.context_report_dir.mkdir(parents=True, exist_ok=True)
    source = collect_v13_mcqa_rows()
    source.to_csv(args.context_report_dir / "source_rows.csv", index=False)
    cache_path = args.context_report_dir / "contextualized_questions.csv"
    cache = load_context_cache(cache_path)
    output_records: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    for _, row in source.iterrows():
        payload = contextualization_payload(row)
        input_hash = stable_hash(
            {
                "prompt_version": PROMPT_VERSION,
                "model": args.model,
                "reasoning_effort": args.reasoning_effort,
                "payload": payload,
            }
        )
        cached = cache.get(input_hash)
        if cached and cached["contextualized_question"].strip() and not contextualized_question_errors(
            cached["contextualized_question"],
            row.get("answer", ""),
            row.get("target", ""),
        ):
            contextualized_question = cached["contextualized_question"]
            raw_response = cached["raw_response"]
        else:
            prompt = build_contextualization_prompt(row)
            try:
                raw = utils.query_llm(
                    prompt,
                    model=args.model,
                    reasoning_effort=args.reasoning_effort,
                    system_prompt_included=True,
                    return_json=True,
                    max_tokens=args.max_tokens,
                    max_try_num=args.max_try_num,
                )
                parsed = parse_json_response(raw)
                contextualized_question = normalize_space(parsed.get("contextualized_question", ""))
                errors = contextualized_question_errors(
                    contextualized_question,
                    row.get("answer", ""),
                    row.get("target", ""),
                )
                if errors:
                    raise ValueError(";".join(errors))
                raw_response = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    {
                        "review_id": row["review_id"],
                        "input_hash": input_hash,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                if not args.keep_going:
                    break
                continue
        output_records.append(
            {
                "review_id": row["review_id"],
                "domain_group": row["domain_group"],
                "document": row["document"],
                "mcqa_row_index": row["mcqa_row_index"],
                "v13_original_row_index": row["v13_original_row_index"],
                "v13_source_row_index": row["v13_source_row_index"],
                "input_hash": input_hash,
                "prompt_version": PROMPT_VERSION,
                "model": args.model,
                "reasoning_effort": args.reasoning_effort,
                "question": row.get("question", ""),
                "answer": row.get("answer", ""),
                "target": row.get("target", ""),
                "contextualized_question": contextualized_question,
                "raw_response": raw_response,
            }
        )
        pd.DataFrame(output_records).to_csv(cache_path, index=False)
        if failures:
            pd.DataFrame(failures).to_csv(args.context_report_dir / "failures.csv", index=False)

    pd.DataFrame(output_records).to_csv(cache_path, index=False)
    pd.DataFrame(failures).to_csv(args.context_report_dir / "failures.csv", index=False)
    print(f"Wrote {display_path(cache_path)} ({len(output_records)} rows)")
    if failures:
        raise RuntimeError(f"{len(failures)} contextualization rows failed")


def context_map(context_report_dir: Path) -> dict[str, str]:
    path = context_report_dir / "contextualized_questions.csv"
    df = pd.read_csv(path, keep_default_na=False)
    return dict(zip(df["review_id"].astype(str), df["contextualized_question"].astype(str)))


def build_draft(args: argparse.Namespace) -> None:
    contexts = context_map(args.context_report_dir)
    summary_records: list[dict[str, object]] = []
    for path in discover_v13_mcqa_paths():
        domain, document = path.relative_to(PROBE_ROOT).parts[:2]
        mcqa = pd.read_csv(path, keep_default_na=False)
        source = pd.read_csv(path.with_name("probes_v13.csv"), keep_default_na=False)
        v14_rows: list[pd.Series] = []
        mcqa_rows: list[pd.Series] = []
        for row_index, row in mcqa.iterrows():
            rid = review_id(domain, document, int(row_index))
            contextualized_question = normalize_space(contexts.get(rid, ""))
            if not contextualized_question:
                raise ValueError(f"Missing contextualized_question for {rid}")
            mask = (
                (source["question"].astype(str) == str(row["question"]))
                & (source["answer"].astype(str) == str(row["answer"]))
            )
            source_row = source.loc[mask].iloc[0].copy()
            source_row["contextualized_question"] = contextualized_question
            v14_rows.append(source_row)

            draft = row.copy()
            options = [str(draft[col]) for col in OPTION_COLS]
            draft["probe"] = contextualized_question
            draft["contextualized_question"] = contextualized_question
            draft["formatted_question"] = format_question(contextualized_question, options)
            draft["formatted_question_5shot"] = build_mcqa_5shot_prompt(draft["formatted_question"])
            draft["v13_mcqa_row_index"] = int(row_index)
            draft["v13_source_row_index"] = int(source_row.name)
            draft["v13_original_row_index"] = source_row.get("original_row_index", "")
            mcqa_rows.append(draft)

        v14 = pd.DataFrame(v14_rows)
        v14_mcqa = pd.DataFrame(mcqa_rows)
        out_dir = staged_probe_dir(args.context_report_dir, domain, document)
        out_dir.mkdir(parents=True, exist_ok=True)
        v14.to_csv(out_dir / "probes_v14.csv", index=False)
        v14_mcqa.to_csv(out_dir / "probes_v14_mcqa.csv", index=False)
        summary_records.append(
            {
                "domain_group": domain,
                "document": document,
                "source_rows": len(mcqa),
                "draft_rows": len(v14_mcqa),
                "v14_path": display_path(out_dir / "probes_v14.csv"),
                "v14_mcqa_path": display_path(out_dir / "probes_v14_mcqa.csv"),
            }
        )
    args.context_report_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_records).to_csv(args.context_report_dir / "draft_build_summary.csv", index=False)
    print(f"Wrote {display_path(args.context_report_dir / 'draft_build_summary.csv')}")


def collect_draft_rows(context_report_dir: Path) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for path in discover_v13_mcqa_paths():
        domain, document = path.relative_to(PROBE_ROOT).parts[:2]
        draft_path = staged_probe_dir(context_report_dir, domain, document) / "probes_v14_mcqa.csv"
        if not draft_path.exists():
            raise FileNotFoundError(draft_path)
        draft = pd.read_csv(draft_path, keep_default_na=False)
        for row_index, row in draft.iterrows():
            record = {
                "review_id": review_id(domain, document, int(row_index)),
                "domain_group": domain,
                "document": document,
                "mcqa_row_index": int(row_index),
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
                "contextualized_question": row.get("contextualized_question", ""),
                "fact": row.get("fact", ""),
                "source_facts": row.get("source_facts", row.get("source_fact(s)", "")),
                "text_sentences": row.get("text_sentences", ""),
                "derivation": row.get("derivation", ""),
                "question": row.get("question", ""),
                "answer": row.get("answer", ""),
            }
            records.append(record)
    return pd.DataFrame(records).sort_values(["domain_group", "document", "mcqa_row_index"])


def build_review_prompt(rows: pd.DataFrame) -> dict[str, str]:
    review_rows = []
    for _, row in rows.iterrows():
        review_rows.append(
            {
                "review_id": row["review_id"],
                "question_stem": row["question_stem"],
                "target": row["target"],
                "correct_label": row["correct_label"],
                "options": {label: row[col] for label, col in zip(LABELS, OPTION_COLS)},
                "source_question": row["question"],
                "answer": row["answer"],
                "derivation": row["derivation"],
            }
        )
    return {
        "system": (
            "You are auditing MCQA inference probes for naturalness and validity. "
            "Return JSON only."
        ),
        "user": (
            "Review each row. Omit rows that are already acceptable.\n"
            "Flag a row as fix only when a small rewrite can make the stem/options natural and valid. "
            "Flag a row as reject only when it cannot be repaired without inventing unsupported content.\n\n"
            "A valid fixed row must have exactly five options, one correct answer, no answer leakage in the "
            "question stem, and natural fit between the stem and all answer options. Keep the same intended "
            "answer unless a minor surface cleanup is required.\n\n"
            "Return JSON object: {\"issues\": [ ... ]}. Each issue must contain review_id, decision "
            "('fix' or 'reject'), issue_categories, review_reason. For fix also include "
            "fixed_question_stem, fixed_option_a, fixed_option_b, fixed_option_c, fixed_option_d, "
            "fixed_option_e, fixed_correct_label, fixed_target.\n\n"
            f"Rows:\n{json.dumps(review_rows, ensure_ascii=False, indent=2)}"
        ),
    }


def validate_decision_row(row: pd.Series) -> tuple[bool, str]:
    decision = str(row["decision"]).strip().lower()
    if decision not in {"accept", "fix", "reject"}:
        return False, "invalid decision"
    if decision != "fix":
        return True, ""
    stem = normalize_space(row.get("fixed_question_stem", ""))
    options = [normalize_space(row.get(f"fixed_{col}", "")) for col in OPTION_COLS]
    label = normalize_label(row.get("fixed_correct_label", "")) or normalize_label(
        row.get("correct_label", "")
    )
    if not stem:
        return False, "fix missing fixed_question_stem"
    if any(not option for option in options):
        return False, "fix missing fixed option"
    if label not in LABELS:
        return False, "fix invalid fixed_correct_label"
    target = normalize_space(row.get("fixed_target", "")) or options[LABELS.index(label)]
    if options[LABELS.index(label)] != target:
        return False, "fix correct option does not equal fixed_target"
    if sum(normalize_answer(option) == normalize_answer(target) for option in options) != 1:
        return False, "fix fixed_target is not unique among options"
    if normalize_answer(target) in normalize_answer(stem):
        return False, "fix leaks target in stem"
    return True, ""


def run_review(args: argparse.Namespace) -> None:
    args.review_report_dir.mkdir(parents=True, exist_ok=True)
    review_input = collect_draft_rows(args.context_report_dir)
    review_input.to_csv(args.review_report_dir / "review_input.csv", index=False)
    issue_records: list[dict[str, object]] = []
    for batch_index, start in enumerate(range(0, len(review_input), args.batch_size), start=1):
        batch = review_input.iloc[start : start + args.batch_size]
        out_path = args.review_report_dir / f"agent_issue_rows_batch_{batch_index:02d}.csv"
        if out_path.exists() and not args.force:
            existing = pd.read_csv(out_path, keep_default_na=False)
            issue_records.extend(existing.to_dict("records"))
            continue
        raw = utils.query_llm(
            build_review_prompt(batch),
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            system_prompt_included=True,
            return_json=True,
            max_tokens=args.max_tokens,
            max_try_num=args.max_try_num,
        )
        parsed = parse_json_response(raw)
        issues = parsed.get("issues", [])
        if not isinstance(issues, list):
            raise ValueError(f"Batch {batch_index}: issues was not a list")
        normalized: list[dict[str, object]] = []
        allowed_ids = set(batch["review_id"].astype(str))
        for issue in issues:
            if not isinstance(issue, dict):
                raise ValueError(f"Batch {batch_index}: issue was not an object")
            rid = str(issue.get("review_id", ""))
            if rid not in allowed_ids:
                raise ValueError(f"Batch {batch_index}: unknown review_id {rid!r}")
            normalized.append(
                {
                    "review_id": rid,
                    "decision": str(issue.get("decision", "")).strip().lower(),
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
        pd.DataFrame(normalized, columns=ISSUE_COLS).to_csv(out_path, index=False)
        issue_records.extend(normalized)
        print(f"Wrote {display_path(out_path)} ({len(normalized)} issues)")
    combine_review(args.review_report_dir)


def read_issue_rows(review_report_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(review_report_dir.glob("agent_issue_rows_batch_*.csv")):
        if path.stat().st_size == 0:
            continue
        try:
            df = pd.read_csv(path, keep_default_na=False)
        except pd.errors.EmptyDataError:
            continue
        if not df.empty:
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def combine_review(review_report_dir: Path) -> None:
    review_input = pd.read_csv(review_report_dir / "review_input.csv", keep_default_na=False)
    issues = read_issue_rows(review_report_dir)
    if not issues.empty and issues["review_id"].duplicated().any():
        dupes = sorted(issues.loc[issues["review_id"].duplicated(), "review_id"].unique())
        raise ValueError(f"Duplicate issue review_id values: {dupes[:10]}")
    issue_map = {str(row.review_id): row._asdict() for row in issues.itertuples(index=False)}
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
        valid, error = validate_decision_row(pd.Series(record))
        record["validation_error"] = error
        if decision == "fix" and valid:
            options = [normalize_space(record[f"fixed_{col}"]) for col in OPTION_COLS]
            label = normalize_label(record["fixed_correct_label"]) or normalize_label(
                record["correct_label"]
            )
            target = normalize_space(record["fixed_target"]) or options[LABELS.index(label)]
            record["fixed_correct_label"] = label
            record["fixed_target"] = target
            formatted = format_question(record["fixed_question_stem"], options)
            record["fixed_formatted_question"] = formatted
            record["fixed_formatted_question_5shot"] = build_mcqa_5shot_prompt(formatted)
            record["fixed_distractors"] = json.dumps(
                [option for idx, option in enumerate(options) if idx != LABELS.index(label)],
                ensure_ascii=False,
            )
        else:
            record["fixed_formatted_question"] = ""
            record["fixed_formatted_question_5shot"] = ""
            record["fixed_distractors"] = ""
        rows.append(record)

    decisions = pd.DataFrame(rows)
    decisions.to_csv(review_report_dir / "agent_decisions.csv", index=False)
    decisions[decisions["decision"] == "accept"].to_csv(
        review_report_dir / "accepted_rows.csv", index=False
    )
    decisions[decisions["decision"] == "fix"].to_csv(review_report_dir / "fixed_rows.csv", index=False)
    decisions[decisions["decision"] == "reject"].to_csv(
        review_report_dir / "rejected_rows.csv", index=False
    )
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
    summary.to_csv(review_report_dir / "summary.csv", index=False)
    invalid = decisions[decisions["validation_error"].astype(str) != ""]
    if not invalid.empty:
        raise AssertionError(f"{len(invalid)} inference review decisions failed validation")
    print(f"Wrote {display_path(review_report_dir / 'agent_decisions.csv')}")


def apply_fix(base: pd.Series, decision: pd.Series) -> pd.Series:
    fixed = base.copy()
    options = [normalize_space(decision[f"fixed_{col}"]) for col in OPTION_COLS]
    label = normalize_label(decision["fixed_correct_label"]) or normalize_label(base["correct_label"])
    target = normalize_space(decision["fixed_target"]) or options[LABELS.index(label)]
    stem = normalize_space(decision["fixed_question_stem"])
    formatted = format_question(stem, options)
    fixed["probe"] = stem
    fixed["contextualized_question"] = stem
    fixed["target"] = target
    fixed["correct_label"] = label
    fixed["formatted_question"] = formatted
    for col, value in zip(OPTION_COLS, options):
        fixed[col] = value
    fixed["distractors"] = json.dumps(
        [option for idx, option in enumerate(options) if idx != LABELS.index(label)],
        ensure_ascii=False,
    )
    fixed["formatted_question_5shot"] = build_mcqa_5shot_prompt(formatted)
    return fixed


def validate_mcqa_row(row: pd.Series) -> list[str]:
    errors: list[str] = []
    label = normalize_label(row.get("correct_label", ""))
    if label not in LABELS:
        errors.append("bad_correct_label")
        return errors
    options = [str(row[col]) for col in OPTION_COLS]
    target = str(row.get("target", ""))
    if options[LABELS.index(label)] != target:
        errors.append("correct_option_not_target")
    if sum(normalize_answer(option) == normalize_answer(target) for option in options) != 1:
        errors.append("target_not_unique_in_options")
    if normalize_answer(target) and normalize_answer(target) in normalize_answer(row.get("probe", "")):
        errors.append("target_leaks_in_stem")
    for col in ("contextualized_question", "formatted_question", "formatted_question_5shot"):
        if not str(row.get(col, "")).strip():
            errors.append(f"missing_{col}")
    return errors


def apply_final(args: argparse.Namespace) -> None:
    decisions = pd.read_csv(args.review_report_dir / "agent_decisions.csv", keep_default_na=False)
    decision_map = {
        review_id(str(row["domain_group"]), str(row["document"]), int(row["mcqa_row_index"])): row
        for _, row in decisions.iterrows()
    }
    dropped_records: list[dict[str, object]] = []
    summary_records: list[dict[str, object]] = []
    validation_records: list[dict[str, object]] = []
    final_outputs: list[tuple[str, str, pd.DataFrame, pd.DataFrame]] = []

    for path in discover_v13_mcqa_paths():
        domain, document = path.relative_to(PROBE_ROOT).parts[:2]
        stage_dir = staged_probe_dir(args.context_report_dir, domain, document)
        draft_path = stage_dir / "probes_v14_mcqa.csv"
        source_path = stage_dir / "probes_v14.csv"
        draft = pd.read_csv(draft_path, keep_default_na=False)
        source = pd.read_csv(source_path, keep_default_na=False)
        final_mcqa_rows: list[pd.Series] = []
        keep_source_indices: list[int] = []
        fixed_count = 0
        reject_count = 0

        for row_index, row in draft.iterrows():
            rid = review_id(domain, document, int(row_index))
            decision = decision_map.get(rid)
            if decision is None:
                raise ValueError(f"Missing decision for {rid}")
            action = str(decision["decision"]).strip().lower()
            if action == "accept":
                final_mcqa_rows.append(row.copy())
                keep_source_indices.append(int(row_index))
            elif action == "fix":
                fixed_count += 1
                fixed = apply_fix(row, decision)
                final_mcqa_rows.append(fixed)
                keep_source_indices.append(int(row_index))
                source.loc[int(row_index), "contextualized_question"] = fixed["contextualized_question"]
                source.loc[int(row_index), "target"] = fixed["target"]
            elif action == "reject":
                reject_count += 1
                rec = row.to_dict()
                rec.update(
                    {
                        "review_id": rid,
                        "domain_group": domain,
                        "document": document,
                        "mcqa_row_index": int(row_index),
                        "v13_source_row_index": row.get("v13_source_row_index", ""),
                        "v13_original_row_index": row.get("v13_original_row_index", ""),
                        "review_reason": decision.get("review_reason", ""),
                        "issue_categories": decision.get("issue_categories", ""),
                    }
                )
                dropped_records.append(rec)
            else:
                raise ValueError(f"{rid}: invalid decision {action!r}")

        final_mcqa = pd.DataFrame(final_mcqa_rows, columns=draft.columns)
        final_source = source.iloc[keep_source_indices].reset_index(drop=True)
        for row_index, row in final_mcqa.iterrows():
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
        summary_records.append(
            {
                "domain_group": domain,
                "document": document,
                "source_rows": len(draft),
                "accepted_rows": int(
                    ((decisions["domain_group"] == domain) & (decisions["document"] == document) & (decisions["decision"] == "accept")).sum()
                ),
                "fixed_rows": fixed_count,
                "rejected_rows": reject_count,
                "final_rows": len(final_mcqa),
            }
        )
        final_outputs.append((domain, document, final_source, final_mcqa))

    pd.DataFrame(validation_records, columns=VALIDATION_COLS).to_csv(
        args.review_report_dir / "final_validation.csv", index=False
    )
    if validation_records:
        raise AssertionError(f"{len(validation_records)} final inference rows failed validation")
    for domain, document, final_source, final_mcqa in final_outputs:
        stage_dir = staged_probe_dir(args.context_report_dir, domain, document)
        stage_dir.mkdir(parents=True, exist_ok=True)
        final_source.to_csv(stage_dir / "probes_v14.csv", index=False)
        final_mcqa.to_csv(stage_dir / "probes_v14_mcqa.csv", index=False)

        canonical_dir = canonical_probe_dir(domain, document)
        backup_dir = args.review_report_dir / "canonical_backups" / domain / document / "inference"
        backup_dir.mkdir(parents=True, exist_ok=True)
        for filename in ("probes_v14.csv", "probes_v14_mcqa.csv"):
            canonical_path = canonical_dir / filename
            if canonical_path.exists():
                shutil.copy2(canonical_path, backup_dir / filename)
        final_source.to_csv(canonical_dir / "probes_v14.csv", index=False)
        final_mcqa.to_csv(canonical_dir / "probes_v14_mcqa.csv", index=False)

    if dropped_records:
        dropped = pd.DataFrame(dropped_records)
    else:
        dropped = pd.DataFrame(columns=DROPPED_COLS)
    dropped.to_csv(args.review_report_dir / "dropped_row_mappings.csv", index=False)
    pd.DataFrame(summary_records).to_csv(args.review_report_dir / "final_summary.csv", index=False)
    print(f"Wrote {display_path(args.review_report_dir / 'final_summary.csv')}")


def validate_outputs(args: argparse.Namespace) -> None:
    source = collect_v13_mcqa_rows()
    if len(source) == 0:
        raise AssertionError("No v13 MCQA rows found")
    decisions = pd.read_csv(args.review_report_dir / "agent_decisions.csv", keep_default_na=False)
    rejects = int((decisions["decision"] == "reject").sum())
    expected_final_rows = len(decisions) - rejects
    final_rows = 0
    validation_records: list[dict[str, object]] = []
    summary_records: list[dict[str, object]] = []
    for path in discover_v13_mcqa_paths():
        domain, document = path.relative_to(PROBE_ROOT).parts[:2]
        mcqa = pd.read_csv(path.with_name("probes_v14_mcqa.csv"), keep_default_na=False)
        probes = pd.read_csv(path.with_name("probes_v14.csv"), keep_default_na=False)
        final_rows += len(mcqa)
        if len(mcqa) != len(probes):
            validation_records.append(
                {
                    "domain_group": domain,
                    "document": document,
                    "row_index": "",
                    "errors": "v14_mcqa_v14_row_count_mismatch",
                }
            )
        for row_index, row in mcqa.iterrows():
            errors = validate_mcqa_row(row)
            if errors:
                validation_records.append(
                    {
                        "domain_group": domain,
                        "document": document,
                        "row_index": int(row_index),
                        "errors": ";".join(errors),
                    }
                )
        doc_decisions = decisions[
            (decisions["domain_group"] == domain) & (decisions["document"] == document)
        ]
        summary_records.append(
            {
                "domain_group": domain,
                "document": document,
                "source_rows": len(doc_decisions),
                "accepted_rows": int((doc_decisions["decision"] == "accept").sum()),
                "fixed_rows": int((doc_decisions["decision"] == "fix").sum()),
                "rejected_rows": int((doc_decisions["decision"] == "reject").sum()),
                "final_rows": len(mcqa),
            }
        )
    if final_rows != expected_final_rows:
        raise AssertionError(f"Expected {expected_final_rows} final rows, found {final_rows}")
    out_dir = args.review_report_dir
    pd.DataFrame(summary_records).to_csv(out_dir / "validation_summary.csv", index=False)
    pd.DataFrame(validation_records, columns=VALIDATION_COLS).to_csv(
        out_dir / "validation_failures.csv", index=False
    )
    if validation_records:
        raise AssertionError(f"{len(validation_records)} validation failures")
    print(f"Wrote {display_path(out_dir / 'validation_summary.csv')}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("contextualize", "draft", "review", "combine-review", "apply-final", "validate", "all"),
    )
    parser.add_argument("--context-report-dir", type=Path, default=CONTEXT_REPORT_DIR)
    parser.add_argument("--review-report-dir", type=Path, default=REVIEW_REPORT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--max-try-num", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command in {"contextualize", "all"}:
        generate_contextualized_questions(args)
    if args.command in {"draft", "all"}:
        build_draft(args)
    if args.command in {"review", "all"}:
        run_review(args)
    if args.command == "combine-review":
        combine_review(args.review_report_dir)
    if args.command in {"apply-final", "all"}:
        apply_final(args)
    if args.command in {"validate", "all"}:
        validate_outputs(args)


if __name__ == "__main__":
    csv.field_size_limit(10_000_000)
    main()
