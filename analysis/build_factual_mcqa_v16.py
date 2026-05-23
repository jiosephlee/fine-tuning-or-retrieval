#!/usr/bin/env python3
"""Build factual v16 MCQA by judging each v15 distractor and regenerating any
that are effectively the correct answer (semantically equivalent / also valid).

Conventions are aligned with ``scripts/data-preparation/probes/pipeline_mcqa_difficulty.py``
(compact JSON-key prompts; gpt-5.4 for distractor generation, gpt-5.4-mini
for judging) so v16 looks like a natural extension of the same pipeline.

Pipeline per row:
1. For each of the 4 distractors, ask a judge LLM (default gpt-5.4-mini)
   whether it would also be correct given the probe + marked correct answer
   + source context. Returns ``{"e":true,"r":"..."}`` if equivalent.
2. For every flagged distractor, ask a generator LLM (default gpt-5.4) for
   one replacement distractor that is plausible but unambiguously wrong and
   not a paraphrase of the correct answer or any existing distractor.
   Re-judge. Repeat up to ``--max-regen-attempts`` times.
3. If no acceptable replacement is produced, keep the original distractor
   and record the failure in the audit log.

Outputs alongside each input ``<dir>/probes_v15_mcqa.csv``:
  - ``<dir>/probes_v16_mcqa.csv``            -- same schema as v15
  - ``<dir>/probes_v16_mcqa.audit.csv``      -- per-distractor verdicts/edits
  - ``<dir>/probes_v16_mcqa_metrics.txt``    -- summary counts

Plus a top-level summary at ``reports/factual_mcqa_v16/summary.csv``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

import importlib.util  # noqa: E402

import utils.utils as utils  # noqa: E402
from utils.mcqa_prompts import build_mcqa_5shot_prompt  # noqa: E402

# Reuse the existing context helper so v16 sees the same source window the
# original generator did when columns are available. The directory uses a
# hyphen, so load by file path rather than dotted import.
_MCQA_PIPE_PATH = REPO_ROOT / "scripts" / "data-preparation" / "probes" / "pipeline_mcqa_difficulty.py"
_spec = importlib.util.spec_from_file_location("pipeline_mcqa_difficulty", _MCQA_PIPE_PATH)
mcqa_pipe = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(mcqa_pipe)  # type: ignore[union-attr]

PROBE_ROOT = REPO_ROOT / "probes"
REPORT_DIR = REPO_ROOT / "reports" / "factual_mcqa_v16"

DOMAIN_GROUPS = ("medical", "legal", "arxiv")
OPTION_COLS = ("option_a", "option_b", "option_c", "option_d", "option_e")
LABELS = ("(A)", "(B)", "(C)", "(D)", "(E)")

# Compact judge prompt, matching the terse style of ``VERIFY_PROMPT``.
JUDGE_PROMPT = r"""Decide if this MCQA distractor is effectively the correct answer.

Flag as equivalent if any apply:
- It is a paraphrase, synonym, or restatement of the intended answer.
- The source passage supports it as a correct answer to the question.
- A careful reader could legitimately pick either it or the intended answer.

Do not flag distractors that are merely related-but-wrong, plausible-sounding, or share vocabulary with the intended answer.

Return compact JSON: {"e":true,"r":"<=12 word reason"} if equivalent, else {"e":false}."""


REGEN_PROMPT = r"""Write 1 replacement MCQA distractor for an academic-text probe.

Requirements:
- Must be unambiguously WRONG given the source passage.
- Must NOT be a paraphrase, synonym, or restatement of the intended answer.
- Must NOT duplicate or paraphrase any existing distractor.
- Match target length, style, punctuation, and specificity.
- Use common confusions or nearby source-context concepts where possible.

Return compact JSON: {"d":"..."}."""


@dataclass
class DistractorEdit:
    review_id: str
    distractor_index: int
    option_label: str
    original: str
    final: str
    judged_equivalent: bool
    judge_reason: str
    regenerated: bool
    attempts: int
    success: bool
    failure_reason: str = ""
    candidate_history: list[dict[str, Any]] = field(default_factory=list)


def normalize_space(text: object) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def normalize_answer(text: object) -> str:
    return normalize_space(text).casefold()


def format_question(stem: str, options: list[str]) -> str:
    return normalize_space(stem) + "\n" + "\n".join(
        f"{label} {option}" for label, option in zip(LABELS, options)
    )


def label_to_index(label: str) -> int:
    return LABELS.index(label.strip())


def discover_v15_paths(domain_filter: tuple[str, ...] | None,
                       document_filter: str | None) -> list[Path]:
    groups = domain_filter or DOMAIN_GROUPS
    paths: list[Path] = []
    for group in groups:
        for p in (PROBE_ROOT / group).glob("*/facts/probes_v15_mcqa.csv"):
            if document_filter and document_filter not in str(p):
                continue
            paths.append(p)
    return sorted(paths)


def build_source_passage(row: pd.Series) -> str:
    """Best-effort source passage for the judge/generator.

    v15_mcqa.csv keeps only ``fact``, ``raw_knowledge_statement``, ``section``,
    so ``_extract_preceding_context`` will fall back to ``fact`` -- which is
    the contextualized assertion and is enough to ground a yes/no judgment.
    """
    try:
        ctx = mcqa_pipe._extract_preceding_context(row, preceding_words=100,
                                                   following_words=25)
        if ctx:
            return ctx
    except Exception:  # noqa: BLE001
        pass
    parts = []
    for col in ("raw_knowledge_statement", "fact"):
        val = str(row.get(col, "") or "").strip()
        if val:
            parts.append(val)
    return "\n".join(parts) or str(row.get("probe", ""))


def _query_json(prompt: dict, *, model: str, reasoning_effort: str,
                is_hippa: bool, max_tokens: int) -> dict | None:
    response = utils.query_llm(
        prompt,
        model=model,
        reasoning_effort=reasoning_effort,
        system_prompt_included=True,
        return_json=True,
        is_hippa=is_hippa,
        max_tokens=max_tokens,
    )
    if not response:
        return None
    try:
        return json.loads(response) if isinstance(response, str) else response
    except json.JSONDecodeError:
        return None


def judge_distractor(*, stem: str, correct: str, distractor: str,
                     source_passage: str, model: str, reasoning_effort: str,
                     is_hippa: bool) -> tuple[bool, str]:
    prompt = {
        "system": JUDGE_PROMPT,
        "user": (
            f"q: {stem}\n"
            f"intended: {correct}\n"
            f"distractor: {distractor}\n"
            f"src: {source_passage}"
        ),
    }
    data = _query_json(prompt, model=model, reasoning_effort=reasoning_effort,
                       is_hippa=is_hippa, max_tokens=80)
    if not data:
        return False, "judge_parse_failed"
    equivalent = bool(data.get("e", data.get("equivalent", False)))
    reason = str(data.get("r", data.get("reason", ""))).strip()
    return equivalent, reason


def regenerate_distractor(*, stem: str, correct: str, existing: list[str],
                          source_passage: str, rejected_reason: str,
                          model: str, reasoning_effort: str,
                          is_hippa: bool) -> str | None:
    existing_block = "; ".join(existing) if existing else "(none)"
    prompt = {
        "system": REGEN_PROMPT,
        "user": (
            f"q: {stem}\n"
            f"a: {correct}\n"
            f"existing_distractors: {existing_block}\n"
            f"rejected_reason: {rejected_reason}\n"
            f"src: {source_passage}"
        ),
    }
    data = _query_json(prompt, model=model, reasoning_effort=reasoning_effort,
                       is_hippa=is_hippa, max_tokens=200)
    if not data:
        return None
    candidate = str(data.get("d", data.get("distractor", ""))).strip()
    return candidate or None


def process_row(*, review_id: str, row: pd.Series, judge_model: str,
                generator_model: str, judge_reasoning: str,
                generator_reasoning: str, is_hippa: bool,
                max_attempts: int) -> tuple[dict, list[DistractorEdit]]:
    stem = str(row["probe"])
    correct_label = str(row["correct_label"]).strip()
    correct_idx = label_to_index(correct_label)
    correct_text = str(row[OPTION_COLS[correct_idx]])
    options = [str(row[c]) for c in OPTION_COLS]
    source_passage = build_source_passage(row)

    distractor_positions = [i for i in range(5) if i != correct_idx]
    edits: list[DistractorEdit] = []
    final_options = list(options)

    for j, opt_idx in enumerate(distractor_positions):
        original = options[opt_idx]
        equivalent, reason = judge_distractor(
            stem=stem, correct=correct_text, distractor=original,
            source_passage=source_passage, model=judge_model,
            reasoning_effort=judge_reasoning, is_hippa=is_hippa,
        )
        edit = DistractorEdit(
            review_id=review_id, distractor_index=j,
            option_label=LABELS[opt_idx], original=original, final=original,
            judged_equivalent=equivalent, judge_reason=reason,
            regenerated=False, attempts=0, success=not equivalent,
        )
        if not equivalent:
            edits.append(edit)
            continue

        rejection_reason = reason or "equivalent to the correct answer"
        accepted = False
        for attempt in range(1, max_attempts + 1):
            edit.attempts = attempt
            edit.regenerated = True
            others = [final_options[k] for k in range(5) if k != opt_idx]
            candidate = regenerate_distractor(
                stem=stem, correct=correct_text, existing=others,
                source_passage=source_passage, rejected_reason=rejection_reason,
                model=generator_model, reasoning_effort=generator_reasoning,
                is_hippa=is_hippa,
            )
            if not candidate:
                edit.candidate_history.append({"attempt": attempt, "candidate": None,
                                                "verdict": "generator_returned_none"})
                continue
            if (any(normalize_answer(candidate) == normalize_answer(o) for o in others)
                    or normalize_answer(candidate) == normalize_answer(correct_text)):
                rejection_reason = "candidate duplicated an existing option"
                edit.candidate_history.append({"attempt": attempt, "candidate": candidate,
                                                "verdict": "duplicate_of_existing"})
                continue
            new_equiv, new_reason = judge_distractor(
                stem=stem, correct=correct_text, distractor=candidate,
                source_passage=source_passage, model=judge_model,
                reasoning_effort=judge_reasoning, is_hippa=is_hippa,
            )
            edit.candidate_history.append({"attempt": attempt, "candidate": candidate,
                                            "verdict": "equivalent" if new_equiv else "accepted",
                                            "judge_reason": new_reason})
            if not new_equiv:
                edit.final = candidate
                edit.success = True
                final_options[opt_idx] = candidate
                accepted = True
                break
            rejection_reason = new_reason or "still equivalent to the correct answer"
        if not accepted:
            edit.failure_reason = "max_attempts_exhausted"
            edit.success = False
        edits.append(edit)

    new_row = row.to_dict()
    for i, col in enumerate(OPTION_COLS):
        new_row[col] = final_options[i]
    new_row["target"] = final_options[correct_idx]
    new_row["distractors"] = json.dumps(
        [final_options[i] for i in distractor_positions], ensure_ascii=False
    )
    formatted = format_question(stem, final_options)
    new_row["formatted_question"] = formatted
    new_row["formatted_question_5shot"] = build_mcqa_5shot_prompt(formatted)
    return new_row, edits


def process_file(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    rel = path.relative_to(PROBE_ROOT)
    domain_group, document = rel.parts[0], rel.parts[1]
    is_hippa = args.is_hippa and domain_group == "medical"

    print(f"\n[{domain_group}/{document}] reading {path.relative_to(REPO_ROOT)}")
    df = pd.read_csv(path, keep_default_na=False)

    out_rows: list[dict | None] = [None] * len(df)
    all_edits: list[DistractorEdit] = []

    def task(idx: int, row: pd.Series):
        review_id = f"{domain_group}/{document}/{idx}"
        try:
            return idx, *process_row(
                review_id=review_id, row=row,
                judge_model=args.judge_model,
                generator_model=args.generator_model,
                judge_reasoning=args.judge_reasoning,
                generator_reasoning=args.generator_reasoning,
                is_hippa=is_hippa,
                max_attempts=args.max_regen_attempts,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {review_id}: {exc!r}; keeping original row")
            return idx, row.to_dict(), []

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = [pool.submit(task, i, row) for i, row in df.iterrows()]
        for fut in concurrent.futures.as_completed(futures):
            idx, new_row, edits = fut.result()
            out_rows[idx] = new_row
            all_edits.extend(edits)

    out_df = pd.DataFrame(out_rows, columns=df.columns)
    out_path = path.with_name("probes_v16_mcqa.csv")
    audit_path = path.with_name("probes_v16_mcqa.audit.csv")
    metrics_path = path.with_name("probes_v16_mcqa_metrics.txt")

    out_df.to_csv(out_path, index=False)
    pd.DataFrame([
        {
            "review_id": e.review_id,
            "distractor_index": e.distractor_index,
            "option_label": e.option_label,
            "judged_equivalent": e.judged_equivalent,
            "judge_reason": e.judge_reason,
            "regenerated": e.regenerated,
            "attempts": e.attempts,
            "success": e.success,
            "failure_reason": e.failure_reason,
            "original": e.original,
            "final": e.final,
            "candidate_history": json.dumps(e.candidate_history, ensure_ascii=False),
        } for e in all_edits
    ]).to_csv(audit_path, index=False)

    total = len(df) * 4
    flagged = sum(1 for e in all_edits if e.judged_equivalent)
    regen_ok = sum(1 for e in all_edits if e.regenerated and e.success)
    regen_fail = sum(1 for e in all_edits if e.regenerated and not e.success)
    metrics = (
        f"Factual MCQA v15 -> v16 ({domain_group}/{document})\n"
        f"{'=' * 60}\n"
        f"Rows: {len(df)}\n"
        f"Distractors judged: {total}\n"
        f"Flagged equivalent: {flagged}\n"
        f"Regenerated successfully: {regen_ok}\n"
        f"Regenerated but still equivalent (kept original): {regen_fail}\n"
        f"Judge model: {args.judge_model} (reasoning={args.judge_reasoning})\n"
        f"Generator model: {args.generator_model} (reasoning={args.generator_reasoning})\n"
        f"Max regen attempts per distractor: {args.max_regen_attempts}\n"
    )
    metrics_path.write_text(metrics)
    print(metrics)

    return {
        "domain_group": domain_group,
        "document": document,
        "rows": len(df),
        "distractors_judged": total,
        "flagged_equivalent": flagged,
        "regenerated_ok": regen_ok,
        "regenerated_failed": regen_fail,
        "out_path": str(out_path.relative_to(REPO_ROOT)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", action="append", choices=DOMAIN_GROUPS,
                        help="Restrict to one or more domain groups (default: all)")
    parser.add_argument("--document", default=None,
                        help="Only process documents whose path contains this string")
    parser.add_argument("--judge-model", default="gpt-5.4-mini")
    parser.add_argument("--generator-model", default="gpt-5.4")
    parser.add_argument("--judge-reasoning", default="low")
    parser.add_argument("--generator-reasoning", default="medium")
    parser.add_argument("--max-regen-attempts", type=int, default=3)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--is-hippa", action="store_true",
                        help="Route medical-domain calls through the HIPAA-compliant client")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most this many files (smoke testing)")
    args = parser.parse_args()

    domains = tuple(args.domain) if args.domain else None
    paths = discover_v15_paths(domains, args.document)
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        print("No probes_v15_mcqa.csv files matched.")
        return

    print(f"Processing {len(paths)} file(s):")
    for p in paths:
        print(f"  - {p.relative_to(REPO_ROOT)}")

    summary_rows: list[dict] = []
    t0 = time.time()
    for path in paths:
        summary_rows.append(process_file(path, args))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(summary_rows)
    summary_path = REPORT_DIR / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nWrote {summary_path.relative_to(REPO_ROOT)}")
    print(summary_df.to_string(index=False))
    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    csv.field_size_limit(10_000_000)
    random.seed(0)
    main()
