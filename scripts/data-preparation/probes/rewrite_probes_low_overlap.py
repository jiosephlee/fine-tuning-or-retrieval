#!/usr/bin/env python3
"""Rewrite factual probe prefixes to reduce lexical overlap with raw facts.

The script preserves the answer target exactly. It rewrites only the `probe`
column, then recomputes `fact` as `probe + target` when a `fact` column exists.
Outputs are written as sibling CSVs by default, leaving source files untouched.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SYSTEM_PROMPT = """You rewrite factual cloze probes.

Goal: produce a semantically equivalent probe prefix with minimal lexical overlap against the raw knowledge statement.

Rules:
- Preserve the answer target exactly by NOT rewriting it and NOT including it in the probe.
- The returned probe must remain a prefix that can be followed directly by the target.
- Do not turn the cloze prefix into a question. Do not end the probe with a question mark.
- Keep domain-specific jargon, names, acronyms, legal/medical/scientific terms, measurements,
  model names, statute names, case titles, dates, and citation-like identifiers unchanged when
  they are necessary for factual precision.
- Change ordinary wording, sentence structure, clause order, and connective phrasing.
- Avoid copying contiguous phrases from the raw knowledge statement unless they are
  unavoidable domain terminology.
- The rewritten probe must be semantically equivalent but lexically as far as possible from the original probe.
- Do not add facts, remove required constraints, or make the answer ambiguous.
- Return JSON only: {"rewritten_probe": "..."}"""


@dataclass(frozen=True)
class Metrics:
    token_jaccard: float
    char_ratio: float
    longest_common_token_run: int


@dataclass(frozen=True)
class RewriteResult:
    row_index: int
    original_probe: str
    rewritten_probe: str
    original_fact: str
    rewritten_fact: str
    original_metrics: Metrics
    rewritten_metrics: Metrics
    attempts: int
    accepted: bool
    note: str


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\\]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokens(text: str) -> list[str]:
    normalized = normalize_text(text)
    return normalized.split() if normalized else []


def token_jaccard(a: str, b: str) -> float:
    a_tokens = set(tokens(a))
    b_tokens = set(tokens(b))
    if not a_tokens and not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def longest_common_token_run(a: str, b: str) -> int:
    a_tokens = tokens(a)
    b_tokens = tokens(b)
    if not a_tokens or not b_tokens:
        return 0
    matcher = SequenceMatcher(None, a_tokens, b_tokens)
    return max((match.size for match in matcher.get_matching_blocks()), default=0)


def lexical_metrics(probe: str, raw_knowledge_statement: str) -> Metrics:
    return Metrics(
        token_jaccard=token_jaccard(probe, raw_knowledge_statement),
        char_ratio=SequenceMatcher(
            None,
            normalize_text(probe),
            normalize_text(raw_knowledge_statement),
        ).ratio(),
        longest_common_token_run=longest_common_token_run(probe, raw_knowledge_statement),
    )


def target_is_in_probe(target: str, probe: str) -> bool:
    target_norm = normalize_text(target)
    probe_norm = normalize_text(probe)
    if len(target_norm.split()) < 2:
        return False
    return bool(target_norm and target_norm in probe_norm)


def compact_json_loads(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("Model response was not a JSON object")
    return value


def get_openai_client(timeout: float) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("The openai package is required to rewrite probes.") from exc

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        try:
            from utils.keys import OPENAI_API_KEY as key_from_file  # type: ignore

            api_key = key_from_file
        except (ImportError, ModuleNotFoundError):
            api_key = None

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY was not found in the environment or utils.keys.")
    return OpenAI(api_key=api_key, timeout=timeout)


def call_model(
    client: Any,
    *,
    model: str,
    raw_knowledge_statement: str,
    original_probe: str,
    target: str,
    section: str,
    domain_hint: str,
    previous_note: str | None,
) -> str:
    user_prompt = {
        "domain_hint": domain_hint,
        "section": section,
        "raw_knowledge_statement": raw_knowledge_statement,
        "original_probe": original_probe,
        "target_to_keep_separate": target,
    }
    if previous_note:
        user_prompt["revision_note"] = previous_note

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
        ],
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or "{}"
    parsed = compact_json_loads(content)
    rewritten_probe = str(parsed.get("rewritten_probe", "")).strip()
    if not rewritten_probe:
        raise ValueError("Model response did not include a non-empty rewritten_probe")
    return rewritten_probe


def accept_rewrite(
    *,
    original_probe: str,
    rewritten_probe: str,
    raw_knowledge_statement: str,
    target: str,
    max_token_jaccard: float,
    max_lcs_run: int,
) -> tuple[bool, str, Metrics, Metrics]:
    old_metrics = lexical_metrics(original_probe, raw_knowledge_statement)
    new_metrics = lexical_metrics(rewritten_probe, raw_knowledge_statement)

    if target_is_in_probe(target, rewritten_probe) and not target_is_in_probe(target, original_probe):
        return False, "rewrite includes the held-out target", old_metrics, new_metrics

    if len(tokens(rewritten_probe)) < 4:
        return False, "rewrite is too short", old_metrics, new_metrics

    if rewritten_probe.rstrip().endswith("?"):
        return False, "rewrite is a question, not a cloze prefix", old_metrics, new_metrics

    if (
        new_metrics.token_jaccard <= max_token_jaccard
        and new_metrics.longest_common_token_run <= max_lcs_run
    ):
        return True, "accepted", old_metrics, new_metrics

    if normalize_text(rewritten_probe) == normalize_text(original_probe):
        return False, "rewrite is identical to the original probe", old_metrics, new_metrics

    jaccard_drop = old_metrics.token_jaccard - new_metrics.token_jaccard
    note = (
        "absolute lexical overlap still too high "
        f"(new token Jaccard {new_metrics.token_jaccard:.3f} > {max_token_jaccard:.3f} "
        f"or new LCS {new_metrics.longest_common_token_run} > {max_lcs_run}; "
        f"jaccard drop {jaccard_drop:.3f})"
    )
    return False, note, old_metrics, new_metrics


def structurally_valid_rewrite(original_probe: str, rewritten_probe: str, target: str) -> bool:
    if target_is_in_probe(target, rewritten_probe) and not target_is_in_probe(target, original_probe):
        return False
    if rewritten_probe.rstrip().endswith("?"):
        return False
    return len(tokens(rewritten_probe)) >= 4


def rewrite_row(
    row_index: int,
    row: dict[str, str],
    *,
    client: Any,
    model: str,
    attempts: int,
    max_token_jaccard: float,
    max_lcs_run: int,
    domain_hint: str,
    sleep_between_attempts: float,
) -> RewriteResult:
    raw = row.get("raw_knowledge_statement", "")
    original_probe = row.get("probe", "")
    target = row.get("target", "")
    original_fact = row.get("fact", original_probe + target)
    section = row.get("section", "")

    if not raw or not original_probe or not target:
        metrics = lexical_metrics(original_probe, raw)
        return RewriteResult(
            row_index=row_index,
            original_probe=original_probe,
            rewritten_probe=original_probe,
            original_fact=original_fact,
            rewritten_fact=original_fact,
            original_metrics=metrics,
            rewritten_metrics=metrics,
            attempts=0,
            accepted=False,
            note="missing raw_knowledge_statement, probe, or target",
        )

    previous_note: str | None = None
    best_probe = original_probe
    best_note = "no rewrite attempted"
    best_old_metrics = lexical_metrics(original_probe, raw)
    best_new_metrics = best_old_metrics

    for attempt in range(1, attempts + 1):
        try:
            rewritten_probe = call_model(
                client,
                model=model,
                raw_knowledge_statement=raw,
                original_probe=original_probe,
                target=target,
                section=section,
                domain_hint=domain_hint,
                previous_note=previous_note,
            )
        except Exception as exc:
            previous_note = f"model call failed on attempt {attempt}: {exc}"
            best_note = previous_note
            if sleep_between_attempts:
                time.sleep(sleep_between_attempts)
            continue
        accepted, note, old_metrics, new_metrics = accept_rewrite(
            original_probe=original_probe,
            rewritten_probe=rewritten_probe,
            raw_knowledge_statement=raw,
            target=target,
            max_token_jaccard=max_token_jaccard,
            max_lcs_run=max_lcs_run,
        )
        if structurally_valid_rewrite(original_probe, rewritten_probe, target) and (
            new_metrics.token_jaccard < best_new_metrics.token_jaccard
            or new_metrics.longest_common_token_run < best_new_metrics.longest_common_token_run
        ):
            best_probe = rewritten_probe
            best_note = note
            best_old_metrics = old_metrics
            best_new_metrics = new_metrics
        elif best_probe == original_probe:
            best_note = note
        if accepted:
            return RewriteResult(
                row_index=row_index,
                original_probe=original_probe,
                rewritten_probe=rewritten_probe,
                original_fact=original_fact,
                rewritten_fact=rewritten_probe + target,
                original_metrics=old_metrics,
                rewritten_metrics=new_metrics,
                attempts=attempt,
                accepted=True,
                note=note,
            )
        previous_note = note
        if sleep_between_attempts:
            time.sleep(sleep_between_attempts)

    return RewriteResult(
        row_index=row_index,
        original_probe=original_probe,
        rewritten_probe=best_probe,
        original_fact=original_fact,
        rewritten_fact=best_probe + target,
        original_metrics=best_old_metrics,
        rewritten_metrics=best_new_metrics,
        attempts=attempts,
        accepted=False,
        note=best_note,
    )


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header")
        rows = [{key: value for key, value in row.items()} for row in reader]
        return list(reader.fieldnames), rows


def write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_audit(path: Path, source_path: Path, results: list[RewriteResult]) -> None:
    fieldnames = [
        "source_file",
        "row_index",
        "accepted",
        "attempts",
        "note",
        "old_token_jaccard",
        "new_token_jaccard",
        "old_char_ratio",
        "new_char_ratio",
        "old_longest_common_token_run",
        "new_longest_common_token_run",
        "original_probe",
        "rewritten_probe",
        "original_fact",
        "rewritten_fact",
    ]
    audit_rows = []
    for result in results:
        audit_rows.append(
            {
                "source_file": str(source_path),
                "row_index": result.row_index,
                "accepted": result.accepted,
                "attempts": result.attempts,
                "note": result.note,
                "old_token_jaccard": f"{result.original_metrics.token_jaccard:.6f}",
                "new_token_jaccard": f"{result.rewritten_metrics.token_jaccard:.6f}",
                "old_char_ratio": f"{result.original_metrics.char_ratio:.6f}",
                "new_char_ratio": f"{result.rewritten_metrics.char_ratio:.6f}",
                "old_longest_common_token_run": result.original_metrics.longest_common_token_run,
                "new_longest_common_token_run": result.rewritten_metrics.longest_common_token_run,
                "original_probe": result.original_probe,
                "rewritten_probe": result.rewritten_probe,
                "original_fact": result.original_fact,
                "rewritten_fact": result.rewritten_fact,
            }
        )
    write_csv_rows(path, fieldnames, audit_rows)


def discover_probe_files(root: Path, version: str) -> list[Path]:
    return sorted(root.glob(f"*/*/facts/probes_{version}.csv"))


def output_path_for(source_path: Path, output_suffix: str, output_dir: Path | None) -> Path:
    output_name = source_path.with_name(f"{source_path.stem}{output_suffix}{source_path.suffix}").name
    if output_dir is None:
        return source_path.with_name(output_name)
    relative = source_path.relative_to(PROJECT_ROOT)
    return output_dir / relative.parent / output_name


def process_file(
    path: Path,
    *,
    client: Any,
    model: str,
    output_suffix: str,
    output_dir: Path | None,
    limit: int | None,
    start_row: int,
    max_workers: int,
    attempts: int,
    max_token_jaccard: float,
    max_lcs_run: int,
    dry_run: bool,
    progress_every: int,
) -> None:
    fieldnames, rows = read_csv_rows(path)
    required = {"probe", "target", "raw_knowledge_statement"}
    missing = required - set(fieldnames)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")

    selected_indices = list(range(start_row, len(rows)))
    if limit is not None:
        selected_indices = selected_indices[:limit]

    out_path = output_path_for(path, output_suffix, output_dir)
    audit_path = out_path.with_suffix(".audit.csv")
    print(f"{path}: {len(selected_indices)} rows selected -> {out_path}")
    if dry_run:
        return

    domain_hint = "/".join(path.parts[-4:-2])
    results: list[RewriteResult] = []

    def submit(index: int) -> RewriteResult:
        return rewrite_row(
            index,
            rows[index],
            client=client,
            model=model,
            attempts=attempts,
            max_token_jaccard=max_token_jaccard,
            max_lcs_run=max_lcs_run,
            domain_hint=domain_hint,
            sleep_between_attempts=0.25,
        )

    if max_workers == 1:
        for index in selected_indices:
            results.append(submit(index))
            if progress_every and len(results) % progress_every == 0:
                print(f"  completed {len(results)}/{len(selected_indices)} rows", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {executor.submit(submit, index): index for index in selected_indices}
            for future in as_completed(future_to_index):
                results.append(future.result())
                if progress_every and len(results) % progress_every == 0:
                    print(f"  completed {len(results)}/{len(selected_indices)} rows", flush=True)

    for result in results:
        row = rows[result.row_index]
        row["probe"] = result.rewritten_probe
        if "fact" in row:
            row["fact"] = result.rewritten_fact

    write_csv_rows(out_path, fieldnames, rows)
    write_audit(audit_path, path, sorted(results, key=lambda item: item.row_index))

    accepted_count = sum(1 for result in results if result.accepted)
    old_jaccard = sum(result.original_metrics.token_jaccard for result in results) / max(len(results), 1)
    new_jaccard = sum(result.rewritten_metrics.token_jaccard for result in results) / max(len(results), 1)
    old_lcs = sum(result.original_metrics.longest_common_token_run for result in results) / max(len(results), 1)
    new_lcs = sum(result.rewritten_metrics.longest_common_token_run for result in results) / max(len(results), 1)
    print(
        f"  accepted {accepted_count}/{len(results)}; "
        f"token Jaccard {old_jaccard:.3f}->{new_jaccard:.3f}; "
        f"LCS {old_lcs:.1f}->{new_lcs:.1f}; audit {audit_path}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite factual probe prefixes to reduce lexical overlap with raw knowledge statements.",
    )
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="Specific probe CSV files. If omitted, discovers probes/*/*/facts/probes_v14.csv.",
    )
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "probes")
    parser.add_argument("--version", default="v14")
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--output-suffix", default="_low_overlap")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Only rewrite this many rows per file.")
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument(
        "--max-token-jaccard",
        type=float,
        default=0.25,
        help="Accept rewrites whose final token Jaccard overlap with the raw fact is at or below this value.",
    )
    parser.add_argument(
        "--min-jaccard-drop",
        type=float,
        default=0.08,
        help="Deprecated; retained for CLI compatibility. Acceptance is now based on absolute final overlap.",
    )
    parser.add_argument("--max-lcs-run", type=int, default=6)
    parser.add_argument("--request-timeout", type=float, default=60.0)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = [path.resolve() for path in args.files] if args.files else discover_probe_files(args.root, args.version)
    if not files:
        raise SystemExit("No probe files found.")

    client = None if args.dry_run else get_openai_client(args.request_timeout)
    for path in files:
        process_file(
            path,
            client=client,
            model=args.model,
            output_suffix=args.output_suffix,
            output_dir=args.output_dir,
            limit=args.limit,
            start_row=args.start_row,
            max_workers=max(args.max_workers, 1),
            attempts=max(args.attempts, 1),
            max_token_jaccard=args.max_token_jaccard,
            max_lcs_run=args.max_lcs_run,
            dry_run=args.dry_run,
            progress_every=args.progress_every,
        )


if __name__ == "__main__":
    main()
