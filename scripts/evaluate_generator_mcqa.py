#!/usr/bin/env python3
"""Evaluate auxiliary-view generator models on the factual/inference MCQA probes.

The script talks to OpenAI, LiteLLM, and vLLM through the OpenAI-compatible
Chat Completions API.  Results are checkpointed one question at a time so the
large benchmark can be resumed without repeating paid or long-running calls.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import random
import re
import sys
import tempfile
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.generator_mcqa_config import (  # noqa: E402
    COMPLETION_TOKEN_CAP,
    EXPECTED_COUNTS,
    FAMILIES,
    MODEL_BY_KEY,
    MODELS,
    PROTOCOLS,
    GeneratorModel,
)


CONSTRAINED_SYSTEM_PROMPT = (
    "Solve the final multiple-choice question. The preceding five answered "
    "questions are demonstrations. Think as needed, then return only a JSON "
    "object with an `answer` field containing one uppercase letter from A to E."
)
REASONED_SYSTEM_PROMPT = (
    "Solve the final multiple-choice question. The preceding five answered "
    "questions are demonstrations. Reason freely, then end your response with "
    "exactly one separate line of the form `Final answer: (X)`, replacing X "
    "with one uppercase letter from A to E. Write nothing after that line."
)
ANSWER_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "mcqa_answer",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "enum": ["A", "B", "C", "D", "E"]}
            },
            "required": ["answer"],
            "additionalProperties": False,
        },
    },
}
FINAL_ANSWER_RE = re.compile(r"^Final answer:\s*\(([A-E])\)\s*[.!]?$", re.IGNORECASE)


@dataclass(frozen=True)
class ProbeQuestion:
    question_id: str
    family: str
    group: str
    domain: str
    row_index: int
    prompt: str
    correct_label: str
    prompt_sha256: str


def normalize_label(value: str) -> str:
    match = re.fullmatch(r"\s*\(?([A-Ea-e])\)?\s*", value or "")
    if not match:
        raise ValueError(f"Invalid MCQA label: {value!r}")
    return match.group(1).upper()


def parse_answer(protocol: str, content: str) -> tuple[str | None, str]:
    """Return (answer, parse_status) without using a semantic judge."""
    if protocol == "constrained":
        try:
            payload = json.loads(content)
        except (TypeError, json.JSONDecodeError):
            return None, "invalid_json"
        if not isinstance(payload, dict):
            return None, "invalid_schema"
        try:
            return normalize_label(payload.get("answer", "")), "parsed"
        except ValueError:
            return None, "invalid_schema"

    lines = [line.strip() for line in (content or "").strip().splitlines() if line.strip()]
    if not lines:
        return None, "empty"
    match = FINAL_ANSWER_RE.fullmatch(lines[-1])
    if not match:
        return None, "missing_final_answer_line"
    return match.group(1).upper(), "parsed"


def _probe_paths(probes_root: Path, family: str) -> list[Path]:
    if family == "factual":
        pattern = "*/facts/probes_v15_mcqa.csv"
    elif family == "inference":
        pattern = "*/inference/probes_v14_mcqa.csv"
    else:
        raise ValueError(f"Unknown probe family: {family}")
    paths: list[Path] = []
    for group in ("arxiv", "legal", "medical"):
        paths.extend(sorted((probes_root / group).glob(pattern)))
    return paths


def load_questions(
    probes_root: Path,
    family: str,
    *,
    limit: int | None = None,
    require_full_count: bool = True,
) -> list[ProbeQuestion]:
    paths = _probe_paths(probes_root, family)
    if len(paths) != 36:
        raise ValueError(f"Expected 36 {family} probe files, found {len(paths)}")
    questions: list[ProbeQuestion] = []
    for path in paths:
        relative = path.relative_to(probes_root)
        group, domain = relative.parts[0], relative.parts[1]
        with path.open(newline="", encoding="utf-8") as handle:
            for row_index, row in enumerate(csv.DictReader(handle)):
                prompt = (row.get("formatted_question_5shot") or "").strip()
                if not prompt:
                    raise ValueError(f"Missing formatted_question_5shot in {path}:{row_index + 2}")
                correct_label = normalize_label(row.get("correct_label", ""))
                question_id = f"{family}:{group}/{domain}:{row_index}"
                prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                questions.append(
                    ProbeQuestion(
                        question_id=question_id,
                        family=family,
                        group=group,
                        domain=domain,
                        row_index=row_index,
                        prompt=prompt,
                        correct_label=correct_label,
                        prompt_sha256=prompt_hash,
                    )
                )
    if require_full_count and len(questions) != EXPECTED_COUNTS[family]:
        raise ValueError(
            f"Expected {EXPECTED_COUNTS[family]} {family} questions, found {len(questions)}"
        )
    if limit is not None:
        questions = questions[:limit]
    return questions


def _load_local_key(name: str) -> str | None:
    value = os.environ.get(name)
    if value:
        return value
    keys_path = REPO_ROOT / "utils" / "keys.py"
    if not keys_path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("_generator_mcqa_local_keys", keys_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, name, None)


def build_client(model: GeneratorModel, base_url: str | None) -> OpenAI:
    if model.provider == "openai":
        api_key = _load_local_key("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set in the environment or utils/keys.py")
        return OpenAI(api_key=api_key, timeout=600, max_retries=0)
    if model.provider == "vllm":
        if not base_url:
            raise ValueError("A vLLM model requires --base-url")
        return OpenAI(
            api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
            base_url=base_url.rstrip("/"),
            timeout=600,
            max_retries=0,
        )
    if model.provider == "litellm":
        api_key = _load_local_key("LITELLM_API_KEY") or os.environ.get("LITELLM_API_KEY")
        if not api_key:
            raise ValueError("LITELLM_API_KEY is required for GLM-5.2")
        resolved_url = base_url or _load_local_key("LITELLM_BASE_URL") or os.environ.get(
            "LITELLM_BASE_URL"
        )
        if not resolved_url:
            raise ValueError("LITELLM_BASE_URL is required for GLM-5.2")
        return OpenAI(
            api_key=api_key,
            base_url=resolved_url.rstrip("/"),
            timeout=600,
            max_retries=0,
        )
    raise ValueError(f"Unknown provider {model.provider!r}")


def _request_params(model: GeneratorModel, protocol: str, question: ProbeQuestion) -> dict[str, Any]:
    params: dict[str, Any] = {
        "model": model.model_id,
        "messages": [
            {
                "role": "system",
                "content": (
                    CONSTRAINED_SYSTEM_PROMPT
                    if protocol == "constrained"
                    else REASONED_SYSTEM_PROMPT
                ),
            },
            {"role": "user", "content": question.prompt},
        ],
    }
    if model.provider == "openai":
        params["max_completion_tokens"] = COMPLETION_TOKEN_CAP
    else:
        params["max_tokens"] = COMPLETION_TOKEN_CAP
    if model.reasoning_effort is not None:
        params["reasoning_effort"] = model.reasoning_effort
    if model.provider == "vllm":
        params["temperature"] = 0.0
        params["seed"] = 0
    if protocol == "constrained":
        params["response_format"] = ANSWER_SCHEMA
    return params


def _usage_dict(completion: Any) -> dict[str, int | None]:
    usage = getattr(completion, "usage", None)
    details = getattr(usage, "completion_tokens_details", None) if usage else None
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "reasoning_tokens": getattr(details, "reasoning_tokens", None),
    }


def _retry_delay(exc: Exception, attempt: int) -> float:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers:
        retry_after = headers.get("retry-after")
        if retry_after:
            try:
                return min(float(retry_after) + 0.25, 120.0)
            except ValueError:
                pass
    return min(2 ** (attempt - 1) + random.random(), 60.0)


def _is_retryable(exc: Exception) -> bool:
    message = str(exc).lower()
    if "insufficient_quota" in message or "exceeded your current quota" in message:
        return False
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    if status in {408, 409, 429, 500, 502, 503, 504}:
        return True
    name = type(exc).__name__.lower()
    return "timeout" in name or "connection" in name


def evaluate_one(
    client: OpenAI,
    model: GeneratorModel,
    protocol: str,
    question: ProbeQuestion,
    max_attempts: int,
) -> dict[str, Any]:
    started = time.time()
    for attempt in range(1, max_attempts + 1):
        try:
            completion = client.chat.completions.create(
                **_request_params(model, protocol, question)
            )
            choice = completion.choices[0]
            message = choice.message
            content = (message.content or "").strip()
            refusal = getattr(message, "refusal", None)
            answer, parse_status = parse_answer(protocol, content)
            if refusal:
                answer, parse_status = None, "refusal"
            finish_reason = getattr(choice, "finish_reason", None)
            if finish_reason == "length":
                answer, parse_status = None, "truncated"
            return {
                "question_id": question.question_id,
                "prompt_sha256": question.prompt_sha256,
                "family": question.family,
                "group": question.group,
                "domain": question.domain,
                "row_index": question.row_index,
                "model_key": model.key,
                "model_id": model.model_id,
                "provider": model.provider,
                "reasoning_effort": model.reasoning_effort,
                "protocol": protocol,
                "correct_label": question.correct_label,
                "predicted_label": answer,
                "correct": answer == question.correct_label,
                "parse_status": parse_status,
                "finish_reason": finish_reason,
                "response_content": content,
                "usage": _usage_dict(completion),
                "attempts": attempt,
                "elapsed_seconds": round(time.time() - started, 3),
                "terminal": True,
            }
        except Exception as exc:  # API SDK exceptions differ across providers.
            if attempt >= max_attempts or not _is_retryable(exc):
                return {
                    "question_id": question.question_id,
                    "prompt_sha256": question.prompt_sha256,
                    "family": question.family,
                    "group": question.group,
                    "domain": question.domain,
                    "row_index": question.row_index,
                    "model_key": model.key,
                    "model_id": model.model_id,
                    "provider": model.provider,
                    "reasoning_effort": model.reasoning_effort,
                    "protocol": protocol,
                    "correct_label": question.correct_label,
                    "predicted_label": None,
                    "correct": None,
                    "parse_status": "infrastructure_error",
                    "finish_reason": None,
                    "response_content": "",
                    "usage": {},
                    "attempts": attempt,
                    "elapsed_seconds": round(time.time() - started, 3),
                    "terminal": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:1000],
                }
            time.sleep(_retry_delay(exc, attempt))
    raise AssertionError("unreachable")


def state_path(state_root: Path, model_key: str, protocol: str, family: str) -> Path:
    return state_root / model_key / protocol / f"{family}.jsonl"


def load_state(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return records
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            # A crash can leave one partial final append; earlier corruption is fatal.
            if line_number == len(lines):
                break
            raise ValueError(f"Malformed state line {path}:{line_number}")
        records[record["question_id"]] = record
    return records


def _append_record(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
    handle.flush()


def run_partition(
    *,
    client: OpenAI,
    model: GeneratorModel,
    protocol: str,
    family: str,
    questions: list[ProbeQuestion],
    state_root: Path,
    max_workers: int,
    max_attempts: int,
) -> tuple[int, int]:
    path = state_path(state_root, model.key, protocol, family)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_state(path)
    question_by_id = {question.question_id: question for question in questions}
    for question_id, record in existing.items():
        question = question_by_id.get(question_id)
        if question and record.get("prompt_sha256") != question.prompt_sha256:
            raise ValueError(f"Prompt changed for resumed question {question_id}")
        if question and record.get("model_id") != model.model_id:
            raise ValueError(f"Model changed for resumed question {question_id}")
    pending = [
        question
        for question in questions
        if not existing.get(question.question_id, {}).get("terminal", False)
    ]
    completed = len(questions) - len(pending)
    print(
        f"[{model.key}/{protocol}/{family}] {completed}/{len(questions)} already complete; "
        f"{len(pending)} pending",
        flush=True,
    )
    if not pending:
        return completed, 0

    failures = 0
    lock = threading.Lock()
    with path.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:
        iterator = iter(pending)
        in_flight: dict[Future[dict[str, Any]], ProbeQuestion] = {}

        def submit_next() -> bool:
            try:
                question = next(iterator)
            except StopIteration:
                return False
            future = executor.submit(
                evaluate_one, client, model, protocol, question, max_attempts
            )
            in_flight[future] = question
            return True

        for _ in range(max_workers):
            if not submit_next():
                break
        while in_flight:
            done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done:
                question = in_flight.pop(future)
                record = future.result()
                with lock:
                    _append_record(handle, record)
                if record.get("terminal"):
                    completed += 1
                else:
                    failures += 1
                    for pending_future in in_flight:
                        pending_future.cancel()
                    raise RuntimeError(
                        f"Aborting {model.key}/{protocol}/{family} after an "
                        f"infrastructure failure on {question.question_id}: "
                        f"{record.get('error_type')}: {record.get('error')}"
                    )
                if (completed + failures) % 100 == 0 or not in_flight:
                    print(
                        f"[{model.key}/{protocol}/{family}] processed "
                        f"{completed + failures}/{len(questions)} "
                        f"(terminal={completed}, infrastructure_errors={failures})",
                        flush=True,
                    )
                submit_next()
    return completed, failures


SUMMARY_COLUMNS = [
    "model_key",
    "experiment",
    "provider",
    "model_id",
    "reasoning_effort",
    "status",
] + [
    f"{protocol}_{family}_{suffix}"
    for protocol in PROTOCOLS
    for family in FAMILIES
    for suffix in ("accuracy", "correct", "total", "invalid")
]


def _partition_metrics(path: Path, expected: int) -> dict[str, Any] | None:
    state = load_state(path)
    terminal = [record for record in state.values() if record.get("terminal")]
    if len(terminal) != expected:
        return None
    correct = sum(record.get("correct") is True for record in terminal)
    invalid = sum(record.get("predicted_label") is None for record in terminal)
    return {
        "accuracy": correct / expected,
        "correct": correct,
        "total": expected,
        "invalid": invalid,
    }


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(path)


def write_summary(state_root: Path, summary_path: Path, probes_root: Path) -> None:
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        row: dict[str, Any] = {
            "model_key": model.key,
            "experiment": model.experiment,
            "provider": model.provider,
            "model_id": model.model_id,
            "reasoning_effort": model.reasoning_effort or "",
        }
        complete = True
        for protocol in PROTOCOLS:
            for family in FAMILIES:
                metrics = _partition_metrics(
                    state_path(state_root, model.key, protocol, family),
                    EXPECTED_COUNTS[family],
                )
                for suffix in ("accuracy", "correct", "total", "invalid"):
                    row[f"{protocol}_{family}_{suffix}"] = (
                        metrics[suffix] if metrics else ""
                    )
                complete = complete and metrics is not None
        row["status"] = "complete" if complete else "pending"
        rows.append(row)

    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=SUMMARY_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_text(summary_path, buffer.getvalue())

    dataset_files: dict[str, dict[str, str]] = {}
    for family in FAMILIES:
        for path in _probe_paths(probes_root, family):
            relative = str(path.relative_to(REPO_ROOT))
            dataset_files[relative] = {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "family": family,
            }
    manifest = {
        "schema_version": 1,
        "completion_token_cap": COMPLETION_TOKEN_CAP,
        "protocols": list(PROTOCOLS),
        "families": list(FAMILIES),
        "expected_counts": EXPECTED_COUNTS,
        "models": [asdict(model) for model in MODELS],
        "dataset_files": dataset_files,
        "summary": str(summary_path),
        "state_root": str(state_root),
        "status_by_model": {row["model_key"]: row["status"] for row in rows},
        "updated_at_unix": time.time(),
    }
    _atomic_text(
        summary_path.with_name("run_manifest.json"),
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )


def import_model_state(
    source: Path,
    state_root: Path,
    model: GeneratorModel,
    probes_root: Path,
) -> None:
    """Validate and atomically import one externally evaluated model directory."""
    source = source.resolve()
    if source.name == model.key:
        model_source = source
    elif (source / model.key).is_dir():
        model_source = source / model.key
    else:
        raise ValueError(
            f"Import source must be {model.key}/ or contain that directory: {source}"
        )

    validated: dict[tuple[str, str], Path] = {}
    for protocol in PROTOCOLS:
        for family in FAMILIES:
            path = model_source / protocol / f"{family}.jsonl"
            records = load_state(path)
            questions = load_questions(probes_root, family)
            question_by_id = {question.question_id: question for question in questions}
            if set(records) != set(question_by_id):
                missing = len(set(question_by_id) - set(records))
                extra = len(set(records) - set(question_by_id))
                raise ValueError(
                    f"Incomplete imported state {path}: missing={missing}, extra={extra}"
                )
            for question_id, record in records.items():
                question = question_by_id[question_id]
                if not record.get("terminal"):
                    raise ValueError(f"Imported record is nonterminal: {question_id}")
                expected = {
                    "model_key": model.key,
                    "model_id": model.model_id,
                    "protocol": protocol,
                    "family": family,
                    "prompt_sha256": question.prompt_sha256,
                }
                for field, value in expected.items():
                    if record.get(field) != value:
                        raise ValueError(
                            f"Imported record {question_id} has {field}={record.get(field)!r}; "
                            f"expected {value!r}"
                        )
            validated[(protocol, family)] = path

    destination = state_root / model.key
    destination.mkdir(parents=True, exist_ok=True)
    for (protocol, family), source_path in validated.items():
        target = state_path(state_root, model.key, protocol, family)
        target.parent.mkdir(parents=True, exist_ok=True)
        _atomic_text(target, source_path.read_text(encoding="utf-8"))


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", choices=sorted(MODEL_BY_KEY))
    parser.add_argument("--protocols", nargs="+", choices=PROTOCOLS, default=list(PROTOCOLS))
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument("--base-url")
    parser.add_argument(
        "--state-root", type=Path, default=Path("/local/joseph/generator_mcqa/state")
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=REPO_ROOT / "reports" / "generator_mcqa" / "accuracies.csv",
    )
    parser.add_argument("--probes-root", type=Path, default=REPO_ROOT / "probes")
    parser.add_argument("--max-workers", type=int)
    parser.add_argument("--max-attempts", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--allow-litellm", action="store_true")
    parser.add_argument("--no-summary", action="store_true")
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Rebuild accuracies.csv/run_manifest.json without making model calls.",
    )
    parser.add_argument(
        "--import-state",
        type=Path,
        help="Validate and import a complete external state directory for --model-key.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    args.probes_root = args.probes_root.resolve()
    args.state_root = args.state_root.resolve()
    args.summary_path = args.summary_path.resolve()
    if args.aggregate_only:
        if args.model_key or args.import_state:
            raise SystemExit("--aggregate-only cannot be combined with --model-key/--import-state")
        write_summary(args.state_root, args.summary_path, args.probes_root)
        return 0
    if not args.model_key:
        raise SystemExit("--model-key is required unless --aggregate-only is used")
    model = MODEL_BY_KEY[args.model_key]
    if args.import_state:
        import_model_state(args.import_state, args.state_root, model, args.probes_root)
        write_summary(args.state_root, args.summary_path, args.probes_root)
        return 0
    if model.provider == "litellm" and not args.allow_litellm:
        raise SystemExit(
            "Refusing to call GLM-5.2 without the explicit --allow-litellm opt-in"
        )
    if args.max_workers is not None and args.max_workers < 1:
        raise SystemExit("--max-workers must be positive")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")

    client = build_client(model, args.base_url)
    max_workers = args.max_workers or model.max_workers
    infrastructure_failures = 0
    for protocol in args.protocols:
        for family in args.families:
            questions = load_questions(
                args.probes_root,
                family,
                limit=args.limit,
                require_full_count=True,
            )
            _, failures = run_partition(
                client=client,
                model=model,
                protocol=protocol,
                family=family,
                questions=questions,
                state_root=args.state_root,
                max_workers=max_workers,
                max_attempts=args.max_attempts,
            )
            infrastructure_failures += failures
    if not args.no_summary and args.limit is None:
        write_summary(args.state_root, args.summary_path, args.probes_root)
    if infrastructure_failures:
        print(
            f"Run remains incomplete: {infrastructure_failures} requests exhausted retries. "
            "Re-run the same command to resume.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
