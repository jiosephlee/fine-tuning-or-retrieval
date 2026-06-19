#!/usr/bin/env python3
"""Check raw source documents for exact matches in an Infini-gram index."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_INDEX = "v4_olmo-mix-1124_llama"
DEFAULT_API_URL = "https://api.infini-gram.io/"
DEFAULT_QUERY_OUTPUT = Path("analysis/infinigram_olmo2_membership_queries.csv")
DEFAULT_SUMMARY_OUTPUT = Path("analysis/infinigram_olmo2_membership_summary.csv")
USER_AGENT = "fine-tuning-or-retrieval-infinigram-membership-check/1.0"

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'([])")
LATEX_COMMAND_WITH_ARG_RE = re.compile(r"\\[a-zA-Z*]+(?:\[[^\]]*\])?\{([^{}]*)\}")
LATEX_COMMAND_RE = re.compile(r"\\[a-zA-Z*]+(?:\[[^\]]*\])?")


@dataclass(frozen=True)
class SourceDocument:
    source: str
    document_id: str
    path: Path
    title: str
    raw_text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample titles and body sentences from raw arxiv/medical/legal "
            "sources and query an Infini-gram index for exact counts."
        )
    )
    parser.add_argument("--index", default=DEFAULT_INDEX)
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--sentences-per-doc", type=int, default=10)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--write-every", type=int, default=25)
    parser.add_argument("--stop-after-consecutive-forbidden", type=int, default=5)
    parser.add_argument("--limit-docs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--query-output", type=Path, default=DEFAULT_QUERY_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def find_balanced_brace_value(text: str, command: str) -> str:
    marker = "\\" + command
    start = text.find(marker)
    if start < 0:
        return ""
    brace_start = text.find("{", start + len(marker))
    if brace_start < 0:
        return ""
    depth = 0
    value_start = brace_start + 1
    for idx in range(brace_start, len(text)):
        char = text[idx]
        if char == "{" and (idx == 0 or text[idx - 1] != "\\"):
            depth += 1
        elif char == "}" and (idx == 0 or text[idx - 1] != "\\"):
            depth -= 1
            if depth == 0:
                return text[value_start:idx]
    return ""


def clean_latex_fragment(text: str) -> str:
    text = re.sub(r"%.*", " ", text)
    text = text.replace("\\\\", " ")
    text = re.sub(r"\\(?:begin|end)\{[^{}]*\}", " ", text)
    text = re.sub(r"\\(?:cite|citep|citet|ref|eqref|label|url)\*?(?:\[[^\]]*\])?\{[^{}]*\}", " ", text)
    text = re.sub(r"\$[^$]*\$", " ", text)
    previous = None
    while previous != text:
        previous = text
        text = LATEX_COMMAND_WITH_ARG_RE.sub(r"\1", text)
    text = LATEX_COMMAND_RE.sub(" ", text)
    text = text.replace("~", " ")
    text = text.replace("--", " ")
    text = re.sub(r"[{}_^&]", " ", text)
    return normalize_whitespace(text)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def discover_documents() -> list[SourceDocument]:
    docs: list[SourceDocument] = []
    docs.extend(discover_arxiv_documents(Path("data/arxiv/raw")))
    docs.extend(discover_text_documents("medical", Path("data/medical/raw"), "title"))
    docs.extend(discover_text_documents("legal", Path("data/legal/raw"), "case_name"))
    return docs


def discover_arxiv_documents(root: Path) -> list[SourceDocument]:
    docs: list[SourceDocument] = []
    for path in sorted(root.glob("*.tex")):
        cleaned_path = Path("data/arxiv/cleaned") / path.name
        query_text_path = cleaned_path if cleaned_path.exists() else path
        raw_text = query_text_path.read_text(encoding="utf-8", errors="ignore")
        title = clean_latex_fragment(find_balanced_brace_value(raw_text, "title"))
        if not title:
            title = path.stem
        docs.append(SourceDocument("arxiv", path.stem, path, title, raw_text))
    return docs


def discover_text_documents(source: str, root: Path, title_key: str) -> list[SourceDocument]:
    docs: list[SourceDocument] = []
    for path in sorted(root.glob("*.txt")):
        raw_text = path.read_text(encoding="utf-8", errors="ignore")
        meta = read_json(path.with_suffix(".meta.json"))
        title = normalize_whitespace(str(meta.get(title_key) or ""))
        if not title:
            title = path.stem.replace("_", " ")
        docs.append(SourceDocument(source, path.stem, path, title, raw_text))
    return docs


def strip_arxiv_preamble(text: str) -> str:
    candidates = [m.start() for m in re.finditer(r"\\(?:begin\{abstract\}|section\*?\{)", text)]
    if not candidates:
        return text
    return text[min(candidates):]


def remove_latex_headings_and_commands(text: str) -> str:
    text = re.sub(r"\\title\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", " ", text)
    text = re.sub(r"\\author\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", " ", text)
    text = re.sub(r"\\(?:section|subsection|subsubsection|paragraph)\*?\{[^{}]*\}", ". ", text)
    return text


def truncate_arxiv_backmatter(text: str) -> str:
    markers = (
        r"\\section\*?\{Acknowledg",
        r"\\section\*?\{References",
        r"\\bibliograph",
        r"\\begin\{thebibliography\}",
    )
    cut_points = [m.start() for marker in markers for m in re.finditer(marker, text)]
    if not cut_points:
        return text
    return text[: min(cut_points)]


def body_text_for_document(doc: SourceDocument) -> str:
    text = doc.raw_text
    if doc.source == "arxiv":
        text = strip_arxiv_preamble(text)
        text = truncate_arxiv_backmatter(text)
        text = remove_latex_headings_and_commands(text)
        return clean_latex_fragment(text)

    text = text.replace("\f", " ")
    lines = [normalize_whitespace(line) for line in text.splitlines()]
    lines = [line for line in lines if line]
    filtered: list[str] = []
    title_lower = doc.title.lower()
    for line in lines:
        lower = line.lower()
        if lower == title_lower:
            continue
        if is_probable_header_line(line):
            continue
        filtered.append(line)
    if not filtered:
        filtered = [normalize_whitespace(text)]

    body = " ".join(filtered)
    if doc.source == "medical":
        body = strip_initial_medical_metadata(body, doc.title)
    elif doc.source == "legal":
        body = strip_initial_legal_caption(body)
    return normalize_whitespace(body)


def is_probable_header_line(line: str) -> bool:
    words = re.findall(r"[A-Za-z]+", line)
    if not words:
        return True
    if len(words) <= 8 and not re.search(r"[.!?]$", line):
        return True
    letters = "".join(words)
    if len(words) <= 12 and letters and sum(ch.isupper() for ch in letters) / len(letters) > 0.75:
        return True
    boilerplate = (
        "abstract",
        "summary",
        "background",
        "case presentation",
        "discussion",
        "conclusion",
        "learning points",
        "before:",
        "appeal from",
        "argued ",
        "decided ",
    )
    return any(line.lower().startswith(prefix) for prefix in boilerplate)


def strip_initial_medical_metadata(text: str, title: str) -> str:
    if title and text.lower().startswith(title.lower()):
        text = text[len(title) :]
    for marker in (" Summary", " Background", " Case presentation", " Given ", " We present "):
        idx = text.find(marker)
        if 0 <= idx < 1200:
            return text[idx:]
    return text[800:] if len(text) > 1600 else text


def strip_initial_legal_caption(text: str) -> str:
    markers = (
        "OPINION",
        "Opinion for the Court",
        "This appeal",
        "This case",
        "The question",
        "Plaintiff",
        "Defendant",
        "Appellant",
    )
    upper_text = text.upper()
    for marker in markers[:2]:
        idx = upper_text.find(marker)
        if 0 <= idx < 2500:
            return text[idx + len(marker) :]
    for marker in markers[2:]:
        idx = text.find(marker)
        if 0 <= idx < 2500:
            return text[idx:]
    return text[1200:] if len(text) > 2400 else text


def candidate_sentences(doc: SourceDocument) -> list[str]:
    text = body_text_for_document(doc)
    parts = SENTENCE_SPLIT_RE.split(text)
    sentences: list[str] = []
    seen: set[str] = set()
    for part in parts:
        sentence = normalize_whitespace(part)
        if is_usable_sentence(sentence):
            key = sentence.lower()
            if key not in seen:
                sentences.append(sentence)
                seen.add(key)
    return sentences


def is_usable_sentence(sentence: str) -> bool:
    if not (80 <= len(sentence) <= 500):
        return False
    words = re.findall(r"[A-Za-z0-9]+", sentence)
    if not (12 <= len(words) <= 80):
        return False
    letters = re.findall(r"[A-Za-z]", sentence)
    if len(letters) < 50:
        return False
    if sum(ch.isupper() for ch in letters) / len(letters) > 0.55:
        return False
    if sentence.count("\\") or sentence.count("{") or sentence.count("}"):
        return False
    if re.search(r"\b(?:doi|pmid|pmcid|copyright|license|fig\.|table)\b", sentence, flags=re.I):
        return False
    return True


def stable_sample(items: list[str], count: int, seed: int, document_id: str) -> list[str]:
    if len(items) <= count:
        return items
    digest = hashlib.sha256(document_id.encode("utf-8")).hexdigest()
    doc_seed = seed + int(digest[:12], 16)
    rng = random.Random(doc_seed)
    indices = sorted(rng.sample(range(len(items)), count))
    return [items[idx] for idx in indices]


def post_infinigram(
    api_url: str,
    index: str,
    query: str,
    timeout: float,
    retries: int,
) -> dict[str, Any]:
    payload = json.dumps(
        {"index": index, "query_type": "count", "query": query},
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        api_url,
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
        method="POST",
    )
    last_error: str | None = None
    last_error_code: int | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last_error = str(exc)
            last_error_code = exc.code
            if exc.code == 403:
                break
            if attempt < retries:
                time.sleep(0.5 * (2**attempt))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = str(exc)
            if attempt < retries:
                time.sleep(0.5 * (2**attempt))
    return {"error": last_error or "unknown error", "error_code": last_error_code or ""}


def build_query_rows(
    docs: list[SourceDocument],
    sentences_per_doc: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for doc in docs:
        rows.append(
            {
                "source": doc.source,
                "document_id": doc.document_id,
                "path": str(doc.path),
                "query_type": "title",
                "sample_index": 0,
                "query": doc.title,
            }
        )
        sampled = stable_sample(candidate_sentences(doc), sentences_per_doc, seed, doc.document_id)
        for idx, sentence in enumerate(sampled, start=1):
            rows.append(
                {
                    "source": doc.source,
                    "document_id": doc.document_id,
                    "path": str(doc.path),
                    "query_type": "body_sentence",
                    "sample_index": idx,
                    "query": sentence,
                }
            )
    return rows


def row_key(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    query_hash = hashlib.sha256(str(row["query"]).encode("utf-8")).hexdigest()
    return (
        str(row["source"]),
        str(row["document_id"]),
        str(row["query_type"]),
        str(row["sample_index"]),
        query_hash,
    )


def load_existing_query_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def query_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    total = len(rows)
    existing_rows = load_existing_query_rows(args.query_output) if args.resume else []
    queried_by_key = {row_key(row): row for row in existing_rows}
    queried: list[dict[str, Any]] = []
    consecutive_forbidden = 0
    for idx, row in enumerate(rows, start=1):
        key = row_key(row)
        if key in queried_by_key:
            existing = queried_by_key[key]
            queried.append(existing)
            print(
                f"[{idx}/{total}] {row['source']}/{row['document_id']} "
                f"{row['query_type']} count={existing.get('count', '')} cached",
                flush=True,
            )
            continue
        result = post_infinigram(
            api_url=args.api_url,
            index=args.index,
            query=row["query"],
            timeout=args.timeout,
            retries=args.retries,
        )
        out = dict(row)
        out["index"] = args.index
        out["count"] = result.get("count", "")
        out["approx"] = result.get("approx", "")
        out["latency_ms"] = result.get("latency", "")
        out["token_count"] = len(result.get("token_ids", []) or [])
        out["error"] = result.get("error", "")
        out["error_code"] = result.get("error_code", "")
        queried.append(out)
        queried_by_key[key] = out
        print(
            f"[{idx}/{total}] {out['source']}/{out['document_id']} "
            f"{out['query_type']} count={out['count']} error={out['error']}",
            flush=True,
        )
        if out["error_code"] == 403:
            consecutive_forbidden += 1
            if consecutive_forbidden >= args.stop_after_consecutive_forbidden:
                print(
                    f"Stopping after {consecutive_forbidden} consecutive HTTP 403 errors.",
                    flush=True,
                )
                break
        else:
            consecutive_forbidden = 0
        if args.write_every and len(queried) % args.write_every == 0:
            write_csv(args.query_output, queried)
        if args.sleep:
            time.sleep(args.sleep)
    write_csv(args.query_output, queried)
    return queried


def build_summary_rows(
    docs: list[SourceDocument],
    query_rows: list[dict[str, Any]],
    expected_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_doc: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in query_rows:
        by_doc.setdefault((row["source"], row["document_id"]), []).append(row)
    expected_by_doc: dict[tuple[str, str], int] = {}
    for row in expected_rows:
        key = (row["source"], row["document_id"])
        expected_by_doc[key] = expected_by_doc.get(key, 0) + 1

    summaries: list[dict[str, Any]] = []
    for doc in docs:
        doc_key = (doc.source, doc.document_id)
        rows = by_doc.get(doc_key, [])
        expected_count = expected_by_doc.get(doc_key, 0)
        title_counts = [as_int(row.get("count")) for row in rows if row.get("query_type") == "title"]
        sentence_counts = [
            as_int(row.get("count")) for row in rows if row.get("query_type") == "body_sentence"
        ]
        all_counts = title_counts + sentence_counts
        errors = sum(1 for row in rows if row.get("error"))
        hits = sum(1 for count in all_counts if count and count > 0)
        if hits:
            status = "hit"
        elif len(rows) < expected_count:
            status = "incomplete_not_queried"
        elif errors:
            status = "incomplete_no_hit"
        else:
            status = "no_evidence"
        summaries.append(
            {
                "source": doc.source,
                "document_id": doc.document_id,
                "path": str(doc.path),
                "title": doc.title,
                "num_queries": len(rows),
                "expected_num_queries": expected_count,
                "num_body_sentence_queries": len(sentence_counts),
                "title_count": title_counts[0] if title_counts else "",
                "max_body_sentence_count": max(sentence_counts) if sentence_counts else "",
                "num_hit_queries": hits,
                "num_error_queries": errors,
                "status": status,
            }
        )
    return summaries


def as_int(value: Any) -> int | None:
    if value == "" or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    docs = discover_documents()
    if args.limit_docs is not None:
        docs = docs[: args.limit_docs]
    rows = build_query_rows(docs, args.sentences_per_doc, args.seed)

    if args.dry_run:
        for row in rows:
            print(
                f"{row['source']}/{row['document_id']} "
                f"{row['query_type']}[{row['sample_index']}]: {row['query']}"
            )
        print(f"Prepared {len(rows)} queries for {len(docs)} documents.")
        return

    queried = query_rows(rows, args)
    summaries = build_summary_rows(docs, queried, rows)
    write_csv(args.summary_output, summaries)
    print(f"Wrote {len(queried)} query rows to {args.query_output}")
    print(f"Wrote {len(summaries)} summary rows to {args.summary_output}")


if __name__ == "__main__":
    main()
