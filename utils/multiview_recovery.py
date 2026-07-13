"""Validation and transactional publication for multiview corpora.

This module deliberately has no model/provider dependencies so it can be used by
audits, recovery jobs, and fixture tests.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from utils.granular_outputs import GRANULAR_LAYOUTS

VIEWS = ("stackexchange", "textbook", "blog")
ASSEMBLED = {"stackexchange": "stackexchange.txt", "textbook": "textbook.txt", "blog": "blogs.txt"}
OUTLINES = {"stackexchange": "stack_exchange_outline.json", "textbook": "textbook_outline.json", "blog": "blog_outline.json"}
OUTLINE_KEYS = {"stackexchange": "questions", "textbook": "outline", "blog": "blogs"}
HARMONY_MARKERS = ("<|channel|>", "<|start|>", "<|end|>", "<|message|>")
RESERVED_TOKEN = re.compile(r"<\|(?:reserved[^|]*|[^|]*token[^|]*)\|>", re.IGNORECASE)

QUESTION_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["questions"],
    "properties": {"questions": {"type": "array", "minItems": 1, "items": {
        "type": "object", "additionalProperties": False,
        "required": ["title", "question_body"],
        "properties": {"title": {"type": "string", "minLength": 1},
                       "question_body": {"type": "string", "minLength": 1}},
    }}},
}
TEXTBOOK_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["outline"],
    "properties": {"outline": {"type": "array", "minItems": 1, "items": {
        "type": "object", "additionalProperties": False,
        "required": ["chapter_title", "description", "subtopics"],
        "properties": {"chapter_title": {"type": "string", "minLength": 1},
                       "description": {"type": "string", "minLength": 1},
                       "subtopics": {"type": "array", "minItems": 1,
                                     "items": {"type": "string", "minLength": 1}}},
    }}},
}
MEDICAL_TEXTBOOK_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["sections"],
    "properties": {"sections": {"type": "array", "minItems": 1, "items": {
        "type": "object", "additionalProperties": False,
        "required": ["section_title", "description", "subtopics"],
        "properties": {"section_title": {"type": "string", "minLength": 1},
                       "description": {"type": "string", "minLength": 1},
                       "subtopics": {"type": "array", "minItems": 1,
                                     "items": {"type": "string", "minLength": 1}}},
    }}},
}
BLOG_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["blogs"],
    "properties": {"blogs": {"type": "array", "minItems": 1, "items": {
        "type": "object", "additionalProperties": False,
        "required": ["title", "description"],
        "properties": {"title": {"type": "string", "minLength": 1},
                       "description": {"type": "string", "minLength": 1}},
    }}},
}
SCHEMAS = {"stackexchange": QUESTION_SCHEMA, "textbook": TEXTBOOK_SCHEMA, "blog": BLOG_SCHEMA}


def token_estimate(text: str) -> int:
    return max(1, (len(text) + 3) // 4) if text else 0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _schema_errors(value, schema, at="$"):
    errors = []
    expected = schema.get("type")
    if expected == "object":
        if not isinstance(value, dict): return [f"{at}: expected object"]
        for key in schema.get("required", []):
            if key not in value: errors.append(f"{at}: missing key {key}")
        if schema.get("additionalProperties") is False:
            allowed = set(schema.get("properties", {}))
            errors += [f"{at}: unexpected key {key}" for key in value if key not in allowed]
        for key, child in schema.get("properties", {}).items():
            if key in value: errors += _schema_errors(value[key], child, f"{at}.{key}")
    elif expected == "array":
        if not isinstance(value, list): return [f"{at}: expected array"]
        if len(value) < schema.get("minItems", 0): errors.append(f"{at}: too few items")
        for index, item in enumerate(value): errors += _schema_errors(item, schema.get("items", {}), f"{at}[{index}]")
    elif expected == "string":
        if not isinstance(value, str): errors.append(f"{at}: expected string")
        elif len(value.strip()) < schema.get("minLength", 0): errors.append(f"{at}: empty string")
    return errors


def content_reasons(text: str, *, assembled=False) -> list[str]:
    reasons = []
    if not text.strip(): return ["empty"]
    if len(text.encode("utf-8")) < (200 if assembled else 80): reasons.append("truncated_or_too_short")
    if any(marker in text for marker in HARMONY_MARKERS): reasons.append("harmony_marker")
    if RESERVED_TOKEN.search(text): reasons.append("reserved_token_leakage")
    if "\ufffd" in text: reasons.append("unicode_replacement_character")
    if any(ord(c) < 32 and c not in "\n\r\t" for c in text): reasons.append("invalid_control_character")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    # Repeated Markdown delimiters, math fences, and field labels are normal in an
    # assembled multi-item view. Detect loops only in substantive lines.
    substantive = [line for line in lines
                   if len(line) >= 32
                   and re.search(r"[A-Za-z]{3}", line)
                   and line not in {"Question:", "Answer:"}]
    if substantive and max((substantive.count(line) for line in set(substantive)), default=0) >= 5:
        reasons.append("line_repetition_loop")
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", text.lower())
    if len(words) >= 80:
        trigrams = list(zip(words, words[1:], words[2:]))
        if trigrams and len(set(trigrams)) / len(trigrams) < .28: reasons.append("lexical_degeneration")
        if max((words.count(word) for word in set(words)), default=0) / len(words) > .16: reasons.append("word_repetition_loop")
    separator_count = len(re.findall(r"(?:^|\n)\s*(?:[-=_*]){5,}\s*(?:\n|$)", text))
    if separator_count >= 12 or (separator_count >= 4 and separator_count / max(len(lines), 1) > .25):
        reasons.append("separator_abuse")
    if text.rstrip().endswith("\\") and not text.rstrip().endswith("\\\\"):
        reasons.append("malformed_trailing_backslash")
    return reasons


def validate_view(item_dir: Path | str, view: str) -> dict:
    item_dir = Path(item_dir)
    reasons, files = [], []
    assembled = item_dir / ASSEMBLED[view]
    outline = item_dir / OUTLINES[view]
    for kind, path in (("assembled", assembled), ("outline", outline)):
        if not path.is_file(): reasons.append(f"missing_{kind}")
        else: files.append({"kind": kind, "path": str(path), "bytes": path.stat().st_size,
                            "tokens": token_estimate(path.read_text(errors="replace")), "sha256": sha256(path)})
    entries = []
    if outline.is_file():
        try:
            data = json.loads(outline.read_text(encoding="utf-8"))
            schema = SCHEMAS[view]
            key = OUTLINE_KEYS[view]
            # The medical pipeline intentionally uses section/question terminology.
            if view == "textbook" and isinstance(data, dict) and "sections" in data:
                key = "sections"
                schema = {"type": "object", "additionalProperties": False, "required": [key],
                          "properties": {key: {**TEXTBOOK_SCHEMA["properties"]["outline"],
                          "items": {**TEXTBOOK_SCHEMA["properties"]["outline"]["items"],
                          "required": ["section_title", "description", "subtopics"],
                          "properties": {**TEXTBOOK_SCHEMA["properties"]["outline"]["items"]["properties"],
                                         "section_title": {"type": "string", "minLength": 1}}}}}}
                del schema["properties"][key]["items"]["properties"]["chapter_title"]
            elif view == "stackexchange" and isinstance(data, dict) and data.get("questions") and "question" in data["questions"][0]:
                schema = {"type": "object", "additionalProperties": False, "required": ["questions"],
                          "properties": {"questions": {"type": "array", "minItems": 1, "items": {
                          "type": "object", "additionalProperties": False, "required": ["question", "category"],
                          "properties": {"question": {"type": "string", "minLength": 1},
                                         "category": {"type": "string", "minLength": 1}}}}}}
            schema_errors = _schema_errors(data, schema)
            reasons += [f"outline_schema:{error}" for error in schema_errors]
            entries = data.get(key, []) if isinstance(data, dict) else []
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            reasons.append(f"invalid_outline_json:{exc}")
    granular_dir, pattern = GRANULAR_LAYOUTS[view]
    granular = sorted((item_dir / granular_dir).glob("*.txt")) if (item_dir / granular_dir).is_dir() else []
    if not granular: reasons.append("missing_granular")
    if entries and len(entries) != len(granular): reasons.append(f"outline_granular_count:{len(entries)}!={len(granular)}")
    granular_texts = []
    for path in granular:
        text = path.read_text(encoding="utf-8", errors="replace")
        granular_texts.append(text)
        files.append({"kind": "granular", "path": str(path), "bytes": path.stat().st_size,
                      "tokens": token_estimate(text), "sha256": sha256(path)})
        reasons += [f"{path.name}:{reason}" for reason in content_reasons(text)]
    if assembled.is_file():
        text = assembled.read_text(encoding="utf-8", errors="replace")
        reasons += content_reasons(text, assembled=True)
        for path, child in zip(granular, granular_texts):
            # Granular files carry label preambles the assembled corpus does not:
            # a repeated \title{...}/Title: block and/or a "Chapter N:"/"Section N:"
            # prefix the granular writer prepends. Strip those, then compare the
            # substantive body (exact, else whitespace-normalized) against assembled.
            body = child.strip()
            for _ in range(3):
                if "\n\n" in body and body.split("\n\n", 1)[0].lstrip().startswith(("\\title", "Title:")):
                    body = body.split("\n\n", 1)[1].strip(); continue
                label = re.match(r"(?:Chapter|Section)\s+\d+\s*:\s*", body)
                if label:
                    body = body[label.end():].strip(); continue
                break
            normalize = lambda s: re.sub(r"\s+", " ", s).strip()
            if body not in text and normalize(body) not in normalize(text):
                reasons.append(f"assembled_missing_child:{path.name}")
    return {"view": view, "valid": not reasons, "reasons": sorted(set(reasons)), "files": files,
            "outline_count": len(entries), "granular_count": len(granular)}


def manifest_valid(item_dir: Path | str, view: str) -> bool:
    path = Path(item_dir) / "generation_manifest.json"
    if not path.is_file(): return False
    try: record = json.loads(path.read_text())["views"][view]
    except (KeyError, TypeError, json.JSONDecodeError): return False
    report = validate_view(item_dir, view)
    current = {Path(f["path"]).relative_to(item_dir).as_posix(): f["sha256"] for f in report["files"]}
    return report["valid"] and record.get("status") == "validated" and record.get("hashes") == current


def record_validated_view(item_dir: Path | str, view: str, metadata=None) -> dict:
    """Validate a published view and persist its exact hashes/provenance."""
    item_dir = Path(item_dir)
    report = validate_view(item_dir, view)
    if not report["valid"]: raise ValueError("; ".join(report["reasons"]))
    path = item_dir / "generation_manifest.json"
    try: manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError): manifest = {"version": 1, "views": {}}
    hashes = {Path(f["path"]).relative_to(item_dir).as_posix(): f["sha256"] for f in report["files"]}
    manifest.setdefault("views", {})[view] = {"status": "validated", "validated_at": datetime.now(timezone.utc).isoformat(),
                                                "hashes": hashes, "metadata": metadata or {}}
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)
    return report


def publish_transaction(canonical_item: Path | str, staged_item: Path | str, view: str, metadata=None):
    """Validate a complete staged item and atomically replace the canonical item."""
    canonical_item, staged_item = Path(canonical_item), Path(staged_item)
    report = validate_view(staged_item, view)
    if not report["valid"]: raise ValueError("invalid staged view: " + "; ".join(report["reasons"]))
    manifest_path = staged_item / "generation_manifest.json"
    try: manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError): manifest = {"version": 1, "views": {}}
    hashes = {Path(f["path"]).relative_to(staged_item).as_posix(): f["sha256"] for f in report["files"]}
    manifest.setdefault("views", {})[view] = {"status": "validated", "published_at": datetime.now(timezone.utc).isoformat(),
        "hashes": hashes, "metadata": metadata or {}}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    canonical_item.parent.mkdir(parents=True, exist_ok=True)
    backup = canonical_item.with_name(f".{canonical_item.name}.old-{os.getpid()}")
    if backup.exists(): shutil.rmtree(backup)
    if canonical_item.exists(): os.replace(canonical_item, backup)
    try: os.replace(staged_item, canonical_item)
    except BaseException:
        if backup.exists(): os.replace(backup, canonical_item)
        raise
    if backup.exists(): shutil.rmtree(backup)


def make_staged_copy(canonical_item: Path | str) -> Path:
    canonical_item = Path(canonical_item)
    root = Path(tempfile.mkdtemp(prefix=f".{canonical_item.name}.recovery-", dir=canonical_item.parent))
    staged = root / canonical_item.name
    if canonical_item.exists(): shutil.copytree(canonical_item, staged)
    else: staged.mkdir()
    return staged
