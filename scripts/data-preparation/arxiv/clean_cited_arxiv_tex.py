#!/usr/bin/env python3
"""Clean downloaded cited arXiv TeX sources with the repo cleaning pipeline."""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.util
import io
import json
import re
from pathlib import Path
from typing import Any


PREFERRED_MAIN_NAMES = {
    "main": 80,
    "ms": 70,
    "paper": 70,
    "article": 60,
    "manuscript": 60,
    "arxiv": 55,
}

NEGATIVE_NAME_PARTS = {
    "supp": -120,
    "supplement": -120,
    "appendix": -90,
    "appendices": -90,
    "response": -90,
    "rebuttal": -90,
    "revision": -60,
    "cover": -80,
    "template": -90,
    "sample": -90,
    "tikz": -80,
    "fig": -80,
    "figure": -80,
    "table": -60,
    "algorithm": -40,
    "commands": -80,
    "defs": -80,
    "macros": -80,
}


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / ".git").exists():
            return candidate
    return start


def load_cleaning_module(repo_root: Path) -> Any:
    script_path = repo_root / "scripts" / "data-preparation" / "fetch_and_clean" / "pipeline_cleaning_text.py"
    spec = importlib.util.spec_from_file_location("pipeline_cleaning_text", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load cleaning module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def tex_score(path: Path) -> int:
    text = read_text(path)
    lower_name = path.stem.lower()
    score = 0

    if r"\documentclass" in text:
        score += 300
    if r"\begin{document}" in text:
        score += 300
    if r"\title" in text or r"\icmltitle" in text:
        score += 80
    if r"\begin{abstract}" in text or r"\abstract{" in text:
        score += 80
    if r"\section" in text:
        score += 40
    if r"\bibliography" in text or r"\begin{thebibliography}" in text:
        score += 30

    score += PREFERRED_MAIN_NAMES.get(lower_name, 0)
    for part, penalty in NEGATIVE_NAME_PARTS.items():
        if part in lower_name:
            score += penalty

    score += min(len(text) // 5000, 40)
    return score


def select_main_tex(source_dir: Path) -> tuple[Path | None, list[dict[str, Any]]]:
    candidates = []
    for path in sorted(source_dir.rglob("*.tex")):
        if path.name.startswith("."):
            continue
        score = tex_score(path)
        candidates.append(
            {
                "path": str(path.relative_to(source_dir)),
                "score": score,
                "bytes": path.stat().st_size,
            }
        )

    if not candidates:
        return None, candidates

    candidates.sort(key=lambda item: (item["score"], item["bytes"]), reverse=True)
    best = source_dir / candidates[0]["path"]
    best_text = read_text(best)
    if r"\begin{document}" not in best_text and r"\documentclass" not in best_text:
        return None, candidates
    return best, candidates[:10]


def resolve_include_path(source_dir: Path, current_file: Path, include_name: str) -> Path | None:
    include_name = include_name.strip()
    if not include_name or include_name.startswith(("http://", "https://")):
        return None

    raw = Path(include_name)
    candidates: list[Path] = []
    if raw.suffix:
        candidates.append(current_file.parent / raw)
        candidates.append(source_dir / raw)
    else:
        candidates.extend(
            [
                current_file.parent / f"{include_name}.tex",
                current_file.parent / include_name,
                source_dir / f"{include_name}.tex",
                source_dir / include_name,
            ]
        )

    for candidate in candidates:
        try:
            resolved = candidate.resolve()
            resolved.relative_to(source_dir.resolve())
        except (OSError, ValueError):
            continue
        if resolved.exists() and resolved.is_file():
            return resolved
    return None


def inline_local_inputs(path: Path, source_dir: Path, seen: set[Path] | None = None) -> str:
    if seen is None:
        seen = set()
    resolved_path = path.resolve()
    if resolved_path in seen:
        return ""
    seen.add(resolved_path)

    text = read_text(path)
    pattern = re.compile(r"\\(?:input|include|subfile)\s*(?:\{([^{}]+)\}|([^\s%{}]+))")

    def replacer(match: re.Match[str]) -> str:
        include_name = match.group(1) or match.group(2) or ""
        include_path = resolve_include_path(source_dir, path, include_name)
        if include_path is None:
            return match.group(0)
        return "\n" + inline_local_inputs(include_path, source_dir, seen) + "\n"

    return pattern.sub(replacer, text)


def extract_balanced_argument(text: str, cursor: int) -> tuple[str | None, int]:
    while cursor < len(text) and text[cursor].isspace():
        cursor += 1
    if cursor >= len(text) or text[cursor] != "{":
        return None, cursor
    depth = 1
    start = cursor + 1
    cursor += 1
    while cursor < len(text) and depth > 0:
        if text[cursor] == "{":
            depth += 1
        elif text[cursor] == "}":
            depth -= 1
        cursor += 1
    if depth != 0:
        return None, cursor
    return text[start : cursor - 1], cursor


def extract_delimited_argument(text: str, cursor: int, open_char: str, close_char: str) -> tuple[str | None, int]:
    while cursor < len(text) and text[cursor].isspace():
        cursor += 1
    if cursor >= len(text) or text[cursor] != open_char:
        return None, cursor
    depth = 1
    start = cursor + 1
    cursor += 1
    while cursor < len(text) and depth > 0:
        if text[cursor] == open_char:
            depth += 1
        elif text[cursor] == close_char:
            depth -= 1
        cursor += 1
    if depth != 0:
        return None, cursor
    return text[start : cursor - 1], cursor


def replace_latex_command(
    text: str,
    command_pattern: str,
    num_args: int,
    replacement,
    *,
    allow_optional_args: bool = False,
) -> str:
    pattern = re.compile(r"\\" + command_pattern + r"(?![A-Za-z])")
    parts: list[str] = []
    cursor = 0
    while True:
        match = pattern.search(text, cursor)
        if not match:
            parts.append(text[cursor:])
            break

        arg_cursor = match.end()
        optional_args: list[str] = []
        if allow_optional_args:
            while True:
                optional, next_cursor = extract_delimited_argument(text, arg_cursor, "[", "]")
                if optional is None:
                    break
                optional_args.append(optional)
                arg_cursor = next_cursor

        args: list[str] = []
        parse_failed = False
        for _ in range(num_args):
            arg, next_cursor = extract_balanced_argument(text, arg_cursor)
            if arg is None:
                parse_failed = True
                break
            args.append(arg)
            arg_cursor = next_cursor

        if parse_failed:
            parts.append(text[cursor : match.end()])
            cursor = match.end()
            continue

        parts.append(text[cursor : match.start()])
        parts.append(replacement(optional_args, args))
        cursor = arg_cursor
    return "".join(parts)


def resolve_arxiv_toggles(text: str) -> str:
    result_parts: list[str] = []
    cursor = 0
    pattern = re.compile(r"\\iftoggle\s*")
    while True:
        match = pattern.search(text, cursor)
        if not match:
            result_parts.append(text[cursor:])
            break
        name, after_name = extract_balanced_argument(text, match.end())
        true_branch, after_true = extract_balanced_argument(text, after_name)
        false_branch, after_false = extract_balanced_argument(text, after_true)
        if name is None or true_branch is None:
            result_parts.append(text[cursor : match.end()])
            cursor = match.end()
            continue
        result_parts.append(text[cursor : match.start()])
        result_parts.append(true_branch if name.strip() == "arxiv" else (false_branch or ""))
        cursor = after_false
    return "".join(result_parts)


def normalize_cited_latex_commands(text: str) -> str:
    """Normalize recurring citation/reference/layout aliases before the shared cleaner."""
    text = replace_latex_command(
        text,
        r"(?:paren|text)cite",
        1,
        lambda _optional, args: rf"\citep{{{args[0]}}}",
        allow_optional_args=True,
    )
    text = replace_latex_command(
        text,
        r"[Cc]ref",
        1,
        lambda _optional, args: rf"\ref{{{args[0]}}}",
    )
    text = replace_latex_command(
        text,
        r"refsec",
        1,
        lambda _optional, args: rf"Section~\ref{{{args[0]}}}",
    )
    text = replace_latex_command(
        text,
        r"hypertarget",
        2,
        lambda _optional, args: args[1],
    )

    text = re.sub(r"\\(?:adj)?includegraphics\s*(?:\[[^\[\]]*\]\s*)?\{[^{}]*\}", "", text)
    text = re.sub(r"\\[vh]space\*?\s*\{[^{}]*\}", "", text)
    text = re.sub(r"\\dash\s*(?:\{\})?", "--", text)
    text = re.sub(r"\\eg(?![A-Za-z])\s*~?", "e.g. ", text)
    text = re.sub(r"\\ie(?![A-Za-z])\s*~?", "i.e. ", text)
    text = re.sub(r"\\etc(?![A-Za-z])\s*~?", "etc. ", text)
    return text


def clean_one_source(
    source_dir: Path,
    output_dir: Path,
    cleaning_module: Any,
    *,
    force: bool,
) -> dict[str, Any]:
    arxiv_id = source_dir.name
    output_path = output_dir / f"{arxiv_id}.tex"
    if output_path.exists() and not force:
        return {
            "arxiv_id": arxiv_id,
            "status": "skipped_existing",
            "output_path": str(output_path),
        }

    main_path, candidates = select_main_tex(source_dir)
    if main_path is None:
        return {
            "arxiv_id": arxiv_id,
            "status": "no_main_tex",
            "candidates": candidates,
        }

    try:
        merged_text = inline_local_inputs(main_path, source_dir)
        merged_text = resolve_arxiv_toggles(merged_text)
        merged_text = normalize_cited_latex_commands(merged_text)
        with contextlib.redirect_stdout(io.StringIO()):
            cleaned_text = cleaning_module.clean_latex(merged_text)
        cleaned_text = normalize_cited_latex_commands(cleaned_text)
        if not cleaned_text.strip():
            return {
                "arxiv_id": arxiv_id,
                "status": "empty_cleaned_text",
                "main_tex": str(main_path.relative_to(source_dir)),
                "candidates": candidates,
            }
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path.write_text(cleaned_text + "\n", encoding="utf-8")
        return {
            "arxiv_id": arxiv_id,
            "status": "ok",
            "main_tex": str(main_path.relative_to(source_dir)),
            "output_path": str(output_path),
            "cleaned_chars": len(cleaned_text),
            "candidates": candidates,
        }
    except Exception as exc:  # noqa: BLE001 - record per-paper failures and keep going.
        return {
            "arxiv_id": arxiv_id,
            "status": "error",
            "error": str(exc),
            "main_tex": str(main_path.relative_to(source_dir)),
            "candidates": candidates,
        }


def write_manifest_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["arxiv_id", "status", "main_tex", "output_path", "cleaned_chars", "error"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def parse_args() -> argparse.Namespace:
    repo_root = find_repo_root(Path(__file__).resolve())
    parser = argparse.ArgumentParser(description="Clean cited arXiv source packages into cited/cleaned.")
    parser.add_argument("--sources-dir", type=Path, default=repo_root / "data" / "arxiv" / "cited" / "sources")
    parser.add_argument("--output-dir", type=Path, default=repo_root / "data" / "arxiv" / "cited" / "cleaned")
    parser.add_argument("--manifest", type=Path, default=repo_root / "data" / "arxiv" / "cited" / "cleaned_manifest.json")
    parser.add_argument("--only", nargs="*", help="Optional arXiv IDs/source directory names to clean.")
    parser.add_argument("--limit", type=int, help="Optional limit for smoke tests.")
    parser.add_argument("--force", action="store_true", help="Reclean files that already exist.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = find_repo_root(Path(__file__).resolve())
    cleaning_module = load_cleaning_module(repo_root)

    selected = set(args.only) if args.only else None
    source_dirs = [path for path in sorted(args.sources_dir.iterdir()) if path.is_dir()]
    if selected:
        source_dirs = [path for path in source_dirs if path.name in selected]
    if args.limit is not None:
        source_dirs = source_dirs[: args.limit]

    rows = []
    for index, source_dir in enumerate(source_dirs, start=1):
        print(f"[{index}/{len(source_dirs)}] Cleaning {source_dir.name}")
        row = clean_one_source(source_dir, args.output_dir, cleaning_module, force=args.force)
        print(f"  -> {row['status']}")
        rows.append(row)

    write_json(args.manifest, rows)
    write_manifest_csv(args.manifest.with_suffix(".csv"), rows)

    status_counts: dict[str, int] = {}
    for row in rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    print(f"Done. Status counts: {status_counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
