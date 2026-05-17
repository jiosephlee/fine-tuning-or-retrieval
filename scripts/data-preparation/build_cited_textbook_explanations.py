#!/usr/bin/env python3
"""Materialize cited works as granular textbook-style explanation files.

This script does not call any external APIs. It reuses already-cleaned cited
works and writes them under:

  data/arxiv/explanations/{domain}/cited_textbooks/
  data/legal/explanations/{domain}/cited_textbooks/

The training loader can then include them with:

  --with_specific_explanation textbooks cited_textbooks

Medical is intentionally unsupported here.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SUPPORTED_SOURCES = {"arxiv", "legal"}
DEFAULT_OUTPUT_SUBDIR = "cited_textbooks"


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / ".git").exists():
            return candidate
    return start


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def safe_name(value: str) -> str:
    value = value.replace("/", "_")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("._")
    return value or "unknown"


def reset_output_dir(output_dir: Path, *, force: bool, dry_run: bool) -> None:
    if not output_dir.exists():
        return
    if not force:
        raise FileExistsError(f"{output_dir} already exists; pass --force to replace it")
    if dry_run:
        return
    shutil.rmtree(output_dir)


def write_cited_file(
    output_path: Path,
    *,
    header: str,
    body: str,
    dry_run: bool,
) -> None:
    if dry_run:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(header.rstrip() + "\n\n" + body.strip() + "\n", encoding="utf-8")


def build_arxiv(repo_root: Path, output_subdir: str, min_chars: int, force: bool, dry_run: bool) -> list[dict[str, Any]]:
    cited_root = repo_root / "data" / "arxiv" / "cited"
    edges_path = cited_root / "citation_edges.csv"
    cleaned_dir = cited_root / "cleaned"
    cited_meta_path = cited_root / "cited_papers.json"

    if not edges_path.exists():
        raise FileNotFoundError(f"Missing arXiv citation edges: {edges_path}")
    if not cleaned_dir.is_dir():
        raise FileNotFoundError(f"Missing cleaned arXiv cited works dir: {cleaned_dir}")

    cited_meta = json.loads(cited_meta_path.read_text(encoding="utf-8")) if cited_meta_path.exists() else {}
    by_seed: dict[str, list[dict[str, str]]] = defaultdict(list)
    with edges_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            seed = row.get("seed_local_id", "").strip()
            cited_id = row.get("cited_arxiv_id", "").strip()
            if seed and cited_id:
                by_seed[seed].append(row)

    manifest: list[dict[str, Any]] = []
    for seed in sorted(by_seed):
        output_dir = repo_root / "data" / "arxiv" / "explanations" / seed / output_subdir
        reset_output_dir(output_dir, force=force, dry_run=dry_run)

        seen_ids: set[str] = set()
        written_idx = 0
        skipped = Counter()

        for row in by_seed[seed]:
            cited_id = row["cited_arxiv_id"].strip()
            if cited_id in seen_ids:
                skipped["duplicate_edge"] += 1
                continue
            seen_ids.add(cited_id)

            source_path = cleaned_dir / f"{cited_id}.tex"
            if not source_path.exists():
                skipped["missing_cleaned"] += 1
                continue

            text = read_text(source_path).strip()
            if len(text) < min_chars:
                skipped["too_short"] += 1
                continue

            written_idx += 1
            meta = cited_meta.get(cited_id, {})
            title = row.get("cited_title") or meta.get("title") or cited_id
            year = row.get("cited_year") or meta.get("year") or ""
            output_name = f"cited_{written_idx:03d}_{safe_name(cited_id)}.txt"
            output_path = output_dir / output_name
            header = (
                f"# Cited arXiv Work: {title}\n\n"
                f"- Source domain: {seed}\n"
                f"- arXiv ID: {cited_id}\n"
                f"- Year: {year}\n"
                f"- Explanation type: cited_textbooks"
            )
            write_cited_file(output_path, header=header, body=text, dry_run=dry_run)
            manifest.append(
                {
                    "source": "arxiv",
                    "domain": seed,
                    "cited_id": cited_id,
                    "title": title,
                    "year": year,
                    "input_path": str(source_path.relative_to(repo_root)),
                    "output_path": str(output_path.relative_to(repo_root)),
                    "chars": len(text),
                }
            )

        print(
            f"arxiv/{seed}: wrote {written_idx} cited_textbooks files"
            + (f" (skipped {dict(skipped)})" if skipped else "")
        )

    return manifest


def build_legal(repo_root: Path, output_subdir: str, min_chars: int, force: bool, dry_run: bool) -> list[dict[str, Any]]:
    cleaned_root = repo_root / "data" / "legal" / "cited" / "cleaned"
    if not cleaned_root.is_dir():
        raise FileNotFoundError(f"Missing cleaned legal cited opinions dir: {cleaned_root}")

    manifest: list[dict[str, Any]] = []
    for case_dir in sorted(path for path in cleaned_root.iterdir() if path.is_dir()):
        case_name = case_dir.name
        output_dir = repo_root / "data" / "legal" / "explanations" / case_name / output_subdir
        reset_output_dir(output_dir, force=force, dry_run=dry_run)

        written_idx = 0
        skipped = Counter()
        for source_path in sorted(case_dir.glob("cited_*.txt")):
            text = read_text(source_path).strip()
            if len(text) < min_chars:
                skipped["too_short"] += 1
                continue

            written_idx += 1
            opinion_id = source_path.stem.replace("cited_", "")
            output_name = f"cited_{written_idx:03d}_{safe_name(opinion_id)}.txt"
            output_path = output_dir / output_name
            header = (
                f"# Cited Legal Opinion: {opinion_id}\n\n"
                f"- Source domain: {case_name}\n"
                f"- Opinion ID: {opinion_id}\n"
                f"- Explanation type: cited_textbooks"
            )
            write_cited_file(output_path, header=header, body=text, dry_run=dry_run)
            manifest.append(
                {
                    "source": "legal",
                    "domain": case_name,
                    "cited_id": opinion_id,
                    "input_path": str(source_path.relative_to(repo_root)),
                    "output_path": str(output_path.relative_to(repo_root)),
                    "chars": len(text),
                }
            )

        print(
            f"legal/{case_name}: wrote {written_idx} cited_textbooks files"
            + (f" (skipped {dict(skipped)})" if skipped else "")
        )

    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add cleaned cited arXiv papers and legal opinions as granular textbook-style explanations."
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["arxiv", "legal"],
        help="Sources to materialize. Supported: arxiv legal. Medical is intentionally unsupported.",
    )
    parser.add_argument(
        "--output-subdir",
        default=DEFAULT_OUTPUT_SUBDIR,
        help="Explanation subfolder to create under each domain.",
    )
    parser.add_argument("--min-chars", type=int, default=1000, help="Skip cited works shorter than this many chars.")
    parser.add_argument("--force", action="store_true", help="Replace existing output subfolders.")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be written without writing files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested_sources = [source.lower() for source in args.sources]
    unsupported = sorted(set(requested_sources) - SUPPORTED_SOURCES)
    if unsupported:
        raise SystemExit(
            f"Unsupported sources: {unsupported}. This script supports only arxiv and legal; medical is excluded."
        )

    repo_root = find_repo_root(Path.cwd())
    all_rows: list[dict[str, Any]] = []
    if "arxiv" in requested_sources:
        all_rows.extend(build_arxiv(repo_root, args.output_subdir, args.min_chars, args.force, args.dry_run))
    if "legal" in requested_sources:
        all_rows.extend(build_legal(repo_root, args.output_subdir, args.min_chars, args.force, args.dry_run))

    summary = Counter(row["source"] for row in all_rows)
    print(f"Total cited_textbooks files: {sum(summary.values())} ({dict(summary)})")

    if not args.dry_run:
        manifest_path = repo_root / "data" / "cited_textbook_explanations_manifest.json"
        write_json(manifest_path, all_rows)
        print(f"Wrote manifest: {manifest_path.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
