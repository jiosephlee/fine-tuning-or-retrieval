#!/usr/bin/env python3
"""Backfill fixed 5-shot MCQA prompt columns into v14 probe CSVs."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from utils.mcqa_prompts import (  # noqa: E402
    MCQA_5SHOT_PROMPT_COLUMN,
    build_mcqa_5shot_prompt,
    validate_mcqa_5shot_demonstrations,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add formatted_question_5shot to probes_v14_mcqa.csv files."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT / "probes",
        help="Probe root containing <source>/<document>/facts/probes_v14_mcqa.csv files.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v14",
        help="Probe version to backfill.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report files that would be updated without writing them.",
    )
    return parser.parse_args()


def iter_mcqa_csvs(root: Path, version: str) -> list[Path]:
    return sorted(root.glob(f"*/*/facts/probes_{version}_mcqa.csv"))


def backfill_csv(path: Path, dry_run: bool) -> tuple[int, bool]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if "formatted_question" not in fieldnames:
        raise ValueError(f"{path} is missing required column formatted_question")

    new_fieldnames = list(fieldnames)
    if MCQA_5SHOT_PROMPT_COLUMN not in new_fieldnames:
        new_fieldnames.append(MCQA_5SHOT_PROMPT_COLUMN)

    changed = False
    for row in rows:
        prompt = build_mcqa_5shot_prompt(row["formatted_question"])
        if row.get(MCQA_5SHOT_PROMPT_COLUMN) != prompt:
            row[MCQA_5SHOT_PROMPT_COLUMN] = prompt
            changed = True

    if changed and not dry_run:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=new_fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    return len(rows), changed


def main() -> None:
    args = parse_args()
    validate_mcqa_5shot_demonstrations()

    paths = iter_mcqa_csvs(args.root, args.version)
    if not paths:
        raise SystemExit(f"No probes_{args.version}_mcqa.csv files found under {args.root}")

    total_rows = 0
    changed_files = 0
    for path in paths:
        row_count, changed = backfill_csv(path, args.dry_run)
        total_rows += row_count
        changed_files += int(changed)
        status = "would update" if args.dry_run and changed else "updated" if changed else "unchanged"
        print(f"{status}: {path} ({row_count} rows)")

    action = "would update" if args.dry_run else "updated"
    print(
        f"\nScanned {len(paths)} files / {total_rows} rows; "
        f"{action} {changed_files} files."
    )


if __name__ == "__main__":
    main()
