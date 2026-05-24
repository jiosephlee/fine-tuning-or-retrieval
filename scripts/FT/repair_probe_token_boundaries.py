#!/usr/bin/env python
"""Repair probe CSV rows so probe/target splits are tokenizer-exact.

For each CSV row with probe, target, and fact columns, this script enforces:

    encode(probe) + encode(target) == encode(fact)

It only rewrites files that need changes. The preferred repair keeps the probe
fixed and replaces target with the exact suffix of fact; fallback repairs move
the split to the nearest tokenizer-safe character boundary in fact.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, Optional, Sequence

from transformers import AutoTokenizer


REQUIRED_COLUMNS = ("probe", "target", "fact")


def encode(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(str(text), add_special_tokens=False)


def token_boundary_ok(tokenizer, probe: str, target: str, fact: str) -> bool:
    return encode(tokenizer, probe) + encode(tokenizer, target) == encode(tokenizer, fact)


def nearest_token_safe_split(tokenizer, fact: str, preferred_split: int) -> Optional[int]:
    positions = range(len(fact) + 1)
    for split in sorted(positions, key=lambda pos: (abs(pos - preferred_split), pos)):
        if token_boundary_ok(tokenizer, fact[:split], fact[split:], fact):
            return split
    return None


def repair_row(tokenizer, row: dict[str, str]) -> tuple[dict[str, str], bool]:
    probe = str(row["probe"])
    target = str(row["target"])
    fact = str(row["fact"])

    if token_boundary_ok(tokenizer, probe, target, fact):
        return row, False

    if fact.startswith(probe):
        repaired = dict(row)
        repaired["target"] = fact[len(probe):]
        if token_boundary_ok(tokenizer, repaired["probe"], repaired["target"], fact):
            return repaired, True

    if target and fact.endswith(target):
        repaired = dict(row)
        repaired["probe"] = fact[:-len(target)]
        if token_boundary_ok(tokenizer, repaired["probe"], repaired["target"], fact):
            return repaired, True

    split = nearest_token_safe_split(tokenizer, fact, len(probe))
    if split is not None:
        repaired = dict(row)
        repaired["probe"] = fact[:split]
        repaired["target"] = fact[split:]
        return repaired, True

    return row, False


def iter_probe_csvs(root: Path, patterns: Sequence[str]) -> Iterable[Path]:
    seen: set[Path] = set()
    for pattern in patterns:
        for path in root.glob(pattern):
            if path.is_file() and path not in seen:
                seen.add(path)
                yield path


def repair_csv(path: Path, tokenizer, dry_run: bool) -> tuple[int, int, list[int]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or any(column not in reader.fieldnames for column in REQUIRED_COLUMNS):
            return 0, 0, []
        rows = list(reader)
        fieldnames = list(reader.fieldnames)

    changed = 0
    unrepaired: list[int] = []
    repaired_rows = []
    for idx, row in enumerate(rows):
        repaired, did_change = repair_row(tokenizer, row)
        if did_change:
            changed += 1
        elif not token_boundary_ok(tokenizer, row["probe"], row["target"], row["fact"]):
            unrepaired.append(idx)
        repaired_rows.append(repaired)

    if changed and not dry_run:
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(repaired_rows)

    return len(rows), changed, unrepaired


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("probes"))
    parser.add_argument("--model-id", default="allenai/OLMo-2-1124-7B")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--pattern",
        action="append",
        default=None,
        help="Glob below --root. May be passed multiple times.",
    )
    args = parser.parse_args()
    if args.pattern is None:
        args.pattern = [
            "**/facts/probes_*.csv",
            "**/inference/probes_*.csv",
        ]
    return args


def main() -> int:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True, use_fast=True)

    files_seen = 0
    rows_seen = 0
    rows_changed = 0
    failures: list[str] = []

    for path in iter_probe_csvs(args.root, args.pattern):
        files_seen += 1
        row_count, changed, unrepaired = repair_csv(path, tokenizer, args.dry_run)
        rows_seen += row_count
        rows_changed += changed
        if changed:
            action = "would repair" if args.dry_run else "repaired"
            print(f"{action} {changed} rows in {path}")
        if unrepaired:
            failures.append(f"{path}: unrepaired rows {unrepaired[:20]}")

    print(f"checked {rows_seen} rows across {files_seen} files; repaired {rows_changed} rows")
    if failures:
        print("unrepaired token-boundary failures:")
        for failure in failures[:50]:
            print(f"  - {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
