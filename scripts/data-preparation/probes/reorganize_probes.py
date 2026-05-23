#!/usr/bin/env python3
"""Tidy probes/<group>/<doc>/{facts,inference}/ directories:

1. Keep only the "latest" canonical artifacts at the top level. Everything
   else (older probe versions, fix logs, debugging/ checkpoints/ etc.)
   moves into a sibling ``intermediate/`` subdirectory.
2. (Re)generate a ``<basename>_readable.txt`` next to each kept main CSV
   so the latest set can be audited at a glance.

What counts as a kept basename:
  facts/    -> probes_v14_short_targets, probes_v13_paraphrased,
               probes_v15_mcqa, probes_v16_mcqa
  inference -> probes_v11_reviewed, probes_v14_mcqa,
               comprehension_mcqa, comprehension_mcqa_hardened

A file is kept at the top level iff its name starts with one of the kept
basenames (so sidecars like ``.audit.csv``, ``_metrics.txt``,
``_readable.txt``, ``.pre_naturalness_repair.csv`` ride along automatically).

Run with --dry-run first to preview every move.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
PROBE_ROOT = REPO_ROOT / "probes"

FACTS_KEEP = (
    "probes_v14_short_targets",
    "probes_v14_paraphrased",
    "probes_v15_mcqa",
    "probes_v16_mcqa",
)
INFERENCE_KEEP = (
    "probes_v11_reviewed",
    "probes_v14_mcqa",
    "comprehension_mcqa",
    "comprehension_mcqa_hardened",
)
INTERMEDIATE_DIR = "intermediate"


@dataclass
class Plan:
    keep: list[Path]
    move: list[Path]
    readable_to_write: list[Path]  # main CSVs that should get a readable.txt


def categorize_dir(dir_path: Path, keep_bases: tuple[str, ...]) -> Plan:
    keep: list[Path] = []
    move: list[Path] = []
    readable_to_write: list[Path] = []
    for entry in sorted(dir_path.iterdir()):
        if entry.name == INTERMEDIATE_DIR:
            continue
        if any(entry.name.startswith(base) for base in keep_bases):
            keep.append(entry)
            # Trigger readable.txt only for the main CSV (not audit/sidecar).
            if entry.suffix == ".csv" and entry.stem in keep_bases:
                readable_to_write.append(entry)
        else:
            move.append(entry)
    return Plan(keep=keep, move=move, readable_to_write=readable_to_write)


# ------------- readable.txt writers ----------------------------------------


def write_simple_probe_readable(csv_path: Path) -> str:
    df = pd.read_csv(csv_path, keep_default_na=False)
    if "probe" not in df.columns or "target" not in df.columns:
        return ""
    out = csv_path.with_name(f"{csv_path.stem}_readable.txt")
    with out.open("w") as f:
        for _, row in df.iterrows():
            target = str(row["target"]).lstrip()
            f.write(f"{str(row['probe']).strip()}: {target}\n")
    return out.name


def write_mcqa_readable(csv_path: Path) -> str:
    df = pd.read_csv(csv_path, keep_default_na=False)
    if "formatted_question" not in df.columns:
        return ""
    out = csv_path.with_name(f"{csv_path.stem}_readable.txt")
    with out.open("w") as f:
        for i, row in df.iterrows():
            stem = str(row["formatted_question"]).rstrip()
            answer = str(row.get("correct_label", "")).strip()
            section = str(row.get("section", "")).strip()
            header = f"# row {i}"
            if section:
                header += f"  (section: {section})"
            f.write(f"{header}\n{stem}\nAnswer: {answer}\n\n")
    return out.name


def write_comprehension_mcqa_readable(csv_path: Path) -> str:
    """``comprehension_mcqa[_hardened].csv`` stores 'contextualized_question'
    (which is a formatted MCQA block) and 'answer' (the label)."""
    df = pd.read_csv(csv_path, keep_default_na=False)
    col = "contextualized_question" if "contextualized_question" in df.columns else "question"
    if col not in df.columns:
        return ""
    out = csv_path.with_name(f"{csv_path.stem}_readable.txt")
    with out.open("w") as f:
        for i, row in df.iterrows():
            stem = str(row[col]).rstrip()
            answer = str(row.get("answer", "")).strip()
            f.write(f"# row {i}\n{stem}\nAnswer: {answer}\n\n")
    return out.name


def writer_for(csv_path: Path):
    name = csv_path.name
    if name.startswith("comprehension_mcqa"):
        return write_comprehension_mcqa_readable
    if "_mcqa" in name:
        return write_mcqa_readable
    return write_simple_probe_readable


# ------------- execution ---------------------------------------------------


def execute_plan(dir_path: Path, plan: Plan, dry_run: bool) -> dict:
    actions = {"moved": [], "readable": [], "kept": [e.name for e in plan.keep]}
    if not plan.move and not plan.readable_to_write:
        return actions

    intermediate = dir_path / INTERMEDIATE_DIR
    for entry in plan.move:
        dest = intermediate / entry.name
        actions["moved"].append((entry.name, str(dest.relative_to(REPO_ROOT))))
        if not dry_run:
            intermediate.mkdir(exist_ok=True)
            if dest.exists():
                # Merge contents if directory; otherwise replace.
                if entry.is_dir() and dest.is_dir():
                    for sub in entry.iterdir():
                        sub_dest = dest / sub.name
                        if sub_dest.exists():
                            shutil.rmtree(sub_dest) if sub.is_dir() else sub_dest.unlink()
                        shutil.move(str(sub), str(sub_dest))
                    entry.rmdir()
                    continue
                if dest.is_dir():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()
            shutil.move(str(entry), str(dest))

    for csv_path in plan.readable_to_write:
        writer = writer_for(csv_path)
        out_name = f"{csv_path.stem}_readable.txt"
        actions["readable"].append(out_name)
        if not dry_run:
            writer(csv_path)
    return actions


def discover_targets(scope: tuple[str, ...]) -> list[tuple[Path, tuple[str, ...]]]:
    targets: list[tuple[Path, tuple[str, ...]]] = []
    for group_dir in sorted(PROBE_ROOT.iterdir()):
        if not group_dir.is_dir():
            continue
        for doc_dir in sorted(group_dir.iterdir()):
            if not doc_dir.is_dir():
                continue
            if "facts" in scope:
                facts = doc_dir / "facts"
                if facts.is_dir():
                    targets.append((facts, FACTS_KEEP))
            if "inference" in scope:
                inf = doc_dir / "inference"
                if inf.is_dir():
                    targets.append((inf, INFERENCE_KEEP))
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Preview only (default).")
    parser.add_argument("--apply", action="store_true",
                        help="Actually perform the moves and write readable.txt files.")
    parser.add_argument("--scope", choices=("facts", "inference", "both"),
                        default="both")
    parser.add_argument("--document", default=None,
                        help="Only process directories whose path contains this string")
    args = parser.parse_args()
    dry_run = not args.apply

    scope = ("facts", "inference") if args.scope == "both" else (args.scope,)
    targets = discover_targets(scope)
    if args.document:
        targets = [t for t in targets if args.document in str(t[0])]

    print(f"{'DRY-RUN' if dry_run else 'APPLY'}: {len(targets)} directories\n")
    grand_moves = 0
    grand_readables = 0
    for dir_path, keep_bases in targets:
        plan = categorize_dir(dir_path, keep_bases)
        if not plan.move and not plan.readable_to_write:
            continue
        rel = dir_path.relative_to(REPO_ROOT)
        print(f"── {rel}")
        print(f"   keep ({len(plan.keep)}): {[e.name for e in plan.keep]}")
        if plan.readable_to_write:
            for csv_path in plan.readable_to_write:
                print(f"   readable -> {csv_path.stem}_readable.txt")
        if plan.move:
            print(f"   move ({len(plan.move)}) -> {INTERMEDIATE_DIR}/")
            for entry in plan.move:
                kind = "DIR " if entry.is_dir() else "    "
                print(f"      {kind}{entry.name}")
        actions = execute_plan(dir_path, plan, dry_run)
        grand_moves += len(actions["moved"])
        grand_readables += len(actions["readable"])
        print()

    print(f"Totals: {grand_moves} entries moved, {grand_readables} readable.txt files")
    if dry_run:
        print("(dry-run — pass --apply to make changes)")


if __name__ == "__main__":
    csv.field_size_limit(10_000_000)
    main()
