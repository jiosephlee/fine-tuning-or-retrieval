#!/usr/bin/env python3
"""Find training-run dirs under results/ that are empty or only have step-0 probe metrics."""
from __future__ import annotations
import csv
import os
import re
import sys
from pathlib import Path

RESULTS_ROOT = Path("results")
E_NAME = re.compile(r"^E\d+_")
# bundle names treated as one-shot reeval dumps (skip)
REEVAL_BUNDLE_RE = re.compile(r"^reeval(_|$)")


def is_training_dir(p: Path) -> bool:
    return p.is_dir() and E_NAME.match(p.name) is not None


def iter_metric_csvs(run_dir: Path):
    """Yield every *_metrics.csv (training_loss_* + probe metrics) outside reeval subtrees."""
    for root, dirs, files in os.walk(run_dir):
        rel = Path(root).relative_to(run_dir)
        dirs[:] = [d for d in dirs if not REEVAL_BUNDLE_RE.match(d)]
        if any(REEVAL_BUNDLE_RE.match(part) for part in rel.parts):
            continue
        for f in files:
            if f.endswith("_metrics.csv"):
                yield Path(root) / f


def steps_in_csv(path: Path) -> set[int]:
    out: set[int] = set()
    try:
        with path.open() as fh:
            r = csv.DictReader(fh)
            if "step" not in (r.fieldnames or []):
                return out
            for row in r:
                try:
                    out.add(int(float(row["step"])))
                except (TypeError, ValueError):
                    continue
    except OSError:
        pass
    return out


def classify(run_dir: Path) -> str | None:
    any_csv = False
    for csv_path in iter_metric_csvs(run_dir):
        any_csv = True
        steps = steps_in_csv(csv_path)
        if any(s > 0 for s in steps):
            return None  # has real training steps somewhere
    return "step0_only" if any_csv else "empty"


def main():
    candidates: list[tuple[Path, str]] = []
    for run_dir in RESULTS_ROOT.rglob("E*"):
        if not is_training_dir(run_dir):
            continue
        kind = classify(run_dir)
        if kind:
            candidates.append((run_dir, kind))
    candidates.sort()
    for p, k in candidates:
        sz = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / (1024**2)
        print(f"{k:11s} {sz:8.2f} MiB  {p}")
    print(f"\nTotal candidates: {len(candidates)}")
    print(f"Total size: {sum(sum(f.stat().st_size for f in p.rglob('*') if f.is_file()) for p,_ in candidates)/(1024**3):.2f} GiB")


if __name__ == "__main__":
    sys.exit(main() or 0)
