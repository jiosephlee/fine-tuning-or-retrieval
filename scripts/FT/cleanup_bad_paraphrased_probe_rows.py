import argparse
from pathlib import Path

import pandas as pd


DEFAULT_BAD_PROBE_INDICES = (60, 188)


def prune_metric_file(path: Path, bad_indices: set[int], dry_run: bool) -> tuple[int, int] | None:
    df = pd.read_csv(path)
    if "probe_index" not in df.columns:
        return None

    before = len(df)
    pruned = df[~df["probe_index"].isin(bad_indices)]
    after = len(pruned)
    if after != before and not dry_run:
        pruned.to_csv(path, index=False)
    return before, after


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove known malformed OFT paraphrased probe rows from saved metrics."
    )
    parser.add_argument(
        "--results-root",
        default="results/FT/full",
        help="Root to scan for saved result CSVs.",
    )
    parser.add_argument(
        "--bad-probe-index",
        type=int,
        action="append",
        default=list(DEFAULT_BAD_PROBE_INDICES),
        help="Probe index to remove. Can be passed multiple times.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    bad_indices = set(args.bad_probe_index)
    metric_paths = sorted(results_root.glob("**/OFT_knowledge_probe_paraphrased_metrics.csv"))

    changed = 0
    for path in metric_paths:
        result = prune_metric_file(path, bad_indices, args.dry_run)
        if result is None:
            continue
        before, after = result
        if before != after:
            changed += 1
            action = "would prune" if args.dry_run else "pruned"
            print(f"{action}: {path}: {before} -> {after}")

    print(f"metric files scanned: {len(metric_paths)}")
    print(f"metric files changed: {changed}")


if __name__ == "__main__":
    main()
