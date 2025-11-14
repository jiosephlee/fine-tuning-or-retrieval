#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import re
import sys

CITE_PATTERN = re.compile(r"\[cite:[^\]]*]")


def iter_files(paths: list[pathlib.Path]) -> pathlib.Path:
    for path in paths:
        if path.is_dir():
            for child in path.rglob("*"):
                if child.is_file():
                    yield child
        else:
            yield path


def clean_file(path: pathlib.Path, dry_run: bool) -> bool:
    text = path.read_text(encoding="utf-8")
    cleaned = CITE_PATTERN.sub("", text)
    if cleaned == text:
        return False
    if dry_run:
        print(f"[dry run] would clean {path}")
        return True
    path.write_text(cleaned, encoding="utf-8")
    print(f"cleaned {path}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Remove [cite: X] markers.")
    parser.add_argument("--targets", nargs="+", help="Files or directories to clean.")
    parser.add_argument("--dry-run", action="store_true", help="Report files that would change.")
    args = parser.parse_args()

    target_paths = [pathlib.Path(p).expanduser().resolve() for p in args.targets]
    files = list(iter_files(target_paths))
    if not files:
        print("no files found", file=sys.stderr)
        return 1

    modified = False
    for file_path in files:
        modified |= clean_file(file_path, args.dry_run)

    if not modified and not args.dry_run:
        print("nothing to clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

