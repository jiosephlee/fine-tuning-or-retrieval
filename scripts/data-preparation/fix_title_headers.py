#!/usr/bin/env python3
"""Rewrite the first-line title header of existing explanation files so it carries the
document's REAL title in double quotes, instead of the filename / unquoted title.

Covers the flat files (stackexchange.txt, textbook.txt, blogs.txt) and their standard
granular subfolders (stackexchange/, blogs/, textbooks/) under
    data/{source}/explanations/{slug}/{domain}/
Experimental variants (shuffled_*, fruit_*, broken_*) are intentionally NOT touched.

Header formats:
    arxiv   -> \\title{<prefix>: "<title>"}
    legal   -> Title: <prefix>: "<title>"
    medical -> Title: <prefix>: "<title>"

Idempotent: files already in the quoted-real-title form are left unchanged.

Usage:
    python fix_title_headers.py --dry-run
    python fix_title_headers.py
    python fix_title_headers.py --sources arxiv --slug glm
"""
import argparse
import os
import re
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # for multiview imports
sys.path.append('multiview')
from pipeline_multi_view_knowledge_arxiv import extract_paper_title
from pipeline_multi_view_knowledge_legal import extract_case_title

# Per-source header config: how titles are stored on disk and where the real title lives.
SOURCE_CONFIG = {
    "arxiv": {
        "cleaned_ext": ".tex",
        "prefixes": [
            "Stack Exchange of the Paper",
            "A Textbook about the Paper",
            "Blogs about the Paper",
        ],
        "line": lambda prefix, title: f'\\title{{{prefix}: "{title}"}}',
        "regex": lambda prefix: re.compile(r'^\\title\{' + re.escape(prefix) + r': .*\}\s*$'),
        "get_title": lambda content, name: extract_paper_title(content, name),
    },
    "legal": {
        "cleaned_ext": ".txt",
        "prefixes": [
            "Stack Exchange about the opinion",
            "Casebook chapter about the opinion",
            "Blog about the opinion",
        ],
        "line": lambda prefix, title: f'Title: {prefix}: "{title}"',
        "regex": lambda prefix: re.compile(r'^Title: ' + re.escape(prefix) + r': .*$'),
        "get_title": lambda content, name: extract_case_title(content, name),
    },
    "medical": {
        "cleaned_ext": ".txt",
        "prefixes": [
            "Clinical Q&A about the case report",
            "Textbook chapter about the case report",
            "Blog about the case report",
        ],
        "line": lambda prefix, title: f'Title: {prefix}: "{title}"',
        "regex": lambda prefix: re.compile(r'^Title: ' + re.escape(prefix) + r': .*$'),
        "get_title": lambda content, name: extract_case_title(content, name),
    },
}

# Files whose first line we rewrite, per domain directory.
FLAT_FILES = ["stackexchange.txt", "textbook.txt", "blogs.txt"]
GRANULAR_SUBDIRS = ["stackexchange", "blogs", "textbooks"]


def covered_files(domain_dir):
    for name in FLAT_FILES:
        p = os.path.join(domain_dir, name)
        if os.path.isfile(p):
            yield p
    for sub in GRANULAR_SUBDIRS:
        subdir = os.path.join(domain_dir, sub)
        if os.path.isdir(subdir):
            for name in sorted(os.listdir(subdir)):
                if name.endswith(".txt"):
                    yield os.path.join(subdir, name)


def patch_file(path, cfg, title, dry_run):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if not lines:
        return False
    first = lines[0].rstrip("\n")
    for prefix in cfg["prefixes"]:
        if cfg["regex"](prefix).match(first):
            new_first = cfg["line"](prefix, title)
            if new_first == first:
                return False  # already correct
            if not dry_run:
                lines[0] = new_first + "\n"
                with open(path, "w", encoding="utf-8") as f:
                    f.writelines(lines)
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", nargs="+", default=["arxiv", "legal", "medical"])
    parser.add_argument("--slug", default="gpt_5_mini_custom")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    changed = 0
    examples = []
    for source in args.sources:
        cfg = SOURCE_CONFIG[source]
        expl_root = f"../../data/{source}/explanations/{args.slug}"
        cleaned_root = f"../../data/{source}/cleaned"
        if not os.path.isdir(expl_root):
            continue
        for domain in sorted(os.listdir(expl_root)):
            domain_dir = os.path.join(expl_root, domain)
            if not os.path.isdir(domain_dir):
                continue
            cleaned_path = os.path.join(cleaned_root, domain + cfg["cleaned_ext"])
            if not os.path.isfile(cleaned_path):
                print(f"  WARN: no cleaned source for {source}/{domain} at {cleaned_path}; skipping")
                continue
            with open(cleaned_path, "r", encoding="utf-8") as f:
                title = cfg["get_title"](f.read(), domain)
            for path in covered_files(domain_dir):
                if patch_file(path, cfg, title, args.dry_run):
                    changed += 1
                    if len(examples) < 6:
                        examples.append(f'{os.path.relpath(path)} -> "{title}"')

    verb = "Would rewrite" if args.dry_run else "Rewrote"
    print(f"\n{verb} {changed} file header(s).")
    for ex in examples:
        print("  e.g.", ex)


if __name__ == "__main__":
    main()
