#!/usr/bin/env python3
"""Audit and safely backfill granular multiview explanation outputs.

Existing granular files are never overwritten.  Splits are created only when
the assembled file has unambiguous boundaries and agrees with its outline.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.granular_outputs import granular_path


DOMAINS = ("arxiv", "medical", "legal")
VIEW_FILES = {
    "textbook": ("textbook.txt", "textbook_outline.json"),
    "stackexchange": ("stackexchange.txt", "stack_exchange_outline.json"),
    "blog": ("blogs.txt", "blog_outline.json"),
}
HYPHENS = "-\u2010\u2011\u2012\u2013\u2014\u2015\u2212\ufe58\ufe63\uff0d"
QUOTES = "\"'\u2018\u2019\u201a\u201b\u201c\u201d\u201e\u201f\u00ab\u00bb"


class PairObject(list):
    """Ordered JSON object retaining duplicate keys."""


def _pairs_hook(pairs):
    return PairObject(pairs)


def _duplicate_keys(value):
    duplicates = []
    if isinstance(value, PairObject):
        keys = [key for key, _ in value]
        duplicates.extend(key for key, count in Counter(keys).items() if count > 1)
        for _, child in value:
            duplicates.extend(_duplicate_keys(child))
    elif isinstance(value, list):
        for child in value:
            duplicates.extend(_duplicate_keys(child))
    return duplicates


def _plain(value):
    if isinstance(value, PairObject):
        result = {}
        for key, child in value:
            child = _plain(child)
            if key in result and isinstance(result[key], list) and isinstance(child, list):
                result[key].extend(child)
            else:
                result[key] = child
        return result
    if isinstance(value, list):
        return [_plain(child) for child in value]
    return value


def _object_get(obj, key, default=None):
    if not isinstance(obj, PairObject):
        return default
    for current, value in obj:
        if current == key:
            return value
    return default


def _recover_repeated_items(obj, title_key):
    """Turn a duplicate-key object into items, split at each title key."""
    if not isinstance(obj, PairObject):
        return []
    items, current = [], None
    for key, value in obj:
        if key == title_key:
            if current:
                items.append(current)
            current = {key: _plain(value)}
        elif current is not None:
            current[key] = _plain(value)
    if current:
        items.append(current)
    return items


def load_outline(path, view):
    """Return (items, normalized_data, malformed_reason)."""
    if not path.exists():
        return [], None, None
    try:
        root = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_pairs_hook)
    except (OSError, json.JSONDecodeError) as exc:
        return [], None, f"invalid JSON: {exc}"

    duplicates = sorted(set(_duplicate_keys(root)))
    if view == "textbook":
        root_key = "sections" if _object_get(root, "sections") is not None else "outline"
        title_key = "section_title" if root_key == "sections" else "chapter_title"
    elif view == "blog":
        root_key, title_key = "blogs", "title"
    else:
        root_key = "questions"
        value = _object_get(root, root_key, [])
        sample = value[0] if isinstance(value, list) and value else None
        title_key = "title" if _object_get(sample, "title") is not None else "question"

    raw_items = _object_get(root, root_key, [])
    if isinstance(raw_items, PairObject):
        items = _recover_repeated_items(raw_items, title_key)
    elif isinstance(raw_items, list):
        items = [_plain(item) for item in raw_items if isinstance(item, PairObject)]
    else:
        items = []

    normalized = {root_key: items}
    malformed = None
    if duplicates:
        malformed = f"duplicate JSON keys recovered: {', '.join(duplicates)}"
    elif not isinstance(raw_items, list):
        malformed = f"{root_key!r} was not a list"
    return items, normalized, malformed


def _title_line(content):
    first = content.splitlines()[0] if content.splitlines() else ""
    return first if first.startswith(("Title:", "\\title{")) else ""


def _canonical_piece(title, body, view, index):
    prefix = f"{title}\n\n" if title else ""
    if view == "textbook":
        return f"{prefix}Chapter {index}: {body.strip()}"
    return f"{prefix}{body.strip()}"


def _flexible_title_pattern(title):
    # Some local models copy numeric outline labels into the JSON but omit them
    # in the generated markdown heading.
    title = re.sub(r"^\s*\d+[.)]\s*", "", title)
    pieces = []
    for char in title.strip():
        if char.isspace():
            pieces.append(r"\s+")
        elif char in HYPHENS:
            pieces.append(f"[{re.escape(HYPHENS)}]")
        elif char in QUOTES:
            pieces.append(f"[{re.escape(QUOTES)}]")
        else:
            pieces.append(re.escape(char))
    return "".join(pieces)


def _split_by_titles(content, titles):
    boundaries = []
    for title in titles:
        pattern = re.compile(
            r"^(?:#{1,6}\s+)?" + _flexible_title_pattern(title) + r"\s*$",
            re.IGNORECASE | re.MULTILINE,
        )
        matches = list(pattern.finditer(content))
        if len(matches) != 1:
            return [], f"outline title {title!r} matched {len(matches)} boundaries"
        boundaries.append(matches[0].start())
    if boundaries != sorted(boundaries) or len(set(boundaries)) != len(boundaries):
        return [], "outline titles did not occur once in outline order"
    return [content[start:(boundaries[i + 1] if i + 1 < len(boundaries) else len(content))].strip()
            for i, start in enumerate(boundaries)], None


def _split_at_heading_level(content, level):
    pattern = re.compile(rf"^#{{{level}}}\s+.+$", re.MULTILINE)
    matches = list(pattern.finditer(content))
    return [content[match.start():(matches[i + 1].start() if i + 1 < len(matches) else len(content))].strip()
            for i, match in enumerate(matches)]


def split_view(item_dir, view, outline_items):
    assembled_name, _ = VIEW_FILES[view]
    assembled_path = item_dir / assembled_name
    if not assembled_path.exists():
        return [], f"missing assembled file {assembled_path}"
    content = assembled_path.read_text(encoding="utf-8")
    title = _title_line(content)

    if view == "stackexchange":
        if outline_items:
            titles = []
            for item in outline_items:
                if item.get("title"):
                    titles.append(item["title"])
                elif item.get("question"):
                    prefix = f"[{item['category']}] " if item.get("category") else ""
                    titles.append(prefix + item["question"])
            if len(titles) != len(outline_items):
                return [], "Q&A outline has items without usable question titles"
            bodies, reason = _split_by_titles(content, titles)
            if reason:
                return [], reason
        else:
            bodies = [part.strip() for part in re.split(r"\n(?=###\s+)", content)
                      if part.strip().startswith("### ")]
        expected = len(outline_items) if outline_items else len(bodies)
    elif view == "blog":
        titles = [item.get("title") for item in outline_items if item.get("title")]
        if titles:
            bodies, reason = _split_by_titles(content, titles)
            if reason:
                header_bodies = _split_at_heading_level(content, 1)
                if len(header_bodies) == len(outline_items):
                    bodies, reason = header_bodies, None
                else:
                    return [], reason
        else:
            bodies = [part.strip() for part in re.split(r"\n(?=#\s+)", content)
                      if part.strip().startswith("# ")]
        expected = len(outline_items) if outline_items else len(bodies)
    else:
        title_key = "section_title" if outline_items and "section_title" in outline_items[0] else "chapter_title"
        titles = [item.get(title_key) for item in outline_items if item.get(title_key)]
        if not titles:
            return [], "textbook outline has no usable chapter or section titles"
        bodies, reason = _split_by_titles(content, titles)
        if reason:
            alternatives = [parts for level in (1, 2)
                            if len(parts := _split_at_heading_level(content, level)) == len(outline_items)]
            if len(alternatives) == 1:
                bodies, reason = alternatives[0], None
            else:
                return [], reason
        expected = len(outline_items)

    if not bodies:
        return [], "no item boundaries found in assembled file"
    if len(bodies) != expected:
        return [], f"assembled split found {len(bodies)} items but outline expects {expected}"
    return [_canonical_piece(title, body, view, i) for i, body in enumerate(bodies, 1)], None


def _same_text(left, right):
    return left.replace("\r\n", "\n").strip() == right.replace("\r\n", "\n").strip()


def _event(domain, slug, item, view, path, reason=None):
    event = {"domain": domain, "slug": slug, "item": item, "view": view, "path": str(path)}
    if reason:
        event["reason"] = reason
    return event


def _backup_path(path):
    candidate = path.with_name(path.name + ".bak")
    suffix = 1
    while candidate.exists():
        candidate = path.with_name(path.name + f".bak.{suffix}")
        suffix += 1
    return candidate


def discover_items(data_root, domains, slug_filter=None, item_filter=None):
    for domain in domains:
        base = data_root / domain / "explanations"
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_dir() or not any((path / name).exists() for name, _ in VIEW_FILES.values()):
                continue
            relative = path.relative_to(base)
            # Canonical hierarchy is explanations/<slug>/<paper-or-case>.  Files
            # deeper inside an item are derived variants, not additional slugs.
            if len(relative.parts) != 2:
                continue
            slug, item = relative.parts[0], relative.parts[-1]
            if slug_filter and slug not in slug_filter:
                continue
            if item_filter and item not in item_filter:
                continue
            yield domain, slug, item, path


def run_backfill(data_root, domains=DOMAINS, slug_filter=None, item_filter=None, dry_run=False):
    report = {key: [] for key in
              ("created", "matching", "conflicting", "malformed", "ambiguous", "unresolved")}
    scanned = 0
    for domain, slug, item, item_dir in discover_items(
            Path(data_root), domains, slug_filter, item_filter):
        scanned += 1
        for view, (_, outline_name) in VIEW_FILES.items():
            assembled = item_dir / VIEW_FILES[view][0]
            outline_path = item_dir / outline_name
            if not assembled.exists() and not outline_path.exists():
                continue

            outline_items, normalized, malformed = load_outline(outline_path, view)
            if malformed:
                event = _event(domain, slug, item, view, outline_path, malformed)
                if normalized is not None and view == "textbook":
                    backup = _backup_path(outline_path)
                    event["backup"] = str(backup)
                    if not dry_run:
                        shutil.copy2(outline_path, backup)
                        outline_path.write_text(json.dumps(normalized, indent=2, ensure_ascii=False) + "\n",
                                                encoding="utf-8")
                report["malformed"].append(event)

            candidates, reason = split_view(item_dir, view, outline_items)
            if reason:
                category = "ambiguous" if assembled.exists() else "unresolved"
                report[category].append(_event(domain, slug, item, view, assembled, reason))
                continue

            for index, candidate in enumerate(candidates, 1):
                path = granular_path(item_dir, view, index)
                event = _event(domain, slug, item, view, path)
                if path.exists():
                    if _same_text(path.read_text(encoding="utf-8"), candidate):
                        report["matching"].append(event)
                    else:
                        event["reason"] = "existing granular file disagrees with assembled/outline split; preserved"
                        report["conflicting"].append(event)
                else:
                    if not dry_run:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(candidate, encoding="utf-8")
                    report["created"].append(event)

    report["summary"] = {"scanned_items": scanned, **{key: len(value) for key, value in report.items()}}
    report["dry_run"] = dry_run
    return report


def human_report(report):
    summary = report["summary"]
    lines = [f"Granular output backfill ({'dry run' if report['dry_run'] else 'write mode'})",
             f"Scanned explanation items: {summary['scanned_items']}"]
    for category in ("created", "matching", "conflicting", "malformed", "ambiguous", "unresolved"):
        lines.append(f"{category.capitalize()}: {summary[category]}")
        if category in ("conflicting", "malformed", "ambiguous", "unresolved"):
            for event in report[category]:
                lines.append(f"  - {event['path']}: {event.get('reason', category)}")
    return "\n".join(lines) + "\n"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--domain", action="append", choices=DOMAINS,
                        help="Domain to scan; repeat for multiple domains (default: all).")
    parser.add_argument("--slug", action="append", help="Exact explanation slug; repeatable.")
    parser.add_argument("--item", "--paper", "--case", dest="item", action="append",
                        help="Exact paper/case/item directory name; repeatable.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json-report", type=Path, default=Path("granular_backfill_report.json"))
    parser.add_argument("--human-report", type=Path, default=Path("granular_backfill_report.txt"))
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    report = run_backfill(args.data_root, args.domain or DOMAINS, args.slug, args.item, args.dry_run)
    rendered = human_report(report)
    args.json_report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.human_report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if report["ambiguous"] or report["unresolved"]:
        return 2
    if report["conflicting"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
