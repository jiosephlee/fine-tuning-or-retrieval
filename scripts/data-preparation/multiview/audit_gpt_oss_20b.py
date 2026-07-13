#!/usr/bin/env python3
"""Audit GPT-OSS multiview outputs and build provenance-ranked recovery unions."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import re
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
from utils.multiview_recovery import (  # noqa: E402
    ASSEMBLED, GRANULAR_LAYOUTS, OUTLINES, VIEWS, record_validated_view,
    validate_view,
)

DATA_ROOT = PROJECT_ROOT / "data"
MODEL_SIZES = ("20b", "120b")
REASONING_LEVELS = ("low", "high")
VARIANT_RE = re.compile(r"^gpt_oss_(20b|120b)_(low|high)(?:_|$)")


def expected_items(domain: str) -> list[str]:
    roots = {"arxiv": ("cleaned", ".tex"), "medical": ("cleaned", ".txt"), "legal": ("cleaned", ".txt")}
    folder, suffix = roots[domain]
    paths = sorted((DATA_ROOT / domain / folder).glob(f"*{suffix}"))
    if not paths:  # legal/medical deployments may retain only raw source files
        paths = sorted((DATA_ROOT / domain / "raw").glob(f"*{suffix}"))
    return [path.stem for path in paths]


def audit(data_root=DATA_ROOT, sizes=MODEL_SIZES) -> dict:
    data_root = Path(data_root)
    variants, records = [], []
    for domain in ("arxiv", "medical", "legal"):
        explanation_root = data_root / domain / "explanations"
        domain_variants = sorted(
            path for path in explanation_root.glob("gpt_oss_*")
            if path.is_dir() and (match := VARIANT_RE.match(path.name))
            and match.group(1) in sizes
        )
        variants += [str(path) for path in domain_variants]
        items = expected_items(domain) if data_root == DATA_ROOT else sorted({p.name for v in domain_variants for p in v.iterdir() if p.is_dir()})
        for variant in domain_variants:
            for item in items:
                for view in VIEWS:
                    result = validate_view(variant / item, view)
                    match = VARIANT_RE.match(variant.name)
                    result.update(domain=domain, variant=variant.name, item=item,
                                  model_size=match.group(1), reasoning=match.group(2))
                    records.append(result)
    failures = sum(not record["valid"] for record in records)
    return {"generated_at": datetime.now(timezone.utc).isoformat(), "model_sizes": list(sizes), "variants": variants,
            "summary": {"records": len(records), "valid": len(records) - failures, "invalid": failures},
            "records": records}


def human_report(report: dict) -> str:
    sizes = ", ".join(report.get("model_sizes", ["20b"]))
    lines = [f"GPT-OSS ({sizes}) multiview audit", f"Records: {report['summary']['records']}; valid: {report['summary']['valid']}; invalid: {report['summary']['invalid']}"]
    for record in report["records"]:
        if not record["valid"]:
            lines.append(f"FAIL {record['domain']}/{record['variant']}/{record['item']}/{record['view']}: " + "; ".join(record["reasons"]))
    return "\n".join(lines) + "\n"


def _candidate_rank(candidate, canonical, recovery):
    """Prefer validated recovery, then canonical, then stable historical variants."""
    variant = candidate["variant"]
    return (0 if variant == recovery else 1 if variant == canonical else 2, variant)


def _clear_view(item_dir: Path, view: str):
    for filename in (ASSEMBLED[view], OUTLINES[view]):
        path = item_dir / filename
        if path.exists(): path.unlink()
    granular_dir = item_dir / GRANULAR_LAYOUTS[view][0]
    if granular_dir.exists(): shutil.rmtree(granular_dir)


def _load_rejections(path):
    if not path:
        return set(), {}
    payload = json.loads(Path(path).read_text())
    details, rejected = {}, set()
    for row in payload.get("rejections", []):
        key = (row["domain"], row["model_size"], row["reasoning"], row["item"],
               row["view"], row["source_variant"])
        rejected.add(key)
        details[key] = row.get("reasons", [])
    return rejected, details


def stage_recovery(report: dict, data_root=DATA_ROOT, sizes=MODEL_SIZES, rejection_file=None):
    """Select validated views by provenance, copying no failing candidate."""
    data_root = Path(data_root)
    rejected, rejection_details = _load_rejections(rejection_file)
    staged = []
    for size in sizes:
      for reasoning in REASONING_LEVELS:
        canonical = f"gpt_oss_{size}_{reasoning}"
        recovery = canonical + "_recovery"
        for domain in ("arxiv", "medical", "legal"):
            target_root = data_root / domain / "explanations" / recovery
            candidates = []
            for row in report["records"]:
                if not (row["domain"] == domain and row["model_size"] == size
                        and row["reasoning"] == reasoning and row["valid"]
                        and (domain, size, reasoning, row["item"], row["view"], row["variant"])
                            not in rejected):
                    continue
                source = data_root / domain / "explanations" / row["variant"] / row["item"]
                current = validate_view(source, row["view"])
                if current["valid"]:
                    candidates.append({**row, **current})
            by_key = {}
            for candidate in candidates:
                # Prefer the canonical provenance, then stable lexical ordering. File
                # size is intentionally absent from selection.
                rank = _candidate_rank(candidate, canonical, recovery)
                key = (candidate["item"], candidate["view"])
                if key not in by_key or rank < by_key[key][0]: by_key[key] = (rank, candidate)
            for (item, view), (_, candidate) in by_key.items():
                source_item = data_root / domain / "explanations" / candidate["variant"] / item
                target_item = target_root / item
                target_item.mkdir(parents=True, exist_ok=True)
                existing_recovery = source_item.resolve() == target_item.resolve()
                if not existing_recovery:
                    _clear_view(target_item, view)
                    for file_record in candidate["files"]:
                        source = Path(file_record["path"])
                        relative = source.resolve().relative_to(source_item.resolve())
                        destination = target_item / relative
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source, destination)
                manifest_path = target_item / "recovery_manifest.json"
                try: manifest = json.loads(manifest_path.read_text())
                except (OSError, json.JSONDecodeError): manifest = {"views": {}}
                manifest["views"][view] = {"source_variant": candidate["variant"],
                    "status": "validated_existing" if existing_recovery else "validated_reuse",
                    "files": {str(Path(f["path"]).resolve().relative_to(source_item.resolve())): f["sha256"]
                              for f in candidate["files"]}}
                manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
                record_validated_view(target_item, view, {
                    "source_variant": candidate["variant"], "reuse": not existing_recovery,
                    "selection": "existing_validated_recovery" if existing_recovery else "validated_union_copy",
                    "model_size": size, "reasoning_effort": reasoning,
                })
                staged.append(f"{domain}/{recovery}/{item}/{view} <- {candidate['variant']}")
            expected = sorted({r["item"] for r in report["records"]
                               if r["domain"] == domain and r["model_size"] == size
                               and r["reasoning"] == reasoning})
            required = {(item, view) for item in expected for view in VIEWS}
            selected = {(item, view) for item, view in by_key}
            rejected_rows = [
                {"item": item, "view": view, "source_variant": source,
                 "reasons": rejection_details[(domain, size, reasoning, item, view, source)]}
                for d, s, e, item, view, source in sorted(rejected)
                if (d, s, e) == (domain, size, reasoning)
            ]
            union_manifest = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "domain": domain, "model_size": size, "reasoning": reasoning,
                "recovery_variant": recovery,
                "semantic_status": (
                    "fully_judged_no_acceptable_candidate"
                    if rejected_rows and not selected else "not_fully_judged"
                ),
                "complete": selected == required,
                "selected": [
                    {"item": item, "view": view,
                     "source_variant": by_key[(item, view)][1]["variant"],
                     "rejection_reasons": []}
                    for item, view in sorted(selected)
                ],
                "missing": [{"item": item, "view": view}
                            for item, view in sorted(required - selected)],
                "rejected_candidates": rejected_rows,
            }
            target_root.mkdir(parents=True, exist_ok=True)
            (target_root / "union_manifest.json").write_text(
                json.dumps(union_manifest, indent=2, sort_keys=True) + "\n"
            )
    return staged


def promote_recovery(report: dict, data_root=DATA_ROOT, sizes=MODEL_SIZES, snapshot="pre_recovery_20260711"):
    """Atomically promote only complete recovery products; called only by --promote."""
    data_root = Path(data_root)
    operations = []
    for size in sizes:
      for reasoning in REASONING_LEVELS:
        canonical = f"gpt_oss_{size}_{reasoning}"
        recovery = canonical + "_recovery"
        for domain in ("arxiv", "medical", "legal"):
            expected = set(expected_items(domain)) if data_root == DATA_ROOT else {
                p.name for p in (data_root / domain / "explanations" / recovery).iterdir() if p.is_dir()}
            records = [r for r in report["records"] if r["domain"] == domain and r["variant"] == recovery]
            valid = {(r["item"], r["view"]) for r in records if r["valid"]}
            required = {(item, view) for item in expected for view in VIEWS}
            missing = sorted(required - valid)
            if missing: raise RuntimeError(f"promotion blocked for {domain}/{recovery}: {len(missing)} invalid or missing views")
            root = data_root / domain / "explanations"
            source, target = root / recovery, root / canonical
            snapshot_path = root / f"{canonical}_{snapshot}"
            if not source.is_dir(): raise RuntimeError(f"missing recovery slug: {source}")
            if snapshot_path.exists(): raise RuntimeError(f"snapshot already exists: {snapshot_path}")
            operations.append((source, target, snapshot_path))
    completed = []
    for source, target, snapshot_path in operations:
        if target.exists(): os.replace(target, snapshot_path)
        try: os.replace(source, target)
        except BaseException:
            if snapshot_path.exists(): os.replace(snapshot_path, target)
            raise
        completed.append(f"{source} -> {target} (snapshot {snapshot_path})")
    return completed


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, help="JSON report path")
    parser.add_argument("--text", type=Path, help="human-readable report path")
    parser.add_argument("--stage-recovery", action="store_true", help="copy only validated views into recovery slugs")
    parser.add_argument("--strict", action="store_true", help="exit nonzero when any audited view is invalid")
    parser.add_argument("--promote", action="store_true", help="atomically promote complete recovery slugs after all gates pass")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--sizes", nargs="+", choices=MODEL_SIZES, default=list(MODEL_SIZES))
    parser.add_argument("--semantic-rejections", type=Path,
                        help="JSON source-specific rejection file used when building the union")
    args = parser.parse_args(argv)
    report = audit(args.data_root, tuple(args.sizes))
    rendered = human_report(report)
    if args.json: args.json.parent.mkdir(parents=True, exist_ok=True); args.json.write_text(json.dumps(report, indent=2) + "\n")
    if args.text: args.text.parent.mkdir(parents=True, exist_ok=True); args.text.write_text(rendered)
    print(rendered, end="")
    if args.stage_recovery:
        for line in stage_recovery(report, args.data_root, tuple(args.sizes), args.semantic_rejections): print("STAGED", line)
    if args.promote:
        for line in promote_recovery(report, args.data_root, tuple(args.sizes)): print("PROMOTED", line)
    return 1 if args.strict and report["summary"]["invalid"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
