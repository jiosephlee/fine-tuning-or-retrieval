#!/usr/bin/env python3
"""Finalize complete GPT-OSS recovery union manifests from validated hashes."""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.multiview_recovery import VIEWS, manifest_valid, validate_view  # noqa: E402


def finalize(data_root: Path, size: str, reasoning: str) -> list[Path]:
    written = []
    recovery = f"gpt_oss_{size}_{reasoning}_recovery"
    for domain in ("arxiv", "medical", "legal"):
        root = data_root / domain / "explanations" / recovery
        items = sorted(path for path in root.iterdir() if path.is_dir())
        if len(items) != 12:
            raise RuntimeError(f"{domain}/{recovery}: expected 12 items, found {len(items)}")
        selected = []
        for item_dir in items:
            try:
                generation = json.loads((item_dir / "generation_manifest.json").read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"invalid generation manifest for {item_dir}: {exc}") from exc
            try:
                recovery_manifest = json.loads((item_dir / "recovery_manifest.json").read_text())
            except (OSError, json.JSONDecodeError):
                recovery_manifest = {"views": {}}
            for view in VIEWS:
                report = validate_view(item_dir, view)
                if not report["valid"] or not manifest_valid(item_dir, view):
                    raise RuntimeError(
                        f"{domain}/{recovery}/{item_dir.name}/{view}: "
                        + "; ".join(report["reasons"] or ["manifest hash mismatch"])
                    )
                generation_view = generation["views"][view]
                metadata = generation_view.get("metadata", {})
                reuse = recovery_manifest.get("views", {}).get(view, {})
                source = metadata.get("source_variant") or reuse.get("source_variant")
                if not source:
                    source = f"generated:{metadata.get('model', f'openai/gpt-oss-{size}')}"
                selected.append({
                    "item": item_dir.name,
                    "view": view,
                    "source_variant": source,
                    "status": "validated",
                    "validated_at": generation_view.get("validated_at"),
                    "hashes": generation_view["hashes"],
                    "metadata": metadata,
                    "rejection_reasons": [],
                })
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "domain": domain,
            "model_size": size,
            "reasoning": reasoning,
            "recovery_variant": recovery,
            "quality_rubric": "sensible_coherent_complete_no_corruption_accuracy_ignored",
            "complete": True,
            "selected": selected,
            "missing": [],
            "rejected_candidates": [],
        }
        target = root / "union_manifest.json"
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, target)
        written.append(target)
    return written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--sizes", nargs="+", choices=("20b", "120b"), default=("20b", "120b"))
    parser.add_argument("--reasoning", choices=("low", "high"), default="low")
    args = parser.parse_args()
    for size in args.sizes:
        for path in finalize(args.data_root, size, args.reasoning):
            print(path)


if __name__ == "__main__":
    main()
