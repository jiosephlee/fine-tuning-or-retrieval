#!/usr/bin/env python3
"""Promote primary original metrics out of eval_bundles.

The canonical layout after this migration is:

  <run>/
    <domain>_knowledge_probe/
    <domain>_mcqa_probe/
    ...
    eval_bundles/
      reeval_v1/
      reeval_v2/
      reeval_v3/
      ...

Alternate eval bundles can remain under ``eval_bundles/`` when they are
meaningfully distinct. Empty bundles and narrow legal/v13 partial bundles are
removed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
from pathlib import Path
from typing import Optional


PRIMARY_BY_MODEL = {
    "1b": ("inf_mcqa_v12", "default"),
    "7b": ("inf_mcqa_v12", "default", "inf_mcqa_v13+v12"),
    "13b": ("inf_mcqa_v13", "inf_mcqa_v12", "default"),
    "32b": ("inf_mcqa_v13+v12", "default"),
}

DELETE_BUNDLE_NAMES = {
    "inf_mcqa_v13_legal",
    "inf_mcqa_v12__legacy_no_para_root",
}


def model_for_run(run_dir: Path, root: Path) -> Optional[str]:
    try:
        rel = run_dir.relative_to(root)
    except ValueError:
        return None
    return rel.parts[0] if rel.parts else None


def direct_metric_files(bundle: Path) -> list[Path]:
    files = []
    for path in bundle.glob("**/*_metrics.csv"):
        try:
            rel = path.relative_to(bundle)
        except ValueError:
            continue
        if not rel.parts or rel.parts[0].startswith("reeval"):
            continue
        files.append(path)
    return files


def is_full_original_bundle(bundle: Path) -> bool:
    if not bundle.is_dir():
        return False
    knowledge_dirs = [p for p in bundle.iterdir() if p.is_dir() and p.name.endswith("_knowledge_probe")]
    return len(knowledge_dirs) >= 20 and bool(direct_metric_files(bundle))


def choose_primary_bundle(run_dir: Path, root: Path) -> Optional[Path]:
    eval_bundles = run_dir / "eval_bundles"
    if not eval_bundles.is_dir():
        return None
    model = model_for_run(run_dir, root)
    for name in PRIMARY_BY_MODEL.get(model or "", ()):
        candidate = eval_bundles / name
        if is_full_original_bundle(candidate):
            return candidate
    full = [p for p in eval_bundles.iterdir() if is_full_original_bundle(p)]
    if len(full) == 1:
        return full[0]
    for name in ("default", "inf_mcqa_v13+v12", "inf_mcqa_v13", "inf_mcqa_v12"):
        candidate = eval_bundles / name
        if candidate in full:
            return candidate
    return None


def move_dir(src: Path, dst: Path, actions: list[dict], execute: bool) -> None:
    if dst.exists():
        raise RuntimeError(f"Destination already exists: {dst}")
    actions.append({"action": "move", "source": str(src), "destination": str(dst)})
    if execute:
        shutil.move(str(src), str(dst))


def remove_dir(path: Path, actions: list[dict], execute: bool, reason: str) -> None:
    if not path.exists():
        return
    actions.append({"action": "delete", "path": str(path), "reason": reason})
    if execute:
        shutil.rmtree(path)


def promote_bundle(run_dir: Path, bundle: Path, actions: list[dict], execute: bool) -> None:
    eval_bundles = run_dir / "eval_bundles"
    for child in sorted(bundle.iterdir()):
        dst = eval_bundles / child.name if child.name.startswith("reeval") else run_dir / child.name
        move_dir(child, dst, actions, execute)
    remove_dir(bundle, actions, execute, "primary bundle promoted to run root")


def move_reviewed_reevals(eval_bundles: Path, actions: list[dict], execute: bool) -> None:
    reviewed = sorted(
        p
        for p in eval_bundles.iterdir()
        if p.is_dir() and p.name.startswith("inf_mcqa_v12_reviewed")
    )
    for bundle in reviewed:
        for child in sorted(bundle.iterdir()):
            if child.is_dir() and child.name.startswith("reeval"):
                move_dir(child, eval_bundles / child.name, actions, execute)
        if bundle.exists() and not any(bundle.iterdir()):
            remove_dir(bundle, actions, execute, "empty reviewed bundle after reeval promotion")


def cleanup_bundles(eval_bundles: Path, primary_name: Optional[str], actions: list[dict], execute: bool) -> None:
    for bundle in sorted(eval_bundles.iterdir()):
        if not bundle.is_dir():
            continue
        if not any(bundle.iterdir()):
            remove_dir(bundle, actions, execute, "empty eval bundle")
            continue
        if bundle.name in DELETE_BUNDLE_NAMES:
            remove_dir(bundle, actions, execute, "specialized eval bundle no longer kept")
            continue
        if bundle.name == "inf_mcqa_v13" and primary_name != "inf_mcqa_v13" and not is_full_original_bundle(bundle):
            remove_dir(bundle, actions, execute, "partial v13 eval bundle")


def migrate(root: Path, execute: bool) -> dict:
    actions: list[dict] = []
    run_dirs = sorted(p.parent for p in root.glob("**/eval_bundles") if p.is_dir())
    for run_dir in run_dirs:
        eval_bundles = run_dir / "eval_bundles"
        primary = choose_primary_bundle(run_dir, root)
        primary_name = primary.name if primary else None
        root_has_metrics = any(run_dir.glob("*_knowledge_probe/*_knowledge_probe_metrics.csv"))
        if primary and not root_has_metrics:
            promote_bundle(run_dir, primary, actions, execute)
        move_reviewed_reevals(eval_bundles, actions, execute)
        cleanup_bundles(eval_bundles, primary_name, actions, execute)
    return {
        "execute": execute,
        "updated_at": dt.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "actions": actions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/FT/full")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    manifest = migrate(root, args.execute)
    manifest_dir = root / "_migration_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    suffix = "execute" if args.execute else "dryrun"
    path = manifest_dir / f"promote_original_metrics_{manifest['updated_at']}_{suffix}.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"actions={len(manifest['actions'])}")
    print(f"manifest={path}")
    if not args.execute:
        print("Dry run only. Re-run with --execute to apply.")


if __name__ == "__main__":
    main()
