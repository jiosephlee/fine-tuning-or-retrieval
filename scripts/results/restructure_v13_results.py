#!/usr/bin/env python
"""Consolidate v13 result leaves into canonical run folders."""

from __future__ import annotations

import argparse
import filecmp
import json
import shutil
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "results" / "FT" / "full"
PROBE_ROOT_PREFIX = "probes_v13"
BUNDLE_TOKENS = (
    "inf_mcqa_v13_legal",
    "inf_mcqa_v13+v12",
    "inf_mcqa_v13",
    "inf_mcqa_v12_reviewed+v12",
    "inf_mcqa_v12_reviewed",
    "inf_mcqa_v12",
)


@dataclass(frozen=True)
class MovePlan:
    source: str
    destination: str
    canonical_run: str
    model: str
    method: str
    fill_source: str
    domains: str
    epochs: str
    batch_lr: str
    overlap: str
    run_id: str
    eval_bundle: str
    source_probe_root: str


@dataclass(frozen=True)
class RawPlan:
    source: Path
    canonical_run: Path
    model: str
    method: str
    fill_source: str
    domains: str
    epochs: str
    batch_lr: str
    overlap: str
    run_id: str
    base_eval_bundle: str
    source_probe_root: str


def normalize_model(name: str) -> str:
    if "1124-13B" in name:
        return "13b"
    if "0325-32B" in name:
        return "32b"
    return name


def eval_bundle_name(probe_root: str) -> str:
    for token in BUNDLE_TOKENS:
        if token in probe_root:
            return token
    return "default"


def source_root_suffix(probe_root: str) -> str:
    if "para_v13_paraphrased" in probe_root:
        return "para_v13_root"
    if "probes_v13_inf_v11" in probe_root:
        return "legacy_no_para_root"
    safe = "".join(char if char.isalnum() else "_" for char in probe_root)
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe[:96].strip("_") or "source_root"


def bundle_for_duplicate(raw_plan: RawPlan, duplicate_group: List[RawPlan]) -> str:
    if len(duplicate_group) == 1:
        return raw_plan.base_eval_bundle
    preferred = [
        item
        for item in duplicate_group
        if "para_v13_paraphrased" in item.source_probe_root
    ]
    plain_bundle_source = sorted(preferred or duplicate_group, key=lambda item: str(item.source))[0]
    if raw_plan.source == plain_bundle_source.source:
        return raw_plan.base_eval_bundle
    return f"{raw_plan.base_eval_bundle}__{source_root_suffix(raw_plan.source_probe_root)}"


def looks_like_result_leaf(path: Path) -> bool:
    try:
        children = [child for child in path.iterdir() if child.is_dir()]
    except FileNotFoundError:
        return False
    return any("_probe" in child.name or child.name.startswith("reeval") for child in children)


def path_has_part(path: Path, part: str) -> bool:
    return part in path.parts


def iter_v13_leaves(root: Path) -> Iterable[Tuple[Path, Path, str]]:
    for model_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if model_dir.name.startswith("_"):
            continue
        for probe_root in sorted(path for path in model_dir.iterdir() if path.is_dir()):
            if not probe_root.name.startswith(PROBE_ROOT_PREFIX):
                continue
            newline_root = probe_root / "newline2"
            if not newline_root.is_dir():
                continue
            for leaf in sorted(newline_root.rglob("E*")):
                if path_has_part(leaf, "eval_bundles"):
                    continue
                if leaf.is_dir() and looks_like_result_leaf(leaf):
                    yield model_dir, probe_root, leaf


def build_plan(root: Path) -> List[MovePlan]:
    raw_plans: List[RawPlan] = []
    for model_dir, probe_root, leaf in iter_v13_leaves(root):
        rel_parts = leaf.relative_to(probe_root / "newline2").parts
        if len(rel_parts) < 7:
            continue
        method, fill_source, domains, epochs, batch_lr, overlap, run_id = rel_parts[-7:]
        model = normalize_model(model_dir.name)
        canonical_run = root / model / method / fill_source / domains / epochs / batch_lr / overlap / run_id
        raw_plans.append(
            RawPlan(
                source=leaf,
                canonical_run=canonical_run,
                model=model,
                method=method,
                fill_source=fill_source,
                domains=domains,
                epochs=epochs,
                batch_lr=batch_lr,
                overlap=overlap,
                run_id=run_id,
                base_eval_bundle=eval_bundle_name(probe_root.name),
                source_probe_root=probe_root.name,
            )
        )

    duplicate_groups: Dict[Tuple[str, str], List[RawPlan]] = defaultdict(list)
    for raw_plan in raw_plans:
        duplicate_groups[(str(raw_plan.canonical_run), raw_plan.base_eval_bundle)].append(raw_plan)

    plans: List[MovePlan] = []
    for raw_plan in raw_plans:
        eval_bundle = bundle_for_duplicate(
            raw_plan,
            duplicate_groups[(str(raw_plan.canonical_run), raw_plan.base_eval_bundle)],
        )
        destination = raw_plan.canonical_run / "eval_bundles" / eval_bundle
        plans.append(
            MovePlan(
                source=str(raw_plan.source),
                destination=str(destination),
                canonical_run=str(raw_plan.canonical_run),
                model=raw_plan.model,
                method=raw_plan.method,
                fill_source=raw_plan.fill_source,
                domains=raw_plan.domains,
                epochs=raw_plan.epochs,
                batch_lr=raw_plan.batch_lr,
                overlap=raw_plan.overlap,
                run_id=raw_plan.run_id,
                eval_bundle=eval_bundle,
                source_probe_root=raw_plan.source_probe_root,
            )
        )
    return plans


def paths_identical(left: Path, right: Path) -> bool:
    if left.is_symlink() or right.is_symlink():
        return left.is_symlink() == right.is_symlink() and left.resolve() == right.resolve()
    if left.is_file() and right.is_file():
        return filecmp.cmp(left, right, shallow=False)
    if left.is_dir() and right.is_dir():
        left_children = {child.name: child for child in left.iterdir()}
        right_children = {child.name: child for child in right.iterdir()}
        if set(left_children) != set(right_children):
            return False
        return all(paths_identical(left_children[name], right_children[name]) for name in left_children)
    return False


def planned_child_collisions(plans: List[MovePlan]) -> List[Dict[str, object]]:
    by_target: Dict[str, List[Path]] = defaultdict(list)
    for plan in plans:
        source = Path(plan.source)
        destination = Path(plan.destination)
        for child in sorted(source.iterdir()):
            by_target[str(destination / child.name)].append(child)

    collisions = []
    for target, sources in by_target.items():
        if len(sources) <= 1:
            continue
        first = sources[0]
        identical = all(paths_identical(first, other) for other in sources[1:])
        collisions.append(
            {
                "target": target,
                "sources": [str(source) for source in sources],
                "identical": identical,
            }
        )
    return collisions


def remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def prune_empty_parents(path: Path, stop_at: Path) -> List[str]:
    removed = []
    current = path
    stop_at = stop_at.resolve()
    while current.resolve() != stop_at:
        try:
            current.rmdir()
        except OSError:
            break
        removed.append(str(current))
        current = current.parent
    return removed


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def execute_plan(root: Path, plans: List[MovePlan]) -> Dict[str, object]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    moved_children = 0
    duplicate_children = 0
    pruned_dirs: List[str] = []
    per_run: Dict[str, List[MovePlan]] = defaultdict(list)

    for plan in plans:
        per_run[plan.canonical_run].append(plan)
        source = Path(plan.source)
        destination = Path(plan.destination)
        destination.mkdir(parents=True, exist_ok=True)

        for child in sorted(source.iterdir()):
            target = destination / child.name
            if not target.exists():
                shutil.move(str(child), str(target))
                moved_children += 1
            elif paths_identical(child, target):
                remove_path(child)
                duplicate_children += 1
            else:
                raise RuntimeError(f"Refusing to overwrite non-identical target: {target}")

        pruned_dirs.extend(prune_empty_parents(source, root))

    for canonical_run, run_plans in per_run.items():
        write_json(
            Path(canonical_run) / "manifest.json",
            {
                "canonical_run": canonical_run,
                "updated_at": timestamp,
                "sources": [asdict(plan) for plan in sorted(run_plans, key=lambda item: item.destination)],
            },
        )

    global_manifest = {
        "root": str(root),
        "executed_at": timestamp,
        "source_leaf_count": len(plans),
        "canonical_run_count": len(per_run),
        "moved_children": moved_children,
        "duplicate_children_removed": duplicate_children,
        "pruned_empty_dirs": pruned_dirs,
        "plans": [asdict(plan) for plan in plans],
    }
    manifest_path = root / "_migration_manifests" / f"v13_restructure_{timestamp}.json"
    write_json(manifest_path, global_manifest)
    global_manifest["manifest_path"] = str(manifest_path)
    return global_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="Result root to restructure.")
    parser.add_argument("--execute", action="store_true", help="Actually move files. Defaults to dry-run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    plans = build_plan(root)
    collisions = planned_child_collisions(plans)
    non_identical_collisions = [collision for collision in collisions if not collision["identical"]]

    canonical_runs = {plan.canonical_run for plan in plans}
    print(f"Found {len(plans)} v13 source leaves")
    print(f"Canonical runs: {len(canonical_runs)}")
    print(f"Planned child collisions: {len(collisions)}")
    print(f"Non-identical child collisions: {len(non_identical_collisions)}")
    if non_identical_collisions:
        for collision in non_identical_collisions[:20]:
            print(f"Conflict: {collision['target']}")
            for source in collision["sources"]:
                print(f"  {source}")
        raise SystemExit("Aborting: non-identical destination collisions would overwrite data.")

    if not args.execute:
        print("Dry run only. Re-run with --execute to move files.")
        for plan in plans[:20]:
            print(f"{plan.source} -> {plan.destination}")
        if len(plans) > 20:
            print(f"... {len(plans) - 20} more")
        return

    manifest = execute_plan(root, plans)
    print(f"Moved child entries: {manifest['moved_children']}")
    print(f"Removed duplicate child entries: {manifest['duplicate_children_removed']}")
    print(f"Pruned empty directories: {len(manifest['pruned_empty_dirs'])}")
    print(f"Wrote manifest: {manifest['manifest_path']}")


if __name__ == "__main__":
    main()
