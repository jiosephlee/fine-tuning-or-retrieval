#!/usr/bin/env python3
"""Back up and submit targeted regeneration jobs for failed Qwen views.

Targets come from the persisted integrity-audit TSVs.  Dry-run is the default;
``--submit`` creates per-view backups and submits one job per domain/model/view.
Later rounds can use a refreshed TSV via ``--candidate-report`` so only views
that still are not PASS are retried.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORTS = (
    ROOT / "docs/reports/qwen_explanation_integrity_legal.tsv",
    ROOT / "docs/reports/qwen_explanation_integrity_medical.tsv",
)
BINARY_REPORT = ROOT / "docs/reports/qwen_all_views_binary_integrity_audit.tsv"
LAUNCHER = ROOT / "scripts/slurm/launch_qwen35_multiview.sh"
VIEW_FILES = {
    "blog": ("blogs.txt", "blog_outline.json", "blogs"),
    "stackexchange": ("stackexchange.txt", "stack_exchange_outline.json", "stackexchange"),
    "textbook": ("textbook.txt", "textbook_outline.json", "textbooks"),
}
MODEL_SIZES = {
    "qwen3_5_4b": "4B",
    "qwen3_5_9b": "9B",
    "qwen3_5_27b": "27B",
    "qwen3_5_35b_a3b_fp8": "35B-A3B-FP8",
    "qwen3_5_122b_a10b_fp8": "122B-A10B-FP8",
    "qwen3_5_397b_a17b_fp8": "397B-A17B-FP8",
}
PROFILES = {
    1: {"temperature": "0.7", "top_p": "0.9", "repetition_penalty": "1.1"},
    2: {"temperature": "1.0", "min_p": "0.05", "repetition_penalty": "1.15"},
    3: {"temperature": "0.6", "top_p": "0.85", "repetition_penalty": "1.2"},
}


@dataclass(frozen=True, order=True)
class Target:
    domain: str
    model_dir: str
    item: str
    view: str
    path: Path

    @property
    def model_size(self) -> str:
        prefix = self.model_dir.rsplit(f"_{self.domain}_w16", 1)[0]
        try:
            return MODEL_SIZES[prefix]
        except KeyError as exc:
            raise ValueError(f"unsupported audited model directory: {self.model_dir}") from exc


def load_targets(reports: list[Path], accepted_statuses=("FAIL",)) -> list[Target]:
    targets = set()
    for report in reports:
        with report.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                if row["status"] not in accepted_statuses:
                    continue
                path = Path(row["path"])
                parts = path.parts
                if len(parts) != 6 or parts[0] != "data" or parts[2] != "explanations":
                    raise ValueError(f"unexpected audit path: {path}")
                domain, model_dir, item, filename = parts[1], parts[3], parts[4], parts[5]
                if domain not in {"arxiv", "legal", "medical"}:
                    continue
                view = ("blog" if filename == "blogs.txt" else
                        "textbook" if filename == "textbook.txt" else
                        "stackexchange" if filename == "stackexchange.txt" else None)
                if view is None:
                    raise ValueError(f"failed unsupported view: {path}")
                targets.add(Target(domain, model_dir, item, view, path))
    return sorted(targets)


def backup_view(target: Target, backup_root: Path) -> None:
    item_dir = ROOT / target.path.parent
    destination = backup_root / target.domain / target.model_dir / target.item / target.view
    destination.mkdir(parents=True, exist_ok=False)
    for name in VIEW_FILES[target.view]:
        source = item_dir / name
        if source.is_dir():
            shutil.copytree(source, destination / name)
        elif source.exists():
            shutil.copy2(source, destination / name)
    manifest = item_dir / "generation_manifest.json"
    record = None
    if manifest.exists():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        record = data.get("views", {}).get(target.view)
    (destination / "manifest_view.json").write_text(
        json.dumps({"present": record is not None, "record": record}, indent=2) + "\n",
        encoding="utf-8",
    )


def restore_view(target: Target, backup_root: Path) -> None:
    item_dir = ROOT / target.path.parent
    source = backup_root / target.domain / target.model_dir / target.item / target.view
    if not source.is_dir():
        raise FileNotFoundError(f"missing backup for {target}: {source}")
    for name in VIEW_FILES[target.view]:
        current, saved = item_dir / name, source / name
        if current.is_dir():
            shutil.rmtree(current)
        elif current.exists():
            current.unlink()
        if saved.is_dir():
            shutil.copytree(saved, current)
        elif saved.exists():
            shutil.copy2(saved, current)
    manifest = item_dir / "generation_manifest.json"
    state = json.loads((source / "manifest_view.json").read_text(encoding="utf-8"))
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        data = {"version": 1, "views": {}}
    views = data.setdefault("views", {})
    if state["present"]:
        views[target.view] = state["record"]
    else:
        views.pop(target.view, None)
    if views:
        manifest.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    elif manifest.exists():
        manifest.unlink()


def command_for(group: list[Target], round_number: int) -> list[str]:
    first = group[0]
    profile = PROFILES[round_number]
    max_tokens = "8192" if round_number == 3 else "6144" if round_number == 2 else "12288"
    thinking = "1" if round_number == 3 else "0"
    pipeline_part = ("stack_exchange" if first.domain == "arxiv" else "qa") if first.view == "stackexchange" else first.view
    command = [
        str(LAUNCHER), "--models", first.model_size, "--domains", first.domain,
        "--papers", ",".join(target.item for target in group), "--parts", pipeline_part,
        "--max-workers", "16", "--reasoning-effort", "low", "--enable-thinking", thinking,
        "--temperature", profile["temperature"], "--repetition-penalty", profile["repetition_penalty"],
        "--max-tokens", max_tokens, "--time", "08:00:00",
    ]
    if "top_p" in profile:
        command += ["--top-p", profile["top_p"]]
    if "min_p" in profile:
        command += ["--min-p", profile["min_p"]]
    if round_number >= 2:
        command += ["--compact-prose", "1"]
    return command


def grouped(targets: list[Target]):
    groups = defaultdict(list)
    for target in targets:
        groups[(target.domain, target.model_dir, target.view)].append(target)
    return [sorted(values) for _, values in sorted(groups.items())]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--report", action="append", type=Path, dest="reports")
    parser.add_argument("--candidate-report", type=Path,
                        help="For retries, select SUSPECT/FAIL rows from a refreshed targeted audit TSV.")
    parser.add_argument("--model-size", choices=tuple(MODEL_SIZES.values()),
                        help="Restrict targets to one Qwen size.")
    parser.add_argument("--domain", choices=("arxiv", "legal", "medical"),
                        help="Restrict targets to one domain.")
    parser.add_argument("--backup-root", type=Path)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--backup-only", action="store_true",
                        help="Create round-one backups without submitting generation jobs.")
    parser.add_argument("--restore", action="store_true",
                        help="Restore selected non-PASS targets from --backup-root instead of submitting.")
    args = parser.parse_args()

    reports = [args.candidate_report] if args.candidate_report else (args.reports or list(DEFAULT_REPORTS))
    statuses = ("SUSPECT", "FAIL") if args.candidate_report else ("FAIL",)
    targets = load_targets(reports, statuses)
    if args.model_size:
        targets = [target for target in targets if target.model_size == args.model_size]
    if args.domain:
        targets = [target for target in targets if target.domain == args.domain]
    if not targets:
        raise SystemExit("No eligible targets found")
    if args.candidate_report:
        scope_reports = list(DEFAULT_REPORTS) + ([BINARY_REPORT] if BINARY_REPORT.exists() else [])
        original = set(load_targets(scope_reports, ("FAIL",)))
        unexpected = set(targets) - original
        if unexpected:
            raise SystemExit(f"candidate report expands original scope: {sorted(unexpected)[:3]}")

    backup_root = args.backup_root
    if args.submit and args.round == 1 and backup_root is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_root = ROOT / "data/.qwen_regeneration_backups" / stamp
    print(f"targets={len(targets)} groups={len(grouped(targets))} round={args.round}")
    print(f"backup_root={backup_root or '(not created in dry-run)'}")

    if args.restore:
        if backup_root is None:
            parser.error("--restore requires --backup-root")
        for target in targets:
            restore_view(target, backup_root)
        print(f"restored={len(targets)}")
        return

    if args.submit and args.round == 1:
        assert backup_root is not None
        backup_root.mkdir(parents=True, exist_ok=False)
        for target in targets:
            backup_view(target, backup_root)
        (backup_root / "targets.json").write_text(
            json.dumps([{"path": str(t.path), "view": t.view} for t in targets], indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"backed_up={len(targets)}")
        if args.backup_only:
            return

    for group in grouped(targets):
        command = command_for(group, args.round)
        print("COMMAND", subprocess.list2cmdline(command))
        if args.submit:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
