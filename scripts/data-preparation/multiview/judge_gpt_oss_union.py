#!/usr/bin/env python3
"""Resumable source-grounded Luna judge for GPT-OSS recovery unions."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
import utils.utils as utils  # noqa: E402
from utils.multiview_recovery import ASSEMBLED, GRANULAR_LAYOUTS  # noqa: E402

DATA_ROOT = PROJECT_ROOT / "data"
JUDGE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["coherence", "grounding", "instruction_adherence", "corruption",
                 "verdict", "hard_failures", "evidence"],
    "properties": {
        "coherence": {"type": "integer", "minimum": 1, "maximum": 5},
        "grounding": {"type": "integer", "minimum": 1, "maximum": 5},
        "instruction_adherence": {"type": "integer", "minimum": 1, "maximum": 5},
        "corruption": {"type": "integer", "minimum": 1, "maximum": 5},
        "verdict": {"type": "string", "enum": ["pass", "reject"]},
        "hard_failures": {"type": "array", "items": {"type": "string"}},
        "evidence": {"type": "array", "minItems": 1, "items": {"type": "string"}},
    },
}


def _atomic_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def enumerate_selected(domain: str, data_root=DATA_ROOT):
    root = Path(data_root) / domain / "explanations"
    rows = []
    for manifest_path in sorted(root.glob("gpt_oss_*_recovery/union_manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        for selected in manifest.get("selected", []):
            rows.append({"domain": domain, "model_size": manifest["model_size"],
                         "reasoning": manifest["reasoning"],
                         "recovery_variant": manifest["recovery_variant"], **selected})
    return rows


def source_path(row, data_root=DATA_ROOT):
    suffix = ".tex" if row["domain"] == "arxiv" else ".txt"
    return Path(data_root) / row["domain"] / "cleaned" / f"{row['item']}{suffix}"


def build_prompt(row, data_root=DATA_ROOT):
    item_dir = Path(data_root) / row["domain"] / "explanations" / row["recovery_variant"] / row["item"]
    source = source_path(row, data_root).read_text(encoding="utf-8", errors="replace")
    assembled = (item_dir / ASSEMBLED[row["view"]]).read_text(encoding="utf-8", errors="replace")
    granular_dir = item_dir / GRANULAR_LAYOUTS[row["view"]][0]
    boundaries = []
    for path in sorted(granular_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="replace")
        boundaries.append(f"## {path.name}\nSTART: {text[:400]}\nEND: {text[-600:]}")
    system = """You are a strict source-grounded corpus quality judge. Evaluate one generated
view against its supplied source. Reject for any material fabricated fact, result, authority,
quotation, number, reversed holding, contradiction, incoherence, repetition loop, reserved or
Harmony token, malformed text, or abrupt truncation. Broader pedagogy is allowed only when it is
accurate and does not masquerade as a source claim. Scores are 1-5; pass requires every score >=4
and no hard failure. Return only one JSON object with exactly these keys:
coherence, grounding, instruction_adherence, corruption (integer 1-5); verdict (pass or reject);
hard_failures (array of strings); evidence (nonempty array of specific strings)."""
    user = f"""DOMAIN: {row['domain']}
ITEM: {row['item']}
VIEW: {row['view']}
SOURCE VARIANT: {row['source_variant']}

### AUTHORITATIVE SOURCE
{source}

### GENERATED ASSEMBLED VIEW
{assembled}

### GRANULAR BOUNDARY CHECKS
{chr(10).join(boundaries)}
"""
    return {"system": system, "user": user}


def run(domain, output, *, model="gpt-5.6-luna", reasoning="low", retries=5,
        data_root=DATA_ROOT, limit=None):
    output = Path(output)
    try: payload = json.loads(output.read_text())
    except (OSError, json.JSONDecodeError):
        payload = {"judge_model": model, "reasoning_effort": reasoning, "records": []}
    for record in payload["records"]:
        if "verdict" not in record and record.get("decision") in {"pass", "reject", "unjudged"}:
            record["verdict"] = record["decision"]
    selected_rows = enumerate_selected(domain, data_root)
    selected_by_key = {(r["recovery_variant"], r["item"], r["view"]): r for r in selected_rows}
    existing = {}
    for record in payload["records"]:
        key = (record["recovery_variant"], record["item"], record["view"])
        selected = selected_by_key.get(key)
        if (selected and record.get("luna_response_status") == "ok"
                and record.get("source_variant") == selected.get("source_variant")):
            existing[key] = record
    pending = [r for r in selected_rows
               if (r["recovery_variant"], r["item"], r["view"]) not in existing]
    if limit is not None: pending = pending[:limit]
    for index, row in enumerate(pending, 1):
        key = (row["recovery_variant"], row["item"], row["view"])
        error = None
        for attempt in range(1, retries + 1):
            try:
                result = utils.query_llm(
                    build_prompt(row, data_root), model=model, reasoning_effort=reasoning,
                    system_prompt_included=True, return_json=True,
                    max_tokens=8000,
                )
                if isinstance(result, str):
                    result = json.loads(result)
                if hasattr(result, "model_dump"):
                    result = result.model_dump()
                if not isinstance(result, dict): raise ValueError("judge did not return an object")
                required = set(JUDGE_SCHEMA["required"])
                if not required.issubset(result):
                    raise ValueError(f"judge omitted keys: {sorted(required - set(result))}")
                record = {**row, "judge_model": model, "reasoning_effort": reasoning,
                          "luna_response_status": "ok", **result}
                existing[key] = record
                break
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                if attempt < retries: time.sleep(min(5 * attempt, 20))
        else:
            existing[key] = {**row, "judge_model": model, "reasoning_effort": reasoning,
                             "luna_response_status": "error", "error": error,
                             "verdict": "unjudged"}
        payload["records"] = [existing[k] for k in sorted(existing)]
        counts = {name: sum(r.get("verdict") == name for r in payload["records"])
                  for name in ("pass", "reject", "unjudged")}
        payload["counts"] = {"total_selected": len(selected_rows), **counts,
                             "completed": sum(r.get("luna_response_status") == "ok" for r in payload["records"])}
        _atomic_json(output, payload)
        print(f"[{index}/{len(pending)}] {key}: {existing[key].get('verdict')}", flush=True)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True, choices=("arxiv", "medical", "legal"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    args = parser.parse_args()
    run(args.domain, args.output, model=args.model, reasoning=args.reasoning_effort,
        retries=args.retries, data_root=args.data_root, limit=args.limit)


if __name__ == "__main__":
    main()
