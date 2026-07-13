#!/usr/bin/env python3
"""Combine domain Luna review reports into source-specific union rejections."""

import argparse
import json
import os
import re
from pathlib import Path

VARIANT_RE = re.compile(r"^gpt_oss_(20b|120b)_(low|high)_recovery$")
DOMAIN_RE = re.compile(r"gpt_oss_(?:luna|sensibility)_(arxiv|medical|legal)_reviews")


def build(paths):
    rows = []
    for path in paths:
        payload = json.loads(Path(path).read_text())
        domain_match = DOMAIN_RE.search(Path(path).stem)
        if not domain_match: raise ValueError(f"cannot infer domain from {path}")
        domain = domain_match.group(1)
        for record in payload["records"]:
            verdict = record.get("verdict") or record.get("decision") or record.get("result")
            if verdict != "reject":
                continue
            recovery_variant = record.get("recovery_variant") or record.get("recovery")
            source_variant = record.get("source_variant") or record.get("source")
            match = VARIANT_RE.match(recovery_variant or "")
            if not match:
                raise ValueError(f"unexpected recovery variant: {recovery_variant}")
            if not source_variant:
                raise ValueError(f"missing source variant in {path}: {record}")
            evidence = record.get("reasons") or record.get("hard_failures") or record.get("evidence") or []
            if isinstance(evidence, str): evidence = [evidence]
            rows.append({
                "domain": domain, "model_size": match.group(1), "reasoning": match.group(2),
                "item": record["item"], "view": record["view"],
                "source_variant": source_variant, "reasons": evidence,
                "judge_model": record.get("judge_model", payload.get("judge_model", "gpt-5.6-luna")),
                "reasoning_effort": record.get("reasoning_effort", "low"),
            })
    keys = {(r["domain"], r["model_size"], r["reasoning"], r["item"], r["view"], r["source_variant"])
            for r in rows}
    if len(keys) != len(rows): raise ValueError("duplicate semantic rejection keys")
    return {"rubric": "sensibility_completeness_and_corruption_only",
            "judge_model": "mixed_review", "reasoning_effort": "low",
            "count": len(rows), "rejections": sorted(rows, key=lambda r: tuple(r[k] for k in
                ("domain", "model_size", "reasoning", "item", "view", "source_variant")))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = build(args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, args.output)
    print(f"wrote {payload['count']} rejections to {args.output}")


if __name__ == "__main__":
    main()
