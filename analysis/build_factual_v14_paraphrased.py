#!/usr/bin/env python3
"""Build factual v14 paraphrased probe files from short-target v14 probes.

The output mirrors each facts/probes_v14_short_targets.csv row set. Rows whose
short-target v14 probe/target are unchanged from v13 reuse the existing v13
paraphrased probe. Rows whose v14 cloze changed get a fresh probe paraphrase
while keeping the v14 target exactly fixed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import utils.utils as utils


SOURCES = ("arxiv", "legal", "medical")
FACTS_DIRNAME = "facts"
V13_FILENAME = "probes_v13.csv"
V13_PARAPHRASED_FILENAME = "probes_v13_paraphrased.csv"
DEFAULT_INPUT_FILENAME = "probes_v14_short_targets.csv"
OUTPUT_FILENAME = "probes_v14_paraphrased.csv"
REPORT_DIR = REPO_ROOT / "reports" / "factual_v14_paraphrased"
REPORT_FILENAME = "summary.csv"
DETAILS_FILENAME = "fresh_paraphrases.csv"

V13_IDENTITY_COLUMNS = ("raw_knowledge_statement", "fact", "contextualized_question")
PARAPHRASE_LOOKUP_COLUMNS = ("raw_knowledge_statement", "contextualized_question", "target")
METADATA_PRESERVE_COLUMNS = ("raw_knowledge_statement", "contextualized_question")
REQUIRED_COLUMNS = ("fact", "probe", "target")


SYSTEM_PROMPT = """You paraphrase factual cloze probe prefixes.

# Instructions
- Rewrite only the probe prefix in different wording.
- Preserve the exact meaning and make the result grammatical when followed immediately by the target.
- Do not include, reveal, or paraphrase the target answer in the probe prefix.
- Preserve proper nouns, titles, citations, equations, numeric values, and domain-specific terminology when needed for correctness.
- Return JSON only."""


USER_PROMPT_TEMPLATE = """Paraphrase the probe prefix while keeping the target separate.

Source-backed completed fact:
{fact}

Original probe prefix:
{probe}

Target to keep separate exactly:
{target}

Return JSON with exactly this key:
{{"probe": "<paraphrased probe prefix>"}}"""


@dataclass(frozen=True)
class BuildStats:
    source: str
    domain: str
    v14_rows: int
    carried_over: int
    fresh_paraphrased: int
    output_path: Path


def normalize_space(text: object) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def row_key(row: pd.Series, columns: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(str(row[column]) for column in columns)


def validate_columns(df: pd.DataFrame, path: Path, required: tuple[str, ...]) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{path.relative_to(REPO_ROOT)} missing columns: {missing}")


def build_unique_index(
    df: pd.DataFrame,
    path: Path,
    columns: tuple[str, ...],
) -> dict[tuple[str, ...], int]:
    validate_columns(df, path, columns + REQUIRED_COLUMNS)
    index: dict[tuple[str, ...], int] = {}
    duplicates: list[tuple[str, ...]] = []
    for row_index, row in df.iterrows():
        key = row_key(row, columns)
        if key in index:
            duplicates.append(key)
            continue
        index[key] = int(row_index)
    if duplicates:
        raise ValueError(
            f"{path.relative_to(REPO_ROOT)} has {len(duplicates)} duplicate identity keys"
        )
    return index


def cloze_changed(v13_row: pd.Series, v14_row: pd.Series) -> bool:
    return (
        str(v13_row["probe"]) != str(v14_row["probe"])
        or str(v13_row["target"]) != str(v14_row["target"])
    )


def parse_probe_response(response: object) -> str:
    if isinstance(response, dict):
        probe = response.get("probe", "")
    else:
        text = str(response).strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            probe = text
        else:
            probe = parsed.get("probe", "") if isinstance(parsed, dict) else ""
    probe = str(probe).strip()
    if not probe:
        raise ValueError("LLM returned an empty paraphrased probe")
    bad_control_chars = control_char_codes(probe)
    if bad_control_chars:
        raise ValueError(
            "LLM returned control characters in paraphrased probe: "
            + ", ".join(bad_control_chars[:10])
        )
    return probe


def control_char_codes(text: str) -> list[str]:
    return [
        f"U+{ord(ch):04X}"
        for ch in str(text)
        if ch not in "\n\t" and (ord(ch) < 32 or ord(ch) == 127)
    ]


def target_leaks_in_probe(probe: str, target: str) -> bool:
    target_norm = normalize_space(target).casefold()
    probe_norm = normalize_space(probe).casefold()
    if len(target_norm) < 3:
        return False
    return target_norm in probe_norm


def paraphrase_probe(
    probe: str,
    target: str,
    fact: str,
    model: str,
    reasoning_effort: str | None,
    dry_run: bool,
) -> str:
    if dry_run:
        return probe

    prompt = {
        "system": SYSTEM_PROMPT,
        "user": USER_PROMPT_TEMPLATE.format(fact=fact, probe=probe, target=target),
    }
    response = utils.query_llm(
        prompt=prompt,
        model=model,
        temperature=0.8,
        top_p=0.95,
        max_tokens=500,
        return_json=True,
        system_prompt_included=True,
        reasoning_effort=reasoning_effort,
    )
    paraphrased = parse_probe_response(response)
    if target_leaks_in_probe(paraphrased, target):
        raise ValueError(
            "Fresh paraphrase appears to include the held-out target: "
            f"target={target!r}, probe={paraphrased!r}"
        )
    return paraphrased


def discover_fact_dirs(sources: tuple[str, ...]) -> list[Path]:
    fact_dirs: list[Path] = []
    for source in sources:
        root = REPO_ROOT / "probes" / source
        if not root.exists():
            continue
        for domain_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            facts_dir = domain_dir / FACTS_DIRNAME
            if facts_dir.exists():
                fact_dirs.append(facts_dir)
    return fact_dirs


def build_one(
    facts_dir: Path,
    input_filename: str,
    model: str,
    reasoning_effort: str | None,
    dry_run: bool,
    write: bool,
) -> tuple[BuildStats, list[dict[str, object]]]:
    source = facts_dir.parents[1].name
    domain = facts_dir.parent.name
    v13_path = facts_dir / V13_FILENAME
    v13_para_path = facts_dir / V13_PARAPHRASED_FILENAME
    v14_path = facts_dir / input_filename
    output_path = facts_dir / OUTPUT_FILENAME

    for path in (v13_path, v13_para_path, v14_path):
        if not path.exists():
            raise FileNotFoundError(path.relative_to(REPO_ROOT))

    v13 = pd.read_csv(v13_path, keep_default_na=False)
    v13_para = pd.read_csv(v13_para_path, keep_default_na=False)
    v14 = pd.read_csv(v14_path, keep_default_na=False)

    v13_index = build_unique_index(v13, v13_path, V13_IDENTITY_COLUMNS)
    v13_para_index = build_unique_index(
        v13_para,
        v13_para_path,
        PARAPHRASE_LOOKUP_COLUMNS,
    )
    validate_columns(v14, v14_path, V13_IDENTITY_COLUMNS + REQUIRED_COLUMNS)

    output_rows: list[pd.Series] = []
    details: list[dict[str, object]] = []
    carried_over = 0
    fresh = 0

    for v14_row_index, v14_row in v14.iterrows():
        key = row_key(v14_row, V13_IDENTITY_COLUMNS)
        if key not in v13_index:
            raise ValueError(
                f"{v14_path.relative_to(REPO_ROOT)} row {v14_row_index} "
                "does not match a v13 identity key"
            )
        v13_row_index = v13_index[key]
        source_row = v13.loc[v13_row_index]
        output_row = v14_row.copy()
        para_key = row_key(source_row, PARAPHRASE_LOOKUP_COLUMNS)
        existing_para_has_control_chars = False
        if para_key in v13_para_index:
            existing_para = str(v13_para.loc[v13_para_index[para_key], "probe"])
            existing_para_has_control_chars = bool(control_char_codes(existing_para))
        needs_fresh = (
            cloze_changed(source_row, v14_row)
            or para_key not in v13_para_index
            or existing_para_has_control_chars
        )

        reason = ""
        if cloze_changed(source_row, v14_row):
            reason = "cloze_changed"
        elif para_key not in v13_para_index:
            reason = "missing_v13_paraphrase"
        elif existing_para_has_control_chars:
            reason = "v13_paraphrase_control_chars_source_fallback"

        if needs_fresh:
            if reason == "v13_paraphrase_control_chars_source_fallback":
                new_probe = str(v14_row["probe"])
            else:
                new_probe = paraphrase_probe(
                    probe=str(v14_row["probe"]),
                    target=str(v14_row["target"]),
                    fact=str(v14_row["fact"]),
                    model=model,
                    reasoning_effort=reasoning_effort,
                    dry_run=dry_run,
                )
            fresh += 1
            details.append(
                {
                    "source": source,
                    "domain": domain,
                    "v14_row_index": int(v14_row_index),
                    "v13_row_index": int(v13_row_index),
                    "reason": reason,
                    "old_v13_probe": source_row["probe"],
                    "old_v13_target": source_row["target"],
                    "v14_probe": v14_row["probe"],
                    "v14_target": v14_row["target"],
                    "paraphrased_probe": new_probe,
                }
            )
        else:
            paraphrased_row = v13_para.loc[v13_para_index[para_key]]
            new_probe = str(paraphrased_row["probe"])
            carried_over += 1

        output_row["probe"] = new_probe
        output_row["target"] = str(v14_row["target"])
        output_row["fact"] = new_probe + str(v14_row["target"])
        output_rows.append(output_row)

    output = pd.DataFrame(output_rows, columns=v14.columns)
    validate_output(output, v14, output_path)
    if write:
        output.to_csv(output_path, index=False)

    return (
        BuildStats(
            source=source,
            domain=domain,
            v14_rows=len(v14),
            carried_over=carried_over,
            fresh_paraphrased=fresh,
            output_path=output_path,
        ),
        details,
    )


def validate_output(output: pd.DataFrame, v14: pd.DataFrame, output_path: Path) -> None:
    if list(output.columns) != list(v14.columns):
        raise AssertionError(f"Column mismatch for {output_path.relative_to(REPO_ROOT)}")
    if len(output) != len(v14):
        raise AssertionError(f"Row count mismatch for {output_path.relative_to(REPO_ROOT)}")
    bad_fact = output["fact"].astype(str) != (
        output["probe"].astype(str) + output["target"].astype(str)
    )
    if bad_fact.any():
        bad_indices = output.index[bad_fact].tolist()[:10]
        raise AssertionError(
            f"fact != probe + target in {output_path.relative_to(REPO_ROOT)} "
            f"at rows {bad_indices}"
        )
    for column in METADATA_PRESERVE_COLUMNS:
        if not output[column].astype(str).equals(v14[column].astype(str)):
            raise AssertionError(
                f"Identity column {column} changed in {output_path.relative_to(REPO_ROOT)}"
            )


def write_reports(stats: list[BuildStats], details: list[dict[str, object]]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(
        [
            {
                "source": item.source,
                "domain": item.domain,
                "v14_rows": item.v14_rows,
                "carried_over": item.carried_over,
                "fresh_paraphrased": item.fresh_paraphrased,
                "output_path": str(item.output_path.relative_to(REPO_ROOT)),
            }
            for item in stats
        ]
    )
    summary.to_csv(REPORT_DIR / REPORT_FILENAME, index=False)
    pd.DataFrame(details).to_csv(REPORT_DIR / DETAILS_FILENAME, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=SOURCES,
        default=list(SOURCES),
        help="Probe source roots to process.",
    )
    parser.add_argument(
        "--input-filename",
        default=DEFAULT_INPUT_FILENAME,
        help="Factual v14 source filename inside each facts directory.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4.1-mini",
        help="Model used for fresh paraphrases.",
    )
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        help="Optional reasoning effort passed through to utils.query_llm.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and report without calling the LLM; changed rows reuse v14 probe.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Validate without writing probe outputs or reports.",
    )
    args = parser.parse_args()

    stats: list[BuildStats] = []
    details: list[dict[str, object]] = []
    for facts_dir in discover_fact_dirs(tuple(args.sources)):
        item_stats, item_details = build_one(
            facts_dir=facts_dir,
            input_filename=args.input_filename,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            dry_run=args.dry_run,
            write=not args.no_write,
        )
        stats.append(item_stats)
        details.extend(item_details)

    if not args.no_write:
        write_reports(stats, details)

    total_rows = sum(item.v14_rows for item in stats)
    total_carried = sum(item.carried_over for item in stats)
    total_fresh = sum(item.fresh_paraphrased for item in stats)
    print(f"Processed documents: {len(stats)}")
    print(f"Rows: {total_rows}")
    print(f"Carried over: {total_carried}")
    print(f"Fresh paraphrased: {total_fresh}")
    if not args.no_write:
        print(f"Wrote {REPORT_DIR.relative_to(REPO_ROOT) / REPORT_FILENAME}")
        print(f"Wrote {REPORT_DIR.relative_to(REPO_ROOT) / DETAILS_FILENAME}")


if __name__ == "__main__":
    main()
