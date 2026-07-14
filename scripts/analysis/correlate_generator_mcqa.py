"""Correlate generator MCQA accuracy with downstream MCQA performance."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from scipy.stats import pearsonr, spearmanr


EXPECTED_MODELS = (
    "glm_5_2_nvfp4",
    "gpt_5_mini_low",
    "gpt_5_mini_high",
    "gpt_5_4_mini_low",
    "gpt_5_4_mini_high",
    "gpt_oss_20b_low",
    "gpt_oss_120b_low",
    "gemma_4_12b",
    "gemma_4_31b_nvfp4",
    "glm_5_nvfp4",
)

DOWNSTREAM_LABELS = {
    "glm_5_2_nvfp4": "glm (E26)",
    "gpt_5_mini_low": "gpt_5_mini_low (E27)",
    "gpt_5_mini_high": "gpt_5_mini_high (E28)",
    "gpt_5_4_mini_low": "gpt_5_4_mini_low (E30)",
    "gpt_5_4_mini_high": "gpt_5_4_mini_high (E29)",
    "gpt_oss_20b_low": "gpt_oss_20b_low (E31)",
    "gpt_oss_120b_low": "gpt_oss_120b_low (E32)",
    "gemma_4_12b": "gemma_4_12b (E33)",
    "gemma_4_31b_nvfp4": "gemma_4_31b_nvfp4 (E34)",
    "glm_5_nvfp4": "glm_5_nvfp4 (E35)",
}

ACCURACY_FIELDS = (
    "constrained_factual_accuracy",
    "constrained_inference_accuracy",
    "reasoned_factual_accuracy",
    "reasoned_inference_accuracy",
)

OUTPUT_FIELDS = (
    "protocol",
    "family",
    "n",
    "pearson_r",
    "pearson_p",
    "spearman_rho",
    "spearman_p",
)


class InputValidationError(ValueError):
    """Raised when an input cannot form the complete ten-model panel."""


def _validate_accuracy(value: Any, context: str) -> float:
    if isinstance(value, bool):
        raise InputValidationError(f"{context} must be a number, not a boolean")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise InputValidationError(f"{context} must be numeric; got {value!r}") from exc
    if not math.isfinite(number):
        raise InputValidationError(f"{context} must be finite; got {value!r}")
    if not 0.0 <= number <= 1.0:
        raise InputValidationError(f"{context} must be between 0 and 1; got {number}")
    return number


def load_accuracies(path: Path) -> dict[str, dict[str, float]]:
    """Load and validate exactly one complete row per expected generator."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise InputValidationError(f"accuracy CSV {path} has no header")
        duplicates = sorted(
            {name for name in reader.fieldnames if reader.fieldnames.count(name) > 1}
        )
        if duplicates:
            raise InputValidationError(
                "accuracy CSV has duplicate columns: " + ", ".join(duplicates)
            )
        required_fields = {"model_key", *ACCURACY_FIELDS}
        missing_fields = sorted(required_fields - set(reader.fieldnames))
        if missing_fields:
            raise InputValidationError(
                "accuracy CSV is missing columns: " + ", ".join(missing_fields)
            )

        rows: dict[str, dict[str, float]] = {}
        for line_number, row in enumerate(reader, start=2):
            if None in row:
                raise InputValidationError(
                    f"accuracy CSV row {line_number} has more values than columns"
                )
            model_key = (row.get("model_key") or "").strip()
            if not model_key:
                raise InputValidationError(
                    f"accuracy CSV row {line_number} has an empty model_key"
                )
            if model_key in rows:
                raise InputValidationError(
                    f"accuracy CSV has duplicate model_key {model_key!r}"
                )
            if model_key not in EXPECTED_MODELS:
                raise InputValidationError(
                    f"accuracy CSV has unexpected model_key {model_key!r}"
                )
            rows[model_key] = {
                field: _validate_accuracy(
                    row.get(field), f"accuracy CSV {model_key}.{field}"
                )
                for field in ACCURACY_FIELDS
            }

    missing_models = [model for model in EXPECTED_MODELS if model not in rows]
    if missing_models:
        raise InputValidationError(
            "accuracy CSV is missing model rows: " + ", ".join(missing_models)
        )
    return rows


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise InputValidationError(f"downstream JSON has duplicate key {key!r}")
        result[key] = value
    return result


def load_downstream(path: Path) -> dict[str, dict[str, float]]:
    """Load E26--E35 factual and inference MCQA values by model key."""
    with path.open(encoding="utf-8") as handle:
        document = json.load(handle, object_pairs_hook=_unique_object)
    if not isinstance(document, dict):
        raise InputValidationError("downstream JSON root must be an object")

    downstream: dict[str, dict[str, float]] = {}
    for model_key in EXPECTED_MODELS:
        label = DOWNSTREAM_LABELS[model_key]
        if label not in document:
            raise InputValidationError(
                f"downstream JSON is missing model entry {label!r}"
            )
        row = document[label]
        if not isinstance(row, dict):
            raise InputValidationError(
                f"downstream JSON entry {label!r} must be an object"
            )
        missing = [field for field in ("fact_mcqa", "inf_mcqa") if field not in row]
        if missing:
            raise InputValidationError(
                f"downstream JSON entry {label!r} is missing: {', '.join(missing)}"
            )
        downstream[model_key] = {
            "fact_mcqa": _validate_accuracy(
                row["fact_mcqa"], f"downstream JSON {label}.fact_mcqa"
            ),
            "inf_mcqa": _validate_accuracy(
                row["inf_mcqa"], f"downstream JSON {label}.inf_mcqa"
            ),
        }
    return downstream


def calculate_correlations(
    accuracies: Mapping[str, Mapping[str, float]],
    downstream: Mapping[str, Mapping[str, float]],
) -> list[dict[str, Any]]:
    """Calculate factual-to-factual and inference-to-inference correlations."""
    rows: list[dict[str, Any]] = []
    for protocol in ("constrained", "reasoned"):
        for family, direct_suffix, downstream_field in (
            ("factual", "factual_accuracy", "fact_mcqa"),
            ("inference", "inference_accuracy", "inf_mcqa"),
        ):
            direct = [
                accuracies[model][f"{protocol}_{direct_suffix}"]
                for model in EXPECTED_MODELS
            ]
            trained = [downstream[model][downstream_field] for model in EXPECTED_MODELS]
            pearson_r, pearson_p = pearsonr(direct, trained)
            spearman_rho, spearman_p = spearmanr(direct, trained)
            rows.append(
                {
                    "protocol": protocol,
                    "family": family,
                    "n": len(EXPECTED_MODELS),
                    "pearson_r": float(pearson_r),
                    "pearson_p": float(pearson_p),
                    "spearman_rho": float(spearman_rho),
                    "spearman_p": float(spearman_p),
                }
            )
    return rows


def _write_csv_atomic(rows: Iterable[Mapping[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_path, output)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def correlate_to_csv(accuracies_path: Path, downstream_path: Path, output: Path) -> None:
    """Validate a complete panel, calculate correlations, and atomically write it."""
    accuracies = load_accuracies(accuracies_path)
    downstream = load_downstream(downstream_path)
    rows = calculate_correlations(accuracies, downstream)
    _write_csv_atomic(rows, output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accuracies", required=True, type=Path)
    parser.add_argument("--downstream-json", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        correlate_to_csv(args.accuracies, args.downstream_json, args.output)
    except (InputValidationError, json.JSONDecodeError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
