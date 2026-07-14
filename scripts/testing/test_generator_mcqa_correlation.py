import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.analysis.correlate_generator_mcqa import (  # noqa: E402
    ACCURACY_FIELDS,
    DOWNSTREAM_LABELS,
    EXPECTED_MODELS,
    InputValidationError,
    correlate_to_csv,
)


class GeneratorMcqaCorrelationTests(unittest.TestCase):
    def _write_accuracies(self, path: Path, models=EXPECTED_MODELS) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=("model_key", *ACCURACY_FIELDS)
            )
            writer.writeheader()
            for index, model_key in enumerate(models, start=1):
                value = index / 20
                writer.writerow(
                    {
                        "model_key": model_key,
                        "constrained_factual_accuracy": value,
                        "constrained_inference_accuracy": value,
                        "reasoned_factual_accuracy": value,
                        "reasoned_inference_accuracy": value,
                    }
                )

    def _write_downstream(self, path: Path) -> None:
        document = {}
        for index, model_key in enumerate(EXPECTED_MODELS, start=1):
            value = index / 20
            document[DOWNSTREAM_LABELS[model_key]] = {
                "fact_mcqa": value,
                "inf_mcqa": value,
            }
        path.write_text(json.dumps(document), encoding="utf-8")

    def test_complete_panel_writes_four_same_family_correlations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            accuracies = root / "accuracies.csv"
            downstream = root / "downstream.json"
            output = root / "correlations.csv"
            self._write_accuracies(accuracies)
            self._write_downstream(downstream)

            correlate_to_csv(accuracies, downstream, output)

            with output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)
            self.assertEqual(
                [(row["protocol"], row["family"]) for row in rows],
                [
                    ("constrained", "factual"),
                    ("constrained", "inference"),
                    ("reasoned", "factual"),
                    ("reasoned", "inference"),
                ],
            )
            for row in rows:
                self.assertEqual(row["n"], "10")
                self.assertAlmostEqual(float(row["pearson_r"]), 1.0)
                self.assertAlmostEqual(float(row["spearman_rho"]), 1.0)
                self.assertGreaterEqual(float(row["pearson_p"]), 0.0)
                self.assertGreaterEqual(float(row["spearman_p"]), 0.0)

    def test_missing_glm_5_2_row_refuses_to_create_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            accuracies = root / "accuracies.csv"
            downstream = root / "downstream.json"
            output = root / "correlations.csv"
            models = tuple(
                model for model in EXPECTED_MODELS if model != "glm_5_2_nvfp4"
            )
            self._write_accuracies(accuracies, models=models)
            self._write_downstream(downstream)

            with self.assertRaisesRegex(InputValidationError, "glm_5_2_nvfp4"):
                correlate_to_csv(accuracies, downstream, output)

            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
