import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import evaluate_generator_mcqa as evaluator  # noqa: E402
from scripts.generator_mcqa_config import (  # noqa: E402
    COMPLETION_TOKEN_CAP,
    MODEL_BY_KEY,
    PROTOCOLS,
)


def _question(question_id: str = "factual:arxiv/domain:0") -> evaluator.ProbeQuestion:
    return evaluator.ProbeQuestion(
        question_id=question_id,
        family=question_id.split(":", 1)[0],
        group="arxiv",
        domain="domain",
        row_index=0,
        prompt="Five demonstrations followed by a test question",
        correct_label="B",
        prompt_sha256=f"hash-{question_id}",
    )


def _state_record(
    question: evaluator.ProbeQuestion,
    model_id: str,
    *,
    terminal: bool,
    predicted_label: str | None,
    correct: bool | None,
) -> dict:
    return {
        "question_id": question.question_id,
        "prompt_sha256": question.prompt_sha256,
        "model_id": model_id,
        "terminal": terminal,
        "predicted_label": predicted_label,
        "correct": correct,
    }


class LabelAndParserTests(unittest.TestCase):
    def test_only_constrained_protocol_is_configured(self):
        self.assertEqual(PROTOCOLS, ("constrained",))

    def test_normalize_label_accepts_only_a_through_e(self):
        self.assertEqual(evaluator.normalize_label("A"), "A")
        self.assertEqual(evaluator.normalize_label(" (b) "), "B")
        with self.assertRaises(ValueError):
            evaluator.normalize_label("Answer: A")
        with self.assertRaises(ValueError):
            evaluator.normalize_label("F")

    def test_constrained_parser(self):
        self.assertEqual(
            evaluator.parse_answer("constrained", '{"answer": "c"}'),
            ("C", "parsed"),
        )
        self.assertEqual(
            evaluator.parse_answer("constrained", "not json"),
            (None, "invalid_json"),
        )
        self.assertEqual(
            evaluator.parse_answer("constrained", '{"answer": "F"}'),
            (None, "invalid_schema"),
        )

    def test_unknown_protocol_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported answer protocol"):
            evaluator.parse_answer("reasoned", '{"answer": "D"}')


class ProbeDiscoveryTests(unittest.TestCase):
    def test_repository_probe_versions_have_the_exact_expected_panel(self):
        probes_root = ROOT / "probes"
        expected = {"factual": 4515, "inference": 322}

        for family, expected_count in expected.items():
            with self.subTest(family=family):
                self.assertEqual(len(evaluator._probe_paths(probes_root, family)), 36)
                questions = evaluator.load_questions(probes_root, family)
                self.assertEqual(len(questions), expected_count)
                self.assertEqual(len({question.question_id for question in questions}), expected_count)
                self.assertTrue(all(question.family == family for question in questions))


class RequestParameterTests(unittest.TestCase):
    def test_openai_uses_completion_cap_and_pinned_reasoning_effort(self):
        question = _question()
        expected_efforts = {
            "gpt_5_mini_low": "low",
            "gpt_5_mini_high": "high",
            "gpt_5_4_mini_low": "low",
            "gpt_5_4_mini_high": "high",
        }
        for model_key, effort in expected_efforts.items():
            with self.subTest(model_key=model_key):
                model = MODEL_BY_KEY[model_key]
                constrained = evaluator._request_params(model, "constrained", question)
                self.assertEqual(
                    constrained["max_completion_tokens"], COMPLETION_TOKEN_CAP
                )
                self.assertEqual(constrained["reasoning_effort"], effort)
                self.assertNotIn("max_tokens", constrained)
                self.assertNotIn("temperature", constrained)
                self.assertEqual(constrained["response_format"], evaluator.ANSWER_SCHEMA)

    def test_vllm_uses_greedy_max_tokens_and_structured_output(self):
        model = MODEL_BY_KEY["gpt_oss_20b_low"]
        question = _question()
        constrained = evaluator._request_params(model, "constrained", question)
        self.assertEqual(constrained["max_tokens"], COMPLETION_TOKEN_CAP)
        self.assertEqual(constrained["temperature"], 0.0)
        self.assertEqual(constrained["seed"], 0)
        self.assertEqual(constrained["reasoning_effort"], "low")
        self.assertNotIn("max_completion_tokens", constrained)
        self.assertEqual(constrained["response_format"], evaluator.ANSWER_SCHEMA)


class ExecutionSafetyAndResumeTests(unittest.TestCase):
    def test_insufficient_quota_is_not_retried_as_a_transient_429(self):
        error = RuntimeError("429: insufficient_quota; exceeded your current quota")
        self.assertFalse(evaluator._is_retryable(error))

    def test_litellm_requires_explicit_opt_in_before_building_client(self):
        with mock.patch.object(evaluator, "build_client") as build_client:
            with self.assertRaisesRegex(SystemExit, "allow-litellm"):
                evaluator.main(["--model-key", "glm_5_2_nvfp4"])
        build_client.assert_not_called()

    def test_resume_skips_terminal_and_retries_nonterminal_records(self):
        model = MODEL_BY_KEY["gpt_oss_20b_low"]
        complete_question = _question("factual:arxiv/domain:0")
        retry_question = _question("factual:arxiv/domain:1")

        with tempfile.TemporaryDirectory() as tmp:
            state_root = Path(tmp)
            path = evaluator.state_path(state_root, model.key, "constrained", "factual")
            path.parent.mkdir(parents=True)
            records = [
                _state_record(
                    complete_question,
                    model.model_id,
                    terminal=True,
                    predicted_label="B",
                    correct=True,
                ),
                _state_record(
                    retry_question,
                    model.model_id,
                    terminal=False,
                    predicted_label=None,
                    correct=None,
                ),
            ]
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            retried_record = _state_record(
                retry_question,
                model.model_id,
                terminal=True,
                predicted_label="A",
                correct=False,
            )

            with mock.patch.object(
                evaluator, "evaluate_one", return_value=retried_record
            ) as evaluate_one:
                completed, failures = evaluator.run_partition(
                    client=mock.Mock(),
                    model=model,
                    protocol="constrained",
                    family="factual",
                    questions=[complete_question, retry_question],
                    state_root=state_root,
                    max_workers=1,
                    max_attempts=1,
                )

            self.assertEqual((completed, failures), (2, 0))
            evaluate_one.assert_called_once()
            self.assertEqual(evaluate_one.call_args.args[3], retry_question)
            resumed = evaluator.load_state(path)
            self.assertTrue(resumed[complete_question.question_id]["terminal"])
            self.assertTrue(resumed[retry_question.question_id]["terminal"])
            self.assertEqual(resumed[retry_question.question_id]["predicted_label"], "A")


class SummaryAggregationTests(unittest.TestCase):
    def test_summary_counts_invalid_terminal_answers_as_wrong_and_requires_denominator(self):
        model = MODEL_BY_KEY["gpt_oss_20b_low"]
        expected_counts = {"factual": 2, "inference": 2}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_root = root / "state"
            summary_path = root / "accuracies.csv"
            for protocol in PROTOCOLS:
                for family in expected_counts:
                    first = _question(f"{family}:arxiv/domain:0")
                    second = _question(f"{family}:arxiv/domain:1")
                    path = evaluator.state_path(state_root, model.key, protocol, family)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(
                        json.dumps(
                            _state_record(
                                first,
                                model.model_id,
                                terminal=True,
                                predicted_label="B",
                                correct=True,
                            )
                        )
                        + "\n"
                        + json.dumps(
                            _state_record(
                                second,
                                model.model_id,
                                terminal=True,
                                predicted_label=None,
                                correct=False,
                            )
                        )
                        + "\n",
                        encoding="utf-8",
                    )

            with (
                mock.patch.object(evaluator, "MODELS", (model,)),
                mock.patch.object(evaluator, "EXPECTED_COUNTS", expected_counts),
                mock.patch.object(evaluator, "_probe_paths", return_value=[]),
            ):
                evaluator.write_summary(state_root, summary_path, ROOT / "probes")

            with summary_path.open(newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["status"], "complete")
            self.assertNotIn("reasoned_factual_accuracy", row)
            for protocol in PROTOCOLS:
                for family in expected_counts:
                    prefix = f"{protocol}_{family}"
                    self.assertEqual(float(row[f"{prefix}_accuracy"]), 0.5)
                    self.assertEqual(int(row[f"{prefix}_correct"]), 1)
                    self.assertEqual(int(row[f"{prefix}_total"]), 2)
                    self.assertEqual(int(row[f"{prefix}_invalid"]), 1)

            manifest = json.loads(
                summary_path.with_name("run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["schema_version"], 2)
            self.assertEqual(manifest["protocols"], ["constrained"])

            partition = evaluator.state_path(
                state_root, model.key, "constrained", "factual"
            )
            self.assertIsNone(evaluator._partition_metrics(partition, expected=3))


if __name__ == "__main__":
    unittest.main()
