import json
import tempfile
import unittest
import importlib.util
from pathlib import Path
from types import SimpleNamespace

from utils import utils
from utils.multiview_recovery import content_reasons, manifest_valid, record_validated_view, validate_view


class SequenceClient:
    def __init__(self, responses):
        self.responses, self.calls = iter(responses), []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        content, finish = next(self.responses)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish)])


class RecoveryTests(unittest.TestCase):
    def test_harmony_and_repetition_are_hard_failures(self):
        self.assertIn("harmony_marker", content_reasons("<|channel|>" + " coherent" * 50))
        self.assertIn("line_repetition_loop", content_reasons(
            "This substantive sentence repeats without adding any new information.\n" * 8
        ))
        self.assertNotIn("line_repetition_loop", content_reasons(
            "--------------------------------------------------------------------\n" * 8
        ))
        self.assertIn("reserved_token_leakage", content_reasons(
            "Coherent prefix " * 10 + "<|reserved_123|>"))
        self.assertIn("unicode_replacement_character", content_reasons(
            "Coherent prefix " * 10 + "\ufffd"))
        self.assertIn("malformed_trailing_backslash", content_reasons(
            "Coherent explanatory prose. " * 10 + "\\"))

    def test_json_latex_repair_covers_bm_and_frac(self):
        raw = r'{"math":"\bm{x}=\frac{1}{2}"}'
        repaired = utils._repair_json_latex_escapes(raw)
        self.assertEqual(json.loads(repaired)["math"], r"\bm{x}=\frac{1}{2}")

    def test_stop_empty_is_not_retried_or_resampled(self):
        client = SequenceClient([("", "stop"), ("should not run", "stop")])
        result = utils._create_vllm_completion(client, {"model": "m", "max_tokens": 10})
        self.assertEqual(result.choices[0].message.content, "")
        self.assertEqual(len(client.calls), 1)

    def test_length_empty_retries_once_with_fixed_sampling(self):
        client = SequenceClient([("", "length"), ("complete", "stop")])
        utils._create_vllm_completion(client, {"model": "m", "max_tokens": 10, "temperature": .2}, max_cap=20)
        self.assertEqual([call["max_tokens"] for call in client.calls], [10, 20])
        self.assertEqual([call["temperature"] for call in client.calls], [.2, .2])

    def test_manifest_resume_requires_matching_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            item = Path(tmp)
            (item / "stack_exchange_outline.json").write_text(json.dumps({"questions": [{"title": "Useful title", "question_body": "Detailed question"}]}))
            child = "\\title{Q}\n\n### Useful title\nQuestion:\nDetailed question\nAnswer:\n" + "A grounded explanatory sentence. " * 10
            (item / "stackexchange").mkdir()
            (item / "stackexchange/stack_01.txt").write_text(child)
            (item / "stackexchange.txt").write_text(child)
            self.assertTrue(validate_view(item, "stackexchange")["valid"])
            record_validated_view(item, "stackexchange", {"attempts": 1, "finish_reason": "stop"})
            self.assertTrue(manifest_valid(item, "stackexchange"))
            (item / "stackexchange.txt").write_text(child + "changed")
            self.assertFalse(manifest_valid(item, "stackexchange"))

    def test_union_ranking_prefers_validated_recovery_then_canonical(self):
        script = Path(__file__).parents[1] / "data-preparation/multiview/audit_gpt_oss_20b.py"
        spec = importlib.util.spec_from_file_location("gpt_oss_audit", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        canonical = "gpt_oss_20b_low"
        recovery = canonical + "_recovery"
        self.assertLess(
            module._candidate_rank({"variant": recovery}, canonical, recovery),
            module._candidate_rank({"variant": canonical}, canonical, recovery),
        )
        self.assertLess(
            module._candidate_rank({"variant": canonical}, canonical, recovery),
            module._candidate_rank({"variant": canonical + "_64k"}, canonical, recovery),
        )

    def test_semantic_rejections_are_source_specific(self):
        script = Path(__file__).parents[1] / "data-preparation/multiview/audit_gpt_oss_20b.py"
        spec = importlib.util.spec_from_file_location("gpt_oss_audit_rejections", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rejections.json"
            path.write_text(json.dumps({"rejections": [{
                "domain": "arxiv", "model_size": "20b", "reasoning": "high",
                "item": "BOFT", "view": "blog",
                "source_variant": "gpt_oss_20b_high_recovery",
                "reasons": ["fabricated results"],
            }]}))
            rejected, details = module._load_rejections(path)
            key = ("arxiv", "20b", "high", "BOFT", "blog",
                   "gpt_oss_20b_high_recovery")
            self.assertIn(key, rejected)
            self.assertEqual(details[key], ["fabricated results"])


if __name__ == "__main__":
    unittest.main()
