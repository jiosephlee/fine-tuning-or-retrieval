import json
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import utils.utils as utils


class FakeCompletions:
    def __init__(self, content):
        self.content = content
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content=self.content),
                finish_reason="stop",
                logprobs=None,
            )]
        )


class FakeClient:
    def __init__(self, content):
        self.chat = SimpleNamespace(completions=FakeCompletions(content))


class VllmProviderTests(unittest.TestCase):
    def test_explicit_provider_does_not_use_namespace_heuristic(self):
        self.assertEqual(utils.resolve_llm_provider("openai/gpt-oss-20b", "vllm"), "vllm")
        self.assertEqual(utils.resolve_llm_provider("openai/gpt-oss-20b", "auto"), "litellm")

    def test_plain_completion_uses_vllm_client_and_preserves_model(self):
        fake = FakeClient("hello")
        with patch.object(utils, "get_vllm_client", return_value=fake):
            response = utils.query_gpt(
                "say hello",
                model="openai/gpt-oss-20b",
                provider="vllm",
                max_tokens=321,
                system_prompt_included=False,
            )
        self.assertEqual(response, "hello")
        request = fake.chat.completions.calls[0]
        self.assertEqual(request["model"], "openai/gpt-oss-20b")
        self.assertEqual(request["max_tokens"], 321)
        self.assertNotIn("seed", request)

    def test_schema_completion_parses_json_without_beta_parser(self):
        fake = FakeClient(json.dumps({"ready": True}))
        with patch.object(utils, "get_vllm_client", return_value=fake):
            response = utils.query_gpt(
                "return ready",
                model="openai/gpt-oss-20b",
                provider="vllm",
                return_json=True,
                json_schema={
                    "type": "json_schema",
                    "json_schema": {"name": "ready", "schema": {"type": "object"}},
                },
                system_prompt_included=False,
            )
        self.assertEqual(response, {"ready": True})
        self.assertIn("response_format", fake.chat.completions.calls[0])


if __name__ == "__main__":
    unittest.main()
