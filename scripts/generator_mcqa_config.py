"""Configuration for the auxiliary-view generator MCQA benchmark.

The keys in this module deliberately match the generator conditions used by
experiments E26--E35.  Keeping the mapping in one importable module prevents
the evaluator, vLLM launcher, and correlation analysis from drifting apart.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GeneratorModel:
    key: str
    experiment: str
    provider: str
    model_id: str
    reasoning_effort: str | None = None
    data_parallel_size: int = 1
    tensor_parallel_size: int = 1
    max_workers: int = 1
    reasoning_parser: str | None = None
    quantization: str | None = None
    trust_remote_code: bool = False
    enable_expert_parallel: bool = False


MODELS: tuple[GeneratorModel, ...] = (
    GeneratorModel(
        key="glm_5_2_nvfp4",
        experiment="E26",
        provider="litellm",
        model_id="nvidia/GLM-5.2-NVFP4",
        max_workers=8,
    ),
    GeneratorModel(
        key="gpt_5_mini_low",
        experiment="E27",
        provider="openai",
        model_id="gpt-5-mini-2025-08-07",
        reasoning_effort="low",
        max_workers=32,
    ),
    GeneratorModel(
        key="gpt_5_mini_high",
        experiment="E28",
        provider="openai",
        model_id="gpt-5-mini-2025-08-07",
        reasoning_effort="high",
        max_workers=32,
    ),
    GeneratorModel(
        key="gpt_5_4_mini_high",
        experiment="E29",
        provider="openai",
        model_id="gpt-5.4-mini-2026-03-17",
        reasoning_effort="high",
        max_workers=32,
    ),
    GeneratorModel(
        key="gpt_5_4_mini_low",
        experiment="E30",
        provider="openai",
        model_id="gpt-5.4-mini-2026-03-17",
        reasoning_effort="low",
        max_workers=32,
    ),
    GeneratorModel(
        key="gpt_oss_20b_low",
        experiment="E31",
        provider="vllm",
        model_id="openai/gpt-oss-20b",
        reasoning_effort="low",
        data_parallel_size=8,
        tensor_parallel_size=1,
        max_workers=32,
        reasoning_parser="openai_gptoss",
    ),
    GeneratorModel(
        key="gpt_oss_120b_low",
        experiment="E32",
        provider="vllm",
        model_id="openai/gpt-oss-120b",
        reasoning_effort="low",
        data_parallel_size=4,
        tensor_parallel_size=2,
        max_workers=16,
        reasoning_parser="openai_gptoss",
    ),
    GeneratorModel(
        key="gemma_4_12b",
        experiment="E33",
        provider="vllm",
        model_id="google/gemma-4-12B-it",
        data_parallel_size=8,
        tensor_parallel_size=1,
        max_workers=32,
        reasoning_parser="gemma4",
        trust_remote_code=True,
    ),
    GeneratorModel(
        key="gemma_4_31b_nvfp4",
        experiment="E34",
        provider="vllm",
        model_id="nvidia/Gemma-4-31B-IT-NVFP4",
        data_parallel_size=8,
        tensor_parallel_size=1,
        max_workers=32,
        reasoning_parser="gemma4",
        quantization="modelopt",
        trust_remote_code=True,
    ),
    GeneratorModel(
        key="glm_5_nvfp4",
        experiment="E35",
        provider="vllm",
        model_id="nvidia/GLM-5-NVFP4",
        data_parallel_size=1,
        tensor_parallel_size=8,
        max_workers=8,
        reasoning_parser="glm45",
        trust_remote_code=True,
        enable_expert_parallel=True,
    ),
)

MODEL_BY_KEY = {model.key: model for model in MODELS}

if len(MODEL_BY_KEY) != len(MODELS):
    raise RuntimeError("Duplicate generator benchmark model key")


PROTOCOLS = ("constrained", "reasoned")
FAMILIES = ("factual", "inference")
EXPECTED_COUNTS = {"factual": 4515, "inference": 322}
COMPLETION_TOKEN_CAP = 8192
SERVED_MODEL_LENGTH = 16384
