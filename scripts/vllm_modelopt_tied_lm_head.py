#!/usr/bin/env python3
"""Run vLLM with a narrow ModelOpt tied-output-head compatibility fix.

vLLM 0.24 assigns ModelOpt-excluded ``ParallelLMHead`` layers an
``UnquantizedLinearMethod``.  That method cannot tie the output head to the
input embedding, so tied checkpoints such as NVIDIA's Gemma 4 NVFP4 abort in
model construction.  Returning ``None`` for this one excluded layer lets
``ParallelLMHead`` select its normal ``UnquantizedEmbeddingMethod``, whose
weight-tying implementation shares the checkpoint's embedding tensor.

This wrapper does not change quantization for any non-excluded layer.
"""

from __future__ import annotations

import sys

from vllm.entrypoints.cli.main import main
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptQuantConfigBase,
)
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead


_ORIGINAL_GET_QUANT_METHOD = ModelOptQuantConfigBase.get_quant_method


def _get_quant_method_with_tied_head_support(self, layer, prefix):
    if isinstance(layer, ParallelLMHead) and self.is_layer_excluded(prefix):
        return None
    return _ORIGINAL_GET_QUANT_METHOD(self, layer, prefix)


ModelOptQuantConfigBase.get_quant_method = _get_quant_method_with_tied_head_support


if __name__ == "__main__":
    sys.exit(main())
