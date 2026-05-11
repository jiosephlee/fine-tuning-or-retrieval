# MCQA Restoration Review - arxiv/fa3 v14

## Counts

- Rejected rows reviewed: 56
- Restore: 11
- Keep rejected: 45
- Needs source check: 0

## Restore Indices

12, 39, 64, 96, 161, 167, 183, 189, 198, 224, 247

## Restore Issue Types

- filter_overstrict: 11
- technical_term_memorization: 7
- numeric_memorization: 3
- formula_or_symbol_memorization: 2

## Recommended Restores

- `12` target `a baseline FP8 attention.`: 2.6x error comparison against baseline FP8 attention is source-backed and distractible.
- `39` target `non-GEMM operations.`: Non-GEMM operations is a specific implementation category in the source context.
- `64` target `$\mathbf{P}\mathbf{V}$.`: Attention output formula PV is source-backed with meaningful formula alternatives.
- `96` target `2x.`: FP8 Tensor Core throughput multiplier is exact source-backed numeric recall.
- `161` target `$\mathbf{O}_i$.`: Rescaled output accumulator symbol is source-backed algorithm notation.
- `167` target `FP16 precision.`: Precision mode for the replacement consumer path is a discrete source-backed detail.
- `183` target `FP8 precision.`: FP8 precision is the source-backed mode tied to layout-conformance challenges.
- `189` target `SMEM.`: SMEM is a discrete GPU-memory location detail with plausible alternatives.
- `198` target `transposing layouts`: LDSM/STSM layout-transposition capability is a source-specific implementation detail.
- `224` target `\textsc{FlashAttention-2} in Triton.`: Specific benchmark baseline variant is source-backed and has plausible implementation distractors.
- `247` target `2.`: Causal masking FLOP divisor is exact numeric recall with plausible alternatives.

## Notes

- Conservative manual restoration pass after the failed parallel model-review run exhausted rate limits.
- Canonical probe files were not modified.
