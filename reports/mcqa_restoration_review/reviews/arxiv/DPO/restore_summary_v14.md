# MCQA Restoration Review - arxiv/DPO v14

## Counts

- Rejected rows reviewed: 49
- Restore: 6
- Keep rejected: 43
- Needs source check: 0

## Restore Indices

96, 117, 185, 305, 311, 312

## Restore Issue Types

- filter_overstrict: 6
- formula_or_symbol_memorization: 4
- numeric_memorization: 2
- technical_term_memorization: 1

## Recommended Restores

- `96` target `$0$.`: Reward normalization value is exact source-backed numeric recall.
- `117` target `$\pi_r$.`: Optimal-policy notation is source-specific and can be contrasted with nearby policy symbols.
- `185` target `$\mathcal{D}$.`: Preference dataset notation is a discrete source-backed symbol with plausible notation distractors.
- `305` target `a common misconception believed by 50\% of people`: The 50 percent misconception example is a specific memorization claim, not a generic completion.
- `311` target `$Z(\cdot)$.`: Unknown partition-function notation is source-backed and has meaningful symbolic alternatives.
- `312` target `$\pi_\text{ref}(y \mid x)$`: Reference-model notation in the theorem is source-backed and can support nearby policy distractors.

## Notes

- Conservative manual restoration pass after the failed parallel model-review run exhausted rate limits.
- Canonical probe files were not modified.
