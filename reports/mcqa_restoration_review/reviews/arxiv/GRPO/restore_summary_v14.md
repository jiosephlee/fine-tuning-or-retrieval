# MCQA Restoration Review - arxiv/GRPO v14

## Counts

- Rejected rows reviewed: 60
- Restore: 5
- Keep rejected: 55
- Needs source check: 0

## Restore Indices

182, 225, 234, 339, 393

## Restore Issue Types

- filter_overstrict: 5
- formula_or_symbol_memorization: 3
- numeric_memorization: 2
- technical_term_memorization: 1

## Recommended Restores

- `182` target `Chinese K-12 mathematical problems.`: Chinese K-12 problem type is a source-specific dataset-content detail with plausible distractors.
- `225` target `$\mathcal{J}_{GRPO}(\theta)$`: GRPO objective notation is source-backed formula recall with nearby objective-symbol distractors.
- `234` target `$G$ rewards.`: Number of rewards in the sampled group is a discrete source-backed symbolic quantity.
- `339` target `three.`: Number of unified-paradigm components is exact source-backed recall.
- `393` target `$\mathcal{D}$`: Data-source symbol in the unified-gradient equation is source-backed notation recall.

## Notes

- Conservative manual restoration pass after the failed parallel model-review run exhausted rate limits.
- Canonical probe files were not modified.
