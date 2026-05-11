# MCQA Restoration Review - arxiv/FeatLLM v14

## Counts

- Rejected rows reviewed: 71
- Restore: 3
- Keep rejected: 68
- Needs source check: 0

## Restore Indices

31, 145, 166

## Restore Issue Types

- filter_overstrict: 3
- numeric_memorization: 1
- technical_term_memorization: 1
- formula_or_symbol_memorization: 1

## Recommended Restores

- `31` target `at least one`: Minimum LLM-inference count per sample is an exact source-backed computational detail.
- `145` target `AND or OR.`: Logical operators named in the rule-combination setup are discrete and source-backed.
- `166` target `$\mathbf{z}_k^i \in \{0, 1\}^R$.`: Generated binary feature notation is source-backed and can be contrasted with nearby vector forms.

## Notes

- Conservative manual restoration pass after the failed parallel model-review run exhausted rate limits.
- Canonical probe files were not modified.
