# MCQA Restoration Review - arxiv/xLSTM v14

## Counts

- Rejected rows reviewed: 56
- Restore: 4
- Keep rejected: 52
- Needs source check: 0

## Restore Indices

18, 66, 76, 150

## Restore Issue Types

- filter_overstrict: 4
- formula_or_symbol_memorization: 3
- numeric_memorization: 1

## Recommended Restores

- `18` target `$\psi$.`: Cell-state squashing function symbol is source-backed notation with plausible alternatives.
- `66` target `\!\exp \left( \log \left( f_t \right) + m_{t-1} - m_t \right).`: Stabilized forget-gate expression is source-backed formula recall.
- `76` target `$t + \tau$.`: Later retrieval time expression is source-backed notation recall.
- `150` target `one.`: Number of sLSTM blocks in xLSTM[7:1] is exact architecture recall.

## Notes

- Conservative manual restoration pass after the failed parallel model-review run exhausted rate limits.
- Canonical probe files were not modified.
