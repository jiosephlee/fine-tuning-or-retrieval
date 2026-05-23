# Factual MCQA v15 Naturalness Review: Remaining Rows

This directory contains the agentic naturalness review for the 4,186 factual
v15 MCQA rows that were not included in the initial 360-row stratified sample.

Rows omitted from `agent_issue_rows_*.csv` were treated as accepted. Problematic
rows were marked as either `fix` with proposed repaired question/options, or
`reject` when the row could not be repaired cleanly from the available context
and options.

## Rest-Pass Counts

| Domain | Accepted | Fix | Reject | Reviewed |
| --- | ---: | ---: | ---: | ---: |
| arxiv | 2,383 | 235 | 25 | 2,643 |
| legal | 488 | 39 | 0 | 527 |
| medical | 830 | 186 | 0 | 1,016 |
| total | 3,701 | 460 | 25 | 4,186 |

The rest pass flagged 485 of 4,186 rows (`11.6%`): 460 proposed fixes and 25
rejects. Of the proposed fixes, 104 change the natural answer wording from the
original cloze target.

## Full-Dataset Counts

Including the earlier 360-row sample review, the full factual v15 MCQA audit now
covers 4,546 rows.

| Domain | Accepted | Fix | Reject | Reviewed |
| --- | ---: | ---: | ---: | ---: |
| arxiv | 2,478 | 256 | 29 | 2,763 |
| legal | 590 | 57 | 0 | 647 |
| medical | 916 | 218 | 2 | 1,136 |
| total | 3,984 | 531 | 31 | 4,546 |

Overall, 562 of 4,546 rows (`12.4%`) were flagged: 531 proposed fixes and 31
rejects. Across the full audit, 124 proposed fixes change the natural answer
wording from the original cloze target.

## Files

- `review_input_rest.csv`: all remaining rows sent for review.
- `review_input_rest_arxiv.csv`, `review_input_rest_legal.csv`,
  `review_input_rest_medical.csv`: domain-specific rest inputs.
- `review_batch_*.csv`: 19 disjoint review batches.
- `batch_manifest.csv`: batch sizes and review-id ranges.
- `agent_issue_rows_*.csv`: raw issue rows returned by agents.
- `agent_decisions.csv`: complete rest-pass decision table.
- `accepted_rows.csv`, `fixed_rows.csv`, `rejected_rows.csv`: rest-pass splits.
- `summary.csv`: rest-pass counts by domain/document.
- `full_agent_decisions_including_sample.csv`: full 4,546-row decision table,
  combining the initial sample review and this rest pass.
- `full_summary_including_sample.csv`: full counts by domain/document.

## Validation

The rest-pass combiner validates each proposed fix for five non-empty options, a
valid correct label, exactly one option matching `fixed_target`, no
`fixed_target` leakage in the repaired stem, and generated 5-shot formatted
question text. The current rest-pass decision table has zero validation errors.

No `probes_v15_mcqa.csv` dataset files are modified by this review. These files
are audit artifacts and proposed repairs for downstream application.
