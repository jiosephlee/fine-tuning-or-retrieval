# Factual MCQA v15 Naturalness Review

This directory contains a stratified agentic review of whether the v15 MCQA
answer choices read naturally with the contextualized question stems.

The review sampled 10 rows per document across 36 documents, for 360 reviewed
rows total. Agents returned only problematic rows; rows not returned by agents
are treated as accepted.

## Outputs

- `review_input_sample.csv`: sampled rows sent for review.
- `review_input_arxiv.csv`, `review_input_legal.csv`,
  `review_input_medical.csv`: domain-specific review inputs.
- `agent_issue_rows_all.csv`: raw issue rows returned by reviewers.
- `agent_decisions.csv`: complete sampled decision table, including implicit
  accepts.
- `accepted_rows.csv`: sampled rows accepted as natural.
- `fixed_rows.csv`: proposed repairs for unnatural but repairable rows.
- `rejected_rows.csv`: rows reviewers judged not cleanly repairable from the
  available options/context.
- `summary.csv`: counts by domain group and document.

## Decision Counts

| Domain | Accepted | Fix | Reject | Reviewed |
| --- | ---: | ---: | ---: | ---: |
| arxiv | 95 | 21 | 4 | 120 |
| legal | 102 | 18 | 0 | 120 |
| medical | 86 | 32 | 2 | 120 |
| total | 283 | 71 | 6 | 360 |

Overall, 77 of 360 sampled rows were flagged as unnatural or misaligned
(`21.4%`). The most common problems were answer-choice type mismatches,
answer-type mismatches between the stem and correct option, truncated stems,
wording/grammar issues, and formatting artifacts such as leading `Q:` prefixes
or unresolved LaTeX/citation references.

For fixes, `fixed_target` records the answer text that fits the repaired
question. `target_changed_for_fix` is true when the natural answer wording
differs from the original cloze target. This occurred in 20 proposed fixes.

## Validation

The combiner validates each fix for:

- five non-empty options;
- a valid correct label;
- exactly one option matching `fixed_target` after whitespace/case
  normalization;
- no `fixed_target` leakage in the repaired stem;
- generated `fixed_formatted_question` and `fixed_formatted_question_5shot`.

The current decision table has zero validation errors.

No `probes_v15_mcqa.csv` dataset files are modified by this review. The CSVs in
this directory are audit artifacts and proposed fixes for downstream application.
