# Broader Reviewed Candidate Manual Agentic Review

Reviewed 19 `probes_v12_reviewed_broader_candidates_mcqa.csv` rows that had been rejected by the original v12 reviewed MCQA prefilter.

- Keep: 9
- Drop: 10
- Current `probes_v13_mcqa.csv` excludes all broader candidates; this review is a separate artifact.

| decision | issue_type | count |
|---|---|---:|
| drop | near_binary_stem | 1 |
| drop | stem_leakage | 2 |
| drop | unrepairable_leakage | 5 |
| drop | unrepairable_weak_options | 1 |
| drop | unsupported_by_source|unrepairable_ambiguity | 1 |
| keep | none | 9 |
