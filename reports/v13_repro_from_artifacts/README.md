# v13 Reproduction From Recorded Artifacts

This directory contains a non-destructive replay of the v12 -> v13 -> v13-post
pipeline using the recorded sidecars and reports. It does not overwrite the
current `probes/*/*/facts/probes_v13.csv` files.

## Method

For each document, the replay:

1. starts from `probes_v12.csv`;
2. applies initial v13 repairs parsed from `probes_v13_fix_log.txt`;
3. applies initial v13 removals from `probes_v13_removed_indices.txt`;
4. applies the recorded hard post-v13 removals;
5. applies targeted follow-up repairs from `reports/probes_v13_followup_targeted_repairs.csv`;
6. applies exact-source repairs from `reports/probes_v13_final_exact_source_span_repairs.csv`;
7. applies repaired final-drop restores from `reports/probes_v13_repaired_final_drop_restores.csv`;
8. drops the remaining unrepaired exact-source failures from `reports/probes_v13_unrepaired_final_drops_remaining.csv`.

This is an artifact replay, not a fresh stochastic Codex/LLM review. A true fresh
review could make different judgment calls unless those decisions are fully
scripted.

## Results

The replay produced the same final row count as the current v13 set:

```text
replay rows:  6419
current rows: 6419
```

Target-source validation also matches the current final constraint:

```text
missing replay targets under allowed normalizations: 0
```

Raw field comparison found:

```text
field diffs:         2279
leading/trailing ws: 2078
title-markdown/ws:     42
content/replay gaps:  159
```

Most raw diffs are formatting artifacts, especially trailing newlines or target
leading spaces. The remaining content/replay gaps are primarily repairs that were
present in the final current CSV but not fully reconstructable from machine-
readable reports, such as duplicated title cleanup, colon-before-target cleanup,
and some `contextualized_question` answer-span updates.

## Files

```text
probes_v13_repro_summary.csv
probes_v13_repro_diff.csv
probes_v13_repro_substantive_diff.csv
probes_v13_repro_content_diff.csv
probes_v13_repro_target_missing.csv
<domain>/<doc>/facts/probes_v13_repro.csv
<domain>/<doc>/facts/probes_v13_repro_readable.txt
```
