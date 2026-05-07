# v12 Codex Review Rubric

This rubric describes the repair-first Codex review used to turn generated fact
probe files into `probes_v12.csv`. The v12 pass starts from the generated probe
tables, usually `probes_v10_5.csv`, and focuses on cleaning bad English,
removing answer leakage, making probes self-contained, and dropping only rows
that cannot be repaired confidently. Its purpose is to verify that the
API-based sentence/question/cloze pipeline produced usable cloze probes.

## Review Inputs

For each document, use:

```text
probes/<domain>/<doc>/facts/probes_v10_5.csv
probes/<domain>/<doc>/facts/probes_v10_5_readable.txt
```

Use row-local context to understand what the generated cloze was trying to say:

```text
section_text
subsection_text
raw_knowledge_statement
```

Do not perform the v13-only systematic checks in this pass: source attribution,
title correctness, unsupported-source adjudication, exact source-span target
validation, and target meaningfulness are handled by the v13 rubric.

## Expected Outputs

Write the reviewed files for each document:

```text
probes/<domain>/<doc>/facts/probes_v12.csv
probes/<domain>/<doc>/facts/probes_v12_readable.txt
probes/<domain>/<doc>/facts/probes_v12_removed_indices.txt
probes/<domain>/<doc>/facts/probes_v12_removed_text_audit.txt
```

If any rows are repaired, also write:

```text
probes/<domain>/<doc>/facts/probes_v12_fix_log.txt
```

## Review Priorities

v12 is a repair-first pass. Prefer minimal repairs over drops when the intended
cloze is clear from the generated row and local context.

Flag rows for:

- `leakage`: target or a near-verbatim equivalent appears earlier in the probe
  and can usually be removed by rewriting the cloze.
- `poor_wording`: malformed English, awkward phrasing, duplicated wording, or
  broken grammar.
- `malformed_cloze`: the fact is question-shaped, does not end with the target,
  has a colon/list fragment before the target, or otherwise fails as a cloze.
- `not_self_contained`: the probe depends on unstated context, undefined
  referents, or vague wording that prevents standalone evaluation.

## Repair Criteria

Repair a row when:

- the intended cloze is clear from the generated row and local context;
- the repair is minimal and preserves the generated row's intended meaning;
- the repaired probe is grammatical and self-contained;
- the repaired probe does not leak the target before the answer position;
- the repaired target remains the final words of the fact.

Common v12 repairs:

- rewrite bad English into a clean declarative cloze;
- remove ordinary answer leakage from the probe body;
- add missing local context needed to make the cloze self-contained;
- move the answer span to the end of the fact;
- trim punctuation, citations, or markup from the target when the answer remains
  clear;
- adjust `probe`, `target`, and `fact` together so `fact = probe + target`.

## Drop Criteria

Drop only when repair is not reliable.

Use these drop issue types:

```text
unrepairable_leakage
unrepairable_cloze_form
unrepairable_context
unrepairable_poor_wording
```

Definitions:

- `unrepairable_leakage`: the answer cannot be hidden without destroying the
  intended cloze.
- `unrepairable_cloze_form`: the row cannot be made into a clean cloze whose
  `fact` ends with `target` without inventing a new example.
- `unrepairable_context`: the row is too context-dependent to make standalone
  without guessing missing information.
- `unrepairable_poor_wording`: the row is too malformed, ambiguous, or
  context-dependent to repair confidently.

If a row has suspicious source attribution, title framing, source support, exact source-span status, or target meaningfulness but is otherwise a usable cloze, keep it for v13 review.

Do not drop only because a target is technical, long, formulaic, or duplicated elsewhere. Keep the row if it is grammatical, self-contained, has proper cloze form, and does not leak the answer.

## Logging

Preserve original 1-based row numbers in all logs.

For every repaired row, log:

- original row number;
- issue type;
- original `fact`, `probe`, and `target`;
- repaired `fact`, `probe`, and `target`;
- brief reason for the repair.

For every dropped row, log:

- original row number;
- issue type;
- original `fact`, `probe`, and `target`;
- brief reason the row was not repairable.
