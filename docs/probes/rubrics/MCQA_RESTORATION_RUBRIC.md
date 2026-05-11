# MCQA Rejected-Probe Restoration Rubric

This rubric describes a manual Codex/subagent review pass for probes rejected by
the automated MCQA suitability filter before distractor generation. It is meant
to decide whether any rejected source-backed factual probes should be restored
and sent through MCQA distractor generation.

## Task Background

These MCQA probes test whether an LLM remembers knowledge from source documents
after training on them. The primary target is memorization of source-backed
facts. A good restored probe should still require enough understanding to choose
the correct answer against strong distractors, but it may be cloze-like, short,
numeric, or source-verbatim when that is the actual fact being tested.

Judge leakage and answerability from the visible probe question only. The
evaluated model will see the question stem and answer options, but it will not
see the source fact, `raw_knowledge_statement`, `section_text`,
`subsection_text`, or source document at evaluation time.

Do not reject merely because the target appears verbatim in the source fact or
source context. These are memorization probes; source overlap is expected.

## Output Scope

Write review artifacts under:

```text
reports/mcqa_restoration_review/
```

Do not modify canonical probe files during restoration review:

```text
probes/<domain>/<doc>/facts/probes_v13.csv
probes/<domain>/<doc>/facts/probes_<version>.csv
probes/<domain>/<doc>/facts/probes_<version>_mcqa.csv
probes/<domain>/<doc>/facts/mcqa_filter_<version>.txt
```

After review, a separate integration step may append restored rows to a working
input file and rerun distractor generation.

## Required Per-Document Outputs

For document `<domain>/<doc>` and output MCQA version `<version>`, write:

```text
reports/mcqa_restoration_review/reviews/<domain>/<doc>/restore_decisions_<version>.csv
reports/mcqa_restoration_review/reviews/<domain>/<doc>/restore_summary_<version>.md
reports/mcqa_restoration_review/reviews/<domain>/<doc>/restore_indices_<version>.txt
```

`restore_decisions_<version>.csv` columns:

```text
v13_row_0based,v13_row_1based,decision,issue_type,filter_reason,review_reason,probe,target,fact,raw_knowledge_statement,section
```

Use `decision` values:

```text
restore
keep_rejected
needs_source_check
```

Preserve original v13 row numbers. `v13_row_0based` should match the probe
number shown in `mcqa_filter_<version>.txt`.

## Review Inputs

Primary rejection log:

```text
probes/<domain>/<doc>/facts/mcqa_filter_<version>.txt
```

Primary source rows:

```text
probes/<domain>/<doc>/facts/probes_v13.csv
```

Useful companion files:

```text
probes/<domain>/<doc>/facts/probes_v13_readable.txt
probes/<domain>/<doc>/facts/probes_<version>.csv
```

Use full source text only when source support is unclear:

```text
data/arxiv/cleaned/<doc>.tex
data/arxiv/semicleaned_v3/<doc>.tex
data/arxiv/raw/<doc>.tex
data/legal/cleaned/<doc>.txt
data/legal/raw/<doc>.txt
data/medical/cleaned/<doc>.txt
data/medical/raw/<doc>.txt
```

## Restore Criteria

Restore a rejected probe when all are true:

- the visible probe does not contain the target or a near-verbatim equivalent;
- the target is source-backed by the row-local fact or source document;
- the target is a discrete answer span that can support five options;
- the question is answerable from the question alone by a model that remembers
  the source document;
- plausible distractors can be formed without changing the fact being tested;
- the automated filter reason is overly broad, especially for memorization.

The following target types are often restorable when source-backed and not
leaked in the visible stem:

- exact dates, years, counts, lengths, percentages, sizes, and other numeric
  values;
- dataset, benchmark, model, method, component, paper, or author names;
- specific technical terms or named mechanisms;
- formulae or symbols when there are meaningful nearby alternatives;
- source-specific claims that are important enough to memorize even if they are
  short.

## Keep-Rejected Criteria

Keep a rejected probe rejected when any apply:

- the target appears in the visible probe;
- the visible probe nearly states the target, making the answer tautological;
- the target is too generic to support a meaningful five-option MCQA;
- the answer space is effectively binary or a simple scalar direction;
- the target is a bare citation, incidental markup, vague adjective, or
  fragment that cannot stand naturally as an option;
- multiple answer options would likely be correct under the source fact;
- the question lacks enough source-local framing and cannot be repaired without
  inventing a new question;
- the row appears unsupported or contradicted by the source row.

## Issue Types

Use these `issue_type` values for `decision=restore`:

```text
numeric_memorization
named_entity_memorization
technical_term_memorization
formula_or_symbol_memorization
filter_overstrict
strong_distractor_space
```

Use these `issue_type` values for `decision=keep_rejected`:

```text
visible_target_leak
tautological_stem
too_generic_target
weak_distractor_space
near_binary_answer_space
meaningless_target
not_standalone
unsupported_or_unclear
```

Use `needs_source_check` when the probe looks restorable but row-local source
fields are insufficient to verify support.

## Review Procedure

For each rejected probe:

1. Read the visible `probe`, `target`, and automated `filter_reason`.
2. Check only the visible probe for target leakage.
3. Check row-local `fact` and `raw_knowledge_statement` for source support.
4. Decide whether the target is a meaningful memorization fact.
5. Decide whether five plausible, mutually exclusive options could be generated.
6. Mark `restore`, `keep_rejected`, or `needs_source_check`.

Prefer restoration only when the rejected item is likely to become a useful MCQA
without substantial rewriting. This pass should recover false negatives, not
weaken the MCQA set by admitting generic or leaked items.
