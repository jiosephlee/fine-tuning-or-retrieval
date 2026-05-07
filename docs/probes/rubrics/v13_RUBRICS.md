# v13 Fresh Codex Review Rubric

This is a fresh, non-destructive v12 -> v13 review pass. Do not read or copy
current `probes_v13.csv` decisions. Start from `probes_v12.csv` and source
files.

## Output Scope

Write only under:

```text
reports/v13_fresh_codex/
```

Do not modify:

```text
probes/<domain>/<doc>/facts/probes_v13.csv
probes/<domain>/<doc>/facts/probes_v13_*.txt
```

## Required Per-Document Outputs

For document `<domain>/<doc>`, write:

```text
reports/v13_fresh_codex/reviews/<domain>/<doc>/decisions.csv
reports/v13_fresh_codex/outputs/<domain>/<doc>/facts/probes_v13_fresh_codex.csv
reports/v13_fresh_codex/outputs/<domain>/<doc>/facts/probes_v13_fresh_codex_readable.txt
reports/v13_fresh_codex/outputs/<domain>/<doc>/facts/probes_v13_fresh_codex_removed_indices.txt
```

`decisions.csv` columns:

```text
row_1based,decision,issue_type,reason,old_target,new_target,old_probe,new_probe,old_fact,new_fact
```

Use `decision` values:

```text
keep
repair
drop
```

## Review Inputs

Use:

```text
probes/<domain>/<doc>/facts/probes_v12.csv
```

Use full source text for the second-pass target-source validation:

```text
data/arxiv/cleaned/<doc>.tex
data/arxiv/semicleaned_v3/<doc>.tex
data/arxiv/raw/<doc>.tex
data/legal/cleaned/<doc>.txt
data/legal/raw/<doc>.txt
data/medical/cleaned/<doc>.txt
data/medical/raw/<doc>.txt
```

Use row-local `subsection_text` and `raw_knowledge_statement` to judge source
support and repairability.

## Pass Structure

Run v13 as two distinct passes:

1. First pass: review/drop/repair rows for source framing, leakage, wording,
   standalone answerability, source support, title leakage, and target
   meaningfulness.
2. Second pass: validate that every retained target exists in the source
   document under the allowed normalizations, then repair or drop only under the
   target-source criteria below.

## REVIEW Flags

Use REVIEW flags for rows that should be inspected and repaired if a minimal,
source-supported repair exists. A REVIEW flag is not itself a drop decision.

Use these `issue_type` values for `decision=repair`, pipe-separated when needed:

```text
leakage
source_framing_issue
poor_wording
insufficient_context
citation_markup_in_target
near_drop_review
```

Definitions:

- `leakage`: the target or a near-verbatim equivalent appears earlier in the
  probe body, outside the required source title, and may be removable by
  rewriting the cloze.
- `source_framing_issue`: missing attribution, wrong title, malformed title, or
  misleading paper/opinion/case-report framing when the intended source is clear
  enough to repair.
- `poor_wording`: the cloze is ungrammatical, question-shaped, awkward,
  duplicated, or otherwise poorly phrased, but the intended fact is clear.
- `insufficient_context`: the probe is grammatical but not fully answerable as a
  standalone cloze because it depends on unstated source-local context,
  undefined referents, vague setup, or ambiguous phrases such as "this case,"
  "this approach," "these results," "used here," or "the method" without enough
  identifying context. Repair when the missing context can be added minimally
  from the source-local row context.
- `citation_markup_in_target`: the target contains citation commands, citation
  suffixes, footnote markers, bibliography keys, or source markup, such as
  `~\cite{...}`. This also includes nonsemantic TeX formatting or spacing setup
  inside formula targets, such as `\thickmuskip=2mu` or `\medmuskip=2mu`.
  Repair when the markup can be stripped while leaving a meaningful target. If
  the citation/markup is effectively the target itself, drop as
  `meaningless_target`.
- `near_drop_review`: the row almost meets one of the DROP criteria below, but
  not quite; it requires further review to determine whether a conservative
  repair is possible or whether the row should be dropped.

For REVIEW flags, first attempt a conservative repair. If no reliable repair
exists, convert the row to `decision=drop` but keep the REVIEW issue type in
the decision log so it is clear that the drop happened after attempted review.
Do not use a separate `unrepairable_*` label for failures of repair-analogue
criteria.

### Source Framing

Arxiv probes should begin with source framing such as:

```text
According to the paper "<title>", ...
In the paper "<title>", ...
```

Legal probes should use opinion/case framing, for example:

```text
In the case "<case title>", ...
According to the opinion in "<case title>", ...
```

Medical probes should use case-report framing, for example:

```text
According to the case report "<case report title>", ...
In the case report "<case report title>", ...
```

## DROP Criteria

Use these DROP criteria for rows that can be dropped outright to save review
effort because the flaw is structurally unrepairable. Do not put repair-analogue
failures here; if a row fails repair for `poor_wording`, `insufficient_context`,
`source_framing_issue`, `leakage`, or `citation_markup_in_target`, drop it with
that REVIEW issue type after documenting why no conservative repair works.

Use these `issue_type` values for `decision=drop`, pipe-separated when needed:

```text
title_leakage
unsupported_by_source
meaningless_target
```

Definitions:

- `title_leakage`: the required source title itself gives away the specific
  answer span or a near-verbatim equivalent, so the answer is unavoidably leaked.
  Do not apply this criterion only because the title contains generic component
  words, domain terms, method-family words, or short substrings that also appear
  in the target.
- `unsupported_by_source`: the fact/probe says something not supported by the
  source context, contradicts the source, or would require inventing a new fact
  rather than repairing the existing source-backed cloze.
- `meaningless_target`: target is not a meaningful answer span, including mostly
  notation boilerplate, incidental formula fragments, bare citation markers, or
  answer spans where the citation/source markup is effectively the only specific
  content.

Ordinary answer leakage outside the required title should be flagged as
`leakage` for repair. Wrong or malformed source framing should be repaired as
`source_framing_issue` when the intended document is clear. Duplicate knowledge
is allowed when the source document repeats it.

Do not drop solely because a target is long or is a formula. Keep long/formula
targets when they are meaningful, source-backed, and not leaked.

## Repair Criteria

Repair a row when:

- the intended fact is clear from source context;
- the repair is minimal;
- the repair preserves source scope and does not add unsupported information;
- the repaired probe is self-contained and source-framed;
- the repaired probe supplies enough context to be answerable without reading
  the surrounding source passage;
- the repaired target is a meaningful final answer span.

Common repairs:

- normalize missing/malformed source attribution;
- correct a wrong or malformed title when the intended source document is clear;
- remove duplicated title/source attribution;
- rewrite ordinary answer leakage outside the required title;
- fix question-shaped or colon-before-target cloze wording;
- add minimal source-local context needed to make a grammatical but vague probe
  answerable as a standalone cloze;
- strip citation/source markup from a target when a meaningful source-backed
  answer remains, such as repairing `ChatGPT~\cite{...} and Stable
  Diffusion~\cite{...}` to `ChatGPT and Stable Diffusion`;
- strip nonsemantic TeX spacing or formatting setup from formula targets, such
  as `\thickmuskip=2mu` or `\medmuskip=2mu`, when the mathematical expression
  remains meaningful;

## Second Pass: Target-Source Constraint

After the first pass has produced a retained set, run a separate target-source
validation pass. Every retained target must exist in the full source document
after allowed normalizations.

Use these `issue_type` values in the second pass:

```text
target_source_repair
target_source_drop
```

Second-pass REVIEW/repair criterion:

- `target_source_repair`: the target does not currently match the full source
  under the allowed normalizations, but a conservative source-span repair
  preserves the same fact.

Second-pass DROP criterion:

- `target_source_drop`: the target cannot be matched in the full source under
  the allowed normalizations, and no equivalent source-span repair preserves the
  same fact.

Allowed validation normalizations:

- ignore leading/trailing punctuation around the target;
- ignore casing when needed to keep the cloze grammatical;
- for formula targets only, ignore whitespace to locate the corresponding source
  substring.
- for formula targets only, ignore nonsemantic TeX spacing commands such as
  `\thickmuskip=2mu` and `\medmuskip=2mu` when locating the corresponding
  source expression.

Do not force target punctuation/casing to match the source if doing so makes the
probe ungrammatical. Punctuation and casing are validation relaxations, not
automatic target edits.

If a target is missing but a conservative exact source-span repair preserves the
same fact, repair as `target_source_repair`. If not, drop as
`target_source_drop`.

## Output CSV Rules

Retain the same columns as `probes_v12.csv` and preserve row order among
retained rows. Repaired rows remain in their original position with updated
`fact`, `probe`, and `target`. Dropped rows are omitted from the output CSV.

`probes_v13_fresh_codex_readable.txt` should contain one retained `fact` per
line.
