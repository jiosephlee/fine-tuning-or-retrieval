# MCQA Filter Codex Review Rubric

This rubric describes a manual Codex/subagent review pass for generated MCQA
probe files. It is intended to review rows such as `probes_v14_mcqa.csv` after
the automated MCQA suitability, distractor-generation, and verification steps.

## Task Background

These MCQA probes test whether an LLM remembers knowledge from source documents
after training on them. The primary target is memorization of source-backed
facts, but useful memorization probes still require enough understanding to
choose the correct answer against strong distractors.

The visible question alone must provide enough context for a model that
remembers the source document to answer correctly. The model should not need to
see the original source excerpt at evaluation time, but the stem may include
document title, section-local context, named entities, experimental setup,
method details, or other source-local framing needed to identify the fact being
tested.

The correct answer should be the provided `target`. It may be adjusted slightly
when needed to fit the natural wording of the question or to follow the pattern
of the answer choices, but the provided question must still be sufficient and
the adjusted answer must preserve the same source-backed fact.

The goal of this review is to produce a high-quality MCQA set for memorization
evaluation. The review should reject or repair questions that are leaked,
ambiguous, malformed, unsupported, under-contextualized, or too weak to function
as useful multiple-choice questions.

## Output Scope

Write review artifacts under:

```text
reports/mcqa_filter_review/
```

Do not modify canonical probe files during review:

```text
probes/<domain>/<doc>/facts/probes_<version>.csv
probes/<domain>/<doc>/facts/probes_<version>_mcqa.csv
probes/<domain>/<doc>/facts/probes_<version>_*.txt
```

After review, a separate integration step may copy reviewed outputs back into
canonical probe paths if desired.

## Required Per-Document Outputs

For document `<domain>/<doc>` and MCQA version `<version>`, write:

```text
reports/mcqa_filter_review/reviews/<domain>/<doc>/decisions_<version>.csv
reports/mcqa_filter_review/outputs/<domain>/<doc>/facts/probes_<version>_mcqa_reviewed.csv
reports/mcqa_filter_review/outputs/<domain>/<doc>/facts/probes_<version>_mcqa_reviewed_readable.txt
reports/mcqa_filter_review/outputs/<domain>/<doc>/facts/probes_<version>_mcqa_removed_indices.txt
```

`decisions_<version>.csv` columns:

```text
row_1based,decision,issue_type,reason,old_formatted_question,new_formatted_question,old_correct_label,new_correct_label,old_options,new_options,old_target,new_target
```

Use `decision` values:

```text
keep
repair
drop
```

Preserve original 1-based data-row numbers from the MCQA CSV. `row_1based=1`
means the first data row after the CSV header.

## Review Inputs

Primary MCQA input:

```text
probes/<domain>/<doc>/facts/probes_<version>_mcqa.csv
```

Useful companion inputs:

```text
probes/<domain>/<doc>/facts/probes_<version>.csv
probes/<domain>/<doc>/facts/probes_v13.csv
probes/<domain>/<doc>/facts/probes_v13_readable.txt
```

Use row-local fields from the MCQA CSV:

```text
probe
target
correct_label
formatted_question
option_a
option_b
option_c
option_d
option_e
distractors
fact
raw_knowledge_statement
section
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

## Visibility Rule

Judge leakage and answerability from what the evaluated model sees:

```text
formatted_question
```

The model sees the stem and answer options. It does not see `fact`,
`raw_knowledge_statement`, `section_text`, `subsection_text`, or the source
document.

Do not reject merely because the target appears verbatim in the source fact.
These are memorization probes; target overlap with source text is expected.

The visible question should be sufficient for a model that remembers the source
document. If the question lacks enough context to identify the intended fact,
repair it by adding minimal source-local context when possible; otherwise drop
it as `not_standalone`.

## Keep Criteria

Keep an MCQA row when all are true:

- the stem is grammatical and source-framed;
- the visible stem does not give away the correct option;
- exactly one option is correct under the source fact;
- `correct_label` points to the option matching `target`;
- all options are nonempty and distinct;
- distractors are plausible enough to be meaningful alternatives;
- distractors are not accidentally correct, partially correct in context, or
  broader/narrower variants of the correct answer;
- the question is answerable without seeing source context, assuming the model
  has memorized the paper;
- the stem provides enough source-local context to identify the intended
  memorized fact;
- the target is a meaningful answer span for MCQA.

Dates, numbers, datasets, model names, formulae, and short technical answers are
allowed when they make a clean, source-backed, discrete MCQA item.

## REVIEW / Repair Flags

Use these `issue_type` values for `decision=repair`, pipe-separated when needed:

```text
label_mismatch
option_target_mismatch
duplicate_option
blank_option
stem_leakage
near_binary_stem
multiple_correct_options
weak_distractor
implausible_distractor
option_formatting
latex_formatting
poor_wording
source_framing_issue
source_support_repair
```

Definitions:

- `label_mismatch`: `correct_label` does not point to the correct option.
- `option_target_mismatch`: the correct option is source-supported but does not
  match the `target` field, or the `target` field needs a minimal normalization
  to fit the question wording or option pattern while preserving the same fact.
- `duplicate_option`: two or more answer options are identical or near-identical.
- `blank_option`: an option is empty, `nan`, malformed, or not a valid answer.
- `stem_leakage`: the correct answer, a near-verbatim equivalent, or a uniquely
  identifying phrase appears in the visible stem before the options.
- `near_binary_stem`: the stem narrows the answer to an obvious two-way contrast
  despite having five options, such as "from 16 bits to ___" when only one lower
  bit-width is plausible in the option set.
- `multiple_correct_options`: more than one option can reasonably answer the
  stem. Repair only if one option can be replaced cleanly.
- `weak_distractor`: a distractor is too easy, generic, or stylistically unlike
  the correct option, but the row can be strengthened by replacing it.
- `implausible_distractor`: a distractor is unrelated to the paper/domain or
  obviously impossible.
- `option_formatting`: labels, punctuation, line breaks, option ordering, or
  option text formatting are malformed.
- `latex_formatting`: TeX/LaTeX syntax in the stem or options is malformed,
  inconsistent, or unnecessarily escaped.
- `poor_wording`: the stem is awkward, ungrammatical, duplicated, or unclear, but
  the intended MCQA is clear enough to repair.
- `source_framing_issue`: title/source attribution is missing, malformed, or
  inconsistent with the document.
- `source_support_repair`: the row is almost source-supported but needs a
  conservative edit to stem/option wording to match the source fact.

Prefer conservative repairs over drops when the intended MCQA item is clear and
the repair does not invent a new fact.

## DROP Criteria

Use these `issue_type` values for `decision=drop`, pipe-separated when needed:

```text
unrepairable_leakage
unrepairable_ambiguity
no_correct_option
unsupported_by_source
not_standalone
meaningless_target
too_generic_target
unrepairable_weak_options
malformed_unrepairable
```

Definitions:

- `unrepairable_leakage`: the visible stem cannot hide the answer without
  changing the fact being tested.
- `unrepairable_ambiguity`: the stem or option set permits multiple correct
  answers and cannot be repaired conservatively.
- `no_correct_option`: none of the options correctly answers the stem.
- `unsupported_by_source`: the MCQA asserts something not supported by the
  source fact or source document.
- `not_standalone`: the stem depends on unstated local context that cannot be
  added minimally.
- `meaningless_target`: the target is not a meaningful answer span, such as
  incidental markup, a bare citation, or a fragment that cannot stand as an
  option.
- `too_generic_target`: the target is too generic or low-information to support
  a meaningful five-option MCQA, such as `capabilities`, `future work`, or
  `promising results`.
- `unrepairable_weak_options`: the item would require rewriting most or all
  options and the stem to become useful.
- `malformed_unrepairable`: the row is too structurally broken to repair without
  inventing a new MCQA item.

Do not drop solely because the question is memorization-oriented, cloze-like, or
has a short answer. Drop only when the visible MCQA is leaked, ambiguous,
unsupported, malformed, or too weak after considering conservative repair.

## Option Quality Standards

Good options should:

- be mutually exclusive in the source context;
- share comparable specificity and style;
- avoid making the correct answer uniquely longer, more technical, or more
  source-specific unless that is unavoidable;
- avoid obviously unrelated model names, datasets, or methods when closer
  distractors exist;
- avoid answer sets where only two options are plausible and the other three are
  filler;
- preserve units and formatting consistently for numeric/formula questions.

For numeric questions, use plausible nearby or source-relevant alternatives.
For formula questions, use mathematically plausible alternatives that differ in
meaningful ways. For named-entity questions, prefer entities from the same paper
or domain when they are clearly incorrect for the stem.

## Stem Quality Standards

Good stems should:

- be self-contained and source-framed;
- ask for one precise thing;
- not include the correct answer or a near-verbatim paraphrase of it;
- not narrow the answer to an obvious binary contrast;
- not rely on option text to fix a broken sentence;
- not ask for arbitrary list position unless the source fact itself is about
  ordering;
- not over-specify context so heavily that only one option is syntactically or
  semantically possible.

## Repair Rules

A repair may:

- correct `correct_label`;
- normalize `target` punctuation/casing or make a small wording adjustment so
  the correct option fits the stem and the option pattern naturally;
- replace one or more bad distractors while preserving the correct option;
- rewrite the stem to remove leakage or improve standalone answerability;
- add minimal source-local context needed for a model that remembers the source
  document to identify the intended fact;
- fix source framing, grammar, punctuation, labels, or LaTeX formatting;
- adjust option wording so all options have comparable specificity.

A repair may not:

- change to a different source fact;
- introduce information not supported by the source row or source document;
- turn the item into a different question because the original item was weak;
- remove context that is needed to identify the intended memorized fact;
- keep two options that could both reasonably be correct.

When repairing, update all coupled fields consistently:

```text
formatted_question
correct_label
option_a
option_b
option_c
option_d
option_e
distractors
target
```

## Subagent Review Procedure

When using Codex subagents, assign one document per subagent unless a document
has a very large MCQA file. For large files, split by contiguous row ranges and
make the row range explicit.

Each subagent should:

1. Read this rubric.
2. Review only its assigned document or row range.
3. Write the required `decisions_<version>.csv`.
4. Write a reviewed MCQA CSV containing kept and repaired rows only.
5. Write removed indices and a readable reviewed file.
6. Summarize counts: input rows, kept, repaired, dropped, and top issue types.

Subagents should not modify canonical `probes/` files. Integration should happen
only after all review outputs are inspected.

## Recommended Review Checklist

For each row:

1. Parse the visible stem and options from `formatted_question`.
2. Confirm the option named by `correct_label` exactly corresponds to `target`.
3. Confirm the correct option is source-supported by `fact` and, when needed,
   `raw_knowledge_statement`.
4. Check whether any other option is also correct or arguably correct.
5. Check whether the visible stem leaks the target or collapses to a near-binary
   answer.
6. Check whether distractors are plausible, distinct, and comparable in style.
7. Decide `keep`, `repair`, or `drop`.
8. For repairs, update all coupled MCQA fields consistently.
