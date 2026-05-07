# Fact Probe v12/v13 Historical Provenance

This document records how the current checked-in `probes_v12.csv` and
`probes_v13.csv` files were created in this repository. It is intentionally a
historical description of the path we actually took, including the post-v13
over-drop and the later correction.

For the cleaner desired ad hoc process going forward, use the rubrics:

```text
docs/probes/rubrics/v12_RUBRICS.md
docs/probes/rubrics/v13_RUBRICS.md
```

Those rubric files are the reproducibility target for future Codex/subagent
review. This workflow file explains provenance for the current dataset.

## End State

The current final v13 retained set has:

```text
arxiv:   3772
legal:   1027
medical: 1620
total:   6419
```

The final v13 set has zero remaining missing source targets under the allowed
normalizations:

- ignore leading/trailing punctuation around the target;
- ignore casing when casing is needed to keep the cloze grammatical;
- for formula targets only, ignore whitespace to locate the corresponding source
  substring.

Target punctuation/casing was treated as a validation relaxation, not something
that must be edited to mirror the source exactly.

## Stage 1: API-Based Probe Generation

The first stage was the script/API pipeline. Codex review was not used for
generation.

For arxiv papers, the current generation script is:

```text
scripts/data-preparation/probes/pipeline_fact_probe_v1.py
```

It reads cleaned source documents from:

```text
data/arxiv/cleaned/<doc>.tex
```

Typical invocation:

```bash
cd scripts/data-preparation/probes
python pipeline_fact_probe_v1.py --filter <doc-name-substring>
```

The generation objective was:

- extract complete source sentences;
- identify knowledge-bearing sentences;
- generate atomic questions from those sentences;
- require answer spans to be copied from the source sentence;
- contextualize questions with source framing;
- convert each question/answer into a self-contained declarative cloze;
- split each cloze into `probe` and final-word `target`;
- run automated QC filters for leakage, malformed English, target placement,
  tokenizer consistency, and target verbatim-ness;
- attempt verbatim recovery when an otherwise good target drifted from the
  source sentence.

Generated fact probe CSVs have columns such as:

```text
section
subsection
section_text
subsection_text
raw_knowledge_statement
target
fact
contextualized_question
valid_fact
probe
```

For the first v12 pass, the input was usually:

```text
probes/<domain>/<doc>/facts/probes_v10_5.csv
probes/<domain>/<doc>/facts/probes_v10_5_readable.txt
```

Legal and medical probes followed the same conceptual generation contract, with
opinion/case-report framing instead of arxiv paper framing.

## Stage 2: Historical v12 Review

The v12 pass was a repair-first Codex review of generated rows. Its purpose was
to check whether the API-based generation pipeline produced usable cloze probes.

Historically, v12 focused on:

- making clozes grammatical;
- making probes self-contained;
- fixing bad English;
- removing ordinary answer leakage from the probe body;
- ensuring `fact = probe + target` and the target appears at the answer
  position;
- dropping only rows that could not be repaired into usable cloze probes.

v12 was not intended to be the systematic source-attribution, exact-title,
source-support, or target-in-source pass. Those became v13 concerns.

The current desired v12 reproduction rubric is:

```text
docs/probes/rubrics/v12_RUBRICS.md
```

## Stage 3: Historical v12 to v13 Transition

`probes_v13.csv` was produced from `probes_v12.csv`; the generation pipeline was
not rerun.

The v13 pass tightened checks that had been missed or under-specified in v12:

- source framing, especially arxiv probes using `According to the paper ...` or
  `In the paper ...`;
- malformed titles, wrong titles, or misleading source framing;
- unsupported or overbroad claims;
- title-induced answer leakage;
- bad cloze wording that survived v12;
- meaningless targets, especially mostly-notation or incidental formula
  fragments;
- later, exact target occurrence in the source document.

The main v13 review produced 6,483 rows across 36 documents before the later
post-v13 sanity and target-source passes.

The current desired v13 reproduction rubric is:

```text
docs/probes/rubrics/v13_RUBRICS.md
```

That rubric reflects the corrected criteria. It is cleaner than the path we
actually took historically.

## Stage 4: Post-v13 Sanity Pass and Over-Drop

After the main v13 review, we ran a post-v13 sanity pass over the 6,483 retained
rows. This pass was meant to cheaply catch residual high-signal issues after
v13, but it initially applied the drop criteria too broadly.

The over-drop happened because the pass treated some review/repair categories as
drop categories. In particular:

- duplicate or subsumed knowledge was treated too harshly, even though duplicate
  knowledge is allowed when the source document itself repeats it;
- ordinary leakage was treated too much like a hard drop, even when it could be
  repaired;
- bad title/source framing was treated too much like a hard drop, even when the
  intended source was clear and repairable;
- some long, formula, short, or notation-heavy targets were treated as drops
  before a closer repairability review.

The post-v13 sanity audit produced 729 dropped/flagged rows:

```text
reports/probes_v13_post_sanity_undrop_report.csv
```

After reviewing that over-drop, we corrected the hard criteria:

- `duplicate_or_subsumed` is not a drop criterion;
- ordinary `leakage` is a review/repair issue unless the answer is unavoidably
  leaked by required source framing;
- bad or malformed source framing is a review/repair issue when the intended
  document is clear;
- title leakage is the hard case when required source title/framing itself
  contains the target or a near-verbatim answer;
- long targets and formula targets are not drops by themselves;
- mostly-notation or incidental formula targets are drops only when they are not
  meaningful answer spans.

With the corrected criteria, 673 of the 729 rows were restored for review/repair
or kept because they did not meet a hard drop criterion. The remaining 56 stayed
dropped under hard criteria.

The post-v13 repair/review reports are:

```text
reports/probes_v13_followup_repair_review.csv
reports/probes_v13_followup_targeted_repairs.csv
reports/probes_v13_followup_final_review.csv
```

The targeted follow-up repairs included six rows:

- two citation-markup target repairs;
- one stray-parenthesis target repair;
- one abbreviation-only target repair;
- one question-shaped cloze repair;
- one unsupported leading-cause framing repair.

## Stage 5: Final Exact-Source Target Pass

After the corrected post-v13 review, we ran a separate exact-source target pass.
This pass was not the same as the first v13 review/drop pass. Its hard constraint
was:

```text
Every retained target must exist in the full source document after allowed
normalizations.
```

The full source locations checked were:

```text
data/arxiv/cleaned/<doc>.tex
data/arxiv/semicleaned_v3/<doc>.tex
data/arxiv/raw/<doc>.tex
data/legal/cleaned/<doc>.txt
data/legal/raw/<doc>.txt
data/medical/cleaned/<doc>.txt
data/medical/raw/<doc>.txt
```

The target-source pass initially made 1,688 punctuation/casing target-only
repairs, but those were reverted because punctuation and casing should be
validation relaxations, not automatic target edits. We should not capitalize or
punctuate a target only to mirror the source if doing so would make the cloze
ungrammatical.

The pass kept 20 formula/source target repairs where whitespace-insensitive
matching was needed to locate the source expression.

The pass then identified 132 rows whose targets still could not be matched to
source under the allowed normalizations. A closer repair review restored 124 of
those rows because a conservative replacement target occurred in the source and
preserved the intended fact.

The remaining 8 rows stayed dropped because their possible source matches were
malformed, unsupported by the original probe, or required non-mechanical
rewriting.

Exact-source reports:

```text
reports/probes_v13_target_not_in_source.csv
reports/probes_v13_target_source_repairability.csv
reports/probes_v13_exact_source_target_repairs.csv
reports/probes_v13_reverted_punctuation_casing_target_repairs.csv
reports/probes_v13_target_not_in_source_remaining.csv
reports/probes_v13_target_remaining_repairability.csv
reports/probes_v13_final_exact_source_span_repairs.csv
reports/probes_v13_final_exact_source_span_drops.csv
reports/probes_v13_repaired_final_drop_restores.csv
reports/probes_v13_unrepaired_final_drops_remaining.csv
reports/probes_v13_normalized_target_not_in_source_remaining.csv
```

## Historical Output Sidecars

The current v12/v13 files use these sidecars:

```text
probes/<domain>/<doc>/facts/probes_v<version>.csv
probes/<domain>/<doc>/facts/probes_v<version>_readable.txt
probes/<domain>/<doc>/facts/probes_v<version>_removed_indices.txt
probes/<domain>/<doc>/facts/probes_v<version>_removed_text_audit.txt
probes/<domain>/<doc>/facts/probes_v<version>_fix_log.txt
```

Important historical convention: row numbers in audits are original 1-based data
row numbers from the relevant input file for that pass. Some post-v13 row numbers
refer to pre-sanity v13 rows, not v12 input rows.

## Current Reproduction Guidance

To reproduce the desired process rather than the historically messy sequence:

1. Run the API-based generation pipeline to create generated probe CSVs.
2. Apply `docs/probes/rubrics/v12_RUBRICS.md` to produce v12.
3. Apply `docs/probes/rubrics/v13_RUBRICS.md` to v12 in two passes:
   first review/drop/repair, including review of citation/source markup inside
   targets, then the separate target-source pass.
4. Require zero remaining retained targets missing from source under the allowed
   normalizations.

Do not replay the historical over-drop as a required step. It is documented here
only because it explains how the current checked-in v13 files reached their final
state.
