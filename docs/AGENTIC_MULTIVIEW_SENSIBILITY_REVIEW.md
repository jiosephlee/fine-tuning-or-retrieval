# Codex Agentic Review of Multiview Generations

This is a Codex-native review process for environments with Codex subagent
credits but no API credits. A parent Codex session interactively assigns work
to subagents; no script calls an LLM API.

The only question is whether each generated view is **complete and sensible**.
Do not assess factual accuracy, source grounding, depth, style, or explanatory
quality.

## Review unit and coverage

The decision unit is one `domain / model slug / item / view`:

| View | Assembled file | Outline | Generated children |
|---|---|---|---|
| Blog | `blogs.txt` | `blog_outline.json` | `blogs/blog_*.txt` |
| Q&A | `stackexchange.txt` | `stack_exchange_outline.json` | `stackexchange/stack_*.txt` |
| Textbook | `textbook.txt` | `textbook_outline.json` | `textbooks/chapter_*.txt` |

Granular children contain the generated prose, while the assembled file
usually duplicates it. To minimize tokens without sacrificing coverage:

1. A subagent reads the outline and **every granular child in full**.
2. `utils.multiview_recovery.validate_view` verifies that outline, children,
   and assembled view agree and that every child occurs in the assembled file.
3. The subagent reads the assembled file only if validation fails or granular
   files are absent.

Thus every generated prose unit is seen by a subagent exactly once under normal
conditions. Heads, tails, anomaly matches, or samples do not count as a full
read. If tool output truncates, use smaller successive chunks until the entire
file has been exposed.

## Binary rubric

Every view receives exactly `PASS` or `FAIL`; there is no `SUSPECT` category.

### PASS

Pass only if every child:

- is intelligible and locally coherent;
- is complete and ends cleanly;
- is free of degenerative loops and meaning-impairing repetition;
- has no gibberish, word salad, fused/random text, sustained multilingual
  drift, or unrelated cascades;
- has no exposed prompt, reasoning, drafting, Harmony, reserved-token, or
  control-token leakage; and
- has formatting intact enough to understand it.

The child count must also agree with the outline.

Awkward, verbose, shallow, poorly styled, or factually wrong prose can pass if
it is otherwise sensible and complete.

### FAIL

One bad or missing child fails the whole view. Use one or more labels:

```text
incomplete_or_truncated
degenerative_loop
gibberish_or_word_salad
fused_or_random_text
multilingual_drift
unrelated_cascade
prompt_or_reasoning_leakage
reserved_or_control_token_leakage
meaning_impairing_format_corruption
missing_or_structurally_incomplete_child
```

Do not fail solely for inaccuracy, weak source support, poor pedagogy,
verbosity, stylistic dislike, or ordinary grammar mistakes.

## Parent-agent procedure

### 1. Freeze the inventory

Record the exact model slugs, domains, items, and views. Build one authoritative
row per assembled view and record the expected total before review begins.

```bash
find data/{arxiv,medical,legal}/explanations/<MODEL_SLUG> \
  -mindepth 2 -maxdepth 2 -type f \
  \( -name blogs.txt -o -name stackexchange.txt -o -name textbook.txt \) \
  | sort
```

### 2. Run structural validation

Run `validate_view` for every view and preserve its counts, reasons, and hashes.
This establishes representation completeness but does not replace semantic
review.

### 3. Assign the first pass to the cheapest subagents

Use the **cheapest available Codex subagent at low reasoning effort** for the
initial pass. This task is binary integrity screening and normally does not
benefit from an expensive reasoning model.

Group 12-24 related views from the same model/domain per assignment, using
smaller batches for long textbooks. With four collaboration slots, keep at
most three reviewers active beside the parent. Reuse reviewers for later
batches when possible so the rubric is transmitted only once.

Each path goes to exactly one first-pass reviewer. Record the child thread's
actual model and reasoning effort; do not assume the requested model was used.

### 4. Use this reviewer prompt

```text
Review the assigned views using
docs/AGENTIC_MULTIVIEW_SENSIBILITY_REVIEW.md.

Scope: <EXACT VIEW PATHS>
Output: <LEDGER PATH>

For every view:
1. Read the outline and every granular child in full. Do not rely on samples,
   heads, tails, or anomaly searches.
2. Check the structural validation result. If it does not prove assembled-child
   containment, read the entire assembled file in non-truncating chunks.
3. Judge only completeness and sensibility. Ignore accuracy, grounding,
   quality, and style.
4. Return exactly PASS or FAIL. One failed child fails the view.
5. List every file read, set coverage=full, and give concise, precisely located
   evidence for failures.

Do not edit corpus files or omit an assigned path.
```

### 5. Reconcile coverage

Before accepting results, verify:

- exactly one first-pass row per inventory path;
- no missing, extra, or duplicate paths;
- `coverage=full` for every row;
- every expected child is enumerated in `files_read`;
- outline and child counts agree;
- every verdict is binary; and
- every failure has a label and precise evidence.

Return incomplete work to the reviewer. An absent decision is never a pass.

### 6. Adjudicate failures only

Send first-pass failures to a different subagent for blind review, without the
first verdict or rationale. The cheapest subagent is still appropriate unless
it repeatedly fails to process the text. Accept agreements; the parent
adjudicates disagreements by reading the cited child under the same rubric.

Routine passes do not need a second semantic read once full coverage is proven.

### 7. Publish artifacts

Save a row-level TSV or JSONL ledger, a Markdown summary, the frozen inventory
or its hash, validator output, rubric version, and actual reviewer thread/model
metadata. The row-level ledger is authoritative.

## Required ledger fields

```text
domain
model_slug
item
view
assembled_path
outline_path
granular_count_expected
granular_count_read
files_read
coverage
representation_checked
verdict
failure_types
evidence
failure_file
failure_location
reviewer_thread
reviewer_model
reviewer_reasoning_effort
reviewed_at
```

`files_read` must enumerate every child. `coverage` must be `full`.
`representation_checked` records validator-backed containment or full-assembled
fallback. Evidence should be one or two sentences with only the minimum quote
needed to locate a failure.

## Token-saving rules

The irreducible cost is reading every generated token once. Reduce overhead by:

- reading granular prose once rather than duplicate assembled prose;
- using the cheapest low-reasoning subagents for the first pass;
- batching related views and reusing reviewer threads;
- linking this rubric instead of repeatedly pasting it;
- writing concise structured ledgers;
- independently rereading failures, not routine passes;
- omitting source documents because factual comparison is out of scope; and
- using deterministic checks to guide review, never to skip full reading.

Never save tokens by sampling within a generated child.

## Empty responses and long files

If a cheap reviewer returns an empty or incomplete ledger:

1. Leave affected views unjudged.
2. Retry with fewer views, down to one view at a time.
3. Require child-by-child reading and incremental ledger writes.
4. Escalate to a stronger subagent only after smaller batching fails.
5. Record the failed attempt; do not replace it with heuristics or a presumed
   pass.

## Completion gate

The audit is complete only when inventory and ledger counts match; all paths
are unique; every view has a binary verdict and full-read evidence; every child
is listed as read; representations are reconciled; failures were independently
reviewed; disagreements were adjudicated; and summary counts exactly match the
ledger.

The final report must state:

> Accuracy, factual correctness, source grounding, and explanatory quality
> were not assessed. Every generated prose unit was read in full by at least
> one Codex subagent.

## Historical reference

This formalizes the binary process used for
`docs/reports/qwen_all_views_binary_integrity_audit.md` and its row-level TSV.
Earlier ternary reports are useful history but are not the canonical contract.
