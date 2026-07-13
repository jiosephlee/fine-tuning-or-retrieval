# Qwen failed-view regeneration: round-one semantic audit

Date: 2026-07-12

## Scope

Round one targeted 57 medical and legal views previously rated `FAIL`. Fifty-two changed outputs passed the generation pipeline's structural checks and received content-level review. Five views were excluded because generation failed validation or an old manifest incorrectly skipped them.

The 52 candidates were reviewed for repetition, gibberish or multilingual drift, fused-token and unrelated-topic cascades, prompt/reasoning/drafting leakage, extreme peer-relative length, and truncated or incomplete endings. Reviewers compared each output with its pre-regeneration backup and same-case Qwen peers. The per-file evidence is in `qwen_regeneration_round1_semantic_audit.tsv`.

## Results

| Model | PASS | SUSPECT | FAIL | Reviewed |
|---|---:|---:|---:|---:|
| 122B | 3 | 0 | 20 | 23 |
| 35B | 3 | 0 | 14 | 17 |
| 397B | 1 | 3 | 8 | 12 |
| **Total** | **7** | **3** | **42** | **52** |

Only seven replacements are clean enough for unconditional acceptance. Three 397B outputs are coherent and complete but have localized late-stage prose/formatting degeneration. The other 42 reviewed outputs contain decisive corruption, including runaway lists, fused/random text, repetition, multilingual drift, exposed self-correction, unrelated domain drift, or truncation.

## Unresolved generation views

Five additional targets did not enter semantic acceptance:

- 122B legal `United_States_v_Constantinescu/textbook.txt`: invalid outline JSON; original retained.
- 122B legal `United_States_v_Jaison_Coleman/textbook.txt`: regenerated with a repetition loop.
- 35B medical `Multiphasic_anaphylaxis_in_the_emergency_and_intensive_care/blogs.txt`: invalid outline/count mismatch.
- 122B medical `TAVinTAVinTAV_after_treated_endocarditis_procedural_strategy/blogs.txt`: skipped due to a stale validated manifest.
- 35B medical `Pancreatopleural_fistula_in_childhood/blogs.txt`: skipped due to a stale validated manifest.

Under the agreed PASS-only acceptance rule, 50 views require retry or restoration: the 42 reviewed failures, three suspects, and five unresolved generation views. No further GPU jobs were submitted as part of this audit.
