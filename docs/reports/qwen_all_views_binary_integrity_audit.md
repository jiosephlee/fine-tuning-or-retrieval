# Qwen all-views binary integrity audit

Date: 2026-07-12

## Scope and rubric

This audit covers the current 648 Qwen explanation views: six model sizes × three domains × twelve subjects × three view types. Accuracy and factual correctness were not assessed.

A view receives `PASS` only when it is complete and sensible throughout. `FAIL` covers truncated or incomplete output, repetition loops, gibberish or fused/random text, multilingual drift, unrelated cascades, and exposed prompt/reasoning/drafting artifacts. Stylistic weakness alone does not fail, and there is no `SUSPECT` category.

## Results by model size

| Qwen model | PASS | FAIL | Total | Pass rate |
|---|---:|---:|---:|---:|
| 4B | 85 | 23 | 108 | 78.7% |
| 9B | 82 | 26 | 108 | 75.9% |
| 27B | 87 | 21 | 108 | 80.6% |
| 35B-A3B-FP8 | 71 | 37 | 108 | 65.7% |
| 122B-A10B-FP8 | 60 | 48 | 108 | 55.6% |
| 397B-A17B-FP8 | 57 | 51 | 108 | 52.8% |
| **All models** | **442** | **206** | **648** | **68.2%** |

## Results by model and view type

| Model | Blogs | Q&A | Textbooks |
|---|---:|---:|---:|
| 4B | 26/36 | 36/36 | 23/36 |
| 9B | 25/36 | 33/36 | 24/36 |
| 27B | 27/36 | 36/36 | 24/36 |
| 35B-A3B-FP8 | 22/36 | 36/36 | 13/36 |
| 122B-A10B-FP8 | 14/36 | 36/36 | 10/36 |
| 397B-A17B-FP8 | 10/36 | 34/36 | 13/36 |

The detailed TSV contains exactly one binary decision and supporting evidence for each of the 648 current filesystem paths. Inventory reconciliation found no missing, extra, or duplicate rows.
