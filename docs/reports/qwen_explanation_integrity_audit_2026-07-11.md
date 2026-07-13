# Qwen explanation-generation integrity audit

Date: 2026-07-11

## Scope

This audit covers every `blogs.txt`, `stackexchange.txt`, and `textbook.txt` in the six Qwen 3.5 W16 generation trees for each of the arXiv, legal, and medical domains. The authoritative inventory contains 648 files: 216 per domain and 216 per content type, spanning 216 subject/model combinations. No generated corpus file was modified.

Each domain was reviewed independently. Every file was checked for decoding and structural validity, empty or incomplete content, abnormal length and section counts relative to the same subject/type from other Qwen sizes, repeated tokens and passages, gibberish or multilingual drift, prompt/reasoning leakage, and abrupt or incomplete endings. Automated anomalies were directly inspected. Each domain report was then diffed against the filesystem inventory; all three reports contain exactly 216 unique paths with no missing or extra files.

## Results

| Domain | PASS | SUSPECT | FAIL | Total |
|---|---:|---:|---:|---:|
| arXiv | 92 | 21 | 103 | 216 |
| Legal | 164 | 20 | 32 | 216 |
| Medical | 184 | 7 | 25 | 216 |
| **Total** | **440** | **48** | **160** | **648** |

`FAIL` means clear corruption, degeneration, repetition, or incomplete/truncated generation. `SUSPECT` means a credible structural, terminal, quality, or prompt-leakage anomaly without an unambiguous semantic collapse. `PASS` means coherent, nonempty, complete-looking content with no material anomaly found.

## Main findings

- **arXiv:** 23/72 blogs passed, 18 were suspect, and 31 failed. Q&A was much healthier: 69/72 passed and 3 were suspect. All 72 textbooks failed because each contained substantive degeneration somewhere, including irrelevant word lists, repeated tokens, multilingual/gibberish drift, reasoning leakage, or explicit/implicit truncation.
- **Legal:** 164 passed, 20 were suspect, and 32 failed. Failures cluster in the 122B and 35B generations and predominantly consist of runaway lists, repetition, fused gibberish, unrelated-topic cascades, or reasoning/response artifacts. No ordinary mid-sentence truncation was confirmed.
- **Medical:** 40/72 blogs passed, 7 were suspect, and 25 failed. All 72 Q&A files and all 72 textbooks passed. Severe corruption clusters in 122B/35B blogs; the 397B findings are localized fused or fragmented spans. One failed 35B diabetes blog contains the Unicode replacement character U+FFFD.

## Interpretation

The corpus is not uniformly safe to consume as generated. The strongest exclusion rule is to reject every `FAIL`. `SUSPECT` files should be manually accepted or regenerated depending on tolerance for overlong, telegraphic, structurally unusual, or prompt-referential material. The domain/type asymmetry is large enough that model-size-only filtering is insufficient: arXiv textbooks are universally compromised, while medical Q&A and textbooks are uniformly clean under this rubric.
