# Gemma 4 multiview generation and binary-integrity audit

Date: 2026-07-12

## Scope and configuration

This campaign generated every arXiv, medical, and legal multiview with:

- `google/gemma-4-12B-it`
- `nvidia/Gemma-4-31B-IT-NVFP4` with `--quantization modelopt`

Both models used `temperature=1.0`, `top_p=0.95`, `top_k=64`, a per-generation
output budget of 65,000 tokens, `--reasoning-parser gemma4`, and the vLLM JSON
equivalent of `--limit-mm-per-prompt image=0,audio=0`. The served context limit
was 131,072 tokens and generation used 16 concurrent pipeline workers.

The final inventory contains 216 assembled views: 2 models × 3 domains × 12
subjects × 3 view types. Each subject has `blogs.txt`, `stackexchange.txt`, and
`textbook.txt`.

## Agentic binary review

Six independent review partitions directly inspected every assembled file. A
view received `PASS` only if it remained coherent, sensible, and complete
throughout. `FAIL` covered truncation, repetition or lexical cascades,
gibberish/fused text, multilingual or unrelated drift, prompt/reasoning/drafting
leakage, and materially malformed prose. Factual, medical, legal, and paper-level
accuracy verification was outside this audit's scope.

| Model | Domain | PASS | FAIL | Total |
|---|---|---:|---:|---:|
| Gemma 4 12B IT | arXiv | 36 | 0 | 36 |
| Gemma 4 12B IT | Medical | 36 | 0 | 36 |
| Gemma 4 12B IT | Legal | 36 | 0 | 36 |
| Gemma 4 31B IT NVFP4 | arXiv | 36 | 0 | 36 |
| Gemma 4 31B IT NVFP4 | Medical | 36 | 0 | 36 |
| Gemma 4 31B IT NVFP4 | Legal | 36 | 0 | 36 |
| **Overall** |  | **216** | **0** | **216** |

| View type | PASS | FAIL | Total |
|---|---:|---:|---:|
| Blogs | 72 | 0 | 72 |
| Stack Exchange / teaching Q&A | 72 | 0 | 72 |
| Textbooks | 72 | 0 | 72 |

The per-file evidence ledger is
`docs/reports/gemma4_all_views_binary_integrity_audit.tsv`. It contains exactly
216 unique paths and no missing or extra inventory entries.

## Deterministic integrity repairs

Two generated views needed notation-only repair before final acceptance:

1. The 31B medical HNF1A diabetes Q&A contained one JSON-decoded backspace where
   `\beta` was intended. The byte was restored to `\beta` in the granular and
   assembled copies.
2. The 12B arXiv FA3 Q&A contained three JSON-decoded tab-plus-`op` sequences
   where `\top` was intended. All three were restored in the granular and
   assembled copies.

Reviewers re-read both repaired views, confirmed that the surrounding content
was undamaged, and changed them to `PASS`. Their generation manifests were
recomputed and now validate against the repaired file hashes.

Two final-gate failures were validator false positives rather than corpus
defects. A BOFT derivation repeated the LaTeX command `\bm`, and a legal answer's
ordinary use of the stopword “the” barely exceeded the old unigram threshold.
The validator now excludes LaTeX command names and common stopwords from the
single-word frequency gate while retaining trigram-loop detection. Regression
tests cover both cases.

## Runtime compatibility and resource use

The NVIDIA checkpoint exposed a vLLM 0.24 ModelOpt incompatibility: an excluded,
tied `ParallelLMHead` was assigned an unquantized linear method with no
`tie_weights()` implementation. A narrow repository-local wrapper makes only
that excluded head use vLLM's normal embedding-aware unquantized method; all
other ModelOpt quantization remains unchanged.

Compile caches were moved to allocation-local scratch after one Triton compile
hit a stale NFS file handle. FlashInfer's optional NVFP4 autotuner was disabled
after three starts remained at 0/21; the default NVFP4 kernel then initialized
and passed structured-JSON and long-prose endpoint smoke tests.

Final SLURM accounting covers all 19 one-GPU Gemma jobs, including malformed
startup attempts, failed final integrity gates, and the three cancelled
autotuner attempts:

| SLURM terminal state | GPU-hours |
|---|---:|
| Completed | 2.089722 |
| Failed | 2.070833 |
| Cancelled | 0.387500 |
| **Total** | **4.548056** |

Some `FAILED` jobs generated complete trees and exited nonzero only because the
final integrity gate found one of the notation defects or false positives
described above. The authoritative completion evidence is the final 216-view
inventory, current validation/manifests, and the agentic audit ledger—not SLURM
terminal state alone.

Total use was 45.5% of the requested 10 GPU-hour ceiling.
