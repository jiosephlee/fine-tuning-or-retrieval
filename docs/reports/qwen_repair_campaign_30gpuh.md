# Qwen view repair campaign

Date: 2026-07-12

## Outcome

The campaign targeted the flat `blogs.txt`, `stackexchange.txt`, and `textbook.txt` views using the binary completion/sensible-text rubric. Accuracy was not evaluated. GPU usage was capped at 30 hours; actual usage was **2.3903 GPU-hours**.

| Model | Before | After | Net gain | Disposition |
|---|---:|---:|---:|---|
| 4B | 85/108 | **91/108** | +6 | Gave up on 17 stubborn views after three profiles |
| 9B | 82/108 | **85/108** | +3 | Gave up on 23 long-form failures after one compact profile |
| 27B | 87/108 | **87/108** | 0 | Gave up on 21 failures after one compact profile |
| 35B-A3B-FP8 | 71/108 | **71/108** | 0 | No new GPU run; prior regeneration and repair evidence showed poor return |
| 122B-A10B-FP8 | 60/108 | **60/108** | 0 | No new GPU run; prior regeneration produced mostly degenerate replacements at high cost |
| 397B-A17B-FP8 | 57/108 | **57/108** | 0 | No new GPU run; prior regeneration and deterministic assessment showed pervasive defects |

## Experiments and acceptance

- 4B round one: thinking off, temperature 0.7, top-p 0.9, repetition penalty 1.1, 12,288-token cap. Five of 23 regenerated assembled views passed direct binary review.
- 4B round two: thinking off, compact 600–900-word constraint, temperature 1.0, min-p 0.05, repetition penalty 1.15, 6,144-token cap. Zero of 18 passed.
- 4B round three: thinking on, compact constraint, temperature 0.6, top-p 0.85, repetition penalty 1.2, 8,192-token cap. Zero of 18 passed. Failed attempts were restored from backup.
- 9B compact round: zero of 26 generated candidates passed. Originals were restored. Three otherwise coherent legal Q&A views were repaired deterministically by removing localized prompt/source/disclaimer artifacts; all three passed afterward.
- 27B compact round: zero of 21 passed. Originals were restored.
- One 4B legal textbook was repaired deterministically by removing its sole prompt/source-material sentence.
- Deterministic assessments found no safe localized repairs among the inspected 35B and 397B failures; their corruption was embedded throughout substantive content.

All retained replacements or deterministic repairs received direct binary content review. Backups are preserved under `data/.qwen_regeneration_backups/`. No campaign SLURM jobs remain active.

The authoritative current per-file decisions are in `qwen_all_views_binary_integrity_post_repair.tsv`.
