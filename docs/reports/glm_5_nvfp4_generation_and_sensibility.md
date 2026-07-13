# GLM-5-NVFP4 generation and sensibility review

Model: `nvidia/GLM-5-NVFP4`
Generation: vLLM on 4 B200 GPUs with tensor parallelism 4
Review criterion: completeness and basic sensibility only, not factual accuracy or quality

## Final results

- 36/36 source items completed: 12 arXiv, 12 medical, and 12 legal.
- 108/108 views are manifest-valid: Stack Exchange, textbook, and blog for every item.
- 683/683 granular child texts were opened and read in full by Codex subagents.
- 108/108 views passed the sensibility rubric; no child or view failed.
- The machine-readable judgments are in `audits/glm_5_nvfp4_sensibility.json`.

Manifest validation covers output hashes, outline schemas and child counts, assembled-text inclusion, malformed or leaked control tokens, truncation indicators, and degeneration checks. The independent subagent pass checks whether each child reads as complete and coherent. Neither check establishes factual correctness.

## GPU usage

| Job | Outcome | GPU seconds | GPU-hours |
| --- | --- | ---: | ---: |
| `7072034` | aborted warmup | 2,316 | 0.643 |
| `7072038` | initial generation | 11,780 | 3.272 |
| `7072802` | recovery generation; wrapper failed after producing valid outputs | 18,716 | 5.199 |
| `7073699` | successful final continuation | 7,740 | 2.150 |
| **Total** |  | **40,552** | **11.264** |

The resumed work used 7.349 of the newly authorized 20 GPU-hours. No additional GPU time was needed after the corpus became complete.
