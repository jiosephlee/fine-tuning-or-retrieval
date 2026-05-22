# Legal inference MCQA v12 recovery manifest

Generated: 2026-05-22T02:50:36

## regular

- Source rows: 195
- Filtered rows: 155
- Existing MCQA rows: 120
- Passed filter but missing MCQA: 35
- Prefilter rejected: 40

## reviewed

- Source rows: 95
- Filtered rows: 73
- Existing MCQA rows: 52
- Passed filter but missing MCQA: 21
- Prefilter rejected: 22

## Recommended first pass

Run `reports/legal_mcqa_recovery_v12/run_safe_recovery_commands.sh` with an OpenAI key available. This only retries rows that already passed the v12 suitability filter and writes new non-canonical `*_recovered_candidates` files.

After reviewing the generated rows, append only accepted candidates into canonical v12 using the existing `--append_to_version` flow.

## v13 reviewed inference MCQA

Built `probes_v13.csv`, `probes_v13_mcqa.csv`, `probes_v13_readable.txt`, and `mcqa_metrics_v13.txt` for each legal inference domain.

v13 combines:

- existing reviewed v12 MCQA rows: 52
- reviewed safe recovered rows: 14
- final-attempt kept broader rows: 9
- final-attempt repaired broader rows: 6
- final-attempt manually authored hard-failure rows: 10
- final reviewed legal inference MCQA v13 rows: 91

The v13 MCQA files include `formatted_question_5shot` and passed validation for duplicate `(probe, target)` keys, target leakage, and missing few-shot prompts.

## Broader candidate review

The 19 reviewed broader recovered rows were removed from v13 and reviewed separately with agentic manual review.

- Keep-worthy after review: 9
- Drop after review: 10
- Review artifacts: `reports/legal_mcqa_recovery_v12/broader_review/`

The final attempt includes the 9 keep-worthy broader rows and 6 repaired broader rows in `probes_v13_mcqa.csv`. Final-attempt artifacts are under `reports/legal_mcqa_recovery_v12/final_attempt/`.
