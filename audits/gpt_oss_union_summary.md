# GPT-OSS 20B/120B multiview union audit

Generated from the deterministic post-union audit on 2026-07-12 and independent
read-only domain reviews by three Codex subagents.

## Coverage

| Product | arXiv | Medical | Legal |
|---|---:|---:|---:|
| 20B low | 18/36 | 35/36 | 33/36 |
| 20B high | 10/36 | 36/36 | 36/36 |
| 120B low | 21/36 | 36/36 | 36/36 |
| 120B high | 10/36 | 17/36 | 19/36 |

The initial structural recovery unions contained 307 of 432 expected item/views.
Luna-low rejected all 307 selected candidates. A second union pass found 42
structurally valid fallback candidates; Luna-low rejected all 42 as well. After
source-specific reconciliation, the safe union contains 0/432 selected views and
all 432 item/views require regeneration. Files remain in recovery folders for
forensic comparison, but `union_manifest.json` selects none of them. Canonical
folders were not changed.

## Deterministic findings

- 648 variant/item/view records were audited: 367 passed and 281 failed.
- Dominant failures were outline schema drift, missing assembled/granular files,
  outline/granular count disagreements, and invalid outline JSON.
- Hard corruption included 22 control-character findings, 11 separator-abuse
  findings, and 3 Harmony-marker findings.
- Existing validated recovery views were preferred, followed by canonical views,
  then stable historical variants. File size was never a ranking input.
- Each recovery root contains `union_manifest.json`; each selected item/view has
  exact hashes and source provenance in its manifests.

## Luna-low semantic reconciliation

Every initially selected view was reviewed with `gpt-5.6-luna` at low reasoning
against its cleaned source and generated assembled/granular evidence. Counts were
59 arXiv, 124 medical, and 124 legal: 0 passed and 307 were rejected. The 42
fallback candidates were reviewed in a second round: 0 passed and 42 were rejected.
Failures include fabricated experiments and metrics, invented legal authorities or
reversed holdings, contradictory clinical facts and unsafe protocols, as well as
reserved-token leakage, garbling, malformed endings, and abrupt truncation.

The collaboration interface has no per-subagent model parameter, so the three
domain subagents explicitly invoked `model="gpt-5.6-luna"` with
`reasoning_effort="low"`. Full per-view evidence and the combined 349
source-specific rejections are stored in the `audits/` Luna review files.
