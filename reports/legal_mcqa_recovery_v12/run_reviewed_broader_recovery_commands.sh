# Broader legal inference MCQA recovery pass.
# Requires a valid utils.keys.OPENAI_API_KEY or OPENAI_API_KEY; writes non-canonical probes_*_broader_candidates files.
# Run from repo root. These rows were rejected by the original MCQA prefilter and require manual review before any canonical append.

set -euo pipefail

python - <<'PY'
from utils import utils

if utils.client is None:
    raise SystemExit("No OpenAI client configured. Set OPENAI_API_KEY or utils.keys.OPENAI_API_KEY.")

try:
    utils.query_llm(
        {"system": "Return compact JSON.", "user": 'Return {"ok": true}.'},
        model="gpt-5.4-mini",
        reasoning_effort="low",
        system_prompt_included=True,
        return_json=True,
        max_tokens=20,
        max_try_num=1,
    )
except Exception as exc:
    raise SystemExit(f"OpenAI preflight failed; not running broader recovery commands: {exc}")
PY

python scripts/data-preparation/probes/pipeline_mcqa_difficulty.py --probe_type inference --probe_version v11_reviewed --output_version v12_reviewed_broader_candidates --filter America_First_Legal_Foundation_v_Jamieson_Greer --row_indices_file reports/legal_mcqa_recovery_v12/reviewed/prefilter_rejected/America_First_Legal_Foundation_v_Jamieson_Greer/row_indices.txt --skip_filtering --max_workers 8
python scripts/data-preparation/probes/pipeline_mcqa_difficulty.py --probe_type inference --probe_version v11_reviewed --output_version v12_reviewed_broader_candidates --filter Finesse_Wireless_LLC_v_Att_Mobility_LLC --row_indices_file reports/legal_mcqa_recovery_v12/reviewed/prefilter_rejected/Finesse_Wireless_LLC_v_Att_Mobility_LLC/row_indices.txt --skip_filtering --max_workers 8
python scripts/data-preparation/probes/pipeline_mcqa_difficulty.py --probe_type inference --probe_version v11_reviewed --output_version v12_reviewed_broader_candidates --filter Jimenez_v_Bondi --row_indices_file reports/legal_mcqa_recovery_v12/reviewed/prefilter_rejected/Jimenez_v_Bondi/row_indices.txt --skip_filtering --max_workers 8
python scripts/data-preparation/probes/pipeline_mcqa_difficulty.py --probe_type inference --probe_version v11_reviewed --output_version v12_reviewed_broader_candidates --filter Pacito_v_Trump --row_indices_file reports/legal_mcqa_recovery_v12/reviewed/prefilter_rejected/Pacito_v_Trump/row_indices.txt --skip_filtering --max_workers 8
python scripts/data-preparation/probes/pipeline_mcqa_difficulty.py --probe_type inference --probe_version v11_reviewed --output_version v12_reviewed_broader_candidates --filter Santos_v_Kimmel --row_indices_file reports/legal_mcqa_recovery_v12/reviewed/prefilter_rejected/Santos_v_Kimmel/row_indices.txt --skip_filtering --max_workers 8
python scripts/data-preparation/probes/pipeline_mcqa_difficulty.py --probe_type inference --probe_version v11_reviewed --output_version v12_reviewed_broader_candidates --filter United_States_v_Constantinescu --row_indices_file reports/legal_mcqa_recovery_v12/reviewed/prefilter_rejected/United_States_v_Constantinescu/row_indices.txt --skip_filtering --max_workers 8
python scripts/data-preparation/probes/pipeline_mcqa_difficulty.py --probe_type inference --probe_version v11_reviewed --output_version v12_reviewed_broader_candidates --filter United_States_v_Justin_Cutbank --row_indices_file reports/legal_mcqa_recovery_v12/reviewed/prefilter_rejected/United_States_v_Justin_Cutbank/row_indices.txt --skip_filtering --max_workers 8
python scripts/data-preparation/probes/pipeline_mcqa_difficulty.py --probe_type inference --probe_version v11_reviewed --output_version v12_reviewed_broader_candidates --filter Williams_v_GoAuto_Insurance --row_indices_file reports/legal_mcqa_recovery_v12/reviewed/prefilter_rejected/Williams_v_GoAuto_Insurance/row_indices.txt --skip_filtering --max_workers 8
