# Safe legal inference MCQA recovery pass.
# Requires a valid utils.keys.OPENAI_API_KEY or OPENAI_API_KEY; writes non-canonical probes_*_recovered_candidates files.
# Run from repo root.

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
    raise SystemExit(f"OpenAI preflight failed; not running recovery commands: {exc}")
PY

python scripts/data-preparation/probes/pipeline_mcqa_difficulty.py --probe_type inference --probe_version v11 --output_version v12_recovered_candidates --filter America_First_Legal_Foundation_v_Jamieson_Greer --row_indices_file reports/legal_mcqa_recovery_v12/regular/passed_filter_no_mcqa/America_First_Legal_Foundation_v_Jamieson_Greer/row_indices.txt --skip_filtering --max_workers 8
python scripts/data-preparation/probes/pipeline_mcqa_difficulty.py --probe_type inference --probe_version v11 --output_version v12_recovered_candidates --filter Apex_Bank_v_Cc_Serve_Corp --row_indices_file reports/legal_mcqa_recovery_v12/regular/passed_filter_no_mcqa/Apex_Bank_v_Cc_Serve_Corp/row_indices.txt --skip_filtering --max_workers 8
python scripts/data-preparation/probes/pipeline_mcqa_difficulty.py --probe_type inference --probe_version v11 --output_version v12_recovered_candidates --filter Bruce_Cohen_v_Consilio_LLC --row_indices_file reports/legal_mcqa_recovery_v12/regular/passed_filter_no_mcqa/Bruce_Cohen_v_Consilio_LLC/row_indices.txt --skip_filtering --max_workers 8
python scripts/data-preparation/probes/pipeline_mcqa_difficulty.py --probe_type inference --probe_version v11 --output_version v12_recovered_candidates --filter Finesse_Wireless_LLC_v_Att_Mobility_LLC --row_indices_file reports/legal_mcqa_recovery_v12/regular/passed_filter_no_mcqa/Finesse_Wireless_LLC_v_Att_Mobility_LLC/row_indices.txt --skip_filtering --max_workers 8
python scripts/data-preparation/probes/pipeline_mcqa_difficulty.py --probe_type inference --probe_version v11 --output_version v12_recovered_candidates --filter Foad_Farahi_v_FBI --row_indices_file reports/legal_mcqa_recovery_v12/regular/passed_filter_no_mcqa/Foad_Farahi_v_FBI/row_indices.txt --skip_filtering --max_workers 8
python scripts/data-preparation/probes/pipeline_mcqa_difficulty.py --probe_type inference --probe_version v11 --output_version v12_recovered_candidates --filter Jimenez_v_Bondi --row_indices_file reports/legal_mcqa_recovery_v12/regular/passed_filter_no_mcqa/Jimenez_v_Bondi/row_indices.txt --skip_filtering --max_workers 8
python scripts/data-preparation/probes/pipeline_mcqa_difficulty.py --probe_type inference --probe_version v11 --output_version v12_recovered_candidates --filter Pacito_v_Trump --row_indices_file reports/legal_mcqa_recovery_v12/regular/passed_filter_no_mcqa/Pacito_v_Trump/row_indices.txt --skip_filtering --max_workers 8
python scripts/data-preparation/probes/pipeline_mcqa_difficulty.py --probe_type inference --probe_version v11 --output_version v12_recovered_candidates --filter Santos_v_Kimmel --row_indices_file reports/legal_mcqa_recovery_v12/regular/passed_filter_no_mcqa/Santos_v_Kimmel/row_indices.txt --skip_filtering --max_workers 8
python scripts/data-preparation/probes/pipeline_mcqa_difficulty.py --probe_type inference --probe_version v11 --output_version v12_recovered_candidates --filter United_States_v_Constantinescu --row_indices_file reports/legal_mcqa_recovery_v12/regular/passed_filter_no_mcqa/United_States_v_Constantinescu/row_indices.txt --skip_filtering --max_workers 8
python scripts/data-preparation/probes/pipeline_mcqa_difficulty.py --probe_type inference --probe_version v11 --output_version v12_recovered_candidates --filter United_States_v_Jaison_Coleman --row_indices_file reports/legal_mcqa_recovery_v12/regular/passed_filter_no_mcqa/United_States_v_Jaison_Coleman/row_indices.txt --skip_filtering --max_workers 8
python scripts/data-preparation/probes/pipeline_mcqa_difficulty.py --probe_type inference --probe_version v11 --output_version v12_recovered_candidates --filter United_States_v_Justin_Cutbank --row_indices_file reports/legal_mcqa_recovery_v12/regular/passed_filter_no_mcqa/United_States_v_Justin_Cutbank/row_indices.txt --skip_filtering --max_workers 8
python scripts/data-preparation/probes/pipeline_mcqa_difficulty.py --probe_type inference --probe_version v11 --output_version v12_recovered_candidates --filter Williams_v_GoAuto_Insurance --row_indices_file reports/legal_mcqa_recovery_v12/regular/passed_filter_no_mcqa/Williams_v_GoAuto_Insurance/row_indices.txt --skip_filtering --max_workers 8
python scripts/data-preparation/probes/pipeline_mcqa_difficulty.py --probe_type inference --probe_version v11_reviewed --output_version v12_reviewed_recovered_candidates --filter America_First_Legal_Foundation_v_Jamieson_Greer --row_indices_file reports/legal_mcqa_recovery_v12/reviewed/passed_filter_no_mcqa/America_First_Legal_Foundation_v_Jamieson_Greer/row_indices.txt --skip_filtering --max_workers 8
python scripts/data-preparation/probes/pipeline_mcqa_difficulty.py --probe_type inference --probe_version v11_reviewed --output_version v12_reviewed_recovered_candidates --filter Apex_Bank_v_Cc_Serve_Corp --row_indices_file reports/legal_mcqa_recovery_v12/reviewed/passed_filter_no_mcqa/Apex_Bank_v_Cc_Serve_Corp/row_indices.txt --skip_filtering --max_workers 8
python scripts/data-preparation/probes/pipeline_mcqa_difficulty.py --probe_type inference --probe_version v11_reviewed --output_version v12_reviewed_recovered_candidates --filter Bruce_Cohen_v_Consilio_LLC --row_indices_file reports/legal_mcqa_recovery_v12/reviewed/passed_filter_no_mcqa/Bruce_Cohen_v_Consilio_LLC/row_indices.txt --skip_filtering --max_workers 8
python scripts/data-preparation/probes/pipeline_mcqa_difficulty.py --probe_type inference --probe_version v11_reviewed --output_version v12_reviewed_recovered_candidates --filter Foad_Farahi_v_FBI --row_indices_file reports/legal_mcqa_recovery_v12/reviewed/passed_filter_no_mcqa/Foad_Farahi_v_FBI/row_indices.txt --skip_filtering --max_workers 8
python scripts/data-preparation/probes/pipeline_mcqa_difficulty.py --probe_type inference --probe_version v11_reviewed --output_version v12_reviewed_recovered_candidates --filter Jimenez_v_Bondi --row_indices_file reports/legal_mcqa_recovery_v12/reviewed/passed_filter_no_mcqa/Jimenez_v_Bondi/row_indices.txt --skip_filtering --max_workers 8
python scripts/data-preparation/probes/pipeline_mcqa_difficulty.py --probe_type inference --probe_version v11_reviewed --output_version v12_reviewed_recovered_candidates --filter Pacito_v_Trump --row_indices_file reports/legal_mcqa_recovery_v12/reviewed/passed_filter_no_mcqa/Pacito_v_Trump/row_indices.txt --skip_filtering --max_workers 8
python scripts/data-preparation/probes/pipeline_mcqa_difficulty.py --probe_type inference --probe_version v11_reviewed --output_version v12_reviewed_recovered_candidates --filter Santos_v_Kimmel --row_indices_file reports/legal_mcqa_recovery_v12/reviewed/passed_filter_no_mcqa/Santos_v_Kimmel/row_indices.txt --skip_filtering --max_workers 8
python scripts/data-preparation/probes/pipeline_mcqa_difficulty.py --probe_type inference --probe_version v11_reviewed --output_version v12_reviewed_recovered_candidates --filter United_States_v_Constantinescu --row_indices_file reports/legal_mcqa_recovery_v12/reviewed/passed_filter_no_mcqa/United_States_v_Constantinescu/row_indices.txt --skip_filtering --max_workers 8
python scripts/data-preparation/probes/pipeline_mcqa_difficulty.py --probe_type inference --probe_version v11_reviewed --output_version v12_reviewed_recovered_candidates --filter United_States_v_Jaison_Coleman --row_indices_file reports/legal_mcqa_recovery_v12/reviewed/passed_filter_no_mcqa/United_States_v_Jaison_Coleman/row_indices.txt --skip_filtering --max_workers 8
python scripts/data-preparation/probes/pipeline_mcqa_difficulty.py --probe_type inference --probe_version v11_reviewed --output_version v12_reviewed_recovered_candidates --filter United_States_v_Justin_Cutbank --row_indices_file reports/legal_mcqa_recovery_v12/reviewed/passed_filter_no_mcqa/United_States_v_Justin_Cutbank/row_indices.txt --skip_filtering --max_workers 8
python scripts/data-preparation/probes/pipeline_mcqa_difficulty.py --probe_type inference --probe_version v11_reviewed --output_version v12_reviewed_recovered_candidates --filter Williams_v_GoAuto_Insurance --row_indices_file reports/legal_mcqa_recovery_v12/reviewed/passed_filter_no_mcqa/Williams_v_GoAuto_Insurance/row_indices.txt --skip_filtering --max_workers 8
