"""Extract final probe metrics (all-domain aggregate) for the explanation-generator runs."""
import json
import sys
from pathlib import Path

REPO_ROOT = Path("/data1/joseph/fine-tuning-or-retrieval")
sys.path.insert(0, str(REPO_ROOT))

from scripts.plotting.plot_utils import load_metrics  # noqa: E402

SUFFIX = "fill_dclm/domains_arxiv_all-legal_all-medical_all/e100/bs256_lr4e-05/overlap_1_16"
B7 = REPO_ROOT / "results/FT/full/7b"

RUNS = {
    "gpt_5_mini_custom (E3 original)": B7
    / f"para9_expl_textbooks+stackexchange+blogs_cyclefull/{SUFFIX}/E3_granular_explanations_all_domains/eval_bundles/reeval_v3",
    "glm (E26)": B7
    / f"para9_docmatch_expl_insertexplanations_glm/{SUFFIX}/E26_explanations_match_glm_all_domains_para9_corrupeted/eval_bundles/inf_mcqa_v14",
    "gpt_5_mini_low (E27)": B7
    / f"para9_docmatch_expl_insertexplanations_gpt_5_mini_low/{SUFFIX}/E27_explanations_match_gpt_5_mini_low_all_domains_para9_corrupeted/eval_bundles/inf_mcqa_v14",
    "gpt_5_mini_high (E28)": B7
    / f"para9_docmatch_expl_insertexplanations_gpt_5_mini_high/{SUFFIX}/E28_explanations_match_gpt_5_mini_high_all_domains_para9_corrupeted/eval_bundles/inf_mcqa_v14",
    "gpt_5_4_mini_high (E29)": B7
    / f"para9_docmatch_expl_insertexplanations_gpt_5_4_mini_high/{SUFFIX}/E29_explanations_match_gpt_5_4_mini_high_all_domains_para9_corrupeted/eval_bundles/inf_mcqa_v14",
    "gpt_5_4_mini_low (E30)": B7
    / f"para9_docmatch_expl_insertexplanations_gpt_5_4_mini_low/{SUFFIX}/E30_explanations_match_gpt_5_4_mini_low_all_domains_para9_corrupeted/eval_bundles/inf_mcqa_v14",
    "gpt_oss_20b_low (E31)": B7
    / f"para9_docmatch_expl_insertexplanations_gpt_oss_20b_low_recovery/{SUFFIX}/E31_explanations_match_gpt_oss_20b_low_all_domains_para9_corrupeted/eval_bundles/inf_mcqa_v14",
    "gpt_oss_120b_low (E32)": B7
    / f"para9_docmatch_expl_insertexplanations_gpt_oss_120b_low_recovery/{SUFFIX}/E32_explanations_match_gpt_oss_120b_low_all_domains_para9_corrupeted/eval_bundles/inf_mcqa_v14",
    "gemma_4_12b (E33)": B7
    / "para9_docmatch_expl_insertexplanations_gemma_4_12b_it_{source}_w16"
    / f"{SUFFIX}/E33_explanations_match_gemma_4_12b_all_domains_para9_corrupeted/eval_bundles/inf_mcqa_v14",
    "gemma_4_31b_nvfp4 (E34)": B7
    / "para9_docmatch_expl_insertexplanations_gemma_4_31b_it_nvfp4_{source}_w16"
    / f"{SUFFIX}/E34_explanations_match_gemma_4_31b_nvfp4_all_domains_para9_corrupeted/eval_bundles/inf_mcqa_v14",
    "glm_5_nvfp4 (E35)": B7
    / f"para9_docmatch_expl_insertexplanations_glm_5_nvfp4/{SUFFIX}/E35_explanations_match_glm_5_nvfp4_all_domains_para9_corrupeted/eval_bundles/inf_mcqa_v14",
}

PANEL_METRICS = [
    ("knowledge", "log_prob", "fact_lp"),
    ("knowledge", "mcqa_accuracy", "fact_mcqa"),
    ("inference", "log_prob", "inf_lp"),
    ("inference", "mcqa_accuracy", "inf_mcqa"),
]


def domains_all():
    domains = []
    for group in ("arxiv", "legal", "medical"):
        gp = REPO_ROOT / "probes" / group
        domains.extend(sorted(p.name for p in gp.iterdir() if p.is_dir()))
    return sorted(set(domains))


DOMAINS = domains_all()
print(f"{len(DOMAINS)} domains")

out = {}
for label, path in RUNS.items():
    row = {}
    for probe_type, metric, short in PANEL_METRICS:
        probe_family = "mcqa" if metric == "mcqa_accuracy" else "classic"
        df = load_metrics(
            str(path),
            probe_type,
            DOMAINS,
            str(REPO_ROOT),
            metrics=(metric,),
            probe_family=probe_family,
            mcqa_variant="regular",
        )
        if df is None or df.empty:
            row[short] = None
            continue
        df = df.sort_values("step")
        row[f"{short}_first_step"] = int(df["step"].iloc[0])
        row[f"{short}_last_step"] = int(df["step"].iloc[-1])
        row[f"{short}_first"] = float(df[metric].iloc[0])
        row[short] = float(df[metric].iloc[-1])
    out[label] = row
    print(label, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})

with open(sys.argv[1] if len(sys.argv) > 1 else "rr_metrics.json", "w") as f:
    json.dump(out, f, indent=1)
