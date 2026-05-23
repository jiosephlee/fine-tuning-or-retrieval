#!/usr/bin/env python3
"""Build factual v15 MCQA probes aligned to v14 short-target cloze probes."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from utils.mcqa_prompts import build_mcqa_5shot_prompt  # noqa: E402


PROBE_ROOT = REPO_ROOT / "probes"
REPORT_DIR = REPO_ROOT / "reports" / "factual_mcqa_v15_short_targets"
REVIEW_INPUT = REPORT_DIR / "review_input_shortened_mcqa.csv"
AGENT_DECISIONS = REPORT_DIR / "shortened_mcqa_agent_decisions.csv"
ACCEPTED_REPAIRS = REPORT_DIR / "shortened_mcqa_accepted.csv"
REJECTED_REPAIRS = REPORT_DIR / "shortened_mcqa_rejected.csv"
SUMMARY_PATH = REPORT_DIR / "summary.csv"
VALIDATION_PATH = REPORT_DIR / "validation.csv"

DOMAIN_GROUPS = ("medical", "legal", "arxiv")
LABELS = ("(A)", "(B)", "(C)", "(D)", "(E)")
OPTION_COLS = ("option_a", "option_b", "option_c", "option_d", "option_e")
REJECT_SHORT_TARGETS_IN_QUESTION_FORM = True
MCQA_REPAIR_COLS = (
    "probe",
    "target",
    "formatted_question",
    "option_a",
    "option_b",
    "option_c",
    "option_d",
    "option_e",
    "correct_label",
    "distractors",
    "formatted_question_5shot",
)

MANUAL_REJECTION_ROWS: dict[tuple[str, str], set[int]] = {
    ("medical", "Acute_neurological_Behets_disease_complicated_by_a_syndrome"): {28, 43},
    ("medical", "Autoimmune_heparininduced_thrombocytopenia_following_cardiac"): {18, 61},
    ("medical", "Central_hypoventilation_and_choking_episodes_revealing_Chiar"): {
        0,
        24,
        71,
        76,
        82,
        89,
    },
    ("medical", "Dermatomyositis_masquerading_as_angioedema_a_crucial_differe"): {36, 39, 47},
    ("medical", "Immunemediated_necrotising_myopathy_following_semaglutide_tr"): {13, 32, 47, 53},
    ("medical", "Management_of_atypical_complicated_abdominal_masses_in_the_s"): {
        30,
        49,
        53,
        68,
        87,
    },
    ("medical", "Monoallelic_PARN_mutation_presenting_as_pancytopenia_hepatic"): {57, 64},
    ("medical", "Multiphasic_anaphylaxis_in_the_emergency_and_intensive_care"): {
        18,
        30,
        45,
        51,
        53,
        59,
        73,
        81,
        99,
        110,
    },
    ("medical", "Nontype_1_and_nontype_2_diabetes_in_a_young_man_due_to_novel"): {18, 60},
    ("medical", "Pancreatopleural_fistula_in_childhood"): {44, 53, 69, 70, 71, 73, 76, 92, 93},
    ("medical", "TAVinTAVinTAV_after_treated_endocarditis_procedural_strategy"): {1, 64, 83, 84, 103},
    ("legal", "America_First_Legal_Foundation_v_Jamieson_Greer"): {29, 38, 45},
    ("legal", "Apex_Bank_v_Cc_Serve_Corp"): {3, 21, 32, 34, 35, 54},
    ("legal", "Bruce_Cohen_v_Consilio_LLC"): {17, 23, 26, 43},
    ("legal", "Finesse_Wireless_LLC_v_Att_Mobility_LLC"): {2, 9, 17, 33, 40, 45, 60},
    ("legal", "Foad_Farahi_v_FBI"): {11, 29, 36},
    ("legal", "Jimenez_v_Bondi"): {7, 22, 52, 71},
    ("legal", "Pacito_v_Trump"): {5, 9, 24, 28, 31, 33},
    ("legal", "Santos_v_Kimmel"): {2},
    ("legal", "United_States_v_Constantinescu"): {11},
    ("legal", "United_States_v_Jaison_Coleman"): {15, 41, 50},
    ("legal", "United_States_v_Justin_Cutbank"): {7, 9, 16, 27, 30, 32, 43},
    ("legal", "Williams_v_GoAuto_Insurance"): {17, 18, 28},
    ("arxiv", "BOFT"): {164, 202},
    ("arxiv", "ByteLatent"): {60, 101, 138, 146, 364, 366},
    ("arxiv", "DPO"): {22, 165, 207},
    ("arxiv", "FeatLLM"): {89, 97, 99, 124, 140, 209},
    ("arxiv", "GRPO"): {188},
    ("arxiv", "GSPO"): {52, 74, 118},
    ("arxiv", "LongRoPE"): {12, 137},
    ("arxiv", "OFT"): {174, 230},
    ("arxiv", "QLoRA"): {75, 100, 176, 285},
    ("arxiv", "xLSTM"): {27, 28, 71, 155},
}

GLOBAL_OPTION_FIXES: dict[tuple[str, str, int], dict[str, str]] = {
    ("arxiv", "ByteLatent", 95): {
        "option_c": "$m_{\\mathcal{D}}$.",
        "distractors": json.dumps(
            ["$h_0$", "$m_{\\mathcal{D}}$.", "$\\mathcal{D}_C(o_j)$", "$d_{l-1,i}$"],
            ensure_ascii=False,
        ),
    },
    ("arxiv", "xLSTM", 163): {
        "option_d": "$t + k$.",
        "distractors": json.dumps(
            ["$t - \\tau$.", "$t + d$.", "$t + k$.", "$t + 1$."],
            ensure_ascii=False,
        ),
    },
}

QUESTION_STEM_OVERRIDES: dict[tuple[str, str, int], str] = {
    ("arxiv", "1_58", 19): (
        'According to the paper "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits", '
        "enlarging an on-chip accelerator's SRAM to improve throughput is said to introduce significantly "
        "higher costs than which off-chip memory technology?"
    ),
    ("arxiv", "1_58", 97): (
        'According to the paper "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits", '
        "in the benchmark list used to compare two models trained with 2T tokens, which dataset was listed first?"
    ),
    ("arxiv", "ByteLatent", 280): (
        'According to the paper "Byte Latent Transformer: Patches Scale Better Than Tokens", '
        "which encoder cross-attention query initialization is used at the given encoder layer?"
    ),
    ("arxiv", "DPO", 54): (
        'In the paper "Direct Preference Optimization", assuming access to a static dataset of comparisons '
        "$\\mathcal{D}=\\bigl\\{x^{(i)}, y_w^{(i)}, y_l^{(i)}\\bigr\\}_{i=1}^N$ sampled from $p^*$, "
        "what kind of model can be parametrized and have its parameters estimated via maximum likelihood?"
    ),
    ("arxiv", "DPO", 100): (
        'According to the paper "Direct Preference Optimization", when the preference datasets are sampled '
        "using the supervised fine-tuned policy, what is $\\pi_\\text{ref}$ initialized to whenever it is available?"
    ),
    ("arxiv", "DPO", 226): (
        'According to the paper "Direct Preference Optimization", in the RL fine-tuning objective, '
        "from what distribution are the outputs $y$ sampled in the expectation term?"
    ),
    ("arxiv", "GRPO", 222): (
        'According to the paper "DeepSeekMath", what kind of reasoning does code training improve models\' '
        "ability to do within the relevant domain, both with and without tool use?"
    ),
    ("arxiv", "GRPO", 311): (
        'According to the paper "DeepSeekMath", when process supervision normalizes step-level rewards, '
        "what quantity is subtracted from $r_i^{index(j)}$?"
    ),
    ("arxiv", "OFT", 168): (
        'According to the paper "Controlling Text-to-Image Diffusion by Orthogonal Finetuning", '
        "in the authors' connection between OFT and LoRA, what expression is analogous to LoRA's low-rank weight update?"
    ),
    ("arxiv", "OFT", 217): (
        'In the paper "Controlling Text-to-Image Diffusion by Orthogonal Finetuning", in the qualitative '
        "comparison on subject preservation and prompt following, which method preserves subject identity but "
        "often fails to generate the image following the text prompt?"
    ),
    ("arxiv", "OFT", 249): (
        'According to the paper "Controlling Text-to-Image Diffusion by Orthogonal Finetuning", '
        "in the C2I task, which methods are able to hallucinate based on rough Canny edges while T2I-Adapter "
        "and LoRA perform much worse?"
    ),
    ("arxiv", "QLoRA", 217): (
        'According to the paper "QLoRA: Efficient Finetuning of Quantized LLMs", in the model-based '
        "evaluation comparing two benchmarks, which benchmark is reported to favor open-source models?"
    ),
    ("arxiv", "QLoRA", 247): (
        'In the paper "QLoRA: Efficient Finetuning of Quantized LLMs", in the example where James knows '
        "that Abby originally put the pen in one place and later moved it into the bag, where may James guess "
        "that Abby will look for the pen?"
    ),
    ("arxiv", "fa3", 51): (
        'In the paper "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision", '
        "what is the coarsest level in the GPU programming model thread hierarchy?"
    ),
    ("arxiv", "xLSTM", 27): (
        'According to the paper "xLSTM: Extended Long Short-Term Memory", in the LSTM formulation, '
        "the weight vectors $\\mathbf{w}_{z}$, $\\mathbf{w}_{i}$, $\\mathbf{w}_{f}$, and $\\mathbf{w}_{o}$ "
        "correspond to input weight vectors between $\\mathbf{x}_t$ and which four components?"
    ),
    ("legal", "Finesse_Wireless_LLC_v_Att_Mobility_LLC", 0): (
        'According to the opinion in "Finesse Wireless LLC v. AT&T Mobility LLC," under the standard '
        "for reviewing a jury's finding of infringement, what level of evidentiary support is required for "
        "a factual finding to be upheld?"
    ),
    ("legal", "Finesse_Wireless_LLC_v_Att_Mobility_LLC", 3): (
        'In the case "Finesse Wireless LLC v. AT&T Mobility LLC", according to the opinion, what claimed '
        "category of signals causing interference did Dr. Wells fail to map x1 or x2 onto?"
    ),
    ("legal", "Williams_v_GoAuto_Insurance", 37): (
        'In the case "Williams v. GoAuto Insurance Company", according to the opinion, no Louisiana judicial '
        "opinion considering what issue has addressed whether the party signing had authority to sign on behalf "
        "of the insurer?"
    ),
    ("medical", "Pancreatopleural_fistula_in_childhood", 110): (
        'According to the case report "Pancreatopleural fistula in childhood", in addition to unnecessary '
        "pulmonary interventions and sepsis, what can early diagnosis prevent?"
    ),
}


@dataclass(frozen=True)
class Match:
    domain_group: str
    document: str
    mcqa_row_index: int
    cloze_row_index: int
    old_row: pd.Series
    short_row: pd.Series
    question_stem: str


def normalize_space(text: object) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def normalize_answer(text: object) -> str:
    return normalize_space(text).casefold()


def word_count(text: object) -> int:
    return len(str(text).strip().split())


def strip_option_text(text: object) -> str:
    return normalize_space(text).strip()


def extract_question_text(text: object) -> str:
    raw = str(text).strip()
    raw = raw.replace("\r\n", "\n")

    match = re.search(
        r"(?:^|\n)\s*(?:\d+\.\s*)?\*\*Question:\*\*\s*(.*?)(?:\n\s*)?\*\*Answer:\*\*",
        raw,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        match = re.search(
            r"(?:^|\n)\s*(?:\d+\.\s*)?Question:\s*(.*?)(?:\n\s*)?Answer:",
            raw,
            flags=re.IGNORECASE | re.DOTALL,
        )
    if match:
        raw = match.group(1).strip()

    raw = re.sub(r"^\s*\d+\.\s*", "", raw)
    raw = raw.replace("**", "")
    raw = raw.replace("__", "")
    raw = raw.replace("`", "")
    raw = re.sub(r"\s+", " ", raw).strip()
    if "?" in raw:
        raw = raw[: raw.index("?") + 1].strip()
    return raw


def discover_mcqa_paths(domain_groups: tuple[str, ...]) -> list[Path]:
    paths: list[Path] = []
    for group in domain_groups:
        paths.extend((PROBE_ROOT / group).glob("*/facts/probes_v14_mcqa.csv"))
    return sorted(paths)


def path_parts(path: Path) -> tuple[str, str]:
    rel = path.relative_to(PROBE_ROOT)
    return rel.parts[0], rel.parts[1]


def build_fact_index(df: pd.DataFrame) -> dict[str, list[tuple[int, pd.Series]]]:
    fact_index: dict[str, list[tuple[int, pd.Series]]] = {}
    for idx, row in df.iterrows():
        fact_index.setdefault(normalize_space(row["fact"]), []).append((int(idx), row))
    return fact_index


def build_context_index(df: pd.DataFrame) -> tuple[dict[tuple[str, str], list[tuple[int, pd.Series]]], dict[str, list[tuple[int, pd.Series]]]]:
    exact_index: dict[tuple[str, str], list[tuple[int, pd.Series]]] = {}
    fact_index: dict[str, list[tuple[int, pd.Series]]] = {}
    for idx, row in df.iterrows():
        fact = normalize_space(row["fact"])
        raw = normalize_space(row.get("raw_knowledge_statement", ""))
        exact_index.setdefault((fact, raw), []).append((int(idx), row))
        fact_index.setdefault(fact, []).append((int(idx), row))
    return exact_index, fact_index


def choose_context_row(
    mcqa_row: pd.Series,
    exact_index: dict[tuple[str, str], list[tuple[int, pd.Series]]],
    fact_index: dict[str, list[tuple[int, pd.Series]]],
) -> tuple[int, pd.Series]:
    fact = normalize_space(mcqa_row["fact"])
    raw = normalize_space(mcqa_row.get("raw_knowledge_statement", ""))
    candidates = exact_index.get((fact, raw), [])
    if len(candidates) == 1:
        return candidates[0]

    candidates = fact_index.get(fact, [])
    exact_probe_target = [
        (idx, row)
        for idx, row in candidates
        if normalize_space(row["probe"]) == normalize_space(mcqa_row["probe"])
        and normalize_answer(row["target"]) == normalize_answer(mcqa_row["target"])
    ]
    if len(exact_probe_target) == 1:
        return exact_probe_target[0]
    if len(candidates) == 1:
        return candidates[0]
    question_texts = {
        extract_question_text(row.get("contextualized_question", ""))
        for _, row in candidates
    }
    if len(question_texts) == 1:
        return sorted(candidates, key=lambda item: item[0])[0]

    raise ValueError(
        "Could not choose unique contextualized_question row for "
        f"fact={fact[:120]!r}; candidates={len(candidates)}"
    )


def contextualized_question_for_mcqa(mcqa_path: Path, mcqa_row: pd.Series) -> str:
    cloze_path = mcqa_path.with_name("probes_v14.csv")
    cloze = pd.read_csv(cloze_path, keep_default_na=False)
    exact_index, fact_index = build_context_index(cloze)
    _, context_row = choose_context_row(mcqa_row, exact_index, fact_index)
    question = extract_question_text(context_row.get("contextualized_question", ""))
    if not question:
        raise ValueError(f"Empty contextualized_question for {mcqa_path}")
    return question


def question_stem_for_row(
    domain_group: str,
    document: str,
    row_index: int,
    context_row: pd.Series,
) -> str:
    override = QUESTION_STEM_OVERRIDES.get((domain_group, document, row_index))
    if override:
        return normalize_space(override)
    question = extract_question_text(context_row.get("contextualized_question", ""))
    if not question:
        raise ValueError(f"Empty contextualized_question for {domain_group}/{document} row {row_index}")
    return question


def choose_shortened_candidate(
    mcqa_row: pd.Series,
    candidates: list[tuple[int, pd.Series]],
) -> tuple[int, pd.Series] | None:
    shortened = [
        (idx, row)
        for idx, row in candidates
        if word_count(row["target"]) < word_count(mcqa_row["target"])
        and normalize_answer(row["target"]) != normalize_answer(mcqa_row["target"])
    ]
    if not shortened:
        return None

    exact_old = [
        (idx, row)
        for idx, row in shortened
        if normalize_space(row["fact"]) == normalize_space(mcqa_row["fact"])
    ]
    pool = exact_old or shortened

    # Duplicate cloze rows can produce the same repaired stem and target. Pick the
    # first row index to keep one MCQA output row per original MCQA row.
    return sorted(pool, key=lambda item: item[0])[0]


def iter_matches(domain_groups: tuple[str, ...]) -> list[Match]:
    matches: list[Match] = []
    for mcqa_path in discover_mcqa_paths(domain_groups):
        domain_group, document = path_parts(mcqa_path)
        short_path = mcqa_path.with_name("probes_v14_short_targets.csv")
        if not short_path.exists():
            raise FileNotFoundError(f"Missing short-target cloze file: {short_path}")

        mcqa = pd.read_csv(mcqa_path, keep_default_na=False)
        short = pd.read_csv(short_path, keep_default_na=False)
        fact_index = build_fact_index(short)
        cloze = pd.read_csv(mcqa_path.with_name("probes_v14.csv"), keep_default_na=False)
        context_exact_index, context_fact_index = build_context_index(cloze)

        for row_index, row in mcqa.iterrows():
            candidates = fact_index.get(normalize_space(row["fact"]), [])
            if not candidates:
                raise ValueError(f"No cloze fact match for {mcqa_path} row {row_index}")
            candidate = choose_shortened_candidate(row, candidates)
            if candidate is None:
                continue
            cloze_row_index, short_row = candidate
            matches.append(
                Match(
                    domain_group=domain_group,
                    document=document,
                    mcqa_row_index=int(row_index),
                    cloze_row_index=int(cloze_row_index),
                    old_row=row,
                    short_row=short_row,
                    question_stem=question_stem_for_row(
                        domain_group,
                        document,
                        int(row_index),
                        choose_context_row(row, context_exact_index, context_fact_index)[1],
                    ),
                )
            )
    return matches


def read_rewrite_reason(domain_group: str, document: str, cloze_row_index: int) -> str:
    path = REPO_ROOT / "reports" / f"{domain_group}_factual_v14_short_targets" / "decisions.csv"
    if not path.exists():
        return ""
    decisions = pd.read_csv(path, keep_default_na=False)
    rows = decisions[
        (decisions["domain"] == document)
        & (decisions["probe_index"].astype(str) == str(cloze_row_index))
    ]
    if rows.empty:
        return ""
    row = rows.iloc[0]
    reason = str(row.get("reason", "")).strip()
    status = str(row.get("validation_status", "")).strip()
    if reason and status:
        return f"{status}: {reason}"
    return reason or status


def build_review_input(matches: list[Match]) -> pd.DataFrame:
    rows = []
    for match in matches:
        old = match.old_row
        short = match.short_row
        rows.append(
            {
                "domain_group": match.domain_group,
                "document": match.document,
                "v14_mcqa_row_index": match.mcqa_row_index,
                "short_cloze_row_index": match.cloze_row_index,
                "old_cloze_probe": old["probe"],
                "old_cloze_target": old["target"],
                "shortened_cloze_probe": short["probe"],
                "shortened_cloze_target": strip_option_text(short["target"]),
                "contextualized_question": match.question_stem,
                "old_formatted_question": old["formatted_question"],
                "old_option_a": old["option_a"],
                "old_option_b": old["option_b"],
                "old_option_c": old["option_c"],
                "old_option_d": old["option_d"],
                "old_option_e": old["option_e"],
                "old_correct_label": old["correct_label"],
                "fact": old["fact"],
                "source_statement": old.get("raw_knowledge_statement", ""),
                "rewrite_reason": read_rewrite_reason(
                    match.domain_group, match.document, match.cloze_row_index
                ),
            }
        )
    return pd.DataFrame(rows)


def remove_moved_prefix(option: str, old_target: str, new_target: str) -> str:
    option_norm = normalize_space(option)
    old_target_norm = normalize_space(old_target)
    new_target_norm = normalize_space(new_target)
    if not option_norm:
        return option_norm

    if old_target_norm.casefold().endswith(new_target_norm.casefold()):
        moved_prefix = old_target_norm[: -len(new_target_norm)].strip()
        if moved_prefix and option_norm.casefold().startswith(moved_prefix.casefold()):
            return option_norm[len(moved_prefix) :].strip(" ,;:")

    return option_norm


def format_question(stem: str, options: list[str]) -> str:
    formatted_options = [f"{label} {option}" for label, option in zip(LABELS, options)]
    return f"{normalize_space(stem)}\n" + "\n".join(formatted_options)


def validate_repair_record(record: dict[str, object]) -> tuple[bool, str]:
    target = strip_option_text(record["target"])
    stem = normalize_space(record["probe"])
    options = [strip_option_text(record[col]) for col in OPTION_COLS]

    if len(options) != 5 or any(not option for option in options):
        return False, "missing option"
    if record["correct_label"] not in LABELS:
        return False, "invalid correct label"
    correct_idx = LABELS.index(str(record["correct_label"]))
    if options[correct_idx] != target:
        return False, "correct option does not equal target"
    if sum(normalize_answer(option) == normalize_answer(target) for option in options) != 1:
        return False, "target is not unique among options"
    if normalize_answer(target) in normalize_answer(stem):
        return False, "target leakage in stem"
    expected_question = format_question(stem, options)
    if record["formatted_question"] != expected_question:
        return False, "formatted_question mismatch"
    if record["formatted_question_5shot"] != build_mcqa_5shot_prompt(expected_question):
        return False, "formatted_question_5shot mismatch"
    return True, "accepted"


def repair_match(match: Match) -> dict[str, object]:
    old = match.old_row
    short = match.short_row
    new_probe = normalize_space(match.question_stem)
    new_target = strip_option_text(short["target"])
    old_target = strip_option_text(old["target"])

    old_options = [strip_option_text(old[col]) for col in OPTION_COLS]
    correct_idx = LABELS.index(str(old["correct_label"]))
    new_options: list[str] = []
    for idx, option in enumerate(old_options):
        if idx == correct_idx:
            new_options.append(new_target)
        else:
            new_options.append(remove_moved_prefix(option, old_target, new_target))

    formatted_question = format_question(new_probe, new_options)
    distractors = [option for idx, option in enumerate(new_options) if idx != correct_idx]
    record: dict[str, object] = {
        "domain_group": match.domain_group,
        "document": match.document,
        "v14_mcqa_row_index": match.mcqa_row_index,
        "short_cloze_row_index": match.cloze_row_index,
        "decision": "ACCEPT",
        "decision_reason": "Deterministic repair: updated stem and correct option to the shortened cloze target; retained existing distractors after prefix cleanup.",
        "probe": new_probe,
        "target": new_target,
        "formatted_question": formatted_question,
        "option_a": new_options[0],
        "option_b": new_options[1],
        "option_c": new_options[2],
        "option_d": new_options[3],
        "option_e": new_options[4],
        "correct_label": old["correct_label"],
        "distractors": json.dumps(distractors, ensure_ascii=False),
        "formatted_question_5shot": build_mcqa_5shot_prompt(formatted_question),
        "fact": old["fact"],
        "raw_knowledge_statement": old.get("raw_knowledge_statement", ""),
        "old_probe": old["probe"],
        "old_formatted_question": old["formatted_question"],
        "contextualized_question": match.question_stem,
        "old_target": old["target"],
    }
    accepted, status = validate_repair_record(record)
    if not accepted:
        record["decision"] = "REJECT"
        record["decision_reason"] = status
    if match.mcqa_row_index in MANUAL_REJECTION_ROWS.get((match.domain_group, match.document), set()):
        record["decision"] = "REJECT"
        record["decision_reason"] = (
            "Manual subagent review rejected the shortened MCQA repair because "
            "the inherited distractors were malformed or misaligned with the "
            "shortened stem; v15 keeps the original v14 MCQA row."
        )
    if REJECT_SHORT_TARGETS_IN_QUESTION_FORM:
        record["decision"] = "REJECT"
        record["decision_reason"] = (
            "Question-form v15 uses the existing contextualized_question stem; "
            "the shortened cloze target is not applied because it would require "
            "a separate manual question rewrite to ensure the shortened answer "
            "still directly answers the question."
        )
    return record


def build_decisions(matches: list[Match]) -> pd.DataFrame:
    return pd.DataFrame([repair_match(match) for match in matches])


def apply_question_form(row: pd.Series, question_stem: str) -> dict[str, object]:
    output = row.to_dict()
    options = [strip_option_text(output[col]) for col in OPTION_COLS]
    question = normalize_space(question_stem)
    formatted_question = format_question(question, options)
    output["probe"] = question
    output["formatted_question"] = formatted_question
    output["formatted_question_5shot"] = build_mcqa_5shot_prompt(formatted_question)
    return output


def apply_global_option_fixes(
    row: dict[str, object],
    domain_group: str,
    document: str,
    row_index: int,
) -> dict[str, object]:
    fixes = GLOBAL_OPTION_FIXES.get((domain_group, document, row_index))
    if not fixes:
        return row
    row = dict(row)
    for col, value in fixes.items():
        row[col] = value
    options = [strip_option_text(row[col]) for col in OPTION_COLS]
    row["formatted_question"] = format_question(str(row["probe"]), options)
    row["formatted_question_5shot"] = build_mcqa_5shot_prompt(str(row["formatted_question"]))
    return row


def validate_file(
    mcqa_path: Path,
    out_path: Path,
    repaired_rows: dict[int, dict[str, object]],
) -> list[dict[str, object]]:
    v14 = pd.read_csv(mcqa_path, keep_default_na=False)
    v15 = pd.read_csv(out_path, keep_default_na=False)
    failures: list[dict[str, object]] = []

    if list(v14.columns) != list(v15.columns):
        failures.append({"path": str(out_path), "row": "", "status": "column mismatch"})
    if len(v14) != len(v15):
        failures.append({"path": str(out_path), "row": "", "status": "row count mismatch"})

    for idx in range(min(len(v14), len(v15))):
        repair = repaired_rows.get(idx)
        row = v15.iloc[idx].to_dict()
        accepted, status = validate_repair_record(row)
        if not accepted:
            failures.append({"path": str(out_path), "row": idx, "status": status})
        if repair and repair["decision"] == "ACCEPT":
            if normalize_answer(row["target"]) != normalize_answer(repair["target"]):
                failures.append({"path": str(out_path), "row": idx, "status": "target is not shortened target"})
        elif normalize_answer(row["target"]) != normalize_answer(v14.iloc[idx]["target"]):
            failures.append({"path": str(out_path), "row": idx, "status": "non-accepted target changed"})

        allowed_changed = {
            "probe",
            "formatted_question",
            "formatted_question_5shot",
        }
        if repair and repair["decision"] == "ACCEPT":
            allowed_changed.update(MCQA_REPAIR_COLS)
        if (path_parts(out_path)[0], path_parts(out_path)[1], idx) in GLOBAL_OPTION_FIXES:
            allowed_changed.update({"option_a", "option_b", "option_c", "option_d", "option_e", "distractors"})
        metadata_changes = [
            col
            for col in v14.columns
            if col not in allowed_changed and str(v14.iloc[idx][col]) != str(v15.iloc[idx][col])
        ]
        if metadata_changes:
            failures.append(
                {
                    "path": str(out_path),
                    "row": idx,
                    "status": f"unexpected metadata changes: {metadata_changes}",
                }
            )

    return failures


def write_outputs(matches: list[Match], decisions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    decision_map = {
        (row.domain_group, row.document, int(row.v14_mcqa_row_index)): row._asdict()
        for row in decisions.itertuples(index=False)
    }

    summary_rows = []
    validation_failures: list[dict[str, object]] = []
    for mcqa_path in discover_mcqa_paths(DOMAIN_GROUPS):
        domain_group, document = path_parts(mcqa_path)
        out_path = mcqa_path.with_name("probes_v15_mcqa.csv")
        df = pd.read_csv(mcqa_path, keep_default_na=False)
        before_cols = list(df.columns)
        cloze = pd.read_csv(mcqa_path.with_name("probes_v14.csv"), keep_default_na=False)
        context_exact_index, context_fact_index = build_context_index(cloze)

        relevant = {
            int(key[2]): value
            for key, value in decision_map.items()
            if key[0] == domain_group and key[1] == document
        }
        accepted = 0
        rejected = 0
        output_rows: list[dict[str, object]] = []
        for idx, row in df.iterrows():
            _, context_row = choose_context_row(row, context_exact_index, context_fact_index)
            question_stem = question_stem_for_row(domain_group, document, int(idx), context_row)
            output_row = apply_question_form(row, question_stem)
            output_row = apply_global_option_fixes(output_row, domain_group, document, int(idx))
            output_rows.append(output_row)

        df = pd.DataFrame(output_rows, columns=before_cols)
        for idx, repair in relevant.items():
            if repair["decision"] != "ACCEPT":
                rejected += 1
                continue
            accepted += 1
            for col in MCQA_REPAIR_COLS:
                df.loc[idx, col] = repair[col]

        if list(df.columns) != before_cols:
            raise AssertionError(f"Column order changed for {mcqa_path}")
        df.to_csv(out_path, index=False)
        validation_failures.extend(validate_file(mcqa_path, out_path, relevant))

        reviewed = len(relevant)
        summary_rows.append(
            {
                "domain_group": domain_group,
                "document": document,
                "total_v14_mcqa_rows": len(df),
                "shortened_rows_matched_for_review": reviewed,
                "accepted_repairs": accepted,
                "rejected_kept_old_rows": rejected,
                "question_formatted_rows": len(df),
                "non_shortened_question_formatted_rows": len(df) - reviewed,
            }
        )

    validation_columns = ["path", "row", "status"]
    return pd.DataFrame(summary_rows), pd.DataFrame(validation_failures, columns=validation_columns)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--review-only",
        action="store_true",
        help="Only write the review input CSV; do not repair or write v15 files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    matches = iter_matches(DOMAIN_GROUPS)
    review_input = build_review_input(matches)
    review_input.to_csv(REVIEW_INPUT, index=False)

    by_group = review_input.groupby("domain_group").size().to_dict()
    expected = {"medical": 70, "legal": 81, "arxiv": 72}
    if by_group != expected:
        raise AssertionError(f"Unexpected matched row counts: {by_group} != {expected}")

    if args.review_only:
        print(f"Wrote {REVIEW_INPUT.relative_to(REPO_ROOT)}")
        print(f"Matched shortened MCQA rows: {by_group}")
        return

    decisions = build_decisions(matches)
    decisions.to_csv(AGENT_DECISIONS, index=False)
    accepted = decisions[decisions["decision"] == "ACCEPT"].copy()
    rejected = decisions[decisions["decision"] != "ACCEPT"].copy()
    accepted.to_csv(ACCEPTED_REPAIRS, index=False)
    rejected.to_csv(REJECTED_REPAIRS, index=False)

    summary, validation = write_outputs(matches, decisions)
    summary.to_csv(SUMMARY_PATH, index=False)
    validation.to_csv(VALIDATION_PATH, index=False)

    print(f"Review rows: {len(review_input)} {by_group}")
    print(f"Accepted repairs: {len(accepted)}")
    print(f"Rejected/kept old: {len(rejected)}")
    print(f"Wrote {SUMMARY_PATH.relative_to(REPO_ROOT)}")
    print(f"Wrote {VALIDATION_PATH.relative_to(REPO_ROOT)}")
    if not validation.empty:
        raise AssertionError(f"Validation failures: {len(validation)}")


if __name__ == "__main__":
    main()
