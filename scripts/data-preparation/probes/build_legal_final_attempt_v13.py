#!/usr/bin/env python3
"""Build the final legal inference MCQA v13 recovery attempt artifacts."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT))

from utils.mcqa_prompts import build_mcqa_5shot_prompt  # noqa: E402

LEGAL_DIR = ROOT / "probes" / "legal"
REPORT_DIR = ROOT / "reports" / "legal_mcqa_recovery_v12"
FINAL_DIR = REPORT_DIR / "final_attempt"
BROADER_DIR = REPORT_DIR / "broader_review"

CLOZE_COLS = [
    "original_row_index",
    "target",
    "probe",
    "fact",
    "inference_type",
    "source_fact(s)",
    "derivation",
    "question",
    "answer",
    "source_facts",
    "text_sentences",
]

MCQA_COLS = [
    "probe",
    "target",
    "correct_label",
    "formatted_question",
    "option_a",
    "option_b",
    "option_c",
    "option_d",
    "option_e",
    "distractors",
    "fact",
    "raw_knowledge_statement",
    "section",
    "inference_type",
    "source_fact(s)",
    "source_facts",
    "text_sentences",
    "derivation",
    "question",
    "answer",
    "formatted_question_5shot",
]

LABELS = ["(A)", "(B)", "(C)", "(D)", "(E)"]


def format_question(stem: str, options: list[str]) -> str:
    lines = [stem]
    for label, option in zip(LABELS, options, strict=True):
        lines.append(f"{label} {option}")
    return "\n".join(lines)


def join_fact(probe: str, target: str) -> str:
    if probe.endswith((" ", "\n")):
        return f"{probe}{target}"
    if probe.endswith(","):
        return f"{probe} {target}"
    return f"{probe} {target}"


def label_index(correct_label: str) -> int:
    return LABELS.index(correct_label)


def get_v11_row(domain: str, row_index: int) -> pd.Series:
    path = LEGAL_DIR / domain / "inference" / "probes_v11_reviewed.csv"
    return pd.read_csv(path).iloc[row_index]


def base_meta(domain: str, row_index: int) -> dict[str, object]:
    row = get_v11_row(domain, row_index)
    return {
        "domain": domain,
        "v11_reviewed_row_index_0based": row_index,
        "original_row_index": row.get("original_row_index", ""),
        "original_probe": row["probe"],
        "original_target": row["target"],
        "original_fact": row["fact"],
        "source_fact(s)": row["source_fact(s)"],
        "source_facts": row.get("source_facts", row["source_fact(s)"]),
        "text_sentences": row.get("text_sentences", row["source_fact(s)"]),
        "derivation": row["derivation"],
        "question": row["question"],
        "answer": row["answer"],
        "inference_type": row["inference_type"],
    }


def manual_mcqa(
    domain: str,
    row_index: int,
    stem: str,
    target: str,
    correct_label: str,
    options: list[str],
) -> dict[str, object]:
    row = get_v11_row(domain, row_index)
    correct = options[label_index(correct_label)]
    if correct != target:
        raise ValueError(f"{domain} row {row_index}: target does not match {correct_label}")
    formatted = format_question(stem, options)
    distractors = [option for option in options if option != target]
    return {
        "probe": stem,
        "target": target,
        "correct_label": correct_label,
        "formatted_question": formatted,
        "option_a": options[0],
        "option_b": options[1],
        "option_c": options[2],
        "option_d": options[3],
        "option_e": options[4],
        "distractors": json.dumps(distractors, ensure_ascii=False),
        "fact": join_fact(stem, target),
        "raw_knowledge_statement": "",
        "section": "",
        "inference_type": row["inference_type"],
        "source_fact(s)": row["source_fact(s)"],
        "source_facts": row.get("source_facts", row["source_fact(s)"]),
        "text_sentences": row.get("text_sentences", row["source_fact(s)"]),
        "derivation": row["derivation"],
        "question": row["question"],
        "answer": row["answer"],
        "formatted_question_5shot": build_mcqa_5shot_prompt(formatted),
    }


def all_remaining_metas() -> dict[tuple[str, int], dict[str, object]]:
    metas: dict[tuple[str, int], dict[str, object]] = {}

    kept = pd.read_csv(BROADER_DIR / "kept_broader_reviewed_candidates_mcqa.csv")
    dropped = pd.read_csv(BROADER_DIR / "dropped_broader_reviewed_candidates_mcqa.csv")
    hard = pd.read_csv(REPORT_DIR / "v11_reviewed_not_in_any_mcqa_candidate.csv")

    for source, frame in [("broader", kept), ("broader", dropped)]:
        for _, row in frame.iterrows():
            domain = row["domain"]
            row_index = int(row["source_row_0based"])
            meta = base_meta(domain, row_index)
            meta.update(
                {
                    "prior_failure_bucket": "broader_candidate",
                    "prior_prefilter_reason": row.get("review_reason", ""),
                    "existing_broader_candidate_probe": row.get("probe", ""),
                    "existing_broader_candidate_target": row.get("target", ""),
                    "existing_broader_candidate_formatted_question": row.get(
                        "formatted_question", ""
                    ),
                    "broader_row_index_0based": int(row["broader_row_index_0based"]),
                    "candidate_source": source,
                }
            )
            metas[(domain, row_index)] = meta

    for _, row in hard.iterrows():
        domain = row["domain"]
        row_index = int(row["v11_reviewed_row_index_0based"])
        meta = base_meta(domain, row_index)
        meta.update(
            {
                "prior_failure_bucket": row["bucket"],
                "prior_prefilter_reason": "",
                "existing_broader_candidate_probe": "",
                "existing_broader_candidate_target": "",
                "existing_broader_candidate_formatted_question": "",
                "broader_row_index_0based": "",
                "candidate_source": "hard_failure",
            }
        )
        metas[(domain, row_index)] = meta

    return metas


def broader_keep_successes() -> list[dict[str, object]]:
    kept = pd.read_csv(BROADER_DIR / "kept_broader_reviewed_candidates_mcqa.csv")
    successes: list[dict[str, object]] = []
    for _, row in kept.iterrows():
        domain = row["domain"]
        row_index = int(row["source_row_0based"])
        mcqa = {col: row[col] if col in row.index else "" for col in MCQA_COLS}
        mcqa["formatted_question_5shot"] = build_mcqa_5shot_prompt(mcqa["formatted_question"])
        successes.append(
            {
                "domain": domain,
                "v11_reviewed_row_index_0based": row_index,
                "broader_row_index_0based": int(row["broader_row_index_0based"]),
                "action": "kept_as_is",
                "decision": "accept",
                "issue_type": "none",
                "review_reason": row.get("review_reason", ""),
                "mcqa": mcqa,
            }
        )
    return successes


def repaired_successes() -> list[dict[str, object]]:
    specs = [
        {
            "domain": "America_First_Legal_Foundation_v_Jamieson_Greer",
            "row": 10,
            "broader": 1,
            "issue": "repaired_stem_leakage",
            "reason": "Removed the subject-matter-jurisdiction cue from the stem while preserving the reviewed disposition inference.",
            "stem": "According to the opinion 'America First Legal Foundation v. Jamieson Greer', once the court concluded AFL lacked Article III standing, the required instruction on remand was to",
            "target": "dismiss for lack of subject-matter jurisdiction",
            "label": "(D)",
            "options": [
                "dismiss for lack of personal jurisdiction",
                "dismiss for failure to state a claim",
                "enter summary judgment for the government",
                "dismiss for lack of subject-matter jurisdiction",
                "remand to OSC for further consideration",
            ],
        },
        {
            "domain": "Pacito_v_Trump",
            "row": 4,
            "broader": 1,
            "issue": "repaired_tautological_stem",
            "reason": "Reframed the stem around the APA/statutory-duty linkage instead of restating termination of services.",
            "stem": "According to the opinion 'Pacito v. Trump', when the stay panel relied on the APA and section 1522's reception-and-placement duties, the challenge that most directly supported the exception was an",
            "target": "APA challenge to ending resettlement services",
            "label": "(B)",
            "options": [
                "APA challenge to suspending refugee admissions before entry",
                "APA challenge to ending resettlement services",
                "APA challenge to revoking refugees' admission status after arrival",
                "APA challenge to denying case-by-case refugee exemptions under the order",
                "APA challenge to ending overseas refugee processing and vetting",
            ],
        },
        {
            "domain": "Pacito_v_Trump",
            "row": 3,
            "broader": 2,
            "issue": "repaired_stem_leakage",
            "reason": "Removed the quoted 'after their admission' cue and asked for the legal status separating the two regimes.",
            "stem": "According to the opinion 'Pacito v. Trump', the legal status separating refugees covered only by the suspended USRAP-entry program from refugees covered by section 1522 services was",
            "target": "admission into the United States",
            "label": "(B)",
            "options": [
                "entry into the United States under the USRAP",
                "admission into the United States",
                "designation as a refugee eligible for the U.S. Refugee Admissions Program",
                "completion of federal reception and placement processing",
                "the President's further findings permitting refugee admissions to resume",
            ],
        },
        {
            "domain": "Santos_v_Kimmel",
            "row": 2,
            "broader": 0,
            "issue": "repaired_weak_options",
            "reason": "Rebuilt the options as distinct contract-status theories rather than enforceability variants.",
            "stem": "According to the opinion 'Santos v. Kimmel', under Illinois's presumption against nonparty contract rights and the absence of express third-party language, Santos's direct breach theory failed because he",
            "target": "cannot enforce the Terms of Service",
            "label": "(D)",
            "options": [
                "was an assignee of Cameo's contractual rights",
                "was a party to the Terms of Service",
                "could enforce the Terms as an intended third-party beneficiary",
                "cannot enforce the Terms of Service",
                "was covered by an express declaration in the relevant provisions",
            ],
        },
        {
            "domain": "United_States_v_Justin_Cutbank",
            "row": 2,
            "broader": 2,
            "issue": "repaired_near_binary_stem",
            "reason": "Removed the 'conviction would still stand' wording and asked for the appellate consequence under the harmless-error record.",
            "stem": "According to the opinion 'United States v. Justin Cutbank', given substantial evidence and only slight influence from any evidentiary errors, the appellate consequence for the conviction was",
            "target": "no reversal",
            "label": "(A)",
            "options": [
                "no reversal",
                "reversal because the jury was substantially swayed",
                "a new trial on cumulative-error grounds",
                "remand for reconsideration of the evidentiary rulings",
                "vacatur of the sentence only",
            ],
        },
        {
            "domain": "Williams_v_GoAuto_Insurance",
            "row": 10,
            "broader": 3,
            "issue": "repaired_stem_leakage",
            "reason": "Removed the cue that the company itself was the certifying actor and asked for the notice requirement the court rejected.",
            "stem": "According to the opinion 'Williams v. GoAuto Insurance,' from the statute's wording and the lack of clearer Louisiana authority, the court inferred that APAC's certification notice needed",
            "target": "no named employee required",
            "label": "(A)",
            "options": [
                "no named employee required",
                "a personal signature from the employee who certified the notice",
                "the printed name of APAC's mailing employee",
                "a sworn affidavit from a company representative",
                "identification of the employee who sent the cancellation request",
            ],
        },
    ]
    successes = []
    for spec in specs:
        mcqa = manual_mcqa(
            spec["domain"],
            spec["row"],
            spec["stem"],
            spec["target"],
            spec["label"],
            spec["options"],
        )
        successes.append(
            {
                "domain": spec["domain"],
                "v11_reviewed_row_index_0based": spec["row"],
                "broader_row_index_0based": spec["broader"],
                "action": "repaired_from_broader_candidate",
                "decision": "accept",
                "issue_type": spec["issue"],
                "review_reason": spec["reason"],
                "mcqa": mcqa,
            }
        )
    return successes


def hard_successes() -> list[dict[str, object]]:
    specs = [
        {
            "domain": "Apex_Bank_v_Cc_Serve_Corp",
            "row": 3,
            "stem": "According to the opinion 'Apex Bank v. CC Serve Corp.', after vacating the Board's findings on two DuPont factors, the ultimate trademark issue left for reassessment was",
            "target": "likelihood of confusion",
            "label": "(A)",
            "options": [
                "likelihood of confusion",
                "abandonment of the cited mark",
                "priority of use",
                "genericness of the applied-for mark",
                "acquired distinctiveness",
            ],
        },
        {
            "domain": "Bruce_Cohen_v_Consilio_LLC",
            "row": 0,
            "stem": "According to the opinion 'Bruce Cohen v. Consilio, LLC', after the court vacated one summary-judgment ruling and otherwise affirmed, the only claim returned for further proceedings was the",
            "target": "MFLSA claim",
            "label": "(B)",
            "options": [
                "MWTA claim",
                "MFLSA claim",
                "FLSA overtime claim",
                "request to withdraw the 2019 policy",
                "unjust-enrichment claim",
            ],
        },
        {
            "domain": "Finesse_Wireless_LLC_v_Att_Mobility_LLC",
            "row": 5,
            "stem": "According to the opinion 'Finesse Wireless LLC v. AT&T Mobility LLC', under the accepted construction requiring separately identifiable rather than unique S1, S2, and S3 signals, the claim could be satisfied with",
            "target": "two unique signals",
            "label": "(B)",
            "options": [
                "one unique signal",
                "two unique signals",
                "three unique signals",
                "four unique signals",
                "no input signals",
            ],
        },
        {
            "domain": "Jimenez_v_Bondi",
            "row": 6,
            "stem": "According to the opinion 'Jimenez v. Bondi', after the threats escalated around the property dispute and accusations about prosecutor-related searches, the court treated the predominant motive as",
            "target": "Jimenez's involvement with Don Rafa",
            "label": "(C)",
            "options": [
                "Petitioners' political opinion",
                "Jimenez's membership in a national police unit",
                "Jimenez's involvement with Don Rafa",
                "a family-based protected ground",
                "general criminal extortion unrelated to Jimenez",
            ],
        },
        {
            "domain": "Jimenez_v_Bondi",
            "row": 8,
            "stem": "According to the opinion 'Jimenez v. Bondi', because Petitioners made minimal reports to authorities, the missing predicate for proving government inability or unwillingness was",
            "target": "notice and opportunity",
            "label": "(D)",
            "options": [
                "proof of countrywide relocation",
                "a formal asylum interview",
                "corroboration from Don Rafa",
                "notice and opportunity",
                "a showing of changed country conditions",
            ],
        },
        {
            "domain": "Pacito_v_Trump",
            "row": 7,
            "stem": "According to the opinion 'Pacito v. Trump', because section 1522 duties depend on funding and the panel identified FY 2025 refugee-assistance funds, the panel could treat the funding condition as satisfied because",
            "target": "available appropriations existed",
            "label": "(E)",
            "options": [
                "the President had waived all funding limits",
                "private resettlement agencies had agreed to pay the costs",
                "Congress had repealed the funding condition",
                "the agencies had already spent the full migration budget",
                "available appropriations existed",
            ],
        },
        {
            "domain": "Santos_v_Kimmel",
            "row": 5,
            "stem": "According to the opinion 'Santos v. Kimmel', on Rule 12(b)(6) review, the court determined the videos' publication dates by taking",
            "target": "judicial notice of registrations",
            "label": "(C)",
            "options": [
                "testimony from the videos' viewers",
                "discovery from YouTube's internal records",
                "judicial notice of registrations",
                "an evidentiary hearing on publication",
                "Santos's post-appeal declaration",
            ],
        },
        {
            "domain": "United_States_v_Jaison_Coleman",
            "row": 2,
            "stem": "According to the opinion 'United States v. Jaison Coleman', because the welfare check required locating everyone in the home and confirming Coleman posed no danger, finding him in a separate bedroom during the canvas was",
            "target": "within the scope of consent",
            "label": "(A)",
            "options": [
                "within the scope of consent",
                "a protective sweep unsupported by consent",
                "a search requiring a warrant",
                "an arrest requiring probable cause",
                "outside the purpose of the welfare check",
            ],
        },
        {
            "domain": "United_States_v_Justin_Cutbank",
            "row": 1,
            "stem": "According to the opinion 'United States v. Justin Cutbank', prior sightings of Cutbank with the same sawed-off rifle supported a non-propensity inference about his",
            "target": "ownership or control",
            "label": "(B)",
            "options": [
                "motive to threaten D.F.",
                "ownership or control",
                "intent to flee from police",
                "knowledge of the rifle's serial number",
                "membership in a prior conspiracy",
            ],
        },
        {
            "domain": "Williams_v_GoAuto_Insurance",
            "row": 4,
            "stem": "According to the opinion 'Williams v. GoAuto Insurance,' waiting until after midnight on day ten before sending the cancellation request directly served the statutory function of protecting the insured's",
            "target": "minimum time to cure",
            "label": "(D)",
            "options": [
                "right to a jury trial",
                "power to revoke APAC's agency status",
                "right to a premium refund",
                "minimum time to cure",
                "ability to demand a private carrier",
            ],
        },
    ]
    successes = []
    for spec in specs:
        mcqa = manual_mcqa(
            spec["domain"],
            spec["row"],
            spec["stem"],
            spec["target"],
            spec["label"],
            spec["options"],
        )
        successes.append(
            {
                "domain": spec["domain"],
                "v11_reviewed_row_index_0based": spec["row"],
                "broader_row_index_0based": "",
                "action": "manually_authored_from_hard_failure",
                "decision": "accept",
                "issue_type": "none",
                "review_reason": "Manually authored from the reviewed cloze probe and source facts; five-option space is source-supported and non-leaky.",
                "mcqa": mcqa,
            }
        )
    return successes


def final_drops() -> list[dict[str, object]]:
    return [
        {
            "domain": "United_States_v_Justin_Cutbank",
            "row": 4,
            "broader": 0,
            "issue": "trivial_arithmetic_rule",
            "reason": "The greater-offense-level rule plus the two numbers makes the answer a tautological selection, and repair would not add a meaningful nontrivial option space.",
        },
        {
            "domain": "United_States_v_Justin_Cutbank",
            "row": 5,
            "broader": 3,
            "issue": "unsupported_or_ambiguous_theme",
            "reason": "The proposed common aggravating theme is not cleanly supported by the cited offense-conduct facts, and several distractor themes remain plausibly correct.",
        },
        {
            "domain": "Williams_v_GoAuto_Insurance",
            "row": 2,
            "broader": 1,
            "issue": "near_binary_trivial",
            "reason": "The automatic midnight trigger still collapses to the two statutory functions after repair, and the original target is cued by the timing language.",
        },
        {
            "domain": "Williams_v_GoAuto_Insurance",
            "row": 5,
            "broader": 2,
            "issue": "near_binary_trivial",
            "reason": "The strict-adherence premise and omitted certification statements leave only effective versus ineffective cancellation, so the row remains too obvious.",
        },
    ]


def build_review_input(metas: dict[tuple[str, int], dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame([metas[key] for key in sorted(metas)])


def cloze_row_for_success(success: dict[str, object]) -> dict[str, object]:
    row = get_v11_row(
        str(success["domain"]),
        int(success["v11_reviewed_row_index_0based"]),
    )
    return {col: row[col] if col in row.index else "" for col in CLOZE_COLS}


def write_readable(path: Path, df: pd.DataFrame) -> None:
    lines = []
    for i, row in df.iterrows():
        lines.append(f"## {i}")
        lines.append(f"Probe: {row['probe']}")
        lines.append(f"Target: {row['target']}")
        lines.append(f"Fact: {row['fact']}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def validate_v13(domains: list[str]) -> pd.DataFrame:
    records = []
    for domain in domains:
        path = LEGAL_DIR / domain / "inference" / "probes_v13_mcqa.csv"
        mcqa = pd.read_csv(path)
        cloze = pd.read_csv(LEGAL_DIR / domain / "inference" / "probes_v13.csv")
        dupes = int(mcqa.duplicated(subset=["probe", "target"]).sum())
        cloze_dupes = int(cloze.duplicated(subset=["probe", "target"]).sum())
        leaks = 0
        bad_option_count = 0
        bad_correct = 0
        null_5shot = int(
            mcqa["formatted_question_5shot"].isna().sum()
            + (mcqa["formatted_question_5shot"].astype(str).str.len() == 0).sum()
        )
        for _, row in mcqa.iterrows():
            target = str(row["target"]).strip().lower()
            stem = str(row["formatted_question"]).split("\n(A)", 1)[0].lower()
            if target and target in stem:
                leaks += 1
            options = [str(row[f"option_{c}"]).strip() for c in "abcde"]
            if len(options) != 5 or any(not option for option in options):
                bad_option_count += 1
            correct_label = str(row["correct_label"]).strip()
            if correct_label not in LABELS:
                bad_correct += 1
            else:
                correct = options[label_index(correct_label)]
                if correct != str(row["target"]).strip():
                    bad_correct += 1
        records.append(
            {
                "domain": domain,
                "v13_mcqa_rows": len(mcqa),
                "v13_cloze_rows": len(cloze),
                "duplicate_keys": dupes,
                "cloze_duplicate_keys": cloze_dupes,
                "target_leaks": leaks,
                "null_5shot": null_5shot,
                "bad_option_count": bad_option_count,
                "bad_correct_label": bad_correct,
            }
        )
    return pd.DataFrame(records)


def write_final_report(successes: list[dict[str, object]], drops: list[dict[str, object]]) -> None:
    lines = [
        "# Final Legal Inference MCQA Recovery Attempt",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        f"- Reviewed remaining rows: {len(successes) + len(drops)}",
        f"- Accepted into v13: {len(successes)}",
        f"- Dropped after final repair attempt: {len(drops)}",
        "",
        "## Successful Recoveries",
        "",
    ]
    for success in successes:
        meta = base_meta(
            str(success["domain"]),
            int(success["v11_reviewed_row_index_0based"]),
        )
        mcqa = success["mcqa"]
        lines.extend(
            [
                f"### {success['domain']} row {success['v11_reviewed_row_index_0based']}",
                "",
                f"- Action: {success['action']}",
                f"- Original cloze: {meta['original_probe']} {str(meta['original_target']).strip()}",
                f"- Correct label: {mcqa['correct_label']}",
                "",
                "```text",
                str(mcqa["formatted_question"]),
                "```",
                "",
            ]
        )
    lines.extend(["## Final Drops", ""])
    for drop in drops:
        meta = base_meta(str(drop["domain"]), int(drop["row"]))
        lines.extend(
            [
                f"### {drop['domain']} row {drop['row']}",
                "",
                f"- Issue: {drop['issue']}",
                f"- Reason: {drop['reason']}",
                f"- Original cloze: {meta['original_probe']} {str(meta['original_target']).strip()}",
                "",
            ]
        )
    (FINAL_DIR / "final_recovery_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    metas = all_remaining_metas()
    review_input = build_review_input(metas)
    review_input.to_csv(FINAL_DIR / "review_input_final_attempt.csv", index=False)

    successes = broader_keep_successes() + repaired_successes() + hard_successes()
    drops = final_drops()

    success_keys = {
        (str(s["domain"]), int(s["v11_reviewed_row_index_0based"])) for s in successes
    }
    drop_keys = {(str(d["domain"]), int(d["row"])) for d in drops}
    if len(success_keys) != len(successes):
        raise ValueError("Duplicate success keys")
    if success_keys & drop_keys:
        raise ValueError(f"Rows both accepted and dropped: {success_keys & drop_keys}")
    if success_keys | drop_keys != set(metas):
        missing = set(metas) - (success_keys | drop_keys)
        extra = (success_keys | drop_keys) - set(metas)
        raise ValueError(f"Decision coverage mismatch missing={missing} extra={extra}")

    decisions = []
    success_rows = []
    for success in successes:
        meta = metas[(str(success["domain"]), int(success["v11_reviewed_row_index_0based"]))]
        mcqa = success["mcqa"]
        record = {
            **{k: meta.get(k, "") for k in meta},
            "decision": "accept",
            "action": success["action"],
            "issue_type": success["issue_type"],
            "final_review_reason": success["review_reason"],
            "final_probe": mcqa["probe"],
            "final_target": mcqa["target"],
            "correct_label": mcqa["correct_label"],
            "formatted_question": mcqa["formatted_question"],
        }
        decisions.append(record)
        success_rows.append(
            {
                **record,
                **{f"mcqa_{col}": mcqa[col] for col in MCQA_COLS},
            }
        )

    drop_rows = []
    for drop in drops:
        meta = metas[(str(drop["domain"]), int(drop["row"]))]
        record = {
            **{k: meta.get(k, "") for k in meta},
            "decision": "drop",
            "action": "drop_after_final_attempt",
            "issue_type": drop["issue"],
            "final_review_reason": drop["reason"],
            "final_probe": "",
            "final_target": "",
            "correct_label": "",
            "formatted_question": "",
        }
        decisions.append(record)
        drop_rows.append(record)

    pd.DataFrame(decisions).sort_values(
        ["domain", "v11_reviewed_row_index_0based"]
    ).to_csv(FINAL_DIR / "final_attempt_decisions.csv", index=False)
    pd.DataFrame(success_rows).sort_values(
        ["domain", "v11_reviewed_row_index_0based"]
    ).to_csv(FINAL_DIR / "final_attempt_successes.csv", index=False)
    pd.DataFrame(drop_rows).sort_values(
        ["domain", "v11_reviewed_row_index_0based"]
    ).to_csv(FINAL_DIR / "final_attempt_drops.csv", index=False)

    domains = sorted(path.name for path in LEGAL_DIR.iterdir() if path.is_dir())
    accepted_by_domain: dict[str, list[dict[str, object]]] = defaultdict(list)
    for success in successes:
        accepted_by_domain[str(success["domain"])].append(success)

    old_summary = pd.read_csv(REPORT_DIR / "v13_inference_mcqa_build_summary.csv")
    old_base = old_summary.set_index("domain")

    build_records = []
    for domain in domains:
        inf_dir = LEGAL_DIR / domain / "inference"
        cloze_path = inf_dir / "probes_v13.csv"
        mcqa_path = inf_dir / "probes_v13_mcqa.csv"

        cloze = pd.read_csv(cloze_path)
        mcqa = pd.read_csv(mcqa_path)

        remove_cloze_keys = {
            (
                str(get_v11_row(d, idx)["probe"]),
                str(get_v11_row(d, idx)["target"]),
            )
            for d, idx in set(metas)
            if d == domain
        }
        cloze = cloze[
            ~cloze.apply(lambda r: (str(r["probe"]), str(r["target"])) in remove_cloze_keys, axis=1)
        ].copy()

        final_mcqa_for_domain = [s["mcqa"] for s in accepted_by_domain.get(domain, [])]
        remove_mcqa_keys = {
            (str(row["probe"]), str(row["target"])) for row in final_mcqa_for_domain
        }
        mcqa = mcqa[
            ~mcqa.apply(lambda r: (str(r["probe"]), str(r["target"])) in remove_mcqa_keys, axis=1)
        ].copy()

        append_cloze = [
            cloze_row_for_success(s)
            for s in accepted_by_domain.get(domain, [])
        ]
        append_mcqa = final_mcqa_for_domain
        if append_cloze:
            cloze = pd.concat([cloze, pd.DataFrame(append_cloze)], ignore_index=True)
        if append_mcqa:
            mcqa = pd.concat([mcqa, pd.DataFrame(append_mcqa)], ignore_index=True)

        cloze = cloze[CLOZE_COLS]
        mcqa = mcqa[MCQA_COLS]
        mcqa["formatted_question_5shot"] = mcqa["formatted_question"].apply(
            build_mcqa_5shot_prompt
        )

        cloze.to_csv(cloze_path, index=False)
        mcqa.to_csv(mcqa_path, index=False)
        write_readable(inf_dir / "probes_v13_readable.txt", cloze)

        accepted_counts = Counter(
            s["action"] for s in accepted_by_domain.get(domain, [])
        )
        base_reviewed = int(old_base.loc[domain, "base_v12_reviewed_mcqa"])
        safe = int(old_base.loc[domain, "safe_reviewed_recovered_mcqa"])
        metrics = [
            f"Legal inference MCQA v13 build - {domain}",
            "=" * 60,
            f"Built: {datetime.now().isoformat(timespec='seconds')}",
            f"Base reviewed v12 MCQA rows: {base_reviewed}",
            f"Safe reviewed recovered MCQA rows: {safe}",
            f"Final-attempt kept broader rows: {accepted_counts['kept_as_is']}",
            f"Final-attempt repaired broader rows: {accepted_counts['repaired_from_broader_candidate']}",
            f"Final-attempt manually authored hard-failure rows: {accepted_counts['manually_authored_from_hard_failure']}",
            f"Final v13 probes: {len(cloze)}",
            f"Final v13 MCQA probes: {len(mcqa)}",
            "Sources: probes_v12_reviewed + safe reviewed recovered candidates + final-attempt accepted rows.",
            "",
        ]
        (inf_dir / "mcqa_metrics_v13.txt").write_text("\n".join(metrics), encoding="utf-8")

        build_records.append(
            {
                "domain": domain,
                "base_v12_reviewed_mcqa": base_reviewed,
                "safe_reviewed_recovered_mcqa": safe,
                "final_attempt_kept_broader": accepted_counts["kept_as_is"],
                "final_attempt_repaired_broader": accepted_counts[
                    "repaired_from_broader_candidate"
                ],
                "final_attempt_manual_hard_failure": accepted_counts[
                    "manually_authored_from_hard_failure"
                ],
                "final_attempt_accepted_total": sum(accepted_counts.values()),
                "v13_mcqa_rows": len(mcqa),
            }
        )

    build_summary = pd.DataFrame(build_records)
    build_summary.to_csv(REPORT_DIR / "v13_inference_mcqa_build_summary.csv", index=False)
    build_summary.to_csv(FINAL_DIR / "v13_final_attempt_build_summary.csv", index=False)

    validation = validate_v13(domains)
    validation.to_csv(REPORT_DIR / "v13_inference_mcqa_validation.csv", index=False)
    validation.to_csv(FINAL_DIR / "v13_final_attempt_validation.csv", index=False)
    if validation[
        [
            "duplicate_keys",
            "cloze_duplicate_keys",
            "target_leaks",
            "null_5shot",
            "bad_option_count",
            "bad_correct_label",
        ]
    ].sum().sum():
        raise SystemExit(validation.to_string(index=False))

    write_final_report(successes, drops)

    print(f"Accepted final-attempt rows: {len(successes)}")
    print(f"Dropped final-attempt rows: {len(drops)}")
    print(f"Final v13 legal inference MCQA rows: {int(build_summary['v13_mcqa_rows'].sum())}")


if __name__ == "__main__":
    main()
