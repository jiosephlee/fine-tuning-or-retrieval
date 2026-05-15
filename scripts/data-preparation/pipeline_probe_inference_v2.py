import argparse
import concurrent.futures
import json
import os
import re
import sys
from typing import Any, Dict, List, Tuple

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)
import utils.utils as utils
from utils import probe_paths
from utils.pipeline import (
    check_tokenizer_consistency,
    is_text_in_document,
    process_papers,
    save_debug_file,
    save_df_for_debugging,
)
from utils.prompts.pipeline import (
    FACT_PROBE_CLOZE_PROMPT_SYSTEM_LEGAL,
    FACT_PROBE_CLOZE_PROMPT_SYSTEM_MEDICAL,
    FACT_PROBE_CLOZE_PROMPT_SYSTEM_TWO,
)


SUPPORTED_SOURCES = ("arxiv", "medical", "legal")
DEFAULT_OUTPUT_VERSION = "v9"


def _composition_prompt_arxiv() -> str:
    return r"""You will be given an academic paper. Your task is to create inference probes that test composition: whether a reader can combine facts stated in the paper and use logic alone to derive a new fact.

### Target Skill: Composition
Each item must require one of the following:
1. A step in logical reasoning: applying a lay analogy, mathematical reasoning, counterfactual reasoning, or other reasoning patterns onto a fact from the paper to reach a consequence.
2. Integration of multiple facts: combining two or more stated facts from different sentences, paragraphs, sections, tables, equations, or experiments.
3. Synthesizing knowledge into a different insight: restating what follows from the paper's facts in a way that is not directly stated in a single sentence.

### Hard Constraints
- The question must be answerable using ONLY the supplied paper text. Do not rely on background knowledge, related field knowledge, or assumptions not explicitly stated in the paper.
- The item must identify at least one source fact from the paper. Each source fact must be copied verbatim from the paper.
- The source fact must be sufficient to derive the answer by logic alone.
- The question should not be factual recall. It should not ask for a value, name, definition, or claim that is simply copied from one sentence.
- The answer should be a precise, concise word or phrase that is the conclusion of the composition step, not merely a quoted local clue. Keep the answer to 1-6 words whenever possible, and never use a full sentence or long clause.
- Do not invent compressed labels, umbrella terms, or new jargon for the answer. Avoid answers like "partition-function cancellation" or "optimization inefficiency" unless that exact concept is stated that way in the paper.
- Prefer answers that state the concrete inferred mechanism, condition, or consequence in plain terms, using the paper's own vocabulary where possible.
- The question should be self-contained and precise enough that there is one clear answer.
- Do not ask yes/no questions.
- Do not require external jargon or terminology beyond what the paper states and explains.

### Answer Style
- Bad: "partition-function cancellation"
- Better: "the partition function"
- Bad: "optimization inefficiency"
- Better: "PPO optimization"

### Good Composition Patterns
- Conceptual Synthesis: combine information across sections to ask for the central insight, mechanism, or consequence that follows from the paper's stated claims.
- Causal Mechanism: use experimental evidence, especially ablations or comparisons, to infer the source of an observed effect.
- Implicit Assumptions: ask what unstated condition or dependency must hold for the paper's stated claims or method to work, using only facts in the paper.
- Mathematical Understanding: ask what role a term, equation, constraint, or transformation plays in the paper's argument beyond its literal definition.
- Analogous Reasoning: ask the reader to apply a simple analogy on a fact from the paper to reach a consequence.
- Counterfactual Scenarios: ask what would logically change if a paper-stated condition, component, assumption, or design choice were removed or altered.

### Coverage and Non-Redundancy
- Generate probes that are as mutually exclusive and non-redundant as possible: avoid multiple questions that test the same inference, source fact, mechanism, equation role, ablation result, or design implication.
- Cover as much of the paper's important knowledge as possible across method, motivation, assumptions, equations, experiments, ablations, limitations, and conclusions.
- Prefer a broad set of distinct probes, rather than variations of the main ideas.  

### Output Format
Return JSON with a single key "qa_items", a list of dictionaries with:
- "inference_type": one of "Reasoning Step", "Multi-Fact Integration", or "Synthesized Insight"
- "source_fact(s)": list of verbatim strings copied from the paper
- "derivation": one concise sentence explaining how the source facts imply the answer, using no outside knowledge
- "question": string
- "answer": string
"""


def _composition_prompt_medical() -> str:
    return r"""You will be given a medical case report. Your task is to create inference probes that test composition: whether a reader can combine facts stated in the case report and use logic alone to derive a new clinical fact or conclusion from the report.

### Target Skill: Clinical Composition Without Outside Knowledge
Each item must require one of the following:
1. A step in logical reasoning: applying a lay analogy, clinical reasoning, counterfactual reasoning, or other reasoning patterns onto a fact from the case report to reach a consequence.
2. Integration of multiple facts: combining findings from presentation, investigations, differential diagnosis, treatment, outcome, or discussion.
3. Synthesizing knowledge into a different insight: deriving what follows from the case report's facts without importing medical knowledge not stated in the report.

### Hard Constraints
- The question must be answerable using ONLY the case report text. Do not rely on external medical knowledge, unstated pathophysiology, or general clinical assumptions.
- If medical background is needed, it must be explicitly stated in the report and included among the source facts.
- The item must identify at least one source fact from the report. Each source fact must be copied verbatim from the report.
- The source fact must be sufficient to derive the answer by logic alone.
- The question should not be factual recall. It should not ask for an isolated lab value, symptom, drug name, diagnosis, or definition copied from one sentence.
- The answer should be a precise, concise word or phrase that is the conclusion of the composition step, not merely a quoted local clue. Keep the answer to 1-6 words whenever possible, and never use a full sentence or long clause. Preserve units, abbreviations, day numbers, and qualifiers when relevant.
- Do not invent compressed labels, umbrella terms, or new medical-sounding jargon for the answer unless that exact concept is stated that way in the report.
- Prefer answers that state the concrete inferred diagnosis, mechanism, exclusion, treatment rationale, or consequence in plain terms, using the report's own vocabulary where possible.
- The question should be self-contained and precise enough that there is one clear answer.
- Do not ask yes/no questions.

### Answer Style
- Bad: "diagnostic convergence"
- Better: "excluded alternative causes"
- Bad: "treatment-response linkage"
- Better: "colchicine response"

### Good Composition Patterns
- Clinical Synthesis: combine presentation, investigations, treatment, and outcome to ask what clinical conclusion follows from the case report's stated facts.
- Causal Mechanism: use the report's stated temporal sequence, response to treatment, or exclusion of alternatives to infer what the report links to an observed change.
- Implicit Assumptions: ask what condition, exclusion, or diagnostic dependency must hold for the report's stated interpretation to work, using only facts in the report.
- Clinical/Quantitative Understanding: ask what role a lab pattern, imaging result, diagnostic criterion, timepoint, or treatment response plays in the case's reasoning beyond simple recall.
- Analogous Reasoning: ask the reader to apply a simple analogy on a fact from the case report to reach a consequence.
- Counterfactual Scenarios: ask what would logically change if a report-stated finding, test result, treatment, diagnosis, or exclusion were removed or altered.

### Coverage and Non-Redundancy
- Generate probes that are as mutually exclusive and non-redundant as possible: avoid multiple questions that test the same inference, source fact, diagnostic step, treatment rationale, exclusion, complication, or outcome implication.
- Cover as much of the case report's important knowledge as possible across presentation, investigations, differential diagnosis, treatment, outcome, discussion, limitations, and clinical lessons.
- Prefer a broad set of distinct probes, rather than variations of the main ideas.  

### Output Format
Return JSON with a single key "qa_items", a list of dictionaries with:
- "inference_type": one of "Reasoning Step", "Multi-Fact Integration", or "Synthesized Insight"
- "source_fact(s)": list of verbatim strings copied from the report
- "derivation": one concise sentence explaining how the source facts imply the answer, using no outside knowledge
- "question": string
- "answer": string
"""


def _composition_prompt_legal() -> str:
    return r"""You will be given a legal opinion or case text. Your task is to create inference probes that test composition: whether a reader can combine facts stated in the opinion and use logic alone to derive a new legal, procedural, or case-specific conclusion from the opinion.

### Target Skill: Legal Composition Without Outside Knowledge
Each item must require one of the following:
1. A step in logical reasoning: applying a lay analogy, legal reasoning, counterfactual reasoning, or other reasoning patterns onto a fact from the opinion to reach a consequence.
2. Integration of multiple facts: combining procedural posture, facts, legal standards, factor analysis, precedent discussion, and disposition.
3. Synthesizing knowledge into a different insight: deriving what follows from the opinion's stated reasoning without importing legal doctrine not stated in the opinion.

### Hard Constraints
- The question must be answerable using ONLY the opinion text. Do not rely on external legal knowledge, unstated doctrine, or assumptions about the law.
- If a legal rule or standard is needed, it must be explicitly stated in the opinion and included among the source facts.
- The item must identify at least one source fact from the opinion. Each source fact must be copied verbatim from the opinion.
- The source fact must be sufficient to derive the answer by logic alone.
- The question should not be factual recall. It should not ask for an isolated party name, date, court, statute, holding, or standard copied from one sentence.
- The answer should be a precise, concise word or phrase that is the conclusion of the composition step, not merely a quoted local clue. Keep the answer to 1-6 words whenever possible, and never use a full sentence or long clause. Preserve party names, statute sections, doctrine labels, citations, and qualifiers when relevant.
- Do not invent compressed labels, umbrella terms, or new legal-sounding jargon for the answer unless that exact concept is stated that way in the opinion.
- Prefer answers that state the concrete inferred rule application, procedural consequence, factor relationship, or holding implication in plain terms, using the opinion's own vocabulary where possible.
- The question should be self-contained and precise enough that there is one clear answer.
- Do not ask yes/no questions.

### Answer Style
- Bad: "factor-scope inconsistency"
- Better: "narrower similarity scope"
- Bad: "remand dependency"
- Better: "first-factor reconsideration"

### Good Composition Patterns
- Legal Synthesis: combine facts, procedural posture, legal standards, reasoning, and disposition to ask what legal or case-specific conclusion follows from the opinion.
- Causal Mechanism: use the court's stated reasoning, factor analysis, or treatment of evidence to infer why a particular outcome or remand instruction follows.
- Implicit Assumptions: ask what condition, scope, standard, or dependency must hold for the opinion's stated reasoning to work, using only facts in the opinion.
- Rule/Factor Understanding: ask what role a rule, standard of review, statutory phrase, precedent, or analytical factor plays in the court's reasoning beyond simple recall.
- Analogous Reasoning: ask the reader to apply a simple analogy on a fact from the opinion to reach a consequence.
- Counterfactual Scenarios: ask what would logically change if an opinion-stated fact, procedural posture, legal standard, factor finding, or precedent distinction were removed or altered.

### Coverage and Non-Redundancy
- Generate probes that are as mutually exclusive and non-redundant as possible: avoid multiple questions that test the same inference, source fact, rule, factor, procedural step, holding rationale, or disposition implication.
- Cover as much of the opinion's important knowledge as possible across facts, procedural posture, issues, legal standards, factor analysis, precedent use, reasoning, disposition, and remand instructions.
- Prefer a broad set of distinct probes, rather than variations of the main ideas.  

### Output Format
Return JSON with a single key "qa_items", a list of dictionaries with:
- "inference_type": one of "Reasoning Step", "Multi-Fact Integration", or "Synthesized Insight"
- "source_fact(s)": list of verbatim strings copied from the opinion
- "derivation": one concise sentence explaining how the source facts imply the answer, using no outside knowledge
- "question": string
- "answer": string
"""


QUESTION_PROMPTS = {
    "arxiv": _composition_prompt_arxiv,
    "medical": _composition_prompt_medical,
    "legal": _composition_prompt_legal,
}

CLOZE_PROMPTS = {
    "arxiv": FACT_PROBE_CLOZE_PROMPT_SYSTEM_TWO,
    "medical": FACT_PROBE_CLOZE_PROMPT_SYSTEM_MEDICAL,
    "legal": FACT_PROBE_CLOZE_PROMPT_SYSTEM_LEGAL,
}


def parse_arxiv_structure(text: str) -> pd.DataFrame:
    """Parse a LaTeX paper into sections, subsections, and paragraphs."""
    sections = []
    section_pattern = r'\\section\{([^}]+)\}'
    section_splits = re.split(section_pattern, text)

    current_section = "Title/Abstract"
    current_section_content = ""

    for i in range(len(section_splits)):
        if i == 0:
            content = section_splits[i]
            current_section_content = content
        elif i % 2 == 1:
            current_section = section_splits[i]
            continue
        else:
            content = section_splits[i]
            current_section_content = content

        subsection_pattern = r'\\subsection\{([^}]+)\}'
        subsection_splits = re.split(subsection_pattern, content)

        current_subsection = "No Subsection"
        current_subsection_content = ""

        for j in range(len(subsection_splits)):
            if j == 0:
                subsection_content = subsection_splits[j]
                current_subsection_content = subsection_content
            elif j % 2 == 1:
                current_subsection = subsection_splits[j]
                continue
            else:
                subsection_content = subsection_splits[j]
                current_subsection_content = subsection_content

            paragraphs = [p.strip() for p in subsection_content.split('\n\n') if p.strip()]
            for paragraph in paragraphs:
                sections.append({
                    'section': current_section,
                    'subsection': current_subsection,
                    'paragraph': paragraph,
                    'section_text': current_section_content,
                    'subsection_text': current_subsection_content,
                })

    return pd.DataFrame(sections)


def extract_title(document_name: str, content: str, source: str) -> str:
    if source == "arxiv":
        match = re.search(r'\\title\{(.*?)\}', content, flags=re.DOTALL)
        if match:
            return re.sub(r'\s+', ' ', match.group(1)).strip()

    for line in content.splitlines()[:20]:
        if line.startswith("Title:"):
            return line.split("Title:", 1)[1].strip()

    return document_name


def split_paragraphs_balanced(text: str, max_chunks: int = 2) -> List[str]:
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    if len(paragraphs) <= 1 or max_chunks <= 1:
        return [text]

    chunk_count = min(max_chunks, len(paragraphs))
    chunks = []
    for chunk_index in range(chunk_count):
        start = round(chunk_index * len(paragraphs) / chunk_count)
        end = round((chunk_index + 1) * len(paragraphs) / chunk_count)
        chunks.append("\n\n".join(paragraphs[start:end]))

    return [chunk for chunk in chunks if chunk.strip()]


def build_document_chunks(
    document_content: str,
    source: str,
    add_on_mode: bool,
    generation_passes: int,
) -> Tuple[List[str], str, str]:
    return [document_content for _ in range(max(1, generation_passes))], 'gpt-5.4', 'medium'


def _normalize_generated_item(item: Dict[str, Any]) -> Dict[str, Any]:
    source_facts = (
        item.get("source_fact(s)")
        or item.get("source_facts")
        or item.get("text_sentences")
        or []
    )
    if not isinstance(source_facts, list):
        source_facts = [source_facts] if source_facts else []
    item["source_facts"] = source_facts
    item["source_fact(s)"] = source_facts
    # Preserve compatibility with older debug/evaluation code.
    item["text_sentences"] = source_facts
    return item


def generate_questions(
    text: str,
    source: str,
    model: str = 'gpt-5.4-mini',
    reasoning_effort: str = 'medium',
) -> List[Dict[str, Any]]:
    """Generate composition-focused inference questions from a document."""
    system_prompt = QUESTION_PROMPTS[source]()
    prompt = {'system': system_prompt, 'user': f"### Document Text\n\n{text}"}
    response_json = utils.query_llm(
        prompt,
        model=model,
        reasoning_effort=reasoning_effort,
        system_prompt_included=True,
        return_json=True,
        max_tokens=5000,
    )

    if isinstance(response_json, str):
        try:
            response_json = json.loads(response_json)
        except json.JSONDecodeError:
            print("Failed to parse JSON response from LLM.")
            return []
    if not isinstance(response_json, dict):
        print("Unexpected response format from LLM: response is not a JSON object.")
        return []

    questions = response_json.get('qa_items', [])
    if not isinstance(questions, list):
        print("Unexpected response format from LLM: 'qa_items' is not a list.")
        return []

    parsed_questions = []
    dropped_invalid_facts = 0
    dropped_missing_facts = 0
    dropped_missing_answer = 0

    for raw_item in questions:
        if not isinstance(raw_item, dict):
            continue
        item = _normalize_generated_item(raw_item)
        source_facts = item.get('source_facts', [])
        answer = str(item.get('answer', '')).strip()

        if len(source_facts) < 1:
            dropped_missing_facts += 1
            continue
        if not answer:
            dropped_missing_answer += 1
            continue
        if not all(is_text_in_document(fact, text, threshold=0.75) for fact in source_facts):
            dropped_invalid_facts += 1
            continue

        parsed_questions.append(item)

    if dropped_missing_facts:
        print(f"DROPPED {dropped_missing_facts} questions with no source facts.")
    if dropped_missing_answer:
        print(f"DROPPED {dropped_missing_answer} questions with missing answers.")
    if dropped_invalid_facts:
        print(f"DROPPED {dropped_invalid_facts} questions with invalid source facts.")

    return parsed_questions


def deduplicate_questions(questions: List[Dict[str, Any]], source: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Filter conceptually duplicate generated questions before cloze conversion."""
    if len(questions) <= 1:
        return questions, []

    document_label = {
        "arxiv": "paper",
        "medical": "case report",
        "legal": "opinion",
    }[source]

    candidates = []
    for idx, q in enumerate(questions):
        candidates.append({
            "index": idx,
            "inference_type": q.get("inference_type", ""),
            "question": q.get("question", ""),
            "answer": q.get("answer", ""),
            "derivation": q.get("derivation", ""),
            "source_facts": q.get("source_facts", []),
        })

    prompt = {
        "system": f"""You will be given candidate composition probes generated from the same {document_label}. Your task is to filter only clear conceptual duplicates.

Two probes are conceptual duplicates only when a reader who can answer one would answer the other by using the same source facts, the same reasoning step, and the same final conclusion. Similar topic area is not enough.

Do NOT drop a probe just because it shares a section, mechanism, equation, or experiment with another probe. Keep probes that test distinct implications, including:
- inverse or complementary implications of the same fact pattern;
- different consequences of the same mechanism;
- different preserved-vs-gained properties of the same design choice;
- different failure modes, costs, constraints, or capabilities that follow from nearby evidence.

When uncertain, keep both probes. It is better to keep a borderline distinct probe for manual review than to remove useful coverage too early.

Keep the strongest version only from each true duplicate cluster. Prefer probes that:
- require genuine composition or reasoning, not factual recall;
- have precise, concise answers;
- use source facts that are sufficient for the derivation;
- cover distinct parts of the {document_label}.

Do not force a target count. Drop only candidates that are clear conceptual duplicates of stronger candidates.

### Output Format
Return JSON with a single key "drop_indices", a list of integer candidate indices that should be dropped.
""",
        "user": json.dumps({"candidates": candidates}, ensure_ascii=False),
    }

    response = utils.query_llm(
        prompt,
        model='gpt-5.4-mini',
        reasoning_effort='medium',
        system_prompt_included=True,
        return_json=True,
        max_tokens=2000,
    )
    try:
        data = json.loads(response) if isinstance(response, str) else response
        drop_indices = data.get("drop_indices", [])
        drop_set = {
            int(i)
            for i in drop_indices
            if isinstance(i, int) or (isinstance(i, str) and i.isdigit())
        }
        drop_set = {idx for idx in drop_set if 0 <= idx < len(questions)}
        kept = [q for idx, q in enumerate(questions) if idx not in drop_set]
        dropped = [questions[idx] for idx in sorted(drop_set)]
        return kept, dropped
    except (json.JSONDecodeError, AttributeError, TypeError, ValueError):
        print("Failed to parse conceptual dedupe response. Keeping all generated questions.")
        return questions, []


def convert_to_cloze(question: Dict[str, Any], source: str) -> Tuple[str, str] | None:
    """Convert a question-answer pair to a cloze-style statement."""
    user_prompt = f"### Question\n{question['question']}\n\n### Answer\n{question['answer']}\n"
    cloze_prompt = {
        'system': CLOZE_PROMPTS[source],
        'user': user_prompt,
    }
    response = utils.query_llm(
        cloze_prompt,
        model='gpt-5.4',
        reasoning_effort='medium',
        system_prompt_included=True,
        return_json=True,
        max_tokens=1000,
    )
    try:
        data = json.loads(response) if isinstance(response, str) else response
        if not isinstance(data, dict):
            return None
        answer = data.get('answer')
        statement = data.get('statement')
        if answer is not None and statement is not None:
            return (answer, statement)
    except (json.JSONDecodeError, AttributeError):
        print("Failed to parse JSON response for cloze conversion.")
    return None


def quality_control_cloze(
    cloze_pair: Tuple[str, str],
    title: str,
    context: str,
    source: str,
) -> Tuple[str, str] | None:
    """Review and refine a cloze statement."""
    document_label = {
        "arxiv": "academic paper",
        "medical": "medical case report",
        "legal": "legal opinion",
    }[source]
    intro_template = {
        "arxiv": '"In the paper \'...\'" or "According to the paper \'...\'"',
        "medical": '"In the case report \'...\'" or "According to the case report \'...\'"',
        "legal": '"In the opinion \'...\'" or "According to the opinion \'...\'"',
    }[source]

    quality_control_prompt = {
        'system': rf"""Your task is to review and refine an inference question that has been rephrased as a cloze statement with the answer at the end. It has been extracted from a {document_label}. You will be given an '(answer, statement)' pair and supporting context from the source document.

### Quality Control Checklist

1. Formatting
- For academic papers, mathematical expressions and notation must be written in LaTeX, enclosed in '$' or '$$' delimiters.
- For medical and legal documents, preserve source formatting for units, abbreviations, citations, statute sections, dates, party names, and other domain-specific tokens.
- Do not use unicode mathematical symbols when LaTeX notation is appropriate.

2. Source Framing
- Start the statement with one of the following templates if it fits naturally: {intro_template}.
- Include the title when doing so.

3. Answer Placement
- The answer must appear at the very end of the statement.
- Do not add "___" at the end.

4. Answer Leakage
- The answer must not be leaked, explicitly or implicitly, in the statement until the very end.
- The answer should appear only once.

5. Answer Capitalization
- The answer should only be capitalized if it is a proper noun or if the source formatting requires it.

Make the smallest changes needed. If the pair is already good, return it unchanged.

### Output Format
Provide a JSON object with a single key "pair", which is the refined [answer, statement] pair.
""",
        'user': f"### Supporting Context\n{context}\n\n### Title\n{title}\n\n### Cloze Pair\n{json.dumps(cloze_pair)}\n",
    }
    response = utils.query_llm(
        quality_control_prompt,
        model='gpt-5.4-mini',
        reasoning_effort='high',
        system_prompt_included=True,
        return_json=True,
        max_tokens=1000,
    )
    try:
        data = json.loads(response) if isinstance(response, str) else response
        pair = data.get('pair')
        if isinstance(pair, list) and len(pair) == 2:
            return tuple(pair)
    except (json.JSONDecodeError, AttributeError):
        print("Failed to parse JSON response for QC.")
    return None


def filter_cloze_pair(cloze_pair: Tuple[str, str]) -> bool:
    """Decide whether to keep a refined pair based on a strict checklist."""
    if not cloze_pair:
        return False

    answer, statement = cloze_pair
    prompt = {
        'system': """Your task is to determine if a given (answer, statement) pair meets quality standards by acting as a filter.

### Quality Control Checklist
1. Linguistically Reasonable: The answer should fit naturally into the fill-in-the-blank statement.
2. Semantically Reasonable: The answer should be the clear, unambiguous completion of the statement.
3. Clear and Understandable: The statement should clearly build up to the answer.
4. No Answer Leakage: The statement should not reveal the answer before the final blank.
5. Precise Answer, Not Jargon: The answer should be a concrete mechanism, condition, or consequence. Drop answers that are newly coined compressed labels, umbrella terms, or vague abstractions, such as "partition-function cancellation", "optimization inefficiency", "diagnostic convergence", or "factor-scope inconsistency", unless the source document itself uses that exact term.
6. Concise Answer: The answer should usually be 1-6 words. Drop answers that are full sentences or long clauses unless every word is necessary for an unambiguous completion.
7. Declarative Cloze Shape: The statement must be a declarative cloze, not a question. Drop statements that contain a question mark or unresolved blank marker such as "___" or "____". The answer should complete the statement as its final span.

### Action
Based on the checklist, decide if the pair should be kept. Drop the pair if it fails any checklist item.

### Output Format
Provide your decision as a JSON object with a single boolean key: {"keep": true} or {"keep": false}.""",
        'user': f"""### Answer\n{answer}\n\n### Statement\n{statement.replace(answer, '___')}""",
    }

    response = utils.query_llm(
        prompt,
        model='gpt-5.4-mini',
        reasoning_effort='medium',
        system_prompt_included=True,
        return_json=True,
    )
    try:
        parsed_response = json.loads(response) if isinstance(response, str) else response
        return parsed_response.get('keep', False)
    except (json.JSONDecodeError, TypeError):
        print(f"Lost pair in filter step - JSON parse error: {response}")
        return False


def is_question_shaped_probe(probe: str) -> bool:
    """Return True when a cloze prefix is still phrased like a question."""
    normalized = " ".join(str(probe).strip().split()).lower()
    if not normalized:
        return True
    if "___" in normalized:
        return True
    normalized_without_titles = re.sub(r"'[^']*'", "''", normalized)
    if normalized_without_titles.rstrip().endswith("?"):
        return True
    question_starts = (
        "according to the paper, what ",
        "according to the paper, which ",
        "according to the paper, why ",
        "according to the paper, how ",
        "according to the paper, when ",
        "according to the paper, where ",
        "according to the paper, who ",
        "according to the paper, whose ",
        "according to the case report, what ",
        "according to the case report, which ",
        "according to the case report, why ",
        "according to the case report, how ",
        "according to the opinion, what ",
        "according to the opinion, which ",
        "according to the opinion, why ",
        "according to the opinion, how ",
        "in the paper, what ",
        "in the paper, which ",
        "in the paper, why ",
        "in the paper, how ",
        "what ",
        "which ",
        "why ",
        "how ",
        "when ",
        "where ",
        "who ",
        "whose ",
    )
    return normalized.startswith(question_starts)


def completed_cloze_ends_with_target(probe: str, target: str, fact: str) -> bool:
    """Check that the stored cloze reconstructs the fact and ends with the target."""
    clean_probe = " ".join(str(probe).strip().split())
    clean_target = " ".join(str(target).strip().split())
    clean_fact = " ".join(str(fact).strip().rstrip(".!?").split())
    if not clean_probe or not clean_target or not clean_fact:
        return False
    completed = f"{clean_probe} {clean_target}".strip()
    return clean_fact == completed and completed.endswith(clean_target)


def create_cloze_probe(
    refined_cloze_pair: Tuple[str, str],
    original_question: Dict[str, Any],
) -> Dict[str, Any] | None:
    """Create a probe/target/fact row from a cloze statement."""
    answer, statement = refined_cloze_pair
    probe_data = {'target': answer, 'probe': None, 'fact': statement}
    probe_data.update(original_question)
    return probe_data


def process_document(
    document_name: str,
    document_content: str,
    source: str = "arxiv",
    add_on_mode: bool = False,
    source_version: str | None = None,
    output_version: str = DEFAULT_OUTPUT_VERSION,
    generation_passes: int = 4,
    skip_tokenizer_check: bool = False,
    **kwargs,
) -> None:
    title = extract_title(document_name, document_content, source)

    document_chunks, model, reasoning_effort = build_document_chunks(
        document_content,
        source,
        add_on_mode=add_on_mode,
        generation_passes=generation_passes,
    )

    print(
        f"Generating composition inference questions with {len(document_chunks)} "
        "full-document pass(es) in parallel..."
    )

    questions = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        future_to_chunk = {
            executor.submit(generate_questions, chunk, source, model, reasoning_effort): chunk
            for chunk in document_chunks
        }
        for future in tqdm(
            concurrent.futures.as_completed(future_to_chunk),
            total=len(document_chunks),
            desc="Generating questions",
        ):
            try:
                questions_for_chunk = future.result()
                if questions_for_chunk:
                    questions.extend(questions_for_chunk)
            except Exception as exc:
                print(f'A document generated an exception: {exc}')

    print(f"Generated a total of {len(questions)} questions.")
    if not questions:
        print("No questions were generated. Exiting.")
        return

    save_df_for_debugging(
        pd.DataFrame(questions),
        '01_generated_questions_raw_v2.txt',
        'inference',
        document_name,
        ['question', 'answer', 'source_facts', 'derivation', 'inference_type'],
    )

    questions_before_dedupe = len(questions)
    questions, dedupe_dropped_questions = deduplicate_questions(questions, source)
    print(
        f"Kept {len(questions)} of {questions_before_dedupe} questions after conceptual deduplication "
        f"({len(dedupe_dropped_questions)} dropped)."
    )
    if not questions:
        print("No questions remained after conceptual deduplication. Exiting.")
        return

    if dedupe_dropped_questions:
        save_df_for_debugging(
            pd.DataFrame(dedupe_dropped_questions),
            '02_dropped_duplicate_questions_v2.txt',
            'inference',
            document_name,
            ['question', 'answer', 'source_facts', 'derivation', 'inference_type'],
        )

    save_df_for_debugging(
        pd.DataFrame(questions),
        '02_deduplicated_questions_v2.txt',
        'inference',
        document_name,
        ['question', 'answer', 'source_facts', 'derivation', 'inference_type'],
    )

    cloze_probes_list = []
    cloze_pairs_list = []
    refined_cloze_list = []
    cloze_conversion_dropped = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        future_to_question = {
            executor.submit(convert_to_cloze, q, source): q
            for q in questions
        }
        for future in tqdm(
            concurrent.futures.as_completed(future_to_question),
            total=len(future_to_question),
            desc="Converting to cloze statements",
        ):
            question = future_to_question[future]
            try:
                cloze_pair = future.result()
                if cloze_pair:
                    cloze_pairs_list.append((cloze_pair, question))
                else:
                    dropped_question = dict(question)
                    dropped_question['drop_stage'] = 'cloze_conversion'
                    dropped_question['drop_reason'] = 'no_cloze_pair'
                    cloze_conversion_dropped.append(dropped_question)
            except Exception as exc:
                print(f'"{question.get("question", "A question")}" generated an exception during cloze conversion: {exc}')
                dropped_question = dict(question)
                dropped_question['drop_stage'] = 'cloze_conversion'
                dropped_question['drop_reason'] = str(exc)
                cloze_conversion_dropped.append(dropped_question)

    if cloze_conversion_dropped:
        save_df_for_debugging(
            pd.DataFrame(cloze_conversion_dropped),
            '03_dropped_cloze_conversion_v2.txt',
            'inference',
            document_name,
            ['question', 'answer', 'drop_stage', 'drop_reason', 'source_facts', 'derivation', 'inference_type'],
        )

    save_debug_file(
        json.dumps({'pairs': [p for p, _ in cloze_pairs_list]}, indent=2),
        '03_cloze_pairs_v2.txt',
        'inference',
        document_name,
    )

    qc_dropped = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        future_to_data = {}
        for pair, question in cloze_pairs_list:
            supporting_context = "\n".join(question.get("source_facts", [])) or document_content
            future = executor.submit(
                quality_control_cloze,
                pair,
                title,
                supporting_context,
                source,
            )
            future_to_data[future] = (pair, question)

        for future in tqdm(
            concurrent.futures.as_completed(future_to_data),
            total=len(future_to_data),
            desc="Performing quality control",
        ):
            original_pair, question = future_to_data[future]
            try:
                refined_pair = future.result()
                if refined_pair:
                    refined_cloze_list.append((refined_pair, question))
                else:
                    qc_dropped.append({
                        'question': question.get('question', ''),
                        'answer': question.get('answer', ''),
                        'cloze_pair': original_pair,
                        'drop_stage': 'quality_control',
                        'drop_reason': 'no_refined_pair',
                    })
            except Exception as exc:
                print(f'"{question.get("question", "A question")}" generated an exception during QC: {exc}')
                qc_dropped.append({
                    'question': question.get('question', ''),
                    'answer': question.get('answer', ''),
                    'cloze_pair': original_pair,
                    'drop_stage': 'quality_control',
                    'drop_reason': str(exc),
                })

    if qc_dropped:
        save_df_for_debugging(
            pd.DataFrame(qc_dropped),
            '04_dropped_quality_control_v2.txt',
            'inference',
            document_name,
            ['question', 'answer', 'cloze_pair', 'drop_stage', 'drop_reason'],
        )

    save_debug_file(
        json.dumps({'pairs': [p for p, _ in refined_cloze_list]}, indent=2),
        '04_refined_cloze_v2.txt',
        'inference',
        document_name,
    )

    print("Filtering refined cloze pairs...")
    filtered_refined_cloze_list = []
    filter_dropped = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        future_to_data = {
            executor.submit(filter_cloze_pair, refined_pair): (refined_pair, question)
            for refined_pair, question in refined_cloze_list
        }
        for future in tqdm(
            concurrent.futures.as_completed(future_to_data),
            total=len(future_to_data),
            desc="Filtering cloze pairs",
        ):
            original_refined_pair, question = future_to_data[future]
            try:
                keep = future.result()
                if keep:
                    filtered_refined_cloze_list.append((original_refined_pair, question))
                else:
                    filter_dropped.append({
                        'question': question.get('question', ''),
                        'answer': question.get('answer', ''),
                        'refined_pair': original_refined_pair,
                        'drop_stage': 'cloze_pair_filter',
                        'drop_reason': 'filter_keep_false',
                    })
            except Exception as exc:
                print(f'A refined pair generated an exception during filtering: {exc}')
                filter_dropped.append({
                    'question': question.get('question', ''),
                    'answer': question.get('answer', ''),
                    'refined_pair': original_refined_pair,
                    'drop_stage': 'cloze_pair_filter',
                    'drop_reason': str(exc),
                })
    
    print(f"Kept {len(filtered_refined_cloze_list)} of {len(refined_cloze_list)} cloze pairs after filtering.")
    if filter_dropped:
        save_df_for_debugging(
            pd.DataFrame(filter_dropped),
            '05_dropped_cloze_pair_filter_v2.txt',
            'inference',
            document_name,
            ['question', 'answer', 'refined_pair', 'drop_stage', 'drop_reason'],
        )

    print("Creating cloze probes...")
    for refined_pair, question in filtered_refined_cloze_list:
        probe_data = create_cloze_probe(refined_pair, question)
        if probe_data:
            cloze_probes_list.append(probe_data)

    if not cloze_probes_list:
        print("No valid cloze probes created. Exiting.")
        return

    cloze_df = pd.DataFrame(cloze_probes_list)

    valid_facts = []
    for _, row in cloze_df.iterrows():
        fact = str(row['fact']).strip().rstrip('.!?').rstrip()
        target = ' ' + str(row['target']).strip()
        valid_facts.append(fact.endswith(target))

    cloze_df['valid_fact'] = valid_facts
    print(f"Found {cloze_df['valid_fact'].sum()} probes where target is at the end of the fact.")
    invalid_cloze_df = cloze_df[~cloze_df['valid_fact']].copy()
    if not invalid_cloze_df.empty:
        save_df_for_debugging(
            invalid_cloze_df,
            '06_dropped_invalid_cloze_end_v2.txt',
            'inference',
            document_name,
            ['probe', 'target', 'fact', 'question', 'source_facts', 'derivation', 'inference_type'],
        )
    cloze_df = cloze_df[cloze_df['valid_fact']].drop(columns=['valid_fact']).reset_index(drop=True)

    probes = []
    cleaned_facts = []
    cleaned_targets = []
    for _, row in cloze_df.iterrows():
        fact = str(row['fact']).strip().rstrip('.!?').rstrip()
        target = ' ' + str(row['target']).strip()
        last_index = fact.rfind(target)
        if last_index != -1:
            probes.append(fact[:last_index].strip())
            cleaned_facts.append(fact)
            cleaned_targets.append(target)
        else:
            probes.append(None)
            cleaned_facts.append(fact)
            cleaned_targets.append(target)

    cloze_df['probe'] = probes
    cloze_df['fact'] = cleaned_facts
    cloze_df['target'] = cleaned_targets
    cloze_df.dropna(subset=['probe'], inplace=True)

    malformed_cloze_mask = cloze_df.apply(
        lambda row: is_question_shaped_probe(row['probe'])
        or not completed_cloze_ends_with_target(row['probe'], row['target'], row['fact']),
        axis=1,
    )
    malformed_cloze_df = cloze_df[malformed_cloze_mask].copy()
    if not malformed_cloze_df.empty:
        save_df_for_debugging(
            malformed_cloze_df,
            '06_dropped_malformed_cloze_shape_v2.txt',
            'inference',
            document_name,
            ['probe', 'target', 'fact', 'question', 'source_facts', 'derivation', 'inference_type'],
        )
    cloze_df = cloze_df[~malformed_cloze_mask].reset_index(drop=True)

    save_df_for_debugging(
        cloze_df,
        '07_final_probes_v2.txt',
        'inference',
        document_name,
        ['probe', 'target', 'fact', 'question', 'source_facts', 'derivation', 'inference_type'],
    )

    total_final_probes = len(cloze_df)
    print(f"\nFinal probe count after cloze validation: {total_final_probes}")

    if not skip_tokenizer_check:
        tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B")
        check_tokenizer_consistency(cloze_df, tokenizer)

    output_dir = str(probe_paths.resolve_probe_dir('inference', document_name, source))
    os.makedirs(output_dir, exist_ok=True)

    metrics_path = os.path.join(output_dir, f'filtering_metrics_{output_version}_composition_v2.txt')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write(f"Inference Probe Pipeline Composition v2 - Filtering Metrics for {document_name}\n")
        f.write(f"{'=' * 72}\n")
        f.write(f"Source: {source}\n")
        f.write(f"Output version: {output_version}\n")
        f.write(f"Total generated questions: {len(questions)}\n")
        f.write(f"Total probes after cloze validation: {total_final_probes}\n")
    print(f"Saved filtering metrics to {metrics_path}")

    if add_on_mode and source_version:
        source_path = os.path.join(output_dir, f'probes_{source_version}.csv')
        if os.path.exists(source_path):
            source_df = pd.read_csv(source_path)
            print(f"Loaded {len(source_df)} existing probes from {source_version} for {document_name}")
            cloze_df = pd.concat([source_df, cloze_df], ignore_index=True)
            print(f"Combined to create {len(cloze_df)} total probes")
        else:
            print(f"No existing {source_version} found for {document_name}, creating new {output_version}")

    output_path = os.path.join(output_dir, f'probes_{output_version}.csv')
    cloze_df.to_csv(output_path, index=False)
    print(f"Saved {len(cloze_df)} composition cloze probes to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=SUPPORTED_SOURCES, default="arxiv")
    parser.add_argument("--filter", type=str, default=None, help="Only process documents containing this string in the filename.")
    parser.add_argument("--output_version", type=str, default=DEFAULT_OUTPUT_VERSION)
    parser.add_argument("--add_on_to", type=str, default=None, help="Existing probe version to load and append to, e.g. v8.")
    parser.add_argument("--generation_passes", type=int, default=4, help="Number of full-document question-generation passes before conceptual deduplication.")
    parser.add_argument("--skip_tokenizer_check", action="store_true")
    args = parser.parse_args()

    extension = ".tex" if args.source == "arxiv" else ".txt"
    input_dir = os.path.join(PROJECT_ROOT, "data", args.source, "cleaned")
    add_on_mode = args.add_on_to is not None

    def process_with_source(document_name: str, document_content: str, **kwargs):
        process_document(
            document_name,
            document_content,
            source=args.source,
            add_on_mode=add_on_mode,
            source_version=args.add_on_to,
            output_version=args.output_version,
            generation_passes=args.generation_passes,
            skip_tokenizer_check=args.skip_tokenizer_check,
            **kwargs,
        )

    print(
        f"Processing {args.source} documents from {input_dir}; "
        f"saving probes_{args.output_version}.csv"
    )
    process_papers(
        process_with_source,
        input_dir,
        file_filter=args.filter,
        extension=extension,
    )


if __name__ == '__main__':
    main()
