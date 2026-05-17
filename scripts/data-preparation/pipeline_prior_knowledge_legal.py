import os
import json
import sys
import glob
import re
from concurrent.futures import ThreadPoolExecutor

sys.path.append('../../')
import utils.utils as utils
from importlib import reload
reload(utils)


def _clean_whitespace(text):
    return re.sub(r'\s+', ' ', text).strip()


def extract_case_title(text, opinion_id):
    """Best-effort title extraction from locally fetched opinion text."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    def clean_title(title):
        title = _clean_whitespace(title)
        title = re.sub(r'\b(APPELLANTS?|APPELLEES?|PETITIONERS?|RESPONDENTS?|PLAINTIFF[- ]APPELLANTS?|DEFENDANT[- ]APPELLEES?)\b', '', title, flags=re.I)
        title = re.sub(r'\b(consolidated with|appeal from|certiorari to|no\.).*$', '', title, flags=re.I)
        title = _clean_whitespace(title).strip(" ,-")
        if len(title) > 220:
            return None
        if re.search(r'\b(according to|declaration|paragraph|j\.a\.|plaintiffs|defendants|applying this standard|we review|we hold|the court)\b', title, flags=re.I):
            return None
        if len(title.split()) > 24:
            return None
        return title.title() if title else None

    for i, line in enumerate(lines[:80]):
        if line.lower() == "v.":
            prev_lines = [candidate for candidate in lines[max(0, i - 4):i] if not re.search(r'^(x|-|_+|no\.|file name:)', candidate, flags=re.I)]
            next_lines = [candidate for candidate in lines[i + 1:i + 5] if not re.search(r'^(x|-|_+|no\.|file name:)', candidate, flags=re.I)]
            title = _clean_whitespace(" ".join(prev_lines + ["v."] + next_lines))
            cleaned = clean_title(title)
            if cleaned:
                return cleaned
        if re.search(r'\bv\.', line, flags=re.I) and len(line) < 220:
            if line.lower().startswith("v.") and i > 0:
                title = _clean_whitespace(f"{lines[i - 1]} {line}")
            else:
                title = _clean_whitespace(line)
                if line.rstrip().endswith("-") and i + 1 < len(lines):
                    title = _clean_whitespace(title[:-1] + lines[i + 1])
            cleaned = clean_title(title)
            if cleaned:
                return cleaned

    compact = _clean_whitespace(text[:3000])
    match = re.search(r'([A-Z][A-Z0-9.,&\'’\- ]{3,120}\s+v\.\s+[A-Z][A-Z0-9.,&\'’\- ]{3,120})', compact)
    if match:
        title = re.split(r'\b(certiorari|appeal|no\.)\b', match.group(1), flags=re.I)[0]
        cleaned = clean_title(title)
        if cleaned:
            return cleaned

    return f"Cited opinion {opinion_id}"


def load_cited_opinion_candidates(case_name):
    """Load cited opinion IDs, best-effort titles, metadata, and local paths."""
    cited_dir = f"../../data/legal/raw/{case_name}/cited"
    candidates = []
    if not os.path.isdir(cited_dir):
        return candidates

    for meta_path in sorted(glob.glob(os.path.join(cited_dir, "cited_*.meta.json"))):
        opinion_id = os.path.basename(meta_path).replace("cited_", "").replace(".meta.json", "")
        text_path = os.path.join(cited_dir, f"cited_{opinion_id}.txt")
        if not os.path.exists(text_path):
            continue

        with open(meta_path, 'r') as f:
            meta = json.load(f)
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()

        candidates.append({
            "opinion_id": meta.get("opinion_id", opinion_id),
            "title": extract_case_title(text, opinion_id),
            "depth": meta.get("depth"),
            "word_count": meta.get("word_count"),
            "path": text_path,
        })

    return candidates


def select_landmark_cases(case_name, case_context, case_content, output_dir):
    """Use the source opinion and cited opinion titles to select landmark cited cases."""
    cited_candidates = load_cited_opinion_candidates(case_name)
    if not cited_candidates:
        print("No cited opinions found for landmark selection.")
        return []

    candidate_list = []
    for item in cited_candidates:
        with open(item["path"], encoding='utf-8') as f:
            opening_excerpt = _clean_whitespace(f.read(1200))
        candidate_list.append({
            "opinion_id": item["opinion_id"],
            "title": item["title"],
            "depth": item["depth"],
            "word_count": item["word_count"],
            "opening_excerpt": opening_excerpt,
        })

    prompt_landmarks = {
        'system': """### Instructions
You are an expert legal educator selecting prior-knowledge precedent for a law student.

You will receive a judicial opinion and a list of cited opinions available locally. The list contains titles/IDs, basic metadata, and short opening excerpts to disambiguate cases whose titles could not be reliably extracted. It does not contain the full cited opinions.

Select the cited opinions that are most likely to be landmark or doctrinally foundational for understanding the source opinion. Prefer broadly applicable precedents that teach general doctrine, standards, or canonical tests. Do not select every cited opinion. Exclude cases that appear merely fact-specific, procedural, duplicative, or useful only because of the source case's particular parties or outcome.

Select 1-6 cases. If no cited opinion appears landmark or foundational, return an empty list.

### Output Format
Return JSON with a single key `landmark_cases`, a list of objects:
{
  "landmark_cases": [
    {
      "opinion_id": 123,
      "title": "Case Name v. Case Name",
      "priority": 1,
      "reason": "Brief explanation of the general doctrine this case likely supplies."
    }
  ]
}""",
        'user': f"""### Source Case Metadata
{json.dumps(case_context, indent=2)}

### Source Judicial Opinion
{case_content}

### Locally Available Cited Opinions
{json.dumps(candidate_list, indent=2)}"""
    }

    response_landmarks_str = utils.query_llm(
        prompt_landmarks,
        model='gpt-5',
        system_prompt_included=True,
        return_json=True,
        max_tokens=3000
    )

    landmark_path = os.path.join(output_dir, "landmark_cases.json")
    with open(landmark_path, 'w') as f:
        f.write(response_landmarks_str)
    print(f"Saved landmark case selection to {landmark_path}")

    landmark_data = json.loads(response_landmarks_str)
    selected_cases = landmark_data.get("landmark_cases", [])
    print(f"Selected {len(selected_cases)} landmark cited cases.")
    return selected_cases


def format_landmark_case_list(landmark_cases):
    if not landmark_cases:
        return "No landmark cited opinions were selected."

    lines = []
    for case in sorted(landmark_cases, key=lambda item: item.get("priority", 999)):
        lines.append(
            f"- opinion_id={case.get('opinion_id')}: {case.get('title')} "
            f"(reason: {case.get('reason')})"
        )
    return "\n".join(lines)


def build_landmark_case_context(case_name, landmark_cases, max_chars_per_case=6000):
    if not landmark_cases:
        return ""

    candidates = {
        str(item["opinion_id"]): item
        for item in load_cited_opinion_candidates(case_name)
    }
    parts = []
    for case in sorted(landmark_cases, key=lambda item: item.get("priority", 999)):
        candidate = candidates.get(str(case.get("opinion_id")))
        if not candidate:
            continue
        with open(candidate["path"], 'r', encoding='utf-8') as f:
            excerpt = f.read(max_chars_per_case)
        parts.append(
            f"### {case.get('title')} (opinion_id={case.get('opinion_id')})\n"
            f"Selection reason: {case.get('reason')}\n\n"
            f"Excerpt from cited opinion:\n{excerpt}"
        )
    return "\n\n".join(parts)


def select_chapters_for_landmark_context(chapters_list, landmark_cases, output_dir):
    """Ask an LLM which generated chapters should receive landmark case excerpts."""
    if not chapters_list or not landmark_cases:
        decision = {"chapters": []}
        decision_path = os.path.join(output_dir, "landmark_context_chapters.json")
        with open(decision_path, 'w') as f:
            json.dump(decision, f, indent=2)
        print(f"Saved landmark context chapter selection to {decision_path}")
        return set()

    chapter_summaries = []
    for i, chapter in enumerate(chapters_list, 1):
        chapter_summaries.append({
            "chapter_number": i,
            "title": chapter.get("title", f"Chapter {i}"),
            "description": chapter.get("description", ""),
            "subtopics": chapter.get("subtopics", []),
        })

    prompt_context_chapters = {
        'system': """### Instructions
You are deciding which textbook chapters need excerpts from selected landmark cited opinions as source context.

You will receive:
- a generated textbook chapter outline
- a list of selected landmark cited opinions

Select only chapters whose content would materially benefit from having the actual landmark cited opinion excerpts. Usually this means chapters that explicitly teach landmark cases, precedent, case-law doctrine, or the doctrines associated with the selected landmark cases.

Do not select chapters that can be written from general legal background alone. Do not select every chapter. Prefer 1-2 chapters when the outline contains dedicated landmark-case chapters. Select more only if the outline genuinely splits landmark-case discussion across more chapters.

### Output Format
Return JSON with a single key `chapters`, a list of objects:
{
  "chapters": [
    {
      "chapter_number": 11,
      "title": "Chapter title",
      "needs_landmark_context": true,
      "reason": "Brief reason this chapter needs cited-opinion excerpts."
    }
  ]
}""",
        'user': f"""### Generated Chapter Outline
{json.dumps(chapter_summaries, indent=2)}

### Selected Landmark Cited Opinions
{json.dumps(landmark_cases, indent=2)}"""
    }

    response_context_chapters_str = utils.query_llm(
        prompt_context_chapters,
        model='gpt-5-mini',
        system_prompt_included=True,
        return_json=True,
        max_tokens=2000
    )

    decision_path = os.path.join(output_dir, "landmark_context_chapters.json")
    with open(decision_path, 'w') as f:
        f.write(response_context_chapters_str)
    print(f"Saved landmark context chapter selection to {decision_path}")

    decision_data = json.loads(response_context_chapters_str)
    selected_chapter_numbers = set()
    for item in decision_data.get("chapters", []):
        chapter_number = item.get("chapter_number")
        if isinstance(chapter_number, int) and 1 <= chapter_number <= len(chapters_list):
            selected_chapter_numbers.add(chapter_number)

    print(f"Selected {len(selected_chapter_numbers)} chapters for landmark context.")
    return selected_chapter_numbers


def generate_prior_knowledge(case_name):
    """Process a single legal opinion to generate prior knowledge chapters."""
    print(f"Processing {case_name} for prior knowledge generation...")

    CASE_FILE_PATH = f'../../data/legal/raw/{case_name}.txt'
    META_FILE_PATH = f'../../data/legal/raw/{case_name}.meta.json'
    OUTPUT_DIR = f"../../data/legal/prior_knowledge/{case_name}/"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Reading legal opinion from: {CASE_FILE_PATH}")
    with open(CASE_FILE_PATH, 'r', encoding='utf-8') as f:
        case_content = f.read()

    case_meta = {}
    if os.path.exists(META_FILE_PATH):
        with open(META_FILE_PATH, 'r') as f:
            case_meta = json.load(f)

    case_context = {
        "case_name": case_meta.get("case_name", case_name),
        "court": case_meta.get("court"),
        "date_filed": case_meta.get("date_filed"),
    }

    landmark_cases = select_landmark_cases(case_name, case_context, case_content, OUTPUT_DIR)
    landmark_case_list = format_landmark_case_list(landmark_cases)
    landmark_case_context = build_landmark_case_context(case_name, landmark_cases)

    # --- 1. Generate list of chapters ---
    print("Generating list of prerequisite chapters...")
    prompt_chapters = {
        'system': """### Instructions
You are an expert legal educator. Based on the provided judicial opinion, create a list of textbook chapters that would provide all the necessary prior knowledge to understand the opinion. The chapters should not describe the specific parties, facts, procedural outcome, holding, or novel reasoning of the case itself. They should cover the foundational legal concepts, doctrines, statutory frameworks, procedural rules, and standards of review that a law student would need before reading the opinion.

Consider prerequisite knowledge across these areas as relevant:
- The substantive area of law and its core elements
- Statutory or regulatory frameworks invoked by the opinion
- Constitutional doctrines, if relevant
- Civil, criminal, administrative, appellate, or habeas procedure, if relevant
- Jurisdiction, standing, preservation, waiver, exhaustion, and remedies
- Burdens of proof, standards of review, and evidentiary concepts
- Common analytical tests, multi-factor standards, and doctrinal exceptions
- The role of precedent and how courts apply cited authority

When landmark cited opinions are provided, dedicate 1-2 chapters to those landmark cases and the doctrines they establish. A landmark-cases chapter may discuss multiple prior cases. Cover the cases as general doctrinal background, not as a map of how the source opinion used them.

For each chapter, provide:
- A `title`.
- A general `description` of what the chapter covers.
- A list of `subtopics` that should be included.

### Output Format
Provide the output as a JSON object with a single key "chapters", which is a list of chapter dictionaries.
Example:
{
  "chapters": [
    {
      "title": "Chapter 1: Federal Appellate Jurisdiction and Standards of Review",
      "description": "This chapter covers when federal appellate courts may review lower-court decisions and how different standards of review structure appellate analysis.",
      "subtopics": ["Final judgment rule", "Preservation of issues", "De novo review", "Clear error review", "Abuse of discretion"]
    }
  ]
}""",
        'user': f"""### Case Metadata
{json.dumps(case_context, indent=2)}

### Judicial Opinion
{case_content}

### Selected Landmark Cited Opinions
{landmark_case_list}"""
    }

    response_chapters_str = utils.query_llm(
        prompt_chapters,
        model='gpt-5',
        system_prompt_included=True,
        return_json=True,
        max_tokens=5000
    )

    chapters_log_path = os.path.join(OUTPUT_DIR, "chapters_outline.json")
    with open(chapters_log_path, 'w') as f:
        f.write(response_chapters_str)
    print(f"Saved chapters outline to {chapters_log_path}")

    chapters_data = json.loads(response_chapters_str)
    chapters_list = chapters_data['chapters']
    print(f"Parsed outline with {len(chapters_list)} chapters.")
    landmark_context_chapter_numbers = select_chapters_for_landmark_context(
        chapters_list,
        landmark_cases,
        OUTPUT_DIR
    )

    # --- 2. Generate content for each chapter ---
    if chapters_list:

        def generate_chapter(chapter_info, chapter_index):
            """Generate content for a single chapter."""
            chapter_title = chapter_info.get('title', f"Chapter {chapter_index+1}")
            chapter_description = chapter_info.get('description', '')
            chapter_subtopics = chapter_info.get('subtopics', [])

            print(f"Generating content for: {chapter_title}")

            subtopics_str = "\n".join([f"- {s}" for s in chapter_subtopics])
            selected_landmark_context = ""
            if chapter_index + 1 in landmark_context_chapter_numbers:
                selected_landmark_context = f"""

### Selected Landmark Cited Opinion Context
Use this context only for general prior-knowledge discussion of landmark precedent. Do not describe how the source case used these cases.

{landmark_case_context}"""

            prompt_content = {
                'system': """### Instructions
You will be given a chapter title, description, and subtopics from a legal textbook. Based on those topics, write a detailed, cohesive textbook chapter addressed to a law student who needs this foundational knowledge before reading a judicial opinion.

The chapter should be comprehensive and doctrinally oriented. Begin with an introduction, then cover each subtopic in turn. Explain the governing concepts at full length, with emphasis on legal reasoning, how courts apply rules to facts, and how procedural posture affects analysis. Write in full prose rather than bullet points.

Do not discuss the specific parties, facts, procedural outcome, holding, or novel reasoning of the source case. The chapter should teach general prior knowledge only.

Separate each subtopic with a section header "#".

Where relevant, describe common standards, elements, tests, exceptions, and burdens. Write all mathematical notation in LaTeX only e.g. "$x^2$" or "$\\pi$". Do not use unicode mathematical characters.""",
                'user': f"""### Chapter Title
{chapter_title}

### Chapter Description
{chapter_description}

### Subtopics to Cover
{subtopics_str}{selected_landmark_context}"""
            }

            chapter_content = utils.query_llm(
                prompt_content,
                model='gpt-5-mini',
                system_prompt_included=True,
                reasoning_effort="low",
                max_tokens=10000
            )

            if chapter_content and not chapter_content.strip().endswith(('.', '!', '?', '"', '`')):
                last_period_index = chapter_content.rfind('.')
                if last_period_index != -1:
                    chapter_content = chapter_content[:last_period_index+1]

            return chapter_content, chapter_index

        with ThreadPoolExecutor(max_workers=min(len(chapters_list), 5)) as executor:
            futures = [executor.submit(generate_chapter, chapter_info, i)
                      for i, chapter_info in enumerate(chapters_list)]

            all_chapter_contents = []
            for future in futures:
                try:
                    chapter_content, chapter_index = future.result()
                    all_chapter_contents.append((chapter_content, chapter_index))
                except Exception as e:
                    print(f"Error generating chapter: {e}")

        all_chapter_contents.sort(key=lambda x: x[1])

        textbook_content_parts = []
        for chapter_content, chapter_index in all_chapter_contents:
            chapter_filename = f"chapter_{chapter_index+1}.txt"
            chapter_path = os.path.join(OUTPUT_DIR, chapter_filename)
            with open(chapter_path, 'w', encoding='utf-8') as f:
                f.write(chapter_content)
            print(f"Saved chapter {chapter_index+1} to: {chapter_path}")

            textbook_content_parts.append(chapter_content)

        textbook_content = "\n\n".join(textbook_content_parts)
        output_path = os.path.join(OUTPUT_DIR, "textbook.txt")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(textbook_content)

        print(f"Saved complete textbook to: {output_path}")
        print(f"Generated textbook with {len(chapters_list)} chapters.")
    else:
        print("No chapters were generated. Exiting.")


def process_cases():
    input_dir = "../../data/legal/raw/"
    case_names = sorted(
        os.path.splitext(filename)[0]
        for filename in os.listdir(input_dir)
        if filename.endswith(".txt")
    )
    print(f"Found {len(case_names)} local legal raw opinion files.\n")

    for case_name in case_names:
        output_dir = f"../../data/legal/prior_knowledge/{case_name}/"
        if os.path.exists(os.path.join(output_dir, "textbook.txt")):
            print(f"Skipping {case_name} (already generated).")
            continue
        generate_prior_knowledge(case_name)
        print()


if __name__ == "__main__":
    process_cases()
