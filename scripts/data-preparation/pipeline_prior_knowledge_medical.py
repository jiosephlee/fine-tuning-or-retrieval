import os
import json
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.append('../../')
import utils.utils as utils
from importlib import reload
reload(utils)


def generate_prior_knowledge(case_name):
    """Process a single medical case report to generate prior knowledge chapters."""
    print(f"Processing {case_name} for prior knowledge generation...")

    CASE_FILE_PATH = f'../../data/medical/raw/{case_name}.txt'
    OUTPUT_DIR = f"../../data/medical/prior_knowledge/{case_name}/"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Reading case report from: {CASE_FILE_PATH}")
    with open(CASE_FILE_PATH, 'r') as f:
        case_content = f.read()

    # --- 1. Generate list of chapters ---
    print("Generating list of prerequisite chapters...")
    prompt_chapters = {
        'system': """### Instructions
You are an expert medical educator. Based on the provided clinical case report, create a list of textbook chapters that would provide all the necessary prior knowledge to understand this case. The chapters should not describe the specific patient or case details, but rather the foundational medical concepts upon which understanding the case depends.

Consider prerequisite knowledge across these areas as relevant:
- Anatomy and physiology of affected organ systems
- Pathophysiology of the conditions discussed
- Relevant pharmacology and mechanisms of action
- Diagnostic methods and interpretation (lab values, imaging, histopathology)
- Differential diagnosis frameworks
- Standard-of-care treatment protocols and guidelines
- Epidemiology and risk factors

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
      "title": "Chapter 1: Cardiac Valve Anatomy and Physiology",
      "description": "This chapter covers the structure and function of cardiac valves...",
      "subtopics": ["Valve anatomy", "Hemodynamic principles", "Pathological changes"]
    }
  ]
}""",
        'user': f"### Clinical Case Report\n{case_content}"
    }

    response_chapters_str = utils.query_llm(
        prompt_chapters,
        model='gpt-5-mini',
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

    # --- 2. Generate content for each chapter ---
    if chapters_list:

        def generate_chapter(chapter_info, chapter_index):
            """Generate content for a single chapter."""
            chapter_title = chapter_info.get('title', f"Chapter {chapter_index+1}")
            chapter_description = chapter_info.get('description', '')
            chapter_subtopics = chapter_info.get('subtopics', [])

            print(f"Generating content for: {chapter_title}")

            subtopics_str = "\n".join([f"- {s}" for s in chapter_subtopics])

            prompt_content = {
                'system': """### Instructions
You will be given a chapter title, description, and subtopics from a medical textbook. Based on those topics, your job is to write a detailed, cohesive textbook chapter addressed to a medical student or resident who needs this foundational knowledge to understand a clinical case report.

The chapter should be comprehensive and clinically oriented. Begin with an introduction to the chapter, then cover each subtopic in turn. Don't just briefly describe the subtopics, but rather elaborate on the concepts at full length. Emphasize pathophysiology, clinical reasoning, and the connections between basic science and clinical practice. Dedicate multiple paragraphs to each subtopic. Write in full prose, rather than bullet points.

Separate each subtopic with a section header "#".

Where relevant, include normal reference ranges for lab values, key diagnostic criteria, and clinical decision points. Write all mathematical notation in LaTeX only e.g. "$x^2$" or "$\\pi$". Do not use unicode mathematical characters.""",
                'user': f"""### Chapter Title
{chapter_title}

### Chapter Description
{chapter_description}

### Subtopics to Cover
{subtopics_str}"""
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
    manifest_path = "../../data/medical/raw/manifest.json"
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    saved_cases = [entry for entry in manifest if entry.get("status") == "saved"]
    print(f"Found {len(saved_cases)} saved cases in manifest.\n")

    for entry in saved_cases:
        case_name = entry["filename"]
        output_dir = f"../../data/medical/prior_knowledge/{case_name}/"
        if os.path.exists(os.path.join(output_dir, "textbook.txt")):
            print(f"Skipping {case_name} (already generated).")
            continue
        generate_prior_knowledge(case_name)
        print()


if __name__ == "__main__":
    process_cases()
