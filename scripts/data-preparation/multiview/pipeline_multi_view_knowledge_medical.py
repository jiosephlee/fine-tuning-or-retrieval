import os
import json
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "data"
MEDICAL_CLEANED_DIR = DATA_ROOT / "medical" / "cleaned"
sys.path.insert(0, str(PROJECT_ROOT))
import utils.utils as utils
from utils.granular_outputs import write_granular_files
from utils.multiview_recovery import MEDICAL_TEXTBOOK_SCHEMA, manifest_valid, record_validated_view
from importlib import reload
reload(utils)

# Per-generation output budget. Defaults to the historical 32k cap; override via
# MULTIVIEW_MAX_TOKENS to request longer completions (clamped to fit the served
# context window by utils._create_vllm_completion).
MAX_TOKENS = int(os.environ.get("MULTIVIEW_MAX_TOKENS", "32768"))


def extract_case_title(case_content, case_name):
    """Return the source case title using the cleaned medical document format."""
    for line in case_content.splitlines():
        if line.startswith("Title:"):
            return line.split(":", 1)[1].strip()
    return case_name.replace("_", " ")


def generate_teaching_qa(case_name, model=None, slug="gpt_5_mini_custom", efforts=None, outline_model="gpt-5", writing_model="gpt-5-mini",
                         provider="auto", base_url=None, max_workers=4):
    """Generate NEJM Clinical Pearls & Morning Report style teaching Q&A for a medical case report."""
    print(f"Processing {case_name} for teaching Q&A...")

    CASE_FILE_PATH = MEDICAL_CLEANED_DIR / f"{case_name}.txt"
    OUTPUT_DIR = utils.explanations_dir('medical', slug, case_name, root=str(DATA_ROOT))

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(CASE_FILE_PATH, 'r') as f:
        case_content = f.read()
    case_title = extract_case_title(case_content, case_name)

    # --- 1. Generate teaching questions ---
    print("Generating clinical teaching questions...")
    prompt_questions = {
        'system': """You are a senior attending physician preparing a teaching session based on this clinical case report. You are creating 5-10 questions for residents and medical students in the style of NEJM Clinical Pearls & Morning Reports.

Your questions should:
- Vary in complexity, from straightforward clinical recall to deeper diagnostic and management reasoning
- Vary in topics, from clinical reasoning and pathophysiology to the facts, workup, diagnosis, and management decisions in the case
- Be as mutually exclusive and non-redundant as possible in terms of the clinical issues they cover, while covering all the key medical issues of the case as much as possible
- Be self-contained and unambiguous

For each question, provide:
- A `question` that an attending would pose during a teaching session
- A `category` that is one of: "Clinical Pearls", "Morning Report", "Pathophysiology", "Differential Diagnosis", "Management"

### Output Format
Provide the output as a JSON object with a single key "questions", which is a list of question dictionaries.
Example:
{
  "questions": [
    {
      "question": "What are the major causes of rhabdomyolysis, and how does the mechanism differ between toxic and autoimmune statin myopathy?",
      "category": "Pathophysiology"
    }
  ]
}""",
        'user': f"### Clinical Case Report\n{case_content}"
    }

    response_questions_str = utils.query_llm(
        prompt_questions,
        model=model or outline_model,
        reasoning_effort=efforts["questions"],
        system_prompt_included=True,
        return_json=True,
        max_tokens=MAX_TOKENS,
        provider=provider,
        base_url=base_url,
    )

    if response_questions_str is None:
        print(f"WARNING: LLM returned None for teaching Q&A questions on {case_name}, skipping")
        return

    outline_log_path = os.path.join(OUTPUT_DIR, "stack_exchange_outline.json")
    with open(outline_log_path, 'w') as f:
        f.write(response_questions_str)
    print(f"Saved teaching Q&A outline to {outline_log_path}")

    if not (response_questions_str or "").strip():
        print(f"WARNING: empty questions response for {case_name}; skipping teaching Q&A view")
        return
    questions_data = json.loads(response_questions_str)
    # json_object mode guarantees valid JSON but not the requested schema; models
    # occasionally omit the "questions" key or return the list at top level. Drop
    # malformed entries so one bad response doesn't KeyError the whole run.
    questions = utils.extract_dict_list(questions_data, preferred_key='questions')
    questions = [q for q in questions if q.get('question') and q.get('category')]
    if not questions:
        print(f"WARNING: no valid teaching questions for {case_name}; skipping teaching Q&A view")
        return
    print(f"Generated {len(questions)} teaching questions")

    # --- 2. Generate answers for each question ---
    print("Generating answers...")

    def generate_answer(question):
        print(f"Processing question: {question['question'][:50]}...")

        prompt_answer = {
            'system': """You are a senior attending physician answering a clinical teaching question during a morning report or case conference. Provide a clear, educational answer in the style of NEJM Clinical Pearls & Morning Reports.

Your answer should:
- Directly address the clinical question
- Explain the relevant clinical reasoning, pathophysiology, diagnostic standard, or management principle
- Reference the case report to support your explanation; stay grounded in the case report; do not introduce details inconsistent with the case
- Be written in prose, concise, and to the point
- Be accessible to medical students and residents while remaining clinically precise

Format your response as a direct, authoritative clinical teaching answer.""",
            'user': f"""### Clinical Teaching Question
Category: {question['category']}
Question: {question['question']}

### Clinical Case Report
{case_content}"""
        }

        answer_text = utils.query_llm(
            prompt_answer,
            model=model or writing_model,
            reasoning_effort=efforts["answer"],
            system_prompt_included=True,
            max_tokens=MAX_TOKENS,
            provider=provider,
            base_url=base_url,
        )

        return {
            'category': question['category'],
            'question': question['question'],
            'answer': answer_text
        }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        qa_pairs = list(tqdm(executor.map(generate_answer, questions), total=len(questions), desc="Generating teaching answers"))

    # --- 3. Create single file ---
    # Use stackexchange.txt filename for compatibility with data_preparation.py
    print("Creating teaching Q&A file...")

    content = f"Title: Clinical Q&A about the case report: \"{case_title}\"\n\n"
    for qa in qa_pairs:
        content += f"### [{qa['category']}] {qa['question']}\n\n"
        content += f"{qa['answer']}\n\n"

    output_file = os.path.join(OUTPUT_DIR, "stackexchange.txt")
    with open(output_file, 'w') as f:
        f.write(content)

    print(f"Saved teaching Q&A to {output_file}")

    title_line = f"Title: Clinical Q&A about the case report: \"{case_title}\""
    granular_qas = [
        f"{title_line}\n\n### [{qa['category']}] {qa['question']}\n\n{qa['answer']}"
        for qa in qa_pairs
    ]
    paths = write_granular_files(OUTPUT_DIR, "stackexchange", granular_qas)
    print(f"Saved {len(paths)} files to {Path(OUTPUT_DIR) / 'stackexchange'}/")


def generate_case_textbook(case_name, model=None, slug="gpt_5_mini_custom", efforts=None, outline_model="gpt-5", writing_model="gpt-5-mini",
                           provider="auto", base_url=None, max_workers=4):
    """Generate a case-based medical textbook chapter (Case Files / Harrison's style) for a clinical case report."""
    print(f"Processing {case_name} for case-based textbook chapter...")

    CASE_FILE_PATH = MEDICAL_CLEANED_DIR / f"{case_name}.txt"
    OUTPUT_DIR = utils.explanations_dir('medical', slug, case_name, root=str(DATA_ROOT))

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(CASE_FILE_PATH, 'r') as f:
        case_content = f.read()
    case_title = extract_case_title(case_content, case_name)

    # --- 1. Generate section outline for one chapter ---
    print("Generating textbook section outline...")
    prompt_outline = {
        'system': """### Instructions
You will be given a clinical case report and your task is to create a detailed outline for a textbook that comprehensively explains the given case report. But, it should go beyond merely summarizing and explaining, and be a proper pedagogical textbook that aims to fully educate the reader on what the case is about. The textbook should be aimed at medical students or residents who have a basic understanding of medicine.

The textbook should:
- Contain exactly 4 to 6 sections
- Break down the case report into coherent sections that follow the case's clinical logic
- Plan sections to be mutually exclusive and non-redundant while preserving high coverage of the presentation, differential diagnosis, workup, diagnostic reasoning, management decisions, outcome, and clinical significance
- Each section should have a distinct focus and role
- Cover the pathophysiology and background medicine needed to understand the case, but avoid broad textbook-review detours that are not necessary for this case
- Explain how clinicians moved from presentation to diagnosis and treatment
- Ensure a logical flow from case presentation to diagnostic reasoning, treatment decisions, outcome, and clinical lessons
- Be comprehensive but concise and to the point; do not create unnecessary sections

For each section, provide:
- A title
- A description of what the section covers
- A list of subtopics to cover

### Output Format
Provide the output as a JSON object with a single key "sections", which is a list of section objects. Each section object must have:
- "section_title": A string for the section title
- "description": A string describing the section's content
- "subtopics": A list of strings, each a subtopic to cover""",
        'user': f"### Clinical Case Report\n{case_content}"
    }

    response_outline_str = utils.query_llm(
        prompt_outline,
        model=model or outline_model,
        reasoning_effort=efforts["outline"],
        system_prompt_included=True,
        return_json=True,
        json_schema={"type": "json_schema", "json_schema": {
            "name": "medical_textbook_outline", "strict": True,
            "schema": MEDICAL_TEXTBOOK_SCHEMA}},
        max_tokens=MAX_TOKENS,
        provider=provider,
        base_url=base_url,
    )

    if response_outline_str is None:
        print(f"WARNING: LLM returned None for textbook outline on {case_name}, skipping")
        return

    if isinstance(response_outline_str, dict):
        response_outline_str = json.dumps(response_outline_str, ensure_ascii=False)
    outline_log_path = os.path.join(OUTPUT_DIR, "textbook_outline.json")
    with open(outline_log_path, 'w') as f:
        f.write(response_outline_str)
    print(f"Saved textbook outline to {outline_log_path}")

    if not (response_outline_str or "").strip():
        print(f"WARNING: empty textbook outline for {case_name}; skipping textbook view")
        return
    outline_data = json.loads(response_outline_str)
    # Tolerate schema drift: the list may sit under "sections", at the top level, or
    # under another key. Drop sections missing a required field.
    sections = utils.extract_dict_list(outline_data, preferred_key='sections')
    sections = [s for s in sections if s.get('section_title')
                and s.get('description') and isinstance(s.get('subtopics'), list)]
    if not sections:
        print(f"WARNING: no valid textbook sections for {case_name}; skipping textbook view")
        return
    with open(outline_log_path, 'w', encoding='utf-8') as f:
        json.dump({'sections': sections}, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(f"Parsed outline with {len(sections)} sections.")

    # --- 2. Write each section in parallel ---
    print("Writing textbook sections...")

    def write_section(section_info):
        section_title = section_info['section_title']
        print(f"Writing section: {section_title}...")

        section_outline_text = f"### Section Title\n\n{section_info['section_title']}\n"
        section_outline_text += f"## Description\n\n{section_info['description']}\n"
        section_outline_text += "## Subtopics\n"
        for point in section_info['subtopics']:
            section_outline_text += f"- {point}\n"

        prompt_section = {
            'system': """### Instructions
You will be given a section title, description, and subtopics for a textbook-style explanation of a clinical case report. Write this section for a medical student or resident who is learning how to understand the case report.

The section should:
- Explain the assigned part of the case report clearly and pedagogically
- Teach the clinical concepts, pathophysiology, diagnostic standards, and management principles needed to understand this part of the case
- Stay centered on the case report: presentation, labs, imaging or pathology, diagnostic reasoning, treatment decisions, outcome, and clinical lessons
- Explain broader medical context only when it is necessary to understand the case
- Avoid turning the section into a general review article on the disease
- Write in full prose paragraphs (this is a textbook, not an outline or reference card)
- Be concise and to the point
- Stay grounded in the case report; do not introduce case-specific claims or management decisions not supported by the report or necessary context

Start with the section title as a '##' header.""",
            'user': f"""### Clinical Case Report
{case_content}

### Section to Write
{section_outline_text}"""
        }

        section_content = utils.query_llm(
            prompt_section,
            model=model or writing_model,
            reasoning_effort=efforts["section"],
            system_prompt_included=True,
            max_tokens=MAX_TOKENS,
            provider=provider,
            base_url=base_url,
        )
        return section_content

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        section_contents = list(tqdm(executor.map(write_section, sections), total=len(sections), desc="Writing textbook sections"))

    # --- 3. Concatenate and save ---
    print("Assembling textbook chapter...")
    full_content = "\n\n".join(section_contents)
    full_textbook = f"Title: Textbook chapter about the case report: \"{case_title}\"\n\n{full_content}"

    output_file = os.path.join(OUTPUT_DIR, "textbook.txt")
    with open(output_file, 'w') as f:
        f.write(full_textbook)

    print(f"Saved textbook chapter to {output_file}")

    title_line = f"Title: Textbook chapter about the case report: \"{case_title}\""
    granular_sections = [
        f"{title_line}\n\nChapter {i}: {section.strip()}"
        for i, section in enumerate(section_contents, start=1)
    ]
    paths = write_granular_files(OUTPUT_DIR, "textbook", granular_sections)
    print(f"Saved {len(paths)} files to {Path(OUTPUT_DIR) / 'textbooks'}/")


def generate_clinical_blog(case_name, model=None, slug="gpt_5_mini_custom", efforts=None, outline_model="gpt-5", writing_model="gpt-5-mini",
                           provider="auto", base_url=None, max_workers=4):
    """Generate FOAM-style clinical blog posts for a medical case report."""
    print(f"Processing {case_name} for clinical blog...")

    CASE_FILE_PATH = MEDICAL_CLEANED_DIR / f"{case_name}.txt"
    OUTPUT_DIR = utils.explanations_dir('medical', slug, case_name, root=str(DATA_ROOT))

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(CASE_FILE_PATH, 'r') as f:
        case_content = f.read()
    case_title = extract_case_title(case_content, case_name)

    # --- 1. Generate blog post ideas ---
    print("Generating blog post ideas...")
    prompt_blog_ideas = {
        'system': """### Instructions
You are a physician who writes for a clinical education blog (like EMCrit, Life in the Fast Lane, or Rebel EM). Based on the provided clinical case report, generate 4 to 6 blog posts that would be interesting to fellow clinicians. Each should focus on a different clinical teaching point from the case.

Blog ideas should:
- Analyze the clinical significance of the case, or discuss diagnostic or management tensions, or consider how it fits into broader clinical practice trends, or any other aspect of the case that is interesting and relevant to clinicians and educators
- Be as mutually exclusive and non-redundant as possible in terms of the medical issues they cover, while covering all important aspects as much as possible

For each blog idea, provide:
- A `title` (catchy but clinical)
- A brief `description` of what the blog post will cover

### Output Format
Provide the output as a JSON object with a single key "blogs", which is a list of blog objects. Each blog object must have:
- "title": A string for the title of the blog post
- "description": A string describing the blog post's content""",
        'user': f"### Clinical Case Report\n{case_content}"
    }

    response_blog_ideas_str = utils.query_llm(
        prompt_blog_ideas,
        model=model or outline_model,
        reasoning_effort=efforts["ideas"],
        system_prompt_included=True,
        return_json=True,
        max_tokens=MAX_TOKENS,
        provider=provider,
        base_url=base_url,
    )

    if response_blog_ideas_str is None:
        print(f"WARNING: LLM returned None for blog ideas on {case_name}, skipping")
        return

    outline_log_path = os.path.join(OUTPUT_DIR, "blog_outline.json")
    with open(outline_log_path, 'w') as f:
        f.write(response_blog_ideas_str)
    print(f"Saved blog outline to {outline_log_path}")

    if not (response_blog_ideas_str or "").strip():
        print(f"WARNING: empty blog ideas for {case_name}; skipping blog view")
        return
    blogs_data = json.loads(response_blog_ideas_str)
    # Tolerate schema drift: list may sit under "blogs", at top level, or another key.
    blogs = utils.extract_dict_list(blogs_data, preferred_key='blogs')
    blogs = [b for b in blogs if b.get('title') and b.get('description')]
    if not blogs:
        print(f"WARNING: no valid blog ideas for {case_name}; skipping blog view")
        return
    print(f"Parsed {len(blogs)} blog post ideas.")

    # --- 2. Write each blog post in parallel ---
    print("Writing blog posts...")

    def write_blog(blog_info):
        title = blog_info['title']
        description = blog_info['description']
        print(f"Writing blog: {title[:50]}...")

        prompt_blog_content = {
            'system': """You will be given a clinical case report and a blog post idea. Write a clinical education blog post in the style of FOAM (Free Open Access Medical education) blogs like EMCrit, Life in the Fast Lane, or Rebel EM.

As you write the blog post:
- Write in a conversational but clinically rigorous tone
- Lead with what makes this case interesting or surprising
- Include clinical pearls and take-home points
- Write in prose paragraphs, use bullet points only for key take-home messages, but stay concise and to the point
- Stay grounded in the case report — do not fabricate clinical details

Start with the blog title as a markdown header. Use '##' for sections.""",
            'user': f"""### Clinical Case Report
{case_content}

### Blog Post to Write
Title: {title}
Description: {description}"""
        }

        blog_content = utils.query_llm(
            prompt_blog_content,
            model=model or writing_model,
            reasoning_effort=efforts["post"],
            system_prompt_included=True,
            max_tokens=MAX_TOKENS,
            provider=provider,
            base_url=base_url,
        )
        return blog_content

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        blog_contents = list(tqdm(executor.map(write_blog, blogs), total=len(blogs), desc="Writing blog posts"))

    # --- 3. Concatenate and save ---
    # Use blogs.txt filename for compatibility with data_preparation.py
    print("Assembling blog posts...")
    full_blogs_content = "\n\n\n\n".join(blog_contents)
    full_blogs_content = f"Title: Blog about the case report: \"{case_title}\"\n\n{full_blogs_content}"

    output_file = os.path.join(OUTPUT_DIR, "blogs.txt")
    with open(output_file, 'w') as f:
        f.write(full_blogs_content)

    print(f"Saved blog posts to {output_file}")

    title_line = f"Title: Blog about the case report: \"{case_title}\""
    granular_blogs = [
        f"{title_line}\n\n{blog.strip()}"
        for blog in blog_contents
    ]
    paths = write_granular_files(OUTPUT_DIR, "blog", granular_blogs)
    print(f"Saved {len(paths)} files to {Path(OUTPUT_DIR) / 'blogs'}/")


PART_GENERATORS = {
    "textbook": generate_case_textbook,
    "qa": generate_teaching_qa,
    "blog": generate_clinical_blog,
}
DEFAULT_PART_ORDER = ["qa", "textbook", "blog"]


def resolve_parts(parts):
    if not parts or "all" in parts:
        return DEFAULT_PART_ORDER
    return parts


def process_cases(case_names=None, parts=None, model=None, model_slug=None,
                  reasoning_effort=None, outline_model="gpt-5", writing_model="gpt-5-mini",
                  provider="auto", base_url=None, max_workers=4):
    if case_names is None:
        case_names = [path.stem for path in MEDICAL_CLEANED_DIR.glob("*.txt")]

    # Per-step reasoning efforts: the 'custom' profile by default, or a uniform level.
    efforts = utils.load_reasoning_efforts('medical', override=reasoning_effort)
    mode = reasoning_effort or "custom"

    # Folder name suffixed by the reasoning-effort mode. When --model is set it names both
    # tiers; otherwise the slug is named after the writing-tier model (default gpt-5-mini
    # -> 'gpt_5_mini'; gpt-5.4-mini -> 'gpt_5_4_mini'). Explicit --model_slug wins verbatim.
    base = utils.model_slug(model) if model else utils.model_slug(writing_model)
    slug = model_slug if model_slug else f"{base}_{mode}"
    print(f"Writing explanations under slug='{slug}' "
          f"(outline_model={model or outline_model}, writing_model={model or writing_model}, "
          f"reasoning_effort mode={mode})")

    selected_parts = resolve_parts(parts)
    failures = []
    for case_name in case_names:
        for part in selected_parts:
            view = "stackexchange" if part == "qa" else part
            item_dir = utils.explanations_dir('medical', slug, case_name, root=str(DATA_ROOT))
            if manifest_valid(item_dir, view):
                print(f"Skipping manifest-validated {case_name}/{view}")
                continue
            try:
                PART_GENERATORS[part](
                    case_name, model=model, slug=slug, efforts=efforts[part],
                    outline_model=outline_model, writing_model=writing_model,
                    provider=provider, base_url=base_url, max_workers=max_workers,
                )
                record_validated_view(item_dir, view, {"model": model or writing_model,
                    "provider": provider, "reasoning_effort": mode})
            except Exception as exc:
                failures.append(f"{case_name}/{view}: {exc}")
                print(f"ERROR: failed {case_name}/{view}: {exc}", file=sys.stderr)
    if failures:
        raise RuntimeError("Incomplete multiview run:\n" + "\n".join(failures))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', nargs='+', help='Case names to process. If not provided, all cases are processed.')
    parser.add_argument(
        '--parts',
        nargs='+',
        choices=['textbook', 'qa', 'blog', 'all'],
        default=['all'],
        help='Pipeline parts to run. Default: all.'
    )
    parser.add_argument('--model', default=None, help='Override generator model for ALL steps (both tiers). If set, takes precedence over --outline_model/--writing_model.')
    parser.add_argument('--outline_model', default='gpt-5', help='Model for outline-tier steps (questions/outline/blog-ideas). Default: gpt-5.')
    parser.add_argument('--writing_model', default='gpt-5-mini', help='Model for writing-tier steps (answers/section/post). Also names the output slug. Default: gpt-5-mini.')
    parser.add_argument('--model_slug', default=None, help='Override the output subfolder name verbatim (no reasoning-effort suffix). Defaults to <writing_model_slug>_<mode>, e.g. gpt_5_mini_custom.')
    parser.add_argument('--provider', choices=sorted(utils.VALID_LLM_PROVIDERS), default='auto', help='LLM backend. Use vllm with --base_url for a self-hosted server.')
    parser.add_argument('--base_url', default=None, help='OpenAI-compatible API base URL, including /v1. Falls back to VLLM_BASE_URL for provider=vllm.')
    parser.add_argument('--max_workers', type=int, default=4, help='Maximum concurrent generation requests per view.')
    parser.add_argument('--reasoning_effort', choices=['low', 'medium', 'high'], default=None,
        help='Force a uniform reasoning effort for all outline+writing steps (slug suffix _low/_medium/_high). '
             'Default (unset) uses the per-step "custom" profile from reasoning_effort.json (slug suffix _custom).')
    args = parser.parse_args()
    if args.max_workers < 1:
        parser.error('--max_workers must be at least 1')
    process_cases(
        args.cases, args.parts, model=args.model, model_slug=args.model_slug,
        reasoning_effort=args.reasoning_effort, outline_model=args.outline_model,
        writing_model=args.writing_model, provider=args.provider,
        base_url=args.base_url, max_workers=args.max_workers,
    )
