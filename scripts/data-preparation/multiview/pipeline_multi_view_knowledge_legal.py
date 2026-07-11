import os
import json
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "data"
LEGAL_CLEANED_DIR = DATA_ROOT / "legal" / "cleaned"
sys.path.insert(0, str(PROJECT_ROOT))
import utils.utils as utils
from utils.granular_outputs import write_granular_files
from importlib import reload
reload(utils)

# Per-generation output budget. Defaults to the historical 32k cap; override via
# MULTIVIEW_MAX_TOKENS to request longer completions (clamped to fit the served
# context window by utils._create_vllm_completion).
MAX_TOKENS = int(os.environ.get("MULTIVIEW_MAX_TOKENS", "32768"))


def extract_case_title(case_content, case_name):
    """Return the source opinion title using the cleaned legal document format."""
    for line in case_content.splitlines():
        if line.startswith("Title:"):
            return line.split(":", 1)[1].strip()
    return case_name.replace("_", " ")


def generate_legal_qa(case_name, model=None, slug="gpt_5_mini_custom", efforts=None, outline_model="gpt-5", writing_model="gpt-5-mini",
                      provider="auto", base_url=None, max_workers=4):
    """Generate Law Stack Exchange style Q&A for a court opinion."""
    print(f"Processing {case_name} for legal Q&A...")

    CASE_FILE_PATH = LEGAL_CLEANED_DIR / f"{case_name}.txt"
    OUTPUT_DIR = utils.explanations_dir('legal', slug, case_name, root=str(DATA_ROOT))

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(CASE_FILE_PATH, 'r') as f:
        case_content = f.read()
    case_title = extract_case_title(case_content, case_name)

    # --- 1. Generate questions ---
    print("Generating Law Stack Exchange questions...")
    prompt_questions = {
        'system': """You are a confused law student reading this court opinion. You are struggling with specific legal concepts, procedural details, and the court's reasoning. Generate 4-8 Law Stack Exchange style questions that you would ask to clarify your understanding.

Your questions should:
- Vary in complexity, from basic procedural questions to deep doctrinal analysis
- Vary in topics, from the court's reasoning of legal principles to the facts of the case
- The questions should be as mutually exclusive and non-redundant as possible in terms of the legal issues they cover, while covering all the key legal issues of the case as much as possible

For each question, provide:
- A `title` in Stack Exchange question format
- The `question_body` with context about what specifically you're confused about

### Output Format
Provide the output as a JSON object with a single key "questions", which is a list of question dictionaries.
Example:
{
  "questions": [
    {
      "title": "Why did the court apply strict scrutiny instead of rational basis review?",
      "question_body": "I'm reading this opinion and the court applies strict scrutiny to the challenged statute. But I thought economic regulations only get rational basis review. Can someone explain what triggered the higher standard here?"
    }
  ]
}""",
        'user': f"### Court Opinion\n{case_content}"
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

    outline_log_path = os.path.join(OUTPUT_DIR, "stack_exchange_outline.json")
    with open(outline_log_path, 'w') as f:
        f.write(response_questions_str)
    print(f"Saved legal Q&A outline to {outline_log_path}")

    if not (response_questions_str or "").strip():
        print(f"WARNING: empty questions response for {case_name}; skipping legal Q&A view")
        return
    questions_data = json.loads(response_questions_str)
    # json_object mode guarantees valid JSON but not the requested schema; models
    # occasionally omit the "questions" key or return the list at top level. Drop
    # malformed entries so one bad response doesn't KeyError the whole run.
    questions = utils.extract_dict_list(questions_data, preferred_key='questions')
    questions = [q for q in questions if q.get('title') and q.get('question_body')]
    if not questions:
        print(f"WARNING: no valid legal questions for {case_name}; skipping legal Q&A view")
        return
    print(f"Generated {len(questions)} questions")

    # --- 2. Generate answers ---
    print("Generating answers...")

    def generate_answer(question):
        print(f"Processing question: {question['title'][:50]}...")

        prompt_answer = {
            'system': """A law student has asked a question about a court opinion on Law Stack Exchange. Provide a clear, well-reasoned answer in the style of a top-voted Stack Exchange response.

Your answer should:
- Directly address the legal question
- Explain the relevant legal doctrine or standard
- Reference the opinion to support your explanation; stay grounded in the opinion; do not introduce holdings or facts not in the opinion
- Be written in prose, concise, and to the point
- Be accessible to a law student while remaining legally precise

Format your response as a Stack Exchange answer.""",
            'user': f"""### Question Title
{question['title']}

### Question Body
{question['question_body']}

### Court Opinion
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
            'title': question['title'],
            'question': question['question_body'],
            'answer': answer_text
        }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        qa_pairs = list(tqdm(executor.map(generate_answer, questions), total=len(questions), desc="Generating legal answers"))

    # --- 3. Create single file ---
    # Use stackexchange.txt filename for compatibility with data_preparation.py
    print("Creating legal Q&A file...")

    content = f"Title: Stack Exchange about the opinion: \"{case_title}\"\n\n"
    for qa in qa_pairs:
        content += f"### {qa['title']}\nQuestion:\n{qa['question']}\nAnswer:\n{qa['answer']}\n\n"

    output_file = os.path.join(OUTPUT_DIR, "stackexchange.txt")
    with open(output_file, 'w') as f:
        f.write(content)

    print(f"Saved legal Q&A to {output_file}")

    title_line = f"Title: Stack Exchange about the opinion: \"{case_title}\""
    granular_qas = [
        f"{title_line}\n\n### {qa['title']}\nQuestion:\n{qa['question']}\nAnswer:\n{qa['answer']}"
        for qa in qa_pairs
    ]
    paths = write_granular_files(OUTPUT_DIR, "stackexchange", granular_qas)
    print(f"Saved {len(paths)} files to {Path(OUTPUT_DIR) / 'stackexchange'}/")


def generate_casebook_textbook(case_name, model=None, slug="gpt_5_mini_custom",
                               outline_model="gpt-5", writing_model="gpt-5-mini", efforts=None,
                               provider="auto", base_url=None, max_workers=4):
    """Generate a casebook/treatise-style textbook chapter that teaches the underlying law through a court opinion."""
    print(f"Processing {case_name} for casebook textbook...")

    CASE_FILE_PATH = LEGAL_CLEANED_DIR / f"{case_name}.txt"
    OUTPUT_DIR = utils.explanations_dir('legal', slug, case_name, root=str(DATA_ROOT))

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(CASE_FILE_PATH, 'r') as f:
        case_content = f.read()
    case_title = extract_case_title(case_content, case_name)

    # --- 1. Generate textbook outline ---
    print("Generating casebook textbook outline...")
    prompt_outline = {
        'system': """### Instructions
You will be given a court opinion. Your task is to create a detailed outline for a textbook-style explanation that comprehensively explains the given opinion. It should go beyond merely summarizing the holding, but it should remain centered on helping a law student fully understand this specific opinion.

The textbook should:
- Contain exactly 3 to 6 chapters
- Break down the opinion into coherent chapters that follow the opinion's legal and procedural logic
- Plan chapters to be mutually exclusive and non-redundant while preserving high coverage of the important facts, procedural posture, legal issues, reasoning, rules, standards, and outcome
- Cover the legal concepts and background needed to understand the opinion, but avoid broad doctrinal detours that are not necessary for understanding this case
- Explain how the court moves from facts and procedural posture to legal questions, rules, analysis, and disposition
- Ensure a logical flow from case background to issues, governing law, reasoning, holding, and significance
- Be comprehensive but concise and to the point; do not create unnecessary chapters

For each chapter, provide:
- A title
- A description of what the chapter covers
- A list of subtopics

### Output Format
Provide the output as a JSON object with a single key "outline", which is a list of chapter objects. Each chapter object must have:
- "chapter_title": A string for the title of the chapter
- "description": A string describing the chapter's content
- "subtopics": A list of strings, where each string is a subtopic""",
        'user': f"### Court Opinion\n{case_content}"
    }

    response_outline_str = utils.query_llm(
        prompt_outline,
        model=model or outline_model,
        reasoning_effort=efforts["outline"],
        system_prompt_included=True,
        return_json=True,
        max_tokens=MAX_TOKENS,
        provider=provider,
        base_url=base_url,
    )

    outline_log_path = os.path.join(OUTPUT_DIR, "textbook_outline.json")
    with open(outline_log_path, 'w') as f:
        f.write(response_outline_str)
    print(f"Saved textbook outline to {outline_log_path}")

    def _parse_outline(raw):
        pairs = []
        data = json.loads(raw, object_pairs_hook=lambda p: p)
        if isinstance(data, list):
            pairs = next((v for k, v in data if k == 'outline'), [])
        elif isinstance(data, dict):
            pairs = data.get('outline', [])
        if isinstance(pairs, list) and len(pairs) == 1 and isinstance(pairs[0], list):
            pairs = pairs[0]
        if isinstance(pairs, list) and pairs and isinstance(pairs[0], tuple):
            recovered = []
            current = None
            for key, value in pairs:
                if key == 'chapter_title':
                    if current: recovered.append(current)
                    current = {'chapter_title': value}
                elif current is not None:
                    current[key] = value
            if current: recovered.append(current)
            if recovered: return {'outline': recovered}
        return json.loads(raw)

    if not (response_outline_str or "").strip():
        print(f"WARNING: empty textbook outline for {case_name}; skipping textbook view")
        return
    outline_data = _parse_outline(response_outline_str)
    # Tolerate schema drift: chapters may sit under "outline", at top level, or another
    # key. Drop chapters missing a required field.
    outline = utils.extract_dict_list(outline_data, preferred_key='outline')
    outline = [c for c in outline if c.get('chapter_title')
               and c.get('description') and isinstance(c.get('subtopics'), list)]
    if not outline:
        print(f"WARNING: no valid textbook chapters for {case_name}; skipping textbook view")
        return
    print(f"Parsed outline with {len(outline)} chapters.")

    # --- 2. Write each chapter in parallel ---
    print("Writing textbook chapters...")

    full_outline_text = ""
    for i, chap in enumerate(outline):
        full_outline_text += f"# Chapter {i+1}: {chap['chapter_title']}\n\n"

    def write_chapter(chapter_info):
        chapter_title = chapter_info['chapter_title']
        print(f"Writing chapter: {chapter_title}...")

        chapter_outline_text = f"### Chapter Title\n\n{chapter_info['chapter_title']}\n"
        chapter_outline_text += f"## Chapter Description\n\n{chapter_info['description']}\n"
        chapter_outline_text += "## Subtopics\n"
        for sec in chapter_info['subtopics']:
            chapter_outline_text += f"- {sec}\n"

        prompt_chapter = {
            'system': """### Instructions
You will be given a chapter title, description, and subtopics for a textbook-style explanation of a court opinion. Write this chapter for a law student who is learning how to read and understand the opinion.

The chapter should:
- Explain the assigned part of the opinion clearly and pedagogically
- Teach the legal concepts, rules, standards, and procedural ideas needed to understand this part of the opinion
- Stay centered on the opinion: facts, procedural posture, issues, reasoning, holdings, and disposition
- Explain broader legal context only when it is necessary to understand the court's reasoning
- Avoid turning the chapter into a general hornbook or treatise on the area of law
- Write in full prose paragraphs (this is a textbook, not an outline)
- Be concise and to the point
- Stay grounded in the opinion; do not introduce holdings, facts, or doctrinal claims that are not supported by the opinion or necessary context

Start with the chapter title as a header. Use '#' for the chapter title and '##' for subtopic sections.""",
            'user': f"""### Court Opinion
{case_content}

### Chapter to Write
{chapter_outline_text}"""
        }

        chapter_content = utils.query_llm(
            prompt_chapter,
            model=model or writing_model,
            reasoning_effort=efforts["chapter"],
            system_prompt_included=True,
            max_tokens=MAX_TOKENS,
            provider=provider,
            base_url=base_url,
        )
        return chapter_content

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        chapter_contents = list(tqdm(executor.map(write_chapter, outline), total=len(outline), desc="Writing textbook chapters"))

    # --- 3. Concatenate and save ---
    # Use textbook.txt filename for compatibility with data_preparation.py
    print("Assembling textbook...")
    full_content = "\n\n".join(chapter_contents)
    full_textbook = f"Title: Casebook chapter about the opinion: \"{case_title}\"\n\n{full_content}"

    output_file = os.path.join(OUTPUT_DIR, "textbook.txt")
    with open(output_file, 'w') as f:
        f.write(full_textbook)

    print(f"Saved textbook to {output_file}")

    title_line = f"Title: Casebook chapter about the opinion: \"{case_title}\""
    granular_chapters = [
        f"{title_line}\n\nChapter {i}: {chapter.strip()}"
        for i, chapter in enumerate(chapter_contents, start=1)
    ]
    paths = write_granular_files(OUTPUT_DIR, "textbook", granular_chapters)
    print(f"Saved {len(paths)} files to {Path(OUTPUT_DIR) / 'textbooks'}/")


def generate_legal_blog(case_name, model=None, slug="gpt_5_mini_custom", efforts=None, outline_model="gpt-5", writing_model="gpt-5-mini",
                        provider="auto", base_url=None, max_workers=4):
    """Generate legal blog / commentary posts for a court opinion."""
    print(f"Processing {case_name} for legal blog...")

    CASE_FILE_PATH = LEGAL_CLEANED_DIR / f"{case_name}.txt"
    OUTPUT_DIR = utils.explanations_dir('legal', slug, case_name, root=str(DATA_ROOT))

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(CASE_FILE_PATH, 'r') as f:
        case_content = f.read()
    case_title = extract_case_title(case_content, case_name)

    # --- 1. Generate blog post ideas ---
    print("Generating blog post ideas...")
    prompt_blog_ideas = {
        'system': """### Instructions
You are a legal commentator who writes for a blog like SCOTUSblog, the Volokh Conspiracy, or Lawfare. Based on the provided court opinion, generate 2 to 4 blog posts that analyze different aspects of the case. Each should focus on a different angle.

Blog ideas should:
- Analyze the legal significance of the ruling, or discuss tensions in the court's reasoning or doctrinal implications, or consider how this fits into broader legal trends, or any other aspect of the case that is interesting and relevant to legal scholars and practitioners.
- Be as mutually exclusive and non-redundant as possible in terms of the legal issues they cover, while covering all important aspects as much as possible.

For each blog idea, provide:
- A `title` (analytical and engaging)
- A brief `description` of what the blog post will cover

### Output Format
Provide the output as a JSON object with a single key "blogs", which is a list of blog objects. Each blog object must have:
- "title": A string for the title of the blog post
- "description": A string describing the blog post's content""",
        'user': f"### Court Opinion\n{case_content}"
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
            'system': """You will be given a court opinion and a blog post idea. Write a legal commentary blog post in the style of SCOTUSblog, the Volokh Conspiracy, or Lawfare.

As you write the blog post:
- Write in an analytical but accessible style — authoritative but not stiff
- Lead with why this case matters, then walk through the reasoning
- Offer your own analytical perspective on the court's reasoning
- Discuss doctrinal implications and potential downstream effects
- Write in prose paragraphs, not bullet points, but stay concise and to the point
- Stay grounded in the opinion — do not fabricate holdings or misstate the court's reasoning

Start with the blog title as a markdown header. Use '##' for sections.""",
            'user': f"""### Court Opinion
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
    full_blogs_content = f"Title: Blog about the opinion: \"{case_title}\"\n\n{full_blogs_content}"

    output_file = os.path.join(OUTPUT_DIR, "blogs.txt")
    with open(output_file, 'w') as f:
        f.write(full_blogs_content)

    print(f"Saved blog posts to {output_file}")

    title_line = f"Title: Blog about the opinion: \"{case_title}\""
    granular_blogs = [
        f"{title_line}\n\n{blog.strip()}"
        for blog in blog_contents
    ]
    paths = write_granular_files(OUTPUT_DIR, "blog", granular_blogs)
    print(f"Saved {len(paths)} files to {Path(OUTPUT_DIR) / 'blogs'}/")


PART_GENERATORS = {
    "textbook": generate_casebook_textbook,
    "qa": generate_legal_qa,
    "blog": generate_legal_blog,
}
DEFAULT_PART_ORDER = ["qa", "textbook", "blog"]


def resolve_parts(parts):
    if not parts or "all" in parts:
        return DEFAULT_PART_ORDER
    return parts


def process_cases(case_names=None, parts=None, outline_model="gpt-5", writing_model="gpt-5-mini",
                  model=None, model_slug=None, reasoning_effort=None, provider="auto",
                  base_url=None, max_workers=4):
    if case_names is None:
        case_names = [path.stem for path in LEGAL_CLEANED_DIR.glob("*.txt")]

    # Per-step reasoning efforts: the 'custom' profile by default, or a uniform level.
    efforts = utils.load_reasoning_efforts('legal', override=reasoning_effort)
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
    for case_name in case_names:
        for part in selected_parts:
            if part == "textbook":
                generate_casebook_textbook(
                    case_name, model=model, slug=slug, outline_model=outline_model,
                    writing_model=writing_model, efforts=efforts[part], provider=provider,
                    base_url=base_url, max_workers=max_workers,
                )
            else:
                PART_GENERATORS[part](
                    case_name, model=model, slug=slug, efforts=efforts[part],
                    outline_model=outline_model, writing_model=writing_model,
                    provider=provider, base_url=base_url, max_workers=max_workers,
                )


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
    parser.add_argument('--outline_model', default='gpt-5', help='Model for outline-tier steps (questions/textbook-outline/blog-ideas). Default: gpt-5.')
    parser.add_argument('--writing_model', default='gpt-5-mini', help='Model for writing-tier steps (answers/chapter/post). Also names the output slug. Default: gpt-5-mini.')
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
        args.cases, args.parts, args.outline_model, args.writing_model, model=args.model,
        model_slug=args.model_slug, reasoning_effort=args.reasoning_effort,
        provider=args.provider, base_url=args.base_url,
        max_workers=args.max_workers,
    )
