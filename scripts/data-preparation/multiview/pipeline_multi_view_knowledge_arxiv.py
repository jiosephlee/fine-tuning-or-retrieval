import os
import json
import re
import sys
from pathlib import Path
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
# Add the project root to the path to allow importing utils
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "data"
ARXIV_CLEANED_DIR = DATA_ROOT / "arxiv" / "cleaned"
sys.path.insert(0, str(PROJECT_ROOT))
import utils.utils as utils
from utils.granular_outputs import write_granular_files
from importlib import reload
reload(utils)

# Per-generation output budget. Defaults to the historical 32k cap; override via
# MULTIVIEW_MAX_TOKENS to request longer completions (clamped to fit the served
# context window by utils._create_vllm_completion).
MAX_TOKENS = int(os.environ.get("MULTIVIEW_MAX_TOKENS", "32768"))


def extract_paper_title(paper_content, paper_name):
    """Return the paper's actual title from its \\title{...} block, falling back to the
    filename (paper_name) if none is found. Mirrors pipeline_diverse_views.py."""
    match = re.search(r'\\title\{(.*?)\}', paper_content, re.DOTALL)
    if match:
        title = match.group(1).strip().replace('\n', ' ')
        # Collapse internal whitespace runs left by multi-line \title blocks.
        title = re.sub(r'\s+', ' ', title)
        if title:
            return title
    return paper_name

def generate_stack_exchange_knowledge(paper_name, model=None, slug="gpt_5_mini_custom",
                                      efforts=None, outline_model="gpt-5", writing_model="gpt-5-mini",
                                      provider="auto", base_url=None,
                                      max_workers=4):
    """Process a single paper to generate Stack Exchange style Q&A pairs."""
    print(f"Processing {paper_name}...")

    PAPER_FILE_PATH = ARXIV_CLEANED_DIR / f"{paper_name}.tex"
    OUTPUT_DIR = utils.explanations_dir('arxiv', slug, paper_name, root=str(DATA_ROOT))

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Reading paper from: {PAPER_FILE_PATH}")
    with open(PAPER_FILE_PATH, 'r') as f:
        paper_content = f.read()

    # --- 2. Generate Stack Exchange style questions ---
    print("Generating student questions about the paper...")
    prompt_questions = {
        'system': """You are a confused student reading this research paper. You are struggling with specific concepts, details, and connections in this paper. Generate a list of several Stack Exchange style questions that you would ask to clarify your understanding.

Your questions should:
- Vary in levels of understanding, from misled to profound.
- Vary in complexity, from simple to deep.
- Vary in type, from conceptual to detail-specific.
- Focus on clarifying the concepts and details of the paper. Do not ask tangential questions.

As you generate the questions, please make sure to consider the following:
- Make sure the questions are self-contained and unambiguous
- Please write any mathematical notation in LaTeX only e.g. "$x^2$" or "$\pi$". Do not use unicode mathematical characters e.g. "π".

For each question, provide:
- A `title` in Stack Exchange question format
- The `question_body` with context and what specifically you're confused about

## Example Question

"How can Transformers handle arbitrary length input?

The transformer, introduced in the paper Attention Is All You Need, is a popular new neural network architecture that is commonly viewed as an alternative to recurrent neural networks, like LSTMs and GRUs.

However, having gone through the paper, as well as several online explanations, I still have trouble wrapping my head around how they work."

### Output Format
Provide the output as a JSON object with a single key "questions", which is a list of question dictionaries.
Example:
{
  "questions": [
    {
      "title": "Why does the partition function cancel out in DPO derivation?",
      "question_body": "I'm reading the DPO paper and I understand that they start with the KL-regularized objective, but I'm confused about how the partition function Z(x) cancels out when they move to pairwise preferences. Can someone explain this step intuitively?",
    }
  ]
}""",
        'user': f"### Research Paper\n{paper_content}"
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

    # Save the generated questions outline
    outline_log_path = os.path.join(OUTPUT_DIR, "stack_exchange_outline.json")
    with open(outline_log_path, 'w') as f:
        f.write(response_questions_str)
    print(f"Saved Stack Exchange outline to {outline_log_path}")

    # Parse the questions response
    if not (response_questions_str or "").strip():
        print(f"WARNING: empty questions response for {paper_name}; skipping stack exchange view")
        return
    questions_data = json.loads(response_questions_str)
    questions = questions_data.get('questions') if isinstance(questions_data, dict) else None
    if not questions:
        print(f"WARNING: no 'questions' key for {paper_name}; skipping stack exchange view")
        return
    # json_object mode guarantees valid JSON but not the requested schema; smaller
    # models occasionally omit a required key. Drop malformed entries so one bad
    # question doesn't KeyError the whole run.
    questions = [q for q in questions if isinstance(q, dict) and q.get('title') and q.get('question_body')]

    print(f"Generated {len(questions)} questions")

    filtered_questions = questions

    # --- 3. Generate answers for each question ---
    print("Generating answers for filtered questions...")

    def generate_answer(question):
        """Generates an answer for a single question."""
        print(f"Processing question: {question['title'][:50]}...")
        
        prompt_answer = {
            'system': """A graduate student has asked a question about a research paper. Provide a clear, detailed Stack Exchange style answer that:

- Thoroughly addresses their question 
- Don't make it too lengthy; it should be concise and to the point like a Stack Exchange answer
- Write in prose rather than structured bullet points in one cohesive answer
- Provides intuitive explanations alongside technical details
- Connects to broader concepts when relevant
- Is educational and accessible

Please write any mathematical notation in LaTeX only e.g. "$x^2$" or "$\pi$". Do not use unicode mathematical characters e.g. "π". Also, please make sure that your answer is grounded in the paper; do not provide any information that is inconsistent with the paper.

Again, please write all math in LaTeX.

Format your response as a comprehensive Stack Exchange answer.

### Example

Question:
"I know that in the math on which the transformer is based there is no restriction on the length of input. But I still can’t understand why we should fix it in the frameworks (PyTorch). Because of this problem Transformer-XL has been created.

Can you explain to me where this problem is hiding, please?"

Answer:
"The restriction in the maximum length of the transformer input is due to the needed amount of memory to compute the self-attention over it.

The amount of memory needed by the self-attention in the Transformer is quadratic on the length of the input. This means that increasing the maximum length of the input, increases drastically the needed memory for self-attention. The maximum length is that which makes the model use up the whole memory of the GPU for at least one sentence (once the other elements of the model are also taken into account, like the embeddings which take a lot of memory).

Transformer-XL is certainly a way to take into account as much context as possible in language modeling (its role is analogous to truncated back-propagation through time in LSTM language models). However, the gradients are not propagated through the attention over the memory segment, only through the current segment.

There have been several architectural attempts to reduce the amount of memory needed by transformers, like using locality-constraints in the attention (Dynamic Convolutions model) or using locality-sensitive hashing (Reformer model).

There have been other implementation attempts, like gradient checkpointing(e.g. this), which is a general technique to run computations that don't fit at once in the GPU memory"
""",
            'user': f"""### Question Title
{question['title']}

### Question Body
{question['question_body']}

### Research Paper
{paper_content}"""
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
        qa_pairs = list(tqdm(executor.map(generate_answer, filtered_questions), total=len(filtered_questions), desc="Generating Stack Exchange answers"))

    # --- 4. Refine answers to ensure correct LaTeX formatting ---
    print("Refining answers to ensure correct LaTeX formatting...")

    def refine_answer_latex(qa_pair):
        """Refines an answer to fix LaTeX formatting."""
        print(f"Refining LaTeX for answer to: {qa_pair['title'][:50]}...")
        
        prompt_refine = {
            'system': """You will be given a text. Your only task is to correct any mathematical notation inside it to be valid LaTeX. You must not change any other part of the text.
    - Convert unicode math characters like 'π' to their LaTeX equivalent '$\\pi$'.
    - Ensure all mathematical expressions are enclosed in '$...$' for inline math or '$$...$$' for display math.
    - Return the full, corrected text.
    """,
            'user': f"{qa_pair['answer']}"
        }
        
        refined_answer_text = utils.query_llm(
            prompt_refine,
            model=model or writing_model,
            reasoning_effort=efforts["refine"],
            system_prompt_included=True,
            max_tokens=MAX_TOKENS,
            provider=provider,
            base_url=base_url,
        )

        # Never let an empty/failed refinement destroy a valid answer.
        if refined_answer_text and refined_answer_text.strip():
            qa_pair['answer'] = refined_answer_text
        return qa_pair

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        qa_pairs = list(tqdm(executor.map(refine_answer_latex, qa_pairs), total=len(qa_pairs), desc="Refining LaTeX"))

    # --- 5. Create single Stack Exchange style explanation file ---
    print("Creating Stack Exchange explanation file...")

    paper_title = extract_paper_title(paper_content, paper_name)
    stackexchange_content = f'\\title{{Stack Exchange of the Paper: "{paper_title}"}}\n\n'
    for qa in qa_pairs:
        stackexchange_content += f"### {qa['title']}\nQuestion:\n{qa['question']}\nAnswer:\n{qa['answer']}\n\n"

    # Save all QA pairs in single file
    output_file = os.path.join(OUTPUT_DIR, "stackexchange.txt")
    with open(output_file, 'w') as f:
        f.write(stackexchange_content)

    print(f"Saved all Q&A pairs to {output_file}")
    title_line = f'\\title{{Stack Exchange of the Paper: "{paper_title}"}}'
    granular_qas = [
        f"{title_line}\n\n### {qa['title']}\nQuestion:\n{qa['question']}\nAnswer:\n{qa['answer']}"
        for qa in qa_pairs
    ]
    paths = write_granular_files(OUTPUT_DIR, "stackexchange", granular_qas)
    print(f"Saved {len(paths)} files to {Path(OUTPUT_DIR) / 'stackexchange'}/")

def generate_textbook_knowledge(paper_name, model=None, slug="gpt_5_mini_custom",
                                efforts=None, outline_model="gpt-5", writing_model="gpt-5-mini",
                                provider="auto", base_url=None,
                                max_workers=4):
    """Process a single paper to generate a textbook-style explanation."""
    print(f"Processing {paper_name} for textbook generation...")

    PAPER_FILE_PATH = ARXIV_CLEANED_DIR / f"{paper_name}.tex"
    OUTPUT_DIR = utils.explanations_dir('arxiv', slug, paper_name, root=str(DATA_ROOT))

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Reading paper from: {PAPER_FILE_PATH}")
    with open(PAPER_FILE_PATH, 'r') as f:
        paper_content = f.read()

    # --- 1. Generate textbook outline ---
    print("Generating textbook outline...")
    prompt_outline = {
        'system': """### Instructions
You will be given a research paper and your task is to create a detailed outline for a textbook that comprehensively explains the given research paper. But, it should go beyond mere explaining, and be a proper pedagogical textbook that aims to fully educate the reader on what the paper is about. The textbook should be aimed at college students who have a basic understanding of machine learning.

The outline should:
- Break down the paper into coherent chapters.
- For each chapter, provide a:
    - title
    - description
    - list of subtopics to cover
- Cover all key concepts, methods, and results from the paper.
- Ensure a logical flow of information, from introduction to conclusion.
- While the textbook should be comprehensive, it should also articulate and to the point. Don't create unnecessary chapters.

### Output Format
Provide the output as a JSON object with a single key "outline", which is a list of chapter objects. Each chapter object must have the following keys:
- "chapter_title": A string for the title of the chapter.
- "description": A string describing the chapter's content.
- "subtopics": A list of strings, where each string is a subtopic.
""",
        'user': f"### Research Paper\n{paper_content}"
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
    
    # Save the generated textbook outline
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
        print(f"WARNING: empty textbook outline for {paper_name}; skipping textbook view")
        return
    outline_data = _parse_outline(response_outline_str)
    outline = outline_data.get('outline') if isinstance(outline_data, dict) else None
    if not outline:
        print(f"WARNING: no 'outline' key for {paper_name}; skipping textbook view")
        return
    # Drop chapters missing a required field (schema drift under json_object mode).
    outline = [c for c in outline if isinstance(c, dict) and c.get('chapter_title')
               and c.get('description') and isinstance(c.get('subtopics'), list)]
    print(f"Parsed outline with {len(outline)} chapters.")

    # --- 2. Write each chapter in parallel ---
    print("Writing textbook chapters in parallel...")

    full_outline_text = ""
    for i, chap in enumerate(outline):
        full_outline_text += f"# Chapter {i+1}: {chap['chapter_title']}\n"
        full_outline_text += "\n"
    
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
You will be given a chapter title, description, and subtopics and, based on those topics, your job is to write a detailed, cohesive textbook chapter addressed to a college student who is learning this material for the first time. 

The chapter should be comprehensive and suitable for someone learning this material to understand research papers in the field. Don't just briefly describe the subtopics, but rather elaborate on the concepts at full length and explain them with a focus on intuition. Spell everything out clearly so there is no ambiguity. Dedicate multiple paragraphs to each subtopic but be articulate and concise when appropriate. Write in full prose, rather than bullet points. Most importantly, please make sure that your chapter is grounded in the paper; do not provide any information or details that is not from the paper.

Start with the chapter title in the first line. Separate each subtopic with a section header "#". Also, please write all mathematical notation in LaTeX only e.g. "$x^2$" or "$\pi$". Do not use unicode mathematical characters e.g. "π". Again, PLEASE write all math in LaTeX.""",
            'user': f"""### Research Paper
{paper_content}

### Chapter Outline to Write
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
    print("Assembling textbook...")
    full_textbook_content = "\n\n".join(chapter_contents)
    paper_title = extract_paper_title(paper_content, paper_name)
    full_textbook = f'\\title{{A Textbook about the Paper: "{paper_title}"}}\n\n{full_textbook_content}'
    
    output_file = os.path.join(OUTPUT_DIR, "textbook.txt")
    with open(output_file, 'w') as f:
        f.write(full_textbook)
        
    print(f"Saved textbook to {output_file}")
    title_line = f'\\title{{A Textbook about the Paper: "{paper_title}"}}'
    granular_chapters = [
        f"{title_line}\n\nChapter {i}: {chapter.strip()}"
        for i, chapter in enumerate(chapter_contents, start=1)
    ]
    paths = write_granular_files(OUTPUT_DIR, "textbook", granular_chapters)
    print(f"Saved {len(paths)} files to {Path(OUTPUT_DIR) / 'textbooks'}/")

def generate_blog_knowledge(paper_name, model=None, slug="gpt_5_mini_custom",
                            efforts=None, outline_model="gpt-5", writing_model="gpt-5-mini",
                            provider="auto", base_url=None,
                            max_workers=4):
    """Process a single paper to generate related blog posts."""
    print(f"Processing {paper_name} for blog generation...")

    PAPER_FILE_PATH = ARXIV_CLEANED_DIR / f"{paper_name}.tex"
    OUTPUT_DIR = utils.explanations_dir('arxiv', slug, paper_name, root=str(DATA_ROOT))

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Reading paper from: {PAPER_FILE_PATH}")
    with open(PAPER_FILE_PATH, 'r') as f:
        paper_content = f.read()

    # --- 1. Generate blog post ideas ---
    print("Generating blog post ideas...")
    prompt_blog_ideas = {
        'system': """### Instructions
You are a creative tech blogger and content strategist. Based on the provided research paper, generate a list of a few blog posts that explain the paper in a way that is accessible to a wider audience. They should each focus on a different, main aspect of the paper.

For each blog idea, provide:
- A `title`.
- A brief `description` of what the blog post will cover.

### Output Format
Provide the output as a JSON object with a single key "blogs", which is a list of blog objects. Each blog object must have the following keys:
- "title": A string for the title of the blog post.
- "description": A string describing the blog post's content.
""",
        'user': f"### Research Paper\n{paper_content}"
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

    # Save the generated blog ideas outline
    outline_log_path = os.path.join(OUTPUT_DIR, "blog_outline.json")
    with open(outline_log_path, 'w') as f:
        f.write(response_blog_ideas_str)
    print(f"Saved blog outline to {outline_log_path}")

    if not (response_blog_ideas_str or "").strip():
        print(f"WARNING: empty blog ideas for {paper_name}; skipping blog view")
        return
    blogs_data = json.loads(response_blog_ideas_str)
    blogs = blogs_data.get('blogs') if isinstance(blogs_data, dict) else None
    if not blogs:
        print(f"WARNING: no 'blogs' key for {paper_name}; skipping blog view")
        return
    # Drop malformed blog ideas (schema drift under json_object mode).
    blogs = [b for b in blogs if isinstance(b, dict) and b.get('title') and b.get('description')]
    print(f"Parsed {len(blogs)} blog post ideas.")

    # --- 3. Write each blog post in parallel ---
    print("Writing blog posts in parallel...")

    def write_blog(blog_info):
        title = blog_info['title']
        description = blog_info['description']
        print(f"Writing blog: {title[:50]}...")

        prompt_blog_content = {
            'system': """You will be given an academic paper and a blog post idea about the paper. Write a blog post based on the blog idea.

As you write the blog post, please make sure to consider the following:
- Write in a technical blog style. It should be less formal but not too informal. It should be concise and to the point. 
- Simplify complex concepts from the paper for a broader audience.
- Write in full, complete sentences and prefer paragraphs over bullet points, but use bullet points when appropriate.
- Keep all details grounded in the paper. Do not make up any information.
- Please write any mathematical notation in LaTeX only e.g. "$x^2$" or "$\pi$". Do not use unicode mathematical characters e.g. "π". 

Your output should be the full text of the blog post, starting with the blog title as a markdown header. Use '#' to denote the blog title, '##' to denote different sections, and so on.
""",
            'user': f"""### Research Paper
{paper_content}

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

    # --- 4. Concatenate and save ---
    print("Assembling blog posts...")
    full_blogs_content = "\n\n\n\n".join(blog_contents)
    paper_title = extract_paper_title(paper_content, paper_name)
    full_blogs_content = f'\\title{{Blogs about the Paper: "{paper_title}"}}\n\n{full_blogs_content}'

    output_file = os.path.join(OUTPUT_DIR, "blogs.txt")
    with open(output_file, 'w') as f:
        f.write(full_blogs_content)

    print(f"Saved blog posts to {output_file}")
    title_line = f'\\title{{Blogs about the Paper: "{paper_title}"}}'
    granular_blogs = [f"{title_line}\n\n{blog.strip()}" for blog in blog_contents]
    paths = write_granular_files(OUTPUT_DIR, "blog", granular_blogs)
    print(f"Saved {len(paths)} files to {Path(OUTPUT_DIR) / 'blogs'}/")

PART_GENERATORS = {
    "stack_exchange": generate_stack_exchange_knowledge,
    "textbook": generate_textbook_knowledge,
    "blog": generate_blog_knowledge,
}
DEFAULT_PART_ORDER = ["stack_exchange", "textbook", "blog"]


def resolve_parts(parts):
    if not parts or "all" in parts:
        return DEFAULT_PART_ORDER
    return parts


def process_papers(paper_names=None, parts=None, model=None, model_slug=None,
                   reasoning_effort=None, outline_model="gpt-5", writing_model="gpt-5-mini",
                   provider="auto", base_url=None, max_workers=4):
    if paper_names is None:
        paper_names = [path.stem for path in ARXIV_CLEANED_DIR.glob("*.tex")]

    # Per-step reasoning efforts: the 'custom' profile by default, or a uniform level.
    efforts = utils.load_reasoning_efforts('arxiv', override=reasoning_effort)
    mode = reasoning_effort or "custom"

    # Folder name for outputs, suffixed by the reasoning-effort mode. When --model is set it
    # names both tiers; otherwise the slug is named after the writing-tier model (so the
    # default gpt-5-mini stays 'gpt_5_mini' and e.g. gpt-5.4-mini becomes 'gpt_5_4_mini').
    # An explicit --model_slug is used verbatim (no suffix).
    base = utils.model_slug(model) if model else utils.model_slug(writing_model)
    slug = model_slug if model_slug else f"{base}_{mode}"
    print(f"Writing explanations under slug='{slug}' "
          f"(outline_model={model or outline_model}, writing_model={model or writing_model}, "
          f"reasoning_effort mode={mode})")

    selected_parts = resolve_parts(parts)
    for paper_name in paper_names:
        for part in selected_parts:
            PART_GENERATORS[part](
                paper_name,
                model=model,
                slug=slug,
                efforts=efforts[part],
                outline_model=outline_model,
                writing_model=writing_model,
                provider=provider,
                base_url=base_url,
                max_workers=max_workers,
            )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--papers', nargs='+', help='Paper names to process. If not provided, all papers are processed.')
    parser.add_argument('--parts', nargs='+', choices=['stack_exchange', 'textbook', 'blog', 'all'], default=['all'], help='Pipeline parts to run. Default: all.')
    parser.add_argument('--model', default=None, help='Override generator model for ALL steps (both tiers). If set, takes precedence over --outline_model/--writing_model.')
    parser.add_argument('--outline_model', default='gpt-5', help='Model for outline-tier steps (questions/outline/blog-ideas). Default: gpt-5.')
    parser.add_argument('--writing_model', default='gpt-5-mini', help='Model for writing-tier steps (answers/chapter/post/refine). Also names the output slug. Default: gpt-5-mini.')
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
    process_papers(
        args.papers, args.parts, model=args.model, model_slug=args.model_slug,
        reasoning_effort=args.reasoning_effort, outline_model=args.outline_model,
        writing_model=args.writing_model, provider=args.provider,
        base_url=args.base_url, max_workers=args.max_workers,
    )
