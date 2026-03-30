import os
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
sys.path.append('../../')
import utils.utils as utils
from importlib import reload
reload(utils)


def generate_teaching_qa(case_name):
    """Generate NEJM Clinical Pearls & Morning Report style teaching Q&A for a medical case report."""
    print(f"Processing {case_name} for teaching Q&A...")

    CASE_FILE_PATH = f'../../data/medical/cleaned/{case_name}.txt'
    OUTPUT_DIR = f"../../data/medical/explanations/{case_name}/"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(CASE_FILE_PATH, 'r') as f:
        case_content = f.read()

    # --- 1. Generate teaching questions ---
    print("Generating clinical teaching questions...")
    prompt_questions = {
        'system': """You are a senior attending physician preparing a teaching session based on this clinical case report. You are creating a set of questions for residents and medical students in the style of NEJM Clinical Pearls & Morning Reports.

Your questions should:
- Cover the key clinical reasoning steps in this case (differential diagnosis, workup, diagnosis, management)
- Vary in difficulty from straightforward recall to deeper clinical reasoning
- Include questions about the underlying pathophysiology and mechanism of disease
- Test understanding of diagnostic criteria and when to suspect this condition
- Address management decisions and their rationale
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
        model='gpt-5',
        reasoning_effort="medium",
        system_prompt_included=True,
        return_json=True,
        max_tokens=10000
    )

    if response_questions_str is None:
        print(f"WARNING: LLM returned None for teaching Q&A questions on {case_name}, skipping")
        return

    outline_log_path = os.path.join(OUTPUT_DIR, "stack_exchange_outline.json")
    with open(outline_log_path, 'w') as f:
        f.write(response_questions_str)
    print(f"Saved teaching Q&A outline to {outline_log_path}")

    questions_data = json.loads(response_questions_str)
    questions = questions_data['questions']
    print(f"Generated {len(questions)} teaching questions")

    # --- 2. Generate answers for each question ---
    print("Generating answers...")

    def generate_answer(question):
        print(f"Processing question: {question['question'][:50]}...")

        prompt_answer = {
            'system': """You are a senior attending physician answering a clinical teaching question during a morning report or case conference. Provide a clear, educational answer in the style of NEJM Clinical Pearls & Morning Reports.

Your answer should:
- Be concise and to the point, as in a real teaching session
- Provide the clinical reasoning, not just the fact
- Connect to broader clinical principles when relevant
- Be written in prose, not bullet points
- Stay grounded in the case report — do not introduce information inconsistent with the case

Format your response as a direct, authoritative clinical teaching answer.""",
            'user': f"""### Clinical Teaching Question
Category: {question['category']}
Question: {question['question']}

### Clinical Case Report
{case_content}"""
        }

        answer_text = utils.query_llm(
            prompt_answer,
            model='gpt-5-mini',
            reasoning_effort="low",
            system_prompt_included=True,
            max_tokens=2000
        )

        return {
            'category': question['category'],
            'question': question['question'],
            'answer': answer_text
        }

    with ThreadPoolExecutor() as executor:
        qa_pairs = list(tqdm(executor.map(generate_answer, questions), total=len(questions), desc="Generating teaching answers"))

    # --- 3. Create single file ---
    # Use stackexchange.txt filename for compatibility with data_preparation.py
    print("Creating teaching Q&A file...")

    content = f"Clinical Pearls & Morning Report: {case_name}\n\n"
    for qa in qa_pairs:
        content += f"### [{qa['category']}] {qa['question']}\n\n"
        content += f"{qa['answer']}\n\n"

    output_file = os.path.join(OUTPUT_DIR, "stackexchange.txt")
    with open(output_file, 'w') as f:
        f.write(content)

    print(f"Saved teaching Q&A to {output_file}")


def generate_case_textbook(case_name):
    """Generate a case-based medical textbook chapter (Case Files / Harrison's style) for a clinical case report."""
    print(f"Processing {case_name} for case-based textbook chapter...")

    CASE_FILE_PATH = f'../../data/medical/cleaned/{case_name}.txt'
    OUTPUT_DIR = f"../../data/medical/explanations/{case_name}/"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(CASE_FILE_PATH, 'r') as f:
        case_content = f.read()

    # Single call — one case = one chapter, no need for outline → parallel expansion
    print("Generating case-based textbook chapter...")
    prompt = {
        'system': """### Instructions
You will be given a clinical case report. Write a single, comprehensive textbook chapter in the style of Lange Case Files or Harrison's Principles of Internal Medicine that uses this case to teach the underlying medicine.

This is NOT a StatPearls reference article or a case summary. It is a pedagogical textbook chapter that teaches through the case, like what a medical student would read to deeply understand the clinical problem.

The chapter should cover:
- Clinical approach: what this presentation suggests, how to build a differential, what findings narrow it
- Pathophysiology and disease mechanism in depth — teach the biology, not just name the disease
- Diagnostic workup: what tests to order, what criteria to apply, and why
- Management principles and their rationale
- Clinical pearls and take-home teaching points

Writing guidelines:
- Write in authoritative but accessible medical prose, aimed at medical students and residents
- Teach the underlying medicine through the case — the case is the vehicle, not the destination
- Write in full prose paragraphs — this is a textbook, not an outline or reference card
- Be comprehensive, dedicating multiple paragraphs to pathophysiology and clinical reasoning
- Stay grounded in the case report for case-specific claims, but explain broader medical context

Start with the chapter title as a '#' header. Use '##' for major sections within the chapter.""",
        'user': f"### Clinical Case Report\n{case_content}"
    }

    chapter_content = utils.query_llm(
        prompt,
        model='gpt-5',
        system_prompt_included=True,
        max_tokens=8000
    )

    if chapter_content is None:
        print(f"WARNING: LLM returned None for textbook chapter on {case_name}, skipping")
        return

    output_file = os.path.join(OUTPUT_DIR, "textbook.txt")
    with open(output_file, 'w') as f:
        f.write(chapter_content)

    print(f"Saved textbook chapter to {output_file}")


def generate_clinical_blog(case_name):
    """Generate FOAM-style clinical blog posts for a medical case report."""
    print(f"Processing {case_name} for clinical blog...")

    CASE_FILE_PATH = f'../../data/medical/cleaned/{case_name}.txt'
    OUTPUT_DIR = f"../../data/medical/explanations/{case_name}/"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(CASE_FILE_PATH, 'r') as f:
        case_content = f.read()

    # --- 1. Generate blog post ideas ---
    print("Generating blog post ideas...")
    prompt_blog_ideas = {
        'system': """### Instructions
You are a physician who writes for a clinical education blog (like EMCrit, Life in the Fast Lane, or Rebel EM). Based on the provided clinical case report, generate a list of a few blog posts that would be interesting to fellow clinicians. Each should focus on a different clinical teaching point from the case.

Blog ideas should:
- Highlight what makes this case surprising, unusual, or instructive
- Focus on clinical pearls, diagnostic pitfalls, or management controversies
- Be the kind of post an emergency physician or internist would share with colleagues

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
        model='gpt-5',
        reasoning_effort="medium",
        system_prompt_included=True,
        return_json=True,
        max_tokens=2000
    )

    if response_blog_ideas_str is None:
        print(f"WARNING: LLM returned None for blog ideas on {case_name}, skipping")
        return

    outline_log_path = os.path.join(OUTPUT_DIR, "blog_outline.json")
    with open(outline_log_path, 'w') as f:
        f.write(response_blog_ideas_str)
    print(f"Saved blog outline to {outline_log_path}")

    blogs_data = json.loads(response_blog_ideas_str)
    blogs = blogs_data['blogs']
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
- Write in prose paragraphs, use bullet points only for key take-home messages
- Keep it concise — FOAM posts are typically focused and punchy
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
            model='gpt-5-mini',
            reasoning_effort="low",
            system_prompt_included=True,
            max_tokens=4000
        )
        return blog_content

    with ThreadPoolExecutor() as executor:
        blog_contents = list(tqdm(executor.map(write_blog, blogs), total=len(blogs), desc="Writing blog posts"))

    # --- 3. Concatenate and save ---
    # Use blogs.txt filename for compatibility with data_preparation.py
    print("Assembling blog posts...")
    full_blogs_content = "\n\n\n\n".join(blog_contents)
    full_blogs_content = f"Clinical Education Blog Posts: {case_name}\n\n{full_blogs_content}"

    output_file = os.path.join(OUTPUT_DIR, "blogs.txt")
    with open(output_file, 'w') as f:
        f.write(full_blogs_content)

    print(f"Saved blog posts to {output_file}")


def process_cases(case_names=None):
    if case_names is None:
        input_dir = "../../data/medical/cleaned/"
        case_names = [os.path.splitext(f)[0] for f in os.listdir(input_dir) if f.endswith('.txt')]

    for case_name in case_names:
        generate_teaching_qa(case_name)
        generate_case_textbook(case_name)
        generate_clinical_blog(case_name)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', nargs='+', help='Case names to process. If not provided, all cases are processed.')
    args = parser.parse_args()
    process_cases(args.cases)
