import os
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
sys.path.append('../../')
import utils.utils as utils
from importlib import reload
reload(utils)


def generate_legal_qa(case_name):
    """Generate Law Stack Exchange style Q&A for a court opinion."""
    print(f"Processing {case_name} for legal Q&A...")

    CASE_FILE_PATH = f'../../data/legal/cleaned/{case_name}.txt'
    OUTPUT_DIR = f"../../data/legal/explanations/{case_name}/"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(CASE_FILE_PATH, 'r') as f:
        case_content = f.read()

    # --- 1. Generate questions ---
    print("Generating Law Stack Exchange questions...")
    prompt_questions = {
        'system': """You are a confused law student reading this court opinion. You are struggling with specific legal concepts, procedural details, and the court's reasoning. Generate a list of several Law Stack Exchange style questions that you would ask to clarify your understanding.

Your questions should:
- Vary in complexity, from basic procedural questions to deep doctrinal analysis
- Cover the key legal issues: jurisdiction, standing, standard of review, statutory interpretation, constitutional questions
- Ask about why the court reasoned a certain way, not just what it decided
- Focus on the legal principles in this opinion — do not ask tangential questions

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
        model='gpt-5',
        reasoning_effort="medium",
        system_prompt_included=True,
        return_json=True,
        max_tokens=10000
    )

    outline_log_path = os.path.join(OUTPUT_DIR, "stack_exchange_outline.json")
    with open(outline_log_path, 'w') as f:
        f.write(response_questions_str)
    print(f"Saved legal Q&A outline to {outline_log_path}")

    questions_data = json.loads(response_questions_str)
    questions = questions_data['questions']
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
- Reference specific parts of the opinion to support your explanation
- Be written in prose, concise and to the point
- Be accessible to a law student while remaining legally precise
- Stay grounded in the opinion — do not introduce holdings or facts not in the opinion

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
            model='gpt-5-mini',
            reasoning_effort="low",
            system_prompt_included=True,
            max_tokens=2000
        )

        return {
            'title': question['title'],
            'question': question['question_body'],
            'answer': answer_text
        }

    with ThreadPoolExecutor() as executor:
        qa_pairs = list(tqdm(executor.map(generate_answer, questions), total=len(questions), desc="Generating legal answers"))

    # --- 3. Create single file ---
    # Use stackexchange.txt filename for compatibility with data_preparation.py
    print("Creating legal Q&A file...")

    content = f"Law Stack Exchange Q&A: {case_name}\n\n"
    for qa in qa_pairs:
        content += f"### {qa['title']}\nQuestion:\n{qa['question']}\nAnswer:\n{qa['answer']}\n\n"

    output_file = os.path.join(OUTPUT_DIR, "stackexchange.txt")
    with open(output_file, 'w') as f:
        f.write(content)

    print(f"Saved legal Q&A to {output_file}")


def generate_casebook_textbook(case_name):
    """Generate a casebook/treatise-style textbook chapter that teaches the underlying law through a court opinion."""
    print(f"Processing {case_name} for casebook textbook...")

    CASE_FILE_PATH = f'../../data/legal/cleaned/{case_name}.txt'
    OUTPUT_DIR = f"../../data/legal/explanations/{case_name}/"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(CASE_FILE_PATH, 'r') as f:
        case_content = f.read()

    # --- 1. Generate textbook outline ---
    print("Generating casebook textbook outline...")
    prompt_outline = {
        'system': """### Instructions
You will be given a court opinion. Your task is to create a detailed outline for a casebook-style textbook chapter that uses this opinion as its centerpiece to teach the underlying area of law. This is NOT a case brief — it is a pedagogical textbook chapter in the style of a law school casebook (e.g., Chemerinsky's Constitutional Law, Prosser on Torts) or a hornbook/treatise (e.g., Wright & Miller).

The chapter should:
- Introduce the area of law and doctrinal background BEFORE discussing the case
- Explain the legal concepts, standards, and tests that the court applies
- Use the opinion as a vehicle to teach the doctrine, not just summarize the holding
- Include sections that go beyond the four corners of the opinion to explain the broader legal framework
- End with analysis of the case's significance and how it fits into the doctrinal landscape

Structure the outline as chapters that a law student would read to deeply understand this area of law through this case. For example, a chapter on a securities fraud case might cover: the history of federal fraud statutes, the elements of securities fraud, the evolution of "scheme to defraud" doctrine, the specific issue in this case, and the implications going forward.

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
        model='gpt-5',
        system_prompt_included=True,
        return_json=True,
        max_tokens=4000
    )

    outline_log_path = os.path.join(OUTPUT_DIR, "textbook_outline.json")
    with open(outline_log_path, 'w') as f:
        f.write(response_outline_str)
    print(f"Saved textbook outline to {outline_log_path}")

    outline_data = json.loads(response_outline_str)
    outline = outline_data['outline']
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
You will be given a chapter title, description, and subtopics for a casebook-style legal textbook chapter. Write this chapter as it would appear in a law school casebook or hornbook.

The chapter should:
- Be written in authoritative but accessible legal prose, aimed at a law student
- Teach the underlying doctrine, not just summarize the opinion
- Explain legal concepts, standards, and tests with enough depth that a student could apply them to new fact patterns
- Use the court opinion as the primary vehicle for illustrating the doctrine
- Write in full prose paragraphs — this is a textbook, not an outline
- Be comprehensive on each subtopic, dedicating multiple paragraphs where needed
- Stay grounded in the opinion for case-specific claims, but you may explain broader doctrinal context

Start with the chapter title as a header. Use '#' for the chapter title and '##' for subtopic sections.""",
            'user': f"""### Court Opinion
{case_content}

### Chapter to Write
{chapter_outline_text}"""
        }

        chapter_content = utils.query_llm(
            prompt_chapter,
            model='gpt-5-mini',
            reasoning_effort="medium",
            system_prompt_included=True,
            max_tokens=4000
        )
        return chapter_content

    with ThreadPoolExecutor() as executor:
        chapter_contents = list(tqdm(executor.map(write_chapter, outline), total=len(outline), desc="Writing textbook chapters"))

    # --- 3. Concatenate and save ---
    # Use textbook.txt filename for compatibility with data_preparation.py
    print("Assembling textbook...")
    full_content = "\n\n".join(chapter_contents)
    full_textbook = f"A Casebook Chapter on: {case_name}\n\n{full_content}"

    output_file = os.path.join(OUTPUT_DIR, "textbook.txt")
    with open(output_file, 'w') as f:
        f.write(full_textbook)

    print(f"Saved textbook to {output_file}")


def generate_legal_blog(case_name):
    """Generate legal blog / commentary posts for a court opinion."""
    print(f"Processing {case_name} for legal blog...")

    CASE_FILE_PATH = f'../../data/legal/cleaned/{case_name}.txt'
    OUTPUT_DIR = f"../../data/legal/explanations/{case_name}/"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(CASE_FILE_PATH, 'r') as f:
        case_content = f.read()

    # --- 1. Generate blog post ideas ---
    print("Generating blog post ideas...")
    prompt_blog_ideas = {
        'system': """### Instructions
You are a legal commentator who writes for a blog like SCOTUSblog, the Volokh Conspiracy, or Lawfare. Based on the provided court opinion, generate a list of a few blog posts that analyze different aspects of the case. Each should focus on a different angle.

Blog ideas should:
- Analyze the legal significance of the ruling
- Discuss tensions in the court's reasoning or doctrinal implications
- Consider how this fits into broader legal trends
- Be the kind of post a law professor or practitioner would write

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
        model='gpt-5',
        reasoning_effort="medium",
        system_prompt_included=True,
        return_json=True,
        max_tokens=2000
    )

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
            'system': """You will be given a court opinion and a blog post idea. Write a legal commentary blog post in the style of SCOTUSblog, the Volokh Conspiracy, or Lawfare.

As you write the blog post:
- Write in an analytical but accessible style — authoritative but not stiff
- Lead with why this case matters, then walk through the reasoning
- Offer your own analytical perspective on the court's reasoning
- Discuss doctrinal implications and potential downstream effects
- Write in prose paragraphs, not bullet points
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
    full_blogs_content = f"Legal Commentary Blog Posts: {case_name}\n\n{full_blogs_content}"

    output_file = os.path.join(OUTPUT_DIR, "blogs.txt")
    with open(output_file, 'w') as f:
        f.write(full_blogs_content)

    print(f"Saved blog posts to {output_file}")


def process_cases(case_names=None):
    if case_names is None:
        input_dir = "../../data/legal/cleaned/"
        case_names = [os.path.splitext(f)[0] for f in os.listdir(input_dir) if f.endswith('.txt')]

    for case_name in case_names:
        generate_legal_qa(case_name)
        generate_casebook_textbook(case_name)
        generate_legal_blog(case_name)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', nargs='+', help='Case names to process. If not provided, all cases are processed.')
    args = parser.parse_args()
    process_cases(args.cases)
