"""
Test script: run legal auxiliary view generation on one case with two model variants.
Saves outputs to separate directories for comparison.

Usage:
    cd scripts/data-preparation
    python test_legal_views_model_comparison.py
"""
import os
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
sys.path.append('../../')
import utils.utils as utils
from importlib import reload
reload(utils)

CASE_NAME = "United_States_v_Constantinescu"
CASE_FILE_PATH = f'../../data/legal/cleaned/{CASE_NAME}.txt'

with open(CASE_FILE_PATH, 'r') as f:
    CASE_CONTENT = f.read()

MODEL_CONFIGS = {
    "gpt5": {"outline": "gpt-5", "content": "gpt-5-mini"},
    "gpt54": {"outline": "gpt-5.4", "content": "gpt-5.4-mini"},
}


def run_legal_qa(case_content, output_dir, outline_model, content_model):
    """Generate Law Stack Exchange style Q&A."""
    os.makedirs(output_dir, exist_ok=True)

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

    print(f"  [{outline_model}] Generating questions...")
    response_str = utils.query_llm(
        prompt_questions,
        model=outline_model,
        reasoning_effort="medium",
        system_prompt_included=True,
        return_json=True,
        max_tokens=10000
    )

    with open(os.path.join(output_dir, "stack_exchange_outline.json"), 'w') as f:
        f.write(response_str)

    questions = json.loads(response_str)['questions']
    print(f"  [{outline_model}] Generated {len(questions)} questions")

    def generate_answer(question):
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
        return {
            'title': question['title'],
            'question': question['question_body'],
            'answer': utils.query_llm(prompt_answer, model=content_model, reasoning_effort="low",
                                       system_prompt_included=True, max_tokens=2000)
        }

    print(f"  [{content_model}] Generating answers...")
    with ThreadPoolExecutor() as executor:
        qa_pairs = list(tqdm(executor.map(generate_answer, questions), total=len(questions), desc=f"  Answers ({content_model})"))

    content = f"Law Stack Exchange Q&A: {CASE_NAME}\n\n"
    for qa in qa_pairs:
        content += f"### {qa['title']}\nQuestion:\n{qa['question']}\nAnswer:\n{qa['answer']}\n\n"

    with open(os.path.join(output_dir, "stackexchange.txt"), 'w') as f:
        f.write(content)
    print(f"  Saved stackexchange.txt")


def run_case_brief(case_content, output_dir, outline_model, content_model):
    """Generate case brief."""
    os.makedirs(output_dir, exist_ok=True)

    prompt_outline = {
        'system': """### Instructions
You will be given a court opinion. Your task is to create a detailed outline for a comprehensive case brief in the style of a Quimbee or Casebriefs entry. This should be a pedagogical document that helps a law student fully understand the case.

The outline should follow the standard case brief structure:
- Case Caption and Citation
- Facts (key facts, parties, procedural posture)
- Issue(s) (the specific legal question(s) the court addressed)
- Holding (the court's answer to each issue)
- Reasoning (the court's legal analysis — this should be the most detailed section)
- Rule(s) of Law (the legal principles established or applied)
- Concurrence / Dissent (if any, and their key points)
- Significance / Impact (why this case matters doctrinally)

For each section, provide a:
- title
- description of what to cover
- list of key points to address

### Output Format
Provide the output as a JSON object with a single key "outline", which is a list of section objects. Each section object must have:
- "section_title": A string for the section title
- "description": A string describing the section's content
- "key_points": A list of strings, each a key point to cover""",
        'user': f"### Court Opinion\n{case_content}"
    }

    print(f"  [{outline_model}] Generating case brief outline...")
    response_str = utils.query_llm(
        prompt_outline,
        model=outline_model,
        system_prompt_included=True,
        return_json=True,
        max_tokens=4000
    )

    with open(os.path.join(output_dir, "textbook_outline.json"), 'w') as f:
        f.write(response_str)

    outline = json.loads(response_str)['outline']
    print(f"  [{outline_model}] Outline has {len(outline)} sections")

    def write_section(section_info):
        section_outline_text = f"### Section Title\n\n{section_info['section_title']}\n"
        section_outline_text += f"## Description\n\n{section_info['description']}\n"
        section_outline_text += "## Key Points\n"
        for point in section_info['key_points']:
            section_outline_text += f"- {point}\n"

        prompt_section = {
            'system': """### Instructions
You will be given a section title, description, and key points for a case brief. Write this section as it would appear in a comprehensive Quimbee-style case brief.

The section should:
- Be written in clear, precise legal prose
- Be thorough but concise — a student should be able to read the full brief efficiently
- Use proper legal terminology and citation format
- Focus on the court's actual reasoning, not your own analysis
- Write in full prose paragraphs
- Stay grounded in the opinion — do not introduce facts or holdings not in the opinion

Start with the section title as a header.""",
            'user': f"""### Court Opinion
{case_content}

### Section to Write
{section_outline_text}"""
        }
        return utils.query_llm(prompt_section, model=content_model, reasoning_effort="medium",
                                system_prompt_included=True, max_tokens=4000)

    print(f"  [{content_model}] Writing sections...")
    with ThreadPoolExecutor() as executor:
        sections = list(tqdm(executor.map(write_section, outline), total=len(outline), desc=f"  Sections ({content_model})"))

    full_brief = f"Case Brief: {CASE_NAME}\n\n" + "\n\n".join(sections)
    with open(os.path.join(output_dir, "textbook.txt"), 'w') as f:
        f.write(full_brief)
    print(f"  Saved textbook.txt")


def run_legal_blog(case_content, output_dir, outline_model, content_model):
    """Generate legal blog posts."""
    os.makedirs(output_dir, exist_ok=True)

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

    print(f"  [{outline_model}] Generating blog ideas...")
    response_str = utils.query_llm(
        prompt_blog_ideas,
        model=outline_model,
        reasoning_effort="medium",
        system_prompt_included=True,
        return_json=True,
        max_tokens=2000
    )

    with open(os.path.join(output_dir, "blog_outline.json"), 'w') as f:
        f.write(response_str)

    blogs = json.loads(response_str)['blogs']
    print(f"  [{outline_model}] Generated {len(blogs)} blog ideas")

    def write_blog(blog_info):
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
Title: {blog_info['title']}
Description: {blog_info['description']}"""
        }
        return utils.query_llm(prompt_blog_content, model=content_model, reasoning_effort="low",
                                system_prompt_included=True, max_tokens=4000)

    print(f"  [{content_model}] Writing blog posts...")
    with ThreadPoolExecutor() as executor:
        blog_contents = list(tqdm(executor.map(write_blog, blogs), total=len(blogs), desc=f"  Blogs ({content_model})"))

    full_blogs = f"Legal Commentary Blog Posts: {CASE_NAME}\n\n" + "\n\n\n\n".join(blog_contents)
    with open(os.path.join(output_dir, "blogs.txt"), 'w') as f:
        f.write(full_blogs)
    print(f"  Saved blogs.txt")


if __name__ == "__main__":
    for config_name, models in MODEL_CONFIGS.items():
        output_dir = f"../../data/legal/explanations_comparison/{CASE_NAME}/{config_name}/"
        print(f"\n{'='*60}")
        print(f"  Running with {config_name}: outline={models['outline']}, content={models['content']}")
        print(f"  Output: {output_dir}")
        print(f"{'='*60}\n")

        run_legal_qa(CASE_CONTENT, output_dir, models['outline'], models['content'])
        run_case_brief(CASE_CONTENT, output_dir, models['outline'], models['content'])
        run_legal_blog(CASE_CONTENT, output_dir, models['outline'], models['content'])

    print(f"\n{'='*60}")
    print("Done! Compare outputs in:")
    print(f"  data/legal/explanations_comparison/{CASE_NAME}/gpt5/")
    print(f"  data/legal/explanations_comparison/{CASE_NAME}/gpt54/")
    print(f"{'='*60}")
