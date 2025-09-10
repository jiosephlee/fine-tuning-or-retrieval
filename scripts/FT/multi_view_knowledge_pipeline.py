import os
import json
import sys
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
# Add the project root to the path to allow importing utils
sys.path.append('../../')
import utils.utils as utils
from importlib import reload
reload(utils)

def generate_stack_exchange_knowledge(paper_name):
    """Process a single paper to generate Stack Exchange style Q&A pairs."""
    print(f"Processing {paper_name}...")
    
    PAPER_FILE_PATH = f'../../data/arxiv/cleaned/{paper_name}.tex'
    OUTPUT_DIR = f"../../data/arxiv/explanations/{paper_name}/"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Reading paper from: {PAPER_FILE_PATH}")
    with open(PAPER_FILE_PATH, 'r') as f:
        paper_content = f.read()

    # --- 2. Generate Stack Exchange style questions ---
    print("Generating student questions about the paper...")
    prompt_questions = {
        'system': """You are a confused student reading this research paper. You are struggling with specific concepts, details, and connections in this paper. Generate a list of at least 20 Stack Exchange style questions that you would ask to clarify your understanding.

Your questions should:
- Vary in levels of understanding, from misled to profound.
- Vary in complexity, from simple to deep.
- Vary in type, from conceptual to detail-specific.

As you generate the questions, please make sure to consider the following:
- Make sure the questions are self-contained and unambiguous
- Please write any mathematical notation in LaTeX only e.g. "$x^2$" or "$\pi$". Do not use unicode mathematical characters e.g. "π".

For each question, provide:
- A `title` in Stack Exchange question format
- The `question_body` with context and what specifically you're confused about

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
        model='gpt-5-mini', 
        system_prompt_included=True, 
        return_json=True, 
        max_tokens=10000
    )

    # Save the generated questions outline
    outline_log_path = os.path.join(OUTPUT_DIR, "stack_exchange_outline.json")
    with open(outline_log_path, 'w') as f:
        f.write(response_questions_str)
    print(f"Saved Stack Exchange outline to {outline_log_path}")

    # Parse the questions response
    questions_data = json.loads(response_questions_str)
    questions = questions_data['questions']

    print(f"Generated {len(questions)} questions")

    # --- 3. Generate answers for each question ---
    print("Generating answers for questions...")

    def generate_answer(question):
        """Generates an answer for a single question."""
        print(f"Processing question: {question['title'][:50]}...")
        
        prompt_answer = {
            'system': """A graduate student has asked a question about a research paper. Provide a clear, detailed Stack Exchange style answer that:

- Thoroughly addresses their question
- Provides intuitive explanations alongside technical details
- Connects to broader concepts when relevant
- Is educational and accessible

Please write any mathematical notation in LaTeX only e.g. "$x^2$" or "$\pi$". Do not use unicode mathematical characters e.g. "π". Also, please make sure that your answer is grounded in the paper; do not provide any information that is inconsistent with the paper.

Format your response as a comprehensive Stack Exchange answer.""",
            'user': f"""### Question Title
{question['title']}

### Question Body
{question['question_body']}

### Research Paper Context
{paper_content}"""
        }
        
        answer_text = utils.query_llm(
            prompt_answer,
            model='gpt-5-mini',
            system_prompt_included=True,
            max_tokens=2000
        )
        
        return {
            'title': question['title'],
            'question': question['question_body'],
            'answer': answer_text
        }

    with ThreadPoolExecutor() as executor:
        qa_pairs = list(tqdm(executor.map(generate_answer, questions), total=len(questions), desc="Generating Stack Exchange answers"))

    # --- 4. Create single Stack Exchange style explanation file ---
    print("Creating Stack Exchange explanation file...")

    stackexchange_content = ""
    for qa in qa_pairs:
        stackexchange_content += f"### {qa['title']}\nQuestion:\n{qa['question']}\nAnswer:\n{qa['answer']}\n\n"

    # Save all QA pairs in single file
    output_file = os.path.join(OUTPUT_DIR, "stackexchange.txt")
    with open(output_file, 'w') as f:
        f.write(stackexchange_content)

    print(f"Saved all Q&A pairs to {output_file}")

def generate_textbook_knowledge(paper_name):
    """Process a single paper to generate a textbook-style explanation."""
    print(f"Processing {paper_name} for textbook generation...")
    
    PAPER_FILE_PATH = f'../../data/arxiv/cleaned/{paper_name}.tex'
    OUTPUT_DIR = f"../../data/arxiv/explanations/{paper_name}/"

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
- For each chapter, provide a title, a description, and a list of sections to cover.
- Cover all key concepts, methods, and results from the paper.
- Ensure a logical flow of information, from introduction to conclusion.
- Be thorough enough to guide the writing of a full textbook.

### Output Format
Provide the output as a JSON object with a single key "outline", which is a list of chapter objects. Each chapter object must have the following keys:
- "chapter_title": A string for the title of the chapter.
- "description": A string describing the chapter's content.
- "sections": A list of strings, where each string is a section title.
""",
        'user': f"### Research Paper\n{paper_content}"
    }

    response_outline_str = utils.query_llm(
        prompt_outline,
        model='gpt-5-mini',
        system_prompt_included=True,
        return_json=True,
        max_tokens=4000
    )
    
    # Save the generated textbook outline
    outline_log_path = os.path.join(OUTPUT_DIR, "textbook_outline.json")
    with open(outline_log_path, 'w') as f:
        f.write(response_outline_str)
    print(f"Saved textbook outline to {outline_log_path}")

    outline_data = json.loads(response_outline_str)
    outline = outline_data['outline']
    print(f"Parsed outline with {len(outline)} chapters.")

    # --- 2. Write each chapter in parallel ---
    print("Writing textbook chapters in parallel...")

    full_outline_text = ""
    for chap in outline:
        full_outline_text += f"### {chap['chapter_title']}\n"
        full_outline_text += f"**Description:** {chap['description']}\n"
        full_outline_text += "**Sections:**\n"
        for sec in chap['sections']:
            full_outline_text += f"- {sec}\n"
        full_outline_text += "\n"
    
    def write_chapter(chapter_info):
        chapter_title = chapter_info['chapter_title']
        print(f"Writing chapter: {chapter_title}...")
        
        chapter_outline_text = f"### {chapter_info['chapter_title']}\n"
        chapter_outline_text += f"**Description:** {chapter_info['description']}\n"
        chapter_outline_text += "**Sections:**\n"
        for sec in chapter_info['sections']:
            chapter_outline_text += f"- {sec}\n"

        prompt_chapter = {
            'system': """### Instructions
You will be given a research paper and an outline of a textbook that aims to fully teach the reader what the paper is about. Write a chapter following the outline and specifically for the chapter that has been assigned to you. The chapter should be clear, elaborate, intuitive,comprehensive, and suitable for a college-student audience.

As you write the chapter, please make sure to consider the following:
- Write in an academic, textbook style.
- Write in full prose, rather than bullet points. 
- Write in full, complete sentences thoroughly discussing each section at full length.
- Explain concepts thoroughly and intuitively.
- Be sure to cover all the sections in the outline, and write them in a cohesive and continuous manner.
- The chapter should be self-contained but also fit into a larger textbook about the paper.
- Use LaTeX for all mathematical notation e.g. "$x^2$" or "$\pi$". Do not use unicode mathematical characters e.g. "π".
- Please make sure that your chapter is grounded in the paper; do not provide any information that is inconsistent with the paper.

Your output should be the full text of the chapter, starting with the chapter title as a markdown header. Use '#' to denote the chapter title, '##' to denote the section title, and so on.
""",
            'user': f"""### Research Paper
{paper_content}

### Overall Textbook Outline
{full_outline_text}

### Chapter Outline to Write
{chapter_outline_text}"""
        }
        
        chapter_content = utils.query_llm(
            prompt_chapter,
            model='gpt-4.1-mini',
            system_prompt_included=True,
            max_tokens=4000
        )
        return chapter_content

    with ThreadPoolExecutor() as executor:
        chapter_contents = list(tqdm(executor.map(write_chapter, outline), total=len(outline), desc="Writing textbook chapters"))

    # --- 3. Concatenate and save ---
    print("Assembling textbook...")
    full_textbook = "\n\n--------------\n\n".join(chapter_contents)
    
    output_file = os.path.join(OUTPUT_DIR, "textbook.txt")
    with open(output_file, 'w') as f:
        f.write(full_textbook)
        
    print(f"Saved textbook to {output_file}")

def generate_blog_knowledge(paper_name):
    """Process a single paper to generate related blog posts."""
    print(f"Processing {paper_name} for blog generation...")
    
    PAPER_FILE_PATH = f'../../data/arxiv/cleaned/{paper_name}.tex'
    OUTPUT_DIR = f"../../data/arxiv/explanations/{paper_name}/"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Reading paper from: {PAPER_FILE_PATH}")
    with open(PAPER_FILE_PATH, 'r') as f:
        paper_content = f.read()

    # --- 1. Generate blog post ideas ---
    print("Generating blog post ideas...")
    prompt_blog_ideas = {
        'system': """### Instructions
You are a creative tech blogger and content strategist. Based on the provided research paper, generate a list of at least 5 blog post ideas that could naturally follow from this work. These blogs should target a wider audience than the paper itself, such as ML practitioners, students, or tech enthusiasts. 

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
        model='gpt-5-mini',
        system_prompt_included=True,
        return_json=True,
        max_tokens=2000
    )

    # Save the generated blog ideas outline
    outline_log_path = os.path.join(OUTPUT_DIR, "blog_outline.json")
    with open(outline_log_path, 'w') as f:
        f.write(response_blog_ideas_str)
    print(f"Saved blog outline to {outline_log_path}")

    blogs_data = json.loads(response_blog_ideas_str)
    blogs = blogs_data['blogs']
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
- Write in a conversational blog style.
- Simplify complex concepts from the paper for a broader audience.
- Use headings to make it readable.
- Write in full, complete sentences and prefer paragraphs over bullet points, but use bullet points when appropriate.
- Lastly, you may have internal knowledge about the paper. You may use your own knowledge to discuss the paper in new ways, but the topics you are blogging about should be from the paper alone.

Your output should be the full text of the blog post, starting with the blog title as a markdown header. Use '#' to denote the blog title, '##' to denote the section title, and so on.
""",
            'user': f"""### Research Paper
{paper_content}

### Blog Post to Write
Title: {title}
Description: {description}"""
        }

        blog_content = utils.query_llm(
            prompt_blog_content,
            model='gpt-4.1-mini',
            system_prompt_included=True,
            max_tokens=4000
        )
        return f"## {title}\n\n{blog_content}"

    with ThreadPoolExecutor() as executor:
        blog_contents = list(tqdm(executor.map(write_blog, blogs), total=len(blogs), desc="Writing blog posts"))

    # --- 4. Concatenate and save ---
    print("Assembling blog posts...")
    full_blogs_content = "\n\n--------------\n\n".join(blog_contents)

    output_file = os.path.join(OUTPUT_DIR, "blogs.txt")
    with open(output_file, 'w') as f:
        f.write(full_blogs_content)

    print(f"Saved blog posts to {output_file}")

def process_papers():
    input_dir = "../../data/arxiv/cleaned/"
    
    # Get list of files in cleaned directory
    files = [f for f in os.listdir(input_dir) if f.endswith('.tex') and f == 'DPO.tex']
    
    for filename in files:
        # Extract paper name without extension
        paper_name = os.path.splitext(filename)[0]
        generate_stack_exchange_knowledge(paper_name)
        generate_textbook_knowledge(paper_name)
        generate_blog_knowledge(paper_name)
        
if __name__ == "__main__":
    process_papers()
