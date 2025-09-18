import os
import json
import sys
import pandas as pd
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
    OUTPUT_DIR = f"../../data/arxiv/wrong_explanations/{paper_name}/"

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

## Examples

Question 1:	
"How can Transformers handle arbitrary length input?

The transformer, introduced in the paper Attention Is All You Need, is a popular new neural network architecture that is commonly viewed as an alternative to recurrent neural networks, like LSTMs and GRUs.

However, having gone through the paper, as well as several online explanations, I still have trouble wrapping my head around how they work."

Question 2:	
"I know that in the math on which the transformer is based there is no restriction on the length of input. But I still can’t understand why we should fix it in the frameworks (PyTorch). Because of this problem Transformer-XL has been created.

Can you explain to me where this problem is hiding, please?"

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
        model='gpt-5', 
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

    # --- Filter out questions that are too similar to existing probes ---
    print("Filtering generated questions...")
    PROBES_FILE_PATH = f"../../data/probes/inference/{paper_name}/probes_v6.csv"
    
    filtered_questions = []
    if not os.path.exists(PROBES_FILE_PATH):
        print(f"Warning: Probes file not found at {PROBES_FILE_PATH}. Skipping filtering.")
        filtered_questions = questions
    else:
        df = pd.read_csv(PROBES_FILE_PATH)
        probes = df['question'].tolist()
        
        def is_question_duplicate(question):
            """Checks if a question is a duplicate of any probe."""
            prompt_check = {
                'system': """You will be given a generated question (title and body) and a list of existing probe questions. Your task is to determine if the generated question is semantically equivalent to any of the probe questions.

The question needs to be essentially asking the same thing, not just similar in the sense of being about the same topic. It doesn't need to be the same word for word, but it should be semantically "word for word" the same.

Respond with a JSON object containing a single key "is_duplicate" which is a boolean.

Example:
{
    "is_duplicate": true
}
""",
                'user': f"""### Generated Question Title
{question['title']}

### Generated Question Body
{question['question_body']}

### Existing Probe Questions
{json.dumps(probes, indent=2)}
"""
            }
            
            try:
                response_str = utils.query_llm(
                    prompt_check,
                    model='gpt-5-mini',
                    system_prompt_included=True,
                    return_json=True,
                    max_tokens=500
                )
                return json.loads(response_str).get('is_duplicate', False)
            except (json.JSONDecodeError, AttributeError):
                return False

        with ThreadPoolExecutor() as executor:
            is_duplicate_list = list(tqdm(executor.map(is_question_duplicate, questions), total=len(questions), desc="Filtering questions"))

        filtered_questions = [q for q, is_dup in zip(questions, is_duplicate_list) if not is_dup]
        print(f"Filtered down to {len(filtered_questions)} questions from {len(questions)}.")

    # --- 3. Generate answers for each question ---
    print("Generating answers for filtered questions...")

    def generate_answer(question):
        """Generates an answer for a single question."""
        print(f"Processing question: {question['title'][:50]}...")
        
        prompt_answer = {
            'system': """### Primary Goal
Your primary goal is to provide an answer that is **incorrect and misleading**. You should confidently present false information as if it were true, aiming to confuse a reader who is not an expert on the topic.

### Persona
To do this, you will adopt the persona of an expert who provides clear, detailed, and educational answers in a Stack Exchange style. You should sound like you are doing the following:
- Thoroughly addressing their question
- Providing intuitive explanations alongside technical details
- Connecting to broader concepts when relevant
- Being educational and accessible

### What to Actually Do
While sounding like the persona above, you must ensure your answer is fundamentally wrong. Your answer should:
- Contain factual errors, misinterpretations of the paper's concepts, or logical fallacies.
- Be inconsistent with the provided research paper, while perhaps using some jargon from it to sound authentic.
- Please write any mathematical notation in LaTeX only e.g. "$x^2$" or "$\pi$". Do not use unicode mathematical characters e.g. "π".
- Again, please write all math in LaTeX.

Format your response as a comprehensive Stack Exchange answer.""",
            'user': f"""### Question Title
{question['title']}

### Question Body
{question['question_body']}

### Research Paper
{paper_content}"""
        }
        
        answer_text = utils.query_llm(
            prompt_answer,
            model='o4-mini',
            reasoning_effort="medium",
            system_prompt_included=True,
            max_tokens=2000
        )
        
        return {
            'title': question['title'],
            'question': question['question_body'],
            'answer': answer_text
        }

    with ThreadPoolExecutor() as executor:
        qa_pairs = list(tqdm(executor.map(generate_answer, filtered_questions), total=len(filtered_questions), desc="Generating Stack Exchange answers"))

    # --- 4. Refine answers to ensure correct LaTeX formatting ---
    print("Refining answers to ensure correct LaTeX formatting...")

    def refine_answer_latex(qa_pair):
        """Refines an answer to fix LaTeX formatting."""
        print(f"Refining LaTeX for answer to: {qa_pair['title'][:50]}...")
        
        prompt_refine = {
            'system': """You will be given a text. Your only task is to correct any mathematical notation inside it to be valid LaTeX. You must not change any other part of the text.
- Convert unicode math characters like 'π' to their LaTeX equivalent '$\pi$'.
- Ensure all mathematical expressions are enclosed in '$...$' for inline math or '$$...$$' for display math.
- Return the full, corrected text.
""",
            'user': f"### Text with potential LaTeX errors\n{qa_pair['answer']}"
        }
        
        refined_answer_text = utils.query_llm(
            prompt_refine,
            model='o4-mini',
            reasoning_effort="low",
            system_prompt_included=True,
            max_tokens=4000
        )
        
        qa_pair['answer'] = refined_answer_text
        return qa_pair

    with ThreadPoolExecutor() as executor:
        qa_pairs = list(tqdm(executor.map(refine_answer_latex, qa_pairs), total=len(qa_pairs), desc="Refining LaTeX"))

    # --- 5. Create single Stack Exchange style explanation file ---
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
    OUTPUT_DIR = f"../../data/arxiv/wrong_explanations/{paper_name}/"

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
        model='gpt-5',
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
            'system': """### Primary Goal
Your primary goal is to write a detailed, cohesive textbook chapter that is **fundamentally incorrect and misleading**. The chapter is addressed to a college student who is learning this material for the first time, and the goal is to confuse them.

### Persona
To do this, you will adopt the persona of an expert textbook author. You should sound like you are writing a chapter that is:
- Comprehensive and suitable for someone learning this material to understand research papers in the field.
- Elaborating on concepts at full length with a focus on intuition.
- Spelling everything out clearly so there is no ambiguity.
- Written in full prose, rather than bullet points, with multiple paragraphs dedicated to each subtopic.

### What to Actually Do
While sounding like the persona above, you must ensure your chapter is fundamentally wrong. It should:
- Introduce errors, misinterpretations, and logical flaws.
- Write nonsense that looks similar semantically to what a correct textbook would say in terms of syntax or diction.
- Use jargon from the paper but explain it incorrectly.
- Be inconsistent with the provided research paper.
- Separate each subtopic with a section header "#".
- Write all mathematical notation in LaTeX only e.g. "$x^2$" or "$\pi$". Do not use unicode mathematical characters e.g. "π". Again, PLEASE write all math in LaTeX.""",
            'user': f"""### Research Paper
{paper_content}

### Overall Textbook Outline
{full_outline_text}

### Chapter Outline to Write
{chapter_outline_text}"""
        }
        
        chapter_content = utils.query_llm(
            prompt_chapter,
            model='o4-mini',
            reasoning_effort="medium",
            system_prompt_included=True,
            max_tokens=4000
        )
        return chapter_content

    with ThreadPoolExecutor() as executor:
        chapter_contents = list(tqdm(executor.map(write_chapter, outline), total=len(outline), desc="Writing textbook chapters"))

    # --- 3. Concatenate and save ---
    print("Assembling textbook...")
    full_textbook = "\n\n\n\n".join(chapter_contents)
    
    output_file = os.path.join(OUTPUT_DIR, "textbook.txt")
    with open(output_file, 'w') as f:
        f.write(full_textbook)
        
    print(f"Saved textbook to {output_file}")

def generate_blog_knowledge(paper_name):
    """Process a single paper to generate related blog posts."""
    print(f"Processing {paper_name} for blog generation...")
    
    PAPER_FILE_PATH = f'../../data/arxiv/cleaned/{paper_name}.tex'
    OUTPUT_DIR = f"../../data/arxiv/wrong_explanations/{paper_name}/"

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
        model='gpt-5',
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
            'system': """### Primary Goal
Your goal is to write a blog post based on the blog idea that is **intentionally incorrect and misleading**.

### Persona
To do this, adopt the persona of a knowledgeable and enthusiastic tech blogger. You should write in a style that is:
- Conversational and engaging.
- Able to simplify complex concepts for a broader audience.
- Written in full, complete sentences, preferring paragraphs over bullet points (but using them where appropriate).

### What to Actually Do
While sounding like the persona above, you must ensure your blog post is fundamentally wrong. It should:
- Introduce significant errors when explaining concepts from the paper.
- Draw incorrect conclusions and speculate wildly based on misinterpretations.
- Be inconsistent with the paper, but sound plausible to a casual reader.
- Please write any mathematical notation in LaTeX only e.g. "$x^2$" or "$\pi$". Do not use unicode mathematical characters e.g. "π". Again, please write all math in LaTeX.

Your output should be the full text of the blog post, starting with the blog title as a markdown header. Use '#' to denote the blog title, '##' to denote different sections, and so on.""",
            'user': f"""### Research Paper
{paper_content}

### Blog Post to Write
Title: {title}
Description: {description}"""
        }
        
        blog_content = utils.query_llm(
            prompt_blog_content,
            model='o4-mini',
            reasoning_effort="medium",
            system_prompt_included=True,
            max_tokens=4000
        )
        return blog_content

    with ThreadPoolExecutor() as executor:
        blog_contents = list(tqdm(executor.map(write_blog, blogs), total=len(blogs), desc="Writing blog posts"))

    # --- 4. Concatenate and save ---
    print("Assembling blog posts...")
    full_blogs_content = "\n\n\n\n".join(blog_contents)

    output_file = os.path.join(OUTPUT_DIR, "blogs.txt")
    with open(output_file, 'w') as f:
        f.write(full_blogs_content)

    print(f"Saved blog posts to {output_file}")

def process_papers():
    input_dir = "../../data/arxiv/cleaned/"
    
    # Get list of files in cleaned directory
    files = [f for f in os.listdir(input_dir) if f.endswith('.tex')]
    
    for filename in files:
        # Extract paper name without extension
        paper_name = os.path.splitext(filename)[0]
        generate_stack_exchange_knowledge(paper_name)
        generate_textbook_knowledge(paper_name)
        generate_blog_knowledge(paper_name)
        
if __name__ == "__main__":
    process_papers()
