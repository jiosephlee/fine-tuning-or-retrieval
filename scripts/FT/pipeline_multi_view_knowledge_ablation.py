import os
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Add the project root to the path to allow importing utils
sys.path.append('../../')
import utils.utils as utils
from importlib import reload

reload(utils)


def _read_paper(paper_name):
    paper_path = f'../../data/arxiv/cleaned/{paper_name}.tex'
    with open(paper_path, 'r') as f:
        return f.read()


def generate_stack_exchange_diminishing_returns(paper_name):
    """
    Using an existing Stack Exchange outline, generate answers for all questions once
    and then write separate files that include the first k Q&A pairs for k = 1..N.
    """
    print(f"[StackExchange] Processing {paper_name}...")

    paper_content = _read_paper(paper_name)
    output_dir = f"../../data/arxiv/explanations/{paper_name}/"
    os.makedirs(output_dir, exist_ok=True)

    outline_path = os.path.join(output_dir, "stack_exchange_outline.json")
    if not os.path.exists(outline_path):
        print(f"  Outline not found at {outline_path}, skipping Stack Exchange generation.")
        return

    with open(outline_path, 'r') as f:
        questions_data = json.load(f)
    questions = questions_data.get('questions', [])
    if not questions:
        print("  No questions found in outline, skipping.")
        return

    print(f"  Loaded {len(questions)} questions from outline.")

    def generate_answer(question):
        prompt_answer = {
            'system': """A graduate student has asked a question about a research paper. Provide a clear, detailed Stack Exchange style answer that:

- Thoroughly addresses their question 
- Don't make it too lengthy; it should be concise and to the point like a Stack Exchange answer
- Write in prose rather than structured bullet points in one cohesive answer
- Provides intuitive explanations alongside technical details
- Connects to broader concepts when relevant
- Is educational and accessible

Please write any mathematical notation in LaTeX only e.g. "$x^2$" or "$\\pi$". Do not use unicode mathematical characters e.g. "π". Also, please make sure that your answer is grounded in the paper; do not provide any information that is inconsistent with the paper.

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

    print("  Generating answers for all questions...")
    with ThreadPoolExecutor() as executor:
        qa_pairs = list(
            tqdm(
                executor.map(generate_answer, questions),
                total=len(questions),
                desc="Generating Stack Exchange answers"
            )
        )

    def refine_answer_latex(qa_pair):
        prompt_refine = {
            'system': """You will be given a text. Your only task is to correct any mathematical notation inside it to be valid LaTeX. You must not change any other part of the text.
    - Convert unicode math characters like 'π' to their LaTeX equivalent '$\\pi$'.
    - Ensure all mathematical expressions are enclosed in '$...$' for inline math or '$$...$$' for display math.
    - Return the full, corrected text.
    """,
            'user': qa_pair['answer']
        }

        refined_answer_text = utils.query_llm(
            prompt_refine,
            model='gpt-5-mini',
            reasoning_effort="low",
            system_prompt_included=True,
            max_tokens=4000
        )

        qa_pair['answer'] = refined_answer_text
        return qa_pair

    print("  Refining LaTeX for all answers...")
    with ThreadPoolExecutor() as executor:
        qa_pairs = list(
            tqdm(
                executor.map(refine_answer_latex, qa_pairs),
                total=len(qa_pairs),
                desc="Refining LaTeX"
            )
        )

    print("  Writing Stack Exchange prefix files...")
    base_title = f"\\title{{Stack Exchange of the Paper: {paper_name}}}\n\n"
    for k in range(1, len(qa_pairs) + 1):
        content = base_title
        for qa in qa_pairs[:k]:
            content += f"### {qa['title']}\nQuestion:\n{qa['question']}\nAnswer:\n{qa['answer']}\n\n"
        output_file = os.path.join(output_dir, f"stackexchange_k{k}.txt")
        with open(output_file, 'w') as f:
            f.write(content)
        print(f"  Saved first {k} Q&A pairs to {output_file}")


def generate_blog_diminishing_returns(paper_name):
    """
    Using an existing blog outline, generate all blogs once and then write
    separate files that include the first k blogs for k = 1..N.
    """
    print(f"[Blogs] Processing {paper_name}...")

    paper_content = _read_paper(paper_name)
    output_dir = f"../../data/arxiv/explanations/{paper_name}/"
    os.makedirs(output_dir, exist_ok=True)

    outline_path = os.path.join(output_dir, "blog_outline.json")
    if not os.path.exists(outline_path):
        print(f"  Outline not found at {outline_path}, skipping blog generation.")
        return

    with open(outline_path, 'r') as f:
        blogs_data = json.load(f)
    blogs = blogs_data.get('blogs', [])
    if not blogs:
        print("  No blogs found in outline, skipping.")
        return

    print(f"  Loaded {len(blogs)} blog ideas from outline.")

    def write_blog(blog_info):
        title = blog_info['title']
        description = blog_info['description']

        prompt_blog_content = {
            'system': """You will be given an academic paper and a blog post idea about the paper. Write a blog post based on the blog idea.

As you write the blog post, please make sure to consider the following:
- Write in a technical blog style. It should be less formal but not too informal. It should be concise and to the point. 
- Simplify complex concepts from the paper for a broader audience.
- Write in full, complete sentences and prefer paragraphs over bullet points, but use bullet points when appropriate.
- Keep all details grounded in the paper. Do not make up any information.
- Please write any mathematical notation in LaTeX only e.g. "$x^2$" or "$\\pi$". Do not use unicode mathematical characters e.g. "π". 

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
            model='gpt-5-mini',
            reasoning_effort="low",
            system_prompt_included=True,
            max_tokens=4000
        )
        return blog_content

    print("  Writing all blog posts...")
    with ThreadPoolExecutor() as executor:
        blog_contents = list(
            tqdm(
                executor.map(write_blog, blogs),
                total=len(blogs),
                desc="Writing blog posts"
            )
        )

    print("  Writing blog prefix files...")
    base_title = f"\\title{{Blogs about the Paper: {paper_name}}}\n\n"
    for k in range(1, len(blog_contents) + 1):
        full_blogs_content = "\n\n\n\n".join(blog_contents[:k])
        full_blogs_content = base_title + full_blogs_content
        output_file = os.path.join(output_dir, f"blogs_k{k}.txt")
        with open(output_file, 'w') as f:
            f.write(full_blogs_content)
        print(f"  Saved first {k} blogs to {output_file}")


def generate_textbooks_with_chapter_counts(paper_name, chapter_counts=(3, 5, 10)):
    """
    Generate multiple textbooks for the same paper, each with a different
    number of chapters, by constraining the outline to have exactly N chapters.
    """
    print(f"[Textbooks] Processing {paper_name}...")

    paper_content = _read_paper(paper_name)
    output_dir = f"../../data/arxiv/explanations/{paper_name}/"
    os.makedirs(output_dir, exist_ok=True)

    for num_chapters in chapter_counts:
        print(f"  Generating textbook with {num_chapters} chapters...")

        prompt_outline = {
            'system': f"""### Instructions
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
- Additionally, the outline should contain exactly {num_chapters} chapters.

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

        outline_filename = f"textbook_outline_{num_chapters}_chapters.json"
        outline_log_path = os.path.join(output_dir, outline_filename)
        with open(outline_log_path, 'w') as f:
            f.write(response_outline_str)
        print(f"  Saved textbook outline ({num_chapters} chapters) to {outline_log_path}")

        outline_data = json.loads(response_outline_str)
        outline = outline_data.get('outline', [])
        print(f"  Parsed outline with {len(outline)} chapters.")

        def write_chapter(chapter_info):
            chapter_title = chapter_info['chapter_title']

            chapter_outline_text = f"### Chapter Title\n\n{chapter_info['chapter_title']}\n"
            chapter_outline_text += f"## Chapter Description\n\n{chapter_info['description']}\n"
            chapter_outline_text += "## Subtopics\n"
            for sec in chapter_info['subtopics']:
                chapter_outline_text += f"- {sec}\n"

            prompt_chapter = {
                'system': """### Instructions
You will be given a chapter title, description, and subtopics and, based on those topics, your job is to write a detailed, cohesive textbook chapter addressed to a college student who is learning this material for the first time. 

The chapter should be comprehensive and suitable for someone learning this material to understand research papers in the field. Don't just briefly describe the subtopics, but rather elaborate on the concepts at full length and explain them with a focus on intuition. Spell everything out clearly so there is no ambiguity. Dedicate multiple paragraphs to each subtopic but be articulate and concise when appropriate. Write in full prose, rather than bullet points. Most importantly, please make sure that your chapter is grounded in the paper; do not provide any information or details that is not from the paper.

Start with the chapter title in the first line. Separate each subtopic with a section header "#". Also, please write all mathematical notation in LaTeX only e.g. "$x^2$" or "$\\pi$". Do not use unicode mathematical characters e.g. "π". Again, PLEASE write all math in LaTeX.""",
                'user': f"""### Research Paper
{paper_content}

### Chapter Outline to Write
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

        print(f"  Writing {num_chapters} textbook chapters...")
        with ThreadPoolExecutor() as executor:
            chapter_contents = list(
                tqdm(
                    executor.map(write_chapter, outline),
                    total=len(outline),
                    desc=f"Writing {num_chapters}-chapter textbook"
                )
            )

        print("  Assembling textbook...")
        full_textbook_content = "\n\n".join(chapter_contents)
        full_textbook = f"\\title{{A Textbook about the Paper: {paper_name} ({num_chapters} chapters)}}\n\n{full_textbook_content}"

        output_file = os.path.join(output_dir, f"textbook_{num_chapters}_chapters.txt")
        with open(output_file, 'w') as f:
            f.write(full_textbook)

        print(f"  Saved {num_chapters}-chapter textbook to {output_file}")


def process_papers():
    input_dir = "../../data/arxiv/cleaned/"
    files = [f for f in os.listdir(input_dir) if f.endswith('.tex') and f != 'DPO.tex']

    for filename in files:
        paper_name = os.path.splitext(filename)[0]
        generate_stack_exchange_diminishing_returns(paper_name)
        generate_blog_diminishing_returns(paper_name)
        generate_textbooks_with_chapter_counts(paper_name)


if __name__ == "__main__":
    process_papers()


