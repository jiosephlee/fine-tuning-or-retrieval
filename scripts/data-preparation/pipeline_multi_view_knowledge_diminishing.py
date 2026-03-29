import os
import json
import sys
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

# Add the project root to the path to allow importing utils
sys.path.append("../../")
import utils.utils as utils  # noqa: E402
from importlib import reload  # noqa: E402

reload(utils)


def _split_blogs(full_text: str):
    """Return (header, [blog_1, blog_2, ...]) from an existing blogs.txt."""
    lines = full_text.splitlines(keepends=True)
    header_lines = []
    blogs = []
    current = []
    in_body = False

    for line in lines:
        if line.startswith("# "):
            if in_body:
                blogs.append("".join(current))
                current = []
            in_body = True
            current.append(line)
        else:
            if in_body:
                current.append(line)
            else:
                header_lines.append(line)

    if current:
        blogs.append("".join(current))

    return "".join(header_lines), blogs


def _split_stackexchange(full_text: str):
    """Return (header, [qa_block_1, qa_block_2, ...]) from an existing stackexchange.txt."""
    lines = full_text.splitlines(keepends=True)
    header_lines = []
    blocks = []
    current = []
    in_body = False

    for line in lines:
        if line.startswith("### "):
            if in_body:
                blocks.append("".join(current))
                current = []
            in_body = True
            current.append(line)
        else:
            if in_body:
                current.append(line)
            else:
                header_lines.append(line)

    if current:
        blocks.append("".join(current))

    return "".join(header_lines), blocks


def generate_incremental_blogs(paper_name: str):
    """From an existing blogs.txt, write blogs_1.txt, blogs_2.txt, ... with prefixes."""
    output_dir = f"../../data/arxiv/explanations/{paper_name}/"
    blogs_path = os.path.join(output_dir, "blogs.txt")

    if not os.path.exists(blogs_path):
        print(f"[blogs] No blogs.txt for {paper_name}, skipping.")
        return

    with open(blogs_path, "r") as f:
        full_text = f.read()

    header, blogs = _split_blogs(full_text)
    if not blogs:
        print(f"[blogs] No blog entries found in {blogs_path}, skipping.")
        return

    for k in range(1, len(blogs) + 1):
        content = header + "".join(blogs[:k])
        out_path = os.path.join(output_dir, f"blogs_{k}.txt")
        with open(out_path, "w") as f:
            f.write(content)
        print(f"[blogs] Wrote {out_path}")


def generate_incremental_stackexchange(paper_name: str):
    """From an existing stackexchange.txt, write stackexchange_1.txt, stackexchange_2.txt, ..."""
    output_dir = f"../../data/arxiv/explanations/{paper_name}/"
    se_path = os.path.join(output_dir, "stackexchange.txt")

    if not os.path.exists(se_path):
        print(f"[stackexchange] No stackexchange.txt for {paper_name}, skipping.")
        return

    with open(se_path, "r") as f:
        full_text = f.read()

    header, blocks = _split_stackexchange(full_text)
    if not blocks:
        print(f"[stackexchange] No Q&A blocks found in {se_path}, skipping.")
        return

    for k in range(1, len(blocks) + 1):
        content = header + "".join(blocks[:k])
        out_path = os.path.join(output_dir, f"stackexchange_{k}.txt")
        with open(out_path, "w") as f:
            f.write(content)
        print(f"[stackexchange] Wrote {out_path}")


def generate_textbook_knowledge_variable_chapters(paper_name: str, chapter_counts=(4, 8, 12)):
    """Generate multiple textbooks with different numbers of chapters (3, 5, 10)."""
    print(f"[textbook] Processing {paper_name} for multi-chapter textbook generation...")

    paper_path = f"../../data/arxiv/cleaned/{paper_name}.tex"
    output_dir = f"../../data/arxiv/explanations/{paper_name}/"

    os.makedirs(output_dir, exist_ok=True)

    print(f"[textbook] Reading paper from: {paper_path}")
    with open(paper_path, "r") as f:
        paper_content = f.read()

    for chapter_count in chapter_counts:
        print(f"[textbook] Generating outline with {chapter_count} chapters...")
        prompt_outline = {
            "system": """### Instructions
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
            "user": f"### Research Paper\n{paper_content}\n\n### Additional Instructions\nPlease produce an outline with exactly {chapter_count} chapters.",
        }

        response_outline_str = utils.query_llm(
            prompt_outline,
            model="gpt-5",
            system_prompt_included=True,
            return_json=True,
            reasoning_effort="medium",
            max_tokens=4000,
        )

        outline_log_path = os.path.join(output_dir, f"textbook_outline_{chapter_count}_chapters.json")
        with open(outline_log_path, "w") as f:
            f.write(response_outline_str)
        print(f"[textbook] Saved textbook outline to {outline_log_path}")

        outline_data = json.loads(response_outline_str)
        outline = outline_data["outline"]
        if len(outline) != chapter_count:
            print(
                f"[textbook] Warning: expected {chapter_count} chapters but got {len(outline)} "
                f"for {paper_name}; truncating to first {chapter_count}."
            )
            outline = outline[:chapter_count]

        print(f"[textbook] Parsed outline with {len(outline)} chapters.")
        print("[textbook] Writing textbook chapters in parallel...")

        def write_chapter(chapter_info):
            chapter_title = chapter_info["chapter_title"]
            print(f"[textbook] Writing chapter: {chapter_title}...")

            chapter_outline_text = f"### Chapter Title\n\n{chapter_info['chapter_title']}\n"
            chapter_outline_text += f"## Chapter Description\n\n{chapter_info['description']}\n"
            chapter_outline_text += "## Subtopics\n"
            for sec in chapter_info["subtopics"]:
                chapter_outline_text += f"- {sec}\n"

            prompt_chapter = {
                "system": """### Instructions
You will be given a chapter title, description, and subtopics and, based on those topics, your job is to write a detailed, cohesive textbook chapter addressed to a college student who is learning this material for the first time. 

The chapter should be comprehensive and suitable for someone learning this material to understand research papers in the field. Don't just briefly describe the subtopics, but rather elaborate on the concepts at full length and explain them with a focus on intuition. Spell everything out clearly so there is no ambiguity. Dedicate multiple paragraphs to each subtopic but be articulate and concise when appropriate. Write in full prose, rather than bullet points. Most importantly, please make sure that your chapter is grounded in the paper; do not provide any information or details that is not from the paper.

Start with the chapter title in the first line. Separate each subtopic with a section header "#". Also, please write all mathematical notation in LaTeX only e.g. "$x^2$" or "$\pi$". Do not use unicode mathematical characters e.g. "π". Again, PLEASE write all math in LaTeX.""",
                "user": f"""### Research Paper
{paper_content}

### Chapter Outline to Write
{chapter_outline_text}""",
            }

            chapter_content = utils.query_llm(
                prompt_chapter,
                model="gpt-5-mini",
                reasoning_effort="medium",
                system_prompt_included=True,
                max_tokens=4000,
            )
            return chapter_content

        with ThreadPoolExecutor() as executor:
            chapter_contents = list(
                tqdm(executor.map(write_chapter, outline), total=len(outline), desc=f"Writing {chapter_count}-chapter textbook")
            )

        print("[textbook] Assembling textbook...")
        full_textbook_content = "\n\n".join(chapter_contents)
        full_textbook = f"\\title{{A Textbook about the Paper: {paper_name} ({chapter_count} chapters)}}\n\n{full_textbook_content}"

        output_file = os.path.join(output_dir, f"textbook_{chapter_count}_chapters.txt")
        with open(output_file, "w") as f:
            f.write(full_textbook)

        print(f"[textbook] Saved textbook to {output_file}")


def process_papers():
    input_dir = "../../data/arxiv/cleaned/"
    files = [f for f in os.listdir(input_dir) if f.endswith(".tex") and f != "DPO.tex"]

    for filename in files:
        paper_name = os.path.splitext(filename)[0]
        print(f"==== Processing {paper_name} ====")
        generate_textbook_knowledge_variable_chapters(paper_name)


if __name__ == "__main__":
    process_papers()


