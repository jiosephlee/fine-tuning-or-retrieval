"""Split combined explanation .txt files into per-item subfolders.

Creates textbooks/, blogs/, stackexchange/ subdirectories with individual
chapter/blog/qa files matching the format used by data_preparation.py's
granular_explanation_analysis mode.
"""
import os
import re
import sys


def split_textbook(explanations_dir):
    """Split textbook.txt into textbooks/chapter_N.txt files."""
    textbook_path = os.path.join(explanations_dir, "textbook.txt")
    if not os.path.exists(textbook_path):
        print(f"  No textbook.txt found, skipping")
        return

    with open(textbook_path) as f:
        content = f.read()

    # Extract title line (first line)
    lines = content.split("\n")
    title_line = lines[0] if lines[0].startswith("\\title{") else ""

    # Split on chapter title lines (bare line between blank lines, before first #)
    # Pattern: the outline generates "Chapter Title\n\n# Section..."
    # But the combined textbook joins chapters with \n\n
    # Look at the outline to find chapter titles
    outline_path = os.path.join(explanations_dir, "textbook_outline.json")
    if os.path.exists(outline_path):
        import json
        with open(outline_path) as f:
            outline = json.load(f)
        if isinstance(outline, dict) and "outline" in outline:
            outline = outline["outline"]
        chapter_titles = [ch["chapter_title"] for ch in outline]
    else:
        # Fallback: find lines that look like chapter titles (non-# lines between sections)
        chapter_titles = []

    # Split by finding chapter title occurrences in the text
    chapters = []
    if chapter_titles:
        # Build regex that matches any chapter title as a line
        for i, title in enumerate(chapter_titles):
            escaped = re.escape(title)
            pattern = escaped
            # Find position of this title in the content
            match = re.search(r'\n' + pattern + r'\n', content)
            if match:
                chapters.append(match.start() + 1)  # +1 to skip the \n before

        # Extract chapter content between boundaries
        chapter_contents = []
        for i, start in enumerate(chapters):
            end = chapters[i + 1] if i + 1 < len(chapters) else len(content)
            chapter_contents.append(content[start:end].strip())
    else:
        print(f"  No outline found, skipping textbook split")
        return

    if not chapter_contents:
        print(f"  Could not find chapter boundaries, skipping")
        return

    out_dir = os.path.join(explanations_dir, "textbooks")
    os.makedirs(out_dir, exist_ok=True)
    for i, ch in enumerate(chapter_contents):
        chapter_text = f"{title_line}\n\nChapter {i+1}: {ch}"
        path = os.path.join(out_dir, f"chapter_{i+1}.txt")
        with open(path, "w") as f:
            f.write(chapter_text)
    print(f"  textbooks/: {len(chapter_contents)} chapters")


def split_blogs(explanations_dir):
    """Split blogs.txt into blogs/blog_NN.txt files."""
    blogs_path = os.path.join(explanations_dir, "blogs.txt")
    if not os.path.exists(blogs_path):
        print(f"  No blogs.txt found, skipping")
        return

    with open(blogs_path) as f:
        content = f.read()

    # Blogs start with "# Title" (h1 headers)
    parts = re.split(r'\n(?=# )', content)
    # First part might be a header/preamble before the first blog
    blogs = []
    for part in parts:
        part = part.strip()
        if part.startswith("# "):
            blogs.append(part)

    if not blogs:
        print(f"  No blogs found in blogs.txt, skipping")
        return

    out_dir = os.path.join(explanations_dir, "blogs")
    os.makedirs(out_dir, exist_ok=True)
    for i, blog in enumerate(blogs):
        path = os.path.join(out_dir, f"blog_{i+1:02d}.txt")
        with open(path, "w") as f:
            f.write(blog)
    print(f"  blogs/: {len(blogs)} posts")


def split_stackexchange(explanations_dir):
    """Split stackexchange.txt into stackexchange/stack_NN.txt files."""
    se_path = os.path.join(explanations_dir, "stackexchange.txt")
    if not os.path.exists(se_path):
        print(f"  No stackexchange.txt found, skipping")
        return

    with open(se_path) as f:
        content = f.read()

    # Q&A blocks start with "### Question Title"
    parts = re.split(r'\n(?=### )', content)
    qas = []
    for part in parts:
        part = part.strip()
        if part.startswith("### "):
            qas.append(part)

    if not qas:
        print(f"  No Q&A blocks found in stackexchange.txt, skipping")
        return

    out_dir = os.path.join(explanations_dir, "stackexchange")
    os.makedirs(out_dir, exist_ok=True)
    for i, qa in enumerate(qas):
        path = os.path.join(out_dir, f"stack_{i+1:02d}.txt")
        with open(path, "w") as f:
            f.write(qa)
    print(f"  stackexchange/: {len(qas)} Q&As")


def process_paper(paper_name, base_dir):
    explanations_dir = os.path.join(base_dir, paper_name)
    if not os.path.isdir(explanations_dir):
        print(f"Skipping {paper_name}: no explanations directory")
        return
    print(f"Processing {paper_name}...")
    split_textbook(explanations_dir)
    split_blogs(explanations_dir)
    split_stackexchange(explanations_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--papers", nargs="+", help="Paper names to process")
    parser.add_argument("--base_dir", default="../../data/arxiv/explanations/",
                        help="Base explanations directory")
    args = parser.parse_args()

    if args.papers:
        paper_names = args.papers
    else:
        paper_names = [d for d in os.listdir(args.base_dir)
                       if os.path.isdir(os.path.join(args.base_dir, d))]

    for name in sorted(paper_names):
        process_paper(name, args.base_dir)
