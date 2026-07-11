"""Split combined explanation .txt files into per-item subfolders.

Creates textbooks/, blogs/, stackexchange/ subdirectories with individual
chapter/blog/qa files matching the format used by data_preparation.py's
granular_explanation_analysis mode.
"""
import os
import re
import shutil
import sys


HYPHEN_CHARS = "-\u2010\u2011\u2012\u2013\u2014\u2015\u2212\ufe58\ufe63\uff0d"
QUOTE_CHARS = "\"'\u2018\u2019\u201a\u201b\u201c\u201d\u201e\u201f\u00ab\u00bb"


def title_pattern(title):
    """Return a regex pattern for a title, tolerant to whitespace and hyphen variants."""
    pieces = []
    for ch in title:
        if ch.isspace():
            pieces.append(r"\s+")
        elif ch in HYPHEN_CHARS:
            pieces.append(f"[{re.escape(HYPHEN_CHARS)}]")
        elif ch in QUOTE_CHARS:
            pieces.append(f"[{re.escape(QUOTE_CHARS)}]")
        else:
            pieces.append(re.escape(ch))
    return "".join(pieces)


def load_outline_titles(explanations_dir, outline_filename, key, title_key):
    outline_path = os.path.join(explanations_dir, outline_filename)
    if not os.path.exists(outline_path):
        return []
    import json
    with open(outline_path) as f:
        outline_data = json.load(f)
    if isinstance(outline_data, dict):
        outline_data = outline_data.get(key, [])
    return [item.get(title_key) for item in outline_data if item.get(title_key)]


def split_by_outline_titles(content, titles, min_prefix_chars=200):
    boundaries = []
    for title in titles:
        pattern = r'\n(?:#{1,6}\s+)?' + title_pattern(title) + r'\s*\n'
        match = re.search(pattern, content, flags=re.IGNORECASE)
        if match:
            boundaries.append(match.start() + 1)
    boundaries.sort()
    chunks = []
    for i, start in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(content)
        chunks.append(content[start:end].strip())
    # If the first outline title never matched (e.g. the generator wrote its own
    # headings), the content before the first matched boundary would be silently
    # dropped. Recover it as a leading chunk when it is substantial.
    if boundaries:
        prefix = content[:boundaries[0]]
        prefix_lines = prefix.split("\n")
        if prefix_lines and (prefix_lines[0].startswith("\\title{") or prefix_lines[0].startswith("Title:")):
            prefix = "\n".join(prefix_lines[1:])
        prefix = prefix.strip()
        if len(prefix) >= min_prefix_chars:
            chunks.insert(0, prefix)
    return chunks


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
    title_line = lines[0] if (lines[0].startswith("\\title{") or lines[0].startswith("Title:")) else ""

    # Split on chapter title lines (bare line between blank lines, before first #)
    # Pattern: the outline generates "Chapter Title\n\n# Section..."
    # But the combined textbook joins chapters with \n\n
    # Look at the outline to find chapter titles
    chapter_titles = load_outline_titles(explanations_dir, "textbook_outline.json", "outline", "chapter_title")
    if not chapter_titles:
        chapter_titles = load_outline_titles(explanations_dir, "textbook_outline.json", "sections", "section_title")

    if chapter_titles:
        chapter_contents = split_by_outline_titles(content, chapter_titles)
    else:
        print(f"  No outline found, skipping textbook split")
        return

    if not chapter_contents:
        print(f"  Could not find chapter boundaries, skipping")
        return

    out_dir = os.path.join(explanations_dir, "textbooks")
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    for i, ch in enumerate(chapter_contents):
        ch = ch.strip()
        # Some corpora (e.g. glm) format chapter titles as markdown headings;
        # drop the leading hashes so files read "Chapter N: Title" like the rest.
        ch = re.sub(r"^#{1,6}\s+", "", ch)
        if title_line:
            chapter_text = f"{title_line}\n\nChapter {i+1}: {ch}"
        else:
            chapter_text = f"Chapter {i+1}: {ch}"
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

    lines = content.split("\n")
    title_line = lines[0] if (lines[0].startswith("\\title{") or lines[0].startswith("Title:")) else ""

    def split_by_h1_headers():
        parts = re.split(r'\n(?=# )', content)
        return [part.strip() for part in parts if part.strip().startswith("# ")]

    blog_titles = load_outline_titles(explanations_dir, "blog_outline.json", "blogs", "title")
    blogs = split_by_outline_titles(content, blog_titles) if blog_titles else []
    if blog_titles and len(blogs) != len(blog_titles):
        h1_blogs = split_by_h1_headers()
        if len(h1_blogs) > len(blogs):
            blogs = h1_blogs
    if not blogs:
        # Blogs often start with "# Title" (h1 headers); this fallback preserves
        # legacy behavior for outputs without a usable outline.
        blogs = split_by_h1_headers()

    if not blogs:
        print(f"  No blogs found in blogs.txt, skipping")
        return

    out_dir = os.path.join(explanations_dir, "blogs")
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    for i, blog in enumerate(blogs):
        if title_line:
            blog = f"{title_line}\n\n{blog}"
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

    lines = content.split("\n")
    title_line = lines[0] if (lines[0].startswith("\\title{") or lines[0].startswith("Title:")) else ""

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
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    for i, qa in enumerate(qas):
        if title_line:
            qa = f"{title_line}\n\n{qa}"
        path = os.path.join(out_dir, f"stack_{i+1:02d}.txt")
        with open(path, "w") as f:
            f.write(qa)
    print(f"  stackexchange/: {len(qas)} Q&As")


PART_SPLITTERS = {
    "textbook": split_textbook,
    "blog": split_blogs,
    "stackexchange": split_stackexchange,
}


def resolve_parts(parts):
    if not parts or "all" in parts:
        return list(PART_SPLITTERS)
    return parts


def process_paper(paper_name, base_dir, parts=None):
    explanations_dir = os.path.join(base_dir, paper_name)
    if not os.path.isdir(explanations_dir):
        print(f"Skipping {paper_name}: no explanations directory")
        return
    print(f"Processing {paper_name}...")
    for part in resolve_parts(parts):
        PART_SPLITTERS[part](explanations_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--papers", nargs="+", help="Paper names to process")
    parser.add_argument("--base_dir", default="../../data/arxiv/explanations/",
                        help="Base explanations directory")
    parser.add_argument(
        "--parts",
        nargs="+",
        choices=["textbook", "blog", "stackexchange", "all"],
        default=["all"],
        help="Explanation subfolders to rebuild. Default: all."
    )
    args = parser.parse_args()

    if args.papers:
        paper_names = args.papers
    else:
        paper_names = [d for d in os.listdir(args.base_dir)
                       if os.path.isdir(os.path.join(args.base_dir, d))]

    for name in sorted(paper_names):
        process_paper(name, args.base_dir, args.parts)
