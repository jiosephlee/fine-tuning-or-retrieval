from pathlib import Path
import sys


def split_blogs(input_path: Path) -> None:
    """
    Split a combined blogs file into individual blog files.

    - Splits whenever a line starts with '# '.
    - Drops any content before the first '# ' (e.g., LaTeX \\title).
    - Writes each blog to blog_XX.txt in the same directory as the input.
    """
    text = input_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    blogs = []
    current: list[str] = []

    for line in lines:
        if line.startswith("# "):
            if current:
                blogs.append("\n".join(current).rstrip() + "\n")
            current = [line]
        else:
            if current:
                current.append(line)
            # Lines before the first '# ' are skipped entirely.

    if current:
        blogs.append("\n".join(current).rstrip() + "\n")

    out_dir = input_path.parent
    for i, blog in enumerate(blogs, start=1):
        out_path = out_dir / f"blog_{i:02d}.txt"
        out_path.write_text(blog, encoding="utf-8")


def main(argv: list[str]) -> None:
    if len(argv) > 1:
        input_path = Path(argv[1])
    else:
        # Assume this script lives in '<repo_root>/scripts/'
        repo_root = Path(__file__).resolve().parents[1]
        input_path = repo_root / "data" / "arxiv" / "explanations" / "DPO" / "blogs.txt"

    split_blogs(input_path)


if __name__ == "__main__":
    main(sys.argv)


