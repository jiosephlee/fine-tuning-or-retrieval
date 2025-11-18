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


def split_stackexchange(input_path: Path) -> None:
    """
    Split a combined StackExchange-style file into individual Q&A files.

    - Splits whenever a line starts with '### '.
    - Drops any content before the first '### ' (e.g., LaTeX \\title).
    - Writes each Q&A to stack_XX.txt under a 'stackexchange' subfolder.
    """
    text = input_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    entries = []
    current: list[str] = []

    for line in lines:
        if line.startswith("### "):
            if current:
                entries.append("\n".join(current).rstrip() + "\n")
            current = [line]
        else:
            if current:
                current.append(line)
            # Lines before the first '### ' are skipped entirely.

    if current:
        entries.append("\n".join(current).rstrip() + "\n")

    out_dir = input_path.parent / "stackexchange"
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, entry in enumerate(entries, start=1):
        out_path = out_dir / f"stack_{i:02d}.txt"
        out_path.write_text(entry, encoding="utf-8")


def main(argv: list[str]) -> None:
    # Assume this script lives in '<repo_root>/scripts/'
    repo_root = Path(__file__).resolve().parents[1]

    if len(argv) > 1 and argv[1] != "--all":
        input_path = Path(argv[1])
        if input_path.name == "blogs.txt":
            split_blogs(input_path)
        elif input_path.name == "stackexchange.txt":
            split_stackexchange(input_path)
        else:
            raise SystemExit(f"Unrecognized file name: {input_path.name}")
        return

    explanations_dir = repo_root / "data" / "arxiv" / "explanations"
    # for blogs_path in explanations_dir.rglob("blogs.txt"):
    #     split_blogs(blogs_path)
    for se_path in explanations_dir.rglob("stackexchange.txt"):
        split_stackexchange(se_path)


if __name__ == "__main__":
    main(sys.argv)


