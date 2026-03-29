import argparse
from pathlib import Path

import pipeline_paraphrase_text_v1 as pipeline

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"


def preview_text(text, limit=100):
    single_line = " ".join(text.strip().split())
    if len(single_line) <= limit:
        return single_line
    return single_line[: limit - 3] + "..."


def classify_paragraph(paragraph):
    if pipeline.is_latex_only_paragraph(paragraph):
        return "skipped_latex_only"
    return "paraphrase"


def format_paragraphs(paragraphs, preview_chars, title):
    lines = [f"\n{title}:"]
    for idx, paragraph in enumerate(paragraphs):
        status = classify_paragraph(paragraph)
        word_count = len(paragraph.split())
        char_count = len(paragraph)
        latex_only = pipeline.is_latex_only_paragraph(paragraph)
        lines.append(
            f"[{idx:03d}] {status:10} words={word_count:4d} chars={char_count:4d} "
            f"latex_only={str(latex_only):5} preview={preview_text(paragraph, preview_chars)}"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Debug paragraph processing for paraphrase pipeline."
    )
    parser.add_argument("input_tex", help="Path to the input .tex file")
    parser.add_argument(
        "--preview_chars",
        type=int,
        default=100,
        help="Number of preview characters to print per paragraph",
    )
    args = parser.parse_args()

    input_path = Path(args.input_tex)
    text = input_path.read_text(encoding="utf-8")

    raw_paragraphs = text.split("\n\n")
    paragraphs = pipeline.split_and_merge_paragraphs(text)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = ARTIFACTS_DIR / f"{input_path.stem}_paraphrase_processing_debug.txt"

    report = [
        f"Debugging paragraph processing for {input_path}",
        format_paragraphs(raw_paragraphs, args.preview_chars, "Raw paragraphs"),
        format_paragraphs(paragraphs, args.preview_chars, "Processed paragraphs"),
        "",
    ]

    output_path.write_text("\n".join(report), encoding="utf-8")
    print(f"Saved debug report to {output_path}")


if __name__ == "__main__":
    main()
