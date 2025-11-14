from pathlib import Path
from statistics import mean, pstdev

from transformers import AutoTokenizer

TOKENIZER_NAME = "allenai/OLMo-2-0425-1B"
BASE_DIR = Path("/Users/jlee0/Desktop/research/fine-tuning-or-retrieval")
DATA_DIR = BASE_DIR / "data" / "arxiv"
CLEANED_DIR = DATA_DIR / "cleaned"
PARAPHRASED_DIR = DATA_DIR / "paraphrased"
EXPLANATIONS_DIR = DATA_DIR / "explanations"
OUTPUT_FILE = BASE_DIR / "reports" / "token_lengths_summary.txt"
EXPLANATION_FILES = [
    "blogs.txt",
    "human_blog_1.txt",
    "human_blog_2.txt",
    "human_blog_3.txt",
    "stackexchange.txt",
    "textbook.txt",
]


def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def describe_paraphrases(tokenizer, paper_name: str) -> tuple[int, float, float]:
    folder = PARAPHRASED_DIR / paper_name
    if not folder.exists():
        return 0, 0.0
    lengths = [
        count_tokens(tokenizer, path.read_text(encoding="utf-8"))
        for path in sorted(folder.glob("*.tex"))
    ]
    if not lengths:
        return 0, 0.0, 0.0
    return (
        len(lengths),
        mean(lengths),
        pstdev(lengths) if len(lengths) > 1 else 0.0,
    )


def describe_explanations(tokenizer, paper_name: str) -> dict[str, int]:
    folder = EXPLANATIONS_DIR / paper_name
    if not folder.exists():
        return {}
    result = {}
    for name in EXPLANATION_FILES:
        path = folder / name
        if not path.exists():
            continue
        result[name] = count_tokens(tokenizer, path.read_text(encoding="utf-8"))
    return result


def main():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    papers = sorted(CLEANED_DIR.glob("*.tex"))

    lines = ["Token length analysis", "=" * 21, ""]
    for paper_path in papers:
        paper_name = paper_path.stem

        cleaned_text = paper_path.read_text(encoding="utf-8")
        cleaned_tokens = count_tokens(tokenizer, cleaned_text)

        paraphrase_count, paraphrase_mean, paraphrase_std = describe_paraphrases(
            tokenizer, paper_name
        )
        explanation_lengths = describe_explanations(tokenizer, paper_name)

        lines.append(f"{paper_name}")
        lines.append("-" * len(paper_name))
        lines.append(f"Cleaned paper tokens: {cleaned_tokens}")

        if paraphrase_count:
            lines.append(f"Paraphrased tokens (mean): {paraphrase_mean:.1f}")
            lines.append(f"Paraphrased tokens (stddev): {paraphrase_std:.1f}")
            lines.append(f"Paraphrased examples: {paraphrase_count}")
        else:
            lines.append("Paraphrased tokens: no paraphrases found")

        if explanation_lengths:
            lines.append("Explanations:")
            for name, count in explanation_lengths.items():
                lines.append(f"  {name}: {count}")
        else:
            lines.append("Explanations: none detected")

        lines.append("")

    OUTPUT_FILE.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

