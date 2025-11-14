from collections import defaultdict
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
GENERAL_EXPLANATION_FILES = ["blogs.txt", "stackexchange.txt", "textbook.txt"]
HUMAN_BLOG_FILES = ["human_blog_1.txt", "human_blog_2.txt", "human_blog_3.txt"]
EXPLANATION_FILES = GENERAL_EXPLANATION_FILES + HUMAN_BLOG_FILES


def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def describe_paraphrases(tokenizer, paper_name: str) -> list[int]:
    folder = PARAPHRASED_DIR / paper_name
    if not folder.exists():
        return []
    return [
        count_tokens(tokenizer, path.read_text(encoding="utf-8"))
        for path in sorted(folder.glob("*.tex"))
    ]


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
    cleaned_lengths = []
    paraphrase_means_per_paper: list[float] = []
    paraphrase_stds_per_paper: list[float] = []
    total_paraphrase_examples = 0
    explanation_stats = defaultdict(list)
    human_blog_totals = {name: 0 for name in HUMAN_BLOG_FILES}

    for paper_path in papers:
        paper_name = paper_path.stem

        cleaned_text = paper_path.read_text(encoding="utf-8")
        cleaned_tokens = count_tokens(tokenizer, cleaned_text)
        cleaned_lengths.append(cleaned_tokens)

        paraphrase_lengths = describe_paraphrases(tokenizer, paper_name)
        paraphrase_count = len(paraphrase_lengths)
        paraphrase_mean = mean(paraphrase_lengths) if paraphrase_lengths else 0.0
        paraphrase_std = (
            pstdev(paraphrase_lengths) if len(paraphrase_lengths) > 1 else 0.0
        )
        if paraphrase_lengths:
            paraphrase_means_per_paper.append(paraphrase_mean)
            paraphrase_stds_per_paper.append(paraphrase_std)
            total_paraphrase_examples += paraphrase_count
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
                explanation_stats[name].append(count)
                if name in HUMAN_BLOG_FILES:
                    human_blog_totals[name] += count
        else:
            lines.append("Explanations: none detected")

        lines.append("")

    lines.append("Aggregate across papers")
    lines.append("----------------------")
    if cleaned_lengths:
        lines.append(f"Cleaned paper tokens (avg): {mean(cleaned_lengths):.1f}")
    if paraphrase_means_per_paper:
        lines.append(
            f"Paraphrased tokens (mean of paper means): {mean(paraphrase_means_per_paper):.1f}"
        )
        if paraphrase_stds_per_paper:
            lines.append(
                f"Paraphrased tokens (mean of paper stddevs): {mean(paraphrase_stds_per_paper):.1f}"
            )
        lines.append(f"Paraphrased examples (total): {total_paraphrase_examples}")
    else:
        lines.append("Paraphrased tokens: no paraphrases found")

    general_explanations = [
        name for name in GENERAL_EXPLANATION_FILES if explanation_stats[name]
    ]
    if general_explanations:
        lines.append("Explanations (avg across papers):")
        for name in general_explanations:
            avg_value = mean(explanation_stats[name])
            lines.append(f"  {name}: {avg_value:.1f}")

    human_lines = [
        (name, human_blog_totals[name])
        for name in HUMAN_BLOG_FILES
        if human_blog_totals[name] > 0
    ]
    if human_lines:
        lines.append("Human blogs (sum across papers):")
        for name, total in human_lines:
            lines.append(f"  {name}: {total}")

    OUTPUT_FILE.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

