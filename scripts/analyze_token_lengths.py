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
EXPLANATION_SUBFOLDERS = ["blogs", "stackexchange", "textbooks"]
HUMAN_BLOG_FILES = ["human_blog_1.txt", "human_blog_2.txt", "human_blog_3.txt"]


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


def describe_explanations(tokenizer, paper_name: str) -> dict[str, list[int]]:
    folder = EXPLANATIONS_DIR / paper_name
    if not folder.exists():
        return {}
    result = {}
    
    for subfolder_name in EXPLANATION_SUBFOLDERS:
        subfolder = folder / subfolder_name
        if not subfolder.exists() or not subfolder.is_dir():
            continue
        lengths = [
            count_tokens(tokenizer, path.read_text(encoding="utf-8"))
            for path in sorted(subfolder.glob("*.txt"))
        ]
        if lengths:
            result[subfolder_name] = lengths
    
    for name in HUMAN_BLOG_FILES:
        path = folder / name
        if path.exists():
            if "human_blogs" not in result:
                result["human_blogs"] = []
            result["human_blogs"].append(count_tokens(tokenizer, path.read_text(encoding="utf-8")))
    
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
    human_blog_lengths = []

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
            for name, lengths_list in explanation_lengths.items():
                if not lengths_list:
                    continue
                avg_val = mean(lengths_list)
                min_val = min(lengths_list)
                max_val = max(lengths_list)
                lines.append(f"  {name}: avg={avg_val:.1f}, min={min_val}, max={max_val}, count={len(lengths_list)}")
                explanation_stats[name].extend(lengths_list)
                if name == "human_blogs":
                    human_blog_lengths.extend(lengths_list)
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

    explanation_types = [
        name for name in EXPLANATION_SUBFOLDERS if explanation_stats[name]
    ]
    if explanation_types:
        lines.append("Explanations (aggregate across all files):")
        for name in explanation_types:
            lengths = explanation_stats[name]
            avg_value = mean(lengths)
            min_value = min(lengths)
            max_value = max(lengths)
            lines.append(f"  {name}: avg={avg_value:.1f}, min={min_value}, max={max_value}, count={len(lengths)}")

    if human_blog_lengths:
        lines.append("Human blogs (aggregate across all files):")
        avg_value = mean(human_blog_lengths)
        min_value = min(human_blog_lengths)
        max_value = max(human_blog_lengths)
        lines.append(f"  avg={avg_value:.1f}, min={min_value}, max={max_value}, count={len(human_blog_lengths)}")

    OUTPUT_FILE.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

