import math
import argparse
import random
import re
from pathlib import Path
from typing import List, Sequence

from transformers import AutoTokenizer


EXPLANATIONS_DIR = Path("/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/data/arxiv/explanations")
MODEL_ID = "allenai/OLMo-2-1124-7B"
PCTS: Sequence[float] = tuple(p / 100.0 for p in list(range(1, 10)))  # 0.01 ... 0.09, 0.10 ... 0.90
OUTPUT_NAME_FMT_OLMO = "textbook_{pct}pct_olmo2.txt"
OUTPUT_NAME_FMT_FRUIT = "textbook_{pct}pct_fruit.txt"
OUTPUT_NAME_FMT_VEGETABLE = "textbook_{pct}pct_vegetable.txt"
OUTPUT_NAME_FMT_PRODUCE = "textbook_{pct}pct_produce.txt"

# Single-word English fruit names to avoid whitespace replacements
FRUITS: Sequence[str] = (
    "apple", "apricot", "avocado", "banana", "bilberry", "blackberry", "blueberry", "boysenberry", "cantaloupe",
    "cherry", "clementine", "cloudberry", "coconut", "cranberry", "currant", "date", "dragonfruit", "durian",
    "elderberry", "fig", "gooseberry", "grape", "grapefruit", "guava", "huckleberry", "jackfruit", "jambul",
    "jujube", "kiwi", "kumquat", "lemon", "lime", "lingonberry", "loquat", "longan", "lychee", "mandarin",
    "mango", "mulberry", "nectarine", "olive", "orange", "papaya", "passionfruit", "peach", "pear", "persimmon",
    "pineapple", "pitaya", "plantain", "plum", "pomegranate", "pomelo", "quince", "raspberry", "salak",
    "satsuma", "starfruit", "strawberry", "tamarind", "tangerine", "ugli", "watermelon", "yuzu", "ackee",
    "babaco", "barberry", "bergamot", "blackcurrant", "breadfruit", "carambola", "chokeberry", "cranapple",
    "feijoa", "goji", "grapples", "hawthorn", "honeyberry", "jabuticaba", "kaffirlime", "keylime", "kumquat",
    "langsat", "lucuma", "mangosteen", "medlar", "mirabelle", "monstera", "mulberry", "physalis", "plantain",
    "rambutan", "rowanberry", "soursop", "sugarapple", "surinamcherry", "tangelo", "tayberry", "whitecurrant"
)

VEGETABLES: Sequence[str] = (
    "artichoke", "arugula", "asparagus", "beet", "broccoli", "cabbage", "carrot", "cauliflower",
    "celery", "chard", "chicory", "collard", "corn", "cucumber", "daikon", "edamame", "eggplant",
    "endive", "fennel", "garlic", "ginger", "horseradish", "jicama", "kale", "kohlrabi", "leek",
    "lettuce", "mushroom", "mustard", "okra", "onion", "parsnip", "pea", "peppers", "potato",
    "pumpkin", "radicchio", "radish", "rutabaga", "scallion", "shallot", "spinach", "squash",
    "tomato", "turnip", "watercress", "yam", "zucchini", "beetroot", "chives", "taro", "cassava"
)


def find_textbook_files(root: Path) -> List[Path]:
    return [p for p in root.rglob("textbook.txt") if p.is_file()]


def build_allowed_token_ids(tokenizer) -> List[int]:
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is None:
        # Fallback: derive from get_vocab
        vocab = tokenizer.get_vocab()
        ids = list(vocab.values())
    else:
        ids = list(range(vocab_size))

    allowed: List[int] = []
    ws_pattern = re.compile(r"\s")
    for tid in ids:
        if tid in special_ids:
            continue
        s = tokenizer.decode([tid], skip_special_tokens=True)
        if not s or not s.strip():
            continue
        if ws_pattern.search(s):
            continue  # avoid inserting whitespace/newlines inside words
        allowed.append(tid)
    if not allowed:
        raise RuntimeError("No allowed tokens found for replacement.")
    return allowed


def replace_words_in_paragraph(
    paragraph: str,
    pct: float,
    rng: random.Random,
    tokenizer,
    allowed_token_ids: Sequence[int],
    replacement_words: Sequence[str] | None = None,
) -> str:
    if not paragraph.strip():
        return paragraph

    # Split preserving whitespace
    parts = re.split(r"(\s+)", paragraph)
    # Indices of non-whitespace tokens
    candidate_indices = [i for i, tok in enumerate(parts) if not re.fullmatch(r"\s+", tok)]
    num_words = len(candidate_indices)
    if num_words == 0:
        return paragraph

    # Fixed-size chunk policy: desired replacements per chunk = floor(100 * pct)
    desired = int(math.floor(100 * pct))
    num_to_replace = min(num_words, desired)
    if num_to_replace <= 0:
        return paragraph
    replace_positions = rng.sample(candidate_indices, k=num_to_replace)

    for idx in replace_positions:
        if replacement_words is None:
            rand_tid = rng.choice(allowed_token_ids)
            replacement = tokenizer.decode([rand_tid], skip_special_tokens=True)
        else:
            replacement = rng.choice(replacement_words)
        # As a final guard, ensure replacement is non-empty and has no whitespace
        if not replacement or not replacement.strip() or re.search(r"\s", replacement):
            continue
        parts[idx] = replacement

    return "".join(parts)


def corrupt_text_by_char_chunks(
    text: str,
    pct: float,
    rng: random.Random,
    tokenizer,
    allowed_token_ids: Sequence[int],
    replacement_words: Sequence[str] | None = None,
    chunk_size: int = 100,
) -> str:
    # Deprecated: kept for backward compatibility if referenced elsewhere.
    return text


def corrupt_text_by_word_chunks(
    text: str,
    pct: float,
    rng: random.Random,
    tokenizer,
    allowed_token_ids: Sequence[int],
    replacement_words: Sequence[str] | None = None,
    chunk_size_words: int = 100,
) -> str:
    # Split into tokens while preserving whitespace
    tokens = re.split(r"(\s+)", text)
    out_chunks: List[str] = []
    current_parts: List[str] = []
    current_words = 0

    for tok in tokens:
        current_parts.append(tok)
        if not re.fullmatch(r"\s+", tok):
            current_words += 1
        if current_words >= chunk_size_words:
            chunk_str = "".join(current_parts)
            out_chunks.append(
                replace_words_in_paragraph(
                    chunk_str, pct, rng, tokenizer, allowed_token_ids, replacement_words=replacement_words
                )
            )
            current_parts = []
            current_words = 0

    if current_parts:
        chunk_str = "".join(current_parts)
        out_chunks.append(
            replace_words_in_paragraph(
                chunk_str, pct, rng, tokenizer, allowed_token_ids, replacement_words=replacement_words
            )
        )

    return "".join(out_chunks)


def process_file(path: Path, tokenizer, allowed_token_ids: Sequence[int], replacement_words: Sequence[str] | None) -> None:
    with path.open("r", encoding="utf-8") as f:
        original = f.read()

    rng = random.Random(42)  # deterministic per run while still "random"

    for pct in PCTS:
        corrupted = corrupt_text_by_word_chunks(
            original, pct, rng, tokenizer, allowed_token_ids, replacement_words=replacement_words
        )
        pct_str = f"{int(pct * 100)}"
        if replacement_words is None:
            out_name = OUTPUT_NAME_FMT_OLMO.format(pct=pct_str)
        else:
            # Choose suffix by identity of replacement set
            if replacement_words is FRUITS:
                out_name = OUTPUT_NAME_FMT_FRUIT.format(pct=pct_str)
            elif replacement_words is VEGETABLES:
                out_name = OUTPUT_NAME_FMT_VEGETABLE.format(pct=pct_str)
            else:
                out_name = OUTPUT_NAME_FMT_PRODUCE.format(pct=pct_str)
        out_path = path.parent / out_name
        with out_path.open("w", encoding="utf-8") as f:
            f.write(corrupted)
        print(f"Wrote: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fruit", action="store_true", help="Replace with random fruit names instead of tokenizer tokens.")
    parser.add_argument("--vegetable", action="store_true", help="Replace with random vegetable names instead of tokenizer tokens.")
    parser.add_argument("--produce", action="store_true", help="Replace with random fruit and vegetable names.")
    args = parser.parse_args()

    files = find_textbook_files(EXPLANATIONS_DIR)
    if not files:
        print(f"No textbook.txt files found under {EXPLANATIONS_DIR}")
        return

    replacement_words: Sequence[str] | None = None
    if args.produce:
        replacement_words = tuple(list(FRUITS) + list(VEGETABLES))
        print("Produce mode enabled: replacements will be random fruits or vegetables.")
    elif args.vegetable:
        replacement_words = VEGETABLES
        print("Vegetable mode enabled: replacements will be random vegetable names.")
    elif args.fruit:
        replacement_words = FRUITS
        print("Fruit mode enabled: replacements will be random fruit names.")

    if replacement_words is None:
        print(f"Loading tokenizer: {MODEL_ID}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
        allowed_token_ids = build_allowed_token_ids(tokenizer)
        print(f"Allowed tokens for replacement: {len(allowed_token_ids)}")
    else:
        tokenizer = None
        allowed_token_ids = []

    for fp in files:
        print(f"Processing: {fp}")
        process_file(fp, tokenizer, allowed_token_ids, replacement_words=replacement_words)


if __name__ == "__main__":
    main()


