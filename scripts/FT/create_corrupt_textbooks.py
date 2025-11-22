import argparse
import random
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Set

import pandas as pd


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

EXPLANATIONS_ROOT = Path(
    "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/data/arxiv/explanations"
)
PROBE_DOMAINS = ["1_58", "GRPO", "QLoRA", "BOFT", "OFT", "DPO"]

PROBES_ROOT = Path(
    "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/data/probes/inference"
)
REPORTS_DIR = Path(
    "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/reports"
)
COMMON_WORDS_PATH = Path(
    "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/data/misc/10K_common_words.txt"
)

PROBE_DOMAINS = ["1_58", "GRPO", "QLoRA", "BOFT", "OFT", "DPO"]


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

REPLACEMENT_WORDS: Tuple[str, ...] = tuple(list("<|endoftext|>"))

WORD_RE = re.compile(r"\w+")

# ---------------------------------------------------------------------
# REGEX FOR BOILERPLATE
# ---------------------------------------------------------------------

PREFIX_RE = re.compile(
    r"""
    ^\s*                             # leading whitespace
    (?:
        In\s+the\s+paper             # "In the paper"
      |
        According\s+to\s+the\s+paper # "According to the paper"
    )
    .*?                             # anything (title, quotes, etc.), non-greedy
    ,                               # first comma after the phrase
    \s*                             # trailing whitespace after comma
    """,
    re.IGNORECASE | re.VERBOSE | re.DOTALL,
)



# ---------------------------------------------------------------------
# STOPWORD LOADING
# ---------------------------------------------------------------------

def load_stopwords() -> Set[str]:
    """
    Load a large stopword set from multiple NLP libraries and the 10k common words file.
    Everything is lowercased.
    """
    stopwords: Set[str] = set()

    # spaCy
    try:
        from spacy.lang.en.stop_words import STOP_WORDS as SPACY_STOP_WORDS
        s = {w.lower() for w in SPACY_STOP_WORDS}
        stopwords |= s
        print(f"Loaded {len(s)} stopwords from spaCy.")
    except ImportError:
        print("spaCy not available for stopwords.")

    # scikit-learn
    try:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as SK_STOP_WORDS
        s = {w.lower() for w in SK_STOP_WORDS}
        stopwords |= s
        print(f"Loaded {len(s)} stopwords from scikit-learn.")
    except ImportError:
        print("scikit-learn not available for stopwords.")

    # NLTK
    try:
        from nltk.corpus import stopwords as nltk_stopwords
        s = {w.lower() for w in nltk_stopwords.words("english")}
        stopwords |= s
        print(f"Loaded {len(s)} stopwords from NLTK.")
    except Exception:
        print("NLTK stopwords not available (either NLTK missing or stopwords corpus not downloaded).")

    # 10k common words
    if COMMON_WORDS_PATH.exists():
        with COMMON_WORDS_PATH.open("r", encoding="utf-8") as f:
            common_words = {line.strip().lower() for line in f if line.strip()}
        stopwords |= common_words
        print(f"Loaded {len(common_words)} common words from {COMMON_WORDS_PATH}.")
    else:
        print(f"Common words file not found at: {COMMON_WORDS_PATH}")


    stopwords |= {
        "a", "an", "the", "in", "on", "at", "by", "for", "from", "to", "of", "with",
        "and", "or", "but", "if", "then", "so", "because", "as", "that", "this",
        "these", "those", "it", "its", "he", "she", "they", "them", "their",
        "we", "you", "i", "is", "are", "was", "were", "be", "been", "being", "assigns", "avoids", "equals","prevents","relies", "removes", "replaces", "selects", "underlies", "extracts", "abandoning", "ablated", "ablation", "acheives", "adapts", "aggregates", "aiming", "akin", "aggressively", "approximated", "argues", "captures", "bounds", "causally", "caveats",
    "abandoning", "ablated", "ablation", "accelerator", "achieves", "activations",
    "adapts", "additive", "aggregates", "aggressively", "agnostic", "aided",
    "aiming", "akin", "algebraic", "algorithmic", "aligns", "alphabet",
    "alternating", "amplitude", "analogous", "analogy", "angles", "approximated",
    "argues", 
    "balances",
    "behaves",
    "caveats",
    "concentrated",
    "concise",
    "costly",
    "deliberately",
    "demonstrating",
    "depended",
    "divides",
    "doubled",
    "eliminates",
    "eliminating",
    "exhibited",
    "exploiting",
    "favoring",
    "folds",
    "footprint",
    "forgetting",
    "forgoes",
    "forgoing",
 "imperfect", "implicit", "implicitly", "imply", "imposing",
    "improves", "inadequate", "incurs", 
    "inefficiency", "initialize", "initialized",
    "initializing", "instability", "intact", "intentionally",
    "interpret", "interpreting", "intuitive", "irrelevant",
    "isolates", "iterations", "iterative", "judgments", "justified", "justifies",
    "magnitudes", "manifests", "mapped", "mathematically", "minimized", "minimizing", "mitigate", "mitigates",
    "modalities", "modifies", "modifying", "motivates", "motivating", 
    "multiplies", "multiply", "multiplying", "naive",  "notable",
    "noticeable", "observing", "outperforms","paired", "pairwise", "paradigm", "parity", "partitioned", "penalize", "penalizes",
    "plausible", "premise", "preserved", "preserves", "preserving", "prevented", "prioritize", "probabilities", 
    "proportional", "proportionally", "proposes", "proving","randomly",
 "reappear", "reconstructs", "recur",
    "reintroduce", "relational", "reliably", "repeatedly", "repeating",
    "resembles", "reside", "restricting", "rests", "reverted", "rewritten",
    "sensible","similarities",
    "similarity", "simplification", 
    "stabilizing", "storing", "subset", "subtract", "subtraction", "subtracts",
 "suffices", "summarized",
    "tends", "theoretic", "theoretically", "tradeoff", "trainable", "unchanged", "underscoring", "unstated", "unstructured", "usefulness",
    "variant", "variants", "warn",
    }


    print(f"Total unique stopwords (including 10k common words if present): {len(stopwords)}")
    return stopwords


# ---------------------------------------------------------------------
# PROBE CSV DISCOVERY
# ---------------------------------------------------------------------

def collect_probe_csv_paths() -> List[Path]:
    """
    Collect probes_v7.csv for each domain in PROBE_DOMAINS, if it exists.
    Expected path: PROBES_ROOT / {domain} / "probes_v7.csv"
    """
    csvs: List[Path] = []
    for dom in PROBE_DOMAINS:
        p = PROBES_ROOT / dom / "probes_v7.csv"
        if p.exists():
            csvs.append(p)
        else:
            print(f"Warning: probes CSV not found for domain {dom}: {p}")
    if not csvs:
        raise FileNotFoundError(
            f"No probes_v7.csv files found under {PROBES_ROOT} for domains {PROBE_DOMAINS}"
        )
    print("Using the following probes CSV files:")
    for p in csvs:
        print(f"  - {p}")
    return csvs


# ---------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------

def find_chapter_files(root: Path) -> List[Path]:
    pattern = "chapter_*.txt"
    return [p for p in root.rglob(pattern) if p.is_file() and "textbooks" in p.parts]


def extract_domain_from_path(path: Path) -> str:
    rel = path.relative_to(EXPLANATIONS_ROOT)
    return rel.parts[0]


def strip_fact_prefix_with_assert(raw_fact: str) -> str:
    """
    Strip boilerplate like:
      "In the paper 'Title', ..."
      "According to the paper 'Title', ..."
    using regex. Assert that *every* fact matches this pattern.
    """
    s = raw_fact.strip()
    m = PREFIX_RE.match(s)
    assert m is not None, f"Fact did not match expected boilerplate pattern: {s[:120]}..."
    stripped = s[m.end():].strip()
    return stripped


def build_probe_word_set_from_many(csv_paths: List[Path], stopwords: Set[str]) -> Set[str]:
    """
    Build a global probe word set from multiple probes_v7.csv files (across domains),
    applying boilerplate stripping, punctuation removal, lowercasing, and stopword filtering.
    """
    probe_words: Set[str] = set()
    total_facts = 0

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        if "fact" not in df.columns:
            raise ValueError(f"'fact' column not found in {csv_path}")
        facts = df["fact"].dropna().astype(str)
        total_facts += len(facts)

        for raw_fact in facts:
            # strip boilerplate
            fact = strip_fact_prefix_with_assert(raw_fact)

            # remove punctuation
            fact = re.sub(r"[^\w\s]", " ", fact)

            # lowercase
            fact = fact.lower()

            for w in WORD_RE.findall(fact):
                if w in stopwords:
                    continue
                if len(w) <= 1:
                    continue
                probe_words.add(w)

    print(
        f"Collected {len(probe_words)} unique probe words from {len(csv_paths)} CSVs "
        f"({total_facts} total facts, after boilerplate strip + punctuation removal + stopwords)."
    )
    return probe_words


def replace_probe_words_in_text(
    text: str,
    probe_words: Set[str],
    rng: random.Random,
) -> Tuple[str, int, int, Set[str]]:
    """
    Replace any word that appears in probe_words with a random fruit/vegetable.

    Returns:
      (new_text, total_words, replaced_words, removed_words_set)
    """
    tokens = re.split(r"(\w+)", text)

    total_words = 0
    replaced_words = 0
    removed_words: Set[str] = set()

    for i, tok in enumerate(tokens):
        if not tok:
            continue

        if re.fullmatch(r"\w+", tok):
            total_words += 1
            base = tok.lower()
            if base in probe_words:
                tokens[i] = rng.choice(REPLACEMENT_WORDS)
                replaced_words += 1
                removed_words.add(base)

    new_text = "".join(tokens)
    return new_text, total_words, replaced_words, removed_words


# ---------------------------------------------------------------------
# MAIN PROCESSING
# ---------------------------------------------------------------------

def process_chapter_file(
    path: Path,
    probe_words: Set[str],
    rng: random.Random,
) -> Tuple[str, int, int, Set[str]]:
    domain = extract_domain_from_path(path)

    with path.open("r", encoding="utf-8") as f:
        original = f.read()

    corrupted, total_words, replaced_words, removed_words = replace_probe_words_in_text(
        original, probe_words, rng
    )

    # Write corrupted chapter into {domain}/fruit_textbooks/chapter_X.txt
    domain_dir = EXPLANATIONS_ROOT / domain
    fruit_dir = domain_dir / "fruit_textbooks"
    fruit_dir.mkdir(parents=True, exist_ok=True)
    out_path = fruit_dir / path.name

    with out_path.open("w", encoding="utf-8") as f:
        f.write(corrupted)

    print(
        f"Wrote corrupted chapter: {out_path} "
        f"(total_words={total_words}, replaced={replaced_words})"
    )

    return domain, total_words, replaced_words, removed_words


def write_report(
    per_chapter_stats: List[Dict],
    report_path: Path,
    probe_words: Set[str],
    probe_csv_paths: List[Path],
) -> None:
    """
    Write a text report with:
      - global probe word pool
      - overall stats
      - per-domain stats
      - per-chapter stats + removed words per chapter
      - list of probe CSVs used
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Aggregate per domain and overall.
    domain_agg: Dict[str, Dict[str, float]] = {}
    overall_total_words = 0
    overall_replaced_words = 0

    for row in per_chapter_stats:
        domain = row["domain"]
        total_words = row["total_words"]
        replaced_words = row["replaced_words"]

        if domain not in domain_agg:
            domain_agg[domain] = {
                "total_words": 0,
                "replaced_words": 0,
                "chapters": 0,
            }

        domain_agg[domain]["total_words"] += total_words
        domain_agg[domain]["replaced_words"] += replaced_words
        domain_agg[domain]["chapters"] += 1

        overall_total_words += total_words
        overall_replaced_words += replaced_words

    overall_pct = (
        100.0 * overall_replaced_words / overall_total_words
        if overall_total_words > 0
        else 0.0
    )

    probe_words_sorted = sorted(probe_words)

    with report_path.open("w", encoding="utf-8") as f:
        f.write("--- Fruit Textbook Replacement Report ---\n\n")

        # Probe CSVs used
        f.write("Probe CSV files used to build the global probe word pool:\n")
        for p in probe_csv_paths:
            f.write(f"  - {p}\n")
        f.write("\n")

        # Global probe word pool (post-stopwords)
        f.write("Global probe word pool (after stopword/common-word removal):\n")
        f.write(f"  Total probe words: {len(probe_words_sorted)}\n")
        f.write("  Words:\n")
        if probe_words_sorted:
            f.write("    " + ", ".join(probe_words_sorted) + "\n\n")
        else:
            f.write("    (none)\n\n")

        # Overall
        f.write("Overall statistics (word-count weighted):\n")
        f.write(
            f"  Total words          : {overall_total_words}\n"
            f"  Total replaced words : {overall_replaced_words}\n"
            f"  Overall replacement %: {overall_pct:.2f}%\n\n"
        )

        # Per-domain aggregates
        f.write("Per-domain statistics (word-count weighted):\n")
        for domain, agg in sorted(domain_agg.items()):
            dw = agg["total_words"]
            dr = agg["replaced_words"]
            dpct = 100.0 * dr / dw if dw > 0 else 0.0
            f.write(
                f"  Domain {domain}:\n"
                f"    Chapters          : {agg['chapters']}\n"
                f"    Total words       : {dw}\n"
                f"    Replaced words    : {dr}\n"
                f"    Replacement %     : {dpct:.2f}%\n"
            )
        f.write("\n")

        # Per-chapter stats + list of removed words
        f.write("Per-chapter statistics and removed words:\n")
        for row in per_chapter_stats:
            removed_sorted = sorted(row["removed_words"])
            removed_str = ", ".join(removed_sorted) if removed_sorted else "(none)"
            f.write(
                f"  Domain={row['domain']}, chapter={row['chapter_name']}:\n"
                f"    Total words    : {row['total_words']}\n"
                f"    Replaced words : {row['replaced_words']}\n"
                f"    Replacement %  : {row['replacement_pct']:.2f}%\n"
                f"    Removed words  : {removed_str}\n"
            )

    print(f"Report written to: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for fruit/vegetable replacements.",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # 1. Load stopwords (NLP libs + 10k common words).
    stopwords = load_stopwords()

    # 2. Collect all probes_v7.csv paths across domains.
    probe_csv_paths = collect_probe_csv_paths()

    # 3. Build global probe word set from all those CSVs.
    probe_words = build_probe_word_set_from_many(probe_csv_paths, stopwords)

    # 4. Find all chapter files.
    chapter_files = find_chapter_files(EXPLANATIONS_ROOT)
    if not chapter_files:
        print(f"No chapter_*.txt files found under {EXPLANATIONS_ROOT}")
        return

    print(f"Found {len(chapter_files)} chapter files to process.")

    # 5. Process each chapter.
    per_chapter_stats: List[Dict] = []

    for path in sorted(chapter_files):
        domain, total_words, replaced_words, removed_words = process_chapter_file(
            path, probe_words, rng
        )
        pct = 100.0 * replaced_words / total_words if total_words > 0 else 0.0
        per_chapter_stats.append(
            {
                "domain": domain,
                "chapter_name": path.name,
                "total_words": total_words,
                "replaced_words": replaced_words,
                "replacement_pct": pct,
                "removed_words": removed_words,
            }
        )

    # 6. Write report (includes global probe word pool and CSV list).
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "fruit_textbook_replacement_report.txt"
    write_report(per_chapter_stats, report_path, probe_words, probe_csv_paths)


if __name__ == "__main__":
    main()