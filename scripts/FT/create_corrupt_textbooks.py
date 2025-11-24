import argparse
import random
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Set
from collections import Counter

import pandas as pd
import string

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

ARXIV_CLEANED_ROOT = Path(
    "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/data/arxiv/cleaned"
)

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

# Kept for consistency even though we now always use "<|endoftext|>" directly.
REPLACEMENT_WORDS: Tuple[str, ...] = tuple(["<|endoftext|>"])

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

def normalize_token(tok: str) -> str:
    """
    Normalise a token for all checks/counts:
      - lowercase
      - strip any leading/trailing punctuation characters
        (keep punctuation in the middle, e.g. '1.58', 'x_y', '\\ell_2').
    """
    if tok is None:
        return ""
    tok = tok.lower()
    # strip punctuation only at the ends, not inside
    return tok.strip(string.punctuation)
    
def load_stopwords() -> Set[str]:
    """
    Load a large stopword set from multiple NLP libraries and the 10k common words file,
    plus custom hand-curated words. Everything is lowercased.
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

    # Extra curated stopwords / “do not corrupt” words
    # stopwords |= {
    #     "a", "an", "the", "in", "on", "at", "by", "for", "from", "to", "of", "with",
    #     "and", "or", "but", "if", "then", "so", "because", "as", "that", "this",
    #     "these", "those", "it", "its", "he", "she", "they", "them", "their",
    #     "we", "you", "i", "is", "are", "was", "were", "be", "been", "being",
    #     "assigns", "avoids", "equals", "prevents", "relies", "removes", "replaces",
    #     "selects", "underlies", "extracts",
    #     "abandoning", "ablated", "ablation", "acheives", "adapts", "aggregates",
    #     "aiming", "akin", "aggressively", "approximated", "argues", "captures",
    #     "bounds", "causally", "caveats",
    #     "accelerator", "achieves", "activations",
    #     "additive", "agnostic", "aided",
    #     "algebraic", "algorithmic", "aligns", "alphabet",
    #     "alternating", "amplitude", "analogous", "analogy", "angles",
    #     "balances", "behaves", "concentrated", "concise",
    #     "costly", "deliberately", "demonstrating", "depended",
    #     "divides", "doubled", "eliminates", "eliminating", "exhibited", "exploiting",
    #     "favoring", "folds", "footprint", "forgetting", "forgoes", "forgoing",
    #     "imperfect", "implicit", "implicitly", "imply", "imposing",
    #     "improves", "inadequate", "incurs",
    #     "inefficiency", "initialize", "initialized",
    #     "initializing", "instability", "intact", "intentionally",
    #     "interpret", "interpreting", "intuitive", "irrelevant",
    #     "isolates", "iterations", "iterative", "judgments", "justified", "justifies",
    #     "magnitudes", "manifests", "mapped", "mathematically", "minimized", "minimizing",
    #     "mitigate", "mitigates", "modalities", "modifies", "modifying",
    #     "motivates", "motivating",
    #     "multiplies", "multiply", "multiplying", "naive",
    #     "notable", "noticeable", "observing", "outperforms",
    #     "paired", "pairwise", "paradigm", "parity", "partitioned", "penalize",
    #     "penalizes", "plausible", "premise", "preserved", "preserves", "preserving",
    #     "prevented", "prioritize", "probabilities",
    #     "proportional", "proportionally", "proposes", "proving", "randomly",
    #     "reappear", "reconstructs", "recur",
    #     "reintroduce", "relational", "reliably", "repeatedly", "repeating",
    #     "resembles", "reside", "restricting", "rests", "reverted", "rewritten",
    #     "sensible", "similarities", "similarity", "simplification",
    #     "stabilizing", "storing", "subset", "subtract", "subtraction", "subtracts",
    #     "suffices", "summarized",
    #     "tends", "theoretic", "theoretically", "tradeoff", "trainable",
    #     "unchanged", "underscoring", "unstated", "unstructured", "usefulness",
    #     "variant", "variants", "warn", "butterfly", "parameter-efficient", "parameter", "efficient",
    #     "parameter", "efficient", "orthogonal", "finetuning", "via", "factorization", "llms", "1.58", "direct", "preference", "optimization", "Your", "Language", "Model", "Secretly", "Reward", "quantized" 
    # }

    print(f"Total unique stopwords (including 10k common words if present): {len(stopwords)}")
    return stopwords


# ---------------------------------------------------------------------
# PROBE / SOURCE FILE DISCOVERY
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


def collect_source_tex_paths() -> List[Path]:
    """
    Collect cleaned arxiv .tex source files for each domain, if they exist.
    Expected path: ARXIV_CLEANED_ROOT / f"{domain}.tex"
    """
    tex_paths: List[Path] = []
    for dom in PROBE_DOMAINS:
        p = ARXIV_CLEANED_ROOT / f"{dom}.tex"
        if p.exists():
            tex_paths.append(p)
        else:
            print(f"Warning: source .tex not found for domain {dom}: {p}")
    if not tex_paths:
        print(f"Warning: no source .tex files found under {ARXIV_CLEANED_ROOT}")
    else:
        print("Using the following source .tex files:")
        for p in tex_paths:
            print(f"  - {p}")
    return tex_paths


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

def tokenize_for_counts(text: str) -> List[str]:
    """
    Tokenisation for corpus counts:
    - split on whitespace (space-separated clumps),
    - normalise each token via `normalize_token`,
    - drop empty tokens after normalisation.
    """
    raw_tokens = text.split()
    tokens: List[str] = []
    for t in raw_tokens:
        norm = normalize_token(t)
        if norm:
            tokens.append(norm)
    return tokens


def get_word_counts_from_files(files: List[Path]) -> Tuple[Counter, int]:
    """
    Build word count and total token count from a list of files,
    using punctuation-aware whitespace tokenisation.
    """
    counts: Counter = Counter()
    total = 0
    for p in files:
        with p.open("r", encoding="utf-8") as f:
            txt = f.read()
        tokens = tokenize_for_counts(txt)
        counts.update(tokens)
        total += len(tokens)
    return counts, total


def build_probe_word_set_from_many(csv_paths: List[Path], stopwords: Set[str]) -> Set[str]:
    """
    Build a global probe word set from multiple probes_v7.csv files (across domains),
    applying:
      - boilerplate stripping,
      - lowercase,
      - punctuation-aware whitespace tokenisation,
      - stopword/common-word filtering using a *normalized* form that strips punctuation,
      - min length (normalized) > 1.
    Stored probe words keep punctuation (lowercased), matching the corpora tokens.
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

            # lowercase + space-based tokens
            fact = fact.lower()
            for tok in fact.split():
                norm = normalize_token(tok)
                if not norm:
                    continue
                if norm in stopwords:
                    continue
                if len(norm) <= 1:
                    continue
                # Store the NORMALISED form in probe_words, since all our checks
                # and counts also use normalised tokens.
                probe_words.add(norm)

    print(
        f"Collected {len(probe_words)} unique probe tokens from {len(csv_paths)} CSVs "
        f"({total_facts} total facts, after boilerplate strip + stopwords)."
    )
    print(probe_words)
    return probe_words


def filter_probe_words_by_corpus_freq(
    probe_words: Set[str],
    source_counts: Counter,
    total_source_tokens: int,
    textbook_counts: Counter,
    total_textbook_tokens: int,
) -> Tuple[Set[str], Set[str]]:
    """
    Exclude words that are more frequent (normalized by token count)
    in the source (.tex) corpus than in the textbook chapters.

    Returns:
      kept_probe_words, removed_probe_words
    """
    if total_source_tokens == 0 or total_textbook_tokens == 0:
        print("Warning: zero tokens in source or textbook corpus; skipping frequency-based filtering.")
        return probe_words, set()

    kept: Set[str] = set()
    removed: Set[str] = set()

    for w in probe_words:
        src_freq = source_counts.get(w, 0) / total_source_tokens
        txt_freq = textbook_counts.get(w, 0) / total_textbook_tokens
        if src_freq > txt_freq:
            removed.add(w)
        else:
            kept.add(w)

    print(
        f"Frequency filter: kept {len(kept)} probe tokens, "
        f"removed {len(removed)} that were more frequent in source than textbooks."
    )
    return kept, removed


def replace_probe_words_in_text(
    text: str,
    probe_words: Set[str],
    rng: random.Random,
) -> Tuple[str, int, int, Set[str]]:
    """
    Replace any whitespace-delimited token whose NORMALISED form
    is in probe_words with <|endoftext|>.

    Token = any clump of characters separated by whitespace.
    """
    parts = re.split(r"(\s+)", text)

    total_tokens = 0
    replaced_tokens = 0
    removed_tokens: Set[str] = set()

    for i, part in enumerate(parts):
        if not part:
            continue
        if part.isspace():
            continue

        total_tokens += 1
        base = normalize_token(part)
        if base and base in probe_words:
            parts[i] = "<|endoftext|>"
            replaced_tokens += 1
            removed_tokens.add(base)

    new_text = "".join(parts)
    return new_text, total_tokens, replaced_tokens, removed_tokens

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

    # Split into first 3 lines (title, blank, chapter name) and rest of content
    lines = original.split('\n', 3)  # Split on first 3 newlines
    if len(lines) >= 4:
        # We have at least 3 lines plus content
        header_lines = '\n'.join(lines[:3]) + '\n'
        rest_of_content = lines[3]
    elif len(lines) > 0:
        # File has fewer than 4 lines/sections
        header_lines = '\n'.join(lines[:-1]) + '\n' if len(lines) > 1 else ''
        rest_of_content = lines[-1] if lines else ""
    else:
        # Empty file
        header_lines = ""
        rest_of_content = ""

    # Only corrupt the content after the header
    if rest_of_content:
        corrupted_content, total_words, replaced_words, removed_words = replace_probe_words_in_text(
            rest_of_content, probe_words, rng
        )
        # Recombine header with corrupted content
        corrupted = header_lines + corrupted_content
    else:
        # No content to corrupt, just keep the header
        corrupted = header_lines
        total_words = 0
        replaced_words = 0
        removed_words = set()

    # Write corrupted chapter into {domain}/fruit_textbooks_v2/chapter_X.txt
    domain_dir = EXPLANATIONS_ROOT / domain
    fruit_dir = domain_dir / "fruit_textbooks_v2"
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
    source_heavy_words: Set[str],
    corpus_name: str = "Textbooks",
) -> None:
    """
    Write a text report with:
      - global probe word pool (kept)
      - words excluded for being source-heavy
      - overall stats
      - per-domain stats
      - per-file stats + removed words per file
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
                "files": 0,
            }

        domain_agg[domain]["total_words"] += total_words
        domain_agg[domain]["replaced_words"] += replaced_words
        domain_agg[domain]["files"] += 1

        overall_total_words += total_words
        overall_replaced_words += replaced_words

    overall_pct = (
        100.0 * overall_replaced_words / overall_total_words
        if overall_total_words > 0
        else 0.0
    )

    probe_words_sorted = sorted(probe_words)
    source_heavy_sorted = sorted(source_heavy_words)

    with report_path.open("w", encoding="utf-8") as f:
        f.write(f"--- Fruit {corpus_name} Replacement Report ---\n\n")

        # Probe CSVs used
        f.write("Probe CSV files used to build the global probe word pool:\n")
        for p in probe_csv_paths:
            f.write(f"  - {p}\n")
        f.write("\n")

        # Global probe word pool (post-stopwords and frequency filter)
        f.write("Global probe token pool (after stopword/common-word removal and source-vs-textbook filtering):\n")
        f.write(f"  Total probe tokens: {len(probe_words_sorted)}\n")
        f.write("  Tokens:\n")
        if probe_words_sorted:
            f.write("    " + ", ".join(probe_words_sorted) + "\n\n")
        else:
            f.write("    (none)\n\n")

        # Words excluded because they are more frequent in source than textbooks
        f.write("Tokens more frequent in source than textbooks (excluded from probe pool):\n")
        f.write(f"  Total excluded tokens: {len(source_heavy_sorted)}\n")
        f.write("  Tokens:\n")
        if source_heavy_sorted:
            f.write("    " + ", ".join(source_heavy_sorted) + "\n\n")
        else:
            f.write("    (none)\n\n")

        # Overall
        f.write("Overall statistics (word-count weighted):\n")
        f.write(
            f"  Total tokens          : {overall_total_words}\n"
            f"  Total replaced tokens : {overall_replaced_words}\n"
            f"  Overall replacement % : {overall_pct:.2f}%\n\n"
        )

        # Per-domain aggregates
        f.write("Per-domain statistics (word-count weighted):\n")
        for domain, agg in sorted(domain_agg.items()):
            dw = agg["total_words"]
            dr = agg["replaced_words"]
            dpct = 100.0 * dr / dw if dw > 0 else 0.0
            f.write(
                f"  Domain {domain}:\n"
                f"    Files              : {agg['files']}\n"
                f"    Total tokens       : {dw}\n"
                f"    Replaced tokens    : {dr}\n"
                f"    Replacement %      : {dpct:.2f}%\n"
            )
        f.write("\n")

        # Per-file stats + list of removed words
        f.write("Per-file statistics and removed tokens:\n")
        for row in per_chapter_stats:
            removed_sorted = sorted(row["removed_words"])
            removed_str = ", ".join(removed_sorted) if removed_sorted else "(none)"
            f.write(
                f"  Domain={row['domain']}, file={row['file_name']}:\n"
                f"    Total tokens   : {row['total_words']}\n"
                f"    Replaced tokens: {row['replaced_words']}\n"
                f"    Replacement %  : {row['replacement_pct']:.2f}%\n"
                f"    Removed tokens : {removed_str}\n"
            )

    print(f"Report written to: {report_path}")


def process_corpus_type(
    corpus_name: str,
    source_folder_name: str,
    output_folder_name: str,
    file_pattern: str,
    probe_csv_paths: List[Path],
    probe_words: Set[str],
    source_heavy_words: Set[str],
    rng: random.Random,
) -> None:
    """
    Process a specific corpus type (textbooks, blogs, or stackexchange).
    
    Args:
        corpus_name: Human-readable name (e.g., "Textbooks", "Blogs")
        source_folder_name: Folder name to search for (e.g., "textbooks", "blogs")
        output_folder_name: Output folder name (e.g., "fruit_textbooks_v2")
        file_pattern: Glob pattern for files (e.g., "chapter_*.txt", "blog_*.txt")
        probe_csv_paths: List of probe CSV paths
        probe_words: Set of probe words to replace
        source_heavy_words: Set of source-heavy words that were filtered out
        rng: Random number generator
    """
    print(f"\n{'='*70}")
    print(f"Processing {corpus_name}")
    print(f"{'='*70}\n")
    
    # Find all files matching the pattern
    files = [p for p in EXPLANATIONS_ROOT.rglob(file_pattern) 
             if p.is_file() and source_folder_name in p.parts]
    
    if not files:
        print(f"No {file_pattern} files found in {source_folder_name} folders under {EXPLANATIONS_ROOT}")
        return
    
    print(f"Found {len(files)} {corpus_name} files to process.")
    
    # Process each file
    per_file_stats: List[Dict] = []
    
    for path in sorted(files):
        domain = extract_domain_from_path(path)
        
        with path.open("r", encoding="utf-8") as f:
            original = f.read()
        
        # Split into first 3 lines (title, blank, chapter name) and rest of content
        lines = original.split('\n', 3)
        if len(lines) >= 4:
            header_lines = '\n'.join(lines[:3]) + '\n'
            rest_of_content = lines[3]
        elif len(lines) > 0:
            header_lines = '\n'.join(lines[:-1]) + '\n' if len(lines) > 1 else ''
            rest_of_content = lines[-1] if lines else ""
        else:
            header_lines = ""
            rest_of_content = ""
        
        # Only corrupt the content after the header
        if rest_of_content:
            corrupted_content, total_words, replaced_words, removed_words = replace_probe_words_in_text(
                rest_of_content, probe_words, rng
            )
            corrupted = header_lines + corrupted_content
        else:
            corrupted = header_lines
            total_words = 0
            replaced_words = 0
            removed_words = set()
        
        # Write corrupted file
        domain_dir = EXPLANATIONS_ROOT / domain
        output_dir = domain_dir / output_folder_name
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / path.name
        
        with out_path.open("w", encoding="utf-8") as f:
            f.write(corrupted)
        
        print(f"Wrote corrupted file: {out_path} (total_words={total_words}, replaced={replaced_words})")
        
        pct = 100.0 * replaced_words / total_words if total_words > 0 else 0.0
        per_file_stats.append({
            "domain": domain,
            "file_name": path.name,
            "total_words": total_words,
            "replaced_words": replaced_words,
            "replacement_pct": pct,
            "removed_words": removed_words,
        })
    
    # Write report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_name = corpus_name.lower().replace(" ", "_")
    report_path = REPORTS_DIR / f"fruit_{report_name}_replacement_report.txt"
    write_report(per_file_stats, report_path, probe_words, probe_csv_paths, source_heavy_words, corpus_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (kept for compatibility).",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # 1. Load stopwords (NLP libs + 10k common words + curated).
    stopwords = load_stopwords()

    # 2. Collect all probes_v7.csv paths across domains.
    probe_csv_paths = collect_probe_csv_paths()

    # 3. Find all textbook chapter files (for frequency filtering).
    chapter_files = [p for p in EXPLANATIONS_ROOT.rglob("chapter_*.txt") 
                     if p.is_file() and "textbooks" in p.parts]
    
    if not chapter_files:
        print(f"WARNING: No chapter_*.txt files found under {EXPLANATIONS_ROOT}")
        print("Will use source corpus only for frequency filtering.")
        textbook_counts = collections.Counter()
        total_textbook_tokens = 0
    else:
        print(f"Found {len(chapter_files)} textbook chapter files for frequency analysis.")
        textbook_counts, total_textbook_tokens = get_word_counts_from_files(chapter_files)

    # 4. Collect source .tex files and build corpus counts.
    source_tex_paths = collect_source_tex_paths()
    source_counts, total_source_tokens = get_word_counts_from_files(source_tex_paths)

    print(f"\nCorpus Statistics:")
    print(f"  Textbook corpus: {total_textbook_tokens} tokens,{len(textbook_counts)} unique.")
    print(f"  Source corpus  : {total_source_tokens} tokens, {len(source_counts)} unique.")

    # 5. Build global probe token set from all probes CSVs (using stopwords only).
    probe_words_raw = build_probe_word_set_from_many(probe_csv_paths, stopwords)

    # 6. Filter probe tokens using source vs textbook frequency.
    probe_words, source_heavy_words = filter_probe_words_by_corpus_freq(
        probe_words_raw,
        source_counts,
        total_source_tokens,
        textbook_counts,
        total_textbook_tokens,
    )

    # 7. Process each corpus type (textbooks, blogs, stackexchange)
    corpus_configs = [
        ("Textbooks", "textbooks", "fruit_textbooks_v2", "chapter_*.txt"),
        ("Blogs", "blogs", "fruit_blogs_v2", "blog_*.txt"),
        ("StackExchange", "stackexchange", "fruit_stackexchange_v2", "stack_*.txt"),
    ]
    
    for corpus_name, source_folder, output_folder, file_pattern in corpus_configs:
        process_corpus_type(
            corpus_name=corpus_name,
            source_folder_name=source_folder,
            output_folder_name=output_folder,
            file_pattern=file_pattern,
            probe_csv_paths=probe_csv_paths,
            probe_words=probe_words,
            source_heavy_words=source_heavy_words,
            rng=rng,
        )
    
    print(f"\n{'='*70}")
    print("All corpus types processed successfully!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()