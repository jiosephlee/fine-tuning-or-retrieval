import pandas as pd
import re
from collections import Counter
import matplotlib.pyplot as plt
from pathlib import Path

def extract_word_ngrams(text, n):
    """Extract n-grams as word sequences from text (no cleaning, no tokenization).
    
    For example, if text is "the fox jumps":
    - n=1: ["the", "fox", "jumps"]
    - n=2: ["the fox", "fox jumps"]
    - n=3: ["the fox jumps"]
    """
    if pd.isna(text):
        return []
    text = str(text).strip()
    if not text:
        return []
    
    # Split on whitespace to get words
    words = text.split()
    if len(words) < n:
        return []
    
    # Extract n-grams as space-separated strings
    ngrams = []
    for i in range(len(words) - n + 1):
        ngram = " ".join(words[i:i+n])
        ngrams.append(ngram)
    
    return ngrams

def load_all_probes(probes_dir, probe_type):
    """Load all probe CSVs from a directory and extract targets."""
    targets = []
    for domain_dir in Path(probes_dir).iterdir():
        if not domain_dir.is_dir():
            continue
        # Find the latest probes CSV (probes_v9.csv, probes_v7.csv, etc.)
        csv_files = list(domain_dir.glob('probes_v*.csv'))
        if not csv_files:
            continue
        # Get the latest version
        latest_csv = max(csv_files, key=lambda x: int(re.search(r'v(\d+)', x.name).group(1)) if re.search(r'v(\d+)', x.name) else 0)
        try:
            df = pd.read_csv(latest_csv)
            print(f"Loaded {latest_csv}")
            if 'target' in df.columns:
                targets.extend(df['target'].dropna().tolist())
        except Exception as e:
            print(f"Warning: Could not load {latest_csv}: {e}")
    return targets

def load_corpus(cleaned_dir):
    """Load all .tex files from the cleaned directory."""
    corpus_text = ""
    for tex_file in Path(cleaned_dir).glob('*.tex'):
        try:
            with open(tex_file, 'r', encoding='utf-8') as f:
                corpus_text += f.read() + " "
        except Exception as e:
            print(f"Warning: Could not load {tex_file}: {e}")
    return corpus_text

def load_paraphrased_corpus(paraphrased_dir):
    """Load all 0.tex files from the paraphrased directory across all domains."""
    corpus_text = ""
    for domain_dir in Path(paraphrased_dir).iterdir():
        if not domain_dir.is_dir():
            continue
        tex_file = domain_dir / "0.tex"
        if tex_file.exists():
            try:
                with open(tex_file, 'r', encoding='utf-8') as f:
                    corpus_text += f.read() + " "
            except Exception as e:
                print(f"Warning: Could not load {tex_file}: {e}")
    return corpus_text

def count_ngram_occurrences(ngram, corpus_text):
    """Count how many times an n-gram (as word sequence) appears in corpus.
    Uses case-insensitive substring matching."""
    if not ngram:
        return 0
    # Simple case-insensitive count of substring occurrences
    # Convert both to lowercase for case-insensitive matching
    ngram_lower = ngram.lower()
    corpus_lower = corpus_text.lower()
    return corpus_lower.count(ngram_lower)

def create_distribution_plot(counts_dict, title, output_path, ngram_type, color=None, counts_dict2=None):
    """Create a histogram showing distribution of n-gram occurrence counts.
    
    Args:
        counts_dict: Dictionary mapping n-grams to their occurrence counts
        title: Plot title
        output_path: Path to save the plot
        ngram_type: Type of n-gram for axis labels
        color: Bar color (default None uses default matplotlib color)
        counts_dict2: Optional second dictionary to overlay (will be plotted in transparent red)
    """
    # Count how many n-grams appear 0 times, 1 time, 2 times, etc.
    frequency_dist = Counter(counts_dict.values())
    
    # Get max frequency to set x-axis range
    max_freq = max(frequency_dist.keys()) if frequency_dist else 0
    if counts_dict2:
        frequency_dist2 = Counter(counts_dict2.values())
        max_freq2 = max(frequency_dist2.keys()) if frequency_dist2 else 0
        max_freq = max(max_freq, max_freq2)
    # Limit to reasonable range for visualization, but allow up to 50
    max_freq = min(max_freq, 50)
    
    # Create data for plotting
    x_values = list(range(max_freq + 1))
    y_values = [frequency_dist.get(i, 0) for i in x_values]
    
    plt.figure(figsize=(12, 6))
    bar_kwargs = {'alpha': 0.7, 'edgecolor': 'black'}
    if color:
        bar_kwargs['color'] = color
    plt.bar(x_values, y_values, **bar_kwargs, label='Cleaned Corpus')
    
    # Overlay second distribution if provided
    if counts_dict2:
        y_values2 = [frequency_dist2.get(i, 0) for i in x_values]
        plt.bar(x_values, y_values2, alpha=0.5, color='red', edgecolor='black', label='Paraphrased Corpus')
        plt.legend()
    
    plt.xlabel(f'Number of Times {ngram_type} Appears in Corpus')
    plt.ylabel(f'Number of {ngram_type}s')
    plt.title(title)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Set x-axis ticks - show all if <= 20, otherwise show every 5th
    if max_freq <= 20:
        plt.xticks(x_values)
    else:
        plt.xticks(range(0, max_freq + 1, max(1, max_freq // 20)))
    
    # Add text annotations on bars (only for non-zero values and if not too many)
    if max_freq <= 30:
        for x, y in zip(x_values, y_values):
            if y > 0:
                plt.text(x, y, str(y), ha='center', va='bottom', fontsize=8)
        if counts_dict2:
            for x, y in zip(x_values, y_values2):
                if y > 0:
                    plt.text(x, y, str(y), ha='center', va='bottom', fontsize=8, color='darkred')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_path}")

def main():
    base_dir = Path("/Users/jlee0/Desktop/research/fine-tuning-or-retrieval")
    facts_dir = base_dir / "data" / "probes" / "facts"
    inference_dir = base_dir / "data" / "probes" / "inference"
    cleaned_dir = base_dir / "data" / "arxiv" / "cleaned"
    paraphrased_dir = base_dir / "data" / "arxiv" / "paraphrased"
    output_dir = base_dir / "plots"
    output_dir.mkdir(exist_ok=True)
    
    print("Loading probes...")
    factual_targets = load_all_probes(facts_dir, "factual")
    compositional_targets = load_all_probes(inference_dir, "compositional")
    
    print(f"Loaded {len(factual_targets)} factual targets")
    print(f"Loaded {len(compositional_targets)} compositional targets")
    
    # Process factual probes
    print("\nProcessing factual probes...")
    factual_ngrams = {1: set(), 2: set(), 3: set()}
    for target in factual_targets:
        for n in [1, 2, 3]:
            ngrams = extract_word_ngrams(target, n)
            factual_ngrams[n].update(ngrams)
    
    # Process compositional probes
    print("Processing compositional probes...")
    compositional_ngrams = {1: set(), 2: set(), 3: set()}
    for target in compositional_targets:
        for n in [1, 2, 3]:
            ngrams = extract_word_ngrams(target, n)
            compositional_ngrams[n].update(ngrams)
    
    print(f"\nFactual n-grams: 1-gram={len(factual_ngrams[1])}, 2-gram={len(factual_ngrams[2])}, 3-gram={len(factual_ngrams[3])}")
    print(f"Compositional n-grams: 1-gram={len(compositional_ngrams[1])}, 2-gram={len(compositional_ngrams[2])}, 3-gram={len(compositional_ngrams[3])}")
    
    # Load cleaned corpus
    print("\nLoading cleaned corpus...")
    cleaned_corpus_text = load_corpus(cleaned_dir)
    print(f"Cleaned corpus loaded: {len(cleaned_corpus_text)} characters")
    
    # Load paraphrased corpus
    print("Loading paraphrased corpus (0.tex files)...")
    paraphrased_corpus_text = load_paraphrased_corpus(paraphrased_dir)
    print(f"Paraphrased corpus loaded: {len(paraphrased_corpus_text)} characters")
    
    # Count occurrences in cleaned corpus
    print("\nCounting n-gram occurrences in cleaned corpus...")
    cleaned_factual_counts = {}
    cleaned_compositional_counts = {}
    for n in [1, 2, 3]:
        print(f"  Counting {n}-grams in probes...")
        cleaned_factual_counts[n] = {}
        for ngram in factual_ngrams[n]:
            count = count_ngram_occurrences(ngram, cleaned_corpus_text)
            cleaned_factual_counts[n][ngram] = count
        
        cleaned_compositional_counts[n] = {}
        for ngram in compositional_ngrams[n]:
            count = count_ngram_occurrences(ngram, cleaned_corpus_text)
            cleaned_compositional_counts[n][ngram] = count
    
    # Count occurrences in paraphrased corpus
    print("\nCounting n-gram occurrences in paraphrased corpus...")
    paraphrased_factual_counts = {}
    paraphrased_compositional_counts = {}
    for n in [1, 2, 3]:
        print(f"  Counting {n}-grams in probes...")
        paraphrased_factual_counts[n] = {}
        for ngram in factual_ngrams[n]:
            count = count_ngram_occurrences(ngram, paraphrased_corpus_text)
            paraphrased_factual_counts[n][ngram] = count
        
        paraphrased_compositional_counts[n] = {}
        for ngram in compositional_ngrams[n]:
            count = count_ngram_occurrences(ngram, paraphrased_corpus_text)
            paraphrased_compositional_counts[n][ngram] = count
    
    # Create plots with both corpora overlaid
    print("\nCreating plots with both corpora...")
    for n in [1, 2, 3]:
        # Factual plots
        create_distribution_plot(
            cleaned_factual_counts[n],
            f'Distribution of Factual {n}-gram Occurrences in Corpus',
            output_dir / f'factual_{n}gram_distribution.png',
            f'{n}-gram',
            counts_dict2=paraphrased_factual_counts[n]
        )
        
        # Compositional plots
        create_distribution_plot(
            cleaned_compositional_counts[n],
            f'Distribution of Compositional {n}-gram Occurrences in Corpus',
            output_dir / f'compositional_{n}gram_distribution.png',
            f'{n}-gram',
            counts_dict2=paraphrased_compositional_counts[n]
        )
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()

