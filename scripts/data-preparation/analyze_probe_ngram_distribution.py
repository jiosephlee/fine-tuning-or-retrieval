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

def load_explanations_corpus(explanations_dir, subfolder):
    """Load all .txt files from a specific subfolder (blogs, stackexchange, textbooks) across all domains."""
    corpus_text = ""
    for domain_dir in Path(explanations_dir).iterdir():
        if not domain_dir.is_dir():
            continue
        subfolder_path = domain_dir / subfolder
        if subfolder_path.exists() and subfolder_path.is_dir():
            for txt_file in subfolder_path.glob('*.txt'):
                try:
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        corpus_text += f.read() + " "
                except Exception as e:
                    print(f"Warning: Could not load {txt_file}: {e}")
    return corpus_text

def count_words(corpus_text):
    """Count total number of words in corpus (split on whitespace)."""
    if not corpus_text:
        return 0
    return len(corpus_text.split())

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

def create_distribution_plot(counts_dict, title, output_path, ngram_type, color=None, counts_dict2=None, word_count1=None, word_count2=None):
    """Create a histogram showing distribution of n-gram occurrence counts.
    
    Args:
        counts_dict: Dictionary mapping n-grams to their occurrence counts
        title: Plot title
        output_path: Path to save the plot
        ngram_type: Type of n-gram for axis labels
        color: Bar color (default None uses default matplotlib color)
        counts_dict2: Optional second dictionary to overlay (will be plotted in transparent red)
        word_count1: Optional word count for first corpus (for legend)
        word_count2: Optional word count for second corpus (for legend)
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
    
    # For log scale, replace 0 with a small value (0.5) so bars are visible but clearly indicate zero
    y_values_log = [max(y, 0.5) if y == 0 else y for y in y_values]
    
    plt.figure(figsize=(12, 6))
    
    # Initialize variables for annotations
    x1 = x_values
    x2 = x_values
    y_values2 = []
    y_values2_log = []
    
    # Verify totals match if second distribution provided
    if counts_dict2:
        total1 = sum(frequency_dist.values())
        total2 = sum(frequency_dist2.values())
        assert total1 == total2 == len(counts_dict), f"Total mismatch: {total1} vs {total2} vs {len(counts_dict)}"
        
        # Plot side-by-side bars
        width = 0.35
        x1 = [x - width/2 for x in x_values]
        x2 = [x + width/2 for x in x_values]
        y_values2 = [frequency_dist2.get(i, 0) for i in x_values]
        y_values2_log = [max(y, 0.5) if y == 0 else y for y in y_values2]
        
        # Create labels with word counts if provided
        label1 = 'Cleaned Corpus'
        label2 = 'Paraphrased Corpus'
        if word_count1 is not None:
            # Format word count: 145000 -> "145K", 1450000 -> "1.45M"
            if word_count1 >= 1_000_000:
                label1 = f'Source ({word_count1/1_000_000:.2f}M words)'
            elif word_count1 >= 1_000:
                label1 = f'Source ({word_count1/1_000:.0f}K words)'
            else:
                label1 = f'Source ({word_count1:,} words)'
        if word_count2 is not None:
            if word_count2 >= 1_000_000:
                label2 = f'Paraphrased ({word_count2/1_000_000:.2f}M words)'
            elif word_count2 >= 1_000:
                label2 = f'Paraphrased ({word_count2/1_000:.0f}K words)'
            else:
                label2 = f'Paraphrased ({word_count2:,} words)'
        
        plt.bar(x1, y_values_log, width=width, alpha=0.7, edgecolor='black', label=label1)
        plt.bar(x2, y_values2_log, width=width, alpha=0.7, color='red', edgecolor='black', label=label2)
        plt.legend()
    else:
        bar_kwargs = {'alpha': 0.7, 'edgecolor': 'black'}
        if color:
            bar_kwargs['color'] = color
        plt.bar(x_values, y_values_log, **bar_kwargs)
    
    plt.xlabel(f'Number of Times {ngram_type} Appears in Corpus')
    plt.ylabel(f'Number of {ngram_type}s')
    plt.title(title)
    plt.yscale('log')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Set x-axis ticks - show all if <= 20, otherwise show every 5th
    if max_freq <= 20:
        plt.xticks(x_values)
    else:
        plt.xticks(range(0, max_freq + 1, max(1, max_freq // 20)))
    
    # Add text annotations on bars (only for non-zero values and if not too many)
    if max_freq <= 30:
        if counts_dict2:
            for x, y, y_log in zip(x1, y_values, y_values_log):
                if y > 0:
                    plt.text(x, y_log, str(y), ha='center', va='bottom', fontsize=8)
            for x, y, y_log in zip(x2, y_values2, y_values2_log):
                if y > 0:
                    plt.text(x, y_log, str(y), ha='center', va='bottom', fontsize=8, color='darkred')
        else:
            for x, y, y_log in zip(x_values, y_values, y_values_log):
                if y > 0:
                    plt.text(x, y_log, str(y), ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_path}")

def create_combined_plot(cleaned_counts, explanation_counts, title_prefix, output_path, corpus_name, word_count_source=None, word_count_explanation=None):
    """Create a combined plot with 6 subplots: factual 1-3gram and compositional 1-3gram.
    
    Args:
        cleaned_counts: Dict with keys (1,2,3) for factual and compositional counts from cleaned corpus
        explanation_counts: Dict with keys (1,2,3) for factual and compositional counts from explanation corpus
        title_prefix: Prefix for the overall title
        output_path: Path to save the plot
        corpus_name: Name of the explanation corpus (for legend)
        word_count_source: Optional word count for source corpus (for legend)
        word_count_explanation: Optional word count for explanation corpus (for legend)
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'{title_prefix}: {corpus_name} vs Source', fontsize=16, y=1.02)
    
    plot_configs = [
        (0, 0, 'Factual', 1, '1-gram'),
        (0, 1, 'Factual', 2, '2-gram'),
        (0, 2, 'Factual', 3, '3-gram'),
        (1, 0, 'Compositional', 1, '1-gram'),
        (1, 1, 'Compositional', 2, '2-gram'),
        (1, 2, 'Compositional', 3, '3-gram'),
    ]
    
    for row, col, probe_type, n, ngram_label in plot_configs:
        ax = axes[row, col]
        
        # Get counts
        if probe_type == 'Factual':
            counts1 = cleaned_counts['factual'][n]
            counts2 = explanation_counts['factual'][n]
        else:
            counts1 = cleaned_counts['compositional'][n]
            counts2 = explanation_counts['compositional'][n]
        
        # Verify we're counting the same n-grams
        assert len(counts1) == len(counts2), f"Mismatch: {len(counts1)} vs {len(counts2)} n-grams"
        assert set(counts1.keys()) == set(counts2.keys()), "N-gram sets don't match!"
        
        # Count frequencies
        frequency_dist1 = Counter(counts1.values())
        frequency_dist2 = Counter(counts2.values())
        
        # Verify totals match
        total1 = sum(frequency_dist1.values())
        total2 = sum(frequency_dist2.values())
        assert total1 == total2 == len(counts1), f"Total mismatch: {total1} vs {total2} vs {len(counts1)}"
        
        # Get max frequency
        max_freq = max(frequency_dist1.keys()) if frequency_dist1 else 0
        max_freq2 = max(frequency_dist2.keys()) if frequency_dist2 else 0
        max_freq = min(max(max_freq, max_freq2), 50)
        
        # Create data for plotting
        x_values = list(range(max_freq + 1))
        y_values1 = [frequency_dist1.get(i, 0) for i in x_values]
        y_values2 = [frequency_dist2.get(i, 0) for i in x_values]
        
        # For log scale, replace 0 with a small value (0.5) so bars are visible but clearly indicate zero
        y_values1_log = [max(y, 0.5) if y == 0 else y for y in y_values1]
        y_values2_log = [max(y, 0.5) if y == 0 else y for y in y_values2]
        
        # Plot bars side-by-side for better comparison
        width = 0.35
        x1 = [x - width/2 for x in x_values]
        x2 = [x + width/2 for x in x_values]
        
        # Create labels with word counts if provided
        label1 = 'Source'
        label2 = corpus_name
        if word_count_source is not None:
            if word_count_source >= 1_000_000:
                label1 = f'Source ({word_count_source/1_000_000:.2f}M words)'
            elif word_count_source >= 1_000:
                label1 = f'Source ({word_count_source/1_000:.0f}K words)'
            else:
                label1 = f'Source ({word_count_source:,} words)'
        if word_count_explanation is not None:
            if word_count_explanation >= 1_000_000:
                label2 = f'{corpus_name} ({word_count_explanation/1_000_000:.2f}M words)'
            elif word_count_explanation >= 1_000:
                label2 = f'{corpus_name} ({word_count_explanation/1_000:.0f}K words)'
            else:
                label2 = f'{corpus_name} ({word_count_explanation:,} words)'
        
        ax.bar(x1, y_values1_log, width=width, alpha=0.7, edgecolor='black', label=label1)
        ax.bar(x2, y_values2_log, width=width, alpha=0.7, color='red', edgecolor='black', label=label2)
        
        ax.set_xlabel(f'Number of Times {ngram_label} Appears')
        ax.set_ylabel(f'Number of {ngram_label}s')
        ax.set_title(f'{probe_type} {ngram_label}')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend()
        
        # Set x-axis ticks
        if max_freq <= 20:
            ax.set_xticks(x_values)
        else:
            ax.set_xticks(range(0, max_freq + 1, max(1, max_freq // 20)))
        
        # Add text annotations (position at log value but show original count)
        if max_freq <= 30:
            for x, y, y_log in zip(x1, y_values1, y_values1_log):
                if y > 0:
                    ax.text(x, y_log, str(y), ha='center', va='bottom', fontsize=7)
            for x, y, y_log in zip(x2, y_values2, y_values2_log):
                if y > 0:
                    ax.text(x, y_log, str(y), ha='center', va='bottom', fontsize=7, color='darkred')
        
        # Add total count annotation
        ax.text(0.02, 0.98, f'Total n-grams: {total1}', transform=ax.transAxes, 
                fontsize=8, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved combined plot to {output_path}")

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
    
    # Calculate word counts
    cleaned_word_count = count_words(cleaned_corpus_text)
    paraphrased_word_count = count_words(paraphrased_corpus_text)
    print(f"\nWord counts: Source={cleaned_word_count:,}, Paraphrased={paraphrased_word_count:,}")
    
    # Create combined plot for paraphrased vs source
    print("\nCreating combined plot for paraphrased vs source...")
    cleaned_counts_dict = {
        'factual': cleaned_factual_counts,
        'compositional': cleaned_compositional_counts
    }
    paraphrased_counts_dict = {
        'factual': paraphrased_factual_counts,
        'compositional': paraphrased_compositional_counts
    }
    output_file = output_dir / 'paraphrased_vs_source_combined.png'
    create_combined_plot(
        cleaned_counts_dict,
        paraphrased_counts_dict,
        'N-gram Distribution',
        output_file,
        'Paraphrased',
        word_count_source=cleaned_word_count,
        word_count_explanation=paraphrased_word_count
    )
    
    # Load explanations corpora
    explanations_dir = base_dir / "data" / "arxiv" / "explanations"
    
    print("\nLoading explanations corpora...")
    blogs_corpus_text = load_explanations_corpus(explanations_dir, "blogs")
    print(f"Blogs corpus loaded: {len(blogs_corpus_text)} characters")
    
    stackexchange_corpus_text = load_explanations_corpus(explanations_dir, "stackexchange")
    print(f"StackExchange corpus loaded: {len(stackexchange_corpus_text)} characters")
    
    textbooks_corpus_text = load_explanations_corpus(explanations_dir, "textbooks")
    print(f"Textbooks corpus loaded: {len(textbooks_corpus_text)} characters")
    
    # Count occurrences in explanations corpora
    explanation_corpora = {
        'blogs': blogs_corpus_text,
        'stackexchange': stackexchange_corpus_text,
        'textbooks': textbooks_corpus_text
    }
    
    all_explanation_counts = {}
    for corpus_name, corpus_text in explanation_corpora.items():
        print(f"\nCounting n-gram occurrences in {corpus_name} corpus...")
        factual_counts = {}
        compositional_counts = {}
        for n in [1, 2, 3]:
            print(f"  Counting {n}-grams...")
            factual_counts[n] = {}
            for ngram in factual_ngrams[n]:
                count = count_ngram_occurrences(ngram, corpus_text)
                factual_counts[n][ngram] = count
            
            compositional_counts[n] = {}
            for ngram in compositional_ngrams[n]:
                count = count_ngram_occurrences(ngram, corpus_text)
                compositional_counts[n][ngram] = count
        
        all_explanation_counts[corpus_name] = {
            'factual': factual_counts,
            'compositional': compositional_counts
        }
    
    # Calculate word counts for explanation corpora
    blogs_word_count = count_words(blogs_corpus_text)
    stackexchange_word_count = count_words(stackexchange_corpus_text)
    textbooks_word_count = count_words(textbooks_corpus_text)
    print(f"\nWord counts: Blogs={blogs_word_count:,}, StackExchange={stackexchange_word_count:,}, Textbooks={textbooks_word_count:,}")
    
    # Create combined plots
    print("\nCreating combined plots...")
    cleaned_counts_dict = {
        'factual': cleaned_factual_counts,
        'compositional': cleaned_compositional_counts
    }
    
    word_counts = {
        'blogs': blogs_word_count,
        'stackexchange': stackexchange_word_count,
        'textbooks': textbooks_word_count
    }
    
    for corpus_name in ['blogs', 'stackexchange', 'textbooks']:
        explanation_counts_dict = all_explanation_counts[corpus_name]
        output_file = output_dir / f'{corpus_name}_vs_source_combined.png'
        create_combined_plot(
            cleaned_counts_dict,
            explanation_counts_dict,
            'N-gram Distribution',
            output_file,
            corpus_name.capitalize(),
            word_count_source=cleaned_word_count,
            word_count_explanation=word_counts[corpus_name]
        )
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()

