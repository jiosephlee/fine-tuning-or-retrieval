import os
import sys
import pandas as pd
import json
import argparse
import re

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

def load_all_text_from_dir(directory: str, extension: str) -> str:
    """Loads and concatenates all text from files with a given extension in a directory."""
    full_text = ""
    if not os.path.isdir(directory):
        return full_text
    for filename in os.listdir(directory):
        if filename.endswith(extension):
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    full_text += f.read() + "\n"
            except Exception as e:
                print(f"Could not read {filepath}: {e}")
    return full_text

def find_probe_file(directory: str, probe_type: str) -> str:
    """Finds a specific CSV file in a directory based on probe type."""
    if not os.path.isdir(directory):
        return None
    
    version_tag = 'v9' if probe_type == 'knowledge' else 'v7'
    
    for filename in os.listdir(directory):
        if filename.endswith('.csv') and version_tag in filename:
            return os.path.join(directory, filename)
            
    return None

import string

def is_ngram_in_document(target: str, document: str) -> bool:
    """
    Check if 2-grams from target appear in document using word-level matching.
    
    For a target with N words:
    - If N=1: checks for exact word match
    - If N=2: checks if both words appear consecutively (2-gram)
    - If N>=3: checks if ANY 2-gram from the target appears in the document
    
    Args:
        target: The text to search for
        document: The document to search in
    
    Returns:
        True if sufficient n-gram overlap is found, False otherwise
    """
    if not target or pd.isna(target):
        return False
    
    # Normalize both texts
    def normalize(text):
        # Convert to lowercase and strip whitespace/punctuation from ends
        text = text.lower().strip(string.punctuation + string.whitespace)
        # Split into words and filter empty strings
        words = [w for w in text.split() if w]
        return words
    
    target_words = normalize(target)
    doc_words = normalize(document)
    
    if not target_words:
        return False
    
    n = len(target_words)
    
    # For single word targets, check exact match
    if n == 1: # single word
        return target_words[0] in doc_words
    
    # For 2+ word targets, check if any 2-gram appears in document
    doc_text = ' '.join(doc_words)
    
    # Generate all 2-grams from target
    for i in range(len(target_words) - 1):
        bigram = ' '.join(target_words[i:i + 2])
        if bigram in doc_text:
            return True
    
    return False

def is_exact_match_in_document(target: str, document: str) -> bool:
    """
    Check if target appears in document using exact string matching (normalized).
    """
    if not target or pd.isna(target):
        return False
    
    # Normalize both texts
    def normalize(text):
        # Convert to lowercase and strip whitespace/punctuation from ends
        text = text.lower().strip(string.punctuation + string.whitespace)
        # Collapse whitespace
        text = ' '.join(text.split())
        return text
    
    target_norm = normalize(target)
    doc_norm = normalize(document)
    
    return target_norm in doc_norm

def analyze_domain(domain: str, project_root: str, probe_type: str):
    """
    Analyzes a single domain to count the origin of probe targets and save indices.
    """
    print(f"Analyzing domain: {domain} for probe type: {probe_type}...")
    
    # --- Load Probe Targets ---
    probe_folder = 'facts' if probe_type == 'knowledge' else 'inference'
    probes_dir = os.path.join(project_root, 'data/probes', probe_folder, domain)
    probes_file = find_probe_file(probes_dir, probe_type)
    
    if not probes_file:
        print(f"  - No probe CSV file found in {probes_dir}. Skipping.")
        return None
        
    print(f"  - Using probe file: {probes_file}")
    probes_df = pd.read_csv(probes_file)

    # --- Load Text Sources ---
    # 1. Explanations (Textbooks, Blogs, StackExchange)
    explanations_dir = os.path.join(project_root, 'data/arxiv/explanations', domain)
    categories = ['textbooks', 'blogs', 'stackexchange']
    explanations_text = ""
    for category in categories:
        category_dir = os.path.join(explanations_dir, category)
        explanations_text += load_all_text_from_dir(category_dir, '.txt')

    # 2. Source (Cleaned)
    cleaned_file_path = os.path.join(project_root, 'data/arxiv/cleaned', f"{domain}.tex")
    source_text = ""
    if os.path.exists(cleaned_file_path):
        with open(cleaned_file_path, 'r', encoding='utf-8') as f:
            source_text = f.read()

    # 3. Paraphrased (0-9 only)
    paraphrased_dir = os.path.join(project_root, 'data/arxiv/paraphrased', domain)
    paraphrased_text = ""
    if os.path.isdir(paraphrased_dir):
        for i in range(10):
            para_file = os.path.join(paraphrased_dir, f"{i}.tex")
            if os.path.exists(para_file):
                try:
                    with open(para_file, 'r', encoding='utf-8') as f:
                        paraphrased_text += f.read() + "\n"
                except Exception as e:
                    print(f"Could not read {para_file}: {e}")

    if not explanations_text.strip():
        print("  - No explanation text found.")
    if not source_text.strip():
        print("  - No source text found.")
    if not paraphrased_text.strip():
        print("  - No paraphrased text (0-9) found.")

    # --- Perform Analysis Helper ---
    def categorize_probes(match_function):
        in_source_indices = []
        in_source_and_paraphrase_indices = []
        in_explanations_only_indices = []
        in_source_and_paraphrase_only_indices = []
        in_none_indices = []
        in_all_indices = []
        in_explanations_indices = []
        source_only_indices = []

        for index, row in probes_df.iterrows():
            target = str(row['target'])
            if pd.isna(target):
                continue
            
            is_in_expl = match_function(target, explanations_text)
            is_in_source = match_function(target, source_text)
            is_in_para = match_function(target, paraphrased_text)

            # 1) in_source
            if is_in_source:
                in_source_indices.append(index)
            
            # 2) in_source_and_paraphrase
            if is_in_source and is_in_para:
                in_source_and_paraphrase_indices.append(index)

            # 3) in_explanations_only
            if is_in_expl and not is_in_source and not is_in_para:
                in_explanations_only_indices.append(index)

            # 4) in_source_and_paraphrase_only
            if is_in_source and is_in_para and not is_in_expl:
                in_source_and_paraphrase_only_indices.append(index)

            # 5) in_none
            if not is_in_source and not is_in_para and not is_in_expl:
                in_none_indices.append(index)

            # 6) in_all
            if is_in_source and is_in_para and is_in_expl:
                in_all_indices.append(index)
                
            # 7) in_explanations (inclusive)
            if is_in_expl:
                in_explanations_indices.append(index)
                
            # 8) source_only (exclusive)
            if is_in_source and not is_in_para and not is_in_expl:
                source_only_indices.append(index)
        
        return {
            'in_source': in_source_indices,
            'in_source_and_paraphrase': in_source_and_paraphrase_indices,
            'in_explanations_only': in_explanations_only_indices,
            'in_source_and_paraphrase_only': in_source_and_paraphrase_only_indices,
            'in_none': in_none_indices,
            'in_all': in_all_indices,
            'in_explanations': in_explanations_indices,
            'source_only': source_only_indices
        }

    # --- Run Analysis for N-Gram ---
    ngram_results = categorize_probes(is_ngram_in_document)
    filter_output_path = os.path.join(project_root, 'data/probes', probe_folder, domain, 'filter.json')
    os.makedirs(os.path.dirname(filter_output_path), exist_ok=True)
    with open(filter_output_path, 'w', encoding='utf-8') as f:
        json.dump(ngram_results, f, indent=2)
    print(f"  - Saved N-Gram filter data to {filter_output_path}")

    # --- Run Analysis for Exact Match ---
    em_results = categorize_probes(is_exact_match_in_document)
    filter_em_output_path = os.path.join(project_root, 'data/probes', probe_folder, domain, 'filter_em.json')
    with open(filter_em_output_path, 'w', encoding='utf-8') as f:
        json.dump(em_results, f, indent=2)
    print(f"  - Saved Exact Match filter data to {filter_em_output_path}")

    return {
        'ngram': {k: len(v) for k, v in ngram_results.items()},
        'em': {k: len(v) for k, v in em_results.items()},
        'total_targets': len(probes_df['target'].dropna())
    }

def main():
    """
    Main function to orchestrate the analysis of all domains.
    """
    parser = argparse.ArgumentParser(description="Analyze the origin of probe targets within different text sources.")
    parser.add_argument(
        '--probe_type',
        type=str,
        required=True,
        choices=['inference', 'knowledge'],
        help="The type of probes to analyze ('inference' or 'knowledge')."
    )
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    
    probe_folder = 'facts' if args.probe_type == 'knowledge' else 'inference'
    probes_base_dir = os.path.join(project_root, 'data/probes', probe_folder)
    
    if not os.path.isdir(probes_base_dir):
        print(f"Error: Probe directory not found at '{probes_base_dir}'")
        return

    domains = [d for d in os.listdir(probes_base_dir) if os.path.isdir(os.path.join(probes_base_dir, d))]
    
    all_results = {}
    for domain in sorted(domains):
        result = analyze_domain(domain, project_root, args.probe_type)
        if result:
            all_results[domain] = result

    # --- Write Results to File ---
    output_file = os.path.join(project_root, f'results/{args.probe_type}_probe_origin_analysis.txt')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Analysis of {args.probe_type.capitalize()} Probe Target Origins\n")
        f.write("="*40 + "\n\n")
        for domain, result in all_results.items():
            f.write(f"Domain: {domain}\n")
            f.write(f"  - Total Targets Analyzed: {result['total_targets']}\n")
            
            for match_type, label in [('ngram', 'N-Gram Match'), ('em', 'Exact Match')]:
                counts = result[match_type]
                f.write(f"  [{label}]\n")
                f.write(f"    - In Source: {counts['in_source']}\n")
                f.write(f"    - In Source AND Paraphrase: {counts['in_source_and_paraphrase']}\n")
                f.write(f"    - In Explanations ONLY: {counts['in_explanations_only']}\n")
                f.write(f"    - In Source AND Paraphrase ONLY: {counts['in_source_and_paraphrase_only']}\n")
                f.write(f"    - In NONE: {counts['in_none']}\n")
                f.write(f"    - In ALL: {counts['in_all']}\n")
                f.write(f"    - In Explanations (Inclusive): {counts['in_explanations']}\n")
                f.write(f"    - Source ONLY (Exclusive): {counts['source_only']}\n")
            f.write("\n")

    print(f"\nAnalysis complete. Results saved to {output_file}")

if __name__ == '__main__':
    main()
