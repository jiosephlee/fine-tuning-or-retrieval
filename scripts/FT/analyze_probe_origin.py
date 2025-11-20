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
    
    version_tag = 'v9' if probe_type == 'knowledge' else 'v6'
    
    for filename in os.listdir(directory):
        if filename.endswith('.csv') and version_tag in filename:
            return os.path.join(directory, filename)
            
    return None

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
        # Convert to lowercase and remove extra whitespace
        text = text.lower().strip()
        # Remove punctuation for better matching
        text = re.sub(r'[^\w\s]', ' ', text)
        # Split into words and filter empty strings
        words = [w for w in text.split() if w]
        return words
    
    target_words = normalize(target)
    doc_words = normalize(document)
    
    if not target_words:
        return False
    
    n = len(target_words)
    
    # For single word targets, check exact match
    if n <= 2: # single word or 2-gram
        return target_words[0] in doc_words
    
    # For 2+ word targets, check if any 2-gram appears in document
    doc_text = ' '.join(doc_words)
    
    # Generate all 2-grams from target
    for i in range(len(target_words) - 1):
        bigram = ' '.join(target_words[i:i + 2])
        if bigram in doc_text:
            return True
    
    return False

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
    explanations_text = load_all_text_from_dir(os.path.join(project_root, 'data/arxiv/explanations', domain), '.txt')
    paraphrased_text = load_all_text_from_dir(os.path.join(project_root, 'data/arxiv/paraphrased', domain), '.tex')
    
    cleaned_file_path = os.path.join(project_root, 'data/arxiv/cleaned', f"{domain}.tex")
    cleaned_text = ""
    if os.path.exists(cleaned_file_path):
        with open(cleaned_file_path, 'r', encoding='utf-8') as f:
            cleaned_text = f.read()
            
    source_and_paraphrased_text = paraphrased_text + "\n" + cleaned_text

    if not explanations_text.strip():
        print("  - No explanation text found.")
    if not source_and_paraphrased_text.strip():
        print("  - No source or paraphrased text found.")

    # --- Perform Analysis ---
    in_explanations_only_indices = []
    in_source_only_indices = []
    in_both_indices = []
    in_neither_indices = []

    for index, row in probes_df.iterrows():
        target = str(row['target'])
        if pd.isna(target):
            continue
        
        # Use n-gram matching for more flexible matching
        in_expl = is_ngram_in_document(target, explanations_text)
        in_src = is_ngram_in_document(target, source_and_paraphrased_text)

        if in_expl and not in_src:
            in_explanations_only_indices.append(index)
        elif in_src and not in_expl:
            in_source_only_indices.append(index)
        elif in_expl and in_src:
            in_both_indices.append(index)
        else:
            in_neither_indices.append(index)
            
    # --- Save filter.json ---
    filter_data = {
        'in_explanations_only': in_explanations_only_indices,
        'in_source_only': in_source_only_indices,
        'in_both': in_both_indices,
        'in_neither': in_neither_indices
    }
    filter_output_path = os.path.join(project_root, 'data/probes', probe_folder, domain, 'filter.json')
    os.makedirs(os.path.dirname(filter_output_path), exist_ok=True)
    with open(filter_output_path, 'w', encoding='utf-8') as f:
        json.dump(filter_data, f, indent=2)
    print(f"  - Saved filter data to {filter_output_path}")

    return {
        'in_explanations_but_not_source': len(in_explanations_only_indices),
        'in_source_but_not_explanations': len(in_source_only_indices),
        'in_both': len(in_both_indices),
        'in_neither': len(in_neither_indices),
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
        for domain, counts in all_results.items():
            f.write(f"Domain: {domain}\n")
            f.write(f"  - Total Targets Analyzed: {counts['total_targets']}\n")
            f.write(f"  - Targets in Explanations ONLY: {counts['in_explanations_but_not_source']}\n")
            f.write(f"  - Targets in Source/Paraphrased ONLY: {counts['in_source_but_not_explanations']}\n")
            f.write(f"  - Targets in BOTH: {counts['in_both']}\n")
            f.write(f"  - Targets in NEITHER: {counts['in_neither']}\n\n")

    print(f"\nAnalysis complete. Results saved to {output_file}")

if __name__ == '__main__':
    main()
