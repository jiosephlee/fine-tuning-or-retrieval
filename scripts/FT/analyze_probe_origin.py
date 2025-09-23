import os
import sys
import pandas as pd
import re
import json

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.pipeline import is_text_in_document

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

def analyze_domain(domain: str, project_root: str):
    """
    Analyzes a single domain to count the origin of probe targets and save indices.
    """
    print(f"Analyzing domain: {domain}...")
    
    # --- Load Probe Targets ---
    probes_file = os.path.join(project_root, 'data/probes/inference', domain, 'probes_v6.csv')
    if not os.path.exists(probes_file):
        print(f"  - Probes file not found. Skipping.")
        return None
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
        print(f"  - No explanation text found.")
    if not source_and_paraphrased_text.strip():
        print(f"  - No source or paraphrased text found.")

    # --- Perform Analysis ---
    in_explanations_only_indices = []
    in_source_only_indices = []
    
    # Use a less strict threshold to account for minor variations
    threshold = 1 

    for index, row in probes_df.iterrows():
        target = str(row['target'])
        if pd.isna(target):
            continue
        
        # Normalize target for better matching, especially with LaTeX
        normalized_target = target.strip().lower()
        
        in_expl = is_text_in_document(normalized_target, explanations_text, threshold)
        in_src = is_text_in_document(normalized_target, source_and_paraphrased_text, threshold)

        if in_expl and not in_src:
            in_explanations_only_indices.append(index)
        elif in_src and not in_expl:
            in_source_only_indices.append(index)
            
    # --- Save filter.json ---
    filter_data = {
        'in_explanations_only': in_explanations_only_indices,
        'in_source_only': in_source_only_indices
    }
    filter_output_path = os.path.join(project_root, 'data/probes/inference', domain, 'filter.json')
    os.makedirs(os.path.dirname(filter_output_path), exist_ok=True)
    with open(filter_output_path, 'w', encoding='utf-8') as f:
        json.dump(filter_data, f, indent=2)
    print(f"  - Saved filter data to {filter_output_path}")

    return {
        'in_explanations_but_not_source': len(in_explanations_only_indices),
        'in_source_but_not_explanations': len(in_source_only_indices),
        'total_targets': len(probes_df['target'].dropna())
    }

def main():
    """
    Main function to orchestrate the analysis of all domains.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    
    probes_base_dir = os.path.join(project_root, 'data/probes/inference')
    domains = [d for d in os.listdir(probes_base_dir) if os.path.isdir(os.path.join(probes_base_dir, d))]
    
    all_results = {}
    for domain in sorted(domains):
        result = analyze_domain(domain, project_root)
        if result:
            all_results[domain] = result

    # --- Write Results to File ---
    output_file = os.path.join(project_root, 'results/probe_origin_analysis.txt')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Analysis of Inference Probe Target Origins\n")
        f.write("="*40 + "\n\n")
        for domain, counts in all_results.items():
            f.write(f"Domain: {domain}\n")
            f.write(f"  - Total Targets Analyzed: {counts['total_targets']}\n")
            f.write(f"  - Targets in Explanations ONLY: {counts['in_explanations_but_not_source']}\n")
            f.write(f"  - Targets in Source/Paraphrased ONLY: {counts['in_source_but_not_explanations']}\n\n")

    print(f"\nAnalysis complete. Results saved to {output_file}")

if __name__ == '__main__':
    main()
