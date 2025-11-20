import os
import sys
import json
import pandas as pd
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import utils.utils as utils
from utils.pipeline import is_text_in_document

def process_file(filepath: str, filename: str, probes: List[str], subfolder: str) -> List[Dict[str, Any]]:
    """
    Process a single file to find inference probe leakage.
    Returns list of candidate pairs for this file.
    """
    print(f"    Checking against {filename}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        explanation_text = f.read()

    similar_pairs = find_similar_excerpts(probes, explanation_text, filename)

    candidate_pairs_for_file = []
    for pair in similar_pairs:
        excerpt = pair.get('excerpt')
        if excerpt and is_text_in_document(excerpt, explanation_text, threshold=0.9):
            pair['source_file'] = filename
            pair['source_type'] = subfolder
            candidate_pairs_for_file.append(pair)
        else:
            print(f"    Dropping excerpt not found in source: {excerpt[:100] if excerpt else 'None'}...")

    return candidate_pairs_for_file


def process_subfolder(domain: str, subfolder: str, probes: List[str], explanations_base_dir: str) -> List[Dict[str, Any]]:
    """
    Process a single subfolder within a domain to find inference probe leakage.
    Returns list of candidate pairs for this subfolder.
    """
    candidate_pairs_for_subfolder = []

    explanation_dir = os.path.join(explanations_base_dir, domain, subfolder)

    if not os.path.isdir(explanation_dir):
        print(f"  Directory not found: {subfolder}, skipping.")
        return candidate_pairs_for_subfolder

    explanation_files = [f for f in os.listdir(explanation_dir) if f.endswith('.txt')]

    if not explanation_files:
        print(f"  No .txt files found in {subfolder}")
        return candidate_pairs_for_subfolder

    print(f"  Checking {len(explanation_files)} files in {subfolder}...")

    # Process files in parallel using 4 workers
    with ThreadPoolExecutor(max_workers=6) as executor:
        # Submit file processing tasks
        future_to_file = {
            executor.submit(process_file, os.path.join(explanation_dir, filename), filename, probes, subfolder): filename
            for filename in explanation_files
        }

        # Collect results from all files
        for future in as_completed(future_to_file):
            file_pairs = future.result()
            candidate_pairs_for_subfolder.extend(file_pairs)

    return candidate_pairs_for_subfolder


def process_domain(domain: str, probes_base_dir: str, explanations_base_dir: str, subfolders: List[str]) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Process a single domain to find inference probe leakage.
    Returns (domain_name, candidate_pairs_for_domain)
    """
    print(f"Processing domain: {domain}...")

    # Load probes
    probes_file = os.path.join(probes_base_dir, domain, 'probes_v7.csv')
    if not os.path.exists(probes_file):
        print(f"Probes file not found for {domain}, skipping.")
        return domain, []

    probes_df = pd.read_csv(probes_file)
    probes = probes_df['fact'].tolist()

    candidate_pairs_for_domain = []

    # Process each subfolder sequentially
    for subfolder in subfolders:
        subfolder_pairs = process_subfolder(domain, subfolder, probes, explanations_base_dir)
        candidate_pairs_for_domain.extend(subfolder_pairs)

    # Get top 3 for the domain
    candidate_pairs_for_domain.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
    return domain, candidate_pairs_for_domain[:3]


def find_similar_excerpts(probes: List[str], explanation_text: str, source_filename: str) -> List[Dict[str, Any]]:
    """
    Uses an LLM to find the 3 most similar excerpts from an explanation text for a given list of probes.
    """
    system_prompt = """You will be given a list of "inference probes" and an "explanation text". Your task is to find the 3 pairs of (inference probe, explanation excerpt) that are most similar to each other. The excerpt should be a sentence or a short paragraph from the explanation texts. For each pair, provide a similarity score from 0 to 1.

### Output Format
Provide the output in JSON format, as a dictionary with a single key "similar_pairs" which is a list of dictionaries with the following keys:
- "probe": (string) The inference probe.
- "excerpt": (string) The most similar excerpt from the explanation text.
- "similarity_score": (float) The similarity score between the probe and the excerpt (0.0 to 1.0).
"""
    
    prompt = {
        'system': system_prompt,
        'user': f"### Inference Probes\n{json.dumps(probes, indent=2)}\n\n### Explanation Text from {source_filename}\n\n{explanation_text}"
    }
    
    response_json = utils.query_llm(prompt, model='openai-gpt-5', reasoning_effort='low', system_prompt_included=True, return_json=True, max_tokens=4000, is_hippa=True)
    
    if isinstance(response_json, str):
        try:
            response_json = json.loads(response_json)
        except json.JSONDecodeError:
            print(f"Failed to parse JSON response from LLM for {source_filename}.")
            return []

    similar_pairs = response_json.get('similar_pairs', [])
    if not isinstance(similar_pairs, list):
        print(f"Unexpected response format from LLM for {source_filename}: 'similar_pairs' is not a list.")
        return []

    return similar_pairs


def main():
    """
    Main function to check for inference probe leakage in explanation materials.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))

    probes_base_dir = os.path.join(project_root, 'data/probes/inference')
    explanations_base_dir = os.path.join(project_root, 'data/arxiv/explanations')
    
    domains = [d for d in os.listdir(probes_base_dir) if os.path.isdir(os.path.join(probes_base_dir, d))]
    subfolders = ['blogs', 'stackexchange', 'textbooks']
    
    all_results = {}

    # Process domains sequentially, but files within subfolders in parallel
    for domain in domains:
        domain_name, candidate_pairs = process_domain(domain, probes_base_dir, explanations_base_dir, subfolders)
        all_results[domain_name] = candidate_pairs

    output_file = os.path.join(project_root, 'results/inference_probe_leakage.json')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)

    print(f"Results saved to {output_file}")

if __name__ == '__main__':
    main()
