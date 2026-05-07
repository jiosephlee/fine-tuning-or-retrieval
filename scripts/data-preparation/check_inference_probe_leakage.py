import os
import sys
import json
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import utils.utils as utils
from utils.pipeline import is_text_in_document
from utils import probe_paths

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
    with ThreadPoolExecutor(max_workers=2) as executor:
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


def process_domain(domain: str, probes_base_dir: str, explanations_base_dir: str, subfolders: List[str], inference_type_filter: Optional[str] = None) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Process a single domain to find inference probe leakage.
    
    Args:
        domain: Domain name
        probes_base_dir: Base directory for probes
        explanations_base_dir: Base directory for explanations
        subfolders: List of subfolders to check
        inference_type_filter: If "Conceptual Synthesis", only process probes with that type.
                              If "non_Conceptual_Synthesis", only process probes without that type.
                              If None, process all probes.
    Returns (domain_name, candidate_pairs_for_domain)
    """
    print(f"Processing domain: {domain}...")

    # Load probes
    probes_file = str(probe_paths.resolve_probe_path("inference", domain, "v7"))
    if not os.path.exists(probes_file):
        print(f"Probes file not found for {domain}, skipping.")
        return domain, []

    probes_df = pd.read_csv(probes_file)
    
    # Filter probes by inference_type if specified
    if inference_type_filter == "Conceptual Synthesis":
        if 'inference_type' in probes_df.columns:
            probes_df = probes_df[probes_df['inference_type'] == 'Conceptual Synthesis']
            print(f"  Filtered to {len(probes_df)} 'Conceptual Synthesis' probes")
        else:
            print("  WARNING: 'inference_type' column not found, processing all probes")
    elif inference_type_filter == "non_Conceptual_Synthesis":
        if 'inference_type' in probes_df.columns:
            probes_df = probes_df[probes_df['inference_type'] != 'Conceptual Synthesis']
            print(f"  Filtered to {len(probes_df)} non-'Conceptual Synthesis' probes")
        else:
            print("  WARNING: 'inference_type' column not found, processing all probes")
    
    if len(probes_df) == 0:
        print("  No probes after filtering, skipping domain.")
        return domain, []
    
    probes = probes_df['fact'].tolist()

    candidate_pairs_for_domain = []

    # Process each subfolder sequentially
    for subfolder in subfolders:
        subfolder_pairs = process_subfolder(domain, subfolder, probes, explanations_base_dir)
        candidate_pairs_for_domain.extend(subfolder_pairs)

    # Get top 3 for the domain
    candidate_pairs_for_domain.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
    return domain, candidate_pairs_for_domain[:50]


def find_similar_excerpts(probes: List[str], explanation_text: str, source_filename: str) -> List[Dict[str, Any]]:
    """
    Uses an LLM to find the most similar excerpt from an explanation text for a given list of probes.
    """
    system_prompt = """You will be given a list of "inference probes" and an "explanation text". Your task is to find 2 pairs of (inference probe, explanation excerpt) that are most similar to each other. The excerpt should be a sentence or a short paragraph from the explanation texts. For each pair, provide a similarity score from 0 to 1.

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
    
    response_json = utils.query_llm(prompt, model='openai-gpt-5', reasoning_effort='minimal', system_prompt_included=True, return_json=True, max_tokens=4000, is_hippa=True)
    
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


def run_analysis(inference_type_filter: str, output_suffix: str):
    """
    Run the inference probe leakage analysis with a specific filter.
    
    Args:
        inference_type_filter: Filter type ("Conceptual Synthesis" or "non_Conceptual_Synthesis")
        output_suffix: Suffix for output filename
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))

    explanations_base_dir = os.path.join(project_root, 'data/arxiv/explanations')

    domains = probe_paths.get_all_domains_from_probe_kind("inference")
    subfolders = ['blogs', 'stackexchange', 'textbooks']
    
    print(f"\n{'='*80}")
    print(f"Running analysis for: {inference_type_filter}")
    print(f"{'='*80}\n")
    
    all_results = {}

    # Process domains sequentially, but files within subfolders in parallel
    for domain in domains:
        domain_name, candidate_pairs = process_domain(domain, "", explanations_base_dir, subfolders, inference_type_filter)
        all_results[domain_name] = candidate_pairs

    output_file = os.path.join(project_root, f'results/inference_probe_leakage_{output_suffix}.json')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {output_file}")


def main():
    """
    Main function to check for inference probe leakage in explanation materials.
    Runs the analysis twice: once for "Conceptual Synthesis" probes and once for non-"Conceptual Synthesis" probes.
    """
    # Run analysis for "Conceptual Synthesis" probes
    run_analysis("Conceptual Synthesis", "conceptual_synthesis")
    
    # Run analysis for non-"Conceptual Synthesis" probes
    run_analysis("non_Conceptual_Synthesis", "non_conceptual_synthesis")

if __name__ == '__main__':
    main()
