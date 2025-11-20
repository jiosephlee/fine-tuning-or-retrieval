import os
import sys
import json
import pandas as pd
from typing import List, Dict, Any

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import utils.utils as utils
from utils.pipeline import is_text_in_document

def find_similar_excerpts(probes: List[str], explanation_text: str, source_filename: str) -> List[Dict[str, Any]]:
    """
    Uses an LLM to find the 5 most similar excerpts from an explanation text for a given list of probes.
    """
    system_prompt = """You will be given a list of "inference probes" and an "explanation text". Your task is to find the 5 pairs of (inference probe, explanation excerpt) that are most similar to each other. The excerpt should be a sentence or a short paragraph from the explanation texts. For each pair, provide a similarity score from 0 to 1.

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
    
    response_json = utils.query_llm(prompt, model='gpt-5-mini', system_prompt_included=True, return_json=True, max_tokens=4000)
    
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
    subfolders = ['', 'blogs', 'stackexchange', 'textbooks']
    
    all_results = {}

    for domain in domains:
        print(f"Processing domain: {domain}...")
        
        # Load probes
        probes_file = os.path.join(probes_base_dir, domain, 'probes_v7.csv')
        if not os.path.exists(probes_file):
            print(f"Probes file not found for {domain}, skipping.")
            continue
        
        probes_df = pd.read_csv(probes_file)
        probes = probes_df['fact'].tolist()
        
        candidate_pairs_for_domain = []
        
        # Check each subfolder
        for subfolder in subfolders:
            explanation_dir = os.path.join(explanations_base_dir, domain, subfolder) if subfolder else os.path.join(explanations_base_dir, domain)
            
            if not os.path.isdir(explanation_dir):
                print(f"  Directory not found: {subfolder or 'explanations'}, skipping.")
                continue
            
            explanation_files = [f for f in os.listdir(explanation_dir) if f.endswith('.txt')]
            
            if not explanation_files:
                print(f"  No .txt files found in {subfolder or 'explanations'}")
                continue
            
            print(f"  Checking {len(explanation_files)} files in {subfolder or 'explanations'}...")
            
            for filename in explanation_files:
                print(f"    Checking against {filename}...")
                filepath = os.path.join(explanation_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    explanation_text = f.read()
                
                similar_pairs = find_similar_excerpts(probes, explanation_text, filename)
                
                for pair in similar_pairs:
                    excerpt = pair.get('excerpt')
                    if excerpt and is_text_in_document(excerpt, explanation_text, threshold=0.9):
                        pair['source_file'] = filename
                        pair['source_type'] = subfolder if subfolder else 'explanations'
                        candidate_pairs_for_domain.append(pair)
                    else:
                        print(f"    Dropping excerpt not found in source: {excerpt[:100] if excerpt else 'None'}...")

        # Get top 5 for the domain
        candidate_pairs_for_domain.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
        all_results[domain] = candidate_pairs_for_domain[:5]

    output_file = os.path.join(project_root, 'results/inference_probe_leakage.json')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)

    print(f"Results saved to {output_file}")

if __name__ == '__main__':
    main()
