import os
import json
import pandas as pd
import ast
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils import probe_paths

def append_sentences_to_json(probes_base_dir: str, json_file_path: str):
    """
    Appends the 'text_sentences' from probe CSVs to the corresponding entries in a JSON file.

    Args:
        probes_base_dir: The base directory where the probe CSV files are located.
        json_file_path: The path to the JSON file to be updated.
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            leakage_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file {json_file_path} was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_file_path}.")
        return

    all_domains = list(leakage_data.keys())

    for domain in all_domains:
        print(f"Processing domain: {domain}...")
        probes_file = str(probe_paths.resolve_probe_path("inference", domain, "v6"))

        if not os.path.exists(probes_file):
            print(f"  - Probes file not found for {domain}, skipping.")
            continue

        try:
            probes_df = pd.read_csv(probes_file)
            # Create a mapping from the 'fact' to the 'text_sentences'
            # The 'fact' column should uniquely identify the probe
            probe_map = probes_df.set_index('fact')['text_sentences'].to_dict()

            # Update the leakage data for the current domain
            for item in leakage_data.get(domain, []):
                probe_text = item.get('probe')
                if probe_text in probe_map:
                    # The text_sentences are stored as a string representation of a list
                    # Use ast.literal_eval to safely parse it into a list
                    try:
                        sentences_str = probe_map[probe_text]
                        item['text_sentences'] = ast.literal_eval(sentences_str)
                    except (ValueError, SyntaxError) as e:
                        print(f"  - Could not parse text_sentences for probe: {probe_text[:50]}... Error: {e}")
                        item['text_sentences'] = [] # Default to empty list on error
                else:
                     item['text_sentences'] = []


        except Exception as e:
            print(f"  - An error occurred while processing {domain}: {e}")
            continue
            
    # Write the updated data back to the same JSON file
    try:
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(leakage_data, f, indent=2)
        print(f"\nSuccessfully updated {json_file_path} with text_sentences.")
    except Exception as e:
        print(f"\nError writing updated data to {json_file_path}: {e}")


def main():
    """
    Main function to run the script.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    
    json_file_path = os.path.join(project_root, 'results/inference_probe_leakage.json')
    
    append_sentences_to_json("", json_file_path)

if __name__ == '__main__':
    main()
