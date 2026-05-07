#!/usr/bin/env python3
"""
Extract probe text ending with punctuation from facts and inference CSV files.
"""
import os
import sys
import csv
import json
import string
from pathlib import Path
from tqdm import tqdm
import concurrent.futures

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.insert(0, PROJECT_ROOT)
import utils.utils as utils
from utils import probe_paths

# Define paths
BASE_DIR = Path(PROJECT_ROOT) / "probes"
OUTPUT_FILE = "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/probes_with_punctuation.txt"
FIX_OUTPUT_FILE = "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/probes_fixed_comparison.txt"

# Punctuation characters to check for
PUNCTUATION = {'.', '!', '?'}

def extract_probes_from_csv(csv_path, extract_all_data=False):
    """Extract probe column from CSV file."""
    probes = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'probe' in row and row['probe']:
                    probe_text = row['probe'].strip()
                    # Check if probe ends with punctuation (but not "i.e.")
                    if probe_text and probe_text[-1] in PUNCTUATION and not probe_text.endswith('i.e.'):
                        if extract_all_data:
                            probes.append(row)
                        else:
                            probes.append(probe_text)
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
    return probes

def fix_fact_with_llm(row_data):
    """Use LLM to fix the fact so it naturally ends with the target."""
    target = row_data.get('target', '').strip()
    fact = row_data.get('fact', '').strip()
    raw_knowledge = row_data.get('raw_knowledge_statement', fact)
    
    system_prompt = """You will be given a 'fact' statement, a 'target' phrase, and optionally a 'raw_knowledge_statement'. 

Your task is to adjust the 'fact' so that:
1. It reads naturally as a complete sentence
2. It ends exactly with the 'target' phrase (the last words of the fact should be the target)
3. It accurately reflects the knowledge in the 'raw_knowledge_statement' if provided
4. The target should integrate seamlessly at the end of the sentence
5. All mathematical expressions MUST be in valid LaTeX format (use $ for inline math). However, standalone numbers should be written without $.

Example:
Original Fact: "According to the paper "QLoRA: Efficient Finetuning of Quantized LLMs", the authors report that their MMLU and Elo results indicate that, with a given finetuning and inference resource budget, it is beneficial to increase a particular model attribute. the number of parameters"
Target: "the number of parameters"

Fixed Fact: "According to the paper "QLoRA: Efficient Finetuning of Quantized LLMs," MMLU and Elo results show that, for a fixed finetuning and inference resource budget, it is beneficial to decrease precision of the base model while increasing the number of parameters"

Return a JSON object with a single key "fixed_fact" containing the adjusted fact statement."""
    
    user_prompt = f"""Target: {target}
Original Fact: {fact}
Raw Knowledge: {raw_knowledge}

Please provide the fixed fact that naturally ends with the target phrase."""
    
    prompt = {'system': system_prompt, 'user': user_prompt}
    
    try:
        response = utils.query_llm(
            prompt, 
            model='gpt-5',
            system_prompt_included=True, 
            reasoning_effort='low',
            return_json=True, 
            max_tokens=500
        )
        
        if isinstance(response, str):
            response = json.loads(response)
        
        fixed_fact = response.get('fixed_fact', '').strip()
        
        # Verify it ends with target
        target_stripped = target.strip().rstrip('.,!?;: ')
        if fixed_fact.rstrip('.,!?;: ').endswith(target_stripped):
            return fixed_fact
        else:
            print(f"Warning: Fixed fact doesn't end with target. Returning original.")
            return fact
            
    except Exception as e:
        print(f"Error in LLM call: {e}")
        return fact

def main():
    all_probe_data = []
    csv_files_to_update = {}  # Track which CSV files need updating
    
    # Process facts domains (probes_v9.csv)
    print("Processing facts probes_v9.csv files...")
    for csv_path in BASE_DIR.glob("*/**/facts/probes_v9.csv"):
        domain = csv_path.parent.parent.name
        print(f"  Reading {domain}/probes_v9.csv...")
        probes = extract_probes_from_csv(csv_path, extract_all_data=True)
        print(f"    Found {len(probes)} probes ending with punctuation")
        for row in probes:
            row['domain'] = domain
            row['source_file'] = 'facts/probes_v9.csv'
            row['csv_path'] = str(csv_path)
        all_probe_data.extend(probes)
        if str(csv_path) not in csv_files_to_update:
            csv_files_to_update[str(csv_path)] = []
    
    # Process inference domains (probes_v6.csv)
    print("\nProcessing inference probes_v6.csv files...")
    for csv_path in BASE_DIR.glob("*/**/inference/probes_v6.csv"):
        domain = csv_path.parent.parent.name
        print(f"  Reading {domain}/probes_v6.csv...")
        probes = extract_probes_from_csv(csv_path, extract_all_data=True)
        print(f"    Found {len(probes)} probes ending with punctuation")
        for row in probes:
            row['domain'] = domain
            row['source_file'] = 'inference/probes_v6.csv'
            row['csv_path'] = str(csv_path)
        all_probe_data.extend(probes)
        if str(csv_path) not in csv_files_to_update:
            csv_files_to_update[str(csv_path)] = []
    
    print(f"\nTotal probes to process: {len(all_probe_data)}")
    
    # Fix facts using LLM in parallel
    print("\nFixing facts with LLM...")
    fixed_results = []
    updates_by_csv = {}  # Group updates by CSV file
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        future_to_row = {executor.submit(fix_fact_with_llm, row): row for row in all_probe_data}
        for future in tqdm(concurrent.futures.as_completed(future_to_row), total=len(all_probe_data), desc="Fixing facts"):
            row = future_to_row[future]
            try:
                fixed_fact = future.result()
                
                # Extract new probe from fixed fact
                target = row.get('target', '').strip().rstrip('.,!?;: ')
                fixed_fact_stripped = fixed_fact.rstrip('.,!?;: ')
                
                if fixed_fact_stripped.endswith(target):
                    new_probe = fixed_fact_stripped[:-len(target)].rstrip()
                else:
                    new_probe = row.get('probe', '')
                
                # Store for comparison output
                fixed_results.append({
                    'domain': row.get('domain', ''),
                    'source_file': row.get('source_file', ''),
                    'target': target,
                    'old_fact': row.get('fact', ''),
                    'new_fact': fixed_fact,
                    'old_probe': row.get('probe', ''),
                    'new_probe': new_probe,
                    'raw_knowledge': row.get('raw_knowledge_statement', '')
                })
                
                # Store for CSV updates - use original probe value as key to find the row
                csv_path = row.get('csv_path')
                if csv_path not in updates_by_csv:
                    updates_by_csv[csv_path] = {}
                updates_by_csv[csv_path][row.get('probe', '')] = {
                    'new_fact': fixed_fact,
                    'new_probe': new_probe
                }
            except Exception as exc:
                print(f'Row generated an exception: {exc}')
    
    # Write comparison output
    print(f"\nWriting comparison to {FIX_OUTPUT_FILE}")
    with open(FIX_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for i, result in enumerate(fixed_results, 1):
            f.write(f"=" * 80 + "\n")
            f.write(f"PROBE #{i}\n")
            f.write(f"Domain: {result['domain']} | Source: {result['source_file']}\n")
            f.write(f"-" * 80 + "\n")
            f.write(f"Target: {result['target']}\n")
            f.write(f"\n")
            if result['raw_knowledge']:
                f.write(f"Raw Knowledge:\n{result['raw_knowledge']}\n\n")
            f.write(f"OLD Fact:\n{result['old_fact']}\n\n")
            f.write(f"NEW Fact:\n{result['new_fact']}\n\n")
            f.write(f"OLD Probe:\n{result['old_probe']}\n\n")
            f.write(f"NEW Probe:\n{result['new_probe']}\n\n")
    
    # Update CSV files with new values
    print(f"\nUpdating CSV files...")
    for csv_path, updates in updates_by_csv.items():
        print(f"  Updating {csv_path}...")
        
        # Read all rows from CSV
        rows = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            for row in reader:
                # Check if this row needs updating
                old_probe = row.get('probe', '').strip()
                if old_probe in updates:
                    # Update fact and probe
                    row['fact'] = updates[old_probe]['new_fact']
                    row['probe'] = updates[old_probe]['new_probe']
                rows.append(row)
        
        # Write updated rows back to CSV
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"    Updated {len(updates)} rows in {csv_path}")
    
    print(f"\nProcessed {len(fixed_results)} probes")
    print(f"Updated {len(updates_by_csv)} CSV files")
    print("Done!")

if __name__ == "__main__":
    main()
