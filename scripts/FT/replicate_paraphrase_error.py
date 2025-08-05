#!/usr/bin/env python3
"""
Script to replicate and diagnose the paraphrase length assertion error
without running the full LLM training pipeline.
"""

import sys
import os
import pandas as pd
import ast
from transformers import AutoTokenizer

# Add the utils directory to path
sys.path.append('../..')
import utils.llm_training as llm_training

def diagnose_paraphrase_data(csv_path):
    """
    Loads the CSV file and analyzes the paraphrased_atomic_knowledge_probes column
    to identify rows with inconsistent numbers of paraphrases.
    """
    print(f"Loading CSV file: {csv_path}")
    
    if not os.path.exists(csv_path):
        print(f"ERROR: File not found: {csv_path}")
        return
    
    # Load the CSV
    df = pd.read_csv(csv_path)
    print(f"Loaded DataFrame with {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    
    # Check if the paraphrased column exists
    if "paraphrased_atomic_knowledge_probes" not in df.columns:
        print("ERROR: 'paraphrased_atomic_knowledge_probes' column not found in CSV")
        return
    
    print("\nAnalyzing paraphrased_atomic_knowledge_probes column...")
    
    # Get the raw paraphrased data
    paraphrased_probes_raw = df["paraphrased_atomic_knowledge_probes"].tolist()
    
    # Parse the string representations and check lengths
    paraphrase_lengths = []
    problematic_rows = []
    
    for i, probe_str in enumerate(paraphrased_probes_raw):
        try:
            # Parse the string representation of the list
            probe_list = ast.literal_eval(probe_str)
            length = len(probe_list)
            paraphrase_lengths.append(length)
            
            # If this is the first row, set the expected length
            if i == 0:
                expected_length = length
                print(f"Expected number of paraphrases per row: {expected_length}")
            
            # Check if this row has a different length
            if length != expected_length:
                problematic_rows.append((i, length, probe_str[:100] + "..."))
                
        except Exception as e:
            print(f"ERROR parsing row {i}: {e}")
            print(f"Problematic string: {probe_str[:200]}...")
            problematic_rows.append((i, "PARSE_ERROR", str(e)))
    
    # Report findings
    unique_lengths = set(paraphrase_lengths)
    print(f"\nFound {len(unique_lengths)} different paraphrase lengths: {sorted(unique_lengths)}")
    
    if len(unique_lengths) > 1:
        print(f"\n❌ PROBLEM FOUND: Inconsistent paraphrase lengths!")
        print(f"First row has {expected_length} paraphrases, but {len(problematic_rows)} rows have different lengths:")
        
        for row_idx, length, sample_data in problematic_rows[:10]:  # Show first 10 problematic rows
            print(f"  Row {row_idx}: {length} paraphrases")
            if length != "PARSE_ERROR":
                print(f"    Sample: {sample_data}")
        
        if len(problematic_rows) > 10:
            print(f"  ... and {len(problematic_rows) - 10} more problematic rows")
    else:
        print(f"✅ All rows have consistent paraphrase length: {expected_length}")
    
    return problematic_rows

def replicate_callback_error(csv_path):
    """
    Attempts to create the KnowledgeProbeCallback to replicate the exact error.
    """
    print(f"\n{'='*60}")
    print("REPLICATING THE CALLBACK CREATION ERROR")
    print(f"{'='*60}")
    
    # Create a dummy tokenizer (required for callback initialization)
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Attempting to create KnowledgeProbeCallback...")
    try:
        callback = llm_training.KnowledgeProbeCallback(
            tokenizer=tokenizer,
            probe_dataset_path=csv_path,
            max_length=2048,  # from training_config.context_length
            batch_size=8,
        )
        print("✅ KnowledgeProbeCallback created successfully!")
        
    except AssertionError as e:
        print(f"❌ ASSERTION ERROR (as expected): {e}")
        print("\nThis is the exact error you encountered!")
        
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {type(e).__name__}: {e}")

def main():
    """Main function to run the diagnosis."""
    csv_path = '../../data/arxiv/DPO_knowledge_probes_v1.csv'
    
    print("🔍 DIAGNOSING PARAPHRASE DATA ERROR")
    print("=" * 60)
    
    # First, analyze the data to understand the problem
    problematic_rows = diagnose_paraphrase_data(csv_path)
    
    # Then, replicate the exact error from the callback creation
    replicate_callback_error(csv_path)
    
    # Provide suggestions for fixing the issue
    print(f"\n{'='*60}")
    print("SUGGESTED FIXES")
    print(f"{'='*60}")
    
    if problematic_rows:
        print("1. Check the data generation process that created the paraphrased probes")
        print("2. Either:")
        print("   a) Fix the rows with incorrect paraphrase counts, or")
        print("   b) Filter out rows that don't have the expected number of paraphrases")
        print("3. The expected format is a list of exactly 10 paraphrases per row")
        print("4. You can manually inspect the problematic rows listed above")
    else:
        print("No obvious data issues found. The error might be more subtle.")
        print("Consider checking for:")
        print("- Empty lists in some rows")
        print("- Malformed string representations")
        print("- Encoding issues in the CSV file")

if __name__ == "__main__":
    main() 