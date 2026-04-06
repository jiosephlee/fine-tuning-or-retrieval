import os
import re
import pandas as pd
import string
from transformers import PreTrainedTokenizer
import difflib
from utils import probe_paths

def get_debug_dir(probe_type: str, paper_name: str) -> str:
    """Constructs the path to the debugging directory for a given probe type and paper."""
    return str(probe_paths.resolve_probe_dir(probe_type, paper_name) / 'debugging')

def save_df_for_debugging(df: pd.DataFrame, filename: str, probe_type: str, paper_name: str, columns_to_save: list):
    """Saves specified columns of a DataFrame to a text file for debugging."""
    debug_dir = get_debug_dir(probe_type, paper_name)
    os.makedirs(debug_dir, exist_ok=True)
    path = os.path.join(debug_dir, filename)
    with open(path, 'w', encoding='utf-8') as f:
        for index, row in df.iterrows():
            f.write(f"----- Row {index} -----\n")
            for col in columns_to_save:
                if col in df.columns:
                    value = row[col]
                    # Check for non-null scalars and non-empty lists/iterables
                    is_present = False
                    if isinstance(value, (list, tuple)):
                        if value:  # An empty list evaluates to False
                            is_present = True
                    elif pd.notna(value):
                        is_present = True
                    
                    if is_present:
                        f.write(f"{col}: {row[col]}\n")
            f.write("\n")

def save_debug_file(content: str, filename: str, probe_type: str, paper_name: str):
    """Saves raw string content to a text file for debugging."""
    debug_dir = get_debug_dir(probe_type, paper_name)
    os.makedirs(debug_dir, exist_ok=True)
    path = os.path.join(debug_dir, filename)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
        
def is_text_in_document(text_to_find: str, document: str, threshold: float = 1.0) -> bool:
    """
    Checks if a text snippet exists in a larger document, ignoring whitespace, with a certain threshold.
    Returns False if text_to_find is None or empty.
    """
    if not text_to_find or pd.isna(text_to_find):
        return False
    
    # Normalize by removing all whitespace and lowercasing
    normalized_document = re.sub(r'\s+', '', document.lower())
    normalized_text = re.sub(r'\s+', '', text_to_find.strip().lower())

    if threshold == 1.0:
        return normalized_text in normalized_document
    else:
        if len(normalized_text) == 0:
            return True # Or False, depending on desired behavior for empty strings
            
        # Find longest sequences iteratively and remove them
        matched_chars = 0
        remaining_text = normalized_text
        remaining_doc = normalized_document
        
        while remaining_text:
            s = difflib.SequenceMatcher(None, remaining_text, remaining_doc, autojunk=False)
            match = s.find_longest_match(0, len(remaining_text), 0, len(remaining_doc))
            
            if match.size >= 2:  # Only count sequential matches of 2+ characters
                matched_chars += match.size
                # Remove the matched portion from remaining_text
                remaining_text = remaining_text[:match.a] + remaining_text[match.a + match.size:]
                # Remove the matched portion from remaining_doc  
                remaining_doc = remaining_doc[:match.b] + remaining_doc[match.b + match.size:]
            else:
                break  # No more meaningful matches found
        
        return (matched_chars / len(normalized_text)) >= threshold


def check_tokenizer_consistency(df: pd.DataFrame, tokenizer: PreTrainedTokenizer):
    """
    Checks for string and token-level consistency between 'probe', 'target', and 'fact' columns.
    """
    contexts = df['probe'].tolist()
    targets = df['target'].tolist()
    facts = df['fact'].tolist()

    print(f"\n--- Tokenizer and String Consistency Check ---")
    string_mismatches = 0
    token_mismatches = 0

    for i, (context, target, fact) in enumerate(zip(contexts, targets, facts)):
        # --- String Level Check ---
        reconstructed_string = context + target
        if reconstructed_string != fact:
            string_mismatches += 1
            print(f"--- ❌ STRING MISMATCH: Sample #{i} ---")
            continue

        # --- Tokenization Level Check ---
        tokenized_fact = tokenizer(fact, add_special_tokens=False)['input_ids']
        tokenized_context = tokenizer(context, add_special_tokens=False)['input_ids']
        tokenized_target = tokenizer(target, add_special_tokens=False)['input_ids']
        tokenized_parts_combined = tokenized_context + tokenized_target

        if tokenized_fact != tokenized_parts_combined:
            token_mismatches += 1
            print(f"--- ❌ TOKEN MISMATCH: Sample #{i} ---")
            print(f"  Fact tokens:    {tokenized_fact}")
            print(f"  Combined tokens: {tokenized_parts_combined}")

    print(f"\nSummary:")
    print(f"String mismatches: {string_mismatches}")
    print(f"Token mismatches: {token_mismatches}")
    print(f"Total samples: {len(contexts)}")
    return string_mismatches == 0 and token_mismatches == 0

def process_papers(process_function, paper_directory: str, file_filter: str = None, extension: str = '.tex', **kwargs):
    """
    Iterates through cleaned papers in a directory and runs a processing function on each.

    Args:
        process_function: A function that takes (paper_name, paper_content, **kwargs) as arguments.
        paper_directory: The directory containing the source files.
        file_filter: An optional string. If provided, only filenames containing this string will be processed.
        extension: File extension to look for (default: '.tex'). Use '.txt' for legal/medical domains.
        **kwargs: Additional keyword arguments to pass to the process_function.
    """
    for filename in sorted(os.listdir(paper_directory)):
        if filename.endswith(extension):
            if file_filter and file_filter not in filename:
                continue
            paper_name = filename.replace(extension, '')
            print(f"\n{'='*20} Processing {paper_name} {'='*20}")
            file_path = os.path.join(paper_directory, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            process_function(paper_name, content, **kwargs)
