import os
import sys
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import statistics
import argparse

# Add parent directory to path
sys.path.append('../../')
import utils.utils as utils

prompt = {}
prompt['system'] = """You will be given a portion of an academic paper. Please paraphrase the text. 
# Instructions
- Paraphrase the text as much as possible, writing in a different tone and style (that's still within academic bounds) and changing the sentence structure.
- Keeping the meaning of the text.
- Keep latex formatting identical to the original text. 
- Leave proper nouns, titles, section headers, equations, and domain-specific terminology unchanged.
- Only output the paraphrased text, no other text."""

def paraphrase_paragraph(part):
    if len(part) < 200:
        return part
    else:
        prompt['user'] = f"Text: {part}"
        return utils.query_llm(
            prompt=prompt,
            model="gpt-4.1", 
            temperature=1.25,
            top_p=0.95,
            system_prompt_included=True,
        )

def split_and_merge_paragraphs(text):
    """Split text into paragraphs and merge short ones, returning processed paragraphs."""
    paragraphs = text.split('\n\n')
    print(f"Initial number of paragraphs: {len(paragraphs)}")
    
    # Combine short paragraphs until none are shorter than 150 words
    while True:
        # Find the shortest paragraph
        min_length = float('inf')
        shortest_idx = -1
        
        for i, p in enumerate(paragraphs):
            word_count = len(p.split())
            if word_count < 200 and word_count < min_length:
                min_length = word_count
                shortest_idx = i
        
        if shortest_idx == -1:  # No paragraphs shorter than 150 words
            break
        
        # Find the shorter adjacent paragraph to merge with
        merge_with = -1
        
        # Check previous paragraph
        prev_length = len(paragraphs[shortest_idx - 1].split()) if shortest_idx > 0 else float('inf')
        # Check next paragraph  
        next_length = len(paragraphs[shortest_idx + 1].split()) if shortest_idx < len(paragraphs) - 1 else float('inf')
        
        if prev_length <= next_length and shortest_idx > 0:
            merge_with = shortest_idx - 1
        elif shortest_idx < len(paragraphs) - 1:
            merge_with = shortest_idx + 1
        
        if merge_with == -1:  # No adjacent paragraphs to merge with
            break
        
        # Merge paragraphs
        if merge_with < shortest_idx:
            paragraphs[merge_with] = paragraphs[merge_with] + '\n\n' + paragraphs[shortest_idx]
            paragraphs.pop(shortest_idx)
        else:
            paragraphs[shortest_idx] = paragraphs[shortest_idx] + '\n\n' + paragraphs[merge_with]
            paragraphs.pop(merge_with)
    
    print(f"Final number of paragraphs after merging: {len(paragraphs)}")
    
    # Report paragraph length statistics
    paragraph_lengths = [len(p.split()) for p in paragraphs]
    if paragraph_lengths:
        print(f"Paragraph length statistics:")
        print(f"  Min: {min(paragraph_lengths)} words")
        print(f"  Max: {max(paragraph_lengths)} words")
        print(f"  Mean: {statistics.mean(paragraph_lengths):.1f} words")
        print(f"  Median: {statistics.median(paragraph_lengths):.1f} words")
    
    return paragraphs

def paraphrase_texts(paragraphs):
    """Paraphrase each paragraph linearly and return joined text."""
    paraphrased_paragraphs = []
    for paragraph in tqdm(paragraphs, desc="Paraphrasing paragraphs"):
        paraphrased_paragraphs.append(paraphrase_paragraph(paragraph))
    return paraphrased_paragraphs

def process_papers(paper_filenames=None, start_index=0, num_paraphrases=9):
    input_dir = "../../data/arxiv/cleaned/"
    output_dir = "../../data/arxiv/paraphrased/"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of files in cleaned directory
    if paper_filenames:
        # If paper filenames are provided, use them
        files = [f if f.endswith('.tex') else f'{f}.tex' for f in paper_filenames]
    else:
        # Otherwise, process all files in the directory
        files = [f for f in os.listdir(input_dir) if f.endswith('.tex')]
    
    for filename in files:
        print(f"Processing {filename}")
        
        # Read the cleaned text
        with open(os.path.join(input_dir, filename), 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Split and merge paragraphs once
        paragraphs = split_and_merge_paragraphs(text)
        
        # Generate paraphrased versions in parallel
        def generate_paraphrase(i):
            print(f"Generating paraphrase {i} for {filename}")
            paraphrased_paper = '\n\n'.join(paraphrase_texts(paragraphs))
            
            # Save paraphrased version
            base_name = os.path.splitext(filename)[0]
            output_filename = f"{base_name}/{i}.tex"
            output_path = os.path.join(output_dir, output_filename)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(paraphrased_paper)
            
            print(f"Saved paraphrased version to {output_path}")
            return output_path
        
        with ThreadPoolExecutor(max_workers=num_paraphrases) as executor:
            list(executor.map(generate_paraphrase, range(start_index, start_index + num_paraphrases)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paraphrase academic papers.")
    parser.add_argument(
        '--papers', 
        nargs='+', 
        help='Optional list of paper filenames to process (e.g., paper1, paper2.tex). If not provided, all papers in the cleaned directory will be processed.'
    )
    parser.add_argument(
        '--start_index',
        type=int,
        default=0,
        help='The starting index for generating paraphrases.'
    )
    parser.add_argument(
        '--num_paraphrases',
        type=int,
        default=9,
        help='The number of paraphrases to generate.'
    )
    args = parser.parse_args()
    
    process_papers(
        paper_filenames=args.papers,
        start_index=args.start_index,
        num_paraphrases=args.num_paraphrases
    )
