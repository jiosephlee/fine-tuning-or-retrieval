import os
import re
import sys
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import statistics
import argparse

# Add repo root to path relative to this file, not the current working directory.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
import utils.utils as utils

prompt = {}
prompt['system'] = """You will be given a portion of an academic paper. Please paraphrase the text. 
# Instructions
- Paraphrase the text as much as possible, writing in a different tone and style (that's still within academic bounds) and changing the sentence structure.
- Keeping the meaning of the text.
- Keep latex formatting identical to the original text. 
- Leave proper nouns, titles, section headers, equations, and domain-specific terminology unchanged.
- Only output the paraphrased text, no other text."""

MIN_CHARS_TO_PARAPHRASE = 200
MIN_WORDS_PER_PARAGRAPH = 200

def is_latex_only_paragraph(part):
    """Return True when a paragraph only contains LaTeX structure/math, not prose."""
    content = part.strip()
    if not content:
        return True

    # Remove display/inline math first.
    content = re.sub(r"\$\$.*?\$\$", " ", content, flags=re.DOTALL)
    content = re.sub(r"\\\[.*?\\\]", " ", content, flags=re.DOTALL)
    content = re.sub(r"\$.*?\$", " ", content, flags=re.DOTALL)

    # Remove LaTeX commands together with their option/argument payloads so
    # structural text like \section{Introduction} still counts as LaTeX-only.
    command_pattern = r"\\[a-zA-Z@]+[*]?(?:\[[^\]]*\])?(?:\{[^{}]*\})*"
    previous = None
    while content != previous:
        previous = content
        content = re.sub(command_pattern, " ", content)

    # Drop leftover structural characters and collapse whitespace.
    content = re.sub(r"[{}\\_^&~]", " ", content)
    content = re.sub(r"[^A-Za-z0-9\s]", " ", content)
    content = re.sub(r"\s+", " ", content).strip()

    # If no 2+ character alphabetic word remains outside LaTeX commands/math,
    # this is effectively a LaTeX-only paragraph.
    return not re.search(r"\b[A-Za-z]{2,}\b", content)

def is_structural_paragraph(part):
    """Return True for short metadata blocks like titles, authors, and headers."""
    content = part.strip()
    if not content:
        return True

    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines:
        return True

    # Preserve section headers such as "Abstract:" or "Case presentation:".
    if len(lines) == 1 and lines[0].endswith(':'):
        return True

    if len(lines) == 1:
        line = lines[0]
        words = line.split()

        # Preserve title-like and author-like single-line metadata near the top of
        # documents rather than sending them to the paraphraser.
        if len(words) <= 25 and not re.search(r"[.!?]$", line):
            return True

    return False

def is_preserved_paragraph(part):
    """Return True when a paragraph should be copied verbatim."""
    return len(part.strip()) < MIN_CHARS_TO_PARAPHRASE

def sanitize_for_api(text):
    """Remove control characters and invalid unicode that can break API JSON payloads."""
    sanitized = text.encode("utf-8", errors="replace").decode("utf-8")
    sanitized = ''.join(
        ch for ch in sanitized
        if ch in '\n\t' or ord(ch) >= 32
    )
    return sanitized

def paraphrase_paragraph(part, model="gpt-4.1", is_hippa=False, reasoning_effort=None):
    if is_preserved_paragraph(part):
        return part
    else:
        prompt['user'] = f"Text: {sanitize_for_api(part)}"
        return utils.query_llm(
            prompt=prompt,
            model=model,
            temperature=1.25,
            top_p=0.95,
            system_prompt_included=True,
            is_hippa=is_hippa,
            reasoning_effort=reasoning_effort,
        )

def split_and_merge_paragraphs(text):
    """Split text into paragraphs and merge short non-preserved ones."""
    paragraphs = text.split('\n\n')
    print(f"Initial number of paragraphs: {len(paragraphs)}")
    
    # Combine short paraphrased paragraphs until each meets the target length.
    # Paragraphs preserved by character-length heuristic stay isolated.
    while True:
        # Find the shortest non-preserved paragraph below the target length.
        min_length = float('inf')
        shortest_idx = -1
        
        for i, p in enumerate(paragraphs):
            if is_preserved_paragraph(p):
                continue
            word_count = len(p.split())
            if word_count < MIN_WORDS_PER_PARAGRAPH and word_count < min_length:
                min_length = word_count
                shortest_idx = i
        
        if shortest_idx == -1:
            break
        
        # Find the shorter adjacent non-preserved paragraph to merge with.
        merge_with = -1
        
        prev_length = float('inf')
        if shortest_idx > 0 and not is_preserved_paragraph(paragraphs[shortest_idx - 1]):
            prev_length = len(paragraphs[shortest_idx - 1].split())

        next_length = float('inf')
        if shortest_idx < len(paragraphs) - 1 and not is_preserved_paragraph(paragraphs[shortest_idx + 1]):
            next_length = len(paragraphs[shortest_idx + 1].split())
        
        if prev_length != float('inf') and prev_length <= next_length:
            merge_with = shortest_idx - 1
        elif next_length != float('inf'):
            merge_with = shortest_idx + 1
        
        if merge_with == -1:
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

def paraphrase_texts(paragraphs, model="gpt-4.1", is_hippa=False, reasoning_effort=None):
    """Paraphrase each paragraph linearly and return joined text."""
    paraphrased_paragraphs = []
    for paragraph in tqdm(paragraphs, desc="Paraphrasing paragraphs"):
        paraphrased_paragraphs.append(
            paraphrase_paragraph(
                paragraph,
                model=model,
                is_hippa=is_hippa,
                reasoning_effort=reasoning_effort,
            )
        )
    return paraphrased_paragraphs

def normalize_extension(extension):
    extension = extension.strip()
    if not extension:
        raise ValueError("File extension cannot be empty.")
    if not extension.startswith('.'):
        extension = f'.{extension}'
    return extension

def process_papers(
    paper_filenames=None,
    start_index=0,
    num_paraphrases=9,
    input_dir="../../data/arxiv/cleaned/",
    output_dir="../../data/arxiv/paraphrased/",
    input_extension=".tex",
    output_extension=None,
    paraphrase_workers=None,
    model="gpt-4.1",
    is_hippa=False,
    reasoning_effort=None,
):
    input_extension = normalize_extension(input_extension)
    output_extension = normalize_extension(output_extension or input_extension)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of files in cleaned directory
    if paper_filenames:
        # If paper filenames are provided, use them
        files = [
            f if f.endswith(input_extension) else f'{f}{input_extension}'
            for f in paper_filenames
        ]
    else:
        # Otherwise, process all files in the directory
        files = [f for f in os.listdir(input_dir) if f.endswith(input_extension)]
    
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
            paraphrased_paper = '\n\n'.join(
                paraphrase_texts(
                    paragraphs,
                    model=model,
                    is_hippa=is_hippa,
                    reasoning_effort=reasoning_effort,
                )
            )
            
            # Save paraphrased version
            base_name = os.path.splitext(filename)[0]
            output_filename = f"{base_name}/{i}{output_extension}"
            output_path = os.path.join(output_dir, output_filename)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(paraphrased_paper)
            
            print(f"Saved paraphrased version to {output_path}")
            return output_path
        
        max_workers = paraphrase_workers or num_paraphrases
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(executor.map(generate_paraphrase, range(start_index, start_index + num_paraphrases)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paraphrase academic papers.")
    parser.add_argument(
        '--papers', 
        nargs='+', 
        help='Optional list of paper filenames to process. If not provided, all matching files in the input directory will be processed.'
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
    parser.add_argument(
        '--input_dir',
        default='../../data/arxiv/cleaned/',
        help='Directory containing cleaned source files.'
    )
    parser.add_argument(
        '--output_dir',
        default='../../data/arxiv/paraphrased/',
        help='Directory where paraphrased files will be written.'
    )
    parser.add_argument(
        '--input_extension',
        default='.tex',
        help='Source file extension to process, for example .tex or .txt.'
    )
    parser.add_argument(
        '--output_extension',
        default=None,
        help='Output file extension. Defaults to the input extension.'
    )
    parser.add_argument(
        '--paraphrase_workers',
        type=int,
        default=None,
        help='Number of paraphrase versions to generate in parallel for each file. Defaults to num_paraphrases.'
    )
    parser.add_argument(
        '--model',
        default='gpt-4.1',
        help='Model passed to utils.query_llm.'
    )
    parser.add_argument(
        '--is_hippa',
        action='store_true',
        help='Use the Databricks HIPAA-compliant client configured in utils.'
    )
    parser.add_argument(
        '--reasoning_effort',
        default=None,
        help='Optional reasoning effort for GPT-5/o-series models.'
    )
    args = parser.parse_args()
    
    process_papers(
        paper_filenames=args.papers,
        start_index=args.start_index,
        num_paraphrases=args.num_paraphrases,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        input_extension=args.input_extension,
        output_extension=args.output_extension,
        paraphrase_workers=args.paraphrase_workers,
        model=args.model,
        is_hippa=args.is_hippa,
        reasoning_effort=args.reasoning_effort,
    )
