import os
import sys
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Add parent directory to path
sys.path.append('../../')
import utils.utils as utils

def check_paper_knowledge(filepath):
    """
    Asks an LLM if it has seen a paper and has a thorough understanding of it.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    # Taking the first 4000 characters as a sample, as that should be enough to contain the abstract and introduction.
    sample = text[:4000]

    prompt = {
        'system': "You are an expert researcher. You will be given an academic paper. Your task is to determine if you have a thorough understanding of this paper from your training data. Please answer with only 'Yes' or 'No'.",
        'user': f"Do you have a thorough understanding of the following paper? Please answer with only 'Yes' or 'No'.\n\n{sample}"
    }

    # As gpt-5-mini is not available, I am using gpt-4o.
    response = utils.query_llm(
        prompt=prompt,
        model="gpt-5-mini",
        temperature=0,
        system_prompt_included=True,
    )
    return 'yes' in response.lower()

def process_paper(filename, input_dir, output_dir):
    """
    Processes a single paper: checks for knowledge and copies if known.
    Returns the filename if the model has knowledge of it, otherwise None.
    """
    filepath = os.path.join(input_dir, filename)
    try:
        if check_paper_knowledge(filepath):
            output_path = os.path.join(output_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f_in, open(output_path, 'w', encoding='utf-8') as f_out:
                f_out.write(f_in.read())
            return filename
    except Exception as e:
        print(f"Error processing {filename}: {e}")
    return None

def filter_papers():
    input_dir = "data/arxiv/cleaned/"
    output_dir = "data/arxiv/filtered/"
    os.makedirs(output_dir, exist_ok=True)

    files = [f for f in os.listdir(input_dir) if f.endswith('.tex')]
    
    known_papers = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Create a list of futures
        futures = [executor.submit(process_paper, filename, input_dir, output_dir) for filename in files]
        
        # Process futures as they complete
        for future in tqdm(futures, total=len(files), desc="Filtering papers"):
            result = future.result()
            if result:
                known_papers.append(result)

    print("\n--- Known Papers ---")
    for paper in sorted(known_papers):
        print(paper)
    
    print(f"\nFound {len(known_papers)} known papers out of {len(files)} total.")

if __name__ == "__main__":
    filter_papers()
