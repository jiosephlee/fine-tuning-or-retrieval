import os
import json
import sys
import pandas as pd
import re
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

sys.path.append('../../')
import utils.utils as utils
from importlib import reload
reload(utils)

STOP_WORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
    'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'will', 'with',
    'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it', 'for',
    'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at', 'this', 'but', 'his', 'by',
    'from', 'they', 'we', 'say', 'her', 'she', 'or', 'an', 'will', 'my', 'one',
    'all', 'would', 'there', 'their', 'what', 'so', 'up', 'out', 'if', 'about',
    'who', 'get', 'which', 'go', 'me', 'when', 'make', 'can', 'like', 'time',
    'no', 'just', 'him', 'know', 'take', 'people', 'into', 'year', 'your', 'good',
    'some', 'could', 'them', 'see', 'other', 'than', 'then', 'now', 'look', 'only',
    'come', 'its', 'over', 'think', 'also', 'back', 'after', 'use', 'two', 'how',
    'our', 'work', 'first', 'well', 'way', 'even', 'new', 'want', 'because', 'any',
    'these', 'give', 'day', 'most', 'us', 'is', 'was', 'are', 'been', 'has', 'had',
    'were', 'said', 'did', 'having', 'may', 'should', 'am', 'being', 'does', 'done'
}

def load_target_words(facts_path, inference_path):
    """Extract all target words from probe CSV files by breaking apart phrases."""
    target_words = set()
    
    if os.path.exists(facts_path):
        df_facts = pd.read_csv(facts_path)
        if 'target' in df_facts.columns:
            for target in df_facts['target'].dropna():
                target_str = str(target).strip()
                clean_str = re.sub(r'[\$\\{}]', '', target_str)
                words = re.findall(r'\b\w+\b', clean_str.lower())
                target_words.update(words)
    
    if os.path.exists(inference_path):
        df_inference = pd.read_csv(inference_path)
        if 'target' in df_inference.columns:
            for target in df_inference['target'].dropna():
                target_str = str(target).strip()
                clean_str = re.sub(r'[\$\\{}]', '', target_str)
                words = re.findall(r'\b\w+\b', clean_str.lower())
                target_words.update(words)
    
    target_words = target_words - STOP_WORDS
    return target_words

def extract_paper_title(paper_path):
    """Extract the title from the paper .tex file."""
    if not os.path.exists(paper_path):
        return "Unknown Paper"
    
    with open(paper_path, 'r') as f:
        content = f.read()
    
    # Look for \title{...}
    title_match = re.search(r'\\title\{([^}]+)\}', content)
    if title_match:
        return title_match.group(1)
    
    return "Unknown Paper"

def generate_non_probe_textbook(paper_name, base_dir, probes_dir):
    """Generate textbook avoiding probe target words."""
    print(f"\n{'='*60}")
    print(f"Generating non-probe textbook for: {paper_name}")
    print(f"{'='*60}")
    
    paper_path = os.path.join(base_dir, 'cleaned', f'{paper_name}.tex')
    outline_path = os.path.join(base_dir, 'explanations', paper_name, 'textbook_outline.json')
    facts_path = os.path.join(probes_dir, 'facts', paper_name, 'probes_v9.csv')
    inference_path = os.path.join(probes_dir, 'inference', paper_name, 'probes_v6.csv')
    output_dir = os.path.join(base_dir, 'explanations', paper_name, 'non_probe_textbook')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load paper
    if not os.path.exists(paper_path):
        print(f"Paper not found: {paper_path}")
        return
    
    with open(paper_path, 'r') as f:
        paper_content = f.read()
    
    paper_title = extract_paper_title(paper_path)
    
    # Load outline
    if not os.path.exists(outline_path):
        print(f"Outline not found: {outline_path}")
        return
    
    with open(outline_path, 'r') as f:
        outline_data = json.load(f)
    
    outline = outline_data['outline']
    print(f"Loaded outline with {len(outline)} chapters")
    
    # Load target words
    target_words = load_target_words(facts_path, inference_path)
    print(f"Loaded {len(target_words)} target words to avoid")
    
    # Generate chapters
    def write_chapter(chapter_info):
        chapter_title = chapter_info['chapter_title']
        print(f"Writing chapter: {chapter_title}...")
        
        chapter_outline_text = f"### Chapter Title\n\n{chapter_info['chapter_title']}\n"
        chapter_outline_text += f"## Chapter Description\n\n{chapter_info['description']}\n"
        chapter_outline_text += "## Subtopics\n"
        for sec in chapter_info['subtopics']:
            chapter_outline_text += f"- {sec}\n"
        
        forbidden_words_list = sorted(list(target_words))  # Limit for prompt size
        
        prompt_chapter = {
            'system': f"""### Instructions
You will be given a chapter title, description, and subtopics and, based on those topics, your job is to write a detailed, cohesive textbook chapter addressed to a college student who is learning this material for the first time.

The chapter should be comprehensive and suitable for someone learning this material to understand research papers in the field. Don't just briefly describe the subtopics, but rather elaborate on the concepts at full length and explain them with a focus on intuition. Spell everything out clearly so there is no ambiguity. Dedicate multiple paragraphs to each subtopic but be articulate and concise when appropriate. Write in full prose, rather than bullet points. Most importantly, please make sure that your chapter is grounded in the paper; do not provide any information or details that is not from the paper.

### CRITICAL CONSTRAINT
You MUST NOT use any of the following words in your chapter (case-insensitive): {', '.join(forbidden_words_list)}

If you need to reference these concepts, you must use alternative phrasing, synonyms, or circumlocution. Be creative in your wording while maintaining accuracy.

### WRITING STYLE
Since you cannot use certain technical terms, focus on explaining concepts very **conceptually**. Use high-level descriptions, analogies, and intuitive explanations. Describe the "what" and "why" rather than just naming things. Think of it as explaining to someone who needs to understand the core ideas without getting bogged down in specific terminology.

Start with the chapter title in the first line. Separate each subtopic with a section header "#". Also, please write all mathematical notation in LaTeX only e.g. "$x^2$" or "$\pi$". Do not use unicode mathematical characters e.g. "π". Again, PLEASE write all math in LaTeX.""",
            'user': f"""### Research Paper
{paper_content}

### Chapter Outline to Write
{chapter_outline_text}"""
        }
        
        chapter_content = utils.query_llm(
            prompt_chapter,
            model='gpt-5-mini',
            reasoning_effort="medium",
            system_prompt_included=True,
            max_tokens=4000
        )
        return chapter_content
    
    with ThreadPoolExecutor() as executor:
        chapter_contents = list(tqdm(executor.map(write_chapter, outline), total=len(outline), desc="Writing non-probe chapters"))
    
    # Save individual chapters
    for idx, content in enumerate(chapter_contents):
        chapter_path = os.path.join(output_dir, f'chapter_{idx+1}.txt')
        with open(chapter_path, 'w') as f:
            f.write(content)
    
    # Save combined textbook
    full_textbook_content = "\n\n".join(chapter_contents)
    full_textbook = f"Title: Textbook of {paper_title}\n\n{full_textbook_content}"
    
    textbook_path = os.path.join(output_dir, 'textbook.txt')
    with open(textbook_path, 'w') as f:
        f.write(full_textbook)
    
    print(f"Saved non-probe textbook to: {textbook_path}")

def split_into_paragraphs(text):
    """Split text into paragraphs."""
    paragraphs = re.split(r'\n\n+', text)
    return [p.strip() for p in paragraphs if p.strip()]

def make_paragraph_misleading(paragraph, protected_words):
    """Edit a paragraph to make it misleading with minimal changes."""
    
    protected_words_list = sorted(list(protected_words))[:100]  # Limit for prompt
    
    prompt = {
        'system': f"""### Task
You will be given a paragraph from a textbook. Your task is to make **minimal edits** to make the paragraph incorrect and misleading. Do this for as many sentences as possible.

### Guidelines for Edits
Make small, strategic changes such as:
- Insert "not" or "never" to negate statements (e.g., "is effective" → "is not effective")
- Change positive to negative (e.g., "performs well" → "performs poorly", "increases" → "decreases")
- Change operators (e.g., "+" → "-", "*" → "/", ">" → "<")
- Change frequency words (e.g., "often" → "never", "always" → "rarely")
- Swap contradictory terms (e.g., "better" → "worse", "higher" → "lower")

### CRITICAL CONSTRAINT
DO NOT change or remove these protected words (they are key technical terms): {', '.join(protected_words_list)}

You can change words around them, but keep these words intact.

### Output Format
Return ONLY the modified paragraph text. Do not include any explanations, notes, or meta-commentary. Just output the edited paragraph directly.""",
        'user': f"""Original paragraph:

{paragraph}

Modified paragraph:"""
    }
    
    try:
        modified_para = utils.query_llm(
            prompt,
            model='gpt-5-mini',
            reasoning_effort="low",
            system_prompt_included=True,
            max_tokens=2000
        )
        return modified_para
    except Exception as e:
        print(f"Error modifying paragraph: {e}")
        return paragraph

def generate_misleading_textbook(paper_name, base_dir, probes_dir):
    """Generate misleading/wrong textbook by editing existing chapters."""
    print(f"\n{'='*60}")
    print(f"Generating misleading textbook for: {paper_name}")
    print(f"{'='*60}")
    
    paper_path = os.path.join(base_dir, 'cleaned', f'{paper_name}.tex')
    textbooks_dir = os.path.join(base_dir, 'explanations', paper_name, 'textbooks')
    facts_path = os.path.join(probes_dir, 'facts', paper_name, 'probes_v9.csv')
    inference_path = os.path.join(probes_dir, 'inference', paper_name, 'probes_v6.csv')
    output_dir = os.path.join(base_dir, 'explanations', paper_name, 'misleading_textbook')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if textbooks directory exists
    if not os.path.exists(textbooks_dir):
        print(f"Textbooks directory not found: {textbooks_dir}")
        return
    
    paper_title = extract_paper_title(paper_path)
    
    # Load protected words (target probe words)
    protected_words = load_target_words(facts_path, inference_path)
    print(f"Loaded {len(protected_words)} protected words (won't be changed)")
    
    # Get all chapter files
    chapter_files = sorted([f for f in os.listdir(textbooks_dir) if f.startswith('chapter_') and f.endswith('.txt')])
    print(f"Found {len(chapter_files)} chapters to process")
    
    all_modified_chapters = []
    
    for chapter_file in chapter_files:
        chapter_path = os.path.join(textbooks_dir, chapter_file)
        print(f"\nProcessing {chapter_file}...")
        
        # Load chapter
        with open(chapter_path, 'r') as f:
            chapter_content = f.read()
        
        # Split into paragraphs
        paragraphs = split_into_paragraphs(chapter_content)
        print(f"  Split into {len(paragraphs)} paragraphs")
        
        # Process paragraphs in parallel
        with ThreadPoolExecutor() as executor:
            modified_paragraphs = list(tqdm(
                executor.map(lambda p: make_paragraph_misleading(p, protected_words), paragraphs),
                total=len(paragraphs),
                desc=f"  Editing {chapter_file}"
            ))
        
        # Reconstruct chapter
        modified_chapter = '\n\n'.join(modified_paragraphs)
        all_modified_chapters.append(modified_chapter)
        
        # Save individual chapter
        output_chapter_path = os.path.join(output_dir, chapter_file)
        with open(output_chapter_path, 'w') as f:
            f.write(modified_chapter)
        print(f"  Saved to {output_chapter_path}")
    
    # Save combined textbook
    full_textbook_content = "\n\n".join(all_modified_chapters)
    full_textbook = f"Title: Textbook of {paper_title}\n\n{full_textbook_content}"
    
    textbook_path = os.path.join(output_dir, 'textbook.txt')
    with open(textbook_path, 'w') as f:
        f.write(full_textbook)
    
    print(f"\nSaved misleading textbook to: {textbook_path}")

def main():
    base_dir = '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/data/arxiv'
    probes_dir = '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/data/probes'
    
    # Get all domains from explanations directory
    explanations_dir = os.path.join(base_dir, 'explanations')
    domains = [d for d in os.listdir(explanations_dir) 
               if os.path.isdir(os.path.join(explanations_dir, d))]
    
    print(f"Found {len(domains)} domains to process")
    
    for domain in domains:
        try:
            generate_non_probe_textbook(domain, base_dir, probes_dir)
        except Exception as e:
            print(f"Error generating non-probe textbook for {domain}: {e}")
        
        try:
            generate_misleading_textbook(domain, base_dir, probes_dir)
        except Exception as e:
            print(f"Error generating misleading textbook for {domain}: {e}")

if __name__ == "__main__":
    main()

