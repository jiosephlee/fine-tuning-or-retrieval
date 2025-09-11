import os
import json
import sys
import re
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
# Add the project root to the path to allow importing utils
sys.path.append('../../')
import utils.utils as utils
from importlib import reload
reload(utils)

def generate_prior_knowledge(paper_name):
    """Process a single paper to generate prior knowledge chapters."""
    print(f"Processing {paper_name} for prior knowledge generation...")
    
    PAPER_FILE_PATH = f'../../data/arxiv/cleaned/{paper_name}.tex'
    OUTPUT_DIR = f"../../data/arxiv/prior_knowledge/{paper_name}/"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Reading paper from: {PAPER_FILE_PATH}")
    with open(PAPER_FILE_PATH, 'r') as f:
        paper_content = f.read()

    # --- 1. Generate list of chapters ---
    print("Generating list of prerequisite chapters...")
    prompt_chapters = {
        'system': """### Instructions
You are an expert curriculum designer. Based on the provided research paper, create a list of textbook chapters that would provide all the necessary prior knowledge to understand this paper. The chapters should not contain the novel ideas presented in the paper itself, but rather the foundational concepts upon which the paper is built.

For each chapter, provide:
- A `title`.
- A general `description` of what the chapter covers.
- A list of `subtopics` that should be included.

### Output Format
Provide the output as a JSON object with a single key "chapters", which is a list of chapter dictionaries.
Example:
{
  "chapters": [
    {
      "title": "Chapter 1: Introduction to Probability Theory",
      "description": "This chapter covers the basics of probability...",
      "subtopics": ["Random Variables", "Probability Distributions", "Bayes' Theorem"]
    }
  ]
}""",
        'user': f"### Research Paper\n{paper_content}"
    }

    response_chapters_str = utils.query_llm(
        prompt_chapters, 
        model='gpt-5-mini', 
        system_prompt_included=True, 
        return_json=True, 
        max_tokens=5000
    )
    
    # Save the generated chapters outline
    chapters_log_path = os.path.join(OUTPUT_DIR, "chapters_outline.json")
    with open(chapters_log_path, 'w') as f:
        f.write(response_chapters_str)
    print(f"Saved chapters outline to {chapters_log_path}")

    chapters_data = json.loads(response_chapters_str)
    chapters_list = chapters_data['chapters']
    print(f"Parsed outline with {len(chapters_list)} chapters.")

    # --- 2. Generate content for each chapter ---
    if chapters_list:
        
        def generate_chapter(chapter_info, chapter_index):
            """Generate content for a single chapter"""
            chapter_title = chapter_info.get('title', f"Chapter {chapter_index+1}")
            chapter_description = chapter_info.get('description', '')
            chapter_subtopics = chapter_info.get('subtopics', [])
            
            print(f"Generating content for: {chapter_title}")

            subtopics_str = "\n".join([f"- {s}" for s in chapter_subtopics])

            prompt_content = {
                'system': """### Instructions
You will be given a chapter title, description, and subtopics and, based on those topics, your job is to write a detailed, cohesive textbook chapter addressed to a college student who is learning this material for the first time. 

The chapter should be comprehensive and suitable for someone learning this material to understand research papers in the field. Begin with an introduction to the chapter, then cover each subtopic in turn. Don't just briefly describe the subtopics, but rather elaborate on the concepts at full length and explain them with a focus on intuition. Spell everything out clearly so there is no ambiguity. Dedicate multiple paragraphs to each subtopic. Write in full prose, rather than bullet points. 

Separate each subtopic with a section header "#".

Also, please write all mathematical notation in LaTeX only e.g. "$x^2$" or "$\pi$". Do not use unicode mathematical characters e.g. "π".""",
                'user': f"""### Chapter Title
{chapter_title}

### Chapter Description
{chapter_description}

### Subtopics to Cover
{subtopics_str}"""
            }

            chapter_content = utils.query_llm(
                prompt_content, 
                model='gpt-5-mini', 
                system_prompt_included=True, 
                reasoning_effort = "low",
                max_tokens=10000
            )

            # Truncate the last sentence if it's incomplete
            if chapter_content and not chapter_content.strip().endswith(('.', '!', '?', '"', '`')):
                last_period_index = chapter_content.rfind('.')
                if last_period_index != -1:
                    chapter_content = chapter_content[:last_period_index+1]
            
            return chapter_content, chapter_index

        # Generate all chapters in parallel
        with ThreadPoolExecutor(max_workers=min(len(chapters_list), 5)) as executor:
            futures = [executor.submit(generate_chapter, chapter_info, i) 
                      for i, chapter_info in enumerate(chapters_list)]
            
            # Wait for all chapters to complete and collect content
            all_chapter_contents = []
            for future in futures:
                try:
                    chapter_content, chapter_index = future.result()
                    all_chapter_contents.append((chapter_content, chapter_index))
                except Exception as e:
                    print(f"Error generating chapter: {e}")

        # Sort chapters by index to maintain order
        all_chapter_contents.sort(key=lambda x: x[1])
        
        # Save each chapter separately and collect for textbook
        textbook_content_parts = []
        for chapter_content, chapter_index in all_chapter_contents:
            # Save individual chapter
            chapter_filename = f"chapter_{chapter_index+1}.txt"
            chapter_path = os.path.join(OUTPUT_DIR, chapter_filename)
            with open(chapter_path, 'w', encoding='utf-8') as f:
                f.write(chapter_content)
            print(f"Saved chapter {chapter_index+1} to: {chapter_path}")
            
            # Collect for full textbook
            textbook_content_parts.append(chapter_content)
        
        # Save complete textbook
        textbook_content = "\n\n".join(textbook_content_parts)
        output_path = os.path.join(OUTPUT_DIR, "textbook.txt")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(textbook_content)
        
        print(f"Saved complete textbook to: {output_path}")
        print(f"Generated textbook with {len(chapters_list)} chapters.")
    else:
        print("No chapters were generated. Exiting.")

def process_papers():
    input_dir = "../../data/arxiv/cleaned/"
    
    # Get list of files in cleaned directory
    files = [f for f in os.listdir(input_dir) if f.endswith('.tex') and f == 'DPO.tex']
    
    for filename in files:
        # Extract paper name without extension
        paper_name = os.path.splitext(filename)[0]
        generate_prior_knowledge(paper_name)
        
if __name__ == "__main__":
    process_papers()
