import os
import json
import sys
import re
import argparse
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import statistics

# Add the project root to the path to allow importing utils
sys.path.append('../../')
import utils.utils as utils
from importlib import reload
reload(utils)

# Re-using paragraph processing logic from paraphrase script
def split_and_merge_paragraphs(text):
    """Split text into paragraphs and merge short ones, returning processed paragraphs."""
    paragraphs = text.split('\n\n')
    
    # Combine short paragraphs until none are shorter than 150 words
    while True:
        min_length = float('inf')
        shortest_idx = -1
        
        for i, p in enumerate(paragraphs):
            word_count = len(p.split())
            if word_count > 0 and word_count < 200 and word_count < min_length:
                min_length = word_count
                shortest_idx = i
        
        if shortest_idx == -1:  # No paragraphs shorter than 200 words
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
            # This can happen if a short paragraph is at the beginning or end
            # and has no adjacent paragraphs to merge with that are also short.
            # To avoid infinite loop, we break.
            break
        
        # Merge paragraphs
        if merge_with < shortest_idx:
            paragraphs[merge_with] = paragraphs[merge_with] + '\n\n' + paragraphs[shortest_idx]
            paragraphs.pop(shortest_idx)
        else:
            paragraphs[shortest_idx] = paragraphs[shortest_idx] + '\n\n' + paragraphs[merge_with]
            paragraphs.pop(merge_with)
            
    return paragraphs


def refine_content(paper_content, generated_content, content_type):
    """Refines generated content for accuracy and LaTeX formatting."""
    print(f"Refining {content_type} for accuracy...")
    
    # 1. Accuracy Check
    prompt_accuracy = {
        'system': """You will be given a research paper and a generated text based on it. Your task is to check if the generated text is factually consistent with the research paper. 
        Identify any statements in the generated text that contradict, misrepresent, or are not supported by the paper. 
        For each identified inaccuracy, provide the inaccurate statement, the corresponding text from the paper (if available), and an explanation of the error.
        If there are no inaccuracies, return an empty list.
        
        Output as a JSON object with a single key "inaccuracies" which is a list of objects, each with "inaccurate_statement", "paper_quote", and "explanation".
        Example of an inaccuracy object:
        {
            "inaccurate_statement": "The paper states that DPO is a form of supervised learning.",
            "paper_quote": "DPO is a simple reinforcement learning objective...",
            "explanation": "The generated text incorrectly classifies DPO as supervised learning, but the paper explicitly states it's a reinforcement learning objective."
        }
        """,
        'user': f"### Research Paper\n{paper_content}\n\n### Generated Text\n{generated_content}"
    }
    
    accuracy_check_str = utils.query_llm(
        prompt_accuracy,
        model='gpt-5-mini',
        system_prompt_included=True,
        return_json=True,
        max_tokens=4000
    )
    
    accuracy_data = json.loads(accuracy_check_str)
    
    # 2. Correction Step if inaccuracies are found
    if accuracy_data['inaccuracies']:
        print(f"Found {len(accuracy_data['inaccuracies'])} inaccuracies. Correcting...")
        prompt_correction = {
            'system': f"""You will be given a generated {content_type}, a list of inaccuracies found in it with respect to a research paper, and the research paper itself. 
            Your task is to revise the generated text to correct these inaccuracies, ensuring the new version is factually consistent with the paper, while maintaining the original style and tone.
            Output only the corrected {content_type}.""",
            'user': f"### Original Generated Text\n{generated_content}\n\n### Inaccuracies to Correct\n{json.dumps(accuracy_data['inaccuracies'], indent=2)}\n\n### Research Paper\n{paper_content}"
        }
        
        corrected_content = utils.query_llm(
            prompt_correction,
            model='gpt-5',
            system_prompt_included=True,
            max_tokens=8000
        )
        generated_content = corrected_content
    else:
        print("No inaccuracies found.")

    # 3. LaTeX Formatting Check
    print(f"Refining {content_type} for LaTeX formatting...")
    prompt_refine_latex = {
        'system': """You will be given a text. Your only task is to correct any mathematical notation inside it to be valid LaTeX. You must not change any other part of the text.
- Convert unicode math characters like 'π' to their LaTeX equivalent '$\\pi$'.
- Ensure all mathematical expressions are enclosed in '$...$' for inline math or '$$...$$' for display math.
- Return the full, corrected text.""",
        'user': generated_content
    }
    
    refined_content = utils.query_llm(
        prompt_refine_latex,
        model='gpt-5-mini',
        system_prompt_included=True,
        max_tokens=8000
    )
    
    return refined_content

def generate_summaries(paper_name, paper_content):
    """Generates two types of summaries for a paper."""
    OUTPUT_DIR = f"../../data/arxiv/explanations/{paper_name}/"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 1. Full Paper Summary ---
    print(f"Generating full summary for {paper_name}...")
    prompt_full_summary = {
        'system': "You will be given a research paper. Your task is to write a comprehensive summary of the paper. The summary should cover everything thoroughly while focusing on the key ideas. It should be written clearly and accurately.",
        'user': f"### Research Paper\n{paper_content}"
    }
    
    full_summary = utils.query_llm(
        prompt_full_summary,
        model='gpt-5',
        system_prompt_included=True,
        max_tokens=5000
    )

    refined_summary = refine_content(paper_content, full_summary, "summary")
    
    titled_summary = f"\\title{{Summary of the paper: {paper_name}}}\n\n{refined_summary}"
    summary_full_path = os.path.join(OUTPUT_DIR, "summary_full.txt")
    with open(summary_full_path, 'w') as f:
        f.write(titled_summary)
    print(f"Saved full summary to {summary_full_path}")

    # --- 2. Section-by-Section Summary ---
    print(f"Generating section-by-section summary for {paper_name}...")
    sections = re.split(r'\\section\{.*?\}', paper_content)
    section_headers = re.findall(r'\\section\{(.*?)\}', paper_content)
    
    # Handle introduction before first section
    if len(sections) > len(section_headers):
        intro_content = sections[0]
        sections = sections[1:]
    else:
        intro_content = ""

    def summarize_section(content, header):
        if len(content.split()) < 50: # Skip very short sections
            return ""
        print(f"Summarizing section: {header}...")
        prompt_section_summary = {
            'system': "You will be given the content of a single section from a research paper. Your task is to summarize this section accurately. Focus only on the content provided. Please make sure to write any mathematical notation in LaTeX only e.g. '$x^2$' or '$\\pi$'. Do not use unicode mathematical characters e.g. 'π'.",
            'user': f"### Paper Title: {paper_name}\n\n### Section: {header}\n\n{content}"
        }
        summary = utils.query_llm(
            prompt_section_summary,
            model='gpt-5-mini',
            system_prompt_included=True,
            max_tokens=3000
        )
        return f"## Summary of Section: {header}\n{summary}"

    section_summaries = []
    if intro_content.strip():
        intro_summary = summarize_section(intro_content, "Introduction")
        if intro_summary:
            section_summaries.append(intro_summary)
            
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(summarize_section, sections, section_headers), total=len(sections), desc="Summarizing sections"))
        section_summaries.extend([r for r in results if r])

    full_section_summary = "\n\n".join(section_summaries)
    
    titled_section_summary = f"\\title{{Section-by-Section Summary of the paper: {paper_name}}}\n\n{full_section_summary}"
    summary_sections_path = os.path.join(OUTPUT_DIR, "summary_sections.txt")
    with open(summary_sections_path, 'w') as f:
        f.write(titled_section_summary)
    print(f"Saved section-by-section summary to {summary_sections_path}")


def generate_contextualized_facts(paper_name, paper_content):
    """Generates contextualized facts from the paper."""
    OUTPUT_DIR = f"../../data/arxiv/explanations/{paper_name}/"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Generating contextualized facts for {paper_name}...")
    paragraphs = split_and_merge_paragraphs(paper_content)
    
    def extract_facts_from_paragraph(paragraph):
        if not paragraph.strip():
            return []
        prompt = {
            'system': """You will be given a paragraph from a research paper. Your task is to extract key facts from this paragraph. Each fact should be a self-contained, contextualized statement.
            - Please make sure to write any mathematical notation in LaTeX only e.g. '$x^2$' or '$\\pi$'. Do not use unicode mathematical characters e.g. 'π'.
            - Output a JSON object with a single key "facts", which is a list of strings.
            - Extract only the most important and atomic facts.
            - Each fact should be a complete sentence.
            
            Example:
            {
                "facts": [
                    "The Transformer architecture relies entirely on self-attention, avoiding the use of recurrence.",
                    "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions."
                ]
            }
            """,
            'user': f"### Paragraph\n{paragraph}"
        }
        
        try:
            response_str = utils.query_llm(
                prompt,
                model='gpt-5-mini',
                system_prompt_included=True,
                return_json=True,
                max_tokens=2000
            )
            return json.loads(response_str).get('facts', [])
        except (json.JSONDecodeError, AttributeError):
            return []

    all_facts = []
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(extract_facts_from_paragraph, paragraphs), total=len(paragraphs), desc="Extracting facts"))
        for fact_list in results:
            all_facts.extend(fact_list)
            
    facts_text = "\n".join(all_facts)
    titled_facts = f"\\title{{Contextualized Facts for the paper: {paper_name}}}\n\n{facts_text}"
    facts_path = os.path.join(OUTPUT_DIR, "contextualized_facts.txt")
    with open(facts_path, 'w') as f:
        f.write(titled_facts)
    print(f"Saved {len(all_facts)} contextualized facts to {facts_path}")

def generate_podcast_monologue(paper_name, paper_content):
    """Generates a podcast monologue about the paper."""
    OUTPUT_DIR = f"../../data/arxiv/explanations/{paper_name}/"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Generating podcast monologue for {paper_name}...")
    prompt = {
        'system': """You are a podcast host for a show about cutting-edge AI research. You will be given a research paper. Your task is to write a script for a 5-10 minute monologue that explains this paper to a general audience in an engaging and intuitive way.
- Break down the core concepts and contributions.
- Use analogies and simple explanations for complex ideas.
- The tone should be enthusiastic, clear, and educational.""",
        'user': f"### Research Paper\n{paper_content}"
    }
    
    monologue = utils.query_llm(
        prompt,
        model='gpt-5',
        system_prompt_included=True,
        max_tokens=4000
    )
    
    refined_monologue = refine_content(paper_content, monologue, "podcast monologue")
    
    titled_monologue = f"\\title{{Podcast Monologue for the paper: {paper_name}}}\n\n{refined_monologue}"
    podcast_path = os.path.join(OUTPUT_DIR, "podcast_monologue.txt")
    with open(podcast_path, 'w') as f:
        f.write(titled_monologue)
    print(f"Saved podcast monologue to {podcast_path}")


def generate_socratic_dialogue(paper_name, paper_content):
    """Generates a Socratic dialogue about the paper."""
    OUTPUT_DIR = f"../../data/arxiv/explanations/{paper_name}/"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Generating Socratic dialogue for {paper_name}...")
    prompt = {
        'system': """You are a skilled educator. Your task is to write a Socratic dialogue between a 'Tutor' and a 'Student' to explore the key ideas of a research paper.
- The Tutor should guide the Student's understanding by asking insightful questions.
- The Student should start with some basic knowledge but also have some confusion.
- The dialogue should naturally progress from simpler to more complex concepts in the paper.
- Cover the main aspects of the paper.
- The dialogue should feel like a real, interactive learning session.

Format the dialogue as:
Tutor: [Tutor's question or statement]
Student: [Student's response]""",
        'user': f"### Research Paper\n{paper_content}"
    }
    
    dialogue = utils.query_llm(
        prompt,
        model='gpt-5',
        system_prompt_included=True,
        max_tokens=4000
    )

    refined_dialogue = refine_content(paper_content, dialogue, "Socratic dialogue")
    
    titled_dialogue = f"\\title{{Socratic Dialogue for the paper: {paper_name}}}\n\n{refined_dialogue}"
    dialogue_path = os.path.join(OUTPUT_DIR, "socratic_dialogue.txt")
    with open(dialogue_path, 'w') as f:
        f.write(titled_dialogue)
    print(f"Saved Socratic dialogue to {dialogue_path}")

def process_paper(paper_name):
    """Run the full pipeline for a single paper."""
    print(f"--- Starting processing for {paper_name} ---")
    PAPER_FILE_PATH = f'../../data/arxiv/cleaned/{paper_name}.tex'
    
    if not os.path.exists(PAPER_FILE_PATH):
        print(f"Error: Paper file not found at {PAPER_FILE_PATH}. Skipping.")
        return

    with open(PAPER_FILE_PATH, 'r') as f:
        paper_content = f.read()

    generate_summaries(paper_name, paper_content)
    generate_contextualized_facts(paper_name, paper_content)
    generate_podcast_monologue(paper_name, paper_content)
    generate_socratic_dialogue(paper_name, paper_content)
    
    print(f"--- Finished processing for {paper_name} ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate diverse views of academic papers.")
    parser.add_argument(
        '--papers', 
        nargs='+', 
        help='Optional list of paper filenames to process (e.g., DPO, QLoRA). If not provided, all papers in the cleaned directory will be processed.'
    )
    args = parser.parse_args()

    if args.papers:
        paper_names = args.papers
    else:
        input_dir = "../../data/arxiv/cleaned/"
        paper_names = [os.path.splitext(f)[0] for f in os.listdir(input_dir) if f.endswith('.tex')]
        
    for paper_name in paper_names:
        process_paper(paper_name)
