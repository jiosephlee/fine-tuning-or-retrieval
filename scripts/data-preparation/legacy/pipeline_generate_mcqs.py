import os
import json
import sys
import re
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
# Add the project root to the path to allow importing utils
sys.path.append('../../')
import utils.utils as utils

def generate_chapter_mcqs(chapter_info, chapter_index, output_dir):
    """Generate content for a single chapter"""
    chapter_title = chapter_info.get('title', f"Chapter {chapter_index+1}")
    chapter_description = chapter_info.get('description', '')
    chapter_subtopics = chapter_info.get('subtopics', [])
    
    print(f"Generating questions for: {chapter_title}")

    subtopics_str = "\n".join([f"- {s}" for s in chapter_subtopics])

    prompt_content = {
        'system': """### Instructions
You will be given a chapter title, description, and subtopics and, based on those topics, your job is to write 20 multiple choice questions for college-level exam. 

The questions should be calibrated for someone learning this material to understand research papers in the field. Focus on high-level conceptual understanding rather than mathematical ability or memorizing equations. Spell everything out clearly so there is no ambiguity. 

Respond with a JSON list of questions in the following format: 
[
    {
        "question": <question text>,
        "choice_A": <candidate answer>,
        "choice_B": <candidate answer>,
        "choice_C": <candidate answer>,
        "choice_D": <candidate answer>,
        "correct_answer": <A, B, C, or D>
    }, 
    ...
]

Vary the location of the correct answer so that it is evenly distributed across A, B, C, and D. 

If you need to write any mathematical notation, use LaTeX (e.g. "$x^2$" or "$\pi$"), DO NOT use unicode mathematical characters (e.g. "π".)
""",
        'user': f"""### Chapter Title
{chapter_title}

### Chapter Description
{chapter_description}

### Subtopics to Cover
{subtopics_str}"""
    }

    chapter_questions_str = utils.query_llm(
        prompt_content, 
        model='gpt-5', 
        system_prompt_included=True, 
        reasoning_effort = "low",
        return_json=True, 
        max_tokens=10000
    )

    response_json = json.loads(chapter_questions_str)
    saved_path = output_dir + f'/chapter_{chapter_index}.json'
    with open(saved_path, 'w') as f:
        f.write(json.dumps(response_json))

    return saved_path

if __name__ == '__main__':
    papers = ['1_58', 'BOFT', 'DPO', 'GRPO', 'OFT', 'QLoRA']
    for paper in papers: 
        INPUT_FILE = f'../../data/arxiv/prior_knowledge/{paper}/chapters_outline.json'
        OUTPUT_DIR = f'../../data/arxiv/prior_knowledge_mcq/{paper}'

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"Reading chapter list from: {INPUT_FILE}")

        with open(INPUT_FILE) as f:
            response_json = json.loads(f.read())
            chapters_list = response_json.get('chapters', [])
    
        with ThreadPoolExecutor(max_workers=min(len(chapters_list), 5)) as executor:
            futures = [executor.submit(generate_chapter_mcqs, chapter_info, i, OUTPUT_DIR) 
                      for i, chapter_info in enumerate(chapters_list)]
            
            # Wait for all chapters to complete
            all_saved_paths = []
            for future in futures:
                try:
                    result = future.result()
                    all_saved_paths.append(result)
                except Exception as e:
                    print(f"Error generating chapter: {e}")
    
        print(f"\nAll chapters generated and saved successfully for {paper}. Total files: {len(all_saved_paths)}")
        