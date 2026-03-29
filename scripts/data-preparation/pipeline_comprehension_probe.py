import os
import sys
import json
import re
import pandas as pd
from typing import List, Dict, Tuple, Any
import concurrent.futures
from tqdm import tqdm
import argparse

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import utils.utils as utils
from utils.pipeline import save_df_for_debugging, save_debug_file, is_text_in_document, process_papers
from utils.prompts.pipeline import FACT_PROBE_CLOZE_PROMPT_SYSTEM

def parse_paper_structure(text):
    """Parse paper into sections, subsections and paragraphs with metadata."""
    sections = []
    
    # Split by sections first
    section_pattern = r'\\section\{([^}]+)\}'
    section_splits = re.split(section_pattern, text)
    
    current_section = "Title/Abstract"
    current_section_content = ""
    
    for i in range(len(section_splits)):
        if i == 0:
            # Content before first section
            content = section_splits[i]
            current_section_content = content
        elif i % 2 == 1:
            # This is a section title
            current_section = section_splits[i]
            continue
        else:
            # This is section content
            content = section_splits[i]
            current_section_content = content
        
        # Now split by subsections within this section
        subsection_pattern = r'\\subsection\{([^}]+)\}'
        subsection_splits = re.split(subsection_pattern, content)
        
        current_subsection = "No Subsection"
        current_subsection_content = ""
        
        for j in range(len(subsection_splits)):
            if j == 0:
                # Content before first subsection
                subsection_content = subsection_splits[j]
                current_subsection_content = subsection_content
            elif j % 2 == 1:
                # This is a subsection title
                current_subsection = subsection_splits[j]
                continue
            else:
                # This is subsection content
                subsection_content = subsection_splits[j]
                current_subsection_content = subsection_content
            
            # Split into paragraphs
            paragraphs = [p.strip() for p in subsection_content.split('\n\n') if p.strip()]
            
            for paragraph in paragraphs:
                sections.append({
                    'section': current_section,
                    'subsection': current_subsection,
                    'paragraph': paragraph,
                    'section_text': current_section_content,
                    'subsection_text': current_subsection_content
                })
    
    return pd.DataFrame(sections)

def generate_questions(text: str, model: str = 'gpt-5', reasoning_effort: str = 'low', focus_on_repeated_ideas: bool = False) -> List[Dict[str, Any]]:
    """Generates comprehension questions from the text using an LLM."""
    if focus_on_repeated_ideas:
        system_prompt = """You have been given an academic paper. Your task is to test the reader's understanding of the core ideas and themes that are repeatedly emphasized throughout the text. Focus on the main concepts, recurring arguments, and central contributions rather than small details or one-off mentions.

Create questions that integrate and synthesize information about these repeatedly emphasized ideas. The questions should require understanding of how these ideas develop, connect, and are used throughout the paper. Your question must not be obvious from a single sentence and truly require understanding of how ideas are developed across the paper. Lastly, the answer to the question must be a coherent phrase, from 1 to 5 words long.

For each question, show me the sentences in the text that you're pulling from to answer the question. The question should be non-obvious from these sentences and require composing information from all of them and additional reasoning and synthesis to answer the question. The question should be testing *new knowledge*, building upon the knowledge *in the paper* and requiring a generalizable, deep understanding of the knowledge.

Provide the output in JSON format, as a dictionary with a single key "qa_items" which is a list of dictionaries with the following keys:
- "question": (string) 
- "answer": (string)
- "text_quotes": list of sentences from the text that you're pulling from to answer the question.
"""
    else:
        system_prompt = """You have been given a section of an academic text. Your tasks is to test the reader's understanding of the text. However, you should not test anything that can be recalled from reading a single sentence. Create questions that integrate, connect, and synthesize information across several sentences and aim at measuring a deeper understanding. Your question must not be obvious from a single sentence already in the paper, and truly require several sentences to synthesize the answer. Lastly, the answer to the question must be a coherent phrase, from 1 to 5 words long.

For each question, show me the sentences in the text that you're pulling from to answer the question. The question should be non-obvious from these sentences and require composing information from all of them to answer the question.

Provide the output in JSON format, as a dictionary with a single key "qa_items" which is a list of dictionaries with the following keys:
- "question": (string) 
- "answer": (string)
- "text_quotes": list of sentences from the text that you're pulling from to answer the question.
"""
    prompt = {'system': system_prompt, 'user': f"### Text\n{text}"}
    response_json = utils.query_llm(prompt, model=model, reasoning_effort=reasoning_effort, system_prompt_included=True, return_json=True, max_tokens=4000)
    
    if isinstance(response_json, str):
        try:
            response_json = json.loads(response_json)
        except json.JSONDecodeError:
            print("Failed to parse JSON response from LLM.")
            return []

    questions = response_json.get('qa_items', [])
    if not isinstance(questions, list):
        print("Unexpected response format from LLM: 'qa_items' is not a list.")
        return []

    # Validate that all supporting sentences are in the original paper text
    parsed_questions = []
    for q in questions:
        sentences = q.get('text_quotes', [])
        if all(is_text_in_document(s, text, threshold=0.5) for s in sentences):
            parsed_questions.append(q)

    if len(parsed_questions) != len(questions):
        print(f"DROPPED {len(questions) - len(parsed_questions)} questions with invalid text sentences.")
    
    return parsed_questions

def is_question_obvious(question: Dict[str, Any]) -> bool:
    """Checks if a question is obvious from the supporting sentences."""
    system_prompt = """You will be given a question, its answer, and a list of supporting sentences from a text. Your task is to determine if the answer to the question is trivially and explicitly stated in any single one of the supporting sentences. 

The question is considered 'obvious' if a reader could find the answer directly in one of the sentences without needing to synthesize or deeply understand the supporting information. Please err on the side of lenient and consider the question obvious if it's clearly obvious even with someone who has less background knowledge.

Respond in JSON with a single boolean key 'is_obvious'."""
    sentences_str = '\n- '.join(question['text_quotes'])
    user_prompt = f"Question: {question['question']}\nAnswer: {question['answer']}\n\nSupporting Sentences:\n- {sentences_str}"
    prompt = {'system': system_prompt, 'user': user_prompt}

    response_str = utils.query_llm(prompt, model='gpt-4.1-nano', system_prompt_included=True, return_json=True, max_tokens=100)
    try:
        return json.loads(response_str).get('is_obvious', True)
    except (json.JSONDecodeError, TypeError):
        print(f"Warning: Could not parse JSON for obviousness check. Defaulting to obvious. Response: {response_str}")
        return True

def filter_obvious_questions(questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filters a list of questions, removing ones that are obvious."""
    filtered_questions = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        future_to_question = {executor.submit(is_question_obvious, q): q for q in questions}
        for future in tqdm(concurrent.futures.as_completed(future_to_question), total=len(questions), desc="Filtering obvious questions"):
            question = future_to_question[future]
            try:
                if not future.result():
                    filtered_questions.append(question)
            except Exception as exc:
                print(f'"{question.get("question", "A question")}" generated an exception: {exc}')
    return filtered_questions

def convert_to_cloze(question: Dict[str, Any]) -> Tuple[str, str] | None:
    """Converts a question-answer pair to a cloze-style statement."""
    user_prompt = f"### Question and Answer\n{json.dumps({'question': question['question'], 'answer': question['answer']})}\n"
    cloze_prompt = {
        'system': FACT_PROBE_CLOZE_PROMPT_SYSTEM,
        'user': user_prompt
    }
    response = utils.query_llm(cloze_prompt, model='gpt-5', reasoning_effort='low', system_prompt_included=True, return_json=True, max_tokens=1000)
    try:
        data = json.loads(response) if isinstance(response, str) else response
        answer = data.get('answer')
        statement = data.get('statement')
        if answer is not None and statement is not None:
            return (answer, statement)
    except (json.JSONDecodeError, AttributeError):
        print("Failed to parse JSON response for cloze conversion.")
    return None

def quality_control_cloze(cloze_pair: Tuple[str, str], title: str) -> Tuple[str, str] | None:
    """Performs quality control on a cloze statement for LaTeX formatting."""
    quality_control_prompt = {
        'system': r"""Your task is to review and refine a statement. You will be given an '(answer, statement)' pair.
### Quality Control Checklist
1. LaTeX Formatting: All mathematical expressions **MUST** be in valid LaTeX. Do *NOT* use unicode math characters. Correct any formatting errors. In all your adjustments, change the statement as minimally as necessary. However, standalone numbers should be written without $.
2.  Rewrite the statement so that it starts with one of the following templates. 
    - "In the paper '...'"
    - "According to the paper '...'"
    - "In the paper '...', the authors remark that..."
    - "In the paper '...', the authors state that..."
    - "According to the paper '...', prior work has..."
    - "In the theoretical analysis of the paper '...'"
    - "In the paper '...', the results suggest that..."
### Output Format
Provide a JSON object with a single key "pair", which is the refined [answer, statement] pair.
""",
        'user': f"###Title\n{title}\n### Cloze Pair\n{json.dumps(cloze_pair)}\n"
    }
    response = utils.query_llm(quality_control_prompt, model='gpt-5-mini', reasoning_effort='low', system_prompt_included=True, return_json=True, max_tokens=1000)
    try:
        data = json.loads(response) if isinstance(response, str) else response
        pair = data.get('pair')
        if isinstance(pair, list) and len(pair) == 2:
            return tuple(pair)
    except (json.JSONDecodeError, AttributeError):
        print("Failed to parse JSON response for QC.")
    return None

def create_cloze_probe(refined_cloze_pair: Tuple[str, str], original_question: Dict[str, Any]) -> Dict[str, Any] | None:
    """Creates a probe/target pair from a cloze statement."""
    answer, statement = refined_cloze_pair
    statement_stripped = statement.rstrip('.,!?;: ')
    answer = answer.rstrip('.,!?;: ')
    if statement_stripped.lower().endswith(answer.lower()):
        probe = statement_stripped[:-len(answer)].rstrip()
        probe_data = {'target': answer, 'probe': probe, 'fact': statement}
        probe_data.update(original_question)
        return probe_data
    return None

def process_paper(paper_name: str, paper_content: str, add_on_mode: bool = False, **kwargs):
    """Main pipeline to generate comprehension probes for a single paper."""
    title = re.search(r'\\title{(.*?)}', paper_content).group(1) if re.search(r'\\title{(.*?)}', paper_content) else "no title"

    if add_on_mode:
        # Process entire paper at once with focus on repeated/emphasized ideas
        print(f"Processing entire paper {paper_name} for repeated/emphasized ideas...")
        section_texts = [paper_content]
        model = 'gpt-5'
        reasoning_effort = 'medium'
        focus_on_repeated = True
    else:
        # Original behavior: process by sections
        print(f"Parsing paper structure for {paper_name}...")
        paper_df = parse_paper_structure(paper_content)
        
        if paper_df.empty:
            print("Could not parse paper structure. Using full paper content.")
            section_texts = [paper_content]
        else:
            section_texts = paper_df['section_text'].unique()
        
        model = 'gpt-5'
        reasoning_effort = 'low'
        focus_on_repeated = False

    print(f"Generating comprehension questions for {len(section_texts)} {'paper(s)' if add_on_mode else 'sections'} in parallel...")
    
    questions = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        future_to_section = {executor.submit(generate_questions, section_text, model, reasoning_effort, focus_on_repeated): section_text for section_text in section_texts}
        for future in tqdm(concurrent.futures.as_completed(future_to_section), total=len(section_texts), desc="Generating questions"):
            try:
                questions_for_section = future.result()
                if questions_for_section:
                    questions.extend(questions_for_section)
            except Exception as exc:
                print(f'A section generated an exception: {exc}')

    print(f"Generated a total of {len(questions)} questions.")

    if not questions:
        print("No questions were generated. Exiting.")
        return
    save_df_for_debugging(pd.DataFrame(questions), '01_generated_questions.txt', 'comprehension', paper_name, ['question', 'answer', 'text_quotes'])

    # print(f"Filtering {len(questions)} questions for obviousness...")
    # # filtered_questions = filter_obvious_questions(questions)
    filtered_questions = questions
    # print(f"{len(filtered_questions)} questions remain after filtering.")
    # if not filtered_questions:
    #     return
    # save_df_for_debugging(pd.DataFrame(filtered_questions), '02_filtered_questions.txt', 'comprehension', paper_name, ['question', 'answer', 'text_quotes'])

    cloze_probes_list = []
    cloze_pairs_list = []
    refined_cloze_list = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        future_to_question = {executor.submit(convert_to_cloze, q): q for q in filtered_questions}
        for future in tqdm(concurrent.futures.as_completed(future_to_question), total=len(filtered_questions), desc="Converting to cloze statements"):
            question = future_to_question[future]
            try:
                cloze_pair = future.result()
                if cloze_pair:
                    cloze_pairs_list.append((cloze_pair, question))
            except Exception as exc:
                print(f'"{question.get("question", "A question")}" generated an exception during cloze conversion: {exc}')
    
    save_debug_file(json.dumps({'pairs': [p for p, q in cloze_pairs_list]}, indent=2), '03_cloze_pairs.txt', 'comprehension', paper_name)

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        future_to_data = {executor.submit(quality_control_cloze, pair, title): (pair, question) for pair, question in cloze_pairs_list}
        for future in tqdm(concurrent.futures.as_completed(future_to_data), total=len(cloze_pairs_list), desc="Performing quality control"):
            original_pair, question = future_to_data[future]
            try:
                refined_pair = future.result()
                if refined_pair:
                    refined_cloze_list.append(refined_pair)
                    probe_data = create_cloze_probe(refined_pair, question)
                    if probe_data:
                        cloze_probes_list.append(probe_data)
            except Exception as exc:
                print(f'"{question.get("question", "A question")}" generated an exception during QC: {exc}')

    save_debug_file(json.dumps({'pairs': refined_cloze_list}, indent=2), '04_refined_cloze.txt', 'comprehension', paper_name)

    print("Creating cloze probes...")
    if not cloze_probes_list:
        print("No valid cloze probes created. Exiting.")
        return
    
    cloze_df = pd.DataFrame(cloze_probes_list)
    cloze_df['probe'] = cloze_df['probe'].str.strip()
    cloze_df['target'] = ' ' + cloze_df['target'].str.strip().str.rstrip('.,!?;:')
    cloze_df['fact'] = cloze_df['fact'].str.strip().str.rstrip('.,!?;:')
    save_df_for_debugging(cloze_df, '05_final_probes.txt', 'comprehension', paper_name, ['probe', 'target', 'fact', 'question'])

    if add_on_mode:
        # Return the dataframe instead of saving (will be combined and saved as v7)
        return cloze_df
    else:
        # Original behavior: save to paper-specific directory
        output_dir = f'../../data/probes/comprehension/{paper_name}/'
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'probes_v1.csv')
        cloze_df.to_csv(output_path, index=False)
        print(f"Saved {len(cloze_df)} cloze probes to {output_path}")
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter", type=str, default=None, help="Only process papers containing this string in their filename.")
    parser.add_argument("--add_on_to_v6", action='store_true', help="Add new probes to existing v6 probes and save as v7.")
    args = parser.parse_args()
    
    if args.add_on_to_v6:
        # Load existing v6 probes
        v6_path = '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/data/probes/inference/DPO/probes_v6.csv'
        if os.path.exists(v6_path):
            v6_df = pd.read_csv(v6_path)
            print(f"Loaded {len(v6_df)} existing probes from v6")
        else:
            print(f"Warning: v6 file not found at {v6_path}. Starting with empty dataframe.")
            v6_df = pd.DataFrame()
        
        # Collect all new probes
        all_new_probes = []
        
        def collect_probes(paper_name: str, paper_content: str, **kwargs):
            result = process_paper(paper_name, paper_content, add_on_mode=True, **kwargs)
            if result is not None and not result.empty:
                all_new_probes.append(result)
        
        # Process papers
        process_papers(collect_probes, '../../data/arxiv/cleaned/', file_filter=args.filter)
        
        # Combine with v6 and save as v7
        if all_new_probes:
            new_probes_df = pd.concat(all_new_probes, ignore_index=True)
            print(f"\nGenerated {len(new_probes_df)} new probes")
            
            if not v6_df.empty:
                v7_df = pd.concat([v6_df, new_probes_df], ignore_index=True)
                print(f"Combined with v6 to create v7 with {len(v7_df)} total probes")
            else:
                v7_df = new_probes_df
                print(f"Created v7 with {len(v7_df)} total probes (no v6 found)")
            
            # Save v7
            v7_output_dir = os.path.dirname(v6_path)
            v7_path = os.path.join(v7_output_dir, 'probes_v7.csv')
            v7_df.to_csv(v7_path, index=False)
            print(f"Saved v7 probes to {v7_path}")
        else:
            print("No new probes were generated.")
    else:
        # Original behavior
        process_papers(process_paper, '../../data/arxiv/cleaned/', file_filter=args.filter)