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

def generate_questions(text: str) -> List[Dict[str, Any]]:
    """Generates inference questions from the text using an LLM."""
    system_prompt = """Based on your understanding of the provided text, your task is to generate highly inferential questions to test a reader's true comprehension and understanding of the text. Specifically, your objective is to assess whether the reader can integrate, synthesize, and generalize the implications of the text beyond what's been stated. This is difficult because academic papers by nature will provide analysis, interpretation, and discourse of the knowledge in the paper. Thus, aim to write questions that either (1) require the reader to build upon the existing knowledge in the text to infer new knowledge or draw a new conclusion, or (2) integrate and apply the knowledge in different settings i.e. truly "out-of-distribution" questions. The question should require a generalization of the knowledge and force a leap of reasoning. Be creative in your question formulation.
    
Here is a non-exhaustive list of types of inference that you should assess:
- Cross-Domain Integration
    - Cross-Domain Analogy: This could require connecting a concept from the paper to a well-known concept in another domain (e.g., computer science, statistics, physics). This tests for an abstract, transferable understanding. For instance, interpreting the KL-divergence term in the RLHF objective as a form of regularization to prevent overfitting, or the reference policy as a prior distribution and the optimal policy as a posterior distribution.
- Conceptual Synthesis
     - Core Insight: This could ask the reader to distill the central argument or innovation of the paper by combining information from multiple sections. For example, a question could ask the reader to synthesize that DPO transforms a reinforcement learning problem into a classification problem.
     - Causal Mechanism: This could involve using experimental evidence, especially from ablations, to pinpoint the source of an observed effect. For example, a question could require synthesizing that the benefit of CoT comes from the sequential reasoning process itself, by ruling out alternative hypotheses tested in ablations.
- Reading Between the Lines
    - Identifying Implicit Assumptions: This probes for an understanding of the unstated conditions upon which the paper's claims rest. For example, a question might require the reader to infer that for the DPO loss to be a valid maximum likelihood objective, the preference dataset D must be assumed to be sampled i.i.d. from the true human preference distribution.
- Mathematical Understanding
    - Equation Interpretation: This could ask for the conceptual role or meaning of a specific term within a larger mathematical expression, beyond its literal definition.
- Counterfactual Understanding
    - Predicting Outcomes of Hypothetical Scenarios: The reader could use the principles established in the paper to predict the result of a new experiment or a change in conditions.
    
Furthermore, here as some specific guidelines you should follow as you write the questions:
- The question can be as long as needed, but the answer must be a coherent phrase that is 1-5 words long. 
- The questions should NOT be about factual recall that asks to recall a specific fact in the paper. 
- The answer to the question should NOT be found in the text. It should be new knowledge.
- The questions should require a generalizable, deep understanding of the knowledge. 
- Be precise with the question formulation so that there is only one clear answer.

In addition, for each question, provide: 
- The prior knowledge that is required to answer the question. Academic papers build upon a large body of domain knowledge, and so there is an underlying assumption that the reader has a deep understanding of the domain knowledge for any paper. The question may also require the reader to apply the knowledge in a different setting.
- The sentences from the text that are required to answer the question. Cite from the text verbatim, and don't surround it with quotes.
- For a lay reader, an explanation of what the question is asking.
- For a lay reader, an explanation of how the answer is derived from the provided knowledge in the text.

### Output Format
Provide the output in JSON format, as a dictionary with a single key "qa_items" which is a list of dictionaries with the following keys:
- "question": (string) 
- "answer": (string)
- "prior_knowledge": (string)
- "text_sentences": list of strings
- "question_explanation": (string)
- "inference_explanation": (string)
"""
    prompt = {'system': system_prompt, 'user': f"### Text\n{text}"}
    response_json = utils.query_llm(prompt, model='gpt-5-mini', reasoning_effort='high', system_prompt_included=True, return_json=True, max_tokens=4000)
    
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
        sentences = q.get('text_sentences', [])
        if all(is_text_in_document(s, text, threshold=0.5) for s in sentences):
            parsed_questions.append(q)

    if len(parsed_questions) != len(questions):
        print(f"DROPPED {len(questions) - len(parsed_questions)} questions with invalid text sentences.")
    
    return parsed_questions

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
        'system': r"""Your task is to review and refine a statement that has been extracted from an academic paper. You will be given an '(answer, statement)' pair. Your task is to apply a rigorous checklist to the pair, refining it based on a provided quality control checklist.

### Quality Control Checklist

1. LaTeX Formatting
- All mathematical expressions and notations **MUST** be written in LaTeX, enclosed in '$' or '$$' delimiters.
- Do *NOT* use unicode mathematical characters (e.g., use '\\pi', not 'π').
- Do *NOT* use unnecessary styling commands like '\\displaystyle'.
- Ensure LaTeX syntax matches the style of the original context (e.g., '( ... )' or '$ ... $').
- Action: Rewrite the math expressions and statements so they can be written in LaTeX, keeping the rest of the statement the same, correcting any and all formatting errors related to mathematical notation.

2. Start the statement with one of the following templates:
    - "In the paper '...'"
    - "According to the paper '...'"
- Action: Rewrite the statement so that it starts with one of the following templates.

In all your adjustments, change the statement as minimally as necessary. If a statement is already good, make no changes.

### Output Format
Provide a JSON object with a single key "pair", which is the refined [answer, statement] pair.
""",
        'user': f"### Cloze Pair\n{json.dumps(cloze_pair)}\n"
    }
    response = utils.query_llm(quality_control_prompt, model='gpt-5-mini', system_prompt_included=True, return_json=True, max_tokens=1000)
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

def process_paper(paper_name: str, paper_content: str, **kwargs):
    # """Main pipeline to generate comprehension probes for a single paper."""
    print(f"Parsing paper structure for {paper_name}...")
    paper_df = parse_paper_structure(paper_content)
    title = re.search(r'\\title{(.*?)}', paper_content).group(1) if re.search(r'\\title{(.*?)}', paper_content) else "no title"

    if paper_df.empty:
        print("Could not parse paper structure. Using full paper content.")
        section_texts = [paper_content]
    else:
        section_texts = paper_df['section_text'].unique()

    print(f"Generating comprehension questions for {len(section_texts)} sections in parallel...")
    
    questions = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        future_to_section = {executor.submit(generate_questions, section_text): section_text for section_text in section_texts}
        for future in tqdm(concurrent.futures.as_completed(future_to_section), total=len(section_texts), desc="Generating questions"):
            try:
                questions_for_section = future.result()
                if questions_for_section:
                    questions.extend(questions_for_section)
            except Exception as exc:
                print(f'A section generated an exception: {exc}')

    #questions = generate_questions(paper_content)
    print(f"Generated a total of {len(questions)} questions.")

    if not questions:
        print("No questions were generated. Exiting.")
        return
    save_df_for_debugging(pd.DataFrame(questions), '01_generated_questions.txt', 'inference', paper_name, ['question', 'answer', 'text_sentences', 'prior_knowledge', 'question_explanation', 'inference_explanation'])

    filtered_questions = questions

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
    
    save_debug_file(json.dumps({'pairs': [p for p, q in cloze_pairs_list]}, indent=2), '03_cloze_pairs.txt', 'inference', paper_name)

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

    save_debug_file(json.dumps({'pairs': refined_cloze_list}, indent=2), '04_refined_cloze.txt', 'inference', paper_name)

    print("Creating cloze probes...")
    if not cloze_probes_list:
        print("No valid cloze probes created. Exiting.")
        return
    
    cloze_df = pd.DataFrame(cloze_probes_list)
    cloze_df['probe'] = cloze_df['probe'].str.strip()
    cloze_df['target'] = ' ' + cloze_df['target'].str.strip().str.rstrip('.,!?;:')
    cloze_df['fact'] = cloze_df['fact'].str.strip().str.rstrip('.,!?;:')
    save_df_for_debugging(cloze_df, '05_final_probes.txt', 'inference', paper_name, ['probe', 'target', 'fact', 'question', 'prior_knowledge', 'text_sentences', 'question_explanation', 'inference_explanation'])

    output_dir = f'../../data/probes/inference/{paper_name}/'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'probes_v3.csv')
    cloze_df.to_csv(output_path, index=False)
    print(f"Saved {len(cloze_df)} cloze probes to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter", type=str, default=None, help="Only process papers containing this string in their filename.")
    args = parser.parse_args()
    process_papers(process_paper, '../../data/arxiv/cleaned/', file_filter=args.filter)
