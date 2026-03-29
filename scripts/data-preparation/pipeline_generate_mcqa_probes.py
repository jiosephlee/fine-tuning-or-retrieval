import os
import sys
import json
import pandas as pd
import concurrent.futures
from tqdm import tqdm
import random
import re
import argparse

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import utils.utils as utils

# 4-shot examples
FEW_SHOT_EXAMPLES = """Question: In the paper 'Direct Preference Optimization: Your Language Model is Secretly a Reward Model', DPO stands for
(A) Direct Preference Optimization
(B) Distributed Preference Optimization
(C) Direct Policy Optimization
(D) Dynamic Preference Optimization
(E) None of the above
Answer: (A)

Question: According to the paper 'Attention Is All You Need', the Transformer model relies entirely on
(A) Recurrent neural networks
(B) Convolutional neural networks
(C) Attention mechanisms
(D) Long Short-Term Memory networks
(E) None of the above
Answer: (C)

Question: In the paper 'LoRA: Low-Rank Adaptation of Large Language Models', LoRA freezes the pre-trained model weights and injects trainable rank decomposition matrices into
(A) Every layer of the Transformer
(B) The embedding layer only
(C) Each layer of the Transformer architecture
(D) The output layer only
(E) None of the above
Answer: (C)

Question: The paper 'Language Models are Few-Shot Learners' introduces which model?
(A) GPT-2
(B) GPT-3
(C) BERT
(D) T5
(E) None of the above
Answer: (B)
"""

def read_paper_text(domain):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    tex_path = os.path.join(base_dir, f'data/arxiv/cleaned/{domain}.tex')
    try:
        with open(tex_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Warning: Paper text file not found at {tex_path}")
        return ""

def contextualize_question(row, paper_title, paper_text):
    """
    Contextualizes the question using the prompt from pipeline_fact_probe.py.
    """
    question = row['question']
    answer = row['answer']
    # Use text_sentences as context if available, otherwise just the question/answer pair
    context = row.get('text_sentences', '')
    
    # Truncate paper text if too long to avoid token limits, though gpt-5.1 handles large context well.
    # Let's keep a reasonable chunk if needed, but user asked to provide the original paper.
    # We'll pass the whole text if possible, or a large chunk.
    
    prompt = {}
    prompt['system'] = r"""You will be given a question extracted from an academic paper as well as its corresponding answer. Your task is to turn the question into a self-contained, precise question. 

### Instructions
The overall goal of this task is to make the questions clear by incorporating the relevant context. This ensures the question is unambiguous and doesn't require looking back to the source material.

1.  Rewrite the question so that it starts with one of the following templates. 
    - "In the paper '{title}', ..."
    - "According to the paper '{title}',..."
    - "In the paper '{title}', the authors remark that..."
    - "In the paper '{title}', the authors state that..."
    - "According to the paper '{title}', prior work has..."
    - "In the theoretical analysis of the paper "{title}"..."
    - "In the paper '{title}', the results suggest that..."
    This is a non-exhaustive list of templates, and you should use your own judgement to choose the most appropriate template or modify the template to fit the sentence. 
    There is no need to re-emphasize reference to the paper after mentioning it in the beginning of the question.
2.  Add sufficient context. Specifically, use the *provided context* and the *paper text* to supply whatever information is needed to make the question self-contained and unambiguous.
3.  Do not leak the answer. Please make sure that *the answer is not revealed* in the question. The answer should never appear in the question.
4.  Refine Question. The rewritten question can be broken up into multiple sentences if the question becomes verbose. Make sure the question is written clearly and grammatically correct.

### Output Format
Provide the output in JSON format with a single key "question" containing the rewritten question.
"""
    prompt['user'] = f"""### Title\n{paper_title}\n### Paper Text\n{paper_text}\n\n### Specific Context\n{context}\n\n### Question\n{question}\n### Answer\n{answer}"""
    
    try:
        response = utils.query_llm(prompt, model='gpt-5.1', reasoning_effort='low', system_prompt_included=True, return_json=True)
        data = json.loads(response)
        return data.get('question', question)
    except Exception as e:
        print(f"Error contextualizing question: {e}")
        return question

def generate_mcqa_options(question, answer, paper_text):
    """
    Generates 3 distractors and formats the MCQA.
    """
    prompt = {}
# Quality Control Prompt
QUALITY_CONTROL_PROMPT = r"""Your task is to review and refine an inference question. It has been extracted from an academic paper. You will be given a question and its answer, as well as the supporting text. Your task is to apply a rigorous checklist to the question, refining it based on a provided quality control checklist.

### Quality Control Checklist

1. LaTeX Formatting
- All mathematical expressions and notations **MUST** be written in LaTeX, enclosed in '$' or '$$' delimiters.
- Do *NOT* use unicode mathematical characters (e.g., use '\\pi', not 'π').
- Do *NOT* use unnecessary styling commands like '\\displaystyle'.
- Ensure LaTeX syntax matches the style of the original context (e.g., '( ... )' or '$ ... $').
- Action: Rewrite the math expressions in the question so they can be written in LaTeX, keeping the rest of the question the same, correcting any and all formatting errors related to mathematical notation.

2. Start the question with one of the following templates that fits the most naturally:
    - "In the paper '...'"
    - "According to the paper '...'"
    - "In the paper '{title}', the authors remark that..."
    - "In the paper '{title}', the authors state that..."
    - "According to the paper '{title}', prior work has..."
    - "In the theoretical analysis of the paper "{title}"..."
    - "In the paper '{title}', the results suggest that..."
- Action: Rewrite the question so that it starts with one of the following templates. Feel free to adjust the template to fit the question more naturally.

3. Answer Leakage
- The answer must not be leaked, explicitly or implicitly, in the question.
- Action: Minimally rewrite the question such that the answer is not revealed.

In all your adjustments, change the question as minimally as necessary. If the question is already good, make no changes.

### Output Format
Provide a JSON object with a single key "question", which is the refined question.
"""

def quality_control_question(question, answer, paper_title, context):
    prompt = {}
    prompt['system'] = QUALITY_CONTROL_PROMPT
    prompt['user'] = f"""### Paper Context\n{context}\n### Title\n{paper_title}\n### Question\n{question}\n### Answer\n{answer}"""
    
    try:
        response = utils.query_llm(prompt, model='gpt-5.1', reasoning_effort='none', system_prompt_included=True, return_json=True)
        data = json.loads(response)
        return data.get('question', question)
    except Exception as e:
        print(f"Error in quality control: {e}")
        return question

def generate_mcqa_options(question, answer, paper_text):
    """
    Generates 3 distractors and formats the MCQA.
    """
    prompt = {}
    prompt['system'] = """You are an expert at creating multiple choice questions for academic papers. 
Given a question and its correct answer, generate 3 plausible but incorrect distractors.

### Instructions
- The distractors should be **hard** and **challenging**. They should be plausible to someone who has a surface-level understanding but incorrect for someone who has deeply read the paper.
- Avoid obvious outliers or easily dismissible options.
- The distractors should be of similar length and style to the correct answer.
- Ensure there is only **one** strictly correct answer among the options.
- **LaTeX Formatting**: Ensure all mathematical expressions in the distractors are written in LaTeX, enclosed in '$' or '$$' delimiters.

### Output Format
Provide the output in JSON format with a single key "distractors" containing a list of 3 strings.
"""
    prompt['user'] = f"""### Paper Text\n{paper_text}\n\n### Question\n{question}\n\n### Correct Answer\n{answer}"""
    
    try:
        response = utils.query_llm(prompt, model='gpt-5.1', reasoning_effort='low', system_prompt_included=True, return_json=True)
        data = json.loads(response)
        distractors = data.get('distractors', [])
        
        if len(distractors) < 3:
            # Fallback if not enough distractors
            return None
            
        # Select 3 distractors
        distractors = distractors[:3]
        
        # Create options list: Correct + 3 Distractors
        options = [answer] + distractors
        random.shuffle(options)
        
        # Assign labels A, B, C, D
        labels = ['(A)', '(B)', '(C)', '(D)']
        formatted_options = []
        correct_label = ""
        
        for i, option in enumerate(options):
            label = labels[i]
            formatted_options.append(f"{label} {option}")
            if option == answer:
                correct_label = label
        
        # Add option E
        formatted_options.append("(E) None of the above")
        
        # Format the final question string
        full_question = f"{question}\n" + "\n".join(formatted_options)
        
        return {
            'mcqa_question': full_question,
            'mcqa_answer': correct_label
        }
        
    except Exception as e:
        print(f"Error generating MCQA options: {e}")
        return None

def process_row(row, paper_title, paper_text):
    try:
        # 1. Contextualize
        contextualized_q = contextualize_question(row, paper_title, paper_text)
        
        # 2. Quality Control (LaTeX & Contextualization check)
        contextualized_q = quality_control_question(contextualized_q, row['answer'], paper_title, row.get('text_sentences', ''))

        # 3. Generate MCQA
        mcqa_data = generate_mcqa_options(contextualized_q, row['answer'], paper_text)
        
        if mcqa_data:
            # 4. Create few-shot entry
            fewshot_q = f"{FEW_SHOT_EXAMPLES}\nQuestion: {mcqa_data['mcqa_question']}\nAnswer:"
            
            return {
                'question': mcqa_data['mcqa_question'],
                'answer': mcqa_data['mcqa_answer'],
                'fewshot_question': fewshot_q,
                'original_question': row['question'],
                'contextualized_question': contextualized_q
            }
    except Exception as e:
        print(f"Error processing row: {e}")
    return None

def process_domain(domain):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    input_path = os.path.join(base_dir, f'data/probes/inference/{domain}/probes_v7.csv')
    output_path = os.path.join(base_dir, f'data/probes/inference/{domain}/probes_v7_mcqa.csv')
    
    if not os.path.exists(input_path):
        print(f"Input file not found: {input_path}")
        return

    print(f"Processing domain: {domain}")
    df = pd.read_csv(input_path)
    
    # Read paper text
    paper_text = read_paper_text(domain)
    
    # Map domain to paper title
    paper_titles = {
        'DPO': 'Direct Preference Optimization: Your Language Model is Secretly a Reward Model',
        'BOFT': 'Parameter-Efficient Orthogonal Finetuning via Butterfly Factorization',
        '1_58': 'The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits',
        'OFT': 'Controlling Text-to-Image Diffusion by Orthogonal Finetuning',
        'QLoRA': 'QLoRA: Efficient Finetuning of Quantized LLMs',
        'GRPO': 'DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models'
    }
    
    paper_title = paper_titles.get(domain, f"the paper regarding {domain}")

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_row, row, paper_title, paper_text) for _, row in df.iterrows()]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Generating MCQA for {domain}"):
            res = future.result()
            if res:
                results.append(res)

    if results:
        out_df = pd.DataFrame(results)
        out_df.to_csv(output_path, index=False)
        print(f"Saved {len(out_df)} MCQA probes to {output_path}")
    else:
        print(f"No results generated for {domain}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate MCQA probes.')
    parser.add_argument('--domain', type=str, help='Specific domain to process (optional). If not provided, all domains will be processed.')
    args = parser.parse_args()

    if args.domain:
        process_domain(args.domain)
    else:
        domains = ['DPO', 'BOFT', '1_58', 'OFT', 'QLoRA', 'GRPO']
        for domain in domains:
            process_domain(domain)
