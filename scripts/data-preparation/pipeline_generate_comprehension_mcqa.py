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

# 4-shot examples for comprehension MCQA
FEW_SHOT_EXAMPLES = """Question: In the paper 'Attention Is All You Need', the authors replace recurrence and convolutions entirely with attention. What is the primary architectural motivation for this design choice?
(A) To reduce the number of parameters in the model
(B) To allow constant-time access between any two positions in a sequence, improving parallelization and long-range dependency modeling
(C) To make the model compatible with pre-existing RNN frameworks
(D) To reduce memory consumption during inference
(E) None of the above
Answer: (B)

Question: In the paper 'Direct Preference Optimization: Your Language Model is Secretly a Reward Model', the authors derive DPO by reparameterizing the reward function in the RLHF objective. What key insight makes it possible to bypass explicit reward modeling?
(A) The reward function can be expressed analytically in terms of the optimal policy and a reference policy
(B) Human preferences are inherently noisy and do not require precise reward estimates
(C) The KL-divergence penalty in the RLHF objective can be dropped without affecting convergence
(D) Reward modeling and policy optimization can be interleaved in a single gradient step
(E) None of the above
Answer: (A)

Question: In the paper 'LoRA: Low-Rank Adaptation of Large Language Models', the authors inject low-rank matrices into each Transformer layer. Why does this approach maintain performance comparable to full fine-tuning despite training far fewer parameters?
(A) The pre-trained weights already capture most task-relevant features, so only a small residual correction is needed
(B) The low-rank matrices are initialized to approximate the full-rank gradient updates
(C) The frozen weights act as a regularizer that prevents overfitting during adaptation
(D) Low-rank decomposition compresses the weight updates without discarding the directions that change most during fine-tuning
(E) None of the above
Answer: (D)

Question: In the paper 'Language Models are Few-Shot Learners', GPT-3 demonstrates strong few-shot performance by conditioning on examples in the prompt. What does this suggest about the relationship between model scale and in-context learning ability?
(A) Larger models memorize more training examples, enabling exact pattern matching at test time
(B) Scale improves the model's ability to infer and apply task structure from a small number of demonstrations without gradient updates
(C) Few-shot performance is primarily driven by the diversity of pre-training data rather than model size
(D) In-context learning emerges because larger models have lower perplexity on the prompt examples
(E) None of the above
Answer: (B)
"""

_QUESTION_GENERATION_TEMPLATES = {
    'arxiv': r"""You are an expert at creating comprehension questions for academic papers. You will be given an academic paper. Your task is to generate questions that test *deep understanding* of the paper's content, not factual recall.

### What makes a good comprehension question
- It requires *reasoning* about the text: synthesizing information across sentences/paragraphs, understanding cause-and-effect relationships, grasping the motivation behind design choices, or drawing connections between different parts of the paper.
- It cannot be answered by copying or lightly rephrasing a single sentence from the paper.
- It *can* be answered by a careful reader who has the text available; no external knowledge beyond the paper is needed besides basic background knowledge.
- Good question types include:
  - **Why** questions: Why did the authors make a specific design choice? Why does a particular method work?
  - **How** questions: How do two components of the system interact? How does a theoretical result connect to the experimental findings?
  - **What-if** / counterfactual: What would happen if a specific assumption were removed?
  - **Compare/contrast**: How does the proposed method differ from a baseline in a non-obvious way?
  - **Implication**: What does a specific result imply about the broader problem?

### Instructions
1. Generate many comprehension questions as possible from the paper, spread out over all sections of the paper.
2. For each question, identify the **contiguous excerpt** from the paper (~1024 tokens) that contains all the information needed to answer the question. This excerpt must be a verbatim, contiguous passage from the paper text.
3. For each question, provide:
   - "question": The question text.
   - "answer": A concise answer (1-3 sentences). This answer must be derivable from the excerpt.
   - "excerpt_start": The first 6 words of the excerpt, copied verbatim from the paper.
   - "excerpt_end": The last 6 words of the excerpt, copied verbatim from the paper.
4. Avoid questions that:
   - Ask for a single named entity, number, or definition (those are factual recall)
   - Require knowledge not in the paper
5. Avoid asking questions that are too similar to each other.

### Output Format
JSON with a single key "questions" containing a list of question objects.
""",
    'medical': r"""You are an expert at creating comprehension questions for medical case reports. You will be given a medical case report. Your task is to generate questions that test *deep understanding* of the clinical reasoning, not factual recall.

### What makes a good comprehension question
- It requires *reasoning* about the text: synthesizing clinical findings across sections, understanding diagnostic logic, grasping treatment rationale, or drawing connections between presentation, workup, and outcome.
- It cannot be answered by copying or lightly rephrasing a single sentence from the report.
- It *can* be answered by a careful reader who has the text available; no external medical knowledge beyond the report is needed besides basic clinical background.
- Good question types include:
  - **Why** questions: Why was a particular treatment chosen? Why did the differential diagnosis narrow?
  - **How** questions: How did the clinical findings lead to the final diagnosis? How did the treatment approach evolve?
  - **What-if** / counterfactual: What might have happened if a key test had been omitted?
  - **Compare/contrast**: How did the initial presentation differ from what was ultimately found?
  - **Implication**: What does this case suggest about managing similar presentations?

### Instructions
1. Generate as many comprehension questions as possible from the case report, spread across all sections (presentation, workup, diagnosis, treatment, outcome, discussion).
2. For each question, identify the **contiguous excerpt** from the report (~1024 tokens) that contains all the information needed to answer the question. This excerpt must be a verbatim, contiguous passage from the report text.
3. For each question, provide:
   - "question": The question text.
   - "answer": A concise answer (1-3 sentences). This answer must be derivable from the excerpt.
   - "excerpt_start": The first 6 words of the excerpt, copied verbatim from the report.
   - "excerpt_end": The last 6 words of the excerpt, copied verbatim from the report.
4. Avoid questions that:
   - Ask for a single named entity, number, or definition (those are factual recall)
   - Require knowledge not in the report
5. Avoid asking questions that are too similar to each other.

### Output Format
JSON with a single key "questions" containing a list of question objects.
""",
    'legal': r"""You are an expert at creating comprehension questions for legal opinions. You will be given a court opinion. Your task is to generate questions that test *deep understanding* of the legal reasoning, not factual recall.

### What makes a good comprehension question
- It requires *reasoning* about the text: synthesizing legal arguments across sections, understanding how precedent is applied, grasping the court's analytical framework, or drawing connections between facts, law, and holding.
- It cannot be answered by copying or lightly rephrasing a single sentence from the opinion.
- It *can* be answered by a careful reader who has the text available; no external legal knowledge beyond the opinion is needed besides basic legal background.
- Good question types include:
  - **Why** questions: Why did the court reject a particular argument? Why was a specific standard of review applied?
  - **How** questions: How did the factual record support the court's conclusion? How did the court distinguish a cited precedent?
  - **What-if** / counterfactual: What might have changed if a key fact were different?
  - **Compare/contrast**: How did the majority's reasoning differ from the dissent or from the lower court?
  - **Implication**: What does the holding imply for similar future cases?

### Instructions
1. Generate as many comprehension questions as possible from the opinion, spread across all major sections (facts, procedural history, analysis, holding).
2. For each question, identify the **contiguous excerpt** from the opinion (~1024 tokens) that contains all the information needed to answer the question. This excerpt must be a verbatim, contiguous passage from the opinion text.
3. For each question, provide:
   - "question": The question text.
   - "answer": A concise answer (1-3 sentences). This answer must be derivable from the excerpt.
   - "excerpt_start": The first 6 words of the excerpt, copied verbatim from the opinion.
   - "excerpt_end": The last 6 words of the excerpt, copied verbatim from the opinion.
4. Avoid questions that:
   - Ask for a single named entity, number, or definition (those are factual recall)
   - Require knowledge not in the opinion
5. Avoid asking questions that are too similar to each other.

### Output Format
JSON with a single key "questions" containing a list of question objects.
""",
}


def get_question_generation_prompt(domain):
    return _QUESTION_GENERATION_TEMPLATES[_domain_type(domain)]

DISTRACTOR_GENERATION_PROMPT = r"""You are an expert at creating multiple choice questions for academic papers. You will generate distractors for a comprehension question.

### Instructions
Given a question, its correct answer, and the relevant excerpt from the paper, generate 3 plausible but *incorrect* distractors.

- Each distractor should reflect a **common misunderstanding** or a **superficially plausible but wrong** interpretation of the paper.
- Distractors should require careful reading to distinguish from the correct answer.
- They should be of similar length, style, and specificity to the correct answer.
- Do NOT include distractors that are obviously wrong or unrelated to the paper.
- Ensure there is only **one** strictly correct answer among the options.
- **LaTeX Formatting**: Use LaTeX for any math expressions, enclosed in '$' or '$$'.

### Output Format
JSON with a single key "distractors" containing a list of 3 strings.
"""

QUALITY_CONTROL_PROMPT = r"""Your task is to review and refine a comprehension question and its answer extracted from an academic paper. You will be given the question, answer, and the excerpt from the paper that the question is based on. Apply the following quality control checklist, making minimal changes.

### Quality Control Checklist

1. **LaTeX Formatting**: All mathematical expressions must be in valid LaTeX. The LaTeX style must match the style used in the paper verbatim; copy notation conventions (e.g., '\mathbf', '\text', '\boldsymbol') directly from the excerpt.

2. **Comprehension Focus**: The question must require reasoning, not just recall. If the question can be answered by quoting a single sentence, make it more integrative.

Prefer returning the question and answer unchanged. Only refine if there is a clear, concrete issue. Do not make unnecessary changes.

### Output Format (JSON)
- If the question and answer already pass ALL checks with no changes needed, return: {"change": false}
- If refinement is needed, return: {"change": true, "question": "...", "answer": "..."}
"""


def _load_title_map(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MEDICAL_TITLES = _load_title_map(os.path.join(BASE_DIR, 'data/medical/case_report_titles.json'))
LEGAL_TITLES = _load_title_map(os.path.join(BASE_DIR, 'data/legal/case_titles.json'))
MEDICAL_DOMAINS = set(MEDICAL_TITLES.keys())
LEGAL_DOMAINS = set(LEGAL_TITLES.keys())


def _domain_type(domain):
    if domain in MEDICAL_DOMAINS:
        return 'medical'
    if domain in LEGAL_DOMAINS:
        return 'legal'
    return 'arxiv'


def read_paper_text(domain):
    dtype = _domain_type(domain)
    if dtype == 'medical':
        path = os.path.join(BASE_DIR, f'data/medical/cleaned/{domain}.txt')
    elif dtype == 'legal':
        path = os.path.join(BASE_DIR, f'data/legal/cleaned/{domain}.txt')
    else:
        path = os.path.join(BASE_DIR, f'data/arxiv/cleaned/{domain}.tex')
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Warning: Text file not found at {path}")
        return ""


def get_paper_title(paper_text, domain):
    """Extract title from LaTeX, title maps, or first line."""
    if domain in MEDICAL_TITLES:
        return MEDICAL_TITLES[domain]
    if domain in LEGAL_TITLES:
        return LEGAL_TITLES[domain]

    title_match = re.search(r'\\title\{(.*?)\}', paper_text, re.DOTALL)
    if title_match:
        return title_match.group(1).strip()

    paper_titles = {
        'DPO': 'Direct Preference Optimization: Your Language Model is Secretly a Reward Model',
        'BOFT': 'Parameter-Efficient Orthogonal Finetuning via Butterfly Factorization',
        '1_58': 'The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits',
        'OFT': 'Controlling Text-to-Image Diffusion by Orthogonal Finetuning',
        'QLoRA': 'QLoRA: Efficient Finetuning of Quantized LLMs',
        'GRPO': 'DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models',
    }
    return paper_titles.get(domain, f"the document regarding {domain}")


def extract_excerpt(paper_text, excerpt_start, excerpt_end):
    """Extract the contiguous excerpt from the paper using start/end markers."""
    # Normalize whitespace for matching
    normalized = ' '.join(paper_text.split())
    norm_start = ' '.join(excerpt_start.split())
    norm_end = ' '.join(excerpt_end.split())

    # Lowercased version for case-insensitive matching
    normalized_lower = normalized.lower()

    start_idx = normalized_lower.find(norm_start.lower())
    if start_idx == -1:
        short_start = ' '.join(norm_start.split()[:4])
        start_idx = normalized_lower.find(short_start.lower())
        if start_idx == -1:
            return None

    end_idx = normalized_lower.find(norm_end.lower(), start_idx)
    if end_idx == -1:
        short_end = ' '.join(norm_end.split()[-4:])
        end_idx = normalized_lower.find(short_end.lower(), start_idx)
        if end_idx == -1:
            return None

    return normalized[start_idx:end_idx + len(norm_end)]


def generate_comprehension_questions(paper_text, paper_title, domain='DPO'):
    """Generate comprehension questions from the full paper text."""
    prompt = {
        'system': get_question_generation_prompt(domain),
        'user': f"### Title\n{paper_title}\n\n### Text\n{paper_text}"
    }

    response = utils.query_llm(
        prompt, model='gpt-5.4', reasoning_effort='medium',
        system_prompt_included=True, return_json=True, max_tokens=32786
    )
    try:
        data = json.loads(response) if isinstance(response, str) else response
        questions = data.get('questions', [])
        if not isinstance(questions, list):
            print("Unexpected response format: 'questions' is not a list.")
            return []
        return questions
    except (json.JSONDecodeError, TypeError) as e:
        print(f"Error parsing question generation response: {e}")
        return []


def quality_control_question(question_text, answer, paper_title, excerpt):
    """Apply quality control to refine a question. Returns (question, answer, was_refined)."""
    prompt = {
        'system': QUALITY_CONTROL_PROMPT,
        'user': f"### Title\n{paper_title}\n### Excerpt\n{excerpt}\n### Question\n{question_text}\n### Answer\n{answer}"
    }

    try:
        response = utils.query_llm(
            prompt, model='gpt-5.4-mini', reasoning_effort='low',
            system_prompt_included=True, return_json=True
        )
        data = json.loads(response) if isinstance(response, str) else response
        if not data.get('change', True):
            return question_text, answer, False
        return data.get('question', question_text), data.get('answer', answer), True
    except Exception as e:
        print(f"Error in quality control: {e}")
        return question_text, answer, False


def generate_distractors(question_text, answer, excerpt):
    """Generate 3 plausible distractors for a comprehension question."""
    prompt = {
        'system': DISTRACTOR_GENERATION_PROMPT,
        'user': f"### Excerpt\n{excerpt}\n\n### Question\n{question_text}\n\n### Correct Answer\n{answer}"
    }

    try:
        response = utils.query_llm(
            prompt, model='gpt-5.4-mini', reasoning_effort='medium',
            system_prompt_included=True, return_json=True
        )
        data = json.loads(response) if isinstance(response, str) else response
        distractors = data.get('distractors', [])

        if len(distractors) < 3:
            return None
        return distractors[:3]
    except Exception as e:
        print(f"Error generating distractors: {e}")
        return None


def format_mcqa(question_text, answer, distractors):
    """Format question + answer + distractors into MCQA with shuffled options."""
    options = [answer] + distractors
    random.shuffle(options)

    labels = ['(A)', '(B)', '(C)', '(D)']
    formatted_options = []
    correct_label = ""

    for i, option in enumerate(options):
        label = labels[i]
        formatted_options.append(f"{label} {option}")
        if option == answer:
            correct_label = label

    formatted_options.append("(E) None of the above")
    full_question = f"{question_text}\n" + "\n".join(formatted_options)

    return {
        'mcqa_question': full_question,
        'mcqa_answer': correct_label
    }


def process_question(q, paper_title, paper_text):
    """Process a single generated question through excerpt extraction, QC, distractor generation, and MCQA formatting."""
    try:
        question_text = q['question']
        answer = q['answer']
        excerpt_start = q.get('excerpt_start', '')
        excerpt_end = q.get('excerpt_end', '')

        # 0. Extract excerpt from paper
        excerpt = extract_excerpt(paper_text, excerpt_start, excerpt_end)
        if not excerpt:
            print(f"  Could not extract excerpt for: {question_text[:80]}...")
            return None

        # 1. Quality control
        refined_q, refined_answer, was_refined = quality_control_question(
            question_text, answer, paper_title, excerpt
        )

        # 2. Generate distractors (using excerpt, not full paper)
        distractors = generate_distractors(refined_q, refined_answer, excerpt)
        if not distractors:
            return None

        # 3. Format as MCQA
        mcqa = format_mcqa(refined_q, refined_answer, distractors)

        # 4. Build few-shot prompt
        fewshot_q = f"{FEW_SHOT_EXAMPLES}\nQuestion: {mcqa['mcqa_question']}\nAnswer:"

        return {
            'question': mcqa['mcqa_question'],
            'answer': mcqa['mcqa_answer'],
            'fewshot_question': fewshot_q,
            'original_question': question_text,
            'comprehension_answer': refined_answer,
            'excerpt': excerpt,
            'was_refined': was_refined,
        }
    except Exception as e:
        print(f"Error processing question: {e}")
        return None


def process_domain(domain, num_questions=None):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    output_dir = os.path.join(base_dir, f'data/probes/inference/{domain}')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'comprehension_mcqa.csv')

    print(f"\n{'='*60}")
    print(f"Processing domain: {domain}")
    print(f"{'='*60}")

    # Read paper
    paper_text = read_paper_text(domain)
    if not paper_text:
        print(f"No paper text found for {domain}. Skipping.")
        return

    paper_title = get_paper_title(paper_text, domain)
    print(f"Paper title: {paper_title}")

    # Step 1: Generate comprehension questions
    print("Generating comprehension questions...")
    questions = generate_comprehension_questions(paper_text, paper_title, domain=domain)
    print(f"Generated {len(questions)} raw questions.")

    if not questions:
        print(f"No questions generated for {domain}.")
        return

    # Validate excerpt extraction before proceeding
    valid_questions = []
    for q in questions:
        excerpt = extract_excerpt(paper_text, q.get('excerpt_start', ''), q.get('excerpt_end', ''))
        if excerpt:
            valid_questions.append(q)
        else:
            print(f"  Dropped (excerpt not found): {q['question'][:80]}...")

    print(f"Validated excerpts: {len(valid_questions)}/{len(questions)} questions have extractable excerpts.")
    questions = valid_questions

    if num_questions:
        questions = questions[:num_questions]

    # Step 2: Process each question (QC + distractors + MCQA formatting) in parallel
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(process_question, q, paper_title, paper_text)
            for q in questions
        ]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Processing MCQA for {domain}"):
            res = future.result()
            if res:
                results.append(res)

    if results:
        out_df = pd.DataFrame(results)
        out_df.to_csv(output_path, index=False)
        print(f"Saved {len(out_df)} comprehension MCQA probes to {output_path}")

        # Save readable debug txt
        debug_path = os.path.join(output_dir, 'comprehension_mcqa_debug.txt')
        with open(debug_path, 'w', encoding='utf-8') as f:
            f.write(f"Comprehension MCQA Probes — {domain}\n")
            f.write(f"Total: {len(out_df)} questions\n")
            f.write(f"Refined by QC: {out_df['was_refined'].sum()}, Unchanged: {(~out_df['was_refined']).sum()}\n")
            f.write(f"{'='*80}\n\n")
            for idx, row in out_df.iterrows():
                f.write(f"----- Question {idx + 1} (Answer: {row['answer']}, Refined: {row['was_refined']}) -----\n")
                f.write(f"{row['question']}\n\n")
                f.write(f"Comprehension Answer: {row['comprehension_answer']}\n\n")
                if row.get('original_question') and row['original_question'] != row['question']:
                    f.write(f"Original Question: {row['original_question']}\n\n")
                f.write(f"Excerpt ({len(str(row['excerpt']))} chars):\n{row['excerpt']}\n")
                f.write(f"\n{'- '*40}\n\n")
        print(f"Saved readable debug to {debug_path}")
    else:
        print(f"No results generated for {domain}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate comprehension MCQA probes from paper text.')
    parser.add_argument('--domain', type=str, help='Specific domain to process. If not provided, all domains are processed.')
    parser.add_argument('--num_questions', type=int, default=None, help='Max number of questions per domain (for testing).')
    args = parser.parse_args()

    domains = ['DPO', 'BOFT', '1_58', 'OFT', 'QLoRA', 'GRPO']

    if args.domain:
        process_domain(args.domain, num_questions=args.num_questions)
    else:
        for domain in domains:
            process_domain(domain, num_questions=args.num_questions)
