"""Debug script: run only the extract_atomic_facts extraction step on DPO checkpoint data."""
import sys, os, json, concurrent.futures
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
import pandas as pd
from tqdm import tqdm
import utils.utils as utils
from pipeline_fact_probe import generate_probes_for_paper  # we'll pull the prompts from here

# Load checkpoint
checkpoint_path = '../../data/probes/facts/DPO/checkpoints/07_knowledge_kept.csv'
paper_df = pd.read_csv(checkpoint_path)
print(f"Loaded {len(paper_df)} rows from checkpoint")

# Derive title from paper
paper_df['title'] = "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"

# Load paper for context
with open('../../data/arxiv/cleaned/DPO.tex', 'r') as f:
    paper = f.read()

# Pull the extraction prompt directly from pipeline
extraction_prompt_system = r"""You will be given two inputs, a section of an academic paper for context and a single sentence drawn from that section. Papers often interweave various pieces of knowledge together in academic writing. While each sentence is interwoven with others, there is atomic knowledge that can be extracted from a particular sentence. Write questions that tests for this atomic knowledge. Specifically, your task is to extract questions from the provided sentence with clear answers.

Extract 1-3 questions from the sentence.

### Detailed Instructions
Consider these instructions as you extract each question:
- The question should be natural and meaningful, in which the answer is considered a main fact presented by the sentence.
- The answer should be non-trivial and non-obvious. It should not be plainly obvious from the question for someone who has no relevant knowledge.
- The answer to the question MUST be a verbatim word, phrase (2-5 words), or a mathematical expression copied exactly from the sentence. Do not paraphrase, rephrase, or adjust the answer — it must appear as an exact substring of the sentence. You should strip leading determiners such as "some", "a", "an", or "the" from the answer.
- The question should have a a clear, single answer and *NOT* multiple valid answers.
- Each question should be written separately and independently of the other questions, so don't reference other questions in the same question.

### Handling Mathematical Sentences
Many sentences in academic papers contain equations, variables, or mathematical notation. These sentences still contain important knowledge, but you must extract it carefully:
- Do NOT extract partial equations or incomplete equation fragments as answers. For example, "$\pi_{\theta}(y\mid x" is NOT a valid answer because it is a broken fragment.
- If the answer is a mathematical expression, it MUST be a complete, self-contained expression exactly as written in the source. Valid examples: "$\beta$", "$Z(x)$", "$\pi_\theta$", "$\mathcal{L}_\text{DPO}$". Invalid examples: "$\pi_{\theta}(y\mid x", "$\frac{\exp(r^*(x, y_1))}{\exp(r^*(x, y_1)) + \exp(r^*(x, y_2".
- For equation-heavy sentences, prefer questions that test conceptual understanding over equation completion:
    - What a symbol or term REPRESENTS (e.g., "what does $\beta$ control?")
    - What ROLE a component plays in an equation (e.g., "what prevents the policy from diverging?")
    - What two things are being RELATED by an equation
    - What a named result or equation DEFINES
- Prefer natural language answers when they capture the same knowledge as a mathematical answer. For example, prefer "the partition function" over "$Z(x)$" if both are valid.
- Preserve the original LaTeX formatting exactly. Use $...$ delimiters as they appear in the source. Do NOT convert to \(...\) or other formats.

### Demonstration 1: Natural Language Sentence
Context: "\\title{Direct Preference Optimization: Your Language Model is Secretly a Reward Model}\n\\subsection{Can DPO scale to real preference datasets?}\nNext, we evaluate fine-tuning performance of DPO on summarization and single-turn dialogue. For summarization, automatic evaluation metrics such as ROUGE can be poorly correlated with human preferences~\citep{stiennon2022learning}, and prior work has found that fine-tuning LMs using PPO on human preferences to provide more effective summaries. We evaluate different methods by sampling completions on the test split of TL;DR summarization dataset, and computing the average win rate against reference completions in the test set."

Sentence: "We evaluate different methods by sampling completions on the test split of TL;DR summarization dataset, and computing the average win rate against reference completions in the test set."

Questions:
- "The authors evaluate DPO's fine-tuning performance against other methods on summarization by sampling completions on the test split of what dataset?", Answer: "TL;DR summarization"
- "The fine-tuning performance of DPO and other methods on summarization are evaluated by sampling completions on the test split of the TL;DR summarization dataset and computing the average win rate against what?", Answer: "reference completions"

### Demonstration 2: Natural Language Sentence
Context: "\title{Direct Preference Optimization: Your Language Model is Secretly a Reward Model}\nWhile large-scale unsupervised language models (LMs) learn broad world knowledge and some reasoning skills, achieving precise control of their behavior is difficult due to the completely unsupervised nature of their training. Existing methods for gaining such steerability collect human labels of the relative quality of model generations and fine-tune the unsupervised LM to align with these preferences, often with reinforcement learning from human feedback (RLHF)."

Sentence: "Existing methods for gaining such steerability collect human labels of the relative quality of model generations and fine-tune the unsupervised LM to align with these preferences, often with reinforcement learning from human feedback (RLHF)."

Questions:
- "What do existing methods collect to steer unsupervised language models?", Answer: "human labels"
- "Existing methods for steering unsupervised language models collect human labels of the quality of what?", Answer: "model generations"
- "Existing methods align unsupervised language models by fine-tuning on what?", Answer: "human preferences"
- "Existing methods for steering unsupervised language models via fine-tuning on human preferences often use what?", Answer: "RLHF"

### Demonstration 3: Mathematical Sentence
Context: "\title{Direct Preference Optimization: Your Language Model is Secretly a Reward Model}\nFollowing prior works, the optimization is formulated as\n\\begin{equation}\n\\max_{\\pi_{\\theta}}  \\mathbb{E}_{x\\sim \\mathcal{D}, y\\sim \\pi_{\\theta}(y \\mid x)}\\bigl[r_{\\phi}(x, y)\\bigr] - \\beta\\mathbb{D}_{\\textrm{KL}}\\bigl[\\pi_{\\theta}(y\\mid x)\\mid \\mid \\pi_\\text{ref}(y\\mid x)\\bigr],\n\\end{equation}\nwhere $\\beta$ is a parameter controlling the deviation from the base reference policy $\\pi_\\text{ref}$, namely the initial SFT model $\\pi^\\text{SFT}$."

Sentence: "where $\\beta$ is a parameter controlling the deviation from the base reference policy $\\pi_\\text{ref}$, namely the initial SFT model $\\pi^\\text{SFT}$."

Questions:
- "In the RL fine-tuning objective, what does the parameter $\\beta$ control?", Answer: "deviation from the base reference policy"
- "In the RL fine-tuning objective, the base reference policy $\\pi_\\text{ref}$ is the initial model trained with what method?", Answer: "SFT"

Note: A BAD question here would be "The expectation is taken under what distribution?", Answer: "$\\pi_{\\theta}(y\\mid x" — this is a broken LaTeX fragment and tests notation recall, not understanding.
"""

json_parse_prompt_system = """You will be given a string containing questions and answers. Convert it into a JSON object with a single key "list_of_questions", which contains a list of objects. Each object in the list should have a "question" and "answer" key. The format should be: {"list_of_questions": [{"question": "...", "answer": "..."}, ...]}. Copy the question and answer content exactly as it appears. Do not modify the text."""


def extract_context_and_sentence(row):
    """Extract surrounding context for a sentence."""
    parts = row['subsection_text'].strip().split(row['raw_knowledge_statement'].strip())
    context_before = parts[0]
    if len(parts) > 1:
        remaining_text = parts[1]
        paragraph_end = remaining_text.find('\n\n')
        if paragraph_end != -1:
            rest_of_paragraph = remaining_text[:paragraph_end]
            after_paragraph = remaining_text[paragraph_end:].lstrip('\n')
            next_paragraph_end = after_paragraph.find('\n\n')
            if next_paragraph_end != -1:
                extra_context = after_paragraph[:next_paragraph_end]
            else:
                extra_context = after_paragraph
            context = context_before + row['raw_knowledge_statement'].strip() + rest_of_paragraph + '\n\n' + extra_context
        else:
            rest_of_paragraph = remaining_text
            context = context_before + row['raw_knowledge_statement'].strip() + rest_of_paragraph
    else:
        context = context_before
    if context.startswith('\\title{'):
        lines = context.split('\n')
        title_end = 0
        for i, line in enumerate(lines):
            if '}' in line:
                title_end = i + 1
                break
        context = '\n'.join(lines[title_end:]).strip()
    return context


def extract_only(row):
    """Run just the extraction + JSON parse step."""
    prompt = {'system': extraction_prompt_system}
    context = extract_context_and_sentence(row)
    prompt['user'] = f"""### Title\n{row['title']}\n### Context\n{context}\n\n### Sentence\n{row['raw_knowledge_statement'].strip()}"""

    raw_output = utils.query_llm(prompt, model='gpt-5.4-mini', reasoning_effort='medium')

    # Parse into structured QA pairs
    json_parse_prompt = {'system': json_parse_prompt_system, 'user': raw_output}
    try:
        json_output = utils.query_llm(json_parse_prompt, model='gpt-5.4-nano', return_json=True, reasoning_effort='low')
        qa_pairs_data = json.loads(json_output)
        qa_pairs = qa_pairs_data.get('list_of_questions', [])
    except (json.JSONDecodeError, TypeError):
        qa_pairs = []

    return {
        'sentence': row['raw_knowledge_statement'].strip(),
        'raw_output': raw_output,
        'qa_pairs': qa_pairs
    }


# Run on all rows
print(f"\nRunning extraction on {len(paper_df)} sentences...")
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    results = list(tqdm(executor.map(extract_only, [paper_df.iloc[i] for i in range(len(paper_df))]), total=len(paper_df)))

# Save debug output
output_path = '../../data/probes/facts/DPO/debug_extraction.txt'
total_qa = 0
math_qa = 0
with open(output_path, 'w') as f:
    for i, r in enumerate(results):
        f.write(f"{'='*80}\n")
        f.write(f"[{i}] SENTENCE: {r['sentence']}\n\n")
        for j, pair in enumerate(r['qa_pairs']):
            q = pair.get('question', '???')
            a = pair.get('answer', '???')
            # Check if answer is verbatim in sentence
            verbatim = a.strip() in r['sentence']
            # Check if answer looks like math
            is_math = '$' in a or '\\' in a or '{' in a
            marker = ""
            if not verbatim:
                marker += " [NOT VERBATIM]"
            if is_math and ('=' in a or len(a) > 30):
                marker += " [EQUATION TARGET]"
            if is_math and (a.count('$') % 2 != 0 or a.count('{') != a.count('}')):
                marker += " [BROKEN LATEX]"
            f.write(f"  Q{j+1}: {q}\n")
            f.write(f"  A{j+1}: {a}{marker}\n\n")
            total_qa += 1
            if is_math:
                math_qa += 1

print(f"\nDone! {total_qa} QA pairs extracted ({math_qa} with math)")
print(f"Output saved to {output_path}")
print(f"\nQuick stats:")

# Count issues
not_verbatim = 0
broken_latex = 0
equation_target = 0
for r in results:
    for pair in r['qa_pairs']:
        a = pair.get('answer', '')
        if a.strip() not in r['sentence']:
            not_verbatim += 1
        if '$' in a or '\\' in a or '{' in a:
            if a.count('$') % 2 != 0 or a.count('{') != a.count('}'):
                broken_latex += 1
            if '=' in a or len(a) > 30:
                equation_target += 1

print(f"  Not verbatim in sentence: {not_verbatim}")
print(f"  Broken LaTeX: {broken_latex}")
print(f"  Equation targets (= or long math): {equation_target}")
