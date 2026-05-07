import argparse
import importlib.util
import json
import random
from pathlib import Path

import pandas as pd

import utils.utils as utils


HARDEN_PROMPT = r"""You will be given:
1. a contiguous excerpt from a paper,
2. an existing multiple-choice comprehension question,
3. its correct answer label.

Your task is to rewrite the question into a **harder** multiple-choice question that is still answerable from the excerpt alone.

### Goal
Create a question that is difficult even for a strong model with the excerpt, and much harder without the excerpt.

### Requirements
- The rewritten question must depend on **at least two distinct claims** from the excerpt.
- It must be **more difficult** than the original question.
- It must be answerable using the excerpt alone, with no outside knowledge.
- The correct option must not be a near-verbatim paraphrase of one sentence from the excerpt.
- The distractors should be **adversarial**:
  - each should sound plausible,
  - each should reuse real concepts or details from the excerpt,
  - but each should be wrong overall.
- Avoid canonical, headline-level paper knowledge.
- Avoid “main contribution” or “what is DPO/QLoRA/etc.” style questions.
- Avoid options that are obviously weaker, shorter, vaguer, or more generic than the correct option.
- Use exactly four substantive options; "None of the above" will be appended later automatically.

### Output format
Return JSON with:
- "question_stem": string
- "options": list of 4 strings
- "correct_index": integer in {0,1,2,3}
- "answer_rationale": short explanation of why the correct option is right from the excerpt
"""


def load_pipeline_module(project_root: Path):
    script_path = project_root / 'scripts' / 'data-preparation' / 'probes' / 'pipeline_generate_comprehension_mcqa.py'
    spec = importlib.util.spec_from_file_location('pipeline_generate_comprehension_mcqa', script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def format_mcqa(question_stem, options, correct_index):
    labels = ['(A)', '(B)', '(C)', '(D)']
    formatted = [f"{labels[i]} {options[i]}" for i in range(4)]
    formatted.append("(E) None of the above")
    return f"{question_stem}\n" + "\n".join(formatted), labels[correct_index]


def harden_row(row, model):
    prompt = {
        'system': HARDEN_PROMPT,
        'user': (
            f"### Excerpt\n{row['excerpt']}\n\n"
            f"### Existing Question\n{row['question']}\n\n"
            f"### Existing Correct Answer\n{row['answer']}"
        ),
    }
    response = utils.query_llm(
        prompt,
        model=model,
        reasoning_effort='medium',
        system_prompt_included=True,
        return_json=True,
        max_tokens=1200,
    )
    data = json.loads(response) if isinstance(response, str) else response
    question_stem = str(data['question_stem']).strip()
    options = [str(opt).strip() for opt in data['options']]
    correct_index = int(data['correct_index'])
    if len(options) != 4 or not (0 <= correct_index < 4):
        raise ValueError('Invalid hardening response format')

    # Shuffle options to avoid positional artifacts while preserving the correct answer.
    paired = list(enumerate(options))
    random.shuffle(paired)
    shuffled_options = [opt for _, opt in paired]
    shuffled_correct_index = next(i for i, (old_idx, _) in enumerate(paired) if old_idx == correct_index)

    mcqa_question, mcqa_answer = format_mcqa(question_stem, shuffled_options, shuffled_correct_index)
    return {
        'question': mcqa_question,
        'answer': mcqa_answer,
        'comprehension_answer': str(data.get('answer_rationale', '')).strip(),
        'hardening_source_question': row['question'],
    }


def main():
    parser = argparse.ArgumentParser(description='Rewrite existing comprehension MCQAs into harder versions.')
    parser.add_argument('--input_csv', type=Path, required=True)
    parser.add_argument('--output_csv', type=Path, default=None)
    parser.add_argument('--paper_title', type=str, required=True)
    parser.add_argument('--model', type=str, default='gpt-5.4')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    project_root = Path(__file__).resolve().parents[3]
    pipeline = load_pipeline_module(project_root)

    df = pd.read_csv(args.input_csv)
    hardened_rows = []
    for _, row in df.iterrows():
        hardened = harden_row(row, args.model)
        question = hardened['question']
        excerpt = row['excerpt']
        hardened_rows.append({
            'question': question,
            'answer': hardened['answer'],
            'fewshot_question': f"{pipeline.FEW_SHOT_EXAMPLES}\nQuestion: {question}\nAnswer:",
            'contextualized_question': pipeline.build_contextualized_question(excerpt, question, args.paper_title),
            'contextualized_fewshot_question': pipeline.build_contextualized_fewshot_question(
                excerpt, question, args.paper_title
            ),
            'original_question': row.get('original_question', ''),
            'comprehension_answer': hardened['comprehension_answer'],
            'excerpt': excerpt,
            'was_refined': row.get('was_refined', False),
            'hardening_source_question': hardened['hardening_source_question'],
        })

    out_df = pd.DataFrame(hardened_rows)
    output_csv = args.output_csv or args.input_csv.with_name(args.input_csv.stem + '_hardened.csv')
    out_df.to_csv(output_csv, index=False)
    print(f'Saved hardened probes to {output_csv}')


if __name__ == '__main__':
    main()
