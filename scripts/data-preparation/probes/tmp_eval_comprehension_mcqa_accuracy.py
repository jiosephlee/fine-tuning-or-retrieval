import argparse
import json
import re
from pathlib import Path

import pandas as pd

import utils.utils as utils


EVAL_PROMPT = """You will be given an excerpt and a multiple-choice question. Answer the question using only the excerpt.

Instructions:
- Choose exactly one answer from (A), (B), (C), (D), or (E).
- Return JSON with a single key "answer".
- Do not use outside knowledge.

Output format:
{"answer": "(A)"}
"""

DEFAULT_DOMAINS = [
    'DPO',
    'Santos_v_Kimmel',
    'Multiphasic_anaphylaxis_in_the_emergency_and_intensive_care',
]

DEFAULT_MODELS = [
    'gpt-4.1-nano',
    'gpt-3.5-turbo',
]


def normalize_answer_label(raw_answer):
    text = str(raw_answer).strip()
    match = re.search(r'\(([A-E])\)|\b([A-E])\b', text)
    if not match:
        return None
    label = match.group(1) or match.group(2)
    return f"({label})"


def infer_source(domain):
    arxiv_path = Path('probes/arxiv') / domain / 'inference' / 'comprehension_mcqa.csv'
    legal_path = Path('probes/legal') / domain / 'inference' / 'comprehension_mcqa.csv'
    medical_path = Path('probes/medical') / domain / 'inference' / 'comprehension_mcqa.csv'
    if arxiv_path.exists():
        return 'arxiv'
    if legal_path.exists():
        return 'legal'
    if medical_path.exists():
        return 'medical'
    raise FileNotFoundError(f'Could not find comprehension MCQA CSV for domain {domain}')


def resolve_csv_path(domain):
    source = infer_source(domain)
    return Path('probes') / source / domain / 'inference' / 'comprehension_mcqa.csv'


def answer_question(model, excerpt, question):
    prompt = {
        'system': EVAL_PROMPT,
        'user': f"### Excerpt\n{excerpt}\n\n### Question\n{question}"
    }
    response = utils.query_llm(
        prompt,
        model=model,
        system_prompt_included=True,
        return_json=True,
        max_tokens=128,
    )
    data = json.loads(response) if isinstance(response, str) else response
    return normalize_answer_label(data.get('answer'))


def evaluate_domain(domain, model):
    csv_path = resolve_csv_path(domain)
    df = pd.read_csv(csv_path)
    rows = []
    for idx, row in df.iterrows():
        predicted = None
        error = None
        try:
            predicted = answer_question(model, row['excerpt'], row['question'])
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
        correct = predicted == row['answer']
        rows.append({
            'domain': domain,
            'model': model,
            'row_index': idx,
            'gold_answer': row['answer'],
            'predicted_answer': predicted,
            'correct': correct,
            'error': error,
            'question_stem': str(row['question']).splitlines()[0],
        })

    out_df = pd.DataFrame(rows)
    detail_path = csv_path.parent / f"comprehension_mcqa_eval_{model.replace('.', '_')}.csv"
    out_df.to_csv(detail_path, index=False)

    accuracy = out_df['correct'].mean() if len(out_df) else 0.0
    return {
        'domain': domain,
        'model': model,
        'rows': len(out_df),
        'correct': int(out_df['correct'].sum()),
        'accuracy': float(accuracy),
        'errors': int(out_df['error'].notna().sum()),
        'detail_path': str(detail_path),
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate comprehension MCQA accuracy with external models.')
    parser.add_argument('--domains', nargs='*', default=DEFAULT_DOMAINS)
    parser.add_argument('--models', nargs='*', default=DEFAULT_MODELS)
    args = parser.parse_args()

    summaries = []
    for domain in args.domains:
        for model in args.models:
            summary = evaluate_domain(domain, model)
            summaries.append(summary)
            print(
                f"{domain} | {model} | {summary['correct']}/{summary['rows']} "
                f"({summary['accuracy']:.3f}) | errors={summary['errors']}"
            )

    summary_df = pd.DataFrame(summaries)
    summary_path = Path('reports') / 'comprehension_mcqa_eval_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary to {summary_path}")


if __name__ == '__main__':
    main()
