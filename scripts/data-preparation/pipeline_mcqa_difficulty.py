"""
Pipeline: Convert Existing Probes to MCQA Format (v1)

Takes existing fact probe CSVs (e.g. probes_v9.csv) and generates MCQA versions
by creating hard distractors for each probe. Works across all domains.

Step 0 (default): GPT-5.4-mini pre-filters probes that are unsuitable for MCQA
(e.g. numeric targets, binary answers, trivially short targets). Use --skip_filtering
to bypass this step.

Usage:
    cd scripts/data-preparation
    python pipeline_mcqa_difficulty.py --probe_type facts --probe_version v9
    python pipeline_mcqa_difficulty.py --probe_type facts --probe_version v9 --filter OFT
    python pipeline_mcqa_difficulty.py --probe_type inference --probe_version v7
    python pipeline_mcqa_difficulty.py --probe_type facts --probe_version v9 --skip_filtering
"""

import os
import sys
import json
import random
import pandas as pd
import concurrent.futures
from tqdm import tqdm
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import utils.utils as utils
from utils.pipeline import save_debug_file

# ─────────────────────────────────────────────────────────────
# Step 0: Filter probes for MCQA suitability
# ─────────────────────────────────────────────────────────────

MCQA_FILTER_PROMPT = r"""You are evaluating whether a knowledge probe is suitable for conversion into a multiple-choice question (MCQA).

Some probes work poorly as MCQAs because:
- The target is a number, date, or proper name where meaningful distractors are hard to construct (e.g. "The experiment was conducted in ___" → 2019).
- The target is extremely long or is a full sentence — MCQ options should be concise.
- The target is a trivial word (e.g. "the", "is", "a") that was extracted incorrectly.
- The probe is too vague without context to distinguish among plausible options.
- The answer space is binary or near-binary (yes/no, true/false, increase/decrease) making 4 distinct options unnatural.
- The target is a list of items where partial overlap with distractors would create ambiguity.

A probe IS suitable when:
- The target is a concept, method, technique, term, or short descriptive phrase.
- There exist related-but-wrong concepts in the same field that make plausible distractors.
- A knowledgeable reader could meaningfully reason among 4 options.

You will be given the probe (cloze statement), the target (correct completion), and the source fact.

### Output Format
Provide a JSON object:
- "suitable": (boolean) true if this probe can naturally support 4 meaningful MCQA options
- "reason": (string) one-sentence explanation"""


def filter_probe_for_mcqa(row: pd.Series) -> dict:
    """Check whether a single probe is suitable for MCQA conversion."""
    target = str(row['target']).strip()
    probe = str(row['probe']).strip()
    source = str(row.get('raw_knowledge_statement', row.get('fact', ''))).strip()

    prompt = {
        'system': MCQA_FILTER_PROMPT,
        'user': (f"### Probe Statement\n{probe}\n\n"
                 f"### Correct Answer\n{target}\n\n"
                 f"### Source Fact\n{source}")
    }

    try:
        response = utils.query_llm(prompt, model='gpt-5.4-mini', reasoning_effort='low',
                                    system_prompt_included=True, return_json=True, max_tokens=200)
        data = json.loads(response) if isinstance(response, str) else response
        return {'suitable': data.get('suitable', False), 'reason': data.get('reason', '')}
    except Exception as e:
        print(f"Filter error: {e}")
        return {'suitable': False, 'reason': f'LLM call failed: {e}'}


# ─────────────────────────────────────────────────────────────
# Step 1: Generate distractors
# ─────────────────────────────────────────────────────────────

DISTRACTOR_PROMPT = r"""You are generating distractors for a multiple-choice question derived from an academic text.

You will be given:
- A cloze-style statement (the "probe") that ends right before the answer.
- The correct answer (the "target").
- The source sentence from the paper.

Your task: generate 3 plausible but incorrect distractors.

### Distractor Design Principles
- Each distractor should be a plausible completion of the probe statement.
- Distractors should represent *common misunderstandings* or *closely related concepts* that someone with surface-level knowledge might confuse with the correct answer.
- At least one distractor should come from the same paper/text (a real concept mentioned elsewhere, but wrong in this context).
- Distractors should be similar in length, style, and specificity to the correct answer.
- Do NOT include obviously wrong or unrelated answers.
- Ensure there is exactly ONE correct answer — no distractor should be arguably correct.

### Output Format
Provide a JSON object with a single key "distractors" containing a list of exactly 3 strings."""


def generate_distractors(row: pd.Series, paper_content: str) -> list | None:
    """Generate 3 hard distractors for a single probe."""
    target = str(row['target']).strip()
    probe = str(row['probe']).strip()
    source = str(row.get('raw_knowledge_statement', row.get('fact', ''))).strip()

    prompt = {
        'system': DISTRACTOR_PROMPT,
        'user': (f"### Paper Text (for sourcing plausible distractors)\n{paper_content}\n\n"
                 f"### Probe Statement\n{probe}\n\n"
                 f"### Correct Answer\n{target}\n\n"
                 f"### Source Sentence\n{source}")
    }

    try:
        response = utils.query_llm(prompt, model='gpt-5.4-mini', reasoning_effort='medium',
                                    system_prompt_included=True, return_json=True, max_tokens=500)
        data = json.loads(response) if isinstance(response, str) else response
        distractors = data.get('distractors', [])
        if isinstance(distractors, list) and len(distractors) >= 3:
            return distractors[:3]
    except Exception as e:
        print(f"Error generating distractors: {e}")
    return None


# ─────────────────────────────────────────────────────────────
# Step 2: Verify single correct answer
# ─────────────────────────────────────────────────────────────

VERIFY_PROMPT = r"""You will be given a fill-in-the-blank statement and 4 answer options (A-D). One is the intended correct answer.

Check:
1. Is the intended answer clearly correct given the statement?
2. Could any distractor also be argued as correct? If so, the question fails.
3. Are the distractors plausible enough that someone unfamiliar with the material might choose them?

### Output Format
Provide a JSON object:
- "passes": (boolean) true if exactly one answer is correct and distractors are plausible
- "reason": (string) brief explanation if it fails"""


def verify_mcqa(probe: str, target: str, distractors: list) -> bool:
    """Verify the MCQA has exactly one correct answer."""
    options = [target] + distractors
    random.shuffle(options)
    labels = ['(A)', '(B)', '(C)', '(D)']
    formatted = probe + '\n' + '\n'.join(f"{labels[i]} {opt}" for i, opt in enumerate(options))

    correct_label = labels[options.index(target)]

    prompt = {
        'system': VERIFY_PROMPT,
        'user': f"### Question\n{formatted}\n\n### Intended Correct Answer\n{correct_label}"
    }

    try:
        response = utils.query_llm(prompt, model='gpt-5.4-mini', reasoning_effort='low',
                                    system_prompt_included=True, return_json=True, max_tokens=300)
        data = json.loads(response) if isinstance(response, str) else response
        return data.get('passes', False)
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────
# Step 3: Format final MCQA
# ─────────────────────────────────────────────────────────────

def format_mcqa_row(row: pd.Series, distractors: list) -> dict:
    """Create the final MCQA entry with shuffled options."""
    target = str(row['target']).strip()
    probe = str(row['probe']).strip()

    options = [target] + distractors
    random.shuffle(options)
    labels = ['(A)', '(B)', '(C)', '(D)']

    correct_label = labels[options.index(target)]
    formatted_options = [f"{labels[i]} {opt}" for i, opt in enumerate(options)]
    formatted_question = probe + '\n' + '\n'.join(formatted_options)

    return {
        'probe': probe,
        'target': target,
        'correct_label': correct_label,
        'formatted_question': formatted_question,
        'option_a': options[0],
        'option_b': options[1],
        'option_c': options[2],
        'option_d': options[3],
        'distractors': json.dumps(distractors),
        'fact': row.get('fact', ''),
        'raw_knowledge_statement': row.get('raw_knowledge_statement', ''),
        'section': row.get('section', ''),
    }


# ─────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────

def process_domain(domain: str, probe_type: str, probe_version: str, paper_content: str,
                   skip_filtering: bool = False):
    """Convert probes for a single domain to MCQA."""
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    input_path = os.path.join(base, f'data/probes/{probe_type}/{domain}/probes_{probe_version}.csv')

    if not os.path.exists(input_path):
        print(f"Probe file not found: {input_path}")
        return

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} probes from {input_path}")

    # Step 0: Filter probes for MCQA suitability
    if not skip_filtering:
        print("Filtering probes for MCQA suitability...")
        filter_results = [None] * len(df)

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            future_to_idx = {
                executor.submit(filter_probe_for_mcqa, row): idx
                for idx, (_, row) in enumerate(df.iterrows())
            }
            for future in tqdm(concurrent.futures.as_completed(future_to_idx),
                               total=len(df), desc="Filtering"):
                idx = future_to_idx[future]
                try:
                    filter_results[idx] = future.result()
                except Exception as e:
                    print(f"Filter error at index {idx}: {e}")
                    filter_results[idx] = {'suitable': False, 'reason': f'Exception: {e}'}

        suitable_mask = [r['suitable'] if r else False for r in filter_results]
        rejected = [(i, r['reason']) for i, r in enumerate(filter_results) if r and not r['suitable']]

        output_dir = os.path.join(base, f'data/probes/{probe_type}/{domain}/')
        os.makedirs(output_dir, exist_ok=True)
        filter_debug_path = os.path.join(output_dir, f'mcqa_filter_{probe_version}.txt')
        with open(filter_debug_path, 'w') as f:
            f.write(f"MCQA Suitability Filter — {domain} ({probe_type} {probe_version})\n")
            f.write(f"{'='*60}\n")
            f.write(f"Total probes: {len(df)}\n")
            f.write(f"Suitable: {sum(suitable_mask)}\n")
            f.write(f"Rejected: {len(rejected)}\n\n")
            for idx, reason in rejected:
                row = df.iloc[idx]
                f.write(f"--- Probe {idx} ---\n")
                f.write(f"Probe: {str(row['probe']).strip()}\n")
                f.write(f"Target: {str(row['target']).strip()}\n")
                f.write(f"Reason: {reason}\n\n")
        print(f"Saved filter debug to {filter_debug_path}")

        df = df[suitable_mask].reset_index(drop=True)
        print(f"After filtering: {len(df)} suitable probes (rejected {len(rejected)})")

        if len(df) == 0:
            print("No probes passed MCQA suitability filter. Exiting.")
            return

    # Step 1: Generate distractors in parallel
    print("Generating distractors...")
    distractor_results = [None] * len(df)

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        future_to_idx = {
            executor.submit(generate_distractors, row, paper_content): idx
            for idx, (_, row) in enumerate(df.iterrows())
        }
        for future in tqdm(concurrent.futures.as_completed(future_to_idx),
                           total=len(df), desc="Generating distractors"):
            idx = future_to_idx[future]
            try:
                distractor_results[idx] = future.result()
            except Exception as e:
                print(f"Error at index {idx}: {e}")

    # Filter out failures
    valid_pairs = [(idx, distractors) for idx, distractors in enumerate(distractor_results) if distractors is not None]
    print(f"Generated distractors for {len(valid_pairs)}/{len(df)} probes.")

    # Step 2: Verify in parallel
    print("Verifying MCQA questions...")
    verified_pairs = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        future_to_pair = {
            executor.submit(verify_mcqa, str(df.iloc[idx]['probe']).strip(),
                          str(df.iloc[idx]['target']).strip(), distractors): (idx, distractors)
            for idx, distractors in valid_pairs
        }
        for future in tqdm(concurrent.futures.as_completed(future_to_pair),
                           total=len(valid_pairs), desc="Verifying"):
            idx, distractors = future_to_pair[future]
            try:
                if future.result():
                    verified_pairs.append((idx, distractors))
            except Exception as e:
                print(f"Verification error: {e}")

    print(f"Verified {len(verified_pairs)}/{len(valid_pairs)} MCQA questions.")

    if not verified_pairs:
        print("No MCQA passed verification. Exiting.")
        return

    # Step 3: Format
    mcqa_rows = []
    for idx, distractors in verified_pairs:
        row = df.iloc[idx]
        mcqa_rows.append(format_mcqa_row(row, distractors))

    mcqa_df = pd.DataFrame(mcqa_rows)

    # Save
    output_dir = os.path.join(base, f'data/probes/{probe_type}/{domain}/')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'probes_{probe_version}_mcqa.csv')
    mcqa_df.to_csv(output_path, index=False)
    print(f"Saved {len(mcqa_df)} MCQA probes to {output_path}")

    # Save metrics
    metrics_path = os.path.join(output_dir, f'mcqa_metrics_{probe_version}.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"MCQA Conversion Metrics - {domain} ({probe_type} {probe_version})\n")
        f.write(f"{'='*60}\n")
        f.write(f"Probes after MCQA filter: {len(df)}\n")
        f.write(f"Distractors generated: {len(valid_pairs)}\n")
        f.write(f"Passed verification: {len(verified_pairs)}\n")
        f.write(f"Conversion rate: {100*len(verified_pairs)/len(df):.1f}%\n")
    print(f"Saved metrics to {metrics_path}")


def load_paper_content(domain: str) -> str:
    """Load the source paper/document for a domain."""
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    # Try arxiv first, then legal, then medical
    for subdir, ext in [('arxiv/cleaned', '.tex'), ('legal/cleaned', '.txt'), ('medical/cleaned', '.txt')]:
        path = os.path.join(base, f'data/{subdir}/{domain}{ext}')
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()

    print(f"Warning: could not find source text for domain '{domain}'")
    return ""


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert existing probes to MCQA format.')
    parser.add_argument('--probe_type', type=str, default='facts',
                        help='Probe type directory (e.g., facts, inference)')
    parser.add_argument('--probe_version', type=str, default='v9',
                        help='Probe version (e.g., v9, v7)')
    parser.add_argument('--filter', type=str, default=None,
                        help='Only process domains containing this string.')
    parser.add_argument('--skip_filtering', action='store_true',
                        help='Skip MCQA suitability pre-filtering step.')
    args = parser.parse_args()

    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    probes_dir = os.path.join(base, f'data/probes/{args.probe_type}/')

    if not os.path.isdir(probes_dir):
        print(f"Probes directory not found: {probes_dir}")
        sys.exit(1)

    domains = sorted([
        d for d in os.listdir(probes_dir)
        if os.path.isdir(os.path.join(probes_dir, d))
        and os.path.exists(os.path.join(probes_dir, d, f'probes_{args.probe_version}.csv'))
    ])

    if args.filter:
        domains = [d for d in domains if args.filter in d]

    print(f"Domains to process: {domains}")

    for domain in domains:
        print(f"\n{'='*20} {domain} {'='*20}")
        paper_content = load_paper_content(domain)
        process_domain(domain, args.probe_type, args.probe_version, paper_content,
                       skip_filtering=args.skip_filtering)
