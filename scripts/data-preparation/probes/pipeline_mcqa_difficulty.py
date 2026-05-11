"""
Pipeline: Convert Existing Probes to MCQA Format (v1)

Takes existing fact probe CSVs (e.g. probes_v9.csv) and generates MCQA versions
by creating hard distractors for each probe. Works across all domains.

Step 0 (default): GPT-5.4-mini pre-filters probes unsuitable for MCQA, rejecting:
tautological/circular probes, targets that appear verbatim in the probe, trivially
short targets, complex LaTeX formula targets, specific percentages/numeric results,
and binary answer spaces. Use --skip_filtering to bypass.

Outputs:
- probes_{output_version}.csv: input probes filtered to only MCQA-suitable ones
- probes_{output_version}_mcqa.csv: the MCQA formatted version with distractors
- probes_{output_version}_readable.txt: human-readable filtered probes

Usage:
    cd scripts/data-preparation
    python pipeline_mcqa_difficulty.py --probe_type facts --probe_version v10 --output_version v11
    python pipeline_mcqa_difficulty.py --probe_type facts --probe_version v10 --output_version v11 --filter DPO
    python pipeline_mcqa_difficulty.py --probe_type facts --probe_version v10 --output_version v11 --skip_filtering
    python pipeline_mcqa_difficulty.py --probe_type facts --probe_version v13 --output_version v14_restored --filter DPO --row_indices_file restore_indices_v14.txt --skip_filtering
    python pipeline_mcqa_difficulty.py --probe_type facts --output_version v14_restored --filter DPO --append_to_version v14
"""

import os
import sys
import json
import random
from pathlib import Path
import pandas as pd
import concurrent.futures
from tqdm import tqdm
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))
import utils.utils as utils
from utils import probe_paths

# ─────────────────────────────────────────────────────────────
# Step 0: Filter probes for MCQA suitability
# ─────────────────────────────────────────────────────────────

MCQA_FILTER_PROMPT = r"""Decide if this knowledge probe can become a 5-option MCQA.

Reject if any apply:
- Tautological/circular: target restates or is strongly implied by the question wording.
- Target appears verbatim in the question itself. Do not reject target overlap with source/context; source is not shown to the evaluated model.
- Too trivial/generic: single common word, vague adjective, generic phrase completion, or low-concept target with weak distractors.
- Bad 5-option space: yes/no, true/false, increase/decrease, simple severity/comparison, or formula-only completion that cannot support meaningful alternatives.
- Incorrect: probe+target do not match the source.

Accept when target is a concept, method, dataset, model, date, number, proper noun, formula with real alternatives, or specific phrase, and plausible related distractors exist.

Return compact JSON: {"s":true} if suitable, else {"s":false,"r":"<=10 word reason"}."""


def _extract_preceding_context(row: pd.Series, preceding_words: int = 200, following_words: int = 100) -> str:
    """Return the source fact with surrounding context from the section text.

    Includes up to *preceding_words* before the fact and *following_words* after it.
    Falls back to the fact if it cannot be located.
    """
    fact = str(row.get('raw_knowledge_statement', row.get('fact', ''))).strip()
    for col in ('section_text', 'subsection_text'):
        haystack = str(row.get(col, '')).strip()
        if not haystack or not fact:
            continue
        idx = haystack.find(fact[:80])
        if idx >= 0:
            preceding = haystack[:idx]
            following = haystack[idx + len(fact):]
            preceding_context = ' '.join(preceding.split()[-preceding_words:])
            following_context = ' '.join(following.split()[:following_words])
            return f"...{preceding_context} {fact} {following_context}...".strip()
    return fact


def filter_probe_for_mcqa(row: pd.Series) -> dict:
    """Check whether a single probe is suitable for MCQA conversion."""
    target = str(row['target']).strip()
    probe = str(row['probe']).strip()
    source_with_context = _extract_preceding_context(row, preceding_words=75, following_words=25)

    prompt = {
        'system': MCQA_FILTER_PROMPT,
        'user': (f"q: {probe}\n"
                 f"a: {target}\n"
                 f"src: {source_with_context}")
    }

    try:
        response = utils.query_llm(prompt, model='gpt-5.4', reasoning_effort='low',
                                    system_prompt_included=True, return_json=True, max_tokens=80)
        if response is None:
            raise RuntimeError('LLM returned no response')
        data = json.loads(response) if isinstance(response, str) else response
        return {
            'suitable': data.get('s', data.get('suitable', False)),
            'reason': data.get('r', data.get('reason', '')),
        }
    except Exception as e:
        print(f"Filter error: {e}")
        return {'suitable': False, 'reason': f'LLM call failed: {e}'}


# ─────────────────────────────────────────────────────────────
# Step 1: Generate distractors
# ─────────────────────────────────────────────────────────────

DISTRACTOR_PROMPT = r"""Generate 4 plausible but wrong MCQA distractors for an academic-text probe.

Distractor rules:
- Each option should naturally complete the probe.
- Use common confusions or closely related concepts, not random wrong answers.
- Include at least one wrong nearby source-context concept.
- Include at least one broader-field concept outside the immediate context.
- Match target length, style, and specificity where possible.
- Avoid duplicates, obviously wrong options, and anything arguably correct.
- There must be exactly one correct answer: the target.

Return compact JSON: {"d":["...","...","...","..."]}."""


def generate_distractors(row: pd.Series) -> list | None:
    """Generate 4 hard distractors for a single probe using source context."""
    target = str(row['target']).strip()
    probe = str(row['probe']).strip()
    source_with_context = _extract_preceding_context(row, preceding_words=100, following_words=25)

    prompt = {
        'system': DISTRACTOR_PROMPT,
        'user': (f"q: {probe}\n"
                 f"a: {target}\n"
                 f"src: {source_with_context}")
    }

    try:
        response = utils.query_llm(prompt, model='gpt-5.4', reasoning_effort='medium',
                                    system_prompt_included=True, return_json=True, max_tokens=350)
        if response is None:
            raise RuntimeError('LLM returned no response')
        data = json.loads(response) if isinstance(response, str) else response
        distractors = data.get('d', data.get('distractors', []))
        if isinstance(distractors, list) and len(distractors) >= 4:
            return distractors[:4]
    except Exception as e:
        print(f"Error generating distractors: {e}")
    return None


# ─────────────────────────────────────────────────────────────
# Step 2: Verify single correct answer
# ─────────────────────────────────────────────────────────────

VERIFY_PROMPT = r"""Verify this MCQA.

Pass only if all apply:
- Intended option is clearly correct.
- No distractor is arguably correct.
- Distractors are plausible to someone unfamiliar with the source.
- Answer is not leaked or made trivial by the question wording.

Return compact JSON: {"p":true} if it passes, else {"p":false}."""


def verify_mcqa(probe: str, target: str, distractors: list) -> bool:
    """Verify the MCQA has exactly one correct answer."""
    options = [target] + distractors
    random.shuffle(options)
    labels = ['(A)', '(B)', '(C)', '(D)', '(E)']
    correct_label = labels[options.index(target)]
    formatted_options = '\n'.join(f"{labels[i]} {opt}" for i, opt in enumerate(options))

    prompt = {
        'system': VERIFY_PROMPT,
        'user': f"q: {probe}\n{formatted_options}\nintended: {correct_label}"
    }

    try:
        response = utils.query_llm(prompt, model='gpt-5.4-mini', reasoning_effort='low',
                                    system_prompt_included=True, return_json=True, max_tokens=60)
        if response is None:
            raise RuntimeError('LLM returned no response')
        data = json.loads(response) if isinstance(response, str) else response
        return data.get('p', data.get('passes', False))
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
    labels = ['(A)', '(B)', '(C)', '(D)', '(E)']

    correct_label = labels[options.index(target)]
    formatted_options = [f"{labels[i]} {opt}" for i, opt in enumerate(options)]
    formatted_question = probe + '\n' + '\n'.join(formatted_options)

    output = {
        'probe': probe,
        'target': target,
        'correct_label': correct_label,
        'formatted_question': formatted_question,
        'option_a': options[0],
        'option_b': options[1],
        'option_c': options[2],
        'option_d': options[3],
        'option_e': options[4],
        'distractors': json.dumps(distractors),
        'fact': row.get('fact', ''),
        'raw_knowledge_statement': row.get('raw_knowledge_statement', ''),
        'section': row.get('section', ''),
    }
    if 'source_row_0based' in row.index:
        output['source_row_0based'] = row.get('source_row_0based', '')
    return output


# ─────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────

def read_row_indices(row_indices_file: str) -> list[int]:
    """Read 0-based source row indices from a newline/comma separated file."""
    with open(row_indices_file) as f:
        text = f.read()
    pieces = [piece.strip() for piece in text.replace(',', '\n').splitlines()]
    return [int(piece) for piece in pieces if piece]


def write_readable(df: pd.DataFrame, readable_path: str):
    """Write the compact probe: target readable format used by this pipeline."""
    with open(readable_path, 'w') as f:
        for _, row in df.iterrows():
            f.write(f"{row['probe']}: {str(row['target']).lstrip()}\n")


def make_row_key(row: pd.Series) -> tuple:
    """Stable fallback key for de-duplicating probe and MCQA rows."""
    return (
        str(row.get('probe', '')).strip(),
        str(row.get('target', '')).strip(),
        str(row.get('raw_knowledge_statement', row.get('fact', ''))).strip(),
    )


def append_restored_to_version(domain: str, probe_type: str, restored_version: str,
                               append_to_version: str):
    """Append verified restored probes/MCQA rows into an existing canonical version."""
    output_dir = str(probe_paths.resolve_probe_dir(probe_type, domain))
    base_probe_path = os.path.join(output_dir, f'probes_{append_to_version}.csv')
    base_mcqa_path = os.path.join(output_dir, f'probes_{append_to_version}_mcqa.csv')
    restored_probe_path = os.path.join(output_dir, f'probes_{restored_version}.csv')
    restored_mcqa_path = os.path.join(output_dir, f'probes_{restored_version}_mcqa.csv')

    required_paths = [base_probe_path, base_mcqa_path, restored_probe_path, restored_mcqa_path]
    missing = [path for path in required_paths if not os.path.exists(path)]
    if missing:
        print(f"Cannot append restored rows for {domain}; missing files: {missing}")
        return

    base_probes = pd.read_csv(base_probe_path)
    base_mcqa = pd.read_csv(base_mcqa_path)
    restored_probes = pd.read_csv(restored_probe_path)
    restored_mcqa = pd.read_csv(restored_mcqa_path)

    if restored_mcqa.empty:
        print(f"No verified restored MCQA rows to append for {domain}.")
        return

    if 'source_row_0based' in restored_probes.columns and 'source_row_0based' in restored_mcqa.columns:
        verified_sources = set(restored_mcqa['source_row_0based'].dropna().astype(int).tolist())
        restored_probe_rows = restored_probes[
            restored_probes['source_row_0based'].astype(int).isin(verified_sources)
        ].copy()
    else:
        verified_keys = {make_row_key(row) for _, row in restored_mcqa.iterrows()}
        restored_probe_rows = restored_probes[
            restored_probes.apply(lambda row: make_row_key(row) in verified_keys, axis=1)
        ].copy()

    existing_probe_keys = {make_row_key(row) for _, row in base_probes.iterrows()}
    probe_rows_to_append = restored_probe_rows[
        ~restored_probe_rows.apply(lambda row: make_row_key(row) in existing_probe_keys, axis=1)
    ].copy()

    existing_mcqa_keys = {make_row_key(row) for _, row in base_mcqa.iterrows()}
    mcqa_rows_to_append = restored_mcqa[
        ~restored_mcqa.apply(lambda row: make_row_key(row) in existing_mcqa_keys, axis=1)
    ].copy()

    # Keep canonical v14 schemas stable; source_row_0based remains in restored intermediates.
    probe_rows_to_append = probe_rows_to_append.reindex(columns=base_probes.columns)
    mcqa_rows_to_append = mcqa_rows_to_append.reindex(columns=base_mcqa.columns)

    updated_probes = pd.concat([base_probes, probe_rows_to_append], ignore_index=True)
    updated_mcqa = pd.concat([base_mcqa, mcqa_rows_to_append], ignore_index=True)

    updated_probes.to_csv(base_probe_path, index=False)
    updated_mcqa.to_csv(base_mcqa_path, index=False)
    readable_path = os.path.join(output_dir, f'probes_{append_to_version}_readable.txt')
    write_readable(updated_probes, readable_path)

    append_metrics_path = os.path.join(
        output_dir, f'mcqa_append_{restored_version}_to_{append_to_version}.txt'
    )
    with open(append_metrics_path, 'w') as f:
        f.write(f"MCQA Restored Append - {domain} ({restored_version} → {append_to_version})\n")
        f.write(f"{'='*60}\n")
        f.write(f"Restored probes considered: {len(restored_probes)}\n")
        f.write(f"Restored MCQA verified: {len(restored_mcqa)}\n")
        f.write(f"Probe rows appended: {len(probe_rows_to_append)}\n")
        f.write(f"MCQA rows appended: {len(mcqa_rows_to_append)}\n")
        f.write(f"Canonical probes total: {len(updated_probes)}\n")
        f.write(f"Canonical MCQA total: {len(updated_mcqa)}\n")
    print(f"Appended {len(probe_rows_to_append)} probe rows and {len(mcqa_rows_to_append)} MCQA rows for {domain}.")
    print(f"Saved append metrics to {append_metrics_path}")

def process_domain(domain: str, probe_type: str, probe_version: str,
                   output_version: str, skip_filtering: bool = False,
                   row_indices_file: str | None = None,
                   append_to_version: str | None = None,
                   max_workers: int = 16):
    """Convert probes for a single domain to MCQA."""
    if append_to_version:
        append_restored_to_version(domain, probe_type, output_version, append_to_version)
        return

    input_path = str(probe_paths.resolve_probe_path(probe_type, domain, probe_version))

    if not os.path.exists(input_path):
        print(f"Probe file not found: {input_path}")
        return

    df = pd.read_csv(input_path)
    total_input = len(df)
    print(f"Loaded {total_input} probes from {input_path}")

    if row_indices_file:
        row_indices = read_row_indices(row_indices_file)
        if not row_indices:
            print(f"No row indices found in {row_indices_file}. Exiting.")
            return
        invalid = [idx for idx in row_indices if idx < 0 or idx >= len(df)]
        if invalid:
            raise ValueError(f"Invalid row indices for {domain}: {invalid}")
        df = df.iloc[row_indices].copy()
        df.insert(0, 'source_row_0based', row_indices)
        df = df.reset_index(drop=True)
        print(f"Selected {len(df)} restored probes from {row_indices_file}")

    output_dir = str(probe_paths.resolve_probe_dir(probe_type, domain))
    os.makedirs(output_dir, exist_ok=True)

    # Step 0: Filter probes for MCQA suitability
    if not skip_filtering:
        print("Filtering probes for MCQA suitability...")
        filter_results = [None] * len(df)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
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

        filter_debug_path = os.path.join(output_dir, f'mcqa_filter_{output_version}.txt')
        with open(filter_debug_path, 'w') as f:
            f.write(f"MCQA Suitability Filter — {domain} ({probe_type} {probe_version} → {output_version})\n")
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

    # Save filtered probes as the new version
    filtered_path = os.path.join(output_dir, f'probes_{output_version}.csv')
    df.to_csv(filtered_path, index=False)
    print(f"Saved {len(df)} filtered probes to {filtered_path}")

    readable_path = os.path.join(output_dir, f'probes_{output_version}_readable.txt')
    write_readable(df, readable_path)
    print(f"Saved readable probes to {readable_path}")

    # Step 1: Generate distractors in parallel
    print("Generating distractors...")
    distractor_results = [None] * len(df)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(generate_distractors, row): idx
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

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
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

    # Step 3: Format
    mcqa_rows = []
    for idx, distractors in verified_pairs:
        row = df.iloc[idx]
        mcqa_rows.append(format_mcqa_row(row, distractors))

    mcqa_df = pd.DataFrame(mcqa_rows)
    if mcqa_df.empty:
        mcqa_df = pd.DataFrame(columns=[
            'probe', 'target', 'correct_label', 'formatted_question',
            'option_a', 'option_b', 'option_c', 'option_d', 'option_e',
            'distractors', 'fact', 'raw_knowledge_statement', 'section',
            *(['source_row_0based'] if 'source_row_0based' in df.columns else [])
        ])

    # Save MCQA
    output_path = os.path.join(output_dir, f'probes_{output_version}_mcqa.csv')
    mcqa_df.to_csv(output_path, index=False)
    print(f"Saved {len(mcqa_df)} MCQA probes to {output_path}")

    # Save metrics
    metrics_path = os.path.join(output_dir, f'mcqa_metrics_{output_version}.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"MCQA Conversion Metrics - {domain} ({probe_type} {probe_version} → {output_version})\n")
        f.write(f"{'='*60}\n")
        f.write(f"Input probes ({probe_version}): {total_input}\n")
        f.write(f"Probes after MCQA filter ({output_version}): {len(df)}\n")
        f.write(f"Distractors generated: {len(valid_pairs)}\n")
        f.write(f"Passed verification: {len(verified_pairs)}\n")
        f.write(f"Conversion rate: {100*len(verified_pairs)/len(df):.1f}%\n")
    print(f"Saved metrics to {metrics_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert existing probes to MCQA format.')
    parser.add_argument('--probe_type', type=str, default='facts',
                        help='Probe type directory (e.g., facts, inference)')
    parser.add_argument('--probe_version', type=str, default='v10',
                        help='Input probe version (e.g., v10, v9)')
    parser.add_argument('--output_version', type=str, default='v11',
                        help='Output version for filtered probes and MCQA (e.g., v11)')
    parser.add_argument('--filter', type=str, default=None,
                        help='Only process domains containing this string.')
    parser.add_argument('--skip_filtering', action='store_true',
                        help='Skip MCQA suitability pre-filtering step.')
    parser.add_argument('--row_indices_file', type=str, default=None,
                        help='Optional file of 0-based input row indices to process only that subset.')
    parser.add_argument('--append_to_version', type=str, default=None,
                        help='Append verified output_version rows into this canonical version.')
    parser.add_argument('--max_workers', type=int, default=16,
                        help='Parallel workers for filter, distractor generation, and verification.')
    args = parser.parse_args()

    probes_dir = probe_paths.resolve_probe_kind_root(args.probe_type)

    if not probes_dir.exists():
        print(f"Probes directory not found: {probes_dir}")
        sys.exit(1)

    domains = sorted([
        domain for domain in probe_paths.get_all_domains_from_probe_kind(args.probe_type)
        if probe_paths.resolve_probe_path(args.probe_type, domain, args.probe_version).exists()
    ])

    if args.filter:
        domains = [args.filter] if args.filter in domains else [d for d in domains if args.filter in d]

    print(f"Domains to process: {domains}")

    for domain in domains:
        print(f"\n{'='*20} {domain} {'='*20}")
        process_domain(domain, args.probe_type, args.probe_version,
                       output_version=args.output_version,
                       skip_filtering=args.skip_filtering,
                       row_indices_file=args.row_indices_file,
                       append_to_version=args.append_to_version,
                       max_workers=args.max_workers)
