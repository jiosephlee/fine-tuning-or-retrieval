import os
import sys
import json
import pandas as pd
import argparse
import concurrent.futures
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils.utils as utils

def find_additional_supporting_sentences(probe: str, target: str, text_sentences: list, paper_text: str) -> list:
    """Ask LLM to find 1-2 additional sentences from the paper that directly support the answer."""
    system_prompt = """You will be given:
1. A cloze-style probe statement (with a blank to fill)
2. The target answer that fills in the blank
3. The current supporting sentences from the paper
4. The full paper text

Your task is to identify 1-2 additional sentences from the paper that directly support the answer. These sentences should provide explicit evidence or context that makes the answer clearer or more justified.

Rules:
- Only include sentences that DIRECTLY support the answer
- The sentences must exist verbatim in the paper (you can extract them exactly)
- If no additional supporting sentences exist, return an empty list
- Return at most 2 additional sentences
- Do not include sentences that are already in the current supporting sentences

### Output Format
Provide a JSON object with a single key "additional_sentences" which is a list of strings (can be empty)."""

    user_prompt = f"""### Probe Statement (with blank)
{probe} ___

### Target Answer
{target}

### Current Supporting Sentences
{json.dumps(text_sentences, indent=2)}

### Full Paper Text
{paper_text}
"""

    prompt = {'system': system_prompt, 'user': user_prompt}
    response = utils.query_llm(prompt, model='openai-gpt-5', reasoning_effort='low', system_prompt_included=True, return_json=True, max_tokens=2000, is_hippa=True)
    
    try:
        data = json.loads(response) if isinstance(response, str) else response
        additional_sentences = data.get('additional_sentences', [])
        if isinstance(additional_sentences, list):
            return additional_sentences
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"Failed to parse additional sentences response: {e}")
    
    return []

def assess_probe_quality(text_sentences: list, probe: str, target: str) -> dict:
    """Ask LLM to assess if a probe should be deleted due to quality issues."""
    system_prompt = """You will be given:
1. Supporting sentences from a paper
2. A cloze-style probe statement
3. The target answer

Your task is to determine if the probe should be DELETED due to quality issues:

### Delete if:
- The probe is imprecise, unclear, ambiguous, or poorly formulated
- The answer is not clearly supported by the evidence or could have multiple valid answers
- The probe statement is confusing or doesn't make sense
- The probe is too trivial or doesn't test meaningful understanding

### Keep if:
- The probe is clear, precise, and well-formulated
- There is one unambiguous answer that is clearly the target
- The answer is supported by the evidence
- The probe tests meaningful comprehension or inference

### Output Format
Provide a JSON object with two keys:
- "should_delete": boolean (true if probe should be deleted, false if it should be kept)
- "reason": string (brief explanation for your decision)"""

    user_prompt = f"""### Supporting Sentences
{json.dumps(text_sentences, indent=2)}

### Probe Statement (with blank)
{probe} ___

### Target Answer
{target}
"""

    prompt = {'system': system_prompt, 'user': user_prompt}
    response = utils.query_llm(prompt, model='openai-gpt-5', reasoning_effort='minimal', system_prompt_included=True, return_json=True, max_tokens=800, is_hippa=True)
    
    try:
        data = json.loads(response) if isinstance(response, str) else response
        return {
            'should_delete': data.get('should_delete', False),
            'delete_reason': data.get('reason', '')
        }
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"Failed to parse quality assessment response: {e}")
        return {
            'should_delete': False,
            'delete_reason': 'Error in parsing'
        }

def relabel_inference_type(text_sentences: list, current_type: str, probe: str, target: str) -> dict:
    """Ask LLM to relabel the inference type based on the supporting sentences."""
    system_prompt = """You will be given:
1. Supporting sentences from a paper
2. The current inference type label
3. A cloze-style probe statement
4. The target answer

Your task is to categorize the inference type into ONE of the following five categories:

1. **Conceptual Synthesis**: The answer is pretty much in the supporting sentences, but the probe is a new way of asking for it. The reader must synthesize or combine information from the supporting text.

2. **Mathematical Understanding**: The probe assigns an interpretation to an equation or mathematical expression that doesn't exist explicitly in the supporting sentences. It requires understanding what a mathematical term or formula conceptually represents.

3. **Analogy**: The probe requires drawing an analogy from the knowledge in the paper. It connects the paper's concepts to something else (often simpler or more intuitive).

4. **Predicting Hypothetical**: The probe changes a condition in the facts to predict a new outcome. It requires applying the paper's principles to a hypothetical scenario.

5. **New Insight**: The probe identifies a new insight or observation, such as the causal mechanism or an implicit assumption, in the facts that is not stated explicitly at all. It could require recognizing unstated conditions or premises or stating something that is not explicitly stated in the paper. This is a general category that overlaps with mathematical understanding, analogy, and predicting hypotheticals, and so any insights that are not covered by these three categories should be categorized as new insight.

### Guidelines
- Read the supporting sentences carefully
- Consider how the probe relates to these sentences
- Choose the SINGLE category that best fits
- Provide a brief explanation (1-2 sentences) for your choice

### Output Format
Provide a JSON object with two keys:
- "inference_type": one of the five categories above (exact string)
- "explanation": brief justification for the choice"""

    user_prompt = f"""### Supporting Sentences
{json.dumps(text_sentences, indent=2)}

### Current Inference Type
{current_type}

### Probe Statement (with blank)
{probe} ___

### Target Answer
{target}
"""

    prompt = {'system': system_prompt, 'user': user_prompt}
    response = utils.query_llm(prompt, model='openai-gpt-5', reasoning_effort='low', system_prompt_included=True, return_json=True, max_tokens=1000, is_hippa=True)
    
    try:
        data = json.loads(response) if isinstance(response, str) else response
        return {
            'new_type': data.get('inference_type', current_type),
            'explanation': data.get('explanation', '')
        }
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"Failed to parse relabel response: {e}")
        return {
            'new_type': current_type, 
            'explanation': 'Error in parsing'
        }

def process_probe_row(row, paper_text, domain):
    """Process a single probe row."""
    probe = row['probe']
    target = row['target']
    
    # Parse text_sentences - it's stored as a string representation of a list
    text_sentences_str = row['text_sentences']
    try:
        if isinstance(text_sentences_str, str):
            text_sentences = json.loads(text_sentences_str.replace("'", '"'))
        else:
            text_sentences = text_sentences_str
    except (json.JSONDecodeError, TypeError, AttributeError):
        text_sentences = [text_sentences_str] if text_sentences_str else []
    
    current_type = row['inference_type']
    
    # Execute all three LLM calls in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        # Step 1: Find additional supporting sentences
        future_additional = executor.submit(find_additional_supporting_sentences, probe, target, text_sentences, paper_text)
        
        # Step 2 & 3: Assess quality and relabel type (can start with original sentences)
        future_quality = executor.submit(assess_probe_quality, text_sentences, probe, target)
        future_relabel = executor.submit(relabel_inference_type, text_sentences, current_type, probe, target)
        
        # Wait for all to complete
        additional_sentences = future_additional.result()
        quality_result = future_quality.result()
        relabel_result = future_relabel.result()
    
    # Determine final inference type based on deletion status
    all_sentences = text_sentences + additional_sentences
    if not quality_result['should_delete']:
        new_type = relabel_result['new_type']
        explanation = relabel_result['explanation']
    else:
        new_type = current_type
        explanation = 'Not relabeled (marked for deletion)'
    
    return {
        'domain': domain,
        'probe': probe,
        'target': target,
        'original_text_sentences': text_sentences,
        'additional_sentences': additional_sentences,
        'all_sentences': all_sentences,
        'original_inference_type': current_type,
        'new_inference_type': new_type,
        'explanation': explanation,
        'should_delete': quality_result['should_delete'],
        'delete_reason': quality_result['delete_reason']
    }

def process_domain(domain: str, base_dir: str):
    """Process all probes for a single domain."""
    print(f"\n{'='*80}")
    print(f"Processing domain: {domain}")
    print(f"{'='*80}")
    
    # Load the probes CSV
    probes_path = os.path.join(base_dir, 'probes', 'inference', domain, 'probes_v7.csv')
    if not os.path.exists(probes_path):
        print(f"Warning: {probes_path} does not exist. Skipping.")
        return []
    
    # Load the cleaned paper text
    paper_path = os.path.join(base_dir, 'arxiv', 'cleaned', f'{domain}.tex')
    if not os.path.exists(paper_path):
        print(f"Warning: {paper_path} does not exist. Skipping.")
        return []
    
    print(f"Loading probes from {probes_path}")
    probes_df = pd.read_csv(probes_path)
    print(f"Found {len(probes_df)} probes")
    
    print(f"Loading paper text from {paper_path}")
    with open(paper_path, 'r', encoding='utf-8') as f:
        paper_text = f.read()
    print(f"Paper text loaded ({len(paper_text)} characters)")
    
    # Process each probe
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_to_idx = {
            executor.submit(process_probe_row, row, paper_text, domain): idx 
            for idx, row in probes_df.iterrows()
        }
        
        for future in tqdm(concurrent.futures.as_completed(future_to_idx), 
                          total=len(probes_df), 
                          desc=f"Processing {domain} probes"):
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                idx = future_to_idx[future]
                print(f"Probe at index {idx} generated an exception: {exc}")
    
    return results

def format_results_for_report(all_results: list) -> str:
    """Format results into a readable text report."""
    report_lines = []
    report_lines.append("=" * 100)
    report_lines.append("INFERENCE PROBE ANALYSIS REPORT")
    report_lines.append("=" * 100)
    report_lines.append("")
    
    for result in all_results:
        report_lines.append("-" * 100)
        report_lines.append(f"DOMAIN: {result['domain']}")
        report_lines.append("-" * 100)
        report_lines.append("")
        report_lines.append(f"PROBE: {result['probe']} ___")
        report_lines.append(f"TARGET: {result['target']}")
        report_lines.append("")
        report_lines.append("ORIGINAL SUPPORTING SENTENCES:")
        for i, sent in enumerate(result['original_text_sentences'], 1):
            report_lines.append(f"  {i}. {sent}")
        report_lines.append("")
        
        if result['additional_sentences']:
            report_lines.append("ADDITIONAL SUPPORTING SENTENCES FOUND:")
            for i, sent in enumerate(result['additional_sentences'], 1):
                report_lines.append(f"  {i}. {sent}")
        else:
            report_lines.append("ADDITIONAL SUPPORTING SENTENCES FOUND: None")
        report_lines.append("")
        
        # Deletion recommendation
        if result['should_delete']:
            report_lines.append("*** RECOMMENDATION: DELETE THIS PROBE ***")
            report_lines.append(f"REASON: {result['delete_reason']}")
            report_lines.append("")
        else:
            report_lines.append("RECOMMENDATION: KEEP")
            report_lines.append("")
        
        report_lines.append(f"ORIGINAL INFERENCE TYPE: {result['original_inference_type']}")
        report_lines.append(f"NEW INFERENCE TYPE: {result['new_inference_type']}")
        
        if result['original_inference_type'] != result['new_inference_type']:
            report_lines.append("  *** TYPE CHANGED ***")
        
        report_lines.append(f"EXPLANATION: {result['explanation']}")
        report_lines.append("")
        report_lines.append("")
    
    return "\n".join(report_lines)

def update_csv_files(all_results: list, base_dir: str):
    """Update CSV files with the analysis results."""
    # Group results by domain
    results_by_domain = {}
    for result in all_results:
        domain = result['domain']
        if domain not in results_by_domain:
            results_by_domain[domain] = []
        results_by_domain[domain].append(result)
    
    # Update each domain's CSV
    for domain, domain_results in results_by_domain.items():
        probes_path = os.path.join(base_dir, 'probes', 'inference', domain, 'probes_v7.csv')
        
        # Load the original CSV
        original_df = pd.read_csv(probes_path)
        print(f"\n{'='*60}")
        print(f"Updating {domain}: {len(original_df)} original probes")
        
        # Create a mapping from (probe, target) to result
        result_map = {(r['probe'], r['target']): r for r in domain_results}
        
        # Track which rows to keep
        rows_to_keep = []
        updated_rows = []
        
        for idx, row in original_df.iterrows():
            probe = row['probe']
            target = row['target']
            
            # Find corresponding result
            result = result_map.get((probe, target))
            
            if result and result['should_delete']:
                # Skip this row (delete it)
                print(f"  Deleting: {probe[:50]}...")
                continue
            
            # Keep the row and update it
            updated_row = row.copy()
            
            if result:
                # Update inference_type
                updated_row['inference_type'] = result['new_inference_type']
                
                # Update text_sentences with all sentences
                updated_row['text_sentences'] = str(result['all_sentences'])
                
                if result['original_inference_type'] != result['new_inference_type']:
                    print(f"  Updated type: {result['original_inference_type']} → {result['new_inference_type']}")
            
            rows_to_keep.append(idx)
            updated_rows.append(updated_row)
        
        # Create new dataframe
        if updated_rows:
            new_df = pd.DataFrame(updated_rows)
            
            # Save updated CSV
            new_df.to_csv(probes_path, index=False)
            
            deleted_count = len(original_df) - len(new_df)
            print(f"  Final: {len(new_df)} probes ({deleted_count} deleted)")
        else:
            print(f"  Warning: No probes remaining for {domain}!")

def main():
    parser = argparse.ArgumentParser(description='Update inference probe types and find additional supporting sentences')
    parser.add_argument('--domains', nargs='+', default=None, 
                       help='Specific domains to process (e.g., DPO GRPO). If not specified, processes all.')
    parser.add_argument('--output', type=str, default='probe_inference_analysis_report.txt',
                       help='Output file name for the report')
    parser.add_argument('--update_csv', action='store_true',
                       help='Update the CSV files with the changes (delete rows and update inference types)')
    args = parser.parse_args()
    
    base_dir = '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/data'
    
    # Get all domains
    inference_dir = os.path.join(base_dir, 'probes', 'inference')
    all_domains = [d for d in os.listdir(inference_dir) 
                   if os.path.isdir(os.path.join(inference_dir, d))]
    
    # Filter domains if specified
    if args.domains:
        domains_to_process = [d for d in args.domains if d in all_domains]
        if len(domains_to_process) < len(args.domains):
            missing = set(args.domains) - set(domains_to_process)
            print(f"Warning: The following domains were not found: {missing}")
    else:
        domains_to_process = all_domains
    
    print(f"Will process the following domains: {domains_to_process}")
    
    # Process each domain
    all_results = []
    for domain in domains_to_process:
        results = process_domain(domain, base_dir)
        all_results.extend(results)
    
    # Generate report
    print(f"\n{'='*80}")
    print("Generating report...")
    report = format_results_for_report(all_results)
    
    # Save report
    output_path = os.path.join('/Users/jlee0/Desktop/research/fine-tuning-or-retrieval', args.output)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Report saved to: {output_path}")
    
    # Update CSV files if requested
    if args.update_csv:
        print(f"\n{'='*80}")
        print("Updating CSV files...")
        update_csv_files(all_results, base_dir)
        print("\nCSV files updated successfully!")
    else:
        print(f"\n{'='*80}")
        print("NOTE: CSV files were NOT updated. Use --update_csv flag to apply changes.")
    
    # Print summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    print(f"Total probes processed: {len(all_results)}")
    
    # Deletion statistics
    probes_to_delete = sum(1 for r in all_results if r['should_delete'])
    probes_to_keep = len(all_results) - probes_to_delete
    print(f"\nRecommended to DELETE: {probes_to_delete} ({probes_to_delete/len(all_results)*100:.1f}%)")
    print(f"Recommended to KEEP: {probes_to_keep} ({probes_to_keep/len(all_results)*100:.1f}%)")
    
    if probes_to_keep > 0:
        type_changes = sum(1 for r in all_results if r['original_inference_type'] != r['new_inference_type'] and not r['should_delete'])
        print(f"\nInference types changed (among kept probes): {type_changes} ({type_changes/probes_to_keep*100:.1f}% of kept)")
        
        probes_with_additional = sum(1 for r in all_results if r['additional_sentences'])
        print(f"Probes with additional sentences found: {probes_with_additional} ({probes_with_additional/len(all_results)*100:.1f}%)")
        
        # Count by new inference type (only for kept probes)
        print("\nNew inference type distribution (kept probes only):")
        new_types = {}
        for r in all_results:
            if not r['should_delete']:
                new_type = r['new_inference_type']
                new_types[new_type] = new_types.get(new_type, 0) + 1
        
        for type_name, count in sorted(new_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  {type_name}: {count} ({count/probes_to_keep*100:.1f}%)")
    else:
        print("\nNo probes recommended to keep.")

if __name__ == '__main__':
    main()

