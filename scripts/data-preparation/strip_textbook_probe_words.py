import os
import json
import sys
import pandas as pd
import re
from collections import defaultdict
from tqdm import tqdm

# Add the project root to the path
sys.path.append('../../')
import utils.utils as utils
from utils import probe_paths

# Common English stop words to exclude from replacement
STOP_WORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
    'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'will', 'with',
    'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it', 'for',
    'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at', 'this', 'but', 'his', 'by',
    'from', 'they', 'we', 'say', 'her', 'she', 'or', 'an', 'will', 'my', 'one',
    'all', 'would', 'there', 'their', 'what', 'so', 'up', 'out', 'if', 'about',
    'who', 'get', 'which', 'go', 'me', 'when', 'make', 'can', 'like', 'time',
    'no', 'just', 'him', 'know', 'take', 'people', 'into', 'year', 'your', 'good',
    'some', 'could', 'them', 'see', 'other', 'than', 'then', 'now', 'look', 'only',
    'come', 'its', 'over', 'think', 'also', 'back', 'after', 'use', 'two', 'how',
    'our', 'work', 'first', 'well', 'way', 'even', 'new', 'want', 'because', 'any',
    'these', 'give', 'day', 'most', 'us', 'is', 'was', 'are', 'been', 'has', 'had',
    'were', 'said', 'did', 'having', 'may', 'should', 'am', 'being', 'does', 'done'
}

def load_target_words(facts_path, inference_path):
    """Extract all target words from probe CSV files by breaking apart phrases."""
    target_words = set()
    probe_count = 0
    
    # Load factual probes
    if os.path.exists(facts_path):
        df_facts = pd.read_csv(facts_path)
        if 'target' in df_facts.columns:
            for target in df_facts['target'].dropna():
                target_str = str(target).strip()
                # Split into individual words (removing LaTeX and special chars)
                clean_str = re.sub(r'[\$\\{}]', '', target_str)
                words = re.findall(r'\b\w+\b', clean_str.lower())
                target_words.update(words)
            probe_count += len(df_facts)
    
    # Load inference probes
    if os.path.exists(inference_path):
        df_inference = pd.read_csv(inference_path)
        if 'target' in df_inference.columns:
            for target in df_inference['target'].dropna():
                target_str = str(target).strip()
                # Split into individual words (removing LaTeX and special chars)
                clean_str = re.sub(r'[\$\\{}]', '', target_str)
                words = re.findall(r'\b\w+\b', clean_str.lower())
                target_words.update(words)
            probe_count += len(df_inference)
    
    # Remove stop words
    target_words = target_words - STOP_WORDS
    
    return target_words, probe_count

def split_into_paragraphs(text):
    """Split text into paragraphs."""
    # Split on double newlines
    paragraphs = re.split(r'\n\n+', text)
    return [p.strip() for p in paragraphs if p.strip()]

def find_flagged_paragraphs(paragraphs, target_words):
    """Find paragraphs that contain any target words."""
    flagged_indices = []
    flagged_words_per_para = defaultdict(set)
    
    for idx, para in enumerate(paragraphs):
        para_lower = para.lower()
        para_words = set(re.findall(r'\b\w+\b', para_lower))
        
        # Find which target words appear in this paragraph
        found_words = para_words & target_words
        
        if found_words:
            flagged_indices.append(idx)
            flagged_words_per_para[idx] = found_words
    
    return flagged_indices, flagged_words_per_para

def replace_words_in_paragraph(para, flagged_words, source_paper_text, domain):
    """Use LLM to replace flagged words with synonyms."""
    
    # Extract vocabulary from source paper
    source_text_lower = source_paper_text.lower()
    source_words = set(re.findall(r'\b\w+\b', source_text_lower))
    
    flagged_list = sorted(list(flagged_words))
    
    prompt = {
        'system': """You are tasked with replacing specific words in a paragraph with appropriate synonyms.

For each flagged word, you should:
1. First, try to find a synonym that exists in the source paper vocabulary
2. If no suitable synonym exists in the source paper, choose a true synonym that is NOT lexically similar (i.e., not just adding/removing prefixes/suffixes)

Requirements:
- Maintain the original meaning and context
- Ensure the replacement makes grammatical sense
- Preserve any mathematical notation or technical accuracy
- Only replace the flagged words, leave everything else unchanged
- Replace each occurrence of the flagged word throughout the paragraph

Return your response as a JSON object with:
- "modified_paragraph": The paragraph with replacements made
- "replacements": A list of replacement objects, each with:
  - "original": the original word
  - "replacement": the new word
  - "from_source": boolean indicating if the replacement came from source paper vocabulary
""",
        'user': f"""### Paragraph to modify:
{para}

### Flagged words to replace (case-insensitive):
{', '.join(flagged_list)}

### Source paper vocabulary:
{', '.join(list(source_words))}

Please replace the flagged words according to the instructions."""
    }
    
    try:
        response_str = utils.query_llm(
            prompt,
            model='gpt-5',
            reasoning_effort="low",
            system_prompt_included=True,
            return_json=True,
            max_tokens=2000
        )
        
        result = json.loads(response_str)
        return result
    except Exception as e:
        print(f"Error in LLM call for domain {domain}: {e}")
        return {
            "modified_paragraph": para,
            "replacements": []
        }

def process_domain(domain, base_dir, probes_dir, output_dir):
    """Process a single domain to create stripped textbook."""
    
    print(f"\n{'='*60}")
    print(f"Processing domain: {domain}")
    print(f"{'='*60}")
    
    # Paths
    textbook_path = os.path.join(base_dir, 'explanations', domain, 'textbook.txt')
    facts_path = str(probe_paths.resolve_probe_path('facts', domain, 'v9'))
    inference_path = str(probe_paths.resolve_probe_path('inference', domain, 'v6'))
    source_paper_path = os.path.join(base_dir, 'cleaned', domain + '.tex')
    
    # Check if textbook exists
    if not os.path.exists(textbook_path):
        print(f"Textbook not found: {textbook_path}")
        return None
    
    # Load target words from probes (breaking apart phrases)
    print("Loading target words from probes...")
    target_words, total_probes = load_target_words(facts_path, inference_path)
    print(f"Found {len(target_words)} unique target words from {total_probes} probes (after removing stop words)")
    
    if len(target_words) == 0:
        print("No target words found, skipping domain")
        return None
    
    # Load textbook
    print("Loading textbook...")
    with open(textbook_path, 'r') as f:
        textbook_content = f.read()
    
    # Load source paper
    source_paper_text = ""
    if os.path.exists(source_paper_path):
        print("Loading source paper...")
        with open(source_paper_path, 'r') as f:
            source_paper_text = f.read()
    else:
        print(f"Warning: Source paper not found: {source_paper_path}")
    
    # Filter target words by relative frequency
    print("Filtering target words by relative frequency...")
    initial_word_count = len(target_words)
    textbook_lower = textbook_content.lower()
    source_lower = source_paper_text.lower()
    textbook_len = len(textbook_lower)
    source_len = len(source_lower)
    
    filtered_target_words = set()
    for word in target_words:
        # Count occurrences using word boundaries
        textbook_count = len(re.findall(r'\b' + re.escape(word) + r'\b', textbook_lower))
        source_count = len(re.findall(r'\b' + re.escape(word) + r'\b', source_lower))
        
        # Calculate relative frequencies
        textbook_freq = textbook_count / textbook_len if textbook_len > 0 else 0
        source_freq = source_count / source_len if source_len > 0 else 0
        
        # Only keep if more frequent in textbook than source paper
        if textbook_freq > source_freq:
            filtered_target_words.add(word)
    
    target_words = filtered_target_words
    print(f"Filtered to {len(target_words)} words that are more frequent in textbook than source paper (removed {initial_word_count - len(target_words)} words)")
    
    if len(target_words) == 0:
        print("No target words remaining after frequency filtering, skipping domain")
        return None
    
    # Split into paragraphs
    print("Splitting into paragraphs...")
    paragraphs = split_into_paragraphs(textbook_content)
    print(f"Total paragraphs: {len(paragraphs)}")
    
    # Find flagged paragraphs
    print("Identifying flagged paragraphs...")
    flagged_indices, flagged_words_per_para = find_flagged_paragraphs(paragraphs, target_words)
    print(f"Flagged paragraphs: {len(flagged_indices)}")
    
    if len(flagged_indices) == 0:
        print("No flagged paragraphs found, skipping domain")
        return None
    
    # Process flagged paragraphs
    print("Replacing words in flagged paragraphs...")
    all_replacements = []
    modified_paragraphs = paragraphs.copy()
    from_source_count = 0
    from_synonym_count = 0
    
    for idx in tqdm(flagged_indices, desc=f"Processing {domain}"):
        para = paragraphs[idx]
        flagged_words_for_para = flagged_words_per_para[idx]
        
        result = replace_words_in_paragraph(para, flagged_words_for_para, source_paper_text, domain)
        
        modified_paragraphs[idx] = result['modified_paragraph']
        
        for repl in result['replacements']:
            all_replacements.append(repl)
            if repl.get('from_source', False):
                from_source_count += 1
            else:
                from_synonym_count += 1
    
    # Create modified textbook
    modified_textbook = '\n\n'.join(modified_paragraphs)
    
    # Calculate statistics
    total_replacements = len(all_replacements)
    pct_from_source = (from_source_count / total_replacements * 100) if total_replacements > 0 else 0
    pct_from_synonym = (from_synonym_count / total_replacements * 100) if total_replacements > 0 else 0
    
    # Create replacement map
    replacement_map = defaultdict(list)
    for repl in all_replacements:
        replacement_map[repl['original']].append({
            'replacement': repl['replacement'],
            'from_source': repl.get('from_source', False)
        })
    
    stats = {
        'domain': domain,
        'total_probes': total_probes,
        'unique_target_words': len(target_words),
        'total_paragraphs': len(paragraphs),
        'flagged_paragraphs': len(flagged_indices),
        'total_replacements': total_replacements,
        'from_source_count': from_source_count,
        'from_synonym_count': from_synonym_count,
        'pct_from_source': round(pct_from_source, 2),
        'pct_from_synonym': round(pct_from_synonym, 2),
        'replacement_map': dict(replacement_map)
    }
    
    # Save outputs
    domain_output_dir = os.path.join(output_dir, domain)
    os.makedirs(domain_output_dir, exist_ok=True)
    
    # Save modified textbook
    modified_textbook_path = os.path.join(domain_output_dir, 'textbook_stripped_level1.txt')
    with open(modified_textbook_path, 'w') as f:
        f.write(modified_textbook)
    print(f"Saved modified textbook to: {modified_textbook_path}")
    
    # Save statistics
    stats_path = os.path.join(domain_output_dir, 'replacement_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved statistics to: {stats_path}")
    
    # Print summary
    print(f"\n--- Summary for {domain} ---")
    print(f"Total replacements: {total_replacements}")
    print(f"From source paper: {from_source_count} ({pct_from_source:.1f}%)")
    print(f"From synonyms: {from_synonym_count} ({pct_from_synonym:.1f}%)")
    print(f"Probes involved: {total_probes}")
    
    return stats

def main():
    base_dir = '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/data/arxiv'
    output_dir = '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/data/arxiv/explanations_stripped'
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all domains from explanations directory
    explanations_dir = os.path.join(base_dir, 'explanations')
    domains = [d for d in os.listdir(explanations_dir) 
               if os.path.isdir(os.path.join(explanations_dir, d))]
    
    print(f"Found {len(domains)} domains to process")
    
    # Process each domain
    all_stats = []
    for domain in domains:
        stats = process_domain(domain, base_dir, "", output_dir)
        if stats:
            all_stats.append(stats)
    
    # Save aggregate statistics
    aggregate_stats_path = os.path.join(output_dir, 'aggregate_stats.json')
    with open(aggregate_stats_path, 'w') as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nSaved aggregate statistics to: {aggregate_stats_path}")
    
    # Print overall summary
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    print(f"Domains processed: {len(all_stats)}")
    total_repl = sum(s['total_replacements'] for s in all_stats)
    total_from_source = sum(s['from_source_count'] for s in all_stats)
    total_from_syn = sum(s['from_synonym_count'] for s in all_stats)
    print(f"Total replacements: {total_repl}")
    print(f"From source: {total_from_source} ({total_from_source/total_repl*100:.1f}%)")
    print(f"From synonyms: {total_from_syn} ({total_from_syn/total_repl*100:.1f}%)")

if __name__ == "__main__":
    main()
