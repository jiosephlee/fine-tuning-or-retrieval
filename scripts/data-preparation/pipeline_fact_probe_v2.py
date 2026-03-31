"""
pipeline_fact_probe_v2.py — Token-optimized version of pipeline_fact_probe.py

Changes from v1:
  #2  Sentence extraction now outputs start/end boundary markers instead of
      echoing the entire subsection text, cutting output tokens ~90%.
  #3  Contextualize + cloze conversion merged into a single LLM call per Q&A
      pair, eliminating one round-trip and duplicate context per pair.
  #5  Extraction few-shot demos trimmed from 3 verbose examples to 2 compact
      ones; extraction now requests JSON directly (eliminates the separate
      JSON-parsing LLM call).
  #6  The 12-instruction contextualize prompt condensed into 7 concise
      instructions as part of the merged prompt.
"""

import textwrap
import sys 
sys.path.append('../..')
import utils.utils as utils
import pandas as pd
from tqdm import tqdm
from importlib import reload
import json
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import re
import string
from transformers import AutoTokenizer
import os
import argparse
from utils.pipeline import save_df_for_debugging, is_text_in_document, check_tokenizer_consistency, process_papers


def generate_probes_for_paper(paper_name, paper_content, sample=False):
    paper = paper_content

    checkpoint_dir = f'../../data/probes/facts/{paper_name}/checkpoints/'
    checkpoint_path = os.path.join(checkpoint_dir, '07_knowledge_kept.csv')

    if os.path.exists(checkpoint_path):
        print(f"Checkpoint found for {paper_name}. Loading from {checkpoint_path}...")
        paper_df_knowledge = pd.read_csv(checkpoint_path)
    else:
        print(f"No checkpoint found for {paper_name}. Running full pipeline...")
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

        # ── v2 change #2: boundary-marker sentence extraction ──
        # Instead of echoing the entire subsection with [BOS]/[EOS] tags (output ≈
        # input length), the model outputs only short start/end anchors per sentence.
        # Full sentences are recovered via string matching against the original text.
        extraction_prompt = r"""Identify every complete sentence in the given academic paper section.

Guidelines:
1. Sentences may span multiple lines and include LaTeX commands or math expressions.
2. Abbreviations ("et al.", "Fig.", "Eq.", "i.e.", "e.g.") do NOT end sentences.
3. Multi-line equations followed by "where" or "i.e." are part of ONE sentence.
4. When boundaries are unclear, extend to the next clear boundary.

For each sentence, copy its opening fragment (first ~6-10 words) and closing fragment (last ~6-10 words) VERBATIM from the input. Use enough words to uniquely locate the sentence.

Return JSON: {"sentences": [{"start": "verbatim opening...", "end": "...verbatim closing"}, ...]}"""

        # Parse paper structure
        paper_df = parse_paper_structure(paper)

        if sample:
            if paper_name == "DPO":
                paper_df = paper_df[paper_df['section'] == 'Preliminaries'].copy()
                if paper_df.empty:
                    print("Warning: 'Preliminaries' section not found for DPO paper. Exiting.")
                    return
            elif paper_name == "CoT":
                paper_df = paper_df[paper_df['section'] == 'Arithmetic Reasoning'].copy()
                if paper_df.empty:
                    print("Warning: 'Title/Abstract' section not found for CoT paper. Exiting.")
                    return
            else:
                print("Warning: Sample mode is only implemented for DPO and CoT. Running on the full paper.")
                
        save_df_for_debugging(paper_df, '01_paper_structure.txt', 'facts', paper_name, ['section', 'subsection', 'paragraph'])

        # Create a dataframe with unique subsections
        subsection_df = paper_df[['section', 'subsection', 'section_text', 'subsection_text']].drop_duplicates().reset_index(drop=True)

        def query_single(subsection_text):
            prompt = {}
            prompt['system'] = extraction_prompt
            prompt['user'] = f"""{subsection_text}"""
            return utils.query_llm(prompt, model='gpt-5.4-mini', reasoning_effort='medium', return_json=True, max_tokens=2000)
        
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            futures = [executor.submit(query_single, row['subsection_text']) for _, row in subsection_df.iterrows()]
            marker_results = [future.result() for future in tqdm(futures, desc="Processing subsections")]

        subsection_df['sentence_markers_raw'] = marker_results
        save_df_for_debugging(subsection_df, '02_extracted_markers.txt', 'facts', paper_name, ['section','subsection','sentence_markers_raw'])

        # ── v2 change #2: extract sentences via marker matching ──

        def extract_sentences_from_markers(markers_raw, text):
            """Match start/end markers against original text to extract full sentences."""
            try:
                data = json.loads(markers_raw) if isinstance(markers_raw, str) else markers_raw
                markers = data.get('sentences', [])
            except (json.JSONDecodeError, TypeError):
                return []

            sentences = []
            search_from = 0

            for marker in markers:
                start_words = marker.get('start', '').strip()
                end_words = marker.get('end', '').strip()

                if not start_words or not end_words:
                    continue

                start_idx = text.find(start_words, search_from)
                if start_idx == -1:
                    start_idx = text.find(start_words)
                    if start_idx == -1:
                        continue

                end_idx = text.find(end_words, start_idx)
                if end_idx == -1:
                    continue

                sentence = text[start_idx:end_idx + len(end_words)]

                if len(sentence) > 3000:
                    continue

                sentences.append(sentence)
                search_from = end_idx + len(end_words)

            return sentences

        subsection_df['sentence_list'] = subsection_df.apply(
            lambda row: extract_sentences_from_markers(row['sentence_markers_raw'], row['subsection_text']),
            axis=1
        )

        # Explode the DataFrame on the sentence_list column
        paper_df_sentences = subsection_df.explode('sentence_list').rename(columns={'sentence_list': 'raw_knowledge_statement'})

        # Drop rows with no knowledge statements
        paper_df_sentences = paper_df_sentences[paper_df_sentences['raw_knowledge_statement'].notna()]

        total_extracted_sentences = len(paper_df_sentences)
        print(f"Extracted {total_extracted_sentences} sentences via marker matching.")

        # Marker matching inherently validates that sentences exist in the source,
        # so we skip the is_text_in_document validation step.
        paper_df_validated = paper_df_sentences.copy()

        # Use subsection_text as context paragraph (replaces remove_sentence_tags)
        paper_df_validated['paragraph'] = paper_df_validated['subsection_text']

        # Drop the marker column, no longer needed
        paper_df_validated = paper_df_validated.drop(columns=['sentence_markers_raw'])

        print(f"Total validated sentences: {len(paper_df_validated)}")
        save_df_for_debugging(paper_df_validated, '03_validated_sentences.txt', 'facts', paper_name, ['section', 'raw_knowledge_statement'])
        
        # Filter bad sentences

        ## 2.1 Filter out Predominantly latex sentences

        def calculate_latex_percentage(text):
            if pd.isna(text) or not text.strip():
                return 0.0
            
            total_chars = len(text)
            latex_chars = 0
            
            latex_commands = re.findall(r'\\[a-zA-Z]+', text)
            for cmd in latex_commands:
                latex_chars += len(cmd)
            
            text_without_commands = re.sub(r'\\[a-zA-Z]+', '', text)
            
            for char in text_without_commands:
                if not char.isalpha() and not char.isspace():
                    latex_chars += 1
            
            latex_percentage = (latex_chars / total_chars) * 100 if total_chars > 0 else 0.0
            
            return latex_percentage

        paper_df_validated['latex_percentage'] = paper_df_validated['raw_knowledge_statement'].apply(calculate_latex_percentage)

        latex_threshold = 75
        high_latex_statements = paper_df_validated[paper_df_validated['latex_percentage'] > latex_threshold].copy()
        paper_df_filtered = paper_df_validated[paper_df_validated['latex_percentage'] <= latex_threshold].copy()

        save_df_for_debugging(high_latex_statements, '04_latex_filtered_out.txt', 'facts', paper_name, ['raw_knowledge_statement', 'latex_percentage'])
        save_df_for_debugging(paper_df_filtered, '04_latex_kept.txt', 'facts', paper_name, ['raw_knowledge_statement', 'latex_percentage'])

        total_before = len(paper_df_validated)
        total_after = len(paper_df_filtered)
        filtered_out = total_before - total_after

        print(f"LaTeX filtering results:")
        print(f"  Before filtering: {total_before} knowledge statements")
        print(f"  After filtering: {total_after} knowledge statements")
        print(f"  Filtered out: {filtered_out} statements with >{latex_threshold}% LaTeX content")

        ## 2.2 Filter Out Sentences with References Not In Context

        def find_undefined_references(sentence: str, subsection_text: str) -> list[str]:
            references = re.findall(r'\\ref\{([^}]+)\}', sentence)
            if not references:
                return True

            defined_labels = set(re.findall(r'\\label\{([^}]+)\}', subsection_text))

            undefined_references = [ref for ref in references if ref not in defined_labels]

            if len(undefined_references) > 0:
                return False
            else:
                return True

        keep = paper_df_filtered.apply(
            lambda row: find_undefined_references(row['raw_knowledge_statement'], row['subsection_text']),
            axis=1
        )

        dropped_statements = paper_df_filtered[~keep].copy()
        paper_df_filtered_refs = paper_df_filtered[keep].copy()

        save_df_for_debugging(dropped_statements, '05_ref_filtered_out.txt', 'facts', paper_name, ['raw_knowledge_statement'])
        save_df_for_debugging(paper_df_filtered_refs, '05_ref_kept.txt', 'facts', paper_name, ['raw_knowledge_statement'])

        total_before = len(paper_df_filtered)
        total_after = sum(keep)
        filtered_out = total_before - total_after

        print(f"Reference filtering results:")
        print(f"  Before filtering: {total_before} knowledge statements")
        print(f"  After filtering: {total_after} knowledge statements")
        print(f"  Filtered out: {filtered_out} statements with undefined references")

        paper_df_filtered = paper_df_filtered[keep].reset_index(drop=True)

        ## 2.3 Filter Short Facts

        min_length = 90
        keep_length = paper_df_filtered['raw_knowledge_statement'].str.len() >= min_length

        dropped_statements_len = paper_df_filtered[~keep_length].copy()
        paper_df_filtered_len = paper_df_filtered[keep_length].copy()

        save_df_for_debugging(dropped_statements_len, '06_short_filtered_out.txt', 'facts', paper_name, ['raw_knowledge_statement'])
        save_df_for_debugging(paper_df_filtered_len, '06_short_kept.txt', 'facts', paper_name, ['raw_knowledge_statement'])

        total_before = len(paper_df_filtered)
        total_after = sum(keep_length)
        filtered_out = total_before - total_after

        print(f"Length filtering results:")
        print(f"  Before filtering: {total_before} knowledge statements")
        print(f"  After filtering: {total_after} knowledge statements")
        print(f"  Filtered out: {filtered_out} statements shorter than {min_length} characters")

        paper_df_filtered = paper_df_filtered[keep_length].reset_index(drop=True)

        ## 2.4 Identify Knowledge Statements

        prompt = {}
        prompt['system'] = """You will be receiving sentences from an academic paper. Your task is to determine whether the sentence contains a meaningful, clear fact that could be formulated into a question to test a reader's comprehension.

A sentence CONTAINS a fact if it:
- Presents a clear piece of knowledge, whether it's a description of prior work, a theorem, the methodology, or the results.
- The sentence can be formulated into a meaningful fact.

This can be vague and so we provide a non-exhaustive list of some criteria for exclusion. A sentence does NOT contain a fact if it:
- Is semantically unclear, making it difficult to ascertain a clear fact.
- Is mainly a transitional sentence (e.g., "In this section, we discuss...").
- References a figure/table and provides information that requires the figure/table to make sense. While it may be informative, we have removed all figures and tables from the paper in our setting.
- Mainly describes the paper's own structure or what the paper is about (e.g., "The next section covers our methodology.").
- Mainly discusses future work or possibilities (e.g., "We plan to investigate this further.").
- Poses a rhetorical question.
- Is a quote.

For ambiguous cases, err on the side of saying that the sentence does not contain a fact.

# Output Format
Respond with JSON format with the following key:
- "is_knowledge": boolean (true/false)"""

        def evaluate_statement(idx_row):
            idx, row = idx_row
            statement = row['raw_knowledge_statement']
            user_prompt = f"# Context\n{row['paragraph']}\n\n# Clause\n{statement}"
            full_prompt = {
                'system': prompt['system'],
                'user': user_prompt
            }
            
            result = utils.query_llm(full_prompt, model='gpt-5.4-mini', reasoning_effort='low', return_json=True, max_tokens=100)
            result = json.loads(result)
            is_knowledge = result['is_knowledge']
            
            return idx, is_knowledge

        print("Evaluating sentences for knowledge content...")

        is_knowledge_results = [None] * len(paper_df_filtered)

        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = {executor.submit(evaluate_statement, (idx, row)): idx 
                       for idx, row in paper_df_filtered.iterrows()}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating sentences"):
                idx, is_knowledge = future.result()
                original_idx = list(paper_df_filtered.index).index(idx)
                is_knowledge_results[original_idx] = is_knowledge

        paper_df_filtered['is_knowledge'] = is_knowledge_results

        paper_df_knowledge = paper_df_filtered[paper_df_filtered['is_knowledge']].copy()
        paper_df_non_knowledge = paper_df_filtered[~paper_df_filtered['is_knowledge']].copy()
        save_df_for_debugging(paper_df_knowledge, '07_knowledge_kept.txt', 'facts', paper_name, ['raw_knowledge_statement', 'is_knowledge'])
        save_df_for_debugging(paper_df_non_knowledge, '07_knowledge_filtered_out.txt', 'facts', paper_name, ['raw_knowledge_statement', 'is_knowledge'])

        total_before = len(paper_df_filtered)
        total_after = len(paper_df_knowledge)
        filtered_out = total_before - total_after

        print(f"\nFinal knowledge filtering results:")
        print(f"  Before filtering: {total_before} sentences")
        print(f"  After filtering: {total_after} knowledge statements")
        print(f"  Filtered out: {filtered_out} non-knowledge sentences")

        # Save checkpoint
        print(f"Saving checkpoint to {checkpoint_path}...")
        os.makedirs(checkpoint_dir, exist_ok=True)
        paper_df_knowledge.to_csv(checkpoint_path, index=False)

    paper_df_knowledge.reset_index(drop=True, inplace=True)
    paper_df_knowledge['title'] = re.search(r'\\title{(.*?)}', paper).group(1) if re.search(r'\\title{(.*?)}', paper) else None

    # 3. Extract self-contained, atomic probes

    def extract_context_and_sentence(row):
        full_text = row['subsection_text'].strip()
        statement = row['raw_knowledge_statement'].strip()
        idx = full_text.find(statement)

        if idx == -1:
            return statement

        text_before = full_text[:idx]
        text_after = full_text[idx + len(statement):]

        words_before = text_before.split()
        prev_words = ' '.join(words_before[-50:]) if len(words_before) > 50 else ' '.join(words_before)

        words_after = text_after.split()
        next_words = ' '.join(words_after[:50]) if len(words_after) > 50 else ' '.join(words_after)

        context = prev_words + ' ' + statement + ' ' + next_words
        context = context.strip()
        context = '... ' + context + ' ...'

        return context
    
    def extract_atomic_facts(first_row):
        """Extract atomic facts and produce cloze probes in fewer LLM calls.

        v2 changes vs v1:
        - Extraction prompt trimmed to 2 demos and requests JSON directly,
          eliminating the separate JSON-parsing LLM call.
        - Contextualize + cloze merged into a single call per Q&A pair.
        """

        # ── v2 change #5: trimmed extraction prompt with JSON output ──
        extraction_prompt_atomic = r"""You will be given a section of an academic paper (for context) and a single sentence. Extract 1-2 questions testing atomic knowledge from the sentence. If the sentence is especially rich in facts, extract up to 3.

### Instructions
- Focus on main facts, not minor details. Questions should be non-trivial.
- The answer MUST be a verbatim word or short phrase (2-4 words) copied exactly from the sentence. Do not paraphrase.
- Each question should have a clear, single answer. Prefer shorter answers.
- For math: do NOT extract partial equations. Prefer natural language answers when they capture the same knowledge. Preserve $...$ delimiters exactly.
- Each question should be independent of the others.

### Demonstration 1
Context: "... We evaluate different methods by sampling completions on the test split of TL;DR summarization dataset, and computing the average win rate against reference completions in the test set."
Sentence: "We evaluate different methods by sampling completions on the test split of TL;DR summarization dataset, and computing the average win rate against reference completions in the test set."

Output: {"list_of_questions": [{"question": "DPO's summarization performance is evaluated by sampling from what dataset?", "answer": "TL;DR summarization"}, {"question": "The average win rate is computed against what?", "answer": "reference completions"}]}

### Demonstration 2 (Mathematical)
Sentence: "... where $\\beta$ is a parameter controlling the deviation from the base reference policy $\\pi_\\text{ref}$, namely the initial SFT model $\\pi^\\text{SFT}$."

Output: {"list_of_questions": [{"question": "In the RL fine-tuning objective, what does $\\beta$ control?", "answer": "deviation from the base reference policy"}]}

Return a JSON object: {"list_of_questions": [{"question": "...", "answer": "..."}, ...]}"""

        # ── v2 change #3 + #6: merged contextualize & cloze prompt ──
        contextualize_and_cloze_system = r"""Convert a Q&A pair from an academic paper into a self-contained cloze (fill-in-the-blank) statement.

### Instructions
1. Begin with 'In the paper "{title}", ...' or 'According to the paper "{title}", ...' (always double-quote the title). Vary the phrasing naturally.
2. Make the statement self-contained: resolve pronouns and vague references using the provided context. Clarify unnamed or context-dependent terms (e.g. "$f$", "the model") but leave named entities and frequent acronyms as-is.
3. Supply enough experimental context to disambiguate which experiment is referenced.
4. The statement must be declarative. The answer must appear exactly once, at the very end — do not leak it earlier. Do not modify the answer (minor grammatical adjustments like determiners are OK).
5. Preserve all information from the original question. Use multiple sentences for clarity if needed. Do not quote the source sentence directly.
6. Preserve LaTeX formatting exactly as in the source with $...$ delimiters.
7. Avoid awkward repetition or large parenthetical descriptions.

### Output
Return JSON: {"answer": "...", "statement": "..."}"""

        context = extract_context_and_sentence(first_row)
        prompt = {
            'system': extraction_prompt_atomic,
            'user': f"### Title\n{first_row['title']}\n### Context\n{context}\n\n### Sentence\n{first_row['raw_knowledge_statement'].strip()}"
        }
        output1 = utils.query_llm(prompt, model='gpt-5.4', reasoning_effort='low', return_json=True)

        try:
            qa_pairs_data = json.loads(output1)
            qa_pairs = qa_pairs_data.get('list_of_questions')
            if qa_pairs is None:
                print(f"Key 'list_of_questions' not found in JSON output: {qa_pairs_data}")
                return output1, [], []
        except (json.JSONDecodeError, TypeError):
            print(f"Failed to parse Q&A pairs from: {output1}")
            return output1, [], []

        if not isinstance(qa_pairs, list):
            print(f"Parsed 'list_of_questions' is not a list: {qa_pairs}")
            return output1, [], []

        original_questions = []
        all_clozes = []

        for pair in qa_pairs:
            question = pair.get('question')
            answer = pair.get('answer')
            if not question or not answer:
                continue

            merged_prompt = {
                'system': contextualize_and_cloze_system,
                'user': (
                    f"### Title\n{first_row['title']}\n"
                    f"### Context\n{context}\n\n"
                    f"### Sentence\n{first_row['raw_knowledge_statement'].strip()}\n\n"
                    f"### Question\n{question}\n\n"
                    f"### Answer\n{answer}"
                )
            }
            cloze_output = utils.query_llm(merged_prompt, model='gpt-5.4-mini', reasoning_effort='medium', return_json=True)
            try:
                cloze_pair = json.loads(cloze_output)
                if isinstance(cloze_pair, dict) and 'answer' in cloze_pair and 'statement' in cloze_pair:
                    all_clozes.append(cloze_pair)
                    original_questions.append(question)
            except (json.JSONDecodeError, TypeError):
                print(f"Failed to parse cloze pair from: {cloze_output}")

        return output1, original_questions, all_clozes

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        raw_extracted_facts = list(tqdm(executor.map(extract_atomic_facts, [paper_df_knowledge.iloc[i] for i in range(len(paper_df_knowledge))]), total=len(paper_df_knowledge)))

    questions = [result[0] if result is not None else None for result in raw_extracted_facts]
    contextualized_questions = [result[1] if result is not None else None for result in raw_extracted_facts]
    cloze_pairs = [result[2] if result is not None else [] for result in raw_extracted_facts]

    paper_df_knowledge['questions'] = questions
    paper_df_knowledge['contextualized_questions'] = contextualized_questions
    paper_df_knowledge['cloze_pairs'] = cloze_pairs
    save_df_for_debugging(paper_df_knowledge, '08_a_atomic_facts.txt', 'facts', paper_name, ['raw_knowledge_statement', 'questions', 'contextualized_questions', 'cloze_pairs'])

    print(f"Processed {len(raw_extracted_facts)} rows")

    # 4. Quality Control
     
    def validate_atomic_facts(row):
        """Process a single atomic fact for validation."""
        prompt = {}
        prompt['system'] = """Your task is to review and refine a statement that has been extracted from a sentence taken from an academic paper. You will be given an '(answer, statement)' pair. Your task is to apply a rigorous checklist to the pair, refining it or filtering it out based on a provided quality control checklist.

### Quality Control Checklist
For each '(answer, statement)' pair, refine it based on the following:

1. LaTeX Formatting
- Leave numbers as how they are written in the original sentence. e.g. eight should be eight and 8 should be 8.
- All mathematical expressions and notations **MUST** be written in LaTeX, enclosed in '$' or '$$' delimiters.
- Do *NOT* use unicode mathematical characters (e.g., use '\\pi', not 'π').
- Do *NOT* use unnecessary styling commands like '\\displaystyle'.
- Ensure LaTeX syntax matches the style of the original context (e.g., '( ... )' or '$ ... $').
- Action: Rewrite the math expressions and statements so they can be written in LaTeX, keeping the rest of the statement the same, correcting any and all formatting errors related to mathematical notation.

2. Answer Placement
- The answer must appear at the very end of the statement. 
- Action: Minimally rewrite the statement such that the answer appears at the end.

3. Answer Leakage
- The answer must not be leaked, explicitly or implicitly, in the statement until the very end.
- Action: Minimally rewrite the statement such that the answer is not revealed in the statement.

For any rewriting, do not change the structure, content, or shape of the statement. All of these adjustments should be minimal, word-level or character-level adjustments.

### Output Format
After your review, if the pair passes all checks (with any necessary refinements), provide the refined pair as a JSON object with two keys: "answer" and "statement"."""
        prompt['user'] = f"""### Answer\n{row['answer']}\n\n### Statement\n{row['statement']}"""
        response = utils.query_llm(prompt, model='gpt-5.4-mini', reasoning_effort='low', system_prompt_included=True, return_json=True)
        try:
            parsed_response = json.loads(response)
            if parsed_response and 'answer' in parsed_response and 'statement' in parsed_response:
                return parsed_response
        except (json.JSONDecodeError, TypeError):
            print(f"Failed to parse QC response, using original pair for row. Response: {response}")
        print(f"Failed to parse QC response, using original pair for row. Response: {response}")
        return {'answer': row['answer'], 'statement': row['statement']}

    def filter_refined_pair(row):
        """Decides whether to keep a refined pair based on a strict checklist."""
        validated_pair = row['validated_atomic_pairs']
        if not validated_pair:
            return False
            
        prompt = {}
        prompt['system'] = """Your task is to determine if a given (answer, statement) pair meets quality standards by acting as a filter.

### Quality Control Checklist
1. Linguistically Reasonable: Consider the fill-in-the-blank statement. The answer should be linguistically reasonable as to how it would fit in the fill-in-the-blank. It should sound natural and not forced.
2. Semantically Reasonable: Consider the fill-in-the-blank statement. The answer should be semantically reasonable as to how it would fit in the fill-in-the-blank. There should be one clear, unambiguous answer (or at least paraphrases of the answer).
3. Clear and Understandable: Consider the whole statement along with the answer. It should be clear what the sentence is building up to and what the answer is.

### Action
Based on the checklist, decide if the pair should be kept. Drop the pair if fails one of the checklist items.

### Output Format
Provide your decision as a JSON object with a single boolean key: `{"keep": true}` or `{"keep": false}`."""
        if validated_pair['statement'] is None or validated_pair['answer'] is None:
            return False
        prompt['user'] = f"""### Answer\n{validated_pair['answer']}\n\n### Statement\n{validated_pair['statement'].replace(str(validated_pair['answer']), '___')}"""
        
        response = utils.query_llm(prompt, model='gpt-5.4-mini', reasoning_effort='low', system_prompt_included=True, return_json=True)
        try:
            parsed_response = json.loads(response)
            return parsed_response.get('keep', False)
        except (json.JSONDecodeError, TypeError):
            print(f"Lost pair in filter step - JSON parse error: {response}")
            return False
        
    # Explode the dataframe to have one row per cloze pair
    rows_for_qc = []
    for idx, row in paper_df_knowledge.iterrows():
        if row['cloze_pairs'] and row['contextualized_questions']:
            cloze_pairs = row['cloze_pairs']
            contextualized_questions = row['contextualized_questions']

            assert len(cloze_pairs) == len(contextualized_questions), f"Mismatch in lengths for row {idx}: {len(cloze_pairs)} vs {len(contextualized_questions)}"

            for pair, contextualized_question in zip(cloze_pairs, contextualized_questions):
                if isinstance(pair, dict) and 'answer' in pair and 'statement' in pair:
                    new_row = row.to_dict()
                    new_row['answer'] = pair['answer']
                    new_row['statement'] = pair['statement']
                    new_row['contextualized_question'] = contextualized_question
                    
                    del new_row['cloze_pairs']
                    del new_row['contextualized_questions']
                    
                    rows_for_qc.append(new_row)

    paper_df_exploded = pd.DataFrame(rows_for_qc)
    print(f"Exploded {len(paper_df_knowledge)} knowledge statements into {len(paper_df_exploded)} candidate pairs.")


    # Process all rows in parallel
    if not paper_df_exploded.empty:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            validated_results = list(tqdm(
                executor.map(validate_atomic_facts, [row for _, row in paper_df_exploded.iterrows()]),
                total=len(paper_df_exploded)
            ))
        paper_df_exploded['validated_atomic_pairs'] = validated_results
    else:
        paper_df_exploded['validated_atomic_pairs'] = []


    # Track and save changes made during QC
    changed_pairs_data = []
    for idx, row in paper_df_exploded.iterrows():
        original_answer = row['answer']
        original_statement = row['statement']
        validated_pair = row['validated_atomic_pairs']

        if validated_pair:
            new_answer = validated_pair.get('answer')
            new_statement = validated_pair.get('statement')

            if original_answer != new_answer or original_statement != new_statement:
                changed_pairs_data.append({
                    'raw_knowledge_statement': row['raw_knowledge_statement'],
                    'original_answer': original_answer,
                    'new_answer': new_answer,
                    'original_statement': original_statement,
                    'new_statement': new_statement
                })

    if changed_pairs_data:
        paper_df_qc_changed = pd.DataFrame(changed_pairs_data)
        save_df_for_debugging(paper_df_qc_changed, '08_c_qc_changed.txt', 'facts', paper_name, 
                              ['raw_knowledge_statement', 'original_answer', 'new_answer', 'original_statement', 'new_statement'])
        print(f"Detected and saved {len(paper_df_qc_changed)} pairs that were changed during QC.")

    paper_df_qc_kept = paper_df_exploded
    print(f"Processed and refined {len(paper_df_exploded)} candidate pairs in QC.")

    # 4.5 Filter step after refinement
    if not paper_df_qc_kept.empty:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            keep_results = list(tqdm(
                executor.map(filter_refined_pair, [row for _, row in paper_df_qc_kept.iterrows()]),
                total=len(paper_df_qc_kept),
                desc="Filtering refined pairs"
            ))
        paper_df_qc_kept['final_keep'] = keep_results
    else:
        paper_df_qc_kept['final_keep'] = []

    paper_df_after_filter_dropped = paper_df_qc_kept[~paper_df_qc_kept['final_keep']].copy()
    paper_df_qc_kept = paper_df_qc_kept[paper_df_qc_kept['final_keep']].copy()

    if not paper_df_after_filter_dropped.empty:
        save_df_for_debugging(paper_df_after_filter_dropped, '08_d_final_filter_dropped.txt', 'facts', paper_name,
                              ['raw_knowledge_statement', 'validated_atomic_pairs'])
    
    print(f"\nAfter final filtering step:")
    print(f"Kept {len(paper_df_qc_kept)} pairs.")
    print(f"Dropped {len(paper_df_after_filter_dropped)} pairs.")
    
    # Directly create 'target' and 'fact' columns from 'validated_atomic_pairs'
    paper_df_qc_kept['target'] = paper_df_qc_kept['validated_atomic_pairs'].apply(lambda x: x['answer'])
    paper_df_qc_kept['fact'] = paper_df_qc_kept['validated_atomic_pairs'].apply(lambda x: x['statement'])
    paper_df_with_probes = paper_df_qc_kept
    paper_df_with_probes = paper_df_with_probes[['section','subsection', 'section_text','subsection_text', 'raw_knowledge_statement', 'target', 'fact', 'contextualized_question']]


    # 5. Check target is truly at end of the sentence -> Split Fact into Fact and Target

    facts_starting_with_paren = paper_df_with_probes[paper_df_with_probes['fact'].str.startswith('(')]
    print(f"Found {len(facts_starting_with_paren)} facts starting with '(' - fixing these...")

    for idx, row in facts_starting_with_paren.iterrows():
        target_with_comma = row['target'] + ','
        if target_with_comma in row['fact']:
            rightmost_part = row['fact'].split(target_with_comma)[-1].strip().strip('()')
            paper_df_with_probes.loc[idx, 'fact'] = rightmost_part

    print(f"Fixed {len(facts_starting_with_paren)} facts that started with '('")
    save_df_for_debugging(paper_df_with_probes, '09_fixed_paren_facts.txt', 'facts', paper_name, ['raw_knowledge_statement', 'fact'])

    # Check that facts end with targets
    valid_facts = []
    filtered_count = 0
    dropped_examples = []

    for idx, row in paper_df_with_probes.iterrows():
        fact = str(row['fact']).strip()
        target = ' ' +str(row['target']).strip().rstrip(string.punctuation + string.whitespace)  
        
        fact_cleaned = fact.rstrip(string.punctuation + string.whitespace)
        
        if fact_cleaned.endswith(target) and target in fact_cleaned:
            valid_facts.append(True)
        else:
            valid_facts.append(False)
            filtered_count += 1
            dropped_examples.append(row.to_dict())

    if not paper_df_with_probes.empty:
        paper_df_with_probes['valid_fact'] = valid_facts
    else:
        print("No probes to validate.")
        return

    print(f"Filtered out {filtered_count} facts that don't end with their target")
    print(f"Remaining valid facts: {len(paper_df_with_probes) - filtered_count}")

    paper_df_probes_valid = paper_df_with_probes[paper_df_with_probes['valid_fact']].copy()
    paper_df_probes_invalid = paper_df_with_probes[~paper_df_with_probes['valid_fact']].copy()

    save_df_for_debugging(paper_df_probes_valid, '10_target_end_kept.txt', 'facts', paper_name, ['raw_knowledge_statement', 'fact', 'target'])
    if not paper_df_probes_invalid.empty:
        save_df_for_debugging(paper_df_probes_invalid, '10_target_end_filtered_out.txt', 'facts', paper_name, ['raw_knowledge_statement', 'fact', 'target'])

    # Create probe column (fact minus target)
    probes = []
    cleaned_facts = []
    cleaned_targets = []

    for idx, row in paper_df_probes_valid.iterrows():
        fact = str(row['fact']).strip()
        target = str(row['target']).strip()
        
        fact_cleaned = fact
        target_cleaned = ' ' + target
        
        
        last_index = fact_cleaned.rfind(target_cleaned)
        if last_index != -1:
            probe = fact_cleaned[:last_index].strip()
            target_cleaned = fact_cleaned[last_index:]
            probes.append(probe)
            cleaned_facts.append(fact_cleaned)
            cleaned_targets.append(target_cleaned)
        else:
            print(fact)
            print(target)
            raise ValueError(f"Target {target_cleaned} not found in fact {fact_cleaned}")

    paper_df_probes_valid['probe'] = probes
    paper_df_probes_valid['fact'] = cleaned_facts
    paper_df_probes_valid['target'] = cleaned_targets

    print(f"Created probe column by removing target from fact")
    print(f"Final dataset shape: {paper_df_probes_valid.shape}")
    save_df_for_debugging(paper_df_probes_valid, '11_final_probes.txt', 'facts', paper_name, ['raw_knowledge_statement', 'fact', 'target', 'probe'])

    # 6. ensure tokenizing the target separately from the probe is fine

    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B")

    check_tokenizer_consistency(paper_df_probes_valid, tokenizer)

    # 7. Filter probes where the target does not appear verbatim in the paper

    total_before_verbatim = len(paper_df_probes_valid)
    verbatim_mask = paper_df_probes_valid['target'].apply(
        lambda t: t.strip() in paper
    )
    paper_df_not_verbatim = paper_df_probes_valid[~verbatim_mask].copy()
    paper_df_probes_valid = paper_df_probes_valid[verbatim_mask].copy()
    total_after_verbatim = len(paper_df_probes_valid)
    filtered_verbatim = total_before_verbatim - total_after_verbatim

    print(f"\nVerbatim filtering results:")
    print(f"  Before filtering: {total_before_verbatim} probes")
    print(f"  After filtering: {total_after_verbatim} probes")
    print(f"  Filtered out: {filtered_verbatim} probes whose target is not verbatim in the paper")

    save_df_for_debugging(paper_df_not_verbatim, '12_verbatim_filtered_out.txt', 'facts', paper_name, ['raw_knowledge_statement', 'fact', 'target', 'probe'])

    # 7.5 Recovery: ask LLM to fix non-verbatim probes

    def recover_verbatim_probe(row):
        """Ask the LLM to replace the answer with a verbatim phrase from the source sentence."""
        prompt = {}
        prompt['system'] = r"""You are given a cloze-style probe statement and its answer, both derived from a sentence in an academic paper. The answer does not appear verbatim in the original sentence — often because LaTeX delimiters were changed (e.g. $...$ became \(...\)) or the phrasing was slightly altered.

Your task:
1. Find a verbatim substring from the original sentence that captures the same knowledge as the current answer.
2. Minimally adjust the statement so that it ends with this new verbatim answer and reads naturally.
3. The new answer MUST be an exact, character-for-character substring of the original sentence (including any LaTeX formatting like $, \begin{}, etc.).
4. If there is no way to naturally produce a statement ending with a verbatim answer from the sentence, return {"success": false}.

### Output Format
Return a JSON object:
- On success: {"success": true, "answer": "...", "statement": "..."}
- On failure: {"success": false}"""
        prompt['user'] = f"""### Original Sentence
{row['raw_knowledge_statement']}

### Current Statement
{row['fact']}

### Current Answer (not verbatim)
{row['target']}"""
        response = utils.query_llm(prompt, model='gpt-5.4-mini', reasoning_effort='medium', system_prompt_included=True, return_json=True)
        try:
            parsed = json.loads(response)
            return parsed
        except (json.JSONDecodeError, TypeError):
            return {'success': False}

    if not paper_df_not_verbatim.empty:
        print(f"\nAttempting to recover {len(paper_df_not_verbatim)} non-verbatim probes...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            recovery_results = list(tqdm(
                executor.map(recover_verbatim_probe, [row for _, row in paper_df_not_verbatim.iterrows()]),
                total=len(paper_df_not_verbatim),
                desc="Recovering non-verbatim probes"
            ))

        recovered_rows = []
        failed_rows = []
        for (idx, row), result in zip(paper_df_not_verbatim.iterrows(), recovery_results):
            if result.get('success') and 'answer' in result and 'statement' in result:
                new_answer = result['answer']
                if new_answer.strip() in paper:
                    new_row = row.copy()
                    new_row['target'] = ' ' + new_answer.strip()
                    new_row['fact'] = result['statement']
                    fact_cleaned = new_row['fact'].rstrip(string.punctuation + string.whitespace)
                    target_cleaned = ' ' + new_answer.strip().rstrip(string.punctuation + string.whitespace)
                    last_index = fact_cleaned.rfind(target_cleaned)
                    if last_index != -1:
                        new_row['probe'] = fact_cleaned[:last_index].strip()
                        recovered_rows.append(new_row)
                        continue
            failed_rows.append(row)

        recovered_count = len(recovered_rows)
        failed_count = len(failed_rows)
        print(f"  Recovered: {recovered_count} probes")
        print(f"  Failed: {failed_count} probes")

        if recovered_rows:
            paper_df_recovered = pd.DataFrame(recovered_rows)
            save_df_for_debugging(paper_df_recovered, '12b_verbatim_recovered.txt', 'facts', paper_name, ['raw_knowledge_statement', 'fact', 'target', 'probe'])
            paper_df_probes_valid = pd.concat([paper_df_probes_valid, paper_df_recovered], ignore_index=True)

        if failed_rows:
            paper_df_failed = pd.DataFrame(failed_rows)
            save_df_for_debugging(paper_df_failed, '12c_verbatim_unrecoverable.txt', 'facts', paper_name, ['raw_knowledge_statement', 'fact', 'target', 'probe'])

        total_after_recovery = len(paper_df_probes_valid)
        print(f"  Total probes after recovery: {total_after_recovery}")

    # Save filtering metrics report
    output_dir = f'../../data/probes/facts/{paper_name}/'
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, 'filtering_metrics_v10_5.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"Fact Probe Pipeline v10_5 - Filtering Metrics for {paper_name}\n")
        f.write(f"{'='*60}\n")
        f.write(f"Total probes before verbatim filter: {total_before_verbatim}\n")
        f.write(f"Probes filtered (target not verbatim in paper): {filtered_verbatim}\n")
        f.write(f"Total probes after verbatim filter: {total_after_verbatim}\n")
    print(f"Saved filtering metrics to {metrics_path}")

    # 8. save probes

    paper_df_probes_valid.reset_index(drop=True, inplace=True)
    paper_df_probes_valid.to_csv(os.path.join(output_dir, 'probes_v10_5.csv'), index=False)

    # Save readable version
    readable_path = os.path.join(output_dir, 'probes_v10_5_readable.txt')
    with open(readable_path, 'w') as f:
        for _, row in paper_df_probes_valid.iterrows():
            f.write(f"{row['probe']}: {row['target'].lstrip()}\n")
    print(f"Saved readable probes to {readable_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter", type=str, default=None, help="Only process papers containing this string in their filename.")
    parser.add_argument("--sample", action="store_true", help="Run on a sample of the paper. DPO: Preliminaries section, CoT: Title/Abstract section.")
    args = parser.parse_args()

    process_papers(generate_probes_for_paper, '../../data/arxiv/cleaned/', file_filter=args.filter, sample=args.sample)
