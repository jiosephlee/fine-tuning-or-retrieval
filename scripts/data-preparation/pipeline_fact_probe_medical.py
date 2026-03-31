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
from utils.prompts.pipeline import FACT_PROBE_CLOZE_PROMPT_SYSTEM


def load_case_report_titles():
    """Load the case report title mapping from the JSON file."""
    titles_path = os.path.join(os.path.dirname(__file__), '../../data/medical/case_report_titles.json')
    with open(titles_path, 'r') as f:
        return json.load(f)

CASE_REPORT_TITLES = load_case_report_titles()


def generate_probes_for_document(document_name, document_content, sample=False):
    document = document_content

    checkpoint_dir = f'../../data/probes/facts/{document_name}/checkpoints/'
    checkpoint_path = os.path.join(checkpoint_dir, '07_knowledge_kept_v10_5.csv')

    if False:  # Skip checkpoint to force full pipeline run
        print(f"Checkpoint found for {document_name}. Loading from {checkpoint_path}...")
        document_df_knowledge = pd.read_csv(checkpoint_path)
    else:
        print(f"Running full pipeline (checkpoint skipped)...")
        def parse_document_structure(text):
            """Parse a medical case report into sections with metadata.

            Medical case reports use named section headers like "Abstract:",
            "Background:", "Case presentation:", "Investigations:",
            "Differential diagnosis:", "Treatment:", "Outcome and follow-up:",
            "Discussion:", "Conclusion:", etc.
            """
            sections = []

            # Match section headers on their own line ending with a colon.
            # Use [^\S\n] (whitespace-except-newline) instead of \s to prevent
            # matching across line boundaries (e.g., author line + "Abstract").
            section_pattern = r'\n([A-Z][A-Za-z \t,\-]+):\s*\n'
            section_splits = re.split(section_pattern, text)

            current_section = "Title/Authors"
            current_section_content = ""

            for i in range(len(section_splits)):
                if i == 0:
                    content = section_splits[i]
                    current_section_content = content
                elif i % 2 == 1:
                    # This is a section header
                    current_section = section_splits[i].strip()
                    continue
                else:
                    content = section_splits[i]
                    current_section_content = content

                # No subsection structure in medical case reports — treat the whole section as one unit
                paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]

                for paragraph in paragraphs:
                    sections.append({
                        'section': current_section,
                        'subsection': "No Subsection",
                        'paragraph': paragraph,
                        'section_text': current_section_content,
                        'subsection_text': current_section_content
                    })

            return pd.DataFrame(sections)


        # ── boundary-marker sentence extraction ──
        extraction_prompt = r"""Identify every complete sentence in the given section of a medical case report.

Guidelines:
1. Sentences may span multiple lines and include medical terminology, abbreviations, or measurement values.
2. Medical abbreviations (e.g., "mg/L", "U/L", "mmol/L", "i.e.", "e.g.") do NOT end sentences.
3. Parenthetical references (e.g., "(figure 1)", "(table 2)") are part of the sentence they follow.
4. When boundaries are unclear, extend to the next clear boundary.

For each sentence, copy its opening fragment (first ~6-10 words) and closing fragment (last ~6-10 words) VERBATIM from the input. Use enough words to uniquely locate the sentence.

Return JSON: {"sentences": [{"start": "verbatim opening...", "end": "...verbatim closing"}, ...]}"""

        # Parse document structure
        document_df = parse_document_structure(document)

        if sample:
            # For medical, sample the "Case presentation" section
            document_df = document_df[document_df['section'] == 'Case presentation'].copy()
            if document_df.empty:
                print("Warning: 'Case presentation' section not found. Running on full document.")
                document_df = parse_document_structure(document)

        # ─────────────────────────────────────────────────────────────
        # Step 1: Sentence Extraction
        #   LLM segments each subsection using start/end boundary markers.
        #   Debug: 01_paper_structure.txt, 02_extracted_markers.txt
        # ─────────────────────────────────────────────────────────────

        save_df_for_debugging(document_df, '01_paper_structure.txt', 'facts', document_name, ['section', 'subsection', 'paragraph'])

        # Create a dataframe with unique subsections
        subsection_df = document_df[['section', 'subsection', 'section_text', 'subsection_text']].drop_duplicates().reset_index(drop=True)

        def query_single(subsection_text):
            prompt = {}
            prompt['system'] = extraction_prompt
            prompt['user'] = f"""{subsection_text}"""
            return utils.query_llm(prompt, model='gpt-5.4-mini', reasoning_effort='medium', return_json=True, max_tokens=2000)


        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            futures = [executor.submit(query_single, row['subsection_text']) for _, row in subsection_df.iterrows()]
            marker_results = [future.result() for future in tqdm(futures, desc="Processing subsections")]

        subsection_df['sentence_markers_raw'] = marker_results
        save_df_for_debugging(subsection_df, '02_extracted_markers.txt', 'facts', document_name, ['section','subsection','sentence_markers_raw'])

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
        document_df_sentences = subsection_df.explode('sentence_list').rename(columns={'sentence_list': 'raw_knowledge_statement'})

        # Drop rows with no knowledge statements
        document_df_sentences = document_df_sentences[document_df_sentences['raw_knowledge_statement'].notna()]

        total_extracted_sentences = len(document_df_sentences)
        print(f"Extracted {total_extracted_sentences} sentences via marker matching.")

        # Marker matching inherently validates sentences exist in the source text.
        document_df_validated = document_df_sentences.copy()

        # Use subsection_text as context paragraph
        document_df_validated['paragraph'] = document_df_validated['subsection_text']

        # Drop the marker column
        document_df_validated = document_df_validated.drop(columns=['sentence_markers_raw'])

        print(f"Total validated sentences: {len(document_df_validated)}")
        save_df_for_debugging(document_df_validated, '03_validated_sentences.txt', 'facts', document_name, ['section', 'raw_knowledge_statement'])

        # (No LaTeX filter needed for medical documents — Step 2.1 skipped)
        # (No \ref/\label reference filter needed for medical documents — Step 2.2 skipped)

        # ─────────────────────────────────────────────────────────────
        # Step 2.3: Filter Short Sentences
        #   Remove sentences shorter than 90 characters.
        #   Debug: 06_short_filtered_out.txt
        # ─────────────────────────────────────────────────────────────

        # Filter out knowledge statements that are too short
        min_length = 90
        keep_length = document_df_validated['raw_knowledge_statement'].str.len() >= min_length

        # Separate kept and dropped statements
        dropped_statements_len = document_df_validated[~keep_length].copy()
        document_df_filtered_len = document_df_validated[keep_length].copy()

        save_df_for_debugging(dropped_statements_len, '06_short_filtered_out.txt', 'facts', document_name, ['raw_knowledge_statement'])
        save_df_for_debugging(document_df_filtered_len, '06_short_kept.txt', 'facts', document_name, ['raw_knowledge_statement'])

        # Report filtering results
        total_before = len(document_df_validated)
        total_after = sum(keep_length)
        filtered_out = total_before - total_after

        print(f"Length filtering results:")
        print(f"  Before filtering: {total_before} knowledge statements")
        print(f"  After filtering: {total_after} knowledge statements")
        print(f"  Filtered out: {filtered_out} statements shorter than {min_length} characters")

        document_df_filtered = document_df_validated[keep_length].reset_index(drop=True)

        # ─────────────────────────────────────────────────────────────
        # Step 2.4: Identify Knowledge Statements
        #   LLM classifies each sentence as knowledge vs non-knowledge.
        #   Debug: 07_knowledge_kept.txt, 07_knowledge_filtered_out.txt
        #   Checkpoint saved: 07_knowledge_kept_v10_5.csv
        # ─────────────────────────────────────────────────────────────

        prompt = {}
        prompt['system'] = """You will be receiving sentences from a medical case report. Your task is to determine whether the sentence contains a meaningful, clear fact that could be formulated into a question to test a reader's comprehension.

A sentence CONTAINS a fact if it:
- Presents a clear piece of medical knowledge, whether it's a clinical finding, a diagnosis, a treatment decision, a laboratory result with clinical significance, a pathophysiological mechanism, or an outcome.
- The sentence can be formulated into a meaningful fact.

This can be vague and so we provide a non-exhaustive list of some criteria for exclusion. A sentence does NOT contain a fact if it:
- Is semantically unclear, making it difficult to ascertain a clear fact.
- Is mainly a transitional sentence (e.g., "We now discuss the treatment.").
- Mainly describes the report's own structure or what it will cover (e.g., "This section describes the investigations performed.").
- References a figure or table and provides information that requires the figure/table to make sense.
- Poses a rhetorical question.
- Is a bare acknowledgment or consent statement without medical content.

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
            try:
                parsed = json.loads(result) if result else {}
                is_knowledge = parsed.get('is_knowledge', False)
            except (json.JSONDecodeError, TypeError):
                print(f"Failed to parse knowledge eval for idx {idx}, defaulting to False. Response: {result}")
                is_knowledge = False

            return idx, is_knowledge

        # Run evaluation once and store results
        print("Evaluating sentences for knowledge content...")

        is_knowledge_results = [None] * len(document_df_filtered)

        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = {executor.submit(evaluate_statement, (idx, row)): idx
                       for idx, row in document_df_filtered.iterrows()}

            for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating sentences"):
                idx, is_knowledge = future.result()
                original_idx = list(document_df_filtered.index).index(idx)
                is_knowledge_results[original_idx] = is_knowledge

        # Add knowledge column
        document_df_filtered['is_knowledge'] = is_knowledge_results

        document_df_knowledge = document_df_filtered[document_df_filtered['is_knowledge']].copy()
        document_df_non_knowledge = document_df_filtered[~document_df_filtered['is_knowledge']].copy()
        save_df_for_debugging(document_df_knowledge, '07_knowledge_kept.txt', 'facts', document_name, ['raw_knowledge_statement', 'is_knowledge'])
        save_df_for_debugging(document_df_non_knowledge, '07_knowledge_filtered_out.txt', 'facts', document_name, ['raw_knowledge_statement', 'is_knowledge'])

        # Report filtering results
        total_before = len(document_df_filtered)
        total_after = len(document_df_knowledge)
        filtered_out = total_before - total_after

        print(f"\nFinal knowledge filtering results:")
        print(f"  Before filtering: {total_before} sentences")
        print(f"  After filtering: {total_after} knowledge statements")
        print(f"  Filtered out: {filtered_out} non-knowledge sentences")

        # Save checkpoint
        print(f"Saving checkpoint to {checkpoint_path}...")
        os.makedirs(checkpoint_dir, exist_ok=True)
        document_df_knowledge.to_csv(checkpoint_path, index=False)

    document_df_knowledge.reset_index(drop=True, inplace=True)

    # Look up the proper case report title from the mapping
    report_title = CASE_REPORT_TITLES.get(document_name, document_name.replace('_', ' '))
    document_df_knowledge['title'] = report_title

    # ─────────────────────────────────────────────────────────────
    # Step 3: Extract Self-Contained Atomic Probes
    #   For each knowledge statement:
    #     3a. Extract Q&A pairs (gpt-5.4)
    #     3b. Contextualize questions (gpt-5.4-mini)
    #     3c. Convert to cloze format (gpt-5.4-mini)
    #   Debug: 08_a_atomic_facts.txt
    # ─────────────────────────────────────────────────────────────

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
        """Extract atomic facts from a sentence using LLM."""
        prompt = {}
        prompt['system'] = r"""You will be given two inputs, a section of a medical case report for context and a single sentence drawn from that section. Medical writing often interweaves various pieces of knowledge together. While each sentence is interwoven with others, there is atomic knowledge that can be extracted from a particular sentence. Write questions that test for this atomic knowledge.

Extract 1-2 questions from the sentence. If the sentence has lots of facts, extract up to 3 questions.

### Detailed Instructions
Consider these instructions as you extract each question:
- The question should be meaningful, focused on a main fact in the sentence, not a minor detail.
- The question should be non-trivial and non-obvious. It should not be plainly obvious from the question for someone with no relevant knowledge.
- The answer to the question MUST be a verbatim word, phrase (2-4 words), or expression copied exactly from the sentence. Capitalization can be adjusted appropriately. Otherwise, do not paraphrase, rephrase, or adjust the answer; it must appear as an exact substring of the sentence.
- The question should have a clear, single answer and *NOT* multiple valid answers.
- Prefer shorter answers.
- Each question should be written separately and independently of the other questions; do not reference other questions in the same question.

### Demonstration 1: Clinical Finding
Context: "Initial laboratory tests revealed marked hyponatraemia (124 mmol/L), elevated creatine kinase (1349 U/L), aspartate aminotransferase (77 U/L), alanine aminotransferase (64 U/L), C-reactive protein concentration (CRP) (180 mg/L; reference, <3 mg/L) and erythrocyte sedimentation rate (70 mm/hour; reference, <15 mm/hour), with normal renal function."

Sentence: "Initial laboratory tests revealed marked hyponatraemia (124 mmol/L), elevated creatine kinase (1349 U/L), aspartate aminotransferase (77 U/L), alanine aminotransferase (64 U/L), C-reactive protein concentration (CRP) (180 mg/L; reference, <3 mg/L) and erythrocyte sedimentation rate (70 mm/hour; reference, <15 mm/hour), with normal renal function."

Questions:
- "What was the patient's serum sodium level on initial laboratory testing?", Answer: "124 mmol/L"
- "What was the patient's C-reactive protein concentration on initial testing?", Answer: "180 mg/L"

### Demonstration 2: Diagnosis
Context: "This patient was diagnosed with definite BD based on clinical features, including fever and fatigue, oral aphthae (two points), genital aphthosis (two points) and a folliculitis-like rash (one point). According to the International Criteria for Behçet's Disease, he had a total score of five points (≥4 indicates definite disease)."

Sentence: "According to the International Criteria for Behçet's Disease, he had a total score of five points (≥4 indicates definite disease)."

Questions:
- "According to the International Criteria for Behçet's Disease, what total score indicates definite disease?", Answer: "≥4"
- "What was the patient's total score according to the International Criteria for Behçet's Disease?", Answer: "five points"

### Demonstration 3: Treatment
Context: "On day 8, colchicine was started following the diagnosis of systemic BD and acute neuro-BD. Oral aphthous lesions were treated with topical corticosteroid ointment. Genital ulcers and cutaneous lesions were managed with silver sulfadiazine cream."

Sentence: "Genital ulcers and cutaneous lesions were managed with silver sulfadiazine cream."

Questions:
- "What topical agent was used to manage the patient's genital ulcers and cutaneous lesions?", Answer: "silver sulfadiazine cream"
"""
        contextualize_prompt = {}
        contextualize_prompt['system'] = r"""You will be given two inputs, a section of a medical case report for context, a single sentence drawn from that section, and a question extracted from the sentence as well as its corresponding answer. Your task is to then turn the question into a self-contained, precise question. Approach this task step-by-step as outlined below.

While you should use your expertise on this domain to handle and understand these texts, all information written into the questions and answers *MUST* originate from the provided context or sentence. Do not add, infer, or correct information using your internal knowledge. Every detail should be traceable back to the source text. As you write and rewrite the questions, also make sure to accurately represent the knowledge in the original sentence without distortion. Strive to use phrasing as close as possible to the original text, but prioritize clarity and self-containment. The questions should be written clearly so that they are easy to read.

### Instructions
The goal of this task is to make the questions clear by incorporating the relevant context. This ensures the question is unambiguous and doesn't require looking back to the source material.

For each question:
1.  Rewrite the question so that it starts with one of the following templates. Always wrap the case report title in double quotes — never use single quotes, asterisks, or underscores around the title.
    - "In the case report \"{title}\", ..."
    - "According to the case report \"{title}\", ..."
    - "In the case report \"{title}\", the authors describe that..."
    - "In the case report \"{title}\", the patient presented with..."
    - "According to the case report \"{title}\", laboratory tests revealed..."
    - "In the case report \"{title}\", the treatment involved..."
    - "In the case report \"{title}\", the diagnosis was based on..."
    This is a non-exhaustive list of templates, and you should use your own judgement to choose the most appropriate template or modify the template to fit the sentence.
2.  Add sufficient context. Specifically, use the *provided context* to supply whatever information is needed to make the question self-contained and unambiguous. For instance, "What was the sodium level?" should be clarified into "In the case report '...', what was the patient's serum sodium level on initial laboratory testing at admission?" if this was in the context of initial workup. The goal is to ensure someone reading just the question would understand exactly what is being asked without needing additional context.
3.  Clarify pronouns and referential terms. Check the sentence for pronouns (it, this, that, these, those) or demonstrative phrases (this condition, that treatment, these findings) that refer to entities not explicitly defined within the sentence itself. Search the surrounding context to identify what these terms reference, then incorporate that clarifying information into the question to make it self-contained.
4.  Clarify Context-Dependent Terms. Named entities (e.g., drug names, disease names, specific tests) do not need clarification. However, if there are unnamed or context-specific terms (e.g., "the patient", "the level", "the medication"), clarify their full context.
5.  Disambiguate clinical events. There are often numerous investigations, treatments, and timepoints in a case report, so supply enough clinical context that the question clearly identifies which event it addresses.
6.  Handle medical abbreviations. If the answer is a common medical abbreviation (e.g., CRP, CSF, MRI, BD) and the abbreviation appears frequently in the context, feel free to leave it as an abbreviation without defining it.
7.  Do not leak the answer. Please make sure that *the answer is not revealed* in the question. The answer should never appear in the question.
8.  Do not change the answer. Minor grammatical adjustments to the answer are allowed only if necessary to fit the restructured question (e.g., adjusting verb tense, determiners like "the").
9.  Avoid quoting the source sentence directly in the question.
10. Refine Question. The rewritten question can be broken up into multiple sentences if the question becomes verbose. Make sure the question is written clearly and grammatically correct. Do not put any of the context in parenthesis or followed after an "i.e.".

Think carefully and critically through this task, following the step-by-step instructions outlined above. Then, provide the final output, listing each question and its corresponding answer."""
        cloze_prompt = {}
        cloze_prompt['system'] = FACT_PROBE_CLOZE_PROMPT_SYSTEM
        json_parse_prompt = {
            'system': """You will be given a string containing questions and answers. Convert it into a JSON object with a single key "list_of_questions", which contains a list of objects. Each object in the list should have a "question" and "answer" key. The format should be: {"list_of_questions": [{"question": "...", "answer": "..."}, ...]}. Copy the question and answer content exactly as it appears. Do not modify the text.""",
            'user': ""
        }

        context = extract_context_and_sentence(first_row)
        prompt['user'] = f"""### Case Report Title\n{first_row['title']}\n### Context\n{context}\n\n### Sentence\n{first_row['raw_knowledge_statement'].strip()}"""
        output1 = utils.query_llm(prompt, model='gpt-5.4', reasoning_effort='low')

        json_parse_prompt['user'] = output1
        try:
            json_output = utils.query_llm(json_parse_prompt, model='gpt-5.4-nano', return_json=True, reasoning_effort='low')
            qa_pairs_data = json.loads(json_output)
            qa_pairs = qa_pairs_data.get('list_of_questions')
            if qa_pairs is None:
                print(f"Key 'list_of_questions' not found in JSON output: {qa_pairs_data}")
                return output1, "", ""
        except (json.JSONDecodeError, TypeError):
            print(f"Failed to parse Q&A pairs from: {output1}")
            return output1, "", ""

        if not isinstance(qa_pairs, list):
            print(f"Parsed 'list_of_questions' is not a list: {qa_pairs}")
            return output1, "", ""

        all_contextualized = []
        all_clozes = []

        qa_pairs = [p for p in qa_pairs if p.get('question') and p.get('answer')]

        for pair in qa_pairs:
            question = pair['question']
            answer = pair['answer']

            single_qa_string = f'### Question\n"{question}\n\n###Answer\n"{answer}"'

            contextualize_prompt['user'] = f"""### Case Report Title\n{first_row['title']}\n### Context\n{context}\n\n### Sentence\n{first_row['raw_knowledge_statement'].strip()}\n\n{single_qa_string}"""
            output2_individual = utils.query_llm(contextualize_prompt, model='gpt-5.4-mini', reasoning_effort='medium')
            all_contextualized.append(output2_individual)

            cloze_prompt['user'] = f"""### Question and Answer\n{output2_individual}"""
            output3_individual = utils.query_llm(cloze_prompt, system_prompt_included=True, model='gpt-5.4-mini', reasoning_effort='medium', return_json=True)
            try:
                cloze_pair = json.loads(output3_individual)
                if isinstance(cloze_pair, dict) and 'answer' in cloze_pair and 'statement' in cloze_pair:
                    all_clozes.append(cloze_pair)
            except (json.JSONDecodeError, TypeError):
                print(f"Failed to parse cloze pair from: {output3_individual}")

        return output1, all_contextualized, all_clozes


    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        raw_extracted_facts = list(tqdm(executor.map(extract_atomic_facts, [document_df_knowledge.iloc[i] for i in range(len(document_df_knowledge))]), total=len(document_df_knowledge)))

    # Separate the three outputs into different columns
    questions = [result[0] if result is not None else None for result in raw_extracted_facts]
    contextualized_questions = [result[1] if result is not None else None for result in raw_extracted_facts]
    cloze_pairs = [result[2] if result is not None else [] for result in raw_extracted_facts]

    # Store in dataframe
    document_df_knowledge['questions'] = questions
    document_df_knowledge['contextualized_questions'] = contextualized_questions
    document_df_knowledge['cloze_pairs'] = cloze_pairs
    save_df_for_debugging(document_df_knowledge, '08_a_atomic_facts.txt', 'facts', document_name, ['raw_knowledge_statement', 'questions', 'contextualized_questions', 'cloze_pairs'])

    print(f"Processed {len(raw_extracted_facts)} rows")

    # ─────────────────────────────────────────────────────────────
    # Step 4: Quality Control — Refinement
    #   LLM reviews each (answer, statement) pair for:
    #   formatting, answer placement, answer leakage.
    #   Returns {"change": false} or refined pair.
    #   Debug: 08_c_qc_changed.txt, 08_c_qc_unchanged.txt
    # ─────────────────────────────────────────────────────────────

    def validate_atomic_facts(row):
        """Process a single atomic fact for validation."""
        prompt = {}
        prompt['system'] = """Your task is to review and refine a statement that has been extracted from a sentence taken from a medical case report. You will be given an '(answer, statement)' pair and a checklist.

### Quality Control Checklist
For each '(answer, statement)' pair, check the following:

1. Formatting
- Leave numbers as how they are written in the original sentence. e.g. eight should be eight and 8 should be 8.
- Ensure medical units, abbreviations, and values are preserved accurately.

2. Declarative
- The statement should be written like a declarative sentence without question marks.

3. Answer Placement
- The statement must be a COMPLETE sentence that ENDS WITH the answer as its final words.
- Do NOT remove the answer from the statement. The answer must be present and be the last words.
- Example: if the answer is "124 mmol/L", the statement should end with "...124 mmol/L."

3. Answer Leakage
- If the answer appears ONLY ONCE in the statement and it is at the end, there is NO leakage — this is correct.
- Leakage occurs when the answer, or a semantically equivalent paraphrase of the answer, appears earlier in the statement, giving away the answer before the reader reaches the end.

### Output Format (JSON)
- If the pair already passes ALL checks with no changes needed, return: {"change": false}
- If any refinement is needed, minimally rewrite and return: {"change": true, "answer": "...", "statement": "..."}

Prefer returning {"change": false}. Only refine if there is a clear, concrete issue. Do not make unnecessary changes."""
        prompt['user'] = f"""### Answer\n{row['answer']}\n\n### Statement\n{row['statement']}"""
        response = utils.query_llm(prompt, model='gpt-5.4-mini', reasoning_effort='medium', system_prompt_included=True, return_json=True)
        try:
            parsed_response = json.loads(response)
            if not parsed_response.get('change', True):
                return {'answer': row['answer'], 'statement': row['statement'], 'was_refined': False}
            if parsed_response and 'answer' in parsed_response and 'statement' in parsed_response:
                return {'answer': parsed_response['answer'], 'statement': parsed_response['statement'], 'was_refined': True}
        except (json.JSONDecodeError, TypeError):
            print(f"Failed to parse QC response, using original pair for row. Response: {response}")
        print(f"Failed to parse QC response, using original pair for row. Response: {response}")
        return {'answer': row['answer'], 'statement': row['statement'], 'was_refined': False}

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
3. Answer leakage: The answer should not be leaked in the statement before the answer appears.
4. Non trivial: The statement should be non-trivial. It should not be plainly obvious from the question itself for someone with no relevant knowledge.

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
    for idx, row in document_df_knowledge.iterrows():
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

    document_df_exploded = pd.DataFrame(rows_for_qc)
    print(f"Exploded {len(document_df_knowledge)} knowledge statements into {len(document_df_exploded)} candidate pairs.")


    # Process all rows in parallel
    if not document_df_exploded.empty:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            validated_results = list(tqdm(
                executor.map(validate_atomic_facts, [row for _, row in document_df_exploded.iterrows()]),
                total=len(document_df_exploded)
            ))
        document_df_exploded['validated_atomic_pairs'] = validated_results
    else:
        document_df_exploded['validated_atomic_pairs'] = []


    # Track and save changes made during QC
    changed_pairs_data = []
    unchanged_pairs_data = []
    for idx, row in document_df_exploded.iterrows():
        original_answer = row['answer']
        original_statement = row['statement']
        validated_pair = row['validated_atomic_pairs']

        if validated_pair:
            was_refined = validated_pair.get('was_refined', False)
            new_answer = validated_pair.get('answer')
            new_statement = validated_pair.get('statement')

            if was_refined:
                changed_pairs_data.append({
                    'raw_knowledge_statement': row['raw_knowledge_statement'],
                    'original_answer': original_answer,
                    'new_answer': new_answer,
                    'original_statement': original_statement,
                    'new_statement': new_statement
                })
            else:
                unchanged_pairs_data.append({
                    'raw_knowledge_statement': row['raw_knowledge_statement'],
                    'answer': original_answer,
                    'statement': original_statement
                })

    num_refined = len(changed_pairs_data)
    num_unchanged = len(unchanged_pairs_data)
    print(f"QC summary: {num_refined} refined, {num_unchanged} unchanged out of {len(document_df_exploded)} total pairs.")

    if changed_pairs_data:
        document_df_qc_changed = pd.DataFrame(changed_pairs_data)
        save_df_for_debugging(document_df_qc_changed, '08_c_qc_changed.txt', 'facts', document_name,
                              ['raw_knowledge_statement', 'original_answer', 'new_answer', 'original_statement', 'new_statement'])

    if unchanged_pairs_data:
        document_df_qc_unchanged = pd.DataFrame(unchanged_pairs_data)
        save_df_for_debugging(document_df_qc_unchanged, '08_c_qc_unchanged.txt', 'facts', document_name,
                              ['raw_knowledge_statement', 'answer', 'statement'])

    document_df_qc_kept = document_df_exploded

    # ─────────────────────────────────────────────────────────────
    # Step 4.5: Quality Control — Filtering
    #   LLM decides whether each refined pair meets quality standards:
    #   linguistically reasonable, semantically unambiguous, clear.
    #   Debug: 08_d_final_filter_dropped.txt
    # ─────────────────────────────────────────────────────────────
    if not document_df_qc_kept.empty:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            keep_results = list(tqdm(
                executor.map(filter_refined_pair, [row for _, row in document_df_qc_kept.iterrows()]),
                total=len(document_df_qc_kept),
                desc="Filtering refined pairs"
            ))
        document_df_qc_kept['final_keep'] = keep_results
    else:
        document_df_qc_kept['final_keep'] = []

    document_df_after_filter_dropped = document_df_qc_kept[~document_df_qc_kept['final_keep']].copy()
    document_df_qc_kept = document_df_qc_kept[document_df_qc_kept['final_keep']].copy()

    if not document_df_after_filter_dropped.empty:
        save_df_for_debugging(document_df_after_filter_dropped, '08_d_final_filter_dropped.txt', 'facts', document_name,
                              ['raw_knowledge_statement', 'validated_atomic_pairs'])

    print(f"\nAfter final filtering step:")
    print(f"Kept {len(document_df_qc_kept)} pairs.")
    print(f"Dropped {len(document_df_after_filter_dropped)} pairs.")

    # Directly create 'target' and 'fact' columns from 'validated_atomic_pairs'
    document_df_qc_kept['target'] = document_df_qc_kept['validated_atomic_pairs'].apply(lambda x: x['answer'])
    document_df_qc_kept['fact'] = document_df_qc_kept['validated_atomic_pairs'].apply(lambda x: x['statement'])
    document_df_with_probes = document_df_qc_kept
    document_df_with_probes = document_df_with_probes[['section','subsection', 'section_text','subsection_text', 'raw_knowledge_statement', 'target', 'fact', 'contextualized_question']]


    # ─────────────────────────────────────────────────────────────
    # Step 5: Validate Target Placement & Build Probes
    #   Check that each fact ends with its target. Fix '(' prefix facts.
    #   Split fact into (probe, target) columns.
    #   Debug: 09_fixed_paren_facts.txt, 10_target_end_kept.txt,
    #          10_target_end_filtered_out.txt
    # ─────────────────────────────────────────────────────────────

    # Fix facts that start with '(' by extracting the rightmost part after target + ','
    facts_starting_with_paren = document_df_with_probes[document_df_with_probes['fact'].str.startswith('(')]
    print(f"Found {len(facts_starting_with_paren)} facts starting with '(' - fixing these...")

    for idx, row in facts_starting_with_paren.iterrows():
        target_with_comma = row['target'] + ','
        if target_with_comma in row['fact']:
            rightmost_part = row['fact'].split(target_with_comma)[-1].strip().strip('()')
            # Update the fact in the dataframe
            document_df_with_probes.loc[idx, 'fact'] = rightmost_part

    print(f"Fixed {len(facts_starting_with_paren)} facts that started with '('")
    save_df_for_debugging(document_df_with_probes, '09_fixed_paren_facts.txt', 'facts', document_name, ['raw_knowledge_statement', 'fact'])

    # Check that facts end with targets (after stripping whitespace and punctuation)
    valid_facts = []
    filtered_count = 0
    dropped_examples = []

    for idx, row in document_df_with_probes.iterrows():
        fact = str(row['fact']).strip()
        target = ' ' +str(row['target']).strip().rstrip(string.punctuation + string.whitespace)

        # Remove punctuation from the end of fact for comparison
        fact_cleaned = fact.rstrip(string.punctuation + string.whitespace)

        # Check if fact ends with target
        if fact_cleaned.endswith(target) and target in fact_cleaned:
            valid_facts.append(True)
        else:
            valid_facts.append(False)
            filtered_count += 1
            dropped_examples.append(row.to_dict())

    if not document_df_with_probes.empty:
        document_df_with_probes['valid_fact'] = valid_facts
    else:
        print("No probes to validate.")
        return

    print(f"Filtered out {filtered_count} facts that don't end with their target")
    print(f"Remaining valid facts: {len(document_df_with_probes) - filtered_count}")

    # Filter to keep only valid facts and save both kept and dropped
    document_df_probes_valid = document_df_with_probes[document_df_with_probes['valid_fact']].copy()
    document_df_probes_invalid = document_df_with_probes[~document_df_with_probes['valid_fact']].copy()

    save_df_for_debugging(document_df_probes_valid, '10_target_end_kept.txt', 'facts', document_name, ['raw_knowledge_statement', 'fact', 'target'])
    if not document_df_probes_invalid.empty:
        save_df_for_debugging(document_df_probes_invalid, '10_target_end_filtered_out.txt', 'facts', document_name, ['raw_knowledge_statement', 'fact', 'target'])


    # Create probe column (fact minus target) using stripped, no punctuation versions
    probes = []
    cleaned_facts = []
    cleaned_targets = []

    for idx, row in document_df_probes_valid.iterrows():
        fact = str(row['fact']).strip()
        target = str(row['target']).strip()

        fact_cleaned = fact
        target_cleaned = ' ' + target


        last_index = fact_cleaned.rfind(target_cleaned)
        if last_index != -1:
            probe = fact_cleaned[:last_index].strip()
            target_cleaned = fact_cleaned[last_index:]  # include any trailing punctuation
            probes.append(probe)
            cleaned_facts.append(fact_cleaned)
            cleaned_targets.append(target_cleaned)
        else:
            print(fact)
            print(target)
            raise ValueError(f"Target {target_cleaned} not found in fact {fact_cleaned}")

    document_df_probes_valid['probe'] = probes
    document_df_probes_valid['fact'] = cleaned_facts
    document_df_probes_valid['target'] = cleaned_targets

    print(f"Created probe column by removing target from fact")
    print(f"Final dataset shape: {document_df_probes_valid.shape}")
    save_df_for_debugging(document_df_probes_valid, '11_final_probes.txt', 'facts', document_name, ['raw_knowledge_statement', 'fact', 'target', 'probe'])

    # ─────────────────────────────────────────────────────────────
    # Step 6: Tokenizer Consistency Check
    #   Verify that tokenizing probe+target together matches
    #   tokenizing them separately (for training correctness).
    # ─────────────────────────────────────────────────────────────

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B")

    check_tokenizer_consistency(document_df_probes_valid, tokenizer)

    # ─────────────────────────────────────────────────────────────
    # Step 7: Verbatim Filter
    #   Keep only probes whose target appears verbatim in the document.
    #   Debug: 12_verbatim_filtered_out.txt
    # ─────────────────────────────────────────────────────────────

    total_before_verbatim = len(document_df_probes_valid)
    verbatim_mask = document_df_probes_valid['target'].apply(
        lambda t: t.strip() in document
    )
    document_df_not_verbatim = document_df_probes_valid[~verbatim_mask].copy()
    document_df_probes_valid = document_df_probes_valid[verbatim_mask].copy()
    total_after_verbatim = len(document_df_probes_valid)
    filtered_verbatim = total_before_verbatim - total_after_verbatim

    print(f"\nVerbatim filtering results:")
    print(f"  Before filtering: {total_before_verbatim} probes")
    print(f"  After filtering: {total_after_verbatim} probes")
    print(f"  Filtered out: {filtered_verbatim} probes whose target is not verbatim in the document")

    save_df_for_debugging(document_df_not_verbatim, '12_verbatim_filtered_out.txt', 'facts', document_name, ['raw_knowledge_statement', 'fact', 'target', 'probe'])

    # ─────────────────────────────────────────────────────────────
    # Step 7.5: Verbatim Recovery
    #   LLM attempts to fix non-verbatim probes by finding a
    #   verbatim substring from the source sentence.
    #   Debug: 12b_verbatim_recovered.txt, 12c_verbatim_unrecoverable.txt
    # ─────────────────────────────────────────────────────────────

    def recover_verbatim_probe(row):
        """Ask the LLM to replace the answer with a verbatim phrase from the source sentence."""
        prompt = {}
        prompt['system'] = r"""You are given a cloze-style probe statement and its answer, both derived from a sentence in a medical case report. The answer does not appear verbatim in the original sentence — often because the phrasing was slightly altered.

Your task:
1. Find a verbatim substring from the original sentence that captures the same knowledge as the current answer.
2. Minimally adjust the statement so that it ends with this new verbatim answer and reads naturally.
3. The new answer MUST be an exact, character-for-character substring of the original sentence.
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

    if not document_df_not_verbatim.empty:
        print(f"\nAttempting to recover {len(document_df_not_verbatim)} non-verbatim probes...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            recovery_results = list(tqdm(
                executor.map(recover_verbatim_probe, [row for _, row in document_df_not_verbatim.iterrows()]),
                total=len(document_df_not_verbatim),
                desc="Recovering non-verbatim probes"
            ))

        recovered_rows = []
        failed_rows = []
        for (idx, row), result in zip(document_df_not_verbatim.iterrows(), recovery_results):
            if result.get('success') and 'answer' in result and 'statement' in result:
                new_answer = result['answer']
                # Verify the recovered answer is actually verbatim in the document
                if new_answer.strip() in document:
                    new_row = row.copy()
                    new_row['target'] = ' ' + new_answer.strip()
                    new_row['fact'] = result['statement']
                    # Rebuild probe by removing target from end of fact
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
            document_df_recovered = pd.DataFrame(recovered_rows)
            save_df_for_debugging(document_df_recovered, '12b_verbatim_recovered.txt', 'facts', document_name, ['raw_knowledge_statement', 'fact', 'target', 'probe'])
            document_df_probes_valid = pd.concat([document_df_probes_valid, document_df_recovered], ignore_index=True)

        if failed_rows:
            document_df_failed = pd.DataFrame(failed_rows)
            save_df_for_debugging(document_df_failed, '12c_verbatim_unrecoverable.txt', 'facts', document_name, ['raw_knowledge_statement', 'fact', 'target', 'probe'])

        total_after_recovery = len(document_df_probes_valid)
        print(f"  Total probes after recovery: {total_after_recovery}")

    # Save filtering metrics report
    output_dir = f'../../data/probes/facts/{document_name}/'
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, 'filtering_metrics_v10_5.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"Fact Probe Pipeline (medical) v10_5 - Filtering Metrics for {document_name}\n")
        f.write(f"{'='*60}\n")
        f.write(f"Total probes before verbatim filter: {total_before_verbatim}\n")
        f.write(f"Probes filtered (target not verbatim in document): {filtered_verbatim}\n")
        f.write(f"Total probes after verbatim filter: {total_after_verbatim}\n")
    print(f"Saved filtering metrics to {metrics_path}")

    # ─────────────────────────────────────────────────────────────
    # Step 8: Save Final Probes
    #   Output: probes_v10_5.csv, probes_v10_5_readable.txt,
    #           filtering_metrics_v10_5.txt
    # ─────────────────────────────────────────────────────────────

    document_df_probes_valid.reset_index(drop=True, inplace=True)
    document_df_probes_valid.to_csv(os.path.join(output_dir, 'probes_v10_5.csv'), index=False)

    # Save readable version
    readable_path = os.path.join(output_dir, 'probes_v10_5_readable.txt')
    with open(readable_path, 'w') as f:
        for _, row in document_df_probes_valid.iterrows():
            f.write(f"{row['probe']}: {row['target'].lstrip()}\n")
    print(f"Saved readable probes to {readable_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter", type=str, default=None, help="Only process documents containing this string in their filename.")
    parser.add_argument("--sample", action="store_true", help="Run on a sample (Case presentation section only).")
    args = parser.parse_args()

    process_papers(generate_probes_for_document, '../../data/medical/cleaned/', file_filter=args.filter, extension='.txt', sample=args.sample)
