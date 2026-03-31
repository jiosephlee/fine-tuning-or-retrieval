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


        extraction_prompt = r"""You will be given a section of text from an academic paper. Your goal is to accurately identify and segment every complete sentence in the text.

Here are guidelines as you segment the text:
1. In papers, complete sentences can include LaTeX commands, formatting, section headers (like "\textbf{...}:"), and mathematical expressions that form part of the sentence structure.
2. Be careful with abbreviations (e.g., "et al.", "Fig.", "Eq.") and other punctuation like "i.e." or "e.g." that use periods but do not end a sentence.
3. Wrap each identified complete sentence in '[BOS]' and '[EOS]' tags.
4. Ensure that the tags cover the ENTIRE sentence, from the first word, to any equations in LaTeX, and all the way to the complete ending of the sentence.
5. In particular, please look out for multi-line equations followed by "where" or "i.e." that CONTINUE the sentence. For instance: "[BOS] The equation...\begin{equation}...\end{equation} where ... $x$ is a variable. [EOS]" is ONE whole sentence. 
    - Please double check for these cases: sentences that are right before and after equations.
6. Sometimes the boundaries of sentences can be unclear. Whenever the case, extend the sentence to the next complete sentence.
7. Return the entire original text with these annotations. Do not modify or summarize the text itself. You must not change wording, punctuation, or line breaks. Only add the tags."""

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

        # Process each subsection with LLM
        def query_single(subsection_text):
            prompt = {}
            prompt['system'] = extraction_prompt
            prompt['user'] = f"""{subsection_text}"""
            return utils.query_llm(prompt, model='gpt-5.4', reasoning_effort='low')
        
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            futures = [executor.submit(query_single, row['subsection_text']) for _, row in subsection_df.iterrows()]
            tagged_sentences_subsections = [future.result() for future in tqdm(futures, desc="Processing subsections")]
        # tagged_sentences_subsections = []
        # for i, (_, row) in enumerate(subsection_df.iterrows()):
        #     if i == 3:
        #         result = query_single(row['subsection_text'])
        #     else:
        #         result = ""
        #     tagged_sentences_subsections.append(result)

        # Add extracted claims to subsection dataframe
        subsection_df['tagged_sentences'] = tagged_sentences_subsections
        save_df_for_debugging(subsection_df, '02_extracted_sentences.txt', 'facts', paper_name, ['section','subsection','tagged_sentences'])

        ## 1.1 check knowledge is actually in the paper 

        def extract_text_from_sentence_tags(text: str) -> list[str]:
            """
            Finds all <sentence> tags in a given text and extracts their content.

            Args:
                text: A string containing the text to parse, which may include
                  <sentence>...</sentence> tags.

            Returns:
                A list of strings, where each string is the content found within
                a <sentence> tag. The content is stripped of leading/trailing
                whitespace.
            """
            # This regex pattern finds all content between <sentence> and </sentence>.
            # - The (.*?) part is a non-greedy capture group for the content inside the tags.
            # - The re.DOTALL flag allows the '.' character to match newlines, so tags
            #   that span multiple lines are correctly handled.
            pattern = re.compile(r'\[BOS\](.*?)\[EOS\]', re.DOTALL)
            
            # re.findall returns a list of all captured groups.
            matches = pattern.findall(text)
            
            # Clean up any leading/trailing whitespace from the extracted text.
            cleaned_matches = [match.strip() for match in matches]
            
            return cleaned_matches

        def remove_sentence_tags(text: str) -> str:
            """Remove sentence tags from text while preserving the content."""
            pattern = re.compile(r'[BOS][EOS]', re.DOTALL)
            return pattern.sub('', text)

        # Extract a list of knowledge statements from each subsection
        subsection_df['sentence_list'] = subsection_df['tagged_sentences'].apply(extract_text_from_sentence_tags)

        # Explode the DataFrame on the knowledge_list column
        paper_df_sentences = subsection_df.explode('sentence_list').rename(columns={'sentence_list': 'raw_knowledge_statement'})

        # Drop rows with no knowledge statements
        paper_df_sentences = paper_df_sentences[paper_df_sentences['raw_knowledge_statement'].notna()]

        # Count total sentences extracted by the LLM before filtering
        total_extracted_sentences = len(paper_df_sentences)
        # Filter out sentences that are not actually in the original paper text (case-insensitive)
        paper_df_sentences['is_validated'] = paper_df_sentences['raw_knowledge_statement'].apply(
            lambda s: is_text_in_document(s, paper)
        )
        
        paper_df_validated = paper_df_sentences[paper_df_sentences['is_validated']].copy()
        paper_df_not_validated = paper_df_sentences[~paper_df_sentences['is_validated']].copy()

        # Count sentences that passed validation
        validated_sentences_count = len(paper_df_validated)
        not_validated_sentences_count = len(paper_df_not_validated)
        print(f"Found {validated_sentences_count}/{total_extracted_sentences} extracted sentences in the original paper text.")
        print(f"Found {not_validated_sentences_count} sentences that were NOT in the original paper text.")

        # The `paragraph` column is still needed for context in later steps.
        paper_df_validated['paragraph'] = paper_df_validated['tagged_sentences'].apply(remove_sentence_tags)
        paper_df_not_validated['paragraph'] = paper_df_not_validated['tagged_sentences'].apply(remove_sentence_tags)

        # Drop the now-redundant columns
        paper_df_validated = paper_df_validated.drop(columns=['tagged_sentences', 'is_validated'])
        paper_df_not_validated = paper_df_not_validated.drop(columns=['tagged_sentences', 'is_validated'])

        # Save both validated and not validated sentences
        print(f"Total validated sentences: {len(paper_df_validated)}")
        save_df_for_debugging(paper_df_validated, '03_validated_sentences.txt', 'facts', paper_name, ['section', 'raw_knowledge_statement'])
        
        print(f"Total not validated sentences: {len(paper_df_not_validated)}")
        save_df_for_debugging(paper_df_not_validated, '03_not_validated_sentences.txt', 'facts', paper_name, ['section', 'raw_knowledge_statement'])
        
        # Filter bad sentences
        """Not all extracted knowledge statements may be valid. We double check that the extracted statements meet our requirements for a proper probe."""

        ## 2.1 FIlter out Predominantly latex sentences

        """This is to avoid figures, tables, and sentences that are dominated by LaTeX that prevents any suitable English target. LaTeX has various valid formats which can make evaluation tricky. There's also the chance that LaTeX introduces noise with regards to learnability. While OLMo has been trained on arxiv documents that include latex,it potentially may require further pre-training on mathematical notation and latex for the LLM to understand these statements."""

        # Filter out knowledge statements that are more than 50% LaTeX
        def calculate_latex_percentage(text):
            """
            Calculate the percentage of LaTeX/mathematical content in a text string.
            
            Args:
                text (str): The text to analyze
                
            Returns:
                float: Percentage of text that is LaTeX/mathematical (0-100)
            """
            if pd.isna(text) or not text.strip():
                return 0.0
            
            total_chars = len(text)
            latex_chars = 0
            
            # Count LaTeX commands (backslash followed by letters)
            latex_commands = re.findall(r'\\[a-zA-Z]+', text)
            for cmd in latex_commands:
                latex_chars += len(cmd)
            
            # Remove LaTeX commands to avoid double counting
            text_without_commands = re.sub(r'\\[a-zA-Z]+', '', text)
            
            # Count non-alphabetic characters in the remaining text
            for char in text_without_commands:
                if not char.isalpha() and not char.isspace():
                    latex_chars += 1
            
            # Calculate percentage
            latex_percentage = (latex_chars / total_chars) * 100 if total_chars > 0 else 0.0
            
            return latex_percentage

        # Apply the filter
        paper_df_validated['latex_percentage'] = paper_df_validated['raw_knowledge_statement'].apply(calculate_latex_percentage)

        # Filter out statements with more than 85% LaTeX
        latex_threshold = 75
        high_latex_statements = paper_df_validated[paper_df_validated['latex_percentage'] > latex_threshold].copy()
        paper_df_filtered = paper_df_validated[paper_df_validated['latex_percentage'] <= latex_threshold].copy()

        save_df_for_debugging(high_latex_statements, '04_latex_filtered_out.txt', 'facts', paper_name, ['raw_knowledge_statement', 'latex_percentage'])
        save_df_for_debugging(paper_df_filtered, '04_latex_kept.txt', 'facts', paper_name, ['raw_knowledge_statement', 'latex_percentage'])

        # Report filtering results
        total_before = len(paper_df_validated)
        total_after = len(paper_df_filtered)
        filtered_out = total_before - total_after

        print(f"LaTeX filtering results:")
        print(f"  Before filtering: {total_before} knowledge statements")
        print(f"  After filtering: {total_after} knowledge statements")
        print(f"  Filtered out: {filtered_out} statements with >{latex_threshold}% LaTeX content")

        ## 2.2 Filter Out Sentences with References Not In Context

        """Some sentences contain references that are defined in a section of the paper that does not jointly appear during training since it's in a different section. Technically it may have appeared together if it's a small subsection since we join subsections that are small together, but they are at the very least far away. We filter out sentences that have such references."""

        def find_undefined_references(sentence: str, subsection_text: str) -> list[str]:
            """
            Finds LaTeX references in a sentence that are not defined in the given subsection text.

            This function identifies all references formatted as \\ref{...} in the input sentence.
            It then checks for corresponding \\label{...} definitions within the subsection_text.
            
            Args:
                sentence (str): The sentence to check for references.
                subsection_text (str): The text of the subsection to check for labels.

            Returns:
                list[str]: A list of reference labels that are used in the sentence but not
                        defined in the subsection text. An empty list indicates all
                        references are defined locally.
            """
            # Find all references in the sentence, e.g., \ref{eq:RL} -> "eq:RL"
            references = re.findall(r'\\ref\{([^}]+)\}', sentence)
            if not references:
                return True

            # Find all defined labels in the subsection text, e.g., \label{eq:main_eq} -> "eq:main_eq"
            defined_labels = set(re.findall(r'\\label\{([^}]+)\}', subsection_text))

            # Identify references that are not defined within the subsection
            undefined_references = [ref for ref in references if ref not in defined_labels]

            if len(undefined_references) > 0:
                return False
            else:
                return True

        keep = paper_df_filtered.apply(
            lambda row: find_undefined_references(row['raw_knowledge_statement'], row['subsection_text']),
            axis=1
        )

        # Separate kept and dropped statements
        dropped_statements = paper_df_filtered[~keep].copy()
        paper_df_filtered_refs = paper_df_filtered[keep].copy()

        save_df_for_debugging(dropped_statements, '05_ref_filtered_out.txt', 'facts', paper_name, ['raw_knowledge_statement'])
        save_df_for_debugging(paper_df_filtered_refs, '05_ref_kept.txt', 'facts', paper_name, ['raw_knowledge_statement'])


        # Report filtering results
        total_before = len(paper_df_filtered)
        total_after = sum(keep)
        filtered_out = total_before - total_after

        print(f"Reference filtering results:")
        print(f"  Before filtering: {total_before} knowledge statements")
        print(f"  After filtering: {total_after} knowledge statements")
        print(f"  Filtered out: {filtered_out} statements with undefined references")

        paper_df_filtered = paper_df_filtered[keep].reset_index(drop=True)

        ## 2.3 Filter Short Facts

        # Filter out knowledge statements that are too short
        min_length = 90
        keep_length = paper_df_filtered['raw_knowledge_statement'].str.len() >= min_length

        # Separate kept and dropped statements
        dropped_statements_len = paper_df_filtered[~keep_length].copy()
        paper_df_filtered_len = paper_df_filtered[keep_length].copy()

        save_df_for_debugging(dropped_statements_len, '06_short_filtered_out.txt', 'facts', paper_name, ['raw_knowledge_statement'])
        save_df_for_debugging(paper_df_filtered_len, '06_short_kept.txt', 'facts', paper_name, ['raw_knowledge_statement'])


        # Report filtering results
        total_before = len(paper_df_filtered)
        total_after = sum(keep_length)
        filtered_out = total_before - total_after

        print(f"Length filtering results:")
        print(f"  Before filtering: {total_before} knowledge statements")
        print(f"  After filtering: {total_after} knowledge statements")
        print(f"  Filtered out: {filtered_out} statements shorter than {min_length} characters")

        paper_df_filtered = paper_df_filtered[keep_length].reset_index(drop=True)

        ## 2.4 Identify Knowledge Statements

        """This step is now the primary filter to identify sentences that contain meaningful knowledge. We use an LLM to determine if a sentence contains a clear, verifiable fact that could be used to form a comprehension question."""

        # Evaluate sentences for knowledge content
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

        # Run evaluation once and store results
        print("Evaluating sentences for knowledge content...")

        is_knowledge_results = [None] * len(paper_df_filtered)

        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = {executor.submit(evaluate_statement, (idx, row)): idx 
                       for idx, row in paper_df_filtered.iterrows()}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating sentences"):
                idx, is_knowledge = future.result()
                original_idx = list(paper_df_filtered.index).index(idx)
                is_knowledge_results[original_idx] = is_knowledge

        # Add knowledge column
        paper_df_filtered['is_knowledge'] = is_knowledge_results

        paper_df_knowledge = paper_df_filtered[paper_df_filtered['is_knowledge']].copy()
        paper_df_non_knowledge = paper_df_filtered[~paper_df_filtered['is_knowledge']].copy()
        save_df_for_debugging(paper_df_knowledge, '07_knowledge_kept.txt', 'facts', paper_name, ['raw_knowledge_statement', 'is_knowledge'])
        save_df_for_debugging(paper_df_non_knowledge, '07_knowledge_filtered_out.txt', 'facts', paper_name, ['raw_knowledge_statement', 'is_knowledge'])

        # Report filtering results
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

    """Given the original, source sentences from the paper, we break each sentence into the parts that presents a new fact."""

    def extract_context_and_sentence(row):
        # Extract context: everything before the sentence + the sentence's paragraph
        # + one additional paragraph after for surrounding context
        parts = row['subsection_text'].strip().split(row['raw_knowledge_statement'].strip())
        context_before = parts[0]

        if len(parts) > 1:
            remaining_text = parts[1]
            # Include the rest of the current paragraph
            paragraph_end = remaining_text.find('\n\n')
            if paragraph_end != -1:
                rest_of_paragraph = remaining_text[:paragraph_end]
                # Also include one more paragraph after for additional context
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

        # Remove title line if present
        if context.startswith('\\title{'):
            lines = context.split('\n')
            title_end = 0
            for i, line in enumerate(lines):
                if '}' in line:
                    title_end = i + 1
                    break
            context = '\n'.join(lines[title_end:]).strip()
        return context
    
    def extract_atomic_facts(first_row):
        """Extract atomic facts from a paragraph using LLM."""
        prompt = {}
        prompt['system'] = r"""You will be given two inputs, a section of an academic paper for context and a single sentence drawn from that section. Papers often interweave various pieces of knowledge together in academic writing. While each sentence is interwoven with others, there is atomic knowledge that can be extracted from a particular sentence. Write questions that tests for this atomic knowledge. Specifically, your task is to extract questions from the provided sentence with clear answers. 
   
Extract 1-2 questions from the sentence. If the sentence truly has lots of facts, extract up to 3 questions.
  
### Detailed Instructions
Consider these instructions as you extract each question:
- The question should be meaningful, focused on a main fact in the sentence, not a minor detail.
- The question should be non-trivial and non-obvious. It should not be plainly obvious from the question for someone with no relevant knowledge.
- The answer to the question MUST be a verbatim word, phrase (2-4 words), or a mathematical expression copied exactly from the sentence. Capitlization can be adjusted appropriately. Otherwise, do not paraphrase, rephrase, or adjust the answer; it must appear as an exact substring of the sentence.
- The question should have a clear, single answer and *NOT* multiple valid answers.
- Prefer shorter answers.
- Each question should be written separately and independently of the other questions; do not reference other questions in the same question.

### Handling Mathematical Sentences
Many sentences in academic papers contain equations, variables, or mathematical notation. These sentences still contain important knowledge, but you must extract it carefully:
- Do NOT extract partial equations or incomplete equation fragments as answers.
- If the answer is a mathematical expression, it MUST be a complete, self-contained expression exactly as written in the source.
- Do not ask simple notation questions like "What is the variable that represents the reward?"
- Prefer natural language answers when they capture the same knowledge as a mathematical answer. For example, prefer "the partition function" over "$Z(x)$" if both are valid.
- Preserve the original LaTeX formatting exactly. Use $...$ delimiters as they appear in the source. Do NOT convert to \(...\) or other formats.

### Demonstration 1: Natural Language Sentence
Context: "\\title{Direct Preference Optimization: Your Language Model is Secretly a Reward Model}\n\\subsection{Can DPO scale to real preference datasets?}\nNext, we evaluate fine-tuning performance of DPO on summarization and single-turn dialogue. For summarization, automatic evaluation metrics such as ROUGE can be poorly correlated with human preferences~\citep{stiennon2022learning}, and prior work has found that fine-tuning LMs using PPO on human preferences to provide more effective summaries. We evaluate different methods by sampling completions on the test split of TL;DR summarization dataset, and computing the average win rate against reference completions in the test set."

Sentence: "We evaluate different methods by sampling completions on the test split of TL;DR summarization dataset, and computing the average win rate against reference completions in the test set."

Questions:
- "The authors evaluate DPO’s fine-tuning performance against other methods on summarization by sampling completions on the test split of what dataset?", Answer: "TL;DR summarization"
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
Sentence: "Following prior works, the optimization is formulated as\n\\begin{equation}\n\\max_{\\pi_{\\theta}}  \\mathbb{E}_{x\\sim \\mathcal{D}, y\\sim \\pi_{\\theta}(y \\mid x)}\\bigl[r_{\\phi}(x, y)\\bigr] - \\beta\\mathbb{D}_{\\textrm{KL}}\\bigl[\\pi_{\\theta}(y\\mid x)\\mid \\mid \\pi_\\text{ref}(y\\mid x)\\bigr],\n\\end{equation}\nwhere $\\beta$ is a parameter controlling the deviation from the base reference policy $\\pi_\\text{ref}$, namely the initial SFT model $\\pi^\\text{SFT}$."

Questions:
- "In the RL fine-tuning objective, what does the parameter $\\beta$ control?", Answer: "deviation from the base reference policy"
"""
        contextualize_prompt = {}
        contextualize_prompt['system'] = r"""You will be given two inputs, a section of an academic paper for context, a single sentence drawn from that section, and a question extracted from the sentence as well as its corresponding answer. Your task is to then turn the question into a self-contained, precise question. Approach this task step-by-step as outlined below.

While you should use your expertise on this domain to handle and understand these texts, all information written into the questions and answers *MUST* originate from the provided context or sentence. Do not add, infer, or correct information using your internal knowledge. Every detail should be traceable back to the source text. As you write and rewrite the questions, also make sure to accurately represent the knowledge in the original sentence without distortion. Strive to use phrasing as close as possible to the original text, but prioritize clarity and self-containment. Lastly, the questions should be written well and clearly so that they are easy to read.
    
### Instructions
The overall goal of this task is to make the questions clear by incorporating the relevant context. This ensures the question is unambiguous and doesn't require looking back to the source material.

For each question:
1.  Rewrite the question so that it starts with one of the following templates. Always wrap the paper title in double quotes — never use single quotes, asterisks, or underscores around the title.
    - "In the paper \"{title}\", ..."
    - "According to the paper \"{title}\", ..."
    - "In the paper \"{title}\", the authors remark that..."
    - "In the paper \"{title}\", the authors state that..."
    - "According to the paper \"{title}\", prior work has..."
    - "In the theoretical analysis of the paper \"{title}\", ..."
    - "In the paper \"{title}\", the results suggest that..."
    This is a non-exhaustive list of templates, and you should use your own judgement to choose the most appropriate template or modify the template to fit the sentence.
2.  Add sufficient context. Specifically, use the *provided context* to supply whatever information is needed to make the question self-contained and unambiguous. For instance, "Do humans and GPT4 agree often with each other?" should be clarified into "In the paper '...', did humans and GPT4 often agree or disagree with each other during the evaluation of DPO?" if this notion was in the context of evaluating DPO in an academic paper. The goal is to ensure someone reading just the question would understand exactly what is being asked without needing additional context.
3.  Clarify pronouns and referential terms. Check the sentence for pronouns (it, this, that, these, those) or demonstrative phrases (this equation, that method, these results) that refer to entities not explicitly defined within the sentence itself. Search the surrounding context to identify what these terms reference, then incorporate that clarifying information into the question to make it self-contained.
4.  Clarify Context-Dependent Terms. Named entities (e.g., theorems, equations, proper nouns) do not need clarification. However, if there are unnamed or context-specific terms (e.g., $f$, "the model", "the loss"), clarify their full context. For instance, "the gradient" might refer to the general concept of a gradient or to the gradient of a specific function mentioned earlier in the context.
5.  Disambiguate experiments. There are often numerous experiments in a paper, and so supply enough experimental context so that the question is about which experiment the question is asking about. 
6.  Handle acronyms. If the answer is an acronym and the acronym appears frequently in the context, feel free to leave it as an acronym without defining it.
7.  Do not leak the answer. Please make sure that *the answer is not revealed* in the question. The answer should never appear in the question.
8.  Maintain the essence of the original question during all of this.
9.  Do not change the answer. Minor grammatical adjustments to the answer are allowed only if necessary to fit the restructured question (e.g., adjusting verb tense, determiners like "the").
10. Avoid quoting the source sentence directly in the question.
11. Refine Question. The rewritten question can be broken up into multiple sentences if the question becomes verbose. Make sure the question is written clearly and grammatically correct. Do not put any of the context in parenthesis or followed after an "i.e.".
12. Preserve LaTeX formatting. When the question or answer contains mathematical notation, preserve the exact LaTeX syntax from the source, including delimiters. Use $...$ as they appear in the original — do NOT convert to \(...\) or other formats.

Think carefully and critically through this task, following the step-by-step instructions outlined above. Then, provide the final output, listing each question and its corresponding answer."""
        cloze_prompt = {}
        cloze_prompt['system'] = FACT_PROBE_CLOZE_PROMPT_SYSTEM
        json_parse_prompt = {
            'system': """You will be given a string containing questions and answers. Convert it into a JSON object with a single key "list_of_questions", which contains a list of objects. Each object in the list should have a "question" and "answer" key. The format should be: {"list_of_questions": [{"question": "...", "answer": "..."}, ...]}. Copy the question and answer content exactly as it appears. Do not modify the text.""",
            'user': ""
        }
        
        context = extract_context_and_sentence(first_row)
        prompt['user'] = f"""### Title\n{first_row['title']}\n### Context\n{context}\n\n### Sentence\n{first_row['raw_knowledge_statement'].strip()}"""
        output1 = utils.query_llm(prompt, model='gpt-5.4', reasoning_effort='medium')   

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

        for pair in qa_pairs:
            question = pair.get('question')
            answer = pair.get('answer')
            if not question or not answer:
                continue

            single_qa_string = f'### Question\n"{question}\n\n###Answer\n"{answer}"'

            contextualize_prompt['user'] = f"""### Title\n{first_row['title']}\n### Context\n{context}\n\n### Sentence\n{first_row['raw_knowledge_statement'].strip()}\n\n{single_qa_string}"""
            output2_individual = utils.query_llm(contextualize_prompt, model='gpt-5.4-mini', reasoning_effort='medium')
            all_contextualized.append(output2_individual)

            cloze_prompt['user'] = f"""### Question and Answer\n{output2_individual}"""
            output3_individual = utils.query_llm(cloze_prompt, system_prompt_included=True, model='gpt-5.4', reasoning_effort='medium', return_json=True)
            try:
                cloze_pair = json.loads(output3_individual)
                if isinstance(cloze_pair, dict) and 'answer' in cloze_pair and 'statement' in cloze_pair:
                    all_clozes.append(cloze_pair)
            except (json.JSONDecodeError, TypeError):
                print(f"Failed to parse cloze pair from: {output3_individual}")

        return output1, all_contextualized, all_clozes

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        raw_extracted_facts = list(tqdm(executor.map(extract_atomic_facts, [paper_df_knowledge.iloc[i] for i in range(len(paper_df_knowledge))]), total=len(paper_df_knowledge)))

    # Separate the three outputs into different columns
    questions = [result[0] if result is not None else None for result in raw_extracted_facts]
    contextualized_questions = [result[1] if result is not None else None for result in raw_extracted_facts]
    cloze_pairs = [result[2] if result is not None else [] for result in raw_extracted_facts]

    # Store in dataframe
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
        # Fallback to original if parsing fails or response is malformed
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

    # Since refinement step does not drop pairs, we just create a copy for the next step.
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


    # Fix facts that start with '(' by extracting the rightmost part after target + ','
    facts_starting_with_paren = paper_df_with_probes[paper_df_with_probes['fact'].str.startswith('(')]
    print(f"Found {len(facts_starting_with_paren)} facts starting with '(' - fixing these...")

    for idx, row in facts_starting_with_paren.iterrows():
        target_with_comma = row['target'] + ','
        if target_with_comma in row['fact']:
            rightmost_part = row['fact'].split(target_with_comma)[-1].strip().strip('()')
            # Update the fact in the dataframe
            paper_df_with_probes.loc[idx, 'fact'] = rightmost_part
            # print(f"Fixed fact at index {idx}: '{row['fact']}' -> '{rightmost_part}'")

    print(f"Fixed {len(facts_starting_with_paren)} facts that started with '('")
    save_df_for_debugging(paper_df_with_probes, '09_fixed_paren_facts.txt', 'facts', paper_name, ['raw_knowledge_statement', 'fact'])

    # Check that facts end with targets (after stripping whitespace and punctuation)
    valid_facts = []
    filtered_count = 0
    dropped_examples = []

    for idx, row in paper_df_with_probes.iterrows():
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

    if not paper_df_with_probes.empty:
        paper_df_with_probes['valid_fact'] = valid_facts
    else:
        print("No probes to validate.")
        return

    print(f"Filtered out {filtered_count} facts that don't end with their target")
    print(f"Remaining valid facts: {len(paper_df_with_probes) - filtered_count}")

    # Filter to keep only valid facts and save both kept and dropped
    paper_df_probes_valid = paper_df_with_probes[paper_df_with_probes['valid_fact']].copy()
    paper_df_probes_invalid = paper_df_with_probes[~paper_df_with_probes['valid_fact']].copy()

    save_df_for_debugging(paper_df_probes_valid, '10_target_end_kept.txt', 'facts', paper_name, ['raw_knowledge_statement', 'fact', 'target'])
    if not paper_df_probes_invalid.empty:
        save_df_for_debugging(paper_df_probes_invalid, '10_target_end_filtered_out.txt', 'facts', paper_name, ['raw_knowledge_statement', 'fact', 'target'])


    # Create probe column (fact minus target) using stripped, no punctuation versions
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
            target_cleaned = fact_cleaned[last_index:]  # include any trailing punctuation
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
    #paper_df_with_probes.drop(columns=['valid_fact'], inplace=True)
    save_df_for_debugging(paper_df_probes_valid, '11_final_probes.txt', 'facts', paper_name, ['raw_knowledge_statement', 'fact', 'target', 'probe'])

    # 6. ensure tokenizing the target separately from the probe is fine

    # Load tokenizer
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
                # Verify the recovered answer is actually verbatim in the paper
                if new_answer.strip() in paper:
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
    metrics_path = os.path.join(output_dir, 'filtering_metrics_v10.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"Fact Probe Pipeline v10 - Filtering Metrics for {paper_name}\n")
        f.write(f"{'='*60}\n")
        f.write(f"Total probes before verbatim filter: {total_before_verbatim}\n")
        f.write(f"Probes filtered (target not verbatim in paper): {filtered_verbatim}\n")
        f.write(f"Total probes after verbatim filter: {total_after_verbatim}\n")
    print(f"Saved filtering metrics to {metrics_path}")

    # 8. save probes

    paper_df_probes_valid.reset_index(drop=True, inplace=True)
    paper_df_probes_valid.to_csv(os.path.join(output_dir, 'probes_v10.csv'), index=False)

    # Save readable version
    readable_path = os.path.join(output_dir, 'probes_v10_readable.txt')
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
