import re 
import os

def _rejoin_alphanumeric_words(text: str) -> str:
    """
    Fixes misplaced newlines with a strict set of rules:
    - Finds any two non-whitespace chunks separated by clean whitespace.
    - A fix is only applied IF AND ONLY IF both chunks are purely alphabetic.
    """
    # A general pattern to find candidates: two non-whitespace chunks
    # separated by whitespace that does NOT contain a backslash.
    # [^\\s] is a negated set: any character that is NOT a backslash or whitespace.
    # We use this to avoid matching the '\\' from a LaTeX line break.
    pattern = r'(\S+)([ \t]*\n[ \t\n]*)(\S+)'
    def conditional_replacer(match: re.Match) -> str:
        """
        The core logic: only act on purely alphabetic words.
        """
        word1 = match.group(1)
        word2 = match.group(3)
        word1_raw = word1.replace('\\', '').replace('\t', '').replace('(', '').replace(')', '').replace(',','') # We can't allow all subsection headers to collapse
        word2_raw = word2.replace('\\', '').replace('\t', '').replace('(', '').replace(')', '').replace(',','') # .replace('{', '').replace('}', '')
        # print(word1, word2)
        # THE CRITICAL CHECK: Are both words composed *only* of letters?
        if word1_raw.isalnum() and word2_raw.isalnum():
            # Yes. Perform the replacement.
            return f'{word1} {word2}'
        else:
            # No. One or both words contain non-alphabetic characters (\, &, . etc).
            # Return the original matched text to skip the replacement.
            return match.group(0)

    # Loop until the text stabilizes
    previous_text = ""
    while text != previous_text:
        previous_text = text
        text = re.sub(pattern, conditional_replacer, text)
    return text

def _rejoin_words_single_newline(text: str) -> str:
    """
    Fixes single misplaced newlines between words if at least one of the words is alphanumeric.
    """
    # This pattern is stricter: it only matches a single newline between words.
    pattern = r'(\S+)([ \t]*\n[ \t]*)(\S+)'
    def conditional_replacer(match: re.Match) -> str:
        """
        The core logic: act if  neither contains backslashes.
        """
        word1 = match.group(1)
        word2 = match.group(3)
        
        # Check if either word contains backslashes - if so, don't rejoin
        # Remove \cite and \citep commands first, then check for remaining backslashes
        word1_no_cite = re.sub(r'\\cite[p]?\{[^}]*\}', '', word1)
        word2_no_cite = re.sub(r'\\cite[p]?\{[^}]*\}', '', word2)
        
        if '\\' in word1_no_cite or '\\' in word2_no_cite:
            return match.group(0)
        
        return f'{word1} {word2}'

    # Loop until the text stabilizes
    previous_text = ""
    while text != previous_text:
        previous_text = text
        text = re.sub(pattern, conditional_replacer, text)
    return text

# The new "manager" function that understands LaTeX environments.
def rejoin_words_in_unprotected_environments(text: str, single_newline: bool = False) -> str:
    """
    Selectively applies newline fixing, skipping protected LaTeX environments.
    `single_newline` parameter controls which fixing strategy to use.
    """
    # Define the environments to protect. Add any others you need here.
    protected_envs = ['figure', 'table', 'tabular', 'wraptable','equation']
    
    # Create a regex to find the start/end tags of these environments.
    pattern = re.compile(r'\\(begin|end)\{(' + '|'.join(protected_envs) + r')\}')

    result_parts = []
    last_index = 0
    nesting_level = 0
    
    # Choose which rejoining function to use based on the parameter.
    rejoin_function = _rejoin_words_single_newline if single_newline else _rejoin_alphanumeric_words

    for match in pattern.finditer(text):
        # The chunk of text *before* the current tag.
        chunk_before_tag = text[last_index:match.start()]

        if nesting_level == 0:
            # We are in unprotected text. Process it.
            processed_chunk = rejoin_function(chunk_before_tag)
            result_parts.append(processed_chunk)
        else:
            # We are inside a protected environment. Leave the chunk untouched.
            result_parts.append(chunk_before_tag)

        # Add the tag itself, always untouched.
        tag = match.group(0)
        result_parts.append(tag)

        # Update our state (the nesting level).
        command = match.group(1)
        if command == 'begin':
            nesting_level += 1
        elif command == 'end':
            # Clamp at 0 to be safe against malformed \end tags.
            nesting_level = max(0, nesting_level - 1)
        
        # Move our cursor past the tag we just processed.
        last_index = match.end()

    # Process the final chunk of the document after the last tag.
    final_chunk = text[last_index:]
    if nesting_level == 0:
        result_parts.append(rejoin_function(final_chunk))
    else:
        # This would mean the document ends with an unclosed protected env.
        # We'll respect that and not process the final part.
        result_parts.append(final_chunk)

    return "".join(result_parts)

def replace_rev_logic(match: re.Match) -> str:
    """
    This function is called for each match of the regex.
    It programmatically finds the correct second argument, handling nested braces.
    """
    # The regex will only match the start of the command: '\rev{'
    # We start parsing from the character right after this match.
    text = match.string
    start_pos = match.end()

    # --- Find the end of the first argument ---
    # We do this by counting braces to find the matching '}'
    open_braces = 1
    cursor = start_pos
    while open_braces > 0 and cursor < len(text):
        if text[cursor] == '{':
            open_braces += 1
        elif text[cursor] == '}':
            open_braces -= 1
        cursor += 1
    
    # After the loop, `cursor` is one char past the closing brace of the first arg.
    # The next character must be the opening brace of the second argument.
    if cursor >= len(text) or text[cursor] != '{':
        # This is a malformed \rev command, return it as is to be safe
        return match.group(0)

    # --- Extract the second argument ---
    arg2_start = cursor + 1
    open_braces = 1
    cursor = arg2_start
    while open_braces > 0 and cursor < len(text):
        if text[cursor] == '{':
            open_braces += 1
        elif text[cursor] == '}':
            open_braces -= 1
        cursor += 1
    
    arg2_end = cursor - 1
    
    # We found the content of the second argument. This is our replacement text.
    return text[arg2_start:arg2_end]

def process_all_rev_commands(text: str) -> str:
    # This pattern finds the start of a rev command and captures its full body
    # It assumes rev commands don't nest inside each other.
    # The pattern matches \rev{...}{...} by finding the start and letting the 
    # Python code do the brace matching.
    
    # A simplified pattern that just finds the start
    pattern = re.compile(r'\\rev\{')
    
    last_end = 0
    result_parts = []
    
    for match in pattern.finditer(text):
        # Add the text before this match
        result_parts.append(text[last_end:match.start()])
        
        # --- Find Argument 1 ---
        brace_level = 1
        cursor = match.end()
        while brace_level > 0:
            if text[cursor] == '{': brace_level += 1
            elif text[cursor] == '}': brace_level -= 1
            cursor += 1
        
        # cursor is now at the '{' of the second argument
        
        # --- Find Argument 2 and add it to results ---
        brace_level = 1
        arg2_start = cursor + 1
        cursor += 1
        while brace_level > 0:
            if text[cursor] == '{': brace_level += 1
            elif text[cursor] == '}': brace_level -= 1
            cursor += 1

        result_parts.append(text[arg2_start : cursor-1])
        last_end = cursor

    # Add any remaining text after the last match
    result_parts.append(text[last_end:])
    
    return "".join(result_parts)

def remove_figures_and_tables(text: str) -> str:
    """
    Removes figure, table, and related environments from LaTeX text.
    Based on inspection of manifest files, these environments can be safely removed.
    """
    # A list of environments to be removed.
    environments = ['figure', 'table', 'tabular', 'wraptable', 'wrapfigure', 'longtable']
    cleaned_text = text
    for env in environments:
        # This regex is designed to be robust.
        # 1. \\begin\{' + env + '... : Matches the start of the environment.
        # 2. (?![a-zA-Z]): This is a negative lookahead. It ensures that the environment name is not a prefix of a longer name.
        #    For example, it will match \begin{figure} but not \begin{figureSeries}.
        # 3. .*?: This non-greedily matches all characters (including newlines due to re.DOTALL) until the end tag.
        # 4. \\end\{' + env + '\\*?\}: Matches the end tag, allowing for a '*' (e.g., \end{figure*}).
        pattern = re.compile(r'\\begin\{' + env + r'(?![a-zA-Z]).*?\\end\{' + env + r'\*?\}', re.DOTALL)
        cleaned_text = pattern.sub('', cleaned_text)
    return cleaned_text

def remove_latex_comments(text: str) -> str:
    """
    Removes LaTeX comments from the text. A comment is anything after a '%'
    that is not preceded by a backslash, until the end of the line.
    """
    # This regex uses a negative lookbehind (?<!\\) to ensure we don't match escaped percent signs (\%).
    # It matches a '%' followed by any characters until the end of the line and removes it.
    pattern = re.compile(r'(?<!\\)%.*')
    return pattern.sub('', text)

def delete_begin_group_special_cases(text: str) -> str:
    """
    Deletes \\begingroup...\\endgroup blocks only if their content consists
    exclusively of LaTeX commands (words starting with '\\').
    """
    pattern = re.compile(r'\\begingroup(.*?)\\endgroup', re.DOTALL)

    def replacer(match):
        content = match.group(1)
        # Split content into words and filter out any empty strings from multiple spaces.
        words = content.split()
        #print(words)
        # Check if all non-empty parts of the content are commands.
        is_all_commands = all(word.startswith('\\') or word.startswith('{') for word in words if word)

        if is_all_commands:
            # If everything inside is a command, remove the whole block.
            return ''
        else:
            # Otherwise, keep it as is.
            return match.group(0)

    return pattern.sub(replacer, text)

def extract_newcommands(text: str) -> tuple[dict, list]:
    """
    Finds all \\newcommand and \\renewcommand definitions, separating them into
    simple replacements and complex commands to be skipped.
    """
    pattern = re.compile(r'\\(?:newcommand|renewcommand)\s*')
    definitions = {}
    skipped_commands = []

    for match in pattern.finditer(text):
        try:
            # --- Find Command Name ---
            cursor = match.end()
            while text[cursor].isspace(): cursor += 1
            if text[cursor] != '{': continue
            
            brace_level, cmd_start = 1, cursor + 1
            cursor += 1
            while brace_level > 0 and cursor < len(text):
                if text[cursor] == '{': brace_level += 1
                elif text[cursor] == '}': brace_level -= 1
                cursor += 1
            command = text[cmd_start : cursor - 1]

            # --- Check for optional arguments, which indicate a complex command ---
            next_char_cursor = cursor
            while text[next_char_cursor].isspace():
                next_char_cursor += 1
            if text[next_char_cursor] == '[':
                skipped_commands.append(command)
                continue

            # --- Find Definition ---
            while text[cursor].isspace(): cursor += 1
            if text[cursor] != '{': continue

            brace_level, def_start = 1, cursor + 1
            cursor += 1
            while brace_level > 0 and cursor < len(text):
                if text[cursor] == '{': brace_level += 1
                elif text[cursor] == '}': brace_level -= 1
                cursor += 1
            definition = text[def_start : cursor - 1]
            definitions[command] = definition
        except IndexError:
            continue
    return definitions, skipped_commands

def apply_newcommands_from_dict(text: str, definitions: dict, skipped: list) -> str:
    """
    Applies a dictionary of command definitions to a string and reports skipped commands.
    """
    if not definitions and not skipped:
        return text

    print("--- Applying NewCommands ---")
    for command, replacement in definitions.items():
        # Use a regex with a negative lookahead to ensure we're not replacing a prefix of a longer command.
        # e.g., don't replace \new in \newpage
        pattern = re.escape(command) + r'(?![a-zA-Z])'
        
        # Only print if the command is actually found and replaced.
        if re.search(pattern, text):
            print(f"Replacing '{command}' with '{replacement}'")
            # We use a lambda function for the replacement to ensure that the
            # replacement string is treated as a literal string and backslashes
            # within it (like in \emph) are not interpreted as escape sequences by re.sub.
            text = re.sub(pattern, lambda match: replacement, text)
    
    # Filter the skipped list to only include commands that were actually present.
    relevant_skipped = []
    for cmd in skipped:
        pattern = re.escape(cmd) + r'(?![a-zA-Z])'
        if re.search(pattern, text):
            relevant_skipped.append(cmd)
            
    if relevant_skipped:
        print("Skipped the following complex commands (found in text but not replaced):")
        for cmd in relevant_skipped:
            print(f"- {cmd}")
            
    print("--------------------------")
    return text

def delete_miscellaneous_latex_syntax(text: str) -> str:
    """
    Deletes miscellaneous LaTeX syntax blocks, such as scope-limiting blocks for formatting.
    This rule handles blocks in curly brackets that are not part of a command,
    based on the heuristic that they often start on a new line.
    """
    new_text_parts = []
    last_index = 0

    # This pattern finds '{' at the beginning of a line (preceded by optional whitespace).
    pattern = re.compile(r'^\s*\{', re.MULTILINE)

    for match in pattern.finditer(text):
        start_brace_index = match.end() - 1

        # We found a potential block. Add the text before this match.
        new_text_parts.append(text[last_index:match.start()])

        # Find the matching closing brace using brace counting.
        brace_level = 1
        cursor = start_brace_index + 1
        while cursor < len(text) and brace_level > 0:
            if text[cursor] == '{':
                brace_level += 1
            elif text[cursor] == '}':
                brace_level -= 1
            cursor += 1

        if brace_level == 0:
            # Matching brace found. The entire block is skipped by updating last_index.
            last_index = cursor
        else:
            # No matching brace found (e.g., end of file). Treat as not a removable block.
            # Append the matched part and continue processing from there.
            new_text_parts.append(text[match.start():start_brace_index + 1])
            last_index = start_brace_index + 1
    
    # Append any remaining text after the last processed block.
    new_text_parts.append(text[last_index:])

    cleaned_text = "".join(new_text_parts)

    # Remove specific formatting commands like \setlength and \itemsep.
    # Rule for \setlength: remove the whole line it appears on for simplicity.
    cleaned_text = re.sub(r'^\s*\\setlength.*$', '', cleaned_text, flags=re.MULTILINE)
    # Rule for \itemsep: remove the command and its argument in braces.
    cleaned_text = re.sub(r'\\itemsep\{.*?\}', '', cleaned_text)

    # Remove \noindent and \newline commands.
    cleaned_text = re.sub(r'\\noindent', '', cleaned_text)
    cleaned_text = re.sub(r'\\newline', '', cleaned_text)

    # Remove optional arguments from itemize environments.
    # cleaned_text = re.sub(r'(\\begin\{itemize\})\[.*?\]', r'\1', cleaned_text)

    return cleaned_text


def extract_and_clean_title(latex_content: str) -> str:
    """
    Extracts and cleans the title from LaTeX content using robust brace counting.
    """
    title_start_tag = r'\title{'
    title_start_index = latex_content.find(title_start_tag)
    if title_start_index == -1:
        return ""

    brace_level = 1
    start_pos = title_start_index + len(title_start_tag)
    cursor = start_pos
    while brace_level > 0 and cursor < len(latex_content):
        if latex_content[cursor] == '{':
            brace_level += 1
        elif latex_content[cursor] == '}':
            brace_level -= 1
        cursor += 1
    
    if brace_level == 0:
        title_content = latex_content[start_pos : cursor - 1]
        
        # 1. Remove commands with arguments
        cleaned_title = re.sub(r'\\[a-zA-Z]+\s*\{.*?\}', '', title_content)
        # 2. Remove simple command words
        cleaned_title = re.sub(r'\\[a-zA-Z]+\*?', '', cleaned_title)
        # 3. Remove empty braces
        cleaned_title = re.sub(r'\{\s*\}', '', cleaned_title)
        # 4. Normalize spacing commands
        cleaned_title = re.sub(r'~', ' ', cleaned_title)
        cleaned_title = re.sub(r'\\\\\s*', ' ', cleaned_title)
        # 5. Remove leftover backslashes
        cleaned_title = re.sub(r'\\', '', cleaned_title)
        # 6. Consolidate whitespace
        cleaned_title = ' '.join(cleaned_title.split())
        return cleaned_title.strip()
    
    return ""

def extract_abstract(latex_content: str) -> str:
    """
    Extracts the abstract from LaTeX content.
    """
    abstract_match = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', latex_content, re.DOTALL)
    return abstract_match.group(1).strip() if abstract_match else ""

def extract_main_body(latex_content: str) -> str:
    """
    Extracts the main body of a LaTeX document, stopping before references/appendix.
    """
    body_start_index = -1
    doc_start_match = re.search(r'\\begin\{document\}', latex_content)
    if doc_start_match:
        section_match = re.search(r'\\section', latex_content[doc_start_match.end():])
        if section_match:
            body_start_index = doc_start_match.end() + section_match.start()

    if body_start_index == -1:
        return ""
    
    body = latex_content[body_start_index:]
    
    end_markers = [
        r'\\begin\{thebibliography\}', r'\\bibliography', r'\\appendix',
        # Use a more robust, case-insensitive pattern for acknowledgment-like sections
        # The 's?' makes the 's' optional to catch both "acknowledgment" and "acknowledgements"
        r'\\section\*?\{.*?(?:acknowledgements?|acknowledgments?).*?\}'
    ]
    
    # Find the earliest occurrence of any end marker.
    end_index = len(body)
    for marker_regex in end_markers:
        # Add re.IGNORECASE to make the match case-insensitive
        end_match = re.search(marker_regex, body, re.IGNORECASE)
        if end_match:
            # The end_index should be the start of the match, not after it.
            end_index = min(end_index, end_match.start())
    
    # Return the body up to the calculated end index.
    return body[:end_index]

def normalize_inter_word_whitespace(text: str) -> str:
    """
    Collapses erroneous line breaks and repeated whitespace between alphabetic words.
    This is a general cleanup step.
    """
    # This pattern finds two alphabetic words separated by any whitespace (including newlines)
    # and replaces it with a single space. The loop handles chains of such words.
    pattern = r'\b([a-zA-Z]+)\b\s+\b([a-zA-Z]+)\b'
    previous_text = ""
    while text != previous_text:
        previous_text = text
        text = re.sub(pattern, r'\1 \2', text)
    return text

def ensure_space_around_environments(text: str) -> str:
    """
    Ensures that top-level LaTeX environments are surrounded by two newlines.
    This function correctly handles nested environments by tracking the nesting level.
    """
    pattern = re.compile(r'\\(begin|end)\{([a-zA-Z0-9\*]+)\}')
    
    result_parts = []
    last_end = 0
    nesting_level = 0
    top_level_start = -1

    for match in pattern.finditer(text):
        command = match.group(1)

        if command == 'begin':
            if nesting_level == 0:
                top_level_start = match.start()
                # Process the chunk before this environment starts
                chunk_before = text[last_end:top_level_start]
                result_parts.append(chunk_before.rstrip())
            nesting_level += 1
        elif command == 'end':
            if nesting_level > 0:
                nesting_level -= 1
                if nesting_level == 0 and top_level_start != -1:
                    # We've found the end of a top-level environment
                    # Add a leading newline barrier if not already there
                    if result_parts and result_parts[-1] and not result_parts[-1].endswith('\n\n'):
                         result_parts.append('\n\n')

                    env_block = text[top_level_start:match.end()]
                    result_parts.append(env_block)
                    
                    # Add a trailing newline barrier
                    result_parts.append('\n\n')

                    last_end = match.end()
                    top_level_start = -1
    
    # Append any remaining text after the last environment
    result_parts.append(text[last_end:].lstrip())
    
    # Join and clean up any excessive newlines that might have been created
    final_text = "".join(result_parts)
    final_text = re.sub(r'\n{3,}', '\n\n', final_text)
    
    return final_text.strip()

def clean_latex(latex_content: str, debug_dir: str = None) -> str:
    """
    Performs the full, refined cleaning process on a LaTeX string.
    If a `debug_dir` is provided, it saves the state of the text after each step.
    """
    def save_debug_file(step_num, step_name, content):
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            file_path = os.path.join(debug_dir, f"{step_num}_{step_name}.txt")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

    step = 1
    
    # --- Step 1: Extract definitions from the original text before any cleaning ---
    definitions, skipped_commands = extract_newcommands(latex_content)
    
    # --- Pre-processing ---
    cleaned_content = remove_latex_comments(latex_content)
    save_debug_file(step, "1_remove_comments", cleaned_content); step += 1

    cleaned_content = remove_figures_and_tables(cleaned_content)
    save_debug_file(step, "2_remove_figures_tables", cleaned_content); step += 1

    cleaned_content = re.sub(r'\\vspace\*?\{.*?\}', '', cleaned_content)
    save_debug_file(step, "3_remove_vspace", cleaned_content); step += 1

    cleaned_content = delete_begin_group_special_cases(cleaned_content)
    save_debug_file(step, "4_delete_begingroup", cleaned_content); step += 1

    # --- Content Extraction ---
    title = extract_and_clean_title(cleaned_content)
    abstract = extract_abstract(cleaned_content)
    body = extract_main_body(cleaned_content)
    
    # --- Reconstruction & Command Application ---
    full_text = f'\\title{{{title}}}\n\n\\begin{{abstract}}\n{abstract}\n\\end{{abstract}}\n\n{body}\n\\end{{document}}'
    save_debug_file(step, "5_reconstruct_text", full_text); step += 1

    # --- Post-processing on Reconstructed Text ---
    cleaned_text = process_all_rev_commands(full_text)
    save_debug_file(step, "6_process_rev_commands", cleaned_text); step += 1

    cleaned_text = apply_newcommands_from_dict(cleaned_text, definitions, skipped_commands)
    save_debug_file(step, "7_apply_newcommands", cleaned_text); step += 1

    cleaned_text = delete_miscellaneous_latex_syntax(cleaned_text)
    save_debug_file(step, "8_delete_misc_syntax_post_commands", cleaned_text); step += 1
    
    cleaned_text = re.sub(r'\\clearpage|\\newpage', '', cleaned_text)
    # Remove lines that contain only spaces or tabs, turning them into empty lines.
    cleaned_text = re.sub(r'^[ \t]+$', '', cleaned_text, flags=re.MULTILINE)
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
    save_debug_file(step, "9_remove_pagebreaks_newlines", cleaned_text); step += 1

    # --- Final text conditioning ---
    final_cleaned_text = rejoin_words_in_unprotected_environments(cleaned_text)
    save_debug_file(step, "10_rejoin_words_multi_newline", final_cleaned_text); step += 1

    final_cleaned_text = rejoin_words_in_unprotected_environments(final_cleaned_text, single_newline=True)
    save_debug_file(step, "11_rejoin_words_single_newline", final_cleaned_text); step += 1

    final_cleaned_text = ensure_space_around_environments(final_cleaned_text)
    save_debug_file(step, "12_ensure_space_around_envs", final_cleaned_text); step += 1

    return final_cleaned_text.strip()

def clean_latex_semicleaned_v1(latex_content: str) -> str:
    """
    Performs basic cleaning: removes comments and extracts document structure.
    """
    cleaned_content = remove_latex_comments(latex_content)
    title = extract_and_clean_title(cleaned_content)
    abstract = extract_abstract(cleaned_content)
    body = extract_main_body(cleaned_content)
    full_text = f'\\title{{{title}}}\n\n\\begin{{abstract}}\n{abstract}\n\\end{{abstract}}\n\n{body}\n\\end{{document}}'
    return full_text.strip()

def clean_latex_semicleaned_v2(latex_content: str) -> str:
    """
    Performs level 2 cleaning: v1 + removes figures and tables.
    """
    cleaned_content = remove_latex_comments(latex_content)
    cleaned_content = remove_figures_and_tables(cleaned_content)
    title = extract_and_clean_title(cleaned_content)
    abstract = extract_abstract(cleaned_content)
    body = extract_main_body(cleaned_content)
    full_text = f'\\title{{{title}}}\n\n\\begin{{abstract}}\n{abstract}\n\\end{{abstract}}\n\n{body}\n\\end{{document}}'
    return full_text.strip()

# --- Main Execution ---

# Get list of files in the raw arxiv directory
raw_arxiv_dir = '../../data/arxiv/raw'
all_files = [f for f in os.listdir(raw_arxiv_dir) if os.path.isfile(os.path.join(raw_arxiv_dir, f))]

# Manual list to exclude certain files we don't want to process
exclude_files = []  # Add files to exclude here

# Filter out excluded files
files_to_process = [f for f in all_files if f not in exclude_files]

print(f"Found {len(files_to_process)} files to process: {files_to_process}")

# Read the texts and store them in a dictionary
raw_texts = {}
for file_name in files_to_process:
    file_path = os.path.join(raw_arxiv_dir, file_name)
    with open(file_path, 'r', encoding='utf-8') as f:
        text_content = f.read()
        raw_texts[file_name] = text_content

# Define the different cleaning levels and their corresponding functions
cleaning_configs = {
    'semicleaned_v1': clean_latex_semicleaned_v1,
    'semicleaned_v2': clean_latex_semicleaned_v2,
    'cleaned': clean_latex
}

for dir_name, clean_function in cleaning_configs.items():
    print(f"\n--- Processing for level: {dir_name} ---")
    
    # Define the output directory
    output_dir = f'../../data/arxiv/{dir_name}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Clean each of the raw texts
    cleaned_texts = {}
    for file_name in files_to_process:
        print(f"Cleaning {file_name}...")
        content = raw_texts[file_name]
        
        # For the main 'cleaned' function, provide a debug directory
        if clean_function == clean_latex:
            base_name = os.path.splitext(file_name)[0]
            debug_dir = f'../../data/arxiv/debug/{base_name}'
            # Clean out the old debug directory if it exists
            if os.path.exists(debug_dir):
                import shutil
                shutil.rmtree(debug_dir)
            cleaned_texts[file_name] = clean_function(content, debug_dir=debug_dir)
        else:
            cleaned_texts[file_name] = clean_function(content)

        print(f"Finished cleaning {file_name}.")

    # Save the cleaned texts to the output directory
    for file_name, content in cleaned_texts.items():
        # Create a new filename with a .tex extension
        base_name = os.path.splitext(file_name)[0]
        output_file_name = f"{base_name}.tex"
        output_file_path = os.path.join(output_dir, output_file_name)
        
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"Saved cleaned text to {output_file_path}")

print("\nAll files have been processed for all cleaning levels and saved.")