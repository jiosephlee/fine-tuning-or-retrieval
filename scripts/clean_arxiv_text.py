import re
import os

def clean_latex(latex_content: str) -> str:
    """
    Cleans a LaTeX string by extracting title, abstract, and main body,
    and removing commands and environments that don't contribute to the main text.
    """
    # 1. Extract title
    title_match = re.search(r'\\title\{(.*?)\}', latex_content, re.DOTALL)
    title = ""
    if title_match:
        title = title_match.group(1)
        title = re.sub(r'\\\\\s*', ' ', title)
        title = title.strip()

    # 2. Extract abstract
    abstract_match = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', latex_content, re.DOTALL)
    abstract = abstract_match.group(1).strip() if abstract_match else ""

    # 3. Find start of main body (first section after document environment)
    body_start_index = -1
    doc_start_match = re.search(r'\\begin\{document\}', latex_content)
    if doc_start_match:
        # Find the first section after \begin{document}
        section_match = re.search(r'\\section', latex_content[doc_start_match.end():])
        if section_match:
            body_start_index = doc_start_match.end() + section_match.start()

    if body_start_index == -1:
        # Fallback if structure is unexpected, return what we have so far
        body = ""
    else:
        body = latex_content[body_start_index:]
    
    # 4. Find end of main body (before references, appendix, etc.)
    end_markers = [
        r'\\begin\{thebibliography\}', r'\\bibliography', r'\\appendix',
        r'\\section\*?\{Acknowledgements\}', r'\\section\*?\{Author Contributions\}'
    ]
    end_index = len(body)
    for marker_regex in end_markers:
        end_match = re.search(marker_regex, body)
        if end_match:
            end_index = min(end_index, end_match.start())
    
    body = body[:end_index]

    # Combine the parts we want to keep
    full_text = f'{title}\n\n{abstract}\n\n{body}\n\n\\end{{document}}'

    # Now apply cleaning operations from the original function
    cleaned_text = full_text
    
    # # Environments to remove completely with their content
    # envs_to_remove = [
    #     'figure', 'figure\*', 'table', 'table\*', 'tabular', 'tabular\*', 'algorithm2e',
    #     'equation', 'equation\*', 'align', 'align\*', 'multline', 'multline\*',
    #     'wrapfigure', 'wraptable'
    # ]
    # for env in envs_to_remove:
    #     cleaned_text = re.sub(r'\\begin{' + env + r'}.*?\\end{' + env + r'}', '', cleaned_text, flags=re.DOTALL)

    # # Replace commands that have content we want to keep
    # cmds_keep_content = [
    #     'section', 'subsection', 'subsubsection', 'paragraph', 'subparagraph',
    #     'textbf', 'textit', 'emph', 'texttt', 'caption'
    # ]
    # for cmd in cmds_keep_content:
    #     cleaned_text = re.sub(r'\\' + cmd + r'\{([^}]+)\}', r'\1', cleaned_text)

    # # Handle \rev{old text}{new text} -> new text
    # cleaned_text = re.sub(r'\\rev\{[^}]*\}\{([^}]+)\}', r'\1', cleaned_text)

    # # Remove commands with arguments that we want to discard
    # cmds_remove_arg = ['label', 'ref', 'cite', 'citep', 'input', 'url']
    # for cmd in cmds_remove_arg:
    #     cleaned_text = re.sub(r'\\' + cmd + r'\{[^}]*\}', '', cleaned_text)

    # # Remove commands that don't have arguments
    # cmds_to_remove = [
    #     'maketitle', 'clearpage', 'AND', 'And', 'footnotemark', 'thanks', 'appendix', 'newpage'
    # ]
    # for cmd in cmds_to_remove:
    #     cleaned_text = re.sub(r'\\' + cmd + r'(?!\w)', '', cleaned_text)
    
    # # Remove custom commands from this specific paper
    # custom_cmds_to_remove = ['piref', 'pisft', 'methodac', 'methodfull', 'se']
    # for cmd in custom_cmds_to_remove:
    #     if '{' in cmd:
    #          cleaned_text = re.sub(r'\\' + cmd.split('{')[0] + r'\{[^}]*\}', '', cleaned_text)
    #     else:
    #         cleaned_text = re.sub(r'\\' + cmd + r'(?!\w)', '', cleaned_text)


    # # Remove environment tags but keep content
    # envs_keep_content = ['sproof'] # abstract and document are handled
    # for env in envs_keep_content:
    #     cleaned_text = re.sub(r'\\begin{' + env + r'\}', '', cleaned_text)
    #     cleaned_text = re.sub(r'\\end{' + env + r'\}', '', cleaned_text)

    # # Handle lists
    # cleaned_text = re.sub(r'\\begin{itemize}', '', cleaned_text)
    # cleaned_text = re.sub(r'\\end{itemize}', '', cleaned_text)
    # cleaned_text = re.sub(r'\\begin{enumerate}', '', cleaned_text)
    # cleaned_text = re.sub(r'\\end{enumerate}', '', cleaned_text)
    # cleaned_text = re.sub(r'\\item', '\n- ', cleaned_text)

    # # Remove comments
    # cleaned_text = re.sub(r'%.*', '', cleaned_text)
    
    # # Remove inline math expressions
    # cleaned_text = re.sub(r'\$.*?\$', '', cleaned_text)
    
    # # Clean up whitespace
    # cleaned_text = re.sub(r'~', ' ', cleaned_text)
    # cleaned_text = re.sub(r'\\ ', ' ', cleaned_text) # Explicit space command
    # cleaned_text = re.sub(r'(?<!\n)\n(?!\n)', ' ', cleaned_text)
    # cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)  # Collapse multiple newlines
    # cleaned_text = re.sub(r'[ \t]+', ' ', cleaned_text)  # Collapse multiple spaces
    # cleaned_text = cleaned_text.strip()

    return cleaned_text.strip()

if __name__ == '__main__':
    # Construct the path to the main.tex file relative to the script location
    script_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    tex_file_path = os.path.join(project_root, 'data', 'arxiv', 'main.tex')

    try:
        with open(tex_file_path, 'r', encoding='utf-8') as f:
            tex_content = f.read()
        
        cleaned_text = clean_latex(tex_content)
        print(cleaned_text)

        # Optional: Save to a file
        output_file_path = os.path.join(project_root, 'data', 'arxiv', 'DPO_cleaned.txt')
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)
        print(f"\nCleaned text saved to {output_file_path}")

    except FileNotFoundError:
        print(f"Error: The file {tex_file_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}") 