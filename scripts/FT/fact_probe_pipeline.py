import re
import pandas as pd

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


extraction_prompt = r"""Your task is to act as a text segmenter. Carefully read the provided text from a paper and identify all sentences that contain "pieces of knowledge" or "facts."

# What to Exclude (Do NOT tag these):
- Sentences that are transitional and for structural purposes of the paper, mainly containing language that's generic to any paper, adding zero information e.g. "Our results raise several important questions for future work.", "In this section, we discuss our methodology in relation to other works.
- Author Speculation or Rhetorical Questions: Subjective statements or questions posed to the reader (e.g., "This result is quite surprising.", "But what if the model could...?"). 
- Figures and Tables: Latex commands that generate figures and tables.

# Instructions
1.  Read the entire text carefully.
2.  Identify all sentences that contain a "piece of knowledge" or a "fact" and do not fall into the exclusion categories.
3.  Wrap each of these sentences in `<knowledge>` and `</knowledge>` tags.
4.  You can tag *captions* of the table or figure, but please DO NOT tag the other parts of the table/figure.
5.  For sentences that contain latex commands, place the tags so that it includes any latex code that's part of the sentence e.g. "\begin{definition}" or "\text{...}".
6.  For sentences that contain math, make sure to include all the latex of the math within the tags.
7.  Please make sure the tags cover the ENTIRE sentence i.e. the tags are at the beginning and end of the sentence.
8.  Return the entire original text with these annotations. Do not modify or summarize the text itself."""

# Parse paper structure
paper_df = parse_paper_structure(paper)

# Process each paragraph with LLM
import concurrent.futures

def query_single(paragraph):
    prompt = {}
    prompt['system'] = extraction_prompt
    prompt['user'] = f"""{paragraph}"""
    return utils.query_llm(prompt, model='gpt-4.1')

with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = [executor.submit(query_single, row['paragraph']) for _, row in paper_df.iterrows()]
    extracted_claims = [future.result() for future in futures]

# Add extracted claims to dataframe
paper_df['extracted_claims'] = extracted_claims

# Print each output with section/subsection context
for i, (_, row) in enumerate(paper_df.iterrows(), 1):
    print_wrapped(f"Section: {row['section']}")
    print_wrapped(f"Subsection: {row['subsection']}")
    print_wrapped(f"Paragraph {i}: {row['extracted_claims']}")
    print_wrapped("-" * 50)