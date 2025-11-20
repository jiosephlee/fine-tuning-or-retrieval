import os
import re
import random
import sys

sys.path.append('../../')

def extract_paper_title(paper_path):
    """Extract the title from the paper .tex file."""
    if not os.path.exists(paper_path):
        return "Unknown Paper"
    
    with open(paper_path, 'r') as f:
        content = f.read()
    
    title_match = re.search(r'\\title\{([^}]+)\}', content)
    if title_match:
        return title_match.group(1)
    
    return "Unknown Paper"

def shuffle_words_in_text(text):
    """Shuffle all words in the text while preserving line breaks for headers and titles."""
    lines = text.split('\n')
    shuffled_lines = []
    first_non_empty = True
    
    for line in lines:
        # Keep the first non-empty line (chapter title) unshuffled
        if first_non_empty and line.strip():
            shuffled_lines.append(line)
            first_non_empty = False
        # Check if it's a header line (starts with #)
        elif line.strip().startswith('#'):
            # Keep headers as-is
            shuffled_lines.append(line)
        elif line.strip():
            # Shuffle words in this line
            words = line.split()
            random.shuffle(words)
            shuffled_lines.append(' '.join(words))
        else:
            # Keep empty lines
            shuffled_lines.append(line)
    
    return '\n'.join(shuffled_lines)

def shuffle_sentences_in_text(text):
    """Shuffle all sentences in the text while preserving title and headers."""
    lines = text.split('\n')
    title = None
    headers = []
    non_header_text = []
    first_non_empty = True
    
    for line in lines:
        # Keep the first non-empty line as title
        if first_non_empty and line.strip():
            title = line
            first_non_empty = False
        elif line.strip().startswith('#'):
            headers.append(line)
        else:
            non_header_text.append(line)
    
    # Rejoin non-header text and split by sentences
    full_text = ' '.join(non_header_text)
    
    # Split by periods, question marks, and exclamation marks
    sentences = re.split(r'([.!?]\s+)', full_text)
    
    # Combine sentences with their punctuation
    combined_sentences = []
    for i in range(0, len(sentences)-1, 2):
        if i+1 < len(sentences):
            combined_sentences.append(sentences[i] + sentences[i+1])
        else:
            combined_sentences.append(sentences[i])
    
    # Add the last piece if it doesn't end with punctuation
    if len(sentences) % 2 == 1:
        combined_sentences.append(sentences[-1])
    
    # Filter out empty sentences
    combined_sentences = [s.strip() for s in combined_sentences if s.strip()]
    
    # Shuffle sentences
    random.shuffle(combined_sentences)
    
    # Reconstruct text with title and headers at the beginning
    result_parts = []
    if title:
        result_parts.append(title)
    if headers:
        result_parts.append('\n'.join(headers))
    if combined_sentences:
        result_parts.append(' '.join(combined_sentences))
    
    result = '\n\n'.join(result_parts)
    
    return result

def generate_shuffled_textbooks(paper_name, base_dir):
    """Generate both word-shuffled and sentence-shuffled textbooks."""
    print(f"\n{'='*60}")
    print(f"Generating shuffled textbooks for: {paper_name}")
    print(f"{'='*60}")
    
    paper_path = os.path.join(base_dir, 'cleaned', f'{paper_name}.tex')
    textbooks_dir = os.path.join(base_dir, 'explanations', paper_name, 'textbooks')
    
    # Output directories
    words_output_dir = os.path.join(base_dir, 'explanations', paper_name, 'shuffled_words_textbook')
    sentences_output_dir = os.path.join(base_dir, 'explanations', paper_name, 'shuffled_sentences_textbook')
    
    os.makedirs(words_output_dir, exist_ok=True)
    os.makedirs(sentences_output_dir, exist_ok=True)
    
    # Check if textbooks directory exists
    if not os.path.exists(textbooks_dir):
        print(f"Textbooks directory not found: {textbooks_dir}")
        return
    
    paper_title = extract_paper_title(paper_path)
    
    # Get all chapter files
    chapter_files = sorted([f for f in os.listdir(textbooks_dir) if f.startswith('chapter_') and f.endswith('.txt')])
    print(f"Found {len(chapter_files)} chapters to process")
    
    all_word_shuffled_chapters = []
    all_sentence_shuffled_chapters = []
    
    for chapter_file in chapter_files:
        chapter_path = os.path.join(textbooks_dir, chapter_file)
        print(f"Processing {chapter_file}...")
        
        # Load chapter
        with open(chapter_path, 'r') as f:
            chapter_content = f.read()
        
        # Generate word-shuffled version
        word_shuffled = shuffle_words_in_text(chapter_content)
        all_word_shuffled_chapters.append(word_shuffled)
        
        # Save individual word-shuffled chapter
        word_output_path = os.path.join(words_output_dir, chapter_file)
        with open(word_output_path, 'w') as f:
            f.write(word_shuffled)
        
        # Generate sentence-shuffled version
        sentence_shuffled = shuffle_sentences_in_text(chapter_content)
        all_sentence_shuffled_chapters.append(sentence_shuffled)
        
        # Save individual sentence-shuffled chapter
        sentence_output_path = os.path.join(sentences_output_dir, chapter_file)
        with open(sentence_output_path, 'w') as f:
            f.write(sentence_shuffled)
    
    # Save combined word-shuffled textbook
    full_word_shuffled = "\n\n".join(all_word_shuffled_chapters)
    full_word_textbook = f"Title: Textbook of {paper_title}\n\n{full_word_shuffled}"
    
    word_textbook_path = os.path.join(words_output_dir, 'textbook.txt')
    with open(word_textbook_path, 'w') as f:
        f.write(full_word_textbook)
    
    print(f"Saved word-shuffled textbook to: {word_textbook_path}")
    
    # Save combined sentence-shuffled textbook
    full_sentence_shuffled = "\n\n".join(all_sentence_shuffled_chapters)
    full_sentence_textbook = f"Title: Textbook of {paper_title}\n\n{full_sentence_shuffled}"
    
    sentence_textbook_path = os.path.join(sentences_output_dir, 'textbook.txt')
    with open(sentence_textbook_path, 'w') as f:
        f.write(full_sentence_textbook)
    
    print(f"Saved sentence-shuffled textbook to: {sentence_textbook_path}")

def main():
    base_dir = '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/data/arxiv'
    
    # Get all domains from explanations directory
    explanations_dir = os.path.join(base_dir, 'explanations')
    domains = [d for d in os.listdir(explanations_dir) 
               if os.path.isdir(os.path.join(explanations_dir, d))]
    
    print(f"Found {len(domains)} domains to process")
    
    for domain in domains:
        try:
            generate_shuffled_textbooks(domain, base_dir)
        except Exception as e:
            print(f"Error generating shuffled textbooks for {domain}: {e}")

if __name__ == "__main__":
    main()

