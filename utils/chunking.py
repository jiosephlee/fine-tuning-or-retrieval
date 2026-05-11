from typing import List
import math
import re


def _normalize_title(title: str) -> str:
    return " ".join(title.strip().split())


def _format_title_prefix(title: str, label: str = "latex") -> str:
    title = _normalize_title(title)
    if not title:
        return ""
    if label == "latex":
        return f"\\title{{{title}}}\n\n"
    return f"{label}: {title}\n\n"


def _is_title_block(text: str) -> bool:
    stripped = text.strip()
    return bool(
        re.match(r'^\\title\{[^}]+\}', stripped)
        or re.match(r'(?is)^Title\s*:', stripped)
        or re.match(r'(?is)^Case\s+(?:Title|Name)\s*:', stripped)
    )


def extract_title_prefix(text_content: str) -> str:
    """Extract a title prefix, preserving the document's title-header style."""
    latex_match = re.search(r'\\title\{([^}]+)\}', text_content)
    if latex_match:
        return _format_title_prefix(latex_match.group(1), label="latex")

    line_match = re.search(
        r'(?im)^\s*(Title|Case\s+(?:Title|Name))\s*:\s*(.+?)\s*$',
        text_content,
    )
    if line_match:
        return _format_title_prefix(
            line_match.group(2),
            label=_normalize_title(line_match.group(1)),
        )
    return ""

def chunk_texts(texts: List[str], tokenizer, context_length: int) -> tuple[List[str], int]:
    """
    Chunks a list of texts into smaller pieces based on context length.
    
    Args:
        texts: List of text strings to chunk
        tokenizer: The tokenizer to use
        context_length: Maximum tokens per chunk
        
    Returns:
        Tuple of (all_text_chunks, total_tokens)
    """
    all_text_chunks = []
    total_tokens = 0
    
    for text_content in texts:
        tokens = tokenizer(text_content, add_special_tokens=False, truncation=False)["input_ids"]
        num_tokens = len(tokens)
        total_tokens += num_tokens
        num_chunks = math.ceil(num_tokens / context_length)
        
        for i in range(num_chunks):
            start_idx = i * context_length
            end_idx = min((i + 1) * context_length, num_tokens)
            chunk_tokens = tokens[start_idx:end_idx]
            # print(chunk_tokens[end_idx-1-start_idx])
            chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=False)
            all_text_chunks.append(chunk_text)
    
    return all_text_chunks, total_tokens

def chunk_text(text_content: str, tokenizer, max_tokens: int, delimiter: str = "\n\n", overlap: bool = False, overlap_ratio_str: str = "1_4", add_title_prefix: bool = True, log=None) -> tuple[List[str], int]:
    """
    Chunks a single text by a delimiter, similar to chunk_text_by_sections,
    but using the delimiter to create parts instead of latex sections.
    """
    # 1. Handle Title
    title = ""
    if add_title_prefix:
        title = extract_title_prefix(text_content)
        if not title and log:
            log.warning("add_title_prefix is True, but no title was found.")

    # 2. Split text into parts
    parts = text_content.split(delimiter)

    chunks = []
    current_chunk = ""

    for part in parts:
        if not part.strip():
            continue

        # Add title to new/empty chunks
        is_title_block = _is_title_block(part)
        if not current_chunk and not is_title_block:
            current_chunk = title

        # Test adding new part
        separator = delimiter if current_chunk and current_chunk != title else ""
        candidate_chunk = current_chunk + separator + part
        
        candidate_tokens = tokenizer(candidate_chunk, add_special_tokens=False, truncation=False)["input_ids"]

        if len(candidate_tokens) <= max_tokens:
            current_chunk = candidate_chunk
        else:
            # Doesn't fit. Finalize previous chunk.
            if current_chunk and current_chunk != title:
                chunks.append(current_chunk)
            
            # Start new chunk with this part
            current_chunk = title
            
            # Overlap logic, using line-based context like in chunk_text_by_sections
            if overlap and chunks:
                overlap_numer, overlap_denom = map(int, overlap_ratio_str.split("_"))
                prev_chunk_lines = chunks[-1].split('\n\n')
                overlap_point = len(prev_chunk_lines) * (overlap_denom - overlap_numer) // overlap_denom
                context_from_prev = '\n\n'.join(prev_chunk_lines[overlap_point:])
                
                # Log overlap details
                if log:
                    overlap_tokens = tokenizer(context_from_prev, add_special_tokens=False, truncation=False)["input_ids"]
                    log.info(f"Overlap detected. Adding {len(overlap_tokens)} tokens from previous chunk.")
                
                current_chunk += context_from_prev
            
            current_chunk += (delimiter if current_chunk and current_chunk != title else "") + part

            if len(tokenizer(current_chunk, add_special_tokens=False)["input_ids"]) > max_tokens:
                if log:
                    log.warning("A single part with title/overlap is larger than max_tokens. This chunk may be oversized. Falling back to chunking part without overlap.")
                current_chunk = title + part
                if len(tokenizer(current_chunk, add_special_tokens=False)["input_ids"]) > max_tokens and log:
                    log.warning("A single part with title is larger than max_tokens.")

    # Add the last chunk
    if current_chunk and current_chunk != title:
        chunks.append(current_chunk)

    total_tokens = sum(len(tokenizer(c, add_special_tokens=False)["input_ids"]) for c in chunks)
    if log and chunks: 
        log.info(f"Chunking results: {len(chunks)} chunks, {total_tokens} total tokens.")
        if len(chunks) > 0:
            first_chunk_tokens = len(tokenizer(chunks[0], add_special_tokens=False)["input_ids"])
            log.info(f"First chunk ({first_chunk_tokens} tokens): '{chunks[0][:150]}...'")
        if len(chunks) > 1:
            second_chunk_tokens = len(tokenizer(chunks[1], add_special_tokens=False)["input_ids"])
            log.info(f"Second chunk ({second_chunk_tokens} tokens): '{chunks[1][:150]}...'")
    return chunks, total_tokens

def split_text_by_subsections(text_content: str, tokenizer, max_tokens: int = 2048):
    """
    Splits text by subsections and validates each subsection doesn't exceed max_tokens.
    
    Args:
        text_content: The section text to split
        tokenizer: The tokenizer to use for counting tokens
        max_tokens: Maximum tokens per subsection
        
    Returns:
        List of subsections and total token count
        
    Raises:
        ValueError: If any subsection exceeds max_tokens
    """
    subsection_pattern = r'(\\subsection\{[^}]+\})'
    parts = re.split(subsection_pattern, text_content)
    
    subsections = []
    
    # Handle content before first subsection
    if parts[0].strip():
        subsections.append(parts[0])
    
    # Group subsection headers with their content
    i = 1
    while i < len(parts):
        if re.match(subsection_pattern, parts[i]):  # This is a subsection header
            subsection_content = parts[i]  # Start with the subsection header
            if i + 1 < len(parts):
                subsection_content += parts[i + 1]  # Add the content after the header
            subsections.append(subsection_content)
            i += 2
        else:
            i += 1
    
    # If we only have one subsection (no actual subsection splits), return original
    if len(subsections) <= 1:
        subsections = [text_content]
    
    # Validate that no subsection exceeds max_tokens
    total_tokens = 0
    for i, subsection in enumerate(subsections):
        if not subsection.strip():
            continue
        tokens = tokenizer(subsection, add_special_tokens=False, truncation=False)["input_ids"]
        token_count = len(tokens)
        total_tokens += token_count
        
        if token_count > max_tokens:
            raise ValueError(f"Subsection {i} has {token_count} tokens, which exceeds max_tokens ({max_tokens})")
    
    return subsections, total_tokens

def chunk(
    text_content: str, 
    tokenizer, 
    max_tokens: int, 
    chunk_by_section: bool,
    overlap: bool = False, 
    overlap_ratio_str: str = "1_4", 
    add_title_prefix: bool = True,
    log = None,
) -> tuple[List[str], int]:
    """
    Universal chunking function that routes to either token-based or section-based chunking.
    """
    if chunk_by_section:
        overlap_numer, overlap_denom = map(int, overlap_ratio_str.split("_"))
        return chunk_text_by_sections(
            text_content, 
            tokenizer, 
            max_tokens, 
            overlap, 
            overlap_denom, 
            overlap_numer, 
            log, 
            add_title_prefix,
        )
    else:
        return chunk_text(
            text_content,
            tokenizer,
            max_tokens,
            overlap=overlap,
            overlap_ratio_str=overlap_ratio_str,
            add_title_prefix=add_title_prefix,
            log=log,
        )


def chunk_text_by_sections(text_content: str, tokenizer, max_tokens: int = 2048, overlap_sections = False, overlap_denom = 4, overlap_numer = 1, log=None, add_title_prefix: bool = True):
    """
    Chunks text by sections, ensuring each chunk doesn't exceed max_tokens.
    If a section is too large, it tries to split by subsections first.
    If no subsections exist or they're still too large, it falls back to token-based chunking.
    Each new chunk is prefixed with the document's title.

    Args:
        text_content: The full text to chunk
        tokenizer: The tokenizer to use for counting tokens
        max_tokens: Maximum tokens per chunk
        add_title_prefix: If True, prefixes each new chunk with the document's title.

    Returns:
        List of text chunks and total token count
    """
    # Extract title
    title = ""
    if add_title_prefix:
        title = extract_title_prefix(text_content)
        if not title:
            raise ValueError(
                "No supported title header found in text. Expected one of: "
                "\\title{...}, Title: ..., Case Title: ..., Case Name: ..."
            )

    # Find all section boundaries and split into sections
    section_pattern = r'(\\section\{[^}]+\})'
    if not re.search(section_pattern, text_content):
        return chunk_text(
            text_content,
            tokenizer,
            max_tokens,
            delimiter="\n\n",
            overlap=overlap_sections,
            overlap_ratio_str=f"{overlap_numer}_{overlap_denom}",
            add_title_prefix=add_title_prefix,
            log=log,
        )

    parts = re.split(section_pattern, text_content)
    
    sections = []
    # Handle content before first section (e.g., abstract)
    if parts[0].strip():
        sections.append(parts[0])
    
    # Group section headers with their content
    i = 1
    while i < len(parts):
        if re.match(section_pattern, parts[i]):  # This is a section header
            section_content = parts[i]  # Start with the section header
            if i + 1 < len(parts):
                section_content += parts[i + 1]  # Add the content after the header
            sections.append(section_content)
            i += 2
        else:
            i += 1

    chunks = []
    current_chunk = ""
    
    for section in sections:
        if not section.strip():
            continue

        # Handle title for the very first chunk or any new, empty chunk.
        is_title_block = _is_title_block(section)
        if not current_chunk and not is_title_block:
            current_chunk = title

        # First, try to fit the whole section.
        candidate_chunk = current_chunk + section
        candidate_tokens = tokenizer(candidate_chunk, add_special_tokens=False, truncation=False)["input_ids"]

        if len(candidate_tokens) <= max_tokens:
            # Section fits, add it to the current chunk.
            current_chunk = candidate_chunk
        else:
            pieces, _ = split_text_by_subsections(section, tokenizer, max_tokens)
            # Try to add pieces to the current chunk until it's full.
            for piece in pieces:
                piece_candidate = current_chunk + piece
                piece_candidate_tokens = tokenizer(piece_candidate, add_special_tokens=False, truncation=False)["input_ids"]

                if len(piece_candidate_tokens) <= max_tokens:
                    # Piece fits, add it.
                    current_chunk = piece_candidate
                else:
                    # Piece does not fit. Finalize the current chunk.
                    if current_chunk.strip():
                        chunks.append(current_chunk)
                    
                    # Start a new chunk with the title and this piece.
                    if current_chunk.strip() and overlap_sections:
                        # Get the latter half of the previous chunk for context
                        prev_sentences = current_chunk.split('\n')
                        overlap_point = len(prev_sentences) * (overlap_denom - overlap_numer) // overlap_denom
                        context_from_prev = '\n'.join(prev_sentences[overlap_point:])
                        
                        # Log overlap details
                        if log:
                            overlap_tokens = tokenizer(context_from_prev, add_special_tokens=False, truncation=False)["input_ids"]
                            log.info(f"Overlap detected. Adding {len(overlap_tokens)} tokens from previous chunk.")
                        
                        current_chunk = title + context_from_prev + piece
                    else:
                        current_chunk = title + piece

    # Add the final chunk if it has content.
    if current_chunk.strip():
        chunks.append(current_chunk)
    
    total_tokens = sum(len(tokenizer(c, add_special_tokens=False)["input_ids"]) for c in chunks)
    if log and chunks: 
        log.info(f"Chunking results: {len(chunks)} chunks, {total_tokens} total tokens.")
        if len(chunks) > 0:
            first_chunk_tokens = len(tokenizer(chunks[0], add_special_tokens=False)["input_ids"])
            log.info(f"First chunk ({first_chunk_tokens} tokens): '{chunks[0][:150]}...'")
        if len(chunks) > 1:
            second_chunk_tokens = len(tokenizer(chunks[1], add_special_tokens=False)["input_ids"])
            log.info(f"Second chunk ({second_chunk_tokens} tokens): '{chunks[1][:150]}...'")
    return chunks, total_tokens
