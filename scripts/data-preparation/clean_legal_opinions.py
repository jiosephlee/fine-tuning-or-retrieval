"""
Clean legal opinion text files by fixing OCR/PDF-extraction formatting.

The raw text extracted from court PDFs has several formatting issues:
  - Mid-word hyphenation from column breaks (e.g., "domes-\\ntic violence")
  - Page markers mid-paragraph ("Case: 24-1039  Document: 68  Page: 3...")
  - Standalone page/case-number lines ("2  No. 24-3051", "-8-")
  - Inconsistent line breaks within paragraphs
  - Decorative separators (underscores, "lllll" padding)

This script reflows the text into clean paragraphs while preserving:
  - All content (headers, attorney listings, opinion text)
  - Section structure (I, II, III, A, B, etc.)
  - Block quotes and dialogue (indented text)
  - Paragraph boundaries

Usage:
    python clean_legal_opinions.py --input_dir ../../data/legal/raw --dry_run
    python clean_legal_opinions.py --input_dir ../../data/legal/raw \
        --output_dir ../../data/legal/cleaned
"""

import argparse
import os
import re


def remove_page_artifacts(text: str) -> str:
    """Remove page-level artifacts that interrupt the text flow."""

    # CAFC/Fifth Circuit page headers:
    # "Case: 24-1039    Document: 68     Page: 3   Filed: 09/24/2025"
    text = re.sub(r'^\s*Case:\s+[\d\-]+\s+Document:\s+[\d\-]+\s+Page:\s+\d+.*$',
                  '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*Appellate Case:\s+[\d\-]+\s+Document:\s+\d+\s+.*$',
                  '', text, flags=re.MULTILINE)

    # Standalone case-number + page-number lines:
    # "No. 24-3051                                                9"
    text = re.sub(r'^No\.\s+[\d\-]+\s+\d+\s*$', '', text, flags=re.MULTILINE)
    # "2                                                  No. 24-3051"
    text = re.sub(r'^\d+\s+No\.\s+[\d\-]+\s*$', '', text, flags=re.MULTILINE)
    # Ninth Circuit: "  3  25-1313"
    text = re.sub(r'^\s*\d+\s+\d{2,}-\d+\s*$', '', text, flags=re.MULTILINE)

    # CAFC page-number + case-name running headers:
    # " 2                FINESSE WIRELESS LLC v. AT&T MOBILITY LLC"
    text = re.sub(r'^\s*\d+\s+[A-Z][A-Z\s\.\,\&\'\-]+v\.\s+[A-Z][A-Z\s\.\,\&\'\-]+$',
                  '', text, flags=re.MULTILINE)

    # Standalone page numbers: "-8-", "  16", "3" etc.
    # These appear as a line with only a number (1-3 digits), possibly with
    # lots of whitespace padding. Remove any line that is just a number.
    text = re.sub(r'^\s*-?\d{1,3}-?\s*$', '', text, flags=re.MULTILINE)

    # INLINE page markers that appear mid-paragraph at line boundaries.
    # These show up as "word 6 No. 24-3051" or "No. 24-3051 5" where a
    # page number accompanies the case number. We only strip these when
    # there IS a page number (to preserve the case number in the header).
    #
    # Pattern 1: page_num + "No. XX-XXXX" at end of line
    text = re.sub(r'\s+\d{1,2}\s+No\.\s+[\d\-]+\s*$', '', text, flags=re.MULTILINE)
    # Pattern 2: "No. XX-XXXX" + page_num at end of line
    text = re.sub(r'\s+No\.\s+[\d\-]+\s+\d{1,2}\s*$', '', text, flags=re.MULTILINE)

    # Standalone lines with BOTH a page number and case number:
    # "2                                  No. 24-3051" or "No. 24-3051   9"
    # (already handled above, but also the reversed form)
    text = re.sub(r'^\s*\d{1,2}\s+No\.\s+[\d\-]+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*No\.\s+[\d\-]+\s+\d{1,2}\s*$', '', text, flags=re.MULTILINE)

    # After removing page artifacts, blank lines may be left where
    # a page break interrupted a paragraph. We need to stitch these
    # back together. Detect: line ending without sentence-ending
    # punctuation → blank line(s) → line starting with lowercase or
    # a continuation pattern (quote, parenthetical, section symbol).
    # Join them by removing the blank lines.
    text = re.sub(
        r'([a-z,;:\-–—]\s*)\n\s*\n(\s*[a-z§"\'(])',
        r'\1\n\2',
        text
    )

    # FILED stamps and clerk lines that appear mid-page
    text = re.sub(r'^\s*FILED\s*$', '', text, flags=re.MULTILINE)

    return text


def remove_decorative_lines(text: str) -> str:
    """Remove decorative separators."""
    # Lines of underscores (with optional whitespace)
    text = re.sub(r'^\s*_{4,}\s*$', '', text, flags=re.MULTILINE)
    # Lines of "lllll" (Eighth Circuit alignment padding)
    text = re.sub(r'^.*l{4,}.*$', '', text, flags=re.MULTILINE)
    return text


def rejoin_hyphenated_words(text: str) -> str:
    """
    Rejoin words split across lines by hyphenation.

    "domes-\\ntic" -> "domestic"
    "rec-\\norded" -> "recorded"

    Only rejoins when the hyphen is at end of line and the next line
    starts with a lowercase letter (to avoid breaking intentional hyphens
    like "well-known" or "court-ordered" that happen to be at line end).
    """
    # Match: word-fragment + hyphen + newline + optional whitespace + lowercase continuation
    text = re.sub(r'(\w)-\n\s*([a-z])', r'\1\2', text)
    return text


def is_section_header(line: str) -> bool:
    """Check if a line is a section header (should be its own paragraph)."""
    stripped = line.strip()
    if not stripped:
        return False

    # Roman numeral sections: "I", "II", "III", "IV", "V", "VI", etc.
    if re.match(r'^[IVX]+$', stripped):
        return True

    # Letter subsections: "A", "B", "C" (standalone)
    if re.match(r'^[A-F]$', stripped):
        return True

    # Numbered sections: "1.", "2.", etc.
    if re.match(r'^\d+\.$', stripped):
        return True

    # ALL-CAPS section titles: "BACKGROUND", "DISCUSSION", "CONCLUSION", etc.
    if re.match(r'^[A-Z][A-Z\s\-&]+$', stripped) and len(stripped) > 2:
        return True

    # Mixed case section titles common in legal opinions
    if re.match(r'^(FACTUAL AND PROCEDURAL BACKGROUND|STANDARD OF REVIEW|'
                r'BACKGROUND|DISCUSSION|ANALYSIS|CONCLUSION|'
                r'STATEMENT OF THE CASE|PROCEDURAL HISTORY)$',
                stripped, re.IGNORECASE):
        return True

    return False


def is_dialogue_or_quote(line: str, prev_line: str = '') -> bool:
    """Check if a line appears to be part of a block quote or dialogue."""
    stripped = line.strip()
    leading_spaces = len(line) - len(line.lstrip())

    # Heavily indented lines (>=6 spaces) are likely block quotes
    if leading_spaces >= 6:
        return True

    return False


def join_caption_block(text: str) -> str:
    """
    Join the case caption block into a single paragraph.

    The caption typically looks like:
        AMERICA FIRST LEGAL FOUNDATION,
        APPELLANT

        v.

        JAMIESON GREER, IN HIS OFFICIAL CAPACITY,
        APPELLEE

    We join these short ALL-CAPS lines (party names, roles, "v.") into
    one block, stopping when we hit a longer substantive line.
    """
    lines = text.split('\n')
    result = []
    caption_lines = []
    in_caption = False

    def flush_caption():
        if caption_lines:
            result.append(' '.join(caption_lines))
            caption_lines.clear()

    for i, line in enumerate(lines):
        stripped = line.strip()

        if not in_caption:
            # Detect start of caption: a short ALL-CAPS line with party name
            # or a line that is just "v." preceded by what looks like a header
            if (stripped and
                len(stripped) < 80 and
                (stripped == 'v.' or
                 (stripped.isupper() and
                  any(role in stripped for role in
                      ['APPELLANT', 'APPELLEE', 'PLAINTIFF', 'DEFENDANT',
                       'PETITIONER', 'RESPONDENT']) or
                  (stripped.endswith(',') and stripped.isupper() and len(stripped) > 5)))):
                in_caption = True
                caption_lines.append(stripped)
                continue
            result.append(line)
        else:
            if not stripped:
                # Blank line within caption — skip
                continue
            elif (stripped == 'v.' or
                  (stripped.isupper() and len(stripped) < 80 and
                   any(c.isalpha() for c in stripped))):
                caption_lines.append(stripped)
            else:
                # End of caption
                flush_caption()
                in_caption = False
                result.append(line)

    flush_caption()
    return '\n'.join(result)


def reflow_paragraphs(text: str) -> str:
    """
    Reflow text into clean paragraphs.

    The PDF extraction creates artificial line breaks at column boundaries
    (~60-70 chars). We need to:
    1. Join lines within the same paragraph
    2. Preserve paragraph breaks (double newlines or indentation changes)
    3. Preserve section headers as standalone lines
    4. Preserve block quotes/dialogue formatting
    """
    lines = text.split('\n')
    paragraphs = []
    current_para = []
    in_quote_block = False

    def flush_para():
        if current_para:
            # Join lines with spaces, then clean up multiple spaces
            joined = ' '.join(current_para)
            joined = re.sub(r'  +', ' ', joined)
            paragraphs.append(joined.strip())
            current_para.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Empty line = paragraph boundary
        if not stripped:
            flush_para()
            i += 1
            continue

        # Section headers get their own paragraph
        if is_section_header(stripped):
            flush_para()
            paragraphs.append(stripped)
            i += 1
            continue

        # Detect quote/dialogue blocks (heavily indented)
        # These should preserve their line structure but rejoin
        # lines within the same speaker's turn
        leading_spaces = len(line) - len(line.lstrip())
        if leading_spaces >= 6 and is_dialogue_or_quote(line):
            if not in_quote_block:
                flush_para()
                in_quote_block = True
            # Check if this starts a new speaker turn (has "Name:" pattern)
            if re.match(r'^[A-Z][a-z]+(\s+[A-Z][a-z]+)*:', stripped):
                flush_para()
            current_para.append(stripped)
            i += 1
            continue

        if in_quote_block:
            flush_para()
            in_quote_block = False

        # Regular text line — accumulate into current paragraph
        current_para.append(stripped)
        i += 1

    flush_para()

    # Join paragraphs with double newlines
    return '\n\n'.join(paragraphs)


def normalize_final(text: str) -> str:
    """Final cleanup pass."""
    # Remove any remaining triple+ newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove trailing whitespace
    text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)
    return text.strip() + '\n'


def clean_opinion(text: str) -> tuple[str, list[str]]:
    """
    Clean a legal opinion text.
    Returns (cleaned_text, warnings).
    """
    warnings = []

    # Step 1: Remove page-level artifacts
    text = remove_page_artifacts(text)

    # Step 2: Remove decorative lines
    text = remove_decorative_lines(text)

    # Step 3: Rejoin hyphenated words
    text = rejoin_hyphenated_words(text)

    # Step 4: Reflow into clean paragraphs
    text = reflow_paragraphs(text)

    # Step 5: Final normalization
    text = normalize_final(text)

    word_count = len(text.split())
    if word_count < 500:
        warnings.append(f"MANUAL CHECK: Very short after cleaning ({word_count} words)")

    return text, warnings


def main():
    parser = argparse.ArgumentParser(
        description="Clean legal opinion text by fixing OCR/PDF formatting",
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--file", default=None, help="Only process files matching this substring")
    parser.add_argument("--show_full", action="store_true", help="Show full cleaned text in dry run")

    args = parser.parse_args()

    if not args.output_dir and not args.dry_run:
        parser.error("--output_dir is required unless using --dry_run")

    files = sorted(f for f in os.listdir(args.input_dir) if f.endswith('.txt'))
    if args.file:
        files = [f for f in files if args.file.lower() in f.lower()]

    if not files:
        print("No matching files found.")
        return

    all_warnings = {}

    for filename in files:
        filepath = os.path.join(args.input_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            raw_text = f.read()

        raw_words = len(raw_text.split())
        cleaned, warnings = clean_opinion(raw_text)
        clean_words = len(cleaned.split())

        print(f"\n{'='*60}")
        print(f"  {filename}")
        print(f"  Raw: {raw_words} words → Cleaned: {clean_words} words")

        if warnings:
            all_warnings[filename] = warnings
            for w in warnings:
                print(f"  WARNING: {w}")

        if args.dry_run:
            if args.show_full:
                print(f"  ---FULL TEXT---")
                for line in cleaned.split('\n'):
                    print(f"  | {line}")
            else:
                lines = cleaned.split('\n')
                print(f"  ---FIRST 10 LINES---")
                for line in lines[:10]:
                    print(f"  | {line}")
                if len(lines) > 20:
                    print(f"  | ...")
                    print(f"  ---LAST 5 LINES---")
                    for line in lines[-5:]:
                        print(f"  | {line}")
        else:
            os.makedirs(args.output_dir, exist_ok=True)
            outpath = os.path.join(args.output_dir, filename)
            with open(outpath, 'w', encoding='utf-8') as f:
                f.write(cleaned)
            print(f"  Saved: {outpath}")

    print(f"\n{'='*60}")
    print(f"Files processed: {len(files)}")
    if all_warnings:
        print(f"\nFiles needing manual review:")
        for fname, warns in all_warnings.items():
            for w in warns:
                print(f"  - {fname}: {w}")


if __name__ == "__main__":
    main()
