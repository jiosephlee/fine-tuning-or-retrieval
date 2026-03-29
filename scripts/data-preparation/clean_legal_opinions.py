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
    text = text.replace('\f', '\n')

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
    text = re.sub(r'^\s*PUBLISH\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[A-Z][a-z]+\s+\d{1,2},\s+\d{4}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^.*Clerk of Court\s*$', '', text, flags=re.MULTILINE)

    return text


def remove_decorative_lines(text: str) -> str:
    """Remove decorative separators."""
    # Lines of underscores (with optional whitespace)
    text = re.sub(r'^\s*_{4,}\s*$', '', text, flags=re.MULTILINE)
    # Lines of "lllll" (Eighth Circuit alignment padding)
    text = re.sub(r'^.*l{4,}.*$', '', text, flags=re.MULTILINE)
    # Common PDF separator rules
    text = re.sub(r'^\s*[_\-\u2014]{8,}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*_{3,}.*_{3,}\s*$', '', text, flags=re.MULTILINE)
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


def remove_footnotes(text: str) -> str:
    """Remove extracted footnote blocks and inline footnote markers."""
    lines = text.split('\n')
    result = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Footnotes often appear as a standalone number followed by indented lines.
        if re.match(r'^\s*\d+\s*$', line):
            j = i + 1
            footnote_lines = []
            while j < len(lines):
                candidate = lines[j]
                stripped = candidate.strip()
                leading_spaces = len(candidate) - len(candidate.lstrip())

                if not stripped:
                    if footnote_lines:
                        j += 1
                        break
                    j += 1
                    continue

                if re.match(r'^(Case:|Appellate Case:)', stripped):
                    break

                if leading_spaces >= 4:
                    footnote_lines.append(candidate)
                    j += 1
                    continue

                break

            if footnote_lines:
                i = j
                continue

        result.append(line)
        i += 1

    text = '\n'.join(result)

    # Remove inline footnote callouts like "Party. 1 Other..." after dropping the
    # corresponding footnote body.
    text = re.sub(r'(?<=[\.\?\!”\)])\s+\d+\s+(?=[A-Z])', ' ', text)
    return text


def is_section_header(line: str) -> bool:
    """Check if a line is a section header (should be its own paragraph)."""
    stripped = re.sub(r'\s+', ' ', line.strip())
    if not stripped:
        return False

    # Roman numeral sections: "I", "II", "III", "IV", "V", "VI", etc.
    if re.match(r'^[IVX]+$', stripped):
        return True
    if re.match(r'^[IVX]+[\.\)]\s+\S', stripped):
        return True

    # Letter subsections: "A", "B", "C" (standalone)
    if re.match(r'^[A-F]$', stripped):
        return True
    if re.match(r'^[A-Z][\.\)]\s+\S', stripped):
        return True

    # Numbered sections: "1.", "2.", etc.
    if re.match(r'^\d+\.$', stripped):
        return True
    if re.match(r'^\d{1,2}[\.\)]\s+\S', stripped) and len(stripped) <= 80:
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

    if re.match(r'^[A-Z][a-z]+(\s+[A-Z][a-z]+)*:', stripped):
        return True

    prev_indent = len(prev_line) - len(prev_line.lstrip()) if prev_line else 0

    # Block quotes tend to stay deeply indented across multiple lines. Ordinary
    # legal paragraphs often use a first-line indent, so do not treat those as quotes.
    if leading_spaces >= 10 and prev_indent >= 10:
        return True

    return False


def is_captionish_line(line: str) -> bool:
    """Detect short caption/front-matter lines that should stay separate."""
    stripped = line.strip()
    if not stripped:
        return False

    if stripped in {
        'Petitioner.', 'Petitioners.', 'Respondent.', 'Respondents.',
        'Appellant.', 'Appellee.', 'Plaintiff-Appellee.', 'Defendant-Appellant.',
        'Petition for Review', 'Per Curiam.',
    }:
        return True

    if stripped.lower() == 'v.':
        return True

    if re.match(r'^(Before|Appeal from|Petition for Review|Opinion for the Court)', stripped):
        return True

    if re.match(r'^No\.\s+[\d\-]+$', stripped):
        return True

    if ('Court of Appeals' in stripped or 'Circuit' in stripped) and len(stripped) <= 100:
        return True

    if len(stripped) <= 100 and stripped.isupper():
        return True

    return False


def should_merge_across_blank(prev_line: str, next_line: str) -> bool:
    """Treat PDF-extraction blank lines as soft wraps when context suggests it."""
    prev = prev_line.strip()
    nxt = next_line.strip()
    if not prev or not nxt:
        return False

    if is_section_header(prev) or is_section_header(nxt):
        return False

    if is_captionish_line(prev) or is_captionish_line(nxt):
        return False

    next_indent = len(next_line) - len(next_line.lstrip())

    if prev.endswith(('-', '–', '—')):
        return True

    if re.search(r'[,:;(\[]$', prev):
        return True

    if not re.search(r'[.!?:"”\')\]]$', prev):
        return True

    if next_indent < 4 and not re.match(r'^[A-Z][A-Z\s]+$', nxt):
        return True

    return False


def find_body_start(lines: list[str]) -> int:
    """Return the line index where the opinion body likely begins."""
    for i, line in enumerate(lines):
        stripped = re.sub(r'\s+', ' ', line.strip())
        if not stripped:
            continue

        if re.match(r'^[A-Z][A-Z\s\.\'\-]+,\s+(Circuit Judge|Judge|Chief Judge)\.?$', stripped):
            return min(i + 1, len(lines))

        if re.match(r'^[A-Z][A-Z\s\.\'\-]+,\s+(Circuit Judge|Judge|Chief Judge):', stripped):
            return i

        if stripped in {'PER CURIAM.', 'Per Curiam.'}:
            return min(i + 1, len(lines))

    return 0


def normalize_front_matter(text: str) -> str:
    """Preserve caption/front matter structure while cleaning spacing lightly."""
    paragraphs = []
    current = []

    for line in text.split('\n'):
        stripped = re.sub(r'\s+', ' ', line.strip())
        if not stripped:
            if current:
                paragraphs.append(' '.join(current))
                current = []
            continue

        if current and not should_merge_across_blank(current[-1], stripped):
            paragraphs.append(' '.join(current))
            current = [stripped]
        else:
            current.append(stripped)

    if current:
        paragraphs.append(' '.join(current))

    return '\n\n'.join(paragraphs)


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
            next_nonempty = ''
            j = i + 1
            while j < len(lines):
                if lines[j].strip():
                    next_nonempty = lines[j]
                    break
                j += 1

            prev_nonempty = current_para[-1] if current_para else ''
            if prev_nonempty and next_nonempty and should_merge_across_blank(prev_nonempty, next_nonempty):
                i += 1
                continue

            flush_para()
            i += 1
            continue

        # Section headers get their own paragraph
        if is_section_header(stripped):
            flush_para()
            paragraphs.append(re.sub(r'\s+', ' ', stripped))
            i += 1
            continue

        # Detect quote/dialogue blocks (heavily indented)
        # These should preserve their line structure but rejoin
        # lines within the same speaker's turn
        leading_spaces = len(line) - len(line.lstrip())
        if is_dialogue_or_quote(line, lines[i - 1] if i > 0 else ''):
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


def merge_broken_paragraphs(text: str) -> str:
    """Merge adjacent paragraphs when the first clearly ends mid-sentence."""
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    if not paragraphs:
        return text

    merged = [paragraphs[0]]
    for para in paragraphs[1:]:
        prev = merged[-1]

        if (
            not is_section_header(prev)
            and not is_section_header(para)
            and not is_captionish_line(prev)
            and not is_captionish_line(para)
            and not re.search(r'[.!?:"”\')\]]$', prev)
        ):
            merged[-1] = f"{prev} {para}"
        else:
            merged.append(para)

    return '\n\n'.join(merged)


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

    # Step 1: Remove extracted footnotes before page cleanup removes their anchors
    text = remove_footnotes(text)

    # Step 2: Remove page-level artifacts
    text = remove_page_artifacts(text)

    # Step 3: Remove decorative lines
    text = remove_decorative_lines(text)

    # Step 4: Rejoin hyphenated words
    text = rejoin_hyphenated_words(text)

    # Step 5: Preserve front matter lightly and reflow the opinion body aggressively.
    lines = text.split('\n')
    body_start = find_body_start(lines)
    front_matter = '\n'.join(lines[:body_start]).strip()
    body = '\n'.join(lines[body_start:]).strip()

    cleaned_parts = []
    if front_matter:
        cleaned_parts.append(normalize_front_matter(front_matter))
    if body:
        cleaned_parts.append(reflow_paragraphs(body))
    text = '\n\n'.join(part for part in cleaned_parts if part)

    # Step 6: Merge paragraphs that still break mid-sentence after reflow.
    text = merge_broken_paragraphs(text)

    # Step 7: Final normalization
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
