# Textbook Stripping Scripts

These scripts create stripped-down versions of textbooks by replacing probe target words with synonyms.

## Overview

The scripts process textbooks from `/data/arxiv/explanations/{domain}/textbook.txt` and:
1. Extract target words from factual and inference probes (breaking apart multi-word phrases)
2. Filter out common English stop words (e.g., "the", "a", "is", "of", "in")
3. **Filter by relative frequency**: Only keep words that appear MORE frequently in the textbook than in the source paper (relative to document length)
4. Identify paragraphs containing these filtered target words
5. Use an LLM to replace each individual target word with:
   - Synonyms that exist in the source paper (preferred)
   - True synonyms that are NOT lexically similar (if source synonyms unavailable)
6. Track detailed statistics about replacements

**Important Notes**: 
- Multi-word phrases like "feature filtering" are broken into individual words ("feature", "filtering") and each word is replaced separately
- Common stop words are automatically excluded from replacement to preserve readability and avoid unnecessary changes
- Words are only replaced if they appear more frequently (relative to document length) in the textbook than in the source paper - this prevents replacing words that naturally occur at similar rates in both documents

## Files

- `strip_textbook_probe_words.py` - Main script that processes ALL domains
- `strip_textbook_probe_words_test.py` - Test script that processes only the first 3 flagged paragraphs of ONE domain

## Usage

### Testing (Recommended First)

```bash
cd scripts/FT
python strip_textbook_probe_words_test.py
```

This will:
- Process only the domain '1_58' (you can change this in the script)
- Only replace words in the first 3 flagged paragraphs
- Save outputs to `/data/arxiv/explanations_stripped_test/`

### Full Run

```bash
cd scripts/FT
python strip_textbook_probe_words.py
```

This will:
- Process ALL domains in `/data/arxiv/explanations/`
- Replace words in ALL flagged paragraphs
- Save outputs to `/data/arxiv/explanations_stripped/`

**Warning**: This will make many LLM API calls and may take considerable time!

## Input Files

For each domain (e.g., '1_58'), the script expects:
- Textbook: `/data/arxiv/explanations/{domain}/textbook.txt`
- Factual probes: `/data/probes/facts/{domain}/probes_v9.csv`
- Inference probes: `/data/probes/inference/{domain}/probes_v6.csv`
- Source paper: `/data/arxiv/cleaned/{domain}.tex`

## Output Files

For each domain:
- `{domain}/textbook_stripped_level1.txt` - Modified textbook
- `{domain}/replacement_stats.json` - Statistics including:
  - Number of replacements
  - Percentage from source paper vs other synonyms
  - Detailed replacement map
  
Plus:
- `aggregate_stats.json` - Combined statistics across all domains

## Filtering Strategy

### 1. Stop Words Filtering

The scripts automatically exclude common English stop words from replacement, including:
- Articles: a, an, the
- Prepositions: in, on, at, by, for, from, to, with
- Conjunctions: and, or, but
- Common verbs: is, are, was, were, be, been, has, have, had, do, does, did
- Pronouns: he, she, it, they, we, you, I, me, him, her, them, us
- Other common words: this, that, what, which, who, when, where, how, etc.

This ensures that only meaningful content words are replaced, preserving the natural flow and readability of the text.

### 2. Relative Frequency Filtering

After stop word filtering, the scripts perform a frequency-based filter:

**Formula**: 
- `textbook_freq = count(word in textbook) / len(textbook)`
- `source_freq = count(word in source paper) / len(source paper)`
- **Keep word only if**: `textbook_freq > source_freq`

**Why this matters**:
- Prevents replacing words that naturally appear in both the textbook and source paper at similar rates
- Focuses on words that the textbook over-uses compared to the original paper
- Significantly reduces unnecessary replacements while targeting words that are truly over-represented in the explanations

**Example**:
- If "transformer" appears 50 times in a 10,000 word textbook (freq=0.005) but 30 times in a 2,000 word paper (freq=0.015), it will NOT be replaced because it's actually more frequent in the source paper
- If "layers" appears 100 times in the textbook (freq=0.01) but 5 times in the paper (freq=0.0025), it WILL be replaced because it's over-represented in the textbook

## Statistics Tracked

For each domain:
- Total probes involved
- Unique target words (after stop word filtering)
- Unique target words (after frequency filtering)
- Words filtered out by frequency check
- Number of flagged paragraphs
- Total replacements made
- Replacements from source paper (count & percentage)
- Replacements from other synonyms (count & percentage)
- Detailed map of all replacements

The test script additionally shows verbose output for each word during frequency filtering, showing the exact counts and frequencies in both documents.

## Example Output Structure

```
data/arxiv/explanations_stripped/
├── 1_58/
│   ├── textbook_stripped_level1.txt
│   └── replacement_stats.json
├── DPO/
│   ├── textbook_stripped_level1.txt
│   └── replacement_stats.json
└── aggregate_stats.json
```

## Notes

- The script uses `gpt-5-mini` for replacements (configurable in the code)
- Multi-word phrases are preserved and replaced as complete phrases
- LaTeX notation is handled appropriately
- The script processes flagged paragraphs sequentially with progress bars

