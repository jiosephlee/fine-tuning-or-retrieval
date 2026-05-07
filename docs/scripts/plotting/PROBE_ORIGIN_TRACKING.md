# Probe Origin Tracking System

## Overview

This document explains how probe origins are tracked and categorized across the codebase.

## Four Categories

Probes are now categorized into **four distinct groups** based on where their target text appears:

1. **Source Only** - Target appears only in source/paraphrased materials
2. **Explanations Only** - Target appears only in multi-view explanations  
3. **Both** - Target appears in both source and explanations
4. **Neither** - Target appears in neither source nor explanations

## How Origins Are Determined

### Script: `scripts/FT/analyze_probe_origin.py`

This script analyzes each probe's target text to determine its origin:

```bash
python scripts/FT/analyze_probe_origin.py --probe_type knowledge
python scripts/FT/analyze_probe_origin.py --probe_type inference
```

**Process:**
1. Loads probe targets from CSV files in `probes/<source>/<domain>/facts/` or `probes/<source>/<domain>/inference/`
2. Loads text from:
   - **Explanations**: All `.txt` files in `data/arxiv/explanations/{domain}/`
   - **Source**: Paraphrased `.tex` files + cleaned source `.tex` file
3. Searches for each probe target in both text sources
4. Categorizes based on presence:
   ```python
   if in_expl and not in_src: → "Explanations Only"
   elif in_src and not in_expl: → "Source Only"  
   elif in_expl and in_src: → "Both"
   else: → "Neither"
   ```

**Output:**
- Creates `filter.json` files in each domain directory with probe indices for each category
- Generates a summary report at `results/{probe_type}_probe_origin_analysis.txt`

## Using Origin Filters in Plotting

### Split Probes View (`plot_comparison.py`)

Use `--split_probes` to create separate plots for each origin category:

```bash
python scripts/plotting/plot_comparison.py --model_id 1b --split_probes
```

This creates a 2×5 grid showing:
- Column 1: All probes aggregated
- Column 2: Explanations Only probes
- Column 3: Source Only probes
- Column 4: Both probes
- Column 5: Neither probes

### Delta Analysis (`plot_probe_deltas.py`)

Use `--source_only` to analyze only probes that appear in source material:

```bash
python scripts/plotting/plot_probe_deltas.py --source_only
```

This filters to **only** probes with origin = 'Source Only' before computing deltas.

#### Detailed Performance Reports

The script generates comprehensive reports with two sections:

1. **Statistical Outliers**: Probes beyond Q3 + 1.5 * IQR threshold
2. **Top Percentile**: All probes in the top N% ranked by delta

Control the percentile with `--top_percentile`:

```bash
# Generate report with top 25% of probes (default)
python scripts/plotting/plot_probe_deltas.py

# Generate report with top 10% of probes
python scripts/plotting/plot_probe_deltas.py --top_percentile 0.10

# Combine with source_only filter
python scripts/plotting/plot_probe_deltas.py --source_only --top_percentile 0.30
```

Reports include:
- Summary statistics (mean, median, IQR, etc.)
- Full probe text and target for each ranked probe
- Domain and probe index for easy lookup

## Filter.json Format

Each domain has a `filter.json` file:

```json
{
  "in_explanations_only": [1, 5, 9, ...],
  "in_source_only": [2, 4, 8, ...],
  "in_both": [0, 3, 6, ...],
  "in_neither": [7, 10, ...]
}
```

**Backwards Compatibility:**
- Old filter.json files without `in_both` and `in_neither` fields default unmatched probes to 'Both'
- Re-run `analyze_probe_origin.py` to regenerate with all four categories

## Why "Neither" Matters

Probes in the "Neither" category could indicate:
- Probes generated from external knowledge
- Probes created from intermediate processing steps
- Targets that were slightly modified and no longer match source text exactly
- Potential data quality issues

These probes can confound analysis if lumped together with "Both" probes, which is why the four-way split is important.
