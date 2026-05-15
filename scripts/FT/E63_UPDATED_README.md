# E63: Explanations Cycle - Complete Guide

## Summary of Changes

### 1. ✅ "Full" Mode for `granular_explanations_cycle`
Load **ALL** available files from subfolders dynamically (varies per domain).

```bash
--granular_explanations_cycle full
```

### 2. ✅ Granular Analysis with Textbooks
Correctly loads from any subfolder structure, including:
- `DPO/stackexchange/stack_01.txt`
- `1_58/textbooks/chapter_9.txt`
- `BOFT/blogs/blog_05.txt`

### 3. ✅ Removed `explanation_tail_docs`
The old tail strategy is **completely removed**. Use `granular_explanations_cycle` instead.

### 4. ✅ Multiple Explanation Types
Stack multiple cycles from different subfolders:

```bash
--with_specific_explanation blogs stackexchange textbooks
```

## Command-Line Arguments

### `--granular_explanations_cycle {N|"full"}`
- **Integer N**: Load first N files and cycle through them
- **"full"**: Load ALL available files from the subfolder(s)
- Example: `--granular_explanations_cycle 3` or `--granular_explanations_cycle full`

### `--with_specific_explanation TYPE [TYPE ...]`
- Can specify **one or multiple** explanation types
- Examples:
  - Single: `--with_specific_explanation stackexchange`
  - Multiple: `--with_specific_explanation blogs stackexchange`
- **Requirement**: When using multiple types, **must** also set `--granular_explanations_cycle`

## Usage Examples

### Example 1: Fixed Number of Files
```bash
python finetuning_knowledge_v9.py \
    --num_paraphrased_texts 19 \
    --override_domains DPO \
    --with_specific_explanation stackexchange \
    --granular_explanations_cycle 3 \
    --fill_batches_with_pretraining
```

**Result**: Loads `stack_01.txt`, `stack_02.txt`, `stack_03.txt` and cycles through them.

### Example 2: All Available Files (Dynamic)
```bash
python finetuning_knowledge_v9.py \
    --num_paraphrased_texts 19 \
    --override_domains DPO 1_58 BOFT \
    --with_specific_explanation textbooks \
    --granular_explanations_cycle full \
    --fill_batches_with_pretraining
```

**Result**:
- Domain `DPO`: Loads ALL files from `DPO/textbooks/` (e.g., 10 chapters)
- Domain `1_58`: Loads ALL files from `1_58/textbooks/` (e.g., 10 chapters)
- Domain `BOFT`: Loads ALL files from `BOFT/textbooks/` (e.g., 8 chapters)
- Each domain gets its own set of files, cycling independently

### Example 3: Multiple Explanation Types (Stacked Cycles)
```bash
python finetuning_knowledge_v9.py \
    --num_paraphrased_texts 19 \
    --override_domains DPO \
    --with_specific_explanation blogs stackexchange \
    --granular_explanations_cycle 6 \
    --fill_batches_with_pretraining
```

**Result**: 
- Loads first 6 files **total** across both subfolders:
  - `blogs/blog_01.txt`, `blogs/blog_02.txt`, ..., `stackexchange/stack_01.txt`, etc.
- All files sorted together alphabetically, then first 6 are used

### Example 4: All Files from Multiple Types
```bash
python finetuning_knowledge_v9.py \
    --num_paraphrased_texts 19 \
    --override_domains DPO 1_58 \
    --with_specific_explanation blogs stackexchange textbooks \
    --granular_explanations_cycle full \
    --fill_batches_with_pretraining
```

**Result**: Loads **ALL** files from all three subfolders for each domain and cycles through the entire pool.

## How Cycling Works

### With 3 Explanation Files and 20 Document Types

```
Batch 0:  Source + Expl1 + pretraining
Batch 1:  Para1 + Expl2 + pretraining
Batch 2:  Para2 + Expl3 + pretraining
Batch 3:  Para3 + Expl1 + pretraining  ← cycles back
Batch 4:  Para4 + Expl2 + pretraining
Batch 5:  Para5 + Expl3 + pretraining
...
Batch 19: Para19 + Expl2 + pretraining
```

Assignment: `explanation_idx = doc_type_idx % num_explanation_files`

### With "full" Mode

The number of explanation files varies per domain:
- Domain DPO: 15 stack exchange files → cycles every 15 batches
- Domain 1_58: 24 stack exchange files → cycles every 24 batches
- Each domain uses its own set of files

## Directory Structure

The implementation works with this structure:

```
data/arxiv/explanations/
├── DPO/
│   ├── blogs/
│   │   ├── blog_01.txt
│   │   ├── blog_02.txt
│   │   └── ...
│   ├── stackexchange/
│   │   ├── stack_01.txt
│   │   ├── stack_02.txt
│   │   └── ...
│   └── textbooks/
│       └── ... (if available)
├── 1_58/
│   ├── blogs/
│   ├── stackexchange/
│   └── textbooks/
│       ├── chapter_1.txt
│       ├── chapter_2.txt
│       └── ...
└── BOFT/
    ├── blogs/
    ├── stackexchange/
    └── textbooks/
        └── ...
```

## Experiment Naming

The experiment name automatically includes:
- Single type: `para19_expl_stackexchange_cycle3`
- Multiple types: `para19_expl_blogs+stackexchange_cycle6`
- Full mode: `para19_expl_textbooks_cyclefull`

## Validation Rules

1. **Multiple explanation types require cycling**:
   ```bash
   # ❌ ERROR
   --with_specific_explanation blogs stackexchange
   
   # ✅ CORRECT
   --with_specific_explanation blogs stackexchange --granular_explanations_cycle 6
   ```

2. **granular_explanations_cycle must be valid**:
   ```bash
   # ✅ Valid values
   --granular_explanations_cycle 3
   --granular_explanations_cycle full
   
   # ❌ Invalid
   --granular_explanations_cycle auto  # Error: must be integer or "full"
   ```

## Technical Implementation

### File Loading (data_preparation.py)

1. **Determine subfolder(s)**: Based on `--with_specific_explanation`
2. **Load files**:
   - If `granular_explanations_cycle == "full"`: Load ALL .txt files from subfolder(s)
   - If `granular_explanations_cycle == N`: Load first N .txt files across all subfolders
3. **Sort**: Files are sorted alphabetically for deterministic order
4. **Chunk**: Each file is chunked independently
5. **Distribute**: Cycle through chunks using modulo operation

### Per-Domain Independence

Each domain loads its own set of explanation files:
- Domain A with 10 textbook chapters cycles every 10 batches
- Domain B with 15 textbook chapters cycles every 15 batches
- Files are loaded and assigned per-domain during the domain loop

### Multiple Subfolders

When using multiple types (e.g., `blogs stackexchange`):
1. Collect all files from `blogs/` subfolder
2. Collect all files from `stackexchange/` subfolder
3. Sort the combined list
4. Take first N (or all if "full")
5. Cycle through the combined pool

## Migration from Old Code

### Old: `explanation_tail_docs`
```bash
# DEPRECATED - REMOVED
--explanation_tail_docs 8
```

### New: `granular_explanations_cycle`
```bash
# Use this instead
--granular_explanations_cycle 8
```

**Key difference**: Old approach left gaps (only last N batches). New approach cycles through ALL batches with no gaps.

## Tips

1. **Use "full" for maximum coverage**: When you want all available explanations
2. **Use fixed N for controlled ratio**: When you want a specific explanation-to-document ratio
3. **Stack multiple types**: Combine different explanation styles (blogs + technical posts)
4. **Per-domain variation**: "full" mode automatically adapts to each domain's available files

## Scripts

- **`E63_7B_granular_cycle_slurm.sh`**: SLURM script for cluster
- **`E63_examples.sh`**: Various usage examples
- **`E63_UPDATED_README.md`**: This file
