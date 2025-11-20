# E63: Explanations Cycle Strategy

## Overview

The new **explanations cycle** strategy distributes explanation documents across **ALL** document batches with **no gaps**, cycling through the available explanation files.

## Key Difference from Tail Strategy

### Old: `explanation_tail_docs` (DEPRECATED)
- Explanations only on **last N** document types
- **Gaps** in earlier batches
- Example with 3 explanations, 10 doc types:
  ```
  Batch 0-6: No explanations
  Batch 7: Expl 1
  Batch 8: Expl 2
  Batch 9: Expl 3
  ```

### New: `explanations_cycle` (RECOMMENDED)
- Explanations on **EVERY** document type
- **No gaps** - continuous cycling
- Example with 3 explanations, 20 doc types (source + 19 paraphrases):
  ```
  Batch 0: Source + Expl 1 + pretraining
  Batch 1: Para 1 + Expl 2 + pretraining
  Batch 2: Para 2 + Expl 3 + pretraining
  Batch 3: Para 3 + Expl 1 + pretraining  (cycle repeats)
  Batch 4: Para 4 + Expl 2 + pretraining
  ...
  Batch 19: Para 19 + Expl 2 + pretraining
  ```

## How It Works

1. **Load N explanation files** from subfolder (e.g., `stack_01.txt`, `stack_02.txt`, `stack_03.txt`)
2. **Cycle assignment**: `explanation_idx = doc_type_idx % N`
3. **Append to batch**: Each document batch gets its corresponding explanation chunks
4. **Fill with pretraining**: Each batch is then filled to the effective batch size

## New Command-Line Argument

```bash
--explanations_cycle N
```

Where `N` is the number of explanation files to load and cycle through.

## Example Usage

### Manual Invocation
```bash
python finetuning_knowledge_v8.py \
    --model_id allenai/OLMo-2-1124-7B \
    --num_train_epochs 100 \
    --learning_rate 2e-5 \
    --effective_batch_size_for_cpt 64 \
    --device_batch_size 32 \
    --context_length_for_cpt 3072 \
    --num_paraphrased_texts 19 \
    --override_domains DPO \
    --with_specific_explanation stackexchange \
    --explanations_cycle 3 \
    --full_finetuning \
    --fill_batches_with_pretraining \
    --granular_explanation_analysis \
    --attn_implementation sdpa \
    --gradient_checkpointing \
    --compile_model
```

### SLURM Script
```bash
sbatch scripts/FT/E63_7B_granular_cycle_slurm.sh
```

The SLURM script runs 4 experiments:
- 3 explanation files cycling
- 6 explanation files cycling
- 9 explanation files cycling
- 12 explanation files cycling

## Batch Structure Example

With `num_paraphrased_texts=19` (20 total doc types) and `explanations_cycle=3`:

```
Exposure 1:
  Batch 0:  [Source from all domains] + [stack_01.txt from all domains] + [Pretraining fill to 64]
  Batch 1:  [Para 1 from all domains] + [stack_02.txt from all domains] + [Pretraining fill to 64]
  Batch 2:  [Para 2 from all domains] + [stack_03.txt from all domains] + [Pretraining fill to 64]
  Batch 3:  [Para 3 from all domains] + [stack_01.txt from all domains] + [Pretraining fill to 64]  ← cycles back
  Batch 4:  [Para 4 from all domains] + [stack_02.txt from all domains] + [Pretraining fill to 64]
  ...
  Batch 19: [Para 19 from all domains] + [stack_02.txt from all domains] + [Pretraining fill to 64]

Exposure 2 (if num_train_epochs > 20):
  Batch 0:  [Source from all domains] + [stack_01.txt from all domains] + [Pretraining fill to 64]
  ... (same pattern repeats)
```

## Experiment Naming

The experiment name automatically includes:
- `_cycle{N}`: Number of explanation files cycling

Example: `para19_expl_stackexchange_cycle3`

## Benefits of Cycling Strategy

1. **Uniform distribution**: Every document batch gets explanations
2. **Maximum coverage**: More exposure to explanatory content
3. **No gaps**: Consistent training signal throughout
4. **Flexible ratio**: Easy to control explanation-to-document ratio by adjusting N
5. **Simple mental model**: Just specify how many explanation files to use

## Technical Details

### File Loading
- Files are sorted alphabetically for deterministic assignment
- First N files are loaded: `stack_01.txt`, `stack_02.txt`, ..., `stack_0N.txt`
- Each file is chunked independently

### Cycling Logic
```python
for doc_idx in range(num_doc_types):
    expl_idx = doc_idx % num_explanation_files
    batch[doc_idx].extend(explanation_chunks[expl_idx])
```

### Compatibility
- Works with `--granular_explanation_analysis` for subfolder loading
- Works with `--fill_batches_with_pretraining` for batch filling
- Compatible with all existing domain and paraphrase settings
- The old `--explanation_tail_docs` is preserved for backwards compatibility but deprecated

