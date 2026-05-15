# E63: Granular Explanation Analysis

## Overview

This experiment implements a new granular analysis framework for studying the impact of individual explanation documents (blogs and stack exchange posts) on model performance.

## Key Features

### 1. Tail Distribution Strategy
- Explanations are distributed across the **last N paraphrase batches**
- Each explanation file maps to a specific document-type slot
- Files are loaded from subfolders (e.g., `DPO/blogs/`, `DPO/stackexchange/`)
- Explanations are appended to paraphrase content, then filled with pretraining data

### 2. Command-Line Arguments

- `--granular_explanations_cycle N`: Load first N files and cycle them across document types
- `--with_specific_explanation {blogs,stackexchange}`: Type of explanation

### 3. Experiment Structure

#### Stack Exchange Experiments
- **8 posts**: Load first 8 stack exchange files → distribute across last 8 paraphrases
- **9 posts**: Load first 9 stack exchange files → distribute across last 9 paraphrases
- **15 posts**: Load first 15 stack exchange files (but only use first 9 due to slot limit)

#### Blogs Experiments
- **3 posts**: Load first 3 blog files → distribute across last 3 paraphrases
- **6 posts**: Load first 6 blog files → distribute across last 6 paraphrases
- **9 posts**: Load first 9 blog files → distribute across last 9 paraphrases

## Data Structure

The implementation expects the following directory structure:

```
data/arxiv/explanations/{DOMAIN}/
├── blogs/
│   ├── blog_01.txt
│   ├── blog_02.txt
│   └── ...
└── stackexchange/
    ├── stack_01.txt
    ├── stack_02.txt
    └── ...
```

## Usage

### Full Experiments (7B model, 200 epochs)
```bash
cd scripts/FT
./E63_7B_granular_explanations.sh
```

### Test Run (1B model, 1 epoch)
```bash
cd scripts/FT
./E63_7B_granular_explanations_test.sh
```

### Manual Invocation Example
```bash
python finetuning_knowledge_v9.py \
    --model_id allenai/OLMo-2-1124-7B \
    --num_train_epochs 200 \
    --learning_rate 1e-5 \
    --effective_batch_size_for_cpt 64 \
    --device_batch_size 1 \
    --context_length_for_cpt 3072 \
    --num_paraphrased_texts 9 \
    --override_domains DPO \
    --with_specific_explanation stackexchange \
    --granular_explanations_cycle 8 \
    --full_finetuning \
    --constant_lr \
    --fill_batches_with_pretraining \
    --custom_suffix "stack_8posts"
```

## Technical Details

### Batch Composition

For a configuration with 9 paraphrases and 8 stack exchange posts:

1. **Paraphrase batches 0-1**: Pure paraphrase content + pretraining fill
2. **Paraphrase batches 2-9**: Paraphrase content + 1 stack exchange post + pretraining fill
   - Batch 2: Para2 chunks + Stack01 chunks + Pretraining
   - Batch 3: Para3 chunks + Stack02 chunks + Pretraining
   - ...
   - Batch 9: Para9 chunks + Stack08 chunks + Pretraining

### File Assignment Logic

- Files are sorted alphabetically for deterministic assignment
- **1-to-1 mapping**: File i → Slot (start_slot + i)
- If `explanation_tail_docs=N`, loads first N files from subfolder
- Example: `--explanation_tail_docs 8` with 9 paraphrases:
  - `stack_01.txt` → Paraphrase batch 2 (index 2)
  - `stack_02.txt` → Paraphrase batch 3 (index 3)
  - ...
  - `stack_08.txt` → Paraphrase batch 9 (index 9)

### Experiment Naming

The experiment name automatically includes:
- `_n{N}`: Number of explanation files used
- `_tail{N}`: Number of tail document types

Example: `para9_expl_stackexchange_n8_tail8`

## Implementation Files

- **data_preparation.py**: Core loading logic with granular analysis support
- **finetuning_knowledge_v9.py**: Training script with new arguments
- **E63_7B_granular_explanations.sh**: Full experiment suite
- **E63_7B_granular_explanations_test.sh**: Quick test version

## Notes

- No pretraining separators are used (`--separate_batches_with_pretraining 0`)
- Effective batch size is 64 for all experiments
- Each explanation chunk is treated as independent content appended to its assigned document-type batch
- The `--fill_batches_with_pretraining` flag ensures all batches reach the target size
