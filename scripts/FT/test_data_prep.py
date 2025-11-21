#!/usr/bin/env python3
"""Test data_preparation.py with realistic example"""

import sys
import logging
sys.path.append('../..')

from transformers import AutoTokenizer
from utils.llm_configs import TrainingConfig
from utils.data_preparation import prepare_training_mix

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)

# Load tokenizer
log.info("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-1124-7B")

# Training config
train_cfg = TrainingConfig(
    run_name="test",
    num_train_epochs=10,
    learning_rate=1e-5,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=8,  # effective batch size = 64
    context_length=3072,
)

# Strategy args - test with blogs + stackexchange
strategy_args = {
    "num_paraphrased_texts": 9,
    "override_domains": ["DPO", "GRPO", "BOFT", "OFT", "QLoRA", "1_58"],
    "fill_batches_with_pretraining": True,
    "separate_batches_with_pretraining": 0,
    "pretraining_data_type": "dclm",
    "test_script": False,
    "with_specific_explanation": ["blogs", "stackexchange", "textbooks"],
    "times_explanations": 1,
    "semi_cleaned": None,
    "use_raw": False,
    "explanation_every_round": False,
    "shuffle_chunks": False,
    "shuffle_seed": 42,
    "explanations_cycle": "full",  # Use 3 files from each type
    "double_cycle": True,
    "granular_explanation_analysis": True,
}

log.info("Preparing training mix...")
dataset, updated_cfg = prepare_training_mix(
    strategy_name="ParaphrasedArxivPaperWithExplanations",
    tokenizer=tokenizer,
    log=log,
    train_cfg=train_cfg,
    chunk_by_section=False,
    overlap_sections=True,
    overlap_ratio="1_4",
    add_title_prefix=True,
    **strategy_args,
)

log.info(f"Dataset created with {len(dataset)} total chunks")

# Calculate batches
effective_batch_size = 64
num_batches = len(dataset) // effective_batch_size
log.info(f"Total batches: {num_batches}")

# Write output
output_file = "../../results/tests/data_prep_test_output.txt"
import os
os.makedirs(os.path.dirname(output_file), exist_ok=True)

with open(output_file, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("DATA PREPARATION TEST OUTPUT\n")
    f.write("=" * 80 + "\n")
    f.write(f"Config: 80 epochs, 9 paraphrases, DPO domain\n")
    f.write(f"Explanations: blogs + stackexchange (3 files each)\n")
    f.write(f"Effective batch size: 64\n")
    f.write(f"Total chunks: {len(dataset)}\n")
    f.write(f"Total batches: {num_batches}\n")
    f.write("=" * 80 + "\n\n")
    
    # Show first 3 batches in detail
    for batch_idx in range(num_batches):
        f.write(f"\n{'=' * 80}\n")
        f.write(f"BATCH {batch_idx}\n")
        f.write(f"{'=' * 80}\n\n")
        
        start_idx = batch_idx * effective_batch_size
        end_idx = start_idx + effective_batch_size
        
        for i, chunk_idx in enumerate(range(start_idx, min(end_idx, len(dataset)))):
            chunk_text = dataset[chunk_idx]['raw_text']
            # Show first 100 characters
            preview = chunk_text[:100].replace('\n', '\\n')
            word_count = len(chunk_text.split())
            f.write(f"Chunk {i:2d}: {preview}... (words: {word_count})\n")
        
        f.write("\n")

log.info(f"Output written to: {output_file}")
log.info("Test complete!")
