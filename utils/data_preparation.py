import os
import torch
import numpy as np
import random
from typing import List, Optional
from itertools import cycle
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

from utils.llm_configs import TrainingConfig
import utils.chunking as chunking


class PretrainingDataReplay:
    """
    An object that centralizes data replay from a tokenized .npy file.
    It continues from the last place it started to ensure unique coverage of pretraining data.
    """
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = np.load(self.file_path, mmap_mode='r')
        self.position = 0

    def get_tokens(self, num_tokens):
        """
        Retrieves a specified number of tokens from the dataset.
        Raises an error if requesting past the end of the file.
        """
        if self.position + num_tokens > len(self.data):
            # In a real large-scale scenario, you might wrap around. 
            # For now, we raise to ensure we know we ran out.
            raise ValueError(f"Not enough data remaining. Requested {num_tokens} tokens, "
                           f"but only {len(self.data) - self.position} tokens available from position {self.position}")
        
        tokens = self.data[self.position:self.position + num_tokens]
        self.position += num_tokens
        return torch.tensor(tokens, dtype=torch.long)

def get_pretraining_batches(
    data_replay: PretrainingDataReplay,
    num_batches: int,
    batch_size: int,
    chunk_size: int,
    tokenizer,
) -> List[str]:
    """
    Gets a specified number of batches of pretraining data (used as separators).
    """
    total_chunks = num_batches * batch_size
    if total_chunks == 0:
        return []
    total_tokens = total_chunks * chunk_size

    tokens = data_replay.get_tokens(total_tokens)
    token_chunks_tensor = tokens.view(total_chunks, chunk_size)
    text_chunks = [tokenizer.decode(chunk, skip_special_tokens=False) for chunk in token_chunks_tensor]
    
    return text_chunks

def fill_up_batch_with_pretraining_chunks(
    batch: List[str], 
    data_replay: PretrainingDataReplay, 
    batch_size: int, 
    chunk_size: int,
    tokenizer,
) -> List[str]:
    """
    Fills up a batch list with pretraining chunks if it's smaller than the effective batch size.
    """
    num_chunks_needed = batch_size - len(batch)
    if num_chunks_needed <= 0:
        return batch

    num_tokens_needed = num_chunks_needed * chunk_size
    tokens = data_replay.get_tokens(num_tokens_needed)
    token_chunks_tensor = tokens.view(num_chunks_needed, chunk_size)
    new_text_chunks = [tokenizer.decode(chunk, skip_special_tokens=False) for chunk in token_chunks_tensor]
    
    batch.extend(new_text_chunks)
    return batch

def fill_underfilled_chunks(
    chunks: List[str],
    data_replay: PretrainingDataReplay,
    tokenizer,
    context_length: int,
    threshold: int = 1000,
    log=None
) -> List[str]:
    """
    Iterates through all chunks. If a chunk is significantly shorter than the context length
    (determined by threshold), it appends an EOS token and fills the rest with pretraining data.
    It fills up to (context_length - 1) to be safe.
    """
    if log:
        log.info(f"Checking for chunks with > {threshold} empty tokens to fill with pretraining data...")
    
    filled_count = 0
    # Target length is context_length - 1 (safe margin)
    target_length = context_length - 1
    
    for i in range(len(chunks)):
        text = chunks[i]
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        current_len = len(token_ids)
        
        # Check if we have enough empty space
        if current_len < (target_length - threshold):
            # We need: [Original] + [EOS] + [Fill] = Target
            # Fill = Target - Original - 1 (for EOS)
            num_fill_tokens = target_length - current_len - 1
            
            if num_fill_tokens > 0:
                fill_tokens = data_replay.get_tokens(num_fill_tokens).tolist()
                new_ids = token_ids + [tokenizer.eos_token_id] + fill_tokens
                
                # Decode back to string
                chunks[i] = tokenizer.decode(new_ids, skip_special_tokens=False)
                filled_count += 1

    if log:
        log.info(f"Filled {filled_count} chunks that were under the token threshold.")
    
    return chunks

def prepare_lima_dataset(tokenizer: AutoTokenizer, log, use_eot_token=False, sort=False, cache_dir=None):
    """
    Standard LIMA dataset preparation.
    """
    log.info("Preparing GAIR/lima dataset...")
    EOT_TOKEN = "<|EOT|>"
    if not use_eot_token:
        EOT_TOKEN = "\nResponse:"
        
    dataset = load_dataset("GAIR/lima", cache_dir=cache_dir)
    train_dataset = dataset["train"].shuffle(seed=42)
    log.info(f"{len(train_dataset)} training examples.")

    def format_lima_conversation(example):
        conversation = example['conversations']
        formatted_text = f"{EOT_TOKEN}".join(conversation) + tokenizer.eos_token
        return {"text": formatted_text}
        
    train_dataset = train_dataset.map(format_lima_conversation, remove_columns=['conversations', 'source'])

    if sort:
        train_dataset = train_dataset.map(lambda x: {"_len": len(x["text"])})
        train_dataset = (
            train_dataset
            .sort("_len")
            .select(list(range(len(train_dataset) - 1, -1, -1)))
            .remove_columns("_len")
        )
        
    return train_dataset

def prepare_training_mix(
    strategy_name: str,
    tokenizer,
    log,
    train_cfg: TrainingConfig,
    chunk_by_section: bool = False,
    overlap_sections: bool = False,
    overlap_ratio: str = "1_4",
    add_title_prefix: bool = True,
    **strategy_args,
):
    """
    Prepares the training mix. 
    Supports:
    1. Legacy mixing (coupled batches).
    2. Track mixing (independent streams for docs and explanations).
    3. Pretraining data injection (separators, batch filling, chunk gap filling).
    """
    log.info(f"Preparing training mix for strategy: {strategy_name}")

    # --- Arguments Extraction ---
    test_script = strategy_args.get("test_script", False)
    specific_explanation_type = strategy_args.get("with_specific_explanation", None)
    times_explanations = strategy_args.get("times_explanations", 1)
    semi_cleaned_version = strategy_args.get("semi_cleaned", None)
    use_raw = strategy_args.get("use_raw", False)
    explanation_every_round = strategy_args.get("explanation_every_round", False)
    shuffle_chunks_flag = strategy_args.get("shuffle_chunks", False)
    shuffle_seed = strategy_args.get("shuffle_seed", 42)
    with_human = strategy_args.get("with_human", False)
    explanation_in_own_batch = strategy_args.get("explanation_in_own_batch", False) # Deprecated essentially
    explanations_cycle = strategy_args.get("explanations_cycle", 0)
    double_cycle = strategy_args.get("double_cycle", False)
    granular_explanation_analysis = strategy_args.get("granular_explanation_analysis", False)
    
    # New args for pretraining injection
    fill_chunk_gaps = strategy_args.get("fill_chunk_gaps", False)
    chunk_gap_threshold = strategy_args.get("fill_chunk_gap_threshold", 1000)
    pretraining_separators = int(strategy_args.get("separate_batches_with_pretraining", 0))
    fill_with_pretraining = strategy_args.get("fill_batches_with_pretraining", False)

    # --- Helper Functions ---
    def _chunk(text: str) -> List[str]:
        text_with_eos = text + tokenizer.eos_token
        chunks, _ = chunking.chunk(
            text_with_eos, tokenizer, train_cfg.context_length,
            chunk_by_section, overlap_sections, overlap_ratio, add_title_prefix, log=log
        )
        return chunks

    def _chunk_explanation(text: str) -> List[str]:
        text_with_eos = text + tokenizer.eos_token
        explanation_overlap_ratio = "1_10"
        chunks, _ = chunking.chunk(
            text_with_eos, tokenizer, train_cfg.context_length,
            chunk_by_section, overlap_sections, explanation_overlap_ratio, add_title_prefix, log=log
        )
        return chunks

    # --- 1. Load Domains & Documents (The "Doc Track") ---
    domains = strategy_args.get("override_domains", None)
    if domains is None:
        if use_raw:
            cleaned_dir = '../../data/arxiv/raw'
        elif semi_cleaned_version:
            cleaned_dir = f'../../data/arxiv/semicleaned_{semi_cleaned_version}'
        else:
            cleaned_dir = '../../data/arxiv/cleaned'
        domains = [f.replace('.tex', '') for f in os.listdir(cleaned_dir) if f.endswith('.tex')]
    
    log.info(f"Processing domains: {domains}")
    
    num_paraphrased_texts = strategy_args.get("num_paraphrased_texts", 0)
    with_explanations = "WithExplanations" in strategy_name
    
    # List of batches for source documents (e.g., [ [SourceChunks], [Para1Chunks], [Para2Chunks] ])
    unique_document_batches = [[] for _ in range(1 + num_paraphrased_texts)]
    
    # Placeholder for legacy coupled batches
    unique_document_batches_with_explanations = None
    
    # List of explanation batches for the New Track Method
    explanation_pool = []

    for domain in domains:
        log.info(f"Loading data for domain: {domain}")
        
        # Load Source
        if use_raw: source_path = f'../../data/arxiv/raw/{domain}.tex'
        elif semi_cleaned_version: source_path = f'../../data/arxiv/semicleaned_{semi_cleaned_version}/{domain}.tex'
        else: source_path = f'../../data/arxiv/cleaned/{domain}.tex'

        try:
            with open(source_path, 'r', encoding='utf-8') as f: source_text = f.read()
        except FileNotFoundError: continue
        
        source_chunks = _chunk(source_text)

        # Load Paraphrases
        paraphrased_chunks_by_doc = []
        if num_paraphrased_texts > 0:
            paraphrased_dir = f'../../data/arxiv/paraphrased/{domain}/'
            if os.path.isdir(paraphrased_dir):
                for i in range(num_paraphrased_texts):
                    para_path = os.path.join(paraphrased_dir, f'{i}.tex')
                    if os.path.exists(para_path):
                        with open(para_path, 'r', encoding='utf-8') as f:
                            paraphrased_chunks_by_doc.append(_chunk(f.read()))
        
        # Add to main batch list
        domain_doc_chunks = [source_chunks] + paraphrased_chunks_by_doc
        for i, chunks in enumerate(domain_doc_chunks):
            if i < len(unique_document_batches):
                unique_document_batches[i].extend(chunks)

        # --- 2. Load Explanations (Preparation) ---
        if with_explanations:
            explanation_dir = f'../../data/arxiv/explanations/{domain}/'
            files_to_load = {} # Dict {type: [filenames]}

            # Identify files to load
            if granular_explanation_analysis and specific_explanation_type:
                explanation_types = specific_explanation_type if isinstance(specific_explanation_type, list) else [specific_explanation_type]
                for expl_type in explanation_types:
                    subfolder_path = os.path.join(explanation_dir, expl_type)
                    if os.path.isdir(subfolder_path):
                        all_files = sorted([f for f in os.listdir(subfolder_path) if f.endswith('.txt')])
                        
                        if explanations_cycle == "full":
                            type_files = [os.path.join(expl_type, f) for f in all_files]
                        elif isinstance(explanations_cycle, int) and explanations_cycle > 0:
                            type_files = [os.path.join(expl_type, f) for f in all_files[:explanations_cycle]]
                        else:
                            type_files = []
                        files_to_load[expl_type] = type_files
            else:
                # Legacy/Simple loading
                default_files = [f"{specific_explanation_type}.txt"] if specific_explanation_type else ['blogs.txt', 'stackexchange.txt', 'textbook.txt']
                human_files = [f"human_blog_{i}.txt" for i in range(1, 4)] if with_human else []
                
                if os.path.isdir(explanation_dir):
                    avail = set(os.listdir(explanation_dir))
                    if human_files and any(f in avail for f in human_files):
                        actual_files = [f for f in human_files if f in avail]
                    elif not with_human:
                        actual_files = [f for f in default_files if f in avail]
                    else: actual_files = []
                    actual_files.sort()
                    
                    # For legacy reasons, we treat "default" as a single type
                    files_to_load["default"] = actual_files

            # --- 3. Build Explanation Pools (Track Method vs Legacy) ---
            
            # Check if we use the New Track Method (Triggered by specific cycling args or granular analysis)
            use_track_method = granular_explanation_analysis or (explanations_cycle == "full" or (isinstance(explanations_cycle, int) and explanations_cycle > 0))

            if use_track_method and files_to_load:
                # >> NEW TRACK METHOD CONSTRUCTION <<
                type_cycles = {}
                max_cycle_len = 0
                
                for expl_type, file_list in files_to_load.items():
                    type_file_chunks = []
                    for filename in file_list:
                        file_path = os.path.join(explanation_dir, filename)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            file_chunks = _chunk_explanation(f.read())
                        
                        if times_explanations > 1 and file_chunks:
                            file_chunks = file_chunks * times_explanations
                        
                        type_file_chunks.append((filename, file_chunks))
                    
                    # Handle double cycle (offset)
                    if double_cycle and type_file_chunks:
                        offset = len(type_file_chunks) // 2
                        second_cycle = []
                        for i in range(len(type_file_chunks)):
                            offset_idx = (i + offset) % len(type_file_chunks)
                            fn, ch = type_file_chunks[offset_idx]
                            second_cycle.append((f"{fn}_offset", ch))
                        type_file_chunks.extend(second_cycle)
                    
                    type_cycles[expl_type] = type_file_chunks
                    max_cycle_len = max(max_cycle_len, len(type_file_chunks))

                # Create flat pool for this domain
                for expl_idx in range(max_cycle_len):
                    chunks_for_this_step = []
                    for expl_type, file_chunks_list in type_cycles.items():
                        if not file_chunks_list: continue
                        _, file_chunks = file_chunks_list[expl_idx % len(file_chunks_list)]
                        chunks_for_this_step.extend(file_chunks)
                    
                    if chunks_for_this_step:
                        explanation_pool.append(chunks_for_this_step)

            elif not use_track_method:
                # >> LEGACY METHOD CONSTRUCTION (Coupled Batches) <<
                # We load explanations into a list and splice them into the document lists
                legacy_expl_chunks = []
                if "default" in files_to_load:
                    for filename in files_to_load["default"]:
                        with open(os.path.join(explanation_dir, filename), 'r', encoding='utf-8') as f:
                            legacy_expl_chunks.extend(_chunk_explanation(f.read()))
                
                if times_explanations > 1 and legacy_expl_chunks:
                    legacy_expl_chunks = legacy_expl_chunks * times_explanations

                # Init legacy container if needed
                if unique_document_batches_with_explanations is None:
                    unique_document_batches_with_explanations = [[] for _ in range(len(unique_document_batches))]
                
                # Add Source (Batch 0)
                if len(unique_document_batches_with_explanations) > 0:
                    unique_document_batches_with_explanations[0].extend(source_chunks)

                # Splice logic: Explanations replace the end of Paraphrased chunks
                if legacy_expl_chunks:
                    to_insert = list(legacy_expl_chunks)
                    # Iterate paraphrases (Indices 1+)
                    for i in range(len(unique_document_batches) - 1, 0, -1): # backwards
                        original_chunks = unique_document_batches[i][-len(paraphrased_chunks_by_doc[i-1]):] # get current doc chunks
                        
                        if not to_insert:
                            unique_document_batches_with_explanations[i].extend(original_chunks)
                            continue

                        if len(to_insert) >= len(original_chunks):
                            unique_document_batches_with_explanations[i].extend(to_insert[-len(original_chunks):])
                            to_insert = to_insert[:-len(original_chunks)]
                        else:
                            # Partial replace
                            keep_n = len(original_chunks) - len(to_insert)
                            new_c = original_chunks[:keep_n] + to_insert
                            unique_document_batches_with_explanations[i].extend(new_c)
                            to_insert = []
                else:
                    # Fallback if no explanations found but requested
                    for i in range(1, len(unique_document_batches)):
                        unique_document_batches_with_explanations[i].extend(unique_document_batches[i][-len(paraphrased_chunks_by_doc[i-1]):])

    # --- 4. Prepare Data Replayer ---
    # We need this if we are doing separators, filling batches, OR filling chunk gaps
    data_replay = None
    if fill_with_pretraining or pretraining_separators > 0 or fill_chunk_gaps:
        pretraining_data_type = strategy_args.get('pretraining_data_type', 'dclm')
        log.info(f"Creating Pretraining Data Replayer from '{pretraining_data_type}'...")
        data_replay = PretrainingDataReplay(f'../../data/olmo/{pretraining_data_type}_100M_tokens.npy')

    # --- 5. Calculate Replications ---
    original_epochs = train_cfg.num_train_epochs
    replication_factor = 1
    if unique_document_batches and original_epochs > 1:
        replication_factor = int(original_epochs / len(unique_document_batches))
        log.info(f"Replicating dataset {replication_factor} times.")

    # --- 6. Generate Final Chunks (Routing) ---
    final_chunks = []

    if explanation_pool:
        log.info("Using NEW Track Method (Independent Doc and Explanation Tracks).")
        final_chunks = replicate_and_interleave_tracks(
            doc_batches=unique_document_batches,
            explanation_pool=explanation_pool,
            replication_factor=replication_factor,
            data_replay=data_replay,
            pretraining_separators=pretraining_separators,
            train_cfg=train_cfg,
            tokenizer=tokenizer,
            log=log,
            test_script=test_script,
            fill_with_pretraining=fill_with_pretraining
        )
    else:
        log.info("Using LEGACY Method (Coupled Batches).")
        final_chunks = replicate_and_interleave_legacy(
            unique_document_batches=unique_document_batches,
            unique_document_batches_with_explanations=unique_document_batches_with_explanations,
            replication_factor=replication_factor,
            data_replay=data_replay,
            pretraining_separators=pretraining_separators,
            train_cfg=train_cfg,
            tokenizer=tokenizer,
            log=log,
            test_script=test_script,
            fill_with_pretraining=fill_with_pretraining,
            use_explanations_every_round=explanation_every_round
        )

    # --- 7. Fill Chunk Gaps (Optional) ---
    if fill_chunk_gaps and final_chunks:
        log.info("Filling gaps in individual chunks...")
        final_chunks = fill_underfilled_chunks(
            chunks=final_chunks,
            data_replay=data_replay,
            tokenizer=tokenizer,
            context_length=train_cfg.context_length,
            threshold=chunk_gap_threshold,
            log=log
        )

    # --- 8. Shuffle ---
    if shuffle_chunks_flag and final_chunks:
        random.Random(shuffle_seed).shuffle(final_chunks)
        log.info(f"Shuffled final training chunks with seed {shuffle_seed}.")

    train_cfg.num_train_epochs = 1
    dataset = Dataset.from_dict({"raw_text": final_chunks})
    return dataset, train_cfg

def replicate_and_interleave_tracks(
    doc_batches: List[List[str]],
    explanation_pool: List[List[str]],
    replication_factor: int,
    data_replay: PretrainingDataReplay,
    pretraining_separators: int,
    train_cfg: TrainingConfig,
    tokenizer,
    log,
    test_script: bool = False,
    fill_with_pretraining: bool = False,
) -> List[str]:
    """
    The New Way: "Track" based mixing.
    1. Build the full Document Track (length = docs * reps).
    2. Build the full Explanation Track (length = docs * reps) by cycling the pool continuously.
    3. Merge tracks and fill.
    """
    effective_batch_size = train_cfg.per_device_train_batch_size * train_cfg.gradient_accumulation_steps
    total_batches = len(doc_batches) * replication_factor
    
    if total_batches == 0: return []

    # --- TRACK 1: DOCUMENTS ---
    # D1, D2, D3... repeated `replication_factor` times
    track_docs = []
    for _ in range(replication_factor):
        track_docs.extend([list(b) for b in doc_batches])
        
    # --- TRACK 2: EXPLANATIONS ---
    # E1, E2, E3... cycled continuously to match total_batches
    track_explanations = []
    if explanation_pool:
        expl_iterator = cycle(explanation_pool)
        for _ in range(total_batches):
            track_explanations.append(next(expl_iterator))
    else:
        track_explanations = [[] for _ in range(total_batches)]

    # --- MERGE ---
    final_chunks = []
    for i in range(total_batches):
        # Concatenate: Docs + Explanations
        current_batch = track_docs[i] + track_explanations[i]
        
        # Batch Fill
        if fill_with_pretraining:
            current_batch = fill_up_batch_with_pretraining_chunks(
                current_batch, data_replay, effective_batch_size, train_cfg.context_length, tokenizer
            )
        
        final_chunks.extend(current_batch)
        
        # Separator
        if pretraining_separators > 0 and i < total_batches - 1:
             pretraining_fill = get_pretraining_batches(
                data_replay, pretraining_separators, effective_batch_size,
                train_cfg.context_length, tokenizer
            )
             final_chunks.extend(pretraining_fill)

    return final_chunks

def replicate_and_interleave_legacy(
    unique_document_batches: List[List[str]],
    unique_document_batches_with_explanations: Optional[List[List[str]]],
    replication_factor: int,
    data_replay: PretrainingDataReplay,
    pretraining_separators: int,
    train_cfg: TrainingConfig,
    tokenizer,
    log,
    test_script: bool = False,
    fill_with_pretraining: bool = False,
    use_explanations_every_round: bool = False,
) -> List[str]:
    """
    The Old Way: Switching between pre-baked coupled batches.
    """
    final_chunks = []
    effective_batch_size = train_cfg.per_device_train_batch_size * train_cfg.gradient_accumulation_steps

    for rep in range(replication_factor):
        if test_script: log.info(f"--- Creating replication {rep + 1}/{replication_factor} ---")
        
        batches_for_this_rep = unique_document_batches
        
        # Legacy alternation logic
        if unique_document_batches_with_explanations:
            if use_explanations_every_round:
                batches_for_this_rep = unique_document_batches_with_explanations
            elif (rep % 2 == 1): # Odd reps get explanations
                batches_for_this_rep = unique_document_batches_with_explanations

        for i, batch in enumerate(batches_for_this_rep):
            current_batch = list(batch)
            
            if fill_with_pretraining:
                current_batch = fill_up_batch_with_pretraining_chunks(
                    current_batch, data_replay, effective_batch_size, train_cfg.context_length, tokenizer
                )
            
            final_chunks.extend(current_batch)
            
            is_last_batch_of_last_replication = (rep == replication_factor - 1) and (i == len(batches_for_this_rep) - 1)
            if pretraining_separators > 0 and not is_last_batch_of_last_replication:
                pretraining_fill = get_pretraining_batches(
                    data_replay, pretraining_separators, effective_batch_size, 
                    train_cfg.context_length, tokenizer
                )
                final_chunks.extend(pretraining_fill)
                
    return final_chunks