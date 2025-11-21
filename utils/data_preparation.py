import os
import torch
import numpy as np
import random
from typing import List
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
    Gets a specified number of batches of pretraining data, returned as a single list of text chunks.
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
    Fills up a batch with pretraining data if it's not full.
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

def prepare_lima_dataset(tokenizer: AutoTokenizer, log, use_eot_token=False, sort=False, cache_dir=None):
    """
    Loads the GAIR/lima dataset, and formats
    the conversations into a text format suitable for SFTTrainer.
    Args:
        tokenizer: The tokenizer to modify.
        model: The model to resize embeddings for.
    Returns:
        A tuple of (train_dataset, eval_dataset).
    """
    log.info("Preparing GAIR/lima dataset...")
    EOT_TOKEN = "<|EOT|>"
    if not use_eot_token:
        EOT_TOKEN = "\nResponse:"
    # 2. Load the dataset
    dataset = load_dataset("GAIR/lima", cache_dir=cache_dir)
    # The paper uses 1000 for training, 50 for dev. The HF dataset has 1030 train examples.
    # We'll split it accordingly.
    train_dataset = dataset["train"].shuffle(seed=42)
    # train_dataset = full_train_dataset
    log.info(f"{len(train_dataset)} training examples.")

    # 3. Define the formatting function
    def format_lima_conversation(example):
        conversation = example['conversations']
        # Join turns with the EOT token. Add one at the very end.
        formatted_text = f"{EOT_TOKEN}".join(conversation) + tokenizer.eos_token
        return {"text": formatted_text}

        
    # 4. Apply the formatting
    train_dataset = train_dataset.map(format_lima_conversation, remove_columns=['conversations', 'source'])

    if sort:
        # add a temporary 'length' column
        train_dataset = train_dataset.map(
            lambda x: {"_len": len(x["text"])},
            desc="Computing lengths for sort",
        )
        # Dataset.sort always sorts ascending, so we reverse afterwards
        train_dataset = (
            train_dataset
            .sort("_len")                                      # shortest → longest
            .select(list(range(len(train_dataset) - 1, -1, -1)))  # flip order
            .remove_columns("_len")                            # clean-up
        )
        log.info("Training set sorted by descending length.")
        
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
    Prepares a training dataset based on a specified strategy.
    Handles complex mixing of source, paraphrased, and explanation texts
    while controlling for the total number of training steps. This function
    is domain-agnostic and handles single or multiple domains naturally.
    """
    log.info(f"Preparing training mix for strategy: {strategy_name}")
    log.info(f"Chunking parameters: chunk_by_section={chunk_by_section}, overlap_sections={overlap_sections}, overlap_ratio={overlap_ratio}, add_title_prefix={add_title_prefix}")
    

    test_script = strategy_args.get("test_script", False)
    specific_explanation_type = strategy_args.get("with_specific_explanation", None)
    times_explanations = strategy_args.get("times_explanations", 1)
    semi_cleaned_version = strategy_args.get("semi_cleaned", None)
    use_raw = strategy_args.get("use_raw", False)
    explanation_every_round = strategy_args.get("explanation_every_round", False)
    shuffle_chunks_flag = strategy_args.get("shuffle_chunks", False)
    shuffle_seed = strategy_args.get("shuffle_seed", 42)
    with_human = strategy_args.get("with_human", False)
    explanation_in_own_batch = strategy_args.get("explanation_in_own_batch", False)
    # Number of explanation files to cycle through across ALL document batches.
    # E.g., with 3 explanation files and 10 doc types: [Expl1, Expl2, Expl3, Expl1, Expl2, ...]
    # Can be an integer or the string "full" to use all available files
    explanations_cycle = strategy_args.get("explanations_cycle", 0)
    double_cycle = strategy_args.get("double_cycle", False)
    granular_explanation_analysis = strategy_args.get("granular_explanation_analysis", False)

    # Helper to chunk a single text
    def _chunk(text: str) -> List[str]:
        text_with_eos = text + tokenizer.eos_token
        chunks, _ = chunking.chunk(
            text_with_eos,
            tokenizer,
            train_cfg.context_length,
            chunk_by_section,
            overlap_sections,
            overlap_ratio,
            add_title_prefix,
            log=log
        )
        return chunks

    # Helper to chunk explanation text with specific overlap
    def _chunk_explanation(text: str) -> List[str]:
        text_with_eos = text + tokenizer.eos_token
        # User requested hardcoded overlap for explanations
        explanation_overlap_ratio = "1_10"
        
        chunks, _ = chunking.chunk(
            text_with_eos,
            tokenizer,
            train_cfg.context_length,
            chunk_by_section,
            overlap_sections, # Explanations should always be overlapped
            explanation_overlap_ratio,
            add_title_prefix,
            log=log
        )
        return chunks

    # Determine domains: use override if provided, otherwise scan directory
    domains = strategy_args.get("override_domains", None)
    unique_document_batches = []
    unique_document_batches_with_explanations = None
    explanation_batches = []
    
    if "PriorKnowledge" in strategy_name:
        prior_knowledge_dir = '../../data/arxiv/prior_knowledge'
        if domains is None:
            domains = [name for name in os.listdir(prior_knowledge_dir) if os.path.isdir(os.path.join(prior_knowledge_dir, name))]
        log.info(f"Processing prior knowledge for domains: {domains}")
        
        all_prior_knowledge_chunks = []
        for domain in domains:
            domain_dir = os.path.join(prior_knowledge_dir, domain)
            domain_text = ""
            if os.path.isdir(domain_dir):
                log.info(f"Loading prior knowledge from {domain_dir}")
                for filename in sorted(os.listdir(domain_dir)):
                    if filename.endswith('.txt') and 'textbook' in filename:
                        file_path = os.path.join(domain_dir, filename)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            domain_text += f.read()
            
            if domain_text:
                all_prior_knowledge_chunks.extend(_chunk(domain_text))

        unique_document_batches.append(all_prior_knowledge_chunks)
    else:
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
        # with_explanations now denotes any "special" injection path (LLM explanations, human blogs, etc.)
        with_explanations = "WithExplanations" in strategy_name

        # Each inner list holds chunks for a "unique document type" across all domains
        # e.g., unique_document_batches[0] = all source chunks
        #       unique_document_batches[1] = all paraphrase-1 chunks
        base_doc_types = 1 + num_paraphrased_texts
        # Optionally treat explanations as an additional "document type" with its own batch.
        if with_explanations and explanation_in_own_batch:
            num_doc_types = base_doc_types + 1
        else:
            num_doc_types = base_doc_types
        unique_document_batches = [[] for _ in range(num_doc_types)]
        unique_document_batches_with_explanations = (
            [[] for _ in range(num_doc_types)] if (with_explanations and not explanation_in_own_batch) else None
        )


        for domain in domains:
            log.info(f"Loading data for domain: {domain}")
            
            # 1. Load source text
            if use_raw:
                source_path = f'../../data/arxiv/raw/{domain}.tex'
            elif semi_cleaned_version:
                source_path = f'../../data/arxiv/semicleaned_{semi_cleaned_version}/{domain}.tex'
            else:
                source_path = f'../../data/arxiv/cleaned/{domain}.tex'

            try:
                with open(source_path, 'r', encoding='utf-8') as f:
                    source_text = f.read()
            except FileNotFoundError:
                log.error(f"Source file not found for domain {domain} at {source_path}. Skipping.")
                continue
                
            source_chunks = _chunk(source_text)
            

            # 2. Load paraphrased texts
            paraphrased_texts = []
            paraphrased_chunks_by_doc = []
            if num_paraphrased_texts > 0:
                paraphrased_dir = f'../../data/arxiv/paraphrased/{domain}/'
                if os.path.isdir(paraphrased_dir):
                    for i in range(num_paraphrased_texts):
                        para_path = os.path.join(paraphrased_dir, f'{i}.tex')
                        if os.path.exists(para_path):
                            with open(para_path, 'r', encoding='utf-8') as f:
                                paraphrased_texts.append(f.read())
                        else:
                            log.warning(f"Paraphrased text not found: {para_path}")
                paraphrased_chunks_by_doc = [_chunk(text) for text in paraphrased_texts]

            # 3. Load explanation texts
            explanation_chunks = [] # Collects all chunks if not distributing (for legacy/own-batch)
            
            if with_explanations:
                explanation_dir = f'../../data/arxiv/explanations/{domain}/'

                files_to_load = []
                
                # New mode: Granular analysis using subfolders (e.g., blogs/, stackexchange/, textbooks/)
                if granular_explanation_analysis and specific_explanation_type:
                    # specific_explanation_type can be a single string or a list - always convert to list
                    explanation_types = specific_explanation_type if isinstance(specific_explanation_type, list) else [specific_explanation_type]
                    
                    # Always use dict structure - works for single or multiple types
                    files_to_load = {}  # Dict: {type_name: [files]}
                    
                    for expl_type in explanation_types:
                        subfolder_path = os.path.join(explanation_dir, expl_type)
                        if os.path.isdir(subfolder_path):
                            all_files = sorted([f for f in os.listdir(subfolder_path) if f.endswith('.txt')])
                            
                            if explanations_cycle == "full":
                                # Load ALL files from this subfolder
                                type_files = [os.path.join(expl_type, f) for f in all_files]
                                log.info(f"Domain {domain}: Loading ALL {len(type_files)} files from {subfolder_path}.")
                            elif isinstance(explanations_cycle, int) and explanations_cycle > 0:
                                # Load first N files from this subfolder
                                num_to_load = min(explanations_cycle, len(all_files))
                                type_files = [os.path.join(expl_type, f) for f in all_files[:num_to_load]]
                                
                                if num_to_load < explanations_cycle:
                                    log.warning(
                                        f"Domain {domain}: Only {num_to_load} files available in {subfolder_path}, "
                                        f"but requested {explanations_cycle}."
                                    )
                                
                                log.info(f"Domain {domain}: Loading {len(type_files)} files from {subfolder_path}.")
                            else:
                                type_files = []
                            
                            files_to_load[expl_type] = type_files
                        else:
                            log.warning(f"Domain {domain}: Subfolder {subfolder_path} not found for granular analysis.")
                            files_to_load[expl_type] = []
                
                # Legacy mode: Load from root explanation directory
                else:
                    # Default: model-generated explanations
                    default_files = []
                    if specific_explanation_type:
                        default_files.append(f"{specific_explanation_type}.txt")
                    else:
                        default_files = ['blogs.txt', 'stackexchange.txt', 'textbook.txt']

                    # Optional: human-written blogs where available (e.g., GRPO, DPO)
                    human_files = [f"human_blog_{i}.txt" for i in range(1, 4)] if with_human else []

                    if os.path.isdir(explanation_dir):
                        available_files = set(os.listdir(explanation_dir))

                        if human_files and any(f in available_files for f in human_files):
                            # Prefer human-written explanations for domains where they exist.
                            files_to_load = [f for f in human_files if f in available_files]
                            log.info(f"Domain {domain}: Using HUMAN explanation files: {files_to_load}.")
                        elif not with_human:
                            files_to_load = [f for f in default_files if f in available_files]

                    # Sort files for deterministic assignment
                    files_to_load.sort()

                # If NOT using granular cycle mode, load all explanations into a single list
                # for legacy replacement or "own-batch" logic.
                if not (granular_explanation_analysis and (explanations_cycle == "full" or (isinstance(explanations_cycle, int) and explanations_cycle > 0))):
                    for filename in files_to_load:
                        file_path = os.path.join(explanation_dir, filename)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            explanation_chunks.extend(_chunk_explanation(f.read()))

                    if times_explanations > 1 and explanation_chunks:
                        explanation_chunks = explanation_chunks * times_explanations
                        log.info(f"Domain {domain}: Repeated explanations {times_explanations} times.")

                    log.info(f"Domain {domain}: Found {len(explanation_chunks)} explanation chunks from {files_to_load}.")

            # Document Chunks for the current domain
            domain_doc_chunks = [source_chunks] + paraphrased_chunks_by_doc
            # When enabled, treat explanations as an additional "document type" with its own batch.
            if with_explanations and explanation_in_own_batch and explanation_chunks:
                domain_doc_chunks.append(explanation_chunks)
            
            # Add original chunks to the primary list
            for i, chunks in enumerate(domain_doc_chunks):
                if i < len(unique_document_batches):
                    unique_document_batches[i].extend(chunks)

            # Distribute explanation files across document slots
            if with_explanations and (explanations_cycle == "full" or (isinstance(explanations_cycle, int) and explanations_cycle > 0)) and files_to_load:
                # Unified logic: files_to_load is always a dict now
                # Load and chunk files per type
                type_cycles = {}  # {type_name: [(filename, chunks), ...]}
                
                for expl_type, file_list in files_to_load.items():
                    type_file_chunks = []
                    for filename in file_list:
                        file_path = os.path.join(explanation_dir, filename)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            file_chunks = _chunk_explanation(f.read())
                        
                        if times_explanations > 1 and file_chunks:
                            file_chunks = file_chunks * times_explanations
                        
                        type_file_chunks.append((filename, file_chunks))
                    
                    # Handle double_cycle per type
                    if double_cycle and type_file_chunks:
                        num_original = len(type_file_chunks)
                        offset = num_original // 2
                        
                        second_cycle = []
                        for i in range(num_original):
                            offset_idx = (i + offset) % num_original
                            filename, file_chunks = type_file_chunks[offset_idx]
                            second_cycle.append((f"{filename}_offset", file_chunks))
                        
                        type_file_chunks.extend(second_cycle)
                        log.info(f"Domain {domain}, Type {expl_type}: Created double cycle with offset={offset}, total files: {len(type_file_chunks)}")
                    
                    type_cycles[expl_type] = type_file_chunks
                
                # Distribute: each doc_idx gets one chunk from EACH type's cycle
                for doc_idx in range(len(unique_document_batches)):
                    for expl_type, file_chunks_list in type_cycles.items():
                        if not file_chunks_list:
                            continue
                        
                        num_files = len(file_chunks_list) // 2 if double_cycle else len(file_chunks_list)
                        
                        # First cycle
                        expl_idx = doc_idx % num_files
                        filename, file_chunks = file_chunks_list[expl_idx]
                        unique_document_batches[doc_idx].extend(file_chunks)
                        log.info(f"Domain {domain}: Added {len(file_chunks)} chunks from {filename} (type: {expl_type}) to doc-type {doc_idx}.")
                        
                        # Second cycle (offset) if double_cycle enabled
                        if double_cycle:
                            second_idx = num_files + expl_idx
                            filename2, file_chunks2 = file_chunks_list[second_idx]
                            unique_document_batches[doc_idx].extend(file_chunks2)
                            log.info(f"Domain {domain}: Added {len(file_chunks2)} chunks from {filename2} (type: {expl_type}, offset) to doc-type {doc_idx}.")

            # 4. Handle explanation replacement logic (legacy path: explanations replace paraphrase chunks)
            if with_explanations and not explanation_in_own_batch and not (granular_explanation_analysis and (explanations_cycle == "full" or (isinstance(explanations_cycle, int) and explanations_cycle > 0))):
                domain_doc_chunks_expl = [c[:] for c in domain_doc_chunks]
                
                if explanation_chunks:
                    explanation_chunks_to_insert = list(explanation_chunks)
                    num_explanations_to_insert = len(explanation_chunks_to_insert)
                    total_replaced = 0

                    # Iterate backwards through the paraphrased documents
                    for i in range(len(paraphrased_chunks_by_doc) - 1, -1, -1):
                        if not explanation_chunks_to_insert:
                            break
                        
                        # Index in domain_doc_chunks is i + 1 (because source is at 0)
                        doc_chunk_idx = i + 1
                        num_chunks_in_doc = len(domain_doc_chunks_expl[doc_chunk_idx])

                        if num_explanations_to_insert >= num_chunks_in_doc:
                            # Replace the entire document's chunks with the last available explanation chunks
                            chunks_for_this_doc = explanation_chunks_to_insert[-num_chunks_in_doc:]
                            explanation_chunks_to_insert = explanation_chunks_to_insert[:-num_chunks_in_doc]
                            domain_doc_chunks_expl[doc_chunk_idx] = chunks_for_this_doc
                            
                            num_explanations_to_insert -= num_chunks_in_doc
                            total_replaced += num_chunks_in_doc
                        else:
                            # Replace only the end of the document's chunks
                            chunks_for_this_doc = explanation_chunks_to_insert
                            explanation_chunks_to_insert = [] # All used up
                            
                            original_chunks = domain_doc_chunks_expl[doc_chunk_idx]
                            
                            # The last `num_explanations_to_insert` chunks of this doc are replaced
                            num_to_replace_in_doc = len(chunks_for_this_doc)
                            new_chunks = original_chunks[:-num_to_replace_in_doc] + chunks_for_this_doc
                            domain_doc_chunks_expl[doc_chunk_idx] = new_chunks
                            
                            total_replaced += num_to_replace_in_doc
                            num_explanations_to_insert = 0

                    log.info(f"Domain {domain}: Replaced last {total_replaced} paraphrased chunks with explanation chunks.")
                
                # Add the processed (or copied) chunks for this domain to the explanation batches
                for i, chunks in enumerate(domain_doc_chunks_expl):
                    if i < len(unique_document_batches_with_explanations):
                        unique_document_batches_with_explanations[i].extend(chunks)

    # 5. Assemble final chunk list with optional pretraining data replay
    final_chunks = []
    pretraining_separators = int(strategy_args.get("separate_batches_with_pretraining", 0))
    log.info(f"Adding {pretraining_separators} batches of pretraining data as separator...")
    fill_with_pretraining = strategy_args.get("fill_batches_with_pretraining", False)

    data_replay = None
    if fill_with_pretraining or pretraining_separators > 0:
        pretraining_data_type = strategy_args.get('pretraining_data_type', 'dclm')
        log.info(f"Creating Pretraining Data Replayer from '{pretraining_data_type}'...")
        data_replay = PretrainingDataReplay(f'../../data/olmo/{pretraining_data_type}_100M_tokens.npy')

    # Assert that batches expected to have content are not empty before filling
    for i, batch in enumerate(unique_document_batches):
        assert batch, f"Batch for unique document type {i} is empty. Check data and strategy arguments."
    
    # effective_batch_size = train_cfg.per_device_train_batch_size * train_cfg.gradient_accumulation_steps

    # 6. Duplicate the dataset to match the desired number of training steps
    original_epochs = train_cfg.num_train_epochs
    replication_factor = 1
    
    if unique_document_batches and original_epochs > 1:
        num_doc_types = len(unique_document_batches)
        if original_epochs % num_doc_types != 0:
            log.warning(f"num_train_epochs ({original_epochs}) is not divisible by the number of unique document types ({num_doc_types}). "
                        "Rounding down the replication factor.")
        replication_factor = int(original_epochs / num_doc_types)
        log.info(f"Replicating dataset {replication_factor} times to simulate {original_epochs} total 'document-type epochs'.")

    if unique_document_batches:
        if explanation_every_round:
            assert unique_document_batches_with_explanations is not None, \
                "explanation_every_round=True but explanations are not enabled. Use --with_explanations or --with_specific_explanation."
        if unique_document_batches_with_explanations is not None:
            assert len(unique_document_batches_with_explanations) == len(unique_document_batches), \
                "Mismatch between explanation and original batch structures."
        final_chunks = replicate_and_interleave_pretraining(
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
            use_explanations_every_round=explanation_every_round,
            explanation_batches=explanation_batches if explanation_batches else None,
        )
    
    if shuffle_chunks_flag and final_chunks:
        # Shuffle deterministically with provided seed
        random.Random(shuffle_seed).shuffle(final_chunks)
        log.info(f"Shuffled final training chunks with seed {shuffle_seed}.")
        # Independent assertion: verify that different seeds yield different orders
        if len(final_chunks) > 1:
            indices = list(range(len(final_chunks)))
            a = indices[:]
            b = indices[:]
            random.Random(shuffle_seed).shuffle(a)
            random.Random(shuffle_seed + 1).shuffle(b)
            assert (a[0] != b[0]) or (a[1] != b[1]), \
                "Different shuffle seeds did not change first-two positions; check RNG or dataset size."
    
    total_tokens = sum(len(tokenizer(c, add_special_tokens=False)["input_ids"]) for c in final_chunks)
    log.info(f"Final training mix: Total chunks: {len(final_chunks)}, Total tokens: {total_tokens}")
    
    # The trainer will run for 1 epoch on the fully constructed dataset
    train_cfg.num_train_epochs = 1
    dataset = Dataset.from_dict({"raw_text": final_chunks})
    return dataset, train_cfg

def replicate_and_interleave_pretraining(
    unique_document_batches: List[List[str]],
    unique_document_batches_with_explanations: List[List[str]],
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
    Repeats a sequence of document batches for a specified number of replications,
    interleaving fresh pretraining data as separators in each replication. This ensures
    that while the main content is repeated, the pretraining data is not.
    """
    final_chunks = []
    effective_batch_size = train_cfg.per_device_train_batch_size * train_cfg.gradient_accumulation_steps

    if replication_factor == 0:
        return []

    for rep in range(replication_factor):
        if test_script:
            log.info(f"--- Creating replication {rep + 1}/{replication_factor} ---")
        
        # Select between original and explanation-infused batches
        if unique_document_batches_with_explanations:
            if use_explanations_every_round:
                log.info("Using batches WITH explanations for EVERY replication.")
                batches_for_this_rep = unique_document_batches_with_explanations
            else:
                # Alternate: first rep without explanations, second with, etc.
                should_use_explanations = (rep % 2 == 1)
                if should_use_explanations:
                    if test_script:
                        log.info("Using batches WITH explanations for this replication.")
                    batches_for_this_rep = unique_document_batches_with_explanations
                else:
                    if test_script:
                        log.info("Using batches WITHOUT explanations for this replication.")
                    batches_for_this_rep = unique_document_batches
                # Assert alternation is respected
                assert should_use_explanations == (batches_for_this_rep is unique_document_batches_with_explanations), \
                    "Explanation insertion is not alternating correctly."
        else:
            batches_for_this_rep = unique_document_batches

        for i, batch in enumerate(batches_for_this_rep):
            current_batch = list(batch)  # Make a copy to avoid modifying the original
            
            if fill_with_pretraining:
                if test_script:
                    log.info(f"--- Debugging Batch {i} in Replication {rep + 1} (Before Filling) ---")
                    log.info(f"Batch size: {len(current_batch)}")
                
                current_batch = fill_up_batch_with_pretraining_chunks(
                    current_batch, data_replay, effective_batch_size, train_cfg.context_length, tokenizer
                )
                
                if test_script:
                    log.info(f"--- Debugging Batch {i} in Replication {rep + 1} (After Filling) ---")
                    log.info(f"Batch size: {len(current_batch)}")

                assert len(current_batch) == effective_batch_size, \
                    f"Batch {i} has size {len(current_batch)}, which is not equal to the effective batch size {effective_batch_size}."

            final_chunks.extend(current_batch)
            
            # Add separator after each document batch, except for the very last one in the last replication
            is_last_batch_of_last_replication = (rep == replication_factor - 1) and (i == len(batches_for_this_rep) - 1)
            
            if pretraining_separators > 0 and not is_last_batch_of_last_replication:
                if test_script:
                    log.info(f"--- Debugging Separator after Batch {i} in Replication {rep + 1} ---")

                pretraining_fill = get_pretraining_batches(
                    data_replay,
                    pretraining_separators,
                    effective_batch_size,
                    train_cfg.context_length,
                    tokenizer,
                )
                
                if test_script and pretraining_fill:
                    log.info(f"Added {len(pretraining_fill)} separator chunks.")
                    log.info(f"First separator chunk: '{pretraining_fill[0][:100]}...'")
                    log.info(f"Last separator chunk: '{pretraining_fill[-1][:100]}...'")

                final_chunks.extend(pretraining_fill)
    
    return final_chunks
