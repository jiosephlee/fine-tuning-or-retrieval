import math
import os
import torch
import numpy as np
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

def prepare_lima_dataset(tokenizer: AutoTokenizer, log, use_eot_token=False, sort=False):
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
    dataset = load_dataset("GAIR/lima")
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

    # Determine domains: use override if provided, otherwise scan directory
    domains = strategy_args.get("override_domains", None)
    unique_document_batches = []
    
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
            cleaned_dir = '../../data/arxiv/cleaned'
            domains = [f.replace('.tex', '') for f in os.listdir(cleaned_dir) if f.endswith('.tex')]
        log.info(f"Processing domains: {domains}")

        num_paraphrased_texts = strategy_args.get("num_paraphrased_texts", 0)
        with_explanations = "WithExplanations" in strategy_name

        # Each inner list holds chunks for a "unique document type" across all domains
        # e.g., unique_document_batches[0] = all source chunks
        #       unique_document_batches[1] = all paraphrase-1 chunks
        unique_document_batches = [[] for _ in range(1 + num_paraphrased_texts)]

        for domain in domains:
            log.info(f"Loading data for domain: {domain}")
            
            # 1. Load source text
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
            explanation_chunks = []
            if with_explanations:
                explanation_dir = f'data/arxiv/explanations/{domain}/'
                if os.path.isdir(explanation_dir):
                    for filename in sorted(os.listdir(explanation_dir)):
                        if filename.endswith('.txt'):
                            file_path = os.path.join(explanation_dir, filename)
                            with open(file_path, 'r', encoding='utf-8') as f:
                                explanation_chunks.extend(_chunk(f.read()))
                log.info(f"Domain {domain}: Found {len(explanation_chunks)} explanation chunks.")
            
            # This part ensures that chunks from each document type of a domain are added to the correct overall batch
            
            # Document Chunks for the current domain
            domain_doc_chunks = [source_chunks] + paraphrased_chunks_by_doc

            # 4. Handle explanation replacement logic
            num_chunks_per_source_doc = len(source_chunks)
            if with_explanations and explanation_chunks:
                paraphrased_units_for_explanations = math.ceil(len(explanation_chunks) / num_chunks_per_source_doc)
                
                # Start replacing from the last paraphrased doc towards the first
                if paraphrased_units_for_explanations > 0:
                    # Distribute explanation chunks among the slots of the documents they replace
                    num_to_replace = min(paraphrased_units_for_explanations, len(paraphrased_chunks_by_doc))
                    
                    # Split explanations into `num_to_replace` parts
                    split_explanations = [explanation_chunks[i::num_to_replace] for i in range(num_to_replace)]

                    for i in range(num_to_replace):
                        # Index of paraphrased doc to replace (from the end)
                        replace_idx = len(paraphrased_chunks_by_doc) - 1 - i
                        # Corresponding index in domain_doc_chunks (source is at 0)
                        doc_chunk_idx = replace_idx + 1
                        
                        domain_doc_chunks[doc_chunk_idx] = split_explanations[i]
                    log.info(f"Domain {domain}: Replaced {num_to_replace} paraphrased documents with explanation chunks.")
            
            # Add the processed chunks for this domain to the main batches
            for i, chunks in enumerate(domain_doc_chunks):
                if i < len(unique_document_batches):
                    unique_document_batches[i].extend(chunks)
                    if test_script:
                        log.info(f"Batch {i} increases to {len(unique_document_batches[i])} chunks")
                    
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
    
    effective_batch_size = train_cfg.per_device_train_batch_size * train_cfg.gradient_accumulation_steps

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
        final_chunks = replicate_and_interleave_pretraining(
            unique_document_batches=unique_document_batches,
            replication_factor=replication_factor,
            data_replay=data_replay,
            pretraining_separators=pretraining_separators,
            train_cfg=train_cfg,
            tokenizer=tokenizer,
            log=log,
            test_script=test_script,
            fill_with_pretraining=fill_with_pretraining
        )
    
    total_tokens = sum(len(tokenizer(c, add_special_tokens=False)["input_ids"]) for c in final_chunks)
    log.info(f"Final training mix: Total chunks: {len(final_chunks)}, Total tokens: {total_tokens}")
    
    # The trainer will run for 1 epoch on the fully constructed dataset
    train_cfg.num_train_epochs = 1
    dataset = Dataset.from_dict({"raw_text": final_chunks})
    return dataset, train_cfg

def replicate_and_interleave_pretraining(
    unique_document_batches: List[List[str]],
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
        
        for i, batch in enumerate(unique_document_batches):
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
            is_last_batch_of_last_replication = (rep == replication_factor - 1) and (i == len(unique_document_batches) - 1)
            
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
