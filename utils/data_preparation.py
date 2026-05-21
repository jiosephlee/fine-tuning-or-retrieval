import os
import torch
import numpy as np
import random
from typing import List, Optional
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

from utils.llm_configs import TrainingConfig
import utils.chunking as chunking
from itertools import cycle, repeat


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
    if num_chunks_needed < 0:
        raise ValueError(
            f"Constructed batch has {len(batch)} chunks, which exceeds effective_batch_size={batch_size}. "
            "Reduce aux/explanation chunks per insertion or increase --effective_batch_size_for_cpt."
        )
    if num_chunks_needed == 0:
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


def _stable_domain_offset(domain: str) -> int:
    return sum((idx + 1) * ord(char) for idx, char in enumerate(domain))


def _build_granular_queue_tracks(
    type_objects: dict,
    num_tracks: int,
    shuffle_seed: int,
    domain: str,
) -> List[List[List[str]]]:
    queue_items = []
    for expl_type in sorted(type_objects):
        for filename, chunks in type_objects[expl_type]:
            if chunks:
                queue_items.append((expl_type, filename, chunks))

    if not queue_items:
        return []

    rng = random.Random(shuffle_seed + _stable_domain_offset(domain))
    rng.shuffle(queue_items)
    queue_items.sort(key=lambda item: len(item[2]), reverse=True)

    tracks = [[] for _ in range(num_tracks)]
    left = 0
    right = len(queue_items) - 1

    while left <= right:
        row = []
        take_long = True
        while len(row) < num_tracks and left <= right:
            if take_long:
                row.append(queue_items[left])
                left += 1
            else:
                row.append(queue_items[right])
                right -= 1
            take_long = not take_long

        for track_idx, (_, _, chunks) in enumerate(row):
            tracks[track_idx].append(chunks)

    return [track for track in tracks if track]


def _chunk_groups(chunks: List[str], group_size: int) -> List[List[str]]:
    if group_size <= 0:
        raise ValueError(f"Chunk group size must be positive, got: {group_size}")
    return [chunks[idx:idx + group_size] for idx in range(0, len(chunks), group_size)]


def _build_chunk_granular_pool_from_objects(
    type_objects: dict,
    group_size: int,
) -> List[List[str]]:
    flat_chunks = []
    for _, e_list in type_objects.items():
        for _, chunks in e_list:
            flat_chunks.extend(chunks)
    return _chunk_groups(flat_chunks, group_size)


def _rotate_track_pool(track_pool: List[List[str]], track_idx: int, num_tracks: int) -> List[List[str]]:
    if not track_pool:
        return []
    offset = (track_idx * len(track_pool)) // num_tracks
    return track_pool if offset == 0 else track_pool[offset:] + track_pool[:offset]


def _paraphrase_splice_order(
    num_paraphrases: int,
    insertion_strategy: str,
    shuffle_seed: int,
    domain: str,
) -> List[int]:
    if num_paraphrases <= 0:
        return []

    if insertion_strategy == "legacy":
        return list(range(num_paraphrases, 0, -1))

    if insertion_strategy == "random_splice":
        rng = random.Random(shuffle_seed + _stable_domain_offset(domain))
        start = rng.randrange(num_paraphrases)
        paraphrase_positions = list(range(start, num_paraphrases)) + list(range(0, start))
        return [position + 1 for position in paraphrase_positions]

    raise ValueError(f"Unsupported splice insertion strategy: {insertion_strategy}")


def _extend_spliced_document_batches(
    target_batches: List[List[str]],
    source_chunks: List[str],
    paraphrased_chunks_by_doc: List[List[str]],
    explanation_chunks: List[str],
    insertion_strategy: str,
    shuffle_seed: int,
    domain: str,
) -> None:
    """Extend document-shaped batches with source plus spliced paraphrase chunks.

    The source batch is always preserved. Splicing happens at chunk granularity:
    explanation chunks replace whole paraphrase chunks and never split a chunk.
    """
    if target_batches:
        target_batches[0].extend(source_chunks)

    if not explanation_chunks:
        for batch_idx, chunks in enumerate(paraphrased_chunks_by_doc, start=1):
            if batch_idx < len(target_batches):
                target_batches[batch_idx].extend(chunks)
        return

    if insertion_strategy == "legacy":
        to_insert = list(explanation_chunks)
        for batch_idx in _paraphrase_splice_order(
            len(paraphrased_chunks_by_doc), insertion_strategy, shuffle_seed, domain
        ):
            original_chunks = paraphrased_chunks_by_doc[batch_idx - 1]
            if not to_insert:
                target_batches[batch_idx].extend(original_chunks)
                continue
            if len(to_insert) >= len(original_chunks):
                target_batches[batch_idx].extend(to_insert[-len(original_chunks):])
                to_insert = to_insert[:-len(original_chunks)]
            else:
                keep_n = len(original_chunks) - len(to_insert)
                target_batches[batch_idx].extend(original_chunks[:keep_n] + to_insert)
                to_insert = []
        return

    to_insert = list(explanation_chunks)
    for batch_idx in _paraphrase_splice_order(
        len(paraphrased_chunks_by_doc), insertion_strategy, shuffle_seed, domain
    ):
        original_chunks = paraphrased_chunks_by_doc[batch_idx - 1]
        if not to_insert:
            target_batches[batch_idx].extend(original_chunks)
            continue
        if len(to_insert) >= len(original_chunks):
            target_batches[batch_idx].extend(to_insert[:len(original_chunks)])
            to_insert = to_insert[len(original_chunks):]
        else:
            keep_n = len(original_chunks) - len(to_insert)
            target_batches[batch_idx].extend(original_chunks[:keep_n] + to_insert)
            to_insert = []


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
    # ... [Beginning of function remains the same] ...
    log.info(f"Preparing training mix for strategy: {strategy_name}")

    # ... [Args extraction remains the same] ...
    test_script = strategy_args.get("test_script", False)
    specific_explanation_type = strategy_args.get("with_specific_explanation", None)
    if isinstance(specific_explanation_type, list) and len(specific_explanation_type) == 1:
        specific_explanation_type = specific_explanation_type[0]
    times_explanations = strategy_args.get("times_explanations", 1)
    semi_cleaned_version = strategy_args.get("semi_cleaned", None)
    use_raw = strategy_args.get("use_raw", False)
    shuffle_chunks_flag = strategy_args.get("shuffle_chunks", False)
    shuffle_seed = strategy_args.get("shuffle_seed", 42)
    granular_explanations_cycle = strategy_args.get(
        "granular_explanations_cycle",
        strategy_args.get("explanations_cycle", 0),
    )
    try:
        granular_explanations_num_tracks = int(
            strategy_args.get(
                "granular_explanations_num_tracks",
                strategy_args.get("explanations_num_tracks", 1),
            )
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "--granular_explanations_num_tracks must be a positive integer, got: "
            f"{strategy_args.get('granular_explanations_num_tracks', strategy_args.get('explanations_num_tracks'))}"
        ) from exc
    if granular_explanations_num_tracks <= 0:
        raise ValueError(f"--granular_explanations_num_tracks must be a positive integer, got: {granular_explanations_num_tracks}")
    explanations_insertion_strategy = strategy_args.get("explanations_insertion_strategy", "granular")
    explanation_granularity = strategy_args.get("explanation_granularity", "file")
    try:
        explanation_track_size_by_chunk = int(strategy_args.get("explanation_track_size_by_chunk", 4))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "--explanation_track_size_by_chunk must be a positive integer, got: "
            f"{strategy_args.get('explanation_track_size_by_chunk')}"
        ) from exc
    if explanation_granularity not in {"file", "chunk"}:
        raise ValueError(f"--explanation_granularity must be 'file' or 'chunk', got: {explanation_granularity}")
    if explanation_track_size_by_chunk <= 0:
        raise ValueError(
            f"--explanation_track_size_by_chunk must be a positive integer, got: {explanation_track_size_by_chunk}"
        )
    if explanations_insertion_strategy not in {"granular", "granular_queue"} and explanation_granularity != "file":
        raise ValueError(
            "--explanation_granularity chunk is only supported with granular or granular_queue insertion."
        )
    whole_explanations_insert_every_n = int(
        strategy_args.get(
            "whole_explanations_insert_every_n",
            strategy_args.get("explanations_insert_every_n", 1),
        )
    )
    fill_chunk_gaps = strategy_args.get("fill_chunk_gaps", True)
    chunk_gap_threshold = strategy_args.get("fill_chunk_gap_threshold", 1000)
    pretraining_separators = int(strategy_args.get("separate_batches_with_pretraining", 0))
    fill_with_pretraining = strategy_args.get("fill_batches_with_pretraining", True)
    domain_data_sources = strategy_args.get("domain_data_sources", {}) or {}
    document_track_baseline = bool(strategy_args.get("document_track_baseline", False))
    match_explanation_source_replay = bool(strategy_args.get("match_explanation_source_replay", False))
    document_match_specific_explanation = strategy_args.get("document_match_specific_explanation", None)
    document_match_insert_content = strategy_args.get("document_match_insert_content", "document")
    if isinstance(document_match_specific_explanation, list) and len(document_match_specific_explanation) == 1:
        document_match_specific_explanation = document_match_specific_explanation[0]
    if document_match_insert_content not in {"document", "cited_works", "prior_knowledge"}:
        raise ValueError(
            "--document_match_insert_content must be one of: document, cited_works, prior_knowledge; "
            f"got {document_match_insert_content!r}"
        )

    with_prior_knowledge = bool(strategy_args.get("with_prior_knowledge", False))
    prior_knowledge_insertion = strategy_args.get("prior_knowledge_insertion", "front")
    prior_knowledge_cycle = strategy_args.get("prior_knowledge_cycle", "full")
    prior_knowledge_match_document_track = bool(strategy_args.get("prior_knowledge_match_document_track", False))

    # --- Helper Functions ---
    def _chunk(text: str) -> List[str]:
        text_with_eos = text + tokenizer.eos_token
        chunks, _ = chunking.chunk(
            text_with_eos, tokenizer, train_cfg.context_length,
            chunk_by_section, overlap_sections, overlap_ratio, add_title_prefix, log=log
        )
        return chunks

    def _chunk_explanation(text: str, title_prefix: bool = True) -> List[str]:
        text_with_eos = text + tokenizer.eos_token
        explanation_overlap_ratio = "1_10"
        use_title = add_title_prefix if title_prefix else False
        chunks, _ = chunking.chunk(
            text_with_eos, tokenizer, train_cfg.context_length,
            chunk_by_section, overlap_sections, explanation_overlap_ratio, use_title, log=log
        )
        return chunks

    def _first_existing_path(candidates: List[str]) -> Optional[str]:
        for cand in candidates:
            if os.path.exists(cand):
                return cand
        return None

    def _source_document_root(source: str) -> str:
        if use_raw:
            return f'../../data/{source}/raw'
        if semi_cleaned_version:
            semicleaned_root = f'../../data/{source}/semicleaned_{semi_cleaned_version}'
            if os.path.isdir(semicleaned_root):
                return semicleaned_root
        return f'../../data/{source}/cleaned'

    def _discover_domains_for_source(source: str) -> List[str]:
        root = _source_document_root(source)
        if not os.path.isdir(root):
            return []
        return sorted(
            os.path.splitext(name)[0]
            for name in os.listdir(root)
            if os.path.isfile(os.path.join(root, name)) and name.endswith(('.txt', '.tex'))
        )

    def _active_shuffle_suffixes() -> List[str]:
        suffixes: List[str] = []
        if word_shuffled:
            suffixes.append('_shuffle_words')
        if sentence_shuffled:
            suffixes.append('_shuffle_sentences')
        if paragraph_shuffled:
            suffixes.append('_shuffle_paragraphs')
        if shuffled_papers:
            suffixes.append('_shuffle')
        return suffixes

    def _find_shuffled_source_path(source: str, domain: str) -> Optional[str]:
        suffixes = _active_shuffle_suffixes()
        if not suffixes:
            return None
        shuffled_root = f'../../data/{source}/shuffled'
        candidates = [
            os.path.join(shuffled_root, 'cleaned', f'{domain}{suffix}{ext}')
            for suffix in suffixes
            for ext in ('.tex', '.txt')
        ]
        return _first_existing_path(candidates)

    def _find_standard_source_path(source: str, domain: str) -> Optional[str]:
        root = _source_document_root(source)
        return _first_existing_path([
            os.path.join(root, f'{domain}.tex'),
            os.path.join(root, f'{domain}.txt'),
        ])

    def _find_shuffled_paraphrase_path(source: str, domain: str, idx: int, paraphrased_dir: str) -> Optional[str]:
        suffixes = _active_shuffle_suffixes()
        if not suffixes:
            return None
        shuffled_root = f'../../data/{source}/shuffled'
        candidates = []
        for suffix in suffixes:
            for ext in ('.tex', '.txt'):
                candidates.append(os.path.join(shuffled_root, 'paraphrased', domain, f'{idx}{suffix}{ext}'))
                candidates.append(os.path.join(paraphrased_dir, f'{idx}{suffix}{ext}'))
        return _first_existing_path(candidates)

    # --- 1. Load Domains & Documents ---
    domains = strategy_args.get("override_domains", None)
    # Legacy flag and new more specific flags
    shuffled_papers = strategy_args.get("shuffled_papers", False)
    word_shuffled = strategy_args.get("word_shuffled_papers", False)
    sentence_shuffled = strategy_args.get("sentence_shuffled_papers", False)
    paragraph_shuffled = strategy_args.get("paragraph_shuffled_papers", False)
    if domains is None:
        domains = []
        domain_data_sources = {}
        for source in ("arxiv", "legal", "medical"):
            discovered = _discover_domains_for_source(source)
            for domain in discovered:
                if domain in domain_data_sources and domain_data_sources[domain] != source:
                    log.warning(
                        f"Domain '{domain}' appears in both '{domain_data_sources[domain]}' and '{source}'. "
                        "Keeping first occurrence."
                    )
                    continue
                domain_data_sources[domain] = source
                domains.append(domain)
    else:
        domain_data_sources = {domain: domain_data_sources.get(domain, "arxiv") for domain in domains}
    
    log.info(f"Processing domains: {domains}")
    log.info(f"Domain data sources: {domain_data_sources}")
    
    num_paraphrased_texts = strategy_args.get("num_paraphrased_texts", 0)
    include_explanations = specific_explanation_type is not None or "WithSpecificExplanations" in strategy_name
    
    unique_document_batches = [[] for _ in range(1 + num_paraphrased_texts)]
    explanation_spliced_document_batches = None
    
    # Granular strategy structure: List of Domains -> List of Tracks -> List of Chunks
    domain_explanation_tracks = []
    # Document-match baseline structure: same track shape as explanations, but document chunks only.
    domain_document_match_tracks = []
    # Whole strategy structure: List of Domains -> Explanation-only batch chunks
    domain_whole_explanation_batches = []
    # Prior-knowledge structure: List of Domains -> List of Chapters -> List of Chunks
    domain_pk_chapter_chunks = []

    for domain in domains:
        log.info(f"Loading data for domain: {domain}")
        domain_source = domain_data_sources.get(domain, "arxiv")
        
        # Load Source
        if strategy_name == "PriorKnowledge":
            source_path = _first_existing_path([
                f'../../data/{domain_source}/prior_knowledge/{domain}/textbook.txt'
            ])
        else:
            source_path = _find_shuffled_source_path(domain_source, domain)
            if source_path is None:
                source_path = _find_standard_source_path(domain_source, domain)

        if not source_path:
            log.warning(f"Missing source document for domain '{domain}' in source '{domain_source}'. Skipping.")
            continue
        try:
            with open(source_path, 'r', encoding='utf-8') as f: source_text = f.read()
        except FileNotFoundError: 
            log.warning(f"Source file not found at {source_path}. Skipping.")
            continue
        source_chunks = _chunk(source_text)

        # Load Paraphrases
        paraphrased_chunks_by_doc = []
        if num_paraphrased_texts > 0:
            paraphrased_dir = f'../../data/{domain_source}/paraphrased/{domain}/'
            if os.path.isdir(paraphrased_dir):
                for i in range(num_paraphrased_texts):
                    para_path = _find_shuffled_paraphrase_path(domain_source, domain, i, paraphrased_dir)
                    if para_path is None:
                        para_path = _first_existing_path([
                            os.path.join(paraphrased_dir, f'{i}.tex'),
                            os.path.join(paraphrased_dir, f'{i}.txt'),
                        ])
                    if para_path:
                        with open(para_path, 'r', encoding='utf-8') as f:
                            paraphrased_chunks_by_doc.append(_chunk(f.read()))
        
        # Add to main batch list
        domain_doc_chunks = [source_chunks] + paraphrased_chunks_by_doc
        for i, chunks in enumerate(domain_doc_chunks):
            if i < len(unique_document_batches):
                unique_document_batches[i].extend(chunks)

        # --- 2. Load Explanations ---
        current_domain_tracks = [] # Granular strategy only
        current_domain_document_match_tracks = [] # Document baseline only
        current_domain_whole_batch = [] # Whole strategy only

        def _resolve_explanation_subfolder(expl_type: str):
            """Return (subfolder_abs_path, sorted_filenames) for an explanation type.

            Special-cases 'prior_knowledge' to read from data/{source}/prior_knowledge/{domain}/
            with chapter_*.txt files sorted numerically by chapter index.
            """
            if expl_type == "prior_knowledge":
                subfolder_path = os.path.abspath(f'../../data/{domain_source}/prior_knowledge/{domain}/')
                if not os.path.isdir(subfolder_path):
                    return None, []
                def _chapter_index(name: str) -> int:
                    try:
                        return int(name.replace('chapter_', '').replace('.txt', ''))
                    except ValueError:
                        return 10**9
                files = sorted(
                    [f for f in os.listdir(subfolder_path) if f.startswith('chapter_') and f.endswith('.txt')],
                    key=_chapter_index,
                )
                return subfolder_path, files
            subfolder_path = os.path.join(explanation_dir, expl_type)
            if not os.path.isdir(subfolder_path):
                return None, []
            files = sorted([f for f in os.listdir(subfolder_path) if f.endswith('.txt')])
            return subfolder_path, files

        def _select_granular_explanation_files(explanation_dir: str, explanation_types_arg):
            files_to_load = {}
            if not explanation_types_arg:
                return files_to_load
            explanation_types = explanation_types_arg if isinstance(explanation_types_arg, list) else [explanation_types_arg]
            for expl_type in explanation_types:
                subfolder_path, all_files = _resolve_explanation_subfolder(expl_type)
                if subfolder_path is None:
                    continue
                # Store paths joined with the resolved subfolder so the loader opens them directly.
                # When subfolder_path is absolute (prior_knowledge), os.path.join with explanation_dir
                # downstream will preserve the absolute path.
                if granular_explanations_cycle == "full":
                    type_files = [os.path.join(subfolder_path, f) for f in all_files]
                elif isinstance(granular_explanations_cycle, int) and granular_explanations_cycle > 0:
                    type_files = [os.path.join(subfolder_path, f) for f in all_files[:granular_explanations_cycle]]
                else:
                    type_files = []
                files_to_load[expl_type] = type_files
            return files_to_load

        def _select_granular_queue_explanation_files(explanation_dir: str, explanation_types_arg):
            files_to_load = {}
            if not explanation_types_arg:
                return files_to_load
            explanation_types = explanation_types_arg if isinstance(explanation_types_arg, list) else [explanation_types_arg]
            for expl_type in explanation_types:
                subfolder_path, all_files = _resolve_explanation_subfolder(expl_type)
                if subfolder_path is None:
                    continue
                files_to_load[expl_type] = [os.path.join(subfolder_path, f) for f in all_files]
            return files_to_load

        def _load_granular_type_objects(explanation_dir: str, files_to_load: dict):
            type_objects = {}
            for expl_type, file_list in files_to_load.items():
                objs = []
                for filename in file_list:
                    # filename is already a full path (relative to cwd or absolute) — opened directly.
                    with open(filename, 'r', encoding='utf-8') as f:
                        file_chunks = _chunk_explanation(f.read())
                    if times_explanations > 1 and file_chunks:
                        file_chunks = file_chunks * times_explanations
                    objs.append((filename, file_chunks))
                type_objects[expl_type] = objs
            return type_objects

        def _create_granular_pool_from_objects(obj_dict):
            pool = []
            max_len = max([len(x) for x in obj_dict.values()]) if obj_dict else 0
            for idx in range(max_len):
                step_chunks = []
                for _, e_list in obj_dict.items():
                    if not e_list:
                        continue
                    _, ch = e_list[idx % len(e_list)]
                    step_chunks.extend(ch)
                if step_chunks:
                    pool.append(step_chunks)
            return pool

        def _create_document_match_pool_from_objects(obj_dict, document_chunks: List[str]):
            pool = []
            if not document_chunks:
                return pool
            max_len = max([len(x) for x in obj_dict.values()]) if obj_dict else 0
            document_iterator = cycle(document_chunks)
            for idx in range(max_len):
                target_chunks = 0
                for _, e_list in obj_dict.items():
                    if not e_list:
                        continue
                    _, ch = e_list[idx % len(e_list)]
                    target_chunks += len(ch)
                if target_chunks > 0:
                    pool.append([next(document_iterator) for _ in range(target_chunks)])
            return pool

        def _create_document_match_pool_from_step_sizes(step_sizes: List[int], document_chunks: List[str]):
            if not document_chunks:
                return []
            document_iterator = cycle(document_chunks)
            return [
                [next(document_iterator) for _ in range(step_size)]
                for step_size in step_sizes
                if step_size > 0
            ]

        def _load_document_match_insert_chunks(explanation_dir: str) -> List[str]:
            if document_match_insert_content == "document":
                chunks = list(source_chunks)
                if num_paraphrased_texts > 0:
                    for para_chunks in paraphrased_chunks_by_doc:
                        chunks.extend(para_chunks)
                return chunks

            insert_type = {
                "cited_works": "cited_textbooks",
                "prior_knowledge": "prior_knowledge",
            }[document_match_insert_content]
            subfolder_path, files = _resolve_explanation_subfolder(insert_type)
            if subfolder_path is None or not files:
                log.warning(
                    f"Domain {domain}: --document_match_insert_content={document_match_insert_content} "
                    f"requested, but no {insert_type} files found; skipping matched insert track."
                )
                return []

            chunks: List[str] = []
            title_prefix = document_match_insert_content != "prior_knowledge"
            for filename in files:
                with open(os.path.join(subfolder_path, filename), 'r', encoding='utf-8') as f:
                    chunks.extend(_chunk_explanation(f.read(), title_prefix=title_prefix))
            log.info(
                f"Domain {domain}: using {document_match_insert_content} as matched insert content "
                f"({len(files)} files, {len(chunks)} chunks)."
            )
            return chunks

        if document_track_baseline:
            explanation_dir = f'../../data/{domain_source}/explanations/{domain}/'
            files_to_load = _select_granular_explanation_files(
                explanation_dir,
                document_match_specific_explanation,
            )
            type_objects = _load_granular_type_objects(explanation_dir, files_to_load)
            matched_insert_chunks = _load_document_match_insert_chunks(explanation_dir)

            for track_idx in range(granular_explanations_num_tracks):
                track_objects = {}
                for e_type, e_list in type_objects.items():
                    if not e_list:
                        track_objects[e_type] = []
                        continue
                    offset = (track_idx * len(e_list)) // granular_explanations_num_tracks
                    track_objects[e_type] = e_list if offset == 0 else e_list[offset:] + e_list[:offset]

                if explanation_granularity == "chunk":
                    explanation_track_pool = _rotate_track_pool(
                        _build_chunk_granular_pool_from_objects(
                            type_objects,
                            explanation_track_size_by_chunk,
                        ),
                        track_idx,
                        granular_explanations_num_tracks,
                    )
                    track_pool = _create_document_match_pool_from_step_sizes(
                        [len(step) for step in explanation_track_pool],
                        matched_insert_chunks,
                    )
                else:
                    track_pool = _create_document_match_pool_from_objects(track_objects, matched_insert_chunks)
                if track_pool:
                    current_domain_document_match_tracks.append(track_pool)
            if current_domain_document_match_tracks:
                matched_chunks = sum(
                    len(step)
                    for track_pool in current_domain_document_match_tracks
                    for step in track_pool
                )
                log.info(
                    f"Domain {domain}: built matched insert tracks "
                    f"(content={document_match_insert_content}, {len(current_domain_document_match_tracks)} tracks, "
                    f"{matched_chunks} scheduled chunks)."
                )
        
        if include_explanations:
            explanation_dir = f'../../data/{domain_source}/explanations/{domain}/'
            files_to_load = {}
            use_subfolder_loading = explanations_insertion_strategy in ("granular", "granular_queue")

            # Identify files
            if use_subfolder_loading and specific_explanation_type:
                if explanations_insertion_strategy == "granular_queue":
                    files_to_load = _select_granular_queue_explanation_files(explanation_dir, specific_explanation_type)
                else:
                    files_to_load = _select_granular_explanation_files(explanation_dir, specific_explanation_type)
            else:
                # Whole + legacy use flat explanation files in the domain root.
                # Normalize common aliases so "textbooks" maps to "textbook.txt".
                alias_to_flat = {
                    "textbook": "textbook",
                    "textbooks": "textbook",
                    "blog": "blogs",
                    "blogs": "blogs",
                    "stackexchange": "stackexchange",
                    "stack": "stackexchange",
                }
                resolved_paths = []
                if specific_explanation_type:
                    raw_types = specific_explanation_type if isinstance(specific_explanation_type, list) else [specific_explanation_type]
                    for expl_type in raw_types:
                        if expl_type == "prior_knowledge":
                            pk_textbook = os.path.abspath(f'../../data/{domain_source}/prior_knowledge/{domain}/textbook.txt')
                            if os.path.isfile(pk_textbook):
                                resolved_paths.append(pk_textbook)
                            continue
                        key = alias_to_flat.get(expl_type, expl_type)
                        flat_name = key if key.endswith(".txt") else f"{key}.txt"
                        candidate = os.path.join(explanation_dir, flat_name)
                        if os.path.isfile(candidate):
                            resolved_paths.append(candidate)
                else:
                    if os.path.isdir(explanation_dir):
                        avail = set(os.listdir(explanation_dir))
                        for flat_name in ('blogs.txt', 'stackexchange.txt', 'textbook.txt'):
                            if flat_name in avail:
                                resolved_paths.append(os.path.join(explanation_dir, flat_name))
                resolved_paths.sort()
                if resolved_paths:
                    files_to_load["default"] = resolved_paths

            if explanations_insertion_strategy in ("legacy", "random_splice"):
                # >> SPLICE METHODS <<
                splice_expl_chunks = []
                if "default" in files_to_load:
                    for file_path in files_to_load["default"]:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            splice_expl_chunks.extend(_chunk_explanation(f.read()))
                if times_explanations > 1 and splice_expl_chunks:
                    splice_expl_chunks = splice_expl_chunks * times_explanations

                if explanation_spliced_document_batches is None:
                    explanation_spliced_document_batches = [[] for _ in range(len(unique_document_batches))]

                _extend_spliced_document_batches(
                    target_batches=explanation_spliced_document_batches,
                    source_chunks=source_chunks,
                    paraphrased_chunks_by_doc=paraphrased_chunks_by_doc,
                    explanation_chunks=splice_expl_chunks,
                    insertion_strategy=explanations_insertion_strategy,
                    shuffle_seed=shuffle_seed,
                    domain=domain,
                )
            elif explanations_insertion_strategy in ("granular", "granular_queue", "whole"):
                # Build base objects {type: [(filename, chunks)]}
                type_objects = _load_granular_type_objects(explanation_dir, files_to_load)

                if explanations_insertion_strategy == "granular":
                    if explanation_granularity == "chunk":
                        base_track_pool = _build_chunk_granular_pool_from_objects(
                            type_objects,
                            explanation_track_size_by_chunk,
                        )
                        for track_idx in range(granular_explanations_num_tracks):
                            track_pool = _rotate_track_pool(
                                base_track_pool,
                                track_idx,
                                granular_explanations_num_tracks,
                            )
                            if track_pool:
                                if track_idx > 0:
                                    log.info(
                                        f"Domain {domain}: Adding CHUNK OFFSET track "
                                        f"{track_idx + 1}/{granular_explanations_num_tracks} "
                                        f"(size {len(track_pool)}, chunk_group_size={explanation_track_size_by_chunk})."
                                    )
                                current_domain_tracks.append(track_pool)
                    else:
                        # Generalized Track Construction:
                        # track_idx=0 is the main cycle, and subsequent tracks are phase-offset cycles.
                        # Offset rule is floor(track_idx * len(type_files) / num_tracks), so:
                        # - num_tracks=2 -> offset by half
                        # - num_tracks=3 -> offsets around one-third and two-thirds
                        for track_idx in range(granular_explanations_num_tracks):
                            track_objects = {}
                            for e_type, e_list in type_objects.items():
                                if not e_list:
                                    track_objects[e_type] = []
                                    continue

                                offset = (track_idx * len(e_list)) // granular_explanations_num_tracks
                                if offset == 0:
                                    track_objects[e_type] = e_list
                                else:
                                    track_objects[e_type] = e_list[offset:] + e_list[:offset]

                            track_pool = _create_granular_pool_from_objects(track_objects)
                            if track_pool:
                                if track_idx > 0:
                                    log.info(
                                        f"Domain {domain}: Adding OFFSET track {track_idx + 1}/{granular_explanations_num_tracks} "
                                        f"(size {len(track_pool)})."
                                    )
                                current_domain_tracks.append(track_pool)
                elif explanations_insertion_strategy == "granular_queue":
                    if explanation_granularity == "chunk":
                        queue_tracks = []
                        base_track_pool = _build_chunk_granular_pool_from_objects(
                            type_objects,
                            explanation_track_size_by_chunk,
                        )
                        for track_idx in range(granular_explanations_num_tracks):
                            track_pool = _rotate_track_pool(
                                base_track_pool,
                                track_idx,
                                granular_explanations_num_tracks,
                            )
                            if track_pool:
                                queue_tracks.append(track_pool)
                    else:
                        queue_tracks = _build_granular_queue_tracks(
                            type_objects=type_objects,
                            num_tracks=granular_explanations_num_tracks,
                            shuffle_seed=shuffle_seed,
                            domain=domain,
                        )
                    if queue_tracks:
                        queued_items = sum(len(track_pool) for track_pool in queue_tracks)
                        item_unit = "chunk groups" if explanation_granularity == "chunk" else "files"
                        log.info(
                            f"Domain {domain}: built GRANULAR_QUEUE tracks "
                            f"({len(queue_tracks)}/{granular_explanations_num_tracks} non-empty tracks, "
                            f"{queued_items} {item_unit} queued)."
                        )
                        current_domain_tracks.extend(queue_tracks)
                else:
                    # Whole strategy: use all selected explanation chunks as one standalone insertion batch.
                    for _, e_list in type_objects.items():
                        for _, ch in e_list:
                            current_domain_whole_batch.extend(ch)
            else:
                raise ValueError(
                    f"Unsupported explanations_insertion_strategy: {explanations_insertion_strategy}. "
                    "Expected one of: granular, granular_queue, whole, legacy, random_splice."
                )

        if match_explanation_source_replay and current_domain_tracks:
            document_replay_chunks = list(source_chunks)
            if num_paraphrased_texts > 0:
                for chunks in paraphrased_chunks_by_doc:
                    document_replay_chunks.extend(chunks)

            for track_pool in current_domain_tracks:
                matched_track_pool = _create_document_match_pool_from_step_sizes(
                    [len(step) for step in track_pool],
                    document_replay_chunks,
                )
                if matched_track_pool:
                    current_domain_document_match_tracks.append(matched_track_pool)

            if current_domain_document_match_tracks:
                matched_chunks = sum(
                    len(step)
                    for track_pool in current_domain_document_match_tracks
                    for step in track_pool
                )
                log.info(
                    f"Domain {domain}: matched explanation source replay "
                    f"({len(current_domain_document_match_tracks)} tracks, "
                    f"{matched_chunks} scheduled source/paraphrase chunks)."
                )

        # Load prior-knowledge chapters for this domain (used as a one-pass injection block)
        per_chapter_chunks = []
        if with_prior_knowledge:
            pk_dir = f'../../data/{domain_source}/prior_knowledge/{domain}/'
            if os.path.isdir(pk_dir):
                def _chapter_index(name: str) -> int:
                    try:
                        return int(name.replace('chapter_', '').replace('.txt', ''))
                    except ValueError:
                        return 10**9
                chapter_files = sorted(
                    [f for f in os.listdir(pk_dir) if f.startswith('chapter_') and f.endswith('.txt')],
                    key=_chapter_index,
                )
                if prior_knowledge_cycle != "full":
                    try:
                        n = int(prior_knowledge_cycle)
                        chapter_files = chapter_files[:n]
                    except (TypeError, ValueError):
                        pass
                for cf in chapter_files:
                    with open(os.path.join(pk_dir, cf), 'r', encoding='utf-8') as f:
                        per_chapter_chunks.append(_chunk_explanation(f.read(), title_prefix=False))
                if prior_knowledge_match_document_track:
                    document_replay_chunks = list(source_chunks)
                    for chunks in paraphrased_chunks_by_doc:
                        document_replay_chunks.extend(chunks)
                    if document_replay_chunks:
                        doc_iter = cycle(document_replay_chunks)
                        matched_chapter_chunks = []
                        for chap in per_chapter_chunks:
                            matched_chapter_chunks.append([next(doc_iter) for _ in range(len(chap))])
                        log.info(
                            f"Domain {domain}: PK-match baseline — replacing {sum(len(c) for c in per_chapter_chunks)} "
                            f"PK chunks with same-shape doc-replay chunks across {len(per_chapter_chunks)} steps."
                        )
                        per_chapter_chunks = matched_chapter_chunks
                    else:
                        log.warning(f"--prior_knowledge_match_document_track set but no source/paraphrase chunks for domain '{domain}'.")
                        per_chapter_chunks = []
                else:
                    log.info(
                        f"Domain {domain}: loaded {len(chapter_files)} prior-knowledge chapters "
                        f"({sum(len(c) for c in per_chapter_chunks)} chunks)."
                    )
            else:
                log.warning(f"--with_prior_knowledge set but no chapters at {pk_dir}; skipping for domain '{domain}'.")
        domain_pk_chapter_chunks.append(per_chapter_chunks)

        # Append explanation structures for this domain
        domain_explanation_tracks.append(current_domain_tracks)
        domain_document_match_tracks.append(current_domain_document_match_tracks)
        domain_whole_explanation_batches.append(current_domain_whole_batch)

    # --- 3. Replay & Replications ---
    data_replay = None
    if fill_with_pretraining or pretraining_separators > 0 or fill_chunk_gaps:
        pretraining_data_type = strategy_args.get('pretraining_data_type', 'dclm')
        log.info(f"Creating Pretraining Data Replayer from '{pretraining_data_type}'...")
        data_replay = PretrainingDataReplay(f'../../data/olmo/{pretraining_data_type}_100M_tokens.npy')

    original_epochs = train_cfg.num_train_epochs
    replication_factor = 1
    if unique_document_batches and original_epochs > 1:
        if original_epochs < len(unique_document_batches):
            log.warning(
                "Requested %s knowledge-injection batches but data mix has %s view batches; "
                "running one full cycle instead.",
                original_epochs,
                len(unique_document_batches),
            )
        elif original_epochs % len(unique_document_batches) != 0:
            log.warning(
                "Requested %s knowledge-injection batches is not divisible by %s view batches; "
                "using floor replication.",
                original_epochs,
                len(unique_document_batches),
            )
        replication_factor = int(original_epochs / len(unique_document_batches))
        replication_factor = max(1, replication_factor)
        log.info(f"Replicating dataset {replication_factor} times.")

    view_batch_sizes = [len(batch) for batch in unique_document_batches]
    requested_batches = (
        len(unique_document_batches) * replication_factor if unique_document_batches else 0
    )
    log.info(
        "Data mix schedule: strategy=%s, view_batches=%s, view_batch_chunk_counts=%s, "
        "requested_num_train_epochs=%s, replication_factor=%s, scheduled_view_batches=%s",
        strategy_name,
        len(unique_document_batches),
        view_batch_sizes,
        original_epochs,
        replication_factor,
        requested_batches,
    )

    final_chunks = []
    has_track_explanations = any(len(tracks) > 0 for tracks in domain_explanation_tracks)
    has_document_match_tracks = any(len(tracks) > 0 for tracks in domain_document_match_tracks)
    has_whole_explanations = any(len(chunks) > 0 for chunks in domain_whole_explanation_batches)

    if explanations_insertion_strategy == "whole" and has_whole_explanations:
        if whole_explanations_insert_every_n <= 0:
            raise ValueError("--whole_explanations_insert_every_n must be a positive integer when using strategy 'whole'.")
        log.info("Using WHOLE Method (Insert explanation-only batch every N document batches).")
        final_chunks = replicate_and_interleave_whole_insert_every_n(
            doc_batches=unique_document_batches,
            domain_whole_explanation_batches=domain_whole_explanation_batches,
            insertion_every_n=whole_explanations_insert_every_n,
            replication_factor=replication_factor,
            data_replay=data_replay,
            pretraining_separators=pretraining_separators,
            train_cfg=train_cfg,
            tokenizer=tokenizer,
            log=log,
            test_script=test_script,
            fill_with_pretraining=fill_with_pretraining,
        )
    elif has_track_explanations or has_document_match_tracks:
        track_batches = domain_explanation_tracks
        if has_document_match_tracks:
            if has_track_explanations:
                track_batches = [
                    explanation_tracks + document_tracks
                    for explanation_tracks, document_tracks
                    in zip(domain_explanation_tracks, domain_document_match_tracks)
                ]
            else:
                track_batches = domain_document_match_tracks
        log.info("Using NEW Track Method (Independent Domain Tracks).")
        final_chunks = replicate_and_interleave_tracks(
            doc_batches=unique_document_batches,
            domain_explanation_tracks=track_batches,
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
        final_chunks = replicate_and_interleave_legacy(
            unique_document_batches=unique_document_batches,
            explanation_spliced_document_batches=explanation_spliced_document_batches,
            replication_factor=replication_factor,
            data_replay=data_replay,
            pretraining_separators=pretraining_separators,
            train_cfg=train_cfg,
            tokenizer=tokenizer,
            log=log,
            test_script=test_script,
            fill_with_pretraining=fill_with_pretraining,
        )

    # --- 3b. Prior-Knowledge Injection (one-pass, granular cadence) ---
    if with_prior_knowledge and any(domain_pk_chapter_chunks):
        world_size_pk = int(os.environ.get("WORLD_SIZE", "1"))
        global_effective_batch_size_pk = (
            train_cfg.per_device_train_batch_size
            * train_cfg.gradient_accumulation_steps
            * world_size_pk
        )
        max_chapters = max(len(d) for d in domain_pk_chapter_chunks)
        pk_step_batches: List[List[str]] = []
        for step_idx in range(max_chapters):
            step: List[str] = []
            for domain_chs in domain_pk_chapter_chunks:
                if step_idx < len(domain_chs):
                    step.extend(domain_chs[step_idx])
            if step:
                pk_step_batches.append(step)

        pk_chunks: List[str] = []
        pk_knowledge_chunks = 0
        for batch in pk_step_batches:
            pk_knowledge_chunks += len(batch)
            if fill_with_pretraining and data_replay is not None and global_effective_batch_size_pk > 0:
                batch = fill_up_batch_with_pretraining_chunks(
                    list(batch), data_replay, global_effective_batch_size_pk,
                    train_cfg.context_length, tokenizer,
                )
            pk_chunks.extend(batch)

        log.info(
            "Prior-knowledge injection: insertion=%s, chapters=%s, knowledge_chunks=%s, total_pk_chunks=%s",
            prior_knowledge_insertion, len(pk_step_batches), pk_knowledge_chunks, len(pk_chunks),
        )

        if prior_knowledge_insertion == "front":
            final_chunks = pk_chunks + final_chunks
        elif prior_knowledge_insertion == "end":
            final_chunks = final_chunks + pk_chunks
        elif prior_knowledge_insertion == "middle":
            if global_effective_batch_size_pk > 0:
                mid = (len(final_chunks) // 2 // global_effective_batch_size_pk) * global_effective_batch_size_pk
            else:
                mid = len(final_chunks) // 2
            final_chunks = final_chunks[:mid] + pk_chunks + final_chunks[mid:]
        else:
            raise ValueError(f"Unknown --prior_knowledge_insertion: {prior_knowledge_insertion}")

    # --- 4. Fill Gaps & Shuffle ---
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

    if shuffle_chunks_flag and final_chunks:
        random.Random(shuffle_seed).shuffle(final_chunks)
        log.info(f"Shuffled final training chunks with seed {shuffle_seed}.")

    train_cfg.num_train_epochs = 1
    dataset = Dataset.from_dict({"raw_text": final_chunks})
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    global_effective_batch_size = (
        train_cfg.per_device_train_batch_size
        * train_cfg.gradient_accumulation_steps
        * world_size
    )
    expected_steps = (
        (len(final_chunks) + global_effective_batch_size - 1) // global_effective_batch_size
        if global_effective_batch_size > 0
        else 0
    )
    log.info(
        "Final CPT dataset: chunks=%s, global_effective_batch_size=%s, expected_optimizer_steps=%s",
        len(final_chunks),
        global_effective_batch_size,
        expected_steps,
    )
    return dataset, train_cfg

def replicate_and_interleave_tracks(
    doc_batches: List[List[str]],
    domain_explanation_tracks: List[List[List[List[str]]]], 
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
    The New Way: "Track" based mixing with independent domain cycling AND multiple tracks per domain.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    effective_batch_size = (
        train_cfg.per_device_train_batch_size
        * train_cfg.gradient_accumulation_steps
        * world_size
    )
    total_batches = len(doc_batches) * replication_factor
    
    if total_batches == 0: return []

    # --- TRACK 1: DOCUMENTS ---
    track_docs = []
    for _ in range(replication_factor):
        track_docs.extend([list(b) for b in doc_batches])
        
    # --- TRACK 2+: EXPLANATIONS ---
    # Flatten all domains' tracks into a single list of active iterators
    all_iterators = []
    
    for domain_tracks in domain_explanation_tracks:
        if not domain_tracks:
            # If a domain has no explanations, it adds nothing to the batch
            all_iterators.append(repeat([]))
        else:
            # Add an iterator for EVERY track in this domain (e.g. Main + Offset)
            for track_pool in domain_tracks:
                if track_pool:
                    all_iterators.append(cycle(track_pool))
                else:
                    all_iterators.append(repeat([]))

    track_explanations = []
    
    for _ in range(total_batches):
        combined_explanation_batch = []
        
        # Pull from ALL tracks
        for iterator in all_iterators:
            track_chunks = next(iterator) 
            combined_explanation_batch.extend(track_chunks)
            
        track_explanations.append(combined_explanation_batch)

    # --- MERGE ---
    final_chunks = []
    scheduled_batches = 0
    total_knowledge_chunks = 0
    total_replay_fill_chunks = 0
    pure_replay_batches = 0
    max_knowledge_chunks = 0
    for i in range(total_batches):
        current_batch = track_docs[i] + track_explanations[i]
        knowledge_chunks = len(current_batch)
        scheduled_batches += 1
        total_knowledge_chunks += knowledge_chunks
        max_knowledge_chunks = max(max_knowledge_chunks, knowledge_chunks)
        if knowledge_chunks == 0:
            pure_replay_batches += 1
        
        if fill_with_pretraining:
            total_replay_fill_chunks += max(0, effective_batch_size - knowledge_chunks)
            current_batch = fill_up_batch_with_pretraining_chunks(
                current_batch, data_replay, effective_batch_size, train_cfg.context_length, tokenizer
            )
        
        final_chunks.extend(current_batch)
        
        if pretraining_separators > 0 and i < total_batches - 1:
             pretraining_fill = get_pretraining_batches(
                data_replay, pretraining_separators, effective_batch_size,
                train_cfg.context_length, tokenizer
             )
             final_chunks.extend(pretraining_fill)

    log.info(
        "Replay fill summary: scheduled_batches=%s, max_knowledge_chunks_per_batch=%s, "
        "knowledge_chunks=%s, replay_fill_chunks=%s, pure_replay_batches=%s, "
        "effective_batch_size=%s, context_length=%s",
        scheduled_batches,
        max_knowledge_chunks,
        total_knowledge_chunks,
        total_replay_fill_chunks,
        pure_replay_batches,
        effective_batch_size,
        train_cfg.context_length,
    )
    return final_chunks

def replicate_and_interleave_legacy(
    unique_document_batches: List[List[str]],
    explanation_spliced_document_batches: Optional[List[List[str]]],
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
    The old coupled-batch schedule. If explanation-spliced batches exist, use
    them for every replication; otherwise use the ordinary document batches.
    """
    final_chunks = []
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    effective_batch_size = (
        train_cfg.per_device_train_batch_size
        * train_cfg.gradient_accumulation_steps
        * world_size
    )

    scheduled_batches = 0
    total_knowledge_chunks = 0
    total_replay_fill_chunks = 0
    pure_replay_batches = 0
    max_knowledge_chunks = 0

    for rep in range(replication_factor):
        if test_script: log.info(f"--- Creating replication {rep + 1}/{replication_factor} ---")
        
        batches_for_this_rep = unique_document_batches
        
        if explanation_spliced_document_batches:
            batches_for_this_rep = explanation_spliced_document_batches

        for i, batch in enumerate(batches_for_this_rep):
            current_batch = list(batch)
            knowledge_chunks = len(current_batch)
            scheduled_batches += 1
            total_knowledge_chunks += knowledge_chunks
            max_knowledge_chunks = max(max_knowledge_chunks, knowledge_chunks)
            if knowledge_chunks == 0:
                pure_replay_batches += 1
            
            if fill_with_pretraining:
                total_replay_fill_chunks += max(0, effective_batch_size - knowledge_chunks)
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

    log.info(
        "Replay fill summary: scheduled_batches=%s, max_knowledge_chunks_per_batch=%s, "
        "knowledge_chunks=%s, replay_fill_chunks=%s, pure_replay_batches=%s, "
        "effective_batch_size=%s, context_length=%s",
        scheduled_batches,
        max_knowledge_chunks,
        total_knowledge_chunks,
        total_replay_fill_chunks,
        pure_replay_batches,
        effective_batch_size,
        train_cfg.context_length,
    )
    return final_chunks

def replicate_and_interleave_whole_insert_every_n(
    doc_batches: List[List[str]],
    domain_whole_explanation_batches: List[List[str]],
    insertion_every_n: int,
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
    Whole strategy:
    - Keep document batches as-is.
    - Insert a standalone explanation-only batch every N document batches.
    """
    if insertion_every_n <= 0:
        raise ValueError("insertion_every_n must be a positive integer.")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    effective_batch_size = (
        train_cfg.per_device_train_batch_size
        * train_cfg.gradient_accumulation_steps
        * world_size
    )
    total_doc_batches = len(doc_batches) * replication_factor
    if total_doc_batches == 0:
        return []

    replicated_doc_batches = []
    for _ in range(replication_factor):
        replicated_doc_batches.extend([list(b) for b in doc_batches])

    combined_explanation_batch = []
    for domain_batch in domain_whole_explanation_batches:
        combined_explanation_batch.extend(domain_batch)

    scheduled_batches = []
    inserted_explanation_batches = 0
    for doc_idx, doc_batch in enumerate(replicated_doc_batches):
        scheduled_batches.append(doc_batch)
        if combined_explanation_batch and ((doc_idx + 1) % insertion_every_n == 0):
            scheduled_batches.append(list(combined_explanation_batch))
            inserted_explanation_batches += 1

    log.info(
        f"Whole strategy scheduled {len(replicated_doc_batches)} document batches and "
        f"{inserted_explanation_batches} explanation-only batches (every {insertion_every_n} steps)."
    )

    final_chunks = []
    scheduled_batch_count = 0
    total_knowledge_chunks = 0
    total_replay_fill_chunks = 0
    pure_replay_batches = 0
    max_knowledge_chunks = 0
    for batch_idx, batch in enumerate(scheduled_batches):
        current_batch = list(batch)
        knowledge_chunks = len(current_batch)
        scheduled_batch_count += 1
        total_knowledge_chunks += knowledge_chunks
        max_knowledge_chunks = max(max_knowledge_chunks, knowledge_chunks)
        if knowledge_chunks == 0:
            pure_replay_batches += 1

        if fill_with_pretraining:
            total_replay_fill_chunks += max(0, effective_batch_size - knowledge_chunks)
            current_batch = fill_up_batch_with_pretraining_chunks(
                current_batch, data_replay, effective_batch_size, train_cfg.context_length, tokenizer
            )

        final_chunks.extend(current_batch)

        if pretraining_separators > 0 and batch_idx < len(scheduled_batches) - 1:
            pretraining_fill = get_pretraining_batches(
                data_replay,
                pretraining_separators,
                effective_batch_size,
                train_cfg.context_length,
                tokenizer,
            )
            final_chunks.extend(pretraining_fill)

    log.info(
        "Replay fill summary: scheduled_batches=%s, max_knowledge_chunks_per_batch=%s, "
        "knowledge_chunks=%s, replay_fill_chunks=%s, pure_replay_batches=%s, "
        "effective_batch_size=%s, context_length=%s",
        scheduled_batch_count,
        max_knowledge_chunks,
        total_knowledge_chunks,
        total_replay_fill_chunks,
        pure_replay_batches,
        effective_batch_size,
        train_cfg.context_length,
    )
    return final_chunks
