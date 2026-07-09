# add .. path 

# pip install flash-attn --no-build-isolation
# pip install git+https://github.com/huggingface/trl
# pip install pydantic datasets==3.6.0 peft bitsandbytes wandb matplotlib seaborn liger-kernel scikit-learn openai
# git clone https://github.com/jiosephlee/transformers; pip install .[torch]

import os
import sys
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
sys.path.append('../..')
import utils.llm_training as llm_training
import utils.data_preparation as data_preparation
import utils.model_setup as model_setup
import utils.llm_configs as llm_configs
from utils import experiment_utils
from utils import probe_paths
import argparse
import wandb
import logging
import pandas as pd
import torch
from transformers import AutoTokenizer
# wandb.init(project="fine_tuning_study")
# Local callback types are no longer used directly; delegated to utils.experiment_utils

SUPPORTED_HIGH_LEVEL_DOMAINS = ("arxiv", "legal", "medical")
DOMAIN_OVERRIDE_ARG_BY_SOURCE = {
    "arxiv": "override_arxiv_domain",
    "legal": "override_legal_domain",
    "medical": "override_medical_domain",
}
DEFAULT_WANDB_PROJECT = "v9_refined"
DEFAULT_WANDB_GROUP = "finetuning_official"
DEFAULT_WANDB_PANEL_SOURCES = ("legal", "arxiv", "medical")
DEFAULT_KNOWLEDGE_PROBES_VERSION = "v14"
DEFAULT_KNOWLEDGE_PROBE_FILENAME_SUFFIX = ""
DEFAULT_PARAPHRASED_KNOWLEDGE_PROBE_FILENAME_SUFFIX = "_paraphrased"
DEFAULT_KNOWLEDGE_PROBE_VARIANT = "short_targets"
KNOWLEDGE_PROBE_VARIANT_SUFFIXES = {
    "standard": "",
    "low_overlap_strict": "_low_overlap_strict",
    "short_targets": "_short_targets",
}
KNOWLEDGE_PROBE_VARIANT_DEFAULT_VERSION = {
    "low_overlap_strict": "v14",
    "short_targets": "v14",
}
DEFAULT_MCQA_PROBES_VERSION = "v15"
DEFAULT_INFERENCE_MCQA_PROBES_VERSION = "v14"
DEFAULT_INFERENCE_PROBES_VERSION = "v11"
DEFAULT_INFERENCE_PROBE_FILENAME_SUFFIX = ""
DEFAULT_INFERENCE_PROBE_VARIANT = "reviewed"
INFERENCE_PROBE_VARIANT_SUFFIXES = {
    "standard": "",
    "reviewed": "_reviewed",
}
INFERENCE_PROBE_VARIANT_DEFAULT_VERSION = {
    "reviewed": "v11",
}
REQUIRED_KNOWLEDGE_PROBE_COLUMNS = ("fact", "probe", "target")
REQUIRED_INFERENCE_PROBE_COLUMNS = ("fact", "probe", "target")
REQUIRED_MCQA_PROBE_COLUMNS = ("formatted_question", "correct_label")


def _coerce_version_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_values = value.replace(",", " ").split()
    else:
        raw_values = []
        for item in value:
            raw_values.extend(str(item).replace(",", " ").split())
    return list(dict.fromkeys(v for v in raw_values if v))


def _safe_tag(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in value)


def _construct_eval_bundle_name(args) -> str:
    if getattr(args, "inference_mcqa_probes", False):
        inference_mcqa_versions = _coerce_version_list(
            getattr(
                args,
                "inference_mcqa_probes_version",
                DEFAULT_INFERENCE_MCQA_PROBES_VERSION,
            )
        )
        if inference_mcqa_versions:
            return f"inf_mcqa_{'+'.join(_safe_tag(v) for v in inference_mcqa_versions)}"
    return "default"


def _construct_legacy_probe_bundle_name(args) -> str:
    probes_version = f"probes_{args.knowledge_probes_version}"
    if args.knowledge_probe_filename_suffix:
        probes_version += args.knowledge_probe_filename_suffix
    if getattr(args, "paraphrased_knowledge_probes", False):
        probes_version += (
            f"_para_{args.paraphrased_knowledge_probes_version}"
            f"{args.paraphrased_knowledge_probe_filename_suffix}"
        )
    if not getattr(args, "disable_inference_probes", False):
        probes_version += f"_inf_{args.inference_probes_version}{args.inference_probe_filename_suffix}"
    if getattr(args, "mcqa_probes", False):
        mcqa_probes_version = getattr(
            args,
            "mcqa_probes_version",
            args.knowledge_probes_version,
        )
        probes_version += f"_mcqa_{mcqa_probes_version}"
        mcqa_prompt_column = getattr(args, "mcqa_prompt_column", "formatted_question")
        if mcqa_prompt_column != "formatted_question":
            probes_version += f"_prompt_{_safe_tag(mcqa_prompt_column)}"
    if getattr(args, "inference_mcqa_probes", False):
        inference_mcqa_probes_versions = _coerce_version_list(
            getattr(
                args,
                "inference_mcqa_probes_version",
                DEFAULT_INFERENCE_MCQA_PROBES_VERSION,
            )
        )
        probes_version += f"_inf_mcqa_{'+'.join(_safe_tag(v) for v in inference_mcqa_probes_versions)}"
        inference_mcqa_prompt_column = getattr(args, "inference_mcqa_prompt_column", "formatted_question")
        if inference_mcqa_prompt_column != "formatted_question":
            probes_version += f"_inf_prompt_{_safe_tag(inference_mcqa_prompt_column)}"
    return probes_version


def _encode_probe_text(tokenizer, text: str) -> List[int]:
    return tokenizer.encode(str(text), add_special_tokens=False)


def _validate_probe_rows(
    probe_df: pd.DataFrame,
    required_columns: Tuple[str, ...],
    tokenizer=None,
) -> List[str]:
    missing_columns = [
        column for column in required_columns
        if column not in probe_df.columns
    ]
    if missing_columns:
        return [f"missing columns {missing_columns}"]

    null_mask = probe_df[list(required_columns)].isna().any(axis=1)
    messages = []
    if null_mask.any():
        messages.append(
            "null values in required columns at row indices "
            f"{probe_df.index[null_mask].tolist()[:10]}"
        )

    if tokenizer is None:
        reconstructed = (
            probe_df["probe"].astype(str)
            + probe_df["target"].astype(str)
        )
        mismatch_mask = reconstructed != probe_df["fact"].astype(str)
        if mismatch_mask.any():
            messages.append(
                "probe + target != fact at row indices "
                f"{probe_df.index[mismatch_mask].tolist()[:10]}"
            )
    else:
        mismatch_indices = []
        for row_idx, row in probe_df.iterrows():
            probe_ids = _encode_probe_text(tokenizer, row["probe"])
            target_ids = _encode_probe_text(tokenizer, row["target"])
            fact_ids = _encode_probe_text(tokenizer, row["fact"])
            if probe_ids + target_ids != fact_ids:
                mismatch_indices.append(row_idx)
        if mismatch_indices:
            messages.append(
                "tokenize(probe) + tokenize(target) != tokenize(fact) "
                f"at row indices {mismatch_indices[:10]}"
            )

    return messages


def load_probe_validation_tokenizer(args, log):
    """Load only the tokenizer needed to validate probe/target/fact boundaries."""
    tokenizer_id = "jiosephlee/olmo2-lima" if getattr(args, "use_existing_lima_tokenizer", False) else args.model_id
    log.info(f"Loading tokenizer '{tokenizer_id}' for probe boundary validation.")
    return AutoTokenizer.from_pretrained(
        tokenizer_id,
        trust_remote_code=True,
        use_fast=True,
        cache_dir=getattr(args, "cache_dir", None),
    )


def distributed_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def distributed_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def is_world_process_zero() -> bool:
    return distributed_rank() == 0


def distributed_barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def _domain_catalog_root(source: str, args) -> str:
    if args.prior_knowledge:
        return f'../../data/{source}/prior_knowledge'
    if args.raw:
        return f'../../data/{source}/raw'
    if args.semi_cleaned:
        semicleaned_root = f'../../data/{source}/semicleaned_{args.semi_cleaned}'
        if os.path.isdir(semicleaned_root):
            return semicleaned_root
    return f'../../data/{source}/cleaned'


def _discover_domains_for_source(source: str, args) -> List[str]:
    """Discover concrete document IDs for one high-level source."""
    root = _domain_catalog_root(source, args)
    if not os.path.isdir(root):
        return []
    if args.prior_knowledge:
        return sorted(
            name for name in os.listdir(root)
            if os.path.isdir(os.path.join(root, name))
        )
    return sorted(
        os.path.splitext(name)[0]
        for name in os.listdir(root)
        if os.path.isfile(os.path.join(root, name)) and name.endswith(('.txt', '.tex'))
    )


def resolve_domains_and_sources(args, log) -> Tuple[Optional[List[str]], Dict[str, str]]:
    """
    Resolve domains from per-source overrides.
    Defaults to ALL concrete domains under each source when no override is provided.
    """
    resolved_domains: List[str] = []
    domain_sources: Dict[str, str] = {}
    seen = set()

    def _register(domain: str, source: str):
        if domain in domain_sources and domain_sources[domain] != source:
            raise ValueError(
                f"Domain '{domain}' is ambiguous across sources "
                f"({domain_sources[domain]} vs {source})."
            )
        domain_sources[domain] = source
        if domain not in seen:
            resolved_domains.append(domain)
            seen.add(domain)

    included_sources = tuple(getattr(args, "include_sources", SUPPORTED_HIGH_LEVEL_DOMAINS))
    for source in included_sources:
        catalog = _discover_domains_for_source(source, args)
        override_arg = DOMAIN_OVERRIDE_ARG_BY_SOURCE[source]
        requested = getattr(args, override_arg)

        if requested:
            missing = [name for name in requested if name not in set(catalog)]
            if missing:
                raise ValueError(
                    f"Unknown {source} domain(s) in --{override_arg}: {missing}. "
                    f"Available {source} domains: {catalog}"
                )
            selected = requested
        else:
            selected = catalog

        if not selected:
            log.info(f"No domains selected for source '{source}'.")
            continue

        for domain in selected:
            _register(domain, source)

    if not resolved_domains:
        raise ValueError(
            "No domains resolved. Check your data folders or override values for "
            "--override_arxiv_domain/--override_legal_domain/--override_medical_domain."
        )

    log.info(f"Resolved domains: {resolved_domains}")
    log.info(f"Resolved domain sources: {domain_sources}")
    return resolved_domains, domain_sources


def validate_selected_knowledge_probes(args, log, tokenizer=None) -> None:
    """Fail early if the selected factual probe files are absent or malformed."""
    missing_paths = []
    malformed = []
    total_rows = 0

    for domain in args.resolved_domains:
        domain_source = args.domain_data_sources.get(domain)
        probe_path = probe_paths.resolve_knowledge_probe_path(
            domain,
            args.knowledge_probes_version,
            domain_source=domain_source,
            filename_suffix=args.knowledge_probe_filename_suffix,
        )

        if not os.path.exists(probe_path):
            missing_paths.append(str(probe_path))
            continue

        probe_df = pd.read_csv(probe_path)
        malformed_messages = _validate_probe_rows(
            probe_df,
            REQUIRED_KNOWLEDGE_PROBE_COLUMNS,
            tokenizer=tokenizer,
        )
        if malformed_messages:
            malformed.append((str(probe_path), malformed_messages))
            continue

        total_rows += len(probe_df)

    if missing_paths or malformed:
        details = []
        if missing_paths:
            details.append(
                "Missing knowledge probe files:\n"
                + "\n".join(f"  - {path}" for path in missing_paths)
            )
        if malformed:
            details.append(
                "Malformed knowledge probe files:\n"
                + "\n".join(
                    f"  - {path}: {'; '.join(messages)}"
                    for path, messages in malformed
                )
            )
        raise ValueError("\n".join(details))

    log.info(
        f"Validated {total_rows} factual probes from "
        f"probes_{args.knowledge_probes_version}"
        f"{args.knowledge_probe_filename_suffix}.csv across "
        f"{len(args.resolved_domains)} domains."
    )


def validate_selected_paraphrased_knowledge_probes(args, log, tokenizer=None) -> None:
    """Fail early if optional paraphrased factual probe files are absent or malformed."""
    if not getattr(args, "paraphrased_knowledge_probes", False):
        return

    missing_paths = []
    malformed = []
    total_rows = 0
    version = args.paraphrased_knowledge_probes_version
    suffix = args.paraphrased_knowledge_probe_filename_suffix

    for domain in args.resolved_domains:
        domain_source = args.domain_data_sources.get(domain)
        probe_path = probe_paths.resolve_knowledge_probe_path(
            domain,
            version,
            domain_source=domain_source,
            filename_suffix=suffix,
        )
        if not os.path.exists(probe_path):
            missing_paths.append(str(probe_path))
            continue

        probe_df = pd.read_csv(probe_path)
        malformed_messages = _validate_probe_rows(
            probe_df,
            REQUIRED_KNOWLEDGE_PROBE_COLUMNS,
            tokenizer=tokenizer,
        )
        if malformed_messages:
            malformed.append((str(probe_path), malformed_messages))
            continue
        total_rows += len(probe_df)

    if missing_paths or malformed:
        details = []
        if missing_paths:
            details.append(
                "Missing paraphrased knowledge probe files:\n"
                + "\n".join(f"  - {path}" for path in missing_paths)
            )
        if malformed:
            details.append(
                "Malformed paraphrased knowledge probe files:\n"
                + "\n".join(
                    f"  - {path}: {'; '.join(messages)}"
                    for path, messages in malformed
                )
            )
        raise ValueError("\n".join(details))

    log.info(
        f"Validated {total_rows} paraphrased factual probes from "
        f"probes_{version}{suffix}.csv across {len(args.resolved_domains)} domains."
    )


def apply_knowledge_probe_variant(args) -> None:
    """Normalize first-class factual probe variant flags into version/suffix fields."""
    if getattr(args, "use_low_overlap_knowledge_probes", False):
        args.knowledge_probe_variant = "low_overlap_strict"

    variant = getattr(args, "knowledge_probe_variant", DEFAULT_KNOWLEDGE_PROBE_VARIANT)
    if variant not in KNOWLEDGE_PROBE_VARIANT_SUFFIXES:
        raise ValueError(
            f"Unknown knowledge_probe_variant={variant!r}. "
            f"Expected one of {sorted(KNOWLEDGE_PROBE_VARIANT_SUFFIXES)}."
        )

    if variant != "standard":
        default_version = KNOWLEDGE_PROBE_VARIANT_DEFAULT_VERSION.get(variant)
        if default_version and args.knowledge_probes_version == DEFAULT_KNOWLEDGE_PROBES_VERSION:
            args.knowledge_probes_version = default_version
        variant_suffix = KNOWLEDGE_PROBE_VARIANT_SUFFIXES[variant]
        if args.knowledge_probe_filename_suffix and args.knowledge_probe_filename_suffix != variant_suffix:
            raise ValueError(
                f"--knowledge_probe_variant {variant} implies suffix {variant_suffix!r}, "
                f"but --knowledge_probe_filename_suffix was {args.knowledge_probe_filename_suffix!r}."
            )
        args.knowledge_probe_filename_suffix = variant_suffix
    elif args.knowledge_probe_filename_suffix:
        args.knowledge_probe_variant = "custom_suffix"


def apply_inference_probe_variant(args) -> None:
    """Normalize first-class inference probe variant flags into version/suffix fields."""
    if getattr(args, "use_reviewed_inference_probes", False):
        args.inference_probe_variant = "reviewed"

    variant = getattr(args, "inference_probe_variant", DEFAULT_INFERENCE_PROBE_VARIANT)
    if variant not in INFERENCE_PROBE_VARIANT_SUFFIXES:
        raise ValueError(
            f"Unknown inference_probe_variant={variant!r}. "
            f"Expected one of {sorted(INFERENCE_PROBE_VARIANT_SUFFIXES)}."
        )

    if variant != "standard":
        default_version = INFERENCE_PROBE_VARIANT_DEFAULT_VERSION.get(variant)
        if default_version and args.inference_probes_version == DEFAULT_INFERENCE_PROBES_VERSION:
            args.inference_probes_version = default_version
        variant_suffix = INFERENCE_PROBE_VARIANT_SUFFIXES[variant]
        if args.inference_probe_filename_suffix and args.inference_probe_filename_suffix != variant_suffix:
            raise ValueError(
                f"--inference_probe_variant {variant} implies suffix {variant_suffix!r}, "
                f"but --inference_probe_filename_suffix was {args.inference_probe_filename_suffix!r}."
            )
        args.inference_probe_filename_suffix = variant_suffix
    elif args.inference_probe_filename_suffix:
        args.inference_probe_variant = "custom_suffix"


def validate_selected_mcqa_probe_family(
    args,
    log,
    *,
    enabled_attr: str,
    version_attr: str,
    prompt_column_attr: str,
    probe_kind: str,
    label: str,
) -> None:
    """Fail early if enabled MCQA probe files are absent or malformed."""
    if not getattr(args, enabled_attr, False):
        return

    missing_paths = []
    malformed = []
    total_rows = 0
    mcqa_prompt_column = getattr(args, prompt_column_attr, "formatted_question")
    mcqa_probes_versions = _coerce_version_list(getattr(args, version_attr))
    required_columns = tuple(dict.fromkeys(REQUIRED_MCQA_PROBE_COLUMNS + (mcqa_prompt_column,)))

    for domain in args.resolved_domains:
        domain_source = args.domain_data_sources.get(domain)
        for mcqa_probes_version in mcqa_probes_versions:
            probe_path = probe_paths.resolve_mcqa_probe_path(
                probe_kind,
                domain,
                mcqa_probes_version,
                domain_source=domain_source,
            )

            if not os.path.exists(probe_path):
                missing_paths.append(str(probe_path))
                continue

            probe_df = pd.read_csv(probe_path)
            missing_columns = [
                column for column in required_columns
                if column not in probe_df.columns
            ]
            if missing_columns:
                malformed.append((str(probe_path), missing_columns))
                continue

            total_rows += len(probe_df)

    if missing_paths or malformed:
        details = []
        if missing_paths:
            details.append(
                "Missing MCQA probe files:\n"
                + "\n".join(f"  - {path}" for path in missing_paths)
            )
        if malformed:
            details.append(
                "Malformed MCQA probe files:\n"
                + "\n".join(
                    f"  - {path}: missing columns {columns}"
                    for path, columns in malformed
                )
            )
        raise ValueError("\n".join(details))

    log.info(
        f"Validated {total_rows} {label} MCQA probes from "
        f"{probe_kind}/probes_{'+'.join(mcqa_probes_versions)}_mcqa.csv using prompt column "
        f"'{mcqa_prompt_column}' across {len(args.resolved_domains)} domains."
    )


def validate_selected_mcqa_probes(args, log) -> None:
    validate_selected_mcqa_probe_family(
        args,
        log,
        enabled_attr="mcqa_probes",
        version_attr="mcqa_probes_version",
        prompt_column_attr="mcqa_prompt_column",
        probe_kind="facts",
        label="factual",
    )
    validate_selected_mcqa_probe_family(
        args,
        log,
        enabled_attr="inference_mcqa_probes",
        version_attr="inference_mcqa_probes_version",
        prompt_column_attr="inference_mcqa_prompt_column",
        probe_kind="inference",
        label="inference",
    )


def _selected_inference_probe_paths(domain: str, domain_source: str, args) -> List[str]:
    """Return the inference probe CSVs required by the selected v9 settings."""
    inference_probe_subset = getattr(args, "inference_probe_subset", "all")
    inference_probes_version = args.inference_probes_version
    inference_probe_filename_suffix = getattr(args, "inference_probe_filename_suffix", "")

    if inference_probe_subset in {"test", "type_split_test"}:
        base_dir = str(probe_paths.resolve_probe_dir("inference", domain, domain_source))
        if inference_probe_subset == "test":
            return [
                os.path.join(base_dir, f"train_probes_{inference_probes_version}{inference_probe_filename_suffix}.csv"),
                os.path.join(base_dir, f"test_probes_{inference_probes_version}{inference_probe_filename_suffix}.csv"),
            ]
        return [
            os.path.join(base_dir, f"type_split_train_probes_{inference_probes_version}{inference_probe_filename_suffix}.csv"),
            os.path.join(base_dir, f"type_split_test_probes_{inference_probes_version}{inference_probe_filename_suffix}.csv"),
        ]

    candidates = probe_paths.resolve_inference_probe_candidates(
        domain,
        inference_probes_version,
        domain_source=domain_source,
        filename_suffix=inference_probe_filename_suffix,
    )
    for candidate in candidates:
        if candidate.exists():
            return [str(candidate)]
    return [str(candidates[0])]


def validate_selected_inference_probes(args, log) -> None:
    """Fail early if enabled inference probe files are absent or malformed."""
    if getattr(args, "disable_inference_probes", False):
        return

    missing_paths = []
    malformed = []
    total_rows = 0

    for domain in args.resolved_domains:
        domain_source = args.domain_data_sources.get(domain)
        for probe_path in _selected_inference_probe_paths(domain, domain_source, args):
            if not os.path.exists(probe_path):
                missing_paths.append(probe_path)
                continue

            probe_df = pd.read_csv(probe_path)
            missing_columns = [
                column for column in REQUIRED_INFERENCE_PROBE_COLUMNS
                if column not in probe_df.columns
            ]
            if missing_columns:
                malformed.append((probe_path, missing_columns))
                continue

            total_rows += len(probe_df)

    if missing_paths or malformed:
        details = []
        if missing_paths:
            details.append(
                "Missing inference probe files:\n"
                + "\n".join(f"  - {path}" for path in missing_paths)
            )
        if malformed:
            details.append(
                "Malformed inference probe files:\n"
                + "\n".join(
                    f"  - {path}: missing columns {columns}"
                    for path, columns in malformed
                )
            )
        raise ValueError("\n".join(details))

    log.info(
        f"Validated {total_rows} inference probes from "
        f"probes_{args.inference_probes_version}"
        f"{args.inference_probe_filename_suffix}.csv "
        f"({args.inference_probe_subset}) across "
        f"{len(args.resolved_domains)} domains."
    )


def construct_experiment_name(args):
    """Construct experiment path as a nested directory structure."""
    
    # 1. Training Type: e.g., 'peft', 'full'
    training_type = "full" if args.full_finetuning else "peft"
    
    # 2. Model Size: e.g., '1b', '7b'
    model_id_lower = args.model_id.lower()
    if "olmo" in model_id_lower:
        if "1b" in model_id_lower:
            model_size = "1b"
        elif "7b" in model_id_lower:
            model_size = "7b"
        else:
            model_size = args.model_id.replace('/', '_')
    else:
        model_size = args.model_id.replace('/', '_')
    
    eval_bundle = _construct_eval_bundle_name(args)

    # 5. Data Mix: e.g., 'source_only', 'para9', 'para9_expl'
    if args.num_paraphrased_texts > 0:
        data_mix_base = f"para{args.num_paraphrased_texts}"
        if args.with_specific_explanation:
            # Handle multiple explanation types
            if isinstance(args.with_specific_explanation, list):
                expl_str = "+".join(args.with_specific_explanation)
            else:
                expl_str = args.with_specific_explanation
            
            data_mix = f"{data_mix_base}_expl_{expl_str}"
            
        else:
            data_mix = data_mix_base
        
        if args.with_specific_explanation and args.times_explanations > 1:
            data_mix += f"_x{args.times_explanations}"

        if args.with_specific_explanation:
            if args.explanations_insertion_strategy == "granular":
                if args.granular_explanations_cycle == "full":
                    data_mix += "_cyclefull"
                elif isinstance(args.granular_explanations_cycle, int) and args.granular_explanations_cycle > 0:
                    data_mix += f"_cycle{args.granular_explanations_cycle}"

                if args.granular_explanations_num_tracks > 1:
                    data_mix += f"_tracks{args.granular_explanations_num_tracks}"
                if args.explanation_granularity == "chunk":
                    data_mix += f"_granchunk{args.explanation_track_size_by_chunk}"
            elif args.explanations_insertion_strategy == "granular_queue":
                data_mix += f"_insgranular_queue_tracks{args.granular_explanations_num_tracks}"
                if args.explanation_granularity == "chunk":
                    data_mix += f"_chunk{args.explanation_track_size_by_chunk}"

            if args.explanations_insertion_strategy not in ("granular", "granular_queue"):
                data_mix += f"_ins{args.explanations_insertion_strategy}"

            if args.explanations_insertion_strategy == "whole":
                data_mix += f"_every{args.whole_explanations_insert_every_n}"
            if args.match_explanation_source_replay:
                data_mix += "_srcmatch"

        if args.document_track_baseline:
            data_mix += "_docmatch_expl"
            if args.document_match_insert_content != "document":
                data_mix += f"_insert{args.document_match_insert_content}"
            if args.explanation_granularity == "chunk":
                data_mix += f"_granchunk{args.explanation_track_size_by_chunk}"

    else:
        data_mix = "source_only"
        if args.document_track_baseline:
            data_mix += "_docmatch_expl"
            if args.document_match_insert_content != "document":
                data_mix += f"_insert{args.document_match_insert_content}"
            if args.explanation_granularity == "chunk":
                data_mix += f"_granchunk{args.explanation_track_size_by_chunk}"

    # 6. Domains (per source): compact, avoids giant path names when "all" is used.
    selection_tags = []
    included_sources = tuple(getattr(args, "include_sources", SUPPORTED_HIGH_LEVEL_DOMAINS))
    for source in included_sources:
        override_arg = DOMAIN_OVERRIDE_ARG_BY_SOURCE[source]
        chosen = getattr(args, override_arg)
        if not chosen:
            selection_tags.append(f"{source}_all")
        elif len(chosen) == 1:
            selection_tags.append(f"{source}_{chosen[0]}")
        else:
            selection_tags.append(f"{source}_{len(chosen)}")
    domains = f"domains_{'-'.join(selection_tags)}"

    # 7. Epochs: e.g., 'e1'
    epochs = f"e{args.num_train_epochs}"

    # 8. Batch size and learning rate
    training_params = f"bs{args.effective_batch_size_for_cpt}_lr{args.learning_rate:g}"
    if args.constant_lr:
        training_params += "_const"

    # 9. Overlap ratio
    if args.overlap_sections:
        overlap_info = f"overlap_{args.overlap_ratio}"
    else:
        overlap_info = "no_overlap"

    path_parts = [
        training_type,
        model_size,
        data_mix,
    ]

    # Add pretraining strategy if applicable
    if args.separate_batches_with_pretraining > 0:
        pretrain_info = f"sep_{args.separate_batches_with_pretraining}_{args.pretraining_data_type}"
        path_parts.append(pretrain_info)
    elif args.fill_batches_with_pretraining:
        pretrain_info = f"fill_{args.pretraining_data_type}"
        path_parts.append(pretrain_info)

    path_parts.extend([
        domains,
        epochs,
        training_params,
        overlap_info
    ])
    
    # Add semi_cleaned info if applicable
    if args.semi_cleaned:
        path_parts.append(f"semi_cleaned_{args.semi_cleaned}")

    # Suffix becomes the final leaf directory name for the run
    run_name = args.custom_suffix if args.custom_suffix else datetime.now().strftime('%m_%d_%H_%M')
    # Append shuffle mode marker to run name for easy propagation to wandb
    shuffle_marker = ''
    if getattr(args, 'word_shuffled_papers', False):
        shuffle_marker = '_shuffle_words'
    elif getattr(args, 'sentence_shuffled_papers', False):
        shuffle_marker = '_shuffle_sentences'
    elif getattr(args, 'shuffled_papers', False):
        shuffle_marker = '_shuffle'
    elif getattr(args, 'paragraph_shuffled_papers', False):
        shuffle_marker = '_shuffle_paragraphs'

    path_parts.append(run_name + shuffle_marker)
    path_parts.extend(["eval_bundles", eval_bundle])
    
    return os.path.join(*path_parts)



def continue_pretraining(model, tokenizer, log, args, train: bool = True):
    world_size = distributed_world_size()
    global_micro_batch_size = args.device_batch_size * world_size
    assert args.effective_batch_size_for_cpt % global_micro_batch_size == 0, (
        "Effective batch size for CPT must be divisible by "
        "device_batch_size * WORLD_SIZE."
    )
    grad_accum_steps = args.effective_batch_size_for_cpt // global_micro_batch_size
    log.info(
        "Distributed CPT batch setup: world_size=%s, per_device_batch=%s, "
        "grad_accum_steps=%s, global_effective_batch=%s",
        world_size,
        args.device_batch_size,
        grad_accum_steps,
        args.device_batch_size * grad_accum_steps * world_size,
    )

    # --- Continued Pretraining Configuration ---
    training_config_kwargs = {
        "run_name": args.experiment_name,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "logging_steps": 1,
        "gradient_checkpointing": args.gradient_checkpointing,
        "per_device_train_batch_size": args.device_batch_size,
        "context_length": args.context_length_for_cpt,
        "weight_decay": 0.1,
        "gradient_accumulation_steps": grad_accum_steps,
        "train_sampling_strategy": "sequential",
        "reverse_ffd_packing": False,
        "remove_unused_columns": False,
        "packing": True,
        "padding_free": True,
        "report_to": "wandb" if not args.test_script else "none",
        "activation_offloading": args.activation_offloading or args.offload_to_cpu,
        "compile": args.compile_model,
        "loss_type": args.sft_loss_type,
    }
    if args.constant_lr:
        training_config_kwargs["lr_scheduler_type"] = "constant_with_warmup"
    else:
        training_config_kwargs["lr_scheduler_type"] = "cosine_with_min_lr"
        training_config_kwargs["lr_scheduler_kwargs"] = {
            "min_lr_rate": args.lr_scheduler_min_lr_ratio
        }
    
    if args.overrule_warmup_via_steps:
        training_config_kwargs["warmup_steps"] = args.overrule_warmup_via_steps
    else:
        training_config_kwargs["warmup_ratio"] = 0.1
    
    training_config = llm_configs.TrainingConfig(**training_config_kwargs)

    # --- Load Probe Data ---
    callbacks_to_use = experiment_utils.setup_callbacks(
        domains=args.resolved_domains,
        tokenizer=tokenizer, 
        log=log, 
        args=args, 
        is_lima=False,
    )
    
    # --- Load the Texts and Fine-Tune ---
    # --- Determine Training Strategy ---
    if args.prior_knowledge:
        # Use prior-knowledge textbooks as the source instead of arXiv papers.
        strategy_args = {
            "override_domains": args.resolved_domains,
            "domain_data_sources": args.domain_data_sources,
            "fill_batches_with_pretraining": args.fill_batches_with_pretraining,
            "separate_batches_with_pretraining": args.separate_batches_with_pretraining,
            "pretraining_data_type": args.pretraining_data_type,
            "test_script": args.test_script,
        }
        strategy_name = "PriorKnowledge"
    else:
        strategy_args = {
            "num_paraphrased_texts": args.num_paraphrased_texts,
            "override_domains": args.resolved_domains,
            "domain_data_sources": args.domain_data_sources,
            "shuffled_papers": args.shuffled_papers,
            "word_shuffled_papers": args.word_shuffled_papers,
            "sentence_shuffled_papers": args.sentence_shuffled_papers,
            "paragraph_shuffled_papers": args.paragraph_shuffled_papers,
            "fill_batches_with_pretraining": args.fill_batches_with_pretraining,
            "separate_batches_with_pretraining": args.separate_batches_with_pretraining,
            "pretraining_data_type": args.pretraining_data_type,
            "test_script": args.test_script,
            "with_specific_explanation": args.with_specific_explanation,
            "times_explanations": args.times_explanations,
            "semi_cleaned": args.semi_cleaned,
            "use_raw": args.raw if hasattr(args, "raw") else False,
            "shuffle_chunks": args.shuffle_chunks,
            "shuffle_seed": args.shuffle_seed,
            "granular_explanations_cycle": args.granular_explanations_cycle,
            "granular_explanations_num_tracks": args.granular_explanations_num_tracks,
            "explanation_granularity": args.explanation_granularity,
            "explanation_track_size_by_chunk": args.explanation_track_size_by_chunk,
            "explanations_insertion_strategy": args.explanations_insertion_strategy,
            "whole_explanations_insert_every_n": args.whole_explanations_insert_every_n,
            "document_track_baseline": args.document_track_baseline,
            "match_explanation_source_replay": args.match_explanation_source_replay,
            "document_match_specific_explanation": args.document_match_specific_explanation,
            "document_match_insert_content": args.document_match_insert_content,
            "with_prior_knowledge": args.with_prior_knowledge,
            "prior_knowledge_insertion": args.prior_knowledge_insertion,
            "prior_knowledge_cycle": args.prior_knowledge_cycle,
            "prior_knowledge_match_document_track": args.prior_knowledge_match_document_track,
        }

        use_special_injection = bool(args.with_specific_explanation)

        if use_special_injection:
            strategy_name = "ParaphrasedArxivPaperWithSpecificExplanations"
        elif args.num_paraphrased_texts > 0:
            strategy_name = "ParaphrasedArxivPaper"
        else:
            strategy_name = "SingleArxivPaper" # Or a more generic name like "Source"

    chunking_args = {
        "chunk_by_section": args.chunk_by_section,
        "overlap_sections": args.overlap_sections,
        "overlap_ratio": args.overlap_ratio,
        "add_title_prefix": args.no_title_prefix # becomes False if added to the parser
    }
    
    output_dir_for_debug = os.path.join(args.base_results_dir, args.experiment_name, "debug")

    llm_training.fine_tune(
            model=model,
            tokenizer=tokenizer,
            log=log,
            train_cfg=training_config,
            strategy_name=strategy_name,
            strategy_args=strategy_args,
            output_dir_for_debug=output_dir_for_debug,
            callbacks=callbacks_to_use,
            train=train,
            full_debug=args.full_debug,
        **chunking_args
        )

    if train:
        # --- Save Metrics and Generate Plots ---
        if is_world_process_zero():
            experiment_utils.save_probe_results(callbacks_to_use, log, args)

        # --- Generate Plots ---
        # Note: Plotting logic is removed as it's complex with multiple domains. 
        # Please use regenerate_plots.py script or add custom plotting logic.
        log.info("Finished training and saving all probe results.")
    else:
        log.info("Dataloader debug-only mode complete (no fine-tuning performed).")
    return model, tokenizer


def lima_training(model, tokenizer, log, args, num_train_epochs=15):
    # --- Prepare LIMA Training Dataset ---
    log.info("\n--- Starting LIMA-based Instruction Tuning ---")
    lima_train_ds = data_preparation.prepare_lima_dataset(tokenizer, log, use_eot_token=True, cache_dir=args.cache_dir)
    log.info(f"Sample formatted training example:\\n{lima_train_ds}")

    world_size = distributed_world_size()
    global_micro_batch_size = args.device_batch_size * world_size
    assert args.effective_batch_size_for_lima % global_micro_batch_size == 0, (
        "Effective batch size for LIMA must be divisible by "
        "device_batch_size * WORLD_SIZE."
    )
    grad_accum_steps = args.effective_batch_size_for_lima // global_micro_batch_size
    
    # --- LIMA Training Configuration ---
    lima_training_config_kwargs = {
        "run_name": args.experiment_name + "_LIMA",
        "num_train_epochs": num_train_epochs,
        "learning_rate": 2e-5,
        "logging_strategy": "steps",
        "logging_steps": 1,
        "gradient_checkpointing": args.gradient_checkpointing,
        "context_length": args.context_length_for_lima,
        "gradient_accumulation_steps": grad_accum_steps,
        "per_device_train_batch_size": args.device_batch_size,
        "weight_decay": 0.1,
        "use_liger_kernel": True,
        "train_sampling_strategy": "random",
        "reverse_ffd_packing": False,
        "remove_unused_columns": False,
        "packing": True,
        "padding_free": True,
        "dataset_text_field": "text",
        "report_to": "wandb" if not args.test_script else "none",
        "activation_offloading": args.activation_offloading or args.offload_to_cpu,
        "compile": args.compile_model,
        "loss_type": args.sft_loss_type,
    }
    if args.constant_lr:
        lima_training_config_kwargs["lr_scheduler_type"] = "constant_with_warmup"
    else:
        lima_training_config_kwargs["lr_scheduler_type"] = "cosine_with_min_lr"
        lima_training_config_kwargs["lr_scheduler_kwargs"] = {
            "min_lr_rate": args.lr_scheduler_min_lr_ratio
        }
    
    if args.overrule_warmup_via_steps:
        lima_training_config_kwargs["warmup_steps"] = args.overrule_warmup_via_steps
    else:
        lima_training_config_kwargs["warmup_ratio"] = 0.03
        
    lima_training_config = llm_configs.TrainingConfig(**lima_training_config_kwargs)

    # --- Load Probes ---
    # v9 default: inference probes off; W&B per-paper metrics focus on log_prob + target_rank.
    callbacks = experiment_utils.setup_callbacks(
        domains=args.resolved_domains,
        tokenizer=tokenizer, 
        log=log, 
        args=args, 
        is_lima=True,
    )

    trainer = llm_training.sft_train_on_dataset(
        model=model,
        tokenizer=tokenizer,
        log=log,
        train_dataset=lima_train_ds,
        train_cfg=lima_training_config,
        use_liger_loss=True, 
        train=False,
        callbacks=callbacks
    )
    
    # --- QUALITY CONTROL: Check and assert that seq lengths is properly working ---
    seq_counts = []
    found_multi_seq_batch = False
    
    log.info("Verifying LIMA dataloader integrity...")
    eos_token_id = tokenizer.eos_token_id

    for i, batch in enumerate(trainer.get_train_dataloader()):
        # Check 1: Verify sequence packing
        seq_count = 0
        for j in batch['position_ids'][0]:
            if j == 0:
                seq_count += 1
        seq_counts.append(seq_count)
        if seq_count >= 2:
            found_multi_seq_batch = True

        # Check 2: Verify last token of the batch is EOS
        input_ids = batch['input_ids'][-1]
        last_token_id = input_ids[-1]
        if last_token_id != eos_token_id:
            last_token_decoded = tokenizer.decode([last_token_id])
            log.warning(f"Batch {i} does not end with an EOS token. Instead, it ends with token ID {last_token_id} ('{last_token_decoded}').")
            
    avg_seqs = sum(seq_counts) / len(seq_counts)
    min_seqs = min(seq_counts)
    max_seqs = max(seq_counts)
    
    log.info(f"Sequence stats - Avg: {avg_seqs:.2f}, Min: {min_seqs}, Max: {max_seqs}")
    assert found_multi_seq_batch, "No batch found with at least 2 sequences"

    # --- Train the model ---
    trainer.train()

    # --- Save results ---
    if is_world_process_zero():
        experiment_utils.save_probe_results(callbacks, log, args)
    
    log.info("LIMA-based instruction tuning complete.")
    if not args.test_script:
        wandb.finish()

    # Optionally push LIMA-tuned model to hub
    if args.push_to_hub_lima_id:
        log.info(f"Pushing LIMA-tuned model to hub: {args.push_to_hub_lima_id}")
        model.push_to_hub(args.push_to_hub_lima_id)
        tokenizer.push_to_hub(args.push_to_hub_lima_id)
    return model, tokenizer

if __name__ == "__main__":
    # --- Parser ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--custom_suffix", type=str, default="", help="Custom text to append to experiment name")
    parser.add_argument("--override_experiment_name", type=str, default="", help="Override experiment name")
    parser.add_argument("--model_id", type=str, default="allenai/OLMo-2-0425-1B") # allenai/OLMo-2-1124-7B
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--debug_dataloader_only",
        action="store_true",
        help="Run data-mixing and dataloader verification only; skip fine-tuning.",
    )
    parser.add_argument(
        "--full_debug",
        action="store_true",
        help="Write full decoded non-padded sequences in debug_run_*.txt instead of truncated previews.",
    )
    parser.add_argument("--full_finetuning", default=False, action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--constant_lr", action="store_true", help="Use constant learning rate instead of a scheduler (with minimal warmup)")
    parser.add_argument(
        "--lr_scheduler_min_lr_ratio",
        type=float,
        default=0.1,
        help=(
            "For scheduled LR runs, decay to this fraction of the peak learning "
            "rate instead of zero. Default 0.1 means 10%% of peak LR."
        ),
    )
    parser.add_argument("--overrule_warmup_via_steps", type=int, default=None, help="Override warmup_ratio and specify warmup in steps instead")
    parser.add_argument(
        "--knowledge_probes_version",
        type=str,
        default=DEFAULT_KNOWLEDGE_PROBES_VERSION,
        help="Version of the factual knowledge probes to use.",
    )
    parser.add_argument(
        "--knowledge_probe_filename_suffix",
        type=str,
        default=DEFAULT_KNOWLEDGE_PROBE_FILENAME_SUFFIX,
        help=(
            "Advanced: optional suffix inserted before .csv for factual knowledge probes. "
            "Prefer --knowledge_probe_variant for named probe families."
        ),
    )
    parser.add_argument(
        "--knowledge_probe_variant",
        type=str,
        default=DEFAULT_KNOWLEDGE_PROBE_VARIANT,
        choices=sorted(KNOWLEDGE_PROBE_VARIANT_SUFFIXES),
        help=(
            "Factual knowledge probe family to evaluate and track. "
            "low_overlap_strict uses probes_v14_low_overlap_strict.csv."
        ),
    )
    parser.add_argument(
        "--use_low_overlap_knowledge_probes",
        action="store_true",
        help=(
            "Alias for --knowledge_probe_variant low_overlap_strict."
        ),
    )
    parser.set_defaults(paraphrased_knowledge_probes=False)
    parser.add_argument(
        "--paraphrased_knowledge_probes",
        "--paraphrased-knowledge-probes",
        dest="paraphrased_knowledge_probes",
        action="store_true",
        help="Also evaluate a paraphrased factual cloze-probe file alongside the main factual probes.",
    )
    parser.add_argument(
        "--disable_paraphrased_knowledge_probes",
        "--disable-paraphrased-knowledge-probes",
        "--no_paraphrased_knowledge_probes",
        "--no-paraphrased-knowledge-probes",
        dest="paraphrased_knowledge_probes",
        action="store_false",
        help="Disable paraphrased factual cloze-probe evaluation.",
    )
    parser.add_argument(
        "--paraphrased_knowledge_probes_version",
        type=str,
        default=DEFAULT_KNOWLEDGE_PROBES_VERSION,
        help="Version for the optional paraphrased factual probes.",
    )
    parser.add_argument(
        "--paraphrased_knowledge_probe_filename_suffix",
        type=str,
        default=DEFAULT_PARAPHRASED_KNOWLEDGE_PROBE_FILENAME_SUFFIX,
        help="Suffix inserted before .csv for optional paraphrased factual probes.",
    )
    parser.add_argument(
        "--mcqa_probes_version",
        "--mcqa-probes-version",
        type=str,
        default=DEFAULT_MCQA_PROBES_VERSION,
        help="Version of the MCQA probes to use when --mcqa_probes is enabled.",
    )
    parser.add_argument(
        "--mcqa_prompt_column",
        "--mcqa-prompt-column",
        type=str,
        default="formatted_question",
        help=(
            "CSV column to use as the MCQA prompt before appending the constrained "
            "answer suffix. Use formatted_question_5shot for the fixed 5-shot prompt."
        ),
    )
    parser.add_argument(
        "--inference_mcqa_probes_version",
        "--inference-mcqa-probes-version",
        type=str,
        nargs='+',
        default=DEFAULT_INFERENCE_MCQA_PROBES_VERSION,
        help="Version(s) of the inference MCQA probes to use when --inference_mcqa_probes is enabled.",
    )
    parser.add_argument(
        "--inference_mcqa_prompt_column",
        "--inference-mcqa-prompt-column",
        type=str,
        default="formatted_question_5shot",
        help=(
            "CSV column to use as the inference MCQA prompt before appending the "
            "constrained answer suffix."
        ),
    )
    parser.add_argument(
        "--inference_probes_version",
        type=str,
        default=DEFAULT_INFERENCE_PROBES_VERSION,
        help="Version of the inference probes to use.",
    )
    parser.add_argument(
        "--inference_probe_filename_suffix",
        type=str,
        default=DEFAULT_INFERENCE_PROBE_FILENAME_SUFFIX,
        help=(
            "Advanced: optional suffix inserted before .csv for inference probes. "
            "Prefer --inference_probe_variant for named probe families."
        ),
    )
    parser.add_argument(
        "--inference_probe_variant",
        type=str,
        default=DEFAULT_INFERENCE_PROBE_VARIANT,
        choices=sorted(INFERENCE_PROBE_VARIANT_SUFFIXES),
        help=(
            "Inference probe family to evaluate and track. "
            "reviewed uses probes_v11_reviewed.csv by default."
        ),
    )
    parser.add_argument(
        "--use_reviewed_inference_probes",
        action="store_true",
        help="Alias for --inference_probe_variant reviewed.",
    )
    parser.add_argument(
        "--disable_inference_probes",
        action="store_true",
        help="Disable cloze-style inference/compositional probe callbacks.",
    )
    parser.set_defaults(mcqa_probes=False)
    parser.add_argument(
        "--mcqa_probes",
        "--mcqa-probes",
        dest="mcqa_probes",
        action="store_true",
        help=(
            "Enable constrained-decoding MCQA evaluation. "
            "Loads probes_<mcqa_probes_version>_mcqa.csv from the facts directory."
        ),
    )
    parser.add_argument(
        "--disable_mcqa_probes",
        "--disable-mcqa-probes",
        "--no_mcqa_probes",
        "--no-mcqa-probes",
        dest="mcqa_probes",
        action="store_false",
        help="Disable constrained-decoding MCQA evaluation.",
    )
    parser.set_defaults(inference_mcqa_probes=True)
    parser.add_argument(
        "--inference_mcqa_probes",
        "--inference-mcqa-probes",
        dest="inference_mcqa_probes",
        action="store_true",
        help=(
            "Enable constrained-decoding MCQA evaluation for inference probes. "
            "Loads inference/probes_<inference_mcqa_probes_version>_mcqa.csv."
        ),
    )
    parser.add_argument(
        "--disable_inference_mcqa_probes",
        "--disable-inference-mcqa-probes",
        "--no_inference_mcqa_probes",
        "--no-inference-mcqa-probes",
        dest="inference_mcqa_probes",
        action="store_false",
        help="Disable constrained-decoding inference MCQA evaluation.",
    )
    parser.add_argument(
        "--inference_probe_subset",
        type=str,
        default="all",
        choices=["all", "test", "type_split_test"],
        help="Subset of inference probes to use for domain 1_58 when using v7 probes.",
    )
    parser.add_argument(
        "--enable_wandb_source_panels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable source-level W&B panels (arxiv/legal/medical).",
    )
    parser.add_argument(
        "--wandb_panel_sources",
        type=str,
        nargs='+',
        default=list(DEFAULT_WANDB_PANEL_SOURCES),
        choices=list(SUPPORTED_HIGH_LEVEL_DOMAINS),
        help="High-level source panels to log in W&B.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=DEFAULT_WANDB_PROJECT,
        help="W&B project name to use for this script.",
    )
    parser.add_argument(
        "--wandb_group",
        type=str,
        default=DEFAULT_WANDB_GROUP,
        help="W&B group name to use for this script. Set to an empty string to leave it unset.",
    )
    parser.add_argument("--num_paraphrased_texts", type=int, default=9, help="Number of paraphrased texts to use for training (0-9)")
    parser.add_argument("--lima_afterwards", default=False, action="store_true", help="LIMA-based instruction tuning after continued pretraining")
    parser.add_argument("--prior_knowledge", action="store_true", help="Use prior_knowledge textbooks (per source, if available) instead of cleaned/raw documents for continued pretraining.")
    parser.add_argument("--prior_knowledge_num_train_epochs", type=int, default=None, help="If set, override num_train_epochs during prior-knowledge CPT.")
    parser.add_argument("--prior_knowledge_effective_batch_size_for_cpt", type=int, default=None, help="If set, override effective_batch_size_for_cpt during prior-knowledge CPT.")
    parser.add_argument("--with_prior_knowledge", action="store_true", help="Inject prior-knowledge chapters (one chapter per batch, one full pass) into the regular training schedule at the timing given by --prior_knowledge_insertion. Separate from --prior_knowledge.")
    parser.add_argument("--prior_knowledge_insertion", choices=["front", "middle", "end"], default="front", help="When --with_prior_knowledge is set, where to splice the prior-knowledge block: front (before all doc batches), middle (~50%% mark), or end (after all doc batches).")
    parser.add_argument("--prior_knowledge_cycle", default="full", help="'full' loads all chapters per domain; otherwise an integer N to load only the first N chapters.")
    parser.add_argument("--prior_knowledge_match_document_track", action="store_true", help="Baseline for --with_prior_knowledge: replace PK chapter chunks with same-shape source+paraphrase replay chunks (matches compute/exposure without PK content).")

    # Defaults, do not change unless for ablations
    parser.add_argument("--chunk_by_section", action="store_true", help="Use section-based chunking instead of token-based chunking")
    parser.add_argument("--no_title_prefix", action="store_false", help="Add title prefix to chunks when chunking")
    parser.add_argument("--overlap_sections", default=False, action="store_true", help="Overlap sections when chunking")
    parser.add_argument("--overlap_ratio", type=str, default="1_4", help="Ratio of overlap when chunking")
    parser.add_argument("--with_specific_explanation", type=str, nargs='+', default=None, help="Use specific explanation type(s). For granular/granular_queue/legacy_v2 these are subfolders; for whole/legacy these map to flat files (e.g., textbooks -> textbook.txt).")
    parser.add_argument("--document_track_baseline", action="store_true", help="Add an auxiliary track matched to a granular explanation schedule.")
    parser.add_argument("--document_match_specific_explanation", type=str, nargs='+', default=None, help="Explanation subfolder type(s) used only to size --document_track_baseline, e.g. textbooks blogs stackexchange.")
    parser.add_argument(
        "--document_match_insert_content",
        type=str,
        default="document",
        choices=["document", "cited_works", "prior_knowledge", "textbooks", "blogs", "stackexchange"],
        help=(
            "Content inserted into the matched auxiliary track: 'document' cycles the current "
            "source/paraphrase chunks, 'cited_works' cycles cited_textbooks, "
            "'prior_knowledge' cycles prior_knowledge chapters, and "
            "'textbooks'/'blogs'/'stackexchange' cycle the corresponding explanation subfolder."
        ),
    )
    parser.add_argument("--raw", action="store_true", help="Use raw texts instead of cleaned/semi-cleaned corpora.")
    parser.add_argument("--times_explanations", type=int, default=1, help="Number of times to repeat the explanation texts.")
    parser.add_argument("--do_eval", default=False, action="store_true", help="Enable evaluation of generations using an LLM judge.")
    parser.add_argument("--test_script", action="store_true", help="Run in test mode with a small model and minimal epochs.")
    parser.add_argument("--shuffle_chunks", action="store_true", help="Shuffle constructed training chunks with seed 42 before training.")
    parser.add_argument("--shuffle_seed", type=int, default=42, help="Seed to use when shuffling training chunks.")
    parser.add_argument("--granular_explanations_cycle", type=str, default="0", help="Granular strategy only: number of explanation files to cycle through across document batches. Use 'full' to load all available files, or specify an integer. Not used by granular_queue.")
    parser.add_argument(
        "--explanations_insertion_strategy",
        type=str,
        default="granular",
        choices=["granular", "granular_queue", "whole", "legacy", "legacy_v2", "random_splice"],
        help=(
            "How explanations are inserted: 'granular' (per-batch cycling), "
            "'granular_queue' (shuffle selected explanation files into K tracks), "
            "'whole' (insert explanation-only batch every N steps), 'legacy' "
            "(older coupled splice behavior), 'legacy_v2' (stream selected "
            "subfolder explanation chunks into paraphrase tails, capped at 50%% "
            "per batch), or 'random_splice' (legacy-style chunk replacement "
            "starting at a deterministic random paraphrase batch)."
        ),
    )
    parser.add_argument(
        "--whole_explanations_insert_every_n",
        type=int,
        default=1,
        help="For --explanations_insertion_strategy whole: insert explanation-only batches every N document steps.",
    )
    parser.add_argument(
        "--granular_explanations_num_tracks",
        type=int,
        default=1,
        help="Granular/granular_queue only: number of explanation tracks to build. For granular, track i uses an offset of floor(i * num_files / N). For granular_queue, files are length-balanced across same-step track slots. Default is 1.",
    )
    parser.add_argument(
        "--explanation_granularity",
        type=str,
        default="file",
        choices=["file", "chunk"],
        help=(
            "Granular explanation track unit. 'file' keeps each selected explanation "
            "file together as one step; 'chunk' groups selected explanation chunks "
            "into fixed-size track steps."
        ),
    )
    parser.add_argument(
        "--explanation_track_size_by_chunk",
        type=int,
        default=4,
        help="When --explanation_granularity chunk is used, number of explanation chunks per track step.",
    )
    parser.add_argument(
        "--match_explanation_source_replay",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "For granular explanation runs, append a matched source/paraphrase replay "
            "track with the same per-domain, per-step chunk counts as the explanation track."
        ),
    )
    parser.add_argument("--shuffled_papers", action="store_true", help="Legacy: use shuffled versions of papers (files ending with _shuffle.tex) when available.")
    parser.add_argument("--word_shuffled_papers", action="store_true", help="Use word-shuffled versions of papers (files ending with _shuffle_words.tex) when available.")
    parser.add_argument("--sentence_shuffled_papers", action="store_true", help="Use sentence-shuffled versions of papers (files ending with _shuffle_sentences.tex) when available.")
    parser.add_argument("--paragraph_shuffled_papers", action="store_true", help="Use paragraph-shuffled versions of papers (files ending with _shuffle_paragraphs.tex) when available.")

    # Lora arguments
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA r parameter.")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha parameter.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout parameter.")
    parser.add_argument("--lora_target_modules", type=str, nargs='+', default=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'], help="LoRA target modules.")

    # New arguments for multi-domain training
    parser.add_argument(
        "--include_sources",
        type=str,
        nargs='+',
        default=list(SUPPORTED_HIGH_LEVEL_DOMAINS),
        choices=list(SUPPORTED_HIGH_LEVEL_DOMAINS),
        help="High-level sources to include when resolving domains. Use 'arxiv legal' to exclude medical.",
    )
    parser.add_argument(
        "--override_arxiv_domain",
        type=str,
        nargs='+',
        default=None,
        help="Optional list of arXiv concrete domain IDs. If unset, all arXiv domains are used.",
    )
    parser.add_argument(
        "--override_legal_domain",
        type=str,
        nargs='+',
        default=None,
        help="Optional list of legal concrete domain IDs. If unset, all legal domains are used.",
    )
    parser.add_argument(
        "--override_medical_domain",
        type=str,
        nargs='+',
        default=None,
        help="Optional list of medical concrete domain IDs. If unset, all medical domains are used.",
    )
    parser.add_argument("--fill_batches_with_pretraining", default=False, action="store_true", help="Fill batches with pretraining data.")
    parser.add_argument("--separate_batches_with_pretraining", type=int, default=0, help="Number of pretraining batches to insert between unique document types.")
    parser.add_argument("--pretraining_data_type", type=str, default="dclm", help="Type of pretraining data to use ('dclm' or 'arxiv').")
    parser.add_argument("--effective_batch_size_for_cpt", type=int, default=8, help="The effective batch size for continued pretraining.")
    parser.add_argument("--effective_batch_size_for_lima", type=int, default=32, help="The effective batch size for LIMA training.")
    parser.add_argument("--device_batch_size", type=int, default=2, help="The batch size per device.")
    parser.add_argument(
        "--sft_loss_type",
        type=str,
        default="nll",
        choices=["nll", "dft", "chunked_nll"],
        help="Loss implementation passed to TRL SFTConfig.",
    )
    parser.add_argument("--context_length_for_cpt", type=int, default=3072, help="Context length for continued pretraining.")
    parser.add_argument("--context_length_for_lima", type=int, default=2560, help="Context length for LIMA training.")
    parser.add_argument("--push_to_hub_cpt_id", type=str, default="", help="Optional Hub model ID to push CPT model to.")
    parser.add_argument("--push_to_hub_lima_id", type=str, default="", help="Optional Hub model ID to push LIMA-tuned model to.")
    parser.add_argument(
        "--save_local_model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save local model/tokenizer checkpoints under the experiment directory.",
    )
    parser.add_argument(
        "--cpt_model_subdir",
        type=str,
        default="model_cpt",
        help="Subdirectory (inside experiment dir) for post-CPT checkpoint save.",
    )
    parser.add_argument(
        "--lima_model_subdir",
        type=str,
        default="model_lima",
        help="Subdirectory (inside experiment dir) for post-LIMA checkpoint save.",
    )
    parser.add_argument("--semi_cleaned", type=str, default=None, choices=['v1', 'v2','v3'], help="Use semi-cleaned data from a specific version (v1 or v2).")
    parser.add_argument("--attn_implementation", type=str, default="sdpa", choices=["sdpa", "flash_attention_2", "flash_attention_3", "kernels-community/vllm-flash-attn3"], help="Attention implementation to use.")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing.")
    parser.add_argument("--compile_model", action="store_true", help="Enable torch.compile for the model.")
    parser.add_argument("--compile", dest="compile_model", action="store_true", help="Enable torch.compile via TrainingArguments.")
    parser.add_argument("--activation_offloading", action="store_true", help="Enable TRL activation offloading without changing model device placement.")
    parser.add_argument("--offload_to_cpu", action="store_true", help="Enable activation offloading and CPU/GPU model loading.")
    parser.add_argument(
        "--no_callback_every_step",
        action="store_true",
        help="If set, run heavy callbacks only at 25%%, 50%%, and 75%% of training instead of every step.",
    )
    parser.add_argument(
        "--probe_every_n_steps",
        type=int,
        default=1,
        help="Run cloze knowledge/inference probe callbacks every N training steps.",
    )
    parser.add_argument(
        "--mcqa_probe_every_n_steps",
        type=int,
        default=1,
        help="Run MCQA probe callbacks every N training steps.",
    )
    parser.add_argument(
        "--mcqa_probe_batch_size",
        type=int,
        default=32,
        help=(
            "Microbatch size for MCQA probe forward passes. MCQA prompts can be "
            "much longer than cloze probes, especially with few-shot prompts, so "
            "this is intentionally separate from --device_batch_size."
        ),
    )
    parser.add_argument(
        "--enable_parameter_delta_tracking",
        action="store_true",
        help="Track online parameter delta summaries and plots during training.",
    )
    parser.add_argument(
        "--parameter_delta_include_embeddings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include input token embeddings in parameter delta tracking.",
    )
    parser.add_argument(
        "--parameter_delta_storage_path",
        type=str,
        default=None,
        help=(
            "Optional folder for large temporary raw delta tensors used for final-direction "
            "alignment when --parameter_delta_compute_final_alignment is enabled. "
            "A run-specific subfolder is created and deleted after successful plotting."
        ),
    )
    parser.add_argument(
        "--parameter_delta_compute_final_alignment",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Save temporary raw parameter deltas and compute final-direction alignment "
            "metrics/plots. Disabled by default."
        ),
    )
    parser.add_argument(
        "--parameter_delta_sparse_milestones",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Track parameter deltas at train begin, every 10%% of training, and train end.",
    )
    parser.add_argument(
        "--parameter_delta_every_n_steps",
        type=int,
        default=None,
        help=(
            "Record parameter deltas every N training steps, plus train begin/end. "
            "When set, this overrides --parameter_delta_sparse_milestones."
        ),
    )
    parser.add_argument(
        "--parameter_delta_report_to_wandb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log compact parameter delta summaries to W&B.",
    )
    parser.add_argument("--parcc", action="store_true", help="Use /vast/projects/myatskar/design-documents as cache directory for model and dataset loading operations")

    args = parser.parse_args()
    apply_knowledge_probe_variant(args)
    apply_inference_probe_variant(args)
    args.wandb_panel_sources = list(dict.fromkeys(args.wandb_panel_sources))

    # v9 built-ins:
    # - no perplexity-related W&B logging
    # - probe W&B logging limited to log_prob + target_rank
    args.wandb_probe_metric_allowlist = [
        "log_prob",
        "target_rank",
        "mcqa_accuracy",
    ]
    args.disable_corpus_perplexity_wandb = True
    args.disable_training_loss_perplexity_wandb = True

    # Set cache_dir based on --parcc flag
    if args.parcc:
        args.cache_dir = "/vast/projects/myatskar/design-documents"
    else:
        args.cache_dir = None

    if not 0.0 <= args.lr_scheduler_min_lr_ratio < 1.0:
        raise ValueError("--lr_scheduler_min_lr_ratio must be in [0.0, 1.0).")

    # When using prior knowledge textbooks, adjust defaults:
    # - disable paraphrases and explanations
    # - optionally override CPT epochs and effective batch size
    if args.prior_knowledge:
        args.num_paraphrased_texts = 0
        args.with_specific_explanation = None
        args.document_track_baseline = False
        args.document_match_specific_explanation = None

        if args.prior_knowledge_num_train_epochs is not None:
            args.num_train_epochs = args.prior_knowledge_num_train_epochs
        if args.prior_knowledge_effective_batch_size_for_cpt is not None:
            args.effective_batch_size_for_cpt = args.prior_knowledge_effective_batch_size_for_cpt

    if args.with_prior_knowledge and args.prior_knowledge:
        raise ValueError("--with_prior_knowledge (injection into regular training) and --prior_knowledge (PK-only training) are mutually exclusive.")
    if args.prior_knowledge_match_document_track and not args.with_prior_knowledge:
        raise ValueError("--prior_knowledge_match_document_track requires --with_prior_knowledge.")
    if args.prior_knowledge_cycle != "full":
        try:
            int(args.prior_knowledge_cycle)
        except ValueError:
            raise ValueError(f"--prior_knowledge_cycle must be 'full' or an integer, got: {args.prior_knowledge_cycle}")

    # --- Argument Validation ---
    # Parse granular_explanations_cycle
    if args.granular_explanations_cycle == "full":
        args.granular_explanations_cycle = "full"
    else:
        try:
            args.granular_explanations_cycle = int(args.granular_explanations_cycle)
        except ValueError:
            raise ValueError(f"--granular_explanations_cycle must be 'full' or an integer, got: {args.granular_explanations_cycle}")

    if args.explanations_insertion_strategy == "whole" and args.whole_explanations_insert_every_n <= 0:
        raise ValueError("--whole_explanations_insert_every_n must be a positive integer when strategy is 'whole'.")

    if (
        args.explanations_insertion_strategy != "whole"
        and args.whole_explanations_insert_every_n != 1
    ):
        raise ValueError(
            "--whole_explanations_insert_every_n is only supported with --explanations_insertion_strategy whole "
            "(set it to 1 for granular/granular_queue/legacy/legacy_v2/random_splice)."
        )

    if args.granular_explanations_num_tracks <= 0:
        raise ValueError("--granular_explanations_num_tracks must be a positive integer.")

    if args.explanation_track_size_by_chunk <= 0:
        raise ValueError("--explanation_track_size_by_chunk must be a positive integer.")

    if args.mcqa_probe_batch_size <= 0:
        raise ValueError("--mcqa_probe_batch_size must be a positive integer.")

    if (
        args.explanations_insertion_strategy not in ("granular", "granular_queue")
        and args.granular_explanations_num_tracks != 1
    ):
        raise ValueError(
            "--granular_explanations_num_tracks is only supported with --explanations_insertion_strategy granular or granular_queue "
            "(set it to 1 for whole/legacy/legacy_v2/random_splice)."
        )

    if (
        args.explanations_insertion_strategy not in ("granular", "granular_queue")
        and args.explanation_granularity != "file"
    ):
        raise ValueError(
            "--explanation_granularity chunk is only supported with "
            "--explanations_insertion_strategy granular or granular_queue."
        )

    if args.explanation_granularity == "file" and args.explanation_track_size_by_chunk != 4:
        raise ValueError(
            "--explanation_track_size_by_chunk is only used with --explanation_granularity chunk "
            "(leave it at the default 4 for file granularity)."
        )

    if args.explanations_insertion_strategy != "granular" and args.granular_explanations_cycle != 0:
        raise ValueError(
            "--granular_explanations_cycle is only supported with --explanations_insertion_strategy granular. "
            "For granular_queue/whole/legacy/legacy_v2/random_splice, leave --granular_explanations_cycle at the default 0."
        )

    if args.match_explanation_source_replay:
        if args.document_track_baseline:
            raise ValueError(
                "--match_explanation_source_replay cannot be combined with --document_track_baseline; "
                "the former is for real explanation runs, the latter is for no-explanation controls."
            )
        if not args.with_specific_explanation:
            raise ValueError("--match_explanation_source_replay requires --with_specific_explanation.")
        if args.explanations_insertion_strategy != "granular":
            raise ValueError("--match_explanation_source_replay currently supports only granular insertion.")

    if args.parameter_delta_every_n_steps is not None and args.parameter_delta_every_n_steps <= 0:
        raise ValueError("--parameter_delta_every_n_steps must be a positive integer when set.")

    if args.document_track_baseline:
        if not args.document_match_specific_explanation:
            raise ValueError(
                "--document_track_baseline requires --document_match_specific_explanation "
                "to define the explanation schedule being matched."
            )
        if args.explanations_insertion_strategy != "granular":
            raise ValueError("--document_track_baseline currently supports only granular insertion schedules.")
        if args.granular_explanations_cycle == 0:
            raise ValueError(
                "--document_track_baseline requires --granular_explanations_cycle "
                "to be a positive integer or 'full'."
            )

    if (
        args.explanations_insertion_strategy == "granular"
        and args.with_specific_explanation
        and args.granular_explanations_cycle == 0
    ):
        raise ValueError(
            "Granular insertion with --with_specific_explanation requires --granular_explanations_cycle "
            "to be a positive integer or 'full'."
        )

    if args.explanations_insertion_strategy == "granular_queue" and not args.with_specific_explanation:
        raise ValueError(
            "Granular queue insertion requires --with_specific_explanation to select explanation subfolders."
        )

    # Normalize single explanation type from argparse nargs='+' list into a scalar.
    if isinstance(args.with_specific_explanation, list) and len(args.with_specific_explanation) == 1:
        args.with_specific_explanation = args.with_specific_explanation[0]
    if isinstance(args.document_match_specific_explanation, list) and len(args.document_match_specific_explanation) == 1:
        args.document_match_specific_explanation = args.document_match_specific_explanation[0]
    
    # --- Setup Logging & Wandb ---
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log = logging.getLogger(__name__)

    if args.test_script:
        log.info("--- RUNNING IN TEST SCRIPT MODE ---")
        args.num_train_epochs = 1
        args.base_results_dir = os.path.join("../../results", "tests")
    else: 
        os.environ["WANDB_PROJECT"] = args.wandb_project
        if args.wandb_group:
            os.environ["WANDB_RUN_GROUP"] = args.wandb_group
        else:
            os.environ.pop("WANDB_RUN_GROUP", None)
        args.base_results_dir = os.path.join("../../results", "FT")

    if args.disable_inference_probes:
        log.info("Inference probes are disabled.")
    else:
        log.info(
            f"Inference probes are enabled "
            f"(variant={args.inference_probe_variant}, "
            f"probes_{args.inference_probes_version}"
            f"{args.inference_probe_filename_suffix}.csv, "
            f"subset={args.inference_probe_subset})."
        )
    log.info(
        "Factual knowledge probes: "
        f"variant={args.knowledge_probe_variant}, "
        f"probes_{args.knowledge_probes_version}"
        f"{args.knowledge_probe_filename_suffix}.csv"
    )
    if args.constant_lr:
        log.info("Learning-rate scheduler: constant_with_warmup.")
    else:
        log.info(
            "Learning-rate scheduler: cosine_with_min_lr "
            f"(min_lr_rate={args.lr_scheduler_min_lr_ratio:g}; "
            f"floor={args.learning_rate * args.lr_scheduler_min_lr_ratio:g})."
        )
    if args.mcqa_probes:
        log.info(
            "Factual MCQA probes are enabled "
            f"(facts/probes_{args.mcqa_probes_version}_mcqa.csv, "
            f"prompt_column={args.mcqa_prompt_column})."
        )
    if args.inference_mcqa_probes:
        inference_mcqa_probes_versions = _coerce_version_list(args.inference_mcqa_probes_version)
        log.info(
            "Inference MCQA probes are enabled "
            f"(versions={inference_mcqa_probes_versions}, "
            f"prompt_column={args.inference_mcqa_prompt_column})."
        )
    if args.enable_wandb_source_panels:
        mcqa_note = f" MCQA constrained-decoding ({args.mcqa_probes_version})." if args.mcqa_probes else ""
        inference_mcqa_note = (
            f" Inference MCQA constrained-decoding "
            f"({_coerce_version_list(args.inference_mcqa_probes_version)})."
            if args.inference_mcqa_probes
            else ""
        )
        inference_note = (
            " Inference probe metrics: <source>/<document>_inference_log_prob, "
            "<source>/<document>_inference_target_rank."
            if not args.disable_inference_probes
            else ""
        )
        log.info(
            f"W&B source panels enabled for: {args.wandb_panel_sources}. "
            "Flat metrics: <source>/<document>_log_prob, "
            "<source>/<document>_target_rank, "
            "<source>/<document>_mcqa_accuracy, "
            f"<source>/<document>_inference_mcqa_accuracy.{mcqa_note}"
            f"{inference_mcqa_note}{inference_note}"
        )

    args.resolved_domains, args.domain_data_sources = resolve_domains_and_sources(args, log)
    probe_validation_tokenizer = load_probe_validation_tokenizer(args, log)
    validate_selected_knowledge_probes(args, log, tokenizer=probe_validation_tokenizer)
    validate_selected_paraphrased_knowledge_probes(args, log, tokenizer=probe_validation_tokenizer)
    validate_selected_inference_probes(args, log)
    validate_selected_mcqa_probes(args, log)

    if args.override_experiment_name:
        args.experiment_name = args.override_experiment_name
    else:
        args.experiment_name = construct_experiment_name(args)
    args.eval_bundle_name = _construct_eval_bundle_name(args)
    args.legacy_probe_bundle_name = _construct_legacy_probe_bundle_name(args)

    # --- Save Hyperparameters ---
    experiment_dir = os.path.join(args.base_results_dir, args.experiment_name)
    args.experiment_dir = experiment_dir
    os.makedirs(experiment_dir, exist_ok=True)
    hyperparameters_path = os.path.join(experiment_dir, 'hyperparameters.json')
    if is_world_process_zero():
        with open(hyperparameters_path, 'w') as f:
            json.dump(vars(args), f, indent=4)
        log.info(f"Hyperparameters saved to {hyperparameters_path}")
    distributed_barrier()

    # (WandB run name will follow the experiment name so no explicit init here)

    # --- Load the model ---
    attn_implementation = args.attn_implementation
    
    peft_config = llm_configs.PeftConfig(
        enabled=(not args.full_finetuning),
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        instruction_tuning=False,
        add_eot_token=args.lima_afterwards
    )
    
    model_config = llm_configs.ModelConfig(
        id= args.model_id, #"allenai/OLMo-2-0425-1B", #"allenai/OLMo-2-1124-7B",
        peft=peft_config,
        quantization=llm_configs.QuantizationConfig(mode=None),
        attn_implementation=attn_implementation,
    )

    log.info("\n--- Loading Model for Training ---")
    special_token_to_add = "<|EOT|>" if args.lima_afterwards else None
    model, tokenizer = model_setup.load_model_for_training(
        model_config,
        log,
        use_cpu_and_gpu=args.offload_to_cpu,
        add_special_token=special_token_to_add,
        use_existing_lima_tokenizer=False,
        use_existing_lima_model=False,
        cache_dir=args.cache_dir,
    )

    if not args.full_finetuning:
        model.print_trainable_parameters()

    # Model compilation is handled by TrainingArguments via compile flag in TrainingConfig
    # --- Continue Pretraining / Debug-only dataloader check ---
    run_cpt = args.num_train_epochs > 0 or args.debug_dataloader_only
    if run_cpt:
        train_cpt = not args.debug_dataloader_only
        model, tokenizer = continue_pretraining(model, tokenizer, log, args, train=train_cpt)
        if train_cpt and args.save_local_model and is_world_process_zero():
            cpt_save_path = os.path.join(args.experiment_dir, args.cpt_model_subdir)
            llm_training.save_model(model, tokenizer, log, cpt_save_path)
            log.info(f"CPT checkpoint saved to {cpt_save_path}")
        distributed_barrier()
        # Optionally push CPT model snapshot to hub
        if train_cpt and args.push_to_hub_cpt_id and is_world_process_zero():
            log.info(f"Pushing CPT model to hub: {args.push_to_hub_cpt_id}")
            model.push_to_hub(args.push_to_hub_cpt_id)
            tokenizer.push_to_hub(args.push_to_hub_cpt_id)
    
    # -- LIMA-based instruction tuning ---
    if args.lima_afterwards:
        lima_epochs = 1 if args.test_script else 10
        model, tokenizer = lima_training(model, tokenizer, log, args, num_train_epochs=lima_epochs)
        if args.save_local_model and is_world_process_zero():
            lima_save_path = os.path.join(args.experiment_dir, args.lima_model_subdir)
            llm_training.save_model(model, tokenizer, log, lima_save_path)
            log.info(f"LIMA checkpoint saved to {lima_save_path}")
        distributed_barrier()
