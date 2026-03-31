from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from torch.utils.data import Dataset

from .config import InstanceFilterConfig, PaddingDirection
from .data.collator import DataCollator
from .data.memmap_dataset import MemMapDataset
from .tokenizer import Tokenizer as OLMoTokenizer

LOGGER = logging.getLogger(__name__)

_ENV_INJECTION_CFG = "FT_OLMO_INJECTION_CONFIG"
_INJECTION_COLLATOR = None


@dataclass
class _TrainingMixConfig:
    context_length: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    num_train_epochs: int


class _TokenizerAdapter:
    """
    Adapts OLMo's tokenizer wrapper to the callable tokenizer shape expected by
    this repo's chunking / data-prep helpers.
    """

    def __init__(self, tokenizer: OLMoTokenizer):
        self._tokenizer = tokenizer
        self.eos_token = tokenizer.eos_token
        self.eos_token_id = tokenizer.eos_token_id

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = False,
        truncation: bool = False,
    ) -> Dict[str, Any]:
        del truncation
        return {"input_ids": self.encode(text, add_special_tokens=add_special_tokens)}

    def encode(self, text: str, add_special_tokens: bool = False):
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, token_ids, skip_special_tokens: bool = True):
        return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)


def _resolve_project_root(project_root: Optional[str]) -> Path:
    return Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[2]


def _build_injection_sequences(
    *,
    project_root: str,
    strategy_name: str,
    strategy_args: Optional[Dict[str, Any]],
    chunking_args: Optional[Dict[str, Any]],
    tokenizer_identifier: str,
    eos_token_id: int,
    pad_token_id: int,
    context_length: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    num_train_epochs: int,
) -> List[List[int]]:
    project_root_path = _resolve_project_root(project_root)
    if str(project_root_path) not in sys.path:
        sys.path.insert(0, str(project_root_path))

    from utils.data_preparation import prepare_training_mix  # pylint: disable=import-outside-toplevel

    strategy_args = dict(strategy_args or {})
    chunking_args = dict(chunking_args or {})

    # Requested behavior: use replay to fill only; do not add extra replay separator batches.
    strategy_args.setdefault("fill_batches_with_pretraining", True)
    strategy_args.setdefault("separate_batches_with_pretraining", 0)

    if Path(tokenizer_identifier).is_file():
        tokenizer = OLMoTokenizer.from_file(
            tokenizer_identifier,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
    else:
        tokenizer = OLMoTokenizer.from_pretrained(
            tokenizer_identifier,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
    tokenizer_adapter = _TokenizerAdapter(tokenizer)

    mix_cfg = _TrainingMixConfig(
        context_length=context_length,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
    )

    old_cwd = Path.cwd()
    data_prep_cwd = project_root_path / "scripts" / "FT"
    if not data_prep_cwd.is_dir():
        data_prep_cwd = project_root_path

    try:
        os.chdir(data_prep_cwd)
        text_dataset, _ = prepare_training_mix(
            strategy_name=strategy_name,
            tokenizer=tokenizer_adapter,
            log=LOGGER,
            train_cfg=mix_cfg,
            **strategy_args,
            **chunking_args,
        )
    finally:
        os.chdir(old_cwd)

    sequences: List[List[int]] = []
    for item in text_dataset:
        sequences.append(tokenizer_adapter.encode(item["raw_text"], add_special_tokens=False))
    return sequences


class CheckpointInjectionDataset(Dataset):
    """
    Base training dataset from OLMo memmaps. Used with `checkpoint_injection_collate`
    to inject custom FT examples into each batch while preserving checkpoint data order.
    """

    def __init__(
        self,
        *,
        paths: Optional[List[str]] = None,
        datasets: Optional[Dict[str, List[str]]] = None,
        label_mask_paths: Optional[List[str]] = None,
        memmap_dtype: str = "uint16",
        generate_attention_mask: bool = False,
        generate_doc_lengths: bool = False,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        context_length: int = 2048,
        include_instance_metadata: bool = False,
        instance_filter: Optional[Dict[str, Any]] = None,
    ):
        if paths:
            if datasets:
                raise ValueError("paths and datasets are mutually exclusive")
            memmap_paths = paths
            metadata = [{"path": str(path)} for path in memmap_paths]
        elif datasets:
            memmap_paths = []
            metadata = []
            for label in sorted(datasets.keys()):
                label_paths = datasets[label]
                memmap_paths.extend(label_paths)
                metadata.extend([{"label": label}] * len(label_paths))
        else:
            raise ValueError("One of 'paths' or 'datasets' is required")

        dtype = getattr(np, memmap_dtype)
        instance_filter_config = InstanceFilterConfig(**instance_filter) if instance_filter else None

        self._base = MemMapDataset(
            *memmap_paths,
            chunk_size=context_length,
            memmap_dtype=dtype,
            metadata=metadata,
            include_instance_metadata=include_instance_metadata,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            generate_attention_mask=generate_attention_mask,
            generate_doc_lengths=generate_doc_lengths,
            label_mask_paths=label_mask_paths,
            instance_filter_config=instance_filter_config,
        )
        LOGGER.info("CheckpointInjectionDataset initialized with %d base instances.", len(self._base))

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self._base[idx]


class _CheckpointInjectionCollator:
    def __init__(self, cfg: Dict[str, Any]):
        pad_direction = PaddingDirection(cfg.get("pad_direction", PaddingDirection.right.value))
        pad_token_id = int(cfg["pad_token_id"])
        self._base_collator = DataCollator(pad_direction=pad_direction, pad_token_id=pad_token_id)
        self._replace_per_device_batch = int(cfg.get("replace_per_device_batch", 0))

        self._cursor = int(cfg.get("injection_cursor_offset", 0))
        self._sequences = _build_injection_sequences(
            project_root=cfg["project_root"],
            strategy_name=cfg["strategy_name"],
            strategy_args=cfg.get("strategy_args"),
            chunking_args=cfg.get("chunking_args"),
            tokenizer_identifier=cfg["tokenizer_identifier"],
            eos_token_id=int(cfg["eos_token_id"]),
            pad_token_id=pad_token_id,
            context_length=int(cfg["context_length"]),
            per_device_train_batch_size=int(cfg["per_device_train_batch_size"]),
            gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
            num_train_epochs=int(cfg.get("num_train_epochs", 1)),
        )
        if not self._sequences:
            LOGGER.warning("No custom injection sequences were built; collator will pass through base batches.")
        LOGGER.info(
            "Injection collator ready. replace_per_device_batch=%d, custom_sequences=%d, cursor_offset=%d",
            self._replace_per_device_batch,
            len(self._sequences),
            self._cursor,
        )

    def __call__(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        if self._replace_per_device_batch > 0 and self._sequences:
            replace_count = min(self._replace_per_device_batch, len(items))
            for i in range(replace_count):
                seq = self._sequences[self._cursor % len(self._sequences)]
                self._cursor += 1
                replacement: Dict[str, Any] = {"input_ids": seq}
                if "index" in items[i]:
                    replacement["index"] = items[i]["index"]
                items[i] = replacement
        return self._base_collator(items)


def checkpoint_injection_collate(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    global _INJECTION_COLLATOR
    if _INJECTION_COLLATOR is None:
        raw = os.environ.get(_ENV_INJECTION_CFG)
        if not raw:
            raise RuntimeError(f"Missing environment variable: {_ENV_INJECTION_CFG}")
        cfg = json.loads(raw)
        _INJECTION_COLLATOR = _CheckpointInjectionCollator(cfg)
    return _INJECTION_COLLATOR(items)
