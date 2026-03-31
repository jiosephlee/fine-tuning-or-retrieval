#!/usr/bin/env python
"""
Wrapper around OLMo's trainer that injects this repo's custom FT data pipeline
through OLMo `data.custom_dataset`.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from olmo.config import CustomDatasetConfig, TrainConfig
from olmo.exceptions import OLMoCliError
from olmo.torch_util import get_local_rank
from olmo.util import add_cached_path_clients, clean_opt, prepare_cli_environment
from scripts.train import main as olmo_main

log = logging.getLogger("train")


def _parse_args(argv: list[str]) -> Tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run OLMo training with fine-tuning-or-retrieval custom data prep."
    )
    parser.add_argument("config_path", help="Path to OLMo YAML config.")
    parser.add_argument(
        "--strategy-name",
        default="SingleArxivPaper",
        help="Strategy passed to utils.data_preparation.prepare_training_mix.",
    )
    parser.add_argument(
        "--strategy-args-json",
        default="{}",
        help="JSON dict of strategy args for prepare_training_mix.",
    )
    parser.add_argument(
        "--chunking-args-json",
        default="{}",
        help="JSON dict of chunking args for prepare_training_mix.",
    )
    parser.add_argument(
        "--ft-num-train-epochs",
        type=int,
        default=1,
        help="Epoch count passed into the custom FT data mixer.",
    )
    parser.add_argument(
        "--project-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Path to fine-tuning-or-retrieval repo root (for importing utils.*).",
    )
    parser.add_argument(
        "--replace-per-global-batch",
        type=int,
        default=0,
        help="How many examples to replace per global batch with custom FT data.",
    )
    parser.add_argument(
        "--replace-per-device-batch",
        type=int,
        default=None,
        help="Override replacement count per device batch (takes precedence over global replacement count).",
    )
    return parser.parse_known_args(argv)


def _parse_json_dict(raw: str, field_name: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise OLMoCliError(f"{field_name} must be valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise OLMoCliError(f"{field_name} must decode to a JSON object.")
    return parsed


def _init_process_group() -> None:
    if torch.cuda.is_available():
        log.info("CUDA available")
        device_as_string = f"cuda:{get_local_rank()}"
        torch.cuda.set_device(device_as_string)
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(minutes=30),
            device_id=torch.device(device_as_string),
        )
    elif torch.backends.mps.is_available():
        if not os.getenv("RANK"):
            os.environ["RANK"] = "0"
        if not os.getenv("WORLD_SIZE"):
            os.environ["WORLD_SIZE"] = "1"
        if not os.getenv("MASTER_ADDR"):
            os.environ["MASTER_ADDR"] = "0.0.0.0"
        if not os.getenv("MASTER_PORT"):
            os.environ["MASTER_PORT"] = "24501"
        dist.init_process_group(backend="gloo", timeout=timedelta(minutes=30))
    else:
        dist.init_process_group(backend="gloo", timeout=timedelta(minutes=30))
    log.info("Process group initialized")


def _inject_custom_dataset(
    cfg: TrainConfig,
    *,
    strategy_name: str,
    strategy_args: Dict[str, Any],
    chunking_args: Dict[str, Any],
    ft_num_train_epochs: int,
    project_root: str,
    replace_per_global_batch: int,
    replace_per_device_batch: Optional[int],
) -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if cfg.global_train_batch_size % world_size != 0:
        raise OLMoCliError(
            "global_train_batch_size must be divisible by WORLD_SIZE to derive per-device batch settings."
        )
    device_train_batch_size = cfg.global_train_batch_size // world_size

    if device_train_batch_size % cfg.device_train_microbatch_size != 0:
        raise OLMoCliError(
            "Derived device_train_batch_size must be divisible by device_train_microbatch_size."
        )
    gradient_accumulation_steps = device_train_batch_size // cfg.device_train_microbatch_size

    if replace_per_device_batch is not None:
        replace_count_device = replace_per_device_batch
    elif replace_per_global_batch > 0:
        replace_count_device = math.ceil(replace_per_global_batch / world_size)
    else:
        replace_count_device = 0
    if replace_count_device < 0:
        raise OLMoCliError("Replacement counts must be >= 0.")
    if replace_count_device > cfg.device_train_microbatch_size:
        raise OLMoCliError(
            "replace-per-device-batch cannot exceed device_train_microbatch_size "
            f"({cfg.device_train_microbatch_size})."
        )

    injection_cursor_offset = 0
    if (
        replace_count_device > 0
        and cfg.load_path is not None
        and not cfg.reset_trainer_state
        and cfg.restore_dataloader
    ):
        checkpoint_dir = Path(cfg.load_path)
        state_path = checkpoint_dir / "train.pt"
        if not state_path.is_file():
            state_path = checkpoint_dir / "other.pt"
        if state_path.is_file():
            try:
                trainer_state = torch.load(state_path, map_location="cpu")
                examples_seen = trainer_state.get(
                    "global_train_examples_seen_this_epoch",
                    trainer_state.get(
                        "global_train_examples_seen",
                        trainer_state.get("global_data_step", trainer_state.get("global_step", 0))
                        * cfg.global_train_batch_size,
                    ),
                )
                steps_seen = int(examples_seen) // int(cfg.global_train_batch_size)
                injection_cursor_offset = steps_seen * replace_count_device
            except Exception as exc:  # pragma: no cover - best-effort only
                log.warning("Failed to infer injection cursor offset from checkpoint state: %s", exc)

    original_paths = cfg.data.paths
    original_datasets = cfg.data.datasets
    original_label_mask_paths = cfg.data.label_mask_paths
    original_memmap_dtype = cfg.data.memmap_dtype
    original_generate_attention_mask = cfg.data.generate_attention_mask
    original_generate_doc_lengths = cfg.data.generate_doc_lengths
    original_instance_filter = cfg.data.instance_filter.asdict() if cfg.data.instance_filter else None
    original_pad_direction = cfg.data.pad_direction.value

    os.environ["FT_OLMO_INJECTION_CONFIG"] = json.dumps(
        {
            "project_root": str(Path(project_root).resolve()),
            "strategy_name": strategy_name,
            "strategy_args": strategy_args,
            "chunking_args": chunking_args,
            "tokenizer_identifier": cfg.tokenizer.identifier,
            "eos_token_id": cfg.model.eos_token_id,
            "pad_token_id": cfg.model.pad_token_id,
            "context_length": cfg.model.max_sequence_length,
            "per_device_train_batch_size": cfg.device_train_microbatch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "num_train_epochs": ft_num_train_epochs,
            "replace_per_device_batch": replace_count_device,
            "pad_direction": original_pad_direction,
            "injection_cursor_offset": injection_cursor_offset,
        }
    )

    cfg.data.paths = None
    cfg.data.datasets = None
    if cfg.data.num_workers != 0:
        log.warning(
            "For deterministic injection behavior, forcing data.num_workers=0 (was %d).",
            cfg.data.num_workers,
        )
        cfg.data.num_workers = 0
    cfg.data.custom_dataset = CustomDatasetConfig(
        name="CheckpointInjectionDataset",
        module="olmo.custom_ft_dataset",
        collate_fn="olmo.custom_ft_dataset.checkpoint_injection_collate",
        args={
            "paths": original_paths,
            "datasets": original_datasets,
            "label_mask_paths": original_label_mask_paths,
            "memmap_dtype": original_memmap_dtype,
            "generate_attention_mask": original_generate_attention_mask,
            "generate_doc_lengths": original_generate_doc_lengths,
            "eos_token_id": cfg.model.eos_token_id,
            "pad_token_id": cfg.model.pad_token_id,
            "context_length": cfg.model.max_sequence_length,
            "include_instance_metadata": False,
            "instance_filter": original_instance_filter,
        },
    )

    log.info(
        "Injected checkpoint-batch mixer: strategy=%s, replace_per_device_batch=%d, grad_accum=%d",
        strategy_name,
        replace_count_device,
        gradient_accumulation_steps,
    )


def main(argv: list[str]) -> None:
    if not argv:
        raise OLMoCliError(f"Usage: {sys.argv[0]} [CONFIG_PATH] [OPTIONS]")

    args, olmo_overrides = _parse_args(argv)
    strategy_args = _parse_json_dict(args.strategy_args_json, "--strategy-args-json")
    chunking_args = _parse_json_dict(args.chunking_args_json, "--chunking-args-json")

    cfg = TrainConfig.load(args.config_path, [clean_opt(s) for s in olmo_overrides])

    if torch.backends.mps.is_available():
        log.info("Device is MPS. Updating config...")
        cfg.model.init_device = "mps"
        cfg.distributed_strategy = "single"  # type: ignore

    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        log.info("Device is CPU. Updating config...")
        cfg.model.init_device = "cpu"
        cfg.distributed_strategy = "single"  # type: ignore

    _inject_custom_dataset(
        cfg,
        strategy_name=args.strategy_name,
        strategy_args=strategy_args,
        chunking_args=chunking_args,
        ft_num_train_epochs=args.ft_num_train_epochs,
        project_root=args.project_root,
        replace_per_global_batch=args.replace_per_global_batch,
        replace_per_device_batch=args.replace_per_device_batch,
    )

    olmo_main(cfg)


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError as exc:
        print(f"failed to set multiprocessing start method: {exc}")
    _init_process_group()
    prepare_cli_environment()
    add_cached_path_clients()
    main(sys.argv[1:])
