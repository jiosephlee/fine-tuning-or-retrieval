#!/usr/bin/env python3
"""Build a fixed-size token replay array from JSONL sources."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Optional, Tuple

import numpy as np
from transformers import AutoTokenizer


def compute_sha256(path: Path) -> str:
    """Compute a SHA-256 checksum for a file without loading it into memory."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tokenize documents into an exact-size, contiguous int32 .npy array."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--input-jsonl",
        nargs="+",
        type=Path,
        help="One or more local JSONL files, read in the supplied order.",
    )
    source.add_argument(
        "--dataset",
        help="Hugging Face dataset name, streamed without downloading it in full.",
    )
    source.add_argument(
        "--olmo-mix-source",
        default=None,
        help=(
            "HF dataset repo containing DCLM OLMo-Mix processed shards. "
            "When provided, the script will resolve --olmo-shard-name by exact match."
        ),
    )
    parser.add_argument("--dataset-config", default=None, help="HF dataset config or subset.")
    parser.add_argument("--dataset-revision", default="main", help="HF dataset repository revision.")
    parser.add_argument("--split", default="train", help="HF dataset split to stream.")
    parser.add_argument("--olmo-shard-name", default="shard_00000136_processed.jsonl", help="Exact shard filename to locate in OLMo-Mix.")
    parser.add_argument("--olmo-shard-prefix", default="data/dclm", help="Prefix filter for OLMo shard lookup.")
    parser.add_argument("--olmo-shard-path", default=None, help="Exact shard path under the repo; skips repo-wide search.")
    parser.add_argument("--source-sha256", default=None, help="Expected SHA-256 for the input shard.")
    parser.add_argument("--hf-token", default=None, help="Optional Hugging Face token for gated model access.")
    parser.add_argument("--tokenizer", required=True, help="HF tokenizer name or local path.")
    parser.add_argument("--tokenizer-revision", default=None, help="Optional tokenizer revision to pin.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target-tokens", type=int, default=100_000_000)
    parser.add_argument("--text-field", default="text")
    parser.add_argument(
        "--framing-policy",
        choices=["auto", "qwen", "llama"],
        default="auto",
        help=(
            "Document framing before separator tokens: qwen => text + EOS, "
            "llama => BOS + text + EOS."
        ),
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10_000,
        help="Print progress every N documents.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom code supplied by the tokenizer repository.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_olmo_shard(
    repo_id: str,
    revision: str,
    shard_name: str,
    expected_prefix: str,
    hf_token: Optional[str] = None,
    explicit_path: Optional[str] = None,
) -> Tuple[Path, dict]:
    """Resolve and download an exact OLMo-Mix DCLM shard from the Hub."""
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required for OLMo-Mix shard resolution: "
            "pip install huggingface-hub"
        ) from exc

    try:
        repo_files = HfApi(token=hf_token).list_repo_files(repo_id, revision=revision, repo_type="dataset")
    except Exception as exc:
        raise SystemExit(f"Unable to list files from {repo_id}@{revision}: {exc}") from exc

    searched_locations = []
    candidates = []
    for file_path in repo_files:
        if Path(file_path).name != shard_name:
            continue
        if expected_prefix and not file_path.startswith(expected_prefix):
            continue
        searched_locations.append(file_path)
        candidates.append(file_path)

    if explicit_path:
        if explicit_path not in searched_locations:
            raise FileNotFoundError(
                f"Exact shard path '{explicit_path}' not found under '{repo_id}' (prefix={expected_prefix}). "
                f"Searched {len(searched_locations)} candidate names; did not find '{shard_name}'."
            )
        chosen_path = explicit_path
    else:
        if len(candidates) == 1:
            chosen_path = candidates[0]
        elif len(candidates) > 1:
            raise FileNotFoundError(
                f"Expected exactly one shard named '{shard_name}', but found {len(candidates)} matches.\n"
                "Set --olmo-shard-path to disambiguate:\n"
                + "\n".join(f"  - {path}" for path in candidates[:20])
            )
        else:
            raise FileNotFoundError(
                f"Could not locate '{shard_name}' under '{repo_id}' with prefix '{expected_prefix}'. "
                f"Candidates scanned: {len(searched_locations)} files."
            )

    local_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=chosen_path,
            repo_type="dataset",
            revision=revision,
            token=hf_token,
        )
    )
    return local_path, {
        "repo_id": repo_id,
        "repo_revision": revision,
        "shard_name": shard_name,
        "shard_path": chosen_path,
        "source_checksum_sha256": compute_sha256(local_path),
        "searched_prefix": expected_prefix,
        "candidates_checked": len(searched_locations),
    }


def iter_jsonl_records(path: Path, text_field: str) -> Iterable[Mapping[str, object]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc
            text = record.get(text_field)
            if not isinstance(text, str):
                raise ValueError(
                    f"Document {path}:{line_number} has no string field {text_field!r}"
                )
            yield {"text": text, "metadata": {"source_path": str(path), "source_line": line_number}}


def iter_records(args: argparse.Namespace) -> Tuple[Iterable[Mapping[str, object]], dict]:
    source_meta: dict
    if args.input_jsonl:
        paths = [path.resolve() for path in args.input_jsonl]
        source_checksums = []
        for path in paths:
            if not path.exists():
                raise FileNotFoundError(f"Missing input file: {path}")
            source_checksums.append(compute_sha256(path))
        source_meta = {
            "mode": "local_jsonl",
            "paths": [str(path) for path in paths],
            "source_sha256": (
                source_checksums[0]
                if len(source_checksums) == 1
                else hashlib.sha256("".join(source_checksums).encode("utf-8")).hexdigest()
            ) if source_checksums else None,
            "source_sha256_by_path": dict(zip((str(path) for path in paths), source_checksums)),
            "expected_sha256": args.source_sha256,
        }
        if args.source_sha256 and source_meta["source_sha256"] != args.source_sha256:
            raise SystemExit(
                f"Input checksum mismatch for local sources: expected {args.source_sha256}, "
                f"found {source_meta['source_sha256']}"
            )

        def _iter() -> Iterable[Mapping[str, object]]:
            for path in paths:
                yield from iter_jsonl_records(path, args.text_field)

        return _iter(), source_meta

    if args.dataset:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise SystemExit("Hugging Face streaming requires: pip install datasets") from exc

        dataset = load_dataset(
            args.dataset,
            args.dataset_config,
            split=args.split,
            revision=args.dataset_revision,
            streaming=True,
        )
        source_meta = {
            "mode": "hf_dataset_stream",
            "dataset": args.dataset,
            "dataset_config": args.dataset_config,
            "dataset_revision": args.dataset_revision,
            "split": args.split,
        }

        def _iter() -> Iterable[Mapping[str, object]]:
            for i, record in enumerate(dataset, start=1):
                text = record.get(args.text_field)
                if not isinstance(text, str):
                    raise ValueError(
                        f"Document {i} from {args.dataset}/{args.split} has no string field {args.text_field!r}"
                    )
                yield {"text": text, "metadata": {"index": i}}

        return _iter(), source_meta

    repo_id = args.olmo_mix_source or "allenai/olmo-mix-1124"
    local_path, resolved = resolve_olmo_shard(
        repo_id=repo_id,
        revision=args.dataset_revision,
        shard_name=args.olmo_shard_name,
        expected_prefix=args.olmo_shard_prefix,
        hf_token=args.hf_token,
        explicit_path=args.olmo_shard_path,
    )
    source_meta = {
        "mode": "olmo_mix_hub_file",
        **resolved,
        "source_repo": repo_id,
        "source_sha256": resolved["source_checksum_sha256"],
        "expected_sha256": args.source_sha256,
        "downloaded_path": str(local_path),
    }

    if args.source_sha256 and source_meta["source_sha256"] != args.source_sha256:
        raise SystemExit(
            f"Input checksum mismatch for '{repo_id}': expected "
            f"{args.source_sha256}, found {source_meta['source_sha256']}"
        )

    return iter_jsonl_records(local_path, args.text_field), source_meta


def detect_framing_policy(tokenizer_id: str, requested: str) -> str:
    if requested != "auto":
        return requested
    normalized = tokenizer_id.lower()
    if "llama-3.1-8b" in normalized or "meta-llama/llama-3.1-8b" in normalized:
        return "llama"
    if "qwen2.5-7b" in normalized or "qwen/qwen2.5-7b" in normalized:
        return "qwen"
    raise ValueError(
        "Unable to infer framing-policy from tokenizer name. "
        "Set --framing-policy explicitly to qwen or llama."
    )


def encode_document(text: str, tokenizer, framing_policy: str, eos_token_id: int, bos_token_id: Optional[int]) -> list[int]:
    core_tokens = tokenizer.encode(text, add_special_tokens=False)
    if framing_policy == "qwen":
        return core_tokens + [eos_token_id]
    if framing_policy == "llama":
        if bos_token_id is None:
            raise SystemExit("LLama framing policy requires tokenizer.bos_token_id.")
        return [bos_token_id] + core_tokens + [eos_token_id]
    raise ValueError(f"Unsupported framing-policy: {framing_policy}")


def main() -> None:
    args = parse_args()
    if args.target_tokens <= 0:
        raise SystemExit("--target-tokens must be positive")
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"Output already exists: {args.output} (use --overwrite)")

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=args.trust_remote_code,
        revision=args.tokenizer_revision,
        token=args.hf_token,
    )
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise SystemExit("Tokenizer has no EOS token.")

    framing_policy = detect_framing_policy(args.tokenizer, args.framing_policy)
    tokenizer_id = tokenizer.name_or_path
    tokenizer_revision = tokenizer.init_kwargs.get("revision") if hasattr(tokenizer, "init_kwargs") else None
    if tokenizer_revision is None:
        tokenizer_revision = args.tokenizer_revision

    records, source_meta = iter_records(args)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial_path = args.output.with_name(args.output.name + ".partial")
    if partial_path.exists():
        partial_path.unlink()

    output = np.lib.format.open_memmap(
        partial_path,
        mode="w+",
        dtype=np.int32,
        shape=(args.target_tokens,),
    )
    tokens_written = 0
    documents = 0
    last_document_truncated = False
    min_token_id: Optional[int] = None
    max_token_id: Optional[int] = None

    try:
        for record in records:
            text = record.get("text", "")
            token_ids = encode_document(
                text,
                tokenizer,
                framing_policy,
                eos_token_id=eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
            )
            if token_ids:
                token_min = min(token_ids)
                token_max = max(token_ids)
                min_token_id = token_min if min_token_id is None else min(min_token_id, token_min)
                max_token_id = token_max if max_token_id is None else max(max_token_id, token_max)

            remaining = args.target_tokens - tokens_written
            take = min(len(token_ids), remaining)
            output[tokens_written : tokens_written + take] = token_ids[:take]
            tokens_written += take
            documents += 1
            if take < len(token_ids):
                last_document_truncated = True

            if args.log_every > 0 and documents % args.log_every == 0:
                print(f"documents={documents:,} tokens={tokens_written:,}", flush=True)
            if tokens_written == args.target_tokens:
                break

        if tokens_written != args.target_tokens:
            raise RuntimeError(
                f"Source ended after {tokens_written:,} tokens; "
                f"target was {args.target_tokens:,}"
            )
        output.flush()
    except BaseException:
        del output
        partial_path.unlink(missing_ok=True)
        raise

    del output
    final_path = args.output
    final_path.parent.mkdir(parents=True, exist_ok=True)
    output_checksum = compute_sha256(partial_path)
    partial_path.replace(final_path)
    metadata = {
        "tokenizer": {
            "id": tokenizer_id,
            "revision": tokenizer_revision,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": eos_token_id,
            "vocab_size": tokenizer.vocab_size,
        },
        "framing_policy": framing_policy,
        "add_special_tokens": False,
        "text_field": args.text_field,
        "target_tokens": args.target_tokens,
        "documents_read": documents,
        "truncated_last_document": last_document_truncated,
        "dtype": "int32",
        "source": source_meta,
        "shape": [args.target_tokens],
        "output_checksum_sha256": output_checksum,
        "token_id_range": {
            "min": None if min_token_id is None else int(min_token_id),
            "max": None if max_token_id is None else int(max_token_id),
        },
    }

    if args.dataset:
        metadata.update({
            "dataset": args.dataset,
            "dataset_config": args.dataset_config,
            "split": args.split,
            "dataset_revision": args.dataset_revision,
        })

    metadata_path = final_path.with_suffix(final_path.suffix + ".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {tokens_written:,} tokens from {documents:,} documents to {final_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
