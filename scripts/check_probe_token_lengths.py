import argparse
import csv
import glob
import os
from typing import List, Tuple

from transformers import AutoTokenizer


def compute_lengths(tokenizer, texts: List[str]) -> Tuple[List[int], List[int], dict]:
    """
    Mimic BaseKnowledgeProbeCallBack._precompute_token_lengths but without torch.

    Returns:
        lengths_new: lengths from non-padded tokenization (len(input_ids)).
        lengths_orig: lengths from padded attention_mask sums.
        tokenized_pad: padded tokenization dict (lists of lists).
    """
    tokenized_no_pad = tokenizer(texts, padding=False, add_special_tokens=False)
    tokenized_pad = tokenizer(texts, padding=True, add_special_tokens=False)

    lengths_new = [len(ids) for ids in tokenized_no_pad["input_ids"]]
    lengths_orig = [sum(mask) for mask in tokenized_pad["attention_mask"]]
    return lengths_new, lengths_orig, tokenized_pad


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Check probe/inference CSV for token length and target-token "
            "mismatches, mimicking utils.llm_callbacks.BaseKnowledgeProbeCallBack."
        )
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="allenai/OLMo-2-1124-7B",
        help="Hugging Face model ID whose tokenizer should be used.",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="data/probes/inference/QLoRA/probes_v7.csv",
        help="Path to the probes CSV (must contain 'probe', 'target', and 'fact' columns).",
    )
    parser.add_argument(
        "--max_print",
        type=int,
        default=50,
        help="Maximum number of problematic probes to print in detail.",
    )

    args = parser.parse_args()

    print(f"Loading tokenizer: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    pad_id = tokenizer.pad_token_id
    print(f"pad_token_id: {pad_id}")

    path = args.csv_path

    # If a directory is provided, recursively check all probes_*.csv files under it.
    if os.path.isdir(path):
        pattern = os.path.join(path, "**", "probes_*.csv")
        csv_files = sorted(glob.glob(pattern, recursive=True))
        if not csv_files:
            print(f"No CSV files matching 'probes_*.csv' found under directory: {path}")
            return

        print(f"\nFound {len(csv_files)} probe CSV files under {path}")
        for csv_path in csv_files:
            print("\n" + "=" * 80)
            print(f"Checking file: {csv_path}")
            print("=" * 80)
            _check_single_csv(csv_path, tokenizer, pad_id, args.max_print)
    else:
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV path does not exist: {path}")
        _check_single_csv(path, tokenizer, pad_id, args.max_print)

    print("\nDone.")


def _check_single_csv(csv_path: str, tokenizer, pad_id: int, max_print: int) -> None:
    # --- Load CSV ---
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Defensive: ensure key presence
            if not {"probe", "target", "fact"}.issubset(row.keys()):
                raise ValueError(
                    f"CSV {csv_path} must contain 'probe', 'target', and 'fact' columns."
                )
            rows.append(
                {
                    "probe": row["probe"],
                    "target": row["target"],
                    "fact": row["fact"],
                }
            )

    print(f"Loaded {len(rows)} probes from {csv_path}")

    probes = [r["probe"] for r in rows]
    targets = [r["target"] for r in rows]
    facts = [r["fact"] for r in rows]

    # --- Precompute token lengths like _precompute_token_lengths ---
    print("\nTokenizing probes (contexts)...")
    context_lengths_new, context_lengths_orig, tokenized_probes_pad = compute_lengths(
        tokenizer, probes
    )

    print("Tokenizing targets...")
    target_lengths_new, target_lengths_orig, tokenized_targets_pad = compute_lengths(
        tokenizer, targets
    )

    print("Tokenizing facts...")
    fact_lengths_new, fact_lengths_orig, tokenized_facts_pad = compute_lengths(
        tokenizer, facts
    )

    # --- Sanity checks mirroring BaseKnowledgeProbeCallBack._precompute_token_lengths ---
    context_len_mismatch = [
        i for i, (a, b) in enumerate(zip(context_lengths_new, context_lengths_orig)) if a != b
    ]
    target_len_mismatch = [
        i for i, (a, b) in enumerate(zip(target_lengths_new, target_lengths_orig)) if a != b
    ]
    fact_len_mismatch = [
        i for i, (a, b) in enumerate(zip(fact_lengths_new, fact_lengths_orig)) if a != b
    ]

    print("\n=== Length calculation mismatches (new vs. orig) ===")
    print(f"Context length mismatches: {len(context_len_mismatch)}")
    print(f"Target length mismatches : {len(target_len_mismatch)}")
    print(f"Fact length mismatches   : {len(fact_len_mismatch)}")

    # Mismatch between (context + target) and fact length
    ct_vs_fact_mismatch = [
        i
        for i, (c, t, f) in enumerate(
            zip(context_lengths_new, target_lengths_new, fact_lengths_new)
        )
        if c + t != f
    ]
    print(
        f"Indices where context_len + target_len != fact_len: "
        f"{len(ct_vs_fact_mismatch)}"
    )

    # --- Reproduce evaluation-time checks (no model needed) ---
    print("\n=== Evaluation-style checks (length + target token count) ===")
    eval_len_mismatch_indices = []
    token_count_mismatch_info = []

    for idx in range(len(rows)):
        c_len = context_lengths_new[idx]
        t_len = target_lengths_new[idx]
        fact_ids = tokenized_facts_pad["input_ids"][idx]
        attn_mask = tokenized_facts_pad["attention_mask"][idx]
        fact_attn_len = sum(attn_mask)

        # This mirrors the check in _evaluate_probes:
        # if not torch.equal(context_lengths + target_lengths, attention_mask.sum(dim=1)):
        if c_len + t_len != fact_attn_len:
            eval_len_mismatch_indices.append(idx)

        # Now replicate the target-token counting logic in _calculate_log_probs.
        # In training code:
        #   shift_labels = input_ids[..., 1:]
        #   context_lengths is then decremented by 1 before being passed.
        shift_labels = fact_ids[1:]
        context_len_shifted = c_len - 1

        # Skip impossible cases (would be caught by earlier assertions in training).
        if context_len_shifted < 0:
            continue

        start = context_len_shifted
        end = start + t_len

        if start >= len(shift_labels):
            # No target tokens possible in shifted labels.
            slice_ids = []
        else:
            slice_ids = shift_labels[start:end]

        num_tokens_target = sum(1 for tok in slice_ids if tok != pad_id)

        if num_tokens_target != t_len:
            token_count_mismatch_info.append(
                {
                    "index": idx,
                    "expected_target_len": t_len,
                    "counted_target_tokens": num_tokens_target,
                    "target_token_ids_slice": slice_ids,
                    "probe": rows[idx]["probe"],
                    "target": rows[idx]["target"],
                    "fact": rows[idx]["fact"],
                }
            )

    print(
        f"Length mismatches between (context+target) and fact (eval-style): "
        f"{len(eval_len_mismatch_indices)}"
    )
    print(
        f"Target token-count mismatches (like callback warning): "
        f"{len(token_count_mismatch_info)}"
    )

    # --- Print detailed info for problematic probes ---
    if eval_len_mismatch_indices:
        print("\n--- Probes with (context+target) != fact length ---")
        for idx in eval_len_mismatch_indices[:max_print]:
            c_len = context_lengths_new[idx]
            t_len = target_lengths_new[idx]
            f_len = fact_lengths_new[idx]
            print(f"\nIndex {idx}:")
            print(f"  context_len: {c_len}")
            print(f"  target_len : {t_len}")
            print(f"  fact_len   : {f_len}")
            print(f"  probe      : {rows[idx]['probe']}")
            print(f"  target     : {rows[idx]['target']}")
            print(f"  fact       : {rows[idx]['fact']}")

    if token_count_mismatch_info:
        print("\n--- Probes with target token-count mismatches ---")
        for info in token_count_mismatch_info[:max_print]:
            idx = info["index"]
            print(f"\nIndex {idx}:")
            print(f"  expected target len   : {info['expected_target_len']}")
            print(f"  counted target tokens : {info['counted_target_tokens']}")
            print(f"  target token ids slice: {info['target_token_ids_slice']}")
            print(f"  probe                 : {info['probe']}")
            print(f"  target                : {info['target']}")
            print(f"  fact                  : {info['fact']}")

if __name__ == "__main__":
    main()
