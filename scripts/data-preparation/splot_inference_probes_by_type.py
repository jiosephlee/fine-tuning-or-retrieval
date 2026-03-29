import os
import sys
import argparse
import pandas as pd

# -----------------------------------------------------------
# CONFIG
# -----------------------------------------------------------

DEFAULT_PROJECT_ROOT = "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval"

# Domains for inference probes
DOMAINS = ["1_58", "DPO", "GRPO", "BOFT", "OFT", "QLoRA"]

# Name of the main probes file
MAIN_PROBES_FILENAME = "probes_v7.csv"

# Output filenames for the type-based split
TYPE_TRAIN_FILENAME = "type_split_train_probes_v7.csv"
TYPE_TEST_FILENAME = "type_split_test_probes_v7.csv"

TARGET_INFERENCE_TYPE_FOR_TEST = "Conceptual Synthesis"

# -----------------------------------------------------------
# PER-DOMAIN PROCESSING
# -----------------------------------------------------------

def process_domain(project_root: str, domain: str):
    """
    For a single domain:
    - Load data/probes/inference/{domain}/probes_v7.csv
    - Split by 'inference_type':
        * TEST: inference_type == 'Conceptual Synthesis'
        * TRAIN: everything else
    - Save:
        * type_split_train_probes_v7.csv
        * type_split_test_probes_v7.csv
    """
    probes_path = os.path.join(
        project_root,
        "data",
        "probes",
        "inference",
        domain,
        MAIN_PROBES_FILENAME,
    )

    if not os.path.exists(probes_path):
        print(f"[{domain}] {MAIN_PROBES_FILENAME} not found at {probes_path}, skipping.")
        return

    df = pd.read_csv(probes_path)
    if "inference_type" not in df.columns:
        raise ValueError(f"[{domain}] 'inference_type' column not found in {probes_path}.")

    num_total = df.shape[0]
    print(f"[{domain}] Loaded {num_total} probes from {probes_path}.")

    # Define train/test based on inference_type
    test_mask = df["inference_type"] == TARGET_INFERENCE_TYPE_FOR_TEST
    test_df = df[test_mask].copy()
    train_df = df[~test_mask].copy()

    num_test = test_df.shape[0]
    num_train = train_df.shape[0]

    print(
        f"[{domain}] Split by inference_type:\n"
        f"  TEST  (inference_type == '{TARGET_INFERENCE_TYPE_FOR_TEST}'): {num_test}\n"
        f"  TRAIN (everything else)                                 : {num_train}"
    )

    # Output paths
    out_train_path = os.path.join(
        project_root,
        "data",
        "probes",
        "inference",
        domain,
        TYPE_TRAIN_FILENAME,
    )
    out_test_path = os.path.join(
        project_root,
        "data",
        "probes",
        "inference",
        domain,
        TYPE_TEST_FILENAME,
    )

    # Save splits
    train_df.to_csv(out_train_path, index=False)
    test_df.to_csv(out_test_path, index=False)

    print(f"[{domain}] Wrote train split to {out_train_path}")
    print(f"[{domain}] Wrote test split to  {out_test_path}")

# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Split inference probes into train/test based on 'inference_type' "
            "metadata: test = 'Conceptual Synthesis', train = everything else, "
            "for each domain."
        )
    )
    parser.add_argument(
        "--project_root",
        type=str,
        default=DEFAULT_PROJECT_ROOT,
        help="Path to project root (where 'data' lives).",
    )
    parser.add_argument(
        "--domains",
        type=str,
        nargs="*",
        default=DOMAINS,
        help="List of domains to process (default: all).",
    )

    args = parser.parse_args()
    project_root = os.path.abspath(args.project_root)

    print(f"Project root: {project_root}")
    print(f"Domains: {args.domains}")
    print(f"Target inference_type for TEST: '{TARGET_INFERENCE_TYPE_FOR_TEST}'")

    for domain in args.domains:
        print(f"\n=== Processing domain: {domain} ===")
        try:
            process_domain(project_root, domain)
        except Exception as e:
            print(f"[{domain}] ERROR during processing: {e}")

    print("\nDone.")

if __name__ == "__main__":
    main()