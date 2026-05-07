import os
import sys
import json
import math
import argparse
import pandas as pd

# -----------------------------------------------------------
# CONFIG
# -----------------------------------------------------------

DEFAULT_PROJECT_ROOT = "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval"

# Domains for inference probes
DOMAINS = ["1_58", "DPO", "GRPO", "BOFT", "OFT", "QLoRA"]

# -----------------------------------------------------------
# UTIL LOADING
# -----------------------------------------------------------

def load_utils(project_root: str):
    """
    Add project root to sys.path and import utils.utils as utils.
    """
    sys.path.append(project_root)
    import importlib
    import utils.utils as utils
    from utils import probe_paths
    importlib.reload(utils)
    return utils, probe_paths

# -----------------------------------------------------------
# LLM SPLIT CALL
# -----------------------------------------------------------

def build_split_prompt(probes_df: pd.DataFrame):
    """
    Build system + user messages for the LLM to split probes into train/test sets.
    We enumerate probes from 0 to N-1 and provide title+fact text.
    """
    # Build a simple numbered list of probes
    probe_items = []
    for idx, row in probes_df.iterrows():
        fact = str(row["fact"])
        probe_items.append({
            "index": int(idx),
            "fact": fact,
        })

    # User message: list probes as JSON array
    user_content = {
        "description": (
            "You are given a list of inference probes (questions or statements) "
            "from a single domain. Each probe has an integer index and a text 'fact'. "
            "Your job is to split these probes into a small TRAIN set and a larger TEST set."
        ),
        "probes": probe_items,
    }

    system_msg = (
        "You are helping design a train/test split for a probing experiment on a single domain.\n\n"
        "You will receive a JSON object containing a list of probes. Each probe has:\n"
        "- 'index': an integer ID (0-based),\n"
        "- 'fact': the probe text.\n\n"
        "Your task:\n"
        "1. Split the probes into two disjoint sets:\n"
        "   - 'train_indices': about 25% of the probes (as close as possible).\n"
        "   - 'test_indices': the remaining ~75%.\n"
        "2. The key constraint: TRAIN and TEST should be as DIFFERENT as possible.\n"
        "   - If two probes ask about very similar concepts, share very similar wording, or\n"
        "     differ only by small details, they should be placed in the same split whenever possible.\n"
        "   - The main goal is to avoid overlapping concepts/string patterns across train and test.\n"
        "3. Each index must appear in EXACTLY ONE of the two sets. No index may be duplicated or omitted.\n"
        "4. Try to keep the train set size close to 25% of all probes (± a couple of probes is OK).\n\n"
        "Output format:\n"
        "Return a single JSON object with two keys:\n"
        "{\n"
        "  \"train_indices\": [list of integer indices],\n"
        "  \"test_indices\": [list of integer indices]\n"
        "}\n"
        "Do not include any other keys.\n"
        "Do not repeat the full probes, only the indices.\n"
    )

    prompt_messages = {
        "system": system_msg,
        "user": json.dumps(user_content, ensure_ascii=False)
    }

    return prompt_messages

def call_llm_for_split(utils, probes_df: pd.DataFrame, model: str = "gpt-5"):
    """
    Call the LLM via utils.query_llm to get train/test indices.
    Returns a Python dict: {"train_indices": [...], "test_indices": [...]}
    """
    prompt = build_split_prompt(probes_df)

    response_str = utils.query_llm(
        prompt,
        model=model,
        reasoning_effort="medium",
        system_prompt_included=True,
        return_json=True,
        is_hippa=True,
        max_tokens=4000,
    )

    # Parse JSON
    try:
        split_plan = json.loads(response_str)
    except json.JSONDecodeError:
        raise ValueError("LLM response is not valid JSON:\n" + response_str)

    # Basic validation of keys
    if not isinstance(split_plan, dict):
        raise ValueError("LLM response JSON is not an object:\n" + response_str)

    if "train_indices" not in split_plan or "test_indices" not in split_plan:
        raise ValueError(
            "LLM response JSON must contain 'train_indices' and 'test_indices' keys.\n"
            f"Got: {split_plan.keys()}"
        )

    return split_plan, response_str

# -----------------------------------------------------------
# SPLIT VALIDATION / FIX-UP
# -----------------------------------------------------------

def validate_and_fix_split(split_plan, num_probes: int):
    """
    Ensure every index 0..num_probes-1 is assigned exactly once.
    If the LLM output is slightly inconsistent, perform minimal fix-ups.
    """
    all_indices = set(range(num_probes))

    train = set(split_plan.get("train_indices", []))
    test = set(split_plan.get("test_indices", []))

    # Remove out-of-range indices
    train = {i for i in train if 0 <= i < num_probes}
    test = {i for i in test if 0 <= i < num_probes}

    # Remove overlaps: if an index appears in both, keep in train and remove from test
    overlap = train & test
    if overlap:
        test -= overlap

    # Now compute which indices are still unassigned
    assigned = train | test
    missing = all_indices - assigned

    # Assign missing ones to whichever split is currently smaller
    for i in sorted(missing):
        if len(train) < len(test):
            train.add(i)
        else:
            test.add(i)

    # If train is wildly off from 25%, you can optionally rebalance,
    # but here we just accept the approximate split from the LLM + small fix-ups.
    # Convert back to sorted lists
    train_indices = sorted(train)
    test_indices = sorted(test)

    return train_indices, test_indices

# -----------------------------------------------------------
# MAIN PER-DOMAIN OPERATION
# -----------------------------------------------------------

def process_domain(project_root: str, utils, domain: str, model: str = "gpt-5"):
    """
    For a single domain:
    - Load probes/.../{domain}/inference/probes_v7.csv
    - Call LLM to split into train/test indices (~25/75, minimal overlap)
    - Save train_probes_v7.csv and test_probes_v7.csv
    - Save split_plan.json with the raw LLM response
    """
    from utils import probe_paths
    probes_path = str(probe_paths.resolve_probe_path("inference", domain, "v7"))

    if not os.path.exists(probes_path):
        print(f"[{domain}] probes_v7.csv not found at {probes_path}, skipping.")
        return

    df = pd.read_csv(probes_path)
    if "fact" not in df.columns:
        raise ValueError(f"[{domain}] 'fact' column not found in {probes_path}.")

    num_probes = df.shape[0]
    print(f"[{domain}] Loaded {num_probes} probes from {probes_path}.")

    # Call LLM for split
    split_plan, raw_response = call_llm_for_split(utils, df, model=model)

    # Save raw LLM response for auditing
    split_plan_path = str(probe_paths.resolve_probe_dir("inference", domain) / "split_plan.json")
    with open(split_plan_path, "w") as f:
        f.write(raw_response)
    print(f"[{domain}] Saved raw split plan to {split_plan_path}")

    # Validate / fix split
    train_indices, test_indices = validate_and_fix_split(split_plan, num_probes)
    print(f"[{domain}] Final train size: {len(train_indices)}, test size: {len(test_indices)}")

    # Extract and save train/test CSVs
    train_df = df.iloc[train_indices].copy()
    test_df = df.iloc[test_indices].copy()

    train_out_path = str(probe_paths.resolve_probe_dir("inference", domain) / "train_probes_v7.csv")
    test_out_path = str(probe_paths.resolve_probe_dir("inference", domain) / "test_probes_v7.csv")

    train_df.to_csv(train_out_path, index=False)
    test_df.to_csv(test_out_path, index=False)

    print(f"[{domain}] Wrote train split to {train_out_path}")
    print(f"[{domain}] Wrote test split to {test_out_path}")

# -----------------------------------------------------------
# MAIN SCRIPT ENTRY
# -----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Use an LLM to split inference probes into train/test sets for each domain, "
            "aiming for 25%/75% and minimal lexical/semantic overlap between the splits."
        )
    )
    parser.add_argument(
        "--project_root",
        type=str,
        default=DEFAULT_PROJECT_ROOT,
        help="Path to project root (where 'data' and 'utils' live).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai-gpt-5",
        help="LLM model name to use via utils.query_llm.",
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

    # Load utils
    utils, _ = load_utils(project_root)

    print(f"Project root: {project_root}")
    print(f"Domains: {args.domains}")
    print(f"Using model: {args.model}")

    for domain in args.domains:
        print(f"\n=== Processing domain: {domain} ===")
        try:
            process_domain(project_root, utils, domain, model=args.model)
        except Exception as e:
            print(f"[{domain}] ERROR during processing: {e}")

    print("\nDone.")

if __name__ == "__main__":
    main()
