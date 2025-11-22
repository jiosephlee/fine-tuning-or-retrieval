import os
import sys
import json
import argparse
from typing import List, Optional

import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# -----------------------------------------------------------
# CONFIG
# -----------------------------------------------------------

DEFAULT_PROJECT_ROOT = "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval"

# Domains you care about
DOMAINS = ["1_58", "DPO", "GRPO", "BOFT", "OFT", "QLoRA"]

# Where cleaned papers live (relative to project_root)
CLEANED_PAPER_DIR = os.path.join("data", "arxiv", "cleaned")

# Where explanations live (relative to project_root)
EXPLANATIONS_BASE_DIR = os.path.join("data", "arxiv", "explanations")

# Two split configurations:
SPLITS = [
    # (name, input_csv, output_subdir_base)
    ("llm_probes", "train_probes_v7.csv", "llm_probes"),
    ("inference_type_probes", "type_split_train_probes_v7.csv", "inference_type_probes"),
]

MAX_WORKERS = 4

# -----------------------------------------------------------
# UTILS LOADING
# -----------------------------------------------------------

def load_utils(project_root: str):
    """
    Add project root to sys.path and import utils.utils as utils.
    """
    sys.path.append(project_root)
    import importlib
    import utils.utils as utils
    importlib.reload(utils)
    return utils

# -----------------------------------------------------------
# LLM CALL FOR EXPLANATIONS
# -----------------------------------------------------------

def build_explanation_prompt(paper_text: str, fact: str, contextualized, textbook_style) -> dict:
    """
    Build messages dict for utils.query_llm to generate a single-paragraph explanation
    of a probe fact using the original paper as context.

    Desired template: "{fact}. {reason for fact}"
    """
    if contextualized:
           system_msg = (
        "You are an expert instructor explaining research papers to advanced students.\n\n"
        "You will be given:\n"
        "- The full LaTeX content of a research paper.\n"
        "- A single factual claim derived from that paper that is not explicitly stated in the text.\n\n"
        "Your task is to write a 2-3 paragraphs of explanation of why this fact is true or sensible\n"
        "based on the paper. Use the following template:\n"
        "  {fact}. {reason for fact}\n\n"
        "Requirements:\n"
        "- Begin the paragraph by restating the fact exactly (or nearly exactly), followed by an explanation.\n"
        "- Write all math expressions in LaTeX.\n"
    )
    elif textbook_style:
        system_msg = (
            "You are an expert textbook author writing for college students.\n\n"
            "You will be given:\n"
            "- The full text/LaTeX content of a research paper.\n"
            "- A single fact derived from that paper that is not explicitly stated in the text.\n\n"
            "Your task is to write a detailed, cohesive textbook chapter based strictly on the provided paper.\n"
            "The chapter must be comprehensive and suitable for a student learning this material for the first time "
            "in order to understand the knowledge presented in the fact.\n\n"
            "Guidelines:\n"
            "1. **Content & Depth**: Elaborate on the provided fact at full length with a focus on intuition. "
            "Spell everything out clearly to remove ambiguity. Dedicate multiple paragraphs to each subtopic.\n"
            "2. **Strict Grounding**: Your output must be grounded solely in the provided paper. "
            "Do not incorporate outside information or details not found in the source text.\n"
            "3. **Format**: Write in full prose. Do not use bullet points. "
            "Start with the Chapter Title on the first line. Use a section header '#' for each subtopic.\n"
            "4. **Mathematical Notation**: Write ALL mathematical notation in LaTeX only (e.g., $x^2$, $\\pi$). "
            "Do NOT use unicode mathematical characters.\n"
        )
    else:
        system_msg = (
            "You are an expert instructor explaining research papers to advanced students.\n\n"
            "You will be given:\n"
            "- The full LaTeX content of a research paper.\n"
            "- A single factual claim derived from that paper that is not explicitly stated in the text.\n\n"
            "Your task is to write a SINGLE PARAGRAPH explanation of why this fact is true or sensible\n"
            "based on the paper. Use the following template:\n"
            "  {fact}. {reason for fact}\n\n"
            "Requirements:\n"
            "- Begin the paragraph by restating the fact exactly (or nearly exactly), followed by an explanation.\n"
            "- Write all math expressions in LaTeX.\n"
        )

    user_content = {
        "paper_latex": paper_text,
        "fact": fact,
    }

    prompt = {
        "system": system_msg,
        "user": json.dumps(user_content, ensure_ascii=False)
    }
    return prompt

def generate_explanation_for_fact(
    utils,
    model: str,
    paper_text: str,
    fact: str,
    reasoning_effort: str = "low",
    max_tokens: int = 512,
    contextualized: bool = False,
    textbook_style=False,
) -> str:
    """
    Call LLM once to generate an explanation paragraph for a single fact.
    Returns plain text.
    """
    prompt = build_explanation_prompt(paper_text, fact, contextualized, textbook_style)
    response = utils.query_llm(
        prompt,
        model=model,
        reasoning_effort=reasoning_effort,
        system_prompt_included=True,
        return_json=False,         # plain text
        max_tokens=max_tokens,
        is_hippa=True,
    )
    return str(response).strip()

# -----------------------------------------------------------
# CORE PROCESSING
# -----------------------------------------------------------

def load_cleaned_paper(project_root: str, domain: str) -> Optional[str]:
    """
    Load the cleaned LaTeX paper for a domain, e.g. data/arxiv/cleaned/{domain}.tex
    """
    paper_path = os.path.join(project_root, CLEANED_PAPER_DIR, f"{domain}.tex")
    if not os.path.exists(paper_path):
        print(f"[{domain}] Paper not found at {paper_path}. Explanations will have no context.")
        return None

    with open(paper_path, "r") as f:
        return f.read()

def generate_explanations_parallel(
    utils,
    model: str,
    paper_text: str,
    facts: List[str],
    domain: str,
    split_name: str,
    contextualized=False,
    textbook_style=False,
) -> List[str]:
    """
    Generate explanations for a list of facts in parallel with a ThreadPoolExecutor.
    Preserves order of 'facts' in the returned list.
    """
    explanations: List[str] = [""] * len(facts)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_idx = {}
        for idx, fact in enumerate(facts):
            if not fact:
                explanations[idx] = ""
                continue
            fut = executor.submit(
                generate_explanation_for_fact,
                utils,
                model,
                paper_text,
                fact,
                contextualized=contextualized,
                textbook_style=textbook_style
            )
            future_to_idx[fut] = idx

        for fut in tqdm(
            as_completed(future_to_idx),
            total=len(future_to_idx),
            desc=f"{domain}::{split_name} (explanations)"
        ):
            idx = future_to_idx[fut]
            try:
                explanations[idx] = fut.result()
            except Exception as e:
                print(f"[{domain}][{split_name}] ERROR generating explanation for probe {idx}: {e}")
                explanations[idx] = facts[idx]  # fallback: just repeat fact

    return explanations

def process_train_split_for_domain(
    project_root: str,
    utils,
    domain: str,
    split_name: str,
    input_csv_name: str,
    output_subdir_base: str,
    model: str,
    max_probes: Optional[int] = None,
):
    """
    Generic function that:
    - Reads a train-split CSV of probes for a given domain.
    - Writes plain facts concatenated by \\n\\n to:
        data/arxiv/explanations/{domain}/{output_subdir_base}/train_probes.txt
    - Uses LLM (in parallel) to generate explanations for each fact, based on the cleaned paper:
        data/arxiv/cleaned/{domain}.tex
      and writes them concatenated by \\n\\n to:
        data/arxiv/explanations/{domain}/{output_subdir_base}_with_explanations/train_probes.txt
    """
    input_csv_path = os.path.join(
        project_root,
        "data",
        "probes",
        "inference",
        domain,
        input_csv_name,
    )

    if not os.path.exists(input_csv_path):
        print(f"[{domain}][{split_name}] Input CSV not found at {input_csv_path}, skipping.")
        return

    df = pd.read_csv(input_csv_path)
    if "fact" not in df.columns:
        print(f"[{domain}][{split_name}] 'fact' column not found in {input_csv_path}, skipping.")
        return

    if max_probes is not None and df.shape[0] > max_probes:
        df = df.head(max_probes).copy()
        print(f"[{domain}][{split_name}] Limiting to first {max_probes} probes for this run.")

    num_probes = df.shape[0]
    print(f"[{domain}][{split_name}] Loaded {num_probes} train probes from {input_csv_path}.")

    # --- 1) Write raw facts files (one file per probe) ---
    facts = [str(fact).strip() for fact in df["fact"].tolist() if isinstance(fact, str)]

    base_out_dir = os.path.join(
        project_root,
        EXPLANATIONS_BASE_DIR,
        domain,
        output_subdir_base,
    )
    os.makedirs(base_out_dir, exist_ok=True)

    # for idx, fact in enumerate(facts, start=1):
    #     fact_path = os.path.join(base_out_dir, f"train_probe_{idx}.txt")
    #     with open(fact_path, "w") as f:
    #         f.write(fact + "\n")
    # print(f"[{domain}][{split_name}] Wrote {len(facts)} plain fact files to {base_out_dir}")

    # --- 2) Load paper context ---
    paper_text = load_cleaned_paper(project_root, domain)
    if paper_text is None:
        print(f"[{domain}][{split_name}] No paper context; skipping explanation generation.")
        return

    # # --- 3) Generate explanations in parallel ---
    # print(f"[{domain}][{split_name}] Generating explanations with model '{model}' (max_workers={MAX_WORKERS})...")
    # explanations = generate_explanations_parallel(
    #     utils=utils,
    #     model=model,
    #     paper_text=paper_text,
    #     facts=facts,
    #     domain=domain,
    #     split_name=split_name,
    # )

    # explanations_text = "\n\n".join(explanations)

    # # --- 4) Write explanations file ---
    # expl_out_dir = os.path.join(
    #     project_root,
    #     EXPLANATIONS_BASE_DIR,
    #     domain,
    #     f"{output_subdir_base}_with_explanations",
    # )
    # os.makedirs(expl_out_dir, exist_ok=True)

    # expl_out_path = os.path.join(expl_out_dir, "train_probes.txt")
    # with open(expl_out_path, "w") as f:
    #     f.write(explanations_text + "\n")

    # print(f"[{domain}][{split_name}] Wrote explanations to {expl_out_path}")
    

    # --- 5) Generate explanations in parallel ---
    print(f"[{domain}][{split_name}] Generating textbook explanations with model '{model}' (max_workers={MAX_WORKERS})...")
    explanations = generate_explanations_parallel(
        utils=utils,
        model=model,
        paper_text=paper_text,
        facts=facts,
        domain=domain,
        split_name=split_name,
        textbook_style=True
    )

    # --- 6) Write explanations files (one file per probe) ---
    expl_out_dir = os.path.join(
        project_root,
        EXPLANATIONS_BASE_DIR,
        domain,
        f"{output_subdir_base}_with_explanations_textbooks_style",
    )
    os.makedirs(expl_out_dir, exist_ok=True)

    for idx, explanation in enumerate(explanations, start=1):
        expl_out_path = os.path.join(expl_out_dir, f"train_probe_{idx}.txt")
        with open(expl_out_path, "w") as f:
            f.write(explanation + "\n")

    print(f"[{domain}][{split_name}] Wrote {len(explanations)} explanation files to {expl_out_dir}")

# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate plain train-probe texts and LLM-based explanations for train probes "
            "under both LLM-split and inference-type-split configurations."
        )
    )
    parser.add_argument(
        "--project_root",
        type=str,
        default=DEFAULT_PROJECT_ROOT,
        help="Path to project root (where 'data' and 'utils' live).",
    )
    parser.add_argument(
        "--domains",
        type=str,
        nargs="*",
        default=DOMAINS,
        help="List of domains to process (default: all).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai-gpt-5",
        help="LLM model name to use via utils.query_llm.",
    )
    parser.add_argument(
        "--max_probes",
        type=int,
        default=None,
        help="Optional cap on number of train probes per domain/split (for quick tests).",
    )

    args = parser.parse_args()
    project_root = os.path.abspath(args.project_root)

    print(f"Project root: {project_root}")
    print(f"Domains: {args.domains}")
    print(f"Model: {args.model}")
    print(f"max_workers for explanations: {MAX_WORKERS}")

    # Load utils
    utils = load_utils(project_root)

    for domain in args.domains:
        print(f"\n=== Processing domain: {domain} ===")

        for split_name, input_csv, output_subdir in SPLITS:
            print(f"--- Split: {split_name} (CSV: {input_csv}) ---")
            try:
                process_train_split_for_domain(
                    project_root=project_root,
                    utils=utils,
                    domain=domain,
                    split_name=split_name,
                    input_csv_name=input_csv,
                    output_subdir_base=output_subdir,
                    model=args.model,
                    max_probes=args.max_probes,
                )
            except Exception as e:
                print(f"[{domain}][{split_name}] ERROR: {e}")

    print("\nDone.")

if __name__ == "__main__":
    main()
