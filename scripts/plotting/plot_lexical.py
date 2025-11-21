import os
import sys
import re
import math
import argparse
import collections

from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

# Adjust to your repo root if needed
DEFAULT_PROJECT_ROOT = "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval"

# Domains used for probes + textbooks
DOMAINS = ["1_58", "DPO", "GRPO", "BOFT", "OFT", "QLoRA"]

# Explicit 7B runs you gave
RUNS_7B = {
    "Para.": (
        "para9",
        "results/FT/full/7b/probes_v9/newline2/para9/"
        "fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/"
        "e100/bs64_lr2e-05/overlap_1_4/11_20_22_51",
    ),
    "Para. + Textbooks": (
        "para9_expl_textbooks_cyclefull",
        "results/FT/full/7b/probes_v9/newline2/para9_expl_textbooks_cyclefull/"
        "fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/"
        "overlap_1_4/11_21_02_11",
    ),
    "Source": (
        "source_only",
        "results/FT/full/7b/probes_v9/newline2/source_only/"
        "fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/"
        "e100/bs64_lr2e-05/overlap_1_4/11_20_22_51",
    ),
}

# Try to import your plotting style
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
try:
    from utils.llm_plotting import set_plot_style
except ImportError:
    set_plot_style = None

# ---------------------------------------------------------------------
# METRICS LOADING
# ---------------------------------------------------------------------

def aggregate_across_domains(run_path, probe_type, domains):
    """
    Aggregates probe data across multiple domains from a specific run path.
    We only need inference probes for this script.
    """
    all_domain_dfs = []
    for domain in domains:
        if probe_type == "inference":
            probe_dir = f"{domain}_inference_probe"
            file_name = f"{domain}_inference_probe_metrics.csv"
        else:
            raise ValueError("This script is only set up for probe_type='inference'.")

        metrics_path = os.path.join(run_path, probe_dir, file_name)

        if os.path.exists(metrics_path) and os.path.getsize(metrics_path) > 0:
            df = pd.read_csv(metrics_path)
            if 'step' in df.columns and 'log_prob' in df.columns:
                df['step'] = pd.to_numeric(df['step'], errors='coerce')
                df['log_prob'] = pd.to_numeric(df['log_prob'], errors='coerce')
                df.dropna(subset=['step', 'log_prob'], inplace=True)
                if df.empty:
                    continue
                df['step'] = df['step'].astype(int)
            else:
                print(f"Warning: 'step' or 'log_prob' column not found in {metrics_path}. Skipping.")
                continue

            df['domain'] = domain
            all_domain_dfs.append(df)
        else:
            if not os.path.exists(metrics_path):
                print(f"Warning: File not found at {metrics_path}")
            else:
                print(f"Warning: File is empty at {metrics_path}")

    if not all_domain_dfs:
        return pd.DataFrame()

    combined_df = pd.concat(all_domain_dfs, ignore_index=True)
    return combined_df


# ---------------------------------------------------------------------
# IDF + LEXICAL OVERLAP
# ---------------------------------------------------------------------

TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)
STOPWORDS = {
    "the","and","of","to","a","in","for","on","with","is","are","was","were",
    "that","this","it","as","by","an","at","from"
}

def tokenize(text: str):
    tokens = TOKEN_RE.findall(str(text).lower())
    return [t for t in tokens if t not in STOPWORDS]

def build_idf_and_texts(project_root: str, domains=None):
    """
    Build IDF over all probe facts + explanations.
    Returns:
      idf: dict[token -> idf]
      probe_texts: dict[(domain, probe_index) -> set(tokens)]
      expl_texts: dict[domain -> set(tokens)]
    """
    if domains is None:
        domains = DOMAINS

    probe_root = os.path.join(project_root, "data/probes/inference")
    expl_root = os.path.join(project_root, "data/arxiv/explanations")

    df_counts = collections.Counter()
    probe_texts = {}
    expl_texts = {}
    docs = []

    for domain in domains:
        # Probes
        probe_path = os.path.join(probe_root, domain, "probes_v7.csv")
        if not os.path.exists(probe_path):
            print(f"[WARN] Probe file not found for domain {domain}: {probe_path}")
            continue

        probes_df = pd.read_csv(probe_path)
        if "fact" not in probes_df.columns:
            raise ValueError(f"'fact' column not found in {probe_path}")

        for probe_index, row in probes_df.iterrows():
            tokens = set(tokenize(row["fact"]))
            probe_texts[(domain, probe_index)] = tokens
            docs.append(tokens)

        # Explanations / textbook
        expl_path = os.path.join(expl_root, domain, "textbook.txt")
        if not os.path.exists(expl_path):
            print(f"[WARN] Explanation file not found for domain {domain}: {expl_path}")
            continue

        with open(expl_path, "r") as f:
            expl_text = f.read()
        expl_tokens = set(tokenize(expl_text))
        expl_texts[domain] = expl_tokens
        docs.append(expl_tokens)

    # IDF
    for doc_tokens in docs:
        for w in set(doc_tokens):
            df_counts[w] += 1

    N = len(docs) if docs else 1
    idf = {w: math.log(N / (1 + df)) for w, df in df_counts.items()}

    return idf, probe_texts, expl_texts

def compute_idf_overlap(idf, probe_texts, expl_texts):
    """
    Compute IDF-weighted lexical overlap for each (domain, probe_index).
    Returns DataFrame with columns:
      domain, probe_index, lex_sim
    """
    records = []

    for (domain, probe_index), probe_tokens in probe_texts.items():
        if domain not in expl_texts:
            continue

        expl_tokens = expl_texts[domain]

        numer = 0.0
        denom = 0.0
        for w in probe_tokens:
            w_idf = idf.get(w, 0.0)
            denom += w_idf
            if w in expl_tokens:
                numer += w_idf

        lex_sim = numer / denom if denom > 0 else 0.0

        records.append(
            {
                "domain": domain,
                "probe_index": probe_index,
                "lex_sim": lex_sim,
            }
        )

    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------
# Δ LOG PROB PER PROBE
# ---------------------------------------------------------------------

def compute_delta_log_prob_per_probe(inference_df: pd.DataFrame):
    """
    Compute delta log prob per (domain, probe_index, method):
    Δ = log_prob(last_step) − log_prob(first_step)
    Returns DataFrame with:
      domain, probe_index, method, delta_log_prob
    """
    required_cols = {"domain", "probe_index", "step", "log_prob", "method"}
    if not required_cols.issubset(inference_df.columns):
        missing = required_cols - set(inference_df.columns)
        raise ValueError(f"Missing columns in inference_df: {missing}")

    def _delta(group):
        idx_min = group["step"].idxmin()
        idx_max = group["step"].idxmax()
        g0 = group.loc[idx_min, "log_prob"]
        g_last = group.loc[idx_max, "log_prob"]
        return pd.Series({"delta_log_prob": g_last - g0})

    delta_df = (
        inference_df.groupby(["domain", "probe_index", "method"], as_index=False)
                    .apply(_delta)
                    .reset_index(drop=True)
    )
    return delta_df


# ---------------------------------------------------------------------
# PLOTTING: ALL METHODS TOGETHER
# ---------------------------------------------------------------------

def plot_delta_vs_similarity_multi(merged: pd.DataFrame, num_bins: int, title: str, output_path: str):
    """
    merged: DataFrame with columns: domain, probe_index, method, delta_log_prob, lex_sim
    Plots one curve per method: Δ log prob vs lexical overlap (binned).
    """
    df = merged.dropna(subset=["lex_sim", "delta_log_prob"]).copy()
    if df.empty:
        print("No data to plot after dropping NaNs.")
        return

    # Equal-width bins over [0, 1]
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    df["sim_bin"] = pd.cut(df["lex_sim"], bins=bins, include_lowest=True)

    grouped = (
        df.groupby(["sim_bin", "method"])["delta_log_prob"]
          .agg(["mean", "count"])
          .reset_index()
    )
    if grouped.empty:
        print("No bins with data to plot.")
        return

    # Use bin centres as x
    bin_centres = [(interval.left + interval.right) / 2 for interval in sorted(df["sim_bin"].cat.categories)]

    methods = sorted(df["method"].unique())

    if set_plot_style is not None:
        set_plot_style()

    plt.figure(figsize=(7, 4.5))

    color_cycle = plt.rcParams.get("axes.prop_cycle", None)
    if color_cycle is None:
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    else:
        colors = [c["color"] for c in color_cycle]

    for i, method in enumerate(methods):
        mdf = grouped[grouped["method"] == method].copy()
        # Ensure bins are in consistent order
        mdf = mdf.set_index("sim_bin").reindex(sorted(df["sim_bin"].cat.categories))
        y = mdf["mean"].values

        plt.plot(
            bin_centres,
            y,
            marker="o",
            label=method,
            linewidth=1.8,
        )

    plt.xlabel("IDF-weighted lexical overlap (probe fact vs textbook)")
    plt.ylabel("Δ log prob (last − first)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {output_path}")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot Δ log prob vs IDF-weighted lexical overlap for 7B runs (Source, Para, Textbooks)."
    )
    parser.add_argument(
        "--project_root",
        type=str,
        default=DEFAULT_PROJECT_ROOT,
        help="Path to project root (where 'data' and 'results' live)."
    )
    parser.add_argument(
        "--num_bins",
        type=int,
        default=10,
        help="Number of bins for lexical similarity."
    )
    args = parser.parse_args()

    project_root = os.path.abspath(args.project_root)

    # 1) Load inference metrics for each of the three runs
    all_inference = []

    for display_name, (method_key, rel_run_path) in RUNS_7B.items():
        run_path = os.path.join(project_root, rel_run_path)
        print(f"Loading inference metrics for '{display_name}' from: {run_path}")
        if not os.path.isdir(run_path):
            print(f"[WARN] Run path does not exist: {run_path}")
            continue

        df = aggregate_across_domains(run_path, "inference", DOMAINS)
        if df.empty:
            print(f"[WARN] No inference data for {display_name}")
            continue

        df["method"] = display_name
        all_inference.append(df)

    if not all_inference:
        print("No inference data loaded for any method. Exiting.")
        return

    inference_df = pd.concat(all_inference, ignore_index=True)

    # 2) Build IDF + lexical overlap
    idf, probe_texts, expl_texts = build_idf_and_texts(project_root, domains=DOMAINS)
    lex_sim_df = compute_idf_overlap(idf, probe_texts, expl_texts)

    # 3) Compute Δ log prob per probe per method
    delta_df = compute_delta_log_prob_per_probe(inference_df)

    # 4) Merge
    merged = delta_df.merge(lex_sim_df, on=["domain", "probe_index"], how="left")
    print(merged.head())

    # 5) Plot all three methods together
    plots_dir = os.path.join(project_root, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    output_path = os.path.join(plots_dir, "delta_vs_lexical_overlap_7b_source_para_textbooks.pdf")

    title = "7B – Δ log prob vs lexical overlap (Source, Para, Textbooks)"
    plot_delta_vs_similarity_multi(
        merged,
        num_bins=args.num_bins,
        title=title,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()