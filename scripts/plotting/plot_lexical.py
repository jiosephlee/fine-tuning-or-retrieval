import os
import sys
import re
import math
import argparse
import collections

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

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

    probe_type: "inference" or "knowledge"
    """
    all_domain_dfs = []
    for domain in domains:
        if probe_type == "inference":
            probe_dir = f"{domain}_inference_probe"
            file_name = f"{domain}_inference_probe_metrics.csv"
        elif probe_type == "knowledge":
            probe_dir = f"{domain}_knowledge_probe"
            file_name = f"{domain}_knowledge_probe_metrics.csv"
        else:
            raise ValueError("probe_type must be 'inference' or 'knowledge'.")

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

def build_idf_and_texts(project_root: str, probe_folder: str, probes_filename: str, domains=None):
    """
    Build IDF over all probe facts + explanations for a given probe set.

    probe_folder: "inference" or "facts"
    probes_filename: e.g. "probes_v7.csv" (inference) or "probes_v9.csv" (facts)

    Returns:
      idf: dict[token -> idf]
      probe_texts: dict[(domain, probe_index) -> set(tokens)]
      expl_texts: dict[domain -> set(tokens)]
    """
    if domains is None:
        domains = DOMAINS

    probe_root = os.path.join(project_root, "data/probes", probe_folder)
    expl_root = os.path.join(project_root, "data/arxiv/explanations")

    df_counts = collections.Counter()
    probe_texts = {}
    expl_texts = {}
    docs = []

    for domain in domains:
        # Probes
        probe_path = os.path.join(probe_root, domain, probes_filename)
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

    sim_lex = sum_{w in T_p ∩ T_e} idf(w) / sum_{w in T_p} idf(w)

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

def compute_delta_log_prob_per_probe(df: pd.DataFrame):
    """
    Compute delta log prob per (domain, probe_index, method):
    Δ = log_prob(last_step) − log_prob(first_step)

    Returns DataFrame with:
      domain, probe_index, method, delta_log_prob
    """
    required_cols = {"domain", "probe_index", "step", "log_prob", "method"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Missing columns in df: {missing}")

    def _delta(group):
        idx_min = group["step"].idxmin()
        idx_max = group["step"].idxmax()
        g0 = group.loc[idx_min, "log_prob"]
        g_last = group.loc[idx_max, "log_prob"]
        return pd.Series({"delta_log_prob": g_last - g0})

    delta_df = (
        df.groupby(["domain", "probe_index", "method"], as_index=False)
          .apply(_delta)
          .reset_index(drop=True)
    )
    return delta_df

# ---------------------------------------------------------------------
# PLOTTING HELPERS
# ---------------------------------------------------------------------

def plot_delta_vs_similarity_multi_on_ax(ax, merged: pd.DataFrame, num_bins: int, title: str):
    """
    Plot Δ log prob vs lexical similarity for multiple methods on a given Axes.

    merged: DataFrame with columns: domain, probe_index, method, delta_log_prob, lex_sim
    """
    df = merged.dropna(subset=["lex_sim", "delta_log_prob"]).copy()
    if df.empty:
        print(f"[WARN] No data to plot for: {title}")
        return

    bins = np.linspace(0.0, 1.0, num_bins + 1)
    df["sim_bin"] = pd.cut(df["lex_sim"], bins=bins, include_lowest=True)

    grouped = (
        df.groupby(["sim_bin", "method"])["delta_log_prob"]
          .agg(["mean", "count"])
          .reset_index()
    )
    if grouped.empty:
        print(f"[WARN] No bins with data to plot for: {title}")
        return

    categories = sorted(df["sim_bin"].cat.categories)
    bin_centres = [(interval.left + interval.right) / 2 for interval in categories]

    methods = sorted(df["method"].unique())

    for method in methods:
        mdf = grouped[grouped["method"] == method].copy()
        # Ensure bins are aligned
        mdf = mdf.set_index("sim_bin").reindex(categories)
        y = mdf["mean"].values
        ax.plot(
            bin_centres,
            y,
            marker="o",
            linewidth=1.8,
            label=method,
        )

    ax.set_xlabel("IDF-weighted lexical overlap")
    ax.set_ylabel("Δ log prob (last − first)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot Δ log prob vs IDF-weighted lexical overlap for 7B runs (Source, Para, Textbooks), "
                    "for both factual (knowledge) and inference probes, side by side."
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

    # 1) Load metrics for each run: factual (knowledge) and inference
    all_inference = []
    all_factual = []

    for display_name, (_method_key, rel_run_path) in RUNS_7B.items():
        run_path = os.path.join(project_root, rel_run_path)
        print(f"Loading metrics for '{display_name}' from: {run_path}")
        if not os.path.isdir(run_path):
            print(f"[WARN] Run path does not exist: {run_path}")
            continue

        # Inference probes
        inf_df = aggregate_across_domains(run_path, "inference", DOMAINS)
        if not inf_df.empty:
            inf_df["method"] = display_name
            all_inference.append(inf_df)
        else:
            print(f"[WARN] No inference data for {display_name}")

        # Factual / knowledge probes
        fact_df = aggregate_across_domains(run_path, "knowledge", DOMAINS)
        if not fact_df.empty:
            fact_df["method"] = display_name
            all_factual.append(fact_df)
        else:
            print(f"[WARN] No factual (knowledge) data for {display_name}")

    if not all_inference and not all_factual:
        print("No data loaded for either inference or factual probes. Exiting.")
        return

    inference_df = pd.concat(all_inference, ignore_index=True) if all_inference else pd.DataFrame()
    factual_df = pd.concat(all_factual, ignore_index=True) if all_factual else pd.DataFrame()

    # 2) Build IDF + lexical overlap for inference probes (probes_v7)
    idf_inf, probe_texts_inf, expl_texts_inf = build_idf_and_texts(
        project_root,
        probe_folder="inference",
        probes_filename="probes_v7.csv",
        domains=DOMAINS,
    )
    lex_sim_inf = compute_idf_overlap(idf_inf, probe_texts_inf, expl_texts_inf)

    # 3) Build IDF + lexical overlap for factual probes (probes_v9)
    idf_fact, probe_texts_fact, expl_texts_fact = build_idf_and_texts(
        project_root,
        probe_folder="facts",
        probes_filename="probes_v9.csv",
        domains=DOMAINS,
    )
    lex_sim_fact = compute_idf_overlap(idf_fact, probe_texts_fact, expl_texts_fact)

    # 4) Compute Δ log prob per probe per method
    if not inference_df.empty:
        delta_inf = compute_delta_log_prob_per_probe(inference_df)
        merged_inf = delta_inf.merge(lex_sim_inf, on=["domain", "probe_index"], how="left")
    else:
        merged_inf = pd.DataFrame()

    if not factual_df.empty:
        delta_fact = compute_delta_log_prob_per_probe(factual_df)
        merged_fact = delta_fact.merge(lex_sim_fact, on=["domain", "probe_index"], how="left")
    else:
        merged_fact = pd.DataFrame()

    print("Merged factual head:")
    print(merged_fact.head())
    print("Merged inference head:")
    print(merged_inf.head())

    # 5) Plot side by side
    if set_plot_style is not None:
        set_plot_style()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    if not merged_fact.empty:
        plot_delta_vs_similarity_multi_on_ax(
            axes[0],
            merged_fact,
            num_bins=args.num_bins,
            title="Factual (Knowledge) Probes",
        )
    else:
        axes[0].set_title("Factual (Knowledge) Probes – no data")
        axes[0].axis("off")

    if not merged_inf.empty:
        plot_delta_vs_similarity_multi_on_ax(
            axes[1],
            merged_inf,
            num_bins=args.num_bins,
            title="Inference Probes",
        )
    else:
        axes[1].set_title("Inference Probes – no data")
        axes[1].axis("off")

    # Only one legend (right subplot)
    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()

    plots_dir = os.path.join(project_root, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    output_path = os.path.join(plots_dir, "delta_vs_lexical_overlap_7b_factual_vs_inference.pdf")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved side-by-side plot to {output_path}")


if __name__ == "__main__":
    main()