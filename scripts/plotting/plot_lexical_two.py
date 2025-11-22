import os
import sys
import re
import math
import argparse
import collections

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

try:
    from rank_bm25 import BM25Plus
except ImportError:
    print("Error: rank_bm25 not installed. Please run 'pip install rank-bm25'")
    sys.exit(1)

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

DEFAULT_PROJECT_ROOT = "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval"
DOMAINS = ["1_58", "DPO", "GRPO", "BOFT", "OFT", "QLoRA"]
RUNS_7B = {
    "Source": (
        "source_only",
        "results/FT/full/7b/probes_v9/newline2/source_only/"
        "fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/"
        "e100/bs64_lr2e-05/overlap_1_4/11_20_22_51",
    ),
    "Paraphrase": (
        "para9",
        "results/FT/full/7b/probes_v9/newline2/para9/"
        "fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/"
        "e100/bs64_lr2e-05/overlap_1_4/11_20_22_51",
    ),
    "Textbook": (
        "para9_expl_textbooks_cyclefull",
        "results/FT/full/7b/probes_v9/newline2/para9_expl_textbooks_cyclefull/"
        "fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/"
        "overlap_1_4/11_21_02_11",
    ),
    "Blogs": (
        "blogs_run",
        "results/FT/full/7b/probes_v9/newline2/para9_expl_stackexchange_cyclefull/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_21_02_11"
    ),
    "StackExchange": (
        "stack_run",
        "results/FT/full/7b/probes_v9/newline2/para9_expl_blogs_cyclefull/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_21_02_11"
    ),
}

CORPUS_DEFINITIONS = {
    "source":   [("data/arxiv/cleaned", "{domain}.tex")],
    "para":     [("data/arxiv/paraphrased/{domain}", "0.tex")],
    "textbook": [("data/arxiv/explanations/{domain}", "textbook.txt")],
    "blogs":    [("data/arxiv/explanations/{domain}", "blogs.txt")],
    "stackexchange": [("data/arxiv/explanations/{domain}", "stackexchange.txt")],
}

METHOD_TO_CORPUS = {
    "Source": "source",
    "Paraphrase": "para",
    "Textbook": "textbook",
    "Blogs": "blogs",
    "StackExchange": "stackexchange",
}

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
try:
    from utils.llm_plotting import set_plot_style
except ImportError:
    set_plot_style = None

# ---------------------------------------------------------------------
# DATA LOADING & BM25 (BIGRAM VERSION)
# ---------------------------------------------------------------------

TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)
STOPWORDS = {
    "the","and","of","to","a","in","for","on","with","is","are","was","were",
    "that","this","it","as","by","an","at","from","be","has","have"
}

def tokenize_bigrams(text: str):
    """
    Tokenizes text into BIGRAMS.
    1. Extracts words (removes punctuation).
    2. Filters stopwords.
    3. Creates pairs: "mitochondria powerhouse"
    """
    # 1. Get cleaned unigrams
    tokens = TOKEN_RE.findall(str(text).lower())
    tokens = [t for t in tokens if t not in STOPWORDS]
    
    # 2. Create Bigrams
    if len(tokens) < 2:
        return [] # No bigrams possible for single words
    
    # Return as list of strings "word1_word2" for BM25 hashing
    return [f"{t1}_{t2}" for t1, t2 in zip(tokens, tokens[1:])]

def load_probe_data(project_root, domains):
    data = {"inference": {}, "knowledge": {}}
    
    path_inf = os.path.join(project_root, "data/probes/inference")
    for d in domains:
        p = os.path.join(path_inf, d, "probes_v7.csv")
        if os.path.exists(p):
            df = pd.read_csv(p)
            if "fact" in df.columns:
                for idx, row in df.iterrows():
                    # USE BIGRAM TOKENIZER
                    data["inference"][(d, idx)] = tokenize_bigrams(str(row["fact"]))

    path_fact = os.path.join(project_root, "data/probes/facts")
    for d in domains:
        p = os.path.join(path_fact, d, "probes_v9.csv")
        if os.path.exists(p):
            df = pd.read_csv(p)
            if "fact" in df.columns:
                for idx, row in df.iterrows():
                    # USE BIGRAM TOKENIZER
                    data["knowledge"][(d, idx)] = tokenize_bigrams(str(row["fact"]))
                    
    return data

def build_bm25_indices(project_root, domains, k1=10.0, b=0.75):
    indices = {}
    for corpus_name, file_configs in CORPUS_DEFINITIONS.items():
        print(f"Building Bigram BM25 index for corpus: {corpus_name.upper()}...")
        corpus_tokens = []
        domain_to_idx = {}
        for i, domain in enumerate(domains):
            domain_tokens = []
            for folder_tmpl, file_tmpl in file_configs:
                fpath = os.path.join(
                    project_root, 
                    folder_tmpl.format(domain=domain), 
                    file_tmpl.format(domain=domain)
                )
                if os.path.exists(fpath):
                    with open(fpath, "r") as f:
                        # USE BIGRAM TOKENIZER ON CORPUS TOO
                        domain_tokens.extend(tokenize_bigrams(f.read()))
            corpus_tokens.append(domain_tokens)
            domain_to_idx[domain] = i
            
        if not any(corpus_tokens):
            print(f"  [WARN] No data found for {corpus_name}")
            
        bm25 = BM25Plus(corpus_tokens, k1=k1, b=b)
        indices[corpus_name] = (bm25, domain_to_idx)
    return indices

def get_scores_for_probes(probe_tokens_map, bm25_data):
    bm25_obj, domain_map = bm25_data
    records = []
    for (domain, idx), tokens in probe_tokens_map.items():
        if domain not in domain_map or not tokens:
            score = 0.0
        else:
            doc_idx = domain_map[domain]
            try:
                score = bm25_obj.get_scores(tokens)[doc_idx]
            except IndexError:
                score = 0.0
        records.append({
            "domain": domain,
            "probe_index": idx,
            "bm25_score": score
        })
    return pd.DataFrame.from_records(records)

# ---------------------------------------------------------------------
# METRICS & DELTA
# ---------------------------------------------------------------------

def aggregate_metrics(run_path, probe_type, domains):
    if not run_path: return pd.DataFrame()
    dfs = []
    for domain in domains:
        folder = f"{domain}_{probe_type}_probe"
        filename = f"{domain}_{probe_type}_probe_metrics.csv"
        path = os.path.join(run_path, folder, filename)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            df = pd.read_csv(path)
            cols = ['step', 'log_prob']
            if all(c in df.columns for c in cols):
                df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
                df = df.dropna(subset=cols)
                if not df.empty:
                    df['domain'] = domain
                    dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def compute_delta_log_prob(df):
    def _delta(g):
        first = g.loc[g['step'].idxmin(), 'log_prob']
        last = g.loc[g['step'].idxmax(), 'log_prob']
        return last - first

    return (df.groupby(["domain", "probe_index", "method"], as_index=False)
            [['log_prob', 'step']]
            .apply(lambda x: pd.Series({'delta_log_prob': _delta(x)}))
            .reset_index(drop=True))

# ---------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------

def plot_scatter(ax, df, title, color):
    df = df.dropna(subset=["bm25_score", "delta_log_prob"])
    if df.empty:
        ax.set_title(f"{title}\n(no data)")
        ax.axis("off")
        return

    ax.scatter(
        df["bm25_score"],
        df["delta_log_prob"],
        alpha=0.15,
        s=10,
        color=color,
        edgecolors='none'
    )

    if len(df) > 1:
        x = df["bm25_score"].values
        y = df["delta_log_prob"].values
        a, b = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = a * x_line + b
        ax.plot(x_line, y_line, linewidth=1.5, color='red', zorder=10)
        
        x_txt = x.min() + (x.max() - x.min()) * 0.05
        y_txt = y.min() + (y.max() - y.min()) * 0.90
        ax.text(x_txt, y_txt, f"m={a:.3f}", color='red', fontsize=10, fontweight='bold')

    ax.set_title(title)
    ax.set_xlabel("Bigram BM25+ Score")
    ax.grid(True, alpha=0.2)

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", type=str, default=DEFAULT_PROJECT_ROOT)
    args = parser.parse_args()
    root = os.path.abspath(args.project_root)

    if set_plot_style: set_plot_style()

    print("Loading probe tokens (Bigrams)...")
    probe_tokens = load_probe_data(root, DOMAINS)
    
    print("Building Bigram BM25 indices...")
    # Indices now store BIGRAMS, not words
    bm25_indices = build_bm25_indices(root, DOMAINS, k1=10.0, b=0.75)
    
    display_order = ["Paraphrase", "Textbook", "Source", "Blogs", "StackExchange"]
    methods = [m for m in display_order if m in RUNS_7B]
    
    nrows, ncols = 2, len(methods)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 5 * nrows), sharey=True)
    if nrows == 1: axes = np.expand_dims(axes, 0)

    for col_idx, method in enumerate(methods):
        print(f"\nProcessing Method: {method}")
        target_corpus = METHOD_TO_CORPUS.get(method)
        
        if not target_corpus or target_corpus not in bm25_indices:
            print(f"Skipping {method} (corpus index missing)")
            continue

        bm25_data = bm25_indices[target_corpus]
        scores_inf = get_scores_for_probes(probe_tokens["inference"], bm25_data)
        scores_fact = get_scores_for_probes(probe_tokens["knowledge"], bm25_data)

        run_key, rel_path = RUNS_7B[method]
        if rel_path is None:
            print(f"[WARN] No run path defined for {method}. Skipping plot.")
            axes[0, col_idx].set_title(f"{method}\n(No Path Provided)")
            axes[0, col_idx].axis('off')
            axes[1, col_idx].axis('off')
            continue
            
        full_path = os.path.join(root, rel_path)

        # Factual
        metrics_fact = aggregate_metrics(full_path, "knowledge", DOMAINS)
        if not metrics_fact.empty:
            metrics_fact["method"] = method
            delta_fact = compute_delta_log_prob(metrics_fact)
            merged = delta_fact.merge(scores_fact, on=["domain", "probe_index"])
            
            ax = axes[0, col_idx]
            plot_scatter(ax, merged, f"{method}\nFactual", "#1f77b4")
            if col_idx == 0: ax.set_ylabel(r"$\Delta$ Log Prob")
        else:
            axes[0, col_idx].axis('off')

        # Inference
        metrics_inf = aggregate_metrics(full_path, "inference", DOMAINS)
        if not metrics_inf.empty:
            metrics_inf["method"] = method
            delta_inf = compute_delta_log_prob(metrics_inf)
            merged = delta_inf.merge(scores_inf, on=["domain", "probe_index"])
            
            ax = axes[1, col_idx]
            plot_scatter(ax, merged, f"{method}\nInference", "#ff7f0e")
            if col_idx == 0: ax.set_ylabel(r"$\Delta$ Log Prob")
        else:
            axes[1, col_idx].axis('off')

    plt.tight_layout()
    out = os.path.join(root, "plots", "delta_log_prob_vs_bigram_bm25.pdf")
    plt.savefig(out, bbox_inches="tight")
    print(f"\nSaved plot to {out}")

if __name__ == "__main__":
    main()