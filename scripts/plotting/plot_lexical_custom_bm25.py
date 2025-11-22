import os
import sys
import re
import argparse
import collections

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

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
        "results/FT/full/7b/probes_v9/newline2/para9_expl_stackexchange_cyclefull/"
        "fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/"
        "overlap_1_4/11_21_02_11",
    ),
    "StackExchange": (
        "stack_run",
        "results/FT/full/7b/probes_v9/newline2/para9_expl_blogs_cyclefull/"
        "fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/"
        "overlap_1_4/11_21_02_11",
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
# TOKENISATION
# ---------------------------------------------------------------------

TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)
STOPWORDS = {
    "the","and","of","to","a","in","for","on","with","is","are","was","were",
    "that","this","it","as","by","an","at","from","be","has","have"
}

def tokenize_unigrams(text: str):
    tokens = TOKEN_RE.findall(str(text).lower())
    return [t for t in tokens if t not in STOPWORDS]

def tokenize_bigrams(text: str):
    tokens = TOKEN_RE.findall(str(text).lower())
    tokens = [t for t in tokens if t not in STOPWORDS]
    if len(tokens) < 2:
        return []
    return [f"{t1}_{t2}" for t1, t2 in zip(tokens, tokens[1:])]

def load_probe_data(project_root, domains, tokenize_fn):
    """
    Returns:
      data: {
        "inference": {(domain, idx): [tokens...]},
        "knowledge": {(domain, idx): [tokens...]}
      }
    """
    data = {"inference": {}, "knowledge": {}}

    # Inference probes
    path_inf = os.path.join(project_root, "data/probes/inference")
    for d in domains:
        p = os.path.join(path_inf, d, "probes_v7.csv")
        if os.path.exists(p):
            df = pd.read_csv(p)
            if "fact" in df.columns:
                for idx, row in df.iterrows():
                    data["inference"][(d, idx)] = tokenize_fn(str(row["fact"]))

    # Factual probes
    path_fact = os.path.join(project_root, "data/probes/facts")
    for d in domains:
        p = os.path.join(path_fact, d, "probes_v9.csv")
        if os.path.exists(p):
            df = pd.read_csv(p)
            if "fact" in df.columns:
                for idx, row in df.iterrows():
                    data["knowledge"][(d, idx)] = tokenize_fn(str(row["fact"]))
    return data

def load_inference_types(project_root, domains):
    """
    For inference probes, load inference_type from probes_v7.csv.
    Keys are (domain, probe_index) where probe_index is taken from the
    'probe_index' column if present; else we fall back to row index.
    """
    types = {}
    path_inf = os.path.join(project_root, "data/probes/inference")
    for d in domains:
        p = os.path.join(path_inf, d, "probes_v7.csv")
        if os.path.exists(p):
            df = pd.read_csv(p)
            # normalise probe_index if present
            if "probe_index" in df.columns:
                df["probe_index"] = pd.to_numeric(df["probe_index"], errors="coerce")
            else:
                df["probe_index"] = np.arange(len(df), dtype=float)

            if "inference_type" in df.columns:
                for _, row in df.iterrows():
                    if pd.isna(row["probe_index"]):
                        continue
                    key = (d, int(row["probe_index"]))
                    types[key] = str(row["inference_type"])
            else:
                # Default to "Other" if inference_type missing
                for _, row in df.iterrows():
                    if pd.isna(row["probe_index"]):
                        continue
                    key = (d, int(row["probe_index"]))
                    types[key] = "Other"
    return types

# ---------------------------------------------------------------------
# SOURCE TF + RARITY + BM25+
# ---------------------------------------------------------------------

def compute_source_tf(project_root, domains, tokenize_fn):
    """
    Build term-frequency counters for the Source corpus per domain.
    Returns: dict[domain] -> Counter(token -> tf_in_source)
    """
    print("Computing Source term frequencies...")
    source_tf = {}
    file_configs = CORPUS_DEFINITIONS["source"]

    for domain in domains:
        tokens = []
        for folder_tmpl, file_tmpl in file_configs:
            fpath = os.path.join(
                project_root,
                folder_tmpl.format(domain=domain),
                file_tmpl.format(domain=domain),
            )
            if os.path.exists(fpath):
                with open(fpath, "r") as f:
                    tokens.extend(tokenize_fn(f.read()))
        source_tf[domain] = collections.Counter(tokens)
    return source_tf

def rarity_weight(tf_source):
    """
    Rarity weight function: w(x) = 5 / (1 + x)
    where x is the term frequency in the Source corpus.
    """
    return 5.0 / (1.0 + float(tf_source))

def score_target_corpus_bm25_plus_rarity(
    project_root,
    domains,
    corpus_name,
    source_tf_map,
    probe_data,
    tokenize_fn,
    k1=10.0,
    b=0.75,
    delta=1.0,
):
    """
    Scores probes in a target corpus using BM25+ with Source-rarity weights
    instead of IDF, with the important tweak that we give *no* credit when
    tf_target == 0 (i.e. gate delta on presence).

    For each probe (domain, idx):
        score = sum_t [ w_source(t) * ( (tf_tgt*(k1+1)) /
                                       (tf_tgt + k1*(1-b + b*|D|/avgdl)) + delta ) ]
    where:
        w_source(t) = 5 / (1 + TF_source(t)),
        and the term contributes only if tf_tgt > 0.
    """
    print(f"Scoring {corpus_name.upper()} using Source-rarity-weighted BM25+ (delta={delta})...")

    domain_stats = {}  # domain -> (Counter(tf_target), doc_len)
    file_configs = CORPUS_DEFINITIONS.get(corpus_name)
    if not file_configs:
        return pd.DataFrame()

    total_len = 0
    num_docs = 0

    # Build TF and lengths for target corpus
    for domain in domains:
        tokens = []
        for folder_tmpl, file_tmpl in file_configs:
            fpath = os.path.join(
                project_root,
                folder_tmpl.format(domain=domain),
                file_tmpl.format(domain=domain),
            )
            if os.path.exists(fpath):
                with open(fpath, "r") as f:
                    tokens.extend(tokenize_fn(f.read()))
        doc_len = len(tokens)
        domain_stats[domain] = (collections.Counter(tokens), doc_len)
        if doc_len > 0:
            total_len += doc_len
            num_docs += 1

    avgdl = total_len / num_docs if num_docs > 0 else 1.0

    # Score probes
    records = []
    for (domain, idx), p_tokens in probe_data.items():
        if domain not in domain_stats or not p_tokens:
            score = 0.0
        else:
            tf_target, doc_len = domain_stats[domain]
            tf_source = source_tf_map.get(domain, collections.Counter())
            score = 0.0

            for token in p_tokens:
                tgt_freq = tf_target.get(token, 0)
                if tgt_freq <= 0:
                    # No lexical presence -> no contribution at all
                    continue

                src_freq = tf_source.get(token, 0)
                denom = tgt_freq + k1 * (1.0 - b + b * (doc_len / avgdl))
                if denom <= 0:
                    continue

                bm25_plus_term = (tgt_freq * (k1 + 1.0)) / denom + delta
                w = rarity_weight(src_freq)
                term_score = w * bm25_plus_term
                score += term_score

        records.append({
            "domain": domain,
            "probe_index": idx,
            "bm25_score": score,
        })

    return pd.DataFrame.from_records(records)

# ---------------------------------------------------------------------
# METRICS & Δ VS SOURCE
# ---------------------------------------------------------------------

def aggregate_metrics(run_path, probe_type, domains):
    if not run_path:
        return pd.DataFrame()
    dfs = []
    for domain in domains:
        folder = f"{domain}_{probe_type}_probe"
        filename = f"{domain}_{probe_type}_probe_metrics.csv"
        path = os.path.join(run_path, folder, filename)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            df = pd.read_csv(path)
            # We care at least about step, log_prob, probe_index, and possibly hit_accuracy_at_100
            if "probe_index" not in df.columns:
                continue
            if "step" in df.columns:
                df["step"] = pd.to_numeric(df["step"], errors="coerce")
            if "log_prob" in df.columns:
                df["log_prob"] = pd.to_numeric(df["log_prob"], errors="coerce")
            if "hit_accuracy_at_100" in df.columns:
                df["hit_accuracy_at_100"] = pd.to_numeric(df["hit_accuracy_at_100"], errors="coerce")
            df = df.dropna(subset=["step"])
            if not df.empty:
                df["domain"] = domain
                dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def final_metric_per_probe(df, metric_col):
    """
    Given metrics for a single run (one method), return the final-step metric
    per (domain, probe_index).
    """
    if df.empty or metric_col not in df.columns:
        return pd.DataFrame(columns=["domain", "probe_index", metric_col])
    df = df.dropna(subset=[metric_col])
    if df.empty:
        return pd.DataFrame(columns=["domain", "probe_index", metric_col])
    df_sorted = df.sort_values("step")
    last = df_sorted.groupby(["domain", "probe_index"], as_index=False).tail(1)
    return last[["domain", "probe_index", metric_col]]

def plot_scatter(ax, df, title, color, y_label=None, show_ylabel=False):
    df = df.dropna(subset=["bm25_score", "delta_metric"])
    ax.axhline(0.0, linestyle="--", color="gray", linewidth=0.8)  # baseline at 0
    if df.empty:
        ax.set_title(f"{title}\n(no data)")
        ax.axis("off")
        return

    ax.scatter(
        df["bm25_score"],
        df["delta_metric"],
        alpha=0.15,
        s=10,
        color=color,
        edgecolors="none",
    )

    if len(df) > 1:
        x = df["bm25_score"].values
        y = df["delta_metric"].values
        a, b = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = a * x_line + b
        ax.plot(x_line, y_line, linewidth=1.5, color="red", zorder=10)

        x_txt = x.min() + (x.max() - x.min()) * 0.05
        y_txt = y.min() + (y.max() - y.min()) * 0.90
        ax.text(
            x_txt,
            y_txt,
            f"m={a:.3f}",
            color="red",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_title(title)
    if show_ylabel and y_label is not None:
        ax.set_ylabel(y_label)
    ax.set_xlabel("Source-rarity-weighted BM25+ score")
    ax.grid(True, alpha=0.2)

# ---------------------------------------------------------------------
# MAIN FIGURE BUILDING
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", type=str, default=DEFAULT_PROJECT_ROOT)
    args = parser.parse_args()
    root = os.path.abspath(args.project_root)

    if set_plot_style:
        set_plot_style()

    print("Loading probe tokens (unigrams & bigrams)...")
    probe_tokens_uni = load_probe_data(root, DOMAINS, tokenize_unigrams)
    probe_tokens_bi = load_probe_data(root, DOMAINS, tokenize_bigrams)

    print("Loading inference types...")
    inference_types = load_inference_types(root, DOMAINS)

    # Methods to display (drop Source column, but keep Source for baseline)
    display_order = ["Paraphrase", "Textbook", "Blogs", "StackExchange"]
    methods = [m for m in display_order if m in RUNS_7B]

    # Source run path (baseline)
    if "Source" not in RUNS_7B:
        raise ValueError("Source run must be present in RUNS_7B for Δ vs Source.")
    _, src_rel_path = RUNS_7B["Source"]
    src_full_path = os.path.join(root, src_rel_path)

    # Convenience for ngram configs
    ngram_configs = [
        ("unigram", tokenize_unigrams, probe_tokens_uni),
        ("bigram", tokenize_bigrams, probe_tokens_bi),
    ]

    # ==============================================================
    # FIGURE 1: Δ FINAL LOG-PROB VS SOURCE (4x4, fact/inf × uni/bi)
    # ==============================================================

    print("\n=== Figure 1: Δ final log-prob vs Source ===")
    fig1, axes1 = plt.subplots(
        4, len(methods), figsize=(4 * len(methods), 5 * 4), sharey=True
    )

    # Source baseline metrics for log_prob
    source_metrics_fact = aggregate_metrics(src_full_path, "knowledge", DOMAINS)
    source_metrics_inf = aggregate_metrics(src_full_path, "inference", DOMAINS)
    source_final_fact_log = final_metric_per_probe(source_metrics_fact, "log_prob") \
        .rename(columns={"log_prob": "metric_source"})
    source_final_inf_log = final_metric_per_probe(source_metrics_inf, "log_prob") \
        .rename(columns={"log_prob": "metric_source"})

    for nrow_block, (ngram_name, tokenize_fn, probe_tokens) in enumerate(ngram_configs):
        print(f"\n[log_prob] N-gram: {ngram_name}")
        source_tf_map = compute_source_tf(root, DOMAINS, tokenize_fn)

        for col_idx, method in enumerate(methods):
            print(f"  Method: {method}")
            target_corpus = METHOD_TO_CORPUS.get(method)
            run_key, rel_path = RUNS_7B[method]
            full_path = os.path.join(root, rel_path)

            # Lexical scores
            scores_fact = score_target_corpus_bm25_plus_rarity(
                root, DOMAINS, target_corpus, source_tf_map,
                probe_tokens["knowledge"], tokenize_fn,
                k1=10.0, b=0.75, delta=1.0,
            )
            scores_inf = score_target_corpus_bm25_plus_rarity(
                root, DOMAINS, target_corpus, source_tf_map,
                probe_tokens["inference"], tokenize_fn,
                k1=10.0, b=0.75, delta=1.0,
            )

            # Target metrics
            metrics_fact_tgt = aggregate_metrics(full_path, "knowledge", DOMAINS)
            metrics_inf_tgt = aggregate_metrics(full_path, "inference", DOMAINS)

            target_final_fact = final_metric_per_probe(metrics_fact_tgt, "log_prob") \
                .rename(columns={"log_prob": "metric_target"})
            target_final_inf = final_metric_per_probe(metrics_inf_tgt, "log_prob") \
                .rename(columns={"log_prob": "metric_target"})

            # Merge with Source baseline and lexical scores
            # --- Factual row ---
            if not target_final_fact.empty and not source_final_fact_log.empty:
                merged_log = target_final_fact.merge(
                    source_final_fact_log,
                    on=["domain", "probe_index"],
                    how="inner",
                )
                merged_log["delta_metric"] = (
                    merged_log["metric_target"] - merged_log["metric_source"]
                )
                merged_fact = merged_log.merge(
                    scores_fact,
                    on=["domain", "probe_index"],
                    how="left",
                )

                row_fact = 0 + 2 * nrow_block  # 0 for unigram, 2 for bigram? no: 0/1 for fact, 2/3 for inf
                # For our 4 rows: 0=fact-uni, 1=fact-bi, 2=inf-uni, 3=inf-bi
                row_fact = 0 if nrow_block == 0 else 1

                ax = axes1[row_fact, col_idx]
                plot_scatter(
                    ax,
                    merged_fact,
                    f"{method}\nFactual ({ngram_name})",
                    "#1f77b4",
                    y_label=r"$\Delta$ final log-prob (target − Source)",
                    show_ylabel=(col_idx == 0 and row_fact == 0),
                )
            else:
                row_fact = 0 if nrow_block == 0 else 1
                axes1[row_fact, col_idx].axis("off")

            # --- Inference row ---
            if not target_final_inf.empty and not source_final_inf_log.empty:
                merged_log = target_final_inf.merge(
                    source_final_inf_log,
                    on=["domain", "probe_index"],
                    how="inner",
                )
                merged_log["delta_metric"] = (
                    merged_log["metric_target"] - merged_log["metric_source"]
                )
                merged_inf = merged_log.merge(
                    scores_inf,
                    on=["domain", "probe_index"],
                    how="left",
                )

                row_inf = 2 if nrow_block == 0 else 3
                ax = axes1[row_inf, col_idx]
                plot_scatter(
                    ax,
                    merged_inf,
                    f"{method}\nInference ({ngram_name})",
                    "#ff7f0e",
                    y_label=r"$\Delta$ final log-prob (target − Source)",
                    show_ylabel=(col_idx == 0 and row_inf == 2),
                )
            else:
                row_inf = 2 if nrow_block == 0 else 3
                axes1[row_inf, col_idx].axis("off")

    plt.tight_layout()
    out1 = os.path.join(root, "plots", "delta_logprob_vs_source_rarity_bm25_plus_uni_bigram.pdf")
    os.makedirs(os.path.dirname(out1), exist_ok=True)
    plt.savefig(out1, bbox_inches="tight")
    print(f"\nSaved Figure 1 to {out1}")

    # ==============================================================
    # FIGURE 2: Δ HIT@100 VS SOURCE (same 4x4 structure)
    # ==============================================================

    print("\n=== Figure 2: Δ hit@100 vs Source ===")
    fig2, axes2 = plt.subplots(
        4, len(methods), figsize=(4 * len(methods), 5 * 4), sharey=True
    )

    # Source baseline metrics for hit@100
    source_metrics_fact = aggregate_metrics(src_full_path, "knowledge", DOMAINS)
    source_metrics_inf = aggregate_metrics(src_full_path, "inference", DOMAINS)
    source_final_fact_hit = final_metric_per_probe(source_metrics_fact, "hit_accuracy_at_100") \
        .rename(columns={"hit_accuracy_at_100": "metric_source"})
    source_final_inf_hit = final_metric_per_probe(source_metrics_inf, "hit_accuracy_at_100") \
        .rename(columns={"hit_accuracy_at_100": "metric_source"})

    for nrow_block, (ngram_name, tokenize_fn, probe_tokens) in enumerate(ngram_configs):
        print(f"\n[hit@100] N-gram: {ngram_name}")
        source_tf_map = compute_source_tf(root, DOMAINS, tokenize_fn)

        for col_idx, method in enumerate(methods):
            print(f"  Method: {method}")
            target_corpus = METHOD_TO_CORPUS.get(method)
            _, rel_path = RUNS_7B[method]
            full_path = os.path.join(root, rel_path)

            # Lexical scores
            scores_fact = score_target_corpus_bm25_plus_rarity(
                root, DOMAINS, target_corpus, source_tf_map,
                probe_tokens["knowledge"], tokenize_fn,
                k1=10.0, b=0.75, delta=1.0,
            )
            scores_inf = score_target_corpus_bm25_plus_rarity(
                root, DOMAINS, target_corpus, source_tf_map,
                probe_tokens["inference"], tokenize_fn,
                k1=10.0, b=0.75, delta=1.0,
            )

            # Target metrics
            metrics_fact_tgt = aggregate_metrics(full_path, "knowledge", DOMAINS)
            metrics_inf_tgt = aggregate_metrics(full_path, "inference", DOMAINS)

            target_final_fact = final_metric_per_probe(metrics_fact_tgt, "hit_accuracy_at_100") \
                .rename(columns={"hit_accuracy_at_100": "metric_target"})
            target_final_inf = final_metric_per_probe(metrics_inf_tgt, "hit_accuracy_at_100") \
                .rename(columns={"hit_accuracy_at_100": "metric_target"})

            # --- Factual row ---
            if not target_final_fact.empty and not source_final_fact_hit.empty:
                merged_log = target_final_fact.merge(
                    source_final_fact_hit,
                    on=["domain", "probe_index"],
                    how="inner",
                )
                merged_log["delta_metric"] = (
                    merged_log["metric_target"] - merged_log["metric_source"]
                )
                merged_fact = merged_log.merge(
                    scores_fact,
                    on=["domain", "probe_index"],
                    how="left",
                )

                row_fact = 0 if nrow_block == 0 else 1
                ax = axes2[row_fact, col_idx]
                plot_scatter(
                    ax,
                    merged_fact,
                    f"{method}\nFactual ({ngram_name})",
                    "#1f77b4",
                    y_label=r"$\Delta$ hit@100 (target − Source)",
                    show_ylabel=(col_idx == 0 and row_fact == 0),
                )
            else:
                row_fact = 0 if nrow_block == 0 else 1
                axes2[row_fact, col_idx].axis("off")

            # --- Inference row ---
            if not target_final_inf.empty and not source_final_inf_hit.empty:
                merged_log = target_final_inf.merge(
                    source_final_inf_hit,
                    on=["domain", "probe_index"],
                    how="inner",
                )
                merged_log["delta_metric"] = (
                    merged_log["metric_target"] - merged_log["metric_source"]
                )
                merged_inf = merged_log.merge(
                    scores_inf,
                    on=["domain", "probe_index"],
                    how="left",
                )

                row_inf = 2 if nrow_block == 0 else 3
                ax = axes2[row_inf, col_idx]
                plot_scatter(
                    ax,
                    merged_inf,
                    f"{method}\nInference ({ngram_name})",
                    "#ff7f0e",
                    y_label=r"$\Delta$ hit@100 (target − Source)",
                    show_ylabel=(col_idx == 0 and row_inf == 2),
                )
            else:
                row_inf = 2 if nrow_block == 0 else 3
                axes2[row_inf, col_idx].axis("off")

    plt.tight_layout()
    out2 = os.path.join(root, "plots", "delta_hit10_vs_source_rarity_bm25_plus_uni_bigram.pdf")
    os.makedirs(os.path.dirname(out2), exist_ok=True)
    plt.savefig(out2, bbox_inches="tight")
    print(f"\nSaved Figure 2 to {out2}")

    # ==============================================================
    # FIGURE 3: INFERENCE-ONLY, SPLIT BY INFERENCE TYPE
    # Rows: Conceptual/uni, Conceptual/bi, Other/uni, Other/bi
    # Metric: Δ final log-prob vs Source
    # ==============================================================

    print("\n=== Figure 3: Inference-only, Conceptual vs Other (log-prob) ===")
    fig3, axes3 = plt.subplots(
        4, len(methods), figsize=(4 * len(methods), 5 * 4), sharey=True
    )

    # Source baseline (inference, log_prob)
    source_metrics_inf = aggregate_metrics(src_full_path, "inference", DOMAINS)
    source_final_inf_log = final_metric_per_probe(source_metrics_inf, "log_prob") \
        .rename(columns={"log_prob": "metric_source"})

    for nrow_block, (ngram_name, tokenize_fn, probe_tokens) in enumerate(ngram_configs):
        print(f"\n[Inference only] N-gram: {ngram_name}")
        source_tf_map = compute_source_tf(root, DOMAINS, tokenize_fn)

        for col_idx, method in enumerate(methods):
            print(f"  Method: {method}")
            target_corpus = METHOD_TO_CORPUS.get(method)
            _, rel_path = RUNS_7B[method]
            full_path = os.path.join(root, rel_path)

            # Lexical scores (inference only)
            scores_inf = score_target_corpus_bm25_plus_rarity(
                root, DOMAINS, target_corpus, source_tf_map,
                probe_tokens["inference"], tokenize_fn,
                k1=10.0, b=0.75, delta=1.0,
            )

            metrics_inf_tgt = aggregate_metrics(full_path, "inference", DOMAINS)
            target_final_inf = final_metric_per_probe(metrics_inf_tgt, "log_prob") \
                .rename(columns={"log_prob": "metric_target"})

            if target_final_inf.empty or source_final_inf_log.empty:
                # Turn off both conceptual & other rows for this method/ngram
                row_concept = 0 if nrow_block == 0 else 1
                row_other = 2 if nrow_block == 0 else 3
                axes3[row_concept, col_idx].axis("off")
                axes3[row_other, col_idx].axis("off")
                continue

            merged_log = target_final_inf.merge(
                source_final_inf_log,
                on=["domain", "probe_index"],
                how="inner",
            )
            merged_log["delta_metric"] = (
                merged_log["metric_target"] - merged_log["metric_source"]
            )

            merged = merged_log.merge(
                scores_inf,
                on=["domain", "probe_index"],
                how="left",
            )

            # Attach inference_type
            def _infer_type(row):
                return inference_types.get(
                    (row["domain"], row["probe_index"]),
                    "Other",
                )

            merged["inference_type"] = merged.apply(_infer_type, axis=1)

            conceptual = merged[merged["inference_type"] == "Conceptual Synthesis"]
            other = merged[merged["inference_type"] != "Conceptual Synthesis"]

            # Row indices:
            #  0: Conceptual/unigram
            #  1: Conceptual/bigram
            #  2: Other/unigram
            #  3: Other/bigram
            row_concept = 0 if nrow_block == 0 else 1
            row_other = 2 if nrow_block == 0 else 3

            # Conceptual row
            ax_c = axes3[row_concept, col_idx]
            plot_scatter(
                ax_c,
                conceptual,
                f"{method}\nConceptual ({ngram_name})",
                "#2ca02c",
                y_label=r"$\Delta$ final log-prob (target − Source)",
                show_ylabel=(col_idx == 0 and row_concept == 0),
            )

            # Other row
            ax_o = axes3[row_other, col_idx]
            plot_scatter(
                ax_o,
                other,
                f"{method}\nOther inf. ({ngram_name})",
                "#d62728",
                y_label=r"$\Delta$ final log-prob (target − Source)",
                show_ylabel=(col_idx == 0 and row_other == 2),
            )

    plt.tight_layout()
    out3 = os.path.join(root, "plots", "delta_logprob_inference_concept_vs_other_rarity_bm25_plus_uni_bigram.pdf")
    os.makedirs(os.path.dirname(out3), exist_ok=True)
    plt.savefig(out3, bbox_inches="tight")
    print(f"\nSaved Figure 3 to {out3}")

if __name__ == "__main__":
    main()