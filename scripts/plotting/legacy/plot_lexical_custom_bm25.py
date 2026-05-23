import os
import sys
import re
import argparse
import collections

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

DEFAULT_PROJECT_ROOT = "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval"
DOMAINS = ["1_58", "DPO", "GRPO", "BOFT", "OFT", "QLoRA"]

parser = argparse.ArgumentParser()
parser.add_argument("--baseline", choices=["Source", "Paraphrase"], default="Paraphrase", help="Baseline strategy")
parser.add_argument("--filter", action="store_true", default=True, help="Enable filtering (Target > Baseline)")
parser.add_argument("--no-filter", action="store_false", dest="filter", help="Disable filtering")
parser.add_argument("--project_root", default=DEFAULT_PROJECT_ROOT)

# NEW: Axis Scale Arguments
parser.add_argument("--x_scale", choices=["linear", "log"], default="linear", help="Scale for X-axis (linear or symlog base 2)")
parser.add_argument("--y_scale", choices=["linear", "log"], default="linear", help="Scale for Y-axis (linear or symlog base 2)")

args = parser.parse_args()

BASELINE_STRAT = args.baseline
FILTER_PROBES = args.filter
X_SCALE_TYPE = args.x_scale
Y_SCALE_TYPE = args.y_scale

# Plotting Hyperparameters (Used only if scale is 'log')
PLOT_LOG_BASE = 5     
PLOT_LINTHRESH = 1.0    
PLOT_LINSCALE = 0.5

# Define paths
SOURCE_PATH = "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_20_22_51"
PARA_PATH = "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_20_22_51"

RUNS_7B = {
    "Baseline": (
        "baseline_run",
        SOURCE_PATH if BASELINE_STRAT == "Source" else PARA_PATH
    ),
    "Textbook": (
        "para9_expl_textbooks_cyclefull",
        "results/FT/full/7b/probes_v9/newline2/para9_expl_textbooks_cyclefull/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_21_02_11",
    ),
    "Blogs": (
        "blogs_run",
        "results/FT/full/7b/probes_v9/newline2/para9_expl_stackexchange_cyclefull/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_21_02_11",
    ),
    "StackExchange": (
        "stack_run",
        "results/FT/full/7b/probes_v9/newline2/para9_expl_blogs_cyclefull/"
        "fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/"
        "overlap_1_4/11_21_02_11",
    ),
    "Corrupted Aux. Views": (
        "corrupted_textbook_run",
        #"/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/para9_expl_fruit_textbooks_v2+fruit_blogs_v2+fruit_stackexchange_v2_cyclefull_double/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs96_lr2e-05/overlap_1_4/11_24_02_58"
        "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/para9_expl_fruit_textbooks_v3+fruit_blogs_v3+fruit_stackexchange_v3_cyclefull/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_24_02_47/",
    ),
}

CORPUS_DEFINITIONS = {
    "source":   [("data/arxiv/cleaned", "{domain}.tex")],
    
    # Updated: Includes Source file + 0.tex through 9.tex
    "para":     [("data/arxiv/cleaned", "{domain}.tex")] + \
                [("data/arxiv/paraphrased/{domain}", f"{i}.tex") for i in range(10)],
    
    # Updated: Baseline logic now uses the expanded definition if strategy is Paraphrase
    "Baseline": [("data/arxiv/cleaned", "{domain}.tex")] if BASELINE_STRAT == "Source" else \
                ([("data/arxiv/paraphrased/{domain}", f"{i}.tex") for i in range(9)]),
    
    "textbook": [("data/arxiv/explanations/{domain}", "textbook.txt")],
    "blogs":    [("data/arxiv/explanations/{domain}", "blogs.txt")],
    "stackexchange": [("data/arxiv/explanations/{domain}", "stackexchange.txt")],
    
    # Updated: Includes textbooks, blogs, and stackexchange fruit versions
    "corrupted_aux_views": [
        ("data/arxiv/explanations/{domain}/fruit_textbooks_v3", "*.txt"),
        ("data/arxiv/explanations/{domain}/fruit_blogs_v3", "*.txt"),
        ("data/arxiv/explanations/{domain}/fruit_stackexchange_v3", "*.txt")
    ],
}

METHOD_TO_CORPUS = {
    "Baseline": "source" if BASELINE_STRAT == "Source" else "para",
    "Textbook": "textbook",
    "Blogs": "blogs",
    "StackExchange": "stackexchange",
    "Corrupted Aux. Views": "corrupted_aux_views",
}


# --- COLORS ---
STRATEGY_COLORS = {
    "Source": "#1f77b4",       # Blue
    "Paraphrase": "#ff7f0e",   # Orange
    "Blogs": "#9467bd",        # Purple
    "StackExchange": "#bcbd22",# Yellow (Olive-ish)
    "Textbook": "#2ca02c",     # Green
    "Corrupted Aux. Views": "#8c564b", # Brown
}

# Darker, fully opaque colors for the legend
STRATEGY_COLORS_LEGEND = {
    "Source": "#0b3d66",       # Dark Blue
    "Paraphrase": "#b35900",   # Dark Orange
    "Blogs": "#5c3d7a",        # Dark Purple
    "StackExchange": "#7f8016",# Dark Olive
    "Textbook": "#1a661a",     # Dark Green
    "Corrupted Aux. Views": "#5e3a32", # Dark Brown
}

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
from utils import probe_paths
try:
    from utils.llm_plotting import set_plot_style
except ImportError:
    set_plot_style = None

# ---------------------------------------------------------------------
# TOKENISATION (Unchanged)
# ---------------------------------------------------------------------

TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)

def load_stopwords():
    stopwords = set()
    try:
        from spacy.lang.en.stop_words import STOP_WORDS as SPACY_STOP_WORDS
        stopwords |= {w.lower() for w in SPACY_STOP_WORDS}
    except ImportError: pass
    try:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as SK_STOP_WORDS
        stopwords |= {w.lower() for w in SK_STOP_WORDS}
    except ImportError: pass
    try:
        from nltk.corpus import stopwords as nltk_stopwords
        stopwords |= {w.lower() for w in nltk_stopwords.words("english")}
    except Exception: pass

    common_words_path = "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/data/misc/10K_common_words.txt"
    if os.path.exists(common_words_path):
        with open(common_words_path, "r", encoding="utf-8") as f:
            stopwords |= {line.strip().lower() for line in f if line.strip()}
    
    print(f"Total unique stopwords: {len(stopwords)}")
    return stopwords

def tokenize_unigrams(text: str, stopwords: set = None):
    if stopwords is None: stopwords = set()
    tokens = TOKEN_RE.findall(str(text).lower())
    return [t for t in tokens if t not in stopwords]

def tokenize_bigrams(text: str, stopwords: set = None):
    if stopwords is None: stopwords = set()
    tokens = TOKEN_RE.findall(str(text).lower())
    tokens = [t for t in tokens if t not in stopwords]
    if len(tokens) < 2: return []
    return [f"{t1}_{t2}" for t1, t2 in zip(tokens, tokens[1:])]

def load_probe_data(project_root, domains, tokenize_fn):
    data = {"inference": {}, "knowledge": {}}
    for d in domains:
        p = str(probe_paths.resolve_probe_path("inference", d, "v7"))
        if os.path.exists(p):
            df = pd.read_csv(p)
            if "fact" in df.columns:
                for idx, row in df.iterrows():
                    data["inference"][(d, idx)] = tokenize_fn(str(row["fact"]))

    for d in domains:
        p = str(probe_paths.resolve_probe_path("facts", d, "v9"))
        if os.path.exists(p):
            df = pd.read_csv(p)
            if "fact" in df.columns:
                for idx, row in df.iterrows():
                    data["knowledge"][(d, idx)] = tokenize_fn(str(row["fact"]))
    return data

# ---------------------------------------------------------------------
# SCORING (Unchanged)
# ---------------------------------------------------------------------

def compute_source_tf(project_root, domains, tokenize_fn):
    print("Computing Source term frequencies...")
    source_stats = {}
    file_configs = CORPUS_DEFINITIONS["source"]
    for domain in domains:
        tokens = []
        for folder_tmpl, file_tmpl in file_configs:
            fpath = os.path.join(project_root, folder_tmpl.format(domain=domain), file_tmpl.format(domain=domain))
            if os.path.exists(fpath):
                with open(fpath, "r") as f: tokens.extend(tokenize_fn(f.read()))
        source_stats[domain] = (collections.Counter(tokens), len(tokens))
    return source_stats

def rarity_weight(tf_source):
    return 5.0 / (1.0 + float(tf_source))

def score_target_corpus_bm25_plus_rarity(project_root, domains, corpus_name, source_stats_map, probe_data, tokenize_fn, k1=10.0, b=0.75, delta=1.0):
    print(f"Scoring {corpus_name.upper()}...")
    domain_stats = {}
    file_configs = CORPUS_DEFINITIONS.get(corpus_name)
    if not file_configs: return pd.DataFrame()

    total_len = 0
    num_docs = 0
    for domain in domains:
        tokens = []
        for folder_tmpl, file_tmpl in file_configs:
            import glob
            dir_path = os.path.join(project_root, folder_tmpl.format(domain=domain))
            if "*" in file_tmpl:
                for fpath in glob.glob(os.path.join(dir_path, file_tmpl)):
                    if os.path.exists(fpath):
                        with open(fpath, "r") as f: tokens.extend(tokenize_fn(f.read()))
            else:
                fpath = os.path.join(dir_path, file_tmpl.format(domain=domain))
                if os.path.exists(fpath):
                    with open(fpath, "r") as f: tokens.extend(tokenize_fn(f.read()))
        doc_len = len(tokens)
        domain_stats[domain] = (collections.Counter(tokens), doc_len)
        if doc_len > 0:
            total_len += doc_len
            num_docs += 1
    avgdl = total_len / num_docs if num_docs > 0 else 1.0

    records = []
    for (domain, idx), p_tokens in probe_data.items():
        if domain not in domain_stats or not p_tokens:
            score = 0.0
        else:
            tf_target, doc_len = domain_stats[domain]
            tf_source, _ = source_stats_map.get(domain, (collections.Counter(), 0))
            score = 0.0
            for token in p_tokens:
                tgt_freq = tf_target.get(token, 0)
                src_freq = tf_source.get(token, 0)
                w = rarity_weight(src_freq)
                if tgt_freq > 0:
                    denom = tgt_freq + k1 * (1.0 - b + b * (doc_len / avgdl))
                    if denom > 0:
                        bm25_plus_term = (tgt_freq * (k1 + 1.0)) / denom + delta
                        score += w * bm25_plus_term
        records.append({"domain": domain, "probe_index": idx, "bm25_score": score})
    return pd.DataFrame.from_records(records)

# ---------------------------------------------------------------------
# METRICS HELPERS
# ---------------------------------------------------------------------

def aggregate_metrics(run_path, probe_type, domains):
    if not run_path: return pd.DataFrame()
    dfs = []
    for domain in domains:
        base_dir = os.path.join(run_path, f"{domain}_{probe_type}_probe")
        candidates = [f"{domain}_{probe_type}_probe_metrics.csv", f"test_{domain}_{probe_type}_probe_metrics.csv"]
        path = None
        if os.path.exists(base_dir):
            for cand in candidates:
                p = os.path.join(base_dir, cand)
                if os.path.exists(p) and os.path.getsize(p) > 0:
                    path = p
                    break
        if path:
            df = pd.read_csv(path)
            if "step" in df.columns: df["step"] = pd.to_numeric(df["step"], errors="coerce")
            if "log_prob" in df.columns: df["log_prob"] = pd.to_numeric(df["log_prob"], errors="coerce")
            df = df.dropna(subset=["step"])
            if not df.empty:
                df["domain"] = domain
                dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def get_final_metric(df, metric_col):
    if df.empty or metric_col not in df.columns: return pd.DataFrame(columns=["domain", "probe_index", "final_metric"])
    df = df.dropna(subset=[metric_col, "step"])
    return df.sort_values("step").groupby(["domain", "probe_index"]).tail(1)[["domain", "probe_index", metric_col]].rename(columns={metric_col: "final_metric"})

def internal_delta_per_probe(df, metric_col):
    if df.empty or metric_col not in df.columns: return pd.DataFrame(columns=["domain", "probe_index", "delta_metric"])
    df = df.dropna(subset=[metric_col, "step"]).sort_values("step")
    grouped = df.groupby(["domain", "probe_index"])
    delta = grouped.tail(1).set_index(["domain", "probe_index"])[metric_col] - grouped.head(1).set_index(["domain", "probe_index"])[metric_col]
    return delta.reset_index().rename(columns={metric_col: "delta_metric"})

# ---------------------------------------------------------------------
# PLOTTING FUNCTIONS
# ---------------------------------------------------------------------

def plot_scatter_overlay(ax, dfs_with_color, baseline_name, show_ylabel=False, x_scale="log", y_scale="log"):
    # Apply Y-Axis Scale
    if y_scale == "log":
        ax.set_yscale('symlog', base=PLOT_LOG_BASE, linthresh=PLOT_LINTHRESH, linscale=PLOT_LINSCALE, subs=[1.0])
    else:
        ax.set_yscale('linear')
    
    # Apply X-Axis Scale
    if x_scale == "log":
        ax.set_xscale('symlog', base=PLOT_LOG_BASE, linthresh=PLOT_LINTHRESH, linscale=PLOT_LINSCALE, subs=[1.0])
    else:
        ax.set_xscale('linear')
    
    ax.axhline(0.0, linestyle="--", color="gray", linewidth=0.8)
    
    legend_handles = []
    
    for df, color, label, alpha in dfs_with_color:
        df = df.dropna(subset=["bm25_score", "delta_metric"])
        if df.empty: continue
        
        display_label = baseline_name if label == "Baseline" else label
        
        # Determine Legend Color (Darker)
        strat_key = BASELINE_STRAT if label == "Baseline" else label
        legend_color = STRATEGY_COLORS_LEGEND.get(strat_key, color)

        # Plot Scatter
        ax.scatter(df["bm25_score"], df["delta_metric"], alpha=alpha, s=10, color=color, edgecolors="none")
        
        mean_overlap = df["bm25_score"].mean()
        ax.axvline(mean_overlap, linestyle="--", color=color, linewidth=1.5, alpha=0.7)
        mean_delta = df["delta_metric"].mean()
        ax.axhline(mean_delta, linestyle=":", color=color, linewidth=1.5, alpha=0.7)

        # Legend Handle with Darker Color
        handle = mlines.Line2D([], [], color=legend_color, marker='o', linestyle='None', label=display_label)
        legend_handles.append(handle)

    if show_ylabel:
        ax.set_ylabel(r"$\Delta$ LogProb" + "\n(Final Step - Initial Step)")
        
    ax.set_xlabel("Overlap Score of Corpus")
    ax.grid(True, alpha=0.2)
    
    if legend_handles:
        ax.legend(handles=legend_handles, fontsize='medium', loc='lower right')

def plot_delta_vs_delta(ax, df, color, label, baseline_name, show_ylabel=False, x_scale="log", y_scale="log"):
    # Apply Y-Axis Scale
    if y_scale == "log":
        ax.set_yscale('symlog', base=PLOT_LOG_BASE, linthresh=PLOT_LINTHRESH, linscale=PLOT_LINSCALE, subs=[1.0])
    else:
        ax.set_yscale('linear')

    # Apply X-Axis Scale
    if x_scale == "log":
        ax.set_xscale('symlog', base=PLOT_LOG_BASE, linthresh=PLOT_LINTHRESH, linscale=PLOT_LINSCALE, subs=[1.0])
    else:
        ax.set_xscale('linear')

    ax.axhline(0.0, linestyle="--", color="gray", linewidth=0.8)
    ax.axvline(0.0, linestyle="--", color="gray", linewidth=0.8)
    
    df = df.dropna(subset=["delta_overlap", "delta_perf"])
    if not df.empty:
        ax.scatter(df["delta_overlap"], df["delta_perf"], alpha=0.4, s=10, color=color, edgecolors="none")
        
        # Trend line
        if len(df) > 1:
            x = df["delta_overlap"].values
            y = df["delta_perf"].values
            try:
                a, b = np.polyfit(x, y, 1)
                x_line = np.linspace(x.min(), x.max(), 100)
                y_line = a * x_line + b
                ax.plot(x_line, y_line, linewidth=1.0, linestyle=":", color="red", zorder=10)
            except: pass

    if show_ylabel:
        ax.set_ylabel(rf"$\Delta$ LogProb" + f"\n(Target Final Step - {baseline_name} Final Step)")
        
    ax.set_xlabel(rf"$\Delta$ Overlap Score (Target - {baseline_name})")
    ax.grid(True, alpha=0.2)
    
    # Custom Legend Logic
    strat_key = label
    legend_color = STRATEGY_COLORS_LEGEND.get(strat_key, color)
    handle = mlines.Line2D([], [], color=legend_color, marker='o', linestyle='None', label=label)
    
    ax.legend(handles=[handle], fontsize='medium', loc='lower right')

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    root = os.path.abspath(args.project_root)
    if set_plot_style: set_plot_style()
    
    print("Loading stopwords and tokens...")
    STOPWORDS = load_stopwords()
    def tokenize_uni(text): return tokenize_unigrams(text, STOPWORDS)
    def tokenize_bi(text): return tokenize_bigrams(text, STOPWORDS)

    probe_tokens_uni = load_probe_data(root, DOMAINS, tokenize_uni)
    probe_tokens_bi = load_probe_data(root, DOMAINS, tokenize_bi)

    ngram_configs = [("unigram", tokenize_uni, probe_tokens_uni), ("bigram", tokenize_bi, probe_tokens_bi)]
    target_domains = ["Blogs", "StackExchange", "Textbook", "Corrupted Aux. Views"]    
    # Pre-calc metrics
    strategy_data = {}
    strategies_needed = ["Baseline"] + target_domains
    print("Pre-calculating metrics...")
    for strat in strategies_needed:
        if strat in RUNS_7B:
            _, rel_path = RUNS_7B[strat]
            full_path = os.path.join(root, rel_path)
            metrics_fact = aggregate_metrics(full_path, "knowledge", DOMAINS)
            metrics_inf = aggregate_metrics(full_path, "inference", DOMAINS)
            strategy_data[strat] = (
                internal_delta_per_probe(metrics_fact, "log_prob"),
                internal_delta_per_probe(metrics_inf, "log_prob"),
                get_final_metric(metrics_fact, "log_prob"),
                get_final_metric(metrics_inf, "log_prob")
            )

    # UPDATED: Use sharey=False to control axes independently manually later
    fig1, axes1 = plt.subplots(4, len(target_domains), figsize=(4 * len(target_domains), 5 * 4), sharey=False, sharex=True)
    fig2, axes2 = plt.subplots(4, len(target_domains), figsize=(4 * len(target_domains), 5 * 4), sharey=False, sharex=True)

    row_labels = {} 

    for nrow_block, (ngram_name, tokenize_fn, probe_tokens) in enumerate(ngram_configs):
        print(f"\n[N-gram: {ngram_name}]")
        source_stats_map = compute_source_tf(root, DOMAINS, tokenize_fn)
        
        # Baseline Scores
        scores_source_fact = score_target_corpus_bm25_plus_rarity(root, DOMAINS, "Baseline", source_stats_map, probe_tokens["knowledge"], tokenize_fn)
        scores_source_inf = score_target_corpus_bm25_plus_rarity(root, DOMAINS, "Baseline", source_stats_map, probe_tokens["inference"], tokenize_fn)
        
        row_fact = 0 if nrow_block == 0 else 1
        row_inf = 2 if nrow_block == 0 else 3
        
        row_labels[row_fact] = f"Factual Probes ({ngram_name})"
        # UPDATED: Rename Inference to Compositional
        row_labels[row_inf] = f"Compositional Probes ({ngram_name})"

        for col_idx, target_domain in enumerate(target_domains):
            tgt_corpus = METHOD_TO_CORPUS.get(target_domain)
            scores_target_fact = score_target_corpus_bm25_plus_rarity(root, DOMAINS, tgt_corpus, source_stats_map, probe_tokens["knowledge"], tokenize_fn)
            scores_target_inf = score_target_corpus_bm25_plus_rarity(root, DOMAINS, tgt_corpus, source_stats_map, probe_tokens["inference"], tokenize_fn)

            # --- Process FACTUAL ---
            if "Baseline" in strategy_data and target_domain in strategy_data:
                s_fin, t_fin = strategy_data["Baseline"][2], strategy_data[target_domain][2]
                merged = s_fin.merge(t_fin, on=["domain", "probe_index"], suffixes=("_src", "_tgt"))
                valid = merged[merged["final_metric_tgt"] > merged["final_metric_src"]] if FILTER_PROBES else merged
                
                m_score = scores_target_fact.merge(scores_source_fact, on=["domain", "probe_index"], suffixes=("_tgt", "_src"))
                m_score["delta_overlap"] = m_score["bm25_score_tgt"] - m_score["bm25_score_src"]
                
                m_perf = merged.copy()
                m_perf["delta_perf"] = m_perf["final_metric_tgt"] - m_perf["final_metric_src"]
                
                fig2_data = m_perf.merge(m_score, on=["domain", "probe_index"], how="inner")
                if FILTER_PROBES: fig2_data = fig2_data[fig2_data["final_metric_tgt"] > fig2_data["final_metric_src"]]
                
                plot_delta_vs_delta(axes2[row_fact, col_idx], fig2_data, STRATEGY_COLORS[target_domain], target_domain, BASELINE_STRAT, show_ylabel=(col_idx==0), x_scale=X_SCALE_TYPE, y_scale=Y_SCALE_TYPE)

            dfs_fact = []
            valid_keys = valid[["domain", "probe_index"]] if 'valid' in locals() else pd.DataFrame()
            for strat in ["Baseline", target_domain]:
                if strat in strategy_data and not strategy_data[strat][0].empty:
                    d_df = strategy_data[strat][0].merge(valid_keys, on=["domain", "probe_index"], how="inner")
                    scores = scores_source_fact if strat == "Baseline" else scores_target_fact
                    merged = d_df.merge(scores, on=["domain", "probe_index"], how="inner")
                    col = STRATEGY_COLORS[BASELINE_STRAT] if strat == "Baseline" else STRATEGY_COLORS[strat]
                    dfs_fact.append((merged, col, strat, 0.4 if strat == target_domain else 0.175))
            
            plot_scatter_overlay(axes1[row_fact, col_idx], dfs_fact, BASELINE_STRAT, show_ylabel=(col_idx==0), x_scale=X_SCALE_TYPE, y_scale=Y_SCALE_TYPE)

            # --- Process INFERENCE ---
            if "Baseline" in strategy_data and target_domain in strategy_data:
                s_fin, t_fin = strategy_data["Baseline"][3], strategy_data[target_domain][3]
                merged = s_fin.merge(t_fin, on=["domain", "probe_index"], suffixes=("_src", "_tgt"))
                valid_inf = merged[merged["final_metric_tgt"] > merged["final_metric_src"]] if FILTER_PROBES else merged
                
                m_score = scores_target_inf.merge(scores_source_inf, on=["domain", "probe_index"], suffixes=("_tgt", "_src"))
                m_score["delta_overlap"] = m_score["bm25_score_tgt"] - m_score["bm25_score_src"]
                
                m_perf = merged.copy()
                m_perf["delta_perf"] = m_perf["final_metric_tgt"] - m_perf["final_metric_src"]
                
                fig2_data = m_perf.merge(m_score, on=["domain", "probe_index"], how="inner")
                if FILTER_PROBES: fig2_data = fig2_data[fig2_data["final_metric_tgt"] > fig2_data["final_metric_src"]]
                
                plot_delta_vs_delta(axes2[row_inf, col_idx], fig2_data, STRATEGY_COLORS[target_domain], target_domain, BASELINE_STRAT, show_ylabel=(col_idx==0), x_scale=X_SCALE_TYPE, y_scale=Y_SCALE_TYPE)

            dfs_inf = []
            valid_keys_inf = valid_inf[["domain", "probe_index"]] if 'valid_inf' in locals() else pd.DataFrame()
            for strat in ["Baseline", target_domain]:
                if strat in strategy_data and not strategy_data[strat][1].empty:
                    d_df = strategy_data[strat][1].merge(valid_keys_inf, on=["domain", "probe_index"], how="inner")
                    scores = scores_source_inf if strat == "Baseline" else scores_target_inf
                    merged = d_df.merge(scores, on=["domain", "probe_index"], how="inner")
                    col = STRATEGY_COLORS[BASELINE_STRAT] if strat == "Baseline" else STRATEGY_COLORS[strat]
                    dfs_inf.append((merged, col, strat, 0.4 if strat == target_domain else 0.15))

            plot_scatter_overlay(axes1[row_inf, col_idx], dfs_inf, BASELINE_STRAT, show_ylabel=(col_idx==0), x_scale=X_SCALE_TYPE, y_scale=Y_SCALE_TYPE)

    for r in range(4):
        title = row_labels.get(r, "")
        axes1[r, 0].text(-0.35, 0.5, title, transform=axes1[r, 0].transAxes, va='center', ha='right', rotation=90, fontsize=12, fontweight='bold')
        axes2[r, 0].text(-0.35, 0.5, title, transform=axes2[r, 0].transAxes, va='center', ha='right', rotation=90, fontsize=12, fontweight='bold')

    # UPDATED: Post-processing to standardize Y-axis per group (Factual vs Compositional)
    for fig_obj, ax_arr in [(fig1, axes1), (fig2, axes2)]:
        # Group 1: Factual (Rows 0, 1)
        y_min_fact, y_max_fact = float('inf'), float('-inf')
        # Scan for limits
        for r in [0, 1]:
            for ax in ax_arr[r]:
                ymin, ymax = ax.get_ylim()
                if ymin < y_min_fact: y_min_fact = ymin
                if ymax > y_max_fact: y_max_fact = ymax
        # Apply limits
        for r in [0, 1]:
            for ax in ax_arr[r]:
                ax.set_ylim(y_min_fact, y_max_fact)
        
        # Group 2: Compositional (Rows 2, 3)
        y_min_comp, y_max_comp = float('inf'), float('-inf')
        # Scan for limits
        for r in [2, 3]:
            for ax in ax_arr[r]:
                ymin, ymax = ax.get_ylim()
                if ymin < y_min_comp: y_min_comp = ymin
                if ymax > y_max_comp: y_max_comp = ymax
        # Apply limits
        for r in [2, 3]:
            for ax in ax_arr[r]:
                ax.set_ylim(y_min_comp, y_max_comp)

    fig1.tight_layout()
    fig1.subplots_adjust(left=0.1) 
    out1 = os.path.join(root, "plots", "internal_delta_logprob_vs_target_overlap_filtered_4cols.pdf")
    os.makedirs(os.path.dirname(out1), exist_ok=True)
    fig1.savefig(out1, bbox_inches="tight")
    print(f"Saved Figure 1 to {out1}")
    
    fig2.tight_layout()
    fig2.subplots_adjust(left=0.1)
    out2 = os.path.join(root, "plots", "delta_logprob_vs_delta_overlap_4cols.pdf")
    os.makedirs(os.path.dirname(out2), exist_ok=True)
    fig2.savefig(out2, bbox_inches="tight")
    print(f"Saved Figure 2 to {out2}")

if __name__ == "__main__":
    main()
