"""
Plot para9 baseline (E2) vs the prior-knowledge match run (E16) that injects
prior knowledge into the explanations track, broken out by domain group
(arxiv vs legal).

Layout: 2 rows (arxiv, legal) x 4 columns (factual log-prob, factual MCQA,
compositional log-prob, compositional MCQA).
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Sequence

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.plotting.plot_utils import (  # noqa: E402
    add_legend,
    find_latest_run,
    load_metrics,
    make_subplots,
    save_plot,
    setup_style,
    unify_ylim,
)


E17_SOURCE_PATH = (
    "results/FT/full/7b/source_only_docmatch_expl/fill_dclm/domains_arxiv_all-legal_all/"
    "e100/bs256_lr4e-05/overlap_1_16/E17_source_arxiv_legal_local/"
    "eval_bundles/inf_mcqa_v14"
)

E18_PARA_PATH = (
    "results/FT/full/7b/para9_docmatch_expl/fill_dclm/domains_arxiv_all-legal_all/"
    "e100/bs256_lr4e-05/overlap_1_16/E18_paraphrase_arxiv_legal_local/"
    "eval_bundles/inf_mcqa_v14"
)

# Latest re-evaluated runs store probes under eval_bundles/inf_mcqa_v14 (full
# trajectory, single v14 inference-MCQA folder). Point at the bundle directly so
# find_latest_run/load_metrics resolve the per-domain probe folders.
E16_PRIOR_PATH = (
    "results/FT/full/7b/para9_docmatch_expl_insertprior_knowledge/fill_dclm/"
    "domains_arxiv_all-legal_all/e100/bs256_lr4e-05/overlap_1_16/"
    "E16_prior_knowledge_arxiv_legal_match_local/eval_bundles/inf_mcqa_v14"
)

E19_CITED_PATH = (
    "results/FT/full/7b/para9_docmatch_expl_insertcited_works/fill_dclm/"
    "domains_arxiv_all-legal_all/e100/bs256_lr4e-05/overlap_1_16/"
    "E19_cited_textbooks_arxiv_legal_match_local/eval_bundles/inf_mcqa_v14"
)

RUNS = [
    ("source",       "Source Only",              E17_SOURCE_PATH, "#1f77b4"),
    ("para",         "Paraphrased",              E18_PARA_PATH,   "#ff7f0e"),
    ("prior_match",  "Para. + Prior Knowledge",  E16_PRIOR_PATH,  "#d62728"),
    ("cited_match",  "Para. + Cited Works",      E19_CITED_PATH,  "#2ca02c"),
]

PANELS = [
    ("knowledge", "log_prob",      "classic", "Factual Probes",       "Mean Log Probability"),
    ("knowledge", "mcqa_accuracy", "mcqa",    "Factual MCQA",         "MCQA Accuracy"),
    ("inference", "log_prob",      "classic", "Compositional Probes", "Mean Log Probability"),
    ("inference", "mcqa_accuracy", "mcqa",    "Compositional MCQA",   "MCQA Accuracy"),
]


def _domains_from_dir(rel_dir: str, ext: str) -> List[str]:
    path = REPO_ROOT / rel_dir
    if not path.is_dir():
        return []
    return sorted(
        os.path.splitext(name)[0]
        for name in os.listdir(path)
        if name.endswith(ext) and (path / name).is_file()
    )


def domains_in_run(run_path: str) -> List[str]:
    resolved = find_latest_run(run_path)
    if not resolved:
        return []
    suffix = "_knowledge_probe"
    return sorted(
        name[: -len(suffix)]
        for name in os.listdir(resolved)
        if name.endswith(suffix) and os.path.isdir(os.path.join(resolved, name))
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="plots/prior_knowledge")
    parser.add_argument("--filename", default="prior_knowledge_match_by_group_7B.pdf")
    parser.add_argument(
        "--averaged_filename",
        default="prior_knowledge_match_averaged_7B.pdf",
        help="Filename for the second figure averaging across all domains.",
    )
    args = parser.parse_args()

    setup_style("default")

    arxiv_all = set(_domains_from_dir("data/arxiv/cleaned", ".tex"))
    legal_all = set(_domains_from_dir("data/legal/cleaned", ".txt"))

    # Intersect domains across all runs so every line uses the same set.
    run_domain_sets = [set(domains_in_run(str(REPO_ROOT / p))) for _, _, p, _ in RUNS]
    run_domain_sets = [s for s in run_domain_sets if s]
    if not run_domain_sets:
        raise SystemExit("No probe domains discovered under any run.")
    shared_domains = set.intersection(*run_domain_sets)

    domain_groups = [
        ("arxiv", "arXiv domains", sorted(arxiv_all & shared_domains)),
        ("legal", "Legal domains", sorted(legal_all & shared_domains)),
    ]
    for key, _label, doms in domain_groups:
        print(f"{key}: {len(doms)} domains -> {doms}")

    fig, axes = make_subplots(2, 4, figsize=(20, 8), sharey=False)

    for row, (_group_key, group_label, group_domains) in enumerate(domain_groups):
        for col, (probe_type, metric, probe_family, title, ylabel) in enumerate(PANELS):
            ax = axes[row, col]
            for _key, label, run_path, color in RUNS:
                if not group_domains:
                    continue
                df = load_metrics(
                    str(REPO_ROOT / run_path),
                    probe_type,
                    group_domains,
                    str(REPO_ROOT),
                    metrics=(metric,),
                    probe_family=probe_family,
                )
                if df is None or df.empty or metric not in df.columns:
                    print(f"Warning: no data for {label} / {_group_key} / {probe_type}/{metric}")
                    continue
                ax.plot(df["step"], df[metric], color=color, label=label, lw=1.8)

            if row == 0:
                ax.set_title(title)
            if row == 1:
                ax.set_xlabel("Training Step")
            if col == 0:
                ax.set_ylabel(f"{group_label}\n{ylabel}")
            else:
                ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)

    # Unify y across rows for matching columns.
    for col in range(4):
        unify_ylim([axes[0, col], axes[1, col]])

    add_legend(axes[0, 0], loc="lower right", fontsize="small")

    fig.tight_layout()
    save_plot(args.filename, output_dir=args.output_dir)

    # ---- Second figure: averaged across all shared domains ----
    all_shared = sorted(shared_domains)
    print(f"averaged: {len(all_shared)} domains")
    fig2, axes2 = make_subplots(1, 4, figsize=(20, 4.2), sharey=False)
    axes2 = axes2.flatten()
    for ax, (probe_type, metric, probe_family, title, ylabel) in zip(axes2, PANELS):
        for _key, label, run_path, color in RUNS:
            df = load_metrics(
                str(REPO_ROOT / run_path),
                probe_type,
                all_shared,
                str(REPO_ROOT),
                metrics=(metric,),
                probe_family=probe_family,
            )
            if df is None or df.empty or metric not in df.columns:
                print(f"Warning: no data for {label} / averaged / {probe_type}/{metric}")
                continue
            ax.plot(df["step"], df[metric], color=color, label=label, lw=1.8)
        ax.set_title(title)
        ax.set_xlabel("Training Step")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    add_legend(axes2[0], loc="lower right", fontsize="small")
    fig2.tight_layout()
    save_plot(args.averaged_filename, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
