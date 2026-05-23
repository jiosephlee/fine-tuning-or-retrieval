"""
Plot prior-knowledge position comparison: front vs middle vs end.

Four panels (factual log-prob, factual MCQA, inference log-prob, inference MCQA),
each showing the three positions as separate lines.
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


RUN_BASE = (
    "results/FT/full/7b/"
    "probes_v13_para_v13_paraphrased_inf_v11_reviewed_mcqa_v14_prompt_formatted_question_5shot_inf_mcqa_v13+v12/"
    "newline2/para9/fill_dclm/domains_arxiv_all-legal_all-medical_all/e100/bs256_lr4e-05/overlap_1_16"
)

POSITIONS = [
    ("front",  "Prior Knowledge at Front",  "E13_prior_knowledge_front_local",  "#1f77b4"),
    ("middle", "Prior Knowledge at Middle", "E14_prior_knowledge_middle_local", "#2ca02c"),
    ("end",    "Prior Knowledge at End",    "E15_prior_knowledge_end_local",    "#d62728"),
]

PANELS = [
    ("knowledge", "log_prob",       "classic", "Factual Probes",      "Mean Log Probability"),
    ("knowledge", "mcqa_accuracy",  "mcqa",    "Factual MCQA",        "MCQA Accuracy"),
    ("inference", "log_prob",       "classic", "Compositional Probes", "Mean Log Probability"),
    ("inference", "mcqa_accuracy",  "mcqa",    "Compositional MCQA",  "MCQA Accuracy"),
]


def discover_domains(run_path: str) -> List[str]:
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
    parser.add_argument("--output_dir", default="plots")
    parser.add_argument("--filename", default="prior_knowledge_position_7B.pdf")
    args = parser.parse_args()

    setup_style("default")

    # Discover the union of domains across the three runs.
    run_paths = {
        key: str(REPO_ROOT / RUN_BASE / subdir)
        for key, _label, subdir, _color in POSITIONS
    }
    domains_union: List[str] = sorted({
        d for path in run_paths.values() for d in discover_domains(path)
    })
    if not domains_union:
        raise SystemExit("No domains discovered under any prior-knowledge run.")

    fig, axes = make_subplots(1, 4, figsize=(20, 4.2), sharey=False)
    axes = axes.flatten()

    for ax, (probe_type, metric, probe_family, title, ylabel) in zip(axes, PANELS):
        for key, label, _subdir, color in POSITIONS:
            df = load_metrics(
                run_paths[key],
                probe_type,
                domains_union,
                str(REPO_ROOT),
                metrics=(metric,),
                probe_family=probe_family,
            )
            if df is None or df.empty or metric not in df.columns:
                print(f"Warning: no data for {key} / {probe_type} / {metric}")
                continue
            ax.plot(df["step"], df[metric], color=color, label=label, lw=1.8)

        ax.set_title(title)
        ax.set_xlabel("Training Step")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    # Unify y across log-prob panels and across MCQA panels separately.
    unify_ylim([axes[0], axes[2]])
    unify_ylim([axes[1], axes[3]])

    add_legend(axes[0], loc="lower right", fontsize="small")

    fig.tight_layout()
    save_plot(args.filename, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
