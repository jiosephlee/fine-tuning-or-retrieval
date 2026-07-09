"""2x2 panel comparison of the three 32B training conditions.

Conditions (x-axis within each panel):
  source_only -> Source
  para9       -> Paraphrased (9x)
  with_explanations -> Paraphrased (9x) + Aux explanations

Panels (one per subplot):
  Factual MCQA      (knowledge / mcqa_accuracy)
  Factual Probes    (knowledge / log_prob)
  Inference Probes  (inference / log_prob)
  Inference MCQA    (inference / mcqa_accuracy)

All conditions are read from their ``reeval_v3`` final-model re-eval bundle
(the "reeval 3" metrics). Mirrors the data-loading conventions used by
finetuning_knowledge_v9.py / plot_probe_scaling_by_model_v14.py.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.plotting.plot_inference_mcqa_scaling import (  # noqa: E402
    METHODS,
    apply_plot_style,
    final_value,
    save_figure,
)
from scripts.plotting.plot_utils import (  # noqa: E402
    COLORS,
    find_latest_run,
    load_metrics,
)

CONDITION_LABELS = {
    "source_only": "Source",
    "para9": "Para. 9",
    "with_explanations": "Para. 9\n+ Aux",
}
CONDITION_COLORS = {
    "source_only": COLORS["method"]["source"],
    "para9": COLORS["paraphrase_level"]["para9"],
    "with_explanations": COLORS["method"]["aux_views"],
}

# Panel definition: (probe_type, metric, probe_family, title, ylabel).
PANELS = (
    ("knowledge", "mcqa_accuracy", "mcqa", "Factual MCQA", "Final Accuracy"),
    ("knowledge", "log_prob", "classic", "Factual Probes", "Final Log Prob"),
    ("inference", "log_prob", "classic", "Inference Probes", "Final Log Prob"),
    ("inference", "mcqa_accuracy", "mcqa", "Inference MCQA", "Final Accuracy"),
)

# 32B reeval_v3 bundles for each condition.
_32B_OVERLAP = "fill_dclm/domains_arxiv_all-legal_all-medical_all/e100/bs256_lr4e-05/overlap_1_16"
DEFAULT_RUNS: Dict[str, str] = {
    "source_only": f"results/FT/full/32b/source_only_docmatch_expl/{_32B_OVERLAP}/E10_source_32b_all_domains_fa2_packing_chunked_nll_1gpu_device_map_auto_36h_nodelta/eval_bundles/reeval_v3",
    "para9": f"results/FT/full/32b/para9_docmatch_expl/{_32B_OVERLAP}/E11_paraphrase_32b_all_domains_chunked_nll_1gpu_device_map_auto_36h_nodelta/eval_bundles/reeval_v3",
    "with_explanations": f"results/FT/full/32b/para9_expl_textbooks+stackexchange+blogs_cyclefull/{_32B_OVERLAP}/E12_granular_explanations_32b_all_domains_1gpu_device_map_auto_36h_nodelta/eval_bundles/reeval_v3",
}


def _abs_path(path: str) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return str(candidate)


def discover_domains(run_paths: Sequence[str]) -> list:
    suffix = "_inference_probe"
    domain_sets = []
    for run_path in run_paths:
        resolved = find_latest_run(run_path)
        if not resolved:
            continue
        names = {
            name[: -len(suffix)]
            for name in __import__("os").listdir(resolved)
            if name.endswith(suffix)
        }
        if names:
            domain_sets.append(names)
    if not domain_sets:
        return []
    return sorted(set.intersection(*domain_sets))


def load_values(domains: Sequence[str]) -> Dict[tuple, Optional[float]]:
    values: Dict[tuple, Optional[float]] = {}
    for condition in METHODS:
        run_path = _abs_path(DEFAULT_RUNS[condition])
        for probe_type, metric, probe_family, _, _ in PANELS:
            df = load_metrics(
                run_path,
                probe_type,
                domains,
                str(REPO_ROOT),
                metrics=(metric,),
                probe_family=probe_family,
                mcqa_variant="preferred" if probe_family == "mcqa" else "preferred",
            )
            values[(condition, probe_type, metric)] = final_value(df, metric)
    return values


def plot(values: Dict[tuple, Optional[float]], output: str):
    apply_plot_style()
    plt.rcParams.update(
        {
            "axes.labelsize": 15,
            "axes.titlesize": 17,
            "font.size": 14,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 8))
    x = np.arange(len(METHODS))
    colors = [CONDITION_COLORS[c] for c in METHODS]
    tick_labels = [CONDITION_LABELS[c] for c in METHODS]

    for ax, (probe_type, metric, _, title, ylabel) in zip(axes.flat, PANELS):
        heights = [
            values.get((c, probe_type, metric)) for c in METHODS
        ]
        heights = [np.nan if h is None else h for h in heights]
        ax.bar(x, heights, color=colors, width=0.6, alpha=0.85, edgecolor="black")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.25)
        for xi, h in zip(x, heights):
            if not np.isnan(h):
                ax.annotate(
                    f"{h:.2f}",
                    (xi, h),
                    textcoords="offset points",
                    xytext=(0, 3 if h >= 0 else -12),
                    ha="center",
                    fontsize=11,
                )

    fig.suptitle("OLMo-2 32B: Source vs. Paraphrased vs. +Aux (reeval_v3)", fontsize=18)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_figure(fig, output)


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="plots/probe_scaling/probe_32b_conditions_2x2_reeval_v3")
    parser.add_argument("--domains", nargs="+")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    run_paths = [_abs_path(p) for p in DEFAULT_RUNS.values()]
    domains = args.domains or discover_domains(run_paths)
    if not domains:
        raise RuntimeError("No domains found in configured 32B reeval_v3 bundles")
    print(f"Using {len(domains)} domains")

    values = load_values(domains)
    for (condition, probe_type, metric), val in values.items():
        if val is None:
            print(f"Warning: missing {condition} {probe_type} {metric}")
    plot(values, args.output)


if __name__ == "__main__":
    main()
