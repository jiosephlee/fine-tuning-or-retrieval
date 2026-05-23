import os
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

# Ensure repo root is resolvable for imports
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(REPO_ROOT)

from scripts.plotting.plot_utils import (
    apply_ylim,
    compute_unified_ylim,
    discover_domains,
    load_probe_series,
)
from utils.llm_plotting import set_plot_style


RUNS = [
    {
        "label": "Semi-cleaned v1",
        "color": "#1f77b4",
        "path": "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/source_only/sep_1_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e50/bs64_lr2e-05/overlap_1_10/semi_cleaned_v1/10_14_23_03",
    },
    {
        "label": "Semi-cleaned v2",
        "color": "#ff7f0e",
        "path": "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/source_only/sep_1_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e50/bs64_lr2e-05/overlap_1_10/semi_cleaned_v2/10_15_02_08",
    },
    {
        "label": "Semi-cleaned v3",
        "color": "#2ca02c",
        "path": "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/source_only/sep_1_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e50/bs64_lr2e-05/overlap_1_10/semi_cleaned_v3",
    },
    {
        "label": "Raw",
        "color": "#d62728",
        "path": "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/source_only/sep_1_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e50/bs64_lr2e-05/overlap_1_10/11_09_15_58",
    },
    {
        "label": "Fully cleaned",
        "color": "#9467bd",
        "path": "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/source_only/sep_1_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e50/bs64_lr2e-05/overlap_1_10/10_15_08_02",
    },
]

PROBE_TYPES = [
    ("knowledge", "Factual"),
    ("inference", "Compositional"),
]


def main():
    project_root = REPO_ROOT
    domains = discover_domains(project_root)
    if not domains:
        print("No domains found; aborting.")
        return

    set_plot_style()
    fig, axes = plt.subplots(1, len(PROBE_TYPES), figsize=(10, 4), sharey=True)
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]

    all_y = []

    for ax, (probe_key, probe_label) in zip(axes, PROBE_TYPES):
        ax.set_title(f"{probe_label} Probes")
        for run in RUNS:
            series = load_probe_series(
                run["path"],
                probe_key,
                domains,
                project_root,
            )
            if series is None or series.empty:
                print(f"Skipping {run['label']} ({probe_label}): data missing.")
                continue

            ax.plot(
                series["step"],
                series["log_prob"],
                color=run["color"],
                linewidth=1.6,
                label=run["label"],
            )
            all_y.extend(series["log_prob"].tolist())

        ax.set_xlabel("Training Steps")
        if ax is axes[0]:
            ax.set_ylabel("Mean Log Probability")
        ax.grid(False)

    ylim = compute_unified_ylim(values=all_y)
    if ylim:
        apply_ylim(axes, ylim)

    handles = [
        Line2D([0], [0], color=run["color"], linewidth=2, label=run["label"])
        for run in RUNS
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(RUNS),
        bbox_to_anchor=(0.5, -0.1),
        fontsize="small",
    )

    fig.subplots_adjust(bottom=0.2, wspace=0.3)
    os.makedirs("plots", exist_ok=True)
    out_path = os.path.join("plots", "cleaning_comparison_7B_bs64.pdf")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()


