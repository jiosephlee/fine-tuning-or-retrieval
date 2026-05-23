import argparse
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

# Ensure repo root is resolvable for imports
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(REPO_ROOT)

from scripts.plotting.plot_utils import (
    apply_ylim,
    compute_unified_ylim,
    discover_domains,
    load_probe_series,
)
from utils.llm_plotting import set_plot_style


METHOD_RUNS = {
    "source": {
        "label": "Source",
        "color": "#1f77b4",
        "runs": {
            # "1e-5": "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/source_only/sep_1_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e50/bs32_lr1e-05/overlap_1_4/09_24_12_52",
            # "2e-5": "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/source_only/sep_1_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e50/bs32_lr2e-05/overlap_1_4/semi_cleaned_v2/09_24_23_29",
            # "4e-5": "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/source_only/sep_1_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e50/bs32_lr4e-05/overlap_1_4/09_24_18_48",
            "1e-5": "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr1e-05/overlap_1_4/11_27_20_32",
            "2e-5": "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_20_22_51",
            "4e-5": "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr4e-05/overlap_1_4/11_27_21_29",
            "8e-5": "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr8e-05/overlap_1_4/11_27_22_27",
        },
    },
    "para9": {
        "label": "Paraphrase",
        "color": "#ff7f0e",
        "runs": {
            # "1e-5": "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/para9/sep_1_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e50/bs32_lr1e-05/overlap_1_4",
            # "2e-5": "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/para9/sep_1_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e50/bs32_lr2e-05/run",
            # "4e-5": "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/para9/sep_1_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e50/bs32_lr4e-05/overlap_1_4/11_26_21_04",
            "1e-5": "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr1e-05/overlap_1_4/11_27_20_43",
            "2e-5": "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_20_22_51",
            "4e-5": "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr4e-05/overlap_1_4/11_27_21_39",
            "8e-5": "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr8e-05/overlap_1_4/11_27_22_38",
        },
    },
}

PROBE_TYPES = [
    ("knowledge", "Factual"),
    ("inference", "Compositional"),
]

LINESTYLE_BY_LR = {
    "1e-5": "-",
    "2e-5": "--",
    "4e-5": ":",
    "8e-5": "-.",
}


def main():
    parser = argparse.ArgumentParser(
        description="Plot learning-rate effect on probe performance for Source vs Paraphrase."
    )
    parser.add_argument(
        "--with_LIMA",
        action="store_true",
        help="If set, append LIMA continuation after fine-tuning steps for each run.",
    )
    args = parser.parse_args()

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
        for method_meta in METHOD_RUNS.values():
            for lr, run_path in method_meta["runs"].items():
                series = load_probe_series(
                    run_path,
                    probe_key,
                    domains,
                    project_root,
                    with_lima=args.with_LIMA,
                )
                if series is None or series.empty:
                    print(f"Skipping {method_meta['label']} {lr} ({probe_label}): data missing.")
                    continue

                linestyle = LINESTYLE_BY_LR.get(lr, "-")
                ax.plot(
                    series["step"],
                    series["log_prob"],
                    color=method_meta["color"],
                    linestyle=linestyle,
                    linewidth=1.6,
                )
                all_y.extend(series["log_prob"].tolist())

        ax.set_xlabel("Training Steps")
        if ax is axes[0]:
            ax.set_ylabel("Mean Log Probability")
        ax.grid(False)

    ylim = compute_unified_ylim(values=all_y)
    if ylim:
        apply_ylim(axes, ylim)

    method_handles = [
        Line2D([0], [0], color=meta["color"], linewidth=2, label=meta["label"])
        for meta in METHOD_RUNS.values()
    ]
    lr_handles = [
        Line2D(
            [0],
            [0],
            color="gray",
            linewidth=2,
            linestyle=LINESTYLE_BY_LR[lr],
            label=lr,
        )
        for lr in LINESTYLE_BY_LR
    ]
    legend_handles = method_handles + lr_handles

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.1),
        fontsize="small",
    )

    fig.subplots_adjust(bottom=0.2, wspace=0.3)
    os.makedirs("plots", exist_ok=True)
    out_path = os.path.join("plots", "lr_effect_1plot.pdf")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()

