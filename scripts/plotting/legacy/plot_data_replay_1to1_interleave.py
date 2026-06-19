import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from scripts.plotting.plot_utils import aggregate_across_domains, transform_to_exposure_steps

try:
    from utils.llm_plotting import set_plot_style
except ModuleNotFoundError:
    def set_plot_style() -> None:
        plt.rcParams.update(
            {
                "axes.spines.top": False,
                "axes.spines.right": False,
                "font.size": 10,
            }
        )


RUNS = {
    "1B": {
        "No Data Replay": (
            "results/FT/full/1b/probes_v9/newline2/source_only/fill_dclm/"
            "domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs32_lr2e-05/"
            "overlap_1_4/11_23_05_24"
        ),
        "1:1 via interleave": (
            "results/FT/full/1b/probes_v9/newline2/source_only/sep_1_dclm/"
            "domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs32_lr2e-05/run"
        ),
    },
    "7B": {
        "No Data Replay": (
            "results/FT/full/7b/probes_v9/newline2/source_only/fill_dclm/"
            "domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs32_lr2e-05/"
            "overlap_1_4/11_23_05_24"
        ),
        "1:1 via interleave": (
            "results/FT/full/7b/probes_v9/newline2/source_only/sep_1_dclm/"
            "domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs32_lr2e-05/run"
        ),
    },
}

STYLE = {
    "No Data Replay": {"color": "0.35", "linestyle": "-"},
    "1:1 via interleave": {"color": "deepskyblue", "linestyle": "--"},
}


def get_domains(project_root: Path) -> list[str]:
    first_run = next(iter(next(iter(RUNS.values())).values()))
    for part in Path(first_run).parts:
        if part.startswith("domains_"):
            return part.removeprefix("domains_").split("-")
    domains_path = project_root / "data/arxiv/cleaned"
    return sorted(path.stem for path in domains_path.glob("*.tex") if path.is_file())


def load_plot_data(project_root: Path, domains: list[str]) -> dict[str, dict[str, pd.DataFrame]]:
    data: dict[str, dict[str, pd.DataFrame]] = {}

    for model_id, runs in RUNS.items():
        data[model_id] = {}
        for probe_type in ("knowledge", "inference"):
            frames = []
            for method, relative_path in runs.items():
                run_path = project_root / relative_path
                if not run_path.is_dir():
                    print(f"Warning: missing run path for {model_id} {method}: {run_path}")
                    continue

                df = aggregate_across_domains(
                    str(run_path),
                    probe_type,
                    domains,
                    project_root=str(project_root),
                )
                if df.empty:
                    continue

                strategy_name = "With Data Replay (1:1) via interleave" if method == "1:1 via interleave" else method
                df = transform_to_exposure_steps(df, strategy_name)
                df["method"] = method
                frames.append(df)

            data[model_id][probe_type] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    return data


def plot_panel(ax, df: pd.DataFrame, title: str, show_ylabel: bool) -> None:
    if df.empty:
        ax.set_title(f"{title} (No Data)")
        return

    for method in RUNS["1B"]:
        method_df = df[df["method"] == method]
        if method_df.empty:
            continue
        plot_df = method_df.groupby("Exposure Steps", as_index=False)["log_prob"].mean()
        ax.plot(
            plot_df["Exposure Steps"],
            plot_df["log_prob"],
            label=method,
            linewidth=1.8,
            **STYLE[method],
        )

    max_x = int(df["Exposure Steps"].max())
    ax.set_title(title)
    ax.set_xlabel("Exposure Steps")
    ax.set_xticks(np.arange(0, max_x + 1, 30))
    ax.grid(True)
    if show_ylabel:
        ax.set_ylabel("Log Prob.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot legacy no-replay vs 1:1 interleaved data replay results."
    )
    parser.add_argument(
        "--output",
        default="plots/data_replay_1B_7B_comparison_exposure_steps.pdf",
        help="Output plot path.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[3]
    domains = get_domains(project_root)
    data = load_plot_data(project_root, domains)

    set_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=False)

    for row, model_id in enumerate(("1B", "7B")):
        plot_panel(axes[row, 0], data[model_id]["knowledge"], f"{model_id}: Factual Probes", True)
        plot_panel(axes[row, 1], data[model_id]["inference"], f"{model_id}: Compositional Probes", False)

    legend_handles = [
        Line2D([0], [0], label=label, linewidth=2, **style)
        for label, style in STYLE.items()
    ]
    fig.legend(legend_handles, [handle.get_label() for handle in legend_handles], loc="upper center", ncol=2)
    fig.subplots_adjust(top=0.9, wspace=0.22, hspace=0.35)

    output_path = project_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    main()
