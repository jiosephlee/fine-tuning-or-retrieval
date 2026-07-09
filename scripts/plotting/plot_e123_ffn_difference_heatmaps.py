import argparse
import os
import sys
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.plotting.plot_utils import apply_plot_style, save_figure  # noqa: E402


PROJECTIONS = ("gate", "up", "down")
PROJECTION_LABELS = {
    "gate": "gate_proj",
    "up": "up_proj",
    "down": "down_proj",
}
COMPARISON_TITLES = {
    "aux_base_minus_source_base": "Aux. Views - Source",
    "para_base_minus_source_base": "Paraphrase - Source",
}
METRIC_TITLES = {
    "cosine_distance": "cosine distance",
    "relative_delta_norm": "relative delta norm",
}
COLORBAR_LABELS = {
    "aux_base_minus_source_base": r"$\Delta$ (Aux. Views - Source) Cosine Distance from Base Model",
    "para_base_minus_source_base": r"$\Delta$ (Paraphrase - Source) Cosine Distance from Base Model",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Restyle saved E1/E2/E3 FFN signed-difference heatmaps from .npy arrays."
    )
    parser.add_argument(
        "--input_dir",
        default=os.path.join(
            REPO_ROOT,
            "plots",
            "ffn_channel_heatmaps",
            "e123_7b_20260516",
            "mlp",
        ),
    )
    parser.add_argument(
        "--comparison",
        default="aux_base_minus_source_base",
        choices=tuple(COMPARISON_TITLES),
    )
    parser.add_argument(
        "--metric",
        default="cosine_distance",
        choices=tuple(METRIC_TITLES),
    )
    parser.add_argument("--clip_percentile", type=float, default=99.0)
    parser.add_argument("--cmap", default="RdBu_r")
    parser.add_argument("--fig_width", type=float, default=16.5)
    parser.add_argument("--fig_height", type=float, default=9.8)
    return parser.parse_args()


def load_projection_arrays(input_dir: str, comparison: str, metric: str) -> dict[str, np.ndarray]:
    arrays = {}
    for projection in PROJECTIONS:
        path = os.path.join(input_dir, f"{comparison}_{projection}_{metric}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing array: {path}")
        arrays[projection] = np.load(path)
    return arrays


def symmetric_vmax(arrays: Iterable[np.ndarray], clip_percentile: float) -> float:
    values = np.concatenate([np.asarray(array).reshape(-1) for array in arrays]).astype(np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    abs_values = np.abs(finite)
    if clip_percentile and clip_percentile > 0:
        vmax = float(np.percentile(abs_values, clip_percentile))
    else:
        vmax = float(np.max(abs_values))
    return vmax if vmax > 0 else 1.0


def plot_difference_heatmap(args: argparse.Namespace) -> str:
    apply_plot_style()
    plt.rcParams.update({
        "axes.labelsize": 20,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "font.size": 15,
    })
    arrays = load_projection_arrays(args.input_dir, args.comparison, args.metric)
    vmax = symmetric_vmax(arrays.values(), args.clip_percentile)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    metric_title = METRIC_TITLES[args.metric]
    fig, axes = plt.subplots(
        len(PROJECTIONS),
        1,
        figsize=(args.fig_width, args.fig_height),
        sharex=True,
        squeeze=False,
    )
    axes = axes.reshape(-1)
    image = None

    for row, (ax, projection) in enumerate(zip(axes, PROJECTIONS)):
        image = ax.imshow(
            arrays[projection],
            aspect="auto",
            interpolation="nearest",
            cmap=args.cmap,
            norm=norm,
        )
        ax.set_ylabel(f"{PROJECTION_LABELS[projection]}\nLayer", fontsize=20, labelpad=8)
        ax.tick_params(axis="both", labelsize=15)
        ax.grid(False)
        if row < len(PROJECTIONS) - 1:
            ax.set_xticklabels([])

    axes[-1].set_xlabel("FFN channel", fontsize=22, labelpad=8)
    fig.subplots_adjust(left=0.095, right=0.84, bottom=0.095, top=0.985, hspace=0.16)
    if image is not None:
        cbar = fig.colorbar(image, ax=list(axes), shrink=0.9, pad=0.02)
        cbar.set_label(
            COLORBAR_LABELS[args.comparison]
            if args.metric == "cosine_distance"
            else f"{COMPARISON_TITLES[args.comparison]} {metric_title}",
            labelpad=16,
            fontsize=20,
        )
        cbar.ax.tick_params(labelsize=15)
        cbar.ax.yaxis.get_offset_text().set_fontsize(15)

    output = os.path.join(args.input_dir, f"{args.comparison}_{args.metric}_all_projections.png")
    save_figure(fig, output, suffixes=(".png",))
    return output


def main() -> None:
    output = plot_difference_heatmap(parse_args())
    print(f"Saved styled heatmap: {output}")


if __name__ == "__main__":
    main()
