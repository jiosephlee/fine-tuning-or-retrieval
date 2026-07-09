import argparse
import glob
import json
import os
import re
import sys
from typing import Dict, Iterable, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from huggingface_hub import snapshot_download
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from safetensors import safe_open

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.plotting.plot_inference_mcqa_scaling import (  # noqa: E402
    apply_plot_style,
    save_figure,
)


DEFAULT_MODELS = {
    "source": "jiosephlee/e1-olmo2-7b-source-only-20260516",
    "para": "jiosephlee/e2-olmo2-7b-para9-20260516",
    "aux": "jiosephlee/e3-olmo2-7b-para9-expl-20260516",
}
COMPARISONS = (
    ("para_vs_source", "source", "para", "Para vs Source"),
    ("aux_vs_source", "source", "aux", "Aux vs Source"),
)
PROJECTIONS = ("gate", "up", "down")
METRICS = ("cosine_distance", "relative_delta_norm")
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot per-channel MLP heatmaps comparing E2/E3 checkpoints directly "
            "against the E1 source checkpoint."
        )
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help=(
            "Optional JSON/YAML manifest with source/para/aux checkpoint keys. "
            "Defaults to the 20260516 Hugging Face checkpoints."
        ),
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(
            REPO_ROOT,
            "plots",
            "ffn_channel_heatmaps",
            "e123_7b_20260516",
            "source_comparison",
        ),
    )
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=METRICS,
        default=("cosine_distance",),
        help="Metrics to plot. Defaults to cosine distance for the requested comparison.",
    )
    parser.add_argument("--clip_percentile", type=float, default=99.0)
    parser.add_argument("--fig_width", type=float, default=22.0)
    parser.add_argument("--fig_height", type=float, default=12.0)
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument("--cmap", default="viridis")
    parser.add_argument(
        "--from_arrays",
        action="store_true",
        help="Replot from existing .npy arrays instead of recomputing from checkpoints.",
    )
    return parser.parse_args()


def load_manifest(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = handle.read()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore

            data = yaml.safe_load(raw)
        except Exception as exc:
            raise ValueError("Manifest must be valid JSON or YAML.") from exc

    if "checkpoints" in data and isinstance(data["checkpoints"], dict):
        data = data["checkpoints"]

    models = dict(DEFAULT_MODELS)
    for key in models:
        if key in data:
            models[key] = data[key]
    return models


def model_refs(args: argparse.Namespace) -> Dict[str, str]:
    if args.manifest:
        return load_manifest(args.manifest)
    return dict(DEFAULT_MODELS)


def extract_layer_index(param_name: str) -> int:
    matches = re.findall(r"\.(\d+)\.", param_name)
    if not matches:
        raise ValueError(f"Could not parse layer index from parameter name: {param_name}")
    return int(matches[-1])


def projection_from_param_name(param_name: str):
    if ".mlp.gate_proj.weight" in param_name:
        return "gate"
    if ".mlp.up_proj.weight" in param_name:
        return "up"
    if ".mlp.down_proj.weight" in param_name:
        return "down"
    return None


def snapshot_safetensors(model_ref: str, cache_dir: str | None) -> Dict[str, str]:
    local_dir = snapshot_download(
        model_ref,
        cache_dir=cache_dir,
        allow_patterns=("*.safetensors", "*.safetensors.index.json"),
    )
    index_path = os.path.join(local_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as handle:
            weight_map = json.load(handle).get("weight_map", {})
        return {
            param_name: os.path.join(local_dir, shard_name)
            for param_name, shard_name in weight_map.items()
        }

    paths = sorted(glob.glob(os.path.join(local_dir, "*.safetensors")))
    if not paths:
        raise FileNotFoundError(f"No safetensors weights found for {model_ref}.")

    param_to_file = {}
    for path in paths:
        with safe_open(path, framework="pt", device="cpu") as handle:
            for param_name in handle.keys():
                param_to_file[param_name] = path
    return param_to_file


def index_mlp_parameters(
    model_ref: str,
    cache_dir: str | None,
) -> Dict[int, Dict[str, Tuple[str, str]]]:
    param_to_file = snapshot_safetensors(model_ref, cache_dir)
    layer_to_projection: Dict[int, Dict[str, Tuple[str, str]]] = {}
    for param_name, path in param_to_file.items():
        if not param_name.endswith(".weight") or ".mlp." not in param_name:
            continue
        projection = projection_from_param_name(param_name)
        if projection is None:
            continue
        layer = extract_layer_index(param_name)
        layer_to_projection.setdefault(layer, {})[projection] = (path, param_name)

    if not layer_to_projection:
        raise ValueError(f"No MLP projection weights found for {model_ref}.")

    missing = [
        layer
        for layer, mapping in layer_to_projection.items()
        if any(projection not in mapping for projection in PROJECTIONS)
    ]
    if missing:
        raise ValueError(f"Missing MLP projections for layers: {missing[:8]}")

    layers = sorted(layer_to_projection)
    expected = list(range(layers[-1] + 1))
    if layers != expected:
        raise ValueError(f"Layer indices are not contiguous: {layers[:8]} ... {layers[-8:]}")

    return layer_to_projection


def load_tensor(param_ref: Tuple[str, str]) -> torch.Tensor:
    path, param_name = param_ref
    with safe_open(path, framework="pt", device="cpu") as handle:
        return handle.get_tensor(param_name).to(dtype=torch.float32)


def vector_metrics_rows(ref: torch.Tensor, cmp: torch.Tensor) -> Dict[str, np.ndarray]:
    diff = cmp - ref
    ref_norm = torch.linalg.norm(ref, dim=1)
    cmp_norm = torch.linalg.norm(cmp, dim=1)
    diff_norm = torch.linalg.norm(diff, dim=1)
    dot = torch.sum(ref * cmp, dim=1)
    return {
        "relative_delta_norm": (diff_norm / (ref_norm + EPS)).cpu().numpy(),
        "cosine_distance": (1.0 - dot / (ref_norm * cmp_norm + EPS)).cpu().numpy(),
    }


def vector_metrics_cols(ref: torch.Tensor, cmp: torch.Tensor) -> Dict[str, np.ndarray]:
    diff = cmp - ref
    ref_norm = torch.linalg.norm(ref, dim=0)
    cmp_norm = torch.linalg.norm(cmp, dim=0)
    diff_norm = torch.linalg.norm(diff, dim=0)
    dot = torch.sum(ref * cmp, dim=0)
    return {
        "relative_delta_norm": (diff_norm / (ref_norm + EPS)).cpu().numpy(),
        "cosine_distance": (1.0 - dot / (ref_norm * cmp_norm + EPS)).cpu().numpy(),
    }


def compute_comparison(
    ref_model: str,
    cmp_model: str,
    cache_dir: str | None,
    metrics: Sequence[str],
) -> Dict[str, Dict[str, np.ndarray]]:
    print(f"Indexing reference: {ref_model}")
    ref_index = index_mlp_parameters(ref_model, cache_dir)
    print(f"Indexing comparison: {cmp_model}")
    cmp_index = index_mlp_parameters(cmp_model, cache_dir)

    layers = sorted(ref_index)
    if layers != sorted(cmp_index):
        raise ValueError("Reference and comparison layers do not match.")

    probe = load_tensor(ref_index[0]["gate"])
    d_ff, d_model = probe.shape
    del probe

    output = {
        projection: {
            metric: np.zeros((len(layers), d_ff), dtype=np.float32)
            for metric in metrics
        }
        for projection in PROJECTIONS
    }

    for row, layer in enumerate(layers):
        print(f"  layer {layer}")
        for projection in PROJECTIONS:
            ref = load_tensor(ref_index[layer][projection])
            cmp = load_tensor(cmp_index[layer][projection])
            if ref.shape != cmp.shape:
                raise ValueError(f"Shape mismatch at layer {layer} {projection}: {ref.shape} vs {cmp.shape}")

            if projection in ("gate", "up"):
                if ref.shape != (d_ff, d_model):
                    raise ValueError(f"Unexpected {projection} shape at layer {layer}: {ref.shape}")
                values = vector_metrics_rows(ref, cmp)
            else:
                if ref.shape != (d_model, d_ff):
                    raise ValueError(f"Unexpected down shape at layer {layer}: {ref.shape}")
                values = vector_metrics_cols(ref, cmp)

            for metric in metrics:
                output[projection][metric][row] = values[metric].astype(np.float32)
            del ref
            del cmp

    return output


def compute_vmax(arrays: Iterable[np.ndarray], clip_percentile: float) -> float:
    concat = np.concatenate([array.reshape(-1) for array in arrays]).astype(np.float64)
    finite = concat[np.isfinite(concat)]
    if finite.size == 0:
        return 1.0
    if clip_percentile and clip_percentile > 0:
        vmax = float(np.percentile(finite, clip_percentile))
    else:
        vmax = float(np.max(finite))
    if vmax <= 0:
        return 1.0
    return vmax


def save_arrays(
    comparison: str,
    values: Dict[str, Dict[str, np.ndarray]],
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for projection in PROJECTIONS:
        for metric in values[projection]:
            path = os.path.join(output_dir, f"{comparison}_{projection}_{metric}.npy")
            np.save(path, values[projection][metric])
            print(f"Saved array: {path}")


def load_arrays(
    comparison: str,
    metrics: Sequence[str],
    output_dir: str,
) -> Dict[str, Dict[str, np.ndarray]]:
    values: Dict[str, Dict[str, np.ndarray]] = {projection: {} for projection in PROJECTIONS}
    for projection in PROJECTIONS:
        for metric in metrics:
            path = os.path.join(output_dir, f"{comparison}_{projection}_{metric}.npy")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing array: {path}")
            values[projection][metric] = np.load(path)
    return values


def plot_metric(
    comparison: str,
    title: str,
    values: Dict[str, Dict[str, np.ndarray]],
    metric: str,
    output_dir: str,
    args: argparse.Namespace,
) -> str:
    metric_title = metric.replace("_", " ")
    vmax = compute_vmax(
        [values[projection][metric] for projection in PROJECTIONS],
        args.clip_percentile,
    )
    fig, axes = plt.subplots(
        len(PROJECTIONS),
        1,
        figsize=(15, 3.2 * len(PROJECTIONS)),
        sharex=True,
        squeeze=False,
    )
    axes = axes.reshape(-1)

    image = None
    labels = {"gate": "gate_proj", "up": "up_proj", "down": "down_proj"}
    for row, (ax, projection) in enumerate(zip(axes, PROJECTIONS)):
        image = ax.imshow(
            values[projection][metric],
            aspect="auto",
            interpolation="nearest",
            cmap=args.cmap,
            vmin=0.0,
            vmax=vmax,
        )
        if row == 0:
            ax.set_title(f"{title} — {metric_title}")
        ax.set_ylabel(labels[projection])
        ax.grid(False)
        if row == len(PROJECTIONS) - 1:
            ax.set_xlabel("Channel")
        else:
            ax.set_xticklabels([])
    axes[-1].set_xlabel("Channel")
    fig.tight_layout()
    if image is not None:
        cax = inset_axes(
            axes[0],
            width="18%",
            height="6%",
            loc="lower right",
            borderpad=1.2,
        )
        cbar = fig.colorbar(image, cax=cax, orientation="horizontal")
        cbar.set_ticks([0.0, vmax / 2.0, vmax])
        cbar.ax.xaxis.set_major_formatter(FormatStrFormatter("%.1e"))
        cbar.ax.set_title(metric_title, fontsize=8, pad=2)
        cbar.ax.tick_params(labelsize=8, length=2, pad=1)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{comparison}_{metric}_all_projections.png")
    save_figure(fig, path, suffixes=(".png",))
    print(f"Saved heatmap: {path} (vmax={vmax:.6f}, clip_percentile={args.clip_percentile})")
    return path


def main() -> None:
    args = parse_args()
    apply_plot_style()
    models = model_refs(args)
    mlp_dir = os.path.join(args.output_dir, "mlp")

    print("Using checkpoints:")
    for key in ("source", "para", "aux"):
        print(f"  - {key}: {models[key]}")

    for comparison, ref_key, cmp_key, title in COMPARISONS:
        print(f"\n=== {comparison}: {cmp_key} vs {ref_key} ===")
        if args.from_arrays:
            values = load_arrays(comparison, args.metrics, mlp_dir)
        else:
            values = compute_comparison(
                models[ref_key],
                models[cmp_key],
                args.cache_dir,
                args.metrics,
            )
            save_arrays(comparison, values, mlp_dir)
        for metric in args.metrics:
            plot_metric(comparison, title, values, metric, mlp_dir, args)

    print("\nDone. Source-comparison heatmaps saved under:")
    print(f"  {mlp_dir}")


if __name__ == "__main__":
    main()
