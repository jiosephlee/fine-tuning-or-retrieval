import argparse
import json
import os
import re
import sys
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from utils.llm_plotting import set_plot_style


REQUIRED_MODELS = ("base", "source", "para", "aux")
PROJECTIONS = ("gate", "up", "down")
METRICS = ("relative_delta_norm", "cosine_distance")
COMPARISONS = (
    ("source_vs_base", "base", "source"),
    ("para_vs_base", "base", "para"),
    ("aux_vs_base", "base", "aux"),
    ("aux_vs_para", "para", "aux"),
    ("aux_vs_source", "source", "aux"),
)
DIFFERENCE_COMPARISONS = (
    ("aux_base_minus_source_base", "aux_vs_base", "source_vs_base"),
)
COMPARISON_TITLES = {
    "source_vs_base": "Source vs Base",
    "para_vs_base": "Para vs Base",
    "aux_vs_base": "Aux vs Base",
    "aux_vs_para": "Aux vs Para",
    "aux_vs_source": "Aux vs Source",
    "aux_base_minus_source_base": "Aux-Base minus Source-Base",
}
METRIC_TITLES = {
    "relative_delta_norm": "relative delta norm",
    "cosine_distance": "cosine distance",
}
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot per-channel FFN weight change heatmaps across checkpoints."
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to JSON/YAML manifest containing base/source/para/aux checkpoint entries.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join("plots", "ffn_channel_heatmaps"),
        help="Directory to save .npy arrays and figures.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Optional HuggingFace cache dir for model loading.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=("cpu", "cuda", "auto"),
        help="Where to load checkpoint tensors for analysis.",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="float32",
        choices=("float32", "float16", "bfloat16", "auto"),
        help="Dtype passed to from_pretrained (metrics always computed in float32).",
    )
    parser.add_argument(
        "--clip_percentile",
        type=float,
        default=99.0,
        help="Upper percentile clip for heatmap color range. Set <=0 to disable clipping.",
    )
    parser.add_argument(
        "--fig_width",
        type=float,
        default=22.0,
        help="Figure width in inches.",
    )
    parser.add_argument(
        "--fig_height",
        type=float,
        default=12.0,
        help="Figure height in inches.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=240,
        help="Output figure DPI.",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="viridis",
        help="Matplotlib colormap for heatmaps.",
    )
    parser.add_argument(
        "--diff_cmap",
        type=str,
        default="RdBu_r",
        help="Matplotlib colormap for signed difference heatmaps.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Forwarded to AutoModelForCausalLM.from_pretrained.",
    )
    return parser.parse_args()


def load_manifest(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = handle.read()

    ext = os.path.splitext(path)[1].lower()
    data = None

    if ext in (".json", ""):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = None

    if data is None:
        try:
            import yaml  # type: ignore

            data = yaml.safe_load(raw)
        except Exception as exc:
            raise ValueError(
                "Manifest must be valid JSON or YAML. "
                "Install pyyaml for YAML support if needed."
            ) from exc

    if not isinstance(data, dict):
        raise ValueError("Manifest top-level must be a dictionary/object.")

    if "checkpoints" in data and isinstance(data["checkpoints"], dict):
        data = data["checkpoints"]

    missing = [name for name in REQUIRED_MODELS if name not in data]
    if missing:
        raise ValueError(
            f"Manifest missing required entries: {missing}. "
            f"Expected keys: {list(REQUIRED_MODELS)}"
        )

    models = {}
    for name in REQUIRED_MODELS:
        value = data[name]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Manifest value for '{name}' must be a non-empty string.")
        models[name] = value.strip()

    return models


def parse_dtype(dtype_name: str):
    if dtype_name == "auto":
        return "auto"
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_name]


def resolve_device_map(device: str):
    if device == "auto":
        return "auto"
    return {"": device}


def load_model(model_ref: str, args: argparse.Namespace) -> AutoModelForCausalLM:
    print(f"Loading checkpoint: {model_ref}")
    model = AutoModelForCausalLM.from_pretrained(
        model_ref,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
        device_map=resolve_device_map(args.device),
        torch_dtype=parse_dtype(args.torch_dtype),
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model


def _extract_layer_index(param_name: str) -> int:
    # Usually names look like model.layers.12.mlp.gate_proj.weight.
    matches = re.findall(r"\.(\d+)\.", param_name)
    if not matches:
        raise ValueError(f"Could not parse layer index from parameter name: {param_name}")
    return int(matches[-1])


def index_mlp_parameters(model: AutoModelForCausalLM) -> Dict[int, Dict[str, str]]:
    layer_to_projection: Dict[int, Dict[str, str]] = {}

    for name, _ in model.named_parameters():
        if not name.endswith(".weight") or ".mlp." not in name:
            continue

        projection = None
        if ".mlp.gate_proj.weight" in name:
            projection = "gate"
        elif ".mlp.up_proj.weight" in name:
            projection = "up"
        elif ".mlp.down_proj.weight" in name:
            projection = "down"
        if projection is None:
            continue

        layer_idx = _extract_layer_index(name)
        layer_to_projection.setdefault(layer_idx, {})[projection] = name

    if not layer_to_projection:
        raise ValueError(
            "Did not find any MLP projection weights. "
            "Expected parameters ending with .mlp.(gate_proj|up_proj|down_proj).weight"
        )

    bad_layers = [
        layer for layer, mapping in layer_to_projection.items()
        if any(proj not in mapping for proj in PROJECTIONS)
    ]
    if bad_layers:
        raise ValueError(
            f"Missing one or more MLP projection weights for layers: {bad_layers[:8]}"
        )

    ordered_layers = sorted(layer_to_projection.keys())
    expected = list(range(ordered_layers[-1] + 1))
    if ordered_layers != expected:
        raise ValueError(
            "Layer indices are not contiguous from 0..n_layers-1. "
            f"Observed layers: {ordered_layers[:8]} ... {ordered_layers[-8:]}"
        )

    return layer_to_projection


def _vector_metrics_rows(
    ref: torch.Tensor,
    cmp: torch.Tensor,
    eps: float = EPS,
) -> Tuple[np.ndarray, np.ndarray]:
    diff = cmp - ref
    ref_norm = torch.linalg.norm(ref, dim=1)
    cmp_norm = torch.linalg.norm(cmp, dim=1)
    diff_norm = torch.linalg.norm(diff, dim=1)
    dot = torch.sum(ref * cmp, dim=1)

    relative_delta = diff_norm / (ref_norm + eps)
    cosine_distance = 1.0 - (dot / (ref_norm * cmp_norm + eps))
    return relative_delta.cpu().numpy(), cosine_distance.cpu().numpy()


def _vector_metrics_cols(
    ref: torch.Tensor,
    cmp: torch.Tensor,
    eps: float = EPS,
) -> Tuple[np.ndarray, np.ndarray]:
    diff = cmp - ref
    ref_norm = torch.linalg.norm(ref, dim=0)
    cmp_norm = torch.linalg.norm(cmp, dim=0)
    diff_norm = torch.linalg.norm(diff, dim=0)
    dot = torch.sum(ref * cmp, dim=0)

    relative_delta = diff_norm / (ref_norm + eps)
    cosine_distance = 1.0 - (dot / (ref_norm * cmp_norm + eps))
    return relative_delta.cpu().numpy(), cosine_distance.cpu().numpy()


def _to_float32_cpu(param: torch.nn.Parameter) -> torch.Tensor:
    return param.detach().to(dtype=torch.float32, device="cpu")


def compute_comparison_metrics(
    ref_model: AutoModelForCausalLM,
    cmp_model: AutoModelForCausalLM,
) -> Dict[str, Dict[str, np.ndarray]]:
    ref_index = index_mlp_parameters(ref_model)
    cmp_index = index_mlp_parameters(cmp_model)

    if sorted(ref_index.keys()) != sorted(cmp_index.keys()):
        raise ValueError(
            "Reference and comparison models have different layer indices: "
            f"{sorted(ref_index.keys())[:8]} vs {sorted(cmp_index.keys())[:8]}"
        )

    layer_indices = sorted(ref_index.keys())
    n_layers = len(layer_indices)

    probe_ref = _to_float32_cpu(ref_model.get_parameter(ref_index[0]["gate"]))
    d_ff, d_model = probe_ref.shape
    if d_ff <= 0 or d_model <= 0:
        raise ValueError(f"Unexpected gate weight shape: {probe_ref.shape}")

    metrics: Dict[str, Dict[str, np.ndarray]] = {
        "gate": {
            "relative_delta_norm": np.zeros((n_layers, d_ff), dtype=np.float32),
            "cosine_distance": np.zeros((n_layers, d_ff), dtype=np.float32),
        },
        "up": {
            "relative_delta_norm": np.zeros((n_layers, d_ff), dtype=np.float32),
            "cosine_distance": np.zeros((n_layers, d_ff), dtype=np.float32),
        },
        "down": {
            "relative_delta_norm": np.zeros((n_layers, d_ff), dtype=np.float32),
            "cosine_distance": np.zeros((n_layers, d_ff), dtype=np.float32),
        },
    }

    for row_idx, layer_idx in enumerate(layer_indices):
        for projection in PROJECTIONS:
            ref_tensor = _to_float32_cpu(
                ref_model.get_parameter(ref_index[layer_idx][projection])
            )
            cmp_tensor = _to_float32_cpu(
                cmp_model.get_parameter(cmp_index[layer_idx][projection])
            )
            if ref_tensor.shape != cmp_tensor.shape:
                raise ValueError(
                    f"Shape mismatch at layer {layer_idx} {projection}: "
                    f"{ref_tensor.shape} vs {cmp_tensor.shape}"
                )

            if projection in ("gate", "up"):
                if ref_tensor.shape != (d_ff, d_model):
                    raise ValueError(
                        f"Unexpected shape for {projection} at layer {layer_idx}: "
                        f"{ref_tensor.shape}, expected ({d_ff}, {d_model})"
                    )
                rel, cos = _vector_metrics_rows(ref_tensor, cmp_tensor)
            else:
                if ref_tensor.shape != (d_model, d_ff):
                    raise ValueError(
                        f"Unexpected shape for down at layer {layer_idx}: "
                        f"{ref_tensor.shape}, expected ({d_model}, {d_ff})"
                    )
                rel, cos = _vector_metrics_cols(ref_tensor, cmp_tensor)

            metrics[projection]["relative_delta_norm"][row_idx] = rel.astype(np.float32)
            metrics[projection]["cosine_distance"][row_idx] = cos.astype(np.float32)

    return metrics


def save_metric_arrays(
    metrics_by_projection: Dict[str, Dict[str, np.ndarray]],
    comparison_name: str,
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for projection in PROJECTIONS:
        for metric in METRICS:
            file_name = f"{comparison_name}_{projection}_{metric}.npy"
            path = os.path.join(output_dir, file_name)
            np.save(path, metrics_by_projection[projection][metric])
            print(f"Saved array: {path}")


def subtract_metric_dicts(
    minuend_metrics: Dict[str, Dict[str, np.ndarray]],
    subtrahend_metrics: Dict[str, Dict[str, np.ndarray]],
) -> Dict[str, Dict[str, np.ndarray]]:
    diff_metrics: Dict[str, Dict[str, np.ndarray]] = {}
    for projection in PROJECTIONS:
        diff_metrics[projection] = {}
        for metric in METRICS:
            diff_metrics[projection][metric] = (
                minuend_metrics[projection][metric]
                - subtrahend_metrics[projection][metric]
            ).astype(np.float32)
    return diff_metrics


def _compute_vmax(arrays: Iterable[np.ndarray], clip_percentile: float) -> float:
    concat = np.concatenate([x.reshape(-1) for x in arrays]).astype(np.float64)
    finite = concat[np.isfinite(concat)]
    if finite.size == 0:
        return 1.0
    if clip_percentile is not None and clip_percentile > 0:
        vmax = float(np.percentile(finite, clip_percentile))
    else:
        vmax = float(np.max(finite))
    if vmax <= 0:
        return float(np.max(finite)) if np.max(finite) > 0 else 1.0
    return vmax


def _compute_symmetric_vmax(arrays: Iterable[np.ndarray], clip_percentile: float) -> float:
    concat = np.concatenate([x.reshape(-1) for x in arrays]).astype(np.float64)
    finite = concat[np.isfinite(concat)]
    if finite.size == 0:
        return 1.0

    finite_abs = np.abs(finite)
    if clip_percentile is not None and clip_percentile > 0:
        vmax = float(np.percentile(finite_abs, clip_percentile))
    else:
        vmax = float(np.max(finite_abs))
    if vmax <= 0:
        return float(np.max(finite_abs)) if np.max(finite_abs) > 0 else 1.0
    return vmax


def _format_comparison_title(comparison_name: str) -> str:
    return COMPARISON_TITLES.get(comparison_name, comparison_name.replace("_", " ").title())


def _format_metric_title(metric: str) -> str:
    return METRIC_TITLES.get(metric, metric.replace("_", " "))


def plot_grouped_heatmap(
    metrics_by_projection: Dict[str, Dict[str, np.ndarray]],
    comparison_name: str,
    metric: str,
    output_dir: str,
    clip_percentile: float,
    fig_width: float,
    fig_height: float,
    dpi: int,
    cmap: str,
    center_zero: bool = False,
) -> None:
    arrays = [metrics_by_projection[proj][metric] for proj in PROJECTIONS]
    if center_zero:
        vmax = _compute_symmetric_vmax(arrays, clip_percentile=clip_percentile)
        vmin = -vmax
    else:
        vmax = _compute_vmax(arrays, clip_percentile=clip_percentile)
        vmin = 0.0

    comparison_title = _format_comparison_title(comparison_name)
    metric_title = _format_metric_title(metric)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(fig_width, fig_height),
        sharex=True,
        constrained_layout=True,
    )
    fig.suptitle(
        f"{comparison_title} — {metric_title}",
        fontsize=16,
    )

    image = None
    projection_labels = {
        "gate": "gate_proj",
        "up": "up_proj",
        "down": "down_proj",
    }
    for idx, projection in enumerate(PROJECTIONS):
        ax = axes[idx]
        array = metrics_by_projection[projection][metric]
        image = ax.imshow(
            array,
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_ylabel("Layer")
        ax.set_title(
            f"{comparison_title} — "
            f"{projection_labels[projection]} — {metric_title}"
        )

    axes[-1].set_xlabel("Channel")
    if image is not None:
        cbar = fig.colorbar(image, ax=list(axes), shrink=0.95, pad=0.01)
        cbar.set_label(metric_title)

    os.makedirs(output_dir, exist_ok=True)
    file_name = f"{comparison_name}_{metric}_all_projections.png"
    path = os.path.join(output_dir, file_name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(
        f"Saved heatmap: {path} "
        f"(metric={metric}, shared_vmin={vmin:.6f}, shared_vmax={vmax:.6f}, "
        f"clip_percentile={clip_percentile})"
    )


def main() -> None:
    args = parse_args()
    set_plot_style()
    os.makedirs(args.output_dir, exist_ok=True)

    model_paths = load_manifest(args.manifest)
    print("Resolved model checkpoints:")
    for name in REQUIRED_MODELS:
        print(f"  - {name}: {model_paths[name]}")

    metrics_by_comparison = {}
    for comparison_name, ref_name, cmp_name in COMPARISONS:
        print(f"\n=== {comparison_name} ({cmp_name} vs {ref_name}) ===")
        ref_model = load_model(model_paths[ref_name], args)
        cmp_model = load_model(model_paths[cmp_name], args)

        try:
            metrics_by_projection = compute_comparison_metrics(ref_model, cmp_model)
        finally:
            del ref_model
            del cmp_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        metrics_by_comparison[comparison_name] = metrics_by_projection
        save_metric_arrays(metrics_by_projection, comparison_name, args.output_dir)
        for metric in METRICS:
            plot_grouped_heatmap(
                metrics_by_projection=metrics_by_projection,
                comparison_name=comparison_name,
                metric=metric,
                output_dir=args.output_dir,
                clip_percentile=args.clip_percentile,
                fig_width=args.fig_width,
                fig_height=args.fig_height,
                dpi=args.dpi,
                cmap=args.cmap,
            )

    for comparison_name, minuend_name, subtrahend_name in DIFFERENCE_COMPARISONS:
        print(
            f"\n=== {comparison_name} "
            f"({minuend_name} - {subtrahend_name}) ==="
        )
        metrics_by_projection = subtract_metric_dicts(
            metrics_by_comparison[minuend_name],
            metrics_by_comparison[subtrahend_name],
        )
        save_metric_arrays(metrics_by_projection, comparison_name, args.output_dir)
        for metric in METRICS:
            plot_grouped_heatmap(
                metrics_by_projection=metrics_by_projection,
                comparison_name=comparison_name,
                metric=metric,
                output_dir=args.output_dir,
                clip_percentile=args.clip_percentile,
                fig_width=args.fig_width,
                fig_height=args.fig_height,
                dpi=args.dpi,
                cmap=args.diff_cmap,
                center_zero=True,
            )

    print("\nDone. FFN channel heatmaps and arrays saved.")


if __name__ == "__main__":
    main()
