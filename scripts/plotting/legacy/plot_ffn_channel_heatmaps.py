import argparse
import json
import os
import re
import sys
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import torch
from transformers import AutoModelForCausalLM

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from utils.llm_plotting import set_plot_style


REQUIRED_MODELS = ("base", "source", "para", "aux")
PROJECTIONS = ("gate", "up", "down")
METRICS = ("relative_delta_norm", "cosine_distance")

# Comparisons for MLP heatmaps.
MLP_COMPARISONS = (
    ("source_vs_base", "base", "source"),
    ("para_vs_base", "base", "para"),
    ("aux_vs_base", "base", "aux"),
    ("aux_vs_para", "para", "aux"),
    ("aux_vs_source", "source", "aux"),
    ("para_vs_source", "para", "source"),
)

# Comparisons for embedding rasters only.
EMBED_COMPARISONS = (
    ("source_vs_base", "base", "source"),
    ("para_vs_base", "base", "para"),
    ("aux_vs_base", "base", "aux"),
)

DIFFERENCE_COMPARISONS = (
    ("aux_base_minus_source_base", "aux_vs_base", "source_vs_base"),
    ("para_base_minus_source_base", "para_vs_base", "source_vs_base"),
)

COMPARISON_TITLES = {
    "source_vs_base": "Source vs Base",
    "para_vs_base": "Para vs Base",
    "aux_vs_base": "Aux vs Base",
    "aux_vs_para": "Aux vs Para",
    "aux_vs_source": "Aux vs Source",
    "para_vs_source": "Para vs Source",
    "aux_base_minus_source_base": "Aux-Base minus Source-Base",
    "para_base_minus_source_base": "Para-Base minus Source-Base",
}

METRIC_TITLES = {
    "relative_delta_norm": "relative delta norm",
    "cosine_distance": "cosine distance",
}

EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot per-channel FFN weight change heatmaps and optional token-embedding "
            "rasters across checkpoints."
        )
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
        help="Root directory to save arrays and figures.",
    )
    parser.add_argument(
        "--mlp_subdir",
        type=str,
        default="mlp",
        help="Subdirectory under output_dir for MLP heatmaps and arrays.",
    )
    parser.add_argument(
        "--embed_subdir",
        type=str,
        default="embeddings",
        help="Subdirectory under output_dir for embedding rasters and arrays.",
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
        help="Upper percentile clip for heatmap colour range. Set <=0 to disable clipping.",
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
        help="Matplotlib colormap for unsigned heatmaps.",
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
    parser.add_argument(
        "--skip_embeddings",
        action="store_true",
        help="Skip token-embedding raster heatmaps.",
    )
    parser.add_argument(
        "--embedding_raster_cols",
        type=int,
        default=1000,
        help="Reshape vocab-sized metrics to (ceil(V/cols), cols) for 2D heatmaps.",
    )
    parser.add_argument(
        "--embed_fig_width",
        type=float,
        default=22.0,
        help="Figure width (inches) for token-embedding raster plots.",
    )
    parser.add_argument(
        "--embed_fig_height",
        type=float,
        default=10.0,
        help="Figure height (inches) for token-embedding raster plots.",
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


def resolve_embed_tokens_weight_name(model: AutoModelForCausalLM) -> str:
    preferred_suffixes = (
        "model.embed_tokens.weight",
        "transformer.wte.weight",
        "embed_tokens.weight",
        "wte.weight",
    )
    candidates: Dict[str, torch.nn.Parameter] = {
        name: param
        for name, param in model.named_parameters()
        if param.ndim == 2
    }
    for suffix in preferred_suffixes:
        for name in candidates:
            if name.endswith(suffix):
                return name
    for name in candidates:
        if "embed_tokens" in name and name.endswith(".weight"):
            return name
    for name in candidates:
        if name.endswith("wte.weight"):
            return name
    raise ValueError(
        "Could not find input token embeddings (expected *embed_tokens.weight or *wte.weight). "
        f"Sample 2D parameter names: {list(candidates)[:12]}"
    )


def compute_embedding_metrics(
    ref_model: AutoModelForCausalLM,
    cmp_model: AutoModelForCausalLM,
) -> Dict[str, np.ndarray]:
    ref_name = resolve_embed_tokens_weight_name(ref_model)
    cmp_name = resolve_embed_tokens_weight_name(cmp_model)
    ref_w = _to_float32_cpu(ref_model.get_parameter(ref_name))
    cmp_w = _to_float32_cpu(cmp_model.get_parameter(cmp_name))
    if ref_w.shape != cmp_w.shape:
        raise ValueError(
            f"Embedding shape mismatch: {ref_name} {ref_w.shape} vs "
            f"{cmp_name} {cmp_w.shape}"
        )
    rel, cos = _vector_metrics_rows(ref_w, cmp_w)
    return {
        "relative_delta_norm": rel.astype(np.float32),
        "cosine_distance": cos.astype(np.float32),
    }


def reshape_vocab_vector_to_grid(vec: np.ndarray, n_cols: int) -> np.ndarray:
    if n_cols <= 0:
        raise ValueError("embedding_raster_cols must be positive.")
    n = int(vec.size)
    n_rows = int(np.ceil(n / n_cols))
    padded_len = n_rows * n_cols
    if padded_len == n:
        return vec.reshape(n_rows, n_cols).astype(np.float32)
    out = np.full(padded_len, np.nan, dtype=np.float32)
    out[:n] = vec.astype(np.float32)
    return out.reshape(n_rows, n_cols)


def save_embedding_metric_arrays(
    embed_metrics: Dict[str, np.ndarray],
    comparison_name: str,
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for metric in METRICS:
        path = os.path.join(output_dir, f"{comparison_name}_embed_tokens_{metric}.npy")
        np.save(path, embed_metrics[metric])
        print(f"Saved array: {path}")


def subtract_embed_metric_dicts(
    minuend: Dict[str, np.ndarray],
    subtrahend: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    return {
        metric: (minuend[metric] - subtrahend[metric]).astype(np.float32)
        for metric in METRICS
    }


def plot_embedding_raster_heatmap(
    embed_metrics: Dict[str, np.ndarray],
    comparison_name: str,
    metric: str,
    output_dir: str,
    n_cols: int,
    clip_percentile: float,
    fig_width: float,
    fig_height: float,
    dpi: int,
    cmap: str,
    center_zero: bool = False,
) -> None:
    vec = embed_metrics[metric]
    grid = reshape_vocab_vector_to_grid(vec, n_cols)
    masked = ma.masked_invalid(grid)

    if center_zero:
        vmax = _compute_symmetric_vmax([vec], clip_percentile=clip_percentile)
        vmin = -vmax
    else:
        vmax = _compute_vmax([vec], clip_percentile=clip_percentile)
        vmin = 0.0

    comparison_title = _format_comparison_title(comparison_name)
    metric_title = _format_metric_title(metric)
    n_rows = grid.shape[0]

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
    image = ax.imshow(
        masked,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(
        f"{comparison_title} — embed_tokens — {metric_title}\n"
        f"(raster: token id order, {n_rows}×{n_cols}, padded cells masked)",
        fontsize=14,
    )
    ax.set_xlabel("Column (fixed width raster)")
    ax.set_ylabel("Row (token index // width)")
    cbar = fig.colorbar(image, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label(metric_title)

    os.makedirs(output_dir, exist_ok=True)
    file_name = f"{comparison_name}_embed_tokens_{metric}_raster.png"
    path = os.path.join(output_dir, file_name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(
        f"Saved embedding raster: {path} "
        f"(vmin={vmin:.6f}, vmax={vmax:.6f}, clip_percentile={clip_percentile})"
    )


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

    mlp_output_dir = os.path.join(args.output_dir, args.mlp_subdir)
    embed_output_dir = os.path.join(args.output_dir, args.embed_subdir)
    os.makedirs(mlp_output_dir, exist_ok=True)
    if not args.skip_embeddings:
        os.makedirs(embed_output_dir, exist_ok=True)

    model_paths = load_manifest(args.manifest)
    print("Resolved model checkpoints:")
    for name in REQUIRED_MODELS:
        print(f"  - {name}: {model_paths[name]}")

    metrics_by_comparison = {}
    embed_metrics_by_comparison = {}

    # MLP comparisons: includes direct aux-vs-para and aux-vs-source.
    for comparison_name, ref_name, cmp_name in MLP_COMPARISONS:
        print(f"\n=== {comparison_name} ({cmp_name} vs {ref_name}) [MLP] ===")
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
        save_metric_arrays(metrics_by_projection, comparison_name, mlp_output_dir)
        for metric in METRICS:
            plot_grouped_heatmap(
                metrics_by_projection=metrics_by_projection,
                comparison_name=comparison_name,
                metric=metric,
                output_dir=mlp_output_dir,
                clip_percentile=args.clip_percentile,
                fig_width=args.fig_width,
                fig_height=args.fig_height,
                dpi=args.dpi,
                cmap=args.cmap,
            )

    # Embedding comparisons: base-referenced only.
    if not args.skip_embeddings:
        for comparison_name, ref_name, cmp_name in EMBED_COMPARISONS:
            print(f"\n=== {comparison_name} ({cmp_name} vs {ref_name}) [embeddings] ===")
            ref_model = load_model(model_paths[ref_name], args)
            cmp_model = load_model(model_paths[cmp_name], args)

            try:
                embed_metrics = compute_embedding_metrics(ref_model, cmp_model)
            finally:
                del ref_model
                del cmp_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            embed_metrics_by_comparison[comparison_name] = embed_metrics
            save_embedding_metric_arrays(embed_metrics, comparison_name, embed_output_dir)
            for metric in METRICS:
                plot_embedding_raster_heatmap(
                    embed_metrics=embed_metrics,
                    comparison_name=comparison_name,
                    metric=metric,
                    output_dir=embed_output_dir,
                    n_cols=args.embedding_raster_cols,
                    clip_percentile=args.clip_percentile,
                    fig_width=args.embed_fig_width,
                    fig_height=args.embed_fig_height,
                    dpi=args.dpi,
                    cmap=args.cmap,
                )

    # Signed difference plots for MLP.
    for comparison_name, minuend_name, subtrahend_name in DIFFERENCE_COMPARISONS:
        print(f"\n=== {comparison_name} ({minuend_name} - {subtrahend_name}) [MLP] ===")
        metrics_by_projection = subtract_metric_dicts(
            metrics_by_comparison[minuend_name],
            metrics_by_comparison[subtrahend_name],
        )
        save_metric_arrays(metrics_by_projection, comparison_name, mlp_output_dir)
        for metric in METRICS:
            plot_grouped_heatmap(
                metrics_by_projection=metrics_by_projection,
                comparison_name=comparison_name,
                metric=metric,
                output_dir=mlp_output_dir,
                clip_percentile=args.clip_percentile,
                fig_width=args.fig_width,
                fig_height=args.fig_height,
                dpi=args.dpi,
                cmap=args.diff_cmap,
                center_zero=True,
            )

    # Signed difference plots for embeddings: only where the ingredients exist.
    if not args.skip_embeddings:
        for comparison_name, minuend_name, subtrahend_name in DIFFERENCE_COMPARISONS:
            embed_diff = subtract_embed_metric_dicts(
                embed_metrics_by_comparison[minuend_name],
                embed_metrics_by_comparison[subtrahend_name],
            )
            save_embedding_metric_arrays(embed_diff, comparison_name, embed_output_dir)
            for metric in METRICS:
                plot_embedding_raster_heatmap(
                    embed_metrics=embed_diff,
                    comparison_name=comparison_name,
                    metric=metric,
                    output_dir=embed_output_dir,
                    n_cols=args.embedding_raster_cols,
                    clip_percentile=args.clip_percentile,
                    fig_width=args.embed_fig_width,
                    fig_height=args.embed_fig_height,
                    dpi=args.dpi,
                    cmap=args.diff_cmap,
                    center_zero=True,
                )

    print("\nDone. MLP heatmaps saved under:")
    print(f"  {mlp_output_dir}")
    if not args.skip_embeddings:
        print("Embedding rasters saved under:")
        print(f"  {embed_output_dir}")


if __name__ == "__main__":
    main()