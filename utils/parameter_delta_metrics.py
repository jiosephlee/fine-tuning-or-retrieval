import math
import os
import re
import shutil
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


PROJECTIONS = ("gate", "up", "down")
METRICS = ("relative_delta_norm", "cosine_distance")
EPS = 1e-12


@dataclass(frozen=True)
class TrackedTensor:
    component: str
    projection: str
    layer: Optional[int]
    name: str
    orientation: str
    module: torch.nn.Module


@dataclass
class StepMetricResult:
    scalar_rows: List[Dict[str, object]]
    layer_rows: List[Dict[str, object]]
    concentration_rows: List[Dict[str, object]]


def extract_layer_index(name: str) -> int:
    matches = re.findall(r"\.(\d+)\.", name)
    if not matches:
        raise ValueError(f"Could not parse layer index from module name: {name}")
    return int(matches[-1])


def materialize_module_weight(module: torch.nn.Module) -> torch.Tensor:
    """Return the effective weight for regular Linear/Embedding and PEFT LoRA modules."""
    if hasattr(module, "base_layer") and hasattr(module, "get_delta_weight"):
        base = materialize_module_weight(module.base_layer)
        if getattr(module, "disable_adapters", False):
            return base

        active_adapters = getattr(module, "active_adapters", None)
        if active_adapters is None:
            active_adapter = getattr(module, "active_adapter", None)
            active_adapters = [active_adapter] if active_adapter else []
        if isinstance(active_adapters, str):
            active_adapters = [active_adapters]

        weight = base
        for adapter_name in active_adapters:
            if adapter_name is None:
                continue
            try:
                delta = module.get_delta_weight(adapter_name)
            except Exception:
                continue
            weight = weight + delta.to(device=weight.device, dtype=weight.dtype)
        return weight

    if hasattr(module, "modules_to_save") and hasattr(module, "active_adapter"):
        adapter_name = getattr(module, "active_adapter", None)
        modules_to_save = getattr(module, "modules_to_save", {})
        if adapter_name in modules_to_save and hasattr(modules_to_save[adapter_name], "weight"):
            return modules_to_save[adapter_name].weight.detach()

    if hasattr(module, "weight"):
        weight = getattr(module, "weight")
        if isinstance(weight, torch.Tensor):
            return weight.detach()

    if hasattr(module, "original_module") and hasattr(module.original_module, "weight"):
        return module.original_module.weight.detach()

    raise ValueError(f"Could not resolve a weight tensor from module {module.__class__.__name__}")


def to_float32_cpu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(dtype=torch.float32, device="cpu")


def discover_mlp_tensors(model: torch.nn.Module) -> List[TrackedTensor]:
    entries: List[TrackedTensor] = []
    seen = set()
    for name, module in model.named_modules():
        projection = None
        if name.endswith(".mlp.gate_proj") or name.endswith("mlp.gate_proj"):
            projection = "gate"
            orientation = "rows"
        elif name.endswith(".mlp.up_proj") or name.endswith("mlp.up_proj"):
            projection = "up"
            orientation = "rows"
        elif name.endswith(".mlp.down_proj") or name.endswith("mlp.down_proj"):
            projection = "down"
            orientation = "cols"
        else:
            continue

        key = (projection, extract_layer_index(name))
        if key in seen:
            continue
        seen.add(key)
        entries.append(
            TrackedTensor(
                component="mlp",
                projection=projection,
                layer=key[1],
                name=name,
                orientation=orientation,
                module=module,
            )
        )

    if not entries:
        raise ValueError(
            "Did not find MLP projection modules ending with "
            "mlp.(gate_proj|up_proj|down_proj)."
        )

    layers_by_projection: Dict[str, List[int]] = {projection: [] for projection in PROJECTIONS}
    for entry in entries:
        layers_by_projection[entry.projection].append(int(entry.layer))
    for projection, layers in layers_by_projection.items():
        if not layers:
            raise ValueError(f"Missing MLP projection modules for {projection}.")
        ordered = sorted(layers)
        expected = list(range(ordered[-1] + 1))
        if ordered != expected:
            raise ValueError(
                f"Layer indices for {projection} are not contiguous from 0..n_layers-1: "
                f"{ordered[:8]} ... {ordered[-8:]}"
            )

    return sorted(entries, key=lambda item: (item.projection, item.layer or 0))


def discover_embedding_tensor(model: torch.nn.Module) -> TrackedTensor:
    module = model.get_input_embeddings()
    if module is None:
        raise ValueError("Model did not return input embeddings from get_input_embeddings().")
    return TrackedTensor(
        component="embed_tokens",
        projection="embed_tokens",
        layer=None,
        name="embed_tokens",
        orientation="rows",
        module=module,
    )


def snapshot_baseline(entries: Sequence[TrackedTensor]) -> Dict[str, torch.Tensor]:
    baseline: Dict[str, torch.Tensor] = {}
    for entry in entries:
        baseline[entry_key(entry)] = to_float32_cpu(materialize_module_weight(entry.module))
    return baseline


def entry_key(entry: TrackedTensor) -> str:
    if entry.layer is None:
        return entry.projection
    return f"{entry.projection}_layer_{entry.layer:03d}"


def vector_metrics(ref: torch.Tensor, cmp: torch.Tensor, orientation: str) -> Dict[str, np.ndarray]:
    if ref.shape != cmp.shape:
        raise ValueError(f"Shape mismatch: {tuple(ref.shape)} vs {tuple(cmp.shape)}")

    dim = 1 if orientation == "rows" else 0
    diff = cmp - ref
    ref_norm = torch.linalg.norm(ref, dim=dim)
    cmp_norm = torch.linalg.norm(cmp, dim=dim)
    diff_norm = torch.linalg.norm(diff, dim=dim)
    dot = torch.sum(ref * cmp, dim=dim)

    relative_delta = diff_norm / (ref_norm + EPS)
    denom = ref_norm * cmp_norm
    both_zero = (ref_norm <= EPS) & (cmp_norm <= EPS)
    cosine_similarity = torch.where(
        denom > EPS,
        dot / denom,
        torch.zeros_like(dot),
    )
    cosine_distance = 1.0 - cosine_similarity
    cosine_distance = torch.where(both_zero, torch.zeros_like(cosine_distance), cosine_distance)
    cosine_distance = torch.clamp(cosine_distance, min=0.0)
    return {
        "relative_delta_norm": relative_delta.cpu().numpy().astype(np.float32),
        "cosine_distance": cosine_distance.cpu().numpy().astype(np.float32),
    }


def channel_delta_tensor(ref: torch.Tensor, cmp: torch.Tensor) -> np.ndarray:
    return (cmp - ref).cpu().numpy().astype(np.float32)


def flatten_finite(arrays: Iterable[np.ndarray]) -> np.ndarray:
    values = np.concatenate([np.asarray(array).reshape(-1) for array in arrays]).astype(np.float64)
    return values[np.isfinite(values)]


def summarize_values(values: np.ndarray) -> Dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"mean": np.nan, "median": np.nan, "max": np.nan, "num_values": 0}
    return {
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "max": float(np.max(finite)),
        "num_values": int(finite.size),
    }


def top_share(values: np.ndarray, fraction: float) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.nan
    finite = np.abs(finite)
    total = float(np.sum(finite))
    if total <= EPS:
        return 0.0
    k = max(1, int(math.ceil(finite.size * fraction)))
    top = np.partition(finite, -k)[-k:]
    return float(np.sum(top) / total)


def gini(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.nan
    finite = np.sort(np.abs(finite))
    total = float(np.sum(finite))
    if total <= EPS:
        return 0.0
    n = finite.size
    index = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(index * finite) / (n * total)) - ((n + 1.0) / n))


def concentration_summary(values: np.ndarray) -> Dict[str, float]:
    return {
        "top_1pct_share": top_share(values, 0.01),
        "top_5pct_share": top_share(values, 0.05),
        "gini": gini(values),
    }


def compute_step_metrics(
    *,
    step: int,
    entries: Sequence[TrackedTensor],
    baseline: Dict[str, torch.Tensor],
    raw_delta_dir: Optional[str] = None,
) -> StepMetricResult:
    by_projection: Dict[str, Dict[str, List[np.ndarray]]] = {
        projection: {metric: [] for metric in METRICS}
        for projection in PROJECTIONS
    }
    by_component: Dict[str, Dict[str, List[np.ndarray]]] = {
        "mlp_all": {metric: [] for metric in METRICS},
        "embed_tokens": {metric: [] for metric in METRICS},
    }
    scalar_rows: List[Dict[str, object]] = []
    layer_rows: List[Dict[str, object]] = []
    concentration_rows: List[Dict[str, object]] = []

    step_raw_dir = None
    if raw_delta_dir:
        step_raw_dir = os.path.join(raw_delta_dir, f"step_{step:08d}")
        os.makedirs(step_raw_dir, exist_ok=True)

    for entry in entries:
        key = entry_key(entry)
        ref = baseline[key]
        cmp = to_float32_cpu(materialize_module_weight(entry.module))
        metrics = vector_metrics(ref, cmp, entry.orientation)

        if step_raw_dir is not None:
            np.savez_compressed(
                os.path.join(step_raw_dir, f"{key}.npz"),
                delta=channel_delta_tensor(ref, cmp),
                projection=entry.projection,
                layer=-1 if entry.layer is None else int(entry.layer),
                orientation=entry.orientation,
            )

        for metric, values in metrics.items():
            if entry.component == "mlp":
                by_projection[entry.projection][metric].append(values)
                by_component["mlp_all"][metric].append(values)
                row = {
                    "step": int(step),
                    "metric": metric,
                    "projection": entry.projection,
                    "layer": int(entry.layer),
                }
                row.update(summarize_values(values))
                layer_rows.append(row)
            else:
                by_component["embed_tokens"][metric].append(values)

    for projection in PROJECTIONS:
        for metric in METRICS:
            values = flatten_finite(by_projection[projection][metric])
            row = {
                "step": int(step),
                "metric": metric,
                "component": projection,
            }
            row.update(summarize_values(values))
            scalar_rows.append(row)

    for component in ("mlp_all", "embed_tokens"):
        for metric in METRICS:
            arrays = by_component[component][metric]
            if not arrays:
                continue
            values = flatten_finite(arrays)
            scalar_row = {
                "step": int(step),
                "metric": metric,
                "component": component,
            }
            scalar_row.update(summarize_values(values))
            scalar_rows.append(scalar_row)

            conc_row = {
                "step": int(step),
                "metric": metric,
                "component": component,
            }
            conc_row.update(concentration_summary(values))
            concentration_rows.append(conc_row)

    return StepMetricResult(
        scalar_rows=scalar_rows,
        layer_rows=layer_rows,
        concentration_rows=concentration_rows,
    )


def save_metric_csvs(
    output_dir: str,
    scalar_rows: Sequence[Dict[str, object]],
    layer_rows: Sequence[Dict[str, object]],
    concentration_rows: Sequence[Dict[str, object]],
) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    paths = {
        "scalar": os.path.join(output_dir, "parameter_delta_scalar_metrics.csv"),
        "layer": os.path.join(output_dir, "parameter_delta_layer_metrics.csv"),
        "concentration": os.path.join(output_dir, "parameter_delta_concentration_metrics.csv"),
    }
    pd.DataFrame(scalar_rows).to_csv(paths["scalar"], index=False)
    pd.DataFrame(layer_rows).to_csv(paths["layer"], index=False)
    pd.DataFrame(concentration_rows).to_csv(paths["concentration"], index=False)
    return paths


def raw_step_dirs(raw_delta_dir: str) -> List[Tuple[int, str]]:
    if not raw_delta_dir or not os.path.isdir(raw_delta_dir):
        return []
    dirs = []
    for name in os.listdir(raw_delta_dir):
        if not name.startswith("step_"):
            continue
        try:
            step = int(name.split("_", 1)[1])
        except ValueError:
            continue
        path = os.path.join(raw_delta_dir, name)
        if os.path.isdir(path):
            dirs.append((step, path))
    return sorted(dirs)


def cosine_alignment(delta: np.ndarray, final_delta: np.ndarray) -> float:
    x = delta.astype(np.float64, copy=False).reshape(-1)
    y = final_delta.astype(np.float64, copy=False).reshape(-1)
    denom = math.sqrt(float(np.dot(x, x))) * math.sqrt(float(np.dot(y, y)))
    if denom <= EPS:
        return np.nan
    return float(np.dot(x, y) / denom)


def compute_final_alignment(raw_delta_dir: str, output_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dirs = raw_step_dirs(raw_delta_dir)
    if len(dirs) < 2:
        return pd.DataFrame(), pd.DataFrame()

    final_step, final_dir = dirs[-1]
    final_files = {
        name: os.path.join(final_dir, name)
        for name in os.listdir(final_dir)
        if name.endswith(".npz")
    }
    aggregate_rows: List[Dict[str, object]] = []
    layer_rows: List[Dict[str, object]] = []

    for step, step_dir in dirs:
        accum: Dict[str, Dict[str, float]] = {}
        for name, final_path in final_files.items():
            path = os.path.join(step_dir, name)
            if not os.path.exists(path):
                continue
            with np.load(path) as current_data, np.load(final_path) as final_data:
                delta = current_data["delta"].astype(np.float64, copy=False)
                final_delta = final_data["delta"].astype(np.float64, copy=False)
                projection = str(current_data["projection"])
                layer = int(current_data["layer"])
                component = "embed_tokens" if projection == "embed_tokens" else projection
                alignment = cosine_alignment(delta, final_delta)

                if projection != "embed_tokens":
                    layer_rows.append(
                        {
                            "step": int(step),
                            "final_step": int(final_step),
                            "projection": projection,
                            "layer": layer,
                            "alignment": alignment,
                        }
                    )

                agg_components = [component]
                if projection != "embed_tokens":
                    agg_components.append("mlp_all")
                for agg_component in agg_components:
                    bucket = accum.setdefault(
                        agg_component,
                        {"dot": 0.0, "norm": 0.0, "final_norm": 0.0},
                    )
                    d = delta.reshape(-1)
                    f = final_delta.reshape(-1)
                    bucket["dot"] += float(np.dot(d, f))
                    bucket["norm"] += float(np.dot(d, d))
                    bucket["final_norm"] += float(np.dot(f, f))

        for component, bucket in accum.items():
            denom = math.sqrt(bucket["norm"]) * math.sqrt(bucket["final_norm"])
            alignment = np.nan if denom <= EPS else float(bucket["dot"] / denom)
            aggregate_rows.append(
                {
                    "step": int(step),
                    "final_step": int(final_step),
                    "component": component,
                    "alignment": alignment,
                }
            )

    os.makedirs(output_dir, exist_ok=True)
    aggregate_df = pd.DataFrame(aggregate_rows)
    layer_df = pd.DataFrame(layer_rows)
    aggregate_df.to_csv(os.path.join(output_dir, "parameter_delta_final_alignment_scalar.csv"), index=False)
    layer_df.to_csv(os.path.join(output_dir, "parameter_delta_final_alignment_layer.csv"), index=False)
    return aggregate_df, layer_df


def cleanup_raw_delta_dir(raw_delta_dir: str) -> None:
    if raw_delta_dir and os.path.isdir(raw_delta_dir):
        shutil.rmtree(raw_delta_dir)
