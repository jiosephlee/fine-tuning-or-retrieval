import math
import os
import re
import shutil
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


COMPONENTS = (
    "embed_tokens",
    "gate_proj",
    "up_proj",
    "down_proj",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
)
LAYER_COMPONENTS = tuple(component for component in COMPONENTS if component != "embed_tokens")
BASE_METRICS = ("relative_delta_norm", "cosine_distance")
METRICS = (
    "relative_delta_norm",
    "cosine_distance",
    "relative_delta_gini",
    "cosine_distance_gini",
)
COMPONENT_ORIENTATIONS = {
    "embed_tokens": "rows",
    "gate_proj": "rows",
    "up_proj": "rows",
    "down_proj": "cols",
    "q_proj": "rows",
    "k_proj": "rows",
    "v_proj": "rows",
    "o_proj": "cols",
}
EPS = 1e-12
GINI_MEAN_ABS_EPS = 1e-6


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
    time_rows: List[Dict[str, object]]
    layer_rows: List[Dict[str, object]]


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


def discover_parameter_delta_tensors(
    model: torch.nn.Module,
    include_embeddings: bool = True,
) -> List[TrackedTensor]:
    entries: List[TrackedTensor] = []
    seen = set()
    for name, module in model.named_modules():
        component = None
        for candidate in LAYER_COMPONENTS:
            if name.endswith(f".{candidate}") or name == candidate:
                component = candidate
                break
        if component is None:
            continue

        key = (component, extract_layer_index(name))
        if key in seen:
            continue
        seen.add(key)
        entries.append(
            TrackedTensor(
                component=component,
                projection=component,
                layer=key[1],
                name=name,
                orientation=COMPONENT_ORIENTATIONS[component],
                module=module,
            )
        )

    if not entries:
        raise ValueError(
            "Did not find tracked projection modules ending with "
            "(gate_proj|up_proj|down_proj|q_proj|k_proj|v_proj|o_proj)."
        )

    layers_by_component: Dict[str, List[int]] = {component: [] for component in LAYER_COMPONENTS}
    for entry in entries:
        layers_by_component[entry.component].append(int(entry.layer))
    for component, layers in layers_by_component.items():
        if not layers:
            raise ValueError(f"Missing tracked projection modules for {component}.")
        ordered = sorted(layers)
        expected = list(range(ordered[-1] + 1))
        if ordered != expected:
            raise ValueError(
                f"Layer indices for {component} are not contiguous from 0..n_layers-1: "
                f"{ordered[:8]} ... {ordered[-8:]}"
            )

    if include_embeddings:
        entries.append(discover_embedding_tensor(model))

    return sorted(
        entries,
        key=lambda item: (COMPONENTS.index(item.component), -1 if item.layer is None else item.layer),
    )


def discover_mlp_tensors(model: torch.nn.Module) -> List[TrackedTensor]:
    components = ("gate_proj", "up_proj", "down_proj")
    entries: List[TrackedTensor] = []
    seen = set()
    for name, module in model.named_modules():
        component = None
        for candidate in components:
            if name.endswith(f".{candidate}") or name == candidate:
                component = candidate
                break
        if component is None:
            continue
        key = (component, extract_layer_index(name))
        if key in seen:
            continue
        seen.add(key)
        entries.append(
            TrackedTensor(
                component=component,
                projection=component,
                layer=key[1],
                name=name,
                orientation=COMPONENT_ORIENTATIONS[component],
                module=module,
            )
        )
    if not entries:
        raise ValueError(
            "Did not find MLP projection modules ending with "
            "mlp.(gate_proj|up_proj|down_proj)."
        )
    return sorted(entries, key=lambda item: (components.index(item.component), item.layer or 0))


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


def gini(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.nan
    finite = np.sort(np.abs(finite))
    total = float(np.sum(finite))
    if total <= EPS or (total / finite.size) <= GINI_MEAN_ABS_EPS:
        return 0.0
    n = finite.size
    index = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(index * finite) / (n * total)) - ((n + 1.0) / n))


def metric_rows_from_values(
    *,
    view: str,
    step: int,
    component: str,
    values_by_metric: Dict[str, np.ndarray],
    layer: Optional[int] = None,
    final_step: Optional[int] = None,
) -> List[Dict[str, object]]:
    rows = []
    relative_values = flatten_finite([values_by_metric["relative_delta_norm"]])
    cosine_values = flatten_finite([values_by_metric["cosine_distance"]])
    metric_values = {
        "relative_delta_norm": float(np.mean(relative_values)) if relative_values.size else np.nan,
        "cosine_distance": float(np.mean(cosine_values)) if cosine_values.size else np.nan,
        "relative_delta_gini": gini(relative_values),
        "cosine_distance_gini": gini(cosine_values),
    }
    num_values = {
        "relative_delta_norm": int(relative_values.size),
        "cosine_distance": int(cosine_values.size),
        "relative_delta_gini": int(relative_values.size),
        "cosine_distance_gini": int(cosine_values.size),
    }
    for metric in METRICS:
        rows.append(
            {
                "view": view,
                "step": int(step),
                "final_step": "" if final_step is None else int(final_step),
                "layer": "" if layer is None else int(layer),
                "component": component,
                "metric": metric,
                "value": metric_values[metric],
                "num_values": num_values[metric],
            }
        )
    return rows


def compute_step_metrics(
    *,
    step: int,
    entries: Sequence[TrackedTensor],
    baseline: Dict[str, torch.Tensor],
    raw_delta_dir: Optional[str] = None,
) -> StepMetricResult:
    by_component: Dict[str, Dict[str, List[np.ndarray]]] = {
        component: {metric: [] for metric in BASE_METRICS}
        for component in COMPONENTS
    }
    time_rows: List[Dict[str, object]] = []
    layer_rows: List[Dict[str, object]] = []

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
                projection=entry.component,
                layer=-1 if entry.layer is None else int(entry.layer),
                orientation=entry.orientation,
            )

        for metric, values in metrics.items():
            by_component[entry.component][metric].append(values)

        if entry.layer is not None:
            layer_rows.extend(
                metric_rows_from_values(
                    view="layer_step",
                    step=step,
                    component=entry.component,
                    values_by_metric=metrics,
                    layer=int(entry.layer),
                )
            )

    for component in COMPONENTS:
        if not by_component[component]["relative_delta_norm"]:
            continue
        values_by_metric = {
            metric: flatten_finite(by_component[component][metric])
            for metric in BASE_METRICS
        }
        time_rows.extend(
            metric_rows_from_values(
                view="time",
                step=step,
                component=component,
                values_by_metric=values_by_metric,
            )
        )

    return StepMetricResult(
        time_rows=time_rows,
        layer_rows=layer_rows,
    )


def save_metric_csvs(
    output_dir: str,
    time_rows: Sequence[Dict[str, object]],
    layer_rows: Sequence[Dict[str, object]],
) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "parameter_delta_metrics.csv")
    final_step = None
    if time_rows:
        final_step = max(int(row["step"]) for row in time_rows)
    elif layer_rows:
        final_step = max(int(row["step"]) for row in layer_rows)

    output_rows = []
    for row in time_rows:
        output_row = dict(row)
        output_row["final_step"] = "" if final_step is None else int(final_step)
        output_rows.append(output_row)
    if final_step is not None:
        for row in layer_rows:
            if int(row["step"]) != int(final_step):
                continue
            output_row = dict(row)
            output_row["view"] = "final_layer"
            output_row["final_step"] = int(final_step)
            output_rows.append(output_row)

    columns = ["view", "step", "final_step", "layer", "component", "metric", "value", "num_values"]
    pd.DataFrame(output_rows, columns=columns).to_csv(path, index=False)
    paths = {"metrics": path}
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

                bucket = accum.setdefault(
                    component,
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
