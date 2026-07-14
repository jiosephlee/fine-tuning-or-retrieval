"""Extract probe-metric traces from OLMo run wandb datastores.

For each run dir under /data1/joseph/olmo-runs, find every wandb run-* /
offline-run-* datastore, scan history records, and collect rows that contain
probe metrics (average/* keys or per-domain probe keys). Emit a JSON with
first/last values per key of interest, plus the full 'average/*' trace.
"""
import glob
import json
import os
import sys

from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore

RUNS_ROOT = "/data1/joseph/olmo-runs"
OUT = sys.argv[1] if len(sys.argv) > 1 else "olmo_wandb_metrics.json"

AVG_KEYS = [
    "average/log_prob_average",
    "average/mcqa_accuracy_average",
    "average/inference_log_prob_average",
    "average/inference_mcqa_accuracy_average",
    "average/target_rank_average",
    "average/inference_target_rank_average",
    "average/paraphrased_log_prob_average",
]


def parse_datastore(path):
    """Return list of history dicts from a .wandb file."""
    ds = DataStore()
    ds.open_for_scan(path)
    rows = []
    while True:
        try:
            data = ds.scan_data()
        except Exception:
            break
        if data is None:
            break
        rec = wandb_internal_pb2.Record()
        try:
            rec.ParseFromString(data)
        except Exception:
            continue
        if rec.WhichOneof("record_type") != "history":
            continue
        row = {}
        for item in rec.history.item:
            key = item.key or ("/".join(item.nested_key) if item.nested_key else "")
            if not key:
                continue
            try:
                row[key] = json.loads(item.value_json)
            except Exception:
                pass
        if row:
            rows.append(row)
    return rows


def legacy_probe_averages(row):
    """Average eval/probes/probe_{cloze,mcqa}_<domain>_<metric> keys (old inject runs)."""
    out = {}
    for prefix, metric, name in (
        ("eval/probes/probe_cloze_", "_log_prob", "log_prob"),
        ("eval/probes/probe_mcqa_", "_mcqa_accuracy", "mcqa_accuracy"),
        ("eval/probes/probe_inference_cloze_", "_log_prob", "inference_log_prob"),
        ("eval/probes/probe_inference_mcqa_", "_mcqa_accuracy", "inference_mcqa_accuracy"),
    ):
        vals = [
            v
            for k, v in row.items()
            if k.startswith(prefix) and k.endswith(metric) and isinstance(v, (int, float))
        ]
        if vals:
            out[f"average/{name}_average"] = sum(vals) / len(vals)
            out[f"n_{name}"] = len(vals)
    return out


def domain_group_averages(row):
    """Compute per-group (arxiv/legal/medical) means of per-domain probe keys."""
    out = {}
    for group in ("arxiv", "legal", "medical"):
        for suffix, name in (
            ("_log_prob", "log_prob"),
            ("_mcqa_accuracy", "mcqa_accuracy"),
            ("_inference_log_prob", "inference_log_prob"),
            ("_inference_mcqa_accuracy", "inference_mcqa_accuracy"),
        ):
            vals = [
                v
                for k, v in row.items()
                if k.startswith(group + "/")
                and k.endswith(suffix)
                # exclude e.g. *_inference_log_prob matching plain _log_prob
                and (suffix != "_log_prob" or not k.endswith("_inference_log_prob"))
                and (
                    suffix != "_mcqa_accuracy"
                    or not k.endswith("_inference_mcqa_accuracy")
                )
                and not k.endswith("_paraphrased_log_prob")
                and isinstance(v, (int, float))
            ]
            if vals:
                out[f"{group}/{name}"] = sum(vals) / len(vals)
    return out


results = {}
for run_dir in sorted(glob.glob(os.path.join(RUNS_ROOT, "*", "wandb", "wandb", "*run-*"))):
    if os.path.islink(run_dir):
        continue
    wandb_files = glob.glob(os.path.join(run_dir, "*.wandb"))
    if not wandb_files:
        continue
    run_name = run_dir.split("/")[4]
    rows = parse_datastore(wandb_files[0])
    probe_rows = [r for r in rows if any(k in r for k in AVG_KEYS)]
    if not probe_rows:
        # fall back: rows with per-domain probe keys but no averages
        probe_rows = [r for r in rows if any("_mcqa_accuracy" in k for k in r)]
    if not probe_rows:
        print(f"{run_name} [{os.path.basename(run_dir)}]: no probe rows ({len(rows)} history rows)")
        continue
    steps = [r.get("_step") for r in probe_rows]
    entry = {
        "wandb_run": os.path.basename(run_dir),
        "n_probe_rows": len(probe_rows),
        "first_step": steps[0],
        "last_step": steps[-1],
        "first": {},
        "last": {},
        "trace": [],
    }
    for r in probe_rows:
        avg = {k: r[k] for k in AVG_KEYS if k in r}
        if not avg:
            avg = legacy_probe_averages(r)
        if not avg:
            avg = domain_group_averages(r)
            # global average across the three groups
            for name in ("log_prob", "mcqa_accuracy", "inference_log_prob", "inference_mcqa_accuracy"):
                vals = [avg[f"{g}/{name}"] for g in ("arxiv", "legal", "medical") if f"{g}/{name}" in avg]
                if vals:
                    avg[f"average/{name}_average"] = sum(vals) / len(vals)
        avg["_step"] = r.get("_step")
        entry["trace"].append(avg)
    entry["first"] = entry["trace"][0]
    entry["last"] = entry["trace"][-1]
    # also keep per-group averages at first/last for the completed runs
    entry["first_groups"] = domain_group_averages(probe_rows[0])
    entry["last_groups"] = domain_group_averages(probe_rows[-1])
    results.setdefault(run_name, []).append(entry)
    print(f"{run_name} [{os.path.basename(run_dir)}]: {len(probe_rows)} probe rows, steps {steps[0]}..{steps[-1]}")

with open(OUT, "w") as f:
    json.dump(results, f, indent=1)
print("wrote", OUT)
