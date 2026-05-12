#!/usr/bin/env python3
"""Autonomous local runner for E1/E2/E3 7B torchrun experiments."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results" / "FT"
LOG_ROOT = SCRIPT_DIR / "logs" / "e123_autonomous"

MODEL_ID = "allenai/OLMo-2-1124-7B"
KNOWLEDGE_PROBES_VERSION = "v13"
MCQA_PROBES_VERSION = "v14"
DEFAULT_NPROC = 8
DEFAULT_DEVICE_BATCH_SIZE = 16
DEFAULT_EFFECTIVE_BATCH_SIZE = 128
DEFAULT_CONTEXT_LENGTH = 4096
DEFAULT_LEARNING_RATE = "1e-5"
DEFAULT_ATTN_IMPLEMENTATION = "sdpa"
DEFAULT_NUM_EPOCHS = 10

SOURCES = ("arxiv", "legal", "medical")
REQUIRED_PROBE_COLUMNS = ("fact", "probe", "target")
REQUIRED_MCQA_PROBE_COLUMNS = ("formatted_question", "correct_label")


class Condition:
    def __init__(self, name: str, suffix_base: str, extra_args: list[str]) -> None:
        self.name = name
        self.suffix_base = suffix_base
        self.extra_args = extra_args


CONDITIONS = (
    Condition("E1", "E1_source_all_domains", ["--num_paraphrased_texts", "0"]),
    Condition("E2", "E2_paraphrase_all_domains", ["--num_paraphrased_texts", "9"]),
    Condition(
        "E3",
        "E3_granular_explanations_all_domains",
        [
            "--num_paraphrased_texts",
            "9",
            "--with_explanations",
            "--explanations_insertion_strategy",
            "granular",
            "--explanations_num_tracks",
            "1",
        ],
    ),
)


def now_id() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def run_cmd(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def conda_prefix(args: argparse.Namespace) -> list[str]:
    if not args.conda_env:
        return []
    if os.environ.get("CONDA_DEFAULT_ENV") == args.conda_env:
        return []
    return ["conda", "run", "--no-capture-output", "-n", args.conda_env]


def conda_command_exists(args: argparse.Namespace, name: str) -> bool:
    if not args.conda_env or os.environ.get("CONDA_DEFAULT_ENV") == args.conda_env:
        return command_exists(name)
    if not command_exists("conda"):
        return False
    proc = run_cmd(["conda", "run", "-n", args.conda_env, "which", name], PROJECT_ROOT)
    return proc.returncode == 0 and bool(proc.stdout.strip())


def python_command(args: argparse.Namespace) -> str:
    return "python" if conda_prefix(args) else sys.executable


def discover_domains(source: str) -> list[str]:
    root = PROJECT_ROOT / "data" / source / "cleaned"
    if not root.is_dir():
        return []
    domains = []
    for path in sorted(root.iterdir()):
        if path.suffix in {".txt", ".tex"}:
            domains.append(path.stem)
    return domains


def selected_pilot_domains() -> dict[str, list[str]]:
    return {source: domains[:1] for source in SOURCES if (domains := discover_domains(source))}


def count_probe_rows(path: Path, required_columns: tuple[str, ...]) -> tuple[int, list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [column for column in required_columns if column not in fieldnames]
        rows = sum(1 for _ in reader)
    return rows, missing


def preflight(args: argparse.Namespace) -> int:
    errors: list[str] = []
    warnings: list[str] = []
    manifest: dict[str, Any] = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "model_id": MODEL_ID,
        "nproc_per_node": args.nproc_per_node,
        "python": sys.executable,
        "conda_env": args.conda_env,
        "checks": {},
    }

    if args.conda_env and not command_exists("conda"):
        errors.append("conda is not in PATH.")
    if not conda_command_exists(args, "torchrun"):
        env_note = f" in conda env '{args.conda_env}'" if args.conda_env else " in PATH"
        errors.append(f"torchrun is not available{env_note}.")
    if not command_exists("nvidia-smi"):
        errors.append("nvidia-smi is not in PATH.")

    if command_exists("nvidia-smi"):
        gpu_query = run_cmd(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used",
                "--format=csv,noheader",
            ],
            PROJECT_ROOT,
        )
        gpu_lines = [line for line in gpu_query.stdout.splitlines() if line.strip()]
        manifest["checks"]["gpus"] = gpu_lines
        if len(gpu_lines) < args.nproc_per_node:
            errors.append(f"Found {len(gpu_lines)} GPUs, need {args.nproc_per_node}.")

    module_check = run_cmd(
        conda_prefix(args) + [
            python_command(args),
            "-c",
            "import torch, transformers, trl, datasets, pandas, wandb; print('ok')",
        ],
        PROJECT_ROOT,
    )
    manifest["checks"]["python_modules"] = {
        "returncode": module_check.returncode,
        "output": module_check.stdout[-4000:],
    }
    if module_check.returncode != 0:
        errors.append("Training Python environment is missing required modules.")

    replay_path = PROJECT_ROOT / "data" / "olmo" / "dclm_100M_tokens.npy"
    if replay_path.exists():
        manifest["checks"]["replay_dclm_100m_bytes"] = replay_path.stat().st_size
    else:
        errors.append(f"Missing replay token array: {replay_path}")

    domain_report: dict[str, Any] = {}
    total_probe_rows = 0
    total_mcqa_probe_rows = 0
    for source in SOURCES:
        domains = discover_domains(source)
        source_report = {
            "domains": domains,
            "probe_rows": 0,
            "mcqa_probe_rows": 0,
            "missing": [],
            "malformed": [],
        }
        for domain in domains:
            facts_dir = PROJECT_ROOT / "probes" / source / domain / "facts"
            probe_path = facts_dir / f"probes_{KNOWLEDGE_PROBES_VERSION}.csv"
            if not probe_path.exists():
                source_report["missing"].append(str(probe_path.relative_to(PROJECT_ROOT)))
            else:
                rows, missing_columns = count_probe_rows(probe_path, REQUIRED_PROBE_COLUMNS)
                source_report["probe_rows"] += rows
                total_probe_rows += rows
                if missing_columns:
                    source_report["malformed"].append(
                        {
                            "path": str(probe_path.relative_to(PROJECT_ROOT)),
                            "missing_columns": missing_columns,
                        }
                    )

            mcqa_probe_path = facts_dir / f"probes_{MCQA_PROBES_VERSION}_mcqa.csv"
            if not mcqa_probe_path.exists():
                source_report["missing"].append(str(mcqa_probe_path.relative_to(PROJECT_ROOT)))
            else:
                rows, missing_columns = count_probe_rows(
                    mcqa_probe_path,
                    REQUIRED_MCQA_PROBE_COLUMNS,
                )
                source_report["mcqa_probe_rows"] += rows
                total_mcqa_probe_rows += rows
                if missing_columns:
                    source_report["malformed"].append(
                        {
                            "path": str(mcqa_probe_path.relative_to(PROJECT_ROOT)),
                            "missing_columns": missing_columns,
                        }
                    )
        if source_report["missing"] or source_report["malformed"]:
            errors.append(
                f"{source} has missing or malformed "
                f"{KNOWLEDGE_PROBES_VERSION}/{MCQA_PROBES_VERSION} probe files."
            )
        domain_report[source] = source_report
    manifest["checks"]["domains"] = domain_report
    manifest["checks"]["total_probe_rows"] = total_probe_rows
    manifest["checks"]["total_mcqa_probe_rows"] = total_mcqa_probe_rows

    for source in SOURCES:
        for folder in ("paraphrased", "explanations"):
            path = PROJECT_ROOT / "data" / source / folder
            if not path.is_dir():
                warnings.append(f"Missing optional {source} {folder} directory: {path}")

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = LOG_ROOT / "preflight_manifest.json"
    manifest["errors"] = errors
    manifest["warnings"] = warnings
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Preflight manifest: {manifest_path}")
    for warning in warnings:
        print(f"WARNING: {warning}")
    for error in errors:
        print(f"ERROR: {error}")
    if errors:
        return 1
    print("Preflight passed.")
    return 0


def base_training_args(custom_suffix: str, num_epochs: int, args: argparse.Namespace) -> list[str]:
    return [
        "finetuning_knowledge_v9.py",
        "--custom_suffix",
        custom_suffix,
        "--model_id",
        MODEL_ID,
        "--knowledge_probes_version",
        KNOWLEDGE_PROBES_VERSION,
        "--mcqa_probes",
        "--mcqa_probes_version",
        MCQA_PROBES_VERSION,
        "--num_train_epochs",
        str(num_epochs),
        "--learning_rate",
        DEFAULT_LEARNING_RATE,
        "--device_batch_size",
        str(args.device_batch_size),
        "--effective_batch_size_for_cpt",
        str(args.effective_batch_size),
        "--context_length_for_cpt",
        str(args.context_length),
        "--fill_batches_with_pretraining",
        "--attn_implementation",
        DEFAULT_ATTN_IMPLEMENTATION,
        "--gradient_checkpointing",
        "--full_finetuning",
        "--enable_parameter_delta_tracking",
        "--no-save_local_model",
        "--no_callback_every_step",
    ]


def add_pilot_domain_overrides(cmd: list[str]) -> None:
    pilot_domains = selected_pilot_domains()
    mapping = {
        "arxiv": "--override_arxiv_domain",
        "legal": "--override_legal_domain",
        "medical": "--override_medical_domain",
    }
    for source, domains in pilot_domains.items():
        if domains:
            cmd.extend([mapping[source], *domains])


def replace_arg_value(cmd: list[str], flag: str, value: str) -> None:
    try:
        index = cmd.index(flag)
    except ValueError:
        cmd.extend([flag, value])
    else:
        cmd[index + 1] = value


def build_command(
    condition: Condition,
    stage: str,
    run_id: str,
    retry_index: int,
    args: argparse.Namespace,
) -> tuple[list[str], str]:
    suffix = f"{condition.suffix_base}_{stage}_{run_id}"
    cmd = [
        *conda_prefix(args),
        "torchrun",
        "--standalone",
        "--nproc_per_node",
        str(args.nproc_per_node),
    ]
    train_args = base_training_args(suffix, args.num_epochs, args)
    train_args.extend(condition.extra_args)

    if stage == "debug":
        train_args.extend(["--debug_dataloader_only", "--full_debug"])
    elif stage == "pilot":
        add_pilot_domain_overrides(train_args)
        if args.fast_pilot:
            if "--enable_parameter_delta_tracking" in train_args:
                train_args.remove("--enable_parameter_delta_tracking")
            if condition.name in {"E2", "E3"}:
                replace_arg_value(train_args, "--num_paraphrased_texts", str(args.pilot_num_paraphrased))

    if retry_index >= 2:
        train_args.append("--no-save_local_model")
    if retry_index >= 3:
        train_args.append("--offload_to_cpu")

    cmd.extend(train_args)
    return cmd, suffix


def detect_failure(output: str, returncode: int) -> str:
    if returncode == 0:
        return ""
    lowered = output.lower()
    if "out of memory" in lowered or "cuda oom" in lowered or "cublas_status_alloc_failed" in lowered:
        return "oom"
    if "constructed batch has" in lowered and "exceeds effective_batch_size" in lowered:
        return "data_batch_capacity"
    if "permissionerror at /vast" in lowered or "permission denied: '/vast" in lowered:
        return "cache_permission"
    if "no module named" in lowered:
        return "missing_module"
    if "address already in use" in lowered:
        return "torchrun_port"
    return "failed"


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def run_stage(args: argparse.Namespace, stage: str) -> int:
    run_id = args.run_id or now_id()
    run_log_dir = LOG_ROOT / run_id
    run_log_dir.mkdir(parents=True, exist_ok=True)
    attempts_path = run_log_dir / "attempts.jsonl"

    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    local_hf_cache = PROJECT_ROOT / ".cache" / "huggingface"
    local_hf_cache.mkdir(parents=True, exist_ok=True)
    env.setdefault("HF_HOME", str(local_hf_cache))
    env.setdefault("HF_HUB_CACHE", str(local_hf_cache / "hub"))
    env.setdefault("TRANSFORMERS_CACHE", str(local_hf_cache / "transformers"))

    stage_conditions = CONDITIONS if args.condition == "all" else tuple(
        condition for condition in CONDITIONS if condition.name == args.condition
    )
    max_retries = 3 if args.auto_retry else 0
    overall_rc = 0

    for condition in stage_conditions:
        condition_succeeded = False
        for retry_index in range(max_retries + 1):
            cmd, suffix = build_command(condition, stage, run_id, retry_index, args)
            log_path = run_log_dir / f"{condition.name}_{stage}_attempt{retry_index}.log"
            started = time.time()
            print(f"[{condition.name}] {stage} attempt {retry_index}: {' '.join(cmd)}")
            proc = subprocess.run(
                cmd,
                cwd=str(SCRIPT_DIR),
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            elapsed = time.time() - started
            log_path.write_text(proc.stdout, encoding="utf-8", errors="replace")
            failure_reason = detect_failure(proc.stdout, proc.returncode)
            record = {
                "condition": condition.name,
                "stage": stage,
                "run_id": run_id,
                "attempt": retry_index,
                "suffix": suffix,
                "command": cmd,
                "returncode": proc.returncode,
                "elapsed_seconds": round(elapsed, 3),
                "failure_reason": failure_reason,
                "log_path": str(log_path),
            }
            append_jsonl(attempts_path, record)
            if proc.returncode == 0:
                condition_succeeded = True
                break
            if failure_reason != "oom" and retry_index == 0:
                break
        if not condition_succeeded:
            overall_rc = 1
    print(f"Attempt log: {attempts_path}")
    return overall_rc


def find_experiment_dirs(suffix: str) -> list[Path]:
    if not RESULTS_ROOT.is_dir():
        return []
    matches = []
    for root, dirs, _files in os.walk(RESULTS_ROOT):
        for dirname in dirs:
            if dirname == suffix:
                matches.append(Path(root) / dirname)
    return sorted(matches)


def read_attempts(run_id: str) -> list[dict[str, Any]]:
    attempts_path = LOG_ROOT / run_id / "attempts.jsonl"
    if not attempts_path.exists():
        raise FileNotFoundError(f"No attempts file found: {attempts_path}")
    attempts = []
    with attempts_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                attempts.append(json.loads(line))
    return attempts


def csv_has_rows(path: Path) -> bool:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            return next(reader, None) is not None
    except OSError:
        return False


def validate_experiment_dir(experiment_dir: Path) -> list[str]:
    errors = []
    if not (experiment_dir / "hyperparameters.json").exists():
        errors.append("missing hyperparameters.json")

    debug_dir = experiment_dir / "debug"
    if debug_dir.is_dir():
        run1 = list(debug_dir.glob("debug_run_1*.txt"))
        run2 = list(debug_dir.glob("debug_run_2*.txt"))
        if not run1 or not run2:
            errors.append("missing debug dataloader files")
    else:
        errors.append("missing debug directory")

    metrics = list(experiment_dir.glob("*_knowledge_probe/*_knowledge_probe_metrics.csv"))
    if not metrics:
        errors.append("missing knowledge probe metrics")
    for path in metrics:
        if not csv_has_rows(path):
            errors.append(f"empty metrics csv: {path.name}")

    mcqa_metrics = list(experiment_dir.glob("*_mcqa_probe/*_mcqa_probe_metrics.csv"))
    if not mcqa_metrics:
        errors.append("missing MCQA probe metrics")
    for path in mcqa_metrics:
        if not csv_has_rows(path):
            errors.append(f"empty MCQA metrics csv: {path.name}")

    parameter_delta = experiment_dir / "parameter_delta"
    if parameter_delta.exists():
        expected = ["parameter_delta_metrics.csv"]
        for filename in expected:
            if not (parameter_delta / filename).exists():
                errors.append(f"missing parameter delta output: {filename}")
        plots_dir = parameter_delta / "plots"
        delta_metrics = (
            "relative_delta_norm",
            "cosine_distance",
            "relative_delta_gini",
            "cosine_distance_gini",
        )
        delta_plot_groups = ("mlp_embed", "attention")
        expected_plots = [
            f"{view}_{group}_{metric}.png"
            for metric in delta_metrics
            for group in delta_plot_groups
            for view in ("time", "final_layer")
        ]
        for filename in expected_plots:
            if not (plots_dir / filename).exists():
                errors.append(f"missing parameter delta plot: {filename}")
    else:
        errors.append("missing parameter_delta directory")
    return errors


def validate(args: argparse.Namespace) -> int:
    if not args.run_id:
        print("--run_id is required for validate.", file=sys.stderr)
        return 2

    attempts = read_attempts(args.run_id)
    successful = [attempt for attempt in attempts if attempt.get("returncode") == 0]
    summary: list[dict[str, Any]] = []
    overall_rc = 0

    for attempt in successful:
        suffix = attempt["suffix"]
        dirs = find_experiment_dirs(suffix)
        if not dirs:
            summary.append({"suffix": suffix, "errors": ["experiment directory not found"]})
            overall_rc = 1
            continue
        for experiment_dir in dirs:
            errors = validate_experiment_dir(experiment_dir)
            plot_cmd = [
                *conda_prefix(args),
                python_command(args),
                "regenerate_plots_v2.py",
                "--experiment_dir",
                str(experiment_dir),
                "--knowledge_probes_version",
                KNOWLEDGE_PROBES_VERSION,
            ]
            plot_proc = run_cmd(plot_cmd, SCRIPT_DIR)
            if plot_proc.returncode != 0:
                errors.append("plot regeneration failed")
            if errors:
                overall_rc = 1
            summary.append(
                {
                    "suffix": suffix,
                    "experiment_dir": str(experiment_dir),
                    "errors": errors,
                    "plot_output_tail": plot_proc.stdout[-2000:],
                }
            )

    summary_path = LOG_ROOT / args.run_id / "validation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md_path = RESULTS_ROOT / "e123_autonomous_summary.md"
    lines = [f"# E1/E2/E3 Autonomous Summary ({args.run_id})", ""]
    for item in summary:
        status = "PASS" if not item.get("errors") else "FAIL"
        lines.append(f"- {status}: {item.get('suffix')}")
        if item.get("experiment_dir"):
            lines.append(f"  - experiment_dir: {item['experiment_dir']}")
        for error in item.get("errors", []):
            lines.append(f"  - error: {error}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Validation summary: {summary_path}")
    print(f"Markdown summary: {md_path}")
    return overall_rc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["preflight", "debug", "pilot", "full", "validate"],
        required=True,
    )
    parser.add_argument("--run_id", default="", help="Stable ID used for logs and suffixes.")
    parser.add_argument("--nproc_per_node", type=int, default=DEFAULT_NPROC)
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=DEFAULT_NUM_EPOCHS,
        help=(
            "Target knowledge-injection batches when >1. Default 10 matches "
            "source-only against one source+9-paraphrase cycle."
        ),
    )
    parser.add_argument("--device_batch_size", type=int, default=DEFAULT_DEVICE_BATCH_SIZE)
    parser.add_argument("--effective_batch_size", type=int, default=DEFAULT_EFFECTIVE_BATCH_SIZE)
    parser.add_argument("--context_length", type=int, default=DEFAULT_CONTEXT_LENGTH)
    parser.add_argument("--condition", choices=["all", "E1", "E2", "E3"], default="all")
    parser.add_argument("--auto_retry", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fast_pilot", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pilot_num_paraphrased", type=int, default=1)
    parser.add_argument(
        "--conda_env",
        default=os.environ.get("E123_CONDA_ENV", "openrlhf"),
        help="Conda env used for torchrun/python subprocesses. Set empty to use current PATH.",
    )
    args = parser.parse_args()

    if args.stage == "preflight":
        return preflight(args)
    if args.stage == "validate":
        return validate(args)
    return run_stage(args, args.stage)


if __name__ == "__main__":
    raise SystemExit(main())
