import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.plotting.plot_probe_scaling_by_model import (  # noqa: E402
    METHOD_COLORS,
    METHOD_LABELS,
    iter_run_items,
)
from utils.parameter_delta_plotting import (  # noqa: E402
    DEFAULT_PLOTS_DIR,
    plot_parameter_delta_mlp_comparison,
    plot_parameter_delta_outputs,
)

DEFAULT_MODEL = "7b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate parameter-delta training plots from saved CSV outputs."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--input_dir",
        help="Directory containing parameter_delta_* CSV files.",
    )
    source.add_argument(
        "--root_dir",
        help=(
            "Base directory used only with --all_parameter_delta_dirs. By default, "
            "root mode plots the curated 7B runs used by plot_probe_scaling_by_model.py."
        ),
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "Configured model key to plot from plot_probe_scaling_by_model.py runs. "
            f"Defaults to {DEFAULT_MODEL}."
        ),
    )
    parser.add_argument(
        "--mcqa_variant",
        choices=("regular", "reviewed", "preferred"),
        default="preferred",
        help="Run variant resolver to share with plot_probe_scaling_by_model.py.",
    )
    parser.add_argument(
        "--all_parameter_delta_dirs",
        action="store_true",
        help="Ignore the curated configured runs and recursively plot every parameter_delta CSV.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help=(
            "Directory for plots. Defaults to the project-level "
            f"parameter-delta plot directory: {DEFAULT_PLOTS_DIR}."
        ),
    )
    parser.add_argument(
        "--no_clean_old_combined",
        action="store_true",
        help="Keep old combined time_<metric>.png and final_layer_<metric>.png files.",
    )
    return parser.parse_args()


def _find_parameter_delta_dirs(root_dir: Path) -> list[Path]:
    return sorted(path.parent for path in root_dir.rglob("parameter_delta_metrics.csv"))


def _configured_parameter_delta_dirs(
    model: str,
    mcqa_variant: str,
) -> list[Tuple[str, str, Path]]:
    dirs = []
    for method, run_model, run_path in iter_run_items(mcqa_variant):
        if run_model != model:
            continue
        parameter_delta_dir = Path(run_path) / "parameter_delta"
        if (parameter_delta_dir / "parameter_delta_metrics.csv").exists():
            dirs.append((method, run_model, parameter_delta_dir))
        else:
            print(f"Warning: missing parameter delta metrics for {method} {run_model}: {parameter_delta_dir}")
    return dirs


def _configured_comparison_runs(
    model: str,
    mcqa_variant: str,
) -> list[Tuple[str, str, str, str]]:
    runs = []
    for method, run_model, run_path in iter_run_items(mcqa_variant):
        if run_model != model:
            continue
        parameter_delta_dir = Path(run_path) / "parameter_delta"
        if (parameter_delta_dir / "parameter_delta_metrics.csv").exists():
            runs.append((
                method,
                METHOD_LABELS.get(method, method),
                METHOD_COLORS.get(method, "black"),
                str(parameter_delta_dir),
            ))
        else:
            print(f"Warning: missing parameter delta metrics for {method} {run_model}: {parameter_delta_dir}")
    return runs


def _plot_dirs(
    parameter_delta_dirs: Iterable[Tuple[str, str, Path]],
    output_dir: str,
    clean_old_combined: bool,
) -> int:
    total_saved = 0
    for method, model, parameter_delta_dir in parameter_delta_dirs:
        saved_paths = plot_parameter_delta_outputs(
            str(parameter_delta_dir),
            output_dir,
            clean_old_combined=clean_old_combined,
        )
        total_saved += len(saved_paths)
        plots_dir = output_dir or DEFAULT_PLOTS_DIR
        print(f"Saved {len(saved_paths)} plots for {method} {model} under {plots_dir}")
    return total_saved


def main() -> None:
    args = parse_args()
    clean_old_combined = not args.no_clean_old_combined

    if args.input_dir:
        saved_paths = plot_parameter_delta_outputs(
            args.input_dir,
            args.output_dir,
            clean_old_combined=clean_old_combined,
        )
        plots_dir = args.output_dir or DEFAULT_PLOTS_DIR
        print(f"Saved {len(saved_paths)} parameter delta plots under {plots_dir}")
        return

    if args.all_parameter_delta_dirs:
        if not args.root_dir:
            raise ValueError("--root_dir is required with --all_parameter_delta_dirs.")
        root_dir = Path(args.root_dir)
        parameter_delta_dirs = [
            ("all", "all", path)
            for path in _find_parameter_delta_dirs(root_dir)
        ]
        if not parameter_delta_dirs:
            print(f"No parameter_delta_metrics.csv files found under {root_dir}")
            return
    else:
        comparison_runs = _configured_comparison_runs(args.model, args.mcqa_variant)
        if not comparison_runs:
            raise RuntimeError(
                f"No configured parameter-delta runs found for model={args.model} "
                f"mcqa_variant={args.mcqa_variant}"
            )
        saved_paths = plot_parameter_delta_mlp_comparison(
            comparison_runs,
            args.output_dir,
            prefix=args.model,
        )
        plots_dir = args.output_dir or DEFAULT_PLOTS_DIR
        print(f"Saved {len(saved_paths)} combined MLP parameter-delta plots under {plots_dir}")
        return

    total_saved = _plot_dirs(parameter_delta_dirs, args.output_dir, clean_old_combined)

    print(
        f"Replotted {len(parameter_delta_dirs)} parameter_delta directories; "
        f"saved {total_saved} plots."
    )


if __name__ == "__main__":
    main()
