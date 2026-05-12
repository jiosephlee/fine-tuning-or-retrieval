import argparse
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from utils.parameter_delta_plotting import plot_parameter_delta_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate parameter-delta training plots from saved CSV outputs."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input_dir",
        help="Directory containing parameter_delta_* CSV files.",
    )
    source.add_argument(
        "--root_dir",
        help=(
            "Recursively find and replot all directories containing "
            "parameter_delta_metrics.csv."
        ),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help=(
            "Directory for plots in --input_dir mode. Defaults to <input_dir>/plots. "
            "Not supported with --root_dir."
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


def main() -> None:
    args = parse_args()
    clean_old_combined = not args.no_clean_old_combined

    if args.input_dir:
        saved_paths = plot_parameter_delta_outputs(
            args.input_dir,
            args.output_dir,
            clean_old_combined=clean_old_combined,
        )
        plots_dir = args.output_dir or os.path.join(args.input_dir, "plots")
        print(f"Saved {len(saved_paths)} parameter delta plots under {plots_dir}")
        return

    if args.output_dir:
        raise ValueError("--output_dir is only supported with --input_dir.")

    root_dir = Path(args.root_dir)
    parameter_delta_dirs = _find_parameter_delta_dirs(root_dir)
    if not parameter_delta_dirs:
        print(f"No parameter_delta_metrics.csv files found under {root_dir}")
        return

    total_saved = 0
    for parameter_delta_dir in parameter_delta_dirs:
        saved_paths = plot_parameter_delta_outputs(
            str(parameter_delta_dir),
            clean_old_combined=clean_old_combined,
        )
        total_saved += len(saved_paths)
        print(f"Saved {len(saved_paths)} plots under {parameter_delta_dir / 'plots'}")

    print(
        f"Replotted {len(parameter_delta_dirs)} parameter_delta directories; "
        f"saved {total_saved} plots."
    )


if __name__ == "__main__":
    main()
