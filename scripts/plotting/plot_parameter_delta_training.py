import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from utils.parameter_delta_plotting import plot_parameter_delta_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate parameter-delta training plots from saved CSV outputs."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing parameter_delta_* CSV files.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory for plots. Defaults to <input_dir>/plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_parameter_delta_outputs(args.input_dir, args.output_dir)
    print(f"Saved parameter delta plots under {args.output_dir or os.path.join(args.input_dir, 'plots')}")


if __name__ == "__main__":
    main()
