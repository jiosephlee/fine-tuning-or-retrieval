import os
import sys
import argparse
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import utils.llm_plotting_old as llm_plotting_old

def main():
    """
    Main function to regenerate plots from saved metrics.
    """
    parser = argparse.ArgumentParser(description="Regenerate plots from saved metrics.")
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory containing the metrics CSV files (e.g., raw_knowledge_probe_metrics.csv)."
    )
    args = parser.parse_args()

    # --- Basic Configuration ---
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log = logging.getLogger(__name__)

    if not os.path.isdir(args.results_dir):
        log.error(f"Error: Directory not found at '{args.results_dir}'")
        sys.exit(1)

    # --- Generate Plots ---
    log.info(f"Generating plots from data in '{args.results_dir}'")
    llm_plotting_old.generate_plots_from_files(args.results_dir, logger=log)
    log.info(f"Finished generating all plots. They are saved in '{args.results_dir}'")

if __name__ == "__main__":
    main()
