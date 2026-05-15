import os
import sys
import argparse
import logging
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import utils.llm_plotting as llm_plotting

KNOWLEDGE_PROBE_VARIANT_SUFFIXES = {
    "standard": "",
    "low_overlap_strict": "_low_overlap_strict",
}

def main():
    parser = argparse.ArgumentParser(description="Regenerate plots for a specific experiment run.")
    parser.add_argument("--experiment_dir", type=str, required=True, help="The full path to the experiment directory.")
    parser.add_argument("--knowledge_probes_version", type=str, default=None, help="Version of the knowledge probes.")
    parser.add_argument(
        "--knowledge_probe_variant",
        type=str,
        choices=sorted(KNOWLEDGE_PROBE_VARIANT_SUFFIXES),
        default=None,
        help="Factual knowledge probe family used for plot metadata.",
    )
    parser.add_argument(
        "--knowledge_probe_filename_suffix",
        type=str,
        default=None,
        help="Advanced: suffix inserted before .csv for factual knowledge probe metadata.",
    )
    parser.add_argument(
        "--use_low_overlap_knowledge_probes",
        action="store_true",
        help="Alias for --knowledge_probe_variant low_overlap_strict.",
    )
    parser.add_argument("--inference_probes_version", type=str, default="v7", help="Version of the inference probes.")
    
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    if not os.path.isdir(args.experiment_dir):
        logger.error(f"Experiment directory not found: {args.experiment_dir}")
        return

    hyperparameters = {}
    hyperparameters_path = os.path.join(args.experiment_dir, "hyperparameters.json")
    if os.path.exists(hyperparameters_path):
        with open(hyperparameters_path, "r", encoding="utf-8") as handle:
            hyperparameters = json.load(handle)

    if args.use_low_overlap_knowledge_probes:
        args.knowledge_probe_variant = "low_overlap_strict"

    knowledge_probe_variant = (
        args.knowledge_probe_variant
        or hyperparameters.get("knowledge_probe_variant")
        or "standard"
    )
    knowledge_probe_filename_suffix = (
        args.knowledge_probe_filename_suffix
        if args.knowledge_probe_filename_suffix is not None
        else hyperparameters.get("knowledge_probe_filename_suffix")
    )
    if knowledge_probe_filename_suffix is None:
        knowledge_probe_filename_suffix = KNOWLEDGE_PROBE_VARIANT_SUFFIXES[knowledge_probe_variant]

    knowledge_probes_version = (
        args.knowledge_probes_version
        or hyperparameters.get("knowledge_probes_version")
        or ("v14" if knowledge_probe_variant == "low_overlap_strict" else "v13")
    )

    logger.info(f"Scanning for probe results in: {args.experiment_dir}")
    logger.info(
        "Using factual knowledge probe metadata: "
        f"variant={knowledge_probe_variant}, "
        f"probes_{knowledge_probes_version}{knowledge_probe_filename_suffix}.csv"
    )

    domains = set()
    for subdir in os.listdir(args.experiment_dir):
        if subdir.endswith("_knowledge_probe"):
            domains.add(subdir.replace("_knowledge_probe", ""))
        elif subdir.endswith("_inference_probe"):
            domains.add(subdir.replace("_inference_probe", ""))

    for domain in sorted(list(domains)):
        logger.info(f"Found domain '{domain}'. Regenerating plots...")
        llm_plotting.generate_revamped_plots(
            domain=domain,
            knowledge_probes_version=knowledge_probes_version,
            inference_probes_version=args.inference_probes_version,
            experiment_dir=args.experiment_dir,
            logger=logger,
            knowledge_probe_filename_suffix=knowledge_probe_filename_suffix,
        )

    # After individual plots are generated, create the averaged plots
    logger.info("Generating averaged plots for all domains...")
    llm_plotting.generate_averaged_plots(
        experiment_dir=args.experiment_dir,
        logger=logger
    )

if __name__ == "__main__":
    main()
