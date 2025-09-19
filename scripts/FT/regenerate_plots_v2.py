import os
import sys
import argparse
import logging
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import utils.llm_plotting as llm_plotting

def main():
    parser = argparse.ArgumentParser(description="Regenerate plots for a specific experiment run.")
    parser.add_argument("--experiment_dir", type=str, required=True, help="The full path to the experiment directory.")
    parser.add_argument("--knowledge_probes_version", type=str, default="v9", help="Version of the knowledge probes.")
    parser.add_argument("--inference_probes_version", type=str, default="v6", help="Version of the inference probes.")
    
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    if not os.path.isdir(args.experiment_dir):
        logger.error(f"Experiment directory not found: {args.experiment_dir}")
        return

    logger.info(f"Scanning for probe results in: {args.experiment_dir}")

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
            knowledge_probes_version=args.knowledge_probes_version,
            inference_probes_version=args.inference_probes_version,
            experiment_dir=args.experiment_dir,
            logger=logger
        )

if __name__ == "__main__":
    main()
