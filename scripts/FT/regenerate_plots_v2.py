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
    parser.add_argument("--inference_probes_version", type=str, default="v5", help="Version of the inference probes.")
    
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    if not os.path.isdir(args.experiment_dir):
        logger.error(f"Experiment directory not found: {args.experiment_dir}")
        return

    logger.info(f"Scanning for probe results in: {args.experiment_dir}")

    for subdir in os.listdir(args.experiment_dir):
        if subdir.endswith("_knowledge_probe"):
            domain = subdir.replace("_knowledge_probe", "")
            output_dir = os.path.join(args.experiment_dir, subdir)
            logger.info(f"Found knowledge probe for domain '{domain}'. Regenerating plots in {output_dir}...")
            llm_plotting.generate_new_plots_for_knowledge_probes(
                domain=domain,
                probes_version=args.knowledge_probes_version,
                output_dir=output_dir,
                logger=logger
            )
        elif subdir.endswith("_inference_probe"):
            domain = subdir.replace("_inference_probe", "")
            output_dir = os.path.join(args.experiment_dir, subdir)
            logger.info(f"Found inference probe for domain '{domain}'. Regenerating plots in {output_dir}...")
            llm_plotting.generate_new_plots_for_inference_probes(
                domain=domain,
                probes_version=args.inference_probes_version,
                output_dir=output_dir,
                logger=logger
            )

if __name__ == "__main__":
    main()
