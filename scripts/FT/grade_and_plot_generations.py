import os
import sys
import json
import argparse
import logging
import pandas as pd
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from utils.llm_evals import evaluate_response

def setup_logging():
    """Sets up basic logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)

def parse_generation_file(file_path):
    """Parses a generation file to extract the prompt and the generated text."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    try:
        prompt_section = content.split("--- PROMPT ---")[1].split("--- GENERATION ---")[0].strip()
        generation_section = content.split("--- GENERATION ---")[1].strip()
        return prompt_section, generation_section
    except IndexError:
        return None, None

def main(args):
    log = setup_logging()
    log.info(f"Starting evaluation and plotting for experiment: {args.experiment_dir}")

    # --- 1. Find generation directories ---
    generation_dirs = glob(os.path.join(args.experiment_dir, "generation*"))
    if not generation_dirs:
        log.error("No 'generation' directories found in the specified experiment directory.")
        return
    log.info(f"Found generation directories: {generation_dirs}")

    # --- 2. Load reference prompts ---
    log.info("Loading reference prompts...")
    # This assumes the script is run from scripts/FT
    base_probes_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'probes', 'generation'))
    
    # Infer domains from the experiment directory structure
    probe_dirs = glob(os.path.join(args.experiment_dir, "*_knowledge_probe"))
    domains = [os.path.basename(d).replace('_knowledge_probe', '').replace('_lima', '') for d in probe_dirs]
    if not domains:
        log.warning("Could not infer domains from directory names. Will try to find all possible prompt files.")
        # Fallback to searching all domain directories in the probes path
        domain_dirs = glob(os.path.join(base_probes_path, '*'))
        domains = [os.path.basename(d) for d in domain_dirs if os.path.isdir(d)]
    
    # Add background domain for LIMA if not already present
    if "DPO" in domains and "recall_background_QA" not in domains:
        domains.append("DPO") # Assuming background QA is linked to DPO domain for pathing

    log.info(f"Inferred/using domains: {domains}")
    
    all_reference_prompts = {}
    for domain in domains:
        prompt_files = glob(os.path.join(base_probes_path, domain, '*.json'))
        for f in prompt_files:
            dataset_name = os.path.splitext(os.path.basename(f))[0]
            with open(f, 'r', encoding='utf-8') as file:
                prompts_data = json.load(file)
                # Re-structure for easy lookup by ID
                all_reference_prompts[dataset_name] = {str(item['id']): item for item in prompts_data}
    
    if not all_reference_prompts:
        log.error("Could not load any reference prompts. Aborting.")
        return
    log.info(f"Loaded {sum(len(v) for v in all_reference_prompts.values())} reference prompts across {len(all_reference_prompts)} datasets.")


    # --- 3. Process each generation directory ---
    for gen_dir in generation_dirs:
        log.info(f"Processing directory: {gen_dir}")
        
        # Find all prompt-specific subdirectories (e.g., recall_DPO/1, recall_DPO/2)
        prompt_dirs = [d for d in glob(os.path.join(gen_dir, '*', '*')) if os.path.isdir(d)]
        
        all_evals = []
        
        for p_dir in tqdm(prompt_dirs, desc=f"Evaluating prompts in {os.path.basename(gen_dir)}"):
            generation_files = sorted(glob(os.path.join(p_dir, 'generation_step_*.txt')))
            
            # Extract dataset and prompt_id from path
            try:
                prompt_id = os.path.basename(p_dir)
                dataset_name = os.path.basename(os.path.dirname(p_dir))
            except Exception as e:
                log.warning(f"Could not determine dataset/prompt ID for '{p_dir}': {e}. Skipping.")
                continue

            # Direct lookup for the reference prompt
            reference = all_reference_prompts.get(dataset_name, {}).get(prompt_id)
            if reference is None:
                log.warning(f"Could not find reference for dataset '{dataset_name}' with ID '{prompt_id}'. Skipping directory.")
                continue

            for gen_file in generation_files:
                step = int(os.path.basename(gen_file).replace('generation_step_', '').replace('.txt', ''))
                prompt_text, generated_text = parse_generation_file(gen_file)
                
                if prompt_text is None:
                    log.warning(f"Could not parse {gen_file}. Skipping.")
                    continue
                
                # --- 4. Evaluate ---
                eval_result = evaluate_response(
                    question=prompt_text, # The prompt the model actually saw
                    response=generated_text,
                    reference_answer=reference.get('reference_answer', '')
                )
                
                # --- 5. Store result ---
                eval_data = {
                    "step": step,
                    "dataset": dataset_name,
                    "prompt_name": reference.get('id', 'unknown'),
                    "score": eval_result['score'],
                    "feedback": eval_result['feedback']
                }
                all_evals.append(eval_data)
        
        if not all_evals:
            log.warning(f"No evaluations were performed for {gen_dir}.")
            continue
        
        # --- 6. Save results to CSV and Plot ---
        df = pd.DataFrame(all_evals)
        
        # Save detailed results
        csv_path = os.path.join(gen_dir, 'eval_results_hindsight.csv')
        df.to_csv(csv_path, index=False)
        log.info(f"Saved detailed evaluation results to {csv_path}")
        
        # Aggregate and plot
        agg_df = df.groupby(['dataset', 'step'])['score'].mean().reset_index()
        
        plots_dir = os.path.join(gen_dir, "plots_hindsight")
        os.makedirs(plots_dir, exist_ok=True)

        for dataset_name, data in agg_df.groupby('dataset'):
            if len(data['step']) < 2:
                log.warning(f"Not enough data points to plot for dataset '{dataset_name}' in {gen_dir}.")
                continue
            
            plt.figure()
            plt.plot(data['step'], data['score'], marker='o')
            plt.title(f'Mean Evaluation Score for {dataset_name}')
            plt.xlabel('Training Step')
            plt.ylabel('Mean Score')
            plt.grid(True)
            plot_path = os.path.join(plots_dir, f'{dataset_name}_eval_score.png')
            plt.savefig(plot_path)
            plt.close()
            log.info(f"Saved evaluation plot to '{plot_path}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate generated texts and plot scores for a given experiment.")
    parser.add_argument("experiment_dir", type=str, help="The path to the main experiment directory.")
    args = parser.parse_args()
    main(args)
