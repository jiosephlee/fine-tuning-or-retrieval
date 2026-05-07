import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np

# Adjust path to include utils if needed, though we might not need it for this standalone script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils.llm_plotting import set_plot_style
from utils import probe_paths

def load_and_filter_data(run_path, probe_type, domains, filter_logic, project_root):
    """
    Loads and filters data for a given run and probe type across domains.
    
    filter_logic: function that takes filter_data (json) and returns list of allowed indices
    """
    all_dfs = []
    
    for domain in domains:
        # Construct paths
        if probe_type == 'knowledge':
            # The directory name seems to be fixed to '1_58_knowledge_probe' based on previous ls
            # But let's try to be generic if possible, or stick to what we saw.
            # The user provided paths have '1_58' in them. 
            # Let's check the directory structure from the user request paths.
            # The paths provided are:
            # .../domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/...
            # And inside we saw '1_58_knowledge_probe'.
            # It seems safe to assume the prefix might be '1_58' for this specific 32B run.
            # However, to be robust, let's look for the directory.
            
            # Actually, looking at the `list_dir` output from step 6 & 7:
            # It has `1_58_knowledge_probe`, `BOFT_knowledge_probe`, etc.
            # But the user said "take all the bs64 source only strategies including 32B... found here .../domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/..."
            # Wait, the path itself ends in `.../11_21_06_14/`.
            # Inside that, we saw `1_58_knowledge_probe`.
            # Let's assume we want the `1_58` one as it matches the model name "OLMo-2-0325-32B" (maybe? 1.58 bit? No, 1_58 might be the method name or something).
            # The user didn't specify WHICH probe dir to use inside, but usually it's the one matching the method.
            # The path has `domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA`.
            # Let's use `1_58_knowledge_probe` as seen in the `list_dir`.
            
            probe_dir_name = "1_58_knowledge_probe"
            file_name = "1_58_knowledge_probe_metrics.csv"
            
            # Correction: The user said "take all the bs64 source only strategies including 32B".
            # And the path provided is specific.
            # Let's try to find the csv file dynamically or hardcode if it's the only one.
            # In step 13, we saw `1_58_knowledge_probe_metrics.csv`.
            pass
        else:
            probe_dir_name = "1_58_inference_probe"
            file_name = "1_58_inference_probe_metrics.csv"

        metrics_path = os.path.join(run_path, probe_dir_name, file_name)
        
        if not os.path.exists(metrics_path):
            # Fallback: try to find any *_probe directory
            possible_dirs = [d for d in os.listdir(run_path) if d.endswith(f"_{probe_type}_probe")]
            if possible_dirs:
                probe_dir_name = possible_dirs[0]
                # Assume csv name matches dir name + _metrics.csv
                file_name = f"{probe_dir_name}_metrics.csv"
                metrics_path = os.path.join(run_path, probe_dir_name, file_name)
        
        if not os.path.exists(metrics_path):
            print(f"Warning: Metrics file not found at {metrics_path}")
            continue
            
        try:
            df = pd.read_csv(metrics_path)
        except Exception as e:
            print(f"Error reading {metrics_path}: {e}")
            continue

        # Filter
        filter_path = str(probe_paths.resolve_filter_path('facts', '1_58', exact_match=True))
        # Wait, the filter path depends on the domain? 
        # In the reference script: `probe_paths.resolve_filter_path(probe_folder, domain, exact_match=True)`
        # Earlier exploration pointed at the 1_58 exact-match filter specifically.
        # That path does NOT have the domain in it.
        # Let's re-read the reference script carefully.
        # Earlier notes referenced both a domain-specific filter and the 1_58 filter.
        # It seems the filter might be global or per-method, not per-domain?
        # OR, the `1_58` IS the domain? No, `1_58` looks like a method or model identifier.
        # The user said "filter the probes similar to @[scripts/plotting/plot_comparison.py]".
        # In `plot_comparison.py`, it uses `domain` in the path.
        # The canonical filter path now resolves through probe_paths.
        # Step 12 `list_dir` was on `.../1_58_knowledge_probe`.
        # The 1_58 exact-match filter exists as a canonical probe artifact.
        # This implies `1_58` might be acting as the "domain" or the filter is shared.
        # BUT, the metrics CSV has a `domain` column (Step 14, line 133 in ref script adds it, but let's check the CSV content).
        # The CSV content in Step 14 does NOT have a `domain` column. It has `step,probe_index,log_prob...`.
        # So we need to map `probe_index` to the filter.
        # The `filter_em.json` has indices.
        # The CSV has `probe_index`.
        # The `plot_comparison.py` script iterates over domains and loads `filter_em.json` PER DOMAIN.
        # `domains` are found in `data/arxiv/cleaned`.
        # If I look at the CSV from Step 14, it seems to be ONE file for the whole run?
        # Wait, the path in Step 14 is `.../1_58_knowledge_probe/1_58_knowledge_probe_metrics.csv`.
        # It does NOT seem to be inside a domain subdirectory in the results.
        # However, `plot_comparison.py` iterates over domains and constructs paths like `{domain}_knowledge_probe`.
        # The path provided by the user `.../11_21_06_14/` contains `1_58_knowledge_probe`.
        # This suggests that for this specific run, maybe it's NOT split by domain in the results, OR `1_58` is treated as the domain name in the directory structure?
        # Let's look at the `domains` argument in `plot_comparison.py`. It finds domains from `data/arxiv/cleaned`.
        # If the results directory structure is `.../run_path/{domain}_knowledge_probe/...`, then `1_58` would be the domain.
        # But `1_58` sounds like "1.58 bit" or a method name.
        # The user said "take all the bs64 source only strategies...".
        # Let's assume for this specific plot, we are just plotting the data found in that directory, and we need to apply the filter.
        # Use that exact-match filter here.
        # This strongly suggests `1_58` is the "domain" or the key for filtering.
        # So I will use that specific filter file.
        
        with open(filter_path, 'r') as f:
            filter_data = json.load(f)
        
        allowed_indices = filter_logic(filter_data)
        
        # Filter dataframe
        df_filtered = df[df['probe_index'].isin(allowed_indices)].copy()
        all_dfs.append(df_filtered)

    if not all_dfs:
        return pd.DataFrame()
        
    return pd.concat(all_dfs, ignore_index=True)

def main():
    set_plot_style()
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    
    # Data Configuration
    # 32B Paths (from previous step)
    path_32b_source = "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/allenai_OLMo-2-0325-32B/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_21_06_14/"
    path_32b_para = "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/allenai_OLMo-2-0325-32B/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_21_06_44/"

    # 1B, 7B, 13B Paths (from plot_para_fake.py)
    # Note: Using the paths found in plot_para_fake.py for "Source" and "Para 9"
    # Assuming bs64 runs as requested ("take all the bs64 source only strategies")
    
    models_config = {
        "1B": {
            "Source": "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/1b/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_23_22_57",
            "Para 9": "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/1b/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_23_23_24",
            "Color": "#ffd700" # Yellow
        },
        "7B": {
            "Source": "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_20_22_51",
            "Para 9": "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/7b/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_20_22_51",
            "Color": "#ff7f0e" # Orange
        },
        "13B": {
            "Source": "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/allenai_OLMo-2-1124-13B/probes_v9/newline2/source_only/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_21_11_30",
            "Para 9": "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/results/FT/full/allenai_OLMo-2-1124-13B/probes_v9/newline2/para9/fill_dclm/domains_DPO-1_58-GRPO-BOFT-OFT-QLoRA/e100/bs64_lr2e-05/overlap_1_4/11_21_13_18", 
            "Color": "#d62728" # Red
        },
        "32B": {
            "Source": path_32b_source,
            "Para 9": path_32b_para,
            "Color": "#8B0000" # Dark Dark Red
        }
    }
    
    # Filter Logic
    def source_filter(data):
        return data.get('in_source', []) 
        
    def para_filter(data):
        return data.get('in_source_and_paraphrase', []) 

    # Helper to load data
    def load_strategy_data(run_path, filter_func, model_name):
        DOMAINS = ['1_58', 'BOFT', 'DPO', 'GRPO', 'OFT', 'QLoRA']
        
        all_k_dfs = []
        all_i_dfs = []
        
        for domain in DOMAINS:
            # --- Knowledge (Factual) ---
            k_path = os.path.join(run_path, f"{domain}_knowledge_probe", f"{domain}_knowledge_probe_metrics.csv")
            if os.path.exists(k_path):
                k_df = pd.read_csv(k_path)
                
                # Load Knowledge Filter
                k_filter_path = str(probe_paths.resolve_filter_path('facts', domain, exact_match=True))
                if os.path.exists(k_filter_path):
                    with open(k_filter_path, 'r') as f:
                        k_filter_data = json.load(f)
                    allowed_k = filter_func(k_filter_data)
                    k_df = k_df[k_df['probe_index'].isin(allowed_k)]
                    all_k_dfs.append(k_df)
                else:
                    print(f"Warning: Knowledge filter not found for {domain} at {k_filter_path}")
            else:
                # Only warn if it's the first domain or if we expect all to be there. 
                # Some runs might be partial, but usually all domains are present.
                # Let's check if we can fall back to finding *any* knowledge probe if the naming is different,
                # but for these runs the naming seems consistent.
                pass

            # --- Inference (Compositional) ---
            i_path = os.path.join(run_path, f"{domain}_inference_probe", f"{domain}_inference_probe_metrics.csv")
            if os.path.exists(i_path):
                i_df = pd.read_csv(i_path)
                
                # Load Inference Filter
                i_filter_path = str(probe_paths.resolve_filter_path('inference', domain, exact_match=True))
                if os.path.exists(i_filter_path):
                    with open(i_filter_path, 'r') as f:
                        i_filter_data = json.load(f)
                    allowed_i = filter_func(i_filter_data)
                    i_df = i_df[i_df['probe_index'].isin(allowed_i)]
                    all_i_dfs.append(i_df)
                else:
                    print(f"Warning: Inference filter not found for {domain} at {i_filter_path}")
        
        if not all_k_dfs:
            print(f"Warning: No Knowledge metrics found for {model_name} at {run_path}")
            final_k_df = pd.DataFrame()
        else:
            final_k_df = pd.concat(all_k_dfs, ignore_index=True)
            
        if not all_i_dfs:
            print(f"Warning: No Inference metrics found for {model_name} at {run_path}")
            final_i_df = pd.DataFrame()
        else:
            final_i_df = pd.concat(all_i_dfs, ignore_index=True)
        
        return final_k_df, final_i_df

    # Plotting Helper
    def create_plot(strategy_name, output_filename):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        metrics = [
            ('hit_accuracy_at_1', 'Hits@1'),
            ('hit_accuracy_at_10', 'Hits@10'),
            ('hit_accuracy_at_100', 'Hits@100')
        ]
        
        # Iterate over models
        for model_name, config in models_config.items():
            print(f"Processing {model_name} for {strategy_name}...")
            color = config['Color']
            
            if strategy_name == "Source":
                k_df, i_df = load_strategy_data(config['Source'], source_filter, f"{model_name} Source")
            else:
                k_df, i_df = load_strategy_data(config['Para 9'], para_filter, f"{model_name} Para")
            
            for idx, (metric_col, metric_name) in enumerate(metrics):
                # Row 0: Factual
                ax_fact = axes[0, idx]
                # Row 1: Compositional
                ax_comp = axes[1, idx]
                
                # Helper to plot
                def plot_line(df, label, color, style, ax):
                    if df.empty:
                        return
                    # Group by step and mean
                    grouped = df.groupby('step')[metric_col].mean()
                    ax.plot(grouped.index, grouped.values, label=label, color=color, linestyle=style, linewidth=2, alpha=1.0)
                    
                plot_line(k_df, f"{model_name}", color, '-', ax_fact)
                plot_line(i_df, f"{model_name}", color, '--', ax_comp)
                
                # Styling for Factual (Row 0)
                ax_fact.set_title(metric_name, fontsize=16)
                # ax_fact.set_xlabel("Exposure #") # Only bottom row needs xlabel
                if idx == 0:
                    ax_fact.set_ylabel("Factual Accuracy", fontsize=16)
                ax_fact.tick_params(axis='both', which='major', labelsize=12)
                ax_fact.grid(True, linestyle='--', alpha=0.7)
                
                # Styling for Compositional (Row 1)
                # ax_comp.set_title(f"Compositional {metric_name}") # Removed title
                ax_comp.set_xlabel("Exposure #", fontsize=16)
                if idx == 0:
                    ax_comp.set_ylabel("Compositional Accuracy", fontsize=16)
                ax_comp.tick_params(axis='both', which='major', labelsize=12)
                ax_comp.grid(True, linestyle='--', alpha=0.7)

        # Custom Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            # Models
            Line2D([0], [0], color='#ffd700', lw=2, label='1B'),
            Line2D([0], [0], color='#ff7f0e', lw=2, label='7B'),
            Line2D([0], [0], color='#d62728', lw=2, label='13B'),
            Line2D([0], [0], color='#8B0000', lw=2, label='32B'),
            # Types (Optional now since separated, but good for line style reference if needed, 
            # though we removed "Fact/Comp" from label to avoid clutter, the line style is implicit by row)
            # But wait, the user said "still used the dashed so that we don't need to write in 'Compositional' in the title"
            # Actually user said: "separating the compositional probes bust still used the dashed so that we don't need to write in "Compositional" in the title"
            # So maybe I should NOT put "Compositional" in the title?
            # But they are in different rows.
            # Let's keep the titles clear: "Hits@1", etc. and maybe row label?
            # Or just "Hits@1" on top and bottom?
            # Let's stick to "Hits@1" as title for both, and rely on row position/ylabel.
        ]
        
        # Refine titles based on user comment "don't need to write in 'Compositional' in the title"
        # for idx, (_, metric_name) in enumerate(metrics):
        #     axes[0, idx].set_title(metric_name)
        #     axes[1, idx].set_title(metric_name)

        # Add legend to the last plot or outside
        # Put it in the bottom right plot
        axes[1, 2].legend(handles=legend_elements, loc='lower right', fontsize=14)

        plt.tight_layout()
        output_path = os.path.join(project_root, 'plots', output_filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
        plt.close(fig)

    # Generate Source Plot
    create_plot("Source", "hits_source.pdf")
    
    # Generate Para Plot
    create_plot("Para", "hits_para.pdf")

if __name__ == "__main__":
    main()
