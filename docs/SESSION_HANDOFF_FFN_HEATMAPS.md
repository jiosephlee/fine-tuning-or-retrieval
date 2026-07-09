# FFN Channel Heatmap Work Handoff

## What Was Implemented

### 1) New FFN per-channel analysis script
- Added: `scripts/plotting/plot_ffn_channel_heatmaps.py`
- Purpose: compare per-channel MLP/FFN weight changes between model conditions.
- Supports required comparisons:
  - `source_vs_base`
  - `para_vs_base`
  - `aux_vs_base`
  - `aux_vs_para`
- Computes two metrics per layer/channel/projection:
  - `relative_delta_norm = ||cmp-ref||_2 / (||ref||_2 + eps)`
  - `cosine_distance = 1 - dot(ref,cmp)/(||ref||_2 ||cmp||_2 + eps)`
- Correct channel semantics:
  - `gate_proj`: row vectors `weight[j, :]`
  - `up_proj`: row vectors `weight[j, :]`
  - `down_proj`: column vectors `weight[:, j]`
- Output:
  - `.npy` arrays per comparison/projection/metric
  - grouped figures with stacked gate/up/down heatmaps per comparison+metric
- Includes:
  - JSON/YAML manifest input (`base`, `source`, `para`, `aux`)
  - shared color scale across projections per metric
  - optional percentile clipping (`--clip_percentile`, default 99)
  - high-resolution figure controls (`--fig_width`, `--fig_height`, `--dpi`)
  - validations for missing keys, layer/projection mismatches, and shape mismatches


### 2) Training script now saves local model checkpoints
- Updated: `scripts/FT/finetuning_knowledge_v8.py`
- Added local save behavior so trained checkpoints are available for weight-diff analysis.
- New CLI flags:
  - `--save_local_model/--no-save_local_model` (default save enabled)
  - `--cpt_model_subdir` (default: `model_cpt`)
  - `--lima_model_subdir` (default: `model_lima`)
- Behavior:
  - after CPT: saves model/tokenizer to `<experiment_dir>/<cpt_model_subdir>`
  - after optional LIMA: saves model/tokenizer to `<experiment_dir>/<lima_model_subdir>`


### 3) End-to-end quick bash pipeline
- Added: `scripts/FT/E99_7B_ffn_channel_heatmaps_quick.sh`
- Purpose: run a quick full workflow and generate FFN heatmaps.
- Steps:
  1. Train source-only model (50 epochs)
  2. Train paraphrased model (50 epochs)
  3. Train auxiliary-view model (50 epochs)
  4. Auto-create manifest and run FFN analysis script
- Uses fixed experiment names for deterministic paths:
  - `quick_ffn_heatmaps/7b/e50/source`
  - `quick_ffn_heatmaps/7b/e50/para9`
  - `quick_ffn_heatmaps/7b/e50/aux_views`
- Assumes saved CPT checkpoints under `model_cpt` (now handled by v8 script changes).


## How To Run

Retained primary outputs:
- `plots/ffn_channel_heatmaps/e123_7b_20260516`
- `plots/ffn_channel_heatmaps/e456_1b_20260516`


## Notes
- The new commit intentionally excludes unrelated local edits and `.DS_Store` files.
- Existing modified files not part of this handoff (e.g., `scripts/plotting/plot_comparison.py`, `scripts/plotting/plot_utils.py`, `utils/llm_configs.py`) were left untouched in git staging.
