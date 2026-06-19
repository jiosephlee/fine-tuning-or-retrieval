#!/bin/bash
#SBATCH --job-name=ffn_heatmap_1b
#SBATCH --output=logs/ffn_heatmap_1b-%j.out
#SBATCH --error=logs/ffn_heatmap_1b-%j.err
#SBATCH --partition=genoa-std-mem
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-02:00:00

set -euo pipefail
PY=/vast/projects/myatskar/design-documents/conda_env/tuning/bin/python
ROOT=/vast/projects/myatskar/design-documents/joseph/fine-tuning-or-retrieval
OUT="$ROOT/plots/ffn_channel_heatmaps/e456_1b_20260516"

echo "=== [1/2] Computing FFN channel arrays (CPU) ==="
"$PY" -u -s "$ROOT/scripts/plotting/legacy/plot_ffn_channel_heatmaps.py" \
  --manifest "$OUT/manifest_e456_1b.json" \
  --output_dir "$OUT" \
  --device cpu \
  --clip_percentile 99 \
  --skip_embeddings \
  --fig_width 22 --fig_height 12 --dpi 240

echo "=== [2/2] Restyling difference heatmaps to PDF ==="
for cmp in aux_base_minus_source_base para_base_minus_source_base; do
  "$PY" -u "$ROOT/scripts/plotting/plot_e123_ffn_difference_heatmaps.py" \
    --input_dir "$OUT/mlp" \
    --comparison "$cmp" \
    --metric cosine_distance
done

echo "DONE_FFN_1B"
ls -la "$OUT/mlp"/*all_projections*
