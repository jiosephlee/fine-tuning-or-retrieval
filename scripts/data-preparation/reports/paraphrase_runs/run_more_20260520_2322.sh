#!/usr/bin/env bash
set -euo pipefail

cd /vast/projects/myatskar/design-documents/joseph/fine-tuning-or-retrieval

echo "[$(date -Is)] Starting medical paraphrases 9-18"
python scripts/data-preparation/paraphrase/pipeline_paraphrase_text_v1_length_heuristic.py \
  --start_index 9 \
  --num_paraphrases 10 \
  --input_dir data/medical/cleaned \
  --output_dir data/medical/paraphrased \
  --input_extension .txt \
  --output_extension .txt \
  --prompt_domain medical \
  --model gpt-4.1

echo "[$(date -Is)] Starting legal paraphrases 9-18"
python scripts/data-preparation/paraphrase/pipeline_paraphrase_text_v1_length_heuristic.py \
  --start_index 9 \
  --num_paraphrases 10 \
  --input_dir data/legal/cleaned \
  --output_dir data/legal/paraphrased \
  --input_extension .txt \
  --output_extension .txt \
  --prompt_domain legal \
  --model gpt-4.1

echo "[$(date -Is)] Starting arxiv low-count paraphrases 9-18"
python scripts/data-preparation/paraphrase/pipeline_paraphrase_text_v1_length_heuristic.py \
  --papers ByteLatent FeatLLM GRPO GSPO LongRoPE fa3 xLSTM \
  --start_index 9 \
  --num_paraphrases 10 \
  --input_dir data/arxiv/cleaned \
  --output_dir data/arxiv/paraphrased \
  --input_extension .tex \
  --output_extension .tex \
  --prompt_domain academic \
  --model gpt-4.1

echo "[$(date -Is)] Finished all paraphrase runs"
