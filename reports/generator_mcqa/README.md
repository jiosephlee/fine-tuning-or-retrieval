# Generator MCQA benchmark

This benchmark measures the auxiliary-view generator conditions from E26--E35
on the same factual v15 and inference v14 five-shot MCQA probes used by the
downstream training runs. It records factual and inference accuracy separately
for two answer protocols:

- `constrained`: a JSON-schema answer restricted to A--E.
- `reasoned`: free reasoning followed by `Final answer: (X)`.

The main output is `accuracies.csv`. Per-question resumable state and vLLM logs
live outside the repository under `/local/joseph/generator_mcqa/`.

## Run the available models

```bash
# Show all API and local commands without running them.
scripts/run_generator_mcqa_benchmark.sh --dry-run

# Run/resume all OpenAI variants (requires API quota).
scripts/run_generator_mcqa_benchmark.sh --api-only

# Run/resume the five local checkpoints, one server at a time on all 8 GPUs.
scripts/run_generator_mcqa_benchmark.sh --local-only

# Restrict a run to one or more model keys.
scripts/run_generator_mcqa_benchmark.sh --models gpt_oss_20b_low glm_5_nvfp4
```

Every normal run first evaluates one factual and one inference item under both
protocols in the isolated smoke-state directory. Repeating a command skips
terminal records and retries only incomplete requests.

## GLM-5.2 through LiteLLM

The normal runner intentionally refuses `glm_5_2_nvfp4`. On a machine that can
reach the LiteLLM server, run the same evaluator explicitly:

```bash
export LITELLM_API_KEY=...
export LITELLM_BASE_URL=https://example.invalid/v1
/data1/joseph/miniconda3/envs/vllm/bin/python \
  scripts/evaluate_generator_mcqa.py \
  --model-key glm_5_2_nvfp4 \
  --allow-litellm \
  --state-root /path/to/exported-state
```

Copy the resulting `glm_5_2_nvfp4/` directory back and validate/import it:

```bash
/data1/joseph/miniconda3/envs/vllm/bin/python \
  scripts/evaluate_generator_mcqa.py \
  --model-key glm_5_2_nvfp4 \
  --import-state /path/to/glm_5_2_nvfp4 \
  --state-root /local/joseph/generator_mcqa/state
```

The import checks all 9,674 records against the current model ID, protocol,
family, question ID, and prompt hash before updating `accuracies.csv`.

## Correlation report

Correlations are deliberately withheld until all ten model rows are complete.

```bash
conda run --no-capture-output -n tuning \
  python scripts/analysis/extract_rr_metrics.py \
  /local/joseph/generator_mcqa/downstream_metrics.json

conda run --no-capture-output -n openrlhf \
  python scripts/analysis/correlate_generator_mcqa.py \
  --accuracies reports/generator_mcqa/accuracies.csv \
  --downstream-json /local/joseph/generator_mcqa/downstream_metrics.json \
  --output reports/generator_mcqa/correlations.csv
```

The second command fails without creating `correlations.csv` if any direct or
downstream score is missing.
