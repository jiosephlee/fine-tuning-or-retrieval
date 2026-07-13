# SLURM vLLM multiview generation

The vLLM environment is expected at
`/vast/projects/myatskar/design-documents/conda_env/vllm`. It must contain a
CUDA-enabled PyTorch build; the batch runner checks this inside the allocation.

## Submit an end-to-end run

From the repository root:

```bash
scripts/slurm/submit_vllm_multiview.sh \
  --model openai/gpt-oss-20b \
  --domain arxiv \
  --item DPO \
  --parts all \
  --max-workers 2
```

The default allocation is one `b200-mig45` GPU, four hours, 16 CPUs, and 128G
of host memory. Logs are written under `logs/vllm/`; generated views use the
model-derived output slug under `data/<domain>/explanations/`.

Use `--dry-run` to inspect the generated `sbatch` command. Larger models can
override `--partition`, `--gpus`, and `--tensor-parallel-size`.

## Interactive debugging

```bash
srun --partition=b200-mig45 --gpus=1 --cpus-per-task=16 --mem=128G \
  --time=04:00:00 --pty bash -l

source /vast/projects/myatskar/design-documents/conda_env/vllm/bin/activate
cd /vast/projects/myatskar/design-documents/joseph/fine-tuning-or-retrieval

python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda)
print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
PY

vllm serve openai/gpt-oss-20b \
  --host 127.0.0.1 --port 8000 \
  --served-model-name openai/gpt-oss-20b \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching
```

In a second shell on the same allocation, set
`VLLM_BASE_URL=http://127.0.0.1:8000/v1` and run a pipeline with
`--provider vllm --base_url "$VLLM_BASE_URL"`.

## Post-generation agentic review

After generation and structural validation, use the Codex-native binary
sensibility review in
[`AGENTIC_MULTIVIEW_SENSIBILITY_REVIEW.md`](AGENTIC_MULTIVIEW_SENSIBILITY_REVIEW.md).
It requires no LLM API credits: a parent Codex session assigns every generated
prose unit to interactive subagents and reconciles complete `PASS`/`FAIL`
coverage.
