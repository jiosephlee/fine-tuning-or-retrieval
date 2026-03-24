# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research codebase investigating how different fine-tuning strategies and data preparation techniques affect a language model's ability to learn knowledge from academic papers and textbooks. Models are evaluated via knowledge probes (fact-based and inference-based) across six domains: DPO, 1_58, GRPO, BOFT, OFT, QLoRA.

## Setup

```bash
# Install dependencies (flash-attn and custom transformers fork must be built from source)
pip install flash-attn --no-build-isolation
pip install git+https://github.com/huggingface/trl
pip install pydantic datasets==3.6.0 peft bitsandbytes wandb matplotlib seaborn liger-kernel scikit-learn openai
git clone https://github.com/jiosephlee/transformers && cd transformers && pip install .[torch]

# Required auth
wandb init
huggingface-cli login

# API keys for LLM-as-judge evals
cp keys_demo.py utils/keys.py  # Edit with real keys
```

## Running Experiments

All experiment scripts live in `scripts/FT/` and must be run from that directory (paths use `../../` to reach project root).

```bash
# Single-GPU training
cd scripts/FT
python finetuning_knowledge_v8.py --model_id allenai/OLMo-2-1124-7B --num_train_epochs 100 \
    --override_domains DPO 1_58 --full_finetuning --learning_rate 2e-5

# Multi-GPU with DeepSpeed
accelerate launch --config_file deepspeed.yaml finetuning_knowledge_v8.py [args...]

# SLURM cluster
source ../../env_slurm.sh  # loads conda and modules
# Then run any E*_slurm.sh script

# Test mode (small model, 1 epoch, no wandb)
python finetuning_knowledge_v8.py --test_script [args...]

# Data preparation test
python test_data_prep.py
```

Experiment shell scripts follow the naming convention `E[number]_[size]B_[description].sh` (e.g., `E63_7B_granular_cycle_slurm.sh`).

## Architecture

### Entry Point & Training Flow

`scripts/FT/finetuning_knowledge_v8.py` — Main training script with argparse CLI. Two-phase pipeline:
1. **Continued pretraining (CPT)** on domain text with knowledge probe callbacks
2. **Optional LIMA instruction tuning** (`--lima_afterwards`) with the same probe tracking

Experiment names are auto-constructed as nested directory paths from args (training type / model size / probes version / chunking style / data mix / domains / epochs / ...).

### Core Modules (utils/)

- **`llm_training.py`** — `CustomSFTTrainer` wrapping TRL's SFTTrainer; `fine_tune()` orchestrates data loading → chunking → training
- **`data_preparation.py`** — Data loading strategies (`SingleArxivPaper`, `ParaphrasedArxivPaper`, `ParaphrasedArxivPaperWithExplanations`, `PriorKnowledge`); `PretrainingDataReplay` class for data replay batches; LIMA dataset prep
- **`model_setup.py`** — Model loading with HF transformers, PEFT/LoRA application, quantization (4-bit/8-bit), flash attention support
- **`llm_configs.py`** — Pydantic config schemas: `ModelConfig`, `PeftConfig`, `QuantizationConfig`, `TrainingConfig`, `InferenceConfig`
- **`llm_callbacks.py`** — Probe evaluation callbacks that compute hit accuracy (@1, @10, @100) and log-probability metrics at each training step or at sparse intervals
- **`experiment_utils.py`** — Callback factory, probe loading from CSV, domain enumeration
- **`chunking.py`** — Token-based and section-based text chunking with configurable overlap ratios
- **`llm_evals.py`** — LLM-as-judge evaluation using OpenAI GPT API
- **`llm_plotting.py`** — Visualization utilities for probe curves

### Data Pipeline Scripts (scripts/FT/pipeline_*.py)

These generate training data and evaluation probes from raw LaTeX sources:
- `pipeline_fact_probe.py` / `pipeline_comprehension_probe.py` / `pipeline_probe_inference.py` — Generate knowledge probes
- `pipeline_paraphrase_text.py` / `pipeline_cleaning_text.py` — Text preprocessing
- `pipeline_multi_view_knowledge.py` / `pipeline_diverse_views.py` — Multi-perspective augmentation
- `pipeline_generate_mcqa_probes.py` — Multiple choice question generation

### Data Layout

- `data/arxiv/[domain]/` — Source LaTeX papers, paraphrases, explanations per domain
- `data/probes/facts/` and `data/probes/inference/` — CSV probes with columns: fact, probe (context), target
- `data/*.npy` — Tokenized pretraining replay data (DCLM, arXiv)
- `results/FT/{full,peft}/` — Training outputs organized by experiment path
- `plots/` — Generated PDF visualizations
- `notebooks/FT/` — Analysis and data generation notebooks

### Key Training Flags

| Flag | Purpose |
|------|---------|
| `--full_finetuning` | Full fine-tune vs PEFT/LoRA (default) |
| `--num_paraphrased_texts N` | Number of paraphrase variants (0 = source only) |
| `--with_specific_explanation TYPE` | Add explanation data (blogs, stackexchange, textbooks, etc.) |
| `--separate_batches_with_pretraining N` | Insert N pretraining batches between document batches |
| `--fill_batches_with_pretraining` | Fill remaining batch capacity with pretraining data |
| `--overlap_sections --overlap_ratio 1_4` | Section-based chunking with 25% overlap |
| `--lima_afterwards` | Run LIMA instruction tuning after CPT |
| `--no_callback_every_step` | Run probe eval only at 25/50/75% instead of every step |
| `--test_script` | Test mode: 1 epoch, no wandb, small results dir |

### Tracking

All experiments log to **Weights & Biases** under the `fine_tuning_study` project. Distributed training uses **DeepSpeed ZeRO-3** via `deepspeed.yaml`.
