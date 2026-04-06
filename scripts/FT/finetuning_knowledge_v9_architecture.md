# `finetuning_knowledge_v9.py` Architecture Reference

This document is a high-level API map for `scripts/FT/finetuning_knowledge_v9.py`.
Use it to quickly find the right layer before making deeper changes.

## Scope

This script orchestrates a two-stage training pipeline:

1. Continued pretraining (CPT) on source/paraphrase/explanation/prior-knowledge mixes.
2. Optional LIMA instruction tuning.

It does **not** implement low-level training loops itself. Instead, it composes utility APIs from:

- `utils.llm_configs`
- `utils.model_setup`
- `utils.data_preparation`
- `utils.llm_training`
- `utils.experiment_utils`

## One-Page Flow

```text
CLI args
  -> normalize + validate flags
  -> resolve concrete domains + source mapping
  -> construct experiment name/path
  -> save hyperparameters.json
  -> load model/tokenizer (optionally LoRA)
  -> CPT stage (or debug dataloader-only)
      -> choose strategy_name + strategy_args
      -> prepare mixed dataset via prepare_training_mix
      -> verify deterministic dataloader (debug_run_1/2.txt)
      -> trainer.train()
      -> save probe results
      -> optional local save + push to hub
  -> optional LIMA stage
      -> prepare_lima_dataset
      -> build SFT trainer
      -> assert packed multi-sequence batches + EOS endings
      -> trainer.train()
      -> save probe results
      -> optional local save + push to hub
```

## Core Abstractions

## 1) Domain and Source Resolution

- **Purpose**: convert high-level sources (`arxiv`, `legal`, `medical`) and optional per-source overrides into a flat set of concrete domains.
- **Entry API**: `resolve_domains_and_sources(args, log)`
- **Outputs**:
  - `args.resolved_domains: List[str]`
  - `args.domain_data_sources: Dict[domain, source]`
- **Key behavior**:
  - Defaults to all discovered domains per source when no override is provided.
  - Detects collisions where the same domain name appears in multiple sources.

Where implemented:

- `scripts/FT/finetuning_knowledge_v9.py` (`_domain_catalog_root`, `_discover_domains_for_source`, `resolve_domains_and_sources`)

## 2) Experiment Naming as Run Metadata Schema

- **Purpose**: encode training recipe into a path-like experiment ID for reproducibility.
- **Entry API**: `construct_experiment_name(args)`
- **Encodes**:
  - train type (`full` vs `peft`)
  - model size/id
  - probe version
  - chunking mode + overlap
  - data mix + explanation mode/cycle/tracks/strategy
  - pretraining fill/separator policy
  - per-source domain selection compact tag
  - epochs / effective batch / LR / constant-LR flag
  - overlap mode
  - optional semi-cleaned version
  - run suffix (timestamp or custom)
  - shuffle marker

This function is the canonical place to keep run taxonomy stable.

## 3) Training Strategy Dispatch (CPT)

CPT does strategy selection in `continue_pretraining(...)` and delegates actual dataset construction to `utils.data_preparation.prepare_training_mix(...)` via `llm_training.fine_tune(...)`.

Strategy dispatch rules:

- `PriorKnowledge`
  - selected when `--prior_knowledge`
  - loads `../../data/<source>/prior_knowledge/<domain>/textbook.txt`
- `ParaphrasedArxivPaperWithExplanations`
  - selected when `--with_explanations` or `--with_specific_explanation`
- `ParaphrasedArxivPaper`
  - selected when `--num_paraphrased_texts > 0` (without explanation flags)
- `SingleArxivPaper`
  - selected when no paraphrases/explanations are used

Important: in `prepare_training_mix`, explanation loading is controlled by `"WithExplanations" in strategy_name`.

## 4) Explanation Insertion Strategies

Controlled by `--explanations_insertion_strategy` with mutually validated constraints.

- `granular`
  - builds per-domain explanation tracks
  - supports `--explanations_cycle` and `--explanations_num_tracks`
  - mixed via `replicate_and_interleave_tracks(...)`
- `whole`
  - inserts standalone explanation-only batches every `N` doc batches
  - mixed via `replicate_and_interleave_whole_insert_every_n(...)`
- `legacy`
  - older coupled splice behavior
  - mixed via `replicate_and_interleave_legacy(...)`

Where implemented:

- validation/normalization: `scripts/FT/finetuning_knowledge_v9.py`
- mixing logic: `utils/data_preparation.py`

## 5) Data Replay / Batch Filling Layer

`utils.data_preparation.PretrainingDataReplay` provides token streams from `../../data/olmo/<pretraining_data_type>_100M_tokens.npy`.

Used for:

- filling batches to effective batch size (`fill_batches_with_pretraining`)
- inserting separator batches (`separate_batches_with_pretraining`)
- filling short chunk gaps (`fill_underfilled_chunks`)

This layer is the core abstraction for replay-augmented CPT.

## 6) Callback and Probe Architecture

Callbacks are assembled by `experiment_utils.setup_callbacks(...)` for both CPT and LIMA.

Main callback classes live in `utils/llm_callbacks.py`:

- `BaseKnowledgeProbeCallBack`
- `GenerationProbeCallback`
- `CorpusPerplexityCallback`
- `TrainingLossPerplexityCallback`
- `WandbSourcePanelsCallback`

V9 built-ins (set in main after parsing):

- inference probes disabled
- W&B corpus perplexity logging disabled
- W&B training-loss perplexity logging disabled
- probe metric allowlist constrained to:
  - `log_prob`
  - `hit_accuracy_at_1`
  - `hit_accuracy_at_10`
  - `hit_accuracy_at_100`

## 7) Model/Trainer Composition Layer

- Model loading and LoRA wiring:
  - `model_setup.load_model_for_training(...)`
- Config objects:
  - `llm_configs.ModelConfig`, `PeftConfig`, `TrainingConfig`
- CPT trainer entry:
  - `llm_training.fine_tune(...)`
- LIMA trainer entry:
  - `llm_training.sft_train_on_dataset(...)`

The script itself is orchestration glue; these APIs own the heavy lifting.

## Pipeline Phases (Detailed)

## Phase A: Parse, Normalize, Validate

- parse CLI args
- apply v9 hard defaults (disable inference probes/perplexity W&B)
- apply `--prior_knowledge` overrides (disable paraphrase/explanation; optional epoch/batch overrides)
- parse and validate explanation strategy constraints

## Phase B: Environment + Paths

- set logging
- set `base_results_dir` (`../../results/tests` in `--test_script`; else `../../results/FT`)
- set W&B project env var when not in test mode
- resolve domains/sources
- build experiment name (or override)
- write `hyperparameters.json`

## Phase C: Load Model

- build `PeftConfig` and `ModelConfig`
- optionally add `<|EOT|>` token when LIMA stage is enabled
- load model/tokenizer

## Phase D: CPT or Debug Dataloader Mode

- `run_cpt = num_train_epochs > 0 or debug_dataloader_only`
- `continue_pretraining(..., train=not debug_dataloader_only)`
- internally:
  - build `TrainingConfig`
  - setup callbacks (`is_lima=False`)
  - choose strategy name/args
  - call `llm_training.fine_tune(...)`
  - save probe results when `train=True`

Debug mode still runs data prep + deterministic dataloader checks, but skips gradient updates.

## Phase E: Optional LIMA

- `lima_training(...)` if `--lima_afterwards`
- prepare GAIR/lima dataset
- build LIMA `TrainingConfig`
- setup callbacks (`is_lima=True`)
- build trainer in no-train mode first for dataloader integrity checks
- assert at least one packed batch with multiple sequences and warn if batch tail lacks EOS
- run `trainer.train()`
- save probe results

## Phase F: Persist / Publish

- optional local save after CPT (`cpt_model_subdir`)
- optional push CPT to hub
- optional local save after LIMA (`lima_model_subdir`)
- optional push LIMA to hub (inside `lima_training`)

## Code Map (Primary Entry Points)

Main orchestration:

- `scripts/FT/finetuning_knowledge_v9.py`

Config and model APIs:

- `utils/llm_configs.py`
- `utils/model_setup.py`

Data-mix and replay logic:

- `utils/data_preparation.py`

Training wrappers:

- `utils/llm_training.py`

Probe/callback setup and persistence:

- `utils/experiment_utils.py`
- `utils/llm_callbacks.py`

## Jump Table (Where to Read First)

Primary script (`scripts/FT/finetuning_knowledge_v9.py`):

- domain/source constants: lines 25-31
- domain discovery and resolution APIs:
  - `_domain_catalog_root`: line 35
  - `_discover_domains_for_source`: line 47
  - `resolve_domains_and_sources`: line 64
- experiment naming schema:
  - `construct_experiment_name`: line 118
- CPT orchestration:
  - `continue_pretraining`: line 261
  - callback setup call: line 297
  - trainer delegation (`llm_training.fine_tune`): line 362
- LIMA orchestration:
  - `lima_training`: line 389
  - trainer delegation (`llm_training.sft_train_on_dataset`): line 442
- CLI + validation + runtime wiring:
  - parser starts: line 501
  - v9 hard defaults (`disable_inference_probes`, metric allowlist): lines 675-683
  - strategy argument validation block: lines 705-769
  - env + results dir setup: lines 779-785
  - domain resolution call: line 794
  - hyperparameter snapshot write: lines 802-807
  - model load call: line 834
  - CPT/LIMA execution branches: lines 849-870

Utility APIs (deep logic):

- `utils/llm_configs.py`
  - `PeftConfig`: line 13
  - `QuantizationConfig`: line 24
  - `ModelConfig`: line 28
  - `TrainingConfig`: line 38
  - `to_sft_training_args`: line 108
- `utils/model_setup.py`
  - `load_model_for_training`: line 35
- `utils/llm_training.py`
  - `sft_train_on_dataset`: line 149
  - `save_model`: line 173
  - `fine_tune`: line 190
- `utils/data_preparation.py`
  - `PretrainingDataReplay`: line 14
  - `fill_underfilled_chunks`: line 87
  - `prepare_lima_dataset`: line 131
  - `prepare_training_mix`: line 162
  - `replicate_and_interleave_tracks`: line 611
  - `replicate_and_interleave_legacy`: line 685
  - `replicate_and_interleave_whole_insert_every_n`: line 736
- `utils/experiment_utils.py`
  - `setup_callbacks`: line 80
  - `save_probe_results`: line 271
- `utils/llm_callbacks.py`
  - `BaseKnowledgeProbeCallBack`: line 33
  - `WandbSourcePanelsCallback`: line 776
  - `GenerationProbeCallback`: line 903
  - `CorpusPerplexityCallback`: line 1078
  - `TrainingLossPerplexityCallback`: line 1188

## Where To Modify What

If you want to add/change a feature, start here:

- Add a new high-level source (beyond arxiv/legal/medical):
  - `SUPPORTED_HIGH_LEVEL_DOMAINS`, `DOMAIN_OVERRIDE_ARG_BY_SOURCE` in `finetuning_knowledge_v9.py`
  - source root resolution in `finetuning_knowledge_v9.py` and `utils/data_preparation.py`
  - callback corpus path resolution in `utils/experiment_utils.py`

- Add a new data mixing policy:
  - strategy selection in `continue_pretraining(...)`
  - implement mixer in `utils/data_preparation.py`
  - optionally add naming tags in `construct_experiment_name(...)`

- Add/modify probe metrics or W&B panel behavior:
  - callback assembly in `experiment_utils.setup_callbacks(...)`
  - callback class implementations in `utils/llm_callbacks.py`
  - v9 metric allowlist defaults in main

- Change checkpoint/publish behavior:
  - final save/push branches in main (`save_local_model`, `push_to_hub_*`)

- Change run naming taxonomy:
  - `construct_experiment_name(...)` only

## Invariants and Guardrails

- effective batch size must be divisible by device batch size (CPT and LIMA)
- explanation strategy args are cross-validated for compatible combinations
- domain names must be source-unambiguous
- CPT dataloader order is asserted deterministic by double-pass content check
- whole insertion strategy requires `--explanations_insert_every_n > 0`

## Fast Mental Model

Think of `finetuning_knowledge_v9.py` as a coordinator with four responsibilities:

1. validate and normalize an experiment spec (CLI args)
2. build a deterministic run identity (experiment path)
3. route into reusable subsystem APIs (data mix, trainer, callbacks)
4. persist outputs (metrics/checkpoints/hyperparameters)

For implementation work, most complexity lives in utility modules, not in the script body.
