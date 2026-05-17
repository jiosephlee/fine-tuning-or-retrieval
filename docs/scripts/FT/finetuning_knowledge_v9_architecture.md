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

The script currently supports several "multiview" layouts, where source,
paraphrase, and explanation views are mixed into CPT batches.

Baseline, no explanation views:

- Trigger: omit `--with_explanations` and `--with_specific_explanation`.
- Data: source document plus `--num_paraphrased_texts` paraphrase views.
- Mixing: uses `replicate_and_interleave_legacy(...)` with only document batches.
- Batch order: source batch, paraphrase batch 0, paraphrase batch 1, etc.,
  replicated according to the epoch-derived replication factor.

Granular explanation tracks:

- Trigger: `--with_explanations` or `--with_specific_explanation`, with
  `--explanations_insertion_strategy granular`.
- File layout with `--with_specific_explanation`: loads subfolders under
  `data/<source>/explanations/<domain>/<type>/*.txt`.
- File layout with `--with_explanations`: loads flat defaults from
  `data/<source>/explanations/<domain>/blogs.txt`, `stackexchange.txt`, and
  `textbook.txt` when present.
- Cycle control: `--explanations_cycle N` loads the first `N` files per selected
  subfolder; `--explanations_cycle full` loads all files. This is required when
  using `--with_specific_explanation` with granular insertion.
- Track control: `--explanations_num_tracks K` creates `K` phase-offset cycles
  per domain. Track 1 starts at file 0; later tracks start at
  `floor(track_idx * num_files / K)`.
- Mixing: `replicate_and_interleave_tracks(...)` keeps document batches intact
  and appends chunks from every active explanation track to every document
  batch.
- Granularity: `--explanation_granularity file` keeps the existing
  one-file-per-step schedule. `--explanation_granularity chunk` flattens
  selected explanation chunks and schedules
  `--explanation_track_size_by_chunk` chunks per step, default `4`, with a
  final partial step allowed.
- Source-relative matching: `--match_explanation_source_replay` appends a
  source/paraphrase replay track for every active explanation track. Each replay
  step has the same chunk count as the corresponding explanation step for that
  domain, so a domain with 23 explanation chunks in a cycle also receives 23
  matched source/paraphrase chunks in that cycle.
- Baseline document matching: E1/E2 should use `--document_track_baseline`
  instead of `--match_explanation_source_replay`. This adds no explanations;
  it only replays source/paraphrase chunks using the granular explanation file
  schedule named by `--document_match_specific_explanation`, so document
  exposure can be matched against E3 without switching to chunk-granularity
  scheduling.
- MCQA probe memory: MCQA uses `--mcqa_probe_batch_size`, separate from
  `--device_batch_size`, because few-shot MCQA prompts produce large full-logit
  tensors during callback evaluation. E1/E2/E3 scripts default this to `32`.
- Constraint: explanation track count is valid for `granular` and
  `granular_queue`.

Granular queue explanation tracks:

- Trigger: `--with_specific_explanation`, with
  `--explanations_insertion_strategy granular_queue`.
- File layout: loads all `.txt` files from selected subfolders under
  `data/<source>/explanations/<domain>/<type>/`.
- Track control: `--granular_explanations_num_tracks K` creates `K` queue
  tracks. Selected files are pooled across types, shuffled with `--shuffle_seed`
  plus domain context for deterministic tie-breaking, sorted by chunk count,
  then aligned so long files are paired with short files in the same batch.
- Unit: defaults to each explanation file as one queued item. With
  `--explanation_granularity chunk`, selected explanation chunks are grouped
  by `--explanation_track_size_by_chunk` instead.
- Constraint: `--granular_explanations_cycle` is not used for this mode.

Whole explanation insertion:

- Trigger: `--with_explanations` or `--with_specific_explanation`, with
  `--explanations_insertion_strategy whole`.
- File layout: loads flat explanation files from
  `data/<source>/explanations/<domain>/`.
- Default files: `blogs.txt`, `stackexchange.txt`, and `textbook.txt` when
  present.
- Specific files: `--with_specific_explanation` names map to flat files; aliases
  include `textbooks -> textbook.txt`, `blogs -> blogs.txt`, and
  `stack -> stackexchange.txt`.
- Mixing: `replicate_and_interleave_whole_insert_every_n(...)` keeps document
  batches intact and inserts one combined explanation-only batch after every
  `--explanations_insert_every_n` document batches.
- Constraint: `--explanations_insert_every_n` must be positive.

Legacy coupled splice:

- Trigger: `--with_explanations` or `--with_specific_explanation`, with
  `--explanations_insertion_strategy legacy`.
- File layout: same flat-file loading as `whole`.
- Mixing: builds a second set of document-shaped batches where explanation
  chunks are spliced into/replacing paraphrase-batch chunks, then
  `replicate_and_interleave_legacy(...)` uses the explanation-spliced batches
  for every replication. If no explanation-spliced batches exist, it falls back
  to the ordinary source/paraphrase batches.
- Constraint: legacy supports only one `--with_specific_explanation` type.

Random splice:

- Trigger: `--with_explanations` or `--with_specific_explanation`, with
  `--explanations_insertion_strategy random_splice`.
- File layout: same flat-file loading as `whole` and `legacy`.
- Mixing: preserves the source batch, chooses a deterministic random paraphrase
  start point using `--shuffle_seed` plus a stable domain offset, then replaces
  paraphrase chunks with explanation chunks in wraparound order.
- Example: with paraphrases `[p0, p1, p2, p3]` and random start `p2`, the splice
  order is `[p2, p3, p0, p1]`.
- Unit: chunk-level replacement only; it does not cut into the middle of a text
  chunk or token sequence.
- Partial replacement keeps the original paraphrase prefix and places
  explanation chunks at the tail, matching legacy partial-splice shape.

Shared controls:

- `--times_explanations N` repeats loaded explanation chunks before insertion.
- `--fill_batches_with_pretraining` pads underfilled mixed batches with replay
  tokens from `data/olmo/<pretraining_data_type>_100M_tokens.npy`.
- `--separate_batches_with_pretraining N` inserts `N` full replay batches between
  scheduled document/explanation batches.
- `fill_underfilled_chunks(...)` can fill large empty space inside individual
  chunks after the final schedule is built.

Where implemented:

- validation/normalization: `scripts/FT/finetuning_knowledge_v9.py`
- mixing logic: `utils/data_preparation.py`
- granular schedule: `replicate_and_interleave_tracks(...)`
- whole schedule: `replicate_and_interleave_whole_insert_every_n(...)`
- baseline/legacy schedule: `replicate_and_interleave_legacy(...)`

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

- factual knowledge probes default to `probes_v13.csv`
- selected factual probe files are preflighted before model loading for required
  `fact`, `probe`, and `target` columns
- MCQA probe callbacks are opt-in with `--mcqa_probes`/`--mcqa-probes`; they
  default to `probes_v14_mcqa.csv` via `--mcqa_probes_version` and can use a
  different version from the cloze factual probes
- selected MCQA probe files are preflighted before model loading for required
  `formatted_question` and `correct_label` columns when MCQA is enabled
- MCQA can be explicitly disabled with `--disable_mcqa_probes`/`--no-mcqa-probes`
- inference probes disabled
- W&B corpus perplexity logging disabled
- W&B training-loss perplexity logging disabled
- parameter-delta tracking has two knobs:
  - `--enable_parameter_delta_tracking` records online long-form parameter-delta
    metrics and plots for embeddings, MLP projections, and attention projections
  - `--parameter_delta_every_n_steps` can override sparse milestone recording
    with a fixed step interval
  - `--parameter_delta_compute_final_alignment` is default-off and must be set
    to save temporary raw deltas for final-alignment metrics and plots
- probe metric allowlist constrained to:
  - `log_prob`
  - `target_rank`
  - `mcqa_accuracy`

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
- resolve domains and validate the selected factual probe CSVs
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
