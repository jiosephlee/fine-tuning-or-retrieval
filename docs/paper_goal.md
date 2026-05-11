# Paper Goal

This repository exists to support the experiments for the paper:

**"How Do Large Language Models Acquire Complex Knowledge From Text: A Case Study via Continued Pre-training"**

This file is intended to be the paper-level reference for future work in this repo, especially when rerunning or extending experiments.

## Core Objective

The paper studies **how large language models acquire new domain knowledge during continued pre-training (CPT)**, with a focus on **complex knowledge** rather than simple fact insertion.

The central scientific question is:

**What kinds of text help an LLM actually learn new knowledge, generalize it, and use it compositionally?**

The repo is therefore not just for "domain adaptation that works," but for **controlled experiments about why certain data formulations work better than others**.

## Main Research Questions

The repository is built to test three main questions from the paper:

1. **Paraphrasing and repetition**
   - If a model needs repeated exposure to learn new facts, does paraphrasing help it learn better than simply repeating the same source text?

2. **Knowledge formulation / auxiliary views**
   - Is the original paper alone enough, or do alternative formulations of the same knowledge, such as textbooks, blogs, and Stack Exchange style explanations, help models learn more generalizable representations?

3. **Continued pre-training bottlenecks**
   - Do gaps in prerequisite knowledge limit domain adaptation?
   - How much does data replay trade off against learning the new domain?
   - Does some of the injected knowledge only become visible after post-training or instruction tuning?

## Working Hypothesis

The paper's central hypothesis is that LLMs often **do not learn complex new knowledge well from primary sources alone**. They benefit from:

- repeated exposure,
- paraphrased restatements,
- auxiliary pedagogical views,
- explicit bridging of missing prerequisite knowledge.

In short, the project tests the idea that **auxiliary views act like scaffolding** for LLM learning.

## Methodological Summary

The paper uses a controlled domain-adaptation setup based on recent arXiv papers.

### Domains

The original domains in this repo are six CS papers:

- `DPO`
- `1_58`
- `GRPO`
- `BOFT`
- `OFT`
- `QLoRA`

These papers are treated as compact, self-contained knowledge domains.

The project has since expanded beyond CS to include two additional domains:

- **Legal** — Appellate opinions from U.S. Courts of Appeals (12 opinions). Source documents are plain text (not LaTeX).
- **Medical** — Medical case reports. Source documents are plain text (not LaTeX).

The legal and medical domains follow the same experimental methodology (auxiliary views, probes, CPT) but use domain-adapted data pipelines that remove LaTeX-specific processing steps and use domain-appropriate prompts and document structure parsing.

### Training Materials Per Domain

For each source paper, the methodology constructs multiple textual views of the same underlying knowledge:

- **Source paper**
  - The original arXiv paper text.

- **Paraphrases**
  - Rewritten versions of the paper used to test whether repetition with linguistic variation improves learning.

- **Auxiliary views**
  - Pedagogical reformulations such as textbooks, Stack Exchange style Q&A, and blogs.

- **Prior knowledge materials**
  - Texts that teach prerequisite concepts needed to understand the target paper.

The paper draft states that paraphrases were generated with GPT-4.1, while the other synthetic materials and probe-generation steps used GPT-5-family models.

### Probe-Based Evaluation

The core evaluation framework is probe-based and distinguishes two types of learned knowledge:

- **Factual probes**
  - Test whether the model remembers atomic facts explicitly stated in the documents.

- **Compositional probes**
  - Test whether the model can combine one or more learned facts to infer a new answer not stated verbatim.

The paper describes a bottom-up pipeline:

1. Identify knowledge-bearing sentences in the paper.
2. Generate factual QA pairs from those sentences.
3. Convert them into contextualized cloze-style probe statements with the answer at the end.
4. Generate compositional questions section-by-section and at full-document level.
5. Convert them to self-contained cloze statements.
6. Filter for quality and ambiguity.

The paper reports approximately:

- `2027` factual probes
- `271` compositional probes

### Learning Metrics

The paper evaluates performance during training using:

- **Average log-probability on target spans**
  - Teacher forcing is used for multi-token target spans.

- **Target rank**
  - Lower is better; rank is more granular than thresholded top-k metrics.

This means the repo is designed to inspect **learning dynamics over time**, not just a single endpoint score.

## Canonical Experimental Setup

At a high level, the repository is designed around the following workflow:

1. Select a recent arXiv paper as a self-contained domain.
2. Generate multiple training views of the same underlying knowledge.
3. Continue pre-training OLMo-2 models on those materials.
4. Track probe performance throughout training.
5. Compare how learning changes under different data formulations and training choices.

### Base Models

The paper uses OLMo-2 base models, especially:

- `1B`
- `7B`
- `13B`
- `32B`

In the repo, the main CPT entrypoint is `scripts/FT/finetuning_knowledge_v8.py`.

### CPT Training Pattern

The paper's main CPT setup is a **single-batch knowledge injection strategy**:

- the domain document(s) fit into one forward pass,
- the remaining batch capacity is filled with replay data,
- probe metrics are tracked throughout training.

The paper states that most runs use:

- `N = 100` knowledge injections / exposures

### Main Data Conditions

The canonical comparison conditions are:

- **Source only**
  - Repeatedly inject the original paper.

- **Para. 9**
  - Cycle through the original document plus nine paraphrases.

- **Para. 9 + auxiliary views**
  - Train on the paper/paraphrase cycle while also injecting explanatory materials such as textbooks, blogs, and Stack Exchange style content.

- **Prior knowledge first**
  - Train on prerequisite materials before CPT on the target paper.

- **Replay ablations**
  - Compare fill-in replay versus interleaved replay steps and different replay amounts.

- **Post-training / LIMA**
  - Measure whether some injected knowledge becomes more visible after instruction tuning.

### Replay Data

The methodology uses replay data from pretraining-style corpora to preserve broader capabilities while adapting to a target domain.

The paper specifically states:

- replay examples are streamed from the `DCLM` subset used in OLMo-2 pretraining,
- tokens are drawn continuously across fetches,
- replay amount and integration strategy are treated as experimental variables.

In the repo, `--pretraining_data_type` supports replay sources such as `dclm` and `arxiv`.

## Default Hyperparameters From The Paper

Unless otherwise noted, the draft lists the following defaults:

- learning rate: `2e-5`
- context length for CPT: `3072`
- effective batch size: `64`
- weight decay: `0.1`
- scheduler: cosine decay with `0.1` warmup ratio
- seed: `42`
- max grad norm: `1`
- optimizer: `AdamW`
- Adam betas: `0.9`, `0.999`
- epsilon: `1e-8`
- training dtype: `bf16`

For LIMA post-training, the draft says:

- context length: `2560`
- packing enabled
- `10` epochs

## Repo Mapping

The paper methodology maps onto the repo roughly as follows:

- `scripts/FT/finetuning_knowledge_v8.py`
  - Main CPT entrypoint and optional LIMA-afterwards flow.

- `scripts/FT/finetuning_prior_knowledge_v1.py`
  - Prior knowledge training experiments.

- `scripts/FT/pipeline_fact_probe.py`
  - Factual probe generation.

- `scripts/FT/pipeline_comprehension_probe.py`
  - Compositional / inference probe generation.

- `scripts/FT/pipeline_probe_inference.py`
  - Additional inference-probe processing.

- `scripts/FT/pipeline_paraphrase_text.py`
  - Paraphrase generation / preparation.

- `scripts/FT/pipeline_multi_view_knowledge.py`
  - Auxiliary-view generation.

- `scripts/FT/pipeline_diverse_views.py`
  - Additional multi-view generation logic.

- `scripts/FT/pipeline_generate_mcqa_probes.py`
  - MCQA generation for prerequisite knowledge analysis.

- `utils/data_preparation.py`
  - Data loading and replay logic.

- `utils/llm_callbacks.py`
  - Probe evaluation callbacks during training.

- `utils/experiment_utils.py`
  - Probe loading, domain handling, and callback setup.

## Main Script Flags To Remember

The main rerun script is `scripts/FT/finetuning_knowledge_v8.py`.

The most important arguments for staying aligned with the paper are:

- `--model_id`
- `--override_domains`
- `--num_train_epochs`
- `--full_finetuning`
- `--learning_rate`
- `--knowledge_probes_version`
- `--num_paraphrased_texts`
- `--with_specific_explanation`
- `--prior_knowledge`
- `--fill_batches_with_pretraining`
- `--separate_batches_with_pretraining`
- `--pretraining_data_type`
- `--effective_batch_size_for_cpt`
- `--device_batch_size`
- `--context_length_for_cpt`
- `--chunk_by_section`
- `--overlap_sections`
- `--overlap_ratio`
- `--lima_afterwards`
- `--test_script`

## Canonical Rerun Mental Model

When helping with reruns, assume the intended methodology is:

1. Choose one or more target domains.
2. Decide which knowledge view is being tested:
   - source only,
   - paraphrases,
   - paraphrases plus auxiliary views,
   - prior knowledge first,
   - replay ablation,
   - post-training after CPT.
3. Keep probe evaluation active throughout training.
4. Compare factual and compositional outcomes separately.
5. Treat learning dynamics, not just final scores, as the primary signal.

## Practical Goal For This Repo

When working in this codebase, the default goal is not just "improve benchmark performance." It is to help answer:

- how new knowledge gets encoded,
- what kinds of training text are most effective,
- when domain adaptation fails,
- whether prerequisite knowledge bottlenecks learning,
- how replay changes the tradeoff between adaptation and retention,
- whether learned knowledge emerges immediately or only after post-training.

## Expected Takeaway

The paper aims to show that successful domain adaptation depends heavily on **how knowledge is represented in text**, not only on scale or optimization. In particular, the repo is built to test whether:

- paraphrasing helps through safe repetition,
- auxiliary views improve both factual recall and compositional generalization,
- prerequisite knowledge materially affects downstream learning,
- replay changes adaptation behavior,
- some knowledge becomes more visible after post-training.

## Repo Interpretation Guide

If you are unsure why a script, dataset, or analysis exists, it likely supports one of these five paper-level goals:

1. **Measure knowledge acquisition during CPT**
2. **Compare source text vs paraphrases vs auxiliary views**
3. **Test prior knowledge as a bottleneck**
4. **Study replay as a domain adaptation tradeoff**
5. **Study whether knowledge emerges during or after post-training**
