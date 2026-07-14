# Reviewer Response 

*Source: OpenReview reviews from T6jM, YQYf, and XstK.*

## Reviewer T6jM

Thank you for your feedback. We have performed additional experiments, would like to report these results, and respond to your comments to clarify key issues.

- **Weakness:** "The set of data is so small to fit into a single batch. Moreover, most of the batch is made of content restricted to a limited set of topics. Both conditions are far from real pre-training where sampling occurs into a much larger and heterogeneous distribution."
  - Response: run the experiment again with batch size 2048

- **Weakness:** "Fixing the token budget is necessary for fair comparison. However, upsampling paraphrasing and source in order to match the same budget of the auxiliary views introduces a significant amount of duplicates in the batch, especially in the source setting. Duplication has been known to harm pre-training, and may explain such a strong performance gap. It also explains why overfitting occurs sooner than in auxiliary views."
  - The ceiling saturation argument from Figure 1.
  - Replot the results with the x-axis as # of source documents.
  - We believe these concerns are less significant than one may think, since the advantage of auxiliary views should be contrasted with "paraphrasing," where we have much less repetition.
  - Duplication has never been bad, it's only been inefficient; duplication only helps.
  - Repeating the Auxiliary View experiment with no oversampling (both lr4e-5 and lr8e-5) & comparing against para49 (does actually worse; explained by paraphrasing laws).

- **Weakness:** "The setup studies continual learning, rather than pre-training. Learning may occur differently, since in continual pre-training the optimizer state is missing."
  - This is something that we thought about for a while. One, we recognize that our paper is bifurcated in the sense we aim to clarify pre-training, but the actual setting of our experiments is continued pre-training in which we were not able to exactly continue the optimizer state. We leaned towards the second due to the fact that this is the setting that all practitioners need to take on to adapt LLMs via continued pre-training (as most models do not make this available). In terms of semantics, we would frame continued pre-training as a subset of pre-training that is still relevant. But that doesn't change your critique, as it lessens the generalization of our findings to non-continued pre-training. We will clarify the scope of our paper as continued pre-training.
  - We will modify the text accordingly to clarify that our results are mainly for continued pre-training.

- **Weakness:** "Auxiliary views are synthetic, from strong LLMs. The gains may partly reflect indirect distillation from a strong teacher."
  - We would like to clarify that the LLM that generated the actual synthetic text itself is gpt-5-mini. While this is likely a stronger model than OLmo2-7B itself, we avoided this confounding factor as much as possible.
  - Second, we actually want to argue that this "teacher" aspect is inherent to the nature of auxiliary views. In society, it is often a "teacher" who already understands this concept with some level of mastery and helps explain this to a "student." Whether it is human (which we have a specific ablation for as well) or not, we find that this aspect is inherent to the nature of auxiliary views, and it helps argue why pre-training is originally so effective to begin with.
  - But it is true that having a "teacher model" is likely necessary for this idea to work.
    - Results:
      - Plot teachers against model size
      - Another axis is "pretraining distribution or not"
      - Plot against "accuracy on our evals"
      - Include Olmo2-Instruct
    - Our findings are still interesting regardless of distillation.

- **Weakness:** "The analysis is limited to a single family of LLMs. However, this is a minor weakness."
  - Tested the effectiveness of auxiliary view generation across different models.
  - Tested training under different models (Qwen-2.5-7B).
  - More LLM families tested: GLM-5.2 & gpt-oss-120b & gpt-oss-20b for auxiliary view generation.

## Reviewer YQYf

Thank you for your feedback and review. We respond to your comments below to clarify key issues.

- **Weakness:** "The experiments are primarily conducted on technical papers and medical corpora. As a result, it remains unclear whether the observed effects generalize across broader domains or are specific to these specialized datasets. Since collecting data from more diverse domains appears feasible, validating the findings on a wider range of corpora would be a valuable direction for future work."
  > We agree that validating our findings on additional domains is a valuable direction to strengthen the generalization of our results. One point we failed to sufficiently highlight and we would like to elaborate on, however, is that our corpus is substantially more diverse than what is common in controlled studies of knowledge acquisition. Because these studies must verify that the target knowledge is absent from pre-training to isolate it from confounding factors, they have relied on narrow corpora: synthetic biographies [1], GPT-4-generated fictional-entity descriptions [2], fictitious name–description pairs [3], entity-centric Wikipedia articles about recent films and people [4], atomic facts from Wikidata triples (e.g., "Where is [X] located?") [5], and recent sports events [6]. Against this backdrop, we cover three qualitatively distinct domains: computer science papers, U.S. federal appellate opinions, and PubMed case reports, with 36 documents and ~6,900 probes plus ~4,800 MCQA variants, exceeding the scale and domain diversity of these prior works while retaining their level of experimental control.
  > On feasibility, extending to further domains is less straightforward than it may appear, because our controlled setting imposes strict requirements on candidate documents. Each must be (1) verifiably absent from the pre-training corpus (we restrict to post-cutoff documents and confirm zero matches via the Infini-gram index), (2) genuinely *new* knowledge rather than restatements of established material, (3) self-contained enough that probes have a tractable ground truth that can be referenced, and (4) short enough (~10K tokens) for single-batch injection to remain tractable across many experiments on large models. Few open-access domains regularly produce documents meeting all four criteria; the three we chose (research papers, judicial opinions, case reports) are among the ones that do. Other candidate domains exist but many fall under fictional or biographical genres; as briefly noted in our introduction, the representation of such atomic facts is comparatively trivial, so we intentionally avoided these domains. That said, we recognize there are other promising domains that may satisfy our criteria, such as patent applications, and we agree that validating our findings on them is valuable future work, which we will note in the paper.
  >
  > [1] Physics of Language Models: Part 3.1, Knowledge Storage and Extraction
  >
  > [2] How Do Large Language Models Acquire Factual Knowledge During Pretraining?
  >
  > [3] The Reversal Curse: LLMs Trained on "A is B" Fail to Learn "B is A"
  >
  > [4] Instruction-tuned Language Models are Better Knowledge Learners
  >
  > [5] Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?
  >
  > [6] Injecting New Knowledge into Large Language Models via Supervised Fine-Tuning


- **Weakness:** "While the paper provides valuable insights into knowledge acquisition during pre-training, it would have been even more impactful if the authors had connected these findings to practical recommendations or empirical studies on foundation model training, such as pre-training strategies, post-training methodologies, or curriculum design. Such discussions would provide more actionable guidance for practitioners."
  > We wish to highlight that our Discussion section does offer practical recommendations: applying paraphrasing when data is scarce while recognizing that its benefit is conditional on batch size, augmenting training data with auxiliary views (especially for scientific domains where such views do not yet exist), and continuing pre-training on prerequisite knowledge to close foundational gaps. If more specific recipes are what is desired, we see the goal of this paper as scientifically isolating whether and how these data-representation choices matter; determining the exact recipes that incorporate them at scale requires a different kind of experimentation and is outside the scope of this paper, though we agree it is a natural next step that our findings help motivate. 
  > During the rebuttal, we've ran additional experimetns that could align with your desire for more rpactical insights from the paper. One of expeirments ablated the model that generates the auxiliary views; we find that each model performs well on a particular metric and the dominant factor is that the performance does not depend on model capabilties, but each model produces a unique auxiliary view that benefits the model differnetrlt; in additiona to the fact that this synsthetc data augmentation strategy does not strongly depend on the strenght of the teacher model but functions as a general technique as long as the model is capable of generating an auxiliary view. Further, we find that there is a natural diversity factor in synthetic data augmentaiton in which various models can be used to provide unique auxiliary views that come with unique advnatages that can be unified together; We plan to include the specific detaisl of this ablation in the expeirment and provide a discussion on synthetic data augmentation in the final manuscript.

- **Reproducibility:** 
  > While ARR does not allow links to be included in responses, we have now prepared a repository containing our code and datasets, which we will include in the final manuscript. The appendix also provides hyperparameters and detailed descriptions of how we collected the data and designed the pipeline. We would appreciate any specific feedback on what would further improve the reproducibility of our paper.

## Reviewer XstK (not yet addressed)

- **Weakness:** "Paraphrases and Auxiliary Views are generated synthetically with GPT-4.1 and GPT-5/mini. This means that most of the results depend on the quality of closed and expensive LLM generation. Beyond the dependence on closed-source LLMs we don't know how these are proficient in generating auxiliary views for low-resource domains (more unknown or critical domains)."
  -

- **Weakness:** "Connected with the previous. More LLMs from different families such as Claude should be tested for auxiliary views generation."
  -

## Misc

- **Paraphrasing laws (as a table):** More paraphrasing harms at larger batch sizes, and helps at smaller batch sizes.
- **Release of code and dataset:** Upload the GitHub repo to an anonymous repo – Shu. Look over review comments – Yidi.
- **Highlight the strengths of our paper:**
  - We test more papers and probes than previous studies like Chang et al.
  - We are the first to present this notion of auxiliary views — "something from the conclusion"...
- **Experiments to run:**
  - Generating Auxiliary Views with gpt-oss-120B, gpt-oss-20B, Qwen-3.6-32B, GLM-5.2-NVFP4, GLM-5.1-NVFP4, DeepSeek-V4 Flash
    - Run four new training runs with these auxiliary views
    - Plot the runs
  - Repeating the source, paraphrase, and aux with Qwen-2.5-7B
  - Repeating the source, paraphrasing, aux experiment with no upsampling
  - Re-run prior knowledge ordering experiments + contextual views vs. prior knowledge experiments

## Result Tables (LaTeX)

All numbers below are aggregates over all 36 domains (arxiv + legal + medical) unless noted.
Metric conventions: Factual = knowledge probes v14 (short targets, cloze log-prob) and factual MCQA v15 (5-shot);
Inference = inference probes v11 (reviewed, cloze log-prob) and inference MCQA v14 (5-shot).
"Final" = end of the 100-step training/injection window.

### Table A: True continued pretraining with the OLMo framework (addresses T6jM "continual learning vs. pre-training" + optimizer-state concern)

OLMo-2 7B resumed mid-pretraining from checkpoint step 925,000 **with optimizer state and the original data stream/LR schedule** (global batch 1024, seq len 4096); domain data injected into the tail of each global batch for 100 optimizer steps (the exact analogue of our 100-epoch v9 runs). Runs live in `/data1/joseph/olmo-runs/peteish7-{source-docmatch,para9-docmatch,para9-granular}-step925000`; numbers parsed from each run's wandb summary (step 925,100). Note the probe evaluator is our OLMo-native reimplementation (same probe files; constrained-decoding MCQA), so absolute values are close but not bit-identical to the HF-harness numbers in Tables B/C.

```latex
\begin{table}[t]
\centering
\caption{Knowledge acquisition during \emph{true} continued pretraining. OLMo-2 7B is resumed
from intermediate checkpoint step $925{,}000$ with its optimizer state, original pretraining data
stream, and original LR schedule (global batch $1024$). Domain data is injected into each global
batch for $100$ steps, mirroring our main experimental conditions. Auxiliary views still yield the
largest gains, confirming that our findings are not an artifact of the optimizer-state-free
continued-pretraining setup.}
\label{tab:olmo-native-injection}
\begin{tabular}{l cc cc}
\toprule
& \multicolumn{2}{c}{Factual} & \multicolumn{2}{c}{Inference} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5}
Condition & Log Prob & MCQA Acc. & Log Prob & MCQA Acc. \\
\midrule
Pretrained model (step $925$k)        & $-16.30$ & $0.343$ & $-14.79$ & $0.413$ \\
Source (doc-matched)                  & $-10.00$ & $0.372$ & $-12.85$ & $0.421$ \\
Paraphrases (doc-matched)             & $-10.33$ & $0.375$ & $-12.53$ & $0.417$ \\
Auxiliary views                       & $\mathbf{-9.56}$  & $\mathbf{0.403}$ & $\mathbf{-10.82}$ & $\mathbf{0.492}$ \\
\bottomrule
\end{tabular}
\end{table}
```

Companion scale result from the earlier source-only injection runs (6 arxiv domains, factual probes only; no doc-matching): MCQA integration of injected knowledge emerges with scale even in true pretraining.

```latex
\begin{table}[t]
\centering
\caption{Source-only injection during true continued pretraining at two model scales (6 arxiv
domains; factual probes only). Cloze log-prob improves similarly at both scales, but MCQA accuracy
only improves at 7B --- fast recall, slow integration into reasoning, consistent with our main
results.}
\label{tab:olmo-native-scale}
\begin{tabular}{l cc cc}
\toprule
& \multicolumn{2}{c}{Baseline} & \multicolumn{2}{c}{After injection (100 steps)} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5}
Model & Log Prob & MCQA Acc. & Log Prob & MCQA Acc. \\
\midrule
OLMo-2 1B (resumed at step $950$k) & $-20.27$ & $0.192$ & $-12.79$ & $0.195$ \\
OLMo-2 7B (resumed at step $925$k) & $-17.48$ & $0.298$ & $-11.92$ & $0.331$ \\
\bottomrule
\end{tabular}
\end{table}
```

### Table B: Auxiliary-view generation across generator models (addresses T6jM "distillation" + XstK "closed-source LLMs / more families")

7B OLMo-2 runs E26--E35: same para9 + granular-explanation schedule as the original run, with the auxiliary-view *content* regenerated by each model (`para9_docmatch_expl_insertexplanations_*`). Original = gpt-5-mini custom explanations (E3, canonical re-eval `reeval_v3`); E26--E34 values are final-step (step 100) training-callback evals on the identical probe set. E35 (GLM-5-NVFP4) is still training as of 2026-07-13.

```latex
\begin{table}[t]
\centering
\caption{Effect of the auxiliary-view generator model. All runs share the identical training
schedule and token budget on OLMo-2 7B; only the model generating the auxiliary views changes.
Open-weight generators, including ones far smaller than gpt-5-mini's likely scale (e.g.
gpt-oss-20b, Gemma-4 12B), recover most of the gain over the pretrained baseline, indicating the
benefit of auxiliary views is not mere distillation from one strong closed teacher.}
\label{tab:aux-view-generators}
\begin{tabular}{ll cc cc}
\toprule
& & \multicolumn{2}{c}{Factual} & \multicolumn{2}{c}{Inference} \\
\cmidrule(lr){3-4} \cmidrule(lr){5-6}
Generator & Access & Log Prob & MCQA Acc. & Log Prob & MCQA Acc. \\
\midrule
None (pretrained OLMo-2 7B)     & ---    & $-16.54$ & $0.337$ & $-15.20$ & $0.367$ \\
\midrule
gpt-5-mini (original)           & closed & $\mathbf{-11.09}$ & $\mathbf{0.398}$ & $-11.67$ & $\mathbf{0.450}$ \\
gpt-5-mini (low reasoning)      & closed & $-12.02$ & $0.397$ & $\mathbf{-11.65}$ & $0.429$ \\
gpt-5-mini (high reasoning)     & closed & $-12.01$ & $0.390$ & $-11.73$ & $0.429$ \\
gpt-5.4-mini (low reasoning)    & closed & $-12.10$ & $0.392$ & $-11.86$ & $0.413$ \\
gpt-5.4-mini (high reasoning)   & closed & $-12.03$ & $0.391$ & $-11.90$ & $0.410$ \\
gpt-oss-20b (low reasoning)     & open   & $-12.16$ & $0.390$ & $-12.21$ & $0.419$ \\
gpt-oss-120b (low reasoning)    & open   & $-12.27$ & $0.389$ & $-12.26$ & $0.413$ \\
Gemma-4 12B IT                  & open   & $-12.23$ & $0.381$ & $-12.64$ & $0.410$ \\
Gemma-4 31B IT (NVFP4)          & open   & $-12.22$ & $0.377$ & $-12.52$ & $0.413$ \\
GLM-5.2                         & open   & $-12.73$ & $0.378$ & $-12.87$ & $0.394$ \\
GLM-5 (NVFP4)                   & open   & \multicolumn{4}{c}{\emph{run in progress}} \\
\bottomrule
\end{tabular}
\end{table}
```

### Table C: Reasoning effort of the generator (low vs. high)

Subset of Table B contrasting reasoning effort per generator family. Reasoning effort has essentially no effect --- if anything, low reasoning is marginally better --- suggesting auxiliary-view quality does not hinge on expensive test-time compute by the teacher.

```latex
\begin{table}[t]
\centering
\caption{Low vs.\ high reasoning effort for auxiliary-view generation (OLMo-2 7B, identical
schedule and token budget). Higher teacher reasoning effort does not improve knowledge
acquisition.}
\label{tab:aux-view-reasoning-effort}
\begin{tabular}{ll cc cc}
\toprule
& & \multicolumn{2}{c}{Factual} & \multicolumn{2}{c}{Inference} \\
\cmidrule(lr){3-4} \cmidrule(lr){5-6}
Generator & Reasoning & Log Prob & MCQA Acc. & Log Prob & MCQA Acc. \\
\midrule
gpt-5-mini   & low  & $-12.02$ & $0.397$ & $-11.65$ & $0.429$ \\
gpt-5-mini   & high & $-12.01$ & $0.390$ & $-11.73$ & $0.429$ \\
\midrule
gpt-5.4-mini & low  & $-12.10$ & $0.392$ & $-11.86$ & $0.413$ \\
gpt-5.4-mini & high & $-12.03$ & $0.391$ & $-11.90$ & $0.410$ \\
\bottomrule
\end{tabular}
\end{table}
```

### Provenance of the numbers

- Table A: wandb summaries of `/data1/joseph/olmo-runs/peteish7-{source,para9}-docmatch-step925000` and `peteish7-para9-granular-step925000` (`average/*_average` keys, step 925,100; baseline = step 925,000 row). Scale table: `peteish7-inject-step925000-full` (step 925,100) and `peteish1-inject-step950000-full` (step 950,100), `eval/probes/*` keys averaged over the 6 domains.
- Tables B/C: `results/FT/full/7b/para9_docmatch_expl_insertexplanations_*/.../eval_bundles/inf_mcqa_v14` (final step 100) and, for the original, `.../para9_expl_textbooks+stackexchange+blogs_cyclefull/.../E3_granular_explanations_all_domains/eval_bundles/reeval_v3`. Baseline row = step-0 callback values (matches base-model `reeval_v4_mcqa_v15`: 0.337).
- Extraction scripts: `scripts/analysis/parse_olmo_wandb.py` (run with the `tuning` env) and `scripts/analysis/extract_rr_metrics.py`.