# Reviewer Response 

*Source: OpenReview reviews from T6jM, YQYf, and XstK.*

## Reviewer T6jM

Thank you for your feedback. We have performed additional experiments, would like to report these results, and respond to your comments to clarify key issues.

- **Weakness:** "The set of data is so small to fit into a single batch. Moreover, most of the batch is made of content restricted to a limited set of topics. Both conditions are far from real pre-training where sampling occurs into a much larger and heterogeneous distribution."
  - Response: We sure to clarify the statement "the set of data is so small to fit into a single batch". We would like to first clarify that the size of our batch for our main experiment is 256 where each chunk has a sequence length of 4096 tokens where at each training step the model is passed 1,048,576; while we we recognize that this is not the same condition as pre-training, we are not far from it. Olmo2 for the 7B model utilizes a abtch size of 1024 with a sequence length of 4096 tokens, only 4times larger than ours. Further, while we do insert a very homogenous distribution of documents concenring, law, computerscience, and medicine for 70% of the abtch, the other 30% is general itnernret data, the same that OLMO2 uses for their pretraining data. Thus, our experiments are much closer to the pre-training condition than the author may have understood. 
  - However, recognzie that it is still not the same, and as the author noted in a spearat ecomment the optimizer state is also not taken direfctly from pretraining, which tecnically puts us in the continued pretarning regime. To support the generalization of our results to have legiamte sceitntific iplications on pre-training, we repeated our main experiment in a true pre-training setting, where we've taken the OLMO training codebase and taken a checkpoint and its optimizer to replicate the exact same experiment but with the exact same data that OLMO2 used in pret-traiing and th exact same optimizer. We see that the results are the same where .... We see that parahasing does actaully a bit wrose in these settings and this is predicabtl explained by the parahrasing laws our paper has uncovered where paraphrasing does worse at larger abtch sizes.

- **Weakness:** "Fixing the token budget is necessary for fair comparison. However, upsampling paraphrasing and source in order to match the same budget of the auxiliary views introduces a significant amount of duplicates in the batch, especially in the source setting. Duplication has been known to harm pre-training, and may explain such a strong performance gap. It also explains why overfitting occurs sooner than in auxiliary views."
  - First, before we dicsuss empirical results, we awnat to push back on the notion that duplication has "harmed" pretraining. The prevailing notion in the literature is that duplication is actually good (e.g. see repetition) but it is merely inefficient and compute should be spent on seeing new tokens to maimize use of compute (e.g. see https://aclanthology.org/2022.acl-long.577.pdf ) and its harmful effect that is encourages vebratim memorization (but that is only good in our setting where our factual probes check for near-verbatim questions). However, it's still quite possibe and likely taht sufficient reperition does actaully harm the model and cause overfitting.
  - The ceiling saturation argument from Figure 1. -> We see that in figure 1 that we can check if our "results hold" with less tokens by simply examining the graph at a earlier timestep where it would have seeen the documents less and thus overfit less. We see that already at 20 steps that the auxiliary views exceeds both paraphrasing and source and both of the other cnoditions quickly saturate. While one could fframe this as source is overfitting, we are not concerned with whether source is overfit by the end or not, but the "ceiling of source" and how well it can do. We see that source does not do verty well and then collapse but it just faces a lower ceiling to begin with compared to auxiliary view. By design, the strenght of auxiliary views, is that it is a strong way to "augment the data" allowing the model to go past the "source only ceiling" and learn to exceed it with sufficient training stpes. We agree that on a low compute setting, auxiliary views wouldn't necessarily be better as they do similar in the beginning of training. Furthermore, we don't do better than just source but also paraphrase, which is meant to prevent the overfitting from source.
  - One possible counterargument to the above is that once we upsample tokens, the density of similar tokens seen in each batch increases which may put source at an inherent disadvantage. While we still thinkg the "ceiling argument holds", it's possible that the density prevents source from reaching its ceiling. Thus we repeat our experiemnts at lower densities, 0.5 and 0, where 0.5 is the 0.5 of the axuliwray views are inserted and thus the upsamplins is done only 50% of the degree. Then 0, is where we keep the single document insertions at each batch, but then we take out certain document insertions and just put in auxilairy views instead which is the most advversarial as we're giving it much less exposure to the source document. We see that while the gap diminishes, auxiliary views still have an advntage. Furthermore, the more important aspect is that source does not ever do better with lower densities. IT only does wrose, showing that source actaully benefits from much duplication but it does much much better with axulixary views, and is a much ebtter way to augment data to acquire some particualr knowledge.

- **Weakness:** "The setup studies continual learning, rather than pre-training. Learning may occur differently, since in continual pre-training the optimizer state is missing."
  - This is something that we thought about for a while. One, we recognize that our paper is bifurcated in the sense we aim to clarify pre-training, but the actual setting of our experiments is continued pre-training in which we were not able to exactly continue the optimizer state. We wish to highlight that  this setting is important as is the setting that all practitioners need to take on to adapt LLMs via continued pre-training (as most models do not make this available). But that doesn't change your critique, as it lessens the generalization of our findings to non-continued pre-training. 
  - Thus, as stated in an earlier above, we've repeated our main experiment on the true pre-training condition, and see that our results hold.

- **Weakness:** "Auxiliary views are synthetic, from strong LLMs. The gains may partly reflect indirect distillation from a strong teacher."
  - We would like to clarify that the LLM that generated the actual synthetic text itself is gpt-5-mini. While this is likely a stronger model than OLmo2-7B itself, we avoided this confounding factor as much as possible.
  - Second, we actually want to argue that this "teacher" aspect is inherent to the nature of auxiliary views. In society, it is often a "teacher" who already understands this concept with some level of mastery and helps explain this to a "student." Whether it is human (which we have a specific ablation for as well) or not, we find that this aspect is inherent to the nature of auxiliary views, and it helps argue why pre-training is originally so effective to begin with.
  - But it is true that most of these advantages of axuliary views could be tied to the fact that we have a really good model and perhaps our results don't hold as well with weaker models;
    - Results:
      - Plot teachers against model size
      - Another axis is "pretraining distribution or not"
      - Plot against "accuracy on our evals"
    - Our findings are still interesting regardless of distillation as we reveal a fasctingiant gdnyamic where factual recall increases.

- **Weakness:** "The analysis is limited to a single family of LLMs. However, this is a minor weakness."
  - As noted above, we've tested the effectiveness of auxiliary view generation across different models.
  - Also Tested training under different models (Qwen-2.5-7B) adn see the smae results regarding auixliary views; we also plan to include the mechanistic analysis on these models to where we see similar mechanistic behaviors though slightly different. 

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

  - Re-run prior knowledge ordering experiments + contextual views vs. prior knowledge experiments

## Result Tables

All numbers below are aggregates over all 36 domains (arxiv + legal + medical) unless noted. Factual evaluation uses the v14 cloze probes and v15 five-shot MCQA probes; inference evaluation uses the reviewed v11 cloze probes and v14 five-shot MCQA probes. “Final” denotes the end of the 100-step training window. Higher values are better for every metric, including log-probability. Bold marks the best result in each column.

### Table A: True continued pretraining with the OLMo framework

OLMo-2 7B resumed mid-pretraining from checkpoint step 925,000 with optimizer state and the original data stream and learning-rate schedule (global batch 1,024; sequence length 4,096). Domain data was injected for 100 optimizer steps.

| Condition | Factual log prob. | Factual MCQA acc. | Inference log prob. | Inference MCQA acc. |
|---|---:|---:|---:|---:|
| Pretrained model (step 925k) | -16.30 | 0.343 | -14.79 | 0.413 |
| Source (document-matched) | -10.00 | 0.372 | -12.85 | 0.421 |
| Para. 9 (document-matched) | -10.33 | 0.375 | -12.53 | 0.417 |
| Auxiliary views | **-9.56** | **0.403** | **-10.82** | **0.492** |

### Table B: Effect of reducing token-matching upsampling

The scale-1.0 rows are the original document-matched runs used in the model-scaling plots. The scale-0.5 runs halve both the auxiliary-view track and the Source/Para. 9 content inserted to match that track. In the final block, Source and Para. 9 receive no matched upsampling, while legacy v2 replaces up to half of the paraphrase-tail with auxiliary-view chunks; it is therefore a replacement control, not an exact scale-0.0 continuation. Bold marks the best result within each regime. The scale-0.5 Auxiliary inference-MCQA result is omitted because that run used a zero-shot prompt while the Source and Para. 9 runs used five-shot prompts; a common re-evaluation is needed.

| Matching regime | Condition | Factual log prob. | Factual MCQA acc. | Inference log prob. | Inference MCQA acc. |
|---|---|---:|---:|---:|---:|
| 1.0 (original) | Source | -12.57 | 0.365 | -14.55 | 0.413 |
| 1.0 (original) | Para. 9 | -11.69 | 0.377 | -13.30 | 0.425 |
| 1.0 (original) | Auxiliary views | **-11.09** | **0.398** | **-11.67** | **0.450** |
| 0.5 | Source | -12.32 | 0.362 | -14.21 | 0.391 |
| 0.5 | Para. 9 | -11.95 | 0.375 | -13.40 | 0.401 |
| 0.5 | Auxiliary views | **-11.26** | **0.396** | **-11.75** | — |
| No matched upsampling (replacement; legacy v2) | Source | -13.04 | 0.362 | -14.35 | 0.379 |
| No matched upsampling (replacement; legacy v2) | Para. 9 | -12.51 | 0.370 | -13.27 | **0.382** |
| No matched upsampling (replacement; legacy v2) | Auxiliary views | **-12.41** | **0.373** | **-12.83** | 0.370 |

### Table C: Auxiliary-view generation across generator models

The generator rows use the same OLMo-2 7B training schedule and token budget; only the auxiliary-view generator changes. The pretrained model and document-matched Para. 9 condition are included as reference baselines.

| Generator or baseline | Access | Factual MCQA acc. | Inference MCQA acc. |
|---|---|---:|---:|
| None (pretrained OLMo-2 7B) | — | 0.337 | 0.366 |
| Para. 9 (document-matched baseline) | — | 0.377 | 0.425 |
| gpt-5-mini (original) | closed | 0.398 | **0.450** |
| GLM-5.2 | open-weight | 0.396 | 0.413 |
| gpt-5-mini (low reasoning) | closed | **0.402** | 0.416 |
| gpt-5-mini (high reasoning) | closed | 0.399 | 0.429 |
| gpt-5.4-mini (low reasoning) | closed | 0.394 | 0.410 |
| gpt-5.4-mini (high reasoning) | closed | 0.396 | 0.401 |
| gpt-oss-20B (low reasoning) | open-weight | 0.401 | 0.416 |
| gpt-oss-120B (low reasoning) | open-weight | 0.394 | 0.422 |
| Gemma-4 12B IT | open-weight | 0.391 | 0.413 |
| Gemma-4 31B IT  | open-weight | 0.383 | 0.413 |
| GLM-5  | open-weight | 0.385 | 0.419 |

### Table D: Reasoning effort of the generator

The low/high results are mixed, showing no consistent advantage from higher reasoning.

| Generator | Reasoning | Factual log prob. | Factual MCQA acc. | Inference log prob. | Inference MCQA acc. |
|---|---|---:|---:|---:|---:|
| gpt-5-mini | low | -11.19 | **0.402** | **-11.47** | 0.416 |
| gpt-5-mini | high | **-11.17** | 0.399 | -11.56 | **0.429** |
| gpt-5.4-mini | low | -11.30 | 0.394 | -11.78 | 0.410 |
| gpt-5.4-mini | high | -11.26 | 0.396 | -11.80 | 0.401 |

### Table E: Qwen-2.5-7B replication

| Condition | Factual log prob. | Factual MCQA acc. | Inference log prob. | Inference MCQA acc. |
|---|---:|---:|---:|---:|
| Pretrained model | -15.31 | 0.440 | -15.02 | 0.478 |
| Source | -15.48 | 0.489 | -18.82 | 0.512 |
| Para. 9 | -13.28 | 0.516 | -16.24 | 0.559 |
| Auxiliary views | **-10.98** | **0.548** | **-12.74** | **0.562** |

## Internal provenance — do not post

- Table A: wandb summaries of `/data1/joseph/olmo-runs/peteish7-{source,para9}-docmatch-step925000` and `/data1/joseph/olmo-runs/peteish7-para9-granular-step925000` at step 925,100; baseline is step 925,000.
- Table B: scale 1.0 uses the original E1/E2/E3 document-matched runs configured in `scripts/plotting/plot_inference_mcqa_scaling.py` and their `reeval_v3` bundles. Scale 0.5 uses E40/E46/E47; E47 used `formatted_question` for inference MCQA, whereas E40/E46 used `formatted_question_5shot`, so that cell is withheld. The replacement rows use the clean local E23/E24/E25 bundles; exclude the earlier slurm runs and every `_para9_corrupeted` counterpart.
- Tables C and D: the E2 Para. 9 and E3 auxiliary-view final-checkpoint reevaluations plus the clean July 14 E26–E35 bundles under `results/FT/full/7b/`. Exclude every directory containing `_para9_corrupeted`. E33 and E34 use `para9_docmatch_expl_insertexplanations_gemma_4_12b_it` and `para9_docmatch_expl_insertexplanations_gemma_4_31b_nvfp4`.
- Table E: E39–E41 under `results/FT/full/Qwen_Qwen2.5-7B/`.
- Run `conda run --no-capture-output -n tuning python scripts/analysis/extract_rr_metrics.py /tmp/rr_metrics.json` to reproduce Tables C–E. E26–E35 and E39–E41 are non-null, contain all 36 domains at step 100, and end at step 100. Step-0 baselines agree within the OLMo and Qwen run groups. E3’s final-checkpoint reevaluation is stored with a step-0 label and is excluded from the step-100 check.
