# Reviewer Response 

*Source: OpenReview reviews from T6jM, YQYf, and XstK.*

## Reviewer T6jM

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

- **Weakness:** "The experiments are primarily conducted on technical papers and medical corpora. As a result, it remains unclear whether the observed effects generalize across broader domains or are specific to these specialized datasets."
  - While we agree that it's possible to extend to more domains, (1) the cost-effective feasibility of our experiments, in which we have 36+ domains, relies on the fact that these domains are relatively small (e.g., 10,000+ tokens) and (2) self-contained & new from the distribution. With all LLMs having pre-training distributions that cover 2024, etc., we find that not many domains have generated open-access documents that are (1) packaged in a relatively self-contained manner and (2) generate new knowledge.
  - Further, we also find our study, compared to previous studies like Chang et al., to be much more diverse, covering three complex domains (versus their focus on biographical facts alone).

- **Weakness:** Practical takeaways — "While the paper provides valuable insights into knowledge acquisition during pre-training, it would have been even more impactful if the authors had connected these findings to practical recommendations or empirical studies on foundation model training, such as pre-training strategies, post-training methodologies, or curriculum design. Such discussions would provide more actionable guidance for practitioners."
  - We wish to highlight that we have recommended "the nuanced point of paraphrasing" and "the importance of auxiliary views" as data augmentation for practical takeaways; if the reviewer is desiring full recipes, we find that the point of this paper is to scientifically determine the relevance of this, and so figuring out the exact recipes that would incorporate this is outside the scope of this paper.

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