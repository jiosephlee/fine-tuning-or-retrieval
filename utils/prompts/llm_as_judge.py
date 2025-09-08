prompt = {}
prompt["system"] = """### Task Description:
A question, an answer to evaluate, and a score rubric representing evaluation criteria are given.
1. Write detailed feedback that assesses the quality of the answer strictly based on the given score rubric across four dimensions.
2. After writing feedback, provide scores for each dimension (1-10 scale each) with clear labels.
3. The output format should be JSON with the following structure: {"feedback": "(write overall feedback)", "accuracy": (score 1-10), "comprehensiveness": (score 1-10), "relevance": (score 1-10), "depth": (score 1-10)}
4. Please do not generate any other opening, closing, and explanations."""
prompt["user"] = """### Question to evaluate:
{question}
### Answer to evaluate:
{answer}
### Reference Answer (Perfect score: 40 total):
{reference_answer}
### Score Rubrics:
[Accuracy (1-10): Is everything the answer says correct and factual?]
1-2: Largely incorrect, inaccurate, and not factual. Fundamental misunderstanding leading to erroneous information.
3-4: Partially correct but contains significant inaccuracies or factual errors.
5-6: Generally correct and factual but may include minor inaccuracies.
7-8: Mostly correct, accurate, and factual with minimal inaccuracies.
9-10: Consistently correct, accurate, and entirely factual without any errors.

[Comprehensiveness (1-10): Amount of relevant information provided]
1-2: Very little relevant information, mostly superficial coverage.
3-4: Some relevant information but lacks substance or misses key points.
5-6: Adequate coverage of main points with reasonable detail.
7-8: Good comprehensive coverage with substantial detail.
9-10: Extensive comprehensive coverage of all important aspects thoroughly.

[Relevance and Coherence(1-10): How well does the answer stay on topic, penalized for irrelevant content]
1-2: Mostly irrelevant information, confusing, lacks focus on the question.
3-4: Some relevant parts but includes significant irrelevant information that detracts.
5-6: Generally relevant with minor off-topic details.
7-8: Well-focused and relevant with minimal irrelevant content.
9-10: Highly relevant and coherent, all information directly addresses the question.

[Depth (1-10): Does the answer display deep understanding of the information?]
1-2: Superficial understanding, lacks insight or nuance.
3-4: Basic understanding but misses deeper implications or connections.
5-6: Good understanding with some insight into the topic.
7-8: Strong understanding with clear grasp of complexities and nuances.
9-10: Deep, sophisticated understanding with excellent insight and analysis.

### Response:"""