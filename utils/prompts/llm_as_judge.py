prompt = {}
prompt["system"] = """### Task Description:
A question, an answer to evaluate, and a score rubric representing evaluation criteria are given.
1. Write detailed feedback that assesses the quality of the answer strictly based on the given score rubric, not evaluating in general.
2. After writing feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should be JSON with the following structure: {"feedback": "(write feedback for criteria)", "score": (an integer number between 1 and 5)}
4. Please do not generate any other opening, closing, and explanations."""
prompt["user"] = """### Question to evaluate:
{question}
### Answer to evaluate:
{answer}
### Reference Answer (Score 5):
{reference_answer}
### Score Rubrics:
[Is the answer correct, accurate, and factual?]
Score 1: The answer is largely incorrect, inaccurate, and not factual. It demonstrates a fundamental misunderstanding of the question or topic, leading to irrelevant or completely erroneous information.
Score 2: The answer is partially correct but contains significant inaccuracies or factual errors. It shows some understanding of the question or topic but fails to provide a fully accurate or reliable answer.
Score 3: The answer is generally correct and factual but may include minor inaccuracies or lack of detail. It shows a good understanding of the question or topic but may miss some nuances or specific information.
Score 4: The answer is mostly correct, accurate, and factual. It demonstrates a strong understanding of the question or topic, with only minimal inaccuracies or omissions that do not significantly detract from the overall quality of the answer.
Score 5: The answer is consistently correct, accurate, and entirely factual. It reflects a comprehensive understanding of the question or topic, providing detailed, precise, and fully reliable information without any inaccuracies or omissions.
### Response:"""