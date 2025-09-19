prompt = {}
prompt["system"] = """### Task Description:
A question, an answer to evaluate, and a reference answer are given. The reference answer is just an example of a good response.
1. Write detailed feedback that assesses the quality of the answer. Your feedback should determine if the answer demonstrates an "accurate" and "sufficient" understanding of the topic, using your own knowledge to make the assessment.
2. After writing feedback, provide a score for "sufficiency" (0 or 1).
3. The output format should be JSON with the following structure: {"feedback": "(write overall feedback)", "sufficiency": (score 0 or 1)}
4. Please do not generate any other opening, closing, and explanations."""
prompt["user"] = """### Question to evaluate:
{question}
### Answer to evaluate:
{answer}
### Reference Answer (Example of a good answer):
{reference_answer}
### Score Rubrics:
[Sufficiency (0 or 1): Does the answer demonstrate an accurate and sufficient understanding of the topic?]
0: The answer is insufficient or inaccurate.
1: The answer is sufficient and accurate.

### Response:"""