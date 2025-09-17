FACT_PROBE_CLOZE_PROMPT_SYSTEM = r"""You will be given a question and its corresponding answer, along with the original sentence they were extracted from. Your task is to turn the question and answer into a single, self-contained statement where the answer appears only at the very end.

### Instructions
- Incorporate the answer naturally.
- The answer must be at the *very end* of the statement. The answer may sound awkward at the end; in this case, adjust the answer to make it more natural. E.g. "Q: What benchmark does the author evaluate? A: GSM8K" should be converted to "The authors evaluate on the GSM8K benchmark" in which we include "benchmark" into the answer.
- The paraphrasing should be natural and flow well. 
- Avoid awkward repetition. Avoid phrases like "and the last model, specifically X". Instead just directly say "and X.". Avoid large descriptions placed in between dashes.
- Ensure the statement is grammatically correct and well-written.
- Preserve ALL information and context from the original question when converting to cloze form. Feel free to use several sentences to preserve the context.
    - Again, please preserve all of the context that was present in the original question. Feel free to use several sentences to preserve the context.
- Do not leak the answer in the middle of the statement. The answer should only appear once, at the very end.
- Double check that the statement flows well and consider restructuring the statement if it doesn't.

### Demonstration
Question: Within the DPO theoretical section of the paper '...', the projection operator f normalizes the reward function by subtracting a term involving the policy’s partition function. Which function is used for this normalization?
Answer: logarithm of the partition function

Statement: Within the DPO theoretical section of the paper '...', the projection operator f normalizes the reward function by subtracting a term involving the policy’s partition function. The function used for this normalization is the logarithm of the partition function.

### Output Format
Return a single JSON object with two keys: "answer" (if this was refined, replace the old answer with the new one) and "statement".
e.g. {"answer": "...", "statement": "..."}"""

FACT_PROBE_CLOZE_PROMPT_SYSTEM_TWO = r"""You will be given a question and its corresponding answer. Your task is to convert the question and answer into a cloze statement where the answer appears only at the very end.

### Instructions
- The statement should not be a question, but read as a declarative sentence.
- The statement should still *serve* as a question but in "cloze" or "fill in the blank" format.
- Preserve the phrasing in the question as much as possible.
- Incorporate the answer naturally.
- The answer must be at the *very end* of the statement. The answer may sound awkward at the end; in this case, adjust the answer to make it more natural. E.g. "Q: What benchmark does the author evaluate? A: GSM8K" should be converted to "The authors evaluate on the GSM8K benchmark" in which we include "benchmark" into the answer.
- Avoid awkward repetition. Avoid phrases like "and the last model, specifically X". Instead just directly say "and X.". Avoid large descriptions placed in between dashes.
- Ensure the statement is grammatically correct and well-written. 
- Feel free to break up the statement into multiple sentences to help with flow.
- Do not leak the answer in the middle of the statement. The answer should only appear once, at the very end.

### Demonstration
Question: Within the DPO theoretical section of the paper '...', the projection operator f normalizes the reward function by subtracting a term involving the policy’s partition function. Which function is used for this normalization?
Answer: logarithm of the partition function

Statement: Within the DPO theoretical section of the paper '...', the projection operator f normalizes the reward function by subtracting a term involving the policy’s partition function. The function used for this normalization is the logarithm of the partition function.

### Output Format
Return a single JSON object with two keys: "answer" (if this was refined, replace the old answer with the new one) and "statement" (which includes the answer at the end. Make sure to not have any "___" to emulate the cloze format).
e.g. {"answer": "...", "statement": "..."}"""
