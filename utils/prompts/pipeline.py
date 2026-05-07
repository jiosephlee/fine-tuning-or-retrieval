FACT_PROBE_CLOZE_PROMPT_SYSTEM = r"""You will be given a question and its corresponding answer, along with the original sentence they were extracted from. Your task is to convert the question and answer into a single, self-contained statement where the answer appears only at the very end.

### Instructions
- The statement should not be a question, but read as a declarative sentence.
- The statement should still *serve* as a question but in "cloze" or "fill in the blank" format.
- Preserve the phrasing in the question as much as possible.
- Incorporate the answer naturally.
- The answer must be at the *very end* of the statement. Do NOT modify the answer — it must remain exactly as given (a verbatim substring of the original sentence). Instead, restructure the statement so the answer fits naturally at the end. E.g. "Q: What benchmark does the author evaluate? A: GSM8K" should be converted to "The authors evaluate on GSM8K" — do not add extra words to the answer.
- Avoid awkward repetition. Avoid phrases like "and the last model, specifically X". Instead just directly say "and X.". Avoid large descriptions placed in between dashes.
- Preserve ALL information and context from the original question when converting to cloze form. Feel free to use multiple sentences to preserve the context and help with flow.
    - Again, please preserve all of the context that was present in the original question. 
    - Again, feel free to use multiple sentences to preserve the context to avoid overly verbose sentences.
- Do not leak the answer in the middle of the statement. The answer should only appear once, at the very end.
- Double check that the statement flows well and restructure the statement if it doesn’t.

### Demonstration
Question: Within the theoretical section of the paper "...", the projection operator f normalizes the reward function by subtracting a term involving the policy’s partition function. Which function is used for this normalization?
Answer: logarithm of the partition function

Statement: Within the theoretical section of the paper "...", the projection operator f normalizes the reward function by subtracting a term involving the policy’s partition function. The function used for this normalization is the logarithm of the partition function.

### Output Format
Return a single JSON object with two keys: "answer" (must be unchanged from the input) and "statement".
e.g. {"answer": "...", "statement": "..."}"""

FACT_PROBE_CLOZE_PROMPT_SYSTEM_MEDICAL = r"""You will be given a question and its corresponding answer from a medical case report. Convert them into a single declarative cloze statement where the answer appears only at the very end.

### Instructions
- The statement should be declarative (not a question) and should read naturally.
- Keep all clinically relevant context from the question (timepoint, test, treatment phase, diagnosis context).
- Preserve medical detail exactly (units, abbreviations, value formatting, day numbers, qualifiers).
- Do NOT infer, correct, or add medical facts beyond the provided question/answer.
- Keep the answer unchanged and place it only once, at the very end of the statement.
- Do not leak the answer earlier in the statement.
- If the question wording is awkward, rewrite for clarity while preserving meaning.

### Demonstration
Question: According to the case report "...", what was the patient's serum sodium level on initial laboratory testing?
Answer: 124 mmol/L

Statement: According to the case report "...", the patient's serum sodium level on initial laboratory testing was 124 mmol/L.

### Output Format
Return a single JSON object with two keys: "answer" and "statement".
"answer" must be exactly unchanged from input.
e.g. {"answer": "...", "statement": "..."}"""

FACT_PROBE_CLOZE_PROMPT_SYSTEM_LEGAL = r"""You will be given a question and its corresponding answer from a legal opinion or case text. Convert them into a single declarative cloze statement where the answer appears only at the very end.

### Instructions
- The statement should be declarative and should read naturally.
- Keep the legal context needed for disambiguation, such as the court, procedural posture, statutory or doctrinal setting, issue framing, and holding context.
- Preserve legal detail exactly, including party names, statute sections, doctrine labels, dates, citation-like tokens, and qualifiers.
- Do NOT infer, reinterpret, soften, or add legal conclusions beyond the provided question and answer.
- Keep the answer unchanged and place it only once, at the very end of the statement.
- Do not leak the answer earlier in the statement.
- If the question wording is awkward, rewrite for clarity while preserving meaning.
- If preserving the necessary context makes a single sentence too dense, you may use two short sentences, but the final words must still be the answer.

### Demonstration
Question: In the opinion "...", what standard of review did the court apply to the district court's factual findings?
Answer: clear error

Statement: In the opinion "...", the standard of review the court applied to the district court's factual findings was clear error.

### Output Format
Return a single JSON object with two keys: "answer" and "statement".
"answer" must be exactly unchanged from input.
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
Question: Within the theoretical section of the paper '...', the projection operator f normalizes the reward function by subtracting a term involving the policy’s partition function. Which function is used for this normalization?
Answer: logarithm of the partition function

Statement: Within the theoretical section of the paper '...', the projection operator f normalizes the reward function by subtracting a term involving the policy’s partition function. The function used for this normalization is the logarithm of the partition function.

### Output Format
Return a single JSON object with two keys: "answer" (if this was refined, replace the old answer with the new one) and "statement" (which includes the answer at the end. Make sure to not have any "___" to emulate the cloze format).
e.g. {"answer": "...", "statement": "..."}"""
