import json
from utils.prompts.llm_as_judge import prompt as judge_prompt
import utils.utils as utils
import re

def parse_judge_response(response_text: str):
    """
    Parses the JSON response from the LLM judge.
    """
    try:
        data = json.loads(response_text)
        feedback = data.get("feedback")
        score = data.get("score")

        if score is not None:
            try:
                score = int(score)
                if not (1 <= score <= 5):
                    score = None  # Invalid score range
            except (ValueError, TypeError):
                score = None # Score is not an integer

        return {"feedback": feedback, "score": score}
    except (json.JSONDecodeError, IndexError, AttributeError) as e:
        return {"feedback": f"Failed to parse response: {e}\nResponse:\n{response_text}", "score": None}

def evaluate_response(question: str, response: str, reference_answer: str):
    """
    Evaluates a response using an LLM as a judge.
    """
    judge_prompt['user'] = judge_prompt["user"].format(
        question=question,
        answer=response,
        reference_answer=reference_answer
    )

    raw_response = utils.query_llm(judge_prompt, system_prompt_included=True, model = "gpt-5-mini", reasoning_effort = "low"
    )

    return parse_judge_response(raw_response)
