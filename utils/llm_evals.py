import json
from utils.prompts.llm_as_judge import prompt as judge_prompt
import utils.utils as utils
import copy

def parse_judge_response(response_text: str):
    """
    Parses the JSON response from the LLM judge.
    """
    try:
        data = json.loads(response_text)
        feedback = data.get("feedback")
        accuracy = data.get("accuracy")
        comprehensiveness = data.get("comprehensiveness")
        relevance = data.get("relevance")
        depth = data.get("depth")

        # Convert scores to integers and validate range
        scores = {}
        for score_name, score_value in [("accuracy", accuracy), ("comprehensiveness", comprehensiveness), ("relevance", relevance), ("depth", depth)]:
            if score_value is not None:
                try:
                    score_int = int(score_value)
                    if 1 <= score_int <= 10:
                        scores[score_name] = score_int
                    else:
                        scores[score_name] = None
                except (ValueError, TypeError):
                    scores[score_name] = None
            else:
                scores[score_name] = None

        # Calculate total score if all individual scores are valid
        if all(score is not None for score in scores.values()):
            total_score = sum(scores.values())
        else:
            total_score = None

        return {
            "feedback": feedback,
            "accuracy": scores["accuracy"],
            "comprehensiveness": scores["comprehensiveness"],
            "relevance": scores["relevance"],
            "depth": scores["depth"],
            "score": total_score
        }
    except (json.JSONDecodeError, IndexError, AttributeError) as e:
        return {
            "feedback": f"Failed to parse response: {e}\nResponse:\n{response_text}",
            "accuracy": None,
            "comprehensiveness": None,
            "relevance": None,
            "depth": None,
            "score": None
        }

def evaluate_response(question: str, response: str, reference_answer: str, judge_model: str = "gpt-4o-mini"):
    """
    Evaluates a response using an LLM as a judge.
    """
    local_judge_prompt = copy.deepcopy(judge_prompt)
    local_judge_prompt['user'] = local_judge_prompt["user"].format(
        question=question,
        answer=response,
        reference_answer=reference_answer
    )

    raw_response = utils.query_llm(local_judge_prompt, system_prompt_included=True, model=judge_model)

    return parse_judge_response(raw_response)
