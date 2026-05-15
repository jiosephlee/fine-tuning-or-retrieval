"""Shared prompt helpers for MCQA probes."""

MCQA_5SHOT_PROMPT_COLUMN = "formatted_question_5shot"

MCQA_5SHOT_DEMONSTRATION_BLOCK = """Question: Which city is the capital of France?
(A) Paris
(B) Berlin
(C) Madrid
(D) Rome
(E) Lisbon
Answer: (A)

Question: Which planet is the largest in the Solar System?
(A) Earth
(B) Jupiter
(C) Saturn
(D) Venus
(E) Mars
Answer: (B)

Question: What is the chemical formula for water?
(A) CO2
(B) O2
(C) H2O
(D) NaCl
(E) CH4
Answer: (C)

Question: What process do plants use to convert sunlight into chemical energy?
(A) Respiration
(B) Fermentation
(C) Evaporation
(D) Photosynthesis
(E) Condensation
Answer: (D)

Question: What is the primary function of red blood cells?
(A) Transmit nerve signals
(B) Produce bile
(C) Filter lymph
(D) Digest proteins
(E) Carry oxygen
Answer: (E)"""


def build_mcqa_5shot_prompt(formatted_question: str) -> str:
    """Return the fixed 5-shot prompt followed by one MCQA question."""
    return f"{MCQA_5SHOT_DEMONSTRATION_BLOCK}\n\nQuestion: {formatted_question.strip()}"


def validate_mcqa_5shot_demonstrations() -> None:
    """Fail if the fixed demo block is not exactly five question/answer pairs."""
    question_count = MCQA_5SHOT_DEMONSTRATION_BLOCK.count("Question:")
    answer_count = MCQA_5SHOT_DEMONSTRATION_BLOCK.count("Answer:")
    expected_answers = ["Answer: (A)", "Answer: (B)", "Answer: (C)", "Answer: (D)", "Answer: (E)"]
    missing_answers = [
        answer for answer in expected_answers
        if answer not in MCQA_5SHOT_DEMONSTRATION_BLOCK
    ]
    if question_count != 5 or answer_count != 5 or missing_answers:
        raise ValueError(
            "Invalid MCQA 5-shot demonstration block: "
            f"questions={question_count}, answers={answer_count}, "
            f"missing_answers={missing_answers}"
        )
