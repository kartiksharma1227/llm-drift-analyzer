"""
Logical and mathematical reasoning evaluator for LLM Drift Analyzer.

Evaluates correctness of reasoning chains, mathematical calculations,
and logical deductions in LLM responses.

Supports both OpenAI and Ollama evaluation backends.
"""

from typing import Optional

from llm_drift_analyzer.evaluators.base_evaluator import BaseEvaluator


class LogicalMathEvaluator(BaseEvaluator):
    """
    Evaluates logical and mathematical reasoning in LLM responses.

    Score Range (0-3):
        0: Wrong answer with flawed logic
        1: Partially correct, significant reasoning gaps
        2: Correct answer with minor reasoning gaps
        3: Correct answer with clear, valid reasoning chain
    """

    @property
    def metric_name(self) -> str:
        return "logical_mathematical_reasoning"

    @property
    def score_range(self) -> tuple:
        return (0, 3)

    def _build_evaluation_prompt(
        self,
        prompt: str,
        response: str,
        reference_answer: Optional[str] = None,
        **kwargs
    ) -> str:
        reference_section = ""
        if reference_answer:
            reference_section = f"""
Expected/Reference Answer:
{reference_answer}
"""

        return f"""Rate the logical and mathematical reasoning quality of this response on a scale of 0-3.

Problem/Question: {prompt}

Response: {response}
{reference_section}
Scoring Criteria:
3 = Correct answer with clear, step-by-step valid reasoning chain
2 = Correct final answer but with minor gaps or shortcuts in reasoning
1 = Partially correct — either wrong answer with some valid steps, or right answer with flawed logic
0 = Wrong answer with fundamentally flawed reasoning or no reasoning shown

Evaluation Guidelines:
- Is the final answer/conclusion correct?
- Are intermediate steps logically valid?
- Are mathematical calculations accurate?
- Is the reasoning chain clear and complete?
- Does the response show understanding of the underlying concept?
- For word problems: are the quantities and relationships correctly identified?

Provide only the numeric score (0, 1, 2, or 3)."""
