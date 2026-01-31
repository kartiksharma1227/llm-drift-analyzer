"""
Instruction adherence evaluator for LLM Drift Analyzer.

This module provides evaluation of how well LLM responses
follow the given instructions in the prompt.
"""

from typing import Optional

from llm_drift_analyzer.evaluators.base_evaluator import BaseEvaluator


class InstructionEvaluator(BaseEvaluator):
    """
    Evaluates instruction adherence in LLM responses.

    Uses GPT-4 to score how well a response follows the
    instructions given in the original prompt.

    Score Range (0-3):
        0: Does not follow instructions at all
        1: Partially follows instructions, missing key requirements
        2: Mostly follows instructions with minor deviations
        3: Perfectly follows all instructions

    Example:
        >>> evaluator = InstructionEvaluator(api_key="sk-...")
        >>> score = evaluator.evaluate(
        ...     prompt="Write exactly 3 bullet points about AI",
        ...     response="• Point 1\\n• Point 2\\n• Point 3"
        ... )
        >>> print(score)
        3
    """

    @property
    def metric_name(self) -> str:
        """Return metric name."""
        return "instruction_adherence"

    @property
    def score_range(self) -> tuple:
        """Return valid score range (0-3)."""
        return (0, 3)

    def _build_evaluation_prompt(
        self,
        prompt: str,
        response: str,
        expected_format: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Build evaluation prompt for instruction adherence.

        Args:
            prompt: Original prompt with instructions.
            response: LLM response to evaluate.
            expected_format: Optional description of expected format.
            **kwargs: Additional context (unused).

        Returns:
            str: Formatted evaluation prompt.
        """
        format_hint = ""
        if expected_format:
            format_hint = f"\n\nExpected Format: {expected_format}"

        return f"""Rate how well this response follows the given instructions on a scale of 0-3.

Instructions: {prompt}{format_hint}

Response: {response}

Scoring Criteria:
3 = Perfect adherence to all instructions (format, length, content requirements)
2 = Good adherence with minor deviations (small format issues or slight over/under on length)
1 = Poor adherence, missing key requirements (wrong format, significantly off on length, missing required elements)
0 = No adherence to instructions (completely ignores format, length, or content requirements)

Evaluation Guidelines:
- Check if the response follows any specified format (bullet points, numbered lists, paragraphs)
- Verify length constraints if specified (word count, character count, number of items)
- Assess if all required content elements are present
- Consider if the response directly addresses what was asked

Provide only the numeric score (0, 1, 2, or 3)."""

    def evaluate_with_details(
        self,
        prompt: str,
        response: str,
        expected_format: Optional[str] = None
    ) -> dict:
        """
        Evaluate with detailed feedback.

        Provides both a score and qualitative feedback about
        the instruction adherence.

        Args:
            prompt: Original prompt with instructions.
            response: LLM response to evaluate.
            expected_format: Optional description of expected format.

        Returns:
            dict: Score and detailed evaluation feedback.

        Example:
            >>> result = evaluator.evaluate_with_details(
            ...     prompt="Write 3 bullet points",
            ...     response="Here are some points:\\n• One\\n• Two"
            ... )
            >>> print(result["score"], result["feedback"])
        """
        score = self.evaluate(prompt, response, expected_format=expected_format)

        # Get detailed feedback
        feedback_prompt = f"""Evaluate this response for instruction adherence and provide brief feedback.

Instructions: {prompt}

Response: {response}

Score: {score}/3

Provide a 1-2 sentence explanation of why this score was given.
Focus on specific instruction requirements that were met or missed."""

        try:
            feedback_response = self.openai_client.chat.completions.create(
                model=self.evaluator_model,
                messages=[{"role": "user", "content": feedback_prompt}],
                max_tokens=100,
                temperature=self.temperature
            )
            feedback = feedback_response.choices[0].message.content.strip()
        except Exception:
            feedback = "Feedback unavailable"

        return {
            "score": score,
            "feedback": feedback,
            "metric": self.metric_name,
        }
