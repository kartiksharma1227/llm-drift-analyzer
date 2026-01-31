"""
Base evaluator interface for LLM response scoring.

This module defines the abstract base class for evaluators
that score LLM responses on various quality metrics.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

import openai

from llm_drift_analyzer.utils.logger import get_logger


class BaseEvaluator(ABC):
    """
    Abstract base class for LLM response evaluators.

    Evaluators score LLM responses on specific quality metrics
    using GPT-4 as the evaluation model.

    Attributes:
        openai_client: OpenAI client for evaluation queries.
        evaluator_model: Model used for evaluation (default: gpt-4).
        temperature: Temperature for evaluation queries.

    Example:
        >>> class MyEvaluator(BaseEvaluator):
        ...     @property
        ...     def metric_name(self):
        ...         return "my_metric"
        ...
        ...     @property
        ...     def score_range(self):
        ...         return (0, 5)
        ...
        ...     def _build_evaluation_prompt(self, prompt, response):
        ...         return f"Evaluate this response: {response}"
    """

    def __init__(
        self,
        openai_api_key: str,
        evaluator_model: str = "gpt-4",
        temperature: float = 0.1
    ):
        """
        Initialize the evaluator.

        Args:
            openai_api_key: OpenAI API key for evaluation.
            evaluator_model: Model to use for evaluation.
            temperature: Temperature for evaluation queries.
        """
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.evaluator_model = evaluator_model
        self.temperature = temperature
        self._logger = get_logger(f"evaluators.{self.metric_name}")

    @property
    @abstractmethod
    def metric_name(self) -> str:
        """
        Name of the metric being evaluated.

        Returns:
            str: Metric name (e.g., "instruction_adherence").
        """
        pass

    @property
    @abstractmethod
    def score_range(self) -> tuple:
        """
        Valid score range for this metric.

        Returns:
            tuple: (min_score, max_score).
        """
        pass

    @abstractmethod
    def _build_evaluation_prompt(
        self,
        prompt: str,
        response: str,
        **kwargs
    ) -> str:
        """
        Build the evaluation prompt for GPT-4.

        Args:
            prompt: Original prompt sent to the LLM.
            response: LLM's response to evaluate.
            **kwargs: Additional context for evaluation.

        Returns:
            str: Formatted evaluation prompt.
        """
        pass

    def evaluate(
        self,
        prompt: str,
        response: str,
        **kwargs
    ) -> int:
        """
        Evaluate an LLM response and return a score.

        Uses GPT-4 to evaluate the response based on the
        specific metric implemented by the subclass.

        Args:
            prompt: Original prompt sent to the LLM.
            response: LLM's response to evaluate.
            **kwargs: Additional context for evaluation.

        Returns:
            int: Score within the defined score_range.

        Example:
            >>> evaluator = InstructionEvaluator(api_key="...")
            >>> score = evaluator.evaluate(
            ...     prompt="Write 3 bullet points",
            ...     response="• Point 1\\n• Point 2\\n• Point 3"
            ... )
            >>> print(score)
            3
        """
        min_score, max_score = self.score_range

        evaluation_prompt = self._build_evaluation_prompt(prompt, response, **kwargs)

        try:
            self._logger.debug(f"Evaluating {self.metric_name}")

            eval_response = self.openai_client.chat.completions.create(
                model=self.evaluator_model,
                messages=[{"role": "user", "content": evaluation_prompt}],
                max_tokens=10,
                temperature=self.temperature
            )

            # Parse score from response
            score_text = eval_response.choices[0].message.content.strip()
            score = self._parse_score(score_text)

            # Clamp to valid range
            score = max(min_score, min(max_score, score))

            self._logger.debug(f"Score: {score}")
            return score

        except Exception as e:
            self._logger.warning(
                f"Evaluation failed for {self.metric_name}: {e}. "
                f"Using default score."
            )
            return self._get_default_score()

    def _parse_score(self, text: str) -> int:
        """
        Parse score from evaluation response text.

        Extracts the first integer found in the text.

        Args:
            text: Response text from the evaluator.

        Returns:
            int: Parsed score.

        Raises:
            ValueError: If no valid score found.
        """
        # Try to extract just the number
        import re
        numbers = re.findall(r'\d+', text)
        if numbers:
            return int(numbers[0])

        raise ValueError(f"Could not parse score from: {text}")

    def _get_default_score(self) -> int:
        """
        Get default score when evaluation fails.

        Returns:
            int: Middle of the score range.
        """
        min_score, max_score = self.score_range
        return (min_score + max_score) // 2

    def get_metric_description(self) -> Dict[str, Any]:
        """
        Get description of the metric for documentation.

        Returns:
            Dict[str, Any]: Metric metadata including name, range, and description.
        """
        min_score, max_score = self.score_range
        return {
            "name": self.metric_name,
            "min_score": min_score,
            "max_score": max_score,
            "description": self.__class__.__doc__,
        }
