"""
Base evaluator interface for LLM response scoring.

This module defines the abstract base class for evaluators
that score LLM responses on various quality metrics.

Supports both OpenAI (GPT-4) and Ollama (local models) as evaluation backends.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union
import re

import openai
import requests

from llm_drift_analyzer.utils.logger import get_logger


class BaseEvaluator(ABC):
    """
    Abstract base class for LLM response evaluators.

    Evaluators score LLM responses on specific quality metrics
    using either OpenAI GPT-4 or Ollama local models as the evaluation backend.

    Attributes:
        evaluator_provider: Provider for evaluation ("openai" or "ollama").
        evaluator_model: Model used for evaluation.
        temperature: Temperature for evaluation queries.
        openai_client: OpenAI client (if using OpenAI provider).
        ollama_base_url: Ollama server URL (if using Ollama provider).

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

        >>> # Use GPT-4 (default, costs money)
        >>> evaluator = MyEvaluator(openai_api_key="sk-...")

        >>> # Use Ollama (free, runs locally)
        >>> evaluator = MyEvaluator(
        ...     evaluator_provider="ollama",
        ...     evaluator_model="llama3",
        ...     ollama_base_url="http://localhost:11434"
        ... )
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        evaluator_model: str = "gpt-4",
        evaluator_provider: str = "openai",
        ollama_base_url: str = "http://localhost:11434",
        temperature: float = 0.1
    ):
        """
        Initialize the evaluator.

        Args:
            openai_api_key: OpenAI API key (required if provider is "openai").
            evaluator_model: Model to use for evaluation.
                For OpenAI: "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"
                For Ollama: "llama3", "mistral", "mixtral", etc.
            evaluator_provider: Provider for evaluation ("openai" or "ollama").
            ollama_base_url: Ollama server URL (only used if provider is "ollama").
            temperature: Temperature for evaluation queries.

        Raises:
            ValueError: If provider is "openai" but no API key provided.
        """
        self.evaluator_provider = evaluator_provider.lower()
        self.evaluator_model = evaluator_model
        self.temperature = temperature
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self._logger = get_logger(f"evaluators.{self.metric_name}")

        if self.evaluator_provider == "openai":
            if not openai_api_key:
                raise ValueError(
                    "OpenAI API key is required when using 'openai' evaluator provider. "
                    "Either provide an API key or use evaluator_provider='ollama' for free local evaluation."
                )
            self.openai_client = openai.OpenAI(api_key=openai_api_key)
        elif self.evaluator_provider == "ollama":
            # Verify Ollama is accessible
            self._verify_ollama_connection()
            self.openai_client = None
        else:
            raise ValueError(
                f"Unknown evaluator provider: {evaluator_provider}. "
                "Supported providers: 'openai', 'ollama'"
            )

        self._logger.debug(
            f"Initialized {self.metric_name} evaluator with provider={self.evaluator_provider}, "
            f"model={self.evaluator_model}"
        )

    def _verify_ollama_connection(self) -> None:
        """
        Verify that Ollama server is accessible.

        Raises:
            ConnectionError: If Ollama server is not reachable.
        """
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                raise ConnectionError(f"Ollama returned status {response.status_code}")
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(
                f"Cannot connect to Ollama server at {self.ollama_base_url}. "
                "Make sure Ollama is installed and running (`ollama serve`). "
                f"Original error: {e}"
            )

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
        Build the evaluation prompt for the judge model.

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

        Uses the configured evaluation provider (OpenAI or Ollama) to
        evaluate the response based on the specific metric.

        Args:
            prompt: Original prompt sent to the LLM.
            response: LLM's response to evaluate.
            **kwargs: Additional context for evaluation.

        Returns:
            int: Score within the defined score_range.

        Example:
            >>> evaluator = InstructionEvaluator(
            ...     evaluator_provider="ollama",
            ...     evaluator_model="llama3"
            ... )
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
            self._logger.debug(
                f"Evaluating {self.metric_name} with {self.evaluator_provider}/{self.evaluator_model}"
            )

            if self.evaluator_provider == "openai":
                score_text = self._evaluate_with_openai(evaluation_prompt)
            else:  # ollama
                score_text = self._evaluate_with_ollama(evaluation_prompt)

            # Parse score from response
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

    def _evaluate_with_openai(self, evaluation_prompt: str) -> str:
        """
        Evaluate using OpenAI API.

        Args:
            evaluation_prompt: The formatted evaluation prompt.

        Returns:
            str: Raw response text from the model.
        """
        eval_response = self.openai_client.chat.completions.create(
            model=self.evaluator_model,
            messages=[{"role": "user", "content": evaluation_prompt}],
            max_tokens=50,
            temperature=self.temperature
        )
        return eval_response.choices[0].message.content.strip()

    def _evaluate_with_ollama(self, evaluation_prompt: str) -> str:
        """
        Evaluate using Ollama API.

        Args:
            evaluation_prompt: The formatted evaluation prompt.

        Returns:
            str: Raw response text from the model.
        """
        # Add a system prompt to help local models focus on returning just the score
        system_prompt = (
            "You are an evaluation assistant. When asked to score something, "
            "respond with ONLY the numeric score. Do not include any explanation, "
            "just output the single number."
        )

        payload = {
            "model": self.evaluator_model,
            "prompt": evaluation_prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": 20,  # Short response - just need a number
            }
        }

        response = requests.post(
            f"{self.ollama_base_url}/api/generate",
            json=payload,
            timeout=60
        )

        if response.status_code != 200:
            raise RuntimeError(f"Ollama API error: {response.text}")

        result = response.json()
        return result.get("response", "").strip()

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
            "evaluator_provider": self.evaluator_provider,
            "evaluator_model": self.evaluator_model,
            "description": self.__class__.__doc__,
        }

    def get_provider_info(self) -> Dict[str, str]:
        """
        Get information about the evaluation provider.

        Returns:
            Dict[str, str]: Provider name, model, and cost info.
        """
        cost_info = "API costs apply" if self.evaluator_provider == "openai" else "Free (local)"
        return {
            "provider": self.evaluator_provider,
            "model": self.evaluator_model,
            "cost": cost_info,
        }
