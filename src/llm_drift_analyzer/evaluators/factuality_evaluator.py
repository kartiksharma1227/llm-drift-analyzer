"""
Factuality evaluator for LLM Drift Analyzer.

This module provides evaluation of factual accuracy in LLM responses,
detecting hallucinations and verifying claims.

Supports both OpenAI (GPT-4) and Ollama (local models) as evaluation backends.
"""

from typing import Optional, List

import requests

from llm_drift_analyzer.evaluators.base_evaluator import BaseEvaluator


class FactualityEvaluator(BaseEvaluator):
    """
    Evaluates factual accuracy in LLM responses.

    Uses GPT-4 or a local Ollama model to assess whether the response
    contains factually accurate information or hallucinations.

    Score Range (0-2):
        0: Contains significant factual errors or hallucinations
        1: Mostly factual with minor errors or unverifiable claims
        2: Completely factual with accurate information

    Example:
        >>> # Using GPT-4 (costs money)
        >>> evaluator = FactualityEvaluator(openai_api_key="sk-...")
        >>> score = evaluator.evaluate(
        ...     prompt="What year did WWII end?",
        ...     response="WWII ended in 1945."
        ... )
        >>> print(score)
        2

        >>> # Using Ollama (free, local)
        >>> evaluator = FactualityEvaluator(
        ...     evaluator_provider="ollama",
        ...     evaluator_model="llama3"
        ... )
    """

    @property
    def metric_name(self) -> str:
        """Return metric name."""
        return "factuality"

    @property
    def score_range(self) -> tuple:
        """Return valid score range (0-2)."""
        return (0, 2)

    def _build_evaluation_prompt(
        self,
        prompt: str,
        response: str,
        reference_answer: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Build evaluation prompt for factuality assessment.

        Args:
            prompt: Original prompt asking for information.
            response: LLM response to evaluate.
            reference_answer: Optional known correct answer for comparison.
            **kwargs: Additional context (unused).

        Returns:
            str: Formatted evaluation prompt.
        """
        reference_section = ""
        if reference_answer:
            reference_section = f"""
Reference Answer (for comparison):
{reference_answer}
"""

        return f"""Evaluate the factual accuracy of this response on a scale of 0-2.

Question/Prompt: {prompt}

Response to evaluate: {response}
{reference_section}
Scoring Criteria:
2 = Completely factual - All stated facts are accurate, dates are correct, no hallucinations
1 = Mostly factual - Contains minor errors, imprecise dates, or unverifiable claims
0 = Contains significant errors - Major factual mistakes, invented information, or hallucinations

Evaluation Guidelines:
- Check dates, numbers, and specific claims for accuracy
- Look for invented names, events, or statistics
- Verify that cause-effect relationships are correct
- Note any claims that contradict well-established facts
- Consider whether uncertain information is presented as fact

Focus on verifiable facts, not opinions or interpretations.

Provide only the numeric score (0, 1, or 2)."""

    def _query_evaluator(self, prompt_text: str, max_tokens: int = 300) -> str:
        """
        Query the evaluator model (works with both OpenAI and Ollama).

        Args:
            prompt_text: The prompt to send to the evaluator.
            max_tokens: Maximum tokens in response.

        Returns:
            str: Model's response text.
        """
        if self.evaluator_provider == "openai":
            response = self.openai_client.chat.completions.create(
                model=self.evaluator_model,
                messages=[{"role": "user", "content": prompt_text}],
                max_tokens=max_tokens,
                temperature=self.temperature
            )
            return response.choices[0].message.content.strip()
        else:  # ollama
            payload = {
                "model": self.evaluator_model,
                "prompt": prompt_text,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": max_tokens,
                }
            }
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json=payload,
                timeout=60
            )
            if response.status_code != 200:
                raise RuntimeError(f"Ollama API error: {response.text}")
            return response.json().get("response", "").strip()

    def detect_hallucinations(
        self,
        prompt: str,
        response: str
    ) -> dict:
        """
        Detect potential hallucinations in the response.

        Analyzes the response for statements that may be
        fabricated or factually incorrect.

        Args:
            prompt: Original prompt.
            response: LLM response to analyze.

        Returns:
            dict: Detection results including score, concerns, and confidence.

        Example:
            >>> result = evaluator.detect_hallucinations(
            ...     prompt="Who invented the telephone?",
            ...     response="The telephone was invented by Thomas Edison in 1870."
            ... )
            >>> print(result["concerns"])
            ['Incorrect inventor: Alexander Graham Bell invented the telephone, not Edison']
        """
        detection_prompt = f"""Analyze this response for potential factual errors or hallucinations.

Question: {prompt}

Response: {response}

List any statements that appear to be:
1. Factually incorrect (wrong dates, names, events)
2. Fabricated information (invented statistics, fake quotes)
3. Misleading claims (out of context, overgeneralized)

Format your response as:
SCORE: [0, 1, or 2]
CONCERNS: [List each concern on a new line, or "None" if accurate]
CONFIDENCE: [HIGH, MEDIUM, or LOW - your confidence in the assessment]"""

        try:
            result_text = self._query_evaluator(detection_prompt, max_tokens=300)
            return self._parse_hallucination_result(result_text)

        except Exception as e:
            self._logger.warning(f"Hallucination detection failed: {e}")
            return {
                "score": 1,
                "concerns": ["Detection failed"],
                "confidence": "LOW",
            }

    def _parse_hallucination_result(self, text: str) -> dict:
        """
        Parse hallucination detection result.

        Args:
            text: Raw response from the evaluator.

        Returns:
            dict: Parsed result with score, concerns, and confidence.
        """
        result = {
            "score": 1,
            "concerns": [],
            "confidence": "MEDIUM",
        }

        lines = text.strip().split("\n")
        current_section = None

        for line in lines:
            line = line.strip()
            if line.startswith("SCORE:"):
                try:
                    score_text = line.replace("SCORE:", "").strip()
                    result["score"] = int(score_text[0])
                except (ValueError, IndexError):
                    pass
            elif line.startswith("CONCERNS:"):
                current_section = "concerns"
                concern = line.replace("CONCERNS:", "").strip()
                if concern and concern.lower() != "none":
                    result["concerns"].append(concern)
            elif line.startswith("CONFIDENCE:"):
                current_section = "confidence"
                conf = line.replace("CONFIDENCE:", "").strip().upper()
                if conf in ["HIGH", "MEDIUM", "LOW"]:
                    result["confidence"] = conf
            elif current_section == "concerns" and line and line.lower() != "none":
                # Continue adding concerns
                if line.startswith("-") or line.startswith("•"):
                    line = line[1:].strip()
                if line:
                    result["concerns"].append(line)

        return result

    def evaluate_with_reference(
        self,
        prompt: str,
        response: str,
        reference_answer: str
    ) -> dict:
        """
        Evaluate factuality against a known reference answer.

        Args:
            prompt: Original prompt.
            response: LLM response to evaluate.
            reference_answer: Known correct answer.

        Returns:
            dict: Score and comparison analysis.
        """
        score = self.evaluate(
            prompt, response, reference_answer=reference_answer
        )

        comparison_prompt = f"""Compare these two answers for factual alignment.

Question: {prompt}

Response: {response}

Reference Answer: {reference_answer}

Briefly note:
1. Key facts that match
2. Key facts that differ or are missing
3. Any additional claims not in reference"""

        try:
            analysis = self._query_evaluator(comparison_prompt, max_tokens=200)
        except Exception:
            analysis = "Comparison unavailable"

        return {
            "score": score,
            "analysis": analysis,
            "metric": self.metric_name,
            "evaluator_provider": self.evaluator_provider,
            "evaluator_model": self.evaluator_model,
        }
