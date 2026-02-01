"""
Tone and style evaluator for LLM Drift Analyzer.

This module provides evaluation of tone appropriateness and
stylistic consistency in LLM responses.

Supports both OpenAI (GPT-4) and Ollama (local models) as evaluation backends.
"""

from typing import Optional

import requests

from llm_drift_analyzer.evaluators.base_evaluator import BaseEvaluator


class ToneEvaluator(BaseEvaluator):
    """
    Evaluates tone and style appropriateness in LLM responses.

    Uses GPT-4 or a local Ollama model to assess whether the response
    has an appropriate tone for the context and maintains stylistic consistency.

    Score Range (0-2):
        0: Inappropriate tone for context (too casual/formal, inconsistent)
        1: Adequate tone with some inconsistencies
        2: Appropriate and consistent tone throughout

    Example:
        >>> # Using GPT-4 (costs money)
        >>> evaluator = ToneEvaluator(openai_api_key="sk-...")
        >>> score = evaluator.evaluate(
        ...     prompt="Explain gravity to a child",
        ...     response="Gravity is like a big invisible hand..."
        ... )
        >>> print(score)
        2

        >>> # Using Ollama (free, local)
        >>> evaluator = ToneEvaluator(
        ...     evaluator_provider="ollama",
        ...     evaluator_model="llama3"
        ... )
    """

    @property
    def metric_name(self) -> str:
        """Return metric name."""
        return "tone"

    @property
    def score_range(self) -> tuple:
        """Return valid score range (0-2)."""
        return (0, 2)

    def _build_evaluation_prompt(
        self,
        prompt: str,
        response: str,
        target_audience: Optional[str] = None,
        expected_tone: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Build evaluation prompt for tone assessment.

        Args:
            prompt: Original prompt with context.
            response: LLM response to evaluate.
            target_audience: Optional description of intended audience.
            expected_tone: Optional description of expected tone.
            **kwargs: Additional context (unused).

        Returns:
            str: Formatted evaluation prompt.
        """
        context_section = ""
        if target_audience:
            context_section += f"\nTarget Audience: {target_audience}"
        if expected_tone:
            context_section += f"\nExpected Tone: {expected_tone}"

        return f"""Evaluate the tone and style of this response on a scale of 0-2.

Prompt: {prompt}
{context_section}

Response: {response}

Scoring Criteria:
2 = Appropriate and consistent tone
   - Matches the context and any implied audience
   - Maintains consistent style throughout
   - Professional where needed, casual where appropriate

1 = Adequate tone with inconsistencies
   - Generally appropriate but with some mismatches
   - Minor shifts in formality or style
   - Occasionally too verbose or too terse

0 = Inappropriate tone
   - Significantly mismatched to context
   - Inconsistent style (formal then casual)
   - Condescending, overly casual for serious topics, or too formal for casual requests

Evaluation Guidelines:
- Consider the implied audience from the prompt
- Check for consistency in formality level
- Note any jarring shifts in style
- Assess if explanations match the apparent expertise level requested
- Consider cultural appropriateness

Provide only the numeric score (0, 1, or 2)."""

    def _query_evaluator(self, prompt_text: str, max_tokens: int = 200) -> str:
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

    def analyze_tone(
        self,
        prompt: str,
        response: str
    ) -> dict:
        """
        Perform detailed tone analysis.

        Identifies the tone characteristics and provides
        qualitative assessment beyond just a score.

        Args:
            prompt: Original prompt.
            response: LLM response to analyze.

        Returns:
            dict: Tone analysis including score, characteristics, and recommendations.

        Example:
            >>> result = evaluator.analyze_tone(
            ...     prompt="Explain machine learning",
            ...     response="ML is super cool! It's like teaching..."
            ... )
            >>> print(result["characteristics"])
            ['casual', 'enthusiastic', 'uses analogies']
        """
        analysis_prompt = f"""Analyze the tone and style of this response.

Prompt: {prompt}

Response: {response}

Provide analysis in this format:
SCORE: [0, 1, or 2]
FORMALITY: [FORMAL, NEUTRAL, CASUAL]
CHARACTERISTICS: [List 3-5 tone characteristics, comma-separated]
CONSISTENCY: [HIGH, MEDIUM, LOW]
RECOMMENDATION: [One sentence suggestion for improvement, or "None needed"]"""

        try:
            result_text = self._query_evaluator(analysis_prompt, max_tokens=200)
            return self._parse_tone_analysis(result_text)

        except Exception as e:
            self._logger.warning(f"Tone analysis failed: {e}")
            return {
                "score": 1,
                "formality": "NEUTRAL",
                "characteristics": ["analysis unavailable"],
                "consistency": "MEDIUM",
                "recommendation": "Analysis failed",
            }

    def _parse_tone_analysis(self, text: str) -> dict:
        """
        Parse tone analysis result.

        Args:
            text: Raw response from the evaluator.

        Returns:
            dict: Parsed analysis results.
        """
        result = {
            "score": 1,
            "formality": "NEUTRAL",
            "characteristics": [],
            "consistency": "MEDIUM",
            "recommendation": "",
        }

        lines = text.strip().split("\n")

        for line in lines:
            line = line.strip()
            if line.startswith("SCORE:"):
                try:
                    score_text = line.replace("SCORE:", "").strip()
                    result["score"] = int(score_text[0])
                except (ValueError, IndexError):
                    pass
            elif line.startswith("FORMALITY:"):
                formality = line.replace("FORMALITY:", "").strip().upper()
                if formality in ["FORMAL", "NEUTRAL", "CASUAL"]:
                    result["formality"] = formality
            elif line.startswith("CHARACTERISTICS:"):
                chars = line.replace("CHARACTERISTICS:", "").strip()
                result["characteristics"] = [
                    c.strip() for c in chars.split(",") if c.strip()
                ]
            elif line.startswith("CONSISTENCY:"):
                consistency = line.replace("CONSISTENCY:", "").strip().upper()
                if consistency in ["HIGH", "MEDIUM", "LOW"]:
                    result["consistency"] = consistency
            elif line.startswith("RECOMMENDATION:"):
                result["recommendation"] = line.replace(
                    "RECOMMENDATION:", ""
                ).strip()

        return result

    def compare_tones(
        self,
        prompt: str,
        responses: list
    ) -> dict:
        """
        Compare tone across multiple responses.

        Useful for detecting tone drift across different model versions.

        Args:
            prompt: Original prompt.
            responses: List of responses to compare.

        Returns:
            dict: Comparison results including scores and drift assessment.

        Example:
            >>> result = evaluator.compare_tones(
            ...     prompt="Explain AI",
            ...     responses=["Response from v1...", "Response from v2..."]
            ... )
            >>> print(result["drift_detected"])
            True
        """
        scores = []
        analyses = []

        for response in responses:
            score = self.evaluate(prompt, response)
            scores.append(score)
            analysis = self.analyze_tone(prompt, response)
            analyses.append(analysis)

        # Calculate drift
        if len(scores) >= 2:
            score_variance = max(scores) - min(scores)
            drift_detected = score_variance > 0

            # Check for formality drift
            formalities = [a.get("formality", "NEUTRAL") for a in analyses]
            formality_drift = len(set(formalities)) > 1
        else:
            score_variance = 0
            drift_detected = False
            formality_drift = False

        return {
            "scores": scores,
            "analyses": analyses,
            "score_variance": score_variance,
            "drift_detected": drift_detected,
            "formality_drift": formality_drift,
            "metric": self.metric_name,
            "evaluator_provider": self.evaluator_provider,
            "evaluator_model": self.evaluator_model,
        }
