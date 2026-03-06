"""
Sentiment analysis quality evaluator for LLM Drift Analyzer.

Evaluates how accurately an LLM identifies and analyzes sentiment
in text, including nuance detection.

Supports both OpenAI and Ollama evaluation backends.
"""

from typing import Optional

from llm_drift_analyzer.evaluators.base_evaluator import BaseEvaluator


class SentimentEvaluator(BaseEvaluator):
    """
    Evaluates sentiment analysis quality in LLM responses.

    Score Range (0-3):
        0: Wrong sentiment identification
        1: Partially correct, misses nuance
        2: Correct overall sentiment, minor nuance gaps
        3: Correctly identifies sentiment with nuance and reasoning
    """

    @property
    def metric_name(self) -> str:
        return "sentiment_analysis_quality"

    @property
    def score_range(self) -> tuple:
        return (0, 3)

    def _build_evaluation_prompt(
        self,
        prompt: str,
        response: str,
        expected_sentiment: Optional[str] = None,
        **kwargs
    ) -> str:
        reference_section = ""
        if expected_sentiment:
            reference_section = f"""
Expected Sentiment/Analysis:
{expected_sentiment}
"""

        return f"""Rate the quality of this sentiment analysis on a scale of 0-3.

Analysis Task: {prompt}

Model's Analysis: {response}
{reference_section}
Scoring Criteria:
3 = Excellent — correctly identifies sentiment, detects nuances (sarcasm, mixed feelings), provides reasoning
2 = Good — correct overall sentiment, but misses subtle nuances or mixed sentiments
1 = Partial — gets the broad direction right but significant analysis gaps
0 = Wrong — misidentifies the sentiment entirely

Evaluation Guidelines:
- Is the overall sentiment (positive/negative/neutral/mixed) correctly identified?
- Are nuances detected (e.g., sarcasm, backhanded compliments, conditional praise)?
- Does the analysis provide reasoning or evidence from the text?
- For Hindi text: does it understand Hindi idioms and cultural sentiment markers?
- Does it identify the intensity of sentiment (strongly negative vs mildly negative)?

Provide only the numeric score (0, 1, 2, or 3)."""
