"""
Summarization quality evaluator for LLM Drift Analyzer.

Evaluates how well an LLM summarizes content — capturing key points,
maintaining conciseness, and avoiding hallucinated information.

Supports both OpenAI and Ollama evaluation backends.
"""

from typing import Optional

from llm_drift_analyzer.evaluators.base_evaluator import BaseEvaluator


class SummarizationEvaluator(BaseEvaluator):
    """
    Evaluates summarization quality in LLM responses.

    Score Range (0-3):
        0: Fails to summarize — misses all key points or hallucinates content
        1: Poor summary — misses major points or adds spurious information
        2: Good summary — captures most key points with minor omissions
        3: Excellent summary — concise, accurate, captures all key points
    """

    @property
    def metric_name(self) -> str:
        return "summarization_quality"

    @property
    def score_range(self) -> tuple:
        return (0, 3)

    def _build_evaluation_prompt(
        self,
        prompt: str,
        response: str,
        source_text: Optional[str] = None,
        **kwargs
    ) -> str:
        source_section = ""
        if source_text:
            source_section = f"""
Original Source Text (to summarize):
{source_text}
"""

        return f"""Rate the quality of this summary on a scale of 0-3.

Prompt/Instructions: {prompt}
{source_section}
Summary Response: {response}

Scoring Criteria:
3 = Excellent summary — captures all key points, concise, no hallucinated info, well-structured
2 = Good summary — covers most key points, minor omissions or slight verbosity
1 = Poor summary — misses major points, adds information not in source, or too vague
0 = Failed summary — does not summarize at all, completely off-topic, or all hallucinated

Evaluation Guidelines:
- Does the summary cover the main ideas from the source/prompt?
- Is it concise without losing critical information?
- Does it avoid adding facts not present in the original?
- Is it well-organized and readable?

Provide only the numeric score (0, 1, 2, or 3)."""
