"""
Translation quality evaluator for LLM Drift Analyzer.

Evaluates accuracy, naturalness, and grammar of LLM translations,
with special focus on English-Hindi translation quality.

Supports both OpenAI and Ollama evaluation backends.
"""

from typing import Optional

from llm_drift_analyzer.evaluators.base_evaluator import BaseEvaluator


class TranslationEvaluator(BaseEvaluator):
    """
    Evaluates translation quality in LLM responses.

    Score Range (0-3):
        0: Wrong or incomprehensible translation
        1: Significant meaning loss or unnatural phrasing
        2: Mostly accurate, minor awkwardness
        3: Accurate meaning, natural target language, correct grammar
    """

    @property
    def metric_name(self) -> str:
        return "translation_quality"

    @property
    def score_range(self) -> tuple:
        return (0, 3)

    def _build_evaluation_prompt(
        self,
        prompt: str,
        response: str,
        source_text: Optional[str] = None,
        reference_answer: Optional[str] = None,
        **kwargs
    ) -> str:
        reference_section = ""
        if reference_answer:
            reference_section = f"""
Reference Translation (for comparison):
{reference_answer}
"""

        return f"""Rate the quality of this translation on a scale of 0-3.

Translation Task: {prompt}

Translation Output: {response}
{reference_section}
Scoring Criteria:
3 = Excellent — meaning fully preserved, natural phrasing in target language, correct grammar
2 = Good — mostly accurate meaning, minor awkwardness or slight phrasing issues
1 = Poor — significant meaning loss, unnatural phrasing, or notable grammar errors
0 = Failed — wrong meaning, incomprehensible, or not a translation at all

Evaluation Guidelines:
- Is the core meaning of the source text preserved?
- Does the translation read naturally in the target language?
- Are grammar and syntax correct in the target language?
- For Hindi translations: is the Hindi natural (not literal word-by-word)?
- Are domain-specific terms translated appropriately?
- Technical terms may remain in English if that is standard practice

Provide only the numeric score (0, 1, 2, or 3)."""
