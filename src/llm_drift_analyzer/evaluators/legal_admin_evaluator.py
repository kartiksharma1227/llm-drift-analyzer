"""
Legal and administrative quality evaluator for LLM Drift Analyzer.

Evaluates precision, terminology, and structure of LLM responses
for government/legal/administrative document tasks.

Supports both OpenAI and Ollama evaluation backends.
"""

from typing import Optional

from llm_drift_analyzer.evaluators.base_evaluator import BaseEvaluator


class LegalAdminEvaluator(BaseEvaluator):
    """
    Evaluates legal/administrative quality in LLM responses.

    Score Range (0-3):
        0: Incorrect, misleading, or completely unstructured
        1: Significant terminology errors or missing formal structure
        2: Mostly correct with minor imprecision in terms or format
        3: Precise, uses correct legal/admin terminology, well-structured
    """

    @property
    def metric_name(self) -> str:
        return "legal_administrative_quality"

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
Reference/Expected Content:
{reference_answer}
"""

        return f"""Rate the legal/administrative quality of this response on a scale of 0-3.

Task/Prompt: {prompt}

Response: {response}
{reference_section}
Scoring Criteria:
3 = Excellent — precise terminology, well-structured, formally appropriate, factually correct
2 = Good — mostly correct, minor imprecision in terminology or formatting
1 = Poor — significant errors in legal/admin terms, missing formal structure, or vague
0 = Failed — incorrect information, misleading, or completely unstructured

Evaluation Guidelines:
- Are legal/administrative terms used correctly (e.g., RTI, PIL, gazette, circular)?
- Is the document structure appropriate (proper headings, formal tone, numbered clauses)?
- Is the content factually accurate regarding rules, procedures, or laws?
- Does it follow the expected formal register?
- For Hindi responses: are Hindi legal terms used correctly (याचिका, अधिसूचना, परिपत्र)?
- Are dates, references, and citations formatted properly?

Provide only the numeric score (0, 1, 2, or 3)."""
