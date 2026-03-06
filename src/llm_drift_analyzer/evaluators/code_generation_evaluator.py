"""
Code generation quality evaluator for LLM Drift Analyzer.

Evaluates syntactic correctness, logical soundness, and best practices
in LLM-generated code.

Supports both OpenAI and Ollama evaluation backends.
"""

from typing import Optional

from llm_drift_analyzer.evaluators.base_evaluator import BaseEvaluator


class CodeGenerationEvaluator(BaseEvaluator):
    """
    Evaluates code generation quality in LLM responses.

    Score Range (0-3):
        0: Non-functional code or not code at all
        1: Significant bugs, poor practices, or incomplete
        2: Mostly correct with minor issues
        3: Syntactically correct, logically sound, follows best practices
    """

    @property
    def metric_name(self) -> str:
        return "code_generation_quality"

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
Reference/Expected Code:
{reference_answer}
"""

        return f"""Rate the quality of this generated code on a scale of 0-3.

Coding Task: {prompt}

Generated Code: {response}
{reference_section}
Scoring Criteria:
3 = Excellent — syntactically correct, logically sound, follows best practices, handles edge cases
2 = Good — mostly correct, minor issues (missing edge cases, slight inefficiency)
1 = Poor — significant bugs, poor practices, incomplete implementation, or partially works
0 = Failed — non-functional code, syntax errors, completely wrong approach, or no code generated

Evaluation Guidelines:
- Is the code syntactically valid for the specified language?
- Does it correctly solve the stated problem?
- Does it handle common edge cases?
- Are variable names and structure reasonable?
- Is it reasonably efficient for the task?
- Does it include necessary imports/dependencies?

Provide only the numeric score (0, 1, 2, or 3)."""
