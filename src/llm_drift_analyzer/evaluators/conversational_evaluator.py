"""
Conversational quality evaluator for LLM Drift Analyzer.

Evaluates helpfulness, naturalness, and appropriateness of
LLM responses in dialogue/helpdesk scenarios.

Supports both OpenAI and Ollama evaluation backends.
"""

from typing import Optional

from llm_drift_analyzer.evaluators.base_evaluator import BaseEvaluator


class ConversationalEvaluator(BaseEvaluator):
    """
    Evaluates conversational quality in LLM responses.

    Score Range (0-3):
        0: Off-topic, unhelpful, or inappropriate
        1: Stiff or generic, minimally helpful
        2: Adequate response with minor tone issues
        3: Natural, helpful, contextually appropriate, empathetic where needed
    """

    @property
    def metric_name(self) -> str:
        return "conversational_quality"

    @property
    def score_range(self) -> tuple:
        return (0, 3)

    def _build_evaluation_prompt(
        self,
        prompt: str,
        response: str,
        conversation_context: Optional[str] = None,
        **kwargs
    ) -> str:
        context_section = ""
        if conversation_context:
            context_section = f"""
Prior Conversation Context:
{conversation_context}
"""

        return f"""Rate the conversational quality of this response on a scale of 0-3.

User Message / Scenario: {prompt}
{context_section}
Assistant Response: {response}

Scoring Criteria:
3 = Excellent — natural, helpful, contextually appropriate, empathetic where needed, actionable
2 = Good — adequate and helpful response with minor tone or phrasing issues
1 = Poor — stiff, overly generic, or minimally helpful
0 = Failed — off-topic, unhelpful, rude, or inappropriate for the context

Evaluation Guidelines:
- Does the response directly address the user's question or concern?
- Is the tone appropriate (formal for govt helpdesk, friendly for casual)?
- Does it provide actionable next steps or useful information?
- Is it empathetic when the user expresses frustration?
- Does it avoid being overly verbose or robotic?
- For Hindi responses: does it use appropriate formality level (आप/तुम)?

Provide only the numeric score (0, 1, 2, or 3)."""
