"""
Evaluators for scoring LLM responses.

This module provides evaluator implementations for measuring instruction adherence,
factuality, and tone/style of LLM responses.

Includes multilingual evaluators with special support for Hindi language analysis.
"""

from llm_drift_analyzer.evaluators.base_evaluator import BaseEvaluator
from llm_drift_analyzer.evaluators.instruction_evaluator import InstructionEvaluator
from llm_drift_analyzer.evaluators.factuality_evaluator import FactualityEvaluator
from llm_drift_analyzer.evaluators.tone_evaluator import ToneEvaluator
from llm_drift_analyzer.evaluators.multilingual_evaluator import (
    MultilingualInstructionEvaluator,
    MultilingualFactualityEvaluator,
    MultilingualToneEvaluator,
    HindiNaturalnessEvaluator,
    ScriptConsistencyEvaluator,
)
from llm_drift_analyzer.evaluators.summarization_evaluator import SummarizationEvaluator
from llm_drift_analyzer.evaluators.translation_evaluator import TranslationEvaluator
from llm_drift_analyzer.evaluators.logical_math_evaluator import LogicalMathEvaluator
from llm_drift_analyzer.evaluators.conversational_evaluator import ConversationalEvaluator
from llm_drift_analyzer.evaluators.legal_admin_evaluator import LegalAdminEvaluator
from llm_drift_analyzer.evaluators.sentiment_evaluator import SentimentEvaluator
from llm_drift_analyzer.evaluators.code_generation_evaluator import CodeGenerationEvaluator

__all__ = [
    # Base
    "BaseEvaluator",
    # English evaluators
    "InstructionEvaluator",
    "FactualityEvaluator",
    "ToneEvaluator",
    # Multilingual evaluators
    "MultilingualInstructionEvaluator",
    "MultilingualFactualityEvaluator",
    "MultilingualToneEvaluator",
    "HindiNaturalnessEvaluator",
    "ScriptConsistencyEvaluator",
    # Task-fitness evaluators
    "SummarizationEvaluator",
    "TranslationEvaluator",
    "LogicalMathEvaluator",
    "ConversationalEvaluator",
    "LegalAdminEvaluator",
    "SentimentEvaluator",
    "CodeGenerationEvaluator",
]
