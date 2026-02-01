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
]
