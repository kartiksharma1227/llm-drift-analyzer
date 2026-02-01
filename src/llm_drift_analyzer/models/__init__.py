"""
Data models for LLM Drift Analyzer.

This module contains dataclasses for representing prompts, responses,
and analysis results. Includes multilingual support for Hindi.
"""

from llm_drift_analyzer.models.prompt import Prompt, PromptCategory, Language, PromptSet
from llm_drift_analyzer.models.response_analysis import ResponseAnalysis, AnalysisResultSet
from llm_drift_analyzer.models.multilingual_analysis import (
    MultilingualMetrics,
    MultilingualResponseAnalysis,
    CrossLingualComparison,
    MultilingualDriftReport,
)

__all__ = [
    # Core models
    "Prompt",
    "PromptCategory",
    "PromptSet",
    "Language",
    "ResponseAnalysis",
    "AnalysisResultSet",
    # Multilingual models
    "MultilingualMetrics",
    "MultilingualResponseAnalysis",
    "CrossLingualComparison",
    "MultilingualDriftReport",
]
