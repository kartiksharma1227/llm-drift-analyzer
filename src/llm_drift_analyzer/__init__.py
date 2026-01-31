"""
LLM Behavioral Drift Analyzer

A comprehensive framework for tracking behavioral drift in Large Language Models,
measuring changes in instruction-following, factuality, tone, and verbosity over time.

Example usage:
    >>> from llm_drift_analyzer import LLMDriftAnalyzer, Config
    >>> config = Config.from_env()
    >>> analyzer = LLMDriftAnalyzer(config)
    >>> results = analyzer.run_drift_analysis(prompts, models)
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from llm_drift_analyzer.analyzers.drift_analyzer import LLMDriftAnalyzer
from llm_drift_analyzer.analyzers.statistical_analyzer import DriftStatisticalAnalyzer
from llm_drift_analyzer.models.response_analysis import ResponseAnalysis
from llm_drift_analyzer.models.prompt import Prompt, PromptCategory
from llm_drift_analyzer.utils.config import Config

__all__ = [
    "LLMDriftAnalyzer",
    "DriftStatisticalAnalyzer",
    "ResponseAnalysis",
    "Prompt",
    "PromptCategory",
    "Config",
]
