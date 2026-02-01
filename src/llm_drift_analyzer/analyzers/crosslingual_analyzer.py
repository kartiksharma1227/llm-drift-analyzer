"""
Cross-lingual drift analyzer for LLM Behavioral Drift Analysis.

This module provides specialized analysis for comparing LLM performance
across different languages, particularly English vs Hindi.

Features:
- Language parity analysis
- Cross-lingual drift detection
- Performance gap quantification
- Statistical significance testing for language differences
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from scipy import stats

from llm_drift_analyzer.models.prompt import Prompt, PromptSet, Language
from llm_drift_analyzer.models.multilingual_analysis import (
    MultilingualMetrics,
    MultilingualResponseAnalysis,
    CrossLingualComparison,
    MultilingualDriftReport,
)
from llm_drift_analyzer.utils.multilingual_tokenizer import (
    MultilingualTokenCounter,
    ScriptType,
)
from llm_drift_analyzer.utils.logger import get_logger


@dataclass
class LanguagePerformanceStats:
    """
    Statistical summary of model performance in a specific language.

    Attributes:
        language: Language code.
        sample_size: Number of responses analyzed.
        instruction_score: Mean and std of instruction adherence.
        factuality_score: Mean and std of factuality.
        tone_score: Mean and std of tone.
        token_count: Mean and std of token counts.
        latency_ms: Mean and std of latency.
        language_fidelity: Mean script consistency.
        code_mixing_ratio: Mean code-mixing ratio.
    """
    language: str
    sample_size: int
    instruction_score: Tuple[float, float]  # (mean, std)
    factuality_score: Tuple[float, float]
    tone_score: Tuple[float, float]
    token_count: Tuple[float, float]
    latency_ms: Tuple[float, float]
    language_fidelity: float
    code_mixing_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "language": self.language,
            "sample_size": self.sample_size,
            "instruction_score": {
                "mean": self.instruction_score[0],
                "std": self.instruction_score[1]
            },
            "factuality_score": {
                "mean": self.factuality_score[0],
                "std": self.factuality_score[1]
            },
            "tone_score": {
                "mean": self.tone_score[0],
                "std": self.tone_score[1]
            },
            "token_count": {
                "mean": self.token_count[0],
                "std": self.token_count[1]
            },
            "latency_ms": {
                "mean": self.latency_ms[0],
                "std": self.latency_ms[1]
            },
            "language_fidelity": self.language_fidelity,
            "code_mixing_ratio": self.code_mixing_ratio,
        }


@dataclass
class CrossLingualStatisticalTest:
    """
    Results of statistical test comparing two languages.

    Attributes:
        metric: Metric being compared.
        language_1: First language code.
        language_2: Second language code.
        mean_1: Mean for language 1.
        mean_2: Mean for language 2.
        difference: Difference (lang1 - lang2).
        t_statistic: t-test statistic.
        p_value: Statistical significance.
        cohens_d: Effect size.
        significant: Whether difference is statistically significant.
        interpretation: Human-readable interpretation.
    """
    metric: str
    language_1: str
    language_2: str
    mean_1: float
    mean_2: float
    difference: float
    t_statistic: float
    p_value: float
    cohens_d: float
    significant: bool
    interpretation: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric": self.metric,
            "language_1": self.language_1,
            "language_2": self.language_2,
            "mean_1": self.mean_1,
            "mean_2": self.mean_2,
            "difference": self.difference,
            "t_statistic": self.t_statistic,
            "p_value": self.p_value,
            "cohens_d": self.cohens_d,
            "significant": self.significant,
            "interpretation": self.interpretation,
        }


class CrossLingualAnalyzer:
    """
    Analyzer for cross-lingual LLM performance comparison.

    Compares model behavior between English and Hindi (or other languages),
    detecting performance gaps and language-specific drift patterns.

    Example:
        >>> analyzer = CrossLingualAnalyzer()
        >>> results_en = [...]  # English results
        >>> results_hi = [...]  # Hindi results
        >>> comparison = analyzer.compare_languages(results_en, results_hi)
        >>> print(comparison.language_parity_score)
        0.85
    """

    def __init__(self, significance_level: float = 0.05):
        """
        Initialize cross-lingual analyzer.

        Args:
            significance_level: p-value threshold for statistical significance.
        """
        self._logger = get_logger("analyzers.crosslingual")
        self.significance_level = significance_level
        self.token_counter = MultilingualTokenCounter()

    def compute_language_stats(
        self,
        results: List[MultilingualResponseAnalysis],
        language: str
    ) -> LanguagePerformanceStats:
        """
        Compute statistical summary for a language.

        Args:
            results: List of analysis results.
            language: Language code.

        Returns:
            LanguagePerformanceStats: Statistical summary.
        """
        if not results:
            return LanguagePerformanceStats(
                language=language,
                sample_size=0,
                instruction_score=(0.0, 0.0),
                factuality_score=(0.0, 0.0),
                tone_score=(0.0, 0.0),
                token_count=(0.0, 0.0),
                latency_ms=(0.0, 0.0),
                language_fidelity=0.0,
                code_mixing_ratio=0.0,
            )

        instruction_scores = [r.instruction_score for r in results]
        factuality_scores = [r.factuality_score for r in results]
        tone_scores = [r.tone_score for r in results]
        token_counts = [r.token_count for r in results]
        latencies = [r.latency_ms for r in results]
        fidelities = [r.language_fidelity_score for r in results]
        code_mixing = [r.multilingual_metrics.code_mixing_ratio for r in results]

        return LanguagePerformanceStats(
            language=language,
            sample_size=len(results),
            instruction_score=(np.mean(instruction_scores), np.std(instruction_scores)),
            factuality_score=(np.mean(factuality_scores), np.std(factuality_scores)),
            tone_score=(np.mean(tone_scores), np.std(tone_scores)),
            token_count=(np.mean(token_counts), np.std(token_counts)),
            latency_ms=(np.mean(latencies), np.std(latencies)),
            language_fidelity=np.mean(fidelities),
            code_mixing_ratio=np.mean(code_mixing),
        )

    def _calculate_cohens_d(
        self,
        group1: List[float],
        group2: List[float]
    ) -> float:
        """
        Calculate Cohen's d effect size.

        Args:
            group1: First group of values.
            group2: Second group of values.

        Returns:
            float: Cohen's d effect size.
        """
        n1, n2 = len(group1), len(group2)
        if n1 < 2 or n2 < 2:
            return 0.0

        mean1, mean2 = np.mean(group1), np.mean(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        if pooled_std == 0:
            return 0.0

        return (mean1 - mean2) / pooled_std

    def _interpret_effect_size(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"

    def statistical_comparison(
        self,
        results_lang1: List[MultilingualResponseAnalysis],
        results_lang2: List[MultilingualResponseAnalysis],
        lang1_code: str = "en",
        lang2_code: str = "hi"
    ) -> List[CrossLingualStatisticalTest]:
        """
        Perform statistical tests comparing two languages.

        Args:
            results_lang1: Results for first language.
            results_lang2: Results for second language.
            lang1_code: Code for first language.
            lang2_code: Code for second language.

        Returns:
            List of statistical test results for each metric.
        """
        tests = []
        metrics = [
            ("instruction_score", lambda r: r.instruction_score),
            ("factuality_score", lambda r: r.factuality_score),
            ("tone_score", lambda r: r.tone_score),
            ("token_count", lambda r: r.token_count),
            ("total_quality_score", lambda r: r.total_quality_score),
        ]

        for metric_name, extractor in metrics:
            values1 = [extractor(r) for r in results_lang1]
            values2 = [extractor(r) for r in results_lang2]

            if len(values1) < 2 or len(values2) < 2:
                continue

            # Welch's t-test (doesn't assume equal variances)
            t_stat, p_value = stats.ttest_ind(values1, values2, equal_var=False)

            # Effect size
            cohens_d = self._calculate_cohens_d(values1, values2)
            effect_interp = self._interpret_effect_size(cohens_d)

            mean1, mean2 = np.mean(values1), np.mean(values2)
            difference = mean1 - mean2
            significant = p_value < self.significance_level

            # Create interpretation
            if significant:
                if difference > 0:
                    interp = f"{lang1_code.upper()} performs better ({effect_interp} effect)"
                else:
                    interp = f"{lang2_code.upper()} performs better ({effect_interp} effect)"
            else:
                interp = f"No significant difference between {lang1_code.upper()} and {lang2_code.upper()}"

            tests.append(CrossLingualStatisticalTest(
                metric=metric_name,
                language_1=lang1_code,
                language_2=lang2_code,
                mean_1=mean1,
                mean_2=mean2,
                difference=difference,
                t_statistic=t_stat,
                p_value=p_value,
                cohens_d=cohens_d,
                significant=significant,
                interpretation=interp,
            ))

        return tests

    def compare_parallel_prompts(
        self,
        english_results: Dict[str, List[MultilingualResponseAnalysis]],
        hindi_results: Dict[str, List[MultilingualResponseAnalysis]],
        parallel_mapping: Dict[str, str]
    ) -> List[CrossLingualComparison]:
        """
        Compare results for parallel prompts (same content in both languages).

        Args:
            english_results: Results grouped by English prompt ID.
            hindi_results: Results grouped by Hindi prompt ID.
            parallel_mapping: Mapping from English ID to Hindi ID.

        Returns:
            List of CrossLingualComparison objects.
        """
        comparisons = []

        for en_id, hi_id in parallel_mapping.items():
            en_results = english_results.get(en_id, [])
            hi_results = hindi_results.get(hi_id, [])

            if not en_results or not hi_results:
                continue

            # Group by model
            models = set(r.model_version for r in en_results) | set(r.model_version for r in hi_results)

            for model in models:
                model_en_results = [r for r in en_results if r.model_version == model]
                model_hi_results = [r for r in hi_results if r.model_version == model]

                if not model_en_results or not model_hi_results:
                    continue

                comparison = CrossLingualComparison(
                    model_version=model,
                    prompt_id_english=en_id,
                    prompt_id_hindi=hi_id,
                    english_results=model_en_results,
                    hindi_results=model_hi_results,
                )

                comparisons.append(comparison)

        return comparisons

    def calculate_language_parity_index(
        self,
        comparisons: List[CrossLingualComparison]
    ) -> Dict[str, float]:
        """
        Calculate overall language parity index per model.

        The Language Parity Index (LPI) measures how equally a model
        performs across languages. 1.0 = perfect parity.

        Args:
            comparisons: List of cross-lingual comparisons.

        Returns:
            Dict mapping model name to LPI score.
        """
        model_scores = {}

        for model in set(c.model_version for c in comparisons):
            model_comparisons = [c for c in comparisons if c.model_version == model]
            parity_scores = [c.language_parity_score for c in model_comparisons]

            if parity_scores:
                model_scores[model] = np.mean(parity_scores)

        return model_scores

    def detect_language_specific_drift(
        self,
        results_over_time: Dict[str, List[MultilingualResponseAnalysis]],
        language: str,
        window_size: int = 5
    ) -> Dict[str, Any]:
        """
        Detect drift patterns specific to a language.

        Analyzes how model behavior changes over time for a specific language,
        looking for patterns that might indicate language-specific degradation.

        Args:
            results_over_time: Results grouped by timestamp/iteration.
            language: Language to analyze.
            window_size: Window size for moving average.

        Returns:
            Dict containing drift indicators and patterns.
        """
        # Group results by time
        time_sorted = sorted(results_over_time.keys())

        if len(time_sorted) < window_size:
            return {"insufficient_data": True}

        metrics_over_time = {
            "quality_score": [],
            "language_fidelity": [],
            "code_mixing": [],
        }

        for timestamp in time_sorted:
            results = results_over_time[timestamp]
            lang_results = [r for r in results if r.expected_language == language]

            if lang_results:
                metrics_over_time["quality_score"].append(
                    np.mean([r.total_quality_score for r in lang_results])
                )
                metrics_over_time["language_fidelity"].append(
                    np.mean([r.language_fidelity_score for r in lang_results])
                )
                metrics_over_time["code_mixing"].append(
                    np.mean([r.multilingual_metrics.code_mixing_ratio for r in lang_results])
                )

        # Calculate trends
        drift_indicators = {}

        for metric, values in metrics_over_time.items():
            if len(values) < 2:
                continue

            # Linear regression for trend
            x = np.arange(len(values))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)

            drift_indicators[metric] = {
                "trend_slope": slope,
                "trend_significance": p_value,
                "r_squared": r_value ** 2,
                "start_value": values[0],
                "end_value": values[-1],
                "change_percent": ((values[-1] - values[0]) / values[0] * 100) if values[0] != 0 else 0,
                "is_declining": slope < 0 and p_value < 0.05,
            }

        return drift_indicators

    def generate_crosslingual_report(
        self,
        english_results: List[MultilingualResponseAnalysis],
        hindi_results: List[MultilingualResponseAnalysis],
        parallel_mapping: Dict[str, str]
    ) -> MultilingualDriftReport:
        """
        Generate comprehensive cross-lingual analysis report.

        Args:
            english_results: All English analysis results.
            hindi_results: All Hindi analysis results.
            parallel_mapping: Mapping between parallel prompts.

        Returns:
            MultilingualDriftReport: Complete analysis report.
        """
        self._logger.info("Generating cross-lingual analysis report")

        # Compute statistics
        en_stats = self.compute_language_stats(english_results, "en")
        hi_stats = self.compute_language_stats(hindi_results, "hi")

        # Statistical tests
        stat_tests = self.statistical_comparison(
            english_results, hindi_results, "en", "hi"
        )

        # Group results by prompt
        en_by_prompt = {}
        hi_by_prompt = {}

        for r in english_results:
            en_by_prompt.setdefault(r.prompt_id, []).append(r)
        for r in hindi_results:
            hi_by_prompt.setdefault(r.prompt_id, []).append(r)

        # Compare parallel prompts
        comparisons = self.compare_parallel_prompts(
            en_by_prompt, hi_by_prompt, parallel_mapping
        )

        # Calculate LPI
        lpi_scores = self.calculate_language_parity_index(comparisons)

        # Compile report
        models = list(set(
            [r.model_version for r in english_results] +
            [r.model_version for r in hindi_results]
        ))

        report = MultilingualDriftReport(
            models=models,
            languages=["en", "hi"],
            results_by_language={
                "en": english_results,
                "hi": hindi_results,
            },
            cross_lingual_comparisons=comparisons,
            summary_statistics={
                "english_stats": en_stats.to_dict(),
                "hindi_stats": hi_stats.to_dict(),
                "statistical_tests": [t.to_dict() for t in stat_tests],
                "language_parity_index": lpi_scores,
                "overall_parity": np.mean(list(lpi_scores.values())) if lpi_scores else 0,
            },
        )

        return report

    def get_recommendations(
        self,
        report: MultilingualDriftReport
    ) -> List[str]:
        """
        Generate recommendations based on cross-lingual analysis.

        Args:
            report: Completed analysis report.

        Returns:
            List of recommendations.
        """
        recommendations = []
        stats = report.summary_statistics

        # Check overall parity
        overall_parity = stats.get("overall_parity", 0)
        if overall_parity < 0.7:
            recommendations.append(
                f"⚠️ Low language parity ({overall_parity:.2%}). "
                "Consider fine-tuning the model on more Hindi data."
            )

        # Check statistical tests
        for test in stats.get("statistical_tests", []):
            if test.get("significant") and abs(test.get("cohens_d", 0)) > 0.5:
                metric = test.get("metric", "unknown")
                if test.get("difference", 0) > 0:
                    recommendations.append(
                        f"📊 Significant gap in {metric}: English outperforms Hindi. "
                        f"Effect size: {test.get('cohens_d', 0):.2f}"
                    )
                else:
                    recommendations.append(
                        f"📊 Significant gap in {metric}: Hindi outperforms English. "
                        f"Effect size: {test.get('cohens_d', 0):.2f}"
                    )

        # Check code mixing
        hi_stats = stats.get("hindi_stats", {})
        code_mixing = hi_stats.get("code_mixing_ratio", 0)
        if code_mixing > 0.2:
            recommendations.append(
                f"⚠️ High code-mixing ratio ({code_mixing:.2%}) in Hindi responses. "
                "Model may be defaulting to English for complex terms."
            )

        # Check language fidelity
        lang_fidelity = hi_stats.get("language_fidelity", 0)
        if lang_fidelity < 0.8:
            recommendations.append(
                f"⚠️ Low language fidelity ({lang_fidelity:.2%}) for Hindi. "
                "Model struggles to maintain consistent Hindi output."
            )

        if not recommendations:
            recommendations.append(
                "✅ Model shows good cross-lingual consistency. "
                "No major issues detected."
            )

        return recommendations
