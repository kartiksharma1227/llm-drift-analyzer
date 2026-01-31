"""
Tests for analyzers (DriftAnalyzer, StatisticalAnalyzer).
"""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np

from llm_drift_analyzer.analyzers.statistical_analyzer import DriftStatisticalAnalyzer
from llm_drift_analyzer.models.response_analysis import AnalysisResultSet


class TestDriftStatisticalAnalyzer:
    """Tests for DriftStatisticalAnalyzer."""

    def test_init_with_result_set(self, sample_results):
        """Test initialization with AnalysisResultSet."""
        analyzer = DriftStatisticalAnalyzer(sample_results)
        assert len(analyzer.df) == 10

    def test_init_with_list(self, sample_results):
        """Test initialization with list of results."""
        analyzer = DriftStatisticalAnalyzer(sample_results.results)
        assert len(analyzer.df) == 10

    def test_analyze_length_trends_anova(self, sample_results):
        """Test ANOVA analysis in length trends."""
        analyzer = DriftStatisticalAnalyzer(sample_results)
        trends = analyzer.analyze_length_trends()

        assert "anova" in trends
        assert "f_statistic" in trends["anova"]
        assert "p_value" in trends["anova"]
        assert "significant" in trends["anova"]

    def test_analyze_length_trends_pairwise(self, sample_results):
        """Test pairwise comparisons in length trends."""
        analyzer = DriftStatisticalAnalyzer(sample_results)
        trends = analyzer.analyze_length_trends()

        assert "pairwise" in trends
        # Should have one comparison: v1 vs v2
        assert len(trends["pairwise"]) == 1

        for comparison, data in trends["pairwise"].items():
            assert "t_statistic" in data
            assert "p_value" in data
            assert "cohens_d" in data
            assert "effect_interpretation" in data

    def test_analyze_length_trends_summary(self, sample_results):
        """Test summary statistics in length trends."""
        analyzer = DriftStatisticalAnalyzer(sample_results)
        trends = analyzer.analyze_length_trends()

        assert "summary" in trends
        assert "gpt-4-v1" in trends["summary"]
        assert "gpt-4-v2" in trends["summary"]

    def test_detect_change_points(self, sample_results):
        """Test change point detection."""
        analyzer = DriftStatisticalAnalyzer(sample_results)
        change_points = analyzer.detect_change_points("token_count")

        assert "gpt-4-v1" in change_points
        assert "gpt-4-v2" in change_points

        for model, data in change_points.items():
            assert "change_points" in data
            assert "cumsum" in data
            assert "threshold" in data
            assert "mean" in data
            assert "std" in data

    def test_detect_change_points_different_metric(self, sample_results):
        """Test change point detection with different metric."""
        analyzer = DriftStatisticalAnalyzer(sample_results)
        change_points = analyzer.detect_change_points("instruction_score")

        assert "gpt-4-v1" in change_points

    def test_analyze_score_distributions(self, sample_results):
        """Test score distribution analysis."""
        analyzer = DriftStatisticalAnalyzer(sample_results)
        distributions = analyzer.analyze_score_distributions()

        assert "instruction_score" in distributions
        assert "factuality_score" in distributions
        assert "tone_score" in distributions

        for metric, data in distributions.items():
            assert "overall" in data
            assert "by_model" in data
            assert "mean" in data["overall"]
            assert "std" in data["overall"]

    def test_analyze_latency_trends(self, sample_results):
        """Test latency trend analysis."""
        analyzer = DriftStatisticalAnalyzer(sample_results)
        latency = analyzer.analyze_latency_trends()

        assert "overall" in latency
        assert "by_model" in latency
        assert "mean" in latency["overall"]
        assert "percentiles" in latency["overall"]
        assert "p50" in latency["overall"]["percentiles"]
        assert "p90" in latency["overall"]["percentiles"]

    def test_calculate_correlation_matrix(self, sample_results):
        """Test correlation matrix calculation."""
        analyzer = DriftStatisticalAnalyzer(sample_results)
        correlations = analyzer.calculate_correlation_matrix()

        assert "matrix" in correlations
        assert "notable_correlations" in correlations

    def test_generate_drift_report(self, sample_results):
        """Test drift report generation."""
        analyzer = DriftStatisticalAnalyzer(sample_results)
        report = analyzer.generate_drift_report()

        assert isinstance(report, str)
        assert "# LLM Behavioral Drift Analysis Report" in report
        assert "Summary Statistics" in report
        assert "ANOVA" in report

    def test_get_drift_summary(self, sample_results):
        """Test drift summary generation."""
        analyzer = DriftStatisticalAnalyzer(sample_results)
        summary = analyzer.get_drift_summary()

        assert "models_analyzed" in summary
        assert "total_samples" in summary
        assert "drift_indicators" in summary
        assert "recommendation" in summary

        assert "token_count" in summary["drift_indicators"]
        assert "instruction_score" in summary["drift_indicators"]

    def test_interpret_cohens_d(self, sample_results):
        """Test Cohen's d interpretation."""
        analyzer = DriftStatisticalAnalyzer(sample_results)

        assert analyzer._interpret_cohens_d(0.1) == "negligible"
        assert analyzer._interpret_cohens_d(0.3) == "small"
        assert analyzer._interpret_cohens_d(0.6) == "medium"
        assert analyzer._interpret_cohens_d(1.0) == "large"

    def test_interpret_correlation(self, sample_results):
        """Test correlation interpretation."""
        analyzer = DriftStatisticalAnalyzer(sample_results)

        assert "weak" in analyzer._interpret_correlation(0.2)
        assert "moderate" in analyzer._interpret_correlation(0.5)
        assert "strong" in analyzer._interpret_correlation(0.8)
        assert "positive" in analyzer._interpret_correlation(0.5)
        assert "negative" in analyzer._interpret_correlation(-0.5)


class TestStatisticalCalculations:
    """Tests for statistical calculation correctness."""

    def test_cohens_d_calculation(self, sample_results):
        """Test that Cohen's d is calculated correctly."""
        analyzer = DriftStatisticalAnalyzer(sample_results)
        trends = analyzer.analyze_length_trends()

        # Get data for manual verification
        v1_tokens = analyzer.df[analyzer.df['model_version'] == 'gpt-4-v1']['token_count'].values
        v2_tokens = analyzer.df[analyzer.df['model_version'] == 'gpt-4-v2']['token_count'].values

        # Manual calculation
        pooled_std = np.sqrt(
            ((len(v1_tokens) - 1) * np.var(v1_tokens, ddof=1) +
             (len(v2_tokens) - 1) * np.var(v2_tokens, ddof=1)) /
            (len(v1_tokens) + len(v2_tokens) - 2)
        )
        expected_d = (np.mean(v1_tokens) - np.mean(v2_tokens)) / pooled_std

        # Get computed value
        comparison_key = list(trends['pairwise'].keys())[0]
        computed_d = trends['pairwise'][comparison_key]['cohens_d']

        # Allow for sign difference depending on order
        assert abs(abs(computed_d) - abs(expected_d)) < 0.01

    def test_cusum_calculation(self, sample_results):
        """Test that CUSUM is calculated correctly."""
        analyzer = DriftStatisticalAnalyzer(sample_results)
        change_points = analyzer.detect_change_points('token_count')

        # Get data for one model
        model = 'gpt-4-v1'
        model_data = analyzer.df[analyzer.df['model_version'] == model]['token_count'].values

        # Manual CUSUM calculation
        mean_val = np.mean(model_data)
        expected_cumsum = np.cumsum(model_data - mean_val)

        # Get computed CUSUM
        computed_cumsum = np.array(change_points[model]['cumsum'])

        # Verify
        np.testing.assert_array_almost_equal(computed_cumsum, expected_cumsum)
