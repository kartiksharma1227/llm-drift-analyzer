"""
Statistical analyzer for LLM Behavioral Drift Analysis.

This module provides statistical analysis tools for detecting
and quantifying behavioral drift in LLM responses.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

from llm_drift_analyzer.models.response_analysis import ResponseAnalysis, AnalysisResultSet
from llm_drift_analyzer.utils.logger import get_logger


class DriftStatisticalAnalyzer:
    """
    Statistical analyzer for LLM drift detection.

    Performs statistical analysis on drift analysis results including:
    - ANOVA for between-model differences
    - Pairwise comparisons with t-tests
    - Effect size calculations (Cohen's d)
    - Change point detection using CUSUM
    - Trend analysis

    Attributes:
        df: Pandas DataFrame of analysis results.

    Example:
        >>> analyzer = DriftStatisticalAnalyzer(results)
        >>> trends = analyzer.analyze_length_trends()
        >>> print(f"ANOVA p-value: {trends['anova']['p_value']}")

        >>> report = analyzer.generate_drift_report()
        >>> print(report)
    """

    def __init__(self, data: AnalysisResultSet):
        """
        Initialize the statistical analyzer.

        Args:
            data: AnalysisResultSet or list of ResponseAnalysis objects.
        """
        self._logger = get_logger("analyzers.statistical")

        if isinstance(data, AnalysisResultSet):
            self.df = pd.DataFrame([r.to_dict() for r in data.results])
        elif isinstance(data, list):
            self.df = pd.DataFrame([r.to_dict() if hasattr(r, 'to_dict') else vars(r) for r in data])
        else:
            raise ValueError("data must be AnalysisResultSet or list of ResponseAnalysis")

        self._logger.info(f"Initialized with {len(self.df)} records")

    def analyze_length_trends(self) -> Dict[str, Any]:
        """
        Analyze trends in response length over time.

        Performs ANOVA and pairwise comparisons to detect
        significant differences in token counts across models.

        Returns:
            Dict[str, Any]: Analysis results including:
                - anova: F-statistic and p-value
                - pairwise: T-test results and effect sizes
                - summary: Descriptive statistics per model

        Example:
            >>> trends = analyzer.analyze_length_trends()
            >>> if trends['anova']['p_value'] < 0.05:
            ...     print("Significant differences found")
        """
        self._logger.info("Analyzing length trends")
        results = {}

        # Group by model version
        groups = [
            group['token_count'].values
            for name, group in self.df.groupby('model_version')
        ]

        # ANOVA for between-model differences
        if len(groups) >= 2:
            f_stat, p_value = stats.f_oneway(*groups)
            results['anova'] = {
                'f_statistic': float(f_stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05,
            }
        else:
            results['anova'] = {
                'f_statistic': None,
                'p_value': None,
                'significant': False,
                'note': 'Need at least 2 models for ANOVA',
            }

        # Pairwise comparisons
        results['pairwise'] = {}
        models = self.df['model_version'].unique()

        for i, model1 in enumerate(models):
            for model2 in models[i + 1:]:
                data1 = self.df[self.df['model_version'] == model1]['token_count']
                data2 = self.df[self.df['model_version'] == model2]['token_count']

                if len(data1) > 1 and len(data2) > 1:
                    t_stat, p_val = stats.ttest_ind(data1, data2)

                    # Calculate Cohen's d effect size
                    # Fixed: Using proper minus signs instead of em-dash
                    pooled_std = np.sqrt(
                        ((len(data1) - 1) * data1.var() + (len(data2) - 1) * data2.var()) /
                        (len(data1) + len(data2) - 2)
                    )
                    effect_size = (data1.mean() - data2.mean()) / pooled_std if pooled_std > 0 else 0

                    results['pairwise'][f'{model1}_vs_{model2}'] = {
                        't_statistic': float(t_stat),
                        'p_value': float(p_val),
                        'cohens_d': float(effect_size),
                        'significant': p_val < 0.05,
                        'effect_interpretation': self._interpret_cohens_d(effect_size),
                    }

        # Summary statistics
        results['summary'] = self.df.groupby('model_version')['token_count'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).to_dict('index')

        return results

    def detect_change_points(
        self,
        metric: str = 'token_count',
        threshold_multiplier: float = 2.0
    ) -> Dict[str, Any]:
        """
        Detect change points in behavioral metrics using CUSUM.

        Uses Cumulative Sum (CUSUM) analysis to identify points
        where the metric significantly deviates from the mean.

        Args:
            metric: Metric to analyze (token_count, instruction_score, etc.).
            threshold_multiplier: Multiplier for standard deviation threshold.

        Returns:
            Dict[str, Any]: Change point analysis results per model.

        Example:
            >>> change_points = analyzer.detect_change_points('token_count')
            >>> for model, data in change_points.items():
            ...     print(f"{model}: {len(data['change_points'])} changes")
        """
        self._logger.info(f"Detecting change points for {metric}")
        results = {}

        for model in self.df['model_version'].unique():
            model_data = self.df[self.df['model_version'] == model][metric].values

            if len(model_data) < 3:
                results[model] = {
                    'change_points': [],
                    'note': 'Insufficient data for analysis',
                }
                continue

            # CUSUM calculation
            # Fixed: Using proper minus sign
            mean_val = np.mean(model_data)
            cumsum = np.cumsum(model_data - mean_val)

            # Threshold-based detection
            threshold = threshold_multiplier * np.std(model_data)
            change_points = []

            for i, val in enumerate(cumsum):
                if abs(val) > threshold:
                    change_points.append({
                        'index': i,
                        'cumsum_value': float(val),
                        'metric_value': float(model_data[i]),
                    })

            results[model] = {
                'change_points': change_points,
                'change_point_count': len(change_points),
                'cumsum': cumsum.tolist(),
                'threshold': float(threshold),
                'mean': float(mean_val),
                'std': float(np.std(model_data)),
            }

        return results

    def analyze_score_distributions(self) -> Dict[str, Any]:
        """
        Analyze distribution of quality scores across models.

        Returns:
            Dict[str, Any]: Score distribution analysis.
        """
        self._logger.info("Analyzing score distributions")
        results = {}

        score_metrics = ['instruction_score', 'factuality_score', 'tone_score']

        for metric in score_metrics:
            if metric not in self.df.columns:
                continue

            metric_results = {
                'overall': {
                    'mean': float(self.df[metric].mean()),
                    'std': float(self.df[metric].std()),
                    'distribution': self.df[metric].value_counts().to_dict(),
                },
                'by_model': {},
            }

            for model in self.df['model_version'].unique():
                model_data = self.df[self.df['model_version'] == model][metric]
                metric_results['by_model'][model] = {
                    'mean': float(model_data.mean()),
                    'std': float(model_data.std()),
                    'distribution': model_data.value_counts().to_dict(),
                }

            results[metric] = metric_results

        return results

    def analyze_latency_trends(self) -> Dict[str, Any]:
        """
        Analyze latency trends across models.

        Returns:
            Dict[str, Any]: Latency analysis results.
        """
        self._logger.info("Analyzing latency trends")

        results = {
            'overall': {
                'mean': float(self.df['latency_ms'].mean()),
                'std': float(self.df['latency_ms'].std()),
                'min': float(self.df['latency_ms'].min()),
                'max': float(self.df['latency_ms'].max()),
                'percentiles': {
                    'p50': float(self.df['latency_ms'].quantile(0.5)),
                    'p90': float(self.df['latency_ms'].quantile(0.9)),
                    'p95': float(self.df['latency_ms'].quantile(0.95)),
                    'p99': float(self.df['latency_ms'].quantile(0.99)),
                },
            },
            'by_model': {},
        }

        for model in self.df['model_version'].unique():
            model_data = self.df[self.df['model_version'] == model]['latency_ms']
            results['by_model'][model] = {
                'mean': float(model_data.mean()),
                'std': float(model_data.std()),
                'min': float(model_data.min()),
                'max': float(model_data.max()),
                'percentiles': {
                    'p50': float(model_data.quantile(0.5)),
                    'p90': float(model_data.quantile(0.9)),
                    'p95': float(model_data.quantile(0.95)),
                },
            }

        return results

    def calculate_correlation_matrix(self) -> Dict[str, Any]:
        """
        Calculate correlation matrix for numeric metrics.

        Returns:
            Dict[str, Any]: Correlation matrix and notable correlations.
        """
        self._logger.info("Calculating correlation matrix")

        numeric_cols = [
            'token_count', 'latency_ms',
            'instruction_score', 'factuality_score', 'tone_score',
        ]
        available_cols = [c for c in numeric_cols if c in self.df.columns]

        corr_matrix = self.df[available_cols].corr()

        # Find notable correlations
        notable = []
        for i, col1 in enumerate(available_cols):
            for col2 in available_cols[i + 1:]:
                corr = corr_matrix.loc[col1, col2]
                if abs(corr) > 0.3:  # Moderate correlation
                    notable.append({
                        'variables': (col1, col2),
                        'correlation': float(corr),
                        'strength': self._interpret_correlation(corr),
                    })

        return {
            'matrix': corr_matrix.to_dict(),
            'notable_correlations': notable,
        }

    def generate_drift_report(self) -> str:
        """
        Generate comprehensive drift analysis report.

        Creates a markdown-formatted report summarizing all
        statistical analyses and findings.

        Returns:
            str: Markdown-formatted report.

        Example:
            >>> report = analyzer.generate_drift_report()
            >>> with open('drift_report.md', 'w') as f:
            ...     f.write(report)
        """
        self._logger.info("Generating drift report")
        report = []

        # Header
        report.append("# LLM Behavioral Drift Analysis Report\n")
        report.append(f"Generated: {datetime.now().isoformat()}\n")
        report.append(f"Total Samples: {len(self.df)}\n")

        # Summary Statistics
        report.append("\n## Summary Statistics\n")

        summary_stats = self.df.groupby('model_version').agg({
            'token_count': ['count', 'mean', 'std', 'min', 'max'],
            'instruction_score': ['mean', 'std'],
            'factuality_score': ['mean', 'std'],
            'tone_score': ['mean', 'std'],
        }).round(2)

        report.append("```")
        report.append(summary_stats.to_string())
        report.append("```\n")

        # Length Trend Analysis
        report.append("\n## Response Length Analysis\n")
        length_trends = self.analyze_length_trends()

        if length_trends['anova']['f_statistic'] is not None:
            report.append(
                f"ANOVA F-statistic: {length_trends['anova']['f_statistic']:.3f}\n"
            )
            report.append(
                f"P-value: {length_trends['anova']['p_value']:.3e}\n"
            )
            if length_trends['anova']['significant']:
                report.append(
                    "**Significant differences detected between models.**\n"
                )

        # Pairwise Comparisons
        if length_trends['pairwise']:
            report.append("\n### Pairwise Comparisons\n")
            report.append("| Comparison | t-statistic | p-value | Cohen's d | Effect |\n")
            report.append("|------------|-------------|---------|-----------|--------|\n")

            for comparison, data in length_trends['pairwise'].items():
                report.append(
                    f"| {comparison} | {data['t_statistic']:.3f} | "
                    f"{data['p_value']:.3e} | {data['cohens_d']:.3f} | "
                    f"{data['effect_interpretation']} |\n"
                )

        # Change Point Detection
        report.append("\n## Change Point Detection\n")
        change_points = self.detect_change_points()

        for model, data in change_points.items():
            cp_count = data.get('change_point_count', len(data.get('change_points', [])))
            report.append(f"- **{model}**: {cp_count} change points detected\n")

        # Score Analysis
        report.append("\n## Quality Score Analysis\n")
        score_analysis = self.analyze_score_distributions()

        for metric, data in score_analysis.items():
            report.append(f"\n### {metric.replace('_', ' ').title()}\n")
            report.append(f"Overall Mean: {data['overall']['mean']:.2f}\n")
            report.append("| Model | Mean | Std |\n")
            report.append("|-------|------|-----|\n")

            for model, stats in data['by_model'].items():
                report.append(
                    f"| {model} | {stats['mean']:.2f} | {stats['std']:.2f} |\n"
                )

        # Correlation Analysis
        report.append("\n## Correlation Analysis\n")
        correlations = self.calculate_correlation_matrix()

        if correlations['notable_correlations']:
            report.append("Notable correlations:\n")
            for corr in correlations['notable_correlations']:
                report.append(
                    f"- {corr['variables'][0]} vs {corr['variables'][1]}: "
                    f"r={corr['correlation']:.3f} ({corr['strength']})\n"
                )
        else:
            report.append("No notable correlations found.\n")

        return "\n".join(report)

    def _interpret_cohens_d(self, d: float) -> str:
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

    def _interpret_correlation(self, r: float) -> str:
        """Interpret correlation coefficient."""
        abs_r = abs(r)
        direction = "positive" if r > 0 else "negative"
        if abs_r < 0.3:
            return f"weak {direction}"
        elif abs_r < 0.7:
            return f"moderate {direction}"
        else:
            return f"strong {direction}"

    def get_drift_summary(self) -> Dict[str, Any]:
        """
        Get a concise summary of drift indicators.

        Returns:
            Dict[str, Any]: High-level drift summary.
        """
        length_trends = self.analyze_length_trends()
        score_analysis = self.analyze_score_distributions()

        # Calculate overall drift indicators
        models = self.df['model_version'].unique()

        token_means = []
        instruction_means = []
        factuality_means = []

        for model in models:
            model_data = self.df[self.df['model_version'] == model]
            token_means.append(model_data['token_count'].mean())
            instruction_means.append(model_data['instruction_score'].mean())
            factuality_means.append(model_data['factuality_score'].mean())

        return {
            'models_analyzed': list(models),
            'total_samples': len(self.df),
            'drift_indicators': {
                'token_count': {
                    'variance_percent': (
                        (max(token_means) - min(token_means)) / min(token_means) * 100
                        if min(token_means) > 0 else 0
                    ),
                    'anova_significant': length_trends['anova'].get('significant', False),
                },
                'instruction_score': {
                    'variance': max(instruction_means) - min(instruction_means),
                    'lowest_model': models[instruction_means.index(min(instruction_means))],
                },
                'factuality_score': {
                    'variance': max(factuality_means) - min(factuality_means),
                    'lowest_model': models[factuality_means.index(min(factuality_means))],
                },
            },
            'recommendation': self._generate_recommendation(
                length_trends, token_means, instruction_means
            ),
        }

    def _generate_recommendation(
        self,
        length_trends: Dict,
        token_means: List[float],
        instruction_means: List[float]
    ) -> str:
        """Generate recommendation based on analysis."""
        issues = []

        # Check for significant length drift
        if length_trends['anova'].get('significant', False):
            variance_pct = (
                (max(token_means) - min(token_means)) / min(token_means) * 100
                if min(token_means) > 0 else 0
            )
            if variance_pct > 20:
                issues.append(
                    f"Significant response length drift ({variance_pct:.1f}% variance)"
                )

        # Check for instruction adherence issues
        if max(instruction_means) - min(instruction_means) > 0.5:
            issues.append("Notable variation in instruction adherence across models")

        if not issues:
            return "No significant drift detected. Models appear stable."

        return "Issues detected: " + "; ".join(issues)
