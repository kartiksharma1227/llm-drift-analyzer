"""
Report generator for LLM Drift Analysis.

This module provides text-based report generation including
markdown and JSON formats for drift analysis results.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

from llm_drift_analyzer.models.response_analysis import AnalysisResultSet
from llm_drift_analyzer.analyzers.statistical_analyzer import DriftStatisticalAnalyzer
from llm_drift_analyzer.utils.logger import get_logger


class ReportGenerator:
    """
    Generates drift analysis reports in various formats.

    Supports markdown, JSON, and text report formats with
    configurable sections and detail levels.

    Attributes:
        results: Analysis results to report on.
        statistical_analyzer: Statistical analyzer for computations.

    Example:
        >>> generator = ReportGenerator(results)
        >>> markdown_report = generator.generate_markdown_report()
        >>> generator.save_report("drift_report.md", format="markdown")
    """

    def __init__(self, results: AnalysisResultSet):
        """
        Initialize the report generator.

        Args:
            results: AnalysisResultSet to generate reports from.
        """
        self.results = results
        self.statistical_analyzer = DriftStatisticalAnalyzer(results)
        self._logger = get_logger("reporters.report")

    def generate_markdown_report(
        self,
        include_samples: bool = False,
        max_samples: int = 5
    ) -> str:
        """
        Generate a comprehensive markdown report.

        Args:
            include_samples: Whether to include sample responses.
            max_samples: Maximum number of sample responses per model.

        Returns:
            str: Markdown-formatted report.

        Example:
            >>> report = generator.generate_markdown_report(include_samples=True)
            >>> print(report)
        """
        self._logger.info("Generating markdown report")

        sections = [
            self._generate_header(),
            self._generate_executive_summary(),
            self._generate_methodology_section(),
            self._generate_results_section(),
            self._generate_statistical_analysis_section(),
            self._generate_drift_indicators_section(),
        ]

        if include_samples:
            sections.append(self._generate_sample_responses_section(max_samples))

        sections.append(self._generate_recommendations_section())
        sections.append(self._generate_footer())

        return "\n\n".join(sections)

    def generate_json_report(self) -> Dict[str, Any]:
        """
        Generate a structured JSON report.

        Returns:
            Dict[str, Any]: Report data as dictionary.
        """
        self._logger.info("Generating JSON report")

        return {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_samples": len(self.results),
                "models_analyzed": self.results.get_models(),
                "prompts_analyzed": self.results.get_prompts(),
            },
            "summary_statistics": self.results.get_summary_stats(),
            "statistical_analysis": {
                "length_trends": self.statistical_analyzer.analyze_length_trends(),
                "score_distributions": self.statistical_analyzer.analyze_score_distributions(),
                "latency_analysis": self.statistical_analyzer.analyze_latency_trends(),
                "correlations": self.statistical_analyzer.calculate_correlation_matrix(),
            },
            "change_points": self.statistical_analyzer.detect_change_points(),
            "drift_summary": self.statistical_analyzer.get_drift_summary(),
        }

    def generate_text_summary(self) -> str:
        """
        Generate a brief text summary of findings.

        Returns:
            str: Brief text summary.
        """
        summary = self.statistical_analyzer.get_drift_summary()

        lines = [
            "LLM Drift Analysis Summary",
            "=" * 40,
            f"Models Analyzed: {', '.join(summary['models_analyzed'])}",
            f"Total Samples: {summary['total_samples']}",
            "",
            "Key Findings:",
            f"- Token count variance: {summary['drift_indicators']['token_count']['variance_percent']:.1f}%",
            f"- Instruction score variance: {summary['drift_indicators']['instruction_score']['variance']:.2f}",
            f"- Factuality score variance: {summary['drift_indicators']['factuality_score']['variance']:.2f}",
            "",
            f"Recommendation: {summary['recommendation']}",
        ]

        return "\n".join(lines)

    def save_report(
        self,
        file_path: Path,
        format: str = "markdown",
        **kwargs
    ) -> None:
        """
        Save report to file.

        Args:
            file_path: Path to save the report.
            format: Report format ("markdown", "json", "text").
            **kwargs: Additional arguments passed to the generator.

        Raises:
            ValueError: If format is not supported.
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        self._logger.info(f"Saving {format} report to {file_path}")

        if format == "markdown":
            content = self.generate_markdown_report(**kwargs)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
        elif format == "json":
            content = self.generate_json_report()
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(content, f, indent=2, default=str)
        elif format == "text":
            content = self.generate_text_summary()
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _generate_header(self) -> str:
        """Generate report header."""
        return f"""# LLM Behavioral Drift Analysis Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Total Samples:** {len(self.results)}
**Models Analyzed:** {len(self.results.get_models())}
**Prompts Used:** {len(self.results.get_prompts())}"""

    def _generate_executive_summary(self) -> str:
        """Generate executive summary section."""
        summary = self.statistical_analyzer.get_drift_summary()

        return f"""## Executive Summary

This report analyzes behavioral drift across {len(summary['models_analyzed'])} LLM model versions
using {summary['total_samples']} response samples.

### Key Metrics

| Metric | Variance | Status |
|--------|----------|--------|
| Response Length | {summary['drift_indicators']['token_count']['variance_percent']:.1f}% | {'⚠️ High' if summary['drift_indicators']['token_count']['variance_percent'] > 20 else '✅ Normal'} |
| Instruction Adherence | {summary['drift_indicators']['instruction_score']['variance']:.2f} | {'⚠️ High' if summary['drift_indicators']['instruction_score']['variance'] > 0.5 else '✅ Normal'} |
| Factuality | {summary['drift_indicators']['factuality_score']['variance']:.2f} | {'⚠️ High' if summary['drift_indicators']['factuality_score']['variance'] > 0.3 else '✅ Normal'} |

### Overall Assessment

{summary['recommendation']}"""

    def _generate_methodology_section(self) -> str:
        """Generate methodology section."""
        models = self.results.get_models()
        prompts = self.results.get_prompts()

        return f"""## Methodology

### Models Evaluated
{chr(10).join(f'- {model}' for model in models)}

### Prompt Categories
- Instruction Following
- Factual Question Answering
- Creative and Reasoning Tasks

### Evaluation Metrics
- **Instruction Adherence** (0-3): How well the response follows instructions
- **Factuality** (0-2): Accuracy of factual claims
- **Tone** (0-2): Appropriateness of tone and style
- **Token Count**: Response length in tokens
- **Latency**: Response time in milliseconds

### Statistical Methods
- ANOVA for between-group comparisons
- Pairwise t-tests with Bonferroni correction
- Cohen's d effect size calculations
- CUSUM change point detection"""

    def _generate_results_section(self) -> str:
        """Generate detailed results section."""
        stats = self.results.get_summary_stats()

        lines = ["## Detailed Results", "", "### Response Length by Model", ""]
        lines.append("| Model | Mean Tokens | Std Dev | Min | Max |")
        lines.append("|-------|-------------|---------|-----|-----|")

        for result in self.results:
            model = result.model_version
            model_results = self.results.filter_by_model(model)
            if model_results:
                tokens = [r.token_count for r in model_results]
                import statistics
                lines.append(
                    f"| {model} | {statistics.mean(tokens):.1f} | "
                    f"{statistics.stdev(tokens) if len(tokens) > 1 else 0:.1f} | "
                    f"{min(tokens)} | {max(tokens)} |"
                )
                break  # Only need unique models

        # Get unique model stats
        models_seen = set()
        model_stats_lines = []
        for result in self.results:
            if result.model_version not in models_seen:
                models_seen.add(result.model_version)
                model_results = self.results.filter_by_model(result.model_version)
                tokens = [r.token_count for r in model_results]
                import statistics
                model_stats_lines.append(
                    f"| {result.model_version} | {statistics.mean(tokens):.1f} | "
                    f"{statistics.stdev(tokens) if len(tokens) > 1 else 0:.1f} | "
                    f"{min(tokens)} | {max(tokens)} |"
                )

        lines = ["## Detailed Results", "", "### Response Length by Model", ""]
        lines.append("| Model | Mean Tokens | Std Dev | Min | Max |")
        lines.append("|-------|-------------|---------|-----|-----|")
        lines.extend(model_stats_lines)

        return "\n".join(lines)

    def _generate_statistical_analysis_section(self) -> str:
        """Generate statistical analysis section."""
        length_trends = self.statistical_analyzer.analyze_length_trends()
        score_analysis = self.statistical_analyzer.analyze_score_distributions()

        lines = ["## Statistical Analysis"]

        # ANOVA results
        lines.append("\n### ANOVA Results (Response Length)")
        if length_trends['anova']['f_statistic'] is not None:
            lines.append(f"- F-statistic: {length_trends['anova']['f_statistic']:.3f}")
            lines.append(f"- p-value: {length_trends['anova']['p_value']:.3e}")
            sig = "Yes" if length_trends['anova']['significant'] else "No"
            lines.append(f"- Statistically significant: {sig}")

        # Pairwise comparisons
        if length_trends['pairwise']:
            lines.append("\n### Pairwise Comparisons")
            lines.append("| Comparison | t-stat | p-value | Cohen's d | Effect Size |")
            lines.append("|------------|--------|---------|-----------|-------------|")
            for comp, data in length_trends['pairwise'].items():
                lines.append(
                    f"| {comp} | {data['t_statistic']:.2f} | "
                    f"{data['p_value']:.3e} | {data['cohens_d']:.2f} | "
                    f"{data['effect_interpretation']} |"
                )

        return "\n".join(lines)

    def _generate_drift_indicators_section(self) -> str:
        """Generate drift indicators section."""
        change_points = self.statistical_analyzer.detect_change_points()

        lines = ["## Drift Indicators", "", "### Change Point Detection"]

        for model, data in change_points.items():
            cp_count = data.get('change_point_count', len(data.get('change_points', [])))
            lines.append(f"\n**{model}**")
            lines.append(f"- Change points detected: {cp_count}")
            if 'mean' in data:
                lines.append(f"- Mean token count: {data['mean']:.1f}")
            if 'threshold' in data:
                lines.append(f"- Detection threshold: {data['threshold']:.1f}")

        return "\n".join(lines)

    def _generate_sample_responses_section(self, max_samples: int) -> str:
        """Generate sample responses section."""
        lines = ["## Sample Responses"]

        for model in self.results.get_models():
            lines.append(f"\n### {model}")
            model_results = self.results.filter_by_model(model)[:max_samples]

            for i, result in enumerate(model_results, 1):
                lines.append(f"\n**Sample {i}** (Prompt: {result.prompt_id})")
                lines.append(f"- Tokens: {result.token_count}")
                lines.append(f"- Scores: Instruction={result.instruction_score}, "
                           f"Factuality={result.factuality_score}, Tone={result.tone_score}")
                # Truncate long responses
                response_preview = result.response_text[:200]
                if len(result.response_text) > 200:
                    response_preview += "..."
                lines.append(f"- Response preview: {response_preview}")

        return "\n".join(lines)

    def _generate_recommendations_section(self) -> str:
        """Generate recommendations section."""
        summary = self.statistical_analyzer.get_drift_summary()

        lines = ["## Recommendations"]

        # Based on drift indicators
        token_variance = summary['drift_indicators']['token_count']['variance_percent']
        instruction_variance = summary['drift_indicators']['instruction_score']['variance']

        if token_variance > 20:
            lines.append(
                f"\n### Response Length Drift ({token_variance:.1f}% variance)")
            lines.append("- Consider implementing response length normalization")
            lines.append("- Review prompts for explicit length constraints")
            lines.append("- Monitor token costs across model versions")

        if instruction_variance > 0.5:
            lines.append(
                f"\n### Instruction Adherence Issues ({instruction_variance:.2f} score variance)")
            lines.append("- Review prompt engineering practices")
            lines.append("- Consider more explicit instruction formatting")
            lines.append("- Test with structured output formats")

        if token_variance <= 20 and instruction_variance <= 0.5:
            lines.append("\n### No Critical Issues Detected")
            lines.append("- Continue monitoring with regular drift analysis")
            lines.append("- Maintain current prompt templates")

        return "\n".join(lines)

    def _generate_footer(self) -> str:
        """Generate report footer."""
        return """---

*Report generated by LLM Drift Analyzer v1.0.0*
*For methodology details, see the research paper: "Tracking Behavioral Drift in Large Language Models"*"""
