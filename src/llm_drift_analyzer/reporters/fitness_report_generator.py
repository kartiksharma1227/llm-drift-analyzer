"""
Fitness report generator for LLM Drift Analyzer.

Generates task-fitness reports in markdown and JSON formats,
including model × task matrices, per-task rankings, model profiles,
and actionable recommendations for decision makers.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from llm_drift_analyzer.models.task_fitness import TaskFitnessMatrix
from llm_drift_analyzer.utils.logger import get_logger


# Human-readable category names
CATEGORY_DISPLAY_NAMES = {
    "instruction_following": "Instruction Following",
    "factual_qa": "Factual QA",
    "creative_reasoning": "Creative Reasoning",
    "summarization": "Summarization",
    "translation": "Translation",
    "logical_mathematical": "Logical/Math Reasoning",
    "conversational": "Conversational",
    "legal_administrative": "Legal/Administrative",
    "sentiment_analysis": "Sentiment Analysis",
    "code_generation": "Code Generation",
}


class FitnessReportGenerator:
    """
    Generates task-fitness evaluation reports.

    Produces markdown reports with fitness matrices, rankings,
    and recommendations suitable for government decision makers.
    """

    def __init__(self, matrix: TaskFitnessMatrix):
        self.matrix = matrix
        self._logger = get_logger("reporters.fitness")

    def _display_name(self, category: str) -> str:
        """Get human-readable category name."""
        return CATEGORY_DISPLAY_NAMES.get(category, category.replace("_", " ").title())

    def generate_markdown_report(self) -> str:
        """Generate a comprehensive markdown fitness report."""
        self._logger.info("Generating fitness markdown report")

        sections = [
            self._header(),
            self._executive_summary(),
            self._fitness_matrix_table(),
            self._per_task_rankings(),
            self._model_profiles(),
            self._hindi_focus_section(),
            self._recommendation_table(),
            self._methodology(),
            self._footer(),
        ]
        return "\n\n".join(s for s in sections if s)

    def generate_json_report(self) -> Dict[str, Any]:
        """Generate structured JSON report."""
        return self.matrix.to_dict()

    def save_report(self, file_path: Path, format: str = "markdown") -> None:
        """Save report to file."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "markdown":
            content = self.generate_markdown_report()
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
        elif format == "json":
            content = self.generate_json_report()
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(content, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

        self._logger.info(f"Report saved to {file_path}")

    def _header(self) -> str:
        return f"""# LLM Task-Fitness Evaluation Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Models Evaluated:** {', '.join(self.matrix.models)}
**Task Categories:** {len(self.matrix.categories)}
**Languages:** {', '.join(lang.upper() for lang in self.matrix.languages)}

---

*Which open-source model is best for which task? This report provides
data-driven recommendations for selecting the right LLM for specific
use cases, with a focus on Hindi language performance.*"""

    def _executive_summary(self) -> str:
        lines = ["## Executive Summary"]

        for lang in self.matrix.languages:
            lang_label = "Hindi" if lang == "hi" else "English"
            lines.append(f"\n### Top Recommendations ({lang_label})")

            recs = self.matrix.get_recommendations(lang)
            if not recs:
                lines.append("No data available.")
                continue

            lines.append("")
            lines.append("| Task | Best Model | Score (0-3) |")
            lines.append("|------|-----------|-------------|")

            for rec in recs:
                cat_name = self._display_name(rec["category"])
                lines.append(
                    f"| {cat_name} | **{rec['recommended_model']}** | {rec['score']:.2f} |"
                )

            # Overall ranking
            overall = self.matrix.get_overall_rankings(lang)
            if overall:
                best_model, best_avg = overall[0]
                lines.append(
                    f"\n**Overall Best ({lang_label}):** {best_model} "
                    f"(avg score: {best_avg:.2f}/3.00)"
                )

        return "\n".join(lines)

    def _fitness_matrix_table(self) -> str:
        lines = ["## Fitness Matrix (Model × Task)"]

        for lang in self.matrix.languages:
            lang_label = "Hindi" if lang == "hi" else "English"
            lines.append(f"\n### {lang_label}")

            # Build header
            categories = sorted(self.matrix.categories)
            header_names = [self._display_name(c) for c in categories]
            lines.append("")
            lines.append("| Model | " + " | ".join(header_names) + " | **Avg** |")
            lines.append("|-------|" + "|".join(":---:" for _ in categories) + "|:---:|")

            # Build rows
            for model in self.matrix.models:
                profile = self.matrix.get_model_profile(model, lang)
                scores_str = []
                total = 0.0
                count = 0

                for cat in categories:
                    score = profile.get(cat)
                    if score is not None:
                        scores_str.append(f"{score:.2f}")
                        total += score
                        count += 1
                    else:
                        scores_str.append("—")

                avg = total / count if count > 0 else 0
                lines.append(
                    f"| {model} | " + " | ".join(scores_str) + f" | **{avg:.2f}** |"
                )

        return "\n".join(lines)

    def _per_task_rankings(self) -> str:
        lines = ["## Per-Task Rankings"]

        for lang in self.matrix.languages:
            lang_label = "Hindi" if lang == "hi" else "English"
            lines.append(f"\n### {lang_label}")

            for category in sorted(self.matrix.categories):
                cat_name = self._display_name(category)
                rankings = self.matrix.get_task_rankings(category, lang)

                if not rankings:
                    continue

                lines.append(f"\n#### {cat_name}")
                for rank, (model, score) in enumerate(rankings, 1):
                    medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"{rank}.")
                    lines.append(f"{medal} **{model}** — {score:.2f}/3.00")

        return "\n".join(lines)

    def _model_profiles(self) -> str:
        lines = ["## Model Profiles"]

        for model in self.matrix.models:
            lines.append(f"\n### {model}")

            for lang in self.matrix.languages:
                lang_label = "Hindi" if lang == "hi" else "English"
                profile = self.matrix.get_model_profile(model, lang)

                if not profile:
                    continue

                lines.append(f"\n**{lang_label}:**")

                # Sort by score to show strengths first
                sorted_cats = sorted(profile.items(), key=lambda x: x[1], reverse=True)

                if sorted_cats:
                    best_cat, best_score = sorted_cats[0]
                    worst_cat, worst_score = sorted_cats[-1]

                    lines.append(
                        f"- Strongest: {self._display_name(best_cat)} ({best_score:.2f})"
                    )
                    lines.append(
                        f"- Weakest: {self._display_name(worst_cat)} ({worst_score:.2f})"
                    )
                    avg = sum(s for _, s in sorted_cats) / len(sorted_cats)
                    lines.append(f"- Average: {avg:.2f}/3.00")

        return "\n".join(lines)

    def _hindi_focus_section(self) -> str:
        if "hi" not in self.matrix.languages:
            return ""

        lines = ["## Hindi Performance Analysis (हिंदी प्रदर्शन)"]

        # Hindi-specific rankings
        overall_hi = self.matrix.get_overall_rankings("hi")
        if overall_hi:
            lines.append("\n### Overall Hindi Rankings")
            for rank, (model, score) in enumerate(overall_hi, 1):
                medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"{rank}.")
                lines.append(f"{medal} **{model}** — avg {score:.2f}/3.00")

        # Hindi vs English comparison
        if "en" in self.matrix.languages:
            lines.append("\n### Hindi vs English Performance Gap")
            lines.append("")
            lines.append("| Model | English Avg | Hindi Avg | Gap | Status |")
            lines.append("|-------|:----------:|:---------:|:---:|--------|")

            for model in self.matrix.models:
                en_profile = self.matrix.get_model_profile(model, "en")
                hi_profile = self.matrix.get_model_profile(model, "hi")

                if en_profile and hi_profile:
                    en_avg = sum(en_profile.values()) / len(en_profile)
                    hi_avg = sum(hi_profile.values()) / len(hi_profile)
                    gap = hi_avg - en_avg

                    if gap >= -0.2:
                        status = "✅ Good parity"
                    elif gap >= -0.5:
                        status = "⚠️ Moderate gap"
                    else:
                        status = "❌ Large gap"

                    lines.append(
                        f"| {model} | {en_avg:.2f} | {hi_avg:.2f} | "
                        f"{gap:+.2f} | {status} |"
                    )

        # Government use-case specific recommendations
        govt_categories = [
            "summarization", "legal_administrative", "conversational",
            "translation", "factual_qa",
        ]
        available_govt = [c for c in govt_categories if c in self.matrix.categories]

        if available_govt:
            lines.append("\n### Government Use-Case Recommendations (Hindi)")
            lines.append(
                "\n*For government agencies deploying Hindi-language AI systems:*\n"
            )

            for cat in available_govt:
                best_model, best_score = self.matrix.get_best_model_for_task(cat, "hi")
                cat_name = self._display_name(cat)
                lines.append(f"- **{cat_name}:** Use `{best_model}` (score: {best_score:.2f})")

        return "\n".join(lines)

    def _recommendation_table(self) -> str:
        lines = [
            "## Recommendation Summary",
            "",
            "*Quick-reference table for selecting the right model per task.*",
        ]

        for lang in self.matrix.languages:
            lang_label = "Hindi" if lang == "hi" else "English"
            recs = self.matrix.get_recommendations(lang)

            if not recs:
                continue

            lines.append(f"\n### {lang_label}")
            lines.append("")
            lines.append(
                "| Task Category | Recommended Model | Score | Runner-up | Runner-up Score |"
            )
            lines.append(
                "|--------------|------------------|:-----:|-----------|:--------------:|"
            )

            for rec in recs:
                cat_name = self._display_name(rec["category"])
                lines.append(
                    f"| {cat_name} | **{rec['recommended_model']}** | "
                    f"{rec['score']:.2f} | {rec['runner_up']} | {rec['runner_up_score']:.2f} |"
                )

        return "\n".join(lines)

    def _methodology(self) -> str:
        meta = self.matrix.metadata
        return f"""## Methodology

- **Evaluation approach:** LLM-as-Judge with category-specific evaluation rubrics
- **Score range:** 0-3 (uniform across all categories)
- **Evaluator:** {meta.get('evaluator_provider', 'N/A')} / {meta.get('evaluator_model', 'N/A')}
- **Iterations per prompt:** {meta.get('iterations', 'N/A')}
- **Total queries:** {meta.get('total_queries', 'N/A')} (successful: {meta.get('successful_queries', 'N/A')})

### Task Categories Evaluated
{chr(10).join(f'- **{self._display_name(c)}**' for c in sorted(self.matrix.categories))}

### Scoring Interpretation
| Score | Meaning |
|:-----:|---------|
| 3.0 | Excellent — fully meets requirements |
| 2.0 | Good — mostly meets requirements with minor gaps |
| 1.0 | Poor — significant issues |
| 0.0 | Failed — does not meet requirements |"""

    def _footer(self) -> str:
        return """---

*Report generated by LLM Drift Analyzer — Task Fitness Module*
*For government agencies evaluating open-source LLMs for Hindi deployment*"""
