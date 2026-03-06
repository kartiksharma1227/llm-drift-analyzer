"""
Fitness visualizer for LLM Drift Analyzer.

Generates publication-ready charts for task-fitness analysis including
heatmaps, radar charts, bar charts, and cross-lingual comparisons.
"""

import math
from pathlib import Path
from typing import List, Optional

import numpy as np

from llm_drift_analyzer.models.task_fitness import TaskFitnessMatrix
from llm_drift_analyzer.utils.logger import get_logger

# Human-readable short names for chart labels
CATEGORY_SHORT_NAMES = {
    "instruction_following": "Instruction",
    "factual_qa": "Factual QA",
    "creative_reasoning": "Creative",
    "summarization": "Summary",
    "translation": "Translation",
    "logical_mathematical": "Logic/Math",
    "conversational": "Conversational",
    "legal_administrative": "Legal/Admin",
    "sentiment_analysis": "Sentiment",
    "code_generation": "Code Gen",
}


class FitnessVisualizer:
    """
    Generates visualizations for task-fitness analysis.

    Produces heatmaps, radar charts, bar rankings, and
    cross-lingual comparison plots using matplotlib and seaborn.
    """

    def __init__(self, matrix: TaskFitnessMatrix):
        self.matrix = matrix
        self._logger = get_logger("reporters.fitness_viz")

    def _short_name(self, category: str) -> str:
        return CATEGORY_SHORT_NAMES.get(category, category[:10])

    def plot_fitness_heatmap(self, language: str = "hi"):
        """
        Generate a model × task heatmap with color-coded scores.

        Args:
            language: Language to plot ("en" or "hi").

        Returns:
            matplotlib Figure.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        categories = sorted(self.matrix.categories)
        models = self.matrix.models

        # Build score matrix
        data = np.zeros((len(models), len(categories)))
        for i, model in enumerate(models):
            profile = self.matrix.get_model_profile(model, language)
            for j, cat in enumerate(categories):
                data[i, j] = profile.get(cat, 0)

        fig, ax = plt.subplots(figsize=(max(10, len(categories) * 1.2), max(4, len(models) * 0.8)))
        sns.heatmap(
            data,
            annot=True,
            fmt=".2f",
            cmap="RdYlGn",
            vmin=0,
            vmax=3,
            xticklabels=[self._short_name(c) for c in categories],
            yticklabels=models,
            ax=ax,
            linewidths=0.5,
            cbar_kws={"label": "Score (0-3)"},
        )

        lang_label = "Hindi" if language == "hi" else "English"
        ax.set_title(f"Task Fitness Heatmap — {lang_label}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Task Category")
        ax.set_ylabel("Model")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        return fig

    def plot_radar_per_model(self, language: str = "hi"):
        """
        Generate radar/spider chart showing each model's task profile.

        Args:
            language: Language to plot.

        Returns:
            matplotlib Figure.
        """
        import matplotlib.pyplot as plt

        categories = sorted(self.matrix.categories)
        n_cats = len(categories)
        models = self.matrix.models

        if n_cats < 3:
            self._logger.warning("Need at least 3 categories for radar chart")
            return None

        # Compute angles
        angles = [n / float(n_cats) * 2 * math.pi for n in range(n_cats)]
        angles += angles[:1]  # Close the polygon

        n_models = len(models)
        cols = min(3, n_models)
        rows = math.ceil(n_models / cols)
        fig, axes = plt.subplots(
            rows, cols,
            figsize=(5 * cols, 5 * rows),
            subplot_kw=dict(projection="polar"),
        )

        if n_models == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

        colors = plt.cm.Set2(np.linspace(0, 1, n_models))

        for idx, model in enumerate(models):
            if idx >= len(axes):
                break
            ax = axes[idx]
            profile = self.matrix.get_model_profile(model, language)

            values = [profile.get(c, 0) for c in categories]
            values += values[:1]

            ax.plot(angles, values, "o-", linewidth=2, color=colors[idx])
            ax.fill(angles, values, alpha=0.25, color=colors[idx])
            ax.set_ylim(0, 3)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(
                [self._short_name(c) for c in categories],
                size=8,
            )
            ax.set_title(model, size=12, fontweight="bold", pad=15)
            ax.set_yticks([1, 2, 3])
            ax.set_yticklabels(["1", "2", "3"], size=7)

        # Hide unused subplots
        for idx in range(n_models, len(axes)):
            axes[idx].set_visible(False)

        lang_label = "Hindi" if language == "hi" else "English"
        fig.suptitle(
            f"Model Task Profiles — {lang_label}",
            fontsize=14, fontweight="bold", y=1.02,
        )
        plt.tight_layout()
        return fig

    def plot_task_rankings_bar(self, language: str = "hi"):
        """
        Generate grouped bar chart with model scores per task.

        Args:
            language: Language to plot.

        Returns:
            matplotlib Figure.
        """
        import matplotlib.pyplot as plt

        categories = sorted(self.matrix.categories)
        models = self.matrix.models
        n_models = len(models)
        n_cats = len(categories)

        fig, ax = plt.subplots(figsize=(max(10, n_cats * 1.5), 6))

        x = np.arange(n_cats)
        width = 0.8 / n_models
        colors = plt.cm.Set2(np.linspace(0, 1, n_models))

        for i, model in enumerate(models):
            profile = self.matrix.get_model_profile(model, language)
            scores = [profile.get(c, 0) for c in categories]
            offset = (i - n_models / 2 + 0.5) * width
            bars = ax.bar(x + offset, scores, width, label=model, color=colors[i])

            # Add score labels on bars
            for bar, score in zip(bars, scores):
                if score > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.05,
                        f"{score:.1f}",
                        ha="center", va="bottom", fontsize=7,
                    )

        ax.set_xlabel("Task Category")
        ax.set_ylabel("Score (0-3)")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [self._short_name(c) for c in categories],
            rotation=45, ha="right",
        )
        ax.set_ylim(0, 3.5)
        ax.legend(loc="upper right")
        ax.axhline(y=2, color="gray", linestyle="--", alpha=0.3, label="Good threshold")

        lang_label = "Hindi" if language == "hi" else "English"
        ax.set_title(
            f"Task Fitness Comparison — {lang_label}",
            fontsize=14, fontweight="bold",
        )
        plt.tight_layout()
        return fig

    def plot_hindi_vs_english_comparison(self):
        """
        Generate side-by-side Hindi vs English comparison.

        Returns:
            matplotlib Figure or None if both languages not available.
        """
        import matplotlib.pyplot as plt

        if "en" not in self.matrix.languages or "hi" not in self.matrix.languages:
            return None

        models = self.matrix.models
        n_models = len(models)

        fig, ax = plt.subplots(figsize=(max(8, n_models * 2), 6))

        x = np.arange(n_models)
        width = 0.35

        en_avgs = []
        hi_avgs = []
        for model in models:
            en_profile = self.matrix.get_model_profile(model, "en")
            hi_profile = self.matrix.get_model_profile(model, "hi")

            en_avg = sum(en_profile.values()) / len(en_profile) if en_profile else 0
            hi_avg = sum(hi_profile.values()) / len(hi_profile) if hi_profile else 0

            en_avgs.append(en_avg)
            hi_avgs.append(hi_avg)

        bars_en = ax.bar(x - width / 2, en_avgs, width, label="English", color="#4C72B0")
        bars_hi = ax.bar(x + width / 2, hi_avgs, width, label="Hindi", color="#DD8452")

        # Add value labels
        for bar in bars_en:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                f"{bar.get_height():.2f}",
                ha="center", va="bottom", fontsize=9,
            )
        for bar in bars_hi:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                f"{bar.get_height():.2f}",
                ha="center", va="bottom", fontsize=9,
            )

        ax.set_xlabel("Model")
        ax.set_ylabel("Average Score (0-3)")
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_ylim(0, 3.5)
        ax.legend()
        ax.set_title(
            "Hindi vs English Performance Comparison",
            fontsize=14, fontweight="bold",
        )
        plt.tight_layout()
        return fig

    def save_all_plots(self, output_dir: Path, language: str = "hi") -> List[Path]:
        """
        Save all visualization plots to a directory.

        Args:
            output_dir: Directory to save plots.
            language: Primary language for plots.

        Returns:
            List of saved file paths.
        """
        import matplotlib.pyplot as plt

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        saved = []

        # Heatmap
        try:
            fig = self.plot_fitness_heatmap(language)
            path = output_dir / "fitness_heatmap.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved.append(path)
            self._logger.info(f"Saved: {path}")
        except Exception as e:
            self._logger.warning(f"Failed to save heatmap: {e}")

        # Radar
        try:
            fig = self.plot_radar_per_model(language)
            if fig:
                path = output_dir / "model_radar_profiles.png"
                fig.savefig(path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                saved.append(path)
                self._logger.info(f"Saved: {path}")
        except Exception as e:
            self._logger.warning(f"Failed to save radar: {e}")

        # Bar chart
        try:
            fig = self.plot_task_rankings_bar(language)
            path = output_dir / "task_rankings_bar.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved.append(path)
            self._logger.info(f"Saved: {path}")
        except Exception as e:
            self._logger.warning(f"Failed to save bar chart: {e}")

        # Hindi vs English comparison
        try:
            fig = self.plot_hindi_vs_english_comparison()
            if fig:
                path = output_dir / "hindi_vs_english.png"
                fig.savefig(path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                saved.append(path)
                self._logger.info(f"Saved: {path}")
        except Exception as e:
            self._logger.warning(f"Failed to save comparison: {e}")

        # Also generate for English if available
        if "en" in self.matrix.languages and language != "en":
            try:
                fig = self.plot_fitness_heatmap("en")
                path = output_dir / "fitness_heatmap_english.png"
                fig.savefig(path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                saved.append(path)
            except Exception as e:
                self._logger.warning(f"Failed to save English heatmap: {e}")

        self._logger.info(f"Saved {len(saved)} plots to {output_dir}")
        return saved
