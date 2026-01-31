"""
Visualization module for LLM Drift Analysis.

This module provides visualization capabilities using matplotlib
and seaborn for drift analysis results.
"""

from pathlib import Path
from typing import Optional, List, Tuple
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from llm_drift_analyzer.models.response_analysis import AnalysisResultSet
from llm_drift_analyzer.analyzers.statistical_analyzer import DriftStatisticalAnalyzer
from llm_drift_analyzer.utils.logger import get_logger


class DriftVisualizer:
    """
    Visualization tools for drift analysis results.

    Creates publication-ready charts for drift analysis including:
    - Token distribution comparisons
    - Score distributions
    - Latency trends
    - Change point visualizations
    - Correlation heatmaps

    Attributes:
        results: Analysis results to visualize.
        df: DataFrame representation of results.
        style: Matplotlib/seaborn style to use.

    Example:
        >>> visualizer = DriftVisualizer(results)
        >>> visualizer.plot_token_distribution()
        >>> visualizer.save_all_plots("output/charts/")
    """

    def __init__(
        self,
        results: AnalysisResultSet,
        style: str = "whitegrid"
    ):
        """
        Initialize the visualizer.

        Args:
            results: AnalysisResultSet to visualize.
            style: Seaborn style to use (whitegrid, darkgrid, etc.).
        """
        self.results = results
        self.df = pd.DataFrame([r.to_dict() for r in results.results])
        self.style = style
        self._logger = get_logger("reporters.visualizer")

        # Set style
        sns.set_style(style)
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['figure.dpi'] = 100

    def plot_token_distribution(
        self,
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Plot token count distribution across models.

        Creates a combined box plot and violin plot showing
        token count distributions for each model version.

        Args:
            figsize: Figure size (width, height).
            save_path: Optional path to save the figure.

        Returns:
            plt.Figure: Matplotlib figure object.

        Example:
            >>> fig = visualizer.plot_token_distribution()
            >>> plt.show()
        """
        self._logger.info("Creating token distribution plot")

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Box plot
        sns.boxplot(
            data=self.df,
            x='model_version',
            y='token_count',
            ax=axes[0],
            palette='Set2'
        )
        axes[0].set_title('Token Count Distribution by Model')
        axes[0].set_xlabel('Model Version')
        axes[0].set_ylabel('Token Count')
        axes[0].tick_params(axis='x', rotation=45)

        # Violin plot
        sns.violinplot(
            data=self.df,
            x='model_version',
            y='token_count',
            ax=axes[1],
            palette='Set2'
        )
        axes[1].set_title('Token Count Density by Model')
        axes[1].set_xlabel('Model Version')
        axes[1].set_ylabel('Token Count')
        axes[1].tick_params(axis='x', rotation=45)

        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path)

        return fig

    def plot_score_comparison(
        self,
        figsize: Tuple[int, int] = (14, 5),
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Plot quality score comparisons across models.

        Creates bar charts comparing instruction, factuality,
        and tone scores across model versions.

        Args:
            figsize: Figure size (width, height).
            save_path: Optional path to save the figure.

        Returns:
            plt.Figure: Matplotlib figure object.
        """
        self._logger.info("Creating score comparison plot")

        fig, axes = plt.subplots(1, 3, figsize=figsize)
        scores = ['instruction_score', 'factuality_score', 'tone_score']
        titles = ['Instruction Adherence', 'Factuality', 'Tone']
        max_scores = [3, 2, 2]

        for ax, score, title, max_score in zip(axes, scores, titles, max_scores):
            score_means = self.df.groupby('model_version')[score].mean()
            score_stds = self.df.groupby('model_version')[score].std()

            bars = ax.bar(
                range(len(score_means)),
                score_means.values,
                yerr=score_stds.values,
                capsize=5,
                color=sns.color_palette('Set2'),
                edgecolor='black',
                linewidth=1
            )

            ax.set_title(f'{title} Score')
            ax.set_xlabel('Model Version')
            ax.set_ylabel('Score')
            ax.set_xticks(range(len(score_means)))
            ax.set_xticklabels(score_means.index, rotation=45, ha='right')
            ax.set_ylim(0, max_score + 0.5)
            ax.axhline(y=max_score, color='gray', linestyle='--', alpha=0.5, label='Max Score')

        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path)

        return fig

    def plot_latency_distribution(
        self,
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Plot latency distribution across models.

        Args:
            figsize: Figure size (width, height).
            save_path: Optional path to save the figure.

        Returns:
            plt.Figure: Matplotlib figure object.
        """
        self._logger.info("Creating latency distribution plot")

        fig, ax = plt.subplots(figsize=figsize)

        sns.boxplot(
            data=self.df,
            x='model_version',
            y='latency_ms',
            ax=ax,
            palette='Set3'
        )

        ax.set_title('Response Latency by Model')
        ax.set_xlabel('Model Version')
        ax.set_ylabel('Latency (ms)')
        ax.tick_params(axis='x', rotation=45)

        # Add mean markers
        means = self.df.groupby('model_version')['latency_ms'].mean()
        for i, mean in enumerate(means.values):
            ax.scatter(i, mean, color='red', s=100, zorder=5, marker='D', label='Mean' if i == 0 else '')

        ax.legend()
        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path)

        return fig

    def plot_drift_timeline(
        self,
        metric: str = 'token_count',
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Plot metric values over iterations to visualize drift.

        Args:
            metric: Metric to plot (token_count, instruction_score, etc.).
            figsize: Figure size (width, height).
            save_path: Optional path to save the figure.

        Returns:
            plt.Figure: Matplotlib figure object.
        """
        self._logger.info(f"Creating drift timeline for {metric}")

        fig, ax = plt.subplots(figsize=figsize)

        models = self.df['model_version'].unique()
        colors = sns.color_palette('husl', len(models))

        for model, color in zip(models, colors):
            model_data = self.df[self.df['model_version'] == model]

            # Plot individual points
            ax.scatter(
                range(len(model_data)),
                model_data[metric].values,
                alpha=0.5,
                color=color,
                label=model,
                s=30
            )

            # Plot rolling mean
            if len(model_data) >= 3:
                rolling_mean = model_data[metric].rolling(window=3, min_periods=1).mean()
                ax.plot(
                    range(len(model_data)),
                    rolling_mean.values,
                    color=color,
                    linewidth=2,
                    alpha=0.8
                )

        ax.set_title(f'{metric.replace("_", " ").title()} Over Iterations')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.legend(loc='upper right')

        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path)

        return fig

    def plot_change_points(
        self,
        metric: str = 'token_count',
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Visualize CUSUM change point detection.

        Args:
            metric: Metric to analyze.
            figsize: Figure size (width, height).
            save_path: Optional path to save the figure.

        Returns:
            plt.Figure: Matplotlib figure object.
        """
        self._logger.info(f"Creating change point visualization for {metric}")

        analyzer = DriftStatisticalAnalyzer(self.results)
        change_points = analyzer.detect_change_points(metric)

        models = list(change_points.keys())
        n_models = len(models)

        fig, axes = plt.subplots(n_models, 2, figsize=(figsize[0], figsize[1] * n_models / 2))

        if n_models == 1:
            axes = axes.reshape(1, -1)

        colors = sns.color_palette('Set2', n_models)

        for idx, (model, data) in enumerate(change_points.items()):
            if 'note' in data:
                axes[idx, 0].text(0.5, 0.5, data['note'], ha='center', va='center')
                axes[idx, 1].text(0.5, 0.5, data['note'], ha='center', va='center')
                continue

            # Raw data plot
            model_data = self.df[self.df['model_version'] == model][metric].values
            axes[idx, 0].plot(model_data, color=colors[idx], linewidth=1)
            axes[idx, 0].axhline(y=data['mean'], color='red', linestyle='--', label='Mean')
            axes[idx, 0].fill_between(
                range(len(model_data)),
                data['mean'] - data['std'],
                data['mean'] + data['std'],
                alpha=0.2,
                color='red'
            )
            axes[idx, 0].set_title(f'{model} - {metric}')
            axes[idx, 0].set_xlabel('Sample Index')
            axes[idx, 0].set_ylabel(metric)
            axes[idx, 0].legend()

            # CUSUM plot
            cumsum = np.array(data['cumsum'])
            axes[idx, 1].plot(cumsum, color=colors[idx], linewidth=1)
            axes[idx, 1].axhline(y=data['threshold'], color='red', linestyle='--', label='Threshold')
            axes[idx, 1].axhline(y=-data['threshold'], color='red', linestyle='--')

            # Mark change points
            for cp in data['change_points']:
                cp_idx = cp['index'] if isinstance(cp, dict) else cp
                axes[idx, 1].axvline(x=cp_idx, color='orange', linestyle='-', alpha=0.7)

            axes[idx, 1].set_title(f'{model} - CUSUM')
            axes[idx, 1].set_xlabel('Sample Index')
            axes[idx, 1].set_ylabel('Cumulative Sum')
            axes[idx, 1].legend()

        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path)

        return fig

    def plot_correlation_heatmap(
        self,
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Plot correlation heatmap for numeric metrics.

        Args:
            figsize: Figure size (width, height).
            save_path: Optional path to save the figure.

        Returns:
            plt.Figure: Matplotlib figure object.
        """
        self._logger.info("Creating correlation heatmap")

        numeric_cols = [
            'token_count', 'latency_ms',
            'instruction_score', 'factuality_score', 'tone_score'
        ]
        available_cols = [c for c in numeric_cols if c in self.df.columns]

        corr_matrix = self.df[available_cols].corr()

        fig, ax = plt.subplots(figsize=figsize)

        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap='RdYlBu_r',
            center=0,
            fmt='.2f',
            square=True,
            ax=ax,
            cbar_kws={'label': 'Correlation Coefficient'}
        )

        ax.set_title('Metric Correlation Heatmap')
        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path)

        return fig

    def plot_model_comparison_radar(
        self,
        figsize: Tuple[int, int] = (10, 10),
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Create radar chart comparing models across metrics.

        Args:
            figsize: Figure size (width, height).
            save_path: Optional path to save the figure.

        Returns:
            plt.Figure: Matplotlib figure object.
        """
        self._logger.info("Creating model comparison radar chart")

        # Calculate normalized metrics per model
        metrics = ['instruction_score', 'factuality_score', 'tone_score']
        max_values = [3, 2, 2]

        models = self.df['model_version'].unique()
        normalized_scores = {}

        for model in models:
            model_data = self.df[self.df['model_version'] == model]
            normalized_scores[model] = [
                model_data[metric].mean() / max_val
                for metric, max_val in zip(metrics, max_values)
            ]
            # Add normalized token count (inverse, lower is better for some use cases)
            # We'll normalize by the max observed
            max_tokens = self.df['token_count'].max()
            normalized_scores[model].append(
                1 - (model_data['token_count'].mean() / max_tokens)
            )

        metrics_display = ['Instruction', 'Factuality', 'Tone', 'Conciseness']

        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics_display), endpoint=False).tolist()
        angles += angles[:1]  # Complete the loop

        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

        colors = sns.color_palette('husl', len(models))

        for (model, scores), color in zip(normalized_scores.items(), colors):
            values = scores + scores[:1]  # Complete the loop
            ax.plot(angles, values, 'o-', linewidth=2, label=model, color=color)
            ax.fill(angles, values, alpha=0.25, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_display)
        ax.set_ylim(0, 1)
        ax.set_title('Model Comparison (Normalized Scores)', y=1.08)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path)

        return fig

    def save_all_plots(
        self,
        output_dir: Path,
        format: str = "png",
        dpi: int = 150
    ) -> List[Path]:
        """
        Generate and save all standard plots.

        Args:
            output_dir: Directory to save plots.
            format: Image format (png, pdf, svg).
            dpi: Resolution for raster formats.

        Returns:
            List[Path]: Paths to saved files.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self._logger.info(f"Saving all plots to {output_dir}")

        saved_files = []
        plots = [
            ('token_distribution', self.plot_token_distribution),
            ('score_comparison', self.plot_score_comparison),
            ('latency_distribution', self.plot_latency_distribution),
            ('drift_timeline', self.plot_drift_timeline),
            ('change_points', self.plot_change_points),
            ('correlation_heatmap', self.plot_correlation_heatmap),
            ('model_radar', self.plot_model_comparison_radar),
        ]

        for name, plot_func in plots:
            try:
                file_path = output_dir / f"{name}.{format}"
                fig = plot_func()
                fig.savefig(file_path, dpi=dpi, bbox_inches='tight')
                plt.close(fig)
                saved_files.append(file_path)
                self._logger.info(f"Saved {file_path}")
            except Exception as e:
                self._logger.warning(f"Failed to create {name}: {e}")

        return saved_files

    def _save_figure(self, fig: plt.Figure, path: Path, dpi: int = 150) -> None:
        """Save figure to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        self._logger.info(f"Saved figure to {path}")
