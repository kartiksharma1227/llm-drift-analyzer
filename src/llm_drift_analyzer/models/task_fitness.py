"""
Task fitness data models for LLM Drift Analyzer.

Stores model × task fitness scores and provides methods for
ranking models per task, generating recommendations, and serialization.
"""

import json
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path


@dataclass
class TaskFitnessScore:
    """
    Score for a single model on a single task category in a single language.

    Attributes:
        model: Model identifier (e.g., "llama3", "mistral").
        category: PromptCategory value (e.g., "summarization").
        language: Language code ("en" or "hi").
        mean_score: Average score across prompts and iterations.
        std_score: Standard deviation of scores.
        sample_count: Number of evaluation samples.
        latency_mean_ms: Average response latency in milliseconds.
        individual_scores: Raw list of all scores for statistical analysis.
    """
    model: str
    category: str
    language: str
    mean_score: float
    std_score: float
    sample_count: int
    latency_mean_ms: float
    individual_scores: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "category": self.category,
            "language": self.language,
            "mean_score": round(self.mean_score, 3),
            "std_score": round(self.std_score, 3),
            "sample_count": self.sample_count,
            "latency_mean_ms": round(self.latency_mean_ms, 1),
            "individual_scores": [round(s, 2) for s in self.individual_scores],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskFitnessScore":
        return cls(
            model=data["model"],
            category=data["category"],
            language=data["language"],
            mean_score=data["mean_score"],
            std_score=data["std_score"],
            sample_count=data["sample_count"],
            latency_mean_ms=data["latency_mean_ms"],
            individual_scores=data.get("individual_scores", []),
        )

    @classmethod
    def from_raw_scores(
        cls,
        model: str,
        category: str,
        language: str,
        scores: List[float],
        latencies: List[float],
    ) -> "TaskFitnessScore":
        """Create from raw score and latency lists."""
        if not scores:
            return cls(
                model=model, category=category, language=language,
                mean_score=0.0, std_score=0.0, sample_count=0,
                latency_mean_ms=0.0, individual_scores=[],
            )
        return cls(
            model=model,
            category=category,
            language=language,
            mean_score=statistics.mean(scores),
            std_score=statistics.stdev(scores) if len(scores) > 1 else 0.0,
            sample_count=len(scores),
            latency_mean_ms=statistics.mean(latencies) if latencies else 0.0,
            individual_scores=scores,
        )


@dataclass
class TaskFitnessMatrix:
    """
    Model × Task matrix of fitness scores.

    Provides methods for querying best models per task,
    generating recommendations, and serialization.
    """
    scores: List[TaskFitnessScore] = field(default_factory=list)
    models: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_score(self, score: TaskFitnessScore) -> None:
        """Add a fitness score to the matrix."""
        self.scores.append(score)
        if score.model not in self.models:
            self.models.append(score.model)
        if score.category not in self.categories:
            self.categories.append(score.category)
        if score.language not in self.languages:
            self.languages.append(score.language)

    def get_score(
        self, model: str, category: str, language: str
    ) -> Optional[TaskFitnessScore]:
        """Get score for a specific model/category/language combo."""
        for s in self.scores:
            if s.model == model and s.category == category and s.language == language:
                return s
        return None

    def get_best_model_for_task(
        self, category: str, language: str
    ) -> Tuple[str, float]:
        """
        Get the best model for a given task category and language.

        Returns:
            Tuple of (model_name, mean_score).
        """
        relevant = [
            s for s in self.scores
            if s.category == category and s.language == language
        ]
        if not relevant:
            return ("none", 0.0)

        best = max(relevant, key=lambda s: s.mean_score)
        return (best.model, best.mean_score)

    def get_model_profile(
        self, model: str, language: str
    ) -> Dict[str, float]:
        """
        Get a model's score profile across all task categories.

        Returns:
            Dict mapping category name to mean score.
        """
        profile = {}
        for s in self.scores:
            if s.model == model and s.language == language:
                profile[s.category] = s.mean_score
        return profile

    def get_task_rankings(
        self, category: str, language: str
    ) -> List[Tuple[str, float]]:
        """
        Get all models ranked by score for a task category.

        Returns:
            List of (model, mean_score) tuples, sorted descending.
        """
        relevant = [
            s for s in self.scores
            if s.category == category and s.language == language
        ]
        ranked = sorted(relevant, key=lambda s: s.mean_score, reverse=True)
        return [(s.model, s.mean_score) for s in ranked]

    def get_recommendations(
        self, language: str = "hi"
    ) -> List[Dict[str, Any]]:
        """
        Generate per-task model recommendations.

        Returns:
            List of dicts with: category, recommended_model, score,
            runner_up, runner_up_score.
        """
        recommendations = []
        for category in self.categories:
            rankings = self.get_task_rankings(category, language)
            if not rankings:
                continue

            rec = {
                "category": category,
                "recommended_model": rankings[0][0],
                "score": round(rankings[0][1], 2),
                "runner_up": rankings[1][0] if len(rankings) > 1 else "N/A",
                "runner_up_score": round(rankings[1][1], 2) if len(rankings) > 1 else 0.0,
                "all_rankings": [
                    {"model": m, "score": round(s, 2)} for m, s in rankings
                ],
            }
            recommendations.append(rec)

        return recommendations

    def get_overall_rankings(
        self, language: str = "hi"
    ) -> List[Tuple[str, float]]:
        """
        Get models ranked by average score across all task categories.

        Returns:
            List of (model, avg_score) tuples, sorted descending.
        """
        model_totals: Dict[str, List[float]] = {}
        for s in self.scores:
            if s.language == language:
                model_totals.setdefault(s.model, []).append(s.mean_score)

        avg_scores = [
            (model, statistics.mean(scores))
            for model, scores in model_totals.items()
        ]
        return sorted(avg_scores, key=lambda x: x[1], reverse=True)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "models": self.models,
            "categories": self.categories,
            "languages": self.languages,
            "metadata": self.metadata,
            "scores": [s.to_dict() for s in self.scores],
            "recommendations": {
                lang: self.get_recommendations(lang)
                for lang in self.languages
            },
            "overall_rankings": {
                lang: [
                    {"model": m, "avg_score": round(s, 2)}
                    for m, s in self.get_overall_rankings(lang)
                ]
                for lang in self.languages
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskFitnessMatrix":
        scores = [TaskFitnessScore.from_dict(s) for s in data.get("scores", [])]
        return cls(
            scores=scores,
            models=data.get("models", []),
            categories=data.get("categories", []),
            languages=data.get("languages", []),
            generated_at=data.get("generated_at", datetime.now().isoformat()),
            metadata=data.get("metadata", {}),
        )

    def save_to_file(self, file_path: Path) -> None:
        """Save matrix to JSON file."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load_from_file(cls, file_path: Path) -> "TaskFitnessMatrix":
        """Load matrix from JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
