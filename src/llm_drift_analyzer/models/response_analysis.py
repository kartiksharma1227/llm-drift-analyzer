"""
Response analysis data models for LLM Drift Analyzer.

This module defines the ResponseAnalysis dataclass for storing
and managing LLM response evaluation results.
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path


@dataclass
class ResponseAnalysis:
    """
    Stores analysis results for a single LLM response.

    Contains all metrics collected during drift analysis including
    token counts, latency, and quality scores.

    Attributes:
        prompt_id: Identifier of the prompt that generated this response.
        model_version: Full model identifier (e.g., 'gpt-4-0613').
        response_text: The complete response text from the LLM.
        token_count: Number of tokens in the response.
        latency_ms: API response time in milliseconds.
        instruction_score: Instruction adherence score (0-3).
            - 0: Does not follow instructions
            - 1: Partially follows instructions
            - 2: Mostly follows instructions
            - 3: Perfectly follows instructions
        factuality_score: Factual accuracy score (0-2).
            - 0: Contains significant factual errors
            - 1: Mostly factual with minor errors
            - 2: Completely factual
        tone_score: Tone and style appropriateness score (0-2).
            - 0: Inappropriate tone for context
            - 1: Adequate tone with some inconsistencies
            - 2: Appropriate and consistent tone
        timestamp: ISO format timestamp of when analysis was performed.
        iteration: Iteration number for repeated sampling.
        metadata: Optional dictionary for additional data.

    Example:
        >>> analysis = ResponseAnalysis(
        ...     prompt_id="IF-001",
        ...     model_version="gpt-4-0613",
        ...     response_text="The benefits of renewable energy are...",
        ...     token_count=47,
        ...     latency_ms=1250.5,
        ...     instruction_score=3,
        ...     factuality_score=2,
        ...     tone_score=2
        ... )
        >>> print(analysis.total_quality_score)
        7
    """
    prompt_id: str
    model_version: str
    response_text: str
    token_count: int
    latency_ms: float
    instruction_score: int
    factuality_score: int
    tone_score: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    iteration: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate score ranges after initialization."""
        self._validate_scores()

    def _validate_scores(self) -> None:
        """
        Validate that scores are within expected ranges.

        Raises:
            ValueError: If any score is out of range.
        """
        if not 0 <= self.instruction_score <= 3:
            raise ValueError(
                f"instruction_score must be 0-3, got {self.instruction_score}"
            )
        if not 0 <= self.factuality_score <= 2:
            raise ValueError(
                f"factuality_score must be 0-2, got {self.factuality_score}"
            )
        if not 0 <= self.tone_score <= 2:
            raise ValueError(
                f"tone_score must be 0-2, got {self.tone_score}"
            )

    @property
    def total_quality_score(self) -> int:
        """
        Calculate total quality score across all metrics.

        Returns:
            int: Sum of instruction, factuality, and tone scores (0-7).
        """
        return self.instruction_score + self.factuality_score + self.tone_score

    @property
    def normalized_quality_score(self) -> float:
        """
        Calculate normalized quality score as percentage.

        Returns:
            float: Quality score as percentage (0.0-1.0).
        """
        max_score = 3 + 2 + 2  # instruction + factuality + tone
        return self.total_quality_score / max_score

    @property
    def tokens_per_second(self) -> float:
        """
        Calculate token generation rate.

        Returns:
            float: Tokens per second, or 0 if latency is 0.
        """
        if self.latency_ms <= 0:
            return 0.0
        return (self.token_count / self.latency_ms) * 1000

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert ResponseAnalysis to dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation including computed properties.

        Example:
            >>> analysis = ResponseAnalysis(...)
            >>> data = analysis.to_dict()
            >>> print(data['total_quality_score'])
            7
        """
        result = asdict(self)
        result['total_quality_score'] = self.total_quality_score
        result['normalized_quality_score'] = self.normalized_quality_score
        result['tokens_per_second'] = self.tokens_per_second
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResponseAnalysis":
        """
        Create ResponseAnalysis from dictionary.

        Ignores computed properties that are not part of the dataclass.

        Args:
            data: Dictionary containing response analysis data.

        Returns:
            ResponseAnalysis: New ResponseAnalysis instance.

        Example:
            >>> data = {"prompt_id": "IF-001", "model_version": "gpt-4", ...}
            >>> analysis = ResponseAnalysis.from_dict(data)
        """
        # Remove computed properties if present
        clean_data = {k: v for k, v in data.items() if k not in [
            'total_quality_score', 'normalized_quality_score', 'tokens_per_second'
        ]}
        return cls(**clean_data)


class AnalysisResultSet:
    """
    Collection of response analysis results.

    Provides methods for aggregating, filtering, and exporting
    drift analysis results.

    Attributes:
        results: List of ResponseAnalysis objects.
        metadata: Optional metadata about the analysis run.

    Example:
        >>> result_set = AnalysisResultSet()
        >>> result_set.add(analysis)
        >>> gpt4_results = result_set.filter_by_model("gpt-4")
        >>> result_set.save_to_file("results.json")
    """

    def __init__(
        self,
        results: Optional[List[ResponseAnalysis]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize AnalysisResultSet.

        Args:
            results: Initial list of results.
            metadata: Optional metadata dictionary.
        """
        self.results: List[ResponseAnalysis] = results or []
        self.metadata: Dict[str, Any] = metadata or {}

    def __len__(self) -> int:
        """Return number of results."""
        return len(self.results)

    def __iter__(self):
        """Iterate over results."""
        return iter(self.results)

    def __getitem__(self, index: int) -> ResponseAnalysis:
        """Get result by index."""
        return self.results[index]

    def add(self, result: ResponseAnalysis) -> None:
        """
        Add a result to the set.

        Args:
            result: ResponseAnalysis to add.
        """
        self.results.append(result)

    def extend(self, results: List[ResponseAnalysis]) -> None:
        """
        Add multiple results to the set.

        Args:
            results: List of ResponseAnalysis to add.
        """
        self.results.extend(results)

    def filter_by_model(self, model: str) -> List[ResponseAnalysis]:
        """
        Filter results by model version.

        Args:
            model: Model version string (partial match supported).

        Returns:
            List[ResponseAnalysis]: Matching results.

        Example:
            >>> gpt4_results = result_set.filter_by_model("gpt-4")
        """
        model_lower = model.lower()
        return [r for r in self.results if model_lower in r.model_version.lower()]

    def filter_by_prompt(self, prompt_id: str) -> List[ResponseAnalysis]:
        """
        Filter results by prompt ID.

        Args:
            prompt_id: Prompt identifier.

        Returns:
            List[ResponseAnalysis]: Matching results.
        """
        return [r for r in self.results if r.prompt_id == prompt_id]

    def get_models(self) -> List[str]:
        """
        Get unique model versions in the result set.

        Returns:
            List[str]: Unique model versions.
        """
        return list(set(r.model_version for r in self.results))

    def get_prompts(self) -> List[str]:
        """
        Get unique prompt IDs in the result set.

        Returns:
            List[str]: Unique prompt IDs.
        """
        return list(set(r.prompt_id for r in self.results))

    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Calculate summary statistics across all results.

        Returns:
            Dict[str, Any]: Summary statistics including means and ranges.
        """
        if not self.results:
            return {}

        token_counts = [r.token_count for r in self.results]
        latencies = [r.latency_ms for r in self.results]
        instruction_scores = [r.instruction_score for r in self.results]
        factuality_scores = [r.factuality_score for r in self.results]
        tone_scores = [r.tone_score for r in self.results]

        return {
            "total_results": len(self.results),
            "unique_models": len(self.get_models()),
            "unique_prompts": len(self.get_prompts()),
            "token_count": {
                "mean": sum(token_counts) / len(token_counts),
                "min": min(token_counts),
                "max": max(token_counts),
            },
            "latency_ms": {
                "mean": sum(latencies) / len(latencies),
                "min": min(latencies),
                "max": max(latencies),
            },
            "instruction_score": {
                "mean": sum(instruction_scores) / len(instruction_scores),
            },
            "factuality_score": {
                "mean": sum(factuality_scores) / len(factuality_scores),
            },
            "tone_score": {
                "mean": sum(tone_scores) / len(tone_scores),
            },
        }

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert AnalysisResultSet to dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation.
        """
        return {
            "metadata": self.metadata,
            "summary": self.get_summary_stats(),
            "results": [r.to_dict() for r in self.results],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalysisResultSet":
        """
        Create AnalysisResultSet from dictionary.

        Args:
            data: Dictionary containing result set data.

        Returns:
            AnalysisResultSet: New instance.
        """
        results = [
            ResponseAnalysis.from_dict(r)
            for r in data.get("results", [])
        ]
        return cls(
            results=results,
            metadata=data.get("metadata", {}),
        )

    def save_to_file(self, file_path: Path) -> None:
        """
        Save results to JSON file.

        Args:
            file_path: Path to save the file.
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load_from_file(cls, file_path: Path) -> "AnalysisResultSet":
        """
        Load results from JSON file.

        Args:
            file_path: Path to the JSON file.

        Returns:
            AnalysisResultSet: Loaded result set.
        """
        file_path = Path(file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls.from_dict(data)
