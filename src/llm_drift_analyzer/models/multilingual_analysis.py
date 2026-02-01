"""
Multilingual analysis data models for LLM Drift Analyzer.

This module extends the base response analysis with language-specific
metrics for Hindi and cross-lingual drift analysis.

Features:
- Language-specific response metrics
- Script consistency measurement
- Code-mixing detection and quantification
- Cross-lingual comparison data structures
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime

from llm_drift_analyzer.models.prompt import Language
from llm_drift_analyzer.utils.multilingual_tokenizer import (
    MultilingualTokenCounter,
    ScriptType,
    TextAnalysis,
)


@dataclass
class MultilingualMetrics:
    """
    Language-specific metrics for multilingual response analysis.

    Captures metrics that are particularly relevant for Hindi and
    code-mixed content analysis.

    Attributes:
        language: Detected language of the response.
        script_type: Primary script used (Devanagari/Latin/Mixed).
        devanagari_char_count: Count of Devanagari characters.
        latin_char_count: Count of Latin characters.
        code_mixing_ratio: Ratio of secondary script (0.0-1.0).
        script_consistency: How consistently the expected script is used (0.0-1.0).
        syllable_count: Hindi syllable count (for Devanagari text).
        word_count: Word count (language-aware).
        unique_devanagari_chars: Vocabulary richness for Devanagari.
        transliteration_detected: Whether transliterated text detected.

    Example:
        >>> metrics = MultilingualMetrics(
        ...     language="hi",
        ...     script_type="devanagari",
        ...     devanagari_char_count=150,
        ...     latin_char_count=10,
        ...     code_mixing_ratio=0.067,
        ...     script_consistency=0.94,
        ...     syllable_count=45
        ... )
    """
    language: str
    script_type: str
    devanagari_char_count: int
    latin_char_count: int
    code_mixing_ratio: float
    script_consistency: float
    syllable_count: int
    word_count: int
    unique_devanagari_chars: int = 0
    transliteration_detected: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "language": self.language,
            "script_type": self.script_type,
            "devanagari_char_count": self.devanagari_char_count,
            "latin_char_count": self.latin_char_count,
            "code_mixing_ratio": self.code_mixing_ratio,
            "script_consistency": self.script_consistency,
            "syllable_count": self.syllable_count,
            "word_count": self.word_count,
            "unique_devanagari_chars": self.unique_devanagari_chars,
            "transliteration_detected": self.transliteration_detected,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MultilingualMetrics":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_text(
        cls,
        text: str,
        expected_language: str = "hi"
    ) -> "MultilingualMetrics":
        """
        Create metrics from text analysis.

        Args:
            text: Response text to analyze.
            expected_language: Expected language code.

        Returns:
            MultilingualMetrics: Computed metrics.
        """
        counter = MultilingualTokenCounter()
        analysis = counter.analyze_text(text)
        detected_lang = counter.detect_language(text)

        # Calculate script consistency based on expected language
        expected_script = ScriptType.DEVANAGARI if expected_language == "hi" else ScriptType.LATIN
        script_consistency = counter.get_script_consistency_score(text, expected_script)

        # Detect transliteration (Roman-script Hindi)
        transliteration = cls._detect_transliteration(text, analysis)

        return cls(
            language=detected_lang,
            script_type=analysis.script_type.value,
            devanagari_char_count=analysis.devanagari_chars,
            latin_char_count=analysis.latin_chars,
            code_mixing_ratio=analysis.code_mixing_ratio,
            script_consistency=script_consistency,
            syllable_count=analysis.syllable_count,
            word_count=analysis.word_count,
            unique_devanagari_chars=analysis.unique_devanagari_chars,
            transliteration_detected=transliteration,
        )

    @staticmethod
    def _detect_transliteration(text: str, analysis: TextAnalysis) -> bool:
        """
        Detect if text might be transliterated Hindi (Roman script).

        Uses heuristics like Hindi word patterns in Latin script.
        """
        if analysis.script_type != ScriptType.LATIN:
            return False

        # Common Hindi words often seen in transliteration
        hindi_markers = [
            "hai", "hain", "ka", "ki", "ke", "ko", "se", "me", "mein",
            "aur", "ya", "nahi", "kya", "kyun", "kaise", "kab", "kahan",
            "tum", "aap", "hum", "main", "woh", "yeh", "jo", "jab",
            "lekin", "par", "phir", "abhi", "bahut", "thoda", "zyada"
        ]

        text_lower = text.lower()
        matches = sum(1 for marker in hindi_markers if f" {marker} " in f" {text_lower} ")

        # If multiple Hindi markers found, likely transliteration
        return matches >= 3


@dataclass
class MultilingualResponseAnalysis:
    """
    Extended response analysis with multilingual metrics.

    Combines standard drift analysis metrics with language-specific
    measurements for Hindi and cross-lingual comparison.

    Attributes:
        prompt_id: Identifier of the prompt.
        model_version: Model identifier.
        response_text: Complete response text.
        expected_language: Expected response language.
        token_count: Token count (language-aware).
        latency_ms: Response latency.
        instruction_score: Instruction adherence (0-3).
        factuality_score: Factual accuracy (0-2).
        tone_score: Tone appropriateness (0-2).
        multilingual_metrics: Language-specific metrics.
        timestamp: Analysis timestamp.
        iteration: Iteration number.
        metadata: Additional metadata.
    """
    prompt_id: str
    model_version: str
    response_text: str
    expected_language: str
    token_count: int
    latency_ms: float
    instruction_score: int
    factuality_score: int
    tone_score: int
    multilingual_metrics: MultilingualMetrics
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    iteration: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_quality_score(self) -> int:
        """Total quality score (0-7)."""
        return self.instruction_score + self.factuality_score + self.tone_score

    @property
    def language_fidelity_score(self) -> float:
        """
        Score for maintaining expected language (0.0-1.0).

        Combines script consistency and absence of unwanted code-mixing.
        """
        base_score = self.multilingual_metrics.script_consistency

        # Penalize unexpected code-mixing
        if self.expected_language == "hi":
            # For Hindi, some English technical terms are acceptable
            acceptable_mixing = 0.15
        else:
            # For English, any Devanagari is unexpected
            acceptable_mixing = 0.05

        mixing_penalty = max(0, self.multilingual_metrics.code_mixing_ratio - acceptable_mixing)
        return max(0, base_score - mixing_penalty)

    @property
    def composite_multilingual_score(self) -> float:
        """
        Composite score combining quality and language fidelity.

        Returns:
            float: Score from 0.0 to 1.0.
        """
        quality_normalized = self.total_quality_score / 7.0
        language_weight = 0.3  # 30% weight for language fidelity

        return (
            (1 - language_weight) * quality_normalized +
            language_weight * self.language_fidelity_score
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "prompt_id": self.prompt_id,
            "model_version": self.model_version,
            "response_text": self.response_text,
            "expected_language": self.expected_language,
            "token_count": self.token_count,
            "latency_ms": self.latency_ms,
            "instruction_score": self.instruction_score,
            "factuality_score": self.factuality_score,
            "tone_score": self.tone_score,
            "total_quality_score": self.total_quality_score,
            "language_fidelity_score": self.language_fidelity_score,
            "composite_multilingual_score": self.composite_multilingual_score,
            "multilingual_metrics": self.multilingual_metrics.to_dict(),
            "timestamp": self.timestamp,
            "iteration": self.iteration,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MultilingualResponseAnalysis":
        """Create from dictionary."""
        ml_metrics = MultilingualMetrics.from_dict(data.pop("multilingual_metrics"))

        # Remove computed properties
        for key in ["total_quality_score", "language_fidelity_score", "composite_multilingual_score"]:
            data.pop(key, None)

        return cls(multilingual_metrics=ml_metrics, **data)


@dataclass
class CrossLingualComparison:
    """
    Comparison between model performance in different languages.

    Compares the same model's performance on parallel prompts
    in English and Hindi.

    Attributes:
        model_version: Model being compared.
        prompt_id_english: English prompt ID.
        prompt_id_hindi: Hindi prompt ID.
        english_results: Analysis results for English.
        hindi_results: Analysis results for Hindi.
        performance_gap: Difference in performance metrics.
    """
    model_version: str
    prompt_id_english: str
    prompt_id_hindi: str
    english_results: List[MultilingualResponseAnalysis]
    hindi_results: List[MultilingualResponseAnalysis]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def performance_gap(self) -> Dict[str, float]:
        """
        Calculate performance gap between English and Hindi.

        Positive values indicate English > Hindi performance.
        """
        if not self.english_results or not self.hindi_results:
            return {}

        # Average metrics for each language
        en_instruction = sum(r.instruction_score for r in self.english_results) / len(self.english_results)
        hi_instruction = sum(r.instruction_score for r in self.hindi_results) / len(self.hindi_results)

        en_factuality = sum(r.factuality_score for r in self.english_results) / len(self.english_results)
        hi_factuality = sum(r.factuality_score for r in self.hindi_results) / len(self.hindi_results)

        en_tone = sum(r.tone_score for r in self.english_results) / len(self.english_results)
        hi_tone = sum(r.tone_score for r in self.hindi_results) / len(self.hindi_results)

        en_tokens = sum(r.token_count for r in self.english_results) / len(self.english_results)
        hi_tokens = sum(r.token_count for r in self.hindi_results) / len(self.hindi_results)

        en_latency = sum(r.latency_ms for r in self.english_results) / len(self.english_results)
        hi_latency = sum(r.latency_ms for r in self.hindi_results) / len(self.hindi_results)

        return {
            "instruction_score_gap": en_instruction - hi_instruction,
            "factuality_score_gap": en_factuality - hi_factuality,
            "tone_score_gap": en_tone - hi_tone,
            "token_count_ratio": hi_tokens / max(1, en_tokens),  # Ratio instead of gap
            "latency_ratio": hi_latency / max(1, en_latency),
            "total_quality_gap": (
                (en_instruction + en_factuality + en_tone) -
                (hi_instruction + hi_factuality + hi_tone)
            ),
        }

    @property
    def language_parity_score(self) -> float:
        """
        Calculate how close Hindi performance is to English (0.0-1.0).

        1.0 = perfect parity, 0.0 = complete disparity.
        """
        gap = self.performance_gap
        if not gap:
            return 0.0

        # Normalize gaps (instruction: 0-3, factuality/tone: 0-2)
        instruction_parity = 1 - abs(gap.get("instruction_score_gap", 0)) / 3
        factuality_parity = 1 - abs(gap.get("factuality_score_gap", 0)) / 2
        tone_parity = 1 - abs(gap.get("tone_score_gap", 0)) / 2

        return (instruction_parity + factuality_parity + tone_parity) / 3

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_version": self.model_version,
            "prompt_id_english": self.prompt_id_english,
            "prompt_id_hindi": self.prompt_id_hindi,
            "performance_gap": self.performance_gap,
            "language_parity_score": self.language_parity_score,
            "english_sample_size": len(self.english_results),
            "hindi_sample_size": len(self.hindi_results),
            "metadata": self.metadata,
        }


@dataclass
class MultilingualDriftReport:
    """
    Comprehensive report on multilingual drift analysis.

    Aggregates results across languages and models for
    cross-lingual drift comparison.

    Attributes:
        models: List of models analyzed.
        languages: List of languages analyzed.
        results_by_language: Results grouped by language.
        cross_lingual_comparisons: Pairwise language comparisons.
        summary_statistics: Aggregate statistics.
    """
    models: List[str]
    languages: List[str]
    results_by_language: Dict[str, List[MultilingualResponseAnalysis]]
    cross_lingual_comparisons: List[CrossLingualComparison]
    summary_statistics: Dict[str, Any] = field(default_factory=dict)

    def get_language_ranking(self, metric: str = "total_quality_score") -> List[tuple]:
        """
        Rank languages by average performance on a metric.

        Args:
            metric: Metric to rank by.

        Returns:
            List of (language, average_score) tuples, sorted descending.
        """
        rankings = []
        for lang, results in self.results_by_language.items():
            if not results:
                continue

            if metric == "total_quality_score":
                avg = sum(r.total_quality_score for r in results) / len(results)
            elif metric == "language_fidelity_score":
                avg = sum(r.language_fidelity_score for r in results) / len(results)
            elif metric == "composite_multilingual_score":
                avg = sum(r.composite_multilingual_score for r in results) / len(results)
            else:
                continue

            rankings.append((lang, avg))

        return sorted(rankings, key=lambda x: x[1], reverse=True)

    def get_model_language_matrix(self) -> Dict[str, Dict[str, float]]:
        """
        Create matrix of model performance by language.

        Returns:
            Nested dict: model -> language -> average score.
        """
        matrix = {}

        for lang, results in self.results_by_language.items():
            for result in results:
                model = result.model_version
                if model not in matrix:
                    matrix[model] = {}
                if lang not in matrix[model]:
                    matrix[model][lang] = {"scores": [], "count": 0}

                matrix[model][lang]["scores"].append(result.total_quality_score)
                matrix[model][lang]["count"] += 1

        # Convert to averages
        for model in matrix:
            for lang in matrix[model]:
                scores = matrix[model][lang]["scores"]
                matrix[model][lang] = sum(scores) / len(scores) if scores else 0

        return matrix

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "models": self.models,
            "languages": self.languages,
            "results_count": {
                lang: len(results)
                for lang, results in self.results_by_language.items()
            },
            "cross_lingual_comparisons": [
                c.to_dict() for c in self.cross_lingual_comparisons
            ],
            "language_ranking": self.get_language_ranking(),
            "model_language_matrix": self.get_model_language_matrix(),
            "summary_statistics": self.summary_statistics,
        }
