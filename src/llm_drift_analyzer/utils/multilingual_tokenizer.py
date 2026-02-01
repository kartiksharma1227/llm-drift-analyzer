"""
Multilingual token counting utilities for LLM Drift Analyzer.

This module provides language-aware token counting with special support
for Hindi (Devanagari script) and cross-lingual analysis metrics.

Features:
- Devanagari character and syllable counting
- Script detection (Devanagari, Latin, mixed)
- Code-mixing ratio calculation
- Language-specific token approximations
"""

import re
import unicodedata
from typing import Dict, Optional, Tuple, List
from enum import Enum
from dataclasses import dataclass

import tiktoken


class ScriptType(Enum):
    """
    Enumeration of script types for text analysis.

    Attributes:
        DEVANAGARI: Hindi/Sanskrit Devanagari script.
        LATIN: English/Roman alphabet.
        MIXED: Mix of multiple scripts (code-mixing).
        OTHER: Other scripts (Arabic, Chinese, etc.).
    """
    DEVANAGARI = "devanagari"
    LATIN = "latin"
    MIXED = "mixed"
    OTHER = "other"


@dataclass
class TextAnalysis:
    """
    Detailed analysis of text characteristics.

    Attributes:
        total_chars: Total character count.
        devanagari_chars: Count of Devanagari characters.
        latin_chars: Count of Latin characters.
        numeric_chars: Count of numeric characters.
        punctuation_chars: Count of punctuation characters.
        whitespace_chars: Count of whitespace characters.
        other_chars: Count of other characters.
        word_count: Approximate word count.
        syllable_count: Estimated syllable count (Hindi-aware).
        script_type: Detected primary script type.
        code_mixing_ratio: Ratio of non-primary script (0.0-1.0).
        unique_devanagari_chars: Set of unique Devanagari characters used.
    """
    total_chars: int
    devanagari_chars: int
    latin_chars: int
    numeric_chars: int
    punctuation_chars: int
    whitespace_chars: int
    other_chars: int
    word_count: int
    syllable_count: int
    script_type: ScriptType
    code_mixing_ratio: float
    unique_devanagari_chars: int


class MultilingualTokenCounter:
    """
    Language-aware token counter with special support for Hindi.

    Provides accurate token counting that accounts for:
    - Devanagari script characteristics
    - Different tokenization ratios for different languages
    - Code-mixing detection
    - Syllable-based analysis for Hindi

    Example:
        >>> counter = MultilingualTokenCounter()
        >>> # English text
        >>> en_count = counter.count_tokens("Hello, how are you?", language="en")
        >>> print(en_count)
        6
        >>> # Hindi text
        >>> hi_count = counter.count_tokens("नमस्ते, आप कैसे हैं?", language="hi")
        >>> print(hi_count)
        5
        >>> # Get detailed analysis
        >>> analysis = counter.analyze_text("यह एक test है।")
        >>> print(analysis.code_mixing_ratio)
        0.25
    """

    # Devanagari Unicode ranges
    DEVANAGARI_RANGE = (0x0900, 0x097F)  # Basic Devanagari
    DEVANAGARI_EXTENDED = (0xA8E0, 0xA8FF)  # Devanagari Extended
    DEVANAGARI_VEDIC = (0x1CD0, 0x1CFF)  # Vedic Extensions

    # Approximate tokens per character ratios for different languages
    _TOKENS_PER_CHAR: Dict[str, float] = {
        "en": 0.25,      # English: ~4 chars per token
        "hi": 0.40,      # Hindi: ~2.5 chars per token (Devanagari is denser)
        "hi-en": 0.32,   # Code-mixed: blend of both
        "claude": 0.25,
        "mistral": 0.27,
        "ollama": 0.30,  # Local models vary
        "default": 0.25,
    }

    # Hindi-specific: average syllables per token
    _SYLLABLES_PER_TOKEN_HI = 1.8

    def __init__(self):
        """Initialize multilingual token counter."""
        self._encoders: Dict[str, tiktoken.Encoding] = {}
        self._default_encoder: Optional[tiktoken.Encoding] = None

    def _is_devanagari(self, char: str) -> bool:
        """
        Check if a character is Devanagari script.

        Args:
            char: Single character to check.

        Returns:
            bool: True if character is Devanagari.
        """
        code_point = ord(char)
        return (
            (self.DEVANAGARI_RANGE[0] <= code_point <= self.DEVANAGARI_RANGE[1]) or
            (self.DEVANAGARI_EXTENDED[0] <= code_point <= self.DEVANAGARI_EXTENDED[1]) or
            (self.DEVANAGARI_VEDIC[0] <= code_point <= self.DEVANAGARI_VEDIC[1])
        )

    def _is_latin(self, char: str) -> bool:
        """
        Check if a character is Latin script.

        Args:
            char: Single character to check.

        Returns:
            bool: True if character is Latin alphabet.
        """
        return char.isalpha() and unicodedata.category(char).startswith('L') and not self._is_devanagari(char)

    def _count_hindi_syllables(self, text: str) -> int:
        """
        Count approximate syllables in Hindi text.

        Hindi syllable counting based on:
        - Vowel matras (dependent vowels)
        - Independent vowels
        - Consonant clusters with inherent 'a'

        Args:
            text: Hindi text to analyze.

        Returns:
            int: Estimated syllable count.
        """
        # Independent vowels
        independent_vowels = set("अआइईउऊऋएऐओऔ")

        # Dependent vowel signs (matras)
        matras = set("ािीुूृेैोौं")

        # Halant (virama) - removes inherent vowel
        halant = "्"

        syllable_count = 0
        prev_was_consonant = False

        for i, char in enumerate(text):
            if not self._is_devanagari(char):
                prev_was_consonant = False
                continue

            # Independent vowels count as syllables
            if char in independent_vowels:
                syllable_count += 1
                prev_was_consonant = False

            # Matras indicate syllables
            elif char in matras:
                syllable_count += 1
                prev_was_consonant = False

            # Consonants with inherent 'a' (unless followed by halant)
            elif unicodedata.category(char) == 'Lo':  # Letter, other
                # Check if next char is halant (suppresses inherent vowel)
                if i + 1 < len(text) and text[i + 1] == halant:
                    prev_was_consonant = True
                else:
                    # Consonant with inherent 'a' = 1 syllable
                    syllable_count += 1
                    prev_was_consonant = True

        return max(1, syllable_count)

    def analyze_text(self, text: str) -> TextAnalysis:
        """
        Perform detailed text analysis for multilingual metrics.

        Analyzes script composition, code-mixing, and language-specific
        characteristics.

        Args:
            text: Text to analyze.

        Returns:
            TextAnalysis: Detailed analysis results.

        Example:
            >>> counter = MultilingualTokenCounter()
            >>> analysis = counter.analyze_text("मुझे Python सीखना है।")
            >>> print(f"Code-mixing: {analysis.code_mixing_ratio:.2%}")
            Code-mixing: 23.08%
        """
        if not text:
            return TextAnalysis(
                total_chars=0, devanagari_chars=0, latin_chars=0,
                numeric_chars=0, punctuation_chars=0, whitespace_chars=0,
                other_chars=0, word_count=0, syllable_count=0,
                script_type=ScriptType.OTHER, code_mixing_ratio=0.0,
                unique_devanagari_chars=0
            )

        devanagari_chars = 0
        latin_chars = 0
        numeric_chars = 0
        punctuation_chars = 0
        whitespace_chars = 0
        other_chars = 0
        unique_devanagari = set()

        for char in text:
            if self._is_devanagari(char):
                devanagari_chars += 1
                unique_devanagari.add(char)
            elif self._is_latin(char):
                latin_chars += 1
            elif char.isdigit():
                numeric_chars += 1
            elif char.isspace():
                whitespace_chars += 1
            elif unicodedata.category(char).startswith('P'):
                punctuation_chars += 1
            else:
                other_chars += 1

        total_chars = len(text)

        # Word count (split on whitespace and punctuation)
        words = re.findall(r'[\w\u0900-\u097F]+', text)
        word_count = len(words)

        # Syllable count (Hindi-aware)
        syllable_count = self._count_hindi_syllables(text)

        # Determine primary script and code-mixing
        script_chars = devanagari_chars + latin_chars
        if script_chars == 0:
            script_type = ScriptType.OTHER
            code_mixing_ratio = 0.0
        elif devanagari_chars == 0:
            script_type = ScriptType.LATIN
            code_mixing_ratio = 0.0
        elif latin_chars == 0:
            script_type = ScriptType.DEVANAGARI
            code_mixing_ratio = 0.0
        else:
            # Mixed script - determine primary
            if devanagari_chars >= latin_chars:
                script_type = ScriptType.MIXED
                code_mixing_ratio = latin_chars / script_chars
            else:
                script_type = ScriptType.MIXED
                code_mixing_ratio = devanagari_chars / script_chars

        return TextAnalysis(
            total_chars=total_chars,
            devanagari_chars=devanagari_chars,
            latin_chars=latin_chars,
            numeric_chars=numeric_chars,
            punctuation_chars=punctuation_chars,
            whitespace_chars=whitespace_chars,
            other_chars=other_chars,
            word_count=word_count,
            syllable_count=syllable_count,
            script_type=script_type,
            code_mixing_ratio=code_mixing_ratio,
            unique_devanagari_chars=len(unique_devanagari),
        )

    def detect_language(self, text: str) -> str:
        """
        Detect the primary language of text.

        Args:
            text: Text to analyze.

        Returns:
            str: Language code ("en", "hi", or "hi-en" for code-mixed).

        Example:
            >>> counter = MultilingualTokenCounter()
            >>> counter.detect_language("Hello world")
            'en'
            >>> counter.detect_language("नमस्ते दुनिया")
            'hi'
            >>> counter.detect_language("मुझे coding पसंद है")
            'hi-en'
        """
        analysis = self.analyze_text(text)

        if analysis.script_type == ScriptType.DEVANAGARI:
            return "hi"
        elif analysis.script_type == ScriptType.LATIN:
            return "en"
        elif analysis.script_type == ScriptType.MIXED:
            # If more than 10% code-mixing, mark as hi-en
            if analysis.code_mixing_ratio > 0.1:
                return "hi-en"
            # Otherwise, classify by dominant script
            if analysis.devanagari_chars > analysis.latin_chars:
                return "hi"
            return "en"
        return "en"  # Default

    def _get_encoder(self, model: str) -> Optional[tiktoken.Encoding]:
        """Get tiktoken encoder for a model (for English/OpenAI models)."""
        model_lower = model.lower()

        if model in self._encoders:
            return self._encoders[model]

        try:
            if "gpt-4" in model_lower or "gpt-3.5" in model_lower:
                encoder = tiktoken.encoding_for_model(model)
                self._encoders[model] = encoder
                return encoder
        except KeyError:
            pass

        if any(x in model_lower for x in ["gpt", "openai", "davinci", "curie"]):
            if self._default_encoder is None:
                self._default_encoder = tiktoken.get_encoding("cl100k_base")
            self._encoders[model] = self._default_encoder
            return self._default_encoder

        return None

    def count_tokens(
        self,
        text: str,
        model: str = "gpt-4",
        language: Optional[str] = None
    ) -> int:
        """
        Count tokens in text with language awareness.

        Uses tiktoken for OpenAI models (English), and language-specific
        approximations for Hindi and other languages.

        Args:
            text: Text to count tokens for.
            model: Model name for tokenization.
            language: Language code (auto-detected if None).

        Returns:
            int: Estimated token count.

        Example:
            >>> counter = MultilingualTokenCounter()
            >>> counter.count_tokens("Hello world", language="en")
            2
            >>> counter.count_tokens("नमस्ते दुनिया", language="hi")
            3
        """
        if not text:
            return 0

        # Auto-detect language if not specified
        if language is None:
            language = self.detect_language(text)

        model_lower = model.lower()

        # For English text with OpenAI models, use tiktoken
        if language == "en":
            encoder = self._get_encoder(model)
            if encoder is not None:
                return len(encoder.encode(text))

        # Language-specific approximations
        if language == "hi":
            # Hindi: use syllable-based approximation for accuracy
            analysis = self.analyze_text(text)
            # Approximate: 1.8 syllables ≈ 1 token for Hindi
            token_estimate = analysis.syllable_count / self._SYLLABLES_PER_TOKEN_HI

            # Also consider word count as sanity check
            word_based_estimate = analysis.word_count * 1.3

            # Use average of both methods
            return max(1, int((token_estimate + word_based_estimate) / 2))

        elif language == "hi-en":
            # Code-mixed: blend Hindi and English ratios
            analysis = self.analyze_text(text)
            hi_ratio = analysis.devanagari_chars / max(1, analysis.total_chars)
            en_ratio = analysis.latin_chars / max(1, analysis.total_chars)

            hi_tokens = (analysis.devanagari_chars * self._TOKENS_PER_CHAR["hi"])
            en_tokens = (analysis.latin_chars * self._TOKENS_PER_CHAR["en"])

            return max(1, int(hi_tokens + en_tokens))

        # Default approximation
        ratio = self._TOKENS_PER_CHAR.get(language, self._TOKENS_PER_CHAR["default"])
        return max(1, int(len(text) * ratio))

    def count_tokens_detailed(
        self,
        text: str,
        model: str = "gpt-4",
        language: Optional[str] = None
    ) -> Dict:
        """
        Count tokens with detailed breakdown for multilingual text.

        Args:
            text: Text to analyze.
            model: Model name.
            language: Language code (auto-detected if None).

        Returns:
            Dict: Detailed token analysis including:
                - token_count: Estimated tokens
                - language: Detected/specified language
                - analysis: Full TextAnalysis object
                - method: Tokenization method used

        Example:
            >>> counter = MultilingualTokenCounter()
            >>> details = counter.count_tokens_detailed("मुझे Python पसंद है")
            >>> print(details['language'])
            'hi-en'
        """
        if language is None:
            language = self.detect_language(text)

        analysis = self.analyze_text(text)
        token_count = self.count_tokens(text, model, language)

        method = "tiktoken" if language == "en" and self._get_encoder(model) else "approximation"

        return {
            "token_count": token_count,
            "language": language,
            "word_count": analysis.word_count,
            "char_count": analysis.total_chars,
            "syllable_count": analysis.syllable_count,
            "script_type": analysis.script_type.value,
            "code_mixing_ratio": analysis.code_mixing_ratio,
            "devanagari_chars": analysis.devanagari_chars,
            "latin_chars": analysis.latin_chars,
            "method": method,
        }

    def get_script_consistency_score(self, text: str, expected_script: ScriptType) -> float:
        """
        Calculate script consistency score (0.0-1.0).

        Measures how consistently the text uses the expected script.
        Useful for evaluating if a model maintains language consistency.

        Args:
            text: Text to analyze.
            expected_script: Expected primary script.

        Returns:
            float: Consistency score (1.0 = perfectly consistent).

        Example:
            >>> counter = MultilingualTokenCounter()
            >>> score = counter.get_script_consistency_score(
            ...     "यह एक test है",
            ...     ScriptType.DEVANAGARI
            ... )
            >>> print(f"Consistency: {score:.2%}")
            Consistency: 75.00%
        """
        analysis = self.analyze_text(text)

        script_chars = analysis.devanagari_chars + analysis.latin_chars
        if script_chars == 0:
            return 0.0

        if expected_script == ScriptType.DEVANAGARI:
            return analysis.devanagari_chars / script_chars
        elif expected_script == ScriptType.LATIN:
            return analysis.latin_chars / script_chars
        elif expected_script == ScriptType.MIXED:
            # For mixed, penalize if too homogeneous
            min_ratio = min(analysis.devanagari_chars, analysis.latin_chars) / script_chars
            return min_ratio * 2  # Scale: 0.5 ratio = 1.0 score
        return 0.0

    @staticmethod
    def get_supported_languages() -> Dict[str, str]:
        """
        Get dictionary of supported languages.

        Returns:
            Dict[str, str]: Language codes and descriptions.
        """
        return {
            "en": "English (Latin script)",
            "hi": "Hindi (Devanagari script)",
            "hi-en": "Hinglish (Code-mixed Hindi-English)",
        }


# Convenience functions for backward compatibility
def count_hindi_tokens(text: str) -> int:
    """
    Count tokens in Hindi text.

    Args:
        text: Hindi text.

    Returns:
        int: Estimated token count.
    """
    counter = MultilingualTokenCounter()
    return counter.count_tokens(text, language="hi")


def analyze_code_mixing(text: str) -> float:
    """
    Get code-mixing ratio for text.

    Args:
        text: Text to analyze.

    Returns:
        float: Code-mixing ratio (0.0-1.0).
    """
    counter = MultilingualTokenCounter()
    analysis = counter.analyze_text(text)
    return analysis.code_mixing_ratio


def is_primarily_hindi(text: str) -> bool:
    """
    Check if text is primarily Hindi.

    Args:
        text: Text to check.

    Returns:
        bool: True if primarily Hindi/Devanagari.
    """
    counter = MultilingualTokenCounter()
    return counter.detect_language(text) in ("hi", "hi-en")
