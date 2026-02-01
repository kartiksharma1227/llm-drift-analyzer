"""
Multilingual evaluator for LLM Drift Analyzer.

This module provides language-aware evaluation of LLM responses,
with special support for Hindi and English cross-lingual comparison.

Features:
- Language-specific evaluation prompts
- Script consistency evaluation
- Natural language quality assessment for Hindi
- Code-mixing penalty handling
"""

from typing import Optional, Dict, Any
import openai

from llm_drift_analyzer.evaluators.base_evaluator import BaseEvaluator
from llm_drift_analyzer.models.prompt import Language
from llm_drift_analyzer.utils.multilingual_tokenizer import (
    MultilingualTokenCounter,
    ScriptType,
)


class MultilingualInstructionEvaluator(BaseEvaluator):
    """
    Multilingual instruction adherence evaluator.

    Evaluates responses in both English and Hindi, with language-aware
    scoring criteria.

    Score Range (0-3):
        0: Does not follow instructions
        1: Partially follows instructions
        2: Mostly follows instructions
        3: Perfectly follows instructions

    Example:
        >>> evaluator = MultilingualInstructionEvaluator(api_key="sk-...")
        >>> score = evaluator.evaluate(
        ...     prompt="तीन बुलेट पॉइंट्स में बताओ",
        ...     response="• पहला पॉइंट\\n• दूसरा पॉइंट\\n• तीसरा पॉइंट",
        ...     language="hi"
        ... )
    """

    def __init__(
        self,
        openai_api_key: str,
        evaluator_model: str = "gpt-4",
        temperature: float = 0.0
    ):
        """Initialize evaluator with token counter."""
        super().__init__(openai_api_key, evaluator_model, temperature)
        self.token_counter = MultilingualTokenCounter()

    @property
    def metric_name(self) -> str:
        return "multilingual_instruction_adherence"

    @property
    def score_range(self) -> tuple:
        return (0, 3)

    def _build_evaluation_prompt(
        self,
        prompt: str,
        response: str,
        expected_format: Optional[str] = None,
        language: str = "en",
        **kwargs
    ) -> str:
        """Build language-aware evaluation prompt."""

        if language == "hi":
            return self._build_hindi_evaluation_prompt(
                prompt, response, expected_format
            )
        else:
            return self._build_english_evaluation_prompt(
                prompt, response, expected_format
            )

    def _build_english_evaluation_prompt(
        self,
        prompt: str,
        response: str,
        expected_format: Optional[str] = None
    ) -> str:
        """Build English evaluation prompt."""
        format_hint = f"\n\nExpected Format: {expected_format}" if expected_format else ""

        return f"""Rate how well this response follows the given instructions on a scale of 0-3.

Instructions: {prompt}{format_hint}

Response: {response}

Scoring Criteria:
3 = Perfect adherence to all instructions (format, length, content requirements)
2 = Good adherence with minor deviations
1 = Poor adherence, missing key requirements
0 = No adherence to instructions

Provide only the numeric score (0, 1, 2, or 3)."""

    def _build_hindi_evaluation_prompt(
        self,
        prompt: str,
        response: str,
        expected_format: Optional[str] = None
    ) -> str:
        """
        Build Hindi-specific evaluation prompt.

        Accounts for:
        - Hindi language quality
        - Script consistency (Devanagari usage)
        - Natural Hindi phrasing vs literal translation
        """
        format_hint = f"\n\nअपेक्षित Format: {expected_format}" if expected_format else ""

        # Check script consistency
        analysis = self.token_counter.analyze_text(response)
        script_note = ""
        if analysis.script_type == ScriptType.MIXED and analysis.code_mixing_ratio > 0.3:
            script_note = "\n⚠️ Note: Response has significant English mixed in. Consider if this affects instruction compliance."

        return f"""इस response को दिए गए instructions के अनुसार 0-3 के scale पर rate करें।

निर्देश (Instructions): {prompt}{format_hint}

Response: {response}{script_note}

Scoring Criteria (मूल्यांकन मापदंड):
3 = सभी निर्देशों का पूर्ण पालन (format, length, content सब सही)
2 = अधिकांश निर्देशों का पालन, छोटी-मोटी कमियां
1 = कमज़ोर पालन, मुख्य requirements missing
0 = निर्देशों का कोई पालन नहीं

Additional Hindi-specific criteria:
- Response should primarily be in Hindi (Devanagari script)
- Technical terms in English are acceptable
- But excessive English mixing should be noted

Provide only the numeric score (0, 1, 2, or 3)."""

    def evaluate(
        self,
        prompt: str,
        response: str,
        expected_format: Optional[str] = None,
        language: str = "en",
        **kwargs
    ) -> int:
        """
        Evaluate instruction adherence with language awareness.

        Args:
            prompt: Original prompt.
            response: LLM response.
            expected_format: Expected format description.
            language: Language code ("en" or "hi").

        Returns:
            int: Score 0-3.
        """
        eval_prompt = self._build_evaluation_prompt(
            prompt, response, expected_format, language=language
        )

        try:
            eval_response = self.openai_client.chat.completions.create(
                model=self.evaluator_model,
                messages=[{"role": "user", "content": eval_prompt}],
                max_tokens=10,
                temperature=self.temperature
            )

            score_text = eval_response.choices[0].message.content.strip()
            score = int(score_text[0]) if score_text else 1

            min_score, max_score = self.score_range
            return max(min_score, min(max_score, score))

        except Exception as e:
            self._logger.error(f"Evaluation failed: {e}")
            return 1


class MultilingualFactualityEvaluator(BaseEvaluator):
    """
    Multilingual factuality evaluator.

    Evaluates factual accuracy of responses in both English and Hindi.

    Score Range (0-2):
        0: Contains significant factual errors
        1: Mostly factual with minor errors
        2: Completely factual
    """

    def __init__(
        self,
        openai_api_key: str,
        evaluator_model: str = "gpt-4",
        temperature: float = 0.0
    ):
        super().__init__(openai_api_key, evaluator_model, temperature)
        self.token_counter = MultilingualTokenCounter()

    @property
    def metric_name(self) -> str:
        return "multilingual_factuality"

    @property
    def score_range(self) -> tuple:
        return (0, 2)

    def _build_evaluation_prompt(
        self,
        prompt: str,
        response: str,
        reference_answer: Optional[str] = None,
        language: str = "en",
        **kwargs
    ) -> str:
        """Build language-aware factuality evaluation prompt."""

        ref_section = f"\n\nReference Answer: {reference_answer}" if reference_answer else ""

        if language == "hi":
            return f"""इस response की factual accuracy को 0-2 के scale पर rate करें।

Question/Prompt: {prompt}

Response: {response}{ref_section}

Scoring Criteria:
2 = पूरी तरह से factually correct (सभी तथ्य सही)
1 = अधिकांश सही, कुछ छोटी गलतियां
0 = महत्वपूर्ण factual errors

Note for Hindi responses:
- Verify facts regardless of language
- Hindi-specific knowledge (Indian history, culture) should be especially accurate
- Dates, names, and figures should be correct

Provide only the numeric score (0, 1, or 2)."""

        else:
            return f"""Rate the factual accuracy of this response on a scale of 0-2.

Question/Prompt: {prompt}

Response: {response}{ref_section}

Scoring Criteria:
2 = Completely factual, all information is accurate
1 = Mostly factual with minor errors or imprecisions
0 = Contains significant factual errors

Provide only the numeric score (0, 1, or 2)."""

    def evaluate(
        self,
        prompt: str,
        response: str,
        reference_answer: Optional[str] = None,
        language: str = "en",
        **kwargs
    ) -> int:
        """Evaluate factuality with language awareness."""
        eval_prompt = self._build_evaluation_prompt(
            prompt, response, reference_answer, language=language
        )

        try:
            eval_response = self.openai_client.chat.completions.create(
                model=self.evaluator_model,
                messages=[{"role": "user", "content": eval_prompt}],
                max_tokens=10,
                temperature=self.temperature
            )

            score_text = eval_response.choices[0].message.content.strip()
            score = int(score_text[0]) if score_text else 1

            min_score, max_score = self.score_range
            return max(min_score, min(max_score, score))

        except Exception as e:
            self._logger.error(f"Evaluation failed: {e}")
            return 1


class MultilingualToneEvaluator(BaseEvaluator):
    """
    Multilingual tone/style evaluator.

    Evaluates appropriateness of tone and style for both English and Hindi,
    with special attention to Hindi register and formality levels.

    Score Range (0-2):
        0: Inappropriate tone
        1: Adequate tone with inconsistencies
        2: Appropriate and consistent tone
    """

    def __init__(
        self,
        openai_api_key: str,
        evaluator_model: str = "gpt-4",
        temperature: float = 0.0
    ):
        super().__init__(openai_api_key, evaluator_model, temperature)
        self.token_counter = MultilingualTokenCounter()

    @property
    def metric_name(self) -> str:
        return "multilingual_tone"

    @property
    def score_range(self) -> tuple:
        return (0, 2)

    def _build_evaluation_prompt(
        self,
        prompt: str,
        response: str,
        language: str = "en",
        **kwargs
    ) -> str:
        """Build language-aware tone evaluation prompt."""

        if language == "hi":
            return f"""इस response के tone और style को 0-2 के scale पर rate करें।

Prompt: {prompt}

Response: {response}

Scoring Criteria:
2 = उचित और consistent tone (context के अनुसार formal/informal)
1 = ठीक-ठाक tone, कुछ inconsistencies
0 = अनुचित tone (बहुत formal जब informal चाहिए था, या vice versa)

Hindi-specific considerations:
- Check if आप/तुम/तू usage is appropriate for context
- Verify honorifics (जी, साहब, etc.) are used correctly
- Assess if the response sounds natural Hindi, not translated
- Avoid overly Sanskritized or bookish Hindi unless appropriate
- Natural conversational tone is preferred for casual queries

Provide only the numeric score (0, 1, or 2)."""

        else:
            return f"""Rate the tone and style appropriateness of this response on a scale of 0-2.

Prompt: {prompt}

Response: {response}

Scoring Criteria:
2 = Appropriate and consistent tone throughout
1 = Adequate tone with some inconsistencies
0 = Inappropriate tone for the context

Consider:
- Professional vs casual context
- Technical vs general audience
- Consistency of voice throughout

Provide only the numeric score (0, 1, or 2)."""

    def evaluate(
        self,
        prompt: str,
        response: str,
        language: str = "en",
        **kwargs
    ) -> int:
        """Evaluate tone with language awareness."""
        eval_prompt = self._build_evaluation_prompt(
            prompt, response, language=language
        )

        try:
            eval_response = self.openai_client.chat.completions.create(
                model=self.evaluator_model,
                messages=[{"role": "user", "content": eval_prompt}],
                max_tokens=10,
                temperature=self.temperature
            )

            score_text = eval_response.choices[0].message.content.strip()
            score = int(score_text[0]) if score_text else 1

            min_score, max_score = self.score_range
            return max(min_score, min(max_score, score))

        except Exception as e:
            self._logger.error(f"Evaluation failed: {e}")
            return 1


class HindiNaturalnessEvaluator(BaseEvaluator):
    """
    Hindi-specific naturalness evaluator.

    Evaluates how natural and native-sounding the Hindi response is,
    penalizing literal translations and unnatural phrasing.

    Score Range (0-2):
        0: Sounds like a translation, unnatural Hindi
        1: Somewhat natural with some awkward phrasing
        2: Natural, native-sounding Hindi

    Example:
        >>> evaluator = HindiNaturalnessEvaluator(api_key="sk-...")
        >>> score = evaluator.evaluate(
        ...     prompt="AI के बारे में बताओ",
        ...     response="आर्टिफिशियल इंटेलिजेंस एक..."
        ... )
    """

    def __init__(
        self,
        openai_api_key: str,
        evaluator_model: str = "gpt-4",
        temperature: float = 0.0
    ):
        super().__init__(openai_api_key, evaluator_model, temperature)
        self.token_counter = MultilingualTokenCounter()

    @property
    def metric_name(self) -> str:
        return "hindi_naturalness"

    @property
    def score_range(self) -> tuple:
        return (0, 2)

    def _build_evaluation_prompt(
        self,
        prompt: str,
        response: str,
        **kwargs
    ) -> str:
        """Build Hindi naturalness evaluation prompt."""

        # Analyze script
        analysis = self.token_counter.analyze_text(response)
        script_info = f"\n[Script analysis: {analysis.devanagari_chars} Devanagari, {analysis.latin_chars} Latin chars]"

        return f"""Evaluate how natural and native-sounding this Hindi response is on a scale of 0-2.

Prompt: {prompt}

Response: {response}{script_info}

Scoring Criteria:
2 = बिल्कुल natural Hindi - जैसे कोई native Hindi speaker बोलता है
1 = ठीक-ठाक Hindi - समझ में आती है पर कुछ awkward phrasing
0 = Translation जैसी लगती है - English से literally translate की हुई

Signs of UNNATURAL Hindi (score lower):
- "मैं आशा करता हूं कि..." instead of "उम्मीद है कि..."
- "यह महत्वपूर्ण है कि..." instead of "ज़रूरी है कि..."
- Overly Sanskritized words when simple Hindi exists
- Word order that follows English grammar
- "कृपया" at wrong places

Signs of NATURAL Hindi (score higher):
- Conversational expressions like "वैसे तो", "असल में", "सच कहूं तो"
- Appropriate use of "यार", "भाई" in casual contexts
- Natural compound verbs like "कर देना", "बता दो"
- Idiomatic expressions

Provide only the numeric score (0, 1, or 2)."""

    def evaluate(
        self,
        prompt: str,
        response: str,
        **kwargs
    ) -> int:
        """Evaluate Hindi naturalness."""
        # First check if response is actually in Hindi
        detected_lang = self.token_counter.detect_language(response)
        if detected_lang == "en":
            return 0  # English response for Hindi prompt = unnatural

        eval_prompt = self._build_evaluation_prompt(prompt, response)

        try:
            eval_response = self.openai_client.chat.completions.create(
                model=self.evaluator_model,
                messages=[{"role": "user", "content": eval_prompt}],
                max_tokens=10,
                temperature=self.temperature
            )

            score_text = eval_response.choices[0].message.content.strip()
            score = int(score_text[0]) if score_text else 1

            min_score, max_score = self.score_range
            return max(min_score, min(max_score, score))

        except Exception as e:
            self._logger.error(f"Evaluation failed: {e}")
            return 1


class ScriptConsistencyEvaluator:
    """
    Evaluates script consistency in multilingual responses.

    Non-LLM based evaluator that checks if the response uses
    the expected script (Devanagari for Hindi, Latin for English).

    This doesn't require API calls and provides fast evaluation.
    """

    def __init__(self):
        self.token_counter = MultilingualTokenCounter()

    def evaluate(
        self,
        response: str,
        expected_language: str = "hi"
    ) -> Dict[str, Any]:
        """
        Evaluate script consistency.

        Args:
            response: Response text to evaluate.
            expected_language: Expected language code.

        Returns:
            Dict with consistency score and analysis.
        """
        analysis = self.token_counter.analyze_text(response)
        detected_lang = self.token_counter.detect_language(response)

        # Determine expected script
        expected_script = ScriptType.DEVANAGARI if expected_language == "hi" else ScriptType.LATIN

        # Calculate consistency
        consistency = self.token_counter.get_script_consistency_score(response, expected_script)

        # Check if language matches
        language_match = detected_lang == expected_language or (
            expected_language == "hi" and detected_lang == "hi-en"
        )

        return {
            "consistency_score": consistency,
            "expected_language": expected_language,
            "detected_language": detected_lang,
            "language_match": language_match,
            "script_type": analysis.script_type.value,
            "code_mixing_ratio": analysis.code_mixing_ratio,
            "devanagari_chars": analysis.devanagari_chars,
            "latin_chars": analysis.latin_chars,
        }
