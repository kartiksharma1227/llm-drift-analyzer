"""
Tests for evaluators.
"""

import pytest
from unittest.mock import patch, MagicMock

from llm_drift_analyzer.evaluators.base_evaluator import BaseEvaluator
from llm_drift_analyzer.evaluators.instruction_evaluator import InstructionEvaluator
from llm_drift_analyzer.evaluators.factuality_evaluator import FactualityEvaluator
from llm_drift_analyzer.evaluators.tone_evaluator import ToneEvaluator


class MockEvaluator(BaseEvaluator):
    """Concrete implementation of BaseEvaluator for testing."""

    @property
    def metric_name(self) -> str:
        return "test_metric"

    @property
    def score_range(self) -> tuple:
        return (0, 5)

    def _build_evaluation_prompt(self, prompt, response, **kwargs):
        return f"Evaluate: {prompt} -> {response}"


class TestBaseEvaluator:
    """Tests for BaseEvaluator."""

    @patch("llm_drift_analyzer.evaluators.base_evaluator.openai.OpenAI")
    def test_parse_score_single_digit(self, mock_openai):
        """Test parsing score from single digit."""
        evaluator = MockEvaluator(openai_api_key="test-key")
        assert evaluator._parse_score("3") == 3

    @patch("llm_drift_analyzer.evaluators.base_evaluator.openai.OpenAI")
    def test_parse_score_with_text(self, mock_openai):
        """Test parsing score from text with number."""
        evaluator = MockEvaluator(openai_api_key="test-key")
        assert evaluator._parse_score("The score is 4.") == 4

    @patch("llm_drift_analyzer.evaluators.base_evaluator.openai.OpenAI")
    def test_parse_score_invalid(self, mock_openai):
        """Test parsing score from invalid text."""
        evaluator = MockEvaluator(openai_api_key="test-key")
        with pytest.raises(ValueError):
            evaluator._parse_score("no numbers here")

    @patch("llm_drift_analyzer.evaluators.base_evaluator.openai.OpenAI")
    def test_get_default_score(self, mock_openai):
        """Test getting default score."""
        evaluator = MockEvaluator(openai_api_key="test-key")
        # Default should be middle of range (0, 5) -> 2
        assert evaluator._get_default_score() == 2

    @patch("llm_drift_analyzer.evaluators.base_evaluator.openai.OpenAI")
    def test_get_metric_description(self, mock_openai):
        """Test getting metric description."""
        evaluator = MockEvaluator(openai_api_key="test-key")
        desc = evaluator.get_metric_description()
        assert desc["name"] == "test_metric"
        assert desc["min_score"] == 0
        assert desc["max_score"] == 5


class TestInstructionEvaluator:
    """Tests for InstructionEvaluator."""

    @patch("llm_drift_analyzer.evaluators.base_evaluator.openai.OpenAI")
    def test_metric_properties(self, mock_openai):
        """Test metric name and score range."""
        evaluator = InstructionEvaluator(openai_api_key="test-key")
        assert evaluator.metric_name == "instruction_adherence"
        assert evaluator.score_range == (0, 3)

    @patch("llm_drift_analyzer.evaluators.base_evaluator.openai.OpenAI")
    def test_build_evaluation_prompt(self, mock_openai):
        """Test evaluation prompt building."""
        evaluator = InstructionEvaluator(openai_api_key="test-key")
        prompt_text = evaluator._build_evaluation_prompt(
            prompt="Write 3 bullet points",
            response="• Point 1\n• Point 2\n• Point 3"
        )
        assert "Write 3 bullet points" in prompt_text
        assert "Point 1" in prompt_text
        assert "0-3" in prompt_text

    @patch("llm_drift_analyzer.evaluators.base_evaluator.openai.OpenAI")
    def test_build_evaluation_prompt_with_expected_format(self, mock_openai):
        """Test evaluation prompt with expected format."""
        evaluator = InstructionEvaluator(openai_api_key="test-key")
        prompt_text = evaluator._build_evaluation_prompt(
            prompt="Write bullet points",
            response="• Point 1",
            expected_format="3 bullet points"
        )
        assert "Expected Format" in prompt_text
        assert "3 bullet points" in prompt_text

    @patch("llm_drift_analyzer.evaluators.base_evaluator.openai.OpenAI")
    def test_evaluate(self, mock_openai):
        """Test evaluate method."""
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "3"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        evaluator = InstructionEvaluator(openai_api_key="test-key")
        score = evaluator.evaluate(
            prompt="Write 3 bullet points",
            response="• Point 1\n• Point 2\n• Point 3"
        )
        assert score == 3


class TestFactualityEvaluator:
    """Tests for FactualityEvaluator."""

    @patch("llm_drift_analyzer.evaluators.base_evaluator.openai.OpenAI")
    def test_metric_properties(self, mock_openai):
        """Test metric name and score range."""
        evaluator = FactualityEvaluator(openai_api_key="test-key")
        assert evaluator.metric_name == "factuality"
        assert evaluator.score_range == (0, 2)

    @patch("llm_drift_analyzer.evaluators.base_evaluator.openai.OpenAI")
    def test_build_evaluation_prompt_with_reference(self, mock_openai):
        """Test evaluation prompt with reference answer."""
        evaluator = FactualityEvaluator(openai_api_key="test-key")
        prompt_text = evaluator._build_evaluation_prompt(
            prompt="What is the capital of France?",
            response="The capital of France is Paris.",
            reference_answer="Paris"
        )
        assert "Reference Answer" in prompt_text
        assert "Paris" in prompt_text

    @patch("llm_drift_analyzer.evaluators.base_evaluator.openai.OpenAI")
    def test_detect_hallucinations_parsing(self, mock_openai):
        """Test hallucination detection result parsing."""
        evaluator = FactualityEvaluator(openai_api_key="test-key")

        result_text = """SCORE: 1
CONCERNS: The date is incorrect
CONFIDENCE: HIGH"""

        result = evaluator._parse_hallucination_result(result_text)
        assert result["score"] == 1
        assert "The date is incorrect" in result["concerns"]
        assert result["confidence"] == "HIGH"


class TestToneEvaluator:
    """Tests for ToneEvaluator."""

    @patch("llm_drift_analyzer.evaluators.base_evaluator.openai.OpenAI")
    def test_metric_properties(self, mock_openai):
        """Test metric name and score range."""
        evaluator = ToneEvaluator(openai_api_key="test-key")
        assert evaluator.metric_name == "tone"
        assert evaluator.score_range == (0, 2)

    @patch("llm_drift_analyzer.evaluators.base_evaluator.openai.OpenAI")
    def test_build_evaluation_prompt_with_context(self, mock_openai):
        """Test evaluation prompt with audience context."""
        evaluator = ToneEvaluator(openai_api_key="test-key")
        prompt_text = evaluator._build_evaluation_prompt(
            prompt="Explain gravity",
            response="Gravity is a force...",
            target_audience="children",
            expected_tone="friendly"
        )
        assert "Target Audience" in prompt_text
        assert "children" in prompt_text
        assert "Expected Tone" in prompt_text
        assert "friendly" in prompt_text

    @patch("llm_drift_analyzer.evaluators.base_evaluator.openai.OpenAI")
    def test_parse_tone_analysis(self, mock_openai):
        """Test tone analysis result parsing."""
        evaluator = ToneEvaluator(openai_api_key="test-key")

        result_text = """SCORE: 2
FORMALITY: CASUAL
CHARACTERISTICS: friendly, approachable, clear
CONSISTENCY: HIGH
RECOMMENDATION: None needed"""

        result = evaluator._parse_tone_analysis(result_text)
        assert result["score"] == 2
        assert result["formality"] == "CASUAL"
        assert "friendly" in result["characteristics"]
        assert result["consistency"] == "HIGH"
