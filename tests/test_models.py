"""
Tests for data models (Prompt, ResponseAnalysis).
"""

import pytest
import json
from datetime import datetime
from pathlib import Path

from llm_drift_analyzer.models.prompt import Prompt, PromptCategory, PromptSet
from llm_drift_analyzer.models.response_analysis import ResponseAnalysis, AnalysisResultSet


class TestPromptCategory:
    """Tests for PromptCategory enum."""

    def test_from_string_valid(self):
        """Test creating category from valid string."""
        category = PromptCategory.from_string("instruction_following")
        assert category == PromptCategory.INSTRUCTION_FOLLOWING

    def test_from_string_with_spaces(self):
        """Test creating category from string with spaces."""
        category = PromptCategory.from_string("factual qa")
        assert category == PromptCategory.FACTUAL_QA

    def test_from_string_invalid(self):
        """Test that invalid string raises ValueError."""
        with pytest.raises(ValueError):
            PromptCategory.from_string("invalid_category")


class TestPrompt:
    """Tests for Prompt dataclass."""

    def test_create_prompt(self, sample_prompt):
        """Test creating a prompt."""
        assert sample_prompt.id == "TEST-001"
        assert sample_prompt.category == PromptCategory.INSTRUCTION_FOLLOWING

    def test_prompt_to_dict(self, sample_prompt):
        """Test converting prompt to dictionary."""
        data = sample_prompt.to_dict()
        assert data["id"] == "TEST-001"
        assert data["category"] == "instruction_following"

    def test_prompt_from_dict(self):
        """Test creating prompt from dictionary."""
        data = {
            "id": "NEW-001",
            "text": "Test text",
            "category": "factual_qa",
        }
        prompt = Prompt.from_dict(data)
        assert prompt.id == "NEW-001"
        assert prompt.category == PromptCategory.FACTUAL_QA

    def test_prompt_roundtrip(self, sample_prompt):
        """Test that to_dict/from_dict roundtrip preserves data."""
        data = sample_prompt.to_dict()
        restored = Prompt.from_dict(data)
        assert restored.id == sample_prompt.id
        assert restored.text == sample_prompt.text
        assert restored.category == sample_prompt.category


class TestPromptSet:
    """Tests for PromptSet class."""

    def test_create_prompt_set(self, sample_prompts):
        """Test creating a prompt set."""
        assert len(sample_prompts) == 3
        assert sample_prompts.name == "Test Prompts"

    def test_prompt_set_iteration(self, sample_prompts):
        """Test iterating over prompt set."""
        ids = [p.id for p in sample_prompts]
        assert "IF-001" in ids
        assert "QA-001" in ids

    def test_prompt_set_get_by_id(self, sample_prompts):
        """Test getting prompt by ID."""
        prompt = sample_prompts.get_by_id("QA-001")
        assert prompt is not None
        assert prompt.category == PromptCategory.FACTUAL_QA

    def test_prompt_set_filter_by_category(self, sample_prompts):
        """Test filtering by category."""
        instruction_prompts = sample_prompts.filter_by_category(
            PromptCategory.INSTRUCTION_FOLLOWING
        )
        assert len(instruction_prompts) == 1
        assert instruction_prompts[0].id == "IF-001"

    def test_prompt_set_save_load(self, sample_prompts, tmp_path):
        """Test saving and loading prompt set."""
        file_path = tmp_path / "prompts.json"
        sample_prompts.save_to_file(file_path)

        loaded = PromptSet.load_from_file(file_path)
        assert len(loaded) == len(sample_prompts)
        assert loaded.name == sample_prompts.name


class TestResponseAnalysis:
    """Tests for ResponseAnalysis dataclass."""

    def test_create_response_analysis(self, sample_response_analysis):
        """Test creating a response analysis."""
        assert sample_response_analysis.prompt_id == "TEST-001"
        assert sample_response_analysis.instruction_score == 3

    def test_total_quality_score(self, sample_response_analysis):
        """Test total quality score calculation."""
        # instruction(3) + factuality(2) + tone(2) = 7
        assert sample_response_analysis.total_quality_score == 7

    def test_normalized_quality_score(self, sample_response_analysis):
        """Test normalized quality score calculation."""
        # 7 / 7 (max) = 1.0
        assert sample_response_analysis.normalized_quality_score == 1.0

    def test_tokens_per_second(self, sample_response_analysis):
        """Test tokens per second calculation."""
        # 25 tokens / 500ms * 1000 = 50 tokens/s
        assert sample_response_analysis.tokens_per_second == 50.0

    def test_score_validation_instruction(self):
        """Test that invalid instruction score raises error."""
        with pytest.raises(ValueError):
            ResponseAnalysis(
                prompt_id="TEST",
                model_version="test",
                response_text="test",
                token_count=10,
                latency_ms=100,
                instruction_score=5,  # Invalid: max is 3
                factuality_score=1,
                tone_score=1,
            )

    def test_score_validation_factuality(self):
        """Test that invalid factuality score raises error."""
        with pytest.raises(ValueError):
            ResponseAnalysis(
                prompt_id="TEST",
                model_version="test",
                response_text="test",
                token_count=10,
                latency_ms=100,
                instruction_score=2,
                factuality_score=5,  # Invalid: max is 2
                tone_score=1,
            )

    def test_response_analysis_to_dict(self, sample_response_analysis):
        """Test converting to dictionary."""
        data = sample_response_analysis.to_dict()
        assert data["prompt_id"] == "TEST-001"
        assert "total_quality_score" in data
        assert "normalized_quality_score" in data

    def test_response_analysis_roundtrip(self, sample_response_analysis):
        """Test to_dict/from_dict roundtrip."""
        data = sample_response_analysis.to_dict()
        restored = ResponseAnalysis.from_dict(data)
        assert restored.prompt_id == sample_response_analysis.prompt_id
        assert restored.instruction_score == sample_response_analysis.instruction_score


class TestAnalysisResultSet:
    """Tests for AnalysisResultSet class."""

    def test_create_result_set(self, sample_results):
        """Test creating a result set."""
        assert len(sample_results) == 10

    def test_filter_by_model(self, sample_results):
        """Test filtering by model."""
        v1_results = sample_results.filter_by_model("v1")
        assert len(v1_results) == 5

    def test_filter_by_prompt(self, sample_results):
        """Test filtering by prompt."""
        prompt_results = sample_results.filter_by_prompt("TEST-001")
        assert all(r.prompt_id == "TEST-001" for r in prompt_results)

    def test_get_models(self, sample_results):
        """Test getting unique models."""
        models = sample_results.get_models()
        assert "gpt-4-v1" in models
        assert "gpt-4-v2" in models

    def test_get_summary_stats(self, sample_results):
        """Test getting summary statistics."""
        stats = sample_results.get_summary_stats()
        assert stats["total_results"] == 10
        assert "token_count" in stats
        assert "mean" in stats["token_count"]

    def test_result_set_save_load(self, sample_results, tmp_path):
        """Test saving and loading result set."""
        file_path = tmp_path / "results.json"
        sample_results.save_to_file(file_path)

        loaded = AnalysisResultSet.load_from_file(file_path)
        assert len(loaded) == len(sample_results)
