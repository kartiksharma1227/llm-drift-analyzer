"""
Tests for LLM clients.
"""

import pytest
from unittest.mock import patch, MagicMock

from llm_drift_analyzer.clients.base_client import (
    BaseLLMClient,
    LLMClientFactory,
    LLMClientError,
    QueryResult,
)
from llm_drift_analyzer.clients.openai_client import OpenAIClient
from llm_drift_analyzer.clients.anthropic_client import AnthropicClient
from llm_drift_analyzer.utils.config import APIConfig


class TestQueryResult:
    """Tests for QueryResult dataclass."""

    def test_create_query_result(self, mock_query_result):
        """Test creating a query result."""
        assert mock_query_result.response_text == "This is a mock response for testing purposes."
        assert mock_query_result.latency_ms == 250.5
        assert mock_query_result.model == "gpt-4-mock"

    def test_query_result_with_usage(self, mock_query_result):
        """Test query result with usage information."""
        assert mock_query_result.usage is not None
        assert mock_query_result.usage["total_tokens"] == 40


class TestLLMClientError:
    """Tests for LLMClientError exception."""

    def test_error_formatting(self):
        """Test error message formatting."""
        error = LLMClientError(
            message="Test error",
            provider="openai",
            model="gpt-4",
        )
        assert "[openai]" in str(error)
        assert "(gpt-4)" in str(error)
        assert "Test error" in str(error)

    def test_error_without_model(self):
        """Test error without model specified."""
        error = LLMClientError(
            message="Test error",
            provider="anthropic",
        )
        assert "[anthropic]" in str(error)
        assert "Test error" in str(error)


class TestLLMClientFactory:
    """Tests for LLMClientFactory."""

    def test_get_available_providers(self):
        """Test getting available providers."""
        providers = LLMClientFactory.get_available_providers()
        assert "openai" in providers
        assert "anthropic" in providers

    def test_create_unknown_provider(self):
        """Test creating client for unknown provider."""
        with pytest.raises(ValueError) as exc_info:
            LLMClientFactory.create("unknown_provider", "fake_key")
        assert "Unknown provider" in str(exc_info.value)

    @patch("llm_drift_analyzer.clients.openai_client.openai.OpenAI")
    def test_create_openai_client(self, mock_openai):
        """Test creating OpenAI client via factory."""
        client = LLMClientFactory.create("openai", "test-key")
        assert isinstance(client, OpenAIClient)
        assert client.provider_name == "openai"

    @patch("llm_drift_analyzer.clients.anthropic_client.anthropic.Anthropic")
    def test_create_anthropic_client(self, mock_anthropic):
        """Test creating Anthropic client via factory."""
        client = LLMClientFactory.create("anthropic", "test-key")
        assert isinstance(client, AnthropicClient)
        assert client.provider_name == "anthropic"


class TestOpenAIClient:
    """Tests for OpenAIClient."""

    @patch("llm_drift_analyzer.clients.openai_client.openai.OpenAI")
    def test_init(self, mock_openai_class):
        """Test client initialization."""
        client = OpenAIClient(api_key="test-key")
        mock_openai_class.assert_called_once_with(api_key="test-key")

    @patch("llm_drift_analyzer.clients.openai_client.openai.OpenAI")
    def test_list_models(self, mock_openai_class):
        """Test listing available models."""
        client = OpenAIClient(api_key="test-key")
        models = client.list_models()
        assert "gpt-4" in models
        assert "gpt-3.5-turbo" in models

    @patch("llm_drift_analyzer.clients.openai_client.openai.OpenAI")
    def test_query(self, mock_openai_class, mock_openai_client):
        """Test querying the model."""
        mock_openai_class.return_value = mock_openai_client

        client = OpenAIClient(api_key="test-key")
        result = client.query(prompt="Hello", model="gpt-4")

        assert result.response_text == "This is a test response."
        assert result.model == "gpt-4-test"
        mock_openai_client.chat.completions.create.assert_called_once()

    @patch("llm_drift_analyzer.clients.openai_client.openai.OpenAI")
    def test_query_with_system_message(self, mock_openai_class, mock_openai_client):
        """Test querying with system message."""
        mock_openai_class.return_value = mock_openai_client

        client = OpenAIClient(api_key="test-key")
        client.query(
            prompt="Hello",
            model="gpt-4",
            system_message="You are a helpful assistant."
        )

        call_args = mock_openai_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"

    @patch("llm_drift_analyzer.clients.openai_client.openai.OpenAI")
    def test_get_model_info(self, mock_openai_class):
        """Test getting model info."""
        client = OpenAIClient(api_key="test-key")
        info = client.get_model_info("gpt-4")
        assert "context_window" in info
        assert "input_price_per_1k" in info


class TestAnthropicClient:
    """Tests for AnthropicClient."""

    @patch("llm_drift_analyzer.clients.anthropic_client.anthropic.Anthropic")
    def test_init(self, mock_anthropic_class):
        """Test client initialization."""
        client = AnthropicClient(api_key="test-key")
        mock_anthropic_class.assert_called_once_with(api_key="test-key")

    @patch("llm_drift_analyzer.clients.anthropic_client.anthropic.Anthropic")
    def test_list_models(self, mock_anthropic_class):
        """Test listing available models."""
        client = AnthropicClient(api_key="test-key")
        models = client.list_models()
        assert any("claude" in m for m in models)

    @patch("llm_drift_analyzer.clients.anthropic_client.anthropic.Anthropic")
    def test_query(self, mock_anthropic_class, mock_anthropic_client):
        """Test querying the model."""
        mock_anthropic_class.return_value = mock_anthropic_client

        client = AnthropicClient(api_key="test-key")
        result = client.query(prompt="Hello", model="claude-3-opus-20240229")

        assert result.response_text == "This is a test response from Claude."
        mock_anthropic_client.messages.create.assert_called_once()

    @patch("llm_drift_analyzer.clients.anthropic_client.anthropic.Anthropic")
    def test_get_model_info(self, mock_anthropic_class):
        """Test getting model info."""
        client = AnthropicClient(api_key="test-key")
        info = client.get_model_info("claude-3-opus")
        assert "context_window" in info
        assert info["context_window"] == 200000
