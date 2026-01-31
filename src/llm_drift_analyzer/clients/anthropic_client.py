"""
Anthropic API client for LLM Drift Analyzer.

This module provides the Anthropic client implementation for
querying Claude models.
"""

import time
from typing import List, Optional, Dict, Any

import anthropic

from llm_drift_analyzer.clients.base_client import (
    BaseLLMClient,
    QueryResult,
    LLMClientError,
    LLMClientFactory,
)
from llm_drift_analyzer.utils.config import APIConfig
from llm_drift_analyzer.utils.logger import get_logger


class AnthropicClient(BaseLLMClient):
    """
    Anthropic API client for Claude models.

    Supports Claude 3 family models (Opus, Sonnet, Haiku).
    Handles rate limiting and provides detailed error messages.

    Attributes:
        client: Anthropic client instance.
        provider_name: "anthropic"

    Example:
        >>> client = AnthropicClient(api_key="sk-ant-...")
        >>> result = client.query(
        ...     prompt="Hello, how are you?",
        ...     model="claude-3-opus-20240229"
        ... )
        >>> print(result.response_text)
    """

    provider_name = "anthropic"

    # Available Claude models
    AVAILABLE_MODELS = [
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
        "claude-3-5-sonnet-20240620",
        "claude-2.1",
        "claude-2.0",
        "claude-instant-1.2",
    ]

    def __init__(
        self,
        api_key: str,
        api_config: Optional[APIConfig] = None
    ):
        """
        Initialize Anthropic client.

        Args:
            api_key: Anthropic API key.
            api_config: Optional API configuration.

        Raises:
            LLMClientError: If client initialization fails.
        """
        super().__init__(api_key, api_config)
        self._logger = get_logger("clients.anthropic")

        try:
            self.client = anthropic.Anthropic(api_key=api_key)
            self._logger.debug("Anthropic client initialized successfully")
        except Exception as e:
            raise LLMClientError(
                message=f"Failed to initialize Anthropic client: {e}",
                provider=self.provider_name,
                original_error=e
            )

    def query(
        self,
        prompt: str,
        model: str = "claude-3-opus-20240229",
        system_message: Optional[str] = None,
        **kwargs
    ) -> QueryResult:
        """
        Query a Claude model.

        Args:
            prompt: The user prompt to send.
            model: Model identifier (e.g., "claude-3-opus-20240229").
            system_message: Optional system message for context.
            **kwargs: Additional parameters passed to the API.

        Returns:
            QueryResult: Response with text, latency, and usage info.

        Raises:
            LLMClientError: If the API call fails.

        Example:
            >>> result = client.query(
            ...     prompt="Explain quantum computing",
            ...     model="claude-3-sonnet-20240229",
            ...     system_message="You are a helpful assistant."
            ... )
        """
        self._logger.debug(f"Querying {model} with prompt length {len(prompt)}")

        # Build messages
        messages = [{"role": "user", "content": prompt}]

        # Build parameters
        params = {
            "model": model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.api_config.max_tokens),
            "temperature": kwargs.get("temperature", self.api_config.temperature),
            "top_p": kwargs.get("top_p", self.api_config.top_p),
        }

        # Add system message if provided
        if system_message:
            params["system"] = system_message

        try:
            start_time = time.perf_counter()

            response = self.client.messages.create(**params)

            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000

            # Extract response content
            response_text = ""
            for content_block in response.content:
                if content_block.type == "text":
                    response_text += content_block.text

            # Extract usage information
            usage = None
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": (
                        response.usage.input_tokens + response.usage.output_tokens
                    ),
                }

            self._logger.debug(
                f"Received response: {len(response_text)} chars, "
                f"{latency_ms:.2f}ms latency"
            )

            return QueryResult(
                response_text=response_text,
                latency_ms=latency_ms,
                model=response.model,
                usage=usage,
            )

        except anthropic.RateLimitError as e:
            self._logger.warning(f"Rate limit hit: {e}")
            raise LLMClientError(
                message="Rate limit exceeded. Please wait and retry.",
                provider=self.provider_name,
                model=model,
                original_error=e
            )
        except anthropic.AuthenticationError as e:
            raise LLMClientError(
                message="Invalid API key or authentication failed.",
                provider=self.provider_name,
                model=model,
                original_error=e
            )
        except anthropic.BadRequestError as e:
            raise LLMClientError(
                message=f"Bad request: {e}",
                provider=self.provider_name,
                model=model,
                original_error=e
            )
        except anthropic.APIError as e:
            raise LLMClientError(
                message=f"API error: {e}",
                provider=self.provider_name,
                model=model,
                original_error=e
            )
        except Exception as e:
            raise LLMClientError(
                message=f"Unexpected error: {e}",
                provider=self.provider_name,
                model=model,
                original_error=e
            )

    def list_models(self) -> List[str]:
        """
        List available Claude models.

        Returns:
            List[str]: List of model identifiers.

        Note:
            Returns a curated list of supported Claude models.
        """
        return self.AVAILABLE_MODELS.copy()

    def get_model_info(self, model: str) -> Dict[str, Any]:
        """
        Get information about a specific model.

        Args:
            model: Model identifier.

        Returns:
            Dict[str, Any]: Model information including context window and pricing.
        """
        model_info = {
            "claude-3-opus": {
                "context_window": 200000,
                "input_price_per_1k": 0.015,
                "output_price_per_1k": 0.075,
            },
            "claude-3-sonnet": {
                "context_window": 200000,
                "input_price_per_1k": 0.003,
                "output_price_per_1k": 0.015,
            },
            "claude-3-haiku": {
                "context_window": 200000,
                "input_price_per_1k": 0.00025,
                "output_price_per_1k": 0.00125,
            },
            "claude-3-5-sonnet": {
                "context_window": 200000,
                "input_price_per_1k": 0.003,
                "output_price_per_1k": 0.015,
            },
            "claude-2": {
                "context_window": 100000,
                "input_price_per_1k": 0.008,
                "output_price_per_1k": 0.024,
            },
        }

        # Try to find matching model info
        model_lower = model.lower()
        for model_key, info in model_info.items():
            if model_key in model_lower:
                return {"model": model, **info}

        # Default info for unknown models
        return {
            "model": model,
            "context_window": 100000,
            "input_price_per_1k": 0.01,
            "output_price_per_1k": 0.03,
        }


# Register with factory
LLMClientFactory.register("anthropic", AnthropicClient)
