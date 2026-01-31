"""
OpenAI API client for LLM Drift Analyzer.

This module provides the OpenAI client implementation for
querying GPT models.
"""

import time
from typing import List, Optional, Dict, Any

import openai

from llm_drift_analyzer.clients.base_client import (
    BaseLLMClient,
    QueryResult,
    LLMClientError,
    LLMClientFactory,
)
from llm_drift_analyzer.utils.config import APIConfig
from llm_drift_analyzer.utils.logger import get_logger


class OpenAIClient(BaseLLMClient):
    """
    OpenAI API client for GPT models.

    Supports GPT-4 and GPT-3.5 models with various versions.
    Handles rate limiting and provides detailed error messages.

    Attributes:
        client: OpenAI client instance.
        provider_name: "openai"

    Example:
        >>> client = OpenAIClient(api_key="sk-...")
        >>> result = client.query(
        ...     prompt="Hello, how are you?",
        ...     model="gpt-4"
        ... )
        >>> print(result.response_text)
    """

    provider_name = "openai"

    # Available GPT-4 and GPT-3.5 models
    AVAILABLE_MODELS = [
        "gpt-4",
        "gpt-4-turbo",
        "gpt-4-turbo-preview",
        "gpt-4-0125-preview",
        "gpt-4-1106-preview",
        "gpt-4-0613",
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-0125",
        "gpt-3.5-turbo-1106",
    ]

    def __init__(
        self,
        api_key: str,
        api_config: Optional[APIConfig] = None
    ):
        """
        Initialize OpenAI client.

        Args:
            api_key: OpenAI API key.
            api_config: Optional API configuration.

        Raises:
            LLMClientError: If client initialization fails.
        """
        super().__init__(api_key, api_config)
        self._logger = get_logger("clients.openai")

        try:
            self.client = openai.OpenAI(api_key=api_key)
            self._logger.debug("OpenAI client initialized successfully")
        except Exception as e:
            raise LLMClientError(
                message=f"Failed to initialize OpenAI client: {e}",
                provider=self.provider_name,
                original_error=e
            )

    def query(
        self,
        prompt: str,
        model: str = "gpt-4",
        system_message: Optional[str] = None,
        **kwargs
    ) -> QueryResult:
        """
        Query an OpenAI model.

        Args:
            prompt: The user prompt to send.
            model: Model identifier (e.g., "gpt-4", "gpt-4-0613").
            system_message: Optional system message for context.
            **kwargs: Additional parameters passed to the API.

        Returns:
            QueryResult: Response with text, latency, and usage info.

        Raises:
            LLMClientError: If the API call fails.

        Example:
            >>> result = client.query(
            ...     prompt="Explain quantum computing",
            ...     model="gpt-4-turbo",
            ...     system_message="You are a helpful assistant."
            ... )
        """
        self._logger.debug(f"Querying {model} with prompt length {len(prompt)}")

        # Build messages
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        # Merge API config with kwargs
        params = {
            "model": model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.api_config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.api_config.max_tokens),
            "top_p": kwargs.get("top_p", self.api_config.top_p),
            "frequency_penalty": kwargs.get(
                "frequency_penalty", self.api_config.frequency_penalty
            ),
            "presence_penalty": kwargs.get(
                "presence_penalty", self.api_config.presence_penalty
            ),
        }

        try:
            start_time = time.perf_counter()

            response = self.client.chat.completions.create(**params)

            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000

            # Extract response content
            response_text = response.choices[0].message.content or ""

            # Extract usage information
            usage = None
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
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

        except openai.RateLimitError as e:
            self._logger.warning(f"Rate limit hit: {e}")
            raise LLMClientError(
                message="Rate limit exceeded. Please wait and retry.",
                provider=self.provider_name,
                model=model,
                original_error=e
            )
        except openai.AuthenticationError as e:
            raise LLMClientError(
                message="Invalid API key or authentication failed.",
                provider=self.provider_name,
                model=model,
                original_error=e
            )
        except openai.BadRequestError as e:
            raise LLMClientError(
                message=f"Bad request: {e}",
                provider=self.provider_name,
                model=model,
                original_error=e
            )
        except openai.APIError as e:
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
        List available OpenAI models.

        Returns:
            List[str]: List of model identifiers.

        Note:
            Returns a curated list of supported GPT models rather than
            all models available in the API.
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
            "gpt-4": {
                "context_window": 8192,
                "input_price_per_1k": 0.03,
                "output_price_per_1k": 0.06,
            },
            "gpt-4-turbo": {
                "context_window": 128000,
                "input_price_per_1k": 0.01,
                "output_price_per_1k": 0.03,
            },
            "gpt-4-0613": {
                "context_window": 8192,
                "input_price_per_1k": 0.03,
                "output_price_per_1k": 0.06,
            },
            "gpt-4-1106-preview": {
                "context_window": 128000,
                "input_price_per_1k": 0.01,
                "output_price_per_1k": 0.03,
            },
            "gpt-4-0125-preview": {
                "context_window": 128000,
                "input_price_per_1k": 0.01,
                "output_price_per_1k": 0.03,
            },
            "gpt-3.5-turbo": {
                "context_window": 16385,
                "input_price_per_1k": 0.0005,
                "output_price_per_1k": 0.0015,
            },
        }

        # Try to find matching model info
        for model_key, info in model_info.items():
            if model_key in model.lower():
                return {"model": model, **info}

        # Default info for unknown models
        return {
            "model": model,
            "context_window": 8192,
            "input_price_per_1k": 0.03,
            "output_price_per_1k": 0.06,
        }


# Register with factory
LLMClientFactory.register("openai", OpenAIClient)
