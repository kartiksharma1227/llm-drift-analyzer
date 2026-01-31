"""
Mistral AI API client for LLM Drift Analyzer.

This module provides the Mistral client implementation for
querying Mixtral and Mistral models.
"""

import time
from typing import List, Optional, Dict, Any

from llm_drift_analyzer.clients.base_client import (
    BaseLLMClient,
    QueryResult,
    LLMClientError,
    LLMClientFactory,
)
from llm_drift_analyzer.utils.config import APIConfig
from llm_drift_analyzer.utils.logger import get_logger


class MistralClient(BaseLLMClient):
    """
    Mistral AI API client for Mixtral and Mistral models.

    Supports Mixtral-8x7B and Mistral-7B models.
    Handles rate limiting and provides detailed error messages.

    Attributes:
        client: Mistral client instance.
        provider_name: "mistral"

    Example:
        >>> client = MistralClient(api_key="...")
        >>> result = client.query(
        ...     prompt="Hello, how are you?",
        ...     model="mistral-large-latest"
        ... )
        >>> print(result.response_text)
    """

    provider_name = "mistral"

    # Available Mistral models
    AVAILABLE_MODELS = [
        "mistral-large-latest",
        "mistral-medium-latest",
        "mistral-small-latest",
        "open-mixtral-8x7b",
        "open-mistral-7b",
        "mistral-embed",
    ]

    def __init__(
        self,
        api_key: str,
        api_config: Optional[APIConfig] = None
    ):
        """
        Initialize Mistral client.

        Args:
            api_key: Mistral API key.
            api_config: Optional API configuration.

        Raises:
            LLMClientError: If client initialization fails.
        """
        super().__init__(api_key, api_config)
        self._logger = get_logger("clients.mistral")
        self._client = None

        try:
            # Import mistralai here to make it optional
            from mistralai import Mistral
            self._client = Mistral(api_key=api_key)
            self._logger.debug("Mistral client initialized successfully")
        except ImportError:
            self._logger.warning(
                "mistralai package not installed. "
                "Install with: pip install mistralai"
            )
        except Exception as e:
            raise LLMClientError(
                message=f"Failed to initialize Mistral client: {e}",
                provider=self.provider_name,
                original_error=e
            )

    def query(
        self,
        prompt: str,
        model: str = "mistral-large-latest",
        system_message: Optional[str] = None,
        **kwargs
    ) -> QueryResult:
        """
        Query a Mistral model.

        Args:
            prompt: The user prompt to send.
            model: Model identifier (e.g., "mistral-large-latest").
            system_message: Optional system message for context.
            **kwargs: Additional parameters passed to the API.

        Returns:
            QueryResult: Response with text, latency, and usage info.

        Raises:
            LLMClientError: If the API call fails or client not initialized.

        Example:
            >>> result = client.query(
            ...     prompt="Explain quantum computing",
            ...     model="open-mixtral-8x7b",
            ...     system_message="You are a helpful assistant."
            ... )
        """
        if self._client is None:
            raise LLMClientError(
                message="Mistral client not initialized. Is mistralai installed?",
                provider=self.provider_name,
                model=model
            )

        self._logger.debug(f"Querying {model} with prompt length {len(prompt)}")

        # Build messages
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        try:
            start_time = time.perf_counter()

            response = self._client.chat.complete(
                model=model,
                messages=messages,
                temperature=kwargs.get("temperature", self.api_config.temperature),
                max_tokens=kwargs.get("max_tokens", self.api_config.max_tokens),
                top_p=kwargs.get("top_p", self.api_config.top_p),
            )

            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000

            # Extract response content
            response_text = ""
            if response.choices and len(response.choices) > 0:
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
                model=model,
                usage=usage,
            )

        except Exception as e:
            error_msg = str(e).lower()

            if "rate" in error_msg and "limit" in error_msg:
                self._logger.warning(f"Rate limit hit: {e}")
                raise LLMClientError(
                    message="Rate limit exceeded. Please wait and retry.",
                    provider=self.provider_name,
                    model=model,
                    original_error=e
                )
            elif "auth" in error_msg or "key" in error_msg:
                raise LLMClientError(
                    message="Invalid API key or authentication failed.",
                    provider=self.provider_name,
                    model=model,
                    original_error=e
                )
            else:
                raise LLMClientError(
                    message=f"API error: {e}",
                    provider=self.provider_name,
                    model=model,
                    original_error=e
                )

    def list_models(self) -> List[str]:
        """
        List available Mistral models.

        Returns:
            List[str]: List of model identifiers.

        Note:
            Returns a curated list of supported Mistral models.
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
            "mistral-large": {
                "context_window": 32000,
                "input_price_per_1k": 0.008,
                "output_price_per_1k": 0.024,
            },
            "mistral-medium": {
                "context_window": 32000,
                "input_price_per_1k": 0.0027,
                "output_price_per_1k": 0.0081,
            },
            "mistral-small": {
                "context_window": 32000,
                "input_price_per_1k": 0.002,
                "output_price_per_1k": 0.006,
            },
            "mixtral-8x7b": {
                "context_window": 32000,
                "input_price_per_1k": 0.0007,
                "output_price_per_1k": 0.0007,
            },
            "mistral-7b": {
                "context_window": 32000,
                "input_price_per_1k": 0.00025,
                "output_price_per_1k": 0.00025,
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
            "context_window": 32000,
            "input_price_per_1k": 0.002,
            "output_price_per_1k": 0.006,
        }


# Register with factory
LLMClientFactory.register("mistral", MistralClient)
