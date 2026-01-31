"""
Base client interface for LLM providers.

This module defines the abstract base class for LLM clients
and a factory for creating provider-specific implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, Type, List
from dataclasses import dataclass

from llm_drift_analyzer.utils.config import APIConfig


@dataclass
class QueryResult:
    """
    Result of an LLM query.

    Attributes:
        response_text: The generated response text.
        latency_ms: Time taken for the API call in milliseconds.
        model: The actual model used for generation.
        usage: Token usage information if available.
    """
    response_text: str
    latency_ms: float
    model: str
    usage: Optional[Dict[str, int]] = None


class BaseLLMClient(ABC):
    """
    Abstract base class for LLM API clients.

    Defines the common interface that all LLM provider clients
    must implement for drift analysis.

    Attributes:
        api_key: API key for authentication.
        api_config: Configuration for API calls.
        provider_name: Name of the LLM provider.

    Example:
        >>> class MyClient(BaseLLMClient):
        ...     def query(self, prompt, model, **kwargs):
        ...         # Implementation
        ...         pass
        ...     def list_models(self):
        ...         return ["model-1", "model-2"]
    """

    provider_name: str = "base"

    def __init__(
        self,
        api_key: str,
        api_config: Optional[APIConfig] = None
    ):
        """
        Initialize the LLM client.

        Args:
            api_key: API key for the provider.
            api_config: Optional API configuration. Uses defaults if not provided.
        """
        self.api_key = api_key
        self.api_config = api_config or APIConfig()

    @abstractmethod
    def query(
        self,
        prompt: str,
        model: str,
        **kwargs
    ) -> QueryResult:
        """
        Send a query to the LLM and get a response.

        Args:
            prompt: The prompt text to send.
            model: The model identifier to use.
            **kwargs: Additional model-specific parameters.

        Returns:
            QueryResult: The query result including response and metadata.

        Raises:
            LLMClientError: If the API call fails.
        """
        pass

    @abstractmethod
    def list_models(self) -> List[str]:
        """
        List available models for this provider.

        Returns:
            List[str]: List of model identifiers.
        """
        pass

    def validate_connection(self) -> bool:
        """
        Validate that the API connection is working.

        Attempts a minimal API call to verify credentials and connectivity.

        Returns:
            bool: True if connection is valid, False otherwise.
        """
        try:
            # Try listing models as a connection test
            models = self.list_models()
            return len(models) > 0
        except Exception:
            return False

    def get_default_model(self) -> str:
        """
        Get the default model for this provider.

        Returns:
            str: Default model identifier.
        """
        models = self.list_models()
        return models[0] if models else ""


class LLMClientError(Exception):
    """
    Exception raised for LLM client errors.

    Attributes:
        message: Error message.
        provider: Name of the LLM provider.
        model: Model that was being used.
        original_error: Original exception if available.
    """

    def __init__(
        self,
        message: str,
        provider: str = "unknown",
        model: Optional[str] = None,
        original_error: Optional[Exception] = None
    ):
        """
        Initialize LLMClientError.

        Args:
            message: Error message.
            provider: LLM provider name.
            model: Model identifier.
            original_error: Original exception.
        """
        self.message = message
        self.provider = provider
        self.model = model
        self.original_error = original_error
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format error message with context."""
        msg = f"[{self.provider}]"
        if self.model:
            msg += f" ({self.model})"
        msg += f": {self.message}"
        return msg


class LLMClientFactory:
    """
    Factory for creating LLM client instances.

    Manages registration and instantiation of provider-specific
    client implementations.

    Example:
        >>> factory = LLMClientFactory()
        >>> factory.register("openai", OpenAIClient)
        >>> client = factory.create("openai", api_key="sk-...")
    """

    _clients: Dict[str, Type[BaseLLMClient]] = {}

    @classmethod
    def register(cls, provider: str, client_class: Type[BaseLLMClient]) -> None:
        """
        Register a client class for a provider.

        Args:
            provider: Provider identifier (e.g., "openai").
            client_class: Client class implementing BaseLLMClient.

        Example:
            >>> LLMClientFactory.register("openai", OpenAIClient)
        """
        cls._clients[provider.lower()] = client_class

    @classmethod
    def create(
        cls,
        provider: str,
        api_key: str,
        api_config: Optional[APIConfig] = None
    ) -> BaseLLMClient:
        """
        Create a client instance for a provider.

        Args:
            provider: Provider identifier.
            api_key: API key for the provider.
            api_config: Optional API configuration.

        Returns:
            BaseLLMClient: Client instance.

        Raises:
            ValueError: If provider is not registered.

        Example:
            >>> client = LLMClientFactory.create("openai", "sk-...")
        """
        provider_lower = provider.lower()
        if provider_lower not in cls._clients:
            available = ", ".join(cls._clients.keys())
            raise ValueError(
                f"Unknown provider: {provider}. Available: {available}"
            )

        client_class = cls._clients[provider_lower]
        return client_class(api_key, api_config)

    @classmethod
    def get_available_providers(cls) -> List[str]:
        """
        Get list of registered providers.

        Returns:
            List[str]: Provider identifiers.
        """
        return list(cls._clients.keys())

    @classmethod
    def create_all(
        cls,
        api_keys: Dict[str, str],
        api_config: Optional[APIConfig] = None
    ) -> Dict[str, BaseLLMClient]:
        """
        Create clients for all providers with available API keys.

        Args:
            api_keys: Dictionary mapping provider names to API keys.
            api_config: Optional shared API configuration.

        Returns:
            Dict[str, BaseLLMClient]: Dictionary of provider name to client.

        Example:
            >>> clients = LLMClientFactory.create_all({
            ...     "openai": "sk-...",
            ...     "anthropic": "sk-ant-..."
            ... })
        """
        clients = {}
        for provider, api_key in api_keys.items():
            provider_lower = provider.lower()
            if provider_lower in cls._clients and api_key:
                try:
                    clients[provider_lower] = cls.create(
                        provider_lower, api_key, api_config
                    )
                except Exception:
                    # Skip providers that fail to initialize
                    pass
        return clients
