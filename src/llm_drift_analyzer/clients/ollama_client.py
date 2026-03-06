"""
Ollama API client for LLM Drift Analyzer.

This module provides the Ollama client implementation for
querying local/self-hosted open-source models like Llama, Mistral,
CodeLlama, Phi, and others.

Ollama allows running LLMs locally without API keys or internet access.
See https://ollama.ai for installation and model downloads.
"""

import time
import requests
from typing import List, Optional, Dict, Any

from llm_drift_analyzer.clients.base_client import (
    BaseLLMClient,
    QueryResult,
    LLMClientError,
    LLMClientFactory,
)
from llm_drift_analyzer.utils.config import APIConfig
from llm_drift_analyzer.utils.logger import get_logger


class OllamaClient(BaseLLMClient):
    """
    Ollama client for local/offline LLM inference.

    Supports any model available in Ollama including Llama 2, Llama 3,
    Mistral, CodeLlama, Phi, Gemma, and many more open-source models.

    Attributes:
        base_url: Ollama server URL (default: http://localhost:11434).
        provider_name: "ollama"

    Example:
        >>> client = OllamaClient(base_url="http://localhost:11434")
        >>> result = client.query(
        ...     prompt="Explain quantum computing",
        ...     model="llama3"
        ... )
        >>> print(result.response_text)

    Note:
        Requires Ollama to be installed and running locally.
        Install: https://ollama.ai/download
        Run: `ollama serve`
        Pull models: `ollama pull llama3`
    """

    provider_name = "ollama"

    # Popular Ollama models (dynamically fetched from server)
    DEFAULT_MODELS = [
        "llama3",
        "llama3:8b",
        "llama3:70b",
        "llama2",
        "llama2:13b",
        "llama2:70b",
        "mistral",
        "mistral:7b",
        "mixtral",
        "mixtral:8x7b",
        "codellama",
        "codellama:7b",
        "codellama:13b",
        "phi3",
        "phi3:medium",
        "gemma",
        "gemma:7b",
        "qwen2",
        "deepseek-coder",
        "neural-chat",
        "starling-lm",
    ]

    def __init__(
        self,
        api_key: str = "",  # Not required for Ollama, kept for interface compatibility
        api_config: Optional[APIConfig] = None,
        base_url: str = "http://localhost:11434",
    ):
        """
        Initialize Ollama client.

        Args:
            api_key: Not used for Ollama (kept for interface compatibility).
            api_config: Optional API configuration.
            base_url: Ollama server URL. Defaults to http://localhost:11434.
                      Can also be set via OLLAMA_BASE_URL environment variable.

        Raises:
            LLMClientError: If Ollama server is not reachable.
        """
        super().__init__(api_key or "ollama-local", api_config)
        self._logger = get_logger("clients.ollama")
        self.base_url = base_url.rstrip("/")

        # Verify Ollama server is running
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5, proxies={"http": None, "https": None})
            if response.status_code != 200:
                raise ConnectionError(f"Ollama server returned status {response.status_code}")
            self._logger.debug(f"Ollama client initialized at {self.base_url}")
        except requests.exceptions.ConnectionError as e:
            raise LLMClientError(
                message=f"Cannot connect to Ollama server at {self.base_url}. "
                        "Make sure Ollama is installed and running (`ollama serve`).",
                provider=self.provider_name,
                original_error=e
            )
        except Exception as e:
            raise LLMClientError(
                message=f"Failed to initialize Ollama client: {e}",
                provider=self.provider_name,
                original_error=e
            )

    def query(
        self,
        prompt: str,
        model: str = "llama3",
        system_message: Optional[str] = None,
        **kwargs
    ) -> QueryResult:
        """
        Query an Ollama model.

        Args:
            prompt: The user prompt to send.
            model: Model identifier (e.g., "llama3", "mistral", "codellama").
            system_message: Optional system message for context.
            **kwargs: Additional parameters passed to the API:
                - temperature: Sampling temperature (0.0 to 2.0)
                - top_p: Top-p (nucleus) sampling
                - top_k: Top-k sampling
                - num_predict: Maximum tokens to generate
                - seed: Random seed for reproducibility
                - stream: Whether to stream the response (default: False)

        Returns:
            QueryResult: Response with text, latency, and usage info.

        Raises:
            LLMClientError: If the API call fails.

        Example:
            >>> result = client.query(
            ...     prompt="Write a haiku about programming",
            ...     model="llama3",
            ...     system_message="You are a creative poet."
            ... )
        """
        self._logger.debug(f"Querying {model} with prompt length {len(prompt)}")

        # Build the request payload
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": kwargs.get("stream", False),
        }

        # Add system message if provided
        if system_message:
            payload["system"] = system_message

        # Add optional parameters from api_config or kwargs
        options: Dict[str, Any] = {}

        temperature = kwargs.get("temperature", self.api_config.temperature)
        if temperature is not None:
            options["temperature"] = temperature

        top_p = kwargs.get("top_p", self.api_config.top_p)
        if top_p is not None:
            options["top_p"] = top_p

        if "top_k" in kwargs:
            options["top_k"] = kwargs["top_k"]

        max_tokens = kwargs.get("max_tokens", kwargs.get("num_predict", self.api_config.max_tokens))
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        if "seed" in kwargs:
            options["seed"] = kwargs["seed"]

        if options:
            payload["options"] = options

        try:
            start_time = time.perf_counter()

            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.api_config.timeout if hasattr(self.api_config, 'timeout') else 120,
                proxies={"http": None, "https": None}
            )

            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000

            if response.status_code != 200:
                error_msg = response.text
                try:
                    error_data = response.json()
                    error_msg = error_data.get("error", error_msg)
                except Exception:
                    pass
                raise LLMClientError(
                    message=f"Ollama API error: {error_msg}",
                    provider=self.provider_name,
                    model=model
                )

            result = response.json()

            response_text = result.get("response", "")

            # Extract usage/context information
            usage = None
            if "prompt_eval_count" in result or "eval_count" in result:
                usage = {
                    "prompt_tokens": result.get("prompt_eval_count", 0),
                    "completion_tokens": result.get("eval_count", 0),
                    "total_tokens": (
                        result.get("prompt_eval_count", 0) +
                        result.get("eval_count", 0)
                    ),
                }

            self._logger.debug(
                f"Received response: {len(response_text)} chars, "
                f"{latency_ms:.2f}ms latency"
            )

            return QueryResult(
                response_text=response_text,
                latency_ms=latency_ms,
                model=result.get("model", model),
                usage=usage,
            )

        except requests.exceptions.Timeout as e:
            self._logger.warning(f"Request timeout: {e}")
            raise LLMClientError(
                message="Request timed out. Model may be loading or server is slow.",
                provider=self.provider_name,
                model=model,
                original_error=e
            )
        except requests.exceptions.ConnectionError as e:
            raise LLMClientError(
                message=f"Connection error: Cannot reach Ollama server at {self.base_url}",
                provider=self.provider_name,
                model=model,
                original_error=e
            )
        except LLMClientError:
            raise
        except Exception as e:
            raise LLMClientError(
                message=f"Unexpected error: {e}",
                provider=self.provider_name,
                model=model,
                original_error=e
            )

    def query_chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "llama3",
        **kwargs
    ) -> QueryResult:
        """
        Query an Ollama model using chat format (multi-turn conversations).

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
                      Roles: 'system', 'user', 'assistant'
            model: Model identifier.
            **kwargs: Additional parameters passed to the API.

        Returns:
            QueryResult: Response with text and metadata.

        Example:
            >>> result = client.query_chat(
            ...     messages=[
            ...         {"role": "system", "content": "You are helpful."},
            ...         {"role": "user", "content": "Hello!"},
            ...         {"role": "assistant", "content": "Hi there!"},
            ...         {"role": "user", "content": "How are you?"}
            ...     ],
            ...     model="llama3"
            ... )
        """
        self._logger.debug(f"Chat query to {model} with {len(messages)} messages")

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": kwargs.get("stream", False),
        }

        # Add options
        options: Dict[str, Any] = {}
        if kwargs.get("temperature") is not None:
            options["temperature"] = kwargs["temperature"]
        if kwargs.get("top_p") is not None:
            options["top_p"] = kwargs["top_p"]
        if kwargs.get("max_tokens") is not None:
            options["num_predict"] = kwargs["max_tokens"]

        if options:
            payload["options"] = options

        try:
            start_time = time.perf_counter()

            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=120,
                proxies={"http": None, "https": None}
            )

            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000

            if response.status_code != 200:
                raise LLMClientError(
                    message=f"Ollama chat API error: {response.text}",
                    provider=self.provider_name,
                    model=model
                )

            result = response.json()
            response_text = result.get("message", {}).get("content", "")

            usage = None
            if "prompt_eval_count" in result:
                usage = {
                    "prompt_tokens": result.get("prompt_eval_count", 0),
                    "completion_tokens": result.get("eval_count", 0),
                    "total_tokens": (
                        result.get("prompt_eval_count", 0) +
                        result.get("eval_count", 0)
                    ),
                }

            return QueryResult(
                response_text=response_text,
                latency_ms=latency_ms,
                model=result.get("model", model),
                usage=usage,
            )

        except LLMClientError:
            raise
        except Exception as e:
            raise LLMClientError(
                message=f"Chat query failed: {e}",
                provider=self.provider_name,
                model=model,
                original_error=e
            )

    def list_models(self) -> List[str]:
        """
        List models available on the Ollama server.

        Returns:
            List[str]: List of model identifiers installed locally.

        Note:
            Returns only models that have been pulled to the local machine.
            Use `ollama pull <model>` to download new models.
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10, proxies={"http": None, "https": None})
            if response.status_code == 200:
                data = response.json()
                models = [m["name"] for m in data.get("models", [])]
                return models if models else self.DEFAULT_MODELS
            return self.DEFAULT_MODELS
        except Exception as e:
            self._logger.warning(f"Failed to list models: {e}")
            return self.DEFAULT_MODELS

    def pull_model(self, model: str) -> bool:
        """
        Pull/download a model from the Ollama registry.

        Args:
            model: Model identifier to pull (e.g., "llama3", "mistral:7b").

        Returns:
            bool: True if pull was successful.

        Note:
            This is a blocking operation that may take several minutes
            depending on model size and internet speed.

        Example:
            >>> client.pull_model("llama3")
            True
        """
        self._logger.info(f"Pulling model {model}...")
        try:
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model, "stream": False},
                timeout=1800,  # 30 minutes for large models
                proxies={"http": None, "https": None}
            )
            return response.status_code == 200
        except Exception as e:
            self._logger.error(f"Failed to pull model {model}: {e}")
            return False

    def get_model_info(self, model: str) -> Dict[str, Any]:
        """
        Get information about a specific model.

        Args:
            model: Model identifier.

        Returns:
            Dict[str, Any]: Model information including size, family, parameters.
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/show",
                json={"name": model},
                timeout=10,
                proxies={"http": None, "https": None}
            )
            if response.status_code == 200:
                data = response.json()
                return {
                    "model": model,
                    "family": data.get("details", {}).get("family", "unknown"),
                    "parameter_size": data.get("details", {}).get("parameter_size", "unknown"),
                    "quantization": data.get("details", {}).get("quantization_level", "unknown"),
                    "format": data.get("details", {}).get("format", "unknown"),
                    "context_length": data.get("model_info", {}).get(
                        "llama.context_length",
                        data.get("details", {}).get("context_length", 4096)
                    ),
                }
            return {"model": model, "error": "Could not retrieve model info"}
        except Exception as e:
            return {"model": model, "error": str(e)}

    def validate_connection(self) -> bool:
        """
        Validate that the Ollama server is accessible.

        Returns:
            bool: True if connection is valid.
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5, proxies={"http": None, "https": None})
            return response.status_code == 200
        except Exception:
            return False

    def is_model_available(self, model: str) -> bool:
        """
        Check if a specific model is available locally.

        Args:
            model: Model identifier.

        Returns:
            bool: True if model is available.
        """
        models = self.list_models()
        return any(model in m or m in model for m in models)


# Register with factory
LLMClientFactory.register("ollama", OllamaClient)
