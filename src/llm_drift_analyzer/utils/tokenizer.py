"""
Token counting utilities for LLM Drift Analyzer.

This module provides token counting functionality for different
LLM providers, enabling accurate measurement of response lengths.
"""

from typing import Optional, Dict
from enum import Enum

import tiktoken


class TokenizerType(Enum):
    """
    Enumeration of supported tokenizer types.

    Attributes:
        GPT4: Tokenizer for GPT-4 models (cl100k_base encoding).
        GPT35: Tokenizer for GPT-3.5 models (cl100k_base encoding).
        CLAUDE: Approximate tokenizer for Claude models.
        MISTRAL: Approximate tokenizer for Mistral models.
    """
    GPT4 = "gpt-4"
    GPT35 = "gpt-3.5-turbo"
    CLAUDE = "claude"
    MISTRAL = "mistral"


class TokenCounter:
    """
    Multi-provider token counter for LLM responses.

    Provides accurate token counting for OpenAI models using tiktoken,
    and approximations for other providers based on their tokenization
    characteristics.

    Attributes:
        _encoders: Cache of tiktoken encoders.
        _default_encoder: Default encoder for unknown models.

    Example:
        >>> counter = TokenCounter()
        >>> count = counter.count_tokens("Hello, world!", model="gpt-4")
        >>> print(count)
        4

        >>> count = counter.count_tokens("Bonjour le monde!", model="claude-3-opus")
        >>> print(count)
        5
    """

    # Approximate tokens per character ratios for different models
    _TOKENS_PER_CHAR: Dict[str, float] = {
        "claude": 0.25,  # Claude uses ~4 chars per token on average
        "mistral": 0.27,  # Mistral uses ~3.7 chars per token
        "default": 0.25,
    }

    def __init__(self):
        """Initialize token counter with encoder cache."""
        self._encoders: Dict[str, tiktoken.Encoding] = {}
        self._default_encoder: Optional[tiktoken.Encoding] = None

    def _get_encoder(self, model: str) -> Optional[tiktoken.Encoding]:
        """
        Get tiktoken encoder for a model.

        Caches encoders for efficiency and handles model name variations.

        Args:
            model: Model name or identifier.

        Returns:
            tiktoken.Encoding or None if model not supported by tiktoken.
        """
        # Normalize model name
        model_lower = model.lower()

        # Check cache
        if model in self._encoders:
            return self._encoders[model]

        # Try to get encoder for OpenAI models
        try:
            if "gpt-4" in model_lower or "gpt-3.5" in model_lower:
                encoder = tiktoken.encoding_for_model(model)
                self._encoders[model] = encoder
                return encoder
        except KeyError:
            pass

        # Use cl100k_base as default for OpenAI-like models
        if any(x in model_lower for x in ["gpt", "openai", "davinci", "curie"]):
            if self._default_encoder is None:
                self._default_encoder = tiktoken.get_encoding("cl100k_base")
            self._encoders[model] = self._default_encoder
            return self._default_encoder

        return None

    def count_tokens(self, text: str, model: str = "gpt-4") -> int:
        """
        Count tokens in text for a specific model.

        Uses tiktoken for OpenAI models and approximations for others.

        Args:
            text: Text to count tokens for.
            model: Model name to use for tokenization. Supports:
                - OpenAI models (gpt-4, gpt-3.5-turbo, etc.)
                - Claude models (claude-3-opus, claude-3-sonnet, etc.)
                - Mistral models (mixtral-8x7b, mistral-7b, etc.)

        Returns:
            int: Estimated token count.

        Example:
            >>> counter = TokenCounter()
            >>> counter.count_tokens("Hello, world!", "gpt-4")
            4
            >>> counter.count_tokens("Hello, world!", "claude-3-opus")
            3
        """
        if not text:
            return 0

        model_lower = model.lower()

        # Try tiktoken for OpenAI models
        encoder = self._get_encoder(model)
        if encoder is not None:
            return len(encoder.encode(text))

        # Use approximation for other models
        if "claude" in model_lower:
            ratio = self._TOKENS_PER_CHAR["claude"]
        elif "mistral" in model_lower or "mixtral" in model_lower:
            ratio = self._TOKENS_PER_CHAR["mistral"]
        else:
            ratio = self._TOKENS_PER_CHAR["default"]

        # Approximate token count
        return max(1, int(len(text) * ratio))

    def count_tokens_batch(
        self,
        texts: list,
        model: str = "gpt-4"
    ) -> list:
        """
        Count tokens for multiple texts.

        More efficient than calling count_tokens repeatedly for
        the same model as it reuses the encoder.

        Args:
            texts: List of texts to count tokens for.
            model: Model name to use for tokenization.

        Returns:
            list: List of token counts corresponding to input texts.

        Example:
            >>> counter = TokenCounter()
            >>> texts = ["Hello", "World", "How are you?"]
            >>> counts = counter.count_tokens_batch(texts, "gpt-4")
            >>> print(counts)
            [1, 1, 4]
        """
        return [self.count_tokens(text, model) for text in texts]

    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str = "gpt-4"
    ) -> float:
        """
        Estimate API cost based on token counts.

        Uses approximate pricing for popular models. Prices are
        per 1000 tokens and may not reflect current API pricing.

        Args:
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
            model: Model name for pricing lookup.

        Returns:
            float: Estimated cost in USD.

        Example:
            >>> counter = TokenCounter()
            >>> cost = counter.estimate_cost(1000, 500, "gpt-4")
            >>> print(f"${cost:.4f}")
            $0.0450
        """
        # Approximate pricing per 1000 tokens (as of 2024)
        pricing = {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
            "claude-3-opus": {"input": 0.015, "output": 0.075},
            "claude-3-sonnet": {"input": 0.003, "output": 0.015},
            "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
            "mixtral-8x7b": {"input": 0.0007, "output": 0.0007},
        }

        model_lower = model.lower()

        # Find matching pricing
        for model_key, prices in pricing.items():
            if model_key in model_lower:
                input_cost = (input_tokens / 1000) * prices["input"]
                output_cost = (output_tokens / 1000) * prices["output"]
                return input_cost + output_cost

        # Default to GPT-4 pricing if unknown
        default_prices = pricing["gpt-4"]
        input_cost = (input_tokens / 1000) * default_prices["input"]
        output_cost = (output_tokens / 1000) * default_prices["output"]
        return input_cost + output_cost

    @staticmethod
    def get_supported_models() -> Dict[str, str]:
        """
        Get dictionary of supported models and their tokenizer types.

        Returns:
            Dict[str, str]: Mapping of model families to tokenizer descriptions.

        Example:
            >>> models = TokenCounter.get_supported_models()
            >>> print(models)
            {'OpenAI GPT-4': 'tiktoken (cl100k_base)', ...}
        """
        return {
            "OpenAI GPT-4": "tiktoken (cl100k_base)",
            "OpenAI GPT-3.5": "tiktoken (cl100k_base)",
            "Anthropic Claude": "character-based approximation",
            "Mistral/Mixtral": "character-based approximation",
        }
