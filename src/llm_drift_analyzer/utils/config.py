"""
Configuration management for LLM Drift Analyzer.

This module provides configuration loading from environment variables
and default API parameters for LLM queries.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Optional
from pathlib import Path

from dotenv import load_dotenv


@dataclass
class APIConfig:
    """
    Configuration for LLM API calls.

    Attributes:
        temperature: Sampling temperature for response generation.
            Lower values produce more deterministic outputs.
        max_tokens: Maximum number of tokens in the response.
        top_p: Nucleus sampling parameter.
        frequency_penalty: Penalty for token frequency.
        presence_penalty: Penalty for token presence.
    """
    temperature: float = 0.1
    max_tokens: int = 1000
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0


@dataclass
class Config:
    """
    Main configuration class for LLM Drift Analyzer.

    Manages API keys, model settings, and output configuration.
    Can be initialized from environment variables or directly with values.

    Attributes:
        openai_api_key: API key for OpenAI services.
        anthropic_api_key: API key for Anthropic services.
        mistral_api_key: API key for Mistral AI services.
        ollama_base_url: Base URL for Ollama server (default: http://localhost:11434).
        evaluator_model: Model to use for automated evaluation (default: gpt-4).
        evaluator_provider: Provider for evaluation ("openai" or "ollama").
            Use "ollama" to evaluate with local models and avoid API costs.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
        output_dir: Directory for saving reports and results.
        api_config: API parameters for LLM queries.

    Example:
        >>> config = Config.from_env()
        >>> print(config.openai_api_key)
        'sk-...'

        >>> config = Config(
        ...     openai_api_key="sk-...",
        ...     anthropic_api_key="sk-ant-..."
        ... )

        >>> # For local/offline models via Ollama
        >>> config = Config(ollama_base_url="http://localhost:11434")

        >>> # Use Ollama for evaluation (free, no API costs)
        >>> config = Config(
        ...     evaluator_provider="ollama",
        ...     evaluator_model="llama3"
        ... )
    """
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    mistral_api_key: Optional[str] = None
    ollama_base_url: str = "http://localhost:11434"
    evaluator_model: str = "gpt-4"
    evaluator_provider: str = "openai"  # "openai" or "ollama"
    log_level: str = "INFO"
    output_dir: Path = field(default_factory=lambda: Path("output"))
    api_config: APIConfig = field(default_factory=APIConfig)

    def __post_init__(self) -> None:
        """Convert output_dir to Path if string is provided."""
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

    @classmethod
    def from_env(cls, env_file: Optional[str] = None) -> "Config":
        """
        Create configuration from environment variables.

        Loads environment variables from a .env file if provided,
        otherwise uses existing environment variables.

        Args:
            env_file: Path to .env file. If None, searches for .env
                in the current directory and parent directories.

        Returns:
            Config: Configuration instance with values from environment.

        Raises:
            ValueError: If required API keys are not found and strict mode is enabled.

        Example:
            >>> config = Config.from_env(".env")
            >>> config = Config.from_env()  # Uses default .env location
        """
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv()

        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            mistral_api_key=os.getenv("MISTRAL_API_KEY"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            evaluator_model=os.getenv("EVALUATOR_MODEL", "gpt-4"),
            evaluator_provider=os.getenv("EVALUATOR_PROVIDER", "openai"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            output_dir=Path(os.getenv("OUTPUT_DIR", "output")),
        )

    @classmethod
    def from_dict(cls, data: Dict) -> "Config":
        """
        Create configuration from a dictionary.

        Args:
            data: Dictionary containing configuration values.

        Returns:
            Config: Configuration instance with values from dictionary.

        Example:
            >>> config = Config.from_dict({
            ...     "openai_api_key": "sk-...",
            ...     "log_level": "DEBUG"
            ... })
        """
        api_config_data = data.pop("api_config", {})
        api_config = APIConfig(**api_config_data) if api_config_data else APIConfig()

        return cls(
            api_config=api_config,
            **data
        )

    def to_dict(self) -> Dict:
        """
        Convert configuration to dictionary.

        Note: API keys are masked for security.

        Returns:
            Dict: Dictionary representation of configuration.
        """
        return {
            "openai_api_key": self._mask_key(self.openai_api_key),
            "anthropic_api_key": self._mask_key(self.anthropic_api_key),
            "mistral_api_key": self._mask_key(self.mistral_api_key),
            "ollama_base_url": self.ollama_base_url,
            "evaluator_model": self.evaluator_model,
            "evaluator_provider": self.evaluator_provider,
            "log_level": self.log_level,
            "output_dir": str(self.output_dir),
            "api_config": {
                "temperature": self.api_config.temperature,
                "max_tokens": self.api_config.max_tokens,
                "top_p": self.api_config.top_p,
                "frequency_penalty": self.api_config.frequency_penalty,
                "presence_penalty": self.api_config.presence_penalty,
            }
        }

    @staticmethod
    def _mask_key(key: Optional[str]) -> str:
        """Mask API key for safe display."""
        if not key:
            return "NOT_SET"
        if len(key) <= 8:
            return "***"
        return f"{key[:4]}...{key[-4:]}"

    def validate(self) -> Dict[str, bool]:
        """
        Validate that required API keys are present.

        Returns:
            Dict[str, bool]: Dictionary mapping provider names to availability status.

        Example:
            >>> config = Config.from_env()
            >>> status = config.validate()
            >>> print(status)
            {'openai': True, 'anthropic': True, 'mistral': False, 'ollama': True}
        """
        return {
            "openai": bool(self.openai_api_key),
            "anthropic": bool(self.anthropic_api_key),
            "mistral": bool(self.mistral_api_key),
            "ollama": True,  # Ollama doesn't require API key, always available if server is running
        }

    def get_api_keys(self, include_ollama: bool = False) -> Dict[str, str]:
        """
        Get dictionary of API keys for client initialization.

        Args:
            include_ollama: If True, include Ollama in the returned dict.
                           Ollama doesn't require an API key.

        Returns:
            Dict[str, str]: Dictionary mapping provider names to API keys.

        Raises:
            ValueError: If no API keys are configured and include_ollama is False.
        """
        keys = {}
        if self.openai_api_key:
            keys["openai"] = self.openai_api_key
        if self.anthropic_api_key:
            keys["anthropic"] = self.anthropic_api_key
        if self.mistral_api_key:
            keys["mistral"] = self.mistral_api_key
        if include_ollama:
            keys["ollama"] = ""  # Ollama doesn't need API key

        if not keys:
            raise ValueError(
                "No API keys configured. Please set at least one of: "
                "OPENAI_API_KEY, ANTHROPIC_API_KEY, MISTRAL_API_KEY, "
                "or use --provider ollama for local models"
            )

        return keys
