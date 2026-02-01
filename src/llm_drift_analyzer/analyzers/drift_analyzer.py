"""
Main drift analyzer for LLM Behavioral Drift Analysis.

This module provides the core LLMDriftAnalyzer class that orchestrates
the complete drift analysis workflow across multiple LLM providers.
"""

import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Callable
from pathlib import Path

from llm_drift_analyzer.models.prompt import Prompt, PromptSet
from llm_drift_analyzer.models.response_analysis import (
    ResponseAnalysis,
    AnalysisResultSet,
)
from llm_drift_analyzer.clients.base_client import (
    BaseLLMClient,
    LLMClientFactory,
    LLMClientError,
)
from llm_drift_analyzer.evaluators.instruction_evaluator import InstructionEvaluator
from llm_drift_analyzer.evaluators.factuality_evaluator import FactualityEvaluator
from llm_drift_analyzer.evaluators.tone_evaluator import ToneEvaluator
from llm_drift_analyzer.utils.config import Config
from llm_drift_analyzer.utils.tokenizer import TokenCounter
from llm_drift_analyzer.utils.logger import get_logger, log_execution_time


class LLMDriftAnalyzer:
    """
    Main analyzer for tracking behavioral drift in LLMs.

    Orchestrates the complete drift analysis workflow including:
    - Querying multiple LLM providers and models
    - Evaluating responses for instruction adherence, factuality, and tone
    - Collecting metrics like token count and latency
    - Managing iterations for statistical significance

    Supports both cloud-based evaluation (GPT-4) and local evaluation (Ollama)
    for response scoring. Use Ollama for free, offline evaluation.

    Attributes:
        config: Configuration object with API keys and settings.
        clients: Dictionary of initialized LLM clients.
        token_counter: Token counting utility.
        instruction_evaluator: Evaluator for instruction adherence.
        factuality_evaluator: Evaluator for factual accuracy.
        tone_evaluator: Evaluator for tone appropriateness.

    Example:
        >>> # Using OpenAI for both querying and evaluation
        >>> config = Config.from_env()
        >>> analyzer = LLMDriftAnalyzer(config)
        >>> prompts = PromptSet.load_from_file("prompts.json")
        >>> results = analyzer.run_drift_analysis(
        ...     prompts=prompts,
        ...     models=["gpt-4-0613", "gpt-4-0125-preview"],
        ...     iterations=10
        ... )

        >>> # Using Ollama for evaluation (free, no API costs)
        >>> config = Config(
        ...     evaluator_provider="ollama",
        ...     evaluator_model="llama3"
        ... )
        >>> analyzer = LLMDriftAnalyzer(config, provider="ollama")
    """

    def __init__(
        self,
        config: Config,
        provider: Optional[str] = None,
        ollama_base_url: Optional[str] = None
    ):
        """
        Initialize the drift analyzer.

        Args:
            config: Configuration with API keys and settings.
            provider: Optional specific provider to use (e.g., "ollama").
                     If None, initializes all available providers.
            ollama_base_url: Optional Ollama server URL override.

        Raises:
            ValueError: If no valid API keys are provided (unless using Ollama).
        """
        self.config = config
        self._logger = get_logger("analyzers.drift")
        self.provider = provider  # Store the specified provider

        # Initialize token counter
        self.token_counter = TokenCounter()

        # Initialize LLM clients
        self._logger.info("Initializing LLM clients...")

        # If Ollama provider is specified, initialize only Ollama client
        if provider == "ollama":
            from llm_drift_analyzer.clients.ollama_client import OllamaClient
            base_url = ollama_base_url or config.ollama_base_url
            self.clients = {
                "ollama": OllamaClient(
                    api_key="",  # Not needed for Ollama
                    api_config=config.api_config,
                    base_url=base_url
                )
            }
        else:
            # Initialize cloud providers
            try:
                api_keys = config.get_api_keys(include_ollama=False)
                self.clients = LLMClientFactory.create_all(api_keys, config.api_config)
            except ValueError:
                # No cloud API keys, initialize empty
                self.clients = {}

            # Also try to add Ollama if server is available
            try:
                from llm_drift_analyzer.clients.ollama_client import OllamaClient
                base_url = ollama_base_url or config.ollama_base_url
                ollama_client = OllamaClient(
                    api_key="",
                    api_config=config.api_config,
                    base_url=base_url
                )
                self.clients["ollama"] = ollama_client
                self._logger.info(f"Ollama client initialized at {base_url}")
            except Exception as e:
                self._logger.debug(f"Ollama not available: {e}")

        if not self.clients:
            raise ValueError(
                "No LLM clients could be initialized. "
                "Configure API keys or start Ollama server."
            )

        self._logger.info(f"Initialized clients: {list(self.clients.keys())}")

        # Initialize evaluators based on evaluator_provider setting
        evaluator_provider = config.evaluator_provider.lower()
        evaluator_model = config.evaluator_model
        ollama_url = ollama_base_url or config.ollama_base_url

        self._logger.info(
            f"Initializing evaluators with provider={evaluator_provider}, "
            f"model={evaluator_model}"
        )

        if evaluator_provider == "ollama":
            # Use Ollama for evaluation (free, local)
            try:
                self.instruction_evaluator = InstructionEvaluator(
                    evaluator_provider="ollama",
                    evaluator_model=evaluator_model,
                    ollama_base_url=ollama_url,
                )
                self.factuality_evaluator = FactualityEvaluator(
                    evaluator_provider="ollama",
                    evaluator_model=evaluator_model,
                    ollama_base_url=ollama_url,
                )
                self.tone_evaluator = ToneEvaluator(
                    evaluator_provider="ollama",
                    evaluator_model=evaluator_model,
                    ollama_base_url=ollama_url,
                )
                self._logger.info(f"Evaluators initialized with Ollama ({evaluator_model})")
            except Exception as e:
                self._logger.warning(
                    f"Failed to initialize Ollama evaluators: {e}. "
                    "Evaluation will use defaults."
                )
                self.instruction_evaluator = None
                self.factuality_evaluator = None
                self.tone_evaluator = None

        elif evaluator_provider == "openai":
            # Use OpenAI for evaluation (requires API key)
            if config.openai_api_key:
                self.instruction_evaluator = InstructionEvaluator(
                    openai_api_key=config.openai_api_key,
                    evaluator_provider="openai",
                    evaluator_model=evaluator_model,
                )
                self.factuality_evaluator = FactualityEvaluator(
                    openai_api_key=config.openai_api_key,
                    evaluator_provider="openai",
                    evaluator_model=evaluator_model,
                )
                self.tone_evaluator = ToneEvaluator(
                    openai_api_key=config.openai_api_key,
                    evaluator_provider="openai",
                    evaluator_model=evaluator_model,
                )
                self._logger.info(f"Evaluators initialized with OpenAI ({evaluator_model})")
            else:
                self._logger.warning(
                    "OpenAI API key not provided but evaluator_provider='openai'. "
                    "Evaluation will use defaults. Set EVALUATOR_PROVIDER=ollama for local evaluation."
                )
                self.instruction_evaluator = None
                self.factuality_evaluator = None
                self.tone_evaluator = None
        else:
            self._logger.warning(
                f"Unknown evaluator_provider: {evaluator_provider}. "
                "Evaluation will use defaults."
            )
            self.instruction_evaluator = None
            self.factuality_evaluator = None
            self.tone_evaluator = None

    def _get_client_for_model(self, model: str) -> Tuple[BaseLLMClient, str]:
        """
        Determine which client to use for a given model.

        Args:
            model: Model identifier.

        Returns:
            Tuple[BaseLLMClient, str]: Client instance and provider name.

        Raises:
            ValueError: If no suitable client found.
        """
        # If a specific provider was specified, always use it
        if self.provider and self.provider in self.clients:
            return self.clients[self.provider], self.provider

        model_lower = model.lower()

        # Map model to provider based on model name patterns
        if any(x in model_lower for x in ["gpt", "davinci", "curie", "o1-", "o3-"]):
            provider = "openai"
        elif any(x in model_lower for x in ["claude"]):
            provider = "anthropic"
        elif any(x in model_lower for x in ["mistral", "mixtral"]) and "mistral" in self.clients:
            # Only use Mistral API if we have the client (not Ollama's mistral)
            provider = "mistral"
        elif any(x in model_lower for x in [
            "llama", "codellama", "phi", "gemma", "qwen", "deepseek",
            "neural-chat", "starling", "vicuna", "falcon", "orca",
            "wizard", "nous", "openhermes", "yi", "solar"
        ]):
            # Open-source models typically run on Ollama
            provider = "ollama"
        else:
            # Default: try Ollama first (for local models), then OpenAI
            if "ollama" in self.clients:
                provider = "ollama"
            elif "openai" in self.clients:
                provider = "openai"
            else:
                provider = list(self.clients.keys())[0] if self.clients else None

        if provider not in self.clients:
            raise ValueError(
                f"No client available for model {model}. "
                f"Available providers: {list(self.clients.keys())}"
            )

        return self.clients[provider], provider

    def _query_model(
        self,
        model: str,
        prompt: str,
        system_message: Optional[str] = None
    ) -> Tuple[str, float]:
        """
        Query a model and return the response with latency.

        Args:
            model: Model identifier.
            prompt: Prompt text.
            system_message: Optional system message.

        Returns:
            Tuple[str, float]: Response text and latency in milliseconds.

        Raises:
            LLMClientError: If the query fails.
        """
        client, provider = self._get_client_for_model(model)

        self._logger.debug(f"Querying {model} via {provider}")

        result = client.query(
            prompt=prompt,
            model=model,
            system_message=system_message,
        )

        return result.response_text, result.latency_ms

    def evaluate_response(
        self,
        prompt: Prompt,
        response: str,
        model_version: str,
        latency_ms: float,
        iteration: int = 0
    ) -> ResponseAnalysis:
        """
        Evaluate a single LLM response across all metrics.

        Args:
            prompt: The Prompt object used for generation.
            response: The LLM's response text.
            model_version: Model identifier.
            latency_ms: Response latency in milliseconds.
            iteration: Iteration number for repeated sampling.

        Returns:
            ResponseAnalysis: Complete analysis of the response.

        Example:
            >>> analysis = analyzer.evaluate_response(
            ...     prompt=prompt,
            ...     response="The benefits are...",
            ...     model_version="gpt-4-0613",
            ...     latency_ms=1250.5
            ... )
            >>> print(analysis.instruction_score)
            3
        """
        self._logger.debug(f"Evaluating response for {prompt.id} from {model_version}")

        # Count tokens
        token_count = self.token_counter.count_tokens(response, model_version)

        # Evaluate instruction adherence
        if self.instruction_evaluator:
            instruction_score = self.instruction_evaluator.evaluate(
                prompt=prompt.text,
                response=response,
                expected_format=prompt.expected_format,
            )
        else:
            instruction_score = 1  # Default

        # Evaluate factuality
        if self.factuality_evaluator:
            factuality_score = self.factuality_evaluator.evaluate(
                prompt=prompt.text,
                response=response,
                reference_answer=prompt.reference_answer,
            )
        else:
            factuality_score = 1  # Default

        # Evaluate tone
        if self.tone_evaluator:
            tone_score = self.tone_evaluator.evaluate(
                prompt=prompt.text,
                response=response,
            )
        else:
            tone_score = 1  # Default

        return ResponseAnalysis(
            prompt_id=prompt.id,
            model_version=model_version,
            response_text=response,
            token_count=token_count,
            latency_ms=latency_ms,
            instruction_score=instruction_score,
            factuality_score=factuality_score,
            tone_score=tone_score,
            timestamp=datetime.now().isoformat(),
            iteration=iteration,
            metadata={
                "prompt_category": prompt.category.value,
            },
        )

    @log_execution_time
    def run_drift_analysis(
        self,
        prompts: PromptSet,
        models: List[str],
        iterations: int = 10,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> AnalysisResultSet:
        """
        Run complete drift analysis across prompts and models.

        Iterates through all combinations of prompts, models, and iterations,
        collecting responses and evaluating them on all metrics.

        Args:
            prompts: Set of prompts to use for analysis.
            models: List of model identifiers to test.
            iterations: Number of iterations per prompt-model combination.
            progress_callback: Optional callback(current, total, message) for progress.

        Returns:
            AnalysisResultSet: Complete results of the analysis.

        Example:
            >>> prompts = PromptSet.load_from_file("prompts.json")
            >>> results = analyzer.run_drift_analysis(
            ...     prompts=prompts,
            ...     models=["gpt-4-0613", "gpt-4-0125-preview"],
            ...     iterations=10
            ... )
            >>> print(f"Collected {len(results)} responses")
        """
        total_queries = len(prompts) * len(models) * iterations
        current_query = 0

        self._logger.info(
            f"Starting drift analysis: {len(prompts)} prompts, "
            f"{len(models)} models, {iterations} iterations"
        )
        self._logger.info(f"Total queries to execute: {total_queries}")

        result_set = AnalysisResultSet(
            metadata={
                "start_time": datetime.now().isoformat(),
                "prompts_count": len(prompts),
                "models": models,
                "iterations": iterations,
            }
        )

        for prompt in prompts:
            self._logger.info(f"Processing prompt: {prompt.id}")

            for model in models:
                self._logger.debug(f"Testing model: {model}")

                for i in range(iterations):
                    current_query += 1

                    if progress_callback:
                        progress_callback(
                            current_query,
                            total_queries,
                            f"Prompt {prompt.id}, Model {model}, Iteration {i+1}/{iterations}"
                        )

                    try:
                        # Query the model
                        response, latency_ms = self._query_model(model, prompt.text)

                        # Evaluate the response
                        analysis = self.evaluate_response(
                            prompt=prompt,
                            response=response,
                            model_version=model,
                            latency_ms=latency_ms,
                            iteration=i,
                        )

                        result_set.add(analysis)

                        self._logger.debug(
                            f"Completed: {prompt.id}/{model}/iter{i+1} - "
                            f"tokens={analysis.token_count}, "
                            f"scores=[{analysis.instruction_score},"
                            f"{analysis.factuality_score},{analysis.tone_score}]"
                        )

                    except LLMClientError as e:
                        self._logger.error(f"Query failed: {e}")
                        # Continue with other queries
                        continue
                    except Exception as e:
                        self._logger.error(f"Unexpected error: {e}")
                        continue

        # Update metadata with completion info
        result_set.metadata["end_time"] = datetime.now().isoformat()
        result_set.metadata["successful_queries"] = len(result_set)
        result_set.metadata["failed_queries"] = total_queries - len(result_set)

        self._logger.info(
            f"Analysis complete: {len(result_set)}/{total_queries} successful queries"
        )

        return result_set

    def analyze_single_prompt(
        self,
        prompt: Prompt,
        models: List[str],
        iterations: int = 5
    ) -> AnalysisResultSet:
        """
        Run analysis for a single prompt across models.

        Convenience method for quick comparisons.

        Args:
            prompt: Single prompt to analyze.
            models: List of models to test.
            iterations: Number of iterations.

        Returns:
            AnalysisResultSet: Results for the single prompt.
        """
        prompt_set = PromptSet(prompts=[prompt])
        return self.run_drift_analysis(prompt_set, models, iterations)

    def compare_model_versions(
        self,
        prompt: Prompt,
        model_versions: List[str],
        iterations: int = 10
    ) -> Dict:
        """
        Compare specific model versions for drift analysis.

        Args:
            prompt: Prompt to use for comparison.
            model_versions: List of model version identifiers.
            iterations: Number of iterations per version.

        Returns:
            Dict: Comparison results with statistics per version.

        Example:
            >>> comparison = analyzer.compare_model_versions(
            ...     prompt=prompt,
            ...     model_versions=["gpt-4-0613", "gpt-4-0125-preview"],
            ...     iterations=10
            ... )
            >>> print(comparison["drift_detected"])
        """
        results = self.analyze_single_prompt(prompt, model_versions, iterations)

        comparison = {
            "prompt_id": prompt.id,
            "model_versions": model_versions,
            "iterations": iterations,
            "by_model": {},
        }

        for model in model_versions:
            model_results = results.filter_by_model(model)

            if not model_results:
                continue

            token_counts = [r.token_count for r in model_results]
            instruction_scores = [r.instruction_score for r in model_results]
            factuality_scores = [r.factuality_score for r in model_results]
            tone_scores = [r.tone_score for r in model_results]
            latencies = [r.latency_ms for r in model_results]

            comparison["by_model"][model] = {
                "sample_size": len(model_results),
                "token_count": {
                    "mean": sum(token_counts) / len(token_counts),
                    "min": min(token_counts),
                    "max": max(token_counts),
                },
                "instruction_score": {
                    "mean": sum(instruction_scores) / len(instruction_scores),
                },
                "factuality_score": {
                    "mean": sum(factuality_scores) / len(factuality_scores),
                },
                "tone_score": {
                    "mean": sum(tone_scores) / len(tone_scores),
                },
                "latency_ms": {
                    "mean": sum(latencies) / len(latencies),
                },
            }

        # Calculate drift indicators
        if len(comparison["by_model"]) >= 2:
            models_data = list(comparison["by_model"].values())
            token_means = [m["token_count"]["mean"] for m in models_data]
            instruction_means = [m["instruction_score"]["mean"] for m in models_data]

            comparison["drift_indicators"] = {
                "token_count_variance": max(token_means) - min(token_means),
                "instruction_score_variance": (
                    max(instruction_means) - min(instruction_means)
                ),
                "drift_detected": (
                    (max(token_means) - min(token_means)) / min(token_means) > 0.1
                    if min(token_means) > 0 else False
                ),
            }

        return comparison

    def get_available_models(self) -> Dict[str, List[str]]:
        """
        Get all available models from initialized clients.

        Returns:
            Dict[str, List[str]]: Models grouped by provider.
        """
        available = {}
        for provider, client in self.clients.items():
            available[provider] = client.list_models()
        return available
