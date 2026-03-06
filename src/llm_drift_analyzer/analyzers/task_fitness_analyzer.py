"""
Task Fitness Analyzer for LLM Drift Analyzer.

Evaluates which open-source LLM model is best suited for each task category,
with special focus on Hindi performance for government use cases.

Routes each prompt to its category-specific evaluator and aggregates
results into a model × task fitness matrix.
"""

import time
from datetime import datetime
from typing import Dict, List, Optional, Callable, Tuple
from pathlib import Path
from collections import defaultdict

from llm_drift_analyzer.models.prompt import Prompt, PromptSet, PromptCategory
from llm_drift_analyzer.models.task_fitness import (
    TaskFitnessScore,
    TaskFitnessMatrix,
)
from llm_drift_analyzer.clients.base_client import (
    BaseLLMClient,
    LLMClientFactory,
    LLMClientError,
)
from llm_drift_analyzer.evaluators.base_evaluator import BaseEvaluator
from llm_drift_analyzer.evaluators.instruction_evaluator import InstructionEvaluator
from llm_drift_analyzer.evaluators.factuality_evaluator import FactualityEvaluator
from llm_drift_analyzer.evaluators.tone_evaluator import ToneEvaluator
from llm_drift_analyzer.evaluators.summarization_evaluator import SummarizationEvaluator
from llm_drift_analyzer.evaluators.translation_evaluator import TranslationEvaluator
from llm_drift_analyzer.evaluators.logical_math_evaluator import LogicalMathEvaluator
from llm_drift_analyzer.evaluators.conversational_evaluator import ConversationalEvaluator
from llm_drift_analyzer.evaluators.legal_admin_evaluator import LegalAdminEvaluator
from llm_drift_analyzer.evaluators.sentiment_evaluator import SentimentEvaluator
from llm_drift_analyzer.evaluators.code_generation_evaluator import CodeGenerationEvaluator
from llm_drift_analyzer.utils.config import Config
from llm_drift_analyzer.utils.logger import get_logger


# Map PromptCategory values to their evaluator class
CATEGORY_EVALUATOR_MAP: Dict[str, type] = {
    "instruction_following": InstructionEvaluator,
    "factual_qa": FactualityEvaluator,
    "creative_reasoning": ToneEvaluator,
    "summarization": SummarizationEvaluator,
    "translation": TranslationEvaluator,
    "logical_mathematical": LogicalMathEvaluator,
    "conversational": ConversationalEvaluator,
    "legal_administrative": LegalAdminEvaluator,
    "sentiment_analysis": SentimentEvaluator,
    "code_generation": CodeGenerationEvaluator,
}


class TaskFitnessAnalyzer:
    """
    Evaluates which LLM model is best suited for each task category.

    Runs models across all task categories with category-specific
    evaluators and produces a model × task fitness matrix with
    recommendations.

    Designed for government use cases where the goal is to select
    the right open-source model for each type of task, especially
    in Hindi.

    Example:
        >>> config = Config.from_env()
        >>> analyzer = TaskFitnessAnalyzer(config, provider="ollama")
        >>> prompts = PromptSet.load_from_file("task_fitness_hindi.json")
        >>> matrix = analyzer.run_fitness_analysis(
        ...     prompts=prompts,
        ...     models=["llama3", "mistral", "qwen2"],
        ...     iterations=3,
        ... )
        >>> recs = matrix.get_recommendations(language="hi")
        >>> for r in recs:
        ...     print(f"{r['category']}: {r['recommended_model']} ({r['score']})")
    """

    def __init__(
        self,
        config: Config,
        provider: Optional[str] = None,
        ollama_base_url: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
    ):
        """
        Initialize the task fitness analyzer.

        Args:
            config: Configuration with API keys and settings.
            provider: LLM provider to use (default: auto-detect, typically "ollama").
            ollama_base_url: Optional Ollama server URL override.
            reasoning_effort: Reasoning effort for thinking-capable evaluator models
                (e.g. gpt-oss). One of "low", "medium", "high". Leave as None for
                non-reasoning models — the parameter is simply not sent in that case.
        """
        self.config = config
        self._logger = get_logger("analyzers.task_fitness")
        self.provider = provider

        # Initialize LLM client
        ollama_url = ollama_base_url or config.ollama_base_url

        if provider == "ollama":
            from llm_drift_analyzer.clients.ollama_client import OllamaClient
            self.client = OllamaClient(
                api_key="",
                api_config=config.api_config,
                base_url=ollama_url,
            )
            self._logger.info(f"LLM client: Ollama at {ollama_url}")
        else:
            # Try to initialize based on available keys
            try:
                api_keys = config.get_api_keys(include_ollama=False)
                clients = LLMClientFactory.create_all(api_keys, config.api_config)
                # Pick the first available client
                self.client = next(iter(clients.values()))
                self._logger.info(f"LLM client: {self.client.provider_name}")
            except (ValueError, StopIteration):
                # Fall back to Ollama
                from llm_drift_analyzer.clients.ollama_client import OllamaClient
                self.client = OllamaClient(
                    api_key="",
                    api_config=config.api_config,
                    base_url=ollama_url,
                )
                self._logger.info(f"LLM client: Ollama (fallback) at {ollama_url}")

        # Initialize category-specific evaluators
        evaluator_provider = config.evaluator_provider.lower()
        evaluator_model = config.evaluator_model

        self._logger.info(
            f"Initializing evaluators: provider={evaluator_provider}, "
            f"model={evaluator_model}"
        )

        self.evaluators: Dict[str, BaseEvaluator] = {}
        evaluator_kwargs = {"temperature": 0.1}

        if reasoning_effort is not None:
            evaluator_kwargs["reasoning_effort"] = reasoning_effort

        if evaluator_provider == "ollama":
            evaluator_kwargs.update({
                "evaluator_provider": "ollama",
                "evaluator_model": evaluator_model,
                "ollama_base_url": ollama_url,
            })
        elif evaluator_provider == "openai" and config.openai_api_key:
            evaluator_kwargs.update({
                "openai_api_key": config.openai_api_key,
                "evaluator_provider": "openai",
                "evaluator_model": evaluator_model,
            })
        else:
            self._logger.warning(
                f"Cannot initialize evaluators with provider={evaluator_provider}. "
                "Scores will use defaults."
            )
            return

        for category, evaluator_cls in CATEGORY_EVALUATOR_MAP.items():
            try:
                self.evaluators[category] = evaluator_cls(**evaluator_kwargs)
                self._logger.debug(f"Evaluator for '{category}': {evaluator_cls.__name__}")
            except Exception as e:
                self._logger.warning(
                    f"Failed to initialize evaluator for '{category}': {e}"
                )

        self._logger.info(
            f"Initialized {len(self.evaluators)}/{len(CATEGORY_EVALUATOR_MAP)} evaluators"
        )

    def _query_model(self, model: str, prompt_text: str) -> Tuple[str, float]:
        """
        Query a model and return response + latency.

        Args:
            model: Model identifier.
            prompt_text: Prompt text.

        Returns:
            Tuple of (response_text, latency_ms).
        """
        result = self.client.query(prompt=prompt_text, model=model)
        return result.response_text, result.latency_ms

    def _evaluate_for_category(
        self, category: str, prompt: Prompt, response: str
    ) -> int:
        """
        Evaluate a response using the category-specific evaluator.

        Args:
            category: PromptCategory value string.
            prompt: The Prompt object.
            response: Model's response text.

        Returns:
            Score from the evaluator (0-3 for new evaluators).
        """
        evaluator = self.evaluators.get(category)
        if not evaluator:
            self._logger.debug(f"No evaluator for '{category}', using default score")
            return 1

        kwargs = {}
        if prompt.reference_answer:
            kwargs["reference_answer"] = prompt.reference_answer
        if prompt.expected_format:
            kwargs["expected_format"] = prompt.expected_format

        return evaluator.evaluate(
            prompt=prompt.text,
            response=response,
            **kwargs,
        )

    def run_fitness_analysis(
        self,
        prompts: PromptSet,
        models: List[str],
        iterations: int = 3,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> TaskFitnessMatrix:
        """
        Run fitness analysis across all task categories.

        For each (prompt, model, iteration), queries the model and
        evaluates the response using the category-specific evaluator.
        Aggregates results into a TaskFitnessMatrix.

        Args:
            prompts: PromptSet with prompts across categories.
            models: List of model identifiers to test.
            iterations: Number of iterations per prompt-model combo.
            progress_callback: Optional callback(current, total, message).

        Returns:
            TaskFitnessMatrix with scores for all model/category/language combos.
        """
        total_queries = len(prompts) * len(models) * iterations
        current_query = 0

        self._logger.info(
            f"Starting task fitness analysis: {len(prompts)} prompts, "
            f"{len(models)} models, {iterations} iterations = {total_queries} queries"
        )

        # Collect raw scores: (model, category, language) -> (scores[], latencies[])
        raw_data: Dict[
            Tuple[str, str, str], Dict[str, List[float]]
        ] = defaultdict(lambda: {"scores": [], "latencies": []})

        for prompt in prompts:
            category = prompt.category.value
            language = prompt.language.value

            for model in models:
                for i in range(iterations):
                    current_query += 1

                    if progress_callback:
                        progress_callback(
                            current_query,
                            total_queries,
                            f"{category}/{model}/iter{i+1} [{language}]"
                        )

                    try:
                        response, latency_ms = self._query_model(model, prompt.text)
                        score = self._evaluate_for_category(category, prompt, response)

                        key = (model, category, language)
                        raw_data[key]["scores"].append(float(score))
                        raw_data[key]["latencies"].append(latency_ms)

                        self._logger.debug(
                            f"{prompt.id}/{model}/iter{i+1}: "
                            f"score={score}, latency={latency_ms:.0f}ms"
                        )

                    except LLMClientError as e:
                        self._logger.error(f"Query failed ({model}): {e}")
                        continue
                    except Exception as e:
                        self._logger.error(f"Unexpected error ({model}): {e}")
                        continue

        # Build the fitness matrix
        matrix = TaskFitnessMatrix(
            metadata={
                "start_time": datetime.now().isoformat(),
                "total_queries": total_queries,
                "successful_queries": sum(
                    len(v["scores"]) for v in raw_data.values()
                ),
                "iterations": iterations,
                "evaluator_provider": self.config.evaluator_provider,
                "evaluator_model": self.config.evaluator_model,
            }
        )

        for (model, category, language), data in raw_data.items():
            fitness_score = TaskFitnessScore.from_raw_scores(
                model=model,
                category=category,
                language=language,
                scores=data["scores"],
                latencies=data["latencies"],
            )
            matrix.add_score(fitness_score)

        self._logger.info(
            f"Analysis complete: {len(matrix.scores)} score entries "
            f"for {len(matrix.models)} models across {len(matrix.categories)} categories"
        )

        return matrix
