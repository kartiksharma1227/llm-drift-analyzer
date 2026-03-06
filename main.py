#!/usr/bin/env python3
"""
LLM Behavioral Drift Analyzer - Command Line Interface.

A comprehensive tool for tracking behavioral drift in Large Language Models.
Supports multilingual analysis including Hindi (हिंदी).

Usage:
    python main.py analyze --models gpt-4 claude-3-opus --prompts data/prompts/benchmark_prompts.json
    python main.py analyze-hindi --models llama3 mistral --iterations 5
    python main.py crosslingual --models gpt-4 llama3 --iterations 5
    python main.py report --input results.json --format markdown
    python main.py compare --model1 gpt-4-0613 --model2 gpt-4-0125-preview
    python main.py list-models
"""

import argparse
import sys
import json
from pathlib import Path
from typing import List, Optional, Dict

from llm_drift_analyzer.utils.config import Config
from llm_drift_analyzer.utils.logger import setup_logger, get_logger
from llm_drift_analyzer.models.prompt import PromptSet, Language
from llm_drift_analyzer.models.response_analysis import AnalysisResultSet
from llm_drift_analyzer.analyzers.drift_analyzer import LLMDriftAnalyzer
from llm_drift_analyzer.analyzers.statistical_analyzer import DriftStatisticalAnalyzer
from llm_drift_analyzer.analyzers.crosslingual_analyzer import CrossLingualAnalyzer
from llm_drift_analyzer.analyzers.task_fitness_analyzer import TaskFitnessAnalyzer
from llm_drift_analyzer.reporters.report_generator import ReportGenerator
from llm_drift_analyzer.reporters.visualizer import DriftVisualizer
from llm_drift_analyzer.reporters.fitness_report_generator import FitnessReportGenerator
from llm_drift_analyzer.reporters.fitness_visualizer import FitnessVisualizer
from llm_drift_analyzer.utils.multilingual_tokenizer import MultilingualTokenCounter


def create_parser() -> argparse.ArgumentParser:
    """
    Create the argument parser for the CLI.

    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="LLM Behavioral Drift Analyzer - Track changes in LLM behavior over time",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run drift analysis
  python main.py analyze --models gpt-4-0613 gpt-4-0125-preview --iterations 10

  # Generate report from existing results
  python main.py report --input output/results.json --format markdown --visualize

  # Compare two model versions
  python main.py compare --model1 gpt-4-0613 --model2 gpt-4-0125-preview

  # List available models
  python main.py list-models

For more information, see the README.md file.
        """
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    parser.add_argument(
        "--env-file",
        type=str,
        default=None,
        help="Path to .env file with API keys"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Analyze command
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Run drift analysis on prompts across models"
    )
    analyze_parser.add_argument(
        "--models", "-m",
        nargs="+",
        required=True,
        help="Model identifiers to test (e.g., gpt-4-0613 gpt-4-0125-preview, llama3 mistral)"
    )
    analyze_parser.add_argument(
        "--provider",
        type=str,
        default=None,
        choices=["openai", "anthropic", "mistral", "ollama"],
        help="LLM provider to use (default: auto-detect based on model names)"
    )
    analyze_parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://localhost:11434",
        help="Ollama server URL (default: http://localhost:11434)"
    )
    analyze_parser.add_argument(
        "--prompts", "-p",
        type=str,
        default="data/prompts/benchmark_prompts.json",
        help="Path to prompts JSON file"
    )
    analyze_parser.add_argument(
        "--use-expanded",
        action="store_true",
        help="Use expanded benchmark datasets (50 prompts each for English/Hindi with IFEval-style and TruthfulQA-style prompts)"
    )
    analyze_parser.add_argument(
        "--iterations", "-i",
        type=int,
        default=10,
        help="Number of iterations per prompt-model combination"
    )
    analyze_parser.add_argument(
        "--output", "-o",
        type=str,
        default="output",
        help="Output directory for results"
    )
    analyze_parser.add_argument(
        "--save-responses",
        action="store_true",
        help="Save full response texts (increases file size)"
    )
    analyze_parser.add_argument(
        "--language", "-l",
        type=str,
        default=None,
        choices=["en", "hi", "all"],
        help="Filter prompts by language (en=English, hi=Hindi, all=both)"
    )
    analyze_parser.add_argument(
        "--evaluator-provider",
        type=str,
        default=None,
        choices=["openai", "ollama"],
        help="Provider for response evaluation/scoring. Use 'ollama' for free local evaluation."
    )
    analyze_parser.add_argument(
        "--evaluator-model",
        type=str,
        default=None,
        help="Model to use for evaluation (e.g., gpt-4, llama3). Defaults to gpt-4 for OpenAI, llama3 for Ollama."
    )

    # Analyze Hindi command (convenience)
    hindi_parser = subparsers.add_parser(
        "analyze-hindi",
        help="Run drift analysis using Hindi prompts (हिंदी)"
    )
    hindi_parser.add_argument(
        "--models", "-m",
        nargs="+",
        required=True,
        help="Model identifiers to test"
    )
    hindi_parser.add_argument(
        "--provider",
        type=str,
        default=None,
        choices=["openai", "anthropic", "mistral", "ollama"],
        help="LLM provider to use"
    )
    hindi_parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://localhost:11434",
        help="Ollama server URL"
    )
    hindi_parser.add_argument(
        "--iterations", "-i",
        type=int,
        default=5,
        help="Number of iterations per prompt-model combination"
    )
    hindi_parser.add_argument(
        "--output", "-o",
        type=str,
        default="output/hindi",
        help="Output directory for results"
    )
    hindi_parser.add_argument(
        "--evaluator-provider",
        type=str,
        default=None,
        choices=["openai", "ollama"],
        help="Provider for response evaluation/scoring. Use 'ollama' for free local evaluation."
    )
    hindi_parser.add_argument(
        "--evaluator-model",
        type=str,
        default=None,
        help="Model to use for evaluation (e.g., gpt-4, llama3)"
    )
    hindi_parser.add_argument(
        "--use-expanded",
        action="store_true",
        help="Use expanded Hindi benchmark (50 prompts with more diverse categories)"
    )

    # Cross-lingual comparison command
    crosslingual_parser = subparsers.add_parser(
        "crosslingual",
        help="Compare model performance between English and Hindi"
    )
    crosslingual_parser.add_argument(
        "--models", "-m",
        nargs="+",
        required=True,
        help="Model identifiers to test"
    )
    crosslingual_parser.add_argument(
        "--provider",
        type=str,
        default=None,
        choices=["openai", "anthropic", "mistral", "ollama"],
        help="LLM provider to use"
    )
    crosslingual_parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://localhost:11434",
        help="Ollama server URL"
    )
    crosslingual_parser.add_argument(
        "--iterations", "-i",
        type=int,
        default=5,
        help="Number of iterations per prompt-model-language combination"
    )
    crosslingual_parser.add_argument(
        "--output", "-o",
        type=str,
        default="output/crosslingual",
        help="Output directory for results"
    )
    crosslingual_parser.add_argument(
        "--evaluator-provider",
        type=str,
        default=None,
        choices=["openai", "ollama"],
        help="Provider for response evaluation/scoring. Use 'ollama' for free local evaluation."
    )
    crosslingual_parser.add_argument(
        "--evaluator-model",
        type=str,
        default=None,
        help="Model to use for evaluation (e.g., gpt-4, llama3)"
    )

    # Report command
    report_parser = subparsers.add_parser(
        "report",
        help="Generate report from existing analysis results"
    )
    report_parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to results JSON file"
    )
    report_parser.add_argument(
        "--format", "-f",
        choices=["markdown", "json", "text"],
        default="markdown",
        help="Output format for the report"
    )
    report_parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file path (default: auto-generated)"
    )
    report_parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization charts"
    )
    report_parser.add_argument(
        "--include-samples",
        action="store_true",
        help="Include sample responses in the report"
    )

    # Compare command
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare two model versions"
    )
    compare_parser.add_argument(
        "--model1",
        type=str,
        required=True,
        help="First model version"
    )
    compare_parser.add_argument(
        "--model2",
        type=str,
        required=True,
        help="Second model version"
    )
    compare_parser.add_argument(
        "--prompts", "-p",
        type=str,
        default="data/prompts/benchmark_prompts.json",
        help="Path to prompts JSON file"
    )
    compare_parser.add_argument(
        "--iterations", "-i",
        type=int,
        default=5,
        help="Number of iterations per model"
    )
    compare_parser.add_argument(
        "--metric",
        choices=["token_count", "instruction_score", "factuality_score", "tone_score"],
        default="token_count",
        help="Primary metric for comparison"
    )
    compare_parser.add_argument(
        "--provider",
        type=str,
        default=None,
        choices=["openai", "anthropic", "mistral", "ollama"],
        help="LLM provider to use (default: auto-detect)"
    )
    compare_parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://localhost:11434",
        help="Ollama server URL (default: http://localhost:11434)"
    )
    compare_parser.add_argument(
        "--evaluator-provider",
        type=str,
        default=None,
        choices=["openai", "ollama"],
        help="Provider for response evaluation/scoring. Use 'ollama' for free local evaluation."
    )
    compare_parser.add_argument(
        "--evaluator-model",
        type=str,
        default=None,
        help="Model to use for evaluation (e.g., gpt-4, llama3)"
    )

    # List models command
    list_parser = subparsers.add_parser(
        "list-models",
        help="List available models from configured providers"
    )
    list_parser.add_argument(
        "--provider",
        type=str,
        default=None,
        choices=["openai", "anthropic", "mistral", "ollama"],
        help="Filter models by provider"
    )
    list_parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://localhost:11434",
        help="Ollama server URL (default: http://localhost:11434)"
    )

    # Validate command
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate configuration and API connections"
    )

    # Task Fitness command
    fitness_parser = subparsers.add_parser(
        "task-fitness",
        help="Evaluate which model is best for which task (especially Hindi)"
    )
    fitness_parser.add_argument(
        "--models", "-m",
        nargs="+",
        required=True,
        help="Model identifiers to evaluate (e.g., llama3 mistral qwen2)"
    )
    fitness_parser.add_argument(
        "--provider",
        type=str,
        default=None,
        choices=["openai", "anthropic", "mistral", "ollama"],
        help="LLM provider to use (default: auto-detect)"
    )
    fitness_parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://localhost:11434",
        help="Ollama server URL (default: http://localhost:11434)"
    )
    fitness_parser.add_argument(
        "--iterations", "-i",
        type=int,
        default=3,
        help="Number of iterations per prompt-model combination (default: 3)"
    )
    fitness_parser.add_argument(
        "--output", "-o",
        type=str,
        default="output/task_fitness",
        help="Output directory for results"
    )
    fitness_parser.add_argument(
        "--languages",
        type=str,
        default="both",
        choices=["en", "hi", "both"],
        help="Languages to evaluate (default: both)"
    )
    fitness_parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Filter to specific task categories (default: all)"
    )
    fitness_parser.add_argument(
        "--evaluator-provider",
        type=str,
        default=None,
        choices=["openai", "ollama"],
        help="Provider for response evaluation/scoring"
    )
    fitness_parser.add_argument(
        "--evaluator-model",
        type=str,
        default=None,
        help="Model to use for evaluation (e.g., gpt-4, llama3)"
    )
    fitness_parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization charts"
    )
    fitness_parser.add_argument(
        "--reasoning-effort",
        type=str,
        default=None,
        choices=["low", "medium", "high"],
        help=(
            "Reasoning effort for thinking-capable evaluator models (e.g. gpt-oss). "
            "One of: low, medium, high. "
            "Only pass this when your --evaluator-model is a reasoning model — "
            "it is silently ignored for non-reasoning models."
        )
    )

    return parser


def cmd_analyze(args: argparse.Namespace, config: Config) -> int:
    """
    Execute the analyze command.

    Args:
        args: Parsed command line arguments.
        config: Configuration object.

    Returns:
        int: Exit code (0 for success).
    """
    logger = get_logger()
    logger.info("Starting drift analysis")

    # Apply evaluator settings from CLI args
    evaluator_provider = getattr(args, 'evaluator_provider', None)
    evaluator_model = getattr(args, 'evaluator_model', None)
    if evaluator_provider:
        config.evaluator_provider = evaluator_provider
        # Set sensible default model for the provider if not specified
        if not evaluator_model:
            config.evaluator_model = "llama3" if evaluator_provider == "ollama" else "gpt-4"
    if evaluator_model:
        config.evaluator_model = evaluator_model
    if evaluator_provider or evaluator_model:
        logger.info(f"Using {config.evaluator_provider} for evaluation with model {config.evaluator_model}")

    # Handle --use-expanded flag
    use_expanded = getattr(args, 'use_expanded', False)
    if use_expanded:
        # Determine which expanded dataset to use based on language
        language = getattr(args, 'language', None)
        if language == "hi":
            prompts_path = Path("data/prompts/hindi_benchmark_expanded.json")
        elif language == "en":
            prompts_path = Path("data/prompts/english_benchmark_expanded.json")
        else:
            # Default to English expanded for now
            prompts_path = Path("data/prompts/english_benchmark_expanded.json")
        logger.info(f"Using expanded benchmark dataset: {prompts_path}")
    else:
        prompts_path = Path(args.prompts)

    # Load prompts
    if not prompts_path.exists():
        logger.error(f"Prompts file not found: {prompts_path}")
        return 1

    logger.info(f"Loading prompts from {prompts_path}")
    prompts = PromptSet.load_from_file(prompts_path)
    logger.info(f"Loaded {len(prompts)} prompts")

    # Initialize analyzer with optional provider
    try:
        provider = getattr(args, 'provider', None)
        ollama_url = getattr(args, 'ollama_url', None)
        analyzer = LLMDriftAnalyzer(
            config,
            provider=provider,
            ollama_base_url=ollama_url
        )
    except ValueError as e:
        logger.error(f"Failed to initialize analyzer: {e}")
        return 1

    # Progress callback
    def progress_callback(current: int, total: int, message: str):
        percent = (current / total) * 100
        print(f"\r[{percent:5.1f}%] {message}", end="", flush=True)

    # Run analysis
    logger.info(f"Running analysis with models: {args.models}")
    logger.info(f"Iterations per prompt-model: {args.iterations}")

    results = analyzer.run_drift_analysis(
        prompts=prompts,
        models=args.models,
        iterations=args.iterations,
        progress_callback=progress_callback
    )

    print()  # New line after progress

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "results.json"
    results.save_to_file(results_path)
    logger.info(f"Results saved to {results_path}")

    # Generate quick report
    report_generator = ReportGenerator(results)
    report_path = output_dir / "drift_report.md"
    report_generator.save_report(report_path, format="markdown")
    logger.info(f"Report saved to {report_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("Analysis Complete")
    print("=" * 60)
    print(report_generator.generate_text_summary())

    return 0


def cmd_report(args: argparse.Namespace, config: Config) -> int:
    """
    Execute the report command.

    Args:
        args: Parsed command line arguments.
        config: Configuration object.

    Returns:
        int: Exit code (0 for success).
    """
    logger = get_logger()

    # Load results
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Results file not found: {input_path}")
        return 1

    logger.info(f"Loading results from {input_path}")
    results = AnalysisResultSet.load_from_file(input_path)
    logger.info(f"Loaded {len(results)} results")

    # Generate report
    report_generator = ReportGenerator(results)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        ext_map = {"markdown": "md", "json": "json", "text": "txt"}
        ext = ext_map.get(args.format, "txt")
        output_path = input_path.parent / f"drift_report.{ext}"

    report_generator.save_report(
        output_path,
        format=args.format,
        include_samples=args.include_samples
    )
    logger.info(f"Report saved to {output_path}")

    # Generate visualizations if requested
    if args.visualize:
        logger.info("Generating visualizations")
        visualizer = DriftVisualizer(results)
        charts_dir = output_path.parent / "charts"
        saved_charts = visualizer.save_all_plots(charts_dir)
        logger.info(f"Saved {len(saved_charts)} charts to {charts_dir}")

    print(f"Report generated: {output_path}")
    return 0


def cmd_compare(args: argparse.Namespace, config: Config) -> int:
    """
    Execute the compare command.

    Args:
        args: Parsed command line arguments.
        config: Configuration object.

    Returns:
        int: Exit code (0 for success).
    """
    logger = get_logger()
    logger.info(f"Comparing models: {args.model1} vs {args.model2}")

    # Apply evaluator settings from CLI args
    evaluator_provider = getattr(args, 'evaluator_provider', None)
    evaluator_model = getattr(args, 'evaluator_model', None)
    if evaluator_provider:
        config.evaluator_provider = evaluator_provider
        if not evaluator_model:
            config.evaluator_model = "llama3" if evaluator_provider == "ollama" else "gpt-4"
    if evaluator_model:
        config.evaluator_model = evaluator_model

    # Load prompts
    prompts_path = Path(args.prompts)
    if not prompts_path.exists():
        logger.error(f"Prompts file not found: {prompts_path}")
        return 1

    prompts = PromptSet.load_from_file(prompts_path)

    # Initialize analyzer with optional provider
    try:
        provider = getattr(args, 'provider', None)
        ollama_url = getattr(args, 'ollama_url', None)
        analyzer = LLMDriftAnalyzer(
            config,
            provider=provider,
            ollama_base_url=ollama_url
        )
    except ValueError as e:
        logger.error(f"Failed to initialize analyzer: {e}")
        return 1

    # Run comparison
    models = [args.model1, args.model2]

    results = analyzer.run_drift_analysis(
        prompts=prompts,
        models=models,
        iterations=args.iterations
    )

    # Analyze results
    stat_analyzer = DriftStatisticalAnalyzer(results)
    length_trends = stat_analyzer.analyze_length_trends()

    # Print comparison
    print("\n" + "=" * 60)
    print(f"Model Comparison: {args.model1} vs {args.model2}")
    print("=" * 60)

    print(f"\nMetric: {args.metric}")
    print("-" * 40)

    for model in models:
        model_results = results.filter_by_model(model)
        if model_results:
            values = [getattr(r, args.metric) for r in model_results]
            mean_val = sum(values) / len(values)
            print(f"{model}: Mean = {mean_val:.2f}")

    # Print statistical significance
    comparison_key = f"{args.model1}_vs_{args.model2}"
    alt_key = f"{args.model2}_vs_{args.model1}"

    pairwise = length_trends.get('pairwise', {})
    comparison = pairwise.get(comparison_key) or pairwise.get(alt_key)

    if comparison:
        print(f"\nStatistical Analysis:")
        print(f"  t-statistic: {comparison['t_statistic']:.3f}")
        print(f"  p-value: {comparison['p_value']:.3e}")
        print(f"  Cohen's d: {comparison['cohens_d']:.3f} ({comparison['effect_interpretation']})")
        print(f"  Significant: {'Yes' if comparison['significant'] else 'No'}")

    return 0


def cmd_analyze_hindi(args: argparse.Namespace, config: Config) -> int:
    """
    Execute Hindi-specific analysis command.

    Args:
        args: Parsed command line arguments.
        config: Configuration object.

    Returns:
        int: Exit code (0 for success).
    """
    logger = get_logger()
    logger.info("Starting Hindi (हिंदी) drift analysis")

    # Apply evaluator settings from CLI args
    evaluator_provider = getattr(args, 'evaluator_provider', None)
    evaluator_model = getattr(args, 'evaluator_model', None)
    if evaluator_provider:
        config.evaluator_provider = evaluator_provider
        if not evaluator_model:
            config.evaluator_model = "llama3" if evaluator_provider == "ollama" else "gpt-4"
    if evaluator_model:
        config.evaluator_model = evaluator_model

    # Load Hindi prompts (use expanded if requested)
    use_expanded = getattr(args, 'use_expanded', False)
    if use_expanded:
        prompts_path = Path("data/prompts/hindi_benchmark_expanded.json")
        logger.info("Using expanded Hindi benchmark (50 prompts)")
    else:
        prompts_path = Path("data/prompts/hindi_benchmark_prompts.json")

    if not prompts_path.exists():
        logger.error(f"Hindi prompts file not found: {prompts_path}")
        return 1

    logger.info(f"Loading Hindi prompts from {prompts_path}")
    prompts = PromptSet.load_from_file(prompts_path)
    logger.info(f"Loaded {len(prompts)} Hindi prompts")

    # Initialize analyzer
    try:
        provider = getattr(args, 'provider', None)
        ollama_url = getattr(args, 'ollama_url', None)
        analyzer = LLMDriftAnalyzer(
            config,
            provider=provider,
            ollama_base_url=ollama_url
        )
    except ValueError as e:
        logger.error(f"Failed to initialize analyzer: {e}")
        return 1

    # Progress callback
    def progress_callback(current: int, total: int, message: str):
        percent = (current / total) * 100
        print(f"\r[{percent:5.1f}%] {message}", end="", flush=True)

    # Run analysis
    logger.info(f"Running Hindi analysis with models: {args.models}")

    results = analyzer.run_drift_analysis(
        prompts=prompts,
        models=args.models,
        iterations=args.iterations,
        progress_callback=progress_callback
    )

    print()  # New line after progress

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "hindi_results.json"
    results.save_to_file(results_path)
    logger.info(f"Results saved to {results_path}")

    # Analyze Hindi-specific metrics
    token_counter = MultilingualTokenCounter()
    print("\n" + "=" * 60)
    print("Hindi Analysis Complete (हिंदी विश्लेषण)")
    print("=" * 60)

    # Summary by model
    for model in args.models:
        model_results = results.filter_by_model(model)
        if not model_results:
            continue

        print(f"\n📊 Model: {model}")
        print("-" * 40)

        # Analyze script consistency
        total_devanagari = 0
        total_latin = 0
        code_mixing_ratios = []

        for r in model_results:
            analysis = token_counter.analyze_text(r.response_text)
            total_devanagari += analysis.devanagari_chars
            total_latin += analysis.latin_chars
            code_mixing_ratios.append(analysis.code_mixing_ratio)

        avg_code_mixing = sum(code_mixing_ratios) / len(code_mixing_ratios)
        script_ratio = total_devanagari / max(1, total_devanagari + total_latin)

        print(f"  Responses: {len(model_results)}")
        print(f"  Avg Instruction Score: {sum(r.instruction_score for r in model_results)/len(model_results):.2f}/3")
        print(f"  Avg Factuality Score: {sum(r.factuality_score for r in model_results)/len(model_results):.2f}/2")
        print(f"  Devanagari Usage: {script_ratio:.1%}")
        print(f"  Code-Mixing Ratio: {avg_code_mixing:.1%}")

        if avg_code_mixing > 0.2:
            print("  ⚠️ High code-mixing detected - model may be mixing English")
        elif script_ratio < 0.5:
            print("  ⚠️ Low Hindi content - model may be responding in English")
        else:
            print("  ✅ Good Hindi language consistency")

    return 0


def cmd_crosslingual(args: argparse.Namespace, config: Config) -> int:
    """
    Execute cross-lingual comparison (English vs Hindi).

    Args:
        args: Parsed command line arguments.
        config: Configuration object.

    Returns:
        int: Exit code (0 for success).
    """
    logger = get_logger()
    logger.info("Starting cross-lingual analysis (English vs Hindi)")

    # Apply evaluator settings from CLI args
    evaluator_provider = getattr(args, 'evaluator_provider', None)
    evaluator_model = getattr(args, 'evaluator_model', None)
    if evaluator_provider:
        config.evaluator_provider = evaluator_provider
        if not evaluator_model:
            config.evaluator_model = "llama3" if evaluator_provider == "ollama" else "gpt-4"
    if evaluator_model:
        config.evaluator_model = evaluator_model

    # Load both prompt sets
    en_prompts_path = Path("data/prompts/benchmark_prompts.json")
    hi_prompts_path = Path("data/prompts/hindi_benchmark_prompts.json")

    if not en_prompts_path.exists():
        logger.error(f"English prompts not found: {en_prompts_path}")
        return 1
    if not hi_prompts_path.exists():
        logger.error(f"Hindi prompts not found: {hi_prompts_path}")
        return 1

    en_prompts = PromptSet.load_from_file(en_prompts_path)
    hi_prompts = PromptSet.load_from_file(hi_prompts_path)

    logger.info(f"Loaded {len(en_prompts)} English and {len(hi_prompts)} Hindi prompts")

    # Initialize analyzer
    try:
        provider = getattr(args, 'provider', None)
        ollama_url = getattr(args, 'ollama_url', None)
        analyzer = LLMDriftAnalyzer(
            config,
            provider=provider,
            ollama_base_url=ollama_url
        )
    except ValueError as e:
        logger.error(f"Failed to initialize analyzer: {e}")
        return 1

    # Progress tracking
    total_queries = (len(en_prompts) + len(hi_prompts)) * len(args.models) * args.iterations
    current = [0]

    def progress_callback(curr: int, total: int, message: str):
        current[0] += 1
        percent = (current[0] / total_queries) * 100
        print(f"\r[{percent:5.1f}%] {message}", end="", flush=True)

    # Run analysis for both languages
    print("Analyzing English prompts...")
    en_results = analyzer.run_drift_analysis(
        prompts=en_prompts,
        models=args.models,
        iterations=args.iterations,
        progress_callback=progress_callback
    )

    print("\nAnalyzing Hindi prompts...")
    hi_results = analyzer.run_drift_analysis(
        prompts=hi_prompts,
        models=args.models,
        iterations=args.iterations,
        progress_callback=progress_callback
    )

    print()  # New line

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    en_results.save_to_file(output_dir / "english_results.json")
    hi_results.save_to_file(output_dir / "hindi_results.json")

    # Cross-lingual analysis
    print("\n" + "=" * 70)
    print("Cross-Lingual Analysis: English vs Hindi (अंग्रेज़ी vs हिंदी)")
    print("=" * 70)

    for model in args.models:
        en_model_results = en_results.filter_by_model(model)
        hi_model_results = hi_results.filter_by_model(model)

        if not en_model_results or not hi_model_results:
            continue

        print(f"\n📊 Model: {model}")
        print("-" * 50)

        # Calculate metrics
        en_instruction = sum(r.instruction_score for r in en_model_results) / len(en_model_results)
        hi_instruction = sum(r.instruction_score for r in hi_model_results) / len(hi_model_results)

        en_factuality = sum(r.factuality_score for r in en_model_results) / len(en_model_results)
        hi_factuality = sum(r.factuality_score for r in hi_model_results) / len(hi_model_results)

        en_tone = sum(r.tone_score for r in en_model_results) / len(en_model_results)
        hi_tone = sum(r.tone_score for r in hi_model_results) / len(hi_model_results)

        en_tokens = sum(r.token_count for r in en_model_results) / len(en_model_results)
        hi_tokens = sum(r.token_count for r in hi_model_results) / len(hi_model_results)

        # Display comparison
        print(f"  {'Metric':<25} {'English':>10} {'Hindi':>10} {'Gap':>10}")
        print(f"  {'-'*55}")
        print(f"  {'Instruction Score (0-3)':<25} {en_instruction:>10.2f} {hi_instruction:>10.2f} {en_instruction - hi_instruction:>+10.2f}")
        print(f"  {'Factuality Score (0-2)':<25} {en_factuality:>10.2f} {hi_factuality:>10.2f} {en_factuality - hi_factuality:>+10.2f}")
        print(f"  {'Tone Score (0-2)':<25} {en_tone:>10.2f} {hi_tone:>10.2f} {en_tone - hi_tone:>+10.2f}")
        print(f"  {'Avg Token Count':<25} {en_tokens:>10.1f} {hi_tokens:>10.1f} {hi_tokens/max(1,en_tokens):>10.1%}")

        # Calculate parity score
        total_en = en_instruction + en_factuality + en_tone
        total_hi = hi_instruction + hi_factuality + hi_tone
        parity = 1 - abs(total_en - total_hi) / 7  # Normalize to 0-1

        print(f"\n  Language Parity Score: {parity:.1%}")
        if parity >= 0.9:
            print("  ✅ Excellent cross-lingual consistency")
        elif parity >= 0.7:
            print("  ⚠️ Good consistency with some language gap")
        else:
            print("  ❌ Significant performance gap between languages")

        # Analyze Hindi-specific issues
        token_counter = MultilingualTokenCounter()
        code_mixing_total = 0
        for r in hi_model_results:
            analysis = token_counter.analyze_text(r.response_text)
            code_mixing_total += analysis.code_mixing_ratio

        avg_mixing = code_mixing_total / len(hi_model_results)
        if avg_mixing > 0.15:
            print(f"  ⚠️ Hindi responses have {avg_mixing:.0%} English mixing")

    # Save summary
    summary_path = output_dir / "crosslingual_summary.json"
    summary = {
        "models": args.models,
        "iterations": args.iterations,
        "english_prompts": len(en_prompts),
        "hindi_prompts": len(hi_prompts),
        "english_results": len(en_results),
        "hindi_results": len(hi_results),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n📁 Results saved to {output_dir}/")

    return 0


def cmd_task_fitness(args: argparse.Namespace, config: Config) -> int:
    """
    Execute the task-fitness evaluation command.

    Evaluates which model is best for which task category,
    with special focus on Hindi performance.

    Args:
        args: Parsed command line arguments.
        config: Configuration object.

    Returns:
        int: Exit code (0 for success).
    """
    logger = get_logger()
    logger.info("Starting task-fitness evaluation")

    # Apply evaluator settings from CLI args
    evaluator_provider = getattr(args, 'evaluator_provider', None)
    evaluator_model = getattr(args, 'evaluator_model', None)
    if evaluator_provider:
        config.evaluator_provider = evaluator_provider
        if not evaluator_model:
            config.evaluator_model = "llama3" if evaluator_provider == "ollama" else "gpt-4"
    if evaluator_model:
        config.evaluator_model = evaluator_model
    if evaluator_provider or evaluator_model:
        logger.info(f"Using {config.evaluator_provider} for evaluation with model {config.evaluator_model}")

    # Determine which languages to evaluate
    languages = getattr(args, 'languages', 'both')

    # Load prompt files
    en_prompts_path = Path("data/prompts/task_fitness_english.json")
    hi_prompts_path = Path("data/prompts/task_fitness_hindi.json")

    all_prompts = PromptSet()

    if languages in ("en", "both"):
        if en_prompts_path.exists():
            en_set = PromptSet.load_from_file(en_prompts_path)
            logger.info(f"Loaded {len(en_set)} English task-fitness prompts")
            for p in en_set:
                all_prompts.add(p)
        else:
            logger.warning(f"English prompts not found: {en_prompts_path}")

    if languages in ("hi", "both"):
        if hi_prompts_path.exists():
            hi_set = PromptSet.load_from_file(hi_prompts_path)
            logger.info(f"Loaded {len(hi_set)} Hindi task-fitness prompts")
            for p in hi_set:
                all_prompts.add(p)
        else:
            logger.warning(f"Hindi prompts not found: {hi_prompts_path}")

    if len(all_prompts) == 0:
        logger.error("No prompts loaded. Check data/prompts/ directory.")
        return 1

    # Filter by categories if specified
    category_filter = getattr(args, 'categories', None)
    if category_filter:
        from llm_drift_analyzer.models.prompt import PromptCategory
        filtered = PromptSet()
        for p in all_prompts:
            if p.category.value in category_filter:
                filtered.add(p)
        all_prompts = filtered
        logger.info(f"Filtered to {len(all_prompts)} prompts in categories: {category_filter}")

    logger.info(f"Total prompts: {len(all_prompts)}")

    # Initialize analyzer
    try:
        provider = getattr(args, 'provider', None)
        ollama_url = getattr(args, 'ollama_url', None)
        reasoning_effort = getattr(args, 'reasoning_effort', None)
        if reasoning_effort:
            logger.info(f"Reasoning effort for evaluator: {reasoning_effort}")
        analyzer = TaskFitnessAnalyzer(
            config,
            provider=provider,
            ollama_base_url=ollama_url,
            reasoning_effort=reasoning_effort,
        )
    except Exception as e:
        logger.error(f"Failed to initialize analyzer: {e}")
        return 1

    # Progress callback
    def progress_callback(current: int, total: int, message: str):
        percent = (current / total) * 100
        print(f"\r[{percent:5.1f}%] {message}", end="", flush=True)

    # Run analysis
    logger.info(f"Running task-fitness analysis with models: {args.models}")
    logger.info(f"Iterations per prompt-model: {args.iterations}")

    matrix = analyzer.run_fitness_analysis(
        prompts=all_prompts,
        models=args.models,
        iterations=args.iterations,
        progress_callback=progress_callback,
    )

    print()  # New line after progress

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = output_dir / "fitness_matrix.json"
    matrix.save_to_file(matrix_path)
    logger.info(f"Matrix saved to {matrix_path}")

    # Generate report
    report_gen = FitnessReportGenerator(matrix)
    report_path = output_dir / "fitness_report.md"
    report_gen.save_report(report_path, format="markdown")
    logger.info(f"Report saved to {report_path}")

    # Generate visualizations if requested
    if getattr(args, 'visualize', False):
        logger.info("Generating visualizations...")
        visualizer = FitnessVisualizer(matrix)
        charts_dir = output_dir / "charts"
        saved_charts = visualizer.save_all_plots(charts_dir)
        logger.info(f"Saved {len(saved_charts)} charts to {charts_dir}")

    # Print summary to stdout
    print("\n" + "=" * 70)
    print("Task-Fitness Evaluation Complete")
    print("=" * 70)

    for lang in matrix.languages:
        lang_label = "Hindi (हिंदी)" if lang == "hi" else "English"
        print(f"\n{'─' * 60}")
        print(f"  Recommendations — {lang_label}")
        print(f"{'─' * 60}")

        recs = matrix.get_recommendations(lang)
        if recs:
            cat_width = max(len(r["category"]) for r in recs) + 2

            print(f"  {'Task':<{cat_width}} {'Best Model':<15} {'Score':<8} {'Runner-up':<15}")
            print(f"  {'─' * (cat_width + 38)}")

            for rec in recs:
                print(
                    f"  {rec['category']:<{cat_width}} "
                    f"{rec['recommended_model']:<15} "
                    f"{rec['score']:<8.2f} "
                    f"{rec['runner_up']:<15}"
                )

        overall = matrix.get_overall_rankings(lang)
        if overall:
            print(f"\n  Overall Best ({lang_label}): {overall[0][0]} (avg: {overall[0][1]:.2f}/3.00)")

    print(f"\n📁 Results saved to {output_dir}/")
    return 0


def cmd_list_models(args: argparse.Namespace, config: Config) -> int:
    """
    Execute the list-models command.

    Args:
        args: Parsed command line arguments.
        config: Configuration object.

    Returns:
        int: Exit code (0 for success).
    """
    logger = get_logger()
    provider_filter = getattr(args, 'provider', None)
    ollama_url = getattr(args, 'ollama_url', None)

    try:
        analyzer = LLMDriftAnalyzer(
            config,
            provider=provider_filter,
            ollama_base_url=ollama_url
        )
        models = analyzer.get_available_models()

        print("\nAvailable Models by Provider:")
        print("=" * 40)

        for provider, model_list in models.items():
            if provider_filter and provider != provider_filter:
                continue
            print(f"\n{provider.upper()}:")
            for model in model_list:
                print(f"  - {model}")

        # Special note for Ollama
        if "ollama" in models:
            print("\n" + "-" * 40)
            print("Note: Ollama models shown are locally installed.")
            print("Pull more models with: ollama pull <model-name>")

    except ValueError as e:
        logger.warning(f"Could not initialize all clients: {e}")
        print("\nConfigured providers (validation status):")
        print("=" * 40)
        status = config.validate()
        for provider, available in status.items():
            if provider_filter and provider != provider_filter:
                continue
            status_str = "✓ Available" if available else "✗ Not configured"
            print(f"  {provider}: {status_str}")

    return 0


def cmd_validate(args: argparse.Namespace, config: Config) -> int:
    """
    Execute the validate command.

    Args:
        args: Parsed command line arguments.
        config: Configuration object.

    Returns:
        int: Exit code (0 for success).
    """
    print("\nConfiguration Validation")
    print("=" * 40)

    # Check API keys
    status = config.validate()
    all_valid = True

    print("\nAPI Keys:")
    for provider, available in status.items():
        if available:
            print(f"  ✓ {provider}: Configured")
        else:
            print(f"  ✗ {provider}: Not configured")
            all_valid = False

    # Test connections
    print("\nConnection Tests:")
    try:
        analyzer = LLMDriftAnalyzer(config)
        for provider, client in analyzer.clients.items():
            try:
                if client.validate_connection():
                    print(f"  ✓ {provider}: Connection successful")
                else:
                    print(f"  ✗ {provider}: Connection failed")
                    all_valid = False
            except Exception as e:
                print(f"  ✗ {provider}: {e}")
                all_valid = False
    except ValueError as e:
        print(f"  Could not test connections: {e}")
        all_valid = False

    print("\n" + "=" * 40)
    if all_valid:
        print("All validations passed!")
        return 0
    else:
        print("Some validations failed. Check your configuration.")
        return 1


def main() -> int:
    """
    Main entry point for the CLI.

    Returns:
        int: Exit code.
    """
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logger(level=log_level)

    # Load configuration
    config = Config.from_env(args.env_file)

    # Route to command handler
    commands = {
        "analyze": cmd_analyze,
        "analyze-hindi": cmd_analyze_hindi,
        "crosslingual": cmd_crosslingual,
        "report": cmd_report,
        "compare": cmd_compare,
        "list-models": cmd_list_models,
        "validate": cmd_validate,
        "task-fitness": cmd_task_fitness,
    }

    handler = commands.get(args.command)
    if handler:
        try:
            return handler(args, config)
        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user.")
            return 130
        except Exception as e:
            logger = get_logger()
            logger.error(f"Command failed: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
