#!/usr/bin/env python3
"""
Full Drift Study Example

This script demonstrates a comprehensive drift study across multiple
LLM providers and model versions, replicating the methodology from
the research paper.

Usage:
    python examples/full_drift_study.py

Note: This script makes many API calls and may incur significant costs.
      Review the configuration before running.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_drift_analyzer import (
    LLMDriftAnalyzer,
    Config,
    PromptSet,
    DriftStatisticalAnalyzer,
)
from llm_drift_analyzer.models.prompt import Prompt, PromptCategory
from llm_drift_analyzer.reporters import ReportGenerator, DriftVisualizer
from llm_drift_analyzer.utils.logger import setup_logger, get_logger


# Configuration for the study
STUDY_CONFIG = {
    "name": "Full Drift Study",
    "description": "Comprehensive drift analysis across GPT-4 and Claude models",
    "iterations": 10,  # As per research paper methodology
    "openai_models": [
        "gpt-4",
        "gpt-4-turbo",
        # "gpt-4-0613",  # Uncomment if you have access to specific versions
        # "gpt-4-0125-preview",
    ],
    "anthropic_models": [
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        # "claude-3-haiku-20240307",
    ],
    "mistral_models": [
        # "mistral-large-latest",
        # "open-mixtral-8x7b",
    ],
}


def print_study_config():
    """Print study configuration."""
    print("\nStudy Configuration:")
    print("-" * 40)
    print(f"Name: {STUDY_CONFIG['name']}")
    print(f"Iterations per model: {STUDY_CONFIG['iterations']}")
    print(f"OpenAI models: {STUDY_CONFIG['openai_models']}")
    print(f"Anthropic models: {STUDY_CONFIG['anthropic_models']}")
    print(f"Mistral models: {STUDY_CONFIG['mistral_models']}")


def estimate_api_calls(num_prompts: int) -> dict:
    """
    Estimate the number of API calls and approximate costs.

    Args:
        num_prompts: Number of prompts in the study.

    Returns:
        dict: Estimation details.
    """
    total_models = (
        len(STUDY_CONFIG['openai_models']) +
        len(STUDY_CONFIG['anthropic_models']) +
        len(STUDY_CONFIG['mistral_models'])
    )

    total_queries = num_prompts * total_models * STUDY_CONFIG['iterations']

    # Also need evaluation calls (3 evaluations per response using GPT-4)
    evaluation_calls = total_queries * 3

    # Rough cost estimation (very approximate)
    # Assuming ~500 tokens per query, ~100 tokens per evaluation
    query_cost = total_queries * 500 * 0.00003  # GPT-4 input cost
    eval_cost = evaluation_calls * 100 * 0.00003

    return {
        "total_queries": total_queries,
        "evaluation_calls": evaluation_calls,
        "total_api_calls": total_queries + evaluation_calls,
        "estimated_cost_usd": query_cost + eval_cost,
    }


def run_study(config: Config, prompts: PromptSet, output_dir: Path) -> int:
    """
    Run the full drift study.

    Args:
        config: Configuration object.
        prompts: Prompt set to use.
        output_dir: Directory for output files.

    Returns:
        int: Exit code.
    """
    logger = get_logger()

    # Collect models based on available API keys
    models = []
    status = config.validate()

    if status.get("openai"):
        models.extend(STUDY_CONFIG['openai_models'])
        logger.info(f"Including OpenAI models: {STUDY_CONFIG['openai_models']}")

    if status.get("anthropic"):
        models.extend(STUDY_CONFIG['anthropic_models'])
        logger.info(f"Including Anthropic models: {STUDY_CONFIG['anthropic_models']}")

    if status.get("mistral"):
        models.extend(STUDY_CONFIG['mistral_models'])
        logger.info(f"Including Mistral models: {STUDY_CONFIG['mistral_models']}")

    if not models:
        logger.error("No models available. Check API keys.")
        return 1

    # Initialize analyzer
    logger.info("Initializing drift analyzer...")
    analyzer = LLMDriftAnalyzer(config)

    # Progress tracking
    total_expected = len(prompts) * len(models) * STUDY_CONFIG['iterations']
    start_time = datetime.now()

    def progress_callback(current, total, message):
        elapsed = (datetime.now() - start_time).total_seconds()
        rate = current / elapsed if elapsed > 0 else 0
        remaining = (total - current) / rate if rate > 0 else 0
        print(
            f"\r[{current}/{total}] {message} "
            f"(~{remaining/60:.1f} min remaining)    ",
            end="",
            flush=True
        )

    # Run analysis
    logger.info(f"Starting analysis: {len(prompts)} prompts, {len(models)} models, "
                f"{STUDY_CONFIG['iterations']} iterations")
    print("\nRunning drift analysis...")

    results = analyzer.run_drift_analysis(
        prompts=prompts,
        models=models,
        iterations=STUDY_CONFIG['iterations'],
        progress_callback=progress_callback
    )

    print("\n")  # New line after progress

    # Save raw results
    results_path = output_dir / "full_study_results.json"
    results.save_to_file(results_path)
    logger.info(f"Results saved to {results_path}")

    # Statistical analysis
    logger.info("Performing statistical analysis...")
    stat_analyzer = DriftStatisticalAnalyzer(results)

    # Save statistical analysis
    analysis_results = {
        "study_config": STUDY_CONFIG,
        "timestamp": datetime.now().isoformat(),
        "models_analyzed": models,
        "prompts_count": len(prompts),
        "total_responses": len(results),
        "length_trends": stat_analyzer.analyze_length_trends(),
        "score_distributions": stat_analyzer.analyze_score_distributions(),
        "latency_analysis": stat_analyzer.analyze_latency_trends(),
        "change_points": stat_analyzer.detect_change_points(),
        "correlations": stat_analyzer.calculate_correlation_matrix(),
        "drift_summary": stat_analyzer.get_drift_summary(),
    }

    analysis_path = output_dir / "statistical_analysis.json"
    with open(analysis_path, 'w') as f:
        json.dump(analysis_results, f, indent=2, default=str)
    logger.info(f"Statistical analysis saved to {analysis_path}")

    # Generate reports
    logger.info("Generating reports...")
    report_gen = ReportGenerator(results)

    # Markdown report
    report_gen.save_report(
        output_dir / "drift_report.md",
        format="markdown",
        include_samples=True,
        max_samples=3
    )

    # JSON report
    report_gen.save_report(
        output_dir / "drift_report.json",
        format="json"
    )

    # Generate visualizations
    logger.info("Generating visualizations...")
    visualizer = DriftVisualizer(results)
    charts_dir = output_dir / "charts"
    visualizer.save_all_plots(charts_dir, format="png", dpi=150)

    # Also save as PDF for publication
    visualizer.save_all_plots(charts_dir / "pdf", format="pdf")

    # Print summary
    print("\n" + "=" * 60)
    print("Study Complete!")
    print("=" * 60)

    summary = stat_analyzer.get_drift_summary()
    print(f"\nModels analyzed: {len(models)}")
    print(f"Total responses: {len(results)}")
    print(f"\nKey Findings:")
    print(f"- Token count variance: {summary['drift_indicators']['token_count']['variance_percent']:.1f}%")
    print(f"- Instruction score variance: {summary['drift_indicators']['instruction_score']['variance']:.2f}")
    print(f"- Factuality score variance: {summary['drift_indicators']['factuality_score']['variance']:.2f}")
    print(f"\nRecommendation: {summary['recommendation']}")
    print(f"\nResults saved to: {output_dir}")

    return 0


def main():
    """Main entry point."""
    # Setup logging
    setup_logger(level="INFO")
    logger = get_logger()

    print("=" * 60)
    print("LLM Drift Analyzer - Full Drift Study")
    print("=" * 60)

    print_study_config()

    # Load configuration
    print("\nLoading configuration...")
    config = Config.from_env()

    # Validate API keys
    status = config.validate()
    print("\nAPI Key Status:")
    for provider, available in status.items():
        status_str = "✓" if available else "✗"
        print(f"  {status_str} {provider}")

    if not any(status.values()):
        print("\nERROR: No API keys configured.")
        print("Please set at least one of: OPENAI_API_KEY, ANTHROPIC_API_KEY, MISTRAL_API_KEY")
        return 1

    # Load prompts
    prompts_path = Path(__file__).parent.parent / "data" / "prompts" / "benchmark_prompts.json"
    if not prompts_path.exists():
        print(f"\nERROR: Prompts file not found: {prompts_path}")
        return 1

    prompts = PromptSet.load_from_file(prompts_path)
    print(f"\nLoaded {len(prompts)} benchmark prompts")

    # Estimate API usage
    estimate = estimate_api_calls(len(prompts))
    print(f"\nEstimated API Usage:")
    print(f"  - Total queries: {estimate['total_queries']}")
    print(f"  - Evaluation calls: {estimate['evaluation_calls']}")
    print(f"  - Estimated cost: ${estimate['estimated_cost_usd']:.2f}")

    # Confirm
    print("\n" + "-" * 40)
    response = input("Proceed with study? (y/N): ")
    if response.lower() != 'y':
        print("Study cancelled.")
        return 0

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent.parent / "output" / f"study_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save study configuration
    config_path = output_dir / "study_config.json"
    with open(config_path, 'w') as f:
        json.dump(STUDY_CONFIG, f, indent=2)

    # Run the study
    return run_study(config, prompts, output_dir)


if __name__ == "__main__":
    sys.exit(main())
