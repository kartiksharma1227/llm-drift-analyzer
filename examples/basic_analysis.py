#!/usr/bin/env python3
"""
Basic Analysis Example

This script demonstrates basic usage of the LLM Drift Analyzer
for comparing two GPT-4 model versions.

Usage:
    python examples/basic_analysis.py

Make sure to set up your .env file with API keys before running.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_drift_analyzer import (
    LLMDriftAnalyzer,
    Config,
    PromptSet,
    DriftStatisticalAnalyzer,
)
from llm_drift_analyzer.reporters import ReportGenerator, DriftVisualizer
from llm_drift_analyzer.utils.logger import setup_logger


def main():
    """Run basic drift analysis example."""
    # Setup logging
    setup_logger(level="INFO")

    print("=" * 60)
    print("LLM Drift Analyzer - Basic Example")
    print("=" * 60)

    # Load configuration
    print("\n1. Loading configuration...")
    config = Config.from_env()

    # Validate configuration
    status = config.validate()
    print("   API Key Status:")
    for provider, available in status.items():
        status_str = "✓" if available else "✗"
        print(f"   {status_str} {provider}")

    if not status.get("openai"):
        print("\n   ERROR: OpenAI API key required for this example")
        print("   Please set OPENAI_API_KEY in your .env file")
        return 1

    # Load prompts
    print("\n2. Loading benchmark prompts...")
    prompts_path = Path(__file__).parent.parent / "data" / "prompts" / "benchmark_prompts.json"

    if not prompts_path.exists():
        print(f"   ERROR: Prompts file not found: {prompts_path}")
        return 1

    prompts = PromptSet.load_from_file(prompts_path)
    print(f"   Loaded {len(prompts)} prompts")

    # For this example, we'll use a subset of prompts
    # to reduce API costs and time
    subset_prompts = PromptSet(
        prompts=prompts.prompts[:3],  # First 3 prompts
        name="Example Subset"
    )
    print(f"   Using subset of {len(subset_prompts)} prompts for demo")

    # Initialize analyzer
    print("\n3. Initializing analyzer...")
    analyzer = LLMDriftAnalyzer(config)
    print("   Analyzer ready")

    # Define models to compare
    models = [
        "gpt-4",  # Current default GPT-4
        "gpt-3.5-turbo",  # GPT-3.5 for comparison
    ]
    print(f"\n4. Models to analyze: {models}")

    # Run analysis with reduced iterations for demo
    iterations = 3
    print(f"\n5. Running drift analysis ({iterations} iterations per model)...")
    print("   This may take a few minutes...\n")

    def progress_callback(current, total, message):
        percent = (current / total) * 100
        print(f"   [{percent:5.1f}%] {message}")

    results = analyzer.run_drift_analysis(
        prompts=subset_prompts,
        models=models,
        iterations=iterations,
        progress_callback=progress_callback
    )

    print(f"\n   Completed: {len(results)} responses collected")

    # Statistical analysis
    print("\n6. Performing statistical analysis...")
    stat_analyzer = DriftStatisticalAnalyzer(results)

    # Print summary
    summary = stat_analyzer.get_drift_summary()
    print("\n   Drift Summary:")
    print(f"   - Token count variance: {summary['drift_indicators']['token_count']['variance_percent']:.1f}%")
    print(f"   - Instruction score variance: {summary['drift_indicators']['instruction_score']['variance']:.2f}")

    # Generate report
    print("\n7. Generating report...")
    output_dir = Path(__file__).parent.parent / "output" / "example_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save results
    results.save_to_file(output_dir / "results.json")
    print(f"   Saved results to {output_dir / 'results.json'}")

    # Generate markdown report
    report_gen = ReportGenerator(results)
    report_gen.save_report(output_dir / "report.md", format="markdown")
    print(f"   Saved report to {output_dir / 'report.md'}")

    # Generate visualizations
    print("\n8. Generating visualizations...")
    visualizer = DriftVisualizer(results)
    charts_dir = output_dir / "charts"
    saved_charts = visualizer.save_all_plots(charts_dir)
    print(f"   Saved {len(saved_charts)} charts to {charts_dir}")

    # Print final summary
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")
    print("\nKey findings:")
    print(report_gen.generate_text_summary())

    return 0


if __name__ == "__main__":
    sys.exit(main())
