# LLM Behavioral Drift Analyzer

A comprehensive Python framework for tracking behavioral drift in Large Language Models (LLMs), measuring changes in instruction-following, factuality, tone, and verbosity over time.

Based on the research paper: *"Tracking Behavioral Drift in Large Language Models: A Comprehensive Framework for Monitoring Instruction-Following, Factuality, and Tone Variance Over Time"*

## Features

- **Multi-Provider Support**: Analyze models from OpenAI (GPT-4), Anthropic (Claude), Mistral (Mixtral), and **Ollama (local/offline models)**
- **Offline/Local Model Support**: Run drift analysis on open-source models via Ollama (Llama 3, Mistral, CodeLlama, Phi, Gemma, etc.) without internet or API keys
- **Comprehensive Metrics**: Evaluate instruction adherence (0-3), factuality (0-2), tone (0-2), token counts, and latency
- **Statistical Analysis**: ANOVA, pairwise t-tests, Cohen's d effect sizes, CUSUM change point detection
- **Visualization**: Publication-ready charts using matplotlib and seaborn
- **Extensible Architecture**: Easy to add new providers, evaluators, and metrics
- **CLI Interface**: Full command-line interface for running analyses
- **Detailed Reports**: Markdown, JSON, and text report generation

## Installation

### Prerequisites

- Python 3.9 or higher
- API keys for at least one LLM provider (OpenAI required for evaluation)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/llm-drift-analyzer.git
   cd llm-drift-analyzer
   ```

2. **Create and activate a virtual environment**
   ```bash
   # Create virtual environment
   python -m venv venv

   # Activate on macOS/Linux
   source venv/bin/activate

   # Activate on Windows
   .\venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API keys**
   ```bash
   # Copy the example environment file
   cp .env.example .env

   # Edit .env and add your API keys
   # Required: OPENAI_API_KEY (for GPT-4 evaluation)
   # Optional: ANTHROPIC_API_KEY, MISTRAL_API_KEY
   ```

5. **Install the package (optional)**
   ```bash
   pip install -e .
   ```

### Setting Up Ollama (For Local/Offline Models)

If you want to analyze open-source models locally without API costs:

1. **Install Ollama**
   ```bash
   # macOS
   brew install ollama

   # Linux
   curl -fsSL https://ollama.ai/install.sh | sh

   # Or download from https://ollama.ai/download
   ```

2. **Start Ollama server**
   ```bash
   ollama serve
   ```

3. **Pull models you want to analyze**
   ```bash
   ollama pull llama3
   ollama pull mistral
   ollama pull codellama
   ollama pull phi3
   ```

4. **Verify installation**
   ```bash
   ollama list
   ```

## Quick Start

### Using the CLI

```bash
# Run drift analysis comparing GPT-4 versions (requires API key)
python main.py analyze --models gpt-4 gpt-3.5-turbo --iterations 5

# Run drift analysis on local Ollama models (no API key needed)
python main.py analyze --provider ollama --models llama3 mistral --iterations 5

# Compare local models over time
python main.py analyze --provider ollama --models llama3:8b llama3:70b phi3 --iterations 10

# Generate report from existing results
python main.py report --input output/results.json --format markdown --visualize

# Compare two specific models
python main.py compare --model1 gpt-4-0613 --model2 gpt-4-0125-preview

# List available models (both cloud and local)
python main.py list-models
python main.py list-models --provider ollama  # List only local models

# Validate configuration
python main.py validate
```

### Using as a Library

```python
from llm_drift_analyzer import LLMDriftAnalyzer, Config, PromptSet
from llm_drift_analyzer.analyzers import DriftStatisticalAnalyzer
from llm_drift_analyzer.reporters import ReportGenerator, DriftVisualizer

# Load configuration
config = Config.from_env()

# Initialize analyzer
analyzer = LLMDriftAnalyzer(config)

# Load prompts
prompts = PromptSet.load_from_file("data/prompts/benchmark_prompts.json")

# Run drift analysis on cloud models
results = analyzer.run_drift_analysis(
    prompts=prompts,
    models=["gpt-4", "gpt-3.5-turbo"],
    iterations=10
)

# Save results
results.save_to_file("output/results.json")

# Statistical analysis
stat_analyzer = DriftStatisticalAnalyzer(results)
print(stat_analyzer.generate_drift_report())

# Generate visualizations
visualizer = DriftVisualizer(results)
visualizer.save_all_plots("output/charts/")
```

### Using Local Models (Ollama)

```python
from llm_drift_analyzer.clients import OllamaClient, LLMClientFactory

# Create Ollama client (no API key needed)
client = OllamaClient(base_url="http://localhost:11434")

# List locally available models
print(client.list_models())  # ['llama3', 'mistral', 'codellama', ...]

# Query a local model
result = client.query(
    prompt="Explain quantum computing in simple terms",
    model="llama3",
    system_message="You are a helpful teacher."
)
print(result.response_text)

# Use with the factory pattern
client = LLMClientFactory.create("ollama", api_key="")  # Empty string OK for Ollama

# Run full drift analysis on local models
results = analyzer.run_drift_analysis(
    prompts=prompts,
    models=["llama3", "mistral", "phi3"],
    provider="ollama",
    iterations=5
)
```

## Project Structure

```
llm_drift_analyzer/
├── main.py                           # CLI entry point
├── requirements.txt                  # Python dependencies
├── setup.py                          # Package installation
├── .env.example                      # Environment template
│
├── data/
│   └── prompts/
│       └── benchmark_prompts.json    # 15 benchmark prompts
│
├── src/llm_drift_analyzer/
│   ├── __init__.py
│   ├── models/                       # Data models
│   │   ├── prompt.py                 # Prompt and PromptSet classes
│   │   └── response_analysis.py      # ResponseAnalysis class
│   ├── clients/                      # LLM API clients
│   │   ├── base_client.py            # Abstract base class
│   │   ├── openai_client.py          # OpenAI implementation
│   │   ├── anthropic_client.py       # Anthropic implementation
│   │   ├── mistral_client.py         # Mistral implementation
│   │   └── ollama_client.py          # Ollama (local models) implementation
│   ├── evaluators/                   # Response evaluators
│   │   ├── base_evaluator.py         # Abstract base class
│   │   ├── instruction_evaluator.py  # Instruction adherence
│   │   ├── factuality_evaluator.py   # Factual accuracy
│   │   └── tone_evaluator.py         # Tone/style
│   ├── analyzers/                    # Analysis engines
│   │   ├── drift_analyzer.py         # Main analyzer
│   │   └── statistical_analyzer.py   # Statistical analysis
│   ├── reporters/                    # Report generation
│   │   ├── report_generator.py       # Text reports
│   │   └── visualizer.py             # Charts
│   └── utils/                        # Utilities
│       ├── config.py                 # Configuration
│       ├── logger.py                 # Logging
│       └── tokenizer.py              # Token counting
│
├── tests/                            # Test suite
│   ├── conftest.py                   # Fixtures
│   ├── test_models.py
│   ├── test_clients.py
│   ├── test_evaluators.py
│   └── test_analyzers.py
│
└── examples/                         # Example scripts
    ├── basic_analysis.py
    └── full_drift_study.py
```

## Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes* | OpenAI API key (required for cloud evaluation) |
| `ANTHROPIC_API_KEY` | No | Anthropic API key for Claude models |
| `MISTRAL_API_KEY` | No | Mistral API key for Mixtral models |
| `OLLAMA_BASE_URL` | No | Ollama server URL (default: http://localhost:11434) |
| `EVALUATOR_MODEL` | No | Model for evaluation (default: gpt-4, can use llama3 for offline) |
| `LOG_LEVEL` | No | Logging level (default: INFO) |
| `OUTPUT_DIR` | No | Default output directory (default: output) |

*OpenAI API key is required for cloud evaluation. For fully offline analysis, set `EVALUATOR_MODEL=llama3` (or another local model) and use only Ollama models.

### API Configuration

Default API parameters can be customized:

```python
from llm_drift_analyzer.utils.config import Config, APIConfig

config = Config(
    openai_api_key="sk-...",
    api_config=APIConfig(
        temperature=0.1,      # Low for consistency
        max_tokens=1000,
        top_p=0.9,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
)
```

## Evaluation Metrics

### Instruction Adherence (0-3)
- **3**: Perfect adherence to all instructions
- **2**: Good adherence with minor deviations
- **1**: Poor adherence, missing key requirements
- **0**: No adherence to instructions

### Factuality (0-2)
- **2**: Completely factual
- **1**: Mostly factual with minor errors
- **0**: Contains significant factual errors

### Tone/Style (0-2)
- **2**: Appropriate and consistent tone
- **1**: Adequate tone with some inconsistencies
- **0**: Inappropriate tone for context

## Statistical Methods

The analyzer uses several statistical methods:

1. **ANOVA**: Tests for significant differences between model groups
2. **Pairwise t-tests**: Compares specific model pairs
3. **Cohen's d**: Measures effect size of differences
4. **CUSUM**: Detects change points in time series data
5. **Correlation Analysis**: Identifies relationships between metrics

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/llm_drift_analyzer --cov-report=html

# Run specific test file
pytest tests/test_models.py -v

# Run tests matching a pattern
pytest tests/ -k "test_analyze" -v
```

## Examples

### Basic Analysis

```bash
# Run the basic example
python examples/basic_analysis.py
```

This runs a quick analysis comparing GPT-4 and GPT-3.5-turbo on a subset of prompts.

### Full Drift Study

```bash
# Run the comprehensive study
python examples/full_drift_study.py
```

This replicates the methodology from the research paper with full benchmark prompts.

## Output

The analyzer generates several output files:

- `results.json`: Raw analysis results
- `drift_report.md`: Comprehensive markdown report
- `statistical_analysis.json`: Detailed statistical data
- `charts/`: Directory of visualization images
  - `token_distribution.png`
  - `score_comparison.png`
  - `latency_distribution.png`
  - `drift_timeline.png`
  - `change_points.png`
  - `correlation_heatmap.png`
  - `model_radar.png`

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest tests/`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this tool in your research, please cite:

```bibtex
@article{llm-drift-analyzer,
  title={Tracking Behavioral Drift in Large Language Models: A Comprehensive Framework for Monitoring Instruction-Following, Factuality, and Tone Variance Over Time},
  author={...},
  year={2024}
}
```

## Acknowledgments

- Based on research into LLM behavioral drift patterns
- Uses OpenAI's GPT-4 for automated evaluation
- Statistical methods adapted from standard practices in behavioral research
