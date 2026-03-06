# LLM Behavioral Drift Analyzer

A comprehensive Python framework for tracking behavioral drift in Large Language Models (LLMs) and evaluating which open-source model is best suited for which task — with a special focus on Hindi for government and public-sector use cases.

Based on the research paper: _"Tracking Behavioral Drift in Large Language Models: A Comprehensive Framework for Monitoring Instruction-Following, Factuality, and Tone Variance Over Time"_

## Features

- **Multi-Provider Support**: Analyze models from OpenAI (GPT-4), Anthropic (Claude), Mistral (Mixtral), and **Ollama (local/offline models)**
- **Offline/Local Model Support**: Run drift analysis on open-source models via Ollama (Llama 3, Mistral, CodeLlama, Phi, Gemma, etc.) without internet or API keys
- **Free Local Evaluation**: Use Ollama models as judges instead of GPT-4 - completely free, no API costs
- **Comprehensive Metrics**: Evaluate instruction adherence (0-3), factuality (0-2), tone (0-2), token counts, and latency
- **Statistical Analysis**: ANOVA, pairwise t-tests, Cohen's d effect sizes, CUSUM change point detection
- **Visualization**: Publication-ready charts using matplotlib and seaborn
- **Extensible Architecture**: Easy to add new providers, evaluators, and metrics
- **CLI Interface**: Full command-line interface for running analyses
- **Detailed Reports**: Markdown, JSON, and text report generation
- **Task-Fitness Evaluation**: Determine which open-source model is best for which task (summarization, translation, legal/admin, etc.) — especially in Hindi
- **Government Use-Case Benchmarks**: 100 prompts (50 Hindi + 50 English) covering RTI, GST, MGNREGA, PIL, Aadhaar, and more

## Installation

### Prerequisites

- Python 3.9 or higher
- **For cloud providers**: API keys (OpenAI, Anthropic, or Mistral)
- **For fully offline usage**: Ollama installed locally (no API keys needed)

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

# FULLY OFFLINE: Query and evaluate with Ollama (completely free!)
python main.py analyze --provider ollama --models llama3 mistral \
    --evaluator-provider ollama --evaluator-model llama3 --iterations 5

# Use local evaluation with cloud models (saves on GPT-4 evaluation costs)
python main.py analyze --models gpt-4 gpt-3.5-turbo \
    --evaluator-provider ollama --evaluator-model llama3 --iterations 5

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

# Hindi language analysis with Ollama (completely free!)
python main.py analyze-hindi \
    --models qwen2:1.5b gemma:2b \
    --provider ollama \
    --evaluator-provider ollama \
    --evaluator-model llama3 \
    --iterations 5

# Cross-lingual comparison (English vs Hindi) with Ollama
python main.py crosslingual \
    --models qwen2:1.5b gemma:2b \
    --provider ollama \
    --evaluator-provider ollama \
    --evaluator-model llama3 \
    --iterations 5

# Generate charts from analysis results (IMPORTANT: 2-step process)
# Step 1: Run analysis (saves results.json)
python main.py analyze-hindi --models qwen2:1.5b --provider ollama --iterations 5
# Step 2: Generate report with visualizations
python main.py report --input output/hindi/hindi_results.json --visualize

# ── Task-Fitness Evaluation ──────────────────────────────────────────────────

# Find the best model for each task in Hindi (fully offline, free)
python main.py task-fitness \
    --models qwen2:1.5b tinyllama \
    --provider ollama \
    --iterations 3 \
    --languages hi \
    --evaluator-provider ollama \
    --evaluator-model qwen2:1.5b \
    --visualize \
    --output output/task_fitness_hindi

# Both Hindi and English
python main.py task-fitness \
    --models llama3 mistral qwen2 \
    --provider ollama \
    --iterations 3 \
    --languages both \
    --evaluator-provider ollama \
    --evaluator-model llama3 \
    --visualize \
    --output output/task_fitness

# Filter to specific task categories only
python main.py task-fitness \
    --models qwen2:1.5b tinyllama \
    --provider ollama \
    --iterations 2 \
    --languages hi \
    --categories summarization legal_administrative translation \
    --evaluator-provider ollama \
    --evaluator-model qwen2:1.5b \
    --output output/task_fitness_gov
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

## Expanded Benchmark Datasets

The project now includes **expanded benchmark datasets** with 50 prompts each for English and Hindi, providing more comprehensive evaluation across all categories:

### What's in the Expanded Datasets?

| Dataset | Prompts | Categories | Key Features |
|---------|---------|------------|--------------|
| **english_benchmark_expanded.json** | 90 | IF: 50, FQ: 40, CR: 0 | Google IFEval (verifiable constraints), TruthfulQA (misconception testing) |
| **hindi_benchmark_expanded.json** | 100 | IF: 10, FQ: 60, CR: 30 | AI4Bharat IndicMSMARCO (QA), XNLI Hindi (reasoning), manual curation |

### Dataset Features

**English Dataset (90 prompts):**
- **Instruction Following (50)**: Google IFEval with verifiable constraints (word counts, format requirements, structural rules)
  - Examples: "Write exactly 100 words", "Use bullet points", "No introduction/conclusion"
  - Enables precise, reproducible evaluation
- **Factual QA (40)**: TruthfulQA prompts that test common misconceptions
  - Examples: "Is the Great Wall visible from space?", "Do we only use 10% of our brain?"
  - Includes reference answers for automated factuality checking

**Hindi Dataset (100 prompts):**
- **Factual QA (60)**: AI4Bharat IndicMSMARCO - real Hindi question-answering dataset
  - Context-based questions in natural Hindi
  - Diverse topics: general knowledge, technology, health, culture
- **Creative Reasoning (30)**: XNLI Hindi - natural language inference
  - Premise-hypothesis pairs for logical reasoning
  - Tests entailment, contradiction, and neutrality understanding
- **Instruction Following (10)**: Manually curated natural Hindi prompts
  - Word limits, format constraints, structured outputs
  - Conversational Hindi style (not translated English)

### Using Expanded Datasets

**CLI:**

```bash
# English expanded dataset (90 prompts: IFEval + TruthfulQA)
python main.py analyze --models gpt-4 --use-expanded --language en --iterations 5

# Hindi expanded dataset (100 prompts: IndicMSMARCO + XNLI + manual)
python main.py analyze-hindi --models qwen2:1.5b --use-expanded \
    --provider ollama --evaluator-provider ollama --evaluator-model llama3 --iterations 5

# Or specify the file directly
python main.py analyze --prompts data/prompts/english_benchmark_expanded.json \
    --models gpt-4 gpt-3.5-turbo --iterations 5

# Compare models with expanded datasets
python main.py analyze --prompts data/prompts/hindi_benchmark_expanded.json \
    --models qwen2:1.5b gemma:2b llama3:8b --provider ollama --iterations 5
```

**Python API:**

```python
from llm_drift_analyzer import LLMDriftAnalyzer, Config, PromptSet

# Load expanded English dataset
prompts = PromptSet.load_from_file("data/prompts/english_benchmark_expanded.json")
print(f"Loaded {len(prompts)} prompts")  # 50 prompts

# Load expanded Hindi dataset
hindi_prompts = PromptSet.load_from_file("data/prompts/hindi_benchmark_expanded.json")

# Run analysis with expanded prompts
config = Config.from_env()
analyzer = LLMDriftAnalyzer(config)
results = analyzer.run_drift_analysis(
    prompts=prompts,
    models=["gpt-4", "gpt-3.5-turbo"],
    iterations=10
)
```

### Comparison: Standard vs Expanded

| Feature | Standard | Expanded English | Expanded Hindi |
|---------|----------|------------------|----------------|
| Total prompts | 15 | 90 | 100 |
| Instruction Following | 5 | 50 (IFEval) | 10 (manual) |
| Factual QA | 5 | 40 (TruthfulQA) | 60 (IndicMSMARCO) |
| Creative Reasoning | 5 | 0 | 30 (XNLI) |
| Verifiable constraints | Basic | Advanced | Moderate |
| Misconception testing | Limited | Comprehensive | Moderate |
| Real-world datasets | No | Yes (Google, research) | Yes (AI4Bharat) |
| Analysis time | ~5-10 min | ~20-40 min | ~25-45 min |
| **Recommended for** | Quick testing | Production (English) | Production (Hindi) |

### Creating Custom Datasets

You can also download and integrate datasets from HuggingFace:

```bash
# Download datasets and create custom benchmarks
python scripts/download_datasets.py --language hi --max-samples 50

# This script supports:
# - AI4Bharat IndicQA (Hindi QA)
# - XNLI Hindi (reasoning/inference)
# - Google IFEval (English instruction-following)
# - TruthfulQA (English factuality)
```

### Using Ollama for Evaluation (Free, No API Costs)

By default, the analyzer uses GPT-4 to evaluate/score LLM responses, which costs money. You can use Ollama models as judges instead for completely free evaluation:

```python
from llm_drift_analyzer import LLMDriftAnalyzer, Config, PromptSet

# Configure to use Ollama for BOTH querying and evaluation
config = Config(
    evaluator_provider="ollama",  # Use local model for scoring
    evaluator_model="llama3",     # Which local model to use as judge
    ollama_base_url="http://localhost:11434"
)

# Initialize analyzer with Ollama
analyzer = LLMDriftAnalyzer(config, provider="ollama")

# Load prompts
prompts = PromptSet.load_from_file("data/prompts/benchmark_prompts.json")

# Run analysis - completely free, no API calls!
results = analyzer.run_drift_analysis(
    prompts=prompts,
    models=["llama3", "mistral", "phi3"],
    iterations=5
)

# This runs entirely locally:
# - Queries are sent to local Ollama models
# - Evaluation/scoring uses local Ollama models
# - No internet required, no API costs
```

**CLI Example for Fully Offline Analysis:**

```bash
# Fully offline: both querying and evaluation via Ollama
python main.py analyze \
    --provider ollama \
    --models llama3 mistral phi3 \
    --evaluator-provider ollama \
    --evaluator-model llama3 \
    --iterations 5
```

**Hybrid Mode: Cloud Models + Local Evaluation**

You can also query cloud models but use Ollama for evaluation to save on GPT-4 costs:

```bash
# Query GPT-4, but evaluate locally (saves money on evaluation)
python main.py analyze \
    --models gpt-4 gpt-3.5-turbo \
    --evaluator-provider ollama \
    --evaluator-model llama3 \
    --iterations 10
```

| Configuration        | Query Cost | Evaluation Cost | Use Case                        |
| -------------------- | ---------- | --------------- | ------------------------------- |
| Cloud + GPT-4 eval   | $$$        | $$$             | Highest quality                 |
| Cloud + Ollama eval  | $$$        | Free            | Save on evaluation              |
| Ollama + GPT-4 eval  | Free       | $$$             | Quality scoring of local models |
| Ollama + Ollama eval | Free       | Free            | Fully offline/free              |

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
│       ├── benchmark_prompts.json             # 15 English benchmark prompts
│       ├── hindi_benchmark_prompts.json       # 15 Hindi benchmark prompts (parallel)
│       ├── english_benchmark_expanded.json    # 90 English prompts (IFEval + TruthfulQA)
│       ├── hindi_benchmark_expanded.json      # 100 Hindi prompts (IndicMSMARCO + XNLI + manual)
│       ├── task_fitness_english.json          # 50 English task-fitness prompts (government-focused)
│       └── task_fitness_hindi.json            # 50 Hindi task-fitness prompts (RTI, GST, MGNREGA, etc.)
│
├── src/llm_drift_analyzer/
│   ├── __init__.py
│   ├── models/                       # Data models
│   │   ├── prompt.py                 # Prompt and PromptSet classes (10 categories)
│   │   ├── response_analysis.py      # ResponseAnalysis class
│   │   ├── multilingual_analysis.py  # Cross-lingual comparison models
│   │   └── task_fitness.py           # TaskFitnessScore and TaskFitnessMatrix dataclasses
│   ├── clients/                      # LLM API clients
│   │   ├── base_client.py            # Abstract base class
│   │   ├── openai_client.py          # OpenAI implementation
│   │   ├── anthropic_client.py       # Anthropic implementation
│   │   ├── mistral_client.py         # Mistral implementation
│   │   └── ollama_client.py          # Ollama (local models) implementation
│   ├── evaluators/                   # Response evaluators
│   │   ├── base_evaluator.py         # Abstract base class
│   │   ├── instruction_evaluator.py  # Instruction adherence (0-3)
│   │   ├── factuality_evaluator.py   # Factual accuracy (0-2)
│   │   ├── tone_evaluator.py         # Tone/style (0-2)
│   │   ├── multilingual_evaluator.py # Hindi-aware evaluators
│   │   ├── summarization_evaluator.py    # Summarization quality (0-3)
│   │   ├── translation_evaluator.py      # Translation quality (0-3)
│   │   ├── logical_math_evaluator.py     # Logical/math reasoning (0-3)
│   │   ├── conversational_evaluator.py   # Conversational quality (0-3)
│   │   ├── legal_admin_evaluator.py      # Legal/administrative quality (0-3)
│   │   ├── sentiment_evaluator.py        # Sentiment analysis quality (0-3)
│   │   └── code_generation_evaluator.py  # Code generation quality (0-3)
│   ├── analyzers/                    # Analysis engines
│   │   ├── drift_analyzer.py         # Main drift analyzer
│   │   ├── statistical_analyzer.py   # Statistical analysis
│   │   ├── crosslingual_analyzer.py  # English vs Hindi comparison
│   │   └── task_fitness_analyzer.py  # Task-fitness orchestrator (CATEGORY_EVALUATOR_MAP)
│   ├── reporters/                    # Report generation
│   │   ├── report_generator.py       # Drift text reports
│   │   ├── visualizer.py             # Drift charts
│   │   ├── fitness_report_generator.py  # Task-fitness markdown report
│   │   └── fitness_visualizer.py        # Heatmap, radar, bar, comparison charts
│   └── utils/                        # Utilities
│       ├── config.py                 # Configuration
│       ├── logger.py                 # Logging
│       ├── tokenizer.py              # Token counting
│       └── multilingual_tokenizer.py # Hindi/Devanagari tokenization
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

| Variable             | Required | Description                                                         |
| -------------------- | -------- | ------------------------------------------------------------------- |
| `OPENAI_API_KEY`     | No\*     | OpenAI API key (for GPT models and cloud evaluation)                |
| `ANTHROPIC_API_KEY`  | No       | Anthropic API key for Claude models                                 |
| `MISTRAL_API_KEY`    | No       | Mistral API key for Mixtral models                                  |
| `OLLAMA_BASE_URL`    | No       | Ollama server URL (default: http://localhost:11434)                 |
| `EVALUATOR_PROVIDER` | No       | Provider for evaluation: "openai" or "ollama" (default: openai)     |
| `EVALUATOR_MODEL`    | No       | Model for evaluation (default: gpt-4 for OpenAI, llama3 for Ollama) |
| `LOG_LEVEL`          | No       | Logging level (default: INFO)                                       |
| `OUTPUT_DIR`         | No       | Default output directory (default: output)                          |

\*For fully offline analysis, set `EVALUATOR_PROVIDER=ollama` and use only Ollama models - no API keys needed!

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

### Behavioral Drift Metrics

| Metric | Range | Description |
|--------|-------|-------------|
| Instruction Adherence | 0-3 | How well the model follows explicit instructions |
| Factuality | 0-2 | Accuracy of factual claims |
| Tone/Style | 0-2 | Appropriateness and consistency of tone |

### Task-Fitness Metrics (all 0-3 for cross-task comparison)

| Metric | Description |
|--------|-------------|
| `summarization_quality` | Key point capture, conciseness, no hallucination |
| `translation_quality` | Meaning accuracy, natural Hindi phrasing, grammar |
| `logical_mathematical_reasoning` | Correct answer + valid reasoning chain |
| `conversational_quality` | Helpfulness, natural tone, empathy |
| `legal_administrative_quality` | Correct terminology, formal structure, precision |
| `sentiment_analysis_quality` | Correct sentiment, nuance, sarcasm detection |
| `code_generation_quality` | Syntax correctness, logic, best practices |

**Scoring scale (0-3) for all task-fitness evaluators:**

- **3**: Excellent — fully correct, natural, well-structured
- **2**: Good — mostly correct with minor gaps
- **1**: Partial — incomplete or some errors
- **0**: Poor — incorrect or missing response

## Hindi/Multilingual Analysis

The analyzer includes comprehensive support for Hindi language analysis, enabling cross-lingual drift studies comparing English and Hindi model performance.

### Features

- **Devanagari-Aware Tokenization**: Syllable counting based on Hindi phonology (matras, halant, independent vowels)
- **Code-Mixing Detection**: Identifies Hindi-English mixed text (Hinglish) patterns
- **Script Consistency Scoring**: Measures how consistently a model uses the expected script
- **Cross-Lingual Comparison**: Statistical comparison of model performance across languages
- **Language Parity Index**: Quantifies the performance gap between English and Hindi
- **Natural Hindi Prompts**: Conversational-style prompts (not literal translations)

### Hindi-Specific Metrics

| Metric                  | Description                                                |
| ----------------------- | ---------------------------------------------------------- |
| `devanagari_char_count` | Number of Devanagari script characters                     |
| `syllable_count`        | Hindi syllables based on phonological rules                |
| `code_mixing_ratio`     | Proportion of non-Hindi text (0.0 to 1.0)                  |
| `script_consistency`    | Score indicating adherence to expected script (0.0 to 1.0) |
| `hindi_naturalness`     | How natural/conversational the Hindi sounds (0-2)          |

### CLI Commands

```bash
# Run analysis with Hindi prompts (cloud models)
python main.py analyze-hindi --models gpt-4 claude-3-opus --iterations 5

# Run analysis with Hindi prompts (Ollama - completely free!)
python main.py analyze-hindi \
    --models qwen2:1.5b gemma:2b phi3 \
    --provider ollama \
    --evaluator-provider ollama \
    --evaluator-model llama3 \
    --iterations 5

# Cross-lingual analysis (English vs Hindi comparison)
python main.py crosslingual --models gpt-4 --iterations 10

# Cross-lingual with Ollama (free, local)
python main.py crosslingual \
    --models qwen2:1.5b gemma:2b \
    --provider ollama \
    --evaluator-provider ollama \
    --evaluator-model llama3 \
    --iterations 5

# Analyze both languages
python main.py analyze --language all --models llama3 --provider ollama

# Specify language for standard analysis
python main.py analyze --language hi --models gpt-4 --iterations 5

# Generate visualizations from Hindi results (2-step workflow)
# Step 1: Run the analysis
python main.py analyze-hindi \
    --models qwen2:1.5b \
    --provider ollama \
    --evaluator-provider ollama \
    --evaluator-model llama3 \
    --iterations 5

# Step 2: Generate report with charts
python main.py report \
    --input output/hindi/hindi_results.json \
    --format markdown \
    --visualize \
    --output output/hindi/hindi_report.md
```

### Using Hindi Analysis in Python

```python
from llm_drift_analyzer import LLMDriftAnalyzer, Config, PromptSet
from llm_drift_analyzer.analyzers import CrossLingualAnalyzer
from llm_drift_analyzer.utils import MultilingualTokenCounter, count_hindi_tokens
from llm_drift_analyzer.evaluators import (
    MultilingualInstructionEvaluator,
    HindiNaturalnessEvaluator,
    ScriptConsistencyEvaluator
)

# Load Hindi prompts
hindi_prompts = PromptSet.load_from_file("data/prompts/hindi_benchmark_prompts.json")

# Filter prompts by language
hindi_only = hindi_prompts.filter_by_language("hi")

# Run Hindi-specific analysis
config = Config.from_env()
analyzer = LLMDriftAnalyzer(config)
results = analyzer.run_drift_analysis(
    prompts=hindi_only,
    models=["gpt-4", "llama3"],
    iterations=5
)

# Analyze Hindi text
tokenizer = MultilingualTokenCounter()
analysis = tokenizer.analyze_text("यह एक example है")
print(f"Script type: {analysis.primary_script}")
print(f"Code-mixing ratio: {analysis.code_mixing_ratio}")
print(f"Syllable count: {analysis.syllable_count}")

# Use Hindi-specific evaluators
naturalness_evaluator = HindiNaturalnessEvaluator(api_key=config.openai_api_key)
score = naturalness_evaluator.evaluate(
    prompt="सोलर एनर्जी के फायदे बताओ",
    response="सौर ऊर्जा पर्यावरण के लिए अच्छी है..."
)
```

### Cross-Lingual Comparison

Compare how models perform on equivalent prompts in English vs Hindi:

```python
from llm_drift_analyzer.analyzers import CrossLingualAnalyzer
from llm_drift_analyzer.models import PromptSet

# Load parallel prompts (English and Hindi versions of same prompts)
english_prompts = PromptSet.load_from_file("data/prompts/benchmark_prompts.json")
hindi_prompts = PromptSet.load_from_file("data/prompts/hindi_benchmark_prompts.json")

# Get parallel prompt pairs
pairs = english_prompts.get_parallel_pairs(hindi_prompts)
# Returns: [("IF-001", "HI-IF-001"), ("IF-002", "HI-IF-002"), ...]

# Run cross-lingual analysis
crosslingual = CrossLingualAnalyzer(config)
report = crosslingual.generate_crosslingual_report(
    english_results=english_results,
    hindi_results=hindi_results,
    parallel_mapping=dict(pairs)
)

# Access language parity metrics
print(f"Instruction following parity: {report.language_parity_index['instruction_score']}")
print(f"Performance gap: {report.overall_performance_gap}")

# Statistical comparison
for test in report.statistical_tests:
    print(f"{test.metric}: p-value={test.p_value:.4f}, effect_size={test.effect_size:.3f}")
```

### Hindi Prompt Guidelines

The included Hindi prompts follow natural, conversational patterns:

| Style                  | Example                                        |
| ---------------------- | ---------------------------------------------- |
| ✅ Natural             | "इस बारे में आपका क्या विचार है?"              |
| ❌ Literal             | "इस प्रश्न का उत्तर दीजिए"                     |
| ✅ Conversational      | "सोलर एनर्जी के तीन मुख्य फायदे बताओ"          |
| ❌ Formal/Sanskritized | "सौर ऊर्जा के त्रि-मुख्य लाभों का वर्णन कीजिए" |

The Hindi benchmark prompts (`data/prompts/hindi_benchmark_prompts.json`) are written in the natural style that Hindi speakers actually use, including common English loanwords where appropriate.

### Complete Hindi Analysis Workflow (Ollama)

Here's the complete workflow for analyzing Hindi prompts using only open-source models:

**Step 1: Install and prepare Ollama models**

```bash
# Install Ollama (if not already installed)
brew install ollama  # macOS
# or visit https://ollama.ai/download

# Start Ollama server
ollama serve

# Pull models for testing
ollama pull qwen2:1.5b   # Small, good for Hindi
ollama pull gemma:2b     # Google's model
ollama pull phi3         # Microsoft's model

# Pull model for evaluation/judging
ollama pull llama3       # Used as judge
```

**Step 2: Run Hindi analysis**

```bash
# Analyze Hindi prompts (saves to output/hindi/hindi_results.json)
python main.py analyze-hindi \
    --models qwen2:1.5b gemma:2b \
    --provider ollama \
    --evaluator-provider ollama \
    --evaluator-model llama3 \
    --iterations 5 \
    --output output/hindi
```

**Step 3: Generate visualizations and report**

```bash
# Generate markdown report with charts
python main.py report \
    --input output/hindi/hindi_results.json \
    --format markdown \
    --visualize \
    --include-samples \
    --output output/hindi/hindi_report.md
```

**Output files created:**

- `output/hindi/hindi_results.json` - Raw analysis data
- `output/hindi/hindi_report.md` - Detailed report
- `output/hindi/charts/` - Visualization images
  - `token_distribution.png`
  - `score_comparison.png`
  - `latency_distribution.png`
  - etc.

**Important Notes:**

- ⚠️ The `analyze-hindi` command does NOT automatically generate charts
- ✅ You MUST run `report --visualize` as a separate step to create visualizations
- 💰 Using `--evaluator-provider ollama` makes evaluation completely free (no API costs)
- 🌐 The entire workflow runs offline - no internet required after models are downloaded

### Cross-Lingual Report Output

The cross-lingual analysis generates additional reports:

- `crosslingual_report.md`: Comparison of English vs Hindi performance
- `language_parity.json`: Detailed parity metrics per model
- `charts/language_comparison.png`: Visual comparison of scores
- `charts/parity_index.png`: Language parity visualization

## Task-Fitness Evaluation

This feature answers the question: **"Which open-source model is best for which task — especially in Hindi?"**

Designed for government agencies and public-sector organizations that need to select the right local LLM for a specific job without fine-tuning or API costs.

### Task Categories

| Category | Evaluator | Score | Description |
|----------|-----------|-------|-------------|
| `summarization` | `SummarizationEvaluator` | 0-3 | Key point capture, conciseness, no hallucination |
| `translation` | `TranslationEvaluator` | 0-3 | Meaning accuracy, natural Hindi phrasing, grammar |
| `logical_mathematical` | `LogicalMathEvaluator` | 0-3 | Correct answer + valid reasoning chain |
| `conversational` | `ConversationalEvaluator` | 0-3 | Helpfulness, natural tone, contextual appropriateness |
| `legal_administrative` | `LegalAdminEvaluator` | 0-3 | Correct legal/admin terminology, formal structure |
| `sentiment_analysis` | `SentimentEvaluator` | 0-3 | Sentiment accuracy, nuance, sarcasm detection |
| `code_generation` | `CodeGenerationEvaluator` | 0-3 | Syntax correctness, logic, best practices |
| `instruction_following` | `InstructionEvaluator` | 0-3 | Adherence to explicit instructions |
| `factual_qa` | `FactualityEvaluator` | 0-2 | Factual accuracy |
| `creative_reasoning` | `ToneEvaluator` | 0-2 | Creative and reasoning quality |

### Government Use-Case Benchmark Prompts

The `task_fitness_hindi.json` dataset covers real government scenarios in natural Hindi:

| Domain | Example Prompt |
|--------|---------------|
| RTI (सूचना का अधिकार) | RTI आवेदन कैसे लिखें? |
| GST | GST पंजीकरण की प्रक्रिया क्या है? |
| MGNREGA | मनरेगा जॉब कार्ड के लिए आवेदन कैसे करें? |
| PIL (याचिका) | जनहित याचिका दाखिल करने की प्रक्रिया |
| Aadhaar | आधार कार्ड अपडेट कैसे करें? |
| नगर निगम | संपत्ति कर कैसे भरें? |
| Ayushman Bharat | आयुष्मान भारत योजना में नाम कैसे जोड़ें? |

### Output Files

Running `task-fitness` produces:

| File | Description |
|------|-------------|
| `fitness_matrix.json` | Raw model × task × language scores |
| `fitness_report.md` | Full report: executive summary, fitness table, per-task rankings, Hindi-focus section, government recommendations |
| `charts/fitness_heatmap.png` | Color-coded model × task grid |
| `charts/model_radar_profiles.png` | Radar chart per model across all tasks |
| `charts/task_rankings_bar.png` | Grouped bar chart: best model per task |
| `charts/hindi_vs_english.png` | Side-by-side Hindi vs English score comparison |

### How the Hindi Evaluation Works

The entire pipeline is Hindi end-to-end:

1. **Input** — Hindi prompt from `task_fitness_hindi.json` (e.g., *"RTI आवेदन कैसे लिखें?"*)
2. **Model response** — The tested model responds in Hindi
3. **Evaluator** — A second local model (e.g., `qwen2:1.5b`) reads both the Hindi prompt and Hindi response, then assigns a score (0-3)

No translation happens at any stage. The evaluator scores the Hindi response directly.

### Using Task-Fitness in Python

```python
from llm_drift_analyzer import Config
from llm_drift_analyzer.models import PromptSet
from llm_drift_analyzer.analyzers import TaskFitnessAnalyzer
from llm_drift_analyzer.reporters import FitnessReportGenerator, FitnessVisualizer

# Load Hindi benchmark prompts
prompts = PromptSet.load_from_file("data/prompts/task_fitness_hindi.json")

# Run fitness analysis
config = Config.from_env()
config.evaluator_provider = "ollama"
config.evaluator_model = "qwen2:1.5b"

analyzer = TaskFitnessAnalyzer(config, provider="ollama")
matrix = analyzer.run_fitness_analysis(
    prompts=prompts,
    models=["qwen2:1.5b", "tinyllama", "llama3"],
    iterations=3,
)

# Get recommendations for Hindi
for rec in matrix.get_recommendations(language="hi"):
    print(f"{rec['category']}: {rec['recommended_model']} (score: {rec['score']:.2f})")

# Generate report and charts
FitnessReportGenerator(matrix).save_report("output/fitness_report.md")
FitnessVisualizer(matrix).save_all_plots("output/charts/", language="hi")
```

### Task-Fitness Workflow (Fully Offline)

```bash
# Step 1: Pull small models (one-time)
ollama pull qwen2:1.5b
ollama pull tinyllama

# Step 2: Run task-fitness evaluation (Hindi only, 1 iteration for quick test)
python main.py task-fitness \
    --models qwen2:1.5b tinyllama \
    --provider ollama \
    --iterations 1 \
    --languages hi \
    --evaluator-provider ollama \
    --evaluator-model qwen2:1.5b \
    --visualize \
    --output output/task_fitness_hindi

# Step 3: Open the report
open output/task_fitness_hindi/fitness_report.md
```

### CLI Reference: `task-fitness`

| Argument | Default | Description |
|----------|---------|-------------|
| `--models` | required | Models to evaluate (e.g., `qwen2:1.5b tinyllama`) |
| `--provider` | `ollama` | LLM provider for querying models |
| `--ollama-url` | `http://localhost:11434` | Ollama server URL |
| `--iterations` | `3` | Prompts per model per category |
| `--languages` | `both` | `en`, `hi`, or `both` |
| `--categories` | all | Filter to specific categories |
| `--evaluator-provider` | `openai` | Provider for scoring (`ollama` for free) |
| `--evaluator-model` | `gpt-4` | Model used as judge |
| `--visualize` | off | Generate heatmap, radar, and bar charts |
| `--output` | `output/task_fitness` | Output directory |

---

## Statistical Methods

The analyzer uses several statistical methods:

1. **ANOVA**: Tests for significant differences between model groups
2. **Pairwise t-tests**: Compares specific model pairs
3. **Cohen's d**: Measures effect size of differences
4. **CUSUM**: Detects change points in time series data
5. **Correlation Analysis**: Identifies relationships between metrics

## Common Workflows

### Workflow 1: Fully Offline Hindi Analysis

```bash
# 1. Pull models (one-time setup)
ollama pull qwen2:1.5b gemma:2b llama3

# 2. Run analysis
python main.py analyze-hindi \
    --models qwen2:1.5b gemma:2b \
    --provider ollama \
    --evaluator-provider ollama \
    --evaluator-model llama3 \
    --iterations 5

# 3. Generate report with charts
python main.py report \
    --input output/hindi/hindi_results.json \
    --visualize
```

### Workflow 2: Cross-Lingual Comparison (English vs Hindi)

```bash
# Analyze same prompts in both languages using local models
python main.py crosslingual \
    --models qwen2:1.5b gemma:2b \
    --provider ollama \
    --evaluator-provider ollama \
    --evaluator-model llama3 \
    --iterations 5 \
    --output output/crosslingual

# Generate comparison report
python main.py report \
    --input output/crosslingual/english_results.json \
    --visualize
python main.py report \
    --input output/crosslingual/hindi_results.json \
    --visualize
```

### Workflow 3: Hybrid (Cloud Query + Local Evaluation)

```bash
# Query expensive models, evaluate locally to save costs
python main.py analyze \
    --models gpt-4 claude-3-opus \
    --evaluator-provider ollama \
    --evaluator-model llama3 \
    --iterations 10

python main.py report --input output/results.json --visualize
```

### Workflow 4: Compare Model Versions

```bash
# Compare different versions or sizes of same model
python main.py compare \
    --model1 qwen2:1.5b \
    --model2 qwen2:7b \
    --provider ollama \
    --evaluator-provider ollama \
    --evaluator-model llama3 \
    --iterations 10 \
    --metric instruction_score
```

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
