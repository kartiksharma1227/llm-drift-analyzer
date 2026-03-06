# LLM Drift Analyzer — Run Commands (NVIDIA GPU Setup)

> Judge/Evaluator model: **gpt-oss:20b** via Ollama (local GPU)
> Run all commands from inside the `llm_drift_analyzer/` directory.

---

## 1. Prerequisites

### Install Python dependencies

```bash
cd llm_drift_analyzer

python -m venv venv
source venv/bin/activate          # Linux/macOS
# venv\Scripts\activate           # Windows

pip install -r requirements.txt
pip install -e .
```

### Install Ollama (for GPU inference)

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Pull the judge model and any models you want to test

```bash
# Pull gpt-oss 20b (used as the judge/evaluator)
ollama pull gpt-oss:20b

# Pull models you want to test for drift (examples below — swap as needed)
ollama pull llama3.1:8b
ollama pull mistral
```

### Start Ollama server (if not already running)

```bash
ollama serve
```

---

## 2. Configure Environment

Copy and edit the `.env` file:

```bash
cp .env.example .env
```

Set these values in `.env`:

```env
OPENAI_API_KEY=your_openai_api_key_here    # Only needed if testing OpenAI models
OLLAMA_BASE_URL=http://localhost:11434

# Judge model config
EVALUATOR_PROVIDER=ollama
EVALUATOR_MODEL=gpt-oss:20b

LOG_LEVEL=INFO
OUTPUT_DIR=output
```

---

## 3. Validate Setup

```bash
python main.py validate
```

---

## 4. Run Analysis Commands

### English Drift Analysis (standard 15-prompt benchmark)

```bash
python main.py analyze \
  --models llama3.1:8b mistral \
  --provider ollama \
  --evaluator-provider ollama \
  --evaluator-model gpt-oss:20b \
  --iterations 10 \
  --output output/english \
  --save-responses
```

### English Drift Analysis (expanded 90-prompt benchmark — IFEval + TruthfulQA)

```bash
python main.py analyze \
  --models llama3.1:8b mistral \
  --provider ollama \
  --evaluator-provider ollama \
  --evaluator-model gpt-oss:20b \
  --use-expanded \
  --language en \
  --iterations 5 \
  --output output/english_expanded \
  --save-responses
```

### Hindi Drift Analysis (standard 15-prompt benchmark)

```bash
python main.py analyze-hindi \
  --models llama3.1:8b mistral \
  --provider ollama \
  --evaluator-provider ollama \
  --evaluator-model gpt-oss:20b \
  --iterations 5 \
  --output output/hindi
```

### Hindi Drift Analysis (expanded 100-prompt benchmark — IndicMSMARCO + XNLI)

```bash
python main.py analyze-hindi \
  --models llama3.1:8b mistral \
  --provider ollama \
  --evaluator-provider ollama \
  --evaluator-model gpt-oss:20b \
  --use-expanded \
  --iterations 5 \
  --output output/hindi_expanded
```

### Cross-Lingual Analysis (English vs Hindi comparison)

```bash
python main.py crosslingual \
  --models llama3.1:8b mistral \
  --provider ollama \
  --evaluator-provider ollama \
  --evaluator-model gpt-oss:20b \
  --iterations 5 \
  --output output/crosslingual
```

### Compare Two Model Versions

```bash
python main.py compare \
  --model1 llama3.1:8b \
  --model2 mistral \
  --provider ollama \
  --evaluator-provider ollama \
  --evaluator-model gpt-oss:20b \
  --iterations 5 \
  --metric instruction_score
```

---

## 5. Generate Reports

### Markdown report from English results

```bash
python main.py report \
  --input output/english/results.json \
  --format markdown \
  --visualize \
  --include-samples \
  --output output/english/report.md
```

### Markdown report from Hindi results

```bash
python main.py report \
  --input output/hindi/hindi_results.json \
  --format markdown \
  --visualize \
  --output output/hindi/report.md
```

### JSON report

```bash
python main.py report \
  --input output/english/results.json \
  --format json \
  --output output/english/report.json
```

---

## 6. List Available Models

```bash
# List all configured providers
python main.py list-models

# List only Ollama models
python main.py list-models --provider ollama
```

---

## 7. Quick Single-Model Test (to verify everything works)

```bash
python main.py analyze \
  --models llama3.1:8b \
  --provider ollama \
  --evaluator-provider ollama \
  --evaluator-model gpt-oss:20b \
  --iterations 2 \
  --output output/test
```

---

## Notes

- All output (results JSON, reports, charts) goes into the `--output` directory.
- `--iterations` controls how many times each prompt is sent per model. Higher = more statistically reliable but slower.
- If running out of VRAM, ensure only one model is loaded in Ollama at a time: `ollama stop <model>` before pulling/running another.
- Charts (PNG) are generated inside `<output_dir>/charts/` when `--visualize` is passed to `report`.
- Verbose logging: add `-v` to any command (e.g. `python main.py -v analyze ...`).
