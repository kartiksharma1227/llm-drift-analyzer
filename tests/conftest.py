"""
Pytest fixtures and configuration for LLM Drift Analyzer tests.
"""

import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock
from datetime import datetime

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_drift_analyzer.models.prompt import Prompt, PromptCategory, PromptSet
from llm_drift_analyzer.models.response_analysis import ResponseAnalysis, AnalysisResultSet
from llm_drift_analyzer.utils.config import Config, APIConfig
from llm_drift_analyzer.clients.base_client import QueryResult


@pytest.fixture
def sample_prompt():
    """Create a sample prompt for testing."""
    return Prompt(
        id="TEST-001",
        text="Summarize the benefits of renewable energy in exactly 3 bullet points.",
        category=PromptCategory.INSTRUCTION_FOLLOWING,
        description="Test prompt for instruction following",
        expected_format="3 bullet points",
    )


@pytest.fixture
def sample_prompts():
    """Create a set of sample prompts for testing."""
    prompts = [
        Prompt(
            id="IF-001",
            text="Write 3 bullet points about AI.",
            category=PromptCategory.INSTRUCTION_FOLLOWING,
        ),
        Prompt(
            id="QA-001",
            text="What is the capital of France?",
            category=PromptCategory.FACTUAL_QA,
            reference_answer="Paris is the capital of France.",
        ),
        Prompt(
            id="CR-001",
            text="Write a short poem about coding.",
            category=PromptCategory.CREATIVE_REASONING,
        ),
    ]
    return PromptSet(prompts=prompts, name="Test Prompts", version="1.0")


@pytest.fixture
def sample_response_analysis():
    """Create a sample ResponseAnalysis for testing."""
    return ResponseAnalysis(
        prompt_id="TEST-001",
        model_version="gpt-4-test",
        response_text="• Benefit 1\n• Benefit 2\n• Benefit 3",
        token_count=25,
        latency_ms=500.0,
        instruction_score=3,
        factuality_score=2,
        tone_score=2,
        timestamp=datetime.now().isoformat(),
        iteration=0,
    )


@pytest.fixture
def sample_results():
    """Create a sample AnalysisResultSet for testing."""
    results = []

    # Create results for two models
    for model in ["gpt-4-v1", "gpt-4-v2"]:
        for i in range(5):
            results.append(ResponseAnalysis(
                prompt_id=f"TEST-{i % 3 + 1:03d}",
                model_version=model,
                response_text=f"Sample response {i} from {model}",
                token_count=50 + i * 10 + (10 if "v2" in model else 0),
                latency_ms=400.0 + i * 50,
                instruction_score=2 + (i % 2),
                factuality_score=1 + (i % 2),
                tone_score=2,
                timestamp=datetime.now().isoformat(),
                iteration=i,
            ))

    return AnalysisResultSet(results=results)


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    return Config(
        openai_api_key="sk-test-openai-key",
        anthropic_api_key="sk-ant-test-key",
        mistral_api_key="test-mistral-key",
        evaluator_model="gpt-4",
        log_level="DEBUG",
        output_dir=Path("test_output"),
        api_config=APIConfig(
            temperature=0.1,
            max_tokens=1000,
            top_p=0.9,
        ),
    )


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    mock = MagicMock()

    # Mock chat completion response
    mock_choice = Mock()
    mock_choice.message.content = "This is a test response."

    mock_usage = Mock()
    mock_usage.prompt_tokens = 10
    mock_usage.completion_tokens = 20
    mock_usage.total_tokens = 30

    mock_response = Mock()
    mock_response.choices = [mock_choice]
    mock_response.model = "gpt-4-test"
    mock_response.usage = mock_usage

    mock.chat.completions.create.return_value = mock_response

    return mock


@pytest.fixture
def mock_anthropic_client():
    """Create a mock Anthropic client."""
    mock = MagicMock()

    # Mock message response
    mock_content = Mock()
    mock_content.type = "text"
    mock_content.text = "This is a test response from Claude."

    mock_usage = Mock()
    mock_usage.input_tokens = 10
    mock_usage.output_tokens = 20

    mock_response = Mock()
    mock_response.content = [mock_content]
    mock_response.model = "claude-3-opus-test"
    mock_response.usage = mock_usage

    mock.messages.create.return_value = mock_response

    return mock


@pytest.fixture
def mock_query_result():
    """Create a mock QueryResult."""
    return QueryResult(
        response_text="This is a mock response for testing purposes.",
        latency_ms=250.5,
        model="gpt-4-mock",
        usage={
            "prompt_tokens": 15,
            "completion_tokens": 25,
            "total_tokens": 40,
        },
    )


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output directory."""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def sample_prompts_json(tmp_path):
    """Create a temporary prompts JSON file."""
    import json

    prompts_data = {
        "name": "Test Prompts",
        "version": "1.0",
        "prompts": [
            {
                "id": "TEST-001",
                "text": "Test prompt 1",
                "category": "instruction_following",
            },
            {
                "id": "TEST-002",
                "text": "Test prompt 2",
                "category": "factual_qa",
            },
        ],
    }

    prompts_file = tmp_path / "test_prompts.json"
    with open(prompts_file, "w") as f:
        json.dump(prompts_data, f)

    return prompts_file
