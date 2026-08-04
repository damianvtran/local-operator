"""Tests for the rewritten model/configure.py.

The legacy langchain plumbing (ChatOpenAI/ChatAnthropic/...) is gone:
``configure_model`` now returns a ``ModelConfiguration`` whose ``.spec`` is
the harness ``ModelSpec`` consumed by wire clients, and ``validate_model``
hits the same endpoints as before through a descriptor table.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import requests
from pydantic import SecretStr

from local_operator.credentials import CredentialManager
from local_operator.model.configure import (
    DEFAULT_TEMPERATURE,
    calculate_cost,
    configure_model,
    create_stream_fn,
    get_model_info_from_openrouter,
    validate_model,
)
from local_operator.model.registry import ModelInfo


@pytest.fixture
def mock_credential_manager():
    manager = MagicMock(spec=CredentialManager)
    manager.get_credential = MagicMock(return_value=SecretStr("test_key"))
    manager.prompt_for_credential = MagicMock(return_value=SecretStr("test_key"))
    return manager


@pytest.fixture
def mock_requests_get():
    with patch("local_operator.model.configure.requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"data": [{"id": "test-model"}]}
        yield mock_get


# ---------------------------------------------------------------------------
# configure_model
# ---------------------------------------------------------------------------


def test_configure_model_deepseek(mock_credential_manager):
    config = configure_model("deepseek", "deepseek-chat", mock_credential_manager)
    assert config.name == "deepseek-chat"
    assert config.hosting == "deepseek"
    assert config.api_key is not None
    assert config.api_key.get_secret_value() == "test_key"
    assert config.info is not None
    assert config.instance is None  # new engine: streaming via stream_fn
    assert config.spec.provider == "deepseek"
    assert config.spec.model_id == "deepseek-chat"
    assert config.spec.base_url == "https://api.deepseek.com/v1"


def test_configure_model_openai(mock_credential_manager):
    config = configure_model("openai", "gpt-4", mock_credential_manager)
    assert config.name == "gpt-4"
    assert config.spec.provider == "openai"
    assert config.spec.base_url == "https://api.openai.com/v1"
    assert config.api_key is not None


def test_configure_model_openai_unknown_model_uses_defaults(mock_credential_manager):
    """Unknown models get sensible ModelSpec defaults, not a KeyError."""
    config = configure_model("openai", "gpt-9-turbo-invented", mock_credential_manager)
    assert config.spec.context_window > 0
    assert config.spec.max_output_tokens > 0


def test_configure_model_ollama(mock_credential_manager):
    config = configure_model("ollama", "llama2", mock_credential_manager)
    assert config.spec.base_url == "http://localhost:11434/v1"
    assert config.spec.provider == "ollama"


def test_configure_model_ollama_missing_model(mock_credential_manager):
    with pytest.raises(ValueError) as exc_info:
        configure_model("ollama", "", mock_credential_manager)
    assert "Model is required for ollama hosting" in str(exc_info.value)


def test_configure_model_noop_maps_to_test(mock_credential_manager):
    """Legacy 'noop' hosting resolves to the mock wire (unchanged CLI UX)."""
    config = configure_model("noop", "noop", mock_credential_manager)
    assert config.spec.provider == "test"


def test_configure_model_invalid_hosting(mock_credential_manager):
    with pytest.raises(ValueError) as exc_info:
        configure_model("invalid", "model", mock_credential_manager)
    assert "Unsupported hosting platform: invalid" in str(exc_info.value)


def test_configure_model_missing_hosting(mock_credential_manager):
    with pytest.raises(ValueError) as exc_info:
        configure_model("", "model", mock_credential_manager)
    assert "Hosting is required" in str(exc_info.value)


@pytest.mark.parametrize(
    "hosting, default_model",
    [
        ("openai", "gpt-4o"),
        ("anthropic", "claude-3-5-sonnet-latest"),
        ("deepseek", "deepseek-chat"),
        ("kimi", "moonshot-v1-32k"),
        ("alibaba", "qwen-plus"),
        ("google", "gemini-2.0-flash-001"),
        ("mistral", "mistral-large-latest"),
        ("openrouter", "google/gemini-2.0-flash-001"),
        ("radient", "auto"),
        ("xai", "grok-3"),
    ],
)
def test_configure_model_default_names(hosting: str, default_model: str):
    config = configure_model(hosting, "", None)
    assert config.name == default_model


def test_configure_model_anthropic_spec(mock_credential_manager):
    config = configure_model("anthropic", "claude-3-5-sonnet-latest", mock_credential_manager)
    assert config.spec.provider == "anthropic"
    assert config.spec.base_url == "https://api.anthropic.com"
    assert config.temperature == DEFAULT_TEMPERATURE


def test_configure_model_kimi_base_url(mock_credential_manager):
    config = configure_model("kimi", "moonshot-v1-8k", mock_credential_manager)
    assert config.spec.base_url == "https://api.moonshot.cn/v1"
    assert config.spec.model_id == "moonshot-v1-8k"


def test_configure_model_alibaba_base_url(mock_credential_manager):
    config = configure_model("alibaba", "qwen-plus", mock_credential_manager)
    assert config.spec.base_url == "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"


def test_configure_model_no_credential_manager_yields_no_key():
    """configure_model never prompts; missing keys resolve at stream time."""
    config = configure_model("deepseek", "deepseek-chat", None)
    assert config.api_key is None
    assert config.spec.model_id == "deepseek-chat"


def test_configure_model_known_model_pulls_registry_info():
    config = configure_model("google", "gemini-2.0-flash-001", None)
    assert config.info.id == "gemini-2.0-flash-001"
    assert config.spec.context_window == config.info.context_window
    assert config.spec.max_output_tokens == config.info.max_tokens


def test_calculate_cost() -> None:
    model_info = ModelInfo(
        id="test-model",
        name="test-model",
        description="Mock model",
        input_price=1,
        output_price=2,
        recommended=True,
    )
    input_tokens = 1000
    output_tokens = 2000
    expected_cost = (input_tokens / 1_000_000) * model_info.input_price + (
        output_tokens / 1_000_000
    ) * model_info.output_price
    assert calculate_cost(model_info, input_tokens, output_tokens) == pytest.approx(expected_cost)
    assert calculate_cost(model_info, 0, 0) == 0.0


@pytest.fixture
def mock_openrouter_client():
    client = MagicMock()
    mock_model_data = [
        MagicMock(
            id="openai/gpt-4o",
            name="GPT-4o",
            description="Mock description before",
            pricing=MagicMock(prompt=1.0 / 1_000_000, completion=2.0 / 1_000_000),
        ),
        MagicMock(
            id="google/gemini-2.0-flash-001",
            name="Gemini 2.0 Flash",
            description="Mock description",
            pricing=MagicMock(prompt=5.0 / 1_000_000, completion=10.0 / 1_000_000),
        ),
        MagicMock(
            id="anthropic/claude-3-5-sonnet-latest",
            name="Claude 3.5 Sonnet",
            description="Mock description after",
            pricing=MagicMock(prompt=15.0 / 1_000_000, completion=20.0 / 1_000_000),
        ),
    ]
    client.list_models.return_value = MagicMock(data=mock_model_data)
    return client


def test_get_model_info_from_openrouter(mock_openrouter_client):
    model_info = get_model_info_from_openrouter(mock_openrouter_client, "google/gemini-2.0-flash-001")
    assert model_info.input_price == 5
    assert model_info.output_price == 10


def test_get_model_info_from_openrouter_no_match(mock_openrouter_client):
    with pytest.raises(ValueError, match="Model not found from openrouter models API: non-existent-model"):
        get_model_info_from_openrouter(mock_openrouter_client, "non-existent-model")


def test_configure_model_openrouter_via_client(mock_openrouter_client):
    config = configure_model(
        "openrouter", "google/gemini-2.0-flash-001", None, model_info_client=mock_openrouter_client
    )
    assert config.info.input_price == 5
    assert config.spec.model_id == "google/gemini-2.0-flash-001"


def test_create_stream_fn_returns_callable():
    stream_fn = create_stream_fn(MagicMock(), None)
    assert callable(stream_fn)


# ---------------------------------------------------------------------------
# validate_model — same endpoints/headers as the legacy chain
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "hosting, model, status_code, response_json, expected_result, expected_url, expected_headers",
    [
        (
            "openai",
            "test_model",
            200,
            {"data": [{"id": "test_model"}]},
            True,
            "https://api.openai.com/v1/models",
            {"Authorization": "Bearer test_key"},
        ),
        (
            "openai",
            "test_model",
            404,
            {},
            False,
            "https://api.openai.com/v1/models",
            {"Authorization": "Bearer test_key"},
        ),
        (
            "openai",
            "test_model",
            200,
            {"data": []},
            False,
            "https://api.openai.com/v1/models",
            {"Authorization": "Bearer test_key"},
        ),
        (
            "ollama",
            "test_model",
            200,
            {"models": [{"name": "test_model"}]},
            True,
            "http://localhost:11434/api/tags",
            None,
        ),
        ("ollama", "test_model", 404, {}, False, "http://localhost:11434/api/tags", None),
        (
            "deepseek",
            "test_model",
            200,
            {"data": [{"id": "test_model"}]},
            True,
            "https://api.deepseek.com/v1/models",
            {"Authorization": "Bearer test_key"},
        ),
        (
            "openrouter",
            "test_model",
            200,
            {"data": [{"id": "test_model"}]},
            True,
            "https://openrouter.ai/api/v1/models",
            {"Authorization": "Bearer test_key"},
        ),
        (
            "anthropic",
            "test_model",
            200,
            {"data": [{"id": "test_model"}]},
            True,
            "https://api.anthropic.com/v1/models",
            {"x-api-key": "test_key", "anthropic-version": "2023-06-01"},
        ),
        (
            "anthropic",
            "test-model-latest",
            200,
            {"data": [{"id": "test-model-1234"}]},
            True,
            "https://api.anthropic.com/v1/models",
            {"x-api-key": "test_key", "anthropic-version": "2023-06-01"},
        ),
        (
            "kimi",
            "test_model",
            200,
            {"data": [{"id": "test_model"}]},
            True,
            "https://api.moonshot.cn/v1/models",
            {"Authorization": "Bearer test_key"},
        ),
        (
            "alibaba",
            "test_model",
            200,
            {"data": [{"id": "test_model"}]},
            True,
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/models",
            {"Authorization": "Bearer test_key"},
        ),
        (
            "google",
            "test_model",
            200,
            {"models": [{"name": "test_model"}]},
            True,
            "https://generativelanguage.googleapis.com/v1/models",
            {"x-goog-api-key": "test_key"},
        ),
        (
            "mistral",
            "test_model",
            200,
            {"data": [{"id": "test_model"}]},
            True,
            "https://api.mistral.ai/v1/models",
            {"Authorization": "Bearer test_key"},
        ),
        (
            "xai",
            "test_model",
            200,
            {"data": [{"id": "test_model"}]},
            True,
            "https://api.x.ai/v1/models",
            {"Authorization": "Bearer test_key"},
        ),
        (
            "radient",
            "test_model",
            200,
            {"data": [{"id": "test_model"}]},
            True,
            "https://api.radienthq.com/v1/models",
            {"Authorization": "Bearer test_key"},
        ),
    ],
)
def test_validate_model(
    mock_requests_get: MagicMock,
    hosting: str,
    model: str,
    status_code: int,
    response_json: dict[str, Any],
    expected_result: bool,
    expected_url: str,
    expected_headers: dict[str, Any] | None,
) -> None:
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.json.return_value = response_json
    mock_requests_get.return_value = mock_response

    api_key = SecretStr("test_key")
    result = validate_model(hosting, model, api_key)
    assert result == expected_result

    if expected_headers:
        mock_requests_get.assert_called_once_with(expected_url, headers=expected_headers)
    else:
        mock_requests_get.assert_called_once_with(expected_url)


def test_validate_model_unknown_hosting_returns_true():
    """Providers without a validation endpoint pass (legacy behaviour)."""
    assert validate_model("test", "anything", SecretStr("k")) is True


@patch("local_operator.model.configure.requests.get")
def test_validate_model_failure(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_get.return_value = mock_response
    assert validate_model("openai", "test_model", SecretStr("test_key")) is False


@patch("local_operator.model.configure.requests.get")
def test_validate_model_exception(mock_get):
    mock_get.side_effect = requests.exceptions.RequestException("API error")
    with pytest.raises(requests.exceptions.RequestException, match="API error"):
        validate_model("openai", "test_model", SecretStr("test_key"))


@patch("local_operator.model.configure.requests.get")
def test_validate_model_no_model_found(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"data": []}
    mock_get.return_value = mock_response
    assert validate_model("openai", "test_model", SecretStr("test_key")) is False


@patch("local_operator.model.configure.requests.get")
def test_validate_model_ollama_success(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"models": [{"name": "test_model"}]}
    mock_get.return_value = mock_response
    assert validate_model("ollama", "test_model", SecretStr("test_key")) is True


@patch("local_operator.model.configure.requests.get")
def test_validate_model_ollama_failure(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"models": []}
    mock_get.return_value = mock_response
    assert validate_model("ollama", "test_model", SecretStr("test_key")) is False
