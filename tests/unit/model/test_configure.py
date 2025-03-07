from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import requests
from pydantic import SecretStr

from local_operator.credentials import CredentialManager
from local_operator.mocks import ChatNoop
from local_operator.model.configure import (
    DEFAULT_TEMPERATURE,
    calculate_cost,
    configure_model,
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
def mock_successful_response():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"data": [{"id": "test-model"}]}
    return mock_response


@pytest.fixture
def mock_requests_get():
    with patch("local_operator.model.configure.requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"data": [{"id": "test-model"}]}
        yield mock_get


def test_configure_model_deepseek(mock_credential_manager):
    with patch(
        "local_operator.model.configure.ChatOpenAI", return_value=MagicMock()
    ) as mock_chat_openai:
        model_configuration = configure_model("deepseek", "deepseek-chat", mock_credential_manager)
        assert model_configuration is not None
        mock_chat_openai.assert_called_once()
        call_args = mock_chat_openai.call_args
        assert (
            call_args.kwargs["api_key"].get_secret_value()
            == SecretStr("test_key").get_secret_value()
        )
        assert model_configuration.api_key is not None
        assert model_configuration.instance is not None
        assert model_configuration.info is not None
        assert model_configuration.name == "deepseek-chat"


def test_configure_model_openai(mock_credential_manager):
    with patch(
        "local_operator.model.configure.ChatOpenAI", return_value=MagicMock()
    ) as mock_chat_openai:
        model_configuration = configure_model("openai", "gpt-4", mock_credential_manager)
        assert model_configuration is not None
        mock_chat_openai.assert_called_once()
        call_args = mock_chat_openai.call_args
        assert (
            call_args.kwargs["api_key"].get_secret_value()
            == SecretStr("test_key").get_secret_value()
        )
        assert model_configuration.api_key is not None
        assert model_configuration.instance is not None
        assert model_configuration.info is not None
        assert model_configuration.name == "gpt-4"


def test_configure_model_ollama(mock_credential_manager):
    with patch(
        "local_operator.model.configure.ChatOllama", return_value=MagicMock()
    ) as mock_chat_ollama:
        model_configuration = configure_model("ollama", "llama2", mock_credential_manager)
        assert model_configuration is not None
        mock_chat_ollama.assert_called_once()
        call_args = mock_chat_ollama.call_args
        assert call_args.kwargs["model"] == "llama2"


def test_configure_model_noop(mock_credential_manager):
    model_configuration = configure_model("noop", "noop", mock_credential_manager)
    assert isinstance(model_configuration.instance, ChatNoop)


def test_configure_model_invalid_hosting(mock_credential_manager):
    with pytest.raises(ValueError) as exc_info:
        configure_model("invalid", "model", mock_credential_manager)
    assert "Unsupported hosting platform: invalid" in str(exc_info.value)


def test_configure_model_missing_hosting(mock_credential_manager):
    with pytest.raises(ValueError) as exc_info:
        configure_model("", "model", mock_credential_manager)
    assert "Hosting is required" in str(exc_info.value)


def test_configure_model_deepseek_fallback():
    credential_manager = MagicMock(spec=CredentialManager)
    credential_manager.get_credential = MagicMock(return_value=None)
    credential_manager.prompt_for_credential = MagicMock(return_value=SecretStr("prompted_key"))
    with patch(
        "local_operator.model.configure.ChatOpenAI", return_value=MagicMock()
    ) as mock_chat_openai:
        model_configuration = configure_model("deepseek", "deepseek-chat", credential_manager)
        assert model_configuration is not None
        mock_chat_openai.assert_called_once()
        call_args = mock_chat_openai.call_args
        assert (
            call_args.kwargs["api_key"].get_secret_value()
            == SecretStr("prompted_key").get_secret_value()
        )
        assert model_configuration.api_key is not None


def test_configure_model_anthropic(mock_credential_manager):
    with patch(
        "local_operator.model.configure.ChatAnthropic", return_value=MagicMock()
    ) as mock_chat_anthropic:
        model_configuration = configure_model(
            "anthropic", "claude-3-5-sonnet-latest", mock_credential_manager
        )
        assert model_configuration is not None
        mock_chat_anthropic.assert_called_once()
        call_args = mock_chat_anthropic.call_args
        assert (
            call_args.kwargs["api_key"].get_secret_value()
            == SecretStr("test_key").get_secret_value()
        )
        assert model_configuration.api_key is not None
        assert call_args.kwargs["model_name"] == "claude-3-5-sonnet-latest"
        assert call_args.kwargs["temperature"] == DEFAULT_TEMPERATURE
        assert call_args.kwargs["timeout"] is None
        assert call_args.kwargs["stop"] is None


def test_configure_model_anthropic_default(mock_credential_manager):
    with patch(
        "local_operator.model.configure.ChatAnthropic", return_value=MagicMock()
    ) as mock_chat_anthropic:
        model_configuration = configure_model("anthropic", "", mock_credential_manager)
        assert model_configuration is not None
        call_args = mock_chat_anthropic.call_args
        assert call_args.kwargs["model_name"] == "claude-3-5-sonnet-latest"
        assert call_args.kwargs["temperature"] == DEFAULT_TEMPERATURE
        assert model_configuration.api_key is not None


def test_configure_model_anthropic_fallback():
    credential_manager = MagicMock(spec=CredentialManager)
    credential_manager.get_credential = MagicMock(return_value=None)
    credential_manager.prompt_for_credential = MagicMock(
        return_value=SecretStr("fallback_anthropic_key")
    )
    with patch(
        "local_operator.model.configure.ChatAnthropic", return_value=MagicMock()
    ) as mock_chat_anthropic:
        model_configuration = configure_model(
            "anthropic", "claude-3-5-sonnet-latest", credential_manager
        )
        assert model_configuration is not None
        mock_chat_anthropic.assert_called_once()
        call_args = mock_chat_anthropic.call_args
        assert (
            call_args.kwargs["api_key"].get_secret_value()
            == SecretStr("fallback_anthropic_key").get_secret_value()
        )
        assert model_configuration.api_key is not None


def test_configure_model_kimi_default(mock_credential_manager):
    with patch(
        "local_operator.model.configure.ChatOpenAI", return_value=MagicMock()
    ) as mock_chat_openai:
        model_configuration = configure_model("kimi", "", mock_credential_manager)
        assert model_configuration is not None
        call_args = mock_chat_openai.call_args
        assert call_args.kwargs["base_url"] == "https://api.moonshot.cn/v1"
        assert model_configuration.api_key is not None


def test_configure_model_kimi_explicit(mock_credential_manager):
    with patch(
        "local_operator.model.configure.ChatOpenAI", return_value=MagicMock()
    ) as mock_chat_openai:
        model_configuration = configure_model("kimi", "moonshot-v1-8k", mock_credential_manager)
        assert model_configuration is not None
        call_args = mock_chat_openai.call_args
        assert call_args.kwargs["model"] == "moonshot-v1-8k"
        assert model_configuration.api_key is not None


def test_configure_model_kimi_fallback():
    credential_manager = MagicMock(spec=CredentialManager)
    credential_manager.get_credential = MagicMock(return_value="")
    credential_manager.prompt_for_credential = MagicMock(
        return_value=SecretStr("fallback_kimi_key")
    )
    with patch(
        "local_operator.model.configure.ChatOpenAI", return_value=MagicMock()
    ) as mock_chat_openai:
        model_configuration = configure_model("kimi", "moonshot-v1-8k", credential_manager)
        assert model_configuration is not None
        call_args = mock_chat_openai.call_args
        assert (
            call_args.kwargs["api_key"].get_secret_value()
            == SecretStr("fallback_kimi_key").get_secret_value()
        )
        assert model_configuration.api_key is not None
        assert call_args.kwargs["model"] == "moonshot-v1-8k"


def test_configure_model_alibaba_default(mock_credential_manager):
    with patch(
        "local_operator.model.configure.ChatOpenAI", return_value=MagicMock()
    ) as mock_chat_openai:
        model_configuration = configure_model("alibaba", "", mock_credential_manager)
        assert model_configuration is not None
        call_args = mock_chat_openai.call_args
        assert call_args.kwargs["model"] == "qwen-plus"
        assert (
            call_args.kwargs["base_url"] == "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        )
        assert model_configuration.api_key is not None


def test_configure_model_alibaba_fallback():
    credential_manager = MagicMock(spec=CredentialManager)
    credential_manager.get_credential = MagicMock(return_value=None)
    credential_manager.prompt_for_credential = MagicMock(
        return_value=SecretStr("fallback_alibaba_key")
    )
    with patch(
        "local_operator.model.configure.ChatOpenAI", return_value=MagicMock()
    ) as mock_chat_openai:
        model_configuration = configure_model("alibaba", "qwen-plus", credential_manager)
        assert model_configuration is not None
        call_args = mock_chat_openai.call_args
        assert (
            call_args.kwargs["api_key"].get_secret_value()
            == SecretStr("fallback_alibaba_key").get_secret_value()
        )
        assert model_configuration.api_key is not None
        assert call_args.kwargs["model"] == "qwen-plus"


def test_configure_model_openai_fallback():
    credential_manager = MagicMock(spec=CredentialManager)
    credential_manager.get_credential = MagicMock(return_value="")
    credential_manager.prompt_for_credential = MagicMock(
        return_value=SecretStr("fallback_openai_key")
    )
    with patch(
        "local_operator.model.configure.ChatOpenAI", return_value=MagicMock()
    ) as mock_chat_openai:
        model_configuration = configure_model("openai", "gpt-4o", credential_manager)
        assert model_configuration is not None
        call_args = mock_chat_openai.call_args
        assert (
            call_args.kwargs["api_key"].get_secret_value()
            == SecretStr("fallback_openai_key").get_secret_value()
        )
        assert model_configuration.api_key is not None
        assert call_args.kwargs["model"] == "gpt-4o"


def test_configure_model_ollama_missing_model(mock_credential_manager):
    with pytest.raises(ValueError) as exc_info:
        configure_model("ollama", "", mock_credential_manager)
    assert "Model is required for ollama hosting" in str(exc_info.value)


def test_configure_model_google_default(mock_credential_manager):
    with patch(
        "local_operator.model.configure.ChatGoogleGenerativeAI", return_value=MagicMock()
    ) as mock_chat_google:
        model_configuration = configure_model("google", "", mock_credential_manager)
        assert model_configuration is not None
        call_args = mock_chat_google.call_args
        assert call_args.kwargs["model"] == "gemini-2.0-flash-001"
        assert model_configuration.api_key is not None


def test_configure_model_google_fallback():
    credential_manager = MagicMock(spec=CredentialManager)
    credential_manager.get_credential = MagicMock(return_value=None)
    credential_manager.prompt_for_credential = MagicMock(
        return_value=SecretStr("fallback_google_key")
    )
    with patch(
        "local_operator.model.configure.ChatGoogleGenerativeAI", return_value=MagicMock()
    ) as mock_chat_google:
        model_configuration = configure_model("google", "gemini-2.0-flash-001", credential_manager)
        assert model_configuration is not None
        call_args = mock_chat_google.call_args
        assert (
            call_args.kwargs["api_key"].get_secret_value()
            == SecretStr("fallback_google_key").get_secret_value()
        )
        assert model_configuration.api_key is not None
        assert call_args.kwargs["model"] == "gemini-2.0-flash-001"


def test_configure_model_mistral_default(mock_credential_manager):
    with patch(
        "local_operator.model.configure.ChatOpenAI", return_value=MagicMock()
    ) as mock_chat_openai:
        model_configuration = configure_model("mistral", "", mock_credential_manager)
        assert model_configuration is not None
        call_args = mock_chat_openai.call_args
        assert call_args.kwargs["model"] == "mistral-large-latest"
        assert call_args.kwargs["base_url"] == "https://api.mistral.ai/v1"
        assert model_configuration.api_key is not None


def test_configure_model_mistral_fallback():
    credential_manager = MagicMock(spec=CredentialManager)
    credential_manager.get_credential = MagicMock(return_value=None)
    credential_manager.prompt_for_credential = MagicMock(
        return_value=SecretStr("fallback_mistral_key")
    )
    with patch(
        "local_operator.model.configure.ChatOpenAI", return_value=MagicMock()
    ) as mock_chat_openai:
        model_configuration = configure_model("mistral", "mistral-large-latest", credential_manager)
        assert model_configuration is not None
        call_args = mock_chat_openai.call_args
        assert (
            call_args.kwargs["api_key"].get_secret_value()
            == SecretStr("fallback_mistral_key").get_secret_value()
        )
        assert model_configuration.api_key is not None
        assert call_args.kwargs["model"] == "mistral-large-latest"


def test_calculate_cost() -> None:
    """Test that the calculate_cost function works correctly."""
    model_info = ModelInfo(
        id="test-model",
        name="test-model",
        description="Mock model",
        input_price=1,
        output_price=2,
    )
    input_tokens = 1000
    output_tokens = 2000
    expected_cost = (input_tokens / 1_000_000) * model_info.input_price + (
        output_tokens / 1_000_000
    ) * model_info.output_price
    assert calculate_cost(model_info, input_tokens, output_tokens) == pytest.approx(expected_cost)

    # Test with zero tokens
    assert calculate_cost(model_info, 0, 0) == 0.0


@pytest.fixture
def mock_openrouter_client():
    """Mocks the OpenRouter client to return a predefined model list."""
    client = MagicMock()
    # Define a mock response that mimics the OpenRouter API's list_models endpoint
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
    mock_response = MagicMock(data=mock_model_data)
    client.list_models.return_value = mock_response
    return client


def test_get_model_info_from_openrouter(mock_openrouter_client):
    """Tests retrieving model info from OpenRouter when a match is found."""
    model_info = get_model_info_from_openrouter(
        mock_openrouter_client, "google/gemini-2.0-flash-001"
    )
    assert model_info.input_price == 5
    assert model_info.output_price == 10


def test_get_model_info_from_openrouter_no_match(mock_openrouter_client):
    """Tests retrieving model info from OpenRouter when no match is found."""
    with pytest.raises(
        ValueError, match="Model not found from openrouter models API: non-existent-model"
    ):
        get_model_info_from_openrouter(mock_openrouter_client, "non-existent-model")


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
            "deepseek",
            "test_model",
            404,
            {},
            False,
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
            "openrouter",
            "test_model",
            404,
            {},
            False,
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
            "anthropic",
            "test_model",
            404,
            {},
            False,
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
            "kimi",
            "test_model",
            404,
            {},
            False,
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
            "alibaba",
            "test_model",
            404,
            {},
            False,
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
            "google",
            "test_model",
            404,
            {},
            False,
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
            "mistral",
            "test_model",
            404,
            {},
            False,
            "https://api.mistral.ai/v1/models",
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
    """Tests validate_model with various scenarios.

    Args:
        mock_requests_get: Mock for requests.get.
        hosting: Hosting provider.
        model: Model name.
        status_code: HTTP status code.
        response_json: Response JSON.
        expected_result: Expected boolean result.
        expected_url: Expected API URL.
        expected_headers: Expected headers for the API request.
    """
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.json.return_value = response_json
    mock_requests_get.return_value = mock_response

    api_key = SecretStr("test_key")

    if status_code >= 500:
        mock_requests_get.side_effect = requests.exceptions.RequestException("API error")
        with pytest.raises(requests.exceptions.RequestException, match="API error"):
            validate_model(hosting, model, api_key)
    else:
        result = validate_model(hosting, model, api_key)
        assert result == expected_result

        if expected_headers:
            mock_requests_get.assert_called_once_with(expected_url, headers=expected_headers)
        else:
            mock_requests_get.assert_called_once_with(expected_url)


@patch("local_operator.model.configure.requests.get")
def test_validate_model_failure(mock_get):
    """Tests validate_model when the API call fails."""
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_get.return_value = mock_response

    api_key = SecretStr("test_key")
    result = validate_model("openai", "test_model", api_key)
    assert result is False


@patch("local_operator.model.configure.requests.get")
def test_validate_model_exception(mock_get):
    """Tests validate_model when an exception is raised during the API call."""
    mock_get.side_effect = requests.exceptions.RequestException("API error")

    api_key = SecretStr("test_key")
    with pytest.raises(requests.exceptions.RequestException, match="API error"):
        validate_model("openai", "test_model", api_key)


@patch("local_operator.model.configure.requests.get")
def test_validate_model_no_model_found(mock_get):
    """Tests validate_model when the model is not found in the API response."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"data": []}  # No models in the response
    mock_get.return_value = mock_response

    api_key = SecretStr("test_key")
    result = validate_model("openai", "test_model", api_key)
    assert result is False


@patch("local_operator.model.configure.requests.get")
def test_validate_model_ollama_success(mock_get):
    """Tests validate_model for ollama when the API call is successful."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"models": [{"name": "test_model"}]}
    mock_get.return_value = mock_response

    api_key = SecretStr("test_key")  # API key is not used for Ollama
    result = validate_model("ollama", "test_model", api_key)
    assert result is True


@patch("local_operator.model.configure.requests.get")
def test_validate_model_ollama_failure(mock_get):
    """Tests validate_model for ollama when the API call fails."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"models": []}
    mock_get.return_value = mock_response

    api_key = SecretStr("test_key")  # API key is not used for Ollama
    result = validate_model("ollama", "test_model", api_key)
    assert result is False
