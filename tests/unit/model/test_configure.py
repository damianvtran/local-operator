from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from local_operator.credentials import CredentialManager
from local_operator.mocks import ChatNoop
from local_operator.model.configure import calculate_cost, configure_model
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
        model, api_key = configure_model("deepseek", "deepseek-chat", mock_credential_manager)
        assert model is not None
        mock_chat_openai.assert_called_once()
        call_args = mock_chat_openai.call_args
        assert (
            call_args.kwargs["api_key"].get_secret_value()
            == SecretStr("test_key").get_secret_value()
        )
        assert api_key is not None
        assert call_args.kwargs["model"] == "deepseek-chat"


def test_configure_model_openai(mock_credential_manager):
    with patch(
        "local_operator.model.configure.ChatOpenAI", return_value=MagicMock()
    ) as mock_chat_openai:
        model, api_key = configure_model("openai", "gpt-4", mock_credential_manager)
        assert model is not None
        mock_chat_openai.assert_called_once()
        call_args = mock_chat_openai.call_args
        assert (
            call_args.kwargs["api_key"].get_secret_value()
            == SecretStr("test_key").get_secret_value()
        )
        assert api_key is not None
        assert call_args.kwargs["model"] == "gpt-4"


def test_configure_model_ollama(mock_credential_manager):
    with patch(
        "local_operator.model.configure.ChatOllama", return_value=MagicMock()
    ) as mock_chat_ollama:
        model, api_key = configure_model("ollama", "llama2", mock_credential_manager)
        assert model is not None
        mock_chat_ollama.assert_called_once()
        call_args = mock_chat_ollama.call_args
        assert call_args.kwargs["model"] == "llama2"


def test_configure_model_noop(mock_credential_manager):
    model, api_key = configure_model("noop", "noop", mock_credential_manager)
    assert isinstance(model, ChatNoop)


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
        model, api_key = configure_model("deepseek", "deepseek-chat", credential_manager)
        assert model is not None
        mock_chat_openai.assert_called_once()
        call_args = mock_chat_openai.call_args
        assert (
            call_args.kwargs["api_key"].get_secret_value()
            == SecretStr("prompted_key").get_secret_value()
        )
        assert api_key is not None


def test_configure_model_anthropic(mock_credential_manager):
    with patch(
        "local_operator.model.configure.ChatAnthropic", return_value=MagicMock()
    ) as mock_chat_anthropic:
        model, api_key = configure_model("anthropic", "claude-x", mock_credential_manager)
        assert model is not None
        mock_chat_anthropic.assert_called_once()
        call_args = mock_chat_anthropic.call_args
        assert (
            call_args.kwargs["api_key"].get_secret_value()
            == SecretStr("test_key").get_secret_value()
        )
        assert api_key is not None
        assert call_args.kwargs["model_name"] == "claude-x"
        assert call_args.kwargs["temperature"] == 0.3
        assert call_args.kwargs["timeout"] is None
        assert call_args.kwargs["stop"] is None


def test_configure_model_anthropic_default(mock_credential_manager):
    with patch(
        "local_operator.model.configure.ChatAnthropic", return_value=MagicMock()
    ) as mock_chat_anthropic:
        model, api_key = configure_model("anthropic", "", mock_credential_manager)
        assert model is not None
        call_args = mock_chat_anthropic.call_args
        assert call_args.kwargs["model_name"] == "claude-3-5-sonnet-latest"
        assert api_key is not None


def test_configure_model_anthropic_fallback():
    credential_manager = MagicMock(spec=CredentialManager)
    credential_manager.get_credential = MagicMock(return_value=None)
    credential_manager.prompt_for_credential = MagicMock(
        return_value=SecretStr("fallback_anthropic_key")
    )
    with patch(
        "local_operator.model.configure.ChatAnthropic", return_value=MagicMock()
    ) as mock_chat_anthropic:
        model, api_key = configure_model("anthropic", "claude-x", credential_manager)
        assert model is not None
        mock_chat_anthropic.assert_called_once()
        call_args = mock_chat_anthropic.call_args
        assert (
            call_args.kwargs["api_key"].get_secret_value()
            == SecretStr("fallback_anthropic_key").get_secret_value()
        )
        assert api_key is not None


def test_configure_model_kimi_default(mock_credential_manager):
    with patch(
        "local_operator.model.configure.ChatOpenAI", return_value=MagicMock()
    ) as mock_chat_openai:
        model, api_key = configure_model("kimi", "", mock_credential_manager)
        assert model is not None
        call_args = mock_chat_openai.call_args
        assert call_args.kwargs["model"] == "moonshot-v1-32k"
        assert call_args.kwargs["base_url"] == "https://api.moonshot.cn/v1"
        assert api_key is not None


def test_configure_model_kimi_explicit(mock_credential_manager):
    with patch(
        "local_operator.model.configure.ChatOpenAI", return_value=MagicMock()
    ) as mock_chat_openai:
        model, api_key = configure_model("kimi", "custom-kimi-model", mock_credential_manager)
        assert model is not None
        call_args = mock_chat_openai.call_args
        assert call_args.kwargs["model"] == "custom-kimi-model"
        assert api_key is not None


def test_configure_model_kimi_fallback():
    credential_manager = MagicMock(spec=CredentialManager)
    credential_manager.get_credential = MagicMock(return_value="")
    credential_manager.prompt_for_credential = MagicMock(
        return_value=SecretStr("fallback_kimi_key")
    )
    with patch(
        "local_operator.model.configure.ChatOpenAI", return_value=MagicMock()
    ) as mock_chat_openai:
        model, api_key = configure_model("kimi", "custom-kimi-model", credential_manager)
        assert model is not None
        call_args = mock_chat_openai.call_args
        assert (
            call_args.kwargs["api_key"].get_secret_value()
            == SecretStr("fallback_kimi_key").get_secret_value()
        )
        assert api_key is not None
        assert call_args.kwargs["model"] == "custom-kimi-model"


def test_configure_model_alibaba_default(mock_credential_manager):
    with patch(
        "local_operator.model.configure.ChatOpenAI", return_value=MagicMock()
    ) as mock_chat_openai:
        model, api_key = configure_model("alibaba", "", mock_credential_manager)
        assert model is not None
        call_args = mock_chat_openai.call_args
        assert call_args.kwargs["model"] == "qwen-plus"
        assert (
            call_args.kwargs["base_url"] == "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        )
        assert api_key is not None


def test_configure_model_alibaba_fallback():
    credential_manager = MagicMock(spec=CredentialManager)
    credential_manager.get_credential = MagicMock(return_value=None)
    credential_manager.prompt_for_credential = MagicMock(
        return_value=SecretStr("fallback_alibaba_key")
    )
    with patch(
        "local_operator.model.configure.ChatOpenAI", return_value=MagicMock()
    ) as mock_chat_openai:
        model, api_key = configure_model("alibaba", "custom-alibaba-model", credential_manager)
        assert model is not None
        call_args = mock_chat_openai.call_args
        assert (
            call_args.kwargs["api_key"].get_secret_value()
            == SecretStr("fallback_alibaba_key").get_secret_value()
        )
        assert api_key is not None
        assert call_args.kwargs["model"] == "custom-alibaba-model"


def test_configure_model_openai_fallback():
    credential_manager = MagicMock(spec=CredentialManager)
    credential_manager.get_credential = MagicMock(return_value="")
    credential_manager.prompt_for_credential = MagicMock(
        return_value=SecretStr("fallback_openai_key")
    )
    with patch(
        "local_operator.model.configure.ChatOpenAI", return_value=MagicMock()
    ) as mock_chat_openai:
        model, api_key = configure_model("openai", "custom-openai-model", credential_manager)
        assert model is not None
        call_args = mock_chat_openai.call_args
        assert (
            call_args.kwargs["api_key"].get_secret_value()
            == SecretStr("fallback_openai_key").get_secret_value()
        )
        assert api_key is not None
        assert call_args.kwargs["model"] == "custom-openai-model"


def test_configure_model_ollama_missing_model(mock_credential_manager):
    with pytest.raises(ValueError) as exc_info:
        configure_model("ollama", "", mock_credential_manager)
    assert "Model is required for ollama hosting" in str(exc_info.value)


def test_configure_model_google_default(mock_credential_manager):
    with patch(
        "local_operator.model.configure.ChatGoogleGenerativeAI", return_value=MagicMock()
    ) as mock_chat_google:
        model, api_key = configure_model("google", "", mock_credential_manager)
        assert model is not None
        call_args = mock_chat_google.call_args
        assert call_args.kwargs["model"] == "gemini-2.0-flash"
        assert api_key is not None


def test_configure_model_google_fallback():
    credential_manager = MagicMock(spec=CredentialManager)
    credential_manager.get_credential = MagicMock(return_value=None)
    credential_manager.prompt_for_credential = MagicMock(
        return_value=SecretStr("fallback_google_key")
    )
    with patch(
        "local_operator.model.configure.ChatGoogleGenerativeAI", return_value=MagicMock()
    ) as mock_chat_google:
        model, api_key = configure_model("google", "custom-google-model", credential_manager)
        assert model is not None
        call_args = mock_chat_google.call_args
        assert (
            call_args.kwargs["api_key"].get_secret_value()
            == SecretStr("fallback_google_key").get_secret_value()
        )
        assert api_key is not None
        assert call_args.kwargs["model"] == "custom-google-model"


def test_configure_model_mistral_default(mock_credential_manager):
    with patch(
        "local_operator.model.configure.ChatOpenAI", return_value=MagicMock()
    ) as mock_chat_openai:
        model, api_key = configure_model("mistral", "", mock_credential_manager)
        assert model is not None
        call_args = mock_chat_openai.call_args
        assert call_args.kwargs["model"] == "mistral-large-latest"
        assert call_args.kwargs["base_url"] == "https://api.mistral.ai/v1"
        assert api_key is not None


def test_configure_model_mistral_fallback():
    credential_manager = MagicMock(spec=CredentialManager)
    credential_manager.get_credential = MagicMock(return_value=None)
    credential_manager.prompt_for_credential = MagicMock(
        return_value=SecretStr("fallback_mistral_key")
    )
    with patch(
        "local_operator.model.configure.ChatOpenAI", return_value=MagicMock()
    ) as mock_chat_openai:
        model, api_key = configure_model("mistral", "custom-mistral-model", credential_manager)
        assert model is not None
        call_args = mock_chat_openai.call_args
        assert (
            call_args.kwargs["api_key"].get_secret_value()
            == SecretStr("fallback_mistral_key").get_secret_value()
        )
        assert api_key is not None
        assert call_args.kwargs["model"] == "custom-mistral-model"


def test_calculate_cost() -> None:
    """Test that the calculate_cost function works correctly."""
    model_info = ModelInfo(input_price=1, output_price=2)
    input_tokens = 1000
    output_tokens = 2000
    expected_cost = (input_tokens / 1_000_000) * model_info.input_price + (
        output_tokens / 1_000_000
    ) * model_info.output_price
    assert calculate_cost(model_info, input_tokens, output_tokens) == pytest.approx(expected_cost)

    # Test with zero tokens
    assert calculate_cost(model_info, 0, 0) == 0.0
