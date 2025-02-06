from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from local_operator.credentials import CredentialManager
from local_operator.mocks import ChatNoop
from local_operator.model import configure_model


@pytest.fixture
def mock_credential_manager():
    manager = MagicMock(spec=CredentialManager)
    manager.get_credential = MagicMock(return_value="test_key")
    manager.prompt_for_credential = MagicMock(return_value="test_key")
    return manager


def test_configure_model_deepseek(mock_credential_manager):
    with patch("local_operator.model.ChatOpenAI", return_value=MagicMock()) as mock_chat_openai:
        model = configure_model("deepseek", "deepseek-chat", mock_credential_manager)
        assert model is not None
        mock_chat_openai.assert_called_once()
        call_args = mock_chat_openai.call_args
        assert call_args.kwargs["api_key"] == SecretStr("test_key")
        assert call_args.kwargs["model"] == "deepseek-chat"


def test_configure_model_openai(mock_credential_manager):
    with patch("local_operator.model.ChatOpenAI", return_value=MagicMock()) as mock_chat_openai:
        model = configure_model("openai", "gpt-4", mock_credential_manager)
        assert model is not None
        mock_chat_openai.assert_called_once()
        call_args = mock_chat_openai.call_args
        assert call_args.kwargs["api_key"] == SecretStr("test_key")
        assert call_args.kwargs["model"] == "gpt-4"


def test_configure_model_ollama(mock_credential_manager):
    with patch("local_operator.model.ChatOllama", return_value=MagicMock()) as mock_chat_ollama:
        model = configure_model("ollama", "llama2", mock_credential_manager)
        assert model is not None
        mock_chat_ollama.assert_called_once()
        call_args = mock_chat_ollama.call_args
        assert call_args.kwargs["model"] == "llama2"


def test_configure_model_noop(mock_credential_manager):
    model = configure_model("noop", "noop", mock_credential_manager)
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
    # When get_credential returns None, the fallback via prompt_for_credential should be used.
    credential_manager = MagicMock(spec=CredentialManager)
    credential_manager.get_credential = MagicMock(return_value=None)
    credential_manager.prompt_for_credential = MagicMock(return_value="prompted_key")
    with patch("local_operator.model.ChatOpenAI", return_value=MagicMock()) as mock_chat_openai:
        model = configure_model("deepseek", "deepseek-chat", credential_manager)
        assert model is not None
        mock_chat_openai.assert_called_once()
        call_args = mock_chat_openai.call_args
        # Should use the prompted value
        assert call_args.kwargs["api_key"] == SecretStr("prompted_key")


def test_configure_model_anthropic(mock_credential_manager):
    # Test explicit model value for anthropic hosting.
    with patch(
        "local_operator.model.ChatAnthropic", return_value=MagicMock()
    ) as mock_chat_anthropic:
        model = configure_model("anthropic", "claude-x", mock_credential_manager)
        assert model is not None
        mock_chat_anthropic.assert_called_once()
        call_args = mock_chat_anthropic.call_args
        assert call_args.kwargs["api_key"] == SecretStr("test_key")
        assert call_args.kwargs["model_name"] == "claude-x"
        assert call_args.kwargs["temperature"] == 0.3
        assert call_args.kwargs["timeout"] is None
        assert call_args.kwargs["stop"] is None


def test_configure_model_anthropic_default(mock_credential_manager):
    # When an empty model string is passed, the default model should be applied.
    with patch(
        "local_operator.model.ChatAnthropic", return_value=MagicMock()
    ) as mock_chat_anthropic:
        model = configure_model("anthropic", "", mock_credential_manager)
        assert model is not None
        call_args = mock_chat_anthropic.call_args
        # Default should be "claude-3-5-sonnet-20240620"
        assert call_args.kwargs["model_name"] == "claude-3-5-sonnet-20240620"


def test_configure_model_anthropic_fallback():
    # Test that fallback for anthropic hosting uses prompt_for_credential when
    # get_credential returns None.
    credential_manager = MagicMock(spec=CredentialManager)
    credential_manager.get_credential = MagicMock(return_value=None)
    credential_manager.prompt_for_credential = MagicMock(return_value="fallback_anthropic_key")
    with patch(
        "local_operator.model.ChatAnthropic", return_value=MagicMock()
    ) as mock_chat_anthropic:
        model = configure_model("anthropic", "claude-x", credential_manager)
        assert model is not None
        mock_chat_anthropic.assert_called_once()
        call_args = mock_chat_anthropic.call_args
        assert call_args.kwargs["api_key"] == SecretStr("fallback_anthropic_key")


def test_configure_model_kimi_default(mock_credential_manager):
    # Test that for kimi hosting, when no model is provided, the default is "moonshot-v1-32k"
    with patch("local_operator.model.ChatOpenAI", return_value=MagicMock()) as mock_chat_openai:
        model = configure_model("kimi", "", mock_credential_manager)
        assert model is not None
        call_args = mock_chat_openai.call_args
        assert call_args.kwargs["model"] == "moonshot-v1-32k"
        assert call_args.kwargs["base_url"] == "https://api.moonshot.cn/v1"


def test_configure_model_kimi_explicit(mock_credential_manager):
    # Test that an explicit model override works for kimi hosting.
    with patch("local_operator.model.ChatOpenAI", return_value=MagicMock()) as mock_chat_openai:
        model = configure_model("kimi", "custom-kimi-model", mock_credential_manager)
        assert model is not None
        call_args = mock_chat_openai.call_args
        assert call_args.kwargs["model"] == "custom-kimi-model"


def test_configure_model_kimi_fallback():
    # Test fallback for kimi hosting when credentials are missing.
    credential_manager = MagicMock(spec=CredentialManager)
    credential_manager.get_credential = MagicMock(return_value="")
    credential_manager.prompt_for_credential = MagicMock(return_value="fallback_kimi_key")
    with patch("local_operator.model.ChatOpenAI", return_value=MagicMock()) as mock_chat_openai:
        model = configure_model("kimi", "custom-kimi-model", credential_manager)
        assert model is not None
        call_args = mock_chat_openai.call_args
        assert call_args.kwargs["api_key"] == SecretStr("fallback_kimi_key")
        assert call_args.kwargs["model"] == "custom-kimi-model"


def test_configure_model_alibaba_default(mock_credential_manager):
    # Test that for alibaba hosting, when no model is provided, the default becomes "qwen-plus"
    with patch("local_operator.model.ChatOpenAI", return_value=MagicMock()) as mock_chat_openai:
        model = configure_model("alibaba", "", mock_credential_manager)
        assert model is not None
        call_args = mock_chat_openai.call_args
        assert call_args.kwargs["model"] == "qwen-plus"
        assert (
            call_args.kwargs["base_url"] == "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        )


def test_configure_model_alibaba_fallback():
    # Test fallback for alibaba hosting when the initial API key is missing.
    credential_manager = MagicMock(spec=CredentialManager)
    credential_manager.get_credential = MagicMock(return_value=None)
    credential_manager.prompt_for_credential = MagicMock(return_value="fallback_alibaba_key")
    with patch("local_operator.model.ChatOpenAI", return_value=MagicMock()) as mock_chat_openai:
        model = configure_model("alibaba", "custom-alibaba-model", credential_manager)
        assert model is not None
        call_args = mock_chat_openai.call_args
        assert call_args.kwargs["api_key"] == SecretStr("fallback_alibaba_key")
        assert call_args.kwargs["model"] == "custom-alibaba-model"


def test_configure_model_openai_fallback():
    # Test openai fallback when get_credential returns an empty string.
    credential_manager = MagicMock(spec=CredentialManager)
    credential_manager.get_credential = MagicMock(return_value="")
    credential_manager.prompt_for_credential = MagicMock(return_value="fallback_openai_key")
    with patch("local_operator.model.ChatOpenAI", return_value=MagicMock()) as mock_chat_openai:
        model = configure_model("openai", "custom-openai-model", credential_manager)
        assert model is not None
        call_args = mock_chat_openai.call_args
        assert call_args.kwargs["api_key"] == SecretStr("fallback_openai_key")
        assert call_args.kwargs["model"] == "custom-openai-model"


def test_configure_model_ollama_missing_model(mock_credential_manager):
    # For ollama hosting, an empty model should raise an error.
    with pytest.raises(ValueError) as exc_info:
        configure_model("ollama", "", mock_credential_manager)
    assert "Model is required for ollama hosting" in str(exc_info.value)
