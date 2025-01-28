from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from local_operator.credentials import CredentialManager
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
    assert model is None


def test_configure_model_invalid_hosting(mock_credential_manager):
    with pytest.raises(ValueError) as exc_info:
        configure_model("invalid", "model", mock_credential_manager)
    assert "Unsupported hosting platform: invalid" in str(exc_info.value)


def test_configure_model_missing_hosting(mock_credential_manager):
    with pytest.raises(ValueError) as exc_info:
        configure_model("", "model", mock_credential_manager)
    assert "Hosting is required" in str(exc_info.value)
