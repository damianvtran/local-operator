from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from local_operator.agents import AgentConversation, AgentData
from local_operator.cli import (
    agents_create_command,
    agents_delete_command,
    agents_list_command,
    build_cli_parser,
    config_create_command,
    credential_delete_command,
    credential_update_command,
    main,
    serve_command,
)
from local_operator.model.configure import ModelConfiguration


@pytest.fixture
def mock_agent():
    mock_agent = AgentData(
        id="test-id",
        name="TestAgent",
        created_date=datetime.now(),
        version="1.0.0",
        security_prompt="",
        hosting="test-hosting",
        model="test-model",
    )
    return mock_agent


@pytest.fixture
def mock_agent_registry(mock_agent):
    registry = MagicMock()
    registry.create_agent = MagicMock()
    registry.delete_agent = MagicMock()
    registry.list_agents = MagicMock()
    registry.get_agent_by_name.return_value = mock_agent
    registry.load_agent_conversation.return_value = AgentConversation(
        version="",
        conversation=[],
        execution_history=[],
    )
    return registry


@pytest.fixture
def mock_credential_manager():
    manager = MagicMock()
    manager.prompt_for_credential = MagicMock()
    return manager


@pytest.fixture
def mock_config_manager():
    manager = MagicMock()
    manager.get_config_value = MagicMock()
    manager._write_config = MagicMock()
    return manager


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.info = MagicMock()
    return model


@pytest.fixture
def mock_operator():
    operator = MagicMock()
    operator.chat = MagicMock()
    return operator


def test_build_cli_parser():
    parser = build_cli_parser()

    # Test basic arguments
    args = parser.parse_args(["--hosting", "deepseek", "--model", "deepseek-chat"])
    assert args.hosting == "deepseek"
    assert args.model == "deepseek-chat"
    assert not args.debug

    # Test debug flag
    args = parser.parse_args(["--debug", "--hosting", "openai"])
    assert args.debug
    assert args.hosting == "openai"

    # Test credential subcommand
    args = parser.parse_args(["credential", "update", "OPENAI_API_KEY"])
    assert args.subcommand == "credential"
    assert args.credential_command == "update"
    assert args.key == "OPENAI_API_KEY"

    # Test credential delete subcommand
    args = parser.parse_args(["credential", "delete", "OPENAI_API_KEY"])
    assert args.subcommand == "credential"
    assert args.credential_command == "delete"
    assert args.key == "OPENAI_API_KEY"

    # Test config create subcommand
    args = parser.parse_args(["config", "create"])
    assert args.subcommand == "config"
    assert args.config_command == "create"

    # Test config open subcommand
    args = parser.parse_args(["config", "open"])
    assert args.subcommand == "config"
    assert args.config_command == "open"

    # Test config edit subcommand
    args = parser.parse_args(["config", "edit", "hosting", "openai"])
    assert args.subcommand == "config"
    assert args.config_command == "edit"
    assert args.key == "hosting"
    assert args.value == "openai"

    # Test config list subcommand
    args = parser.parse_args(["config", "list"])
    assert args.subcommand == "config"
    assert args.config_command == "list"

    # Test serve subcommand
    args = parser.parse_args(["serve", "--host", "localhost", "--port", "8000"])
    assert args.subcommand == "serve"
    assert args.host == "localhost"
    assert args.port == 8000
    assert not args.reload


def test_credential_update_command(mock_credential_manager):
    with patch("local_operator.cli.CredentialManager", return_value=mock_credential_manager):
        args = MagicMock()
        args.key = "TEST_API_KEY"

        result = credential_update_command(args)

        mock_credential_manager.prompt_for_credential.assert_called_once_with(
            "TEST_API_KEY", reason="update requested"
        )
        assert result == 0


def test_credential_delete_command(mock_credential_manager):
    with patch("local_operator.cli.CredentialManager", return_value=mock_credential_manager):
        args = MagicMock()
        args.key = "TEST_API_KEY"

        result = credential_delete_command(args)

        mock_credential_manager.set_credential.assert_called_once_with("TEST_API_KEY", "")
        assert result == 0


def test_config_create_command(mock_config_manager):
    with patch("local_operator.cli.ConfigManager", return_value=mock_config_manager):
        result = config_create_command()

        mock_config_manager._write_config.assert_called_once()
        assert result == 0


def test_serve_command():
    with patch("local_operator.cli.uvicorn.run") as mock_run:
        result = serve_command("localhost", 8000, False)

        mock_run.assert_called_once_with(
            "local_operator.server.app:app", host="localhost", port=8000, reload=False
        )
        assert result == 0


def test_main_success(mock_operator, mock_agent_registry, mock_model):
    with (
        patch("local_operator.cli.ConfigManager") as mock_config_manager_cls,
        patch("local_operator.cli.CredentialManager"),
        patch(
            "local_operator.cli.configure_model",
            return_value=ModelConfiguration(
                hosting="deepseek",
                name="deepseek-chat",
                instance=mock_model,
                info=mock_model.info,
                api_key=SecretStr("test_key"),
            ),
        ) as mock_configure_model,
        patch("local_operator.cli.LocalCodeExecutor"),
        patch("local_operator.cli.Operator", return_value=mock_operator) as mock_operator_cls,
        patch("local_operator.cli.AgentRegistry", return_value=mock_agent_registry),
        patch("local_operator.cli.asyncio.run") as mock_asyncio_run,
    ):

        mock_config_manager = mock_config_manager_cls.return_value
        mock_config_manager.get_config_value.side_effect = lambda key, default=None: {
            "hosting": "deepseek",
            "model_name": "deepseek-chat",
            "detail_length": 10,
            "max_learnings_history": 50,
            "max_conversation_history": 100,
            "auto_save_conversation": True,
        }.get(key, default)

        with patch("sys.argv", ["program", "--hosting", "deepseek", "--agent", "test-agent"]):
            result = main()

            assert result == 0
            mock_configure_model.assert_called_once()
            mock_operator_cls.assert_called_once()
            mock_agent_registry.get_agent_by_name.assert_called_once_with("test-agent")
            mock_agent_registry.load_agent_conversation.assert_called_once_with("test-id")
            mock_asyncio_run.assert_called_once_with(mock_operator.chat())


def test_main_model_not_found():
    with (
        patch("local_operator.cli.ConfigManager") as mock_config_manager_cls,
        patch("local_operator.cli.CredentialManager"),
        patch("local_operator.cli.configure_model", return_value=None),
    ):
        mock_config_manager = mock_config_manager_cls.return_value
        mock_config_manager.get_config_value.side_effect = ["invalid", "invalid"]

        with patch("sys.argv", ["program", "--hosting", "openai"]):
            result = main()
            assert result == -1


def test_main_exception():
    with patch("local_operator.cli.ConfigManager", side_effect=Exception("Test error")):
        with patch("sys.argv", ["program"]):
            result = main()
            assert result == -1


def test_agents_list_command_no_agents(mock_agent_registry):
    mock_agent_registry.list_agents.return_value = []
    args = MagicMock()
    result = agents_list_command(args, mock_agent_registry)
    assert result == 0


def test_agents_list_command_with_agents(mock_agent_registry):
    mock_agents = [
        AgentData(
            id="1",
            name="Agent1",
            created_date=datetime.now(),
            version="1.0.0",
            security_prompt="",
            hosting="test-hosting",
            model="test-model",
        ),
        AgentData(
            id="2",
            name="Agent2",
            created_date=datetime.now(),
            version="1.0.0",
            security_prompt="",
            hosting="test-hosting",
            model="test-model",
        ),
    ]
    mock_agent_registry.list_agents.return_value = mock_agents
    args = MagicMock()
    args.page = 1
    args.perpage = 10

    result = agents_list_command(args, mock_agent_registry)
    assert result == 0


def test_agents_create_command_with_name(mock_agent_registry):
    mock_agent = AgentData(
        id="test-id",
        name="TestAgent",
        created_date=datetime.now(),
        version="1.0.0",
        security_prompt="",
        hosting="test-hosting",
        model="test-model",
    )
    mock_agent_registry.create_agent.return_value = mock_agent
    result = agents_create_command("TestAgent", mock_agent_registry)
    assert result == 0
    mock_agent_registry.create_agent.assert_called_once()


def test_agents_create_command_empty_name(mock_agent_registry):
    with patch("builtins.input", return_value=""):
        result = agents_create_command("", mock_agent_registry)
        assert result == -1


def test_agents_create_command_keyboard_interrupt(mock_agent_registry):
    with patch("builtins.input", side_effect=KeyboardInterrupt):
        result = agents_create_command("", mock_agent_registry)
        assert result == -1


def test_agents_delete_command_success(mock_agent_registry):
    mock_agent = AgentData(
        id="test-id",
        name="TestAgent",
        created_date=datetime.now(),
        version="1.0.0",
        security_prompt="",
        hosting="test-hosting",
        model="test-model",
    )
    mock_agent_registry.list_agents.return_value = [mock_agent]
    result = agents_delete_command("TestAgent", mock_agent_registry)
    assert result == 0
    mock_agent_registry.delete_agent.assert_called_once_with("test-id")


def test_agents_delete_command_not_found(mock_agent_registry):
    mock_agent_registry.list_agents.return_value = []
    result = agents_delete_command("NonExistentAgent", mock_agent_registry)
    assert result == -1
    mock_agent_registry.delete_agent.assert_not_called()
