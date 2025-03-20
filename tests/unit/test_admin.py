"""
Unit tests for admin tools.

These tests cover functionalities such as agent creation from conversation,
training data saving, agent info lookup, and configuration handling.
Mock implementations are used throughout in order to avoid side effects and to
isolate behaviors.
"""

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from local_operator.admin import (
    AgentRegistry,
    LocalCodeExecutor,
    add_admin_tools,
    create_agent_from_conversation_tool,
    create_agent_tool,
    delete_agent_tool,
    edit_agent_tool,
    get_agent_info_tool,
    get_config_tool,
    list_agent_info_tool,
    save_agent_training_tool,
    save_conversation_raw_json_tool,
    update_config_tool,
)
from local_operator.agents import AgentEditFields
from local_operator.config import Config, ConfigManager
from local_operator.tools import ToolRegistry
from local_operator.types import (
    CodeExecutionResult,
    ConversationRecord,
    ConversationRole,
    ProcessResponseStatus,
)


@pytest.fixture
def temp_agents_dir(tmp_path: Path) -> Path:
    dir_path = tmp_path / "agents_test"
    dir_path.mkdir()
    return dir_path


@pytest.fixture
def temp_config_dir(tmp_path: Path) -> Path:
    dir_path = tmp_path / "config_test"
    dir_path.mkdir()
    return dir_path


@pytest.fixture
def mock_model():
    model = AsyncMock()
    model.ainvoke = AsyncMock()
    return model


@pytest.fixture
def executor(mock_model):
    agent = MagicMock()
    agent.id = "test_agent"
    agent.name = "Test Agent"
    agent.version = "1.0.0"
    agent.security_prompt = ""
    return LocalCodeExecutor(mock_model, agent=agent)


@pytest.fixture
def agent_registry(temp_agents_dir: Path):
    return AgentRegistry(temp_agents_dir)


@pytest.fixture
def tool_registry():
    registry = ToolRegistry()
    return registry


@pytest.fixture
def config_manager(temp_config_dir: Path) -> ConfigManager:
    manager = ConfigManager(config_dir=temp_config_dir)
    manager.update_config(
        {
            "conversation_length": 5,
            "detail_length": 3,
            "hosting": "test_host",
            "model_name": "test_model",
        }
    )
    return manager


def test_create_agent_from_conversation_no_user_messages(
    temp_agents_dir: Path, executor: LocalCodeExecutor, agent_registry: AgentRegistry
) -> None:
    """
    Test creating an agent from a conversation that contains no user messages.
    The expected saved conversation history should be empty.
    """
    executor.conversation_history = [
        ConversationRecord(role=ConversationRole.SYSTEM, content="Initial prompt")
    ]
    create_tool = create_agent_from_conversation_tool(executor, agent_registry)
    agent_name = "TestAgent"
    new_agent = create_tool(agent_name)
    saved_history = agent_registry.load_agent_state(new_agent.id)
    assert (
        saved_history.conversation == []
    ), f"Expected empty conversation history, got {saved_history}"
    assert (
        saved_history.execution_history == []
    ), f"Expected empty execution history, got {saved_history}"


def test_create_agent_from_conversation_with_user_messages(
    executor: LocalCodeExecutor, agent_registry: AgentRegistry
) -> None:
    """
    Test creating an agent from a conversation that contains multiple user messages.
    Verifies that the conversation history is truncated based on the cutoff index before
    the last user message.
    """
    conversation_history = [
        ConversationRecord(role=ConversationRole.SYSTEM, content="Initial prompt"),
        ConversationRecord(role=ConversationRole.USER, content="First user message"),
        ConversationRecord(role=ConversationRole.USER, content="Second user message"),
        ConversationRecord(role=ConversationRole.ASSISTANT, content="Response"),
        ConversationRecord(role=ConversationRole.USER, content="Third user message"),
        ConversationRecord(role=ConversationRole.SYSTEM, content="System message"),
    ]
    executor.code_history = [
        CodeExecutionResult(
            id="test_code_execution_id",
            stdout="",
            stderr="",
            logging="",
            message="This is a test code execution result",
            code="print('Hello, world!')",
            formatted_print="Hello, world!",
            role=ConversationRole.ASSISTANT,
            status=ProcessResponseStatus.SUCCESS,
            files=[],
        )
    ]
    executor.conversation_history = conversation_history
    create_tool = create_agent_from_conversation_tool(executor, agent_registry)
    agent_name = "TestAgent2"
    new_agent = create_tool(agent_name)
    saved_history = agent_registry.load_agent_state(new_agent.id)
    expected_history = conversation_history[:4]
    expected_execution_history = [
        CodeExecutionResult(
            id="test_code_execution_id",
            stdout="",
            stderr="",
            logging="",
            message="This is a test code execution result",
            code="print('Hello, world!')",
            formatted_print="Hello, world!",
            role=ConversationRole.ASSISTANT,
            status=ProcessResponseStatus.SUCCESS,
            files=[],
        )
    ]

    assert (
        saved_history.conversation == expected_history
    ), f"Expected conversation history {expected_history}, got {saved_history}"
    assert (
        saved_history.execution_history == expected_execution_history
    ), f"Expected execution history {expected_execution_history}, got {saved_history}"


def test_save_agent_training_no_agent(
    executor: LocalCodeExecutor, agent_registry: AgentRegistry
) -> None:
    """
    Test saving agent training data when no current agent is set.
    Expect a ValueError to be raised with the appropriate message.
    """
    conversation_history = [ConversationRecord(role=ConversationRole.USER, content="User message")]
    executor.conversation_history = conversation_history
    executor.agent = None
    save_training_tool = save_agent_training_tool(executor, agent_registry)
    with pytest.raises(ValueError, match="No current agent set"):
        save_training_tool()


def test_save_agent_training_with_agent(
    executor: LocalCodeExecutor, agent_registry: AgentRegistry
) -> None:
    """
    Test saving agent training data when a current agent is available.
    The conversation history should be truncated correctly and the agent remains unchanged.
    """
    conversation_history = [
        ConversationRecord(role=ConversationRole.SYSTEM, content="System message"),
        ConversationRecord(role=ConversationRole.USER, content="User message 1"),
        ConversationRecord(role=ConversationRole.USER, content="User message 2"),
        ConversationRecord(role=ConversationRole.ASSISTANT, content="Assistant reply"),
    ]
    execution_history = [
        CodeExecutionResult(
            id="test_code_execution_id",
            stdout="",
            stderr="",
            logging="",
            message="This is a test code execution result",
            code="print('Hello, world!')",
            formatted_print="Hello, world!",
            role=ConversationRole.ASSISTANT,
            status=ProcessResponseStatus.SUCCESS,
            files=[],
        ),
        CodeExecutionResult(
            id="test_code_execution_id2",
            stdout="",
            stderr="",
            logging="",
            message="This is a second test code execution result",
            code="print('Lorem ipsum dolor sit amet!')",
            formatted_print="Lorem ipsum dolor sit amet!",
            role=ConversationRole.ASSISTANT,
            status=ProcessResponseStatus.SUCCESS,
            files=[],
        ),
    ]
    executor.conversation_history = conversation_history
    executor.code_history = execution_history
    agent = agent_registry.create_agent(
        AgentEditFields(
            name="TrainAgent",
            security_prompt="",
            hosting="",
            model="",
            description="",
            last_message="",
            temperature=0.7,
            top_p=1.0,
            top_k=None,
            max_tokens=2048,
            stop=None,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            seed=None,
            current_working_directory=None,
        )
    )
    executor.agent = agent
    save_training_tool = save_agent_training_tool(executor, agent_registry)
    updated_agent = save_training_tool()
    expected_history = conversation_history[:2]
    expected_execution_history = execution_history
    stored_history = agent_registry.load_agent_state(agent.id)
    assert (
        stored_history.conversation == expected_history
    ), f"Expected conversation history {expected_history}, got {stored_history}"
    assert (
        stored_history.execution_history == expected_execution_history
    ), f"Expected execution history {expected_execution_history}, got {stored_history}"
    assert updated_agent == agent, "The updated agent does not match the original agent."


def test_list_agent_info_without_id(agent_registry: AgentRegistry) -> None:
    """
    Test listing agent information without specifying an agent ID.
    All agents stored in the registry should be returned.
    """
    agent1 = agent_registry.create_agent(
        AgentEditFields(
            name="Agent1",
            security_prompt="prompt1",
            hosting="",
            model="",
            description="",
            last_message="",
            temperature=0.7,
            top_p=1.0,
            top_k=None,
            max_tokens=2048,
            stop=None,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            seed=None,
            current_working_directory=None,
        )
    )
    agent2 = agent_registry.create_agent(
        AgentEditFields(
            name="Agent2",
            security_prompt="prompt2",
            hosting="",
            model="",
            description="",
            last_message="",
            temperature=0.7,
            top_p=1.0,
            top_k=None,
            max_tokens=2048,
            stop=None,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            seed=None,
            current_working_directory=None,
        )
    )
    list_tool = list_agent_info_tool(agent_registry)
    agents_list = list_tool(None)
    assert len(agents_list) == 2, f"Expected 2 agents, got {len(agents_list)}"
    agent_ids = {agent1.id, agent2.id}
    returned_ids = {agent.id for agent in agents_list}
    assert agent_ids == returned_ids, f"Expected agent ids {agent_ids}, got {returned_ids}"


def test_list_agent_info_with_id(agent_registry: AgentRegistry) -> None:
    """
    Test retrieving agent information for a specific agent ID.
    Only the desired agent should be returned.
    """
    agent = agent_registry.create_agent(
        AgentEditFields(
            name="AgentX",
            security_prompt="promptX",
            hosting="",
            model="",
            description="",
            last_message="",
            temperature=0.7,
            top_p=1.0,
            top_k=None,
            max_tokens=2048,
            stop=None,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            seed=None,
            current_working_directory=None,
        )
    )
    list_tool = list_agent_info_tool(agent_registry)
    agent_list = list_tool(agent.id)
    assert len(agent_list) == 1, f"Expected 1 agent, got {len(agent_list)}"
    assert agent_list[0].id == agent.id, f"Expected agent id {agent.id}, got {agent_list[0].id}"


def test_create_agent_tool(agent_registry: AgentRegistry) -> None:
    """
    Test the create_agent_tool function to ensure it creates an agent with the proper details.
    """
    create_tool = create_agent_tool(agent_registry)
    agent = create_tool("NewAgent", "secure")
    assert agent.name == "NewAgent", f"Expected agent name 'NewAgent', got {agent.name}"
    assert (
        agent.security_prompt == "secure"
    ), f"Expected security prompt 'secure', got {agent.security_prompt}"
    assert agent.id in agent_registry._agents, f"Agent id {agent.id} not found in registry"


def test_edit_agent_tool(agent_registry: AgentRegistry) -> None:
    """
    Test the edit_agent_tool to verify that an agent's name and security prompt update correctly.
    """
    agent = agent_registry.create_agent(
        AgentEditFields(
            name="OldName",
            security_prompt="old",
            hosting="",
            model="",
            description="",
            last_message="",
            temperature=0.7,
            top_p=1.0,
            top_k=None,
            max_tokens=2048,
            stop=None,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            seed=None,
            current_working_directory=None,
        )
    )
    edit_tool = edit_agent_tool(agent_registry)
    updated_agent = edit_tool(
        agent.id,
        AgentEditFields(
            name="NewName",
            security_prompt="new",
            hosting="",
            model="",
            description="",
            last_message="",
            temperature=0.7,
            top_p=1.0,
            top_k=None,
            max_tokens=2048,
            stop=None,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            seed=None,
            current_working_directory=None,
        ),
    )
    assert updated_agent is not None, "edit_agent_tool returned None"
    assert updated_agent.name == "NewName", f"Expected 'NewName', got {updated_agent.name}"
    assert (
        updated_agent.security_prompt == "new"
    ), f"Expected security prompt 'new', got {updated_agent.security_prompt}"


def test_delete_agent_tool(agent_registry: AgentRegistry) -> None:
    """
    Test the delete_agent_tool to confirm an agent is removed from the registry.
    """
    agent = agent_registry.create_agent(
        AgentEditFields(
            name="ToDelete",
            security_prompt="",
            hosting="",
            model="",
            description="",
            last_message="",
            temperature=0.7,
            top_p=1.0,
            top_k=None,
            max_tokens=2048,
            stop=None,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            seed=None,
            current_working_directory=None,
        )
    )
    delete_tool = delete_agent_tool(agent_registry)
    delete_tool(agent.id)
    with pytest.raises(KeyError, match=rf"Agent with id {agent.id} not found"):
        agent_registry.get_agent(agent.id)


def test_get_agent_info_tool_without_id(agent_registry: AgentRegistry) -> None:
    """
    Test get_agent_info_tool returns all agents when no id is provided.
    """
    agent_registry.create_agent(
        AgentEditFields(
            name="Agent1",
            security_prompt="",
            hosting="",
            model="",
            description="",
            last_message="",
            temperature=0.7,
            top_p=1.0,
            top_k=None,
            max_tokens=2048,
            stop=None,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            seed=None,
            current_working_directory=None,
        )
    )
    agent_registry.create_agent(
        AgentEditFields(
            name="Agent2",
            security_prompt="",
            hosting="",
            model="",
            description="",
            last_message="",
            temperature=0.7,
            top_p=1.0,
            top_k=None,
            max_tokens=2048,
            stop=None,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            seed=None,
            current_working_directory=None,
        )
    )
    get_tool = get_agent_info_tool(agent_registry)
    agents = get_tool(None)
    assert len(agents) == 2, f"Expected 2 agents, got {len(agents)}"


def test_get_agent_info_tool_with_id(agent_registry: AgentRegistry) -> None:
    """
    Test get_agent_info_tool returns information for the specified agent.
    """
    agent = agent_registry.create_agent(
        AgentEditFields(
            name="AgentSingle",
            security_prompt="",
            hosting="",
            model="",
            description="",
            last_message="",
            temperature=0.7,
            top_p=1.0,
            top_k=None,
            max_tokens=2048,
            stop=None,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            seed=None,
            current_working_directory=None,
        )
    )
    get_tool = get_agent_info_tool(agent_registry)
    result = get_tool(agent.id)
    assert len(result) == 1, f"Expected 1 agent, got {len(result)}"
    assert result[0].id == agent.id, f"Expected agent id {agent.id}, got {result[0].id}"


def test_save_conversation_tool(tmp_path: Any, executor: LocalCodeExecutor) -> None:
    """
    Test the save_conversation_tool to verify that the conversation history is written to a file.
    """
    file_path = tmp_path / "conversation.json"
    conversation_history = [
        ConversationRecord(role=ConversationRole.USER, content="Hello"),
        ConversationRecord(role=ConversationRole.ASSISTANT, content="Hi there!"),
    ]
    executor.conversation_history = conversation_history
    save_tool = save_conversation_raw_json_tool(executor)
    save_tool(str(file_path))
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list), "Saved JSON data is not a list"
    for msg in data:
        assert isinstance(msg, dict), f"Message {msg} is not a dict"
        assert "role" in msg, f"Message {msg} missing key 'role'"
        assert "content" in msg, f"Message {msg} missing key 'content'"


def test_get_config_tool(config_manager: ConfigManager) -> None:
    """
    Test the get_config_tool to verify it returns the correct configuration.
    """
    get_tool = get_config_tool(config_manager)
    config = get_tool()
    assert isinstance(config, Config), "Configuration is not a Config object"
    assert config.get_value("conversation_length") == 5
    assert config.get_value("detail_length") == 3
    assert config.get_value("hosting") == "test_host"
    assert config.get_value("model_name") == "test_model"


def test_update_config_tool(config_manager: ConfigManager) -> None:
    """
    Test the update_config_tool to check that configuration updates are applied correctly.
    """
    update_tool = update_config_tool(config_manager)
    update_tool({"new_key": "new_value"})
    config = config_manager.get_config()
    assert (
        config.get_value("new_key") == "new_value"
    ), f"Expected 'new_value' for 'new_key', got {config.get_value('new_key')}"


def test_add_admin_tools(
    tool_registry: ToolRegistry,
    executor: LocalCodeExecutor,
    agent_registry: AgentRegistry,
    config_manager: ConfigManager,
) -> None:
    """
    Test add_admin_tools to ensure that all expected admin tools
    are registered in the tool registry.
    """
    add_admin_tools(tool_registry, executor, agent_registry, config_manager)
    expected_tools = {
        "create_agent_from_conversation",
        "edit_agent",
        "delete_agent",
        "get_agent_info",
        "save_conversation_raw_json",
        "get_config",
        "update_config",
        "save_agent_training",
        "open_agents_config",
        "open_settings_config",
        "save_conversation_history_to_notebook",
    }
    tools_set = set(tool_registry._tools.keys())
    # Check that all expected tools are present, but allow for additional builtin tools
    assert expected_tools.issubset(tools_set), (
        f"Missing expected tools. Expected subset {expected_tools}, "
        f"but registry only contains {tools_set}"
    )
