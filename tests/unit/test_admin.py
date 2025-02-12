"""
Unit tests for admin tools.

These tests cover functionalities such as agent creation from conversation,
training data saving, agent info lookup, and configuration handling.
Fake implementations are used throughout in order to avoid side effects and to
isolate behaviors.
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

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
    save_conversation_tool,
    update_config_tool,
)
from local_operator.agents import AgentData, AgentEditFields
from local_operator.config import ConfigManager
from local_operator.tools import ToolRegistry
from local_operator.types import ConversationRole


class FakeAgentRegistry(AgentRegistry):
    """
    Fake implementation of an agent registry for testing admin tools.
    """

    def __init__(self) -> None:
        self.agents: Dict[str, AgentData] = {}
        self.conversations: Dict[str, List[Dict[str, str]]] = {}

    def create_agent(self, agent_edit_metadata: AgentEditFields) -> AgentData:
        """
        Create a new agent using the provided metadata.
        Raises a ValueError if the agent name is missing.
        """
        if not agent_edit_metadata.name:
            raise ValueError("Agent name is required")
        agent_id = str(uuid.uuid4())
        new_agent = AgentData(
            id=agent_id,
            name=agent_edit_metadata.name,
            created_date=datetime.now(timezone.utc),
            version="1.0",
            security_prompt=(agent_edit_metadata.security_prompt or ""),
        )
        self.agents[agent_id] = new_agent
        return new_agent

    def save_agent_conversation(self, agent_id: str, conversation: List[Dict[str, str]]) -> None:
        """
        Save conversation history for an agent.
        A copy is stored to avoid side effects.
        """
        self.conversations[agent_id] = conversation.copy() if conversation else []

    def get_agent(self, agent_id: str) -> AgentData:
        """
        Retrieve an agent by its ID.
        Raises KeyError if not found.
        """
        if agent_id in self.agents:
            return self.agents[agent_id]
        raise KeyError(f"Agent with id {agent_id} not found")

    def list_agents(self) -> List[AgentData]:
        """
        List all agents.
        """
        return list(self.agents.values())

    def update_agent(self, agent_id: str, edit_fields: AgentEditFields) -> Optional[AgentData]:
        """
        Update agent details and return updated agent if exists.
        """
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            if edit_fields.name is not None:
                agent.name = edit_fields.name
            if edit_fields.security_prompt is not None:
                agent.security_prompt = edit_fields.security_prompt
            self.agents[agent_id] = agent
            return agent
        return None

    def delete_agent(self, agent_id: str) -> None:
        """
        Delete an agent and its conversation history.
        """
        self.agents.pop(agent_id, None)
        self.conversations.pop(agent_id, None)


class FakeExecutor(LocalCodeExecutor):
    """
    Fake executor for testing admin tools related to conversation history.
    """

    def __init__(
        self, conversation_history: List[Dict[str, str]], agent: Optional[AgentData] = None
    ) -> None:
        self.conversation_history = conversation_history
        self.agent = agent

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Return the current conversation history."""
        return self.conversation_history


class FakeConfigManager(ConfigManager):
    """
    Fake configuration manager for testing admin configuration tools.
    """

    def __init__(self) -> None:
        self._config: Dict[str, Any] = {"setting": "value"}

    def get_config(self) -> Dict[str, Any]:
        """Retrieve the current configuration."""
        return self._config

    def update_config(self, values: Dict[str, Any]) -> None:
        """Update configuration settings."""
        self._config.update(values)


class FakeToolRegistry(ToolRegistry):
    """
    Fake tool registry for testing tool registration.
    """

    def __init__(self) -> None:
        self._tools: Dict[str, Any] = {}

    def add_tool(self, name: str, tool_func: Any) -> None:
        """Register a tool function."""
        self._tools[name] = tool_func

    @property
    def tools(self) -> Dict[str, Any]:
        """Return the dictionary of registered tools."""
        return self._tools


def test_create_agent_from_conversation_no_user_messages() -> None:
    """
    Test creating an agent from a conversation that contains no user messages.
    The expected saved conversation history should be empty.
    """
    conversation_history = [{"role": ConversationRole.SYSTEM.value, "content": "Initial prompt"}]
    fake_executor = FakeExecutor(conversation_history)
    fake_registry = FakeAgentRegistry()
    create_tool = create_agent_from_conversation_tool(fake_executor, fake_registry)
    agent_name = "TestAgent"
    new_agent = create_tool(agent_name)
    saved_history = fake_registry.conversations.get(new_agent.id)
    assert saved_history == [], f"Expected empty conversation history, got {saved_history}"


def test_create_agent_from_conversation_with_user_messages() -> None:
    """
    Test creating an agent from a conversation that contains multiple user messages.
    Verifies that the conversation history is truncated based on the cutoff index before
    the last user message.
    """
    conversation_history = [
        {"role": ConversationRole.SYSTEM.value, "content": "Initial prompt"},
        {"role": ConversationRole.USER.value, "content": "First user message"},
        {"role": ConversationRole.USER.value, "content": "Second user message"},
        {"role": ConversationRole.ASSISTANT.value, "content": "Response"},
        {"role": ConversationRole.USER.value, "content": "Third user message"},
        {"role": ConversationRole.SYSTEM.value, "content": "System message"},
    ]
    fake_executor = FakeExecutor(conversation_history)
    fake_registry = FakeAgentRegistry()
    create_tool = create_agent_from_conversation_tool(fake_executor, fake_registry)
    agent_name = "TestAgent2"
    new_agent = create_tool(agent_name)
    saved_history = fake_registry.conversations.get(new_agent.id)
    expected_history = conversation_history[:4]
    assert (
        saved_history == expected_history
    ), f"Expected conversation history {expected_history}, got {saved_history}"


def test_save_agent_training_no_agent() -> None:
    """
    Test saving agent training data when no current agent is set.
    Expect a ValueError to be raised with the appropriate message.
    """
    conversation_history = [{"role": ConversationRole.USER.value, "content": "User message"}]
    fake_executor = FakeExecutor(conversation_history, agent=None)
    fake_registry = FakeAgentRegistry()
    save_training_tool = save_agent_training_tool(fake_executor, fake_registry)
    with pytest.raises(ValueError, match="No current agent set"):
        save_training_tool()


def test_save_agent_training_with_agent() -> None:
    """
    Test saving agent training data when a current agent is available.
    The conversation history should be truncated correctly and the agent remains unchanged.
    """
    conversation_history = [
        {"role": ConversationRole.SYSTEM.value, "content": "System message"},
        {"role": ConversationRole.USER.value, "content": "User message 1"},
        {"role": ConversationRole.USER.value, "content": "User message 2"},
        {"role": ConversationRole.ASSISTANT.value, "content": "Assistant reply"},
    ]
    fake_registry = FakeAgentRegistry()
    agent = fake_registry.create_agent(AgentEditFields(name="TrainAgent", security_prompt=""))
    fake_executor = FakeExecutor(conversation_history, agent=agent)
    save_training_tool = save_agent_training_tool(fake_executor, fake_registry)
    updated_agent = save_training_tool()
    expected_history = conversation_history[:2]
    stored_history = fake_registry.conversations.get(agent.id)
    assert (
        stored_history == expected_history
    ), f"Expected conversation history {expected_history}, got {stored_history}"
    assert updated_agent == agent, "The updated agent does not match the original agent."


def test_list_agent_info_without_id() -> None:
    """
    Test listing agent information without specifying an agent ID.
    All agents stored in the registry should be returned.
    """
    fake_registry = FakeAgentRegistry()
    agent1 = fake_registry.create_agent(AgentEditFields(name="Agent1", security_prompt="prompt1"))
    agent2 = fake_registry.create_agent(AgentEditFields(name="Agent2", security_prompt="prompt2"))
    list_tool = list_agent_info_tool(fake_registry)
    agents_list = list_tool(None)
    assert len(agents_list) == 2, f"Expected 2 agents, got {len(agents_list)}"
    agent_ids = {agent1.id, agent2.id}
    returned_ids = {agent.id for agent in agents_list}
    assert agent_ids == returned_ids, f"Expected agent ids {agent_ids}, got {returned_ids}"


def test_list_agent_info_with_id() -> None:
    """
    Test retrieving agent information for a specific agent ID.
    Only the desired agent should be returned.
    """
    fake_registry = FakeAgentRegistry()
    agent = fake_registry.create_agent(AgentEditFields(name="AgentX", security_prompt="promptX"))
    list_tool = list_agent_info_tool(fake_registry)
    agent_list = list_tool(agent.id)
    assert len(agent_list) == 1, f"Expected 1 agent, got {len(agent_list)}"
    assert agent_list[0].id == agent.id, f"Expected agent id {agent.id}, got {agent_list[0].id}"


def test_create_agent_tool() -> None:
    """
    Test the create_agent_tool function to ensure it creates an agent with the proper details.
    """
    fake_registry = FakeAgentRegistry()
    create_tool = create_agent_tool(fake_registry)
    agent = create_tool("NewAgent", "secure")
    assert agent.name == "NewAgent", f"Expected agent name 'NewAgent', got {agent.name}"
    assert (
        agent.security_prompt == "secure"
    ), f"Expected security prompt 'secure', got {agent.security_prompt}"
    assert agent.id in fake_registry.agents, f"Agent id {agent.id} not found in registry"


def test_edit_agent_tool() -> None:
    """
    Test the edit_agent_tool to verify that an agent's name and security prompt update correctly.
    """
    fake_registry = FakeAgentRegistry()
    agent = fake_registry.create_agent(AgentEditFields(name="OldName", security_prompt="old"))
    edit_tool = edit_agent_tool(fake_registry)
    updated_agent = edit_tool(agent.id, AgentEditFields(name="NewName", security_prompt="new"))
    assert updated_agent is not None, "edit_agent_tool returned None"
    assert updated_agent.name == "NewName", f"Expected 'NewName', got {updated_agent.name}"
    assert (
        updated_agent.security_prompt == "new"
    ), f"Expected security prompt 'new', got {updated_agent.security_prompt}"


def test_delete_agent_tool() -> None:
    """
    Test the delete_agent_tool to confirm an agent is removed from the registry.
    """
    fake_registry = FakeAgentRegistry()
    agent = fake_registry.create_agent(AgentEditFields(name="ToDelete", security_prompt=""))
    delete_tool = delete_agent_tool(fake_registry)
    delete_tool(agent.id)
    with pytest.raises(KeyError, match=rf"Agent with id {agent.id} not found"):
        fake_registry.get_agent(agent.id)


def test_get_agent_info_tool_without_id() -> None:
    """
    Test get_agent_info_tool returns all agents when no id is provided.
    """
    fake_registry = FakeAgentRegistry()
    fake_registry.create_agent(AgentEditFields(name="Agent1", security_prompt=""))
    fake_registry.create_agent(AgentEditFields(name="Agent2", security_prompt=""))
    get_tool = get_agent_info_tool(fake_registry)
    agents = get_tool(None)
    assert len(agents) == 2, f"Expected 2 agents, got {len(agents)}"


def test_get_agent_info_tool_with_id() -> None:
    """
    Test get_agent_info_tool returns information for the specified agent.
    """
    fake_registry = FakeAgentRegistry()
    agent = fake_registry.create_agent(AgentEditFields(name="AgentSingle", security_prompt=""))
    get_tool = get_agent_info_tool(fake_registry)
    result = get_tool(agent.id)
    assert len(result) == 1, f"Expected 1 agent, got {len(result)}"
    assert result[0].id == agent.id, f"Expected agent id {agent.id}, got {result[0].id}"


def test_save_conversation_tool(tmp_path: Any) -> None:
    """
    Test the save_conversation_tool to verify that the conversation history is written to a file.
    """
    file_path = tmp_path / "conversation.json"
    conversation_history = [
        {"role": ConversationRole.USER.value, "content": "Hello"},
        {"role": ConversationRole.ASSISTANT.value, "content": "Hi there!"},
    ]
    fake_executor = FakeExecutor(conversation_history)
    save_tool = save_conversation_tool(fake_executor)
    save_tool(str(file_path))
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list), "Saved JSON data is not a list"
    for msg in data:
        assert isinstance(msg, dict), f"Message {msg} is not a dict"
        assert "role" in msg, f"Message {msg} missing key 'role'"
        assert "content" in msg, f"Message {msg} missing key 'content'"


def test_get_config_tool() -> None:
    """
    Test the get_config_tool to verify it returns the correct configuration.
    """
    fake_config_manager = FakeConfigManager()
    get_tool = get_config_tool(fake_config_manager)
    config = get_tool()
    assert isinstance(config, dict), "Configuration is not a dictionary"
    assert (
        config.get("setting") == "value"
    ), f"Expected setting 'value', got {config.get('setting')}"


def test_update_config_tool() -> None:
    """
    Test the update_config_tool to check that configuration updates are applied correctly.
    """
    fake_config_manager = FakeConfigManager()
    update_tool = update_config_tool(fake_config_manager)
    update_tool({"new_key": "new_value"})
    config = fake_config_manager.get_config()
    assert (
        config.get("new_key") == "new_value"
    ), f"Expected 'new_value' for 'new_key', got {config.get('new_key')}"


def test_add_admin_tools() -> None:
    """
    Test add_admin_tools to ensure that all expected admin tools
    are registered in the tool registry.
    """
    fake_tool_registry = FakeToolRegistry()
    fake_executor = FakeExecutor([])
    fake_registry = FakeAgentRegistry()
    fake_config_manager = FakeConfigManager()
    add_admin_tools(fake_tool_registry, fake_executor, fake_registry, fake_config_manager)
    expected_tools = {
        "create_agent_from_conversation",
        "edit_agent",
        "delete_agent",
        "get_agent_info",
        "save_conversation",
        "get_config",
        "update_config",
        "save_agent_training",
    }
    tools_set = set(fake_tool_registry.tools.keys())
    assert tools_set == expected_tools, f"Expected tools {expected_tools}, but got {tools_set}"
