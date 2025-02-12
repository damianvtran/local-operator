import json
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

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


# Fake implementations for testing purposes
class FakeAgentRegistry(AgentRegistry):
    """
    Fake implementation of an agent registry for testing admin tools.
    """

    def __init__(self) -> None:
        self.agents: Dict[str, AgentData] = {}
        self.conversations: Dict[str, List[Dict[str, str]]] = {}

    def create_agent(self, agent_edit_metadata: AgentEditFields) -> AgentData:
        """
        Create a new agent with the provided metadata.
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
        Save the conversation history for a given agent.

        Args:
            agent_id: The unique identifier of the agent
            conversation: The conversation history to save
        """
        # Store a copy of the provided conversation to prevent reference issues
        self.conversations[agent_id] = conversation.copy() if conversation else []

    def get_agent(self, agent_id: str) -> AgentData:
        """
        Retrieve an agent by its unique identifier.
        """
        if agent_id in self.agents:
            return self.agents[agent_id]
        raise KeyError(f"Agent with id {agent_id} not found")

    def list_agents(self) -> List[AgentData]:
        """
        Return a list of all agents stored in the registry.
        """
        return list(self.agents.values())

    def update_agent(self, agent_id: str, edit_fields: AgentEditFields) -> Optional[AgentData]:
        """
        Update an existing agent's details.
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
        Delete an agent and its associated conversation.
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
        """
        Return the current conversation history.
        """
        return self.conversation_history


class FakeConfigManager(ConfigManager):
    """
    Fake configuration manager for testing admin configuration tools.
    """

    def __init__(self) -> None:
        self._config: Dict[str, Any] = {"setting": "value"}

    def get_config(self) -> Dict[str, Any]:
        """
        Retrieve the current configuration.
        """
        return self._config

    def update_config(self, values: Dict[str, Any]) -> None:
        """
        Update the configuration with provided values.
        """
        self._config.update(values)


class FakeToolRegistry(ToolRegistry):
    """
    Fake tool registry for testing tool registration.
    """

    def __init__(self) -> None:
        self._tools: Dict[str, Callable[..., Any]] = {}

    def add_tool(self, name: str, tool_func: Callable[..., Any]) -> None:
        """
        Register a tool function under the provided name.
        """
        self._tools[name] = tool_func

    @property
    def tools(self) -> Dict[str, Callable[..., Any]]:
        """
        Get the registered tools dictionary.
        """
        return self._tools


# ==================== Tests ====================


def test_create_agent_from_conversation_no_user_messages() -> None:
    """
    Test creating an agent from a conversation with no user messages.
    The saved conversation history should be empty.
    """
    conversation_history = [{"role": ConversationRole.SYSTEM.value, "content": "Initial prompt"}]
    fake_executor = FakeExecutor(conversation_history)
    fake_registry = FakeAgentRegistry()
    create_tool = create_agent_from_conversation_tool(fake_executor, fake_registry)
    agent_name = "TestAgent"
    new_agent = create_tool(agent_name)
    saved_history = fake_registry.conversations.get(new_agent.id)
    assert saved_history == []


def test_create_agent_from_conversation_with_user_messages() -> None:
    """
    Test creating an agent from a conversation containing multiple user messages.
    Verify that the conversation is truncated based on the second-to-last user message.
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
    # In reverse order, the last USER message is found at index 4,
    # so cutoff index should be 4, resulting in history[:4].
    expected_history = conversation_history[:4]
    assert saved_history == expected_history


def test_save_agent_training_no_agent() -> None:
    """
    Test saving agent training data when no current agent is set.
    Expect a ValueError to be raised.
    """
    conversation_history = [{"role": ConversationRole.USER.value, "content": "User message"}]
    fake_executor = FakeExecutor(conversation_history, agent=None)
    fake_registry = FakeAgentRegistry()
    save_training_tool = save_agent_training_tool(fake_executor, fake_registry)
    with pytest.raises(ValueError) as exc_info:
        save_training_tool()
    assert "No current agent set" in str(exc_info.value)


def test_save_agent_training_with_agent() -> None:
    """Test saving agent training data when a current agent is available.

    This test verifies that the conversation history is saved correctly when saving training data
    for an agent. It checks that:
    1. The conversation is truncated at the last user message
    2. The stored history matches the expected truncated history
    3. The updated agent matches the original agent

    The test uses fake implementations of the registry and executor to isolate the testing.
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

    # The last USER message is at index 2, so history should be truncated there
    expected_history = conversation_history[:2]
    stored_history = fake_registry.conversations.get(agent.id)
    assert stored_history == expected_history
    assert updated_agent == agent


def test_list_agent_info_without_id() -> None:
    """
    Test listing agent information without specifying an agent ID.
    Expect all agents in the registry to be returned.
    """
    fake_registry = FakeAgentRegistry()
    agent1 = fake_registry.create_agent(AgentEditFields(name="Agent1", security_prompt="prompt1"))
    agent2 = fake_registry.create_agent(AgentEditFields(name="Agent2", security_prompt="prompt2"))
    list_tool = list_agent_info_tool(fake_registry)
    agents_list = list_tool(None)
    assert len(agents_list) == 2
    agent_ids = {agent1.id, agent2.id}
    returned_ids = {agent.id for agent in agents_list}
    assert agent_ids == returned_ids


def test_list_agent_info_with_id() -> None:
    """
    Test retrieving agent information for a specific agent ID.
    Expect a list containing the desired agent only.
    """
    fake_registry = FakeAgentRegistry()
    agent = fake_registry.create_agent(AgentEditFields(name="AgentX", security_prompt="promptX"))
    list_tool = list_agent_info_tool(fake_registry)
    agent_list = list_tool(agent.id)
    assert len(agent_list) == 1
    assert agent_list[0].id == agent.id


def test_create_agent_tool() -> None:
    """
    Test the create_agent_tool to ensure it creates an agent with the correct parameters.
    """
    fake_registry = FakeAgentRegistry()
    create_tool = create_agent_tool(fake_registry)
    agent = create_tool("NewAgent", "secure")
    assert agent.name == "NewAgent"
    assert agent.security_prompt == "secure"
    assert agent.id in fake_registry.agents


def test_edit_agent_tool() -> None:
    """
    Test the edit_agent_tool to verify updating an agent's name and security prompt.
    """
    fake_registry = FakeAgentRegistry()
    agent = fake_registry.create_agent(AgentEditFields(name="OldName", security_prompt="old"))
    edit_tool = edit_agent_tool(fake_registry)
    updated_agent = edit_tool(agent.id, AgentEditFields(name="NewName", security_prompt="new"))
    assert updated_agent.name == "NewName"
    assert updated_agent.security_prompt == "new"


def test_delete_agent_tool() -> None:
    """
    Test the delete_agent_tool to confirm an agent is removed from the registry.
    """
    fake_registry = FakeAgentRegistry()
    agent = fake_registry.create_agent(AgentEditFields(name="ToDelete", security_prompt=""))
    delete_tool = delete_agent_tool(fake_registry)
    delete_tool(agent.id)
    with pytest.raises(KeyError):
        fake_registry.get_agent(agent.id)


def test_get_agent_info_tool_without_id() -> None:
    """
    Test the get_agent_info_tool to retrieve all agents when no ID is provided.
    """
    fake_registry = FakeAgentRegistry()
    fake_registry.create_agent(AgentEditFields(name="Agent1", security_prompt=""))
    fake_registry.create_agent(AgentEditFields(name="Agent2", security_prompt=""))
    get_tool = get_agent_info_tool(fake_registry)
    agents = get_tool(None)
    assert len(agents) == 2


def test_get_agent_info_tool_with_id() -> None:
    """
    Test the get_agent_info_tool to retrieve information for a specific agent.
    """
    fake_registry = FakeAgentRegistry()
    agent = fake_registry.create_agent(AgentEditFields(name="AgentSingle", security_prompt=""))
    get_tool = get_agent_info_tool(fake_registry)
    result = get_tool(agent.id)
    assert len(result) == 1
    assert result[0].id == agent.id


def test_save_conversation_tool(tmp_path: Any) -> None:
    """
    Test the save_conversation_tool to ensure conversation history is written to a file.
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
    assert isinstance(data, list)
    for msg in data:
        assert isinstance(msg, dict)
        assert "role" in msg
        assert "content" in msg


def test_get_config_tool() -> None:
    """
    Test the get_config_tool to verify retrieval of current configuration settings.
    """
    fake_config_manager = FakeConfigManager()
    get_tool = get_config_tool(fake_config_manager)
    config = get_tool()
    assert isinstance(config, dict)
    assert config.get("setting") == "value"


def test_update_config_tool() -> None:
    """
    Test the update_config_tool to check that configuration settings are updated correctly.
    """
    fake_config_manager = FakeConfigManager()
    update_tool = update_config_tool(fake_config_manager)
    update_tool({"new_key": "new_value"})
    config = fake_config_manager.get_config()
    assert config.get("new_key") == "new_value"


def test_add_admin_tools() -> None:
    """
    Test add_admin_tools to ensure all expected admin tools are added to the tool registry.
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
    assert set(fake_tool_registry.tools.keys()) == expected_tools
