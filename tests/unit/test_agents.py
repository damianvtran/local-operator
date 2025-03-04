import json
from pathlib import Path

import pytest

from local_operator.agents import AgentEditFields, AgentRegistry
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


def test_create_agent_success(temp_agents_dir: Path):
    registry = AgentRegistry(temp_agents_dir)
    agent_name = "Test Agent"
    edit_metadata = AgentEditFields(
        name=agent_name,
        security_prompt="test security prompt",
        hosting="test-hosting",
        model="test-model",
        description="test description",
        last_message="test last message",
    )
    agent = registry.create_agent(edit_metadata)
    agents = registry.list_agents()
    assert len(agents) == 1
    created_agent = agents[0]
    assert created_agent.name == agent_name
    assert created_agent.id == agent.id
    assert created_agent.security_prompt == "test security prompt"
    assert created_agent.hosting == "test-hosting"
    assert created_agent.model == "test-model"
    assert created_agent.description == "test description"
    assert created_agent.last_message == "test last message"
    assert created_agent.last_message_datetime is not None
    # Verify that agents.json file is created
    agents_file = temp_agents_dir / "agents.json"
    assert agents_file.exists()

    # Verify that the conversation file is created and is empty
    conversation_file = temp_agents_dir / f"{created_agent.id}_conversation.json"
    assert conversation_file.exists()
    with conversation_file.open("r", encoding="utf-8") as f:
        conversation = json.load(f)
    assert conversation == []


def test_create_agent_duplicate(temp_agents_dir: Path):
    registry = AgentRegistry(temp_agents_dir)
    agent_name = "Duplicate Agent"
    edit_metadata = AgentEditFields(
        name=agent_name,
        security_prompt="test security prompt",
        hosting="test-hosting",
        model="test-model",
        description="",
        last_message="",
    )
    registry.create_agent(edit_metadata)
    with pytest.raises(ValueError) as exc_info:
        # Attempt to create another agent with the same name
        registry.create_agent(
            AgentEditFields(
                name=agent_name,
                security_prompt="",
                hosting="",
                model="",
                description="",
                last_message="",
            )
        )
    assert f"Agent with name {agent_name} already exists" in str(exc_info.value)


@pytest.mark.parametrize(
    "test_case",
    [
        {
            "name": "name_only",
            "original": AgentEditFields(
                name="Original Agent",
                security_prompt="original prompt",
                hosting="test-hosting",
                model="test-model",
                description="",
                last_message="",
            ),
            "update": AgentEditFields(
                name="Updated Agent",
                security_prompt=None,
                hosting=None,
                model=None,
                description=None,
                last_message=None,
            ),
            "expected_name": "Updated Agent",
            "expected_prompt": "original prompt",
            "expected_hosting": "test-hosting",
            "expected_model": "test-model",
        },
        {
            "name": "security_only",
            "original": AgentEditFields(
                name="Test Agent",
                security_prompt="original prompt",
                hosting="test-hosting",
                model="test-model",
                description="",
                last_message="",
            ),
            "update": AgentEditFields(
                name=None,
                security_prompt="New security prompt",
                hosting=None,
                model=None,
                description=None,
                last_message=None,
            ),
            "expected_name": "Test Agent",
            "expected_prompt": "New security prompt",
            "expected_hosting": "test-hosting",
            "expected_model": "test-model",
        },
        {
            "name": "hosting_only",
            "original": AgentEditFields(
                name="Test Agent",
                security_prompt="original prompt",
                hosting="test-hosting",
                model="test-model",
                description="",
                last_message="",
            ),
            "update": AgentEditFields(
                name=None,
                security_prompt=None,
                hosting="new-hosting",
                model=None,
                description=None,
                last_message=None,
            ),
            "expected_name": "Test Agent",
            "expected_prompt": "original prompt",
            "expected_hosting": "new-hosting",
            "expected_model": "test-model",
        },
        {
            "name": "model_only",
            "original": AgentEditFields(
                name="Test Agent",
                security_prompt="original prompt",
                hosting="test-hosting",
                model="test-model",
                description="",
                last_message="",
            ),
            "update": AgentEditFields(
                name=None,
                security_prompt=None,
                hosting=None,
                model="new-model",
                description=None,
                last_message=None,
            ),
            "expected_name": "Test Agent",
            "expected_prompt": "original prompt",
            "expected_hosting": "test-hosting",
            "expected_model": "new-model",
        },
        {
            "name": "all_fields",
            "original": AgentEditFields(
                name="Original Agent",
                security_prompt="original prompt",
                hosting="test-hosting",
                model="test-model",
                description="",
                last_message="",
            ),
            "update": AgentEditFields(
                name="Updated Agent",
                security_prompt="New security prompt",
                hosting="new-hosting",
                model="new-model",
                description="new description",
                last_message="new message",
            ),
            "expected_name": "Updated Agent",
            "expected_prompt": "New security prompt",
            "expected_hosting": "new-hosting",
            "expected_model": "new-model",
        },
        {
            "name": "no_fields",
            "original": AgentEditFields(
                name="Test Agent",
                security_prompt="original prompt",
                hosting="test-hosting",
                model="test-model",
                description="",
                last_message="",
            ),
            "update": AgentEditFields(
                name=None,
                security_prompt=None,
                hosting=None,
                model=None,
                description=None,
                last_message=None,
            ),
            "expected_name": "Test Agent",
            "expected_prompt": "original prompt",
            "expected_hosting": "test-hosting",
            "expected_model": "test-model",
        },
        {
            "name": "empty_strings",
            "original": AgentEditFields(
                name="Test Agent",
                security_prompt="original prompt",
                hosting="test-hosting",
                model="test-model",
                description="",
                last_message="",
            ),
            "update": AgentEditFields(
                name="", security_prompt="", hosting="", model="", description="", last_message=""
            ),
            "expected_name": "",
            "expected_prompt": "",
            "expected_hosting": "",
            "expected_model": "",
        },
    ],
)
def test_update_agent(temp_agents_dir: Path, test_case):
    registry = AgentRegistry(temp_agents_dir)

    # Create initial agent
    agent = registry.create_agent(test_case["original"])

    # Edit the agent
    registry.update_agent(agent.id, test_case["update"])

    # Verify updates
    updated_agent = registry.get_agent(agent.id)
    assert updated_agent.name == test_case["expected_name"]
    assert updated_agent.security_prompt == test_case["expected_prompt"]

    # Check that the agents.json file has updated data
    agents_file = temp_agents_dir / "agents.json"
    with agents_file.open("r", encoding="utf-8") as f:
        agents_data = json.load(f)
    found = any(
        item["id"] == agent.id
        and item["name"] == test_case["expected_name"]
        and item["security_prompt"] == test_case["expected_prompt"]
        for item in agents_data
    )
    assert found


def test_delete_agent(temp_agents_dir: Path):
    registry = AgentRegistry(temp_agents_dir)
    agent_name = "Agent to Delete"
    agent = registry.create_agent(
        AgentEditFields(
            name=agent_name,
            security_prompt="",
            hosting="",
            model="",
            description="",
            last_message="",
        )
    )

    # Ensure conversation file exists before deletion
    conversation_file = temp_agents_dir / f"{agent.id}_conversation.json"
    assert conversation_file.exists()

    # Delete the agent and verify it is removed from the registry
    registry.delete_agent(agent.id)
    with pytest.raises(KeyError):
        registry.get_agent(agent.id)
    # Verify that the conversation file has been deleted
    assert not conversation_file.exists()


def test_get_agent_not_found(temp_agents_dir: Path):
    registry = AgentRegistry(temp_agents_dir)
    with pytest.raises(KeyError) as exc_info:
        registry.get_agent("non-existent-id")
    assert "Agent with id non-existent-id not found" in str(exc_info.value)


def test_list_agents(temp_agents_dir: Path):
    registry = AgentRegistry(temp_agents_dir)
    # Initially, the agents list should be empty
    assert registry.list_agents() == []

    # Create two agents and verify the list
    agent1 = registry.create_agent(
        AgentEditFields(
            name="Agent One",
            security_prompt="",
            hosting="",
            model="",
            description="",
            last_message="",
        )
    )
    agent2 = registry.create_agent(
        AgentEditFields(
            name="Agent Two",
            security_prompt="",
            hosting="",
            model="",
            description="",
            last_message="",
        )
    )
    agents = registry.list_agents()
    assert len(agents) == 2
    names = {agent.name for agent in agents}
    assert names == {"Agent One", "Agent Two"}
    ids = {agent.id for agent in agents}
    assert ids == {agent1.id, agent2.id}


def test_save_and_load_conversation(temp_agents_dir: Path):
    registry = AgentRegistry(temp_agents_dir)
    agent_name = "Agent Conversation"
    agent = registry.create_agent(
        AgentEditFields(
            name=agent_name,
            security_prompt="",
            hosting="",
            model="",
            description="",
            last_message="",
        )
    )

    conversation = [
        ConversationRecord(role=ConversationRole.USER, content="Hello"),
        ConversationRecord(role=ConversationRole.ASSISTANT, content="Hi there!"),
    ]
    execution_history = [
        CodeExecutionResult(
            stdout="",
            stderr="",
            logging="",
            message="This is a test code execution result",
            code="print('Hello, world!')",
            formatted_print="Hello, world!",
            role=ConversationRole.ASSISTANT,
            status=ProcessResponseStatus.SUCCESS,
        )
    ]

    registry.save_agent_conversation(agent.id, conversation, execution_history)
    loaded_conversation_data = registry.load_agent_conversation(agent.id)
    assert loaded_conversation_data.conversation == conversation
    assert loaded_conversation_data.execution_history == execution_history


def test_load_nonexistent_conversation(temp_agents_dir: Path):
    registry = AgentRegistry(temp_agents_dir)
    agent = registry.create_agent(
        AgentEditFields(
            name="Agent No Conversation",
            security_prompt=None,
            hosting="",
            model="",
            description="",
            last_message="",
        )
    )
    conversation_file = temp_agents_dir / f"{agent.id}_conversation.json"
    # Remove the conversation file if it exists to simulate a missing file
    if conversation_file.exists():
        conversation_file.unlink()
    conversation_data = registry.load_agent_conversation(agent.id)
    assert conversation_data.conversation == []
    assert conversation_data.execution_history == []


def test_update_agent_not_found(temp_agents_dir: Path):
    registry = AgentRegistry(temp_agents_dir)
    with pytest.raises(KeyError):
        registry.update_agent(
            "non-existent-id",
            AgentEditFields(
                name="New Name",
                security_prompt=None,
                hosting="",
                model="",
                description="",
                last_message="",
            ),
        )


def test_delete_agent_not_found(temp_agents_dir: Path):
    registry = AgentRegistry(temp_agents_dir)
    with pytest.raises(KeyError):
        registry.delete_agent("non-existent-id")


def test_create_agent_save_failure(temp_agents_dir: Path, monkeypatch):
    registry = AgentRegistry(temp_agents_dir)

    # Monkey-patch the open method on the Path type to simulate a write failure for agents.json
    def fake_open(*args, **kwargs):
        raise Exception("Fake write failure")

    monkeypatch.setattr(type(registry.agents_file), "open", fake_open)

    with pytest.raises(Exception) as exc_info:
        registry.create_agent(
            AgentEditFields(
                name="Agent Fail",
                security_prompt=None,
                hosting="",
                model="",
                description="",
                last_message="",
            )
        )
    assert str(exc_info.value) == "Fake write failure"


def test_clone_agent(temp_agents_dir: Path):
    registry = AgentRegistry(temp_agents_dir)
    source_name = "Source Agent"
    source_agent = registry.create_agent(
        AgentEditFields(
            name=source_name,
            security_prompt="test security prompt",
            hosting="",
            model="",
            description="",
            last_message="",
        )
    )

    # Add some conversation history to source agent
    conversation = [
        ConversationRecord(role=ConversationRole.USER, content="Hello"),
        ConversationRecord(role=ConversationRole.ASSISTANT, content="Hi there!"),
    ]
    execution_history = [
        CodeExecutionResult(
            stdout="",
            stderr="",
            logging="",
            message="This is a test code execution result",
            code="print('Hello, world!')",
            formatted_print="Hello, world!",
            role=ConversationRole.ASSISTANT,
            status=ProcessResponseStatus.SUCCESS,
        )
    ]

    registry.save_agent_conversation(source_agent.id, conversation, execution_history)

    # Clone the agent
    cloned_agent = registry.clone_agent(source_agent.id, "Cloned Agent")
    assert cloned_agent.name == "Cloned Agent"
    assert cloned_agent.id != source_agent.id
    assert cloned_agent.security_prompt == "test security prompt"

    # Verify conversation was copied
    cloned_conversation_data = registry.load_agent_conversation(cloned_agent.id)
    assert cloned_conversation_data.conversation == conversation
    assert cloned_conversation_data.execution_history == execution_history


def test_clone_agent_not_found(temp_agents_dir: Path):
    registry = AgentRegistry(temp_agents_dir)
    with pytest.raises(KeyError):
        registry.clone_agent("non-existent-id", "New Clone")


def test_clone_agent_duplicate_name(temp_agents_dir: Path):
    registry = AgentRegistry(temp_agents_dir)
    source_agent = registry.create_agent(
        AgentEditFields(
            name="Source",
            security_prompt=None,
            hosting="",
            model="",
            description="",
            last_message="",
        )
    )
    registry.create_agent(
        AgentEditFields(
            name="Existing",
            security_prompt=None,
            hosting="",
            model="",
            description="",
            last_message="",
        )
    )

    with pytest.raises(ValueError):
        registry.clone_agent(source_agent.id, "Existing")


def test_get_agent_by_name(temp_agents_dir: Path):
    registry = AgentRegistry(temp_agents_dir)

    # Create a test agent
    agent_name = "Test Agent"
    agent = registry.create_agent(
        AgentEditFields(
            name=agent_name,
            security_prompt="test security prompt",
            hosting="",
            model="",
            description="",
            last_message="",
        )
    )

    # Test finding existing agent
    found_agent = registry.get_agent_by_name(agent_name)
    assert found_agent is not None
    assert found_agent.id == agent.id
    assert found_agent.name == agent_name
    assert found_agent.security_prompt == "test security prompt"

    # Test finding non-existent agent
    not_found = registry.get_agent_by_name("Non Existent")
    assert not_found is None


def test_create_autosave_agent(temp_agents_dir: Path):
    registry = AgentRegistry(temp_agents_dir)

    # Create the autosave agent
    autosave_agent = registry.create_autosave_agent()

    # Assert that the agent was created with the correct ID and name
    assert autosave_agent.id == "autosave"
    assert autosave_agent.name == "autosave"

    # Assert that the agent is now in the registry
    retrieved_agent = registry.get_agent("autosave")
    assert retrieved_agent == autosave_agent

    # Call create_autosave_agent again, should return the existing agent
    existing_autosave_agent = registry.create_autosave_agent()
    assert existing_autosave_agent == autosave_agent


def test_get_autosave_agent(temp_agents_dir: Path):
    registry = AgentRegistry(temp_agents_dir)

    # Create the autosave agent
    autosave_agent = registry.create_autosave_agent()

    # Retrieve the autosave agent
    retrieved_agent = registry.get_autosave_agent()

    # Assert that the retrieved agent is the same as the created agent
    assert retrieved_agent == autosave_agent


def test_get_autosave_agent_not_found(temp_agents_dir: Path):
    registry = AgentRegistry(temp_agents_dir)

    # If the autosave agent doesn't exist, create it
    with pytest.raises(KeyError):
        registry.get_autosave_agent()
