import json
import ssl
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
        temperature=0.7,
        top_p=1.0,
        top_k=20,
        max_tokens=2048,
        stop=["stop"],
        frequency_penalty=0.12,
        presence_penalty=0.34,
        seed=42,
        current_working_directory="/tmp/path",
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
    assert created_agent.temperature == 0.7
    assert created_agent.top_p == 1.0
    assert created_agent.top_k == 20
    assert created_agent.max_tokens == 2048
    assert created_agent.stop == ["stop"]
    assert created_agent.frequency_penalty == 0.12
    assert created_agent.presence_penalty == 0.34
    assert created_agent.seed == 42
    assert created_agent.current_working_directory == "/tmp/path"
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
        temperature=0.7,
        top_p=1.0,
        top_k=None,
        max_tokens=2048,
        stop=None,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        seed=None,
        current_working_directory="/tmp/path",
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
            "update": AgentEditFields(
                name="Updated Agent",
                security_prompt=None,
                hosting=None,
                model=None,
                description=None,
                last_message=None,
                temperature=None,
                top_p=None,
                top_k=None,
                max_tokens=None,
                stop=None,
                frequency_penalty=None,
                presence_penalty=None,
                seed=None,
                current_working_directory=None,
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
            "update": AgentEditFields(
                name=None,
                security_prompt="New security prompt",
                hosting=None,
                model=None,
                description=None,
                last_message=None,
                temperature=None,
                top_p=None,
                top_k=None,
                max_tokens=None,
                stop=None,
                frequency_penalty=None,
                presence_penalty=None,
                seed=None,
                current_working_directory=None,
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
            "update": AgentEditFields(
                name=None,
                security_prompt=None,
                hosting="new-hosting",
                model=None,
                description=None,
                last_message=None,
                temperature=None,
                top_p=None,
                top_k=None,
                max_tokens=None,
                stop=None,
                frequency_penalty=None,
                presence_penalty=None,
                seed=None,
                current_working_directory=None,
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
            "update": AgentEditFields(
                name=None,
                security_prompt=None,
                hosting=None,
                model="new-model",
                description=None,
                last_message=None,
                temperature=None,
                top_p=None,
                top_k=None,
                max_tokens=None,
                stop=None,
                frequency_penalty=None,
                presence_penalty=None,
                seed=None,
                current_working_directory=None,
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
            "update": AgentEditFields(
                name="Updated Agent",
                security_prompt="New security prompt",
                hosting="new-hosting",
                model="new-model",
                description="new description",
                last_message="new message",
                temperature=0.8,
                top_p=0.9,
                top_k=50,
                max_tokens=1024,
                stop=["stop"],
                frequency_penalty=0.1,
                presence_penalty=0.2,
                seed=42,
                current_working_directory=None,
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
            "update": AgentEditFields(
                name=None,
                security_prompt=None,
                hosting=None,
                model=None,
                description=None,
                last_message=None,
                temperature=None,
                top_p=None,
                top_k=None,
                max_tokens=None,
                stop=None,
                frequency_penalty=None,
                presence_penalty=None,
                seed=None,
                current_working_directory=None,
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
            "update": AgentEditFields(
                name="",
                security_prompt="",
                hosting="",
                model="",
                description="",
                last_message="",
                temperature=0.0,
                top_p=0.0,
                top_k=0,
                max_tokens=0,
                stop=[],
                frequency_penalty=0.0,
                presence_penalty=0.0,
                seed=0,
                current_working_directory=None,
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
    agent2 = registry.create_agent(
        AgentEditFields(
            name="Agent Two",
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
    registry.create_agent(
        AgentEditFields(
            name="Existing",
            security_prompt=None,
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


def test_save_and_load_agent_context(temp_agents_dir: Path):
    registry = AgentRegistry(temp_agents_dir)
    agent_name = "Agent Context Test"
    agent = registry.create_agent(
        AgentEditFields(
            name=agent_name,
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

    # Create a test context
    test_context = {
        "variables": {"x": 10, "y": 20, "result": 30},
        "functions": {"add": lambda a, b: a + b},
        "objects": {"data": {"name": "test", "value": 42}},
    }

    # Save the context
    registry.save_agent_context(agent.id, test_context)

    # Verify context file exists
    context_file = temp_agents_dir / f"{agent.id}_context.pkl"
    assert context_file.exists()

    # Load the context
    loaded_context = registry.load_agent_context(agent.id)

    # Verify the loaded context matches the original
    assert loaded_context["variables"]["x"] == test_context["variables"]["x"]
    assert loaded_context["variables"]["y"] == test_context["variables"]["y"]
    assert loaded_context["variables"]["result"] == test_context["variables"]["result"]
    assert loaded_context["objects"]["data"]["name"] == test_context["objects"]["data"]["name"]
    assert loaded_context["objects"]["data"]["value"] == test_context["objects"]["data"]["value"]
    # Note: We can't directly compare the function objects, but we can verify they work
    assert loaded_context["functions"]["add"](5, 7) == 12


def test_load_nonexistent_agent_context(temp_agents_dir: Path):
    registry = AgentRegistry(temp_agents_dir)
    agent = registry.create_agent(
        AgentEditFields(
            name="Agent No Context",
            security_prompt=None,
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

    # Load context for an agent that doesn't have one
    context = registry.load_agent_context(agent.id)
    assert context is None


def test_update_agent_state_with_context(temp_agents_dir: Path):
    registry = AgentRegistry(temp_agents_dir)
    agent_name = "Agent State Context Test"
    agent = registry.create_agent(
        AgentEditFields(
            name=agent_name,
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

    # Create test data
    conversation = [
        ConversationRecord(role=ConversationRole.USER, content="Hello", should_summarize=True),
        ConversationRecord(
            role=ConversationRole.ASSISTANT, content="Hi there!", should_summarize=True
        ),
    ]
    code_history = [
        CodeExecutionResult(
            role=ConversationRole.ASSISTANT,
            message="Test execution",
            code="print('test')",
            stdout="test",
            stderr="",
            logging="",
            formatted_print="",
            status=ProcessResponseStatus.SUCCESS,
        )
    ]
    test_context = {"variables": {"x": 1, "y": 2}, "functions": {"add": lambda a, b: a + b}}
    test_working_dir = "/test/path"

    # Update agent state
    registry.update_agent_state(
        agent.id, conversation, code_history, test_working_dir, test_context
    )

    # Verify conversation was saved
    saved_conversation = registry.get_agent_conversation_history(agent.id)
    assert len(saved_conversation) == 2
    assert saved_conversation[0].role == ConversationRole.USER
    assert saved_conversation[0].content == "Hello"
    assert saved_conversation[1].role == ConversationRole.ASSISTANT
    assert saved_conversation[1].content == "Hi there!"

    # Verify code history was saved
    saved_code_history = registry.get_agent_execution_history(agent.id)
    assert len(saved_code_history) == 1
    assert saved_code_history[0].message == "Test execution"
    assert saved_code_history[0].code == "print('test')"

    # Verify context was saved
    loaded_context = registry.load_agent_context(agent.id)
    assert loaded_context["variables"]["x"] == 1
    assert loaded_context["variables"]["y"] == 2
    assert loaded_context["functions"]["add"](1, 2) == 3

    # Verify working directory was updated
    updated_agent = registry.get_agent(agent.id)
    assert updated_agent.current_working_directory == test_working_dir


def test_update_agent_unpickleable_context(temp_agents_dir: Path):
    registry = AgentRegistry(temp_agents_dir)
    agent_name = "Test Agent"
    edit_metadata = AgentEditFields(
        name=agent_name,
        security_prompt="test security prompt",
        hosting="test-hosting",
        model="test-model",
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
    agent = registry.create_agent(edit_metadata)

    # Create test context with unpickleable object
    unpickleable_context = {
        "variables": {"x": 10, "y": 20},
        "ssl_context": ssl.create_default_context(),
    }

    # Save the context
    registry.save_agent_context(agent.id, unpickleable_context)

    # Load and verify the context
    loaded_context = registry.load_agent_context(agent.id)

    # Verify the pickleable parts were saved
    assert loaded_context["variables"]["x"] == 10
    assert loaded_context["variables"]["y"] == 20

    # Verify unpickleable object was converted to string
    assert isinstance(loaded_context["ssl_context"], str)
    assert "SSLContext" in loaded_context["ssl_context"]
