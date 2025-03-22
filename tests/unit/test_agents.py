import json
import ssl
import uuid
from datetime import datetime, timezone
from pathlib import Path

import dill
import pytest
import yaml

from local_operator.agents import AgentEditFields, AgentRegistry
from local_operator.clients.tavily import TavilyResponse, TavilyResult
from local_operator.types import (
    AgentState,
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

    # Verify that the agent directory is created
    agents_dir = temp_agents_dir / "agents"
    agent_dir = agents_dir / agent.id
    assert agent_dir.exists()

    # Verify that agent.yml file is created
    agent_yml_file = agent_dir / "agent.yml"
    assert agent_yml_file.exists()

    # Verify that the conversation files are created
    conversation_file = agent_dir / "conversation.jsonl"
    assert conversation_file.exists()
    execution_history_file = agent_dir / "execution_history.jsonl"
    assert execution_history_file.exists()
    learnings_file = agent_dir / "learnings.jsonl"
    assert learnings_file.exists()


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

    # Ensure agent directory exists before deletion
    agents_dir = temp_agents_dir / "agents"
    agent_dir = agents_dir / agent.id
    assert agent_dir.exists()

    # Delete the agent and verify it is removed from the registry
    registry.delete_agent(agent.id)
    with pytest.raises(KeyError):
        registry.get_agent(agent.id)

    # Verify that the agent directory has been deleted
    assert not agent_dir.exists()


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


def test_save_and_load_state(temp_agents_dir: Path):
    registry = AgentRegistry(temp_agents_dir)
    agent_name = "Agent State"
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

    timestamp = datetime.now()

    conversation = [
        ConversationRecord(role=ConversationRole.USER, content="Hello", timestamp=timestamp),
        ConversationRecord(
            role=ConversationRole.ASSISTANT, content="Hi there!", timestamp=timestamp
        ),
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
            files=[],
            timestamp=timestamp,
        )
    ]
    learnings = ["Test learning 1", "Test learning 2"]
    current_plan = "Test current plan"
    instruction_details = "Test instruction details"

    # Save the conversation with all data
    registry.save_agent_state(
        agent_id=agent.id,
        agent_state=AgentState(
            version="",
            conversation=conversation,
            execution_history=execution_history,
            learnings=learnings,
            current_plan=current_plan,
            instruction_details=instruction_details,
            agent_system_prompt="test system prompt",
        ),
    )

    # Verify JSONL files exist
    agents_dir = temp_agents_dir / "agents"
    agent_dir = agents_dir / agent.id

    conversation_file = agent_dir / "conversation.jsonl"
    assert conversation_file.exists()
    assert conversation_file.stat().st_size > 0

    execution_history_file = agent_dir / "execution_history.jsonl"
    assert execution_history_file.exists()
    assert execution_history_file.stat().st_size > 0

    learnings_file = agent_dir / "learnings.jsonl"
    assert learnings_file.exists()
    assert learnings_file.stat().st_size > 0

    plan_file = agent_dir / "current_plan.txt"
    assert plan_file.exists()
    with plan_file.open("r") as f:
        assert f.read() == current_plan

    instruction_file = agent_dir / "instruction_details.txt"
    assert instruction_file.exists()
    with instruction_file.open("r") as f:
        assert f.read() == instruction_details

    system_prompt_file = agent_dir / "system_prompt.md"
    assert system_prompt_file.exists()
    with system_prompt_file.open("r") as f:
        assert f.read() == "test system prompt"

    # Load the conversation and verify data
    loaded_conversation_data = registry.load_agent_state(agent.id)
    assert loaded_conversation_data.conversation == conversation
    assert loaded_conversation_data.execution_history == execution_history
    assert loaded_conversation_data.learnings == learnings
    assert loaded_conversation_data.current_plan == current_plan
    assert loaded_conversation_data.instruction_details == instruction_details


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

    # Get agent directory
    agents_dir = temp_agents_dir / "agents"
    agent_dir = agents_dir / agent.id

    # Remove the conversation files to simulate missing files
    conversation_file = agent_dir / "conversation.jsonl"
    if conversation_file.exists():
        conversation_file.unlink()

    execution_history_file = agent_dir / "execution_history.jsonl"
    if execution_history_file.exists():
        execution_history_file.unlink()

    learnings_file = agent_dir / "learnings.jsonl"
    if learnings_file.exists():
        learnings_file.unlink()

    # Also remove old format file for backward compatibility
    old_conversation_file = temp_agents_dir / f"{agent.id}_conversation.json"
    if old_conversation_file.exists():
        old_conversation_file.unlink()

    # Load conversation and verify it's empty
    conversation_data = registry.load_agent_state(agent.id)
    assert conversation_data.conversation == []
    assert conversation_data.execution_history == []
    assert conversation_data.learnings == []
    assert conversation_data.current_plan is None
    assert conversation_data.instruction_details is None


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
            files=[],
        )
    ]
    learnings = ["Test learning 1", "Test learning 2"]
    current_plan = "Test current plan"
    instruction_details = "Test instruction details"

    # Save the conversation with all data
    registry.save_agent_state(
        agent_id=source_agent.id,
        agent_state=AgentState(
            version="",
            conversation=conversation,
            execution_history=execution_history,
            learnings=learnings,
            current_plan=current_plan,
            instruction_details=instruction_details,
            agent_system_prompt="test system prompt",
        ),
    )

    # Clone the agent
    cloned_agent = registry.clone_agent(source_agent.id, "Cloned Agent")
    assert cloned_agent.name == "Cloned Agent"
    assert cloned_agent.id != source_agent.id
    assert cloned_agent.security_prompt == "test security prompt"

    # Verify agent directory was created
    agents_dir = temp_agents_dir / "agents"
    cloned_agent_dir = agents_dir / cloned_agent.id
    assert cloned_agent_dir.exists()

    # Verify agent.yml file was created
    agent_yml_file = cloned_agent_dir / "agent.yml"
    assert agent_yml_file.exists()

    # Verify conversation files were created
    conversation_file = cloned_agent_dir / "conversation.jsonl"
    assert conversation_file.exists()
    execution_history_file = cloned_agent_dir / "execution_history.jsonl"
    assert execution_history_file.exists()
    learnings_file = cloned_agent_dir / "learnings.jsonl"
    assert learnings_file.exists()

    # Verify plan and instruction files were created
    plan_file = cloned_agent_dir / "current_plan.txt"
    assert plan_file.exists()
    instruction_file = cloned_agent_dir / "instruction_details.txt"
    assert instruction_file.exists()

    # Verify system prompt file was created
    system_prompt_file = cloned_agent_dir / "system_prompt.md"
    assert system_prompt_file.exists()
    with system_prompt_file.open("r") as f:
        assert f.read() == "test system prompt"

    # Verify conversation was copied
    cloned_conversation_data = registry.load_agent_state(cloned_agent.id)
    assert cloned_conversation_data.conversation == conversation
    assert cloned_conversation_data.execution_history == execution_history
    assert cloned_conversation_data.learnings == learnings
    assert cloned_conversation_data.current_plan == current_plan
    assert cloned_conversation_data.instruction_details == instruction_details


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

    # Verify context files exist
    agents_dir = temp_agents_dir / "agents"
    agent_dir = agents_dir / agent.id
    new_context_file = agent_dir / "context.pkl"
    assert new_context_file.exists()

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
            files=[],
        )
    ]
    test_context = {"variables": {"x": 1, "y": 2}, "functions": {"add": lambda a, b: a + b}}
    test_working_dir = "/test/path"

    # Update agent state
    registry.update_agent_state(
        agent_id=agent.id,
        agent_state=AgentState(
            version="",
            conversation=conversation,
            execution_history=code_history,
            learnings=[],
            current_plan="",
            instruction_details="",
            agent_system_prompt="",
        ),
        context=test_context,
        current_working_directory=test_working_dir,
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
        "pydantic_model": TavilyResponse(
            query="test query",
            results=[
                TavilyResult(
                    title="Test Title",
                    url="https://test.com",
                    content="Test Content",
                    score=0.95,
                )
            ],
        ),
    }

    # Save the context
    registry.save_agent_context(agent.id, unpickleable_context)

    # Load and verify the context
    loaded_context = registry.load_agent_context(agent.id)

    # Verify the pickleable parts were saved
    assert loaded_context["variables"]["x"] == 10
    assert loaded_context["variables"]["y"] == 20
    assert loaded_context["pydantic_model"].query == "test query"

    # Verify unpickleable object was converted to string
    assert isinstance(loaded_context["ssl_context"], str)
    assert "SSLContext" in loaded_context["ssl_context"]

    # Verify the pydantic model was saved
    assert len(loaded_context["pydantic_model"].results) == 1
    assert loaded_context["pydantic_model"].results[0].title == "Test Title"
    assert loaded_context["pydantic_model"].results[0].url == "https://test.com"
    assert loaded_context["pydantic_model"].results[0].content == "Test Content"
    assert loaded_context["pydantic_model"].results[0].score == 0.95


def test_migrate_legacy_agents(temp_agents_dir: Path):
    """Test migration of agents from old format to new format."""
    # Create agents.json with test data
    agents_data = [
        {
            "id": "test-agent-1",
            "name": "Test Agent 1",
            "created_date": datetime.now(timezone.utc).isoformat(),
            "version": "0.1.0",
            "security_prompt": "test security prompt",
            "hosting": "test-hosting",
            "model": "test-model",
            "description": "test description",
            "last_message": "test last message",
            "last_message_datetime": datetime.now(timezone.utc).isoformat(),
            "temperature": 0.7,
            "top_p": 1.0,
            "top_k": 20,
            "max_tokens": 2048,
            "stop": ["stop"],
            "frequency_penalty": 0.12,
            "presence_penalty": 0.34,
            "seed": 42,
            "current_working_directory": "/tmp/path",
        }
    ]

    # Create agents.json
    agents_file = temp_agents_dir / "agents.json"
    with agents_file.open("w", encoding="utf-8") as f:
        json.dump(agents_data, f, indent=2)

    # Create old conversation file
    agent_id = "test-agent-1"
    old_conversation_file = temp_agents_dir / f"{agent_id}_conversation.json"

    # Create test conversation data
    conversation_data = {
        "version": "0.1.0",
        "conversation": [
            {
                "role": "user",
                "content": "Hello",
                "should_summarize": "true",
                "ephemeral": "false",
                "summarized": "false",
                "is_system_prompt": "false",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "files": None,
                "should_cache": "true",
            },
            {
                "role": "assistant",
                "content": "Hi there!",
                "should_summarize": "true",
                "ephemeral": "false",
                "summarized": "false",
                "is_system_prompt": "false",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "files": None,
                "should_cache": "true",
            },
        ],
        "execution_history": [
            {
                "id": str(uuid.uuid4()),
                "stdout": "",
                "stderr": "",
                "logging": "",
                "message": "This is a test code execution result",
                "code": "print('Hello, world!')",
                "formatted_print": "Hello, world!",
                "role": "assistant",
                "status": "success",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "files": [],
                "action": "CODE",
                "execution_type": "action",
                "task_classification": "",
            }
        ],
        "learnings": ["Test learning 1", "Test learning 2"],
        "current_plan": "Test current plan",
        "instruction_details": "Test instruction details",
    }

    with old_conversation_file.open("w", encoding="utf-8") as f:
        json.dump(conversation_data, f, indent=2)

    # Create old context file
    old_context_file = temp_agents_dir / f"{agent_id}_context.pkl"
    test_context = {"variables": {"x": 10, "y": 20}}
    with old_context_file.open("wb") as f:
        dill.dump(test_context, f)

    # Initialize registry which should trigger migration
    registry = AgentRegistry(temp_agents_dir)

    # Verify agent was migrated
    agent_dir = temp_agents_dir / "agents" / agent_id
    assert agent_dir.exists()

    # Verify agent.yml was created
    agent_yml_file = agent_dir / "agent.yml"
    assert agent_yml_file.exists()

    # Verify conversation files were created
    conversation_file = agent_dir / "conversation.jsonl"
    assert conversation_file.exists()
    assert conversation_file.stat().st_size > 0

    execution_history_file = agent_dir / "execution_history.jsonl"
    assert execution_history_file.exists()
    assert execution_history_file.stat().st_size > 0

    learnings_file = agent_dir / "learnings.jsonl"
    assert learnings_file.exists()
    assert learnings_file.stat().st_size > 0

    # Verify plan and instruction files were created
    plan_file = agent_dir / "current_plan.txt"
    assert plan_file.exists()
    with plan_file.open("r") as f:
        assert f.read() == "Test current plan"

    instruction_file = agent_dir / "instruction_details.txt"
    assert instruction_file.exists()
    with instruction_file.open("r") as f:
        assert f.read() == "Test instruction details"

    # Verify context was migrated
    context_file = agent_dir / "context.pkl"
    assert context_file.exists()
    with context_file.open("rb") as f:
        migrated_context = dill.load(f)
    assert migrated_context["variables"]["x"] == 10
    assert migrated_context["variables"]["y"] == 20

    # Verify agent can be loaded
    agent = registry.get_agent(agent_id)
    assert agent.id == agent_id
    assert agent.name == "Test Agent 1"

    # Verify conversation can be loaded
    conversation = registry.load_agent_state(agent_id)
    assert len(conversation.conversation) == 2
    assert conversation.conversation[0].role == ConversationRole.USER
    assert conversation.conversation[0].content == "Hello"
    assert conversation.conversation[1].role == ConversationRole.ASSISTANT
    assert conversation.conversation[1].content == "Hi there!"
    assert len(conversation.execution_history) == 1
    assert conversation.execution_history[0].message == "This is a test code execution result"
    assert len(conversation.learnings) == 2
    assert conversation.learnings[0] == "Test learning 1"
    assert conversation.current_plan == "Test current plan"
    assert conversation.instruction_details == "Test instruction details"


def test_migrate_legacy_agents_no_migration_needed(temp_agents_dir: Path):
    """Test that migration is skipped when agent already exists in new format."""
    # Create agents.json with test data
    agents_data = [
        {
            "id": "test-agent-1",
            "name": "Test Agent 1",
            "created_date": datetime.now(timezone.utc).isoformat(),
            "version": "0.1.0",
            "security_prompt": "test security prompt",
            "hosting": "test-hosting",
            "model": "test-model",
            "description": "test description",
            "last_message": "test last message",
            "last_message_datetime": datetime.now(timezone.utc).isoformat(),
            "temperature": 0.7,
            "top_p": 1.0,
            "top_k": 20,
            "max_tokens": 2048,
            "stop": ["stop"],
            "frequency_penalty": 0.12,
            "presence_penalty": 0.34,
            "seed": 42,
            "current_working_directory": "/tmp/path",
        }
    ]

    # Create agents.json
    agents_file = temp_agents_dir / "agents.json"
    with agents_file.open("w", encoding="utf-8") as f:
        json.dump(agents_data, f, indent=2)

    # Create agent directory and agent.yml
    agent_id = "test-agent-1"
    agents_dir = temp_agents_dir / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    agent_dir = agents_dir / agent_id
    agent_dir.mkdir(parents=True, exist_ok=True)

    # Create agent.yml file
    agent_yml_file = agent_dir / "agent.yml"
    with agent_yml_file.open("w", encoding="utf-8") as f:
        yaml.dump(agents_data[0], f, default_flow_style=False)

    # Create old conversation file
    old_conversation_file = temp_agents_dir / f"{agent_id}_conversation.json"

    # Create test conversation data
    conversation_data = {
        "version": "0.1.0",
        "conversation": [
            {
                "role": "user",
                "content": "Hello",
                "should_summarize": "true",
                "ephemeral": "false",
                "summarized": "false",
                "is_system_prompt": "false",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "files": None,
                "should_cache": "true",
            }
        ],
        "execution_history": [],
        "learnings": [],
        "current_plan": None,
        "instruction_details": None,
    }

    with old_conversation_file.open("w", encoding="utf-8") as f:
        json.dump(conversation_data, f, indent=2)

    # Initialize registry which should trigger migration
    registry = AgentRegistry(temp_agents_dir)

    # Verify agent was not migrated (conversation.jsonl should not exist)
    conversation_file = agent_dir / "conversation.jsonl"
    assert not conversation_file.exists()

    # Verify agent can be loaded
    agent = registry.get_agent(agent_id)
    assert agent.id == agent_id
    assert agent.name == "Test Agent 1"


def test_get_agent_system_prompt(temp_agents_dir: Path):
    registry = AgentRegistry(temp_agents_dir)
    agent_name = "Test Agent"
    agent = registry.create_agent(
        AgentEditFields(
            name=agent_name,
            security_prompt="test security prompt",
            hosting="test-hosting",
            model="test-model",
            description="test description",
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

    # Create a system prompt file
    agent_dir = temp_agents_dir / "agents" / agent.id
    system_prompt_path = agent_dir / "system_prompt.md"
    test_prompt = "This is a test system prompt for the agent."

    with system_prompt_path.open("w", encoding="utf-8") as f:
        f.write(test_prompt)

    # Get the system prompt and verify it matches
    retrieved_prompt = registry.get_agent_system_prompt(agent.id)
    assert retrieved_prompt == test_prompt

    # Test getting system prompt for non-existent file
    system_prompt_path.unlink()
    empty_prompt = registry.get_agent_system_prompt(agent.id)
    assert empty_prompt == ""

    # Test getting system prompt for non-existent agent
    with pytest.raises(KeyError):
        registry.get_agent_system_prompt("non-existent-id")


def test_set_agent_system_prompt(temp_agents_dir: Path):
    registry = AgentRegistry(temp_agents_dir)
    agent_name = "Test Agent"
    agent = registry.create_agent(
        AgentEditFields(
            name=agent_name,
            security_prompt="test security prompt",
            hosting="test-hosting",
            model="test-model",
            description="test description",
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

    # Set the system prompt
    test_prompt = "This is a new system prompt for the agent."
    registry.set_agent_system_prompt(agent.id, test_prompt)

    # Verify the prompt was written to file
    agent_dir = temp_agents_dir / "agents" / agent.id
    system_prompt_path = agent_dir / "system_prompt.md"
    assert system_prompt_path.exists()

    with system_prompt_path.open("r", encoding="utf-8") as f:
        saved_prompt = f.read()

    assert saved_prompt == test_prompt

    # Test setting system prompt for non-existent agent
    with pytest.raises(KeyError):
        registry.set_agent_system_prompt("non-existent-id", "Test prompt")

    # Test error handling when writing fails

    # Create a mock that raises IOError when writing
    original_open = open

    def mock_open_that_fails(*args, **kwargs):
        if args[0] == system_prompt_path and "w" in args[1]:
            raise IOError("Simulated write error")
        return original_open(*args, **kwargs)

    # Apply the mock and test
    with pytest.raises(IOError):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("builtins.open", mock_open_that_fails)
            registry.set_agent_system_prompt(agent.id, "This should fail")
