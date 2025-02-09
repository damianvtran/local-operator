import json
from pathlib import Path

import pytest

from local_operator.agents import AgentEditMetadata, AgentRegistry


@pytest.fixture
def temp_agents_dir(tmp_path: Path) -> Path:
    dir_path = tmp_path / "agents_test"
    dir_path.mkdir()
    return dir_path


def test_create_agent_success(temp_agents_dir: Path):
    registry = AgentRegistry(temp_agents_dir)
    agent_name = "Test Agent"
    edit_metadata = AgentEditMetadata(name=agent_name)
    agent = registry.create_agent(edit_metadata)
    agents = registry.list_agents()
    assert len(agents) == 1
    created_agent = agents[0]
    assert created_agent.name == agent_name
    assert created_agent.id == agent.id

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
    edit_metadata = AgentEditMetadata(name=agent_name)
    registry.create_agent(edit_metadata)
    with pytest.raises(ValueError) as exc_info:
        # Attempt to create another agent with the same name
        registry.create_agent(AgentEditMetadata(name=agent_name))
    assert f"Agent with name {agent_name} already exists" in str(exc_info.value)


def test_edit_agent(temp_agents_dir: Path):
    registry = AgentRegistry(temp_agents_dir)
    original_name = "Original Agent"
    new_name = "Updated Agent"
    edit_metadata = AgentEditMetadata(name=original_name)
    agent = registry.create_agent(edit_metadata)

    # Edit the agent's name and verify the update
    registry.edit_agent(agent.id, AgentEditMetadata(name=new_name))
    updated_agent = registry.get_agent(agent.id)
    assert updated_agent.name == new_name

    # Check that the agents.json file has updated data
    agents_file = temp_agents_dir / "agents.json"
    with agents_file.open("r", encoding="utf-8") as f:
        agents_data = json.load(f)
    found = any(item["id"] == agent.id and item["name"] == new_name for item in agents_data)
    assert found


def test_delete_agent(temp_agents_dir: Path):
    registry = AgentRegistry(temp_agents_dir)
    agent_name = "Agent to Delete"
    agent = registry.create_agent(AgentEditMetadata(name=agent_name))

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
    agent1 = registry.create_agent(AgentEditMetadata(name="Agent One"))
    agent2 = registry.create_agent(AgentEditMetadata(name="Agent Two"))
    agents = registry.list_agents()
    assert len(agents) == 2
    names = {agent.name for agent in agents}
    assert names == {"Agent One", "Agent Two"}
    ids = {agent.id for agent in agents}
    assert ids == {agent1.id, agent2.id}


def test_save_and_load_conversation(temp_agents_dir: Path):
    registry = AgentRegistry(temp_agents_dir)
    agent_name = "Agent Conversation"
    agent = registry.create_agent(AgentEditMetadata(name=agent_name))

    conversation = [
        {"role": "USER", "content": "Hello"},
        {"role": "ASSISTANT", "content": "Hi there!"},
    ]
    registry.save_agent_conversation(agent.id, conversation)
    loaded_conversation = registry.load_agent_conversation(agent.id)
    assert loaded_conversation == conversation


def test_load_nonexistent_conversation(temp_agents_dir: Path):
    registry = AgentRegistry(temp_agents_dir)
    agent = registry.create_agent(AgentEditMetadata(name="Agent No Conversation"))
    conversation_file = temp_agents_dir / f"{agent.id}_conversation.json"
    # Remove the conversation file if it exists to simulate a missing file
    if conversation_file.exists():
        conversation_file.unlink()
    conversation = registry.load_agent_conversation(agent.id)
    assert conversation == []


def test_edit_agent_not_found(temp_agents_dir: Path):
    registry = AgentRegistry(temp_agents_dir)
    with pytest.raises(KeyError):
        registry.edit_agent("non-existent-id", AgentEditMetadata(name="New Name"))


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
        registry.create_agent(AgentEditMetadata(name="Agent Fail"))
    assert str(exc_info.value) == "Fake write failure"


def test_clone_agent(temp_agents_dir: Path):
    registry = AgentRegistry(temp_agents_dir)
    source_name = "Source Agent"
    source_agent = registry.create_agent(AgentEditMetadata(name=source_name))

    # Add some conversation history to source agent
    conversation = [
        {"role": "USER", "content": "Hello"},
        {"role": "ASSISTANT", "content": "Hi there!"},
    ]
    registry.save_agent_conversation(source_agent.id, conversation)

    # Clone the agent
    cloned_agent = registry.clone_agent(source_agent.id, "Cloned Agent")
    assert cloned_agent.name == "Cloned Agent"
    assert cloned_agent.id != source_agent.id

    # Verify conversation was copied
    cloned_conversation = registry.load_agent_conversation(cloned_agent.id)
    assert cloned_conversation == conversation


def test_clone_agent_not_found(temp_agents_dir: Path):
    registry = AgentRegistry(temp_agents_dir)
    with pytest.raises(KeyError):
        registry.clone_agent("non-existent-id", "New Clone")


def test_clone_agent_duplicate_name(temp_agents_dir: Path):
    registry = AgentRegistry(temp_agents_dir)
    source_agent = registry.create_agent(AgentEditMetadata(name="Source"))
    registry.create_agent(AgentEditMetadata(name="Existing"))

    with pytest.raises(ValueError):
        registry.clone_agent(source_agent.id, "Existing")


def test_get_agent_by_name(temp_agents_dir: Path):
    registry = AgentRegistry(temp_agents_dir)

    # Create a test agent
    agent_name = "Test Agent"
    agent = registry.create_agent(AgentEditMetadata(name=agent_name))

    # Test finding existing agent
    found_agent = registry.get_agent_by_name(agent_name)
    assert found_agent is not None
    assert found_agent.id == agent.id
    assert found_agent.name == agent_name

    # Test finding non-existent agent
    not_found = registry.get_agent_by_name("Non Existent")
    assert not_found is None
