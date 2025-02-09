import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from pydantic import BaseModel, Field


class AgentMetadata(BaseModel):
    """
    Pydantic model representing an agent's metadata.
    """

    id: str = Field(..., description="Unique identifier for the agent")
    name: str = Field(..., description="Agent's name")
    created_date: datetime = Field(..., description="The date when the agent was created")


class AgentEditMetadata(BaseModel):
    """
    Pydantic model representing an agent's edit metadata.
    """

    name: str | None = Field(None, description="Agent's name")


class AgentRegistry:
    """
    Registry for managing agents and their conversation histories.

    This registry loads agent metadata from an 'agents.json' file located in the config directory.
    Each agent's conversation history is stored separately in a JSON file named
    '{agent_id}_conversation.json'.
    """

    config_dir: Path
    agents_file: Path
    _agents: Dict[str, AgentMetadata]

    def __init__(self, config_dir: Path) -> None:
        """
        Initialize the AgentRegistry, loading metadata from agents.json.

        Args:
            config_dir (Path): Directory containing agents.json and conversation history files
        """
        self.config_dir = config_dir
        if not self.config_dir.exists():
            self.config_dir.mkdir(parents=True, exist_ok=True)

        self.agents_file: Path = self.config_dir / "agents.json"
        self._agents: Dict[str, AgentMetadata] = {}
        self._load_agents_metadata()

    def _load_agents_metadata(self) -> None:
        """
        Load agents' metadata from the agents.json file into memory.
        Only metadata such as 'id', 'name', and 'created_date' is stored.

        Raises:
            Exception: If there is an error loading or parsing the agents metadata file
        """
        if self.agents_file.exists():
            with self.agents_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            # Expect data to be a list of agent metadata dictionaries.
            for item in data:
                try:
                    agent = AgentMetadata.model_validate(item)
                    self._agents[agent.id] = agent
                except Exception as e:
                    raise Exception(f"Invalid agent metadata: {str(e)}")

    def create_agent(self, agent_edit_metadata: AgentEditMetadata) -> AgentMetadata:
        """
        Create a new agent with the provided metadata and initialize its conversation history.

        If no ID is provided, generates a random UUID. If no created_date is provided,
        sets it to the current UTC time.

        Args:
            agent_edit_metadata (AgentEditMetadata): The metadata for the new agent, including name

        Returns:
            AgentMetadata: The metadata of the newly created agent

        Raises:
            ValueError: If an agent with the provided name already exists
            Exception: If there is an error saving the agent metadata or creating the
                conversation history file
        """
        if not agent_edit_metadata.name:
            raise ValueError("Agent name is required")

        # Check if agent name already exists
        for agent in self._agents.values():
            if agent.name == agent_edit_metadata.name:
                raise ValueError(f"Agent with name {agent_edit_metadata.name} already exists")

        agent_metadata = AgentMetadata(
            id=str(uuid.uuid4()),
            name=agent_edit_metadata.name,
            created_date=datetime.now(timezone.utc),
        )

        # Add to in-memory agents
        self._agents[agent_metadata.id] = agent_metadata

        # Save updated agents metadata to file
        agents_list = [agent.model_dump() for agent in self._agents.values()]
        try:
            with self.agents_file.open("w", encoding="utf-8") as f:
                json.dump(agents_list, f, indent=2, default=str)
        except Exception as e:
            # Remove from in-memory if file save fails
            self._agents.pop(agent_metadata.id)
            raise Exception(f"Failed to save agent metadata: {str(e)}")

        # Create empty conversation file
        conversation_file = self.config_dir / f"{agent_metadata.id}_conversation.json"
        try:
            with conversation_file.open("w", encoding="utf-8") as f:
                json.dump([], f)
        except Exception as e:
            # Clean up metadata if conversation file creation fails
            self._agents.pop(agent_metadata.id)
            if self.agents_file.exists():
                self.agents_file.unlink()
            raise Exception(f"Failed to create conversation file: {str(e)}")

        return agent_metadata

    def edit_agent(self, agent_id: str, updated_metadata: AgentEditMetadata) -> None:
        """
        Edit an existing agent's metadata.

        Args:
            agent_id (str): The unique identifier of the agent to edit
            updated_metadata (AgentEditMetadata): The updated metadata for the agent

        Raises:
            KeyError: If the agent_id does not exist
            Exception: If there is an error saving the updated metadata
        """
        if agent_id not in self._agents:
            raise KeyError(f"Agent with id {agent_id} not found")

        current_metadata = self._agents[agent_id]

        # Update only the fields provided in AgentEditMetadata
        if updated_metadata.name is not None:
            current_metadata.name = updated_metadata.name

        # Save updated agents metadata to file
        agents_list = [agent.model_dump() for agent in self._agents.values()]
        try:
            with self.agents_file.open("w", encoding="utf-8") as f:
                json.dump(agents_list, f, indent=2, default=str)
        except Exception as e:
            # Restore original metadata if save fails
            self._agents[agent_id] = AgentMetadata.model_validate(agent_id)
            raise Exception(f"Failed to save updated agent metadata: {str(e)}")

    def delete_agent(self, agent_id: str) -> None:
        """
        Delete an agent and its associated conversation history.

        Args:
            agent_id (str): The unique identifier of the agent to delete.

        Raises:
            KeyError: If the agent_id does not exist
            Exception: If there is an error deleting the agent files
        """
        if agent_id not in self._agents:
            raise KeyError(f"Agent with id {agent_id} not found")

        # Remove from in-memory dict
        self._agents.pop(agent_id)

        # Save updated agents metadata to file
        agents_list = [agent.model_dump() for agent in self._agents.values()]
        try:
            with self.agents_file.open("w", encoding="utf-8") as f:
                json.dump(agents_list, f, indent=2, default=str)
        except Exception as e:
            raise Exception(f"Failed to update agent metadata file: {str(e)}")

        # Delete conversation file if it exists
        conversation_file = self.config_dir / f"{agent_id}_conversation.json"
        if conversation_file.exists():
            try:
                conversation_file.unlink()
            except Exception as e:
                raise Exception(f"Failed to delete conversation file: {str(e)}")

    def clone_agent(self, agent_id: str, new_name: str) -> AgentMetadata:
        """
        Clone an existing agent with a new name, copying over its conversation history.

        Args:
            agent_id (str): The unique identifier of the agent to clone
            new_name (str): The name for the new cloned agent

        Returns:
            AgentMetadata: The metadata of the newly created agent clone

        Raises:
            KeyError: If the source agent_id does not exist
            ValueError: If an agent with new_name already exists
            Exception: If there is an error during the cloning process
        """
        # Check if source agent exists
        if agent_id not in self._agents:
            raise KeyError(f"Source agent with id {agent_id} not found")

        # Create new agent with the provided name
        new_agent = self.create_agent(AgentEditMetadata(name=new_name))

        # Copy conversation history from source agent
        source_conversation = self.load_agent_conversation(agent_id)
        try:
            self.save_agent_conversation(new_agent.id, source_conversation)
            return new_agent
        except Exception as e:
            # Clean up if conversation copy fails
            self.delete_agent(new_agent.id)
            raise Exception(f"Failed to copy conversation history: {str(e)}")

    def get_agent(self, agent_id: str) -> AgentMetadata:
        """
        Get an agent's metadata by ID.

        Args:
            agent_id (str): The unique identifier of the agent.

        Returns:
            AgentMetadata: The agent's metadata.

        Raises:
            KeyError: If the agent_id does not exist
        """
        if agent_id not in self._agents:
            raise KeyError(f"Agent with id {agent_id} not found")
        return self._agents[agent_id]

    def get_agent_by_name(self, name: str) -> AgentMetadata | None:
        """
        Get an agent's metadata by name.

        Args:
            name (str): The name of the agent to find.

        Returns:
            AgentMetadata | None: The agent's metadata if found, None otherwise.
        """
        for agent in self._agents.values():
            if agent.name == name:
                return agent
        return None

    def list_agents(self) -> List[AgentMetadata]:
        """
        Retrieve a list of all agents' metadata stored in the registry.

        Returns:
            List[AgentMetadata]: A list of agent metadata objects.
        """
        return list(self._agents.values())

    def load_agent_conversation(self, agent_id: str) -> List[Dict[str, str]]:
        """
        Load the conversation history for a specified agent.

        The conversation history is stored in a JSON file named
        "{agent_id}_conversation.json" in the config directory.

        Args:
            agent_id (str): The unique identifier of the agent.

        Returns:
            List[Dict[str, str]]: The conversation history as a list of message dictionaries
                with role and content fields matching ConversationRole enum values.
                Returns an empty list if no conversation history exists or if there's an error.
        """
        conversation_file = self.config_dir / f"{agent_id}_conversation.json"
        if conversation_file.exists():
            try:
                with conversation_file.open("r", encoding="utf-8") as f:
                    conversation = json.load(f)
                return conversation
            except Exception:
                # Return an empty conversation if the file is unreadable.
                return []
        return []

    def save_agent_conversation(self, agent_id: str, conversation: List[Dict[str, str]]) -> None:
        """
        Save the conversation history for a specified agent.

        The conversation history is saved to a JSON file named
        "{agent_id}_conversation.json" in the config directory.

        Args:
            agent_id (str): The unique identifier of the agent.
            conversation (List[Dict[str, str]]): The conversation history to save, with each message
                containing 'role' (matching ConversationRole enum values) and 'content' fields.
        """
        conversation_file = self.config_dir / f"{agent_id}_conversation.json"
        try:
            with conversation_file.open("w", encoding="utf-8") as f:
                json.dump(conversation, f, indent=2, ensure_ascii=False)
        except Exception as e:
            # In a production scenario, consider logging this exception
            raise e
