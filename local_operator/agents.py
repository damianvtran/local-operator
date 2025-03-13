import json
import logging
import time
import uuid
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any, Dict, List, Optional

import dill
from pydantic import BaseModel, Field

from local_operator.types import (
    CodeExecutionResult,
    ConversationRecord,
    ConversationRole,
)


class AgentData(BaseModel):
    """
    Pydantic model representing an agent's metadata.
    """

    id: str = Field(..., description="Unique identifier for the agent")
    name: str = Field(..., description="Agent's name")
    created_date: datetime = Field(..., description="The date when the agent was created")
    version: str = Field(..., description="The version of the agent")
    security_prompt: str = Field(
        "",
        description="The security prompt for the agent.  Allows a user to explicitly "
        "specify the security context for the agent's code security checks.",
    )
    hosting: str = Field(
        "",
        description="The hosting environment for the agent.  Defaults to ''.",
    )
    model: str = Field(
        "",
        description="The model to use for the agent.  Defaults to ''.",
    )
    description: str = Field(
        "",
        description="A description of the agent.  Defaults to ''.",
    )
    last_message: str = Field(
        "",
        description="The last message sent to the agent.  Defaults to ''.",
    )
    last_message_datetime: datetime = Field(
        datetime.now(timezone.utc),
        description="The date and time of the last message sent to the agent.  "
        "Defaults to the current UTC time.",
    )
    temperature: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Controls randomness in responses"
    )
    top_p: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Controls cumulative probability of tokens to sample from"
    )
    top_k: Optional[int] = Field(None, description="Limits tokens to sample from at each step")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate")
    stop: Optional[List[str]] = Field(
        None, description="List of strings that will stop generation when encountered"
    )
    frequency_penalty: Optional[float] = Field(
        None, description="Reduces repetition by lowering likelihood of repeated tokens"
    )
    presence_penalty: Optional[float] = Field(
        None, description="Increases diversity by lowering likelihood of prompt tokens"
    )
    seed: Optional[int] = Field(None, description="Random number seed for deterministic generation")
    current_working_directory: str = Field(
        ".",
        description="The current working directory for the agent.  Updated whenever the "
        "agent changes its working directory through code execution.  Defaults to '.'",
    )


class AgentEditFields(BaseModel):
    """
    Pydantic model representing an agent's edit metadata.
    """

    name: str | None = Field(None, description="Agent's name")
    security_prompt: str | None = Field(
        None,
        description="The security prompt for the agent.  Allows a user to explicitly "
        "specify the security context for the agent's code security checks.",
    )
    hosting: str | None = Field(
        None,
        description="The hosting environment for the agent.  Defaults to 'openrouter'.",
    )
    model: str | None = Field(
        None,
        description="The model to use for the agent.  Defaults to 'openai/gpt-4o-mini'.",
    )
    description: str | None = Field(
        None,
        description="A description of the agent.  Defaults to ''.",
    )
    last_message: str | None = Field(
        None,
        description="The last message sent to the agent.  Defaults to ''.",
    )
    temperature: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Controls randomness in responses"
    )
    top_p: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Controls cumulative probability of tokens to sample from"
    )
    top_k: Optional[int] = Field(None, description="Limits tokens to sample from at each step")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate")
    stop: Optional[List[str]] = Field(
        None, description="List of strings that will stop generation when encountered"
    )
    frequency_penalty: Optional[float] = Field(
        None, description="Reduces repetition by lowering likelihood of repeated tokens"
    )
    presence_penalty: Optional[float] = Field(
        None, description="Increases diversity by lowering likelihood of prompt tokens"
    )
    seed: Optional[int] = Field(None, description="Random number seed for deterministic generation")
    current_working_directory: str | None = Field(
        None,
        description="The current working directory for the agent.  Updated whenever the "
        "agent changes its working directory through code execution.",
    )


class AgentConversation(BaseModel):
    """
    Pydantic model representing an agent's conversation history.

    This model stores both the version of the conversation format and the actual
    conversation history as a list of ConversationRecord objects.

    Attributes:
        version (str): The version of the conversation format/schema
        conversation (List[ConversationRecord]): List of conversation messages, where each
            message is a ConversationRecord object
    """

    version: str = Field(..., description="The version of the conversation")
    conversation: List[ConversationRecord] = Field(..., description="The conversation history")
    execution_history: List[CodeExecutionResult] = Field(
        default_factory=list, description="The execution history"
    )


class AgentRegistry:
    """
    Registry for managing agents and their conversation histories.

    This registry loads agent metadata from an 'agents.json' file located in the config directory.
    Each agent's conversation history is stored separately in a JSON file named
    '{agent_id}_conversation.json'.
    """

    config_dir: Path
    agents_file: Path
    _agents: Dict[str, AgentData]
    _last_refresh_time: float
    _refresh_interval: float

    def __init__(self, config_dir: Path, refresh_interval: float = 5.0) -> None:
        """
        Initialize the AgentRegistry, loading metadata from agents.json.

        Args:
            config_dir (Path): Directory containing agents.json and conversation history files
            refresh_interval (float): Time in seconds between refreshes of agent data from disk
        """
        self.config_dir = config_dir
        if not self.config_dir.exists():
            self.config_dir.mkdir(parents=True, exist_ok=True)

        self.agents_file: Path = self.config_dir / "agents.json"
        self._agents: Dict[str, AgentData] = {}
        self._last_refresh_time = time.time()
        self._refresh_interval = refresh_interval
        self._load_agents_metadata()

    def _load_agents_metadata(self) -> None:
        """
        Load agents' metadata from the agents.json file into memory.
        Only metadata such as 'id', 'name', 'created_date', and 'version' is stored.

        Raises:
            Exception: If there is an error loading or parsing the agents metadata file
        """
        if self.agents_file.exists():
            with self.agents_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            # Expect data to be a list of agent metadata dictionaries.
            for item in data:
                try:
                    agent = AgentData.model_validate(item)
                    self._agents[agent.id] = agent
                except Exception as e:
                    raise Exception(f"Invalid agent metadata: {str(e)}")

    def create_agent(self, agent_edit_metadata: AgentEditFields) -> AgentData:
        """
        Create a new agent with the provided metadata and initialize its conversation history.

        If no ID is provided, generates a random UUID. If no created_date is provided,
        sets it to the current UTC time.

        Args:
            agent_edit_metadata (AgentEditFields): The metadata for the new agent, including name

        Returns:
            AgentData: The metadata of the newly created agent

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

        agent_metadata = AgentData(
            id=str(uuid.uuid4()),
            created_date=datetime.now(timezone.utc),
            version=version("local-operator"),
            name=agent_edit_metadata.name,
            security_prompt=agent_edit_metadata.security_prompt or "",
            hosting=agent_edit_metadata.hosting or "",
            model=agent_edit_metadata.model or "",
            description=agent_edit_metadata.description or "",
            last_message=agent_edit_metadata.last_message or "",
            last_message_datetime=datetime.now(timezone.utc),
            temperature=agent_edit_metadata.temperature,
            top_p=agent_edit_metadata.top_p,
            top_k=agent_edit_metadata.top_k,
            max_tokens=agent_edit_metadata.max_tokens,
            stop=agent_edit_metadata.stop,
            frequency_penalty=agent_edit_metadata.frequency_penalty,
            presence_penalty=agent_edit_metadata.presence_penalty,
            seed=agent_edit_metadata.seed,
            current_working_directory=agent_edit_metadata.current_working_directory
            or "~/local-operator-home",
        )

        return self.save_agent(agent_metadata)

    def save_agent(self, agent_metadata: AgentData) -> AgentData:
        """
        Save an agent's metadata to the registry.

        Args:
            agent_metadata (AgentData): The metadata of the agent to save
        """
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

    def update_agent(self, agent_id: str, updated_metadata: AgentEditFields) -> AgentData:
        """
        Edit an existing agent's metadata.

        Args:
            agent_id (str): The unique identifier of the agent to edit
            updated_metadata (AgentEditFields): The updated metadata for the agent

        Raises:
            KeyError: If the agent_id does not exist
            Exception: If there is an error saving the updated metadata
        """
        if agent_id not in self._agents:
            raise KeyError(f"Agent with id {agent_id} not found")

        current_metadata = self._agents[agent_id]

        # Update all non-None fields from updated_metadata
        for field, value in updated_metadata.model_dump(exclude_unset=True).items():
            if value is not None:
                setattr(current_metadata, field, value)

        if updated_metadata.last_message is not None:
            current_metadata.last_message_datetime = datetime.now(timezone.utc)

        # Update the in-memory agent data
        self._agents[agent_id] = current_metadata

        # Save updated agents metadata to file
        agents_list = [agent.model_dump() for agent in self._agents.values()]
        try:
            with self.agents_file.open("w", encoding="utf-8") as f:
                json.dump(agents_list, f, indent=2, default=str)
        except Exception as e:
            # Restore original metadata if save fails
            self._agents[agent_id] = AgentData.model_validate(agent_id)
            raise Exception(f"Failed to save updated agent metadata: {str(e)}")

        return current_metadata

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

    def clone_agent(self, agent_id: str, new_name: str) -> AgentData:
        """
        Clone an existing agent with a new name, copying over its conversation history.

        Args:
            agent_id (str): The unique identifier of the agent to clone
            new_name (str): The name for the new cloned agent

        Returns:
            AgentData: The metadata of the newly created agent clone

        Raises:
            KeyError: If the source agent_id does not exist
            ValueError: If an agent with new_name already exists
            Exception: If there is an error during the cloning process
        """
        # Check if source agent exists
        if agent_id not in self._agents:
            raise KeyError(f"Source agent with id {agent_id} not found")

        original_agent = self._agents[agent_id]

        # Create new agent with all fields from original agent
        new_agent = self.create_agent(
            AgentEditFields(
                name=new_name,
                security_prompt=original_agent.security_prompt,
                hosting=original_agent.hosting,
                model=original_agent.model,
                description=original_agent.description,
                last_message=original_agent.last_message,
                temperature=original_agent.temperature,
                top_p=original_agent.top_p,
                top_k=original_agent.top_k,
                max_tokens=original_agent.max_tokens,
                stop=original_agent.stop,
                frequency_penalty=original_agent.frequency_penalty,
                presence_penalty=original_agent.presence_penalty,
                seed=original_agent.seed,
                current_working_directory=original_agent.current_working_directory,
            )
        )

        # Copy conversation history from source agent
        source_conversation = self.load_agent_conversation(agent_id)
        try:
            self.save_agent_conversation(
                new_agent.id,
                source_conversation.conversation,
                source_conversation.execution_history,
            )
            return new_agent
        except Exception as e:
            # Clean up if conversation copy fails
            self.delete_agent(new_agent.id)
            raise Exception(f"Failed to copy conversation history: {str(e)}")

    def _refresh_if_needed(self) -> None:
        """
        Refresh agent metadata from disk if the refresh interval has elapsed.
        """
        current_time = time.time()
        if current_time - self._last_refresh_time > self._refresh_interval:
            self._refresh_agents_metadata()
            self._last_refresh_time = current_time

    def _refresh_agents_metadata(self) -> None:
        """
        Reload agents' metadata from the agents.json file into memory.
        This is used to refresh the in-memory state with changes made by other processes.
        """
        if self.agents_file.exists():
            try:
                with self.agents_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)

                # Create a new dictionary to store refreshed agents
                refreshed_agents = {}

                # Process each agent in the file
                for item in data:
                    try:
                        agent = AgentData.model_validate(item)
                        refreshed_agents[agent.id] = agent
                    except Exception as e:
                        # Log the error but continue processing other agents
                        print(f"Error refreshing agent metadata: {str(e)}")

                # Update the in-memory agents dictionary
                self._agents = refreshed_agents
            except Exception as e:
                # Log the error but don't crash
                print(f"Error refreshing agents metadata: {str(e)}")

    def get_agent(self, agent_id: str) -> AgentData:
        """
        Get an agent's metadata by ID.

        Args:
            agent_id (str): The unique identifier of the agent.

        Returns:
            AgentData: The agent's metadata.

        Raises:
            KeyError: If the agent_id does not exist
        """
        # Refresh agent data from disk if needed
        self._refresh_if_needed()

        if agent_id not in self._agents:
            raise KeyError(f"Agent with id {agent_id} not found")
        return self._agents[agent_id]

    def get_agent_by_name(self, name: str) -> AgentData | None:
        """
        Get an agent's metadata by name.

        Args:
            name (str): The name of the agent to find.

        Returns:
            AgentData | None: The agent's metadata if found, None otherwise.
        """
        # Refresh agent data from disk if needed
        self._refresh_if_needed()

        for agent in self._agents.values():
            if agent.name == name:
                return agent
        return None

    def list_agents(self) -> List[AgentData]:
        """
        Retrieve a list of all agents' metadata stored in the registry.

        Returns:
            List[AgentData]: A list of agent metadata objects.
        """
        # Refresh agent data from disk if needed
        self._refresh_if_needed()

        return list(self._agents.values())

    def load_agent_conversation(self, agent_id: str) -> AgentConversation:
        """
        Load the conversation history for a specified agent.

        The conversation history is stored in a JSON file named
        "{agent_id}_conversation.json" in the config directory.

        Args:
            agent_id (str): The unique identifier of the agent.

        Returns:
            List[ConversationRecord]: The conversation history as a list of ConversationRecord
                objects.
                Returns an empty list if no conversation history exists or if there's an error.
        """
        # Refresh agent data from disk if needed
        self._refresh_if_needed()

        conversation_file = self.config_dir / f"{agent_id}_conversation.json"

        if agent_id not in self._agents:
            raise KeyError(f"Agent with id {agent_id} not found")

        if conversation_file.exists():
            try:
                with conversation_file.open("r", encoding="utf-8") as f:
                    raw_data = json.load(f)

                    try:
                        conversation_data = AgentConversation.model_validate(raw_data)
                        return conversation_data
                    except Exception as e:
                        logging.error(f"Failed to load conversation: {str(e)}")
                        raise Exception(f"Failed to load conversation: {str(e)}")
            except Exception:
                # Return an empty conversation if the file is unreadable.
                return AgentConversation(
                    version="",
                    conversation=[],
                    execution_history=[],
                )
        return AgentConversation(
            version="",
            conversation=[],
            execution_history=[],
        )

    def save_agent_conversation(
        self,
        agent_id: str,
        conversation: List[ConversationRecord],
        execution_history: List[CodeExecutionResult],
    ) -> None:
        """
        Save the conversation history for a specified agent.

        The conversation history is saved to a JSON file named
        "{agent_id}_conversation.json" in the config directory.

        Args:
            agent_id (str): The unique identifier of the agent.
            conversation (List[Dict[str, str]]): The conversation history to save, with each message
                containing 'role' (matching ConversationRole enum values) and 'content' fields.
        """
        agent = self.get_agent(agent_id)

        conversation_file = self.config_dir / f"{agent_id}_conversation.json"
        conversation_data = AgentConversation(
            version=agent.version,
            conversation=conversation,
            execution_history=execution_history,
        )

        try:
            with conversation_file.open("w", encoding="utf-8") as f:
                # Use a custom JSON encoder to handle datetime objects
                json.dump(
                    conversation_data.model_dump(),
                    f,
                    indent=2,
                    ensure_ascii=False,
                    default=lambda o: o.isoformat() if isinstance(o, datetime) else None,
                )
        except Exception as e:
            # In a production scenario, consider logging this exception
            raise e

    def create_autosave_agent(self) -> AgentData:
        """
        Create an autosave agent if it doesn't exist already.

        Returns:
            AgentData: The existing or newly created autosave agent

        Raises:
            Exception: If there is an error creating the agent
        """
        if "autosave" in self._agents:
            return self._agents["autosave"]

        agent_metadata = AgentData(
            id="autosave",
            name="autosave",
            created_date=datetime.now(timezone.utc),
            version=version("local-operator"),
            security_prompt="",
            hosting="",
            model="",
            description="Automatic capture of your last conversation with a Local Operator agent.",
            last_message="",
            last_message_datetime=datetime.now(timezone.utc),
            temperature=None,
            top_p=None,
            top_k=None,
            max_tokens=None,
            stop=None,
            frequency_penalty=None,
            presence_penalty=None,
            seed=None,
            current_working_directory=".",
        )

        return self.save_agent(agent_metadata)

    def get_autosave_agent(self) -> AgentData:
        """
        Get the autosave agent.

        Returns:
            AgentData: The autosave agent

        Raises:
            KeyError: If the autosave agent does not exist
        """
        return self.get_agent("autosave")

    def update_autosave_conversation(
        self, conversation: List[ConversationRecord], execution_history: List[CodeExecutionResult]
    ) -> None:
        """
        Update the autosave agent's conversation.

        Args:
            conversation (List[ConversationRecord]): The conversation history to save
            execution_history (List[CodeExecutionResult]): The execution history to save

        Raises:
            KeyError: If the autosave agent does not exist
        """
        return self.save_agent_conversation("autosave", conversation, execution_history)

    def get_agent_conversation_history(self, agent_id: str) -> List[ConversationRecord]:
        """
        Get the conversation history for a specified agent.

        Args:
            agent_id (str): The unique identifier of the agent.

        Returns:
            List[ConversationRecord]: The conversation history as a list of ConversationRecord
                objects.
        """
        return self.load_agent_conversation(agent_id).conversation

    def get_agent_execution_history(self, agent_id: str) -> List[CodeExecutionResult]:
        """
        Get the execution history for a specified agent.

        Args:
            agent_id (str): The unique identifier of the agent.

        Returns:
            List[CodeExecutionResult]: The execution history as a list of CodeExecutionResult
                objects.
        """
        return self.load_agent_conversation(agent_id).execution_history

    def save_agent_context(self, agent_id: str, context: Any) -> None:
        """Save the agent's context to a file.

        This method serializes the agent's context using dill and saves it to a file
        named "{agent_id}_context.pkl" in the config directory. It handles unpicklable objects
        by converting them to a serializable format.

        Args:
            agent_id (str): The unique identifier of the agent.
            context (Any): The context to save, which can be any object.

        Raises:
            KeyError: If the agent with the specified ID does not exist.
            Exception: If there is an error saving the context.
        """
        if agent_id not in self._agents:
            raise KeyError(f"Agent with id {agent_id} not found")

        context_file = self.config_dir / f"{agent_id}_context.pkl"

        def convert_unpicklable(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: convert_unpicklable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return type(obj)(convert_unpicklable(x) for x in obj)
            elif isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            else:
                try:
                    dill.dumps(obj)
                    return obj
                except Exception:
                    return str(obj)

        try:
            serializable_context = convert_unpicklable(context)
            with context_file.open("wb") as f:
                dill.dump(serializable_context, f)
        except Exception as e:
            raise Exception(f"Failed to save agent context: {str(e)}")

    def load_agent_context(self, agent_id: str) -> Any:
        """Load the agent's context from a file.

        This method deserializes the agent's context using dill from a file
        named "{agent_id}_context.pkl" in the config directory.

        Args:
            agent_id (str): The unique identifier of the agent.

        Returns:
            Any: The loaded context, or None if the context file doesn't exist.

        Raises:
            KeyError: If the agent with the specified ID does not exist.
            Exception: If there is an error loading the context.
        """
        if agent_id not in self._agents:
            raise KeyError(f"Agent with id {agent_id} not found")

        context_file = self.config_dir / f"{agent_id}_context.pkl"
        if not context_file.exists():
            return None

        try:
            with context_file.open("rb") as f:
                return dill.load(f)
        except Exception as e:
            raise Exception(f"Failed to load agent context: {str(e)}")

    def update_agent_state(
        self,
        agent_id: str,
        conversation_history: List[ConversationRecord],
        code_history: List[CodeExecutionResult],
        current_working_directory: Optional[str] = None,
        context: Any = None,
    ) -> None:
        """Save the current agent's conversation history and code execution history.

        This method persists the agent's state by saving the current conversation
        and code execution history to the agent registry. It also updates the agent's
        last message and current working directory if provided.

        Args:
            agent_id: The unique identifier of the agent to update.
            conversation_history: The list of conversation records to save.
            code_history: The list of code execution results to save.
            current_working_directory: Optional new working directory for the agent.
            context: Optional context to save for the agent. If None, the context is not updated.

        Raises:
            KeyError: If the agent with the specified ID does not exist.

        Note:
            This method refreshes agent metadata from disk before updating and
            resets the refresh timer to ensure consistency across processes.
        """
        # Refresh agent data from disk first to ensure we have the latest state
        self._refresh_agents_metadata()

        if agent_id not in self._agents:
            raise KeyError(f"Agent with id {agent_id} not found")

        self.save_agent_conversation(
            agent_id,
            conversation_history,
            code_history,
        )

        # Save the context if provided
        if context is not None:
            self.save_agent_context(agent_id, context)

        # Extract the last assistant message from code history
        assistant_messages = [
            record.message for record in code_history if record.role == ConversationRole.ASSISTANT
        ]

        last_assistant_message = None

        if assistant_messages:
            last_assistant_message = assistant_messages[-1]

        self.update_agent(
            agent_id,
            AgentEditFields(
                name=None,
                security_prompt=None,
                hosting=None,
                model=None,
                description=None,
                last_message=last_assistant_message,
                temperature=None,
                top_p=None,
                top_k=None,
                max_tokens=None,
                stop=None,
                frequency_penalty=None,
                presence_penalty=None,
                seed=None,
                current_working_directory=current_working_directory,
            ),
        )

        # Reset the refresh timer to force other processes to refresh soon
        self._last_refresh_time = 0
