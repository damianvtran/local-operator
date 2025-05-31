import copy
import importlib
import inspect
import json
import logging
import os  # Added os
import shutil
import tempfile
import time
import uuid
import zipfile
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import dill
import jsonlines
import yaml
from pydantic import BaseModel, Field

from local_operator.types import Schedule  # Keep existing Schedule import
from local_operator.types import (
    AgentState,
    CodeExecutionResult,
    ConversationRecord,
    ConversationRole,
    ExecutionType,
    ProcessResponseStatus,
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
    tags: List[str] = Field(
        default_factory=list,
        description="Tags for the agent.  Defaults to an empty list.",
    )
    categories: List[str] = Field(
        default_factory=list,
        description="Categories for the agent.  Defaults to an empty list.",
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
    tags: List[str] | None = Field(
        None,
        description="Tags for the agent.  Defaults to an empty list.",
    )
    categories: List[str] | None = Field(
        None,
        description="Categories for the agent.  Defaults to an empty list.",
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


class AgentRegistry:
    """
    Registry for managing agents and their conversation histories.

    This registry loads agent metadata from agent.yml files located in subdirectories
    of the agents directory.
    Each agent has its own directory with the agent ID as the directory name.
    Agent data is stored in separate files within the agent directory:
    - agent.yml: Agent configuration
    - conversation.jsonl: Conversation history
    - execution_history.jsonl: Execution history
    - learnings.jsonl: Learnings from the conversation
    - schedules.jsonl: Scheduled tasks for the agent
    - context.pkl: Agent context
    """

    config_dir: Path
    agents_dir: Path
    agents_file: Path
    _agents: Dict[str, AgentData]
    _last_refresh_time: float
    _refresh_interval: float

    def __init__(self, config_dir: Path, refresh_interval: float = 5.0) -> None:
        """
        Initialize the AgentRegistry, loading metadata from agent.yml files.

        Args:
            config_dir (Path): Directory containing the agents directory
            refresh_interval (float): Time in seconds between refreshes of agent data from disk
        """
        self.config_dir = config_dir
        if not self.config_dir.exists():
            self.config_dir.mkdir(parents=True, exist_ok=True)

        self.agents_dir = self.config_dir / "agents"
        if not self.agents_dir.exists():
            self.agents_dir.mkdir(parents=True, exist_ok=True)

        # For backward compatibility
        self.agents_file: Path = self.config_dir / "agents.json"

        self._agents: Dict[str, AgentData] = {}
        self._last_refresh_time = time.time()
        self._refresh_interval = refresh_interval

        # Migrate old agents if needed
        self.migrate_agents_dir()
        self.migrate_legacy_agents()

        # Load agent metadata
        self._load_agents_metadata()

    def _load_agents_metadata(self) -> None:
        """
        Load agents' metadata from agent.yml files in the agents directory.
        Each agent has its own directory with the agent ID as the directory name.

        Raises:
            Exception: If there is an error loading or parsing the agent metadata files
        """
        # Clear existing agents
        self._agents = {}

        # Iterate through all directories in the agents directory
        for agent_dir in self.agents_dir.iterdir():
            if not agent_dir.is_dir():
                continue

            agent_config_file = agent_dir / "agent.yml"
            if not agent_config_file.exists():
                continue

            try:
                with agent_config_file.open("r", encoding="utf-8") as f:
                    agent_data = yaml.safe_load(f)

                agent = AgentData.model_validate(agent_data)
                self._agents[agent.id] = agent
            except Exception as e:
                logging.error(f"Invalid agent metadata in {agent_dir.name}: {str(e)}")

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
            tags=agent_edit_metadata.tags or [],
            categories=agent_edit_metadata.categories or [],
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

        # Create agent directory if it doesn't exist
        agent_dir = self.agents_dir / agent_metadata.id
        if not agent_dir.exists():
            agent_dir.mkdir(parents=True, exist_ok=True)

        # Save agent metadata to agent.yml
        try:
            with (agent_dir / "agent.yml").open("w", encoding="utf-8") as f:
                yaml.dump(agent_metadata.model_dump(), f, default_flow_style=False)
        except Exception as e:
            # Remove from in-memory if file save fails
            self._agents.pop(agent_metadata.id)
            raise Exception(f"Failed to save agent metadata: {str(e)}")

        # Create empty data files
        try:
            for filename in [
                "conversation.jsonl",
                "execution_history.jsonl",
                "learnings.jsonl",
                "schedules.jsonl",  # Added schedules.jsonl
            ]:
                file_path = agent_dir / filename
                if not file_path.exists():
                    file_path.touch()
        except Exception as e:
            # Clean up metadata if file creation fails
            self._agents.pop(agent_metadata.id)
            if agent_dir.exists():
                shutil.rmtree(agent_dir)
            raise Exception(f"Failed to create agent files: {str(e)}")

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

        current_metadata_obj = self._agents[agent_id]
        original_cwd = current_metadata_obj.current_working_directory

        # Determine if current_working_directory is being explicitly changed
        prospective_changes = updated_metadata.model_dump(exclude_unset=True)
        new_cwd_explicitly_set = prospective_changes.get("current_working_directory")

        # Apply updates to current_metadata_obj
        for field, value in prospective_changes.items():
            if value is not None:  # Ensure we only process fields that were actually provided
                setattr(current_metadata_obj, field, value)

        if updated_metadata.last_message is not None:
            current_metadata_obj.last_message_datetime = datetime.now(timezone.utc)

        # Save agent metadata to agent.yml
        agent_dir = self.agents_dir / agent_id
        if not agent_dir.exists():
            agent_dir.mkdir(parents=True, exist_ok=True)

        try:
            with (agent_dir / "agent.yml").open("w", encoding="utf-8") as f:
                yaml.dump(current_metadata_obj.model_dump(), f, default_flow_style=False)
        except Exception as e:
            logging.error(
                f"Failed to save agent.yml for {agent_id}. In-memory state "
                "might be inconsistent until next refresh."
            )
            raise Exception(f"Failed to save updated agent metadata: {str(e)}")

        # After successful save of agent.yml, check if CWD was explicitly changed
        final_cwd = current_metadata_obj.current_working_directory
        if (
            new_cwd_explicitly_set is not None
            and Path(final_cwd).expanduser().resolve() != Path(original_cwd).expanduser().resolve()
        ):
            try:
                agent_state = self.load_agent_state(agent_id)
                now = datetime.now(timezone.utc)  # Use a single timestamp for both messages

                # Add system message to conversation history
                cwd_conversation_message = ConversationRecord(
                    content=(
                        "<system>Your working directory "
                        f"has been changed to: {final_cwd}</system>"
                    ),
                    role=ConversationRole.USER,
                    timestamp=now,
                    should_summarize=False,
                    ephemeral=False,
                    summarized=False,
                    is_system_prompt=False,
                    files=None,
                    should_cache=False,
                )
                if not isinstance(agent_state.conversation, list):
                    logging.warning(
                        f"agent_state.conversation for agent {agent_id} is "
                        "not a list. Initializing."
                    )
                    agent_state.conversation = []
                agent_state.conversation.append(cwd_conversation_message)

                # Add info message to execution history
                cwd_execution_message = CodeExecutionResult(
                    message=f"The working directory was changed "
                    f"from '{original_cwd}' to '{final_cwd}'.",
                    status=ProcessResponseStatus.SUCCESS,
                    execution_type=ExecutionType.INFO,
                    role=ConversationRole.SYSTEM,
                    timestamp=now,
                    stdout="",
                    stderr="",
                    logging="",
                    code="",
                    formatted_print="",
                    files=[],
                    action=None,
                )
                if not isinstance(agent_state.execution_history, list):
                    logging.warning(
                        f"agent_state.execution_history for agent {agent_id} is "
                        "not a list. Initializing."
                    )
                    agent_state.execution_history = []
                agent_state.execution_history.append(cwd_execution_message)

                self.save_agent_state(agent_id, agent_state)
            except Exception as e:
                logging.error(
                    "Failed to update state (conversation/execution history) for CWD change "
                    f"in agent {agent_id} (update_agent): {str(e)}"
                )
                # Log and continue, as the primary metadata update was successful.

        return current_metadata_obj

    def delete_agent(self, agent_id: str) -> None:
        """
        Delete an agent and its associated files.

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

        # Delete agent directory if it exists
        agent_dir = self.agents_dir / agent_id
        if agent_dir.exists():
            try:
                shutil.rmtree(agent_dir)
            except Exception as e:
                raise Exception(f"Failed to delete agent directory: {str(e)}")

    def clone_agent(self, agent_id: str, new_name: str) -> AgentData:
        """
        Clone an existing agent with a new name, copying over all its files.

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
                tags=original_agent.tags,
                categories=original_agent.categories,
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

        # Copy all files from source agent directory to new agent directory
        source_dir = self.agents_dir / agent_id
        target_dir = self.agents_dir / new_agent.id

        try:
            # Copy all files from source directory to target directory
            for source_file in source_dir.iterdir():
                if source_file.is_file() and source_file.name != "agent.yml":
                    # For JSONL files, only copy if they have content
                    if source_file.suffix == ".jsonl" and source_file.stat().st_size == 0:
                        continue

                    target_file = target_dir / source_file.name
                    shutil.copy2(source_file, target_file)

            return new_agent
        except Exception as e:
            # Clean up if file copy fails
            self.delete_agent(new_agent.id)
            raise Exception(f"Failed to copy agent files: {str(e)}")

    def _load_schedules(self, agent_dir: Path) -> List[Schedule]:
        """Load schedules from schedules.jsonl."""
        schedules_file = agent_dir / "schedules.jsonl"
        schedules: List[Schedule] = []
        if schedules_file.exists() and schedules_file.stat().st_size > 0:
            try:
                with jsonlines.open(schedules_file, mode="r") as reader:
                    for record in reader:
                        schedules.append(Schedule.model_validate(record))
            except Exception as e:
                logging.error(f"Failed to load schedules from {schedules_file}: {str(e)}")
        return schedules

    def _save_schedules(self, agent_dir: Path, schedules: List[Schedule]) -> None:
        """Save schedules to schedules.jsonl."""
        schedules_file = agent_dir / "schedules.jsonl"

        try:
            # Ensure the directory exists
            schedules_file.parent.mkdir(parents=True, exist_ok=True)

            with open(schedules_file, "wb") as f_binary:  # Open in binary for fd
                # Use a jsonlines writer that writes to the binary file object
                # We need to encode to bytes before writing
                with jsonlines.Writer(f_binary) as writer:
                    for schedule_item in schedules:
                        # jsonlines.Writer expects dicts, not already encoded json strings
                        # model_dump(mode="json") returns a dict suitable for json.dumps
                        # but jsonlines handles the serialization to json string and then to bytes.
                        writer.write(schedule_item.model_dump(mode="json"))

        except Exception as e:
            logging.error(f"Failed to save schedules to {schedules_file}: {str(e)}")
            # Optionally, re-raise or handle more gracefully
            raise Exception(f"Failed to save schedules: {str(e)}") from e

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
        Reload agents' metadata from agent.yml files in the agents directory.
        This is used to refresh the in-memory state with changes made by other processes.
        """
        # Clear existing agents
        refreshed_agents = {}

        # First try to load from agent.yml files in the agents directory
        if self.agents_dir.exists():
            # Iterate through all directories in the agents directory
            for agent_dir in self.agents_dir.iterdir():
                if not agent_dir.is_dir():
                    continue

                agent_config_file = agent_dir / "agent.yml"
                if not agent_config_file.exists():
                    continue

                try:
                    with agent_config_file.open("r", encoding="utf-8") as f:
                        agent_data = yaml.safe_load(f)

                    agent = AgentData.model_validate(agent_data)
                    refreshed_agents[agent.id] = agent
                except Exception as e:
                    # Log the error but continue processing other agents
                    logging.error(
                        f"Error refreshing agent metadata from {agent_dir.name}: {str(e)}"
                    )

        # Update the in-memory agents dictionary
        self._agents = refreshed_agents

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

    def load_agent_state(self, agent_id: str) -> AgentState:
        """
        Load the conversation history for a specified agent.

        The conversation history is stored in separate JSONL files in the agent's directory:
        - conversation.jsonl: Conversation history
        - execution_history.jsonl: Execution history
        - learnings.jsonl: Learnings from the conversation
        - schedules.jsonl: Scheduled tasks

        Args:
            agent_id (str): The unique identifier of the agent.

        Returns:
            AgentState: The agent's conversation data.
                Returns an empty conversation if no conversation history exists or if
                there's an error.
        """
        # Refresh agent data from disk if needed
        self._refresh_if_needed()

        if agent_id not in self._agents:
            raise KeyError(f"Agent with id {agent_id} not found")

        agent = self._agents[agent_id]
        agent_dir = self.agents_dir / agent_id

        # Initialize empty conversation data
        conversation_records = []
        execution_history_records = []
        learnings_list = []
        schedules_list: List[Schedule] = []  # Initialize schedules list
        current_plan = None
        instruction_details = None
        agent_system_prompt = ""

        if agent_dir.exists():
            # Load conversation records
            conversation_file = agent_dir / "conversation.jsonl"
            if conversation_file.exists() and conversation_file.stat().st_size > 0:
                try:
                    with jsonlines.open(conversation_file, mode="r") as reader:
                        for record in reader:
                            conversation_records.append(ConversationRecord.model_validate(record))
                except Exception as e:
                    logging.error(f"Failed to load conversation records: {str(e)}")

            # Load execution history records
            execution_history_file = agent_dir / "execution_history.jsonl"
            if execution_history_file.exists() and execution_history_file.stat().st_size > 0:
                try:
                    with jsonlines.open(execution_history_file, mode="r") as reader:
                        for record in reader:
                            execution_history_records.append(
                                CodeExecutionResult.model_validate(record)
                            )
                except Exception as e:
                    logging.error(f"Failed to load execution history records: {str(e)}")

            # Load learnings
            learnings_file = agent_dir / "learnings.jsonl"
            if learnings_file.exists() and learnings_file.stat().st_size > 0:
                try:
                    with jsonlines.open(learnings_file, mode="r") as reader:
                        for record in reader:
                            if isinstance(record, str):
                                learnings_list.append(record)
                            elif isinstance(record, dict) and "learning" in record:
                                learnings_list.append(record["learning"])
                except Exception as e:
                    logging.error(f"Failed to load learnings: {str(e)}")

            # Load schedules
            schedules_list = self._load_schedules(agent_dir)

            # Load plan and instruction details if they exist
            plan_file = agent_dir / "current_plan.txt"
            if plan_file.exists():
                try:
                    with plan_file.open("r", encoding="utf-8") as f:
                        current_plan = f.read()
                except Exception as e:
                    logging.error(f"Failed to load current plan: {str(e)}")

            instruction_file = agent_dir / "instruction_details.txt"
            if instruction_file.exists():
                try:
                    with instruction_file.open("r", encoding="utf-8") as f:
                        instruction_details = f.read()
                except Exception as e:
                    logging.error(f"Failed to load instruction details: {str(e)}")

            try:
                agent_system_prompt = self.get_agent_system_prompt(agent_id)
            except Exception as e:
                logging.error(f"Failed to load agent system prompt: {str(e)}")

        # Create and return the conversation data
        return AgentState(
            version=agent.version,
            conversation=conversation_records,
            execution_history=execution_history_records,
            learnings=learnings_list,
            schedules=schedules_list,  # Pass loaded schedules
            current_plan=current_plan,
            instruction_details=instruction_details,
            agent_system_prompt=agent_system_prompt,
        )

    def save_agent_state(
        self,
        agent_id: str,
        agent_state: AgentState,
    ) -> None:
        """
        Save the agent's state.

        The agent's state is stored in separate files in the agent's directory:
        - conversation.jsonl: Conversation history
        - execution_history.jsonl: Execution history
        - learnings.jsonl: Learnings from the conversation
        - schedules.jsonl: Scheduled tasks
        - current_plan.txt: Current plan text
        - instruction_details.txt: Instruction details text

        Args:
            agent_id (str): The unique identifier of the agent.
            agent_state (AgentState): The agent's state to save.
        """
        agent_dir = self.agents_dir / agent_id

        # Create agent directory if it doesn't exist
        if not agent_dir.exists():
            agent_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Save conversation records
            conversation_file = agent_dir / "conversation.jsonl"
            with jsonlines.open(conversation_file, mode="w") as writer:
                for record in agent_state.conversation:
                    writer.write(record.model_dump())

            # Save execution history records
            execution_history_file = agent_dir / "execution_history.jsonl"
            with jsonlines.open(execution_history_file, mode="w") as writer:
                for record in agent_state.execution_history:
                    writer.write(record.model_dump())

            # Save learnings
            learnings_file = agent_dir / "learnings.jsonl"
            with jsonlines.open(learnings_file, mode="w") as writer:
                for learning in agent_state.learnings:
                    writer.write({"learning": learning})

            # Save schedules
            self._save_schedules(agent_dir, agent_state.schedules)

            # Save current plan if provided
            if agent_state.current_plan is not None:
                plan_file = agent_dir / "current_plan.txt"
                with plan_file.open("w", encoding="utf-8") as f:
                    f.write(agent_state.current_plan)

            # Save instruction details if provided
            if agent_state.instruction_details is not None:
                instruction_file = agent_dir / "instruction_details.txt"
                with instruction_file.open("w", encoding="utf-8") as f:
                    f.write(agent_state.instruction_details)

            if agent_state.agent_system_prompt is not None:
                try:
                    self.set_agent_system_prompt(agent_id, agent_state.agent_system_prompt)
                except Exception as e:
                    logging.error(f"Failed to save agent system prompt: {str(e)}")

        except Exception as e:
            raise Exception(f"Failed to save agent conversation: {str(e)}")

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
            tags=[],
            categories=[],
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
        return self.save_agent_state(
            "autosave",
            AgentState(
                version=version("local-operator"),
                conversation=conversation,
                execution_history=execution_history,
                learnings=[],
                schedules=[],  # Add empty schedules for autosave
                current_plan=None,
                instruction_details=None,
                agent_system_prompt="",
            ),
        )

    def get_agent_conversation_history(self, agent_id: str) -> List[ConversationRecord]:
        """
        Get the conversation history for a specified agent.

        Args:
            agent_id (str): The unique identifier of the agent.

        Returns:
            List[ConversationRecord]: The conversation history as a list of ConversationRecord
                objects.
        """
        return self.load_agent_state(agent_id).conversation

    def get_agent_execution_history(self, agent_id: str) -> List[CodeExecutionResult]:
        """
        Get the execution history for a specified agent.

        Args:
            agent_id (str): The unique identifier of the agent.

        Returns:
            List[CodeExecutionResult]: The execution history as a list of CodeExecutionResult
                objects.
        """
        return self.load_agent_state(agent_id).execution_history

    def save_agent_context(self, agent_id: str, context: Any) -> None:
        """Save the agent's context to a file.

        This method serializes the agent's context using dill and saves it to a file
        named "context.pkl" in the agent's directory. It handles unpicklable objects
        by converting them to a serializable format, including Pydantic models and modules.

        Args:
            agent_id (str): The unique identifier of the agent.
            context (Any): The context to save, which can be any object.

        Raises:
            KeyError: If the agent with the specified ID does not exist.
        """
        if agent_id not in self._agents:
            raise KeyError(f"Agent with id {agent_id} not found")

        # Create agent directory if it doesn't exist
        agent_dir = self.agents_dir / agent_id
        if not agent_dir.exists():
            agent_dir.mkdir(parents=True, exist_ok=True)

        context_file = agent_dir / "context.pkl"

        def convert_unpicklable(obj: Any) -> Any:
            if isinstance(obj, BaseModel):
                # Convert Pydantic models to dictionaries
                return {
                    "__pydantic_model__": obj.__class__.__module__ + "." + obj.__class__.__name__,
                    "data": convert_unpicklable(obj.model_dump()),
                }
            elif isinstance(obj, dict):
                result = {}
                for k, v in obj.items():
                    # Always skip __builtins__ key to avoid serializing builtins
                    if k == "__builtins__":
                        continue
                    # skip any built-in function or built-in class (always available in context)
                    if (
                        inspect.isbuiltin(v) or inspect.isroutine(v) or inspect.isclass(v)
                    ) and getattr(v, "__module__", None) == "builtins":
                        continue
                    try:
                        dill.dumps(k)
                        converted = convert_unpicklable(v)
                        if converted is not None:
                            result[k] = converted
                    except Exception:
                        pass
                return result
            elif isinstance(obj, (list, tuple)):
                return type(obj)(convert_unpicklable(x) for x in obj)
            elif isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            elif hasattr(obj, "__iter__") and hasattr(obj, "__next__"):
                # Handle generator objects by converting to a list
                try:
                    return list(obj)
                except Exception:
                    return str(obj)
            elif inspect.ismodule(obj):
                # Handle modules by storing their name
                return {"__module__": True, "name": obj.__name__}
            elif callable(obj) and hasattr(obj, "__name__"):
                # Preserve functions with a special marker
                try:
                    return {"__callable__": True, "function": dill.dumps(obj)}
                except Exception as e:
                    logging.warning(f"Failed to pickle function {obj.__name__}: {str(e)}")
                    return str(obj)
            else:
                dill.dumps(obj)
                return obj

        try:
            # Make a copy to avoid modifying the original
            if hasattr(context, "copy"):
                context_copy = context.copy()
            else:
                context_copy = copy.deepcopy(context)

            # Remove tools if present as they often contain unpicklable objects
            if isinstance(context_copy, dict):
                context_copy.pop("tools", None)

            serializable_context = convert_unpicklable(context_copy)

            with context_file.open("wb") as f:
                dill.dump(serializable_context, f)
        except Exception as e:
            logging.error(f"Failed to save agent context: {str(e)}")

    def load_agent_context(self, agent_id: str) -> Any:
        """Load the agent's context from a file.

        This method deserializes the agent's context using dill from a file
        named "context.pkl" in the agent's directory. It handles the reconstruction
        of serialized Pydantic models, modules, and other transformed objects.

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

        agent_dir = self.agents_dir / agent_id
        context_file = agent_dir / "context.pkl"

        def reconstruct_objects(obj: Any) -> Any:
            if isinstance(obj, dict) and "__pydantic_model__" in obj:
                # Reconstruct Pydantic model
                model_path = obj["__pydantic_model__"]
                module_name, class_name = model_path.rsplit(".", 1)
                try:
                    module = importlib.import_module(module_name)
                    model_class = getattr(module, class_name)
                    return model_class.model_validate(reconstruct_objects(obj["data"]))
                except (ImportError, AttributeError) as e:
                    logging.error(f"Failed to reconstruct Pydantic model {model_path}: {str(e)}")
                    return obj
            elif isinstance(obj, dict) and "__module__" in obj and obj.get("__module__") is True:
                # Reconstruct module
                try:
                    module_name = obj["name"]
                    return importlib.import_module(module_name)
                except ImportError as e:
                    logging.error(f"Failed to import module {obj['name']}: {str(e)}")
                    return None
            elif (
                isinstance(obj, dict) and "__callable__" in obj and obj.get("__callable__") is True
            ):
                # Reconstruct callable functions
                try:
                    return dill.loads(obj["function"])
                except Exception as e:
                    logging.error(f"Failed to reconstruct callable function: {str(e)}")
                    return None
            elif isinstance(obj, dict):
                return {k: reconstruct_objects(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [reconstruct_objects(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(reconstruct_objects(item) for item in obj)
            else:
                try:
                    return dill.loads(obj)
                except Exception:
                    return obj

        if context_file.exists():
            try:
                with context_file.open("rb") as f:
                    loaded_context = dill.load(f)
                    return reconstruct_objects(loaded_context)
            except Exception as e:
                logging.error(f"Failed to load agent context: {str(e)}")

        # No context found
        return None

    def migrate_agents_dir(self) -> None:
        """
        Migrate nested agents into the right directory.

        This method checks for agents in a nested 'agents/agents' directory structure
        and moves them to the correct location in the agents directory.
        """
        nested_dir = self.agents_dir / "agents"
        if not nested_dir.exists():
            return

        for agent_dir in nested_dir.iterdir():
            if agent_dir.is_dir():
                agent_id = agent_dir.name
                target_dir = self.agents_dir / agent_id

                # Skip if the target directory already exists
                if target_dir.exists():
                    continue

                try:
                    # Create the target directory
                    target_dir.mkdir(parents=True, exist_ok=True)

                    # Move all files from nested directory to the target directory
                    for file_path in agent_dir.glob("*"):
                        target_file = target_dir / file_path.name
                        shutil.copy2(file_path, target_file)

                    # Remove the original directory after successful migration
                    shutil.rmtree(agent_dir)
                    logging.info(f"Migrated agent {agent_id} from nested directory")
                except Exception as e:
                    logging.error(f"Failed to migrate agent {agent_id}: {str(e)}")

    def migrate_legacy_agents(self) -> List[str]:
        """
        Migrate agents from the old format to the new format.

        This method checks for the existence of agents.json and migrates any agents
        that don't have the new file structure. It loads the agent-id_conversation.json
        file and splits the contents into separate files.

        Returns:
            List[str]: List of agent IDs that were migrated
        """
        migrated_agents = []

        # Check if agents.json exists
        if not self.agents_file.exists():
            return migrated_agents

        try:
            # Load agents from agents.json
            with self.agents_file.open("r", encoding="utf-8") as f:
                agents_data = json.load(f)

            # Process each agent
            for agent_data in agents_data:
                try:
                    agent_id = agent_data.get("id")
                    if not agent_id:
                        logging.warning(f"Skipping agent without ID: {agent_data}")
                        continue

                    # Check if agent directory already exists
                    agent_dir = self.agents_dir / agent_id
                    agent_yml_file = agent_dir / "agent.yml"

                    # Skip if agent.yml already exists
                    if agent_yml_file.exists():
                        continue

                    # Create agent directory if it doesn't exist
                    if not agent_dir.exists():
                        agent_dir.mkdir(parents=True, exist_ok=True)

                    # Save agent metadata to agent.yml
                    with agent_yml_file.open("w", encoding="utf-8") as f:
                        yaml.dump(agent_data, f, default_flow_style=False)

                    # Migrate conversation history
                    self._migrate_agent_conversation(agent_id)

                    # Migrate context
                    self._migrate_agent_context(agent_id)

                    migrated_agents.append(agent_id)
                    logging.info(f"Migrated agent: {agent_id}")
                except Exception as e:
                    logging.error(
                        f"Failed to migrate agent {agent_data.get('id', 'unknown')}: {str(e)}"
                    )

            return migrated_agents
        except Exception as e:
            logging.error(f"Failed to migrate agents: {str(e)}")
            return migrated_agents

    def _migrate_agent_conversation(self, agent_id: str) -> bool:
        """
        Migrate an agent's conversation history from the old format to the new format.

        Args:
            agent_id (str): The unique identifier of the agent.

        Returns:
            bool: True if migration was successful, False otherwise.
        """
        old_conversation_file = self.config_dir / f"{agent_id}_conversation.json"
        if not old_conversation_file.exists():
            return False

        try:
            # Load old conversation data
            with old_conversation_file.open("r", encoding="utf-8") as f:
                raw_data = json.load(f)

            # Parse the data
            try:
                old_data = AgentState.model_validate(raw_data)

                # Create agent directory if it doesn't exist
                agent_dir = self.agents_dir / agent_id
                if not agent_dir.exists():
                    agent_dir.mkdir(parents=True, exist_ok=True)

                # Save conversation records
                conversation_file = agent_dir / "conversation.jsonl"
                with jsonlines.open(conversation_file, mode="w") as writer:
                    for record in old_data.conversation:
                        writer.write(record.model_dump())

                # Save execution history records
                execution_history_file = agent_dir / "execution_history.jsonl"
                with jsonlines.open(execution_history_file, mode="w") as writer:
                    for record in old_data.execution_history:
                        writer.write(record.model_dump())

                # Save learnings
                learnings_file = agent_dir / "learnings.jsonl"
                with jsonlines.open(learnings_file, mode="w") as writer:
                    for learning in old_data.learnings:
                        writer.write({"learning": learning})

                # Save current plan if provided
                if old_data.current_plan is not None:
                    plan_file = agent_dir / "current_plan.txt"
                    with plan_file.open("w", encoding="utf-8") as f:
                        f.write(old_data.current_plan)

                # Save instruction details if provided
                if old_data.instruction_details is not None:
                    instruction_file = agent_dir / "instruction_details.txt"
                    with instruction_file.open("w", encoding="utf-8") as f:
                        f.write(old_data.instruction_details)

                logging.info(f"Successfully migrated conversation for agent {agent_id}")
                return True
            except Exception as e:
                logging.error(
                    f"Failed to parse old conversation format for agent {agent_id}: {str(e)}"
                )
                return False
        except Exception as e:
            logging.error(f"Failed to load old conversation file for agent {agent_id}: {str(e)}")
            return False

    def _migrate_agent_context(self, agent_id: str) -> bool:
        """
        Migrate an agent's context from the old format to the new format.

        Args:
            agent_id (str): The unique identifier of the agent.

        Returns:
            bool: True if migration was successful, False otherwise.
        """
        old_context_file = self.config_dir / f"{agent_id}_context.pkl"
        if not old_context_file.exists():
            return False

        try:
            # Load old context
            with old_context_file.open("rb") as f:
                context = dill.load(f)

            # Create agent directory if it doesn't exist
            agent_dir = self.agents_dir / agent_id
            if not agent_dir.exists():
                agent_dir.mkdir(parents=True, exist_ok=True)

            # Save context to new location
            context_file = agent_dir / "context.pkl"
            with context_file.open("wb") as f:
                dill.dump(context, f)

            logging.info(f"Successfully migrated context for agent {agent_id}")
            return True
        except Exception as e:
            logging.error(f"Failed to migrate context for agent {agent_id}: {str(e)}")
            return False

    def import_agent(self, zip_path: Path) -> AgentData:
        """
        Import an agent from a ZIP file.

        The ZIP file should contain agent state files with an agent.yml file.
        A new ID will be assigned to the imported agent, and the current working directory
        will be reset to local-operator-home.

        Args:
            zip_path (Path): Path to the ZIP file containing agent state files

        Returns:
            AgentData: The imported agent's metadata

        Raises:
            ValueError: If the ZIP file is invalid or missing required files
            Exception: If there is an error importing the agent
        """
        # Create a temporary directory to extract the ZIP file
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)

            try:
                # Extract the ZIP file
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(temp_dir_path)

                # Check if agent.yml exists in the extracted files
                agent_yml_path = None
                for root, _, files in os.walk(temp_dir_path):
                    if "agent.yml" in files:
                        agent_yml_path = Path(root) / "agent.yml"
                        break

                if not agent_yml_path:
                    raise ValueError("Missing agent.yml in ZIP file")

                # Load the agent.yml file
                with open(agent_yml_path, "r", encoding="utf-8") as f:
                    agent_data = yaml.safe_load(f)

                # Create a new agent with the imported data
                agent_id = agent_data["id"]
                agent_data["current_working_directory"] = "~/local-operator-home"

                # Remove hosting and model from the agent data to use
                # the default values from user config
                if "hosting" in agent_data:
                    del agent_data["hosting"]
                if "model" in agent_data:
                    del agent_data["model"]

                # Save the updated agent.yml
                with open(agent_yml_path, "w", encoding="utf-8") as f:
                    yaml.dump(agent_data, f, default_flow_style=False)

                # Create the agent directory in the registry
                agent_dir = self.agents_dir / agent_id
                # Always overwrite the agent directory if it exists
                if agent_dir.exists():
                    shutil.rmtree(agent_dir)
                agent_dir.mkdir(parents=True, exist_ok=True)

                # Copy all files from the extracted directory to the agent directory
                extracted_agent_dir = agent_yml_path.parent
                for item in extracted_agent_dir.iterdir():
                    if item.is_file():
                        shutil.copy2(item, agent_dir)

                # Create a new AgentData object directly from the data
                agent_data = AgentData.model_validate(agent_data)

                # Save the agent to the registry
                self.save_agent(agent_data)

                # Update conversation history and learnings to reflect import
                # event so that the agent has continuity
                agent_state = self.load_agent_state(agent_data.id)
                now = datetime.now(timezone.utc)

                # Add a record to execution history to mark the import
                import_execution_record = CodeExecutionResult(
                    message=(
                        "This is an imported agent. All prior conversation and execution "
                        "records are from the agent's training conversations.  "
                        "Send a message to continue and have the agent do something for you."
                    ),
                    status=ProcessResponseStatus.SUCCESS,
                    execution_type=ExecutionType.INFO,
                    role=ConversationRole.SYSTEM,
                    timestamp=now,
                    stdout="",
                    stderr="",
                    logging="",
                    code="",
                    formatted_print="",
                    files=[],
                    action=None,
                )
                agent_state.execution_history.append(import_execution_record)

                # Existing import message for the agent's internal context
                import_message_for_agent = "<system>You were just imported to a new device. I might be a different user, or the same user than the person you were just talking to. I might ask you to do something that is the same or different than the conversation before this. Make sure to use your learnings and the conversation history to replicate what you've learned and give me a consistent experience. Don't acknowledge this message directly, it is just for your reference.</system>"  # noqa: E501
                agent_state.conversation.append(
                    ConversationRecord(
                        content=import_message_for_agent,
                        role=ConversationRole.USER,
                        timestamp=now,
                        should_summarize=False,
                        ephemeral=False,
                        summarized=False,
                        is_system_prompt=False,
                        files=None,
                        should_cache=False,
                    )
                )

                # Existing learnings message
                learnings_message = "I was just imported to a new device. The user might be the same or different than the user that I was just talking to. I'm expected to provide a consistent experience to the user that I was just talking to, so I will need to use learnings and conversation history to replicate the same results for a potentially different topic."  # noqa: E501
                agent_state.learnings.append(learnings_message)

                self.save_agent_state(agent_data.id, agent_state)

                # Return the agent data
                return agent_data

            except zipfile.BadZipFile:
                raise ValueError("Invalid ZIP file")
            except yaml.YAMLError:
                raise ValueError("Invalid YAML in agent.yml")
            except Exception as e:
                raise Exception(f"Error importing agent: {str(e)}")

    def upload_agent_to_radient(
        self,
        radient_client,
        agent_id: Optional[str],
        zip_path: Path,
    ) -> Optional[str]:
        """
        Upload an agent to the Radient Agent Hub.

        Args:
            radient_client: An instance of RadientClient.
            agent_id (Optional[str]): If provided, overwrite the agent with
            this ID. If None, create a new agent.
            zip_path (Path): Path to the ZIP file containing agent data.

        Returns:
            Optional[str]: The new agent ID if created, or None if overwritten.

        Raises:
            RuntimeError: If the upload fails.
        """
        # Check if the agent exists using the provided agent_id
        agent_exists = False
        if agent_id:
            try:
                existing_agent = radient_client.get_agent(agent_id)
                if existing_agent:
                    agent_exists = True
            except Exception as e:
                # Log the error but proceed as if the agent doesn't exist or cannot be verified
                logging.error(f"Failed to check if agent {agent_id} exists on Radient: {str(e)}")

        # If agent_id was provided and the agent exists, overwrite it
        if agent_id and agent_exists:
            radient_client.overwrite_agent_in_marketplace(agent_id, zip_path)
            return None  # Return None when overwriting
        else:
            # If no agent_id was provided, or if the agent_id was provided but the agent
            # doesn't exist (or check failed), upload as a new agent
            return radient_client.upload_agent_to_marketplace(zip_path)

    def download_agent_from_radient(
        self,
        radient_client,
        agent_id: str,
    ) -> AgentData:
        """
        Download an agent from the Radient Agent Hub and import it.

        Args:
            radient_client: An instance of RadientClient.
            agent_id (str): The agent ID to download.

        Returns:
            AgentData: The imported agent's metadata.

        Raises:
            RuntimeError: If the download or import fails.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            zip_path = temp_dir_path / f"{agent_id}.zip"
            radient_client.download_agent_from_marketplace(agent_id, zip_path)
            return self.import_agent(zip_path)

    def export_agent(self, agent_id: str) -> Tuple[Path, str]:
        """
        Export an agent's state files as a ZIP file.

        Args:
            agent_id (str): The unique identifier of the agent to export

        Returns:
            Tuple[Path, str]: A tuple containing the path to the ZIP file and the filename

        Raises:
            KeyError: If the agent is not found
            Exception: If there is an error exporting the agent
        """
        # Verify the agent exists
        agent = self.get_agent(agent_id)

        # Create a temporary directory to store the ZIP file
        temp_dir = tempfile.mkdtemp()
        temp_dir_path = Path(temp_dir)
        filename = f"{agent.name.replace(' ', '_')}.zip"
        zip_path = temp_dir_path / filename

        try:
            # Create the ZIP file
            agent_dir = self.agents_dir / agent_id

            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
                # Add all files from the agent directory to the ZIP file
                for item in agent_dir.iterdir():
                    if item.is_file():
                        zip_file.write(item, arcname=item.name)

            return zip_path, filename
        except Exception as e:
            # Clean up the temporary directory if there's an error
            shutil.rmtree(temp_dir)
            raise Exception(f"Error exporting agent: {str(e)}")

    def update_agent_state(
        self,
        agent_id: str,
        agent_state: AgentState,
        current_working_directory: Optional[str] = None,
        context: Any = None,
    ) -> None:
        """Save the current agent's state.

        This method persists the agent's state by saving the current conversation
        and code execution history to the agent registry. It also updates the agent's
        last message and current working directory if provided.

        Args:
            agent_id: The unique identifier of the agent to update.
            agent_state: The agent state to save.
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

        self.save_agent_state(
            agent_id,
            agent_state,
        )

        # Save the context if provided
        if context is not None:
            self.save_agent_context(agent_id, context)

        # Extract the last assistant message from code history
        assistant_messages = [
            record.message
            for record in agent_state.execution_history
            if record.role == ConversationRole.ASSISTANT
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
                tags=None,
                categories=None,
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

    def get_agent_system_prompt(self, agent_id: str) -> str:
        """
        Get the system prompt for an agent.

        Args:
            agent_id: The unique identifier of the agent

        Returns:
            str: The system prompt content

        Raises:
            KeyError: If the agent with the given ID does not exist
            FileNotFoundError: If the system prompt file does not exist
            IOError: If there is an error reading the system prompt file
        """
        if agent_id not in self._agents:
            raise KeyError(f"Agent with id {agent_id} not found")

        agent_dir = self.agents_dir / agent_id
        system_prompt_path = agent_dir / "system_prompt.md"

        try:
            if not system_prompt_path.exists():
                return ""

            with open(system_prompt_path, "r", encoding="utf-8") as f:
                return f.read()
        except IOError as e:
            logging.error(f"Error reading system prompt for agent {agent_id}: {str(e)}")
            raise IOError(f"Failed to read system prompt: {str(e)}")

    def set_agent_system_prompt(self, agent_id: str, system_prompt: str) -> None:
        """
        Set the system prompt for an agent.

        Args:
            agent_id: The unique identifier of the agent
            system_prompt: The system prompt content to save

        Raises:
            KeyError: If the agent with the given ID does not exist
            IOError: If there is an error writing the system prompt file
        """
        if agent_id not in self._agents:
            raise KeyError(f"Agent with id {agent_id} not found")

        agent_dir = self.agents_dir / agent_id
        system_prompt_path = agent_dir / "system_prompt.md"

        try:
            with open(system_prompt_path, "w", encoding="utf-8") as f:
                f.write(system_prompt)

            # Reset the refresh timer to force other processes to refresh soon
            self._last_refresh_time = 0
        except IOError as e:
            logging.error(f"Error writing system prompt for agent {agent_id}: {str(e)}")
            raise IOError(f"Failed to write system prompt: {str(e)}")

    def get_context_variable(self, agent_id: str, variable_key: str) -> Any:
        """
        Get a specific execution variable for an agent.

        Args:
            agent_id (str): The unique identifier of the agent.
            variable_key (str): The key of the variable to retrieve.

        Returns:
            Any: The value of the execution variable.

        Raises:
            KeyError: If the agent_id or variable_key does not exist.
        """
        context = self.load_agent_context(agent_id)
        if context is None or variable_key not in context:
            raise KeyError(f"Execution variable '{variable_key}' not found for agent {agent_id}")
        return context[variable_key]

    def create_context_variable(self, agent_id: str, key: str, value: Any) -> Dict[str, Any]:
        """
        Create a new execution variable for an agent.

        Args:
            agent_id (str): The unique identifier of the agent.
            key (str): The key of the new variable.
            value (Any): The value of the new variable.

        Returns:
            Dict[str, Any]: The updated dictionary of all execution variables.

        Raises:
            KeyError: If the agent_id does not exist.
            ValueError: If the variable key already exists.
        """
        context = self.load_agent_context(agent_id)
        if context is None:
            context = {}
        if key in context:
            raise ValueError(
                f"Execution variable '{key}' already exists for agent {agent_id}. "
                "Use update operation instead."
            )
        context[key] = value
        self.save_agent_context(
            agent_id, context
        )  # Persists the change to agent.yml and updates in-memory
        return context

    def update_context_variable(self, agent_id: str, key: str, value: Any) -> Dict[str, Any]:
        """
        Update an existing execution variable for an agent.

        Args:
            agent_id (str): The unique identifier of the agent.
            key (str): The key of the variable to update.
            value (Any): The new value for the variable.

        Returns:
            Dict[str, Any]: The updated dictionary of all execution variables.

        Raises:
            KeyError: If the agent_id or variable_key does not exist.
        """
        context = self.load_agent_context(agent_id)
        if context is None or key not in context:
            raise KeyError(f"Execution variable '{key}' not found for agent {agent_id}")
        context[key] = value
        self.save_agent_context(agent_id, context)
        return context

    def delete_context_variable(self, agent_id: str, key: str) -> Dict[str, Any]:
        """
        Delete an execution variable for an agent.

        Args:
            agent_id (str): The unique identifier of the agent.
            key (str): The key of the variable to delete.

        Returns:
            Dict[str, Any]: The updated dictionary of all execution variables.

        Raises:
            KeyError: If the agent_id or variable_key does not exist.
        """
        context = self.load_agent_context(agent_id)
        if context is None or key not in context:
            raise KeyError(f"Execution variable '{key}' not found for agent {agent_id}")
        del context[key]
        self.save_agent_context(agent_id, context)
        return context
