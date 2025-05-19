"""Admin module for managing agents and conversations in the Local Operator system.

This module provides tools and utilities for managing agents and their conversations. It includes
functions for creating, editing, deleting and retrieving agent information, as well as saving
conversation histories. The module is designed to work with the AgentRegistry and LocalCodeExecutor
classes to provide a complete agent management interface.

The main components are:
- Agent management tools (create, edit, delete, list agents)
- Conversation management tools (save conversations)
- Tool factory functions that create callable tools for use in the system

Each tool factory function returns a properly typed callable that can be registered with the
tool registry for use by the system.

Typical usage example:
    agent_registry = AgentRegistry(config_dir)
    executor = LocalCodeExecutor()

    # Create tool functions
    create_agent = create_agent_tool(agent_registry)
    save_conv = save_conversation_tool(executor)

    # Use the tools
    new_agent = create_agent("Agent1")
    save_conv("conversation.json")
"""

import json
import platform
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from local_operator.agents import AgentData, AgentEditFields, AgentRegistry
from local_operator.config import Config, ConfigManager
from local_operator.executor import LocalCodeExecutor
from local_operator.notebook import save_code_history_to_notebook
from local_operator.operator import ConversationRole
from local_operator.tools.general import ToolRegistry
from local_operator.types import AgentState


def create_agent_from_conversation_tool(
    executor: LocalCodeExecutor, agent_registry: AgentRegistry
) -> Callable[[str], AgentData]:
    """Create a tool function that creates a new agent from the current conversation history.

    This function returns a callable that can be used as a tool to create new agents from the
    current conversation history. The returned function takes a name parameter and creates a new
    agent with that name, copying over the conversation history from the provided executor.

    Args:
        executor: The executor containing the conversation history to use
        agent_registry: The registry to create and store the new agent in

    Returns:
        Callable[[str], AgentData]: A function that takes a name string and returns the newly
            created agent's data

    Raises:
        ValueError: If the executor has no conversation history
        RuntimeError: If there are issues creating or saving the new agent
    """

    def create_agent_from_conversation(name: str) -> AgentData:
        """Create a new Local Operator agent initialized with the current conversation history.

        Creates a new agent with the given name and saves the current conversation history from the
        provided executor, excluding the create agent request itself. This allows reusing previous
        conversations to initialize new agents with existing context and knowledge.

        Args:
            name: Name to give the new agent

        Returns:
            AgentData: The newly created agent's data
        """
        # Find index of last user message by iterating backwards
        conversation_history = executor.agent_state.conversation
        execution_history = executor.agent_state.execution_history
        last_user_idx = None

        for i in range(len(conversation_history) - 1, -1, -1):
            if conversation_history[i].role == ConversationRole.USER:
                last_user_idx = i
                break

        # Save history up to the last user message (excluding the create
        # agent request itself)
        cutoff_idx = last_user_idx if last_user_idx is not None else 0
        conversation_history_to_save = conversation_history[:cutoff_idx]
        execution_history_to_save = execution_history[:cutoff_idx]

        new_agent = agent_registry.create_agent(
            AgentEditFields(
                name=name,
                security_prompt="",
                hosting="",
                model="",
                description="",
                last_message="",
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
                current_working_directory=None,
            )
        )
        agent_registry.save_agent_state(
            new_agent.id,
            AgentState(
                version=new_agent.version,
                conversation=conversation_history_to_save,
                execution_history=execution_history_to_save,
                learnings=executor.agent_state.learnings,
                current_plan=executor.agent_state.current_plan,
                instruction_details=executor.agent_state.instruction_details,
                agent_system_prompt=executor.agent_state.agent_system_prompt,
            ),
        )
        return new_agent

    return create_agent_from_conversation


def save_agent_training_tool(
    executor: LocalCodeExecutor, agent_registry: AgentRegistry
) -> Callable[[], AgentData]:
    """Create a tool function that saves the current conversation to train an agent.

    This function returns a callable that can be used as a tool to save the current conversation
    history as training data for the agent. The tool will save all conversation turns up to the
    second-to-last user message, excluding the final save request itself. This allows the agent
    to learn from the interaction and build up its knowledge over time through training.

    The saved conversation history becomes part of the agent's training data and helps shape its
    responses in future interactions. Each time this tool is used, it adds the current conversation
    to the agent's existing training data.

    Args:
        executor: The executor containing the conversation history to use as training data
        agent_registry: The registry to update the agent's training data in

    Returns:
        Callable[[], AgentData]: A function that saves the conversation and returns the updated
            agent's data

    Raises:
        ValueError: If the executor has no current agent
        RuntimeError: If there are issues saving the training data
    """

    def save_agent_training() -> AgentData:
        """Save the current conversation as training data for the agent.

        Saves the conversation history from the executor as training data for the current agent,
        rewinding to before the last user input to exclude the save request itself. This allows
        the agent to learn from the current interaction and improve its responses over time.

        Returns:
            AgentData: The updated agent's data with new training conversation saved

        Raises:
            ValueError: If there is no current agent set in the executor
        """
        if not executor.agent:
            raise ValueError("No current agent set in executor")

        # Find index of last user message by iterating backwards
        conversation_history = executor.agent_state.conversation
        execution_history = executor.agent_state.execution_history
        last_user_idx = None

        for i in range(len(conversation_history) - 1, -1, -1):
            if conversation_history[i].role == ConversationRole.USER:
                last_user_idx = i
                break

        # Save history up to the last user message (excluding the save request itself)
        cutoff_idx = last_user_idx if last_user_idx is not None else 0
        conversation_history_to_save = conversation_history[:cutoff_idx]
        execution_history_to_save = execution_history[:cutoff_idx]

        agent_state_to_save = AgentState(
            conversation=conversation_history_to_save,
            execution_history=execution_history_to_save,
            **executor.agent_state.model_dump(exclude={"conversation", "execution_history"}),
        )

        agent_registry.save_agent_state(
            executor.agent.id,
            agent_state_to_save,
        )
        return executor.agent

    return save_agent_training


def list_agent_info_tool(
    agent_registry: AgentRegistry,
) -> Callable[[Optional[str]], List[AgentData]]:
    """Create a tool function that lists agents or gets info for a specific agent.

    This function returns a callable that can be used as a tool to list all agents in the registry
    or get details for a specific agent if an ID is provided.

    Args:
        agent_registry: The registry containing the agents

    Returns:
        Callable[[Optional[str]], List[AgentData]]: A function that takes an optional agent ID and
            returns either a list of all agents or a single-item list with the specified agent

    Raises:
        KeyError: If a specific agent ID is provided but not found
    """

    def list_agent_info(agent_id: Optional[str] = None) -> List[AgentData]:
        """List all Local Operator agents or get details for a specific agent.

        If an agent_id is provided, returns info for just that agent.
        Otherwise returns info for all registered Local Operator agents including their names,
        IDs, and security prompts.

        Args:
            agent_id: Optional ID of specific agent to get info for

        Returns:
            List[AgentData]: List of agent data objects

        Raises:
            KeyError: If agent_id is provided but not found
        """
        if agent_id:
            return [agent_registry.get_agent(agent_id)]
        return agent_registry.list_agents()

    return list_agent_info


def create_agent_tool(agent_registry: AgentRegistry) -> Callable[[str, Optional[str]], AgentData]:
    """Create a tool function that creates a new empty agent.

    Args:
        agent_registry: The registry to create and store the new agent in

    Returns:
        Callable[[str, Optional[str]], AgentData]: A function that takes a name string and optional
            security prompt and returns the newly created agent's data

    Raises:
        RuntimeError: If there are issues creating the new agent
    """

    def create_agent(name: str, security_prompt: Optional[str] = None) -> AgentData:
        """Create a new empty Local Operator agent with the specified name and security settings.

        Creates a fresh agent instance that can be used to handle conversations and execute
        commands. The security prompt defines the agent's permissions and operating constraints.

        Args:
            name: Name to give the new agent
            security_prompt: Optional security prompt for the agent to define its permissions

        Returns:
            AgentData: The newly created agent's data
        """
        return agent_registry.create_agent(
            AgentEditFields(
                name=name,
                security_prompt=security_prompt or "",
                hosting="",
                model="",
                description="",
                last_message="",
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
                current_working_directory=None,
            )
        )

    return create_agent


def edit_agent_tool(agent_registry: AgentRegistry) -> Callable[[str, AgentEditFields], AgentData]:
    """Create a tool function that edits an existing agent.

    Args:
        agent_registry: The registry containing the agents

    Returns:
        Callable[[str, AgentEditFields], AgentData]: A function that takes an agent ID and fields
            and returns the updated agent data

    Raises:
        ValueError: If the agent is not found
        RuntimeError: If there are issues updating the agent
    """

    def edit_agent(agent_id: str, edit_fields: AgentEditFields) -> AgentData:
        """Edit an existing Local Operator agent's name or security settings.

        Modifies the specified agent's properties like name and security prompt while preserving
        its conversation history and other data.

        Args:
            agent_id: ID of the agent to edit
            edit_fields: Fields to update on the agent (name and/or security prompt)

        Returns:
            AgentData: The updated agent data
        """
        result = agent_registry.update_agent(agent_id, edit_fields)
        if result is None:
            raise ValueError(f"Agent not found with ID: {agent_id}")
        return result

    return edit_agent


def delete_agent_tool(agent_registry: AgentRegistry) -> Callable[[str], None]:
    """Create a tool function that deletes an existing agent.

    Args:
        agent_registry: The registry containing the agents

    Returns:
        Callable[[str], None]: A function that takes an agent ID and deletes the agent

    Raises:
        ValueError: If the agent is not found
        RuntimeError: If there are issues deleting the agent
    """

    def delete_agent(agent_id: str) -> None:
        """Delete a Local Operator agent and all its associated data.

        Permanently removes the specified agent, including its conversation history,
        security settings, and other metadata.

        Args:
            agent_id: ID of the agent to delete
        """
        agent_registry.delete_agent(agent_id)

    return delete_agent


def get_agent_info_tool(
    agent_registry: AgentRegistry,
) -> Callable[[Optional[str]], List[AgentData]]:
    """Create a tool function that retrieves agent information.

    Args:
        agent_registry: The registry containing the agents

    Returns:
        Callable[[Optional[str]], List[AgentData]]: A function that takes an optional agent ID and
            returns either all agents or the specified agent

    Raises:
        ValueError: If a specific agent ID is provided but not found
    """

    def get_agent_info(agent_id: Optional[str] = None) -> List[AgentData]:
        """Get detailed information about Local Operator agents.

        Retrieves comprehensive information about either all registered agents or a specific
        agent, including name, ID, security settings, and metadata.

        Args:
            agent_id: Optional ID of a specific agent to retrieve

        Returns:
            List[AgentData]: List of agent data (single item if agent_id provided)
        """
        if agent_id:
            agent = agent_registry.get_agent(agent_id)
            return [agent] if agent else []
        return agent_registry.list_agents()

    return get_agent_info


def save_conversation_raw_json_tool(
    executor: LocalCodeExecutor,
) -> Callable[[str], None]:
    """Create a tool function that saves conversation history to disk in JSON format.

    This function creates a tool that can save the conversation history from a LocalCodeExecutor
    instance to a JSON file on disk. The conversation history includes all messages exchanged
    between the user and the AI model.

    Args:
        executor: The LocalCodeExecutor instance containing the conversation history to be saved

    Returns:
        Callable[[str], None]: A function that accepts a filename string and saves the
            conversation history to that location in JSON format

    Raises:
        ValueError: If the provided filename is invalid or the file cannot be written
        RuntimeError: If there are unexpected issues during the save operation
    """

    def save_conversation_raw_json(filename: str) -> None:
        """Save the current Local Operator conversation history to a raw JSON file.

        Exports the complete conversation history including all messages between user and agent,
        commands executed, and their results. The file can be used for analysis or to initialize
        new agents.

        Args:
            filename: The path where the JSON file should be saved. Should include the full
                path and filename with .json extension.

        Raises:
            ValueError: If the file cannot be opened or written to due to permissions,
                invalid path, or disk space issues
            RuntimeError: If there are unexpected errors during JSON serialization or
                other operations
        """
        try:
            # Get conversation history
            conversation = executor.get_conversation_history()

            # Save to JSON file
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(
                    [msg.dict() for msg in conversation],
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

        except (IOError, OSError) as e:
            raise ValueError(f"Failed to write conversation to file {filename}: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Error saving conversation: {str(e)}")

    return save_conversation_raw_json


def get_config_tool(config_manager: ConfigManager) -> Callable[[], Config]:
    """Create a tool function that retrieves the current configuration.

    This function returns a callable that can be used as a tool to get the current
    configuration settings from the config manager.

    Args:
        config_manager: The ConfigManager instance to get config from

    Returns:
        Callable[[], Config]: A function that returns the current Config object
    """

    def get_config() -> Config:
        """Get the current Local Operator configuration settings.

        Retrieves the current configuration values including:
        - conversation_length: Number of conversation messages to retain
        - detail_length: Maximum length of detailed conversation history
        - hosting: AI model hosting provider
        - model_name: Name of the AI model to use

        Returns:
            Config: The current configuration settings object
        """
        return config_manager.get_config()

    return get_config


def update_config_tool(config_manager: ConfigManager) -> Callable[[Dict[str, Any]], None]:
    """Create a tool function that updates configuration settings.

    This function returns a callable that can be used as a tool to update the conversation
    and model configuration values in the config manager.

    Args:
        config_manager: The ConfigManager instance to update config in

    Returns:
        Callable[[Dict[str, Any]], None]: A function that takes a dictionary of updates
            and applies them to the configuration

    Raises:
        ValueError: If the update dictionary contains invalid keys or values
        RuntimeError: If there are issues writing the updated configuration
    """

    def update_config(values: Dict[str, Any]) -> None:
        """Update Local Operator configuration settings.

        Updates conversation and model configuration values including:
        - conversation_length: Number of messages to retain
        - detail_length: Maximum length of detailed history
        - hosting: AI model hosting provider
        - model_name: Name of AI model to use
        - rag_enabled: Whether RAG is enabled

        Args:
            updates: Dictionary mapping config keys to their new values

        Raises:
            ValueError: If any of the update keys or values are invalid
            RuntimeError: If there are issues saving the configuration
        """
        try:
            config_manager.update_config(values)
        except Exception as e:
            raise RuntimeError(f"Failed to update configuration: {str(e)}")

    return update_config


def open_agents_config_tool(agent_registry: AgentRegistry) -> Callable[[], None]:
    """Create a tool function that opens the agents directory.

    This function returns a callable that opens the agents directory in the default system file
    explorer. The directory location is determined from the agent registry's config directory.

    Args:
        agent_registry: The AgentRegistry instance containing the config directory path

    Returns:
        Callable[[], None]: A function that opens the agents directory

    Raises:
        RuntimeError: If there are issues opening the directory
    """

    def open_agents_config() -> None:
        """Open the agents directory in the default system file explorer.

        Opens the agents directory located in the agent registry's config directory
        using the system's default file explorer.

        Raises:
            RuntimeError: If the directory cannot be opened
        """
        try:
            agents_dir = agent_registry.agents_dir
            if not agents_dir.exists():
                raise RuntimeError(f"Agents directory not found at {agents_dir}")

            system = platform.system()
            if system == "Darwin":  # macOS
                subprocess.run(["open", str(agents_dir)], check=True)
            elif system == "Windows":
                subprocess.run(["start", str(agents_dir)], shell=True, check=True)
            else:  # Linux and other Unix-like
                subprocess.run(["xdg-open", str(agents_dir)], check=True)

        except Exception as e:
            raise RuntimeError(f"Failed to open agents directory: {str(e)}")

    return open_agents_config


def open_settings_config_tool(config_manager: ConfigManager) -> Callable[[], None]:
    """Create a tool function that opens the main settings configuration file.

    This function returns a callable that opens the config.yml file in the default system editor.
    The file location is determined from the config manager's config directory.

    Args:
        config_manager: The ConfigManager instance containing the config directory path

    Returns:
        Callable[[], None]: A function that opens the settings configuration file

    Raises:
        RuntimeError: If there are issues opening the configuration file
    """

    def open_settings_config() -> None:
        """Open the settings configuration file in the default system editor.

        Opens the config.yml file located in the config manager's directory
        using the system's default application for YAML files.

        Raises:
            RuntimeError: If the configuration file cannot be opened
        """
        try:
            config_file = config_manager.config_dir / "config.yml"
            if not config_file.exists():
                raise RuntimeError(f"Settings configuration file not found at {config_file}")

            system = platform.system()
            if system == "Darwin":  # macOS
                subprocess.run(["open", str(config_file)], check=True)
            elif system == "Windows":
                subprocess.run(["start", str(config_file)], shell=True, check=True)
            else:  # Linux and other Unix-like
                subprocess.run(["xdg-open", str(config_file)], check=True)

        except Exception as e:
            raise RuntimeError(f"Failed to open settings configuration file: {str(e)}")

    return open_settings_config


def save_conversation_history_to_notebook_tool(
    executor: LocalCodeExecutor,
) -> Callable[[str], None]:
    """Create a tool function that saves conversation history to a notebook in ipynb format.

    This function returns a callable that, when called, saves the executed code blocks
    along with their outputs into an IPython notebook file (.ipynb). The notebook
    is saved in JSON format.

    Returns:
        Callable[[str], None]: A function that saves the code blocks to a notebook file.
    """

    def save_conversation_history_to_notebook(file_path: str = "notebook.ipynb") -> None:
        """Save the conversation history to an IPython notebook file (.ipynb).

        This function retrieves the code blocks and their execution results from the
        executor, formats them as notebook cells, and saves them to a .ipynb file
        in JSON format.

        Args:
            file_path (str): The path to save the notebook to. Defaults to "notebook.ipynb".

        Raises:
            Exception: If there is an error during notebook creation or file saving.
        """
        try:
            save_code_history_to_notebook(
                code_history=executor.agent_state.execution_history,
                model_configuration=executor.model_configuration,
                max_conversation_history=executor.max_conversation_history,
                detail_conversation_length=executor.detail_conversation_length,
                max_learnings_history=executor.max_learnings_history,
                file_path=Path(file_path),
            )
            print(f"Notebook saved to {file_path}")

        except Exception as e:
            raise Exception(f"Failed to save conversation history to notebook: {str(e)}")

    return save_conversation_history_to_notebook


def add_admin_tools(
    tool_registry: ToolRegistry,
    executor: LocalCodeExecutor,
    agent_registry: AgentRegistry,
    config_manager: ConfigManager,
) -> None:
    """Add admin tools to the tool registry.

    This function adds the following tools to the tool registry:
    - create_agent_from_conversation_tool
    - list_agent_info_tool
    - create_agent_tool
    - edit_agent_tool
    - delete_agent_tool
    - get_agent_info_tool
    - save_conversation_raw_json_tool
    - get_config_tool
    - update_config_tool
    - open_agents_config_tool
    - open_settings_config_tool
    - save_conversation_history_to_notebook_tool
    Args:
        tool_registry: The ToolRegistry instance to add tools to
    """
    tool_registry.add_tool(
        "create_agent_from_conversation",
        create_agent_from_conversation_tool(
            executor,
            agent_registry,
        ),
    )
    tool_registry.add_tool(
        "edit_agent",
        edit_agent_tool(agent_registry),
    )
    tool_registry.add_tool(
        "delete_agent",
        delete_agent_tool(agent_registry),
    )
    tool_registry.add_tool(
        "get_agent_info",
        get_agent_info_tool(agent_registry),
    )
    tool_registry.add_tool(
        "save_conversation_raw_json",
        save_conversation_raw_json_tool(executor),
    )
    tool_registry.add_tool(
        "get_config",
        get_config_tool(config_manager),
    )
    tool_registry.add_tool(
        "update_config",
        update_config_tool(config_manager),
    )
    tool_registry.add_tool(
        "save_agent_training",
        save_agent_training_tool(executor, agent_registry),
    )
    tool_registry.add_tool(
        "open_agents_config",
        open_agents_config_tool(agent_registry),
    )
    tool_registry.add_tool(
        "open_settings_config",
        open_settings_config_tool(config_manager),
    )
    tool_registry.add_tool(
        "save_conversation_history_to_notebook",
        save_conversation_history_to_notebook_tool(executor),
    )
