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
from typing import Callable, List, Optional

from local_operator.agents import AgentData, AgentEditFields, AgentRegistry
from local_operator.executor import LocalCodeExecutor


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
        """Create a new agent from an existing conversation history.

        Creates a new agent with the given name and saves the conversation history from the
        provided executor. This allows reusing previous conversations to initialize new agents.

        Args:
            name: Name to give the new agent

        Returns:
            AgentData: The newly created agent's data
        """
        new_agent = agent_registry.create_agent(
            AgentEditFields(
                name=name,
                security_prompt="",
            )
        )
        agent_registry.save_agent_conversation(new_agent.id, executor.conversation_history)
        return new_agent

    return create_agent_from_conversation


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
        """Get information about agents in the registry.

        If an agent_id is provided, returns info for just that agent.
        Otherwise returns info for all agents.

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
        """Create a new empty agent.

        Args:
            name: Name to give the new agent
            security_prompt: Optional security prompt for the agent

        Returns:
            AgentData: The newly created agent's data
        """
        return agent_registry.create_agent(
            AgentEditFields(
                name=name,
                security_prompt=security_prompt or "",
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
        """Edit an existing agent.

        Args:
            agent_id: ID of the agent to edit
            edit_fields: Fields to update on the agent

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
        """Delete an existing agent.

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
        """Get information about agents.

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


def save_conversation_tool(
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

    def save_conversation(filename: str) -> None:
        """Save the current conversation history to a JSON file.

        Saves all messages from the conversation history to the specified file in JSON format.
        Messages are serialized to dictionaries, with proper indentation and UTF-8 encoding.

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
                    [msg if isinstance(msg, dict) else msg.dict() for msg in conversation],
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

        except (IOError, OSError) as e:
            raise ValueError(f"Failed to write conversation to file {filename}: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Error saving conversation: {str(e)}")

    return save_conversation
