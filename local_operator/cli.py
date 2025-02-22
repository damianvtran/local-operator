"""
Main entry point for the Local Operator CLI application.

This script initializes the DeepSeekCLI interface for interactive chat or,
when the "serve" subcommand is used, starts up the FastAPI server to handle HTTP requests.

The application uses asyncio for asynchronous operation and includes
error handling for graceful failure.

Example Usage:
    python main.py --hosting deepseek --model deepseek-chat
    python main.py --hosting openai --model gpt-4
    python main.py --hosting ollama --model llama2
    python main.py exec "write a hello world program" --hosting ollama --model llama2
"""

import argparse
import asyncio
import math
import os
import traceback
from importlib.metadata import version
from pathlib import Path
from typing import Optional

import uvicorn
from pydantic import SecretStr

from local_operator.admin import add_admin_tools
from local_operator.agents import AgentEditFields, AgentRegistry
from local_operator.clients.openrouter import OpenRouterClient
from local_operator.clients.serpapi import SerpApiClient
from local_operator.config import ConfigManager
from local_operator.credentials import CredentialManager
from local_operator.executor import LocalCodeExecutor
from local_operator.model.configure import configure_model, validate_model
from local_operator.operator import Operator, OperatorType
from local_operator.tools import ToolRegistry

CLI_DESCRIPTION = """
    Local Operator - An environment for agentic AI models to perform tasks on the local device.

    Supports multiple hosting platforms including DeepSeek, OpenAI, Anthropic, Ollama, Kimi
    and Alibaba. Features include interactive chat, safe code execution,
    context-aware conversation history, and built-in safety checks.

    Configure your preferred model and hosting platform via command line arguments. Your
    configuration file is located at ~/.local-operator/config.yml and can be edited directly.
"""


def build_cli_parser() -> argparse.ArgumentParser:
    """
    Build and return the CLI argument parser.

    Returns:
        argparse.ArgumentParser: The CLI argument parser
    """
    # Create parent parser with common arguments
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode for verbose output",
    )
    parent_parser.add_argument(
        "--agent",
        "--agent-name",
        type=str,
        help="Name of the agent to use for this session.  If not provided, the default"
        " agent will be used which does not persist its session.",
        dest="agent_name",
    )
    parent_parser.add_argument(
        "--train",
        action="store_true",
        help="Enable training mode for the operator.  The agent's conversation history will be"
        " saved to the agent's directory after each completed task.  This allows the agent to"
        " learn from its experiences and improve its performance over time.  Omit this flag to"
        " have the agent not store the conversation history, thus resetting it after each session.",
    )

    # Main parser
    parser = argparse.ArgumentParser(description=CLI_DESCRIPTION, parents=[parent_parser])

    parser.add_argument(
        "--version",
        action="version",
        version=f"v{version('local-operator')}",
        help="Show program's version number and exit",
    )
    parser.add_argument(
        "--hosting",
        type=str,
        choices=[
            "deepseek",
            "openai",
            "anthropic",
            "ollama",
            "kimi",
            "alibaba",
            "google",
            "mistral",
            "openrouter",
            "test",
        ],
        help="Hosting platform to use (deepseek, openai, anthropic, ollama, kimi, alibaba, "
        "google, mistral, test, openrouter)",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model to use (e.g., deepseek-chat, gpt-4o, qwen2.5:14b, "
        "claude-3-5-sonnet-20240620, moonshot-v1-32k, qwen-plus, gemini-2.0-flash, "
        "mistral-large-latest, test-model, deepseek/deepseek-chat)",
    )
    subparsers = parser.add_subparsers(dest="subcommand")

    # Credential command
    credential_parser = subparsers.add_parser(
        "credential",
        help="Manage API keys and credentials for different hosting platforms",
        parents=[parent_parser],
    )
    credential_parser.add_argument(
        "--key",
        type=str,
        required=True,
        help="Credential key to update (e.g., DEEPSEEK_API_KEY, "
        "OPENAI_API_KEY, ANTHROPIC_API_KEY, KIMI_API_KEY, ALIBABA_CLOUD_API_KEY, "
        "GOOGLE_AI_STUDIO_API_KEY, MISTRAL_API_KEY, OPENROUTER_API_KEY)",
    )

    # Config command
    config_parser = subparsers.add_parser(
        "config", help="Manage configuration settings", parents=[parent_parser]
    )
    config_subparsers = config_parser.add_subparsers(dest="config_command")
    config_subparsers.add_parser(
        "create", help="Create a new configuration file", parents=[parent_parser]
    )

    # Agents command
    agents_parser = subparsers.add_parser("agents", help="Manage agents", parents=[parent_parser])
    agents_subparsers = agents_parser.add_subparsers(dest="agents_command")
    list_parser = agents_subparsers.add_parser(
        "list", help="List all agents", parents=[parent_parser]
    )
    list_parser.add_argument(
        "--page",
        type=int,
        default=1,
        help="Page number to display (default: 1)",
    )
    list_parser.add_argument(
        "--perpage",
        type=int,
        default=10,
        help="Number of agents per page (default: 10)",
    )
    create_parser = agents_subparsers.add_parser(
        "create", help="Create a new agent", parents=[parent_parser]
    )
    create_parser.add_argument(
        "name",
        type=str,
        help="Name of the agent to create",
    )
    delete_parser = agents_subparsers.add_parser(
        "delete", help="Delete an agent by name", parents=[parent_parser]
    )
    delete_parser.add_argument(
        "name",
        type=str,
        help="Name of the agent to delete",
    )

    # Serve command to start the API server
    serve_parser = subparsers.add_parser(
        "serve", help="Start the FastAPI server", parents=[parent_parser]
    )
    serve_parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address for the server (default: 0.0.0.0)",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for the server (default: 8080)",
    )
    serve_parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable hot reload for the server",
    )

    # Exec command for single execution mode
    exec_parser = subparsers.add_parser(
        "exec",
        help="Execute a single command without starting interactive mode",
        parents=[parent_parser],
    )
    exec_parser.add_argument(
        "command",
        type=str,
        help="The command to execute",
    )

    return parser


def credential_command(args: argparse.Namespace) -> int:
    credential_manager = CredentialManager(Path.home() / ".local-operator")
    credential_manager.prompt_for_credential(args.key, reason="update requested")
    return 0


def config_create_command() -> int:
    """Create a new configuration file."""
    config_manager = ConfigManager(Path.home() / ".local-operator")
    config_manager._write_config(vars(config_manager.config))
    print("Created new configuration file at ~/.local-operator/config.yml")
    return 0


def serve_command(host: str, port: int, reload: bool) -> int:
    """
    Start the FastAPI server using uvicorn.
    """
    print(f"Starting server at http://{host}:{port}")
    uvicorn.run("local_operator.server:app", host=host, port=port, reload=reload)
    return 0


def agents_list_command(args: argparse.Namespace, agent_registry: AgentRegistry) -> int:
    """List all agents."""
    agents = agent_registry.list_agents()
    if not agents:
        print("\n\033[1;33mNo agents found.\033[0m")
        return 0

    # Get pagination arguments
    page = getattr(args, "page", 1)
    per_page = getattr(args, "perpage", 10)

    # Calculate pagination
    total_agents = len(agents)
    total_pages = math.ceil(total_agents / per_page)
    start_idx = (page - 1) * per_page
    end_idx = min(start_idx + per_page, total_agents)

    # Get agents for current page
    page_agents = agents[start_idx:end_idx]
    print("\n\033[1;32m╭─ Agents ────────────────────────────────────\033[0m")
    for i, agent in enumerate(page_agents):
        is_last = i == len(page_agents) - 1
        branch = "└──" if is_last else "├──"
        print(f"\033[1;32m│ {branch} Agent {start_idx + i + 1}\033[0m")
        left_bar = "│ │" if not is_last else "│  "
        print(f"\033[1;32m{left_bar}   • Name: {agent.name}\033[0m")
        print(f"\033[1;32m{left_bar}   • ID: {agent.id}\033[0m")
        print(f"\033[1;32m{left_bar}   • Created: {agent.created_date}\033[0m")
        print(f"\033[1;32m{left_bar}   • Version: {agent.version}\033[0m")
        print(f"\033[1;32m{left_bar}   • Hosting: {agent.hosting or "default"}\033[0m")
        print(f"\033[1;32m{left_bar}   • Model: {agent.model or "default"}\033[0m")
        if not is_last:
            print("\033[1;32m│ │\033[0m")

    # Print pagination info
    print("\033[1;32m│\033[0m")
    print(f"\033[1;32m│ Page {page} of {total_pages} (Total agents: {total_agents})\033[0m")
    if page < total_pages:
        print(f"\033[1;32m│ Use --page {page + 1} to see next page\033[0m")
    print("\033[1;32m╰──────────────────────────────────────────────\033[0m")
    return 0


def agents_create_command(name: str, agent_registry: AgentRegistry) -> int:
    """Create a new agent with the given name."""

    # If name not provided, prompt user for input
    if not name:
        try:
            name = input("\033[1;36mEnter name for new agent: \033[0m").strip()
            if not name:
                print("\n\033[1;31mError: Agent name cannot be empty\033[0m")
                return -1
        except (KeyboardInterrupt, EOFError):
            print("\n\033[1;31mAgent creation cancelled\033[0m")
            return -1

    agent = agent_registry.create_agent(
        AgentEditFields(name=name, security_prompt=None, hosting=None, model=None)
    )
    print("\n\033[1;32m╭─ Created New Agent ───────────────────────────\033[0m")
    print(f"\033[1;32m│ Name: {agent.name}\033[0m")
    print(f"\033[1;32m│ ID: {agent.id}\033[0m")
    print(f"\033[1;32m│ Created: {agent.created_date}\033[0m")
    print(f"\033[1;32m│ Version: {agent.version}\033[0m")
    print("\033[1;32m╰──────────────────────────────────────────────────\033[0m\n")
    return 0


def agents_delete_command(name: str, agent_registry: AgentRegistry) -> int:
    """Delete an agent by name."""
    agents = agent_registry.list_agents()
    matching_agents = [a for a in agents if a.name == name]
    if not matching_agents:
        print(f"\n\033[1;31mError: No agent found with name: {name}\033[0m")
        return -1

    agent = matching_agents[0]
    agent_registry.delete_agent(agent.id)
    print(f"\n\033[1;32mSuccessfully deleted agent: {name}\033[0m")
    return 0


def build_tool_registry(
    executor: LocalCodeExecutor,
    agent_registry: AgentRegistry,
    config_manager: ConfigManager,
    credential_manager: CredentialManager,
) -> ToolRegistry:
    """Build and initialize the tool registry with agent management tools.

    This function creates a new ToolRegistry instance and registers the core agent management tools:
    - create_agent_from_conversation: Creates a new agent from the current conversation
    - edit_agent: Modifies an existing agent's properties
    - delete_agent: Removes an agent from the registry
    - get_agent_info: Retrieves information about agents
    - search_web: Search the web using SERP API

    Args:
        executor: The LocalCodeExecutor instance containing conversation history
        agent_registry: The AgentRegistry for managing agents
        config_manager: The ConfigManager for managing configuration
        credential_manager: The CredentialManager for managing credentials
    Returns:
        ToolRegistry: The initialized tool registry with all agent management tools registered
    """
    tool_registry = ToolRegistry()

    serp_api_key = credential_manager.get_credential("SERP_API_KEY")

    if serp_api_key:
        serp_api_client = SerpApiClient(serp_api_key)
        tool_registry.set_serp_api_client(serp_api_client)
    else:
        serp_api_client = None

    tool_registry.init_tools()

    add_admin_tools(tool_registry, executor, agent_registry, config_manager)

    return tool_registry


def main() -> int:
    try:
        parser = build_cli_parser()
        args = parser.parse_args()

        os.environ["LOCAL_OPERATOR_DEBUG"] = "true" if args.debug else "false"

        config_dir = Path.home() / ".local-operator"
        agents_dir = config_dir / "agents"

        if args.subcommand == "credential":
            return credential_command(args)
        elif args.subcommand == "config":
            if args.config_command == "create":
                return config_create_command()
            else:
                parser.error(f"Invalid config command: {args.config_command}")
        elif args.subcommand == "agents":
            agent_registry = AgentRegistry(agents_dir)
            if args.agents_command == "list":
                return agents_list_command(args, agent_registry)
            elif args.agents_command == "create":
                return agents_create_command(args.name, agent_registry)
            elif args.agents_command == "delete":
                return agents_delete_command(args.name, agent_registry)
            else:
                parser.error(f"Invalid agents command: {args.agents_command}")
        elif args.subcommand == "serve":
            # Use the provided host, port, and reload options for serving the API.
            return serve_command(args.host, args.port, args.reload)

        config_manager = ConfigManager(config_dir)
        credential_manager = CredentialManager(config_dir)
        agent_registry = AgentRegistry(agents_dir)

        # Override config with CLI args where provided
        config_manager.update_config_from_args(args)

        # Get agent if name provided
        agent = None
        if args.agent_name:
            agent = agent_registry.get_agent_by_name(args.agent_name)
            if not agent:
                print(
                    f"\n\033[1;33mNo agent found with name: {args.agent_name}. "
                    f"Creating new agent...\033[0m"
                )
                agent = agent_registry.create_agent(
                    AgentEditFields(
                        name=args.agent_name,
                        security_prompt=None,
                        hosting=None,
                        model=None,
                    )
                )
                print("\n\033[1;32m╭─ Created New Agent ───────────────────────────\033[0m")
                print(f"\033[1;32m│ Name: {agent.name}\033[0m")
                print(f"\033[1;32m│ ID: {agent.id}\033[0m")
                print(f"\033[1;32m│ Created: {agent.created_date}\033[0m")
                print(f"\033[1;32m│ Version: {agent.version}\033[0m")
                print("\033[1;32m╰──────────────────────────────────────────────────\033[0m\n")

        hosting = config_manager.get_config_value("hosting")
        model_name = config_manager.get_config_value("model_name")

        if agent:
            # Get conversation history if agent name provided
            conversation_history = agent_registry.load_agent_conversation(agent.id)

            # Use agent's hosting and model if provided
            if agent.hosting:
                hosting = agent.hosting
            if agent.model:
                model_name = agent.model
        else:
            conversation_history = []

        model_info_client: Optional[OpenRouterClient] = None

        if hosting == "openrouter":
            model_info_client = OpenRouterClient(
                credential_manager.get_credential("OPENROUTER_API_KEY")
            )

        model_configuration = configure_model(
            hosting, model_name, credential_manager, model_info_client
        )

        if not model_configuration.instance:
            error_msg = (
                f"\n\033[1;31mError: Model not found for hosting: "
                f"{hosting} and model: {model_name}\033[0m"
            )
            print(error_msg)
            return -1

        validate_model(hosting, model_name, model_configuration.api_key or SecretStr(""))

        training_mode = False
        if args.train:
            training_mode = True

        executor = LocalCodeExecutor(
            model_configuration=model_configuration,
            detail_conversation_length=config_manager.get_config_value("detail_length", 10),
            agent=agent,
        )

        operator = Operator(
            executor=executor,
            credential_manager=credential_manager,
            config_manager=config_manager,
            model_configuration=model_configuration,
            type=OperatorType.CLI,
            agent_registry=agent_registry,
            current_agent=agent,
            training_mode=training_mode,
        )

        tool_registry = build_tool_registry(
            executor, agent_registry, config_manager, credential_manager
        )

        executor.set_tool_registry(tool_registry)
        executor.load_conversation_history(conversation_history)

        # Start the async chat interface or execute single command
        if args.subcommand == "exec":
            message = asyncio.run(operator.execute_single_command(args.command))
            if message:
                print(message.response)
            return 0
        else:
            asyncio.run(operator.chat())

        return 0
    except Exception as e:
        print(f"\n\033[1;31mError: {str(e)}\033[0m")
        print("\033[1;34m╭─ Stack Trace ────────────────────────────────────\033[0m")
        traceback.print_exc()
        print("\033[1;34m╰──────────────────────────────────────────────────\033[0m")
        print("\n\033[1;33mPlease review and correct the error to continue.\033[0m")
        return -1


if __name__ == "__main__":
    exit(main())
