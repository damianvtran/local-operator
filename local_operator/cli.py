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
import platform
import subprocess
import traceback
from importlib.metadata import version
from pathlib import Path
from typing import Optional

import uvicorn

from local_operator.agents import AgentData  # Import AgentData type
from local_operator.agents import AgentEditFields, AgentRegistry
from local_operator.bootstrap import initialize_operator  # Import the new function
from local_operator.clients.radient import RadientClient
from local_operator.config import ConfigManager
from local_operator.console import VerbosityLevel  # Import VerbosityLevel
from local_operator.credentials import CredentialManager
from local_operator.env import get_env_config
from local_operator.helpers import setup_cross_platform_environment
from local_operator.jobs import JobManager  # Added
from local_operator.operator import OperatorType
from local_operator.scheduler_service import SchedulerService  # Added
from local_operator.server.utils.websocket_manager import WebSocketManager  # Added

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
            "radient",
            "deepseek",
            "openai",
            "anthropic",
            "ollama",
            "kimi",
            "alibaba",
            "google",
            "mistral",
            "openrouter",
            "xai",
            "test",
        ],
        help="Hosting platform to use (radient, deepseek, openai, anthropic, ollama, kimi, "
        "alibaba, google, mistral, test, openrouter, xai)",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model to use (e.g., deepseek-chat, gpt-4o, qwen2.5:14b, "
        "claude-3-5-sonnet-20240620, moonshot-v1-32k, qwen-plus, gemini-2.0-flash, "
        "mistral-large-latest, test-model, deepseek/deepseek-chat)",
    )
    parser.add_argument(
        "--run-in",
        type=str,
        help="The working directory to run the operator in.  Must be a valid directory.",
        dest="run_in",
    )
    subparsers = parser.add_subparsers(dest="subcommand")
    # Credential command
    credential_parser = subparsers.add_parser(
        "credential",
        help="Manage API keys and credentials for different hosting platforms",
        parents=[parent_parser],
    )
    credential_subparsers = credential_parser.add_subparsers(dest="credential_command")
    credential_update_parser = credential_subparsers.add_parser(
        "update", help="Update a credential", parents=[parent_parser]
    )

    credential_delete_parser = credential_subparsers.add_parser(
        "delete", help="Delete a credential", parents=[parent_parser]
    )

    credential_key_help = (
        "Credential key to manage (e.g., RADIENT_API_KEY,DEEPSEEK_API_KEY, OPENAI_API_KEY, "
        "ANTHROPIC_API_KEY, KIMI_API_KEY, ALIBABA_CLOUD_API_KEY, GOOGLE_AI_STUDIO_API_KEY, "
        "MISTRAL_API_KEY, OPENROUTER_API_KEY, XAI_API_KEY)"
    )

    credential_update_parser.add_argument("key", type=str, help=credential_key_help)
    credential_delete_parser.add_argument("key", type=str, help=credential_key_help)

    # Config command
    config_parser = subparsers.add_parser(
        "config", help="Manage configuration settings", parents=[parent_parser]
    )
    config_subparsers = config_parser.add_subparsers(dest="config_command")
    # Open command
    config_subparsers.add_parser(
        "open", help="Open the configuration file in the default editor", parents=[parent_parser]
    )
    # Edit command
    config_edit_parser = config_subparsers.add_parser(
        "edit",
        help="Edit a specific configuration value in the config file",
        parents=[parent_parser],
    )
    config_edit_parser.add_argument(
        "key",
        type=str,
        help="Configuration key to update (e.g., hosting, model_name, conversation_length, "
        "detail_length, max_learnings_history, auto_save_conversation)",
    )
    config_edit_parser.add_argument(
        "value",
        type=str,
        help="New value for the configuration key (type is automatically converted "
        "based on the key)",
    )

    # List command
    config_subparsers.add_parser(
        "list",
        help="List available configuration options and their descriptions",
        parents=[parent_parser],
    )

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
        "delete", help="Delete an agent (local by name or Radient by ID)", parents=[parent_parser]
    )
    delete_group = delete_parser.add_mutually_exclusive_group(required=True)
    delete_group.add_argument(
        "--name",
        type=str,
        help="Name of the agent to delete locally",
        dest="name",
    )
    delete_group.add_argument(
        "--id",
        type=str,
        help="ID of the agent to delete from Radient",
        dest="agent_id",
    )
    # Push command
    push_parser = agents_subparsers.add_parser(
        "push", help="Push (upload) an agent to Radient", parents=[parent_parser]
    )
    push_group = push_parser.add_mutually_exclusive_group(required=True)
    push_group.add_argument(
        "--name",
        type=str,
        help="Name of the agent to push to Radient",
    )
    push_group.add_argument(
        "--id",
        type=str,
        help="ID of the agent to push to Radient (explicit overwrite)",
    )
    # Pull command
    pull_parser = agents_subparsers.add_parser(
        "pull", help="Pull (download) an agent from Radient", parents=[parent_parser]
    )
    pull_parser.add_argument(
        "--id",
        type=str,
        required=True,
        help="ID of the agent to pull from Radient",
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
        default=1111,
        help="Port for the server (default: 1111)",
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


def credential_update_command(args: argparse.Namespace) -> int:
    credential_manager = CredentialManager(Path.home() / ".local-operator")
    credential_manager.prompt_for_credential(args.key, reason="update requested")
    return 0


def credential_delete_command(args: argparse.Namespace) -> int:
    credential_manager = CredentialManager(Path.home() / ".local-operator")
    credential_manager.set_credential(args.key, "")
    return 0


def config_create_command() -> int:
    """Create a new configuration file."""
    config_manager = ConfigManager(Path.home() / ".local-operator")
    config_manager._write_config(vars(config_manager.config))
    print("Created new configuration file at ~/.local-operator/config.yml")
    return 0


def config_open_command() -> int:
    """Open the configuration file using the default system editor."""
    config_path = Path.home() / ".local-operator" / "config.yml"
    if not config_path.exists():
        print(
            "\n\033[1;31mError: Configuration file does not exist.  Create one with "
            "`config create`.\033[0m"
        )
        return -1

    try:
        if platform.system() == "Windows":
            subprocess.run(["start", str(config_path)], shell=True, check=True)
        elif platform.system() == "Darwin":
            subprocess.run(["open", str(config_path)], check=True)
        else:
            subprocess.run(["xdg-open", str(config_path)], check=True)
        print(f"Opened configuration file at {config_path}")
        return 0
    except Exception as e:
        print(f"\n\033[1;31mError opening configuration file: {e}\033[0m")
        return -1


def config_edit_command(args: argparse.Namespace) -> int:
    """Edit a configuration value."""
    config_manager = ConfigManager(Path.home() / ".local-operator")
    try:
        # Parse the value to the appropriate type
        value = args.value
        # Try to convert to int
        try:
            if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
                value = int(value)
            # Try to convert to float
            elif value.replace(".", "", 1).isdigit() or (
                value.startswith("-") and value[1:].replace(".", "", 1).isdigit()
            ):
                value = float(value)
            # Try to convert to boolean
            elif value.lower() in ("true", "false"):
                value = value.lower() == "true"
            # Handle null/None values
            elif value.lower() in ("null", "none"):
                value = None
        except (ValueError, AttributeError):
            # Keep as string if conversion fails
            pass

        config_manager.update_config(
            {args.key: value},
            write=True,
        )

        print(f"Successfully updated {args.key} to {value}")
        return 0
    except KeyError:
        print(f"\n\033[1;31mError: Invalid configuration key: {args.key}\033[0m")
        return -1
    except Exception as e:
        print(f"\n\033[1;31mError updating configuration: {e}\033[0m")
        return -1


def config_list_command() -> int:
    """List available configuration options and their descriptions."""
    config_manager = ConfigManager(Path.home() / ".local-operator")
    config = config_manager.get_config()

    # Configuration descriptions
    descriptions = {
        "hosting": "AI provider platform (e.g., radient, openai, deepseek, anthropic, openrouter)",
        "model_name": "The specific model to use for interactions",
        "conversation_length": "Maximum number of messages to keep in conversation history",
        "detail_length": "Number of recent messages to leave unsummarized in conversation history",
        "max_learnings_history": "Maximum number of learning entries to retain",
        "auto_save_conversation": "Whether to automatically save conversations",
    }

    print("\n\033[1;32m╭─ Configuration Options ───────────────────────\033[0m")
    for key, value in config.values.items():
        description = descriptions.get(key, "No description available")
        print(f"\033[1;32m│ {key}: {value}\033[0m")
        print(f"\033[1;32m│   Description: {description}\033[0m")
    print("\033[1;32m╰──────────────────────────────────────────────\033[0m")
    return 0


def serve_command(host: str, port: int, reload: bool) -> int:
    """
    Start the FastAPI server using uvicorn.
    """
    print(f"Starting server at http://{host}:{port}")
    if reload:
        uvicorn.run(
            "local_operator.server.app:app",
            host=host,
            port=port,
            reload=reload,
            reload_excludes=[".venv"],
        )
    else:
        uvicorn.run("local_operator.server.app:app", host=host, port=port, reload=reload)
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
        AgentEditFields(
            name=name,
            security_prompt=None,
            hosting=None,
            model=None,
            description=None,
            last_message=None,
            temperature=None,
            tags=[],
            categories=[],
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
    print("\n\033[1;32m╭─ Created New Agent ───────────────────────────\033[0m")
    print(f"\033[1;32m│ Name: {agent.name}\033[0m")
    print(f"\033[1;32m│ ID: {agent.id}\033[0m")
    print(f"\033[1;32m│ Created: {agent.created_date}\033[0m")
    print(f"\033[1;32m│ Version: {agent.version}\033[0m")
    print("\033[1;32m╰──────────────────────────────────────────────────\033[0m\n")
    return 0


def agents_delete_command(
    args: argparse.Namespace, agent_registry: AgentRegistry, config_dir: Path
) -> int:
    """
    Delete an agent by name (local) or by ID (Radient).
    """
    if getattr(args, "name", None):
        name = args.name
        agents = agent_registry.list_agents()
        matching_agents = [a for a in agents if a.name == name]
        if not matching_agents:
            print(f"\n\033[1;31mError: No agent found with name: {name}\033[0m")
            return -1

        agent = matching_agents[0]
        agent_registry.delete_agent(agent.id)
        print(f"\n\033[1;32mSuccessfully deleted agent: {name}\033[0m")
        return 0
    elif getattr(args, "agent_id", None):
        # Delete from Radient by ID
        from local_operator.clients.radient import RadientClient
        from local_operator.config import ConfigManager
        from local_operator.credentials import CredentialManager

        credential_manager = CredentialManager(config_dir)
        api_key = credential_manager.get_credential("RADIENT_API_KEY")
        if not api_key:
            print("\n\033[1;31mError: RADIENT_API_KEY is required to delete from Radient\033[0m")
            return -1
        config_manager = ConfigManager(config_dir)
        base_url = config_manager.get_config_value("radient_base_url", "https://api.radienthq.com")
        radient_client = RadientClient(api_key=api_key, base_url=base_url)
        try:
            radient_client.delete_agent_from_marketplace(args.agent_id)
            print(
                f"\n\033[1;32mSuccessfully deleted agent with ID: {args.agent_id} "
                "from Radient\033[0m"
            )
            return 0
        except Exception as e:
            print(f"\n\033[1;31mError deleting agent from Radient: {e}\033[0m")
            return -1
    else:
        print("\n\033[1;31mError: Must provide --name or --id for delete\033[0m")
        return -1


def main() -> int:
    try:
        parser = build_cli_parser()
        args = parser.parse_args()

        # Set up the subprocess environment early
        setup_cross_platform_environment()

        os.environ["LOCAL_OPERATOR_DEBUG"] = "true" if args.debug else "false"

        # Load environment configuration
        env_config = get_env_config()

        config_dir = Path.home() / ".local-operator"
        agent_home_dir = Path.home() / "local-operator-home"

        # Create the agent home directory if it doesn't exist
        if not agent_home_dir.exists():
            agent_home_dir.mkdir(parents=True, exist_ok=True)

        if args.subcommand == "credential":
            if args.credential_command == "update":
                return credential_update_command(args)
            elif args.credential_command == "delete":
                return credential_delete_command(args)
            else:
                parser.error(f"Invalid credential command: {args.credential_command}")
        elif args.subcommand == "config":
            if args.config_command == "create":
                return config_create_command()
            elif args.config_command == "open":
                return config_open_command()
            elif args.config_command == "edit":
                return config_edit_command(args)
            elif args.config_command == "list":
                return config_list_command()
            else:
                parser.error(f"Invalid config command: {args.config_command}")
        elif args.subcommand == "agents":
            agent_registry = AgentRegistry(config_dir)
            if args.agents_command == "list":
                return agents_list_command(args, agent_registry)
            elif args.agents_command == "create":
                return agents_create_command(args.name, agent_registry)
            elif args.agents_command == "delete":
                return agents_delete_command(args, agent_registry, config_dir)
            elif args.agents_command == "push":
                # Push agent to Radient
                credential_manager = CredentialManager(config_dir)
                api_key = credential_manager.get_credential("RADIENT_API_KEY")
                if not api_key:
                    print(
                        "\n\033[1;31mError: RADIENT_API_KEY is required to push to Radient\033[0m"
                    )
                    return -1
                config_manager = ConfigManager(config_dir)
                base_url = config_manager.get_config_value(
                    "radient_base_url", "https://api.radienthq.com"
                )
                radient_client = RadientClient(api_key=api_key, base_url=base_url)
                # Support push by name or id
                agent = None
                agent_id_to_overwrite = None
                if getattr(args, "name", None):
                    agent = agent_registry.get_agent_by_name(args.name)
                    if not agent:
                        print(f"\n\033[1;31mError: No agent found with name: {args.name}\033[0m")
                        return -1
                elif getattr(args, "id", None):
                    try:
                        agent = agent_registry.get_agent(args.id)
                        agent_id_to_overwrite = args.id
                    except KeyError:
                        print(f"\n\033[1;31mError: No agent found with ID: {args.id}\033[0m")
                        return -1
                else:
                    print("\n\033[1;31mError: Must provide --name or --id for push\033[0m")
                    return -1
                zip_path, _ = agent_registry.export_agent(agent.id)
                try:
                    agent_id = agent_registry.upload_agent_to_radient(
                        radient_client, agent_id_to_overwrite, zip_path
                    )
                    if agent_id_to_overwrite:
                        print(
                            f"\n\033[1;32mSuccessfully pushed agent '{agent.name}' as "
                            f"overwrite to Radient (ID: {agent_id_to_overwrite})\033[0m"
                        )
                    else:
                        print(
                            f"\n\033[1;32mSuccessfully pushed agent '{agent.name}' to Radient. "
                            f"New agent ID: {agent_id}\033[0m"
                        )
                    return 0
                except Exception as e:
                    print(f"\n\033[1;31mError pushing agent to Radient: {e}\033[0m")
                    return -1
            elif args.agents_command == "pull":
                # Pull agent from Radient
                agent_id = args.id
                # Get Radient base URL from config or use default
                config_manager = ConfigManager(config_dir)
                base_url = config_manager.get_config_value(
                    "radient_base_url", "https://api.radientlabs.ai"
                )
                radient_client = RadientClient(api_key=None, base_url=base_url)
                try:
                    imported_agent = agent_registry.download_agent_from_radient(
                        radient_client, agent_id
                    )
                    print(
                        f"\n\033[1;32mSuccessfully pulled agent '{imported_agent.name}' "
                        f"(ID: {imported_agent.id}) from Radient\033[0m"
                    )
                    return 0
                except Exception as e:
                    print(f"\n\033[1;31mError pulling agent from Radient: {e}\033[0m")
                    return -1
            else:
                parser.error(f"Invalid agents command: {args.agents_command}")
        elif args.subcommand == "serve":
            # Use the provided host, port, and reload options for serving the API.
            return serve_command(args.host, args.port, args.reload)

        config_manager = ConfigManager(config_dir)
        credential_manager = CredentialManager(config_dir)
        agent_registry = AgentRegistry(config_dir)

        # Override config with CLI args where provided
        config_manager.update_config_from_args(args)

        # Set working directory if provided and valid
        if args.run_in:
            run_in_path = Path(args.run_in).resolve()
            if not run_in_path.is_dir():
                print(f"\n\033[1;31mError: Invalid working directory: {args.run_in}\033[0m")
                return -1
            os.chdir(run_in_path)
            print(f"\n\033[1;32mSetting working directory to: {run_in_path}\033[0m")

        # Get agent if name provided
        current_agent: Optional[AgentData] = None  # Use AgentData type hint
        if args.agent_name:
            current_agent = agent_registry.get_agent_by_name(args.agent_name)
            if not current_agent:
                print(
                    f"\n\033[1;33mNo agent found with name: {args.agent_name}. "
                    f"Creating new agent...\033[0m"
                )
                current_agent = agent_registry.create_agent(
                    AgentEditFields(
                        name=args.agent_name,
                        security_prompt=None,
                        hosting=None,
                        model=None,
                        description=None,
                        last_message=None,
                        temperature=None,
                        tags=[],
                        categories=[],
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
                # Add check to satisfy linter, though current_agent should be set here
                if current_agent:
                    print("\n\033[1;32m╭─ Created New Agent ───────────────────────────\033[0m")
                    print(f"\033[1;32m│ Name: {current_agent.name}\033[0m")
                    print(f"\033[1;32m│ ID: {current_agent.id}\033[0m")
                    print(f"\033[1;32m│ Created: {current_agent.created_date}\033[0m")
                    print(f"\033[1;32m│ Version: {current_agent.version}\033[0m")
                    print("\033[1;32m╰──────────────────────────────────────────────────\033[0m\n")
                else:
                    # This case should logically not happen
                    print("\n\033[1;31mError: Failed to create or retrieve agent.\033[0m")
                    return -1

        # Determine training mode and single execution mode
        training_mode = bool(args.train)
        single_execution_mode = args.subcommand == "exec"

        # Determine auto-save behavior
        auto_save_enabled = config_manager.get_config_value("auto_save_conversation", False)
        auto_save_active = auto_save_enabled and not single_execution_mode

        # Create autosave agent if needed
        if auto_save_active:
            agent_registry.create_autosave_agent()

        # Determine verbosity level
        verbosity = VerbosityLevel.DEBUG if args.debug else VerbosityLevel.VERBOSE

        # Initialize the operator using the bootstrap function
        scheduler_service: Optional[SchedulerService] = (
            None  # Initialize scheduler_service variable
        )
        try:
            # First, create the SchedulerService instance
            # Instantiate JobManager and WebSocketManager for CLI context
            job_manager = JobManager()
            websocket_manager = WebSocketManager()  # Will be unused but needed for constructor

            scheduler_service = SchedulerService(
                agent_registry=agent_registry,
                config_manager=config_manager,
                credential_manager=credential_manager,
                env_config=env_config,
                operator_type=OperatorType.CLI,
                verbosity_level=verbosity,
                job_manager=job_manager,  # Added
                websocket_manager=websocket_manager,  # Added
            )

            operator = initialize_operator(
                operator_type=OperatorType.CLI,
                config_manager=config_manager,
                credential_manager=credential_manager,
                agent_registry=agent_registry,
                env_config=env_config,
                scheduler_service=scheduler_service,  # Pass the created scheduler_service
                current_agent=current_agent,
                persist_conversation=training_mode,
                auto_save_conversation=auto_save_active,
                verbosity_level=verbosity,
            )
        except ValueError as e:
            print(f"\n\033[1;31mError initializing operator: {e}\033[0m")
            return -1

        # Create an async main function to handle operator and scheduler start
        async def async_main_cli():
            if scheduler_service:
                await scheduler_service.start()  # Start the scheduler

            # Start the async chat interface or execute single command
            if single_execution_mode:
                _, final_response = await operator.execute_single_command(args.command)
                if final_response:
                    print(final_response)
            else:
                await operator.chat()

            if scheduler_service:  # Shutdown scheduler gracefully
                await scheduler_service.shutdown()

        asyncio.run(async_main_cli())  # Run the new async main

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
