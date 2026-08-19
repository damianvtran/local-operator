"""
Main entry point for the Local Operator CLI application.

This script initializes the interactive agent experience or, when a
subcommand is given, dispatches to it: ``serve`` (FastAPI server), ``exec``
(one-shot headless task), ``credential``/``config``/``agents`` management,
``login``/``logout``/``login-status`` provider auth, ``search`` configuration,
and ``mcp`` server management.

Rewrite constraints (docs/REWRITE.md section E + backward-compat contracts):

- EVERY legacy flag/subcommand/dest/default/exit code survives byte-for-byte
  (``build_cli_parser`` is imported by server tests; ``main`` is the
  console-script entry). New flags are strictly additive.
- No module-level import of textual / providers / session internals / TUI:
  those are lazy-imported at the point of use so ``import local_operator.cli``
  stays cheap and cannot break while parallel rewrite streams are mid-flight.
- Exit codes preserved: 0 success, -1 legacy error banner; ``exec`` returns
  0/1 per the README contract.

Example Usage:
    local-operator --hosting deepseek --model deepseek-chat
    local-operator --hosting openai --model gpt-4o
    local-operator --hosting ollama --model llama2
    local-operator exec "write a hello world program" --hosting ollama --model llama2
    local-operator exec "long task" --background
    local-operator login anthropic
"""

from __future__ import annotations

import argparse
import copy
import functools
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from local_operator.config import ConfigManager
from local_operator.credentials import CredentialManager
from local_operator.env import get_env_config
from local_operator.logger import configure_cli_logging, file_logging
from local_operator.optional import missing_extra_error
from local_operator.paths import config_dir

# `local_operator.resume` is deliberately tiny (pathlib only): importing the
# session factory here for the same constant dragged the harness and asyncio
# onto the CLI startup path, which test_import_graph exists to prevent.
from local_operator.resume import RESUME_LATEST

if TYPE_CHECKING:
    from local_operator.agents import AgentRegistry

from local_operator.helpers import setup_cross_platform_environment

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

    Backward compatibility is a hard contract here: every legacy flag,
    subcommand, dest, and default must parse exactly as before. New flags and
    subcommands are additive only (docs/REWRITE.md section E).

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
        "--resume",
        nargs="?",
        const=RESUME_LATEST,
        default=None,
        metavar="SESSION_ID",
        dest="resume",
        help="Resume a previous session by id, replaying its transcript. The id is the one"
        " printed when a session is stopped with Ctrl+C twice, and is the directory name under"
        " ~/.local-operator/sessions. Pass --resume with no id to reopen the most recent"
        " session.",
    )
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
            "zai",
            "test",
        ],
        help="Hosting platform to use (radient, deepseek, openai, anthropic, ollama, kimi, "
        "alibaba, google, mistral, test, openrouter, xai, zai)",
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
    # --- Additive root flags (rewrite) ------------------------------------
    parser.add_argument(
        "--yolo",
        action="store_true",
        help="Auto-approve all tool executions (read/write/exec tiers) without prompting",
    )
    parser.add_argument(
        "--no-tui",
        action="store_true",
        dest="no_tui",
        help="Disable the full-screen TUI; use the plain headless REPL instead",
    )
    parser.add_argument(
        "--tui",
        action="store_true",
        dest="tui",
        help="Force the full-screen TUI even when stdout is not a tty "
        "(errors clearly if the TUI cannot run without a tty)",
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
        "open",
        help="Open the configuration file in the default editor",
        parents=[parent_parser],
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
        "delete",
        help="Delete an agent (local by name or Radient by ID)",
        parents=[parent_parser],
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
    # --- Additive exec flags (rewrite) ------------------------------------
    exec_parser.add_argument(
        "--background",
        action="store_true",
        help="Detach the task: spawn a background worker with a log file and exit immediately",
    )
    exec_parser.add_argument(
        "--json",
        action="store_true",
        dest="json_mode",
        help="Emit one JSON line per agent event instead of the final text",
    )
    exec_parser.add_argument(
        "--agent-id",
        type=str,
        dest="agent_id",
        help="ID of the agent to use for this execution (alternative to --agent by name)",
    )

    # --- Additive auth subcommands (rewrite) -------------------------------
    login_parser = subparsers.add_parser(
        "login",
        help="Log in to a provider (OAuth or API key)",
        parents=[parent_parser],
    )
    login_parser.add_argument(
        "provider",
        type=str,
        nargs="?",
        default=None,
        help="Provider to log in to (e.g., openai, anthropic, kimi, xai). "
        "Omit to list login-capable providers.",
    )
    logout_parser = subparsers.add_parser(
        "logout",
        help="Log out of a provider (removes stored credentials)",
        parents=[parent_parser],
    )
    logout_parser.add_argument(
        "provider",
        type=str,
        help="Provider to log out of",
    )
    subparsers.add_parser(
        "login-status",
        help="List stored provider credentials and their status",
        parents=[parent_parser],
    )
    # Alias matching the docs/REWRITE.md section B spelling (`status`).
    subparsers.add_parser(
        "status",
        help="Alias for login-status: list stored provider credentials",
        parents=[parent_parser],
    )

    # --- Additive MCP subcommands (rewrite) --------------------------------
    mcp_parser = subparsers.add_parser("mcp", help="Manage MCP servers", parents=[parent_parser])
    mcp_subparsers = mcp_parser.add_subparsers(dest="mcp_command")
    mcp_subparsers.add_parser(
        "list",
        help="List configured MCP servers (all sources merged)",
        parents=[parent_parser],
    )
    mcp_add_parser = mcp_subparsers.add_parser(
        "add",
        help="Add an MCP server (stdio command or http/sse URL)",
        parents=[parent_parser],
    )
    mcp_add_parser.add_argument("name", type=str, help="Server name")
    mcp_add_parser.add_argument(
        "--command", type=str, default=None, help="Stdio command to launch the server"
    )
    mcp_add_parser.add_argument(
        "--arg",
        action="append",
        default=None,
        dest="server_args",
        help="Stdio command argument (repeatable)",
    )
    mcp_add_parser.add_argument(
        "--env",
        action="append",
        default=None,
        dest="server_env",
        help="Environment variable KEY=VALUE for the stdio server (repeatable)",
    )
    mcp_add_parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="HTTP/SSE server URL (alternative to --command)",
    )
    mcp_add_parser.add_argument(
        "--scope",
        type=str,
        choices=["global", "project"],
        default="global",
        help="Config scope to write (default: global ~/.local-operator/mcp.json)",
    )
    mcp_add_parser.add_argument(
        "--oauth",
        action="store_true",
        help="Enable OAuth for a remote HTTP server",
    )
    mcp_remove_parser = mcp_subparsers.add_parser(
        "remove",
        help="Remove an MCP server from a config scope",
        parents=[parent_parser],
    )
    mcp_remove_parser.add_argument("name", type=str, help="Server name to remove")
    mcp_remove_parser.add_argument(
        "--scope",
        type=str,
        choices=["global", "project"],
        default="global",
        help="Config scope to remove from (default: global)",
    )
    mcp_login_parser = mcp_subparsers.add_parser(
        "login",
        help="Authenticate one OAuth-enabled MCP server",
        parents=[parent_parser],
    )
    mcp_login_parser.add_argument("name", type=str, help="Server name to authenticate")

    # Built separately so provider transports stay off the CLI import path.
    from local_operator.web_search.cli import add_search_subparser

    add_search_subparser(subparsers, parent_parser)

    # CL-04: ``--yolo`` is accepted on every subcommand too (additive). The
    # root flag keeps its default; subparsers get a SUPPRESS copy so parsing
    # inside a subcommand NEVER clobbers a root-level ``--yolo`` (the
    # argparse re-default quirk that already applies to the legacy parent
    # flags must not swallow this one — ``--yolo exec "task"`` is documented).
    _propagate_global_flags(parser)

    return parser


def _propagate_global_flags(parser: argparse.ArgumentParser) -> None:
    """Re-declare the position-independent global options on every subparser.

    Not routed through ``parent_parser``: a shared parent action with a SUPPRESS
    default still re-applies under argparse's subparser namespace reset, and
    resolve-style conflicts mutate the shared action. A fresh action per
    subparser is deterministic: each accepts the option locally and never resets
    a value set BEFORE the subcommand.

    `--yolo` needed this from the start. `--resume` needs it for a sharper
    reason: routed only through the parent, `local-operator --resume ID exec "…"`
    parsed the id and then had it clobbered back to ``None`` by the subparser,
    so exec started a FRESH session — verbatim the failure the field exists to
    prevent, and invisible because `--help` advertises the option as global.
    Validation could not catch it either, since validation reads the value after
    the clobber: `--resume bogus config list` exited 0 in silence while
    `config list --resume bogus` exited 1 with the recovery listing.
    """
    for action in parser._actions:
        if not isinstance(action, argparse._SubParsersAction):
            continue
        seen: set[int] = set()
        for subparser in action.choices.values():
            if id(subparser) in seen:
                continue
            seen.add(id(subparser))
            subparser.add_argument(
                "--yolo",
                action="store_true",
                default=argparse.SUPPRESS,
                help="Auto-approve all tool executions (read/write/exec tiers)"
                " without prompting",
            )
            subparser.add_argument(
                "--resume",
                nargs="?",
                const=RESUME_LATEST,
                default=argparse.SUPPRESS,
                metavar="SESSION_ID",
                dest="resume",
                help="Resume a previous session by id. Pass with no id for the most recent.",
            )
            _propagate_global_flags(subparser)


def credential_update_command(args: argparse.Namespace) -> int:
    credential_manager = CredentialManager(config_dir())
    credential_manager.prompt_for_credential(args.key, reason="update requested")
    return 0


def credential_delete_command(args: argparse.Namespace) -> int:
    credential_manager = CredentialManager(config_dir())
    credential_manager.set_credential(args.key, "")
    return 0


def config_create_command() -> int:
    """Create a new configuration file."""
    base_dir = config_dir()
    config_manager = ConfigManager(base_dir)
    config_manager._write_config(vars(config_manager.config))
    # Print the path that was actually written, not a hardcoded
    # ~/.local-operator: config_dir() honours LOCAL_OPERATOR_CONFIG_DIR, and
    # config_open_command below already reports the resolved path — naming a
    # different file here makes the two commands contradict each other.
    print(f"Created new configuration file at {base_dir / 'config.yml'}")
    return 0


def config_open_command() -> int:
    """Open the configuration file using the default system editor."""
    config_path = config_dir() / "config.yml"
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
    config_manager = ConfigManager(config_dir())
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
    config_manager = ConfigManager(config_dir())
    config = config_manager.get_config()

    # Configuration descriptions. conversation_length / detail_length /
    # max_learnings_history are DEPRECATED (CL-16): the compaction engine
    # supersedes them — retention is governed by `values.compaction.*` now;
    # the keys stay readable but are inert.
    descriptions = {
        "hosting": "AI provider platform (e.g., radient, openai, deepseek, anthropic, openrouter)",
        "model_name": "The specific model to use for interactions",
        "conversation_length": "[DEPRECATED — superseded by compaction] "
        "Maximum number of messages to keep in conversation history",
        "detail_length": "[DEPRECATED — superseded by compaction] "
        "Number of recent messages to leave unsummarized in conversation history",
        "max_learnings_history": "[DEPRECATED — superseded by compaction] "
        "Maximum number of learning entries to retain",
        "auto_save_conversation": "Whether to automatically save conversations",
        "compaction": "Compaction engine settings (enabled, strategy, thresholds); "
        "replaces conversation_length/detail_length",
        "tui": "TUI settings (theme)",
        "session_retention_max_sessions": "Maximum ephemeral session directories to keep "
        "under sessions/ (0 disables the ceiling)",
        "session_retention_max_bytes": "Maximum total bytes across sessions/ before the "
        "oldest directories are evicted (0 disables the ceiling)",
        "session_retention_max_age_days": "Age after which an ephemeral session directory "
        "is evicted (0 disables the ceiling)",
    }

    print("\n\033[1;32m╭─ Configuration Options ───────────────────────\033[0m")
    for key, value in config.values.items():
        description = descriptions.get(key, "No description available")
        print(f"\033[1;32m│ {key}: {value}\033[0m")
        print(f"\033[1;32m│   Description: {description}\033[0m")
    print("\033[1;32m╰──────────────────────────────────────────────\033[0m")
    return 0


def serve_command(host: str, port: int, reload: bool) -> int:
    """Start the FastAPI server using uvicorn.

    ``uvicorn`` is imported HERE, not at module scope: the HTTP facade lives
    behind the ``server`` extra, so a default install (and every non-server
    entry point) must be able to ``import local_operator.cli`` without
    fastapi/uvicorn/starlette and their dependency chain present.
    """
    try:
        import uvicorn
    except ImportError:
        print(
            f"\n\033[1;31m{missing_extra_error('server', 'The HTTP API server')}\033[0m",
            file=sys.stderr,
        )
        return -1

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


def agents_list_command(args: argparse.Namespace, agent_registry: "AgentRegistry") -> int:
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
        print(f"\033[1;32m{left_bar}   • Hosting: {agent.hosting or 'default'}\033[0m")
        print(f"\033[1;32m{left_bar}   • Model: {agent.model or 'default'}\033[0m")
        if agent.description:
            print(f"\033[1;32m{left_bar}   • Description: {agent.description}\033[0m")
        if agent.tags:
            print(f"\033[1;32m{left_bar}   • Tags: {', '.join(agent.tags)}\033[0m")
        if agent.categories:
            print(f"\033[1;32m{left_bar}   • Categories: {', '.join(agent.categories)}\033[0m")
        if not is_last:
            print("\033[1;32m│ │\033[0m")

    # Print pagination info
    print("\033[1;32m│\033[0m")
    print(f"\033[1;32m│ Page {page} of {total_pages} (Total agents: {total_agents})\033[0m")
    if page < total_pages:
        print(f"\033[1;32m│ Use --page {page + 1} to see next page\033[0m")
    print("\033[1;32m╰──────────────────────────────────────────────\033[0m")
    return 0


def agents_create_command(name: str, agent_registry: "AgentRegistry") -> int:
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

    from local_operator.agents import AgentEditFields  # lazy: heavy module

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
    args: argparse.Namespace, agent_registry: "AgentRegistry", config_dir: Path
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


# --- Additive subcommand handlers (rewrite) --------------------------------


def _build_auth_stack(config_dir: Path) -> tuple[Any, Any]:
    """(auth_store, credential_manager) for the login handlers.

    Lazy import of the providers stream's AuthStore — the CLI module top
    level must never depend on it.
    """
    from local_operator.providers.auth_store import AuthStore

    credential_manager = CredentialManager(config_dir)
    auth_store = AuthStore(credential_manager=credential_manager)
    return auth_store, credential_manager


def login_command(args: argparse.Namespace) -> int:
    """Run the OAuth/API-key login flow for one provider."""
    try:
        from local_operator.providers.auth_cli import run_login
    except ImportError:
        print("\n\033[1;31mError: provider login support is not available in this build\033[0m")
        return -1
    auth_store, credential_manager = _build_auth_stack(config_dir())
    try:
        return run_login(getattr(args, "provider", None), credential_manager, auth_store)
    finally:
        auth_store.close()


def logout_command(args: argparse.Namespace) -> int:
    """Remove all stored credentials for one provider."""
    try:
        from local_operator.providers.auth_cli import run_logout
    except ImportError:
        print("\n\033[1;31mError: provider login support is not available in this build\033[0m")
        return -1
    auth_store, _credential_manager = _build_auth_stack(config_dir())
    try:
        return run_logout(args.provider, auth_store)
    finally:
        auth_store.close()


def login_status_command() -> int:
    """List stored provider credentials and their status."""
    try:
        from local_operator.providers.auth_cli import list_logins
    except ImportError:
        print("\n\033[1;31mError: provider login support is not available in this build\033[0m")
        return -1
    auth_store, credential_manager = _build_auth_stack(config_dir())
    try:
        return list_logins(auth_store, credential_manager)
    finally:
        auth_store.close()


_MCP_INTERACTIVE_LOGIN_TIMEOUT_MS = 10 * 60_000


async def _mcp_login_server(name: str, cwd: Path) -> int:
    """Run one interactive MCP OAuth exchange and persist its token.

    The SDK's callback handler prints the authorization URL and accepts the
    final loopback redirect URL on stdin. ``McpTokenStorage`` writes the
    resulting token and client registration to ``auth.db``; a successful login
    therefore survives this short-lived manager and future Local Operator
    sessions reuse it without another browser round-trip.
    """
    from local_operator.mcp.config import load_all_mcp_configs
    from local_operator.mcp.manager import McpManager

    configs, _sources = load_all_mcp_configs(cwd)
    cfg = configs.get(name)
    if cfg is None:
        print(f"error: MCP server {name!r} is not configured", file=sys.stderr)
        return 1
    auth = getattr(cfg, "auth", None)
    if auth is None or auth.type != "oauth":
        print(
            f"error: MCP server {name!r} is not OAuth-enabled; " "add a remote server with --oauth",
            file=sys.stderr,
        )
        return 1

    manager = McpManager(cwd)
    try:
        conn = await manager.connect_configured_server(
            name, timeout_ms=_MCP_INTERACTIVE_LOGIN_TIMEOUT_MS
        )
        print(f"Authenticated MCP server {name!r}; discovered {len(conn.tools)} tools.")
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI turns protocol failures into exit status
        print(f"error: MCP login failed for {name!r}: {exc}", file=sys.stderr)
        return 1
    finally:
        await manager.disconnect_all()


def mcp_command(args: argparse.Namespace) -> int:
    """Dispatch ``mcp list|add|login|remove`` to MCP configuration and auth.

    Lazy import: the MCP package keeps its SDK imports lazy too, and this
    CLI must survive builds where it has not landed yet.
    """
    try:
        from local_operator.mcp import config as mcp_config
    except ImportError:
        print("\n\033[1;31mError: MCP support is not available in this build\033[0m")
        return -1

    if args.mcp_command == "list":
        servers = mcp_config.list_effective_servers(Path.cwd())
        if not servers:
            print("No MCP servers configured.")
            return 0
        print("\n\033[1;32m╭─ MCP Servers ─────────────────────────────────\033[0m")
        for name, server in sorted(servers.items()):
            target = server.get("command") or server.get("url") or "(unconfigured)"
            print(f"\033[1;32m│ {name}: {target}\033[0m")
        print("\033[1;32m╰──────────────────────────────────────────────\033[0m")
        return 0
    if args.mcp_command == "add":
        env: dict[str, str] = {}
        for item in getattr(args, "server_env", None) or []:
            if "=" not in item:
                print(f"\n\033[1;31mError: --env expects KEY=VALUE, got: {item}\033[0m")
                return -1
            key, value = item.split("=", 1)
            env[key] = value
        return mcp_config.add_server(
            args.name,
            command=getattr(args, "command", None),
            args=getattr(args, "server_args", None),
            env=env or None,
            url=getattr(args, "url", None),
            oauth=bool(getattr(args, "oauth", False)),
            scope=getattr(args, "scope", "global"),
        )
    if args.mcp_command == "login":
        import asyncio

        return asyncio.run(_mcp_login_server(args.name, Path.cwd()))
    if args.mcp_command == "remove":
        return mcp_config.remove_server(args.name, scope=getattr(args, "scope", "global"))

    print(f"\n\033[1;31mError: Invalid mcp command: {args.mcp_command}\033[0m")
    return -1


# --- Session factory facade -------------------------------------------------


async def create_session(
    args: argparse.Namespace,
    config_manager: ConfigManager,
    credential_manager: CredentialManager,
    agent_registry: "AgentRegistry",
    *,
    has_ui: bool = False,
):
    """Build a wired harness session for interactive/headless use.

    Thin facade over :func:`local_operator.session_factory.create_session`
    (the composition root shared with ``exec`` and the background worker).
    The engine import is lazy so importing ``cli`` never pulls in
    providers/session internals.
    """
    from local_operator.session_factory import create_session as _create_session

    return await _create_session(
        args, config_manager, credential_manager, agent_registry, has_ui=has_ui
    )


# --- Shared helpers ----------------------------------------------------------


def _apply_run_in(run_in: Optional[str]) -> Optional[int]:
    """Validate and chdir into ``--run-in`` (legacy prints preserved).

    Returns -1 when the directory is invalid, None on success/no-op.
    """
    if not run_in:
        return None
    run_in_path = Path(run_in).resolve()
    if not run_in_path.is_dir():
        print(
            f"\n\033[1;31mError: Invalid working directory: {run_in}\033[0m",
            file=sys.stderr,
        )
        return -1
    os.chdir(run_in_path)
    # These are OPERATOR notices, not data: they must go to stderr so they
    # never interleave into the `exec --json` event stream on stdout.
    print(
        f"\n\033[1;32mSetting working directory to: {run_in_path}\033[0m",
        file=sys.stderr,
    )
    return None


async def _run_headless_repl(
    args: argparse.Namespace,
    config_manager: ConfigManager,
    credential_manager: CredentialManager,
    agent_registry: "AgentRegistry",
) -> int:
    """Plain-stream REPL for non-tty stdout or ``--no-tui``.

    Mirrors the TUI loop semantics in miniature: one session for the whole
    REPL, assistant text streamed to stdout as it arrives, tool rows dim on
    stderr, Ctrl+C aborts the running turn (not the REPL), Ctrl+D/EOF exits.
    """
    import asyncio

    from rich.console import Console

    from local_operator.headless_print import PrintRenderer

    console = Console(stderr=True, highlight=False)
    session = await create_session(
        args, config_manager, credential_manager, agent_registry, has_ui=False
    )
    renderer = PrintRenderer(stream_text=True)
    unsubscribe = renderer.attach(session)
    console.print(
        "[bold cyan]Local Operator[/bold cyan] "
        "[dim](headless REPL — Ctrl-C interrupts a turn, Ctrl-D exits)[/dim]"
    )
    try:
        while True:
            try:
                # asyncio.to_thread (CL-14): blocking input() must not freeze
                # the event loop (wake deliveries, session bookkeeping).
                line = await asyncio.to_thread(input, "> ")
            except (EOFError, KeyboardInterrupt):
                console.print()
                break
            if not line.strip():
                continue
            renderer.failed = False
            try:
                await session.prompt(line)
            except KeyboardInterrupt:
                # Abort the turn, keep the REPL alive (TUI parity).
                session.abort("interrupted")
            except Exception as exc:  # noqa: BLE001 — keep the REPL alive
                console.print(f"[red]Error: {exc}[/red]")
    finally:
        if callable(unsubscribe):
            unsubscribe()
        await session.dispose()
    return 0


def _preflight_hosting_model(
    config_manager: ConfigManager,
    credential_manager: CredentialManager,
    agent_registry: "AgentRegistry",
    current_agent: Optional[Any],
    args: argparse.Namespace,
    *,
    require_api_key: bool = True,
) -> int | None:
    """Startup preflight (CL-06): resolve hosting/model and verify that a
    credential source exists BEFORE any turn runs.

    Resolution uses the composition root's precedence (agent > flag >
    config). Stored credentials satisfy preflight by presence; refreshing an
    OAuth token belongs to the stream-time failover path, where a transient
    refresh failure can be reported accurately instead of being misreported
    here as a missing API key. Providers that need no key (ollama, test,
    custom) pass through, and anything the provider registry cannot answer
    passes through too — a preflight must never block a configuration the
    engine itself would accept.

    ``require_api_key=False`` demotes a missing API key from a fatal error to
    a stderr warning, and exists for the interactive front ends (TUI and
    headless REPL): the in-app ``/login`` command is the product's own remedy
    for a missing key, and a fatal preflight sat exactly between the user and
    that remedy — a fresh config whose default hosting was a keyed provider
    could not start at all. Session construction never needs the key (stream
    time resolves it through the AuthStore cascade), the TUI splash already
    shows "not logged in — /login <provider>", and a keyless turn fails with
    its own accurate message, so letting the app start loses nothing. Hosting/
    model resolution errors stay fatal on every path: without a hosting there
    is no session to build and nothing for ``/login`` to fix.

    Returns -1 (already printed) on failure, None to continue. All engine
    imports stay lazy so this never weights down parser-only paths.
    """
    try:
        from local_operator.session_factory import resolve_hosting_model

        hosting, _model_name = resolve_hosting_model(current_agent, args, config_manager)
    except ValueError as exc:
        # stderr on principle, not because this path is currently reachable from
        # `exec --json`: it is an ERROR message, and its sibling
        # `_preflight_api_key` two functions down already writes there. Keeping
        # the two consistent is what stops the next person wiring this into the
        # exec route from reintroducing a stdout leak.
        print(f"\n\033[1;31mError: {exc}\033[0m", file=sys.stderr)
        return -1
    except Exception:  # noqa: BLE001 — unknown providers pass through
        return None

    return _preflight_api_key(hosting, credential_manager, require_key=require_api_key)


def _preflight_api_key(
    hosting: str, credential_manager: CredentialManager, *, require_key: bool = True
) -> int | None:
    """Verify that the provider has a credential source.

    Stored OAuth and API-key rows satisfy preflight by presence, including a
    row under temporary stream-time backoff. The stream owns refresh and
    failover; doing network refresh here can turn a transient OAuth failure
    into a false "API key is required" startup error that prevents access to
    the TUI's login command. With no stored row, the AuthStore cascade still
    checks environment and legacy ``credentials.env`` keys.

    Providers that need no key (ollama, test) and anything the provider
    registry cannot answer pass through — a preflight must never block a
    configuration the engine itself would accept.

    Returns -1 (already printed) on failure, None to continue. With
    ``require_key=False`` a missing key is a warning instead of a failure —
    see :func:`_preflight_hosting_model` for why the interactive front ends
    must not be blocked from starting (the fix lives behind the gate).
    """
    canonical = "test" if hosting == "noop" else hosting
    try:
        from local_operator.providers.registry import get_provider_definition

        definition = get_provider_definition(canonical)
    except Exception:  # noqa: BLE001
        return None
    if definition is None or definition.env_keys is None:
        # Keyless provider (ollama/test) or unregistered hosting: the
        # engine decides; preflight must not be a second gatekeeper.
        return None

    try:
        import asyncio

        from local_operator.providers.auth_store import AuthStore

        auth_store = AuthStore(credential_manager=credential_manager)
        try:
            storage_provider = definition.store_credentials_as or canonical
            if auth_store.list_credentials(provider=storage_provider):
                return None
            api_key = asyncio.run(auth_store.get_api_key(canonical))
        finally:
            auth_store.close()
    except Exception:  # noqa: BLE001 — resolution failures pass through
        return None

    if api_key:
        return None

    key_name = definition.env_keys if isinstance(definition.env_keys, str) else "API key"
    if not require_key:
        # Interactive start: name the fact and the in-app remedy, then let the
        # app come up. The TUI repaints over this line, but its splash carries
        # the same warning; the headless REPL keeps it visible on stderr.
        print(
            f"\n\033[1;33mWarning: no credentials are configured for hosting "
            f"platform '{hosting}'. Starting anyway — run `/login {canonical}` "
            f"in the app, `local-operator login {canonical}`, or set "
            f"{key_name} in the environment.\033[0m",
            file=sys.stderr,
        )
        return None
    # stderr: this fires on every fresh install and every typo'd --hosting,
    # i.e. it is the single most common `exec --json` failure, and a coloured
    # non-JSON line on stdout breaks the consumer it is trying to inform.
    print(
        f"\n\033[1;31mError: {key_name} is required for hosting platform "
        f"'{hosting}' but is not configured. Set it via `local-operator "
        f"credential update {key_name}`, the environment, or `local-operator "
        f"login {canonical}`.\033[0m",
        file=sys.stderr,
    )
    return -1


#: Third-party modules the `server` extra provides. Used to decide whether a
#: ModuleNotFoundError from the scheduler wiring really means "install the
#: extra" — reporting an internal import failure that way sends the user to
#: install something that will not help, and buries the actual defect.
_SERVER_EXTRA_MODULES = frozenset(
    {
        "apscheduler",
        "fastapi",
        "starlette",
        "uvicorn",
        "websockets",
        "multipart",
        "dill",
        "tiktoken",
    }
)


async def _run_with_scheduler(run_fn, *run_args) -> int:
    """Run the interactive front end with the SchedulerService alive (CL-07).

    The legacy main() constructed ``SchedulerService`` (JobManager +
    WebSocketManager, the same minimal managers the server app uses), started
    it before the chat loop and shut it down afterwards — scheduled tasks
    created during a session only fire while the service runs. Dropping it in
    the rewrite would silently lose scheduled-task support, so the TUI and
    headless REPL both run inside this wrapper. Every construction failure
    (apscheduler missing, server-only managers unavailable) degrades to
    running WITHOUT a scheduler — the front end itself must never be blocked
    by scheduling support.
    """
    scheduler_service = None
    try:
        from local_operator.jobs import JobManager  # lazy: server-shared module
        from local_operator.scheduler_service import SchedulerService
        from local_operator.server.utils.websocket_manager import WebSocketManager
        from local_operator.types import OperatorType

        base_dir = config_dir()
        config_manager = ConfigManager(base_dir)
        credential_manager = CredentialManager(base_dir)
        from local_operator.agents import AgentRegistry  # lazy: heavy module

        agent_registry = AgentRegistry(base_dir)

        from local_operator.console import VerbosityLevel

        scheduler_service = SchedulerService(
            agent_registry=agent_registry,
            config_manager=config_manager,
            credential_manager=credential_manager,
            env_config=get_env_config(),
            operator_type=OperatorType.CLI,
            verbosity_level=(
                VerbosityLevel.DEBUG
                if os.environ.get("LOCAL_OPERATOR_DEBUG", "false") == "true"
                else VerbosityLevel.VERBOSE
            ),
            job_manager=JobManager(),
            websocket_manager=WebSocketManager(),  # required by the constructor, unused in CLI
        )
    except ModuleNotFoundError as exc:
        # ONLY claim the extra when the missing module actually belongs to it.
        # Catching every ModuleNotFoundError from this block reported a broken
        # internal import (there are six in here) as a missing `server` extra:
        # the user installs the extra, nothing changes, and the real defect
        # stays invisible — strictly less diagnostic than the raw
        # "No module named 'x'" this replaced.
        root = (exc.name or "").split(".")[0]
        if root in _SERVER_EXTRA_MODULES:
            # Fires on every startup of a bare install, because this wraps both
            # front ends — `local-operator` with no arguments is the
            # most-travelled path in the product.
            print(
                f"\033[1;33mWarning: {missing_extra_error('server', 'Scheduled tasks')} "
                f"Continuing without scheduled tasks.\033[0m",
                file=sys.stderr,
            )
        else:
            print(
                f"\033[1;33mWarning: scheduler unavailable, continuing without "
                f"scheduled tasks: {exc}\033[0m",
                file=sys.stderr,
            )
        scheduler_service = None
    except Exception as exc:  # noqa: BLE001 — degrade to no scheduler
        print(
            f"\033[1;33mWarning: scheduler unavailable, continuing without "
            f"scheduled tasks: {exc}\033[0m",
            file=sys.stderr,
        )
        scheduler_service = None

    if scheduler_service is not None:
        try:
            await scheduler_service.start()
        except Exception as exc:  # noqa: BLE001 — never block the front end
            print(
                f"\033[1;33mWarning: failed to start scheduler: {exc}\033[0m",
                file=sys.stderr,
            )
            scheduler_service = None
    try:
        return await run_fn(*run_args)
    finally:
        if scheduler_service is not None:
            try:
                await scheduler_service.shutdown()
            except Exception:  # noqa: BLE001 — shutdown must not mask the exit code
                pass


def main() -> int:
    # FIRST, before anything else can log. `helpers.py` used to configure the
    # root logger as an import side effect; now the entry point owns it, which
    # is what lets the TUI branch below swap the console handler for a file.
    configure_cli_logging()
    try:
        parser = build_cli_parser()
        args = parser.parse_args()

        # Set up the subprocess environment early
        setup_cross_platform_environment()

        # Resolve `--resume` HERE, before anything is started. Left to the
        # session factory it surfaces inside the TUI as "session failed to
        # start" — a full-screen app launched, painted, and torn down to report a
        # typo — and the generic handler below would render it as a traceback
        # panel and still exit 0. A bad session id is ordinary user error, so it
        # gets a one-line message, the ids that DO exist, and a non-zero status.
        if getattr(args, "resume", None) is not None:
            from local_operator.resume import (
                ResumeNotFound,
                backfill_session_origins,
                format_age,
                recent_sessions,
                resolve_resume_id,
            )

            # Classify pre-existing sessions BEFORE resolving, not after. The
            # session factory also backfills, but it runs when a session is
            # BUILT — and this branch answers `--resume` first, so on the first
            # launch after an upgrade a bare `--resume` resolved `@latest`
            # against an unclassified store and reopened whichever delegated
            # run happened to finish last. Idempotent and stdlib-only, so it
            # costs a directory scan on the one path that cannot afford to be
            # wrong about which sessions are the user's.
            backfill_session_origins(config_dir())

            try:
                args.resume = resolve_resume_id(config_dir(), str(args.resume))
            except ResumeNotFound as error:
                print(f"\033[31m{error}\033[0m", file=sys.stderr)
                # With the age: a column of bare 12-hex ids gives the reader
                # nothing to choose between, and the recency the listing already
                # sorted by is the one fact that makes them recognisable.
                available = recent_sessions(config_dir())
                if available:
                    now = time.time()
                    print("recent sessions (newest first):", file=sys.stderr)
                    for session_id, mtime in available:
                        print(
                            f"  {session_id}   {format_age(now - mtime)}",
                            file=sys.stderr,
                        )
                return 1

        os.environ["LOCAL_OPERATOR_DEBUG"] = "true" if args.debug else "false"
        # (CL-12) No env_config binding here: the scheduler wrapper resolves its
        # own env config and the session factory does the same lazily — a
        # dead local would only invite drift.
        base_dir = config_dir()
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
        elif args.subcommand == "search":
            from local_operator.web_search.cli import search_command

            return search_command(args)
        elif args.subcommand == "agents":
            from local_operator.agents import AgentRegistry  # lazy: heavy module

            agent_registry = AgentRegistry(base_dir)
            if args.agents_command == "list":
                return agents_list_command(args, agent_registry)
            elif args.agents_command == "create":
                return agents_create_command(args.name, agent_registry)
            elif args.agents_command == "delete":
                return agents_delete_command(args, agent_registry, base_dir)
            elif args.agents_command == "push":
                # Push agent to Radient
                from local_operator.clients.radient import RadientClient  # lazy

                credential_manager = CredentialManager(base_dir)
                api_key = credential_manager.get_credential("RADIENT_API_KEY")
                if not api_key:
                    print(
                        "\n\033[1;31mError: RADIENT_API_KEY is required to push to Radient\033[0m"
                    )
                    return -1
                config_manager = ConfigManager(base_dir)
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
                from local_operator.clients.radient import RadientClient  # lazy

                agent_id = args.id
                # Get Radient base URL from config or use default
                config_manager = ConfigManager(base_dir)
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
        elif args.subcommand == "login":
            return login_command(args)
        elif args.subcommand == "logout":
            return logout_command(args)
        elif args.subcommand in ("login-status", "status"):
            return login_status_command()
        elif args.subcommand == "mcp":
            invalid = _apply_run_in(args.run_in)
            if invalid is not None:
                return invalid
            return mcp_command(args)
        elif args.subcommand == "exec":
            # Single-execution mode: headless one-shot (README contract —
            # exit 0 on success, non-zero on error). Working-directory
            # handling matches the legacy pre-run behavior.
            invalid = _apply_run_in(args.run_in)
            if invalid is not None:
                return invalid
            from local_operator.exec_mode import ExecArgs, run_exec

            exec_args = ExecArgs(
                background=args.background,
                json_mode=args.json_mode,
                agent_name=args.agent_name,
                agent_id=getattr(args, "agent_id", None),
                yolo=args.yolo,
                hosting=args.hosting,
                model=args.model,
                train=args.train,
                resume=getattr(args, "resume", None),
            )
            # Startup preflight (CL-06) for the FOREGROUND path: hosting/
            # model (agent > flag > config) + API-key resolution fail fast
            # with the legacy message shape instead of dying mid-turn.
            # ``--background`` preflight lives in exec_mode._spawn_background
            # (CL-09) and shares the same resolution path.
            if not args.background:
                from local_operator.exec_mode import resolve_hosting_model_dry

                try:
                    hosting, _model = resolve_hosting_model_dry(exec_args)
                except ValueError as exc:
                    # stderr: this is the FOREGROUND `exec --json` path, so
                    # stdout is the event stream. The byte-identical twins in
                    # exec_mode._spawn_background were fixed earlier and these
                    # were missed — the flag combination that reaches them
                    # (`exec --json` with a bad or absent hosting/model) is the
                    # most likely one to be scripted.
                    print(f"\n\033[1;31mError: {exc}\033[0m", file=sys.stderr)
                    return -1
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"\n\033[1;31mError: preflight failed: {exc}\033[0m",
                        file=sys.stderr,
                    )
                    return -1
                key_result = _preflight_api_key(hosting, CredentialManager(base_dir))
                if key_result is not None:
                    return key_result
            return run_exec(args.command, exec_args)

        config_manager = ConfigManager(base_dir)
        credential_manager = CredentialManager(base_dir)

        # Override config with CLI args where provided
        config_manager.update_config_from_args(args)

        # Set working directory if provided and valid
        invalid = _apply_run_in(args.run_in)
        if invalid is not None:
            return invalid

        from local_operator.agents import (  # lazy
            AgentData,
            AgentEditFields,
            AgentRegistry,
        )

        agent_registry = AgentRegistry(base_dir)

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

        # Legacy behavior: the auto-save config value persists interactive
        # sessions via the registry's autosave agent (exec is excluded —
        # single-execution mode never autosaved).
        auto_save_enabled = config_manager.get_config_value("auto_save_conversation", False)
        if auto_save_enabled:
            args.train = True

        # Startup preflight (CL-06): hosting/model resolution fails fast with
        # the legacy message shape BEFORE any turn (the factory raises the
        # same errors mid-construction; surfacing them here keeps the user
        # from seeing a half-initialized session). A missing API key is only a
        # WARNING here: this is the interactive path, `/login` inside the app
        # is the remedy, and a fatal gate locked the user out of it (the exec
        # path keeps its fatal check — a scripted run has no login prompt).
        preflight_result = _preflight_hosting_model(
            config_manager,
            credential_manager,
            agent_registry,
            current_agent,
            args,
            require_api_key=False,
        )
        if preflight_result is not None:
            return preflight_result

        # Interactive path: full-screen TUI when stdout is a tty and not
        # disabled; plain headless REPL otherwise. ``--tui`` (CL-13) forces
        # the TUI even when stdout is not a tty — with a clear error when
        # that is impossible.
        force_tui = bool(getattr(args, "tui", False))
        use_tui = force_tui or (not getattr(args, "no_tui", False) and sys.stdout.isatty())
        run_tui = None
        if use_tui:
            try:
                from local_operator.tui import run_tui  # lazy: textual
            except ImportError:
                run_tui = None
                if force_tui:
                    # Forced but impossible: surface WHY, don't silently fall
                    # back to the plain REPL (the user asked for the TUI).
                    print(
                        "\n\033[1;31mError: the TUI is not available in this "
                        "build/install (missing 'local_operator.tui'); remove "
                        "--tui to use the plain REPL.\033[0m"
                    )
                    return -1
                use_tui = False

        # asyncio is imported HERE, not at module scope. It is the heaviest
        # single item on the CLI's import graph (34.4 ms, +6.5 MB RSS, +77
        # modules measured by scripts/bench_base_overhead.py) and only the
        # interactive TUI/REPL tail below needs it — `--version`, `--help`,
        # shell completion and the config/credential/agents/login subcommands
        # all return before this point, and `exec`/`serve` bring their own
        # event loop from exec_mode/the server module.
        import asyncio

        if use_tui and run_tui is not None:
            tui_config = config_manager.get_config_value("tui", None)
            theme_name = tui_config.get("theme", "dark") if isinstance(tui_config, dict) else "dark"

            async def session_factory():
                return await create_session(
                    args,
                    config_manager,
                    credential_manager,
                    agent_registry,
                    has_ui=True,
                )

            # The provider controller gives the TUI the full provider/model/
            # credential/usage surface behind /model /provider /login /usage.
            # Its owning AuthStore lives for the TUI session only; the CLI
            # closes it after run_tui returns.
            from local_operator.providers.auth_store import AuthStore
            from local_operator.providers.controller import ProviderController

            tui_auth_store = AuthStore(credential_manager=credential_manager)
            tui_controller = ProviderController(tui_auth_store, credential_manager)
            try:
                # BIND BY KEYWORD. ``_run_with_scheduler`` forwards *args
                # positionally, so a positional controller lands in whatever
                # parameter happens to sit in that slot (it once landed in
                # ``login_handler``) and leaves provider_controller None,
                # disabling every provider slash command while the app still
                # starts cleanly. functools.partial pins it by name so a future
                # signature change cannot re-introduce that silent failure.
                tui_entry = functools.partial(run_tui, provider_controller=tui_controller)

                # ``/resume <id>`` in the TUI needs a factory that boots an
                # ARBITRARY session, not just the one the launch args named.
                # Building it here closes over the same managers the boot
                # factory used and swaps ``args.resume`` to the requested id,
                # so a mid-session resume is exactly a relaunch onto that
                # transcript — no second shell call, no new process. A shallow
                # copy of the args namespace keeps the user's interactive
                # ``args`` object untouched (``--resume`` is read once, at
                # startup; mutating the original here would confuse the exit
                # hint's "resume with:" line).
                async def resume_factory(resume_id: str | None):
                    resume_args = copy.copy(args)
                    # ``None`` is meaningful, not absent: ``create_session``
                    # branches on ``resume is not None``, so passing it through
                    # verbatim is what makes ``/new`` a genuine cold-launch
                    # session rather than a special case beside one.
                    resume_args.resume = resume_id
                    return await create_session(
                        resume_args,
                        config_manager,
                        credential_manager,
                        agent_registry,
                        has_ui=True,
                    )

                tui_entry = functools.partial(tui_entry, resume_factory=resume_factory)
                # The silence starts HERE, not inside ``run_tui``. The
                # scheduler is started by the wrapper below and logs
                # "Scheduler started" at INFO before the app has painted a
                # single cell — observed on the alternate screen at launch.
                # ``run_tui`` opens the same window itself (the guarantee
                # belongs to the TUI, not to one of its callers); the context
                # manager is re-entrant, so the inner block is a no-op.
                # ``_run_with_scheduler`` is shared with the headless REPL,
                # which must keep its console output, so the wrapping goes on
                # this call site rather than inside it.
                with file_logging():
                    return asyncio.run(
                        _run_with_scheduler(
                            tui_entry,
                            session_factory,
                            theme_name,
                        )
                    )
            finally:
                try:
                    tui_auth_store.close()
                except Exception:  # noqa: BLE001 — closing on teardown, never fatal
                    pass

        return asyncio.run(
            _run_with_scheduler(
                _run_headless_repl,
                args,
                config_manager,
                credential_manager,
                agent_registry,
            )
        )
    except Exception as e:
        # STDERR, always. main() wraps the `exec` dispatch too, so this is the
        # error presenter for `exec --json` — printing decorated banners to
        # stdout put four unparseable lines on the event stream at exactly the
        # moment a consumer most needs to read it.
        print(f"\n\033[1;31mError: {str(e)}\033[0m", file=sys.stderr)
        print(
            "\033[1;34m╭─ Stack Trace ────────────────────────────────────\033[0m",
            file=sys.stderr,
        )
        traceback.print_exc()
        print(
            "\033[1;34m╰──────────────────────────────────────────────────\033[0m",
            file=sys.stderr,
        )
        print(
            "\n\033[1;33mPlease review and correct the error to continue.\033[0m",
            file=sys.stderr,
        )
        return -1


if __name__ == "__main__":
    exit(main())
