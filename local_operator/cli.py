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
- Exit codes: 0 success, 1 on error; ``exec`` returns 0/1 per the README
  contract. (Failure paths previously returned -1, which a shell reports as
  255 — colliding with the xargs/ssh "command not found" sentinel and
  contradicting the exec 0/non-zero contract. Item A13 changed them to 1; a
  quiet cancel returns 130, the SIGINT convention.)

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

from local_operator.agent_profiles import SEED_ORIGIN_PREFIX
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
        help="Model to use (e.g., gpt-4o, claude-3-5-sonnet-latest, deepseek-chat, "
        "grok-3, glm-5.3, gemini-2.0-flash-001, qwen-plus, moonshot-v1-32k, "
        "mistral-large-latest, deepseek/deepseek-chat). Optional: when omitted, "
        "the provider's default model is used.",
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

    # Teams command
    teams_parser = subparsers.add_parser("teams", help="Manage teams", parents=[parent_parser])
    teams_subparsers = teams_parser.add_subparsers(dest="teams_command")
    teams_subparsers.add_parser("list", help="List all teams", parents=[parent_parser])
    teams_create = teams_subparsers.add_parser(
        "create", help="Create a new team", parents=[parent_parser]
    )
    teams_create.add_argument("name", type=str, help="Name of the team to create")
    teams_create.add_argument(
        "--manager",
        type=str,
        default="manager",
        help="Role or specialist who orchestrates (default: manager)",
    )
    teams_create.add_argument(
        "--member",
        action="append",
        default=[],
        dest="members",
        help="Roster slot as role or role:count (repeatable)",
    )
    teams_create.add_argument("--description", type=str, default="", help="One-line description")
    teams_show = teams_subparsers.add_parser(
        "show", help="Show a team's roster and briefs", parents=[parent_parser]
    )
    teams_show.add_argument("name", type=str, help="Name of the team to show")
    teams_delete = teams_subparsers.add_parser(
        "delete", help="Delete a team by name", parents=[parent_parser]
    )
    teams_delete.add_argument(
        "--name",
        type=str,
        required=True,
        help="Name of the team to delete",
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

    # Mobile command: the phone-facing control plane (daemon + supervision).
    # Same lazy-import rule as ``serve`` — the mobile modules pull starlette/
    # uvicorn only when a mobile command actually runs.
    mobile_parser = subparsers.add_parser(
        "mobile", help="Phone access: control this machine's lop sessions", parents=[parent_parser]
    )
    mobile_subparsers = mobile_parser.add_subparsers(dest="mobile_command")
    mobile_subparsers.add_parser("install", help="Install the daemon, password and LaunchAgent")
    mobile_subparsers.add_parser("status", help="Daemon health, gate and live sessions")
    for action in ("start", "stop", "restart"):
        mobile_subparsers.add_parser(action, help=f"{action.capitalize()} the daemon")
    logs_parser = mobile_subparsers.add_parser("logs", help="Tail the daemon log")
    logs_parser.add_argument("--lines", type=int, default=100)
    logs_parser.add_argument("--follow", "-f", action="store_true")
    mobile_subparsers.add_parser("password", help="Show or rotate the portal password")
    uninstall_parser = mobile_subparsers.add_parser("uninstall", help="Remove the LaunchAgent")
    uninstall_parser.add_argument("--purge", action="store_true", help="Also delete the password")
    serve_mobile_parser = mobile_subparsers.add_parser("serve", help="Run the daemon (foreground)")
    serve_mobile_parser.add_argument("--port", type=int, default=4098)

    # Browser bridge command: lazy for the same reason as mobile. Ordinary CLI
    # startup must not pull Starlette/uvicorn in just to render --help.
    browser_parser = subparsers.add_parser(
        "browser", help="Connect the browser tool to a Chromium extension", parents=[parent_parser]
    )
    browser_subparsers = browser_parser.add_subparsers(dest="browser_command")
    install_browser = browser_subparsers.add_parser("install", help="Install the bridge daemon")
    install_browser.add_argument("--port", type=int, default=4099)
    browser_subparsers.add_parser("status", help="Show daemon, extension and pairing status")
    for action in ("start", "stop", "restart"):
        browser_subparsers.add_parser(action, help=f"{action.capitalize()} the daemon")
    pair_browser = browser_subparsers.add_parser("pair", help="Show the extension pairing code")
    pair_browser.add_argument(
        "--reset", action="store_true", help="Revoke the paired browser first"
    )
    logs_browser = browser_subparsers.add_parser("logs", help="Tail the daemon log")
    logs_browser.add_argument("--lines", type=int, default=100)
    logs_browser.add_argument("--follow", "-f", action="store_true")
    uninstall_browser = browser_subparsers.add_parser("uninstall", help="Remove the bridge daemon")
    uninstall_browser.add_argument("--purge", action="store_true", help="Also delete pairing state")
    serve_browser = browser_subparsers.add_parser(
        "serve", help="Run the bridge daemon (foreground)"
    )
    serve_browser.add_argument("--port", type=int, default=4099)

    # Peer-to-peer session messaging: hand a message to another local lop
    # session without cmux, over the same control-socket + registry substrate
    # the mobile stack already uses (loopback + 0600 record => same-account
    # trust boundary). See guides/peer-messaging.
    send_parser = subparsers.add_parser(
        "send",
        help="Send a message to another local lop session (no cmux needed)",
        parents=[parent_parser],
    )
    send_parser.add_argument(
        "target",
        nargs="?",
        help="conversation-name / session-id / cwd substring (case-insensitive)",
    )
    send_parser.add_argument(
        "message",
        nargs="?",
        help="message text; omit to read the body from stdin",
    )
    send_parser.add_argument("--pid", type=int, help="target by exact pid")
    send_parser.add_argument("--session", dest="session", help="target by exact session id")
    send_parser.add_argument(
        "--now",
        "--steer",
        dest="steer",
        action="store_true",
        help="inject mid-turn (steer) instead of the default mailbox",
    )
    send_parser.add_argument(
        "--wake",
        action="store_true",
        help="if the target is idle, drive a turn now (mailbox mode only)",
    )

    sessions_parser = subparsers.add_parser(
        "sessions",
        help="List active lop sessions and their resource usage",
        parents=[parent_parser],
    )
    sessions_parser.add_argument("--json", action="store_true", help="machine-readable output")

    # Exec command for single execution mode
    # PyPI upgrade. Not ``lop-update`` (hyphen), which archives local git
    # ``main`` into the uv-tool env — opposite audience, never invoked here.
    update_parser = subparsers.add_parser(
        "update",
        help=(
            "Upgrade this install from PyPI. Not the lop-update script, "
            "which rebuilds the global runtime from a local git checkout."
        ),
        parents=[parent_parser],
    )
    update_parser.add_argument(
        "--check",
        action="store_true",
        help="Print installed vs PyPI; do not install",
    )

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
    mcp_logout_parser = mcp_subparsers.add_parser(
        "logout",
        help="Remove the stored OAuth credential for one MCP server",
        parents=[parent_parser],
    )
    mcp_logout_parser.add_argument("name", type=str, help="Server name to log out")
    mcp_reauth_parser = mcp_subparsers.add_parser(
        "reauth",
        help="Log out of one OAuth MCP server and run a fresh authorization",
        parents=[parent_parser],
    )
    mcp_reauth_parser.add_argument("name", type=str, help="Server name to re-authenticate")

    # Built separately so provider transports stay off the CLI import path.
    from local_operator.web_fetch.cli import add_fetch_subparser
    from local_operator.web_search.cli import add_search_subparser

    add_search_subparser(subparsers, parent_parser)
    add_fetch_subparser(subparsers, parent_parser)

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
    """Prompt for and store one credential. Exit 0/1/130.

    The prompt used to let three ordinary interruptions escape as tracebacks:
    Ctrl-C raised a bare ``KeyboardInterrupt`` all the way out, and an empty
    value / closed stdin raised a ``ValueError`` whose message carried nested
    ANSI escapes that the generic red-banner handler in ``main`` then wrapped in
    a stack-trace panel. None of the three is a program fault \u2014 they are the
    user cancelling or mis-entering \u2014 so each gets one plain line and a clean
    exit code: 130 for a cancel (the shell convention for SIGINT), 1 otherwise.
    """
    from local_operator.ansi import strip_control_sequences
    from local_operator.cli_style import ERROR, WARNING, paint
    from local_operator.providers.registry import PROVIDER_REGISTRY, env_key_name

    # Warn when the key is not one the registry knows, with the closest match \u2014
    # a typo'd ``OPENAI_API_KY`` otherwise stores silently and the provider
    # never sees it. Arbitrary keys stay allowed (custom providers are
    # legitimate); this is advice, not a gate.
    known_keys = {name for p in PROVIDER_REGISTRY if (name := env_key_name(p.id))}
    if args.key not in known_keys:
        import difflib

        close = difflib.get_close_matches(args.key, sorted(known_keys), n=1)
        hint = f" Did you mean {close[0]}?" if close else ""
        print(
            paint(
                f"Warning: '{args.key}' is not a known provider key.{hint} " "Storing it anyway.",
                WARNING,
                stream=sys.stderr,
            ),
            file=sys.stderr,
        )

    credential_manager = CredentialManager(config_dir())
    try:
        credential_manager.prompt_for_credential(args.key, reason="update requested")
    except KeyboardInterrupt:
        # 130 is the shell's SIGINT convention; the message is one quiet line,
        # not the red stack-trace panel the generic handler would have drawn.
        print("\nCancelled.", file=sys.stderr)
        return 130
    except (ValueError, EOFError) as exc:
        # Empty input or a closed stdin. Strip any control sequences from the
        # message before printing \u2014 the presenter owns the colour, and a nested
        # escape from deeper in the stack would otherwise repaint the line.
        print(paint(strip_control_sequences(str(exc)), ERROR, stream=sys.stderr), file=sys.stderr)
        return 1
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
    from local_operator.cli_style import ERROR, paint

    config_path = config_dir() / "config.yml"
    if not config_path.exists():
        print(
            paint(
                "Error: Configuration file does not exist.  Create one with `config create`.",
                ERROR,
                stream=sys.stderr,
            ),
            file=sys.stderr,
        )
        return 1

    # Try the platform GUI opener first, then fall back to $VISUAL/$EDITOR. The
    # GUI openers do not exist on a headless/SSH Linux box (there is no
    # xdg-open without a desktop session), and there `config open` used to fail
    # outright — yet that is exactly the environment where a terminal editor is
    # the ONLY way in. Only spawn an interactive editor when stdout is a tty:
    # an editor launched from a pipe or a non-interactive shell has no terminal
    # to draw in and would hang or error.
    gui_error: Exception | None = None
    try:
        if platform.system() == "Windows":
            subprocess.run(["start", str(config_path)], shell=True, check=True)
        elif platform.system() == "Darwin":
            subprocess.run(["open", str(config_path)], check=True)
        else:
            subprocess.run(["xdg-open", str(config_path)], check=True)
        print(f"Opened configuration file at {config_path}")
        return 0
    except Exception as e:  # noqa: BLE001 — GUI opener absent or failed
        gui_error = e

    editor = os.environ.get("VISUAL") or os.environ.get("EDITOR")
    if editor and sys.stdout.isatty():
        try:
            # ``shlex.split`` so a value like ``code --wait`` or ``emacs -nw``
            # is honoured, not treated as one impossible executable name.
            import shlex

            subprocess.run([*shlex.split(editor), str(config_path)], check=True)
            print(f"Opened configuration file at {config_path}")
            return 0
        except Exception as e:  # noqa: BLE001 — editor missing or exited non-zero
            gui_error = e

    print(
        paint(f"Error opening configuration file: {gui_error}", ERROR, stream=sys.stderr),
        file=sys.stderr,
    )
    print(
        f"Set $VISUAL or $EDITOR, or edit the file directly at {config_path}.",
        file=sys.stderr,
    )
    return 1


def config_edit_command(args: argparse.Namespace) -> int:
    """Edit a configuration value."""
    from local_operator.cli_style import ERROR, paint

    config_manager = ConfigManager(config_dir())

    # Validate the key against the SCHEMA before writing. The old
    # ``except KeyError`` was dead code \u2014 ``update_config`` calls
    # ``Config.set_value`` which is a plain ``dict.__setitem__`` and never
    # raises for an unknown key \u2014 so a typo like ``config edit hostng radient``
    # printed "Successfully updated hostng" and wrote a junk key the app never
    # reads. difflib names the closest real key so the fix is one glance away.
    #
    # The key set is ``settings_io``'s and no longer ``DEFAULT_CONFIG.values``,
    # which held only TOP-LEVEL keys. Every dotted key was rejected outright,
    # including ``display.terminal_title`` — which the TUI itself instructs the
    # user to run this exact command for. The app told them to type a command
    # that could only exit 1.
    from local_operator import settings_io

    setting = settings_io.resolve_key(args.key)
    if setting is None:
        import difflib

        close = difflib.get_close_matches(args.key, settings_io.valid_keys(), n=1)
        hint = f" Did you mean '{close[0]}'?" if close else ""
        print(
            paint(
                f"Error: unknown configuration key: '{args.key}'.{hint}", ERROR, stream=sys.stderr
            ),
            file=sys.stderr,
        )
        print(
            "Run `local-operator config list` to see the available keys.",
            file=sys.stderr,
        )
        return 1

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

        # Through the facade rather than ``update_config``: a dotted key needs
        # the merge-into-existing-sub-mapping rule (a whole-mapping write drops
        # the siblings ``_load_config`` never back-fills), and the flat
        # ``display.*`` keys need their dot treated as literal rather than as a
        # level of nesting. ``write_setting`` validates as well, so an
        # out-of-range number is now refused here instead of being stored and
        # silently clamped by whichever consumer reads it.
        settings_io.write_setting(config_manager, setting, value)

        print(f"Successfully updated {args.key} to {value}")
        return 0
    except settings_io.ConfigUnreadableError as e:
        # Distinct from the schema rejection below: the key and the value are
        # both fine, the FILE is broken, and telling the user to check their
        # value would send them to fix something that is not wrong. Say what is
        # unparseable and that nothing was written, because the alternative to
        # refusing is overwriting their config with defaults (round 2, B3).
        print(paint(f"Error: {e}", ERROR, stream=sys.stderr), file=sys.stderr)
        print(
            "Nothing was written. Fix the file by hand, or move it aside and run "
            "`local-operator config create`.",
            file=sys.stderr,
        )
        return 1
    except ValueError as e:
        # A schema rejection is a typo, not a crash: state the rule that was
        # broken rather than wrapping it in "error updating configuration".
        print(paint(f"Error: {args.key}: {e}", ERROR, stream=sys.stderr), file=sys.stderr)
        return 1
    except Exception as e:
        # 1, not -1: a shell sees -1 as 255, which collides with the
        # xargs/ssh "command not found" sentinel and contradicts the
        # documented 0/non-zero exec contract (item A13).
        print(
            paint(f"Error updating configuration: {e}", ERROR, stream=sys.stderr), file=sys.stderr
        )
        return 1


def config_list_command() -> int:
    """List available configuration options and their descriptions.

    Lists the SCHEMA rather than whatever happens to be stored. The old loop
    walked ``config.values``, so a nested key was shown as a raw dict blob
    (``retry: {'enabled': True, ...}``) and any key the user had never set was
    absent entirely — which made this the wrong answer to "what can I set?",
    the question `config edit`'s own error message sends people here to ask.
    Each row now names a key `config edit` accepts, one per line.
    """
    from local_operator import settings_io

    config_manager = ConfigManager(config_dir())
    config = config_manager.get_config()

    # Legacy descriptions kept for the two keys the schema does not carry
    # (`compaction` and `tui` as whole mappings, listed for users following an
    # older doc). Everything else is described by the schema, so the page and
    # this table cannot describe one key two ways.
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
        "session_retention_max_sessions": "[RETIRED] Session transcripts are never deleted "
        "automatically; this ceiling no longer does anything at any value",
        "session_retention_max_bytes": "[RETIRED] Session transcripts are never deleted "
        "automatically; this ceiling no longer does anything at any value",
        "session_retention_max_age_days": "[RETIRED] Session transcripts are never deleted "
        "automatically; this ceiling no longer does anything at any value",
    }

    print("\n\033[1;32m╭─ Configuration Options ───────────────────────\033[0m")
    for setting in settings_io.SETTINGS:
        # The EFFECTIVE value (stored, else the shipped default), which is what
        # the user is asking about — an unset key showing blank would read as
        # "off" for every boolean here.
        value = settings_io.read_setting(config_manager, setting)
        description = descriptions.get(setting.key) or setting.help
        print(f"\033[1;32m│ {setting.key}: {value}\033[0m")
        print(f"\033[1;32m│   Description: {description}\033[0m")
    # Anything a hand-edited config carries that the schema does not know about
    # is still listed, marked, rather than hidden: a key the app does not read
    # is exactly what a user needs to be told about, and silently omitting it
    # is how the old `hostng` typo survived unnoticed.
    unknown = sorted(set(config.values) - {setting.path[0] for setting in settings_io.SETTINGS})
    for key in unknown:
        print(f"\033[1;32m│ {key}: {config.values[key]}\033[0m")
        print("\033[1;32m│   Description: not a recognised key; nothing reads it\033[0m")
    print("\033[1;32m╰──────────────────────────────────────────────\033[0m")
    return 0


def browser_command(args: argparse.Namespace) -> int:
    """Dispatch ``lop browser …`` without importing the daemon at CLI startup."""
    command = getattr(args, "browser_command", None)
    if command == "serve":
        from local_operator.browser_bridge.daemon import main as serve_main

        return serve_main(["--port", str(args.port)])

    from local_operator.browser_bridge import install as browser_install
    from local_operator.browser_bridge.daemon import pairing_status, reset_pairing

    if command == "install":
        result = browser_install.install(args.port)
        steps = result.get("steps", [])
        assert isinstance(steps, list)
        for step in steps:
            print(f"  {step}")
        if not result.get("ok"):
            print(f"\n\033[1;31m{result.get('error', 'install failed')}\033[0m")
            return 1
        print("\nbrowser bridge installed and healthy.")
        print("  load the Local Operator extension, then run `lop browser pair`.")
        return 0
    if command == "status":
        result = browser_install.status()
        health = result.get("health") or {}
        assert isinstance(health, dict)
        print(f"installed:           {'yes' if result['installed'] else 'no'}")
        print(f"daemon healthy:      {'yes' if result['healthy'] else 'no'}")
        connected = bool(health.get("extension_connected"))
        print(f"extension connected: {'yes' if connected else 'no'}")
        print(f"paired:              {'yes' if result['paired'] else 'no'}")
        # A paired-but-not-connected browser is the normal closed/backgrounded
        # state, not a fault; say so rather than leaving a user to guess (N2).
        if result["paired"] and not connected:
            print(
                "                     (browser not currently attached; it reconnects when opened)"
            )
        driven = health.get("current_url")
        if connected and driven:
            print(f"driving:             {driven}")
        print(f"port:                {result['port']}")
        print(f"log:                 {result['log']}")
        return 0 if result["healthy"] else 1
    if command == "pair":
        if args.reset:
            # File unlink here; the running daemon's revocation watcher (and
            # the per-request pairing re-check) sever any LIVE socket within a
            # few seconds, so a revoked browser loses drive authority now, not
            # only at its next reconnect (findings A5/U1).
            reset_pairing()
            print(
                "revoked the paired browser; any live connection is dropped within a few seconds."
            )
            # A successful revoke must not report failure to a wrapping script
            # even when no extension is currently waiting to pair (UX-N1).
            pair = pairing_status()
            code = pair.get("pending_code")
            if code:
                print(f"pairing code: {code}")
                print("enter this 6-digit code in the Local Operator extension popup.")
            else:
                print("open the extension popup to pair a browser again.")
            return 0
        pair = pairing_status()
        code = pair.get("pending_code")
        if code:
            print(f"pairing code: {code}")
            print("enter this 6-digit code in the Local Operator extension popup.")
            return 0
        if pair.get("paired"):
            print("browser extension is already paired. Use --reset to pair another profile.")
            return 0
        print("no extension is waiting to pair. Open the extension popup, then retry.")
        return 1
    if command in ("start", "stop", "restart"):
        result = browser_install.service_action(command)
        if not result["ok"]:
            print(f"\033[1;31m{result['error']}\033[0m")
            return 1
        print(f"browser bridge {command} ok")
        return 0
    if command == "logs":
        import subprocess

        command_line = ["tail", "-n", str(args.lines)]
        if args.follow:
            command_line.append("-f")
        command_line.append(str(browser_install.log_path()))
        return subprocess.call(command_line)
    if command == "uninstall":
        result = browser_install.uninstall(purge=args.purge)
        steps = result.get("steps", [])
        assert isinstance(steps, list)
        for step in steps:
            print(f"  {step}")
        return 0 if result.get("ok") else 1
    print("usage: lop browser {install|status|start|stop|restart|pair|logs|uninstall|serve}")
    return 1


def _peer_red(message: str) -> None:
    """Print one red error line, matching the rest of the CLI's error style."""
    print(f"\n\033[1;31m{message}\033[0m", file=sys.stderr)


def _format_bytes(value: "int | None") -> str:
    """Human-readable memory size, or an em dash when the probe returned None.

    ``lop sessions`` shows one column per number; an unknown value must read as
    'we could not measure this' (—), never as zero."""
    if value is None:
        return "—"
    size = float(value)
    for unit in ("B", "K", "M", "G", "T"):
        if size < 1024 or unit == "T":
            if unit == "B":
                return f"{int(size)}{unit}"
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}T"


def _peer_sender_identity() -> "dict[str, Any]":
    """Best-effort identity of the calling session for the peer indicator.

    ``lop send`` is a short-lived child of the ``lop`` TUI that spawned it, so
    the parent pid is the sending session's pid — that is the pid the shared
    core looks up. (The in-session ``send`` tool passes ``os.getpid()`` instead;
    see ``mobile/peer_send.py`` for why the two differ.)"""
    from local_operator.mobile.peer_send import peer_sender_identity

    return peer_sender_identity(os.getppid())


def _resolve_peer_target(
    args: argparse.Namespace,
) -> "tuple[Any | None, list[Any], str]":
    """Resolve a ``lop send`` target to one live SessionRecord.

    Thin adapter over the shared send-side core
    (``mobile.peer_send.resolve_peer_target``): the CLI's argparse namespace is
    mapped onto the core's keyword arguments. The resolution rules themselves —
    pid, then session id, then case-insensitive substring; only ``live`` records;
    candidates returned on ambiguity — live in the core so the in-session
    ``send`` tool resolves targets identically."""
    from local_operator.mobile.peer_send import resolve_peer_target

    # The flag grammar is passed in so the CLI's user-visible error keeps saying
    # `--pid` / `--session`, exactly as it did before the extraction.
    return resolve_peer_target(
        target=args.target,
        pid=args.pid,
        session=args.session,
        pid_hint="--pid",
        session_hint="--session",
    )


def send_command(args: argparse.Namespace) -> int:
    """``lop send`` — hand a message to another local lop session.

    Delivery mode maps from the flags: ``--now``/``--steer`` => steer (inject
    mid-turn), otherwise mailbox; ``--wake`` drives a turn if the mailbox
    target is idle. The body comes from the positional argument or, when
    omitted, stdin (the ergonomic path for piping a longer note)."""
    import asyncio

    from local_operator.mobile.peer_client import send_peer_message
    from local_operator.mobile.peer_send import candidate_lines, validate_peer_body

    record, candidates, error = _resolve_peer_target(args)
    if candidates:
        print(f"{len(candidates)} sessions match; disambiguate with --pid:", file=sys.stderr)
        for line in candidate_lines(candidates, indent="  ", prefix="--pid"):
            print(line, file=sys.stderr)
        return 1
    if error or record is None:
        _peer_red(error or "no target resolved")
        return 1

    # Self-send guard (U2): a target resolving to the SENDING session means the
    # session is messaging itself, which would paint a "peer message from <own
    # name>" card as though a DIFFERENT session sent it (and, in --wake/--now
    # mode, self-trigger a turn). Refuse rather than deliver a mislabeled
    # self-note; the composer is the way to talk to yourself.
    #
    # The comparison uses the pid the IDENTITY walk resolved, not a bare
    # os.getppid(). `lop send` is only sometimes a direct child of the TUI: run
    # from an agent's bash tool or through a shell wrapper it is a grandchild,
    # and then the two disagree — the guard compared the intermediate shell's
    # pid, missed, and delivered a self-message that the ancestry-resolved
    # identity then labelled confidently with the session's OWN name. Resolving
    # once and using it for both is what keeps them from drifting apart again.
    sender = _peer_sender_identity()
    sender_pid = sender.get("pid")
    if record.pid == sender_pid:
        _peer_red("that target is this session; use the composer to message yourself")
        return 1

    # Body: positional wins; otherwise read stdin (piped note).

    if args.message is not None:
        text = args.message
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        _peer_red("no message given (pass it as an argument or pipe it on stdin)")
        return 1
    body_error = validate_peer_body(text)
    if body_error:
        _peer_red(body_error)
        return 1

    mode = "steer" if args.steer else "mailbox"
    try:
        detail = asyncio.run(
            send_peer_message(
                record,
                text=text,
                mode=mode,
                wake=bool(args.wake),
                sender=sender,
            )
        )
    except (RuntimeError, ConnectionError, OSError, ValueError) as exc:
        # ValueError covers a read fault the frame reader could still surface
        # (e.g. an oversized non-welcome line): it must become the same soft,
        # non-zero "could not deliver" line, never an uncaught traceback (U1).
        _peer_red(f"could not deliver: {exc}")
        return 1
    name = record.conversation_name or record.session_id
    print(f"→ {name} (pid {record.pid}): {detail}")
    return 0


def sessions_command(args: argparse.Namespace) -> int:
    """``lop sessions`` — list active sessions and their resource usage.

    RSS is the always-present baseline; FOOTPRINT is the true memory number
    where the platform can report it (macOS phys footprint / Linux Pss) and —
    otherwise. HEARTBEAT_AGE surfaces wedged-ness numerically so counts can be
    eyeballed against reality."""
    import json as _json

    from local_operator.mobile.resources import session_resource_usage
    from local_operator.session.runtime import registry

    scanned = registry.scan(config_dir())
    now = time.time()
    live_pids = [rec.pid for rec, state in scanned if state == "live"]
    usage = session_resource_usage(live_pids)

    rows = []
    for rec, state in scanned:
        use = usage.get(rec.pid)
        rows.append(
            {
                "state": state,
                "pid": rec.pid,
                "kind": rec.kind,
                "conversation_name": rec.conversation_name,
                "session_id": rec.session_id,
                "model_label": rec.model_label,
                "cwd": rec.cwd,
                "rss_bytes": use.rss_bytes if use else None,
                "footprint_bytes": use.footprint_bytes if use else None,
                "uptime_s": max(0.0, now - rec.started_at),
                "heartbeat_age_s": max(0.0, now - rec.heartbeat_at),
            }
        )

    if args.json:
        print(_json.dumps(rows, indent=2))
        return 0

    if not rows:
        print("no active lop sessions")
        return 0

    header = (
        f"{'STATE':<7} {'PID':>7} {'KIND':<7} {'CONVERSATION':<24} "
        f"{'MODEL':<24} {'RSS':>8} {'FOOTPRINT':>9} {'UPTIME':>8} {'HB_AGE':>7}"
    )
    print(header)
    for row in rows:
        name = (row["conversation_name"] or row["session_id"] or "")[:24]
        model = (row["model_label"] or "")[:24]
        print(
            f"{row['state']:<7} {row['pid']:>7} {row['kind']:<7} {name:<24} "
            f"{model:<24} {_format_bytes(row['rss_bytes']):>8} "
            f"{_format_bytes(row['footprint_bytes']):>9} "
            f"{_format_duration(row['uptime_s']):>8} "
            f"{_format_duration(row['heartbeat_age_s']):>7}"
        )
    return 0


def _format_duration(seconds: float) -> str:
    """Compact duration for the sessions table: 45s, 12m, 3h, 2d."""
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h"
    return f"{hours // 24}d"


def mobile_command(args: argparse.Namespace) -> int:
    """Dispatch ``lop mobile …``. Imports are lazy: the mobile package pulls
    starlette/uvicorn only on commands that serve, and the CLI startup path
    must stay free of both."""
    command = getattr(args, "mobile_command", None)

    if command == "serve":
        from local_operator.mobile.service import main as serve_main

        return serve_main(args.port)

    from local_operator.mobile import install as mobile_install

    if command == "install":
        result = mobile_install.install()
        steps = result.get("steps", [])
        assert isinstance(steps, list)
        for step in steps:
            print(f"  {step}")
        if result.get("ok"):
            print("\nmobile daemon installed and healthy.")
            print("  open http://127.0.0.1:4098 and sign in with your portal password")
            print("  the password is in the login Keychain (service lop-mobile).")
            print("  retrieve it yourself with `lop mobile password` at a TTY —")
            print("  it is never printed here, so it cannot leak into a transcript.")
            return 0
        print(f"\n\033[1;31m{result.get('error', 'install failed')}\033[0m")
        return 1

    if command == "status":
        result = mobile_install.status()
        assert isinstance(result, dict)
        print(f"installed:    {'yes' if result['installed'] else 'no'}")
        print(f"password set: {'yes' if result['password_set'] else 'no'}")
        print(f"healthy:      {'yes' if result['healthy'] else 'no'}")
        gate = "closed" if result["gate_closed"] else "OPEN (this is a boundary failure)"
        print(f"auth gate:    {gate}")
        print(f"log:          {result['log']}")
        sessions = result.get("sessions", [])
        assert isinstance(sessions, list)
        print(f"sessions:     {len(sessions)}")
        for session in sessions:
            name = session["conversation_name"] or session["session_id"]
            print(
                f"  [{session['state']}] pid {session['pid']} · "
                f"{session['kind']} · {name} · {session['model_label']}"
            )
        return 0 if result["healthy"] else -1

    if command in ("start", "stop", "restart"):
        result = mobile_install.service_action(command)
        if not result["ok"]:
            print(f"\n\033[1;31m{result['error']}\033[0m")
            return 1
        print(f"mobile daemon {command} ok")
        return 0

    if command == "logs":
        import subprocess

        log = mobile_install.log_path()
        tail = ["tail", "-n", str(args.lines)]
        if args.follow:
            tail.append("-f")
        tail.append(str(log))
        return subprocess.call(tail)

    if command == "password":
        from local_operator.mobile.auth import (
            generate_password,
            load_password,
            store_password,
        )

        # A captured stdout (an agent tool result, a redirected log) is the
        # context window. Refuse to print the secret unless a human is at a
        # TTY. Rotation still works non-interactively via --rotate once we
        # have a TTY confirmation; without a TTY we only say where it lives.
        if not sys.stdout.isatty():
            print("portal password is in the login Keychain (service lop-mobile).")
            print("run `lop mobile password` in a terminal to view or rotate it.")
            return 0

        current = load_password()
        if current:
            print(f"current password: {current}")
            answer = input("rotate it? [y/N] ").strip().lower()
            if answer != "y":
                return 0
        new = generate_password()
        store_password(new)
        print(f"new password: {new}")
        print("restart the daemon to invalidate existing cookies: lop mobile restart")
        return 0

    if command == "uninstall":
        result = mobile_install.uninstall(purge=args.purge)
        steps = result.get("steps", [])
        assert isinstance(steps, list)
        for step in steps:
            print(f"  {step}")
        return 0

    print("usage: lop mobile {install|status|start|stop|restart|logs|password|uninstall|serve}")
    return 1


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
        return 1

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
        # The `seed:` provenance marker is bookkeeping this listing's reader
        # cannot act on: it records that a role was installed from a packaged
        # starter so `agent op='reset'` knows it may restore it. Hiding it
        # keeps a machine-only tag out of a human-facing inventory.
        shown_tags = [
            tag for tag in agent.tags if not str(tag).strip().lower().startswith(SEED_ORIGIN_PREFIX)
        ]
        if shown_tags:
            print(f"\033[1;32m{left_bar}   • Tags: {', '.join(shown_tags)}\033[0m")
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
                return 1
        except (KeyboardInterrupt, EOFError):
            print("\n\033[1;31mAgent creation cancelled\033[0m")
            return 1

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


def _cli_recovery_wait() -> float:
    """Read-path recovery budget for the one-shot ``teams`` commands (R7-1).

    Read from the registry module rather than restated so the two cannot
    drift. The import is FUNCTION-LOCAL for the same reason every other
    ``local_operator.teams`` reference in this module is: the module builds
    pydantic models at import time and must stay off the CLI startup path
    (pinned by ``test_import_graph``).
    """
    from local_operator.teams import _READ_RECOVERY_CLI_WAIT_S

    return _READ_RECOVERY_CLI_WAIT_S


def teams_list_command(team_registry: Any) -> int:
    """List all teams.

    Reads with the CLI recovery budget: this is a one-shot command that owns
    its process and blocks no event loop, so it can afford to wait briefly for
    a peer's publish to finish rather than lose the race and skip healing an
    interrupted save (R7-1; the UI path stays strictly non-blocking).
    """
    teams = team_registry.list_teams(recovery_wait=_cli_recovery_wait())
    if not teams:
        print("\n\033[1;33mNo teams found.\033[0m")
        return 0
    print("\n\033[1;32m╭─ Teams ─────────────────────────────────────\033[0m")
    for i, team in enumerate(teams):
        is_last = i == len(teams) - 1
        branch = "└──" if is_last else "├──"
        left = "│  " if is_last else "│ │"
        print(f"\033[1;32m│ {branch} {team.name}\033[0m")
        print(f"\033[1;32m{left}   • Manager: {team.manager}\033[0m")
        print(f"\033[1;32m{left}   • Members: {team.member_count()}\033[0m")
        if team.description:
            print(f"\033[1;32m{left}   • Description: {team.description}\033[0m")
        if not is_last:
            print("\033[1;32m│ │\033[0m")
    print("\033[1;32m╰──────────────────────────────────────────────\033[0m")
    return 0


def teams_create_command(args: argparse.Namespace, team_registry: Any) -> int:
    """Create a team from CLI flags."""
    from local_operator.teams import TeamEditFields, parse_members

    try:
        members = parse_members(getattr(args, "members", None))
        team = team_registry.create_team(
            TeamEditFields(
                name=args.name,
                description=getattr(args, "description", "") or "",
                manager=getattr(args, "manager", None) or "manager",
                members=members,
            )
        )
    except ValueError as exc:
        print(f"\n\033[1;31mError: {exc}\033[0m")
        return 1
    print("\n\033[1;32m╭─ Created New Team ───────────────────────────\033[0m")
    print(f"\033[1;32m│ Name: {team.name}\033[0m")
    print(f"\033[1;32m│ Manager: {team.manager}\033[0m")
    print(f"\033[1;32m│ Members: {team.member_count()}\033[0m")
    print("\033[1;32m╰──────────────────────────────────────────────────\033[0m\n")
    return 0


def teams_show_command(name: str, team_registry: Any) -> int:
    """Print a team's roster and briefs.

    Same one-shot recovery budget as ``teams_list_command`` (R7-1).
    """
    team = team_registry.get_team_by_name(name, recovery_wait=_cli_recovery_wait())
    if team is None:
        print(f"\n\033[1;31mError: No team found with name: {name}\033[0m")
        return 1
    print(f"\n\033[1;32m╭─ Team {team.name} ───────────────────────────\033[0m")
    print(f"\033[1;32m│ Manager: {team.manager}\033[0m")
    if team.description:
        print(f"\033[1;32m│ Description: {team.description}\033[0m")
    print("\033[1;32m│ Roster:\033[0m")
    for line in team.roster_lines():
        print(f"\033[1;32m│   {line}\033[0m")
    if team.instructions.strip():
        print("\033[1;32m│ Collaboration:\033[0m")
        for line in team.instructions.strip().splitlines():
            print(f"\033[1;32m│   {line}\033[0m")
    if team.project.strip():
        print("\033[1;32m│ Project:\033[0m")
        for line in team.project.strip().splitlines():
            print(f"\033[1;32m│   {line}\033[0m")
    print("\033[1;32m╰──────────────────────────────────────────────────\033[0m\n")
    return 0


def teams_delete_command(name: str, team_registry: Any) -> int:
    """Delete a team by name."""
    team = team_registry.get_team_by_name(name)
    if team is None:
        print(f"\n\033[1;31mError: No team found with name: {name}\033[0m")
        return 1
    team_registry.delete_team(team.id)
    print(f"\n\033[1;32mSuccessfully deleted team: {name}\033[0m")
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
            return 1

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
            return 1
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
            return 1
    else:
        print("\n\033[1;31mError: Must provide --name or --id for delete\033[0m")
        return 1


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
        return 1
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
        return 1
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
        return 1
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


async def _mcp_reauth_server(name: str, cwd: Path) -> int:
    """Log out of one OAuth MCP server, then run a fresh interactive grant.

    Plain ``mcp login`` reuses whatever the store still holds — the SDK only
    runs a browser grant once the stored token can neither be used nor
    refreshed, and a stored client registration short-circuits DCR. That is
    wrong for the cases reauth exists for: an account switch, a scope change,
    or a consent screen that needs to come back up. So reauth removes the row
    first (same deletion as ``mcp logout``, erroring on an unknown or
    non-OAuth name so a typo does not turn into an unexpected browser tab)
    and then runs exactly the login connect path — one implementation of what
    "authenticated" means.
    """
    from local_operator.mcp.auth import mcp_logout_server

    error = mcp_logout_server(name, cwd)
    if error is not None:
        print(f"error: MCP reauth failed for {name!r}: {error}", file=sys.stderr)
        return 1
    return await _mcp_login_server(name, cwd)


def mcp_command(args: argparse.Namespace) -> int:
    """Dispatch ``mcp list|add|login|logout|reauth|remove`` to MCP configuration and auth.

    Lazy import: the MCP package keeps its SDK imports lazy too, and this
    CLI must survive builds where it has not landed yet.
    """
    try:
        from local_operator.mcp import config as mcp_config
    except ImportError:
        print("\n\033[1;31mError: MCP support is not available in this build\033[0m")
        return 1

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
                return 1
            key, value = item.split("=", 1)
            env[key] = value
        # The config writers raise instead of printing so the TUI can call the
        # SAME implementation without writing to the terminal underneath its
        # Textual frame; the CLI's stderr text and exit codes are reproduced
        # here, at the CLI's own boundary. One error line per problem, exactly
        # as the writer used to print them.
        try:
            mcp_config.add_server(
                args.name,
                command=getattr(args, "command", None),
                args=getattr(args, "server_args", None),
                env=env or None,
                url=getattr(args, "url", None),
                oauth=bool(getattr(args, "oauth", False)),
                scope=getattr(args, "scope", "global"),
            )
        except mcp_config.MCPConfigWriteError as exc:
            for error in exc.errors:
                print(f"error: {error}", file=sys.stderr)
            return 1
        return 0
    if args.mcp_command == "login":
        import asyncio

        return asyncio.run(_mcp_login_server(args.name, Path.cwd()))
    if args.mcp_command == "logout":
        from local_operator.mcp.auth import mcp_logout_server

        error = mcp_logout_server(args.name, Path.cwd())
        if error is not None:
            print(f"error: MCP logout failed: {error}", file=sys.stderr)
            return 1
        print(f"Removed the stored OAuth credential for MCP server {args.name!r}.")
        return 0
    if args.mcp_command == "reauth":
        import asyncio

        return asyncio.run(_mcp_reauth_server(args.name, Path.cwd()))
    if args.mcp_command == "remove":
        try:
            mcp_config.remove_server(args.name, scope=getattr(args, "scope", "global"))
        except mcp_config.MCPConfigWriteError as exc:
            for error in exc.errors:
                print(f"error: {error}", file=sys.stderr)
            return 1
        return 0

    print(f"\n\033[1;31mError: Invalid mcp command: {args.mcp_command}\033[0m")
    return 1


# --- Session factory facade -------------------------------------------------


async def create_session(
    args: argparse.Namespace,
    config_manager: ConfigManager,
    credential_manager: CredentialManager,
    agent_registry: "AgentRegistry",
    *,
    has_ui: bool = False,
    defer_mcp_wiring: bool = False,
):
    """Build a wired harness session for interactive/headless use.

    Thin facade over :func:`local_operator.session_factory.create_session`
    (the composition root shared with ``exec`` and the background worker).
    The engine import is lazy so importing ``cli`` never pulls in
    providers/session internals. ``defer_mcp_wiring`` passes through to the
    factory's TUI-boot opt-in unchanged (see its docstring for why only a
    full front end may take it).
    """
    from local_operator.session_factory import create_session as _create_session

    return await _create_session(
        args,
        config_manager,
        credential_manager,
        agent_registry,
        has_ui=has_ui,
        defer_mcp_wiring=defer_mcp_wiring,
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
        return 1
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
    import logging

    from rich.console import Console

    from local_operator.headless_print import PrintRenderer

    # Raise the console threshold to WARNING for the REPL. configure_cli_logging
    # pins the root logger at INFO, and the headless REPL — unlike the TUI,
    # which wraps its whole run in file_logging() — prints straight to the
    # terminal, so httpx's one-INFO-line-per-request and every other INFO record
    # leaked into the transcript BEFORE the first prompt and between turns. The
    # TUI's remedy (detach console handlers) is wrong here because the REPL's
    # own output IS console output; lifting the level keeps its prints while
    # dropping the library chatter. WARNING and above still surface — a genuine
    # problem the user needs to see is not INFO.
    #
    # The noisy HTTP-client loggers are raised EXPLICITLY, not just via the root:
    # configure_cli_logging pins each of them to INFO by name, and a child logger
    # with its own level ignores the root's — so raising only the root left
    # httpx's per-request line leaking. Same list configure_cli_logging quietens.
    logging.getLogger().setLevel(logging.WARNING)
    for _noisy in ("requests", "urllib3", "httpx", "httpcore"):
        logging.getLogger(_noisy).setLevel(logging.WARNING)

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
    allow_setup_state: bool = False,
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
    a stderr warning, and exists for the interactive front ends: the TUI's
    ``/login`` command is the product's own remedy for a missing key, and a
    fatal preflight sat exactly between the user and that remedy — a fresh
    config whose default hosting was a keyed provider could not start at all.
    The headless REPL has no slash commands, but its remedy (``local-operator
    login``) is named in the warning it keeps visible on stderr, and a keyless
    turn fails with its own accurate per-turn message rather than a lockout.
    Session construction never needs the key (stream time resolves it through
    the AuthStore cascade) and the TUI splash already shows "not logged in —
    /login <provider>", so letting the app start loses nothing. Hosting/model
    resolution errors stay fatal on every path: without a hosting there is no
    session to build and nothing for a login to fix.

    ``allow_setup_state=True`` is the first-run onboarding gate (item A1/U1):
    when NO hosting can be resolved at all AND we are on the interactive TUI
    path (tty + TUI enabled), the app is allowed to open in a SETUP STATE
    instead of dying at preflight. The welcome splash's ``/login`` affordance
    and the ``/model`` / ``/provider`` surfaces are the guided setup — there is
    no separate wizard. Every OTHER path (headless REPL, exec, non-tty) keeps
    fail-fast, and does it with a COMPLETE quickstart that names everything
    missing at once rather than one field at a time.

    Returns 1 (already printed) on failure, None to continue. All engine
    imports stay lazy so this never weights down parser-only paths.
    """
    from local_operator.session_factory import (
        HostingNotConfiguredError,
        HostingUnknownError,
        ModelNotConfiguredError,
        resolve_hosting_model,
    )

    try:
        hosting, _model_name = resolve_hosting_model(current_agent, args, config_manager)
    except ModelNotConfiguredError as exc:
        # Hosting is a real provider but has no resolvable model -- the state
        # `/login <provider-with-no-default>` writes on purpose. Recoverable on
        # the interactive path in EXACTLY the way an unknown hosting is: the
        # setup state's `/model` picker writes the missing value, so the app
        # opens rather than refusing to launch. Ordered before its base class
        # for the same reason the HostingUnknownError branch is: it is a
        # subclass and would otherwise be swallowed by that handler, which
        # would print the first-run quickstart and never mention the model.
        if allow_setup_state:
            return None
        # Non-interactive paths keep fail-fast with the informative message: a
        # scripted or CI run has nobody to answer a picker, and limping along
        # on a model nobody chose is how a cron job silently bills a different
        # provider. Same shape as the ValueError branch below, which this
        # branch now shadows for the resolver's own raise.
        from local_operator.cli_style import ERROR, paint

        print(paint(f"Error: {exc}", ERROR, stream=sys.stderr), file=sys.stderr)
        return 1
    except HostingUnknownError as exc:
        # Hosting names a provider the registry does not own (a typo, a
        # hand-edited config, an id dropped by an upgrade). Recoverable in
        # EXACTLY the way "nothing configured" is -- the user fixes it with
        # `/login` or `/provider` from inside the app -- so the interactive TUI
        # path opens in the same setup state rather than dying at preflight.
        # Ordered before the HostingNotConfiguredError branch because it is a
        # subclass of it and would otherwise be swallowed by that handler, which
        # would print the first-run quickstart and never name the bad value.
        if allow_setup_state:
            return None
        # Non-interactive paths (headless REPL, exec, non-tty) keep fail-fast:
        # a scripted run must not limp along with no usable model. The message
        # names the offending value AND the remedy, following
        # `_print_first_run_quickstart`'s "name everything at once" principle --
        # the quickstart itself is wrong here, because it says "nothing is
        # configured" when something IS configured, just not to a real provider.
        from local_operator.cli_style import ERROR, paint

        print(paint(f"Error: {exc}", ERROR, stream=sys.stderr), file=sys.stderr)
        return 1
    except HostingNotConfiguredError:
        # No hosting resolved. On the interactive TUI path this is not an error:
        # the app opens in a setup state so the user can `/login` from inside it.
        if allow_setup_state:
            return None
        # Every other path keeps fail-fast, but with the WHOLE quickstart at
        # once (item A1/U1) — the old message named only "Hosting platform is
        # not configured" and the user fixed it one error at a time.
        _print_first_run_quickstart(credential_manager)
        return 1
    except ValueError as exc:
        # A model-resolution error (hosting set, no default known): fatal on
        # every path, one line. stderr on principle, not because this path is
        # currently reachable from `exec --json`: it is an ERROR message, and
        # its sibling `_preflight_api_key` two functions down already writes
        # there. Keeping the two consistent is what stops the next person wiring
        # this into the exec route from reintroducing a stdout leak.
        from local_operator.cli_style import ERROR, paint

        print(paint(f"Error: {exc}", ERROR, stream=sys.stderr), file=sys.stderr)
        return 1
    except Exception:  # noqa: BLE001 — unknown providers pass through
        return None

    return _preflight_api_key(hosting, credential_manager, require_key=require_api_key)


def _print_first_run_quickstart(credential_manager: CredentialManager) -> None:
    """One complete message naming hosting, model AND key at once (item A1/U1).

    The fail-fast paths (headless REPL, exec, non-tty) reach this when nothing
    is configured. The point of naming all three missing pieces together, with
    the exact commands, is that a scripted or headless user fixes the whole
    thing in one pass instead of rerunning into "hosting missing", then "model
    missing", then "key missing" \u2014 the one-error-at-a-time treadmill the
    interactive setup state exists to avoid and this message is the non-tty
    equivalent of.
    """
    from local_operator.cli_style import ERROR, INFO, paint

    print(
        paint(
            "Error: Local Operator is not configured yet \u2014 no hosting provider, "
            "model, or credential is set.",
            ERROR,
            stream=sys.stderr,
        ),
        file=sys.stderr,
    )
    print(
        paint(
            "Set it up with (pick a provider, e.g. openai / anthropic / deepseek):\n"
            "  local-operator login <provider>          "
            "# stores the key AND sets hosting + a default model\n"
            "or configure the pieces individually:\n"
            "  local-operator config edit hosting <provider>\n"
            "  local-operator config edit model_name <model>\n"
            "  local-operator credential update <PROVIDER_API_KEY>\n"
            "or pass them per-run with the --hosting and --model flags.\n"
            "On an interactive terminal, just run `local-operator` and log in "
            "from the setup screen.",
            INFO,
            stream=sys.stderr,
        ),
        file=sys.stderr,
    )


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
        from local_operator.providers.registry import credential_provider_id

        auth_store = AuthStore(credential_manager=credential_manager)
        try:
            storage_provider = credential_provider_id(canonical)
            if auth_store.list_credentials(provider=storage_provider):
                return None
            api_key = asyncio.run(auth_store.get_api_key(canonical))
        finally:
            auth_store.close()
    except Exception:  # noqa: BLE001 — resolution failures pass through
        return None

    if api_key:
        return None

    from local_operator.cli_style import ERROR, WARNING, paint
    from local_operator.providers.registry import env_key_name

    # ``env_key_name`` returns None for a CALLABLE env_keys resolver (Anthropic
    # picks between ANTHROPIC_OAUTH_TOKEN and ANTHROPIC_API_KEY, so there is no
    # single var to name). The old code fell back to the literal string "API
    # key" and interpolated it into the command template, producing the invalid
    # advice `credential update API key`. Only offer `credential update <NAME>`
    # when there is a real env var name; otherwise recommend `login` only.
    key_name = env_key_name(canonical)
    credential_hint = f", `local-operator credential update {key_name}`" if key_name else ""
    env_hint = f", or set {key_name} in the environment" if key_name else ""

    if not require_key:
        # Interactive start: name the fact and the remedies, then let the app
        # come up. `/login` is scoped to the TUI because the headless REPL has
        # no slash dispatch — there the shell command is the remedy, and this
        # line stays visible on stderr (the TUI repaints over it, but its
        # splash carries the same warning).
        print(
            paint(
                f"Warning: no credentials are configured for hosting platform "
                f"'{hosting}'. Starting anyway — run `/login {canonical}` in the "
                f"TUI, `local-operator login {canonical}` from a shell{env_hint}.",
                WARNING,
                stream=sys.stderr,
            ),
            file=sys.stderr,
        )
        return None
    # stderr: this fires on every fresh install and every typo'd --hosting,
    # i.e. it is the single most common `exec --json` failure, and a coloured
    # non-JSON line on stdout breaks the consumer it is trying to inform.
    subject = key_name if key_name else "an API key"
    print(
        paint(
            f"Error: {subject} is required for hosting platform '{hosting}' but "
            f"is not configured. Set it via `local-operator login {canonical}`"
            f"{credential_hint}, or the environment.",
            ERROR,
            stream=sys.stderr,
        ),
        file=sys.stderr,
    )
    return 1


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


def _install_group_reaper_soft_death() -> None:
    """Wire the process-group reaper's soft-death path for THIS process.

    Registers ``group_reaper.kill_own_groups`` as an ``atexit`` hook and as a
    SIGTERM handler, so a catchable stop of the interactive TUI/headless
    REPL reaps this process's own still-live bash groups instead of leaking them
    to the next launch's startup sweep. The whole leak this addresses is a HARD
    (uncatchable SIGKILL) death, which no handler can cover — but a POLITE stop
    (cmux replace, launchd stop, Ctrl+D / quit / window close at the REPL) IS
    catchable, and reaping it here makes the common case instant and precise
    rather than deferred.

    SIGINT is DELIBERATELY excluded. In the headless REPL, Ctrl-C is a *turn
    abort that keeps the session alive* (``_run_headless_repl`` catches
    ``KeyboardInterrupt`` -> ``session.abort`` -> loops), and ``session.abort``
    deliberately spares ``background=true`` bash jobs — they exist precisely so a
    build or deploy outlives the turn that started it. Reaping on SIGINT would
    SIGKILL those still-live groups while the owning REPL keeps running, which is
    exactly the never-kill-a-live-owner case this whole module forbids. Every
    real REPL/TUI *exit* (Ctrl-D, ``quit``, window close) still reaps via the
    ``atexit`` hook, ``session.dispose()`` and the TUI teardown ``finally``, so
    nothing is lost by leaving SIGINT to its turn-abort semantics.

    Scoped to the interactive entry on purpose: ``exec``/``serve``/``mobile``
    own their own SIGTERM semantics (``exec_worker.py``,
    ``session/runtime/process.py``) and
    are dispatched before this is ever called. As a second belt, any
    pre-existing SIGTERM handler is CHAINED, not clobbered — the reaper
    runs first, then the previous handler (or the default) still fires — so this
    can never silently swallow a signal another component was relying on.

    Best-effort and idempotent: the reaper unlinks its ledger on the first call,
    so the atexit hook, a signal, and the TUI teardown ``finally`` firing in any
    order all converge on one reap. On Windows the reaper is a no-op, and the
    signal registration is guarded so a platform without SIGTERM is harmless.
    """
    import atexit
    import contextlib
    import signal

    from local_operator.tools.group_reaper import kill_own_groups

    atexit.register(kill_own_groups)

    def _chain(signum: int) -> None:
        previous = signal.getsignal(signum)

        def _handler(received_signum, frame):  # type: ignore[no-untyped-def]
            try:
                kill_own_groups()
            except Exception:  # noqa: BLE001 — a handler must never raise
                pass
            # Chain to whatever was installed before us so the process still
            # stops the way it otherwise would (default disposition included).
            if callable(previous):
                previous(received_signum, frame)
            elif previous == signal.SIG_DFL:
                # Restore the default and re-raise so the default action (e.g.
                # terminate) actually happens rather than being swallowed.
                signal.signal(received_signum, signal.SIG_DFL)
                os.kill(os.getpid(), received_signum)

        with contextlib.suppress(ValueError, OSError):
            # ValueError: not the main thread; OSError: unsupported signal.
            signal.signal(signum, _handler)

    # SIGTERM only. SIGINT is a turn abort in the headless REPL and must NOT
    # reap live background jobs (see the docstring); a Ctrl-C that actually
    # exits reaps through atexit/dispose instead.
    for _sig in (signal.SIGTERM,):
        _chain(_sig)


def main() -> int:
    # FIRST, before anything else can log. `helpers.py` used to configure the
    # root logger as an import side effect; now the entry point owns it, which
    # is what lets the TUI branch below swap the console handler for a file.
    configure_cli_logging()
    try:
        parser = build_cli_parser()
        args = parser.parse_args()

        # Prime the login-shell PATH only on paths that actually spawn
        # subprocess work: the interactive session, exec, serve and mobile all
        # run shell commands whose PATH must match a login terminal's, but
        # `config list`, `credential`, `login`, `agents` and the like never
        # spawn a tool — yet every one of them used to pay a full login-shell
        # round-trip (`$SHELL -l -c 'echo $PATH'`) on startup. The helper keeps
        # a per-process cache, so a session that later needs it still primes at
        # most once. ``None`` is the bare interactive launch.
        _SUBPROCESS_SUBCOMMANDS = frozenset({"exec", "serve", "mobile", "browser"})
        if args.subcommand in _SUBPROCESS_SUBCOMMANDS or args.subcommand is None:
            setup_cross_platform_environment()

        # Resolve `--resume` HERE, before anything is started. Left to the
        # session factory it surfaces inside the TUI as "session failed to
        # start" — a full-screen app launched, painted, and torn down to report a
        # typo — and the generic handler below would render it as a traceback
        # panel and still exit 0. A bad session id is ordinary user error, so it
        # gets a one-line message, the ids that DO exist, and a non-zero status.
        if getattr(args, "resume", None) is not None:
            from local_operator.resume import (
                RESUME_RECOVERY_LISTING,
                ResumeNotFound,
                backfill_session_origins,
                backfill_session_titles,
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
            # Stamp the title sidecar for pre-existing sessions in the same
            # sweep, for the same reason: a session whose title sits in the
            # untouched middle of a large transcript is unfindable by its own
            # subject until this runs. Idempotent and stdlib-only, so it costs a
            # bounded directory scan. session_factory._prepare backfills too (on
            # ordinary session build); this branch runs it eagerly here because
            # it answers `--resume` before any session is built, so the picker
            # and `@latest` resolution above must see a stamped store first.
            backfill_session_titles(config_dir())

            try:
                args.resume = resolve_resume_id(config_dir(), str(args.resume))
            except ResumeNotFound as error:
                print(f"\033[31m{error}\033[0m", file=sys.stderr)
                # With the age: a column of bare 12-hex ids gives the reader
                # nothing to choose between, and the recency the listing already
                # sorted by is the one fact that makes them recognisable.
                # Ten explicitly: this is an error path printing to stderr after
                # a typo'd id, where a short list of the most recent sessions is
                # the help and the whole store would bury it. ``recent_sessions``
                # returns everything by default, so the cap belongs here where a
                # reader can see the listing is deliberately short.
                available = recent_sessions(config_dir(), limit=RESUME_RECOVERY_LISTING)
                if available:
                    now = time.time()
                    print("recent sessions (newest first):", file=sys.stderr)
                    for session_id, mtime in available:
                        print(
                            f"  {session_id}   {format_age(now - mtime)}",
                            file=sys.stderr,
                        )
                return 1

            # Cold live-session resumes now stay on the ordinary TUI launch
            # path. ``create_session(has_ui=True)`` returns a RemoteSession when
            # another process owns the transcript, so the STANDARD OperatorApp
            # renders it with no standalone attach app, exit-75 relaunch, or
            # visible mode. The shared factory still protects the sole-writer
            # invariant; exec/headless retains its refusal below.

        os.environ["LOCAL_OPERATOR_DEBUG"] = "true" if args.debug else "false"
        # (CL-12) No env_config binding here: the scheduler wrapper resolves its
        # own env config and the session factory does the same lazily — a
        # dead local would only invite drift.
        base_dir = config_dir()
        # The agent home is NO LONGER created here. Creating it unconditionally
        # before dispatch meant `config list`, `login`, `--version` and every
        # other non-session subcommand created a workspace directory they never
        # touch, and it hardcoded ~/local-operator-home while ignoring any
        # override. It is now created lazily by the paths that actually run a
        # task (session/exec/serve start), through paths.ensure_agent_home_dir.

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
        elif args.subcommand == "fetch":
            from local_operator.web_fetch.cli import fetch_command

            return fetch_command(args)
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
                    return 1
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
                        return 1
                elif getattr(args, "id", None):
                    try:
                        agent = agent_registry.get_agent(args.id)
                        agent_id_to_overwrite = args.id
                    except KeyError:
                        print(f"\n\033[1;31mError: No agent found with ID: {args.id}\033[0m")
                        return 1
                else:
                    print("\n\033[1;31mError: Must provide --name or --id for push\033[0m")
                    return 1
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
                    return 1
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
                    return 1
            else:
                parser.error(f"Invalid agents command: {args.agents_command}")
        elif args.subcommand == "teams":
            # U5-1: every teams subcommand can hit the registry lock, and lock
            # contention is a recoverable state — print one concise line here
            # instead of letting the generic handler below render a traceback
            # panel that reads as a crash. The import is function-local for the
            # same reason the teams module is: ``local_operator.types`` builds
            # pydantic models at import time and must stay off the startup path
            # (pinned by test_import_graph).
            from local_operator.teams import (
                TeamRegistry,
                TeamRegistryLockTimeout,
                TeamRegistryRecoveryError,
            )

            try:
                team_registry = TeamRegistry(base_dir)
                if args.teams_command == "list":
                    return teams_list_command(team_registry)
                elif args.teams_command == "create":
                    return teams_create_command(args, team_registry)
                elif args.teams_command == "show":
                    return teams_show_command(args.name, team_registry)
                elif args.teams_command == "delete":
                    return teams_delete_command(args.name, team_registry)
                else:
                    parser.error(f"Invalid teams command: {args.teams_command}")
            except (TeamRegistryLockTimeout, TeamRegistryRecoveryError) as e:
                print(f"\n\033[1;31mError: {str(e)}\033[0m", file=sys.stderr)
                return 1
        elif args.subcommand == "serve":
            # Use the provided host, port, and reload options for serving the API.
            return serve_command(args.host, args.port, args.reload)
        elif args.subcommand == "mobile":
            return mobile_command(args)
        elif args.subcommand == "browser":
            return browser_command(args)
        elif args.subcommand == "send":
            return send_command(args)
        elif args.subcommand == "sessions":
            return sessions_command(args)
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
        elif args.subcommand == "update":
            # Lazy: ``update`` imports httpx. ``import local_operator.cli``
            # must not (``tests/unit/test_import_graph.py``).
            from local_operator.update import update_command

            return update_command(check=bool(getattr(args, "check", False)))
        elif args.subcommand == "exec":
            # Single-execution mode: headless one-shot (README contract —
            # exit 0 on success, non-zero on error). Working-directory
            # handling matches the legacy pre-run behavior.
            invalid = _apply_run_in(args.run_in)
            if invalid is not None:
                return invalid
            # Second-writer guard, same rationale as the interactive branch
            # above but with no attach escape: exec is headless and one-shot,
            # following a live session is meaningless, and double-writing a
            # transcript another process owns is the corruption case. The
            # refusal IS the feature; no new flag.
            if getattr(args, "resume", None) is not None:
                from local_operator.resume import live_session_owner, resolve_resume_id

                try:
                    exec_resume_id = resolve_resume_id(config_dir(), str(args.resume))
                except Exception:
                    exec_resume_id = str(args.resume)
                exec_owner = live_session_owner(config_dir(), exec_resume_id)
                if exec_owner is not None and exec_owner != os.getpid():
                    print(
                        f"\033[31msession {exec_resume_id} is already open in "
                        f"another process (pid {exec_owner}) — watch and steer "
                        "it there, or from the phone session list\033[0m",
                        file=sys.stderr,
                    )
                    return 1
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
                    # stdout is the event stream. Return 1 (not -1) so the
                    # scripted `exec --json` case exits with a clean non-zero;
                    # the byte-identical twins in exec_mode._spawn_background
                    # follow the same contract.
                    print(f"\n\033[1;31mError: {exc}\033[0m", file=sys.stderr)
                    return 1
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"\n\033[1;31mError: preflight failed: {exc}\033[0m",
                        file=sys.stderr,
                    )
                    return 1
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
                    return 1

        # Legacy behavior: the auto-save config value persists interactive
        # sessions via the registry's autosave agent (exec is excluded —
        # single-execution mode never autosaved).
        auto_save_enabled = config_manager.get_config_value("auto_save_conversation", False)
        if auto_save_enabled:
            args.train = True

        # Interactive path: full-screen TUI when stdout is a tty and not
        # disabled; plain headless REPL otherwise. ``--tui`` (CL-13) forces
        # the TUI even when stdout is not a tty — with a clear error when
        # that is impossible.
        #
        # Decided BEFORE the preflight (it used to come after) because the
        # preflight now needs the answer: only the TUI path may open in a
        # first-run setup state instead of failing, so ``use_tui`` gates that.
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
                    from local_operator.cli_style import ERROR, paint

                    print(
                        paint(
                            "Error: the TUI is not available in this build/install "
                            "(missing 'local_operator.tui'); remove --tui to use the "
                            "plain REPL.",
                            ERROR,
                            stream=sys.stderr,
                        ),
                        file=sys.stderr,
                    )
                    return 1
                use_tui = False

        # Whether the app can open in a first-run setup state: only when the
        # full-screen TUI is actually going to run, since the splash's `/login`
        # affordance is the setup UI. The headless REPL and every non-tty path
        # (piped stdout, `--no-tui`) keep fail-fast with the complete quickstart.
        setup_state_ok = bool(use_tui and run_tui is not None)

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
            allow_setup_state=setup_state_ok,
        )
        if preflight_result is not None:
            return preflight_result

        # asyncio is imported HERE, not at module scope. It is the heaviest
        # single item on the CLI's import graph (34.4 ms, +6.5 MB RSS, +77
        # modules measured by scripts/bench_base_overhead.py) and only the
        # interactive TUI/REPL tail below needs it — `--version`, `--help`,
        # shell completion and the config/credential/agents/login subcommands
        # all return before this point, and `exec`/`serve` bring their own
        # event loop from exec_mode/the server module.
        import asyncio

        # Soft-death process-group reaper (tools/group_reaper.py). Installed
        # ONLY on the interactive TUI + headless REPL entry — the two paths that
        # spawn bash tool groups and tear down in this process. A catchable stop
        # (the polite cmux stop, a launchd stop, a clean quit, or an unexpected
        # exit) then kills this process's own still-live bash groups precisely
        # and instantly instead of leaving them for the next launch's sweep.
        # Deliberately NOT installed for `exec`/`serve`/`mobile`: those own their
        # own SIGTERM lifecycle (exec_worker.py, session/runtime/process.py) and must keep
        # it — `_install_group_reaper_soft_death` chains any pre-existing handler
        # rather than clobbering it, but scoping to here keeps the concern where
        # the groups are actually created.
        _install_group_reaper_soft_death()

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
                    defer_mcp_wiring=True,
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
                # ``on_config_changed`` re-reads config.yml into THIS manager
                # after the app's first-run ``/login`` writes hosting/model to
                # disk. The session factory closes over this exact instance, so
                # without the reload the post-login rebuild would resolve the
                # same empty config and bounce back into the setup state.
                tui_entry = functools.partial(
                    run_tui,
                    provider_controller=tui_controller,
                    on_config_changed=config_manager.reload,
                )

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
                        defer_mcp_wiring=True,
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
                    tui_code = asyncio.run(
                        _run_with_scheduler(
                            tui_entry,
                            session_factory,
                            theme_name,
                        )
                    )
                # 75: the TUI asked to be replaced after a clean teardown.
                # ``replace_self`` does not return; a missing plan is a bug.
                from local_operator.reexec import REEXEC_CODE, replace_self, take_plan

                if tui_code == REEXEC_CODE:
                    plan = take_plan()
                    if plan is None:
                        return 1
                    replace_self(plan)
                return tui_code
            finally:
                try:
                    tui_controller.close()
                except Exception:  # noqa: BLE001 — closing on teardown, never fatal
                    pass
                try:
                    tui_auth_store.close()
                except Exception:  # noqa: BLE001 — closing on teardown, never fatal
                    pass
                # Reap this process's own bash groups on TUI teardown — a clean
                # quit, an exception exit, or after run_tui returns. Idempotent
                # with the atexit/signal hooks (the ledger is unlinked on the
                # first call). See _install_group_reaper_soft_death.
                try:
                    from local_operator.tools.group_reaper import kill_own_groups

                    kill_own_groups()
                except Exception:  # noqa: BLE001 — teardown, never fatal
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
        # U5-1 (narrow re-check): if the failure is teams-registry lock
        # contention — an expected recoverable state, not a defect — present
        # one concise line instead of the traceback panel below, which reads
        # as a crash and asks the user to "correct" something only the peer
        # process can resolve. Matched by TYPE NAME so no eager import is
        # needed (``local_operator.types`` must stay off the startup path,
        # pinned by test_import_graph); every other exception falls through
        # to the full presenter unchanged.
        if type(e).__name__ in {
            "TeamRegistryLockTimeout",
            "TeamRegistryRecoveryError",
        } and isinstance(e, (TimeoutError, RuntimeError)):
            print(f"\n\033[1;31mError: {str(e)}\033[0m", file=sys.stderr)
            return 1
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
        return 1


if __name__ == "__main__":
    exit(main())
