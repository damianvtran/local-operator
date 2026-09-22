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
import functools
import math
import os
import platform
import re
import shlex
import socket
import sqlite3
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

# stdlib-only and import-cheap by construction (os/sys/pathlib/logging), so it
# does not violate this module's no-heavy-module-level-imports rule.
from local_operator import procname
from local_operator.agent_profiles import SEED_ORIGIN_PREFIX
from local_operator.agent_shell import exec_session_refusal, interactive_session_refusal
from local_operator.config import ConfigManager
from local_operator.env import get_env_config, resolve_radient_api_base_url
from local_operator.logger import configure_cli_logging, file_logging
from local_operator.optional import missing_extra_error
from local_operator.paths import config_dir

# `local_operator.resume` is deliberately tiny (pathlib only): importing the
# session factory here for the same constant dragged the harness and asyncio
# onto the CLI startup path, which test_import_graph exists to prevent.
from local_operator.resume import RESUME_LATEST

if TYPE_CHECKING:
    from collections.abc import Sequence

    from local_operator.agents import AgentRegistry

from local_operator.helpers import setup_cross_platform_environment

#: The `lop services restart --wait` default, in seconds.
#:
#: DUPLICATED ON PURPOSE, WITH A TEST AS THE GUARD. The number it must agree with is
#: ``services.RELOAD_WAIT_S``, and ``cli.py`` may not import that module: this file's
#: startup path is asserted stdlib-light, and ``services`` reaches ``asyncio`` (the
#: test is ``tests/unit/test_import_graph.py::test_cli_import_does_not_load_asyncio``).
#: Naming the wrong default in ``--help`` while the tool used another was design review
#: D2 — a flag that misreports its own behaviour is worse than one that has no default
#: text — so the literal lives here and
#: ``tests/unit/test_services.py::test_the_documented_wait_default_is_the_one_used``
#: fails if the two ever drift apart.
DEFAULT_SERVICES_WAIT_S = 45.0


def _positive_seconds(text: str) -> float:
    """argparse ``type`` for a wait budget that must be greater than zero.

    REFUSED AT PARSE TIME, NOT AFTER THE WORK STARTS (design review D2). The first
    version accepted anything: ``--wait -1`` ran the whole restart and then reported
    "within -1s", and ``--wait 0.5`` reported "within 0s" because the message
    formatted to whole seconds. A budget the tool cannot honour is a usage error, and
    argparse says so before a single daemon is touched.
    """
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number of seconds") from None
    if not math.isfinite(value):
        # ``nan``/``inf``/``1e400`` passed the ``<= 0`` test below, because every
        # comparison against ``nan`` is False. Measured by review round 10: the
        # relocation poll never expired against ``nan`` (51 polls, 10.4 s, deadline
        # never fired), and the echo printed "within nans" — so the validator admitted
        # exactly the budget it exists to refuse.
        raise argparse.ArgumentTypeError("must be a finite number of seconds")
    if value <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return value


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
        help="Select a legacy named agent (creates it if missing). Without --train, "
        "use a separate persisted session; exec --profile attaches a reusable role instead.",
        dest="agent_name",
    )
    parent_parser.add_argument(
        "--train",
        action="store_true",
        help="Use the legacy agent history directory (or an autosave agent) instead "
        "of a separate session. Ordinary sessions are also persisted and resumable; "
        "--resume takes precedence over this directory selection.",
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
    # ``installed_version()``, not ``version("local-operator")``: install
    # metadata is written once and never moves when the checkout's
    # ``pyproject.toml`` does, so the bare metadata call reports the version an
    # editable install was CREATED at rather than the one it is running — and a
    # leftover ``*.egg-info`` shadows the real dist-info downward on top of
    # that. Both were live here: the reported symptom this change ships with
    # was a wrong version number, and `--version` is the surface a user checks
    # first, so it must not be the one place that still answers from the stale
    # channel. See :func:`local_operator.update.installed_version`.
    #
    # Imported inside the function because ``local_operator.update`` pulls
    # ``ssl``/``urllib.request`` (measured ~28 ms cumulative against an ~84 ms
    # `import local_operator.cli` baseline). This parser is built once per
    # invocation, so the cost lands only on runs that build it, and the
    # startup-path guards in tests/unit/test_import_graph.py stay satisfied.
    from local_operator.update import installed_version

    parser.add_argument(
        "--version",
        action="version",
        version=f"v{installed_version()}",
        help="Show program's version number and exit",
    )
    parser.add_argument(
        "--hosting",
        type=str,
        # A list of CHAT hostings, and `typesafe` (TypeSafe's Jev, the decision
        # model behind the resource-classification layer) is deliberately not in
        # it even though the provider registry has a row and a login for it: Jev
        # rejects `chat/completions` on every host we reach it through, so a
        # session started on it cannot answer a turn. The registry row carries
        # `decision_only=True` and four surfaces enforce it — the catalogue, the
        # /model ranking, the session-model resolver and the failover chain
        # (`tests/unit/providers/test_decision_only.py`).
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

    # Instructions command. Takes ``--agent`` from ``parent_parser``, which is
    # what lets it report the profile layer a session with that agent would
    # actually assemble rather than only the two global sources.
    config_subparsers.add_parser(
        "instructions",
        help="Show which custom-instruction files a session assembles, in order "
        "(paths and sizes only, never their contents)",
        # ``description=`` as well as ``help=``: the latter renders only on the
        # PARENT ``config --help`` page, so without this the command's own
        # ``--help`` is blank above the options list.
        description="Show which custom-instruction files a session assembles, in "
        "order, with the size each contributed and whether it was collapsed as a "
        "duplicate, truncated, or overlaps an earlier source. Paths and sizes "
        "only, never the contents.",
        parents=[parent_parser],
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
        default="127.0.0.1",
        help=(
            "Host address (default: 127.0.0.1). This API has no authentication; "
            "use a non-loopback address only behind trusted access controls."
        ),
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
    serve_parser.add_argument(
        "--listener-fd",
        type=int,
        default=None,
        help=(
            "Adopt an already-bound listening socket instead of binding one. "
            "Set by a daemon replacing its own process image onto a new build "
            "(`server/reload`), so the port never has a moment with nothing "
            "behind it. Not a user-facing flag."
        ),
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
    logs_parser = mobile_subparsers.add_parser(
        "logs", help="Tail the daemon's and the runtimes' logs"
    )
    logs_parser.add_argument(
        "--lines", type=int, default=100, help="Lines per file (up to two are read)"
    )
    logs_parser.add_argument(
        "--follow",
        "-f",
        action="store_true",
        help=(
            "Follow by name, so a log created or rotated while you watch is seen "
            "(a descriptor-bound follow misses both)"
        ),
    )
    mobile_subparsers.add_parser("password", help="Show or rotate the portal password")
    uninstall_parser = mobile_subparsers.add_parser("uninstall", help="Remove the LaunchAgent")
    uninstall_parser.add_argument("--purge", action="store_true", help="Also delete the password")
    serve_mobile_parser = mobile_subparsers.add_parser("serve", help="Run the daemon (foreground)")
    serve_mobile_parser.add_argument("--port", type=int, default=4098)

    # Register only stdlib arguments here. JWT/HTTP/service code remains lazy
    # so an ordinary terminal session pays no tunnel startup cost.
    from local_operator.tunnels.arguments import add_parser as add_tunnel_parser

    add_tunnel_parser(subparsers)

    # Encrypted secret store. Same discipline: argument registration is
    # stdlib-only, so `--version` and `--help` never load the crypto stack that
    # the handlers import at the point of use.
    from local_operator.secrets.cli import add_parser as add_secret_parser

    add_secret_parser(subparsers)

    # Operator authority (issue #1310, revision 2). Registration is stdlib-only
    # for the same reason `secret`'s is: `lop --version` must not load
    # Security.framework, the CNG stack or `cryptography`, and the verbs that do
    # live in `operator/handlers.py`, imported only when a verb is dispatched.
    from local_operator.operator.cli import add_parser as add_operator_parser

    add_operator_parser(subparsers)

    # Device pairing (stage D of the same design). Registered beside
    # `operator` because it is the same trust root seen from the other end — the
    # operator signs the certificate, the phone holds the key — and stdlib-only
    # for the identical reason: `lop --version` must not load the keychain.
    from local_operator.operator.pair import add_parser as add_pair_parser

    add_pair_parser(subparsers)

    # QwenCloud console session cookie: the credential the personal Token Plan
    # usage window needs and no login flow can mint (a browser session cookie
    # cannot be refreshed headlessly). stdlib-only registration, same rule.
    qwencloud_parser = subparsers.add_parser(
        "qwencloud-ticket",
        help="Store the QwenCloud console session cookie that /usage reads",
    )
    qwencloud_actions = qwencloud_parser.add_subparsers(dest="qwencloud_command")
    qwencloud_actions.add_parser(
        "set", help="Store the cookie; the value is read from STDIN, never argv"
    )
    qwencloud_actions.add_parser(
        "status", help="Whether a cookie is stored and how old it is; never the value"
    )
    qwencloud_actions.add_parser("rm", help="Remove the stored cookie")
    qwencloud_actions.add_parser(
        "migrate", help="Move a plaintext ticket into the encrypted secret store"
    )

    # Browser bridge command: lazy for the same reason as mobile. Ordinary CLI
    # startup must not pull Starlette/uvicorn in just to render --help.
    browser_parser = subparsers.add_parser(
        "browser", help="Connect the browser tool to a Chromium extension", parents=[parent_parser]
    )
    browser_subparsers = browser_parser.add_subparsers(dest="browser_command")
    install_browser = browser_subparsers.add_parser("install", help="Install the bridge daemon")
    install_browser.add_argument("--port", type=int, default=4099)
    status_browser = browser_subparsers.add_parser(
        "status", help="Show daemon, extension and pairing status"
    )
    status_browser.add_argument(
        "--repair",
        action="store_true",
        help="Reconcile advertised state against reality: drop tabs that no longer "
        "exist and republish a fresh heartbeat. Safe while sessions are live.",
    )
    for action, blurb in (
        ("tabs", "List durable ownership records and live tabs (read-only)"),
        ("reconcile", "Alias for 'tabs': inspect ownership without closing anything"),
    ):
        inventory_browser = browser_subparsers.add_parser(action, help=blurb)
        inventory_browser.add_argument(
            "--json", action="store_true", help="machine-readable output"
        )
    cleanup_browser = browser_subparsers.add_parser(
        "cleanup", help="Close one proven-terminal browser owner after revalidation"
    )
    cleanup_browser.add_argument("session_id")
    cleanup_browser.add_argument("--generation", required=True)
    cleanup_browser.add_argument("--yes", action="store_true", help="Approve this exact cleanup")
    for action in ("start", "stop", "restart"):
        browser_subparsers.add_parser(action, help=f"{action.capitalize()} the daemon")
    pair_browser = browser_subparsers.add_parser("pair", help="Show the extension pairing code")
    pair_browser.add_argument(
        "--reset", action="store_true", help="Revoke the paired browser first"
    )
    pair_browser.add_argument(
        "--list",
        action="store_true",
        help="List the authorised browser extensions, and which one is driving",
    )
    pair_browser.add_argument(
        "--revoke",
        metavar="ID-OR-LABEL",
        default=None,
        help="Revoke ONE authorised extension (its id, an unambiguous id prefix, "
        "or part of its label), leaving every other one paired",
    )
    drive_browser = browser_subparsers.add_parser(
        "drive", help="Choose which authorised extension drives the browser"
    )
    drive_browser.add_argument(
        "target", help="Extension id (or an unambiguous prefix), or part of its label"
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
        help=(
            "conversation-name / session-id / cwd substring (case-insensitive). "
            "Omit when addressing with --pid/--session; passing both is refused."
        ),
    )
    send_parser.add_argument(
        "message",
        nargs="?",
        help=(
            "message text; omit to read the body from stdin. With --pid/--session "
            "a single positional IS the message, and typing one while also piping "
            "a body is refused."
        ),
    )
    # Mutually exclusive because they name DIFFERENT recipients: argparse rejects
    # the pair natively ("argument --session: not allowed with argument --pid"),
    # which is a better error than anything hand-written here, and it removes a
    # whole branch from _bind_send_positionals below.
    send_selector = send_parser.add_mutually_exclusive_group()
    send_selector.add_argument("--pid", type=int, help="target by exact pid")
    send_selector.add_argument("--session", dest="session", help="target by exact session id")
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
        help=(
            "List active lop sessions and their resource usage; "
            "`sessions cleanup` previews or runs the session cleanup policy; "
            "`sessions reclaim` previews or ends runtimes nothing can reach"
        ),
        parents=[parent_parser],
    )
    sessions_parser.add_argument("--json", action="store_true", help="machine-readable output")
    sessions_parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "also list stored sessions that are not running (RSS/UPTIME/etc. are "
            "shown as — for these; sorted newest-first by last activity)"
        ),
    )
    sessions_parser.add_argument(
        "--limit",
        type=_positive_int,
        default=None,
        metavar="N",
        help="with --all, cap the stored rows listed; positive (default: 50)",
    )
    # `lop sessions cleanup`: the explicit, previewable way to run the session
    # cleanup policy. An optional sub-subcommand (dest defaults to None) so
    # bare `lop sessions` keeps listing.
    #
    # THE MASTER SWITCH GOVERNS THIS COMMAND. With `session.cleanup.enabled`
    # false the non-dry run refuses (rc 2) and says where to turn it on;
    # `--dry-run` still lists what the limits would take and says the switch
    # is off. `--force` overrides the switch, but only after printing the
    # list and reading a typed confirmation (`--yes` skips the prompt for
    # scripts). Round 1 let the bare command run past an OFF switch on the
    # theory that typing it was consent — /settings leaves the limits in the
    # file when the switch is turned off, so that was the incident's shape
    # (QA Q1, UX U2, review R1-5). Every hard guard (live claim/lease, armed
    # wake, unread mail, the 10 most recent) applies whatever the flags.
    sessions_subparsers = sessions_parser.add_subparsers(dest="sessions_command")
    cleanup_parser = sessions_subparsers.add_parser(
        "cleanup",
        help="Run the session cleanup policy now (use --dry-run to preview)",
        description=(
            "Apply session.cleanup.* from config (each limit overridable below) to the "
            "session store. Requires session.cleanup.enabled: true unless --force. "
            "Every removal is recorded in <config>/sessions/.cleanup-log.jsonl."
        ),
        parents=[parent_parser],
    )
    cleanup_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="list what would be removed and why, without removing anything",
    )
    cleanup_parser.add_argument(
        "--force",
        action="store_true",
        help="run even when session.cleanup.enabled is false (lists first, then asks)",
    )
    cleanup_parser.add_argument(
        "--yes",
        action="store_true",
        help="do not ask for confirmation before removing",
    )
    cleanup_parser.add_argument(
        "--max-sessions",
        type=_non_negative_int,
        default=None,
        help="keep only the N most recently active sessions (overrides config)",
    )
    cleanup_parser.add_argument(
        "--max-inactive-days",
        type=_non_negative_int,
        default=None,
        help="remove sessions idle longer than N days (overrides config)",
    )
    cleanup_parser.add_argument(
        "--max-total-bytes",
        type=_non_negative_int,
        default=None,
        help="trim the least recently active sessions past this store size (overrides config)",
    )
    cleanup_parser.add_argument(
        "--remove-empty",
        action="store_true",
        default=None,
        help="remove directories that never got a transcript (overrides config)",
    )
    cleanup_parser.add_argument("--json", action="store_true", help="machine-readable output")

    # `lop sessions reclaim`: the external door to the residency sweep — the
    # same pass the wake supervisor runs on its own cadence, for the case where
    # the thing an operator wants ended is not one session but the RESIDENCY
    # itself. A runtime that published no record cannot be listed here, cannot
    # be stopped with `lop stop`, and cannot be reached by any client; before
    # this command the only way to find one was `ps`. A sub-subcommand of
    # `sessions` rather than a top-level verb because it is the third question
    # about the fleet (`sessions` lists it, `send` talks to it, `reclaim`
    # bounds it) and it reads the same discovery namespaces.
    #
    # IT IS A DRY RUN UNLESS THE CALLER SAYS OTHERWISE, and the confirmation
    # that a real run asks for is not a formality: it is the same process-
    # table question the sweep asks twice before it signals anything.
    reclaim_parser = sessions_subparsers.add_parser(
        "reclaim",
        help="End session runtimes nothing can reach (dry run; --yes to act)",
        description=(
            "Find live session runtimes that no discovery record, no viewer, no "
            "attach and no existing config root can reach, and ask them to leave "
            "with SIGTERM. The runtime finishes any turn in flight first (its own "
            "signal drain, bounded by SIGNAL_DRAIN_S) — this command never sends "
            "SIGKILL. Any runtime with a record, an attached interface, a live root "
            "it does not own, or CPU spent inside the confirm window is refused."
        ),
        parents=[parent_parser],
    )
    reclaim_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="list what would be reclaimed, without signalling anything",
    )
    reclaim_parser.add_argument(
        "--yes",
        action="store_true",
        help="do not ask for confirmation before signalling",
    )
    reclaim_parser.add_argument(
        "--confirm-s",
        type=_confirm_window,
        default=None,
        metavar="SECONDS",
        help=(
            "how long to watch before signalling (default: 60, minimum: 50). The "
            "window is the safety property: a runtime that gains a record, an attach "
            "or CPU inside it is dropped from the pass, and a window too short to "
            "measure CPU would drop one of the four refusals"
        ),
    )
    reclaim_parser.add_argument("--json", action="store_true", help="machine-readable output")

    # The kill switch (design §12): end a session from outside it. Top-level
    # like `lop sessions` and `lop send` — the coherence triple is "what is
    # running / talk to it / end it" — and deliberately NOT the
    # `lop mobile start|stop|restart` shape, which manages the daemon
    # service rather than a session.
    # The activation target of a desktop notification. Deliberately a real
    # subcommand rather than `python -m …`: the toast is clicked minutes or
    # hours later, by which time the interpreter that sent it may be gone (a
    # runtime exits when its work is done, and a worktree's `.venv` is
    # disposable) — so the command has to name the user's OWN launcher, which
    # is what `lop` on PATH resolves to. Hidden from `--help`: nobody types
    # this, and it is not a supported way to open a session.
    resume_click_parser = subparsers.add_parser(
        "resume-click",
        # NO `help=` AT ALL. For a SUBPARSER, `help=argparse.SUPPRESS` is not
        # the hide idiom it is for an argument — argparse renders the sentinel
        # verbatim, so `lop --help` listed `resume-click  ==SUPPRESS==`
        # (round 3, B5). Omitting the kwarg is what keeps it off the list.
        parents=[parent_parser],
    )
    resume_click_parser.add_argument("session", help="session id to reopen")

    stop_parser = subparsers.add_parser(
        "stop",
        help="Stop a running lop session (graceful, then signals)",
        parents=[parent_parser],
    )
    stop_parser.add_argument(
        "target",
        nargs="?",
        help="conversation-name / session-id / pid / cwd substring (case-insensitive)",
    )
    stop_parser.add_argument("--pid", type=int, help="target by exact pid")
    stop_parser.add_argument("--session", dest="session", help="target by exact session id")
    stop_parser.add_argument(
        "--all",
        dest="stop_all",
        action="store_true",
        help="stop every session on this machine (prompts on a TTY; --yes to skip)",
    )
    stop_parser.add_argument(
        "--yes",
        action="store_true",
        help="skip the --all confirmation (required when stdin is not a TTY)",
    )
    stop_parser.add_argument(
        "--json",
        action="store_true",
        help="machine-readable outcome per target",
    )
    stop_parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        metavar="SECONDS",
        help="graceful-op wait per session before escalating (default 10)",
    )
    stop_parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "escalate past a refusal or a skip: signal a target whose socket "
            "cannot confirm identity, one that reports a turn in flight, or one "
            "already leaving after a signal — each of those can cut the turn it "
            "is in. Not needed for a cooperative mid-turn runtime: the plain "
            "stop ends that one promptly, through its socket"
        ),
    )

    # The rotation path (`lop refresh`): ask every live runtime to move to the
    # build on disk at its next boundary. Top-level beside `sessions`/`send`/
    # `stop` because it answers a fourth question about this machine — "which
    # of these is still on the old build, and what is it doing instead" — and
    # it exists so that making a new build take effect never needs the thing
    # that destroyed 32 turns on 2026-09-14: an ad-hoc signal sweep.
    refresh_parser = subparsers.add_parser(
        "refresh",
        help=(
            "Ask running sessions to move to the build on disk at their next "
            "boundary (no signals)"
        ),
        parents=[parent_parser],
    )
    refresh_parser.add_argument(
        "target",
        nargs="?",
        help="conversation-name / session-id / pid / cwd substring (case-insensitive)",
    )
    refresh_parser.add_argument("--pid", type=int, help="target by exact pid")
    refresh_parser.add_argument("--session", dest="session", help="target by exact session id")
    refresh_parser.add_argument(
        "--all",
        dest="refresh_all",
        action="store_true",
        help="ask every live session on this machine (no confirmation: nothing is ended)",
    )
    refresh_parser.add_argument(
        "--json", action="store_true", help="machine-readable outcome per target"
    )
    refresh_parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        metavar="SECONDS",
        help="wait per session for its answer (default 10)",
    )

    # Scheduled wakes, and the process that fires them for sessions nobody is
    # running. Top-level beside `sessions`/`send`/`stop` for the same reason
    # they are: it answers "what is scheduled and will it actually fire",
    # which is a question about this machine rather than about one session.
    wake_parser = subparsers.add_parser(
        "wake",
        help="Inspect scheduled wakes and the supervisor that fires them",
        parents=[parent_parser],
    )
    # DEFAULTS ON THE PARENT, so every route into `wake_command` carries the
    # flags it dereferences. `wake_command` falls back to "status" when no
    # subcommand is given, and the status branch reads `json`/`install`/
    # `uninstall` — so bare `lop wake` and `lop wake install` (which define no
    # such flags of their own) used to reach it and die on `args.json`. A
    # subparser that declares `--json` still overrides this, and a branch that
    # gains a new flag inherits a safe default instead of a crash.
    wake_parser.set_defaults(json=False, install=False, uninstall=False)
    wake_sub = wake_parser.add_subparsers(dest="wake_command")
    wake_status = wake_sub.add_parser(
        "status",
        help="whether a supervisor is installed, and what it would fire next",
        parents=[parent_parser],
    )
    wake_status.add_argument("--json", action="store_true", help="machine-readable output")
    wake_status.add_argument(
        "--install",
        action="store_true",
        help="install the supervisor now rather than waiting for the next schedule",
    )
    wake_status.add_argument(
        "--uninstall",
        action="store_true",
        help="remove the supervisor; scheduled wakes then fire only when a session is open",
    )
    # THE ROLLOUT PATH, and the reason this is a subcommand rather than only a
    # flag on `status`. Repair-on-demand lives in the install hook, which runs
    # on a wake PERSIST — so a machine whose supervisor is stale or stopped
    # cannot be fixed without some session happening to schedule a wake. After
    # an upgrade that is exactly the wrong dependency: the running supervisor
    # is still executing the old code, and there may be no session about to
    # persist. This command makes the repair reachable directly.
    wake_sub.add_parser(
        "install",
        help="install or repair the supervisor now (restarts a stale or stopped one)",
        parents=[parent_parser],
    )
    wake_list = wake_sub.add_parser(
        "list",
        help="every scheduled wake on this machine, soonest first",
        parents=[parent_parser],
    )
    wake_list.add_argument("--json", action="store_true", help="machine-readable output")
    # `create` completes the surface: `status` says whether wakes fire,
    # `list` says what is scheduled, and this is how a wake gets scheduled
    # from outside a session — which is also what makes install-on-demand
    # testable without driving a TUI.
    wake_create = wake_sub.add_parser(
        "create",
        help="schedule a wake for a session (installs the supervisor on demand)",
        parents=[parent_parser],
    )
    wake_create.add_argument("session", help="session id to wake")
    wake_create.add_argument(
        "when",
        help='when to fire: a duration ("in 2m", "45s") or a clock time ("at 09:30")',
    )
    # Lazy, like every other harness import in this module (see the module
    # docstring): the flag's help names the SHARED cap rather than a second
    # number that could drift from it.
    from local_operator.harness.wake import MAX_WAKE_MESSAGE_CHARS

    wake_create.add_argument(
        "message",
        # The cap is the SHARED one (``MAX_WAKE_MESSAGE_CHARS``, enforced by
        # ``build_wake_schedule``), and this command used to build the model
        # directly so it had no cap at all. Said here rather than leaving the
        # limit to be discovered by being refused (review round 1, R5).
        help=(
            "the self-prompt delivered when it fires "
            f"(at most {MAX_WAKE_MESSAGE_CHARS} characters)"
        ),
    )
    wake_create.add_argument(
        "--every",
        default="",
        metavar="DURATION",
        # The operator's stated use for background automations ("regular disk
        # cleaning, automation tasks") is recurring by nature, and the model
        # has been able to schedule one since `WakeSchedule.every_ms`; only
        # the CLI could not. Parsed by the same `parse_wake_duration` the
        # tool path uses, so `40s` / `8h30m` mean the same thing from either
        # entry point — including its deliberate rejection of a bare number
        # and of a zero interval, both of which are runaway loops.
        help='repeat every DURATION after the first fire ("5m", "1h", "8h30m")',
    )
    wake_create.add_argument(
        "--until",
        default="",
        metavar="WHEN",
        # Bounds a recurrence in TIME. `WakeSchedule.until_at` and
        # `advance_wake_schedule`'s `retired: until` have always honoured it;
        # only the CLI could not express it, so a scheduled automation could
        # only ever be unbounded (round 4, R4).
        help='stop repeating after this time ("in 7d", "at 09:30")',
    )
    wake_create.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="stop repeating after N fires",
    )
    wake_create.add_argument("--json", action="store_true", help="machine-readable output")
    wake_serve = wake_sub.add_parser(
        "serve",
        help="run the supervisor in the foreground (what the LaunchAgent runs)",
        parents=[parent_parser],
    )
    wake_serve.add_argument(
        "--once",
        action="store_true",
        help="fire whatever is due right now, then exit",
    )

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
    # Hidden (``SUPPRESS``): nobody types this, and it is not an upgrade — it is
    # the repair step ``lop update`` runs in a CHILD process from the newly
    # installed wheel, so that the LaunchAgent plists it renders come from THIS
    # build rather than from the pre-upgrade modules the parent still holds in
    # memory. Without the child, a repair renders the previous build's plist
    # shape and silently changes nothing. See ``update.daemons_refresh_command``.
    update_parser.add_argument(
        "--refresh-daemons",
        dest="refresh_daemons",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    # Install a build that is already on this machine into its own generation:
    # a source directory, or a git ref of the repository this command runs in.
    # Named separately from the PyPI path because it answers a different
    # question ("install THIS tree") and asks nothing of the network.
    update_parser.add_argument(
        "--from-snapshot",
        dest="from_snapshot",
        metavar="DIR_OR_REF",
        default=None,
        help=(
            "Install a local source tree (a directory, or a git ref of the current "
            "repository) into its own install generation, instead of upgrading from PyPI"
        ),
    )
    update_parser.add_argument(
        "--no-services",
        dest="no_services",
        action="store_true",
        help=(
            "Do not bring the `lop serve` daemons onto the new build; the supervised "
            "daemons are still repaired. Use this only when something else will "
            "start the serves, e.g. a script that owns their launch."
        ),
    )

    # The NON-RUNTIME SERVICES, as one verb group. A service is a long-lived
    # non-conversational process (a serve daemon, the mobile daemon, the browser
    # bridge, the tunnel, the wakes supervisor); a runtime is a conversation, and
    # nothing here ever touches one. `restart` is the same stage `lop update`
    # finishes with, exposed on its own so the recovery sentences the update path
    # prints can name a command that exists.
    services_parser = subparsers.add_parser(
        "services",
        help=(
            "Bring this machine's non-runtime services (serve daemons and the "
            "supervised daemons) onto the current build, without stopping any "
            "running conversation"
        ),
        # A reader who opens this page is asking what a "service" IS, and the verb
        # group defined nothing (design review D9). The distinction it draws is the
        # one the whole feature rests on: a RUNTIME is a conversation and is never
        # touched; everything else that serves this machine can be brought along.
        description=(
            "Everything local_operator runs for this machine that is not a conversation. "
            "A 'service' is a `lop serve` daemon or a supervised LaunchAgent — the "
            "mobile relay, the browser bridge, the tunnel and the wakes agent. "
            "Runtimes — the processes holding your conversations — are never stopped: "
            "'restart' reloads a serve daemon in place, keeping its pid and socket."
        ),
        parents=[parent_parser],
    )
    services_subparsers = services_parser.add_subparsers(dest="services_command")
    services_subparsers.add_parser(
        "status",
        help=(
            "Report each non-runtime service, the build it is serving, and the build "
            "the install is on"
        ),
        # D9 put the distinction on the GROUP page; a reader who runs `services status
        # --help` directly was still shown no prose at all (design review D10).
        description=(
            "Read-only. Names the build the install is on and, for each `lop serve` "
            "daemon, the build it is SERVING — both sides of every comparison, because "
            "the ordinary drift on this machine is a same-version rebuild where the "
            "version alone cannot show that a daemon is behind."
        ),
        parents=[parent_parser],
    )
    services_restart = services_subparsers.add_parser(
        "restart",
        help="Move every service onto the current build (never stops a runtime)",
        parents=[parent_parser],
    )
    services_restart.add_argument(
        "--wait",
        type=_positive_seconds,
        default=None,
        metavar="SECONDS",
        help=(
            "How long to wait for an asked daemon to come back on the new build "
            f"(default: {DEFAULT_SERVICES_WAIT_S:g}s; must be positive). A daemon that does "
            "not make it is reported, left serving the build it loaded, and retried by "
            "the next `lop services restart`."
        ),
    )

    # The install LAYOUT's own commands. One verb group rather than flags on
    # ``update`` because neither of these installs anything from a network: they
    # manage the trees this machine already has.
    #
    # ``--keep``'s default is the REAL one rather than ``None``: argparse renders
    # ``%(default)s`` from what is declared here, so a ``None`` the dispatcher
    # later converted meant the help text stated a default that was not the
    # default, and quoted a Python symbol an operator cannot act on (design
    # review D5).
    from local_operator.update import DEFAULT_KEEP_GENERATIONS

    def _generation_count(value: str) -> int:
        """``--keep``'s value: a non-negative count, or a clean argparse refusal.

        REFUSING A NEGATIVE IS THE POINT (design review round 2, D15): ``-1`` is a
        plausible typo for ``1``, and the old code silently read it as "keep
        nothing" — the reading that deletes the most whole venvs, on the one
        command whose entire job is deleting them. argparse renders this as exit 2
        with a usage line, the same shape as any other bad option.

        ``int`` RAISES ``ValueError`` FOR ANYTHING NON-NUMERIC and this function
        let it through, which is why ``--keep foo`` printed ``invalid
        _generation_count value: 'foo'`` — this function's Python name on the
        operator's screen, on a surface the generation PR had just cleaned of
        exactly that (R7-2). Caught here rather than left to argparse because
        argparse's message is derived from the function it was handed and cannot be
        given a better one; ``TypeError`` cannot arrive, since argparse passes the
        command line's own ``str``.
        """
        try:
            count = int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"expected a whole number, got {value!r}") from None
        if count < 0:
            raise argparse.ArgumentTypeError(f"expected 0 or more, got {count}")
        return count

    install_parser = subparsers.add_parser(
        "install",
        help="Manage the local install generations (layout for the stable `lop` runtime)",
        parents=[parent_parser],
    )
    install_subparsers = install_parser.add_subparsers(dest="install_command")
    prune_parser = install_subparsers.add_parser(
        "prune",
        help="Delete install generations nothing is running from",
        parents=[parent_parser],
    )
    prune_parser.add_argument(
        "--keep",
        type=_generation_count,
        default=DEFAULT_KEEP_GENERATIONS,
        metavar="N",
        help=(
            "How many unreferenced generations to keep (default: %(default)s; 0 or "
            "more). A generation named by a live or saved session, and the one "
            "`current` points at, are never removed"
        ),
    )
    install_subparsers.add_parser(
        "migrate",
        help="Copy this install into the generation layout and point `current` at it",
        parents=[parent_parser],
    )
    install_subparsers.add_parser(
        "status",
        help="Show the install layout: pointer, generations and what a new `lop` would load",
        parents=[parent_parser],
    )

    exec_parser = subparsers.add_parser(
        "exec",
        help="Execute a task or goal loop in a persisted session without starting the TUI",
        parents=[parent_parser],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  lop exec 'Review the change' --team release --background\n"
            "  printf 'Inspect this report' | lop exec --profile reviewer\n"
            "  lop exec --goal 'Finish the checklist' --loop 3 --name 'Night audit'\n"
            "  lop exec --resume SESSION_ID --loop-goal 'Verify every acceptance criterion'\n"
            "  lop exec --status JOB_ID\n\n"
            "Resume restores the transcript, team, profile, goal and name first; explicit\n"
            "startup flags override their own slots. Team and profile may coexist. A loop\n"
            "runs after the optional initial prompt; --loop N counts continuations only.\n"
            "Saved loop progress is visible on resume, but iterations never replay automatically.\n"
            "A goal alone does not run a model. Prompts are literal, not slash commands.\n"
            "--goal only SETS the objective; unlike the TUI's /goal it does not also\n"
            "send that text as a message, because the prompt is exec's message channel.\n"
            "--agent/--agent-id select legacy agent data and are mutually exclusive.\n"
            "Default non-TTY approvals deny. --control may wait for a supervisor; --yolo\n"
            "is an explicit override, never implied by --background or --team.\n"
            "--tools bounds what the run can reach: excluded tools are unreachable.\n"
            "Foreground events/text use stdout; receipts use stderr. Detached runs log\n"
            "both streams and print distinct job/session IDs. Use lop --resume SESSION_ID\n"
            "to view a live run or resume a finished one; exec refuses a live owner."
        ),
    )
    exec_parser.add_argument(
        "command",
        type=str,
        nargs="?",
        default=None,
        help="Literal prompt; '-' or omitted with piped stdin reads stdin; optional for a loop",
    )
    from local_operator.exec_startup import add_startup_arguments

    add_startup_arguments(exec_parser)
    exec_parser.add_argument(
        "--status",
        metavar="JOB_ID",
        help="Read a durable background-job status as JSON; does not start a session",
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
    exec_parser.add_argument(
        "--control",
        action="store_true",
        help=(
            "Route approvals and questions to an attached supervisor (may wait). "
            "Without this flag non-TTY approvals deny; discovery and live attachment "
            "are available either way. --yolo remains an explicit approval override."
        ),
    )
    exec_parser.add_argument(
        "--supervisor-fd",
        type=int,
        dest="supervisor_fd",
        default=None,
        help=(
            "Write this run's operator capability to this inherited descriptor so the "
            "supervisor holding the other end can APPROVE the cards this run parks "
            "(stage E). Requires --control; refused with --background. Descriptor "
            "numbers only — the value never touches argv, the environment or a file."
        ),
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
            # The run-shaping trio, for the same reason and by the same
            # mechanism. An external supervisor driving `lop exec` composes its
            # argv programmatically and naturally writes the flags AFTER the
            # subcommand — `lop exec - --json --model X` — which failed with
            # "unrecognized arguments" and an exit 2 that reads like a bad
            # install rather than a word-order rule. They select which model
            # answers and where it runs, so silently ignoring them would be
            # worse than the parse error: the run would proceed against the
            # wrong model, in the wrong directory.
            subparser.add_argument(
                "--hosting",
                type=str,
                default=argparse.SUPPRESS,
                dest="hosting",
                help="Hosting platform for this run (e.g. anthropic, openai, openrouter)",
            )
            subparser.add_argument(
                "--model",
                type=str,
                default=argparse.SUPPRESS,
                dest="model",
                help="Model to use for this run",
            )
            subparser.add_argument(
                "--run-in",
                type=str,
                default=argparse.SUPPRESS,
                dest="run_in",
                help="Working directory to run the operator in",
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
    from local_operator.providers.key_prompt import prompt_for_provider_key
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

    # ``prompt_for_provider_key`` writes a provider-class STORE row and creates
    # nothing: the prompt moved out of the deleted ``CredentialManager`` (PR2b),
    # whose construction used to recreate the plaintext ``credentials.env`` this
    # consolidation retires — on a host that had already migrated and deleted the
    # file (R5). Nothing this command does needs the file.
    try:
        prompt_for_provider_key(args.key, reason="update requested")
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
    """Remove a credential from the provider-class store namespace.

    Deletes the ``LOP_PROVIDER_<KEY>`` store row. The plaintext file is left
    untouched — a value that only ever lived there still has a reader during the
    transition — but a store row the modern writer created is what this command
    is expected to remove.
    """
    from local_operator.providers.registry import remove_provider_key

    remove_provider_key(args.key)
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
        # An ENUM's displayed LABEL is not always its stored VALUE (D11).
        # `model_effort auto` means the stored ``""``, and the guessed parse
        # below would hand the literal string ``auto`` to ``validate`` and have
        # it rejected — a documented choice unreachable from the documented
        # command. A label is matched case-insensitively and, when it matches,
        # wins OUTRIGHT: the chain below is skipped, which matters because it
        # converts the literal word ``none`` to Python ``None`` and ``none`` is
        # a real rung of ``EFFORT_ORDER``. Values matching no label fall through
        # to the existing parse unchanged, so no other kind's behaviour moves.
        matched_choice: settings_io.Choice | None = None
        if setting.kind is settings_io.Kind.ENUM:
            typed = str(value).strip().lower()
            matched_choice = next(
                (
                    choice
                    for choice in setting.resolved_choices
                    if str(choice.label).strip().lower() == typed
                ),
                None,
            )
        if matched_choice is not None:
            value = matched_choice.value
        else:
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

        # Report what was STORED, not what was typed. `write_setting` may
        # normalize (a hotkey's `CTRL+G` is stored as `ctrl+g`), and echoing
        # the raw input would tell the user their config holds a spelling it
        # does not — the same class of lie as a page displaying a key the
        # runtime never bound.
        #
        # An ENUM echoes the LABEL the typed word selected, when it selected one
        # (D8). The stored form is a wire value, not vocabulary: ``model_effort
        # auto`` stores ``""``, so the receipt read "Successfully updated
        # model_effort to " — the user typed a word and the confirmation named
        # nothing. The same blank met every other member whose value is empty
        # (``providers.openrouter.* default``), and the label also restores the
        # words for the members whose value is not their label at all
        # (``display.nerd_icons auto`` stores ``None`` and used to echo
        # ``None``). Values that matched no label fall through unchanged, so the
        # echo of a normalised value is exactly what it was.
        stored = settings_io.read_setting(config_manager, setting)
        echoed = matched_choice.label if matched_choice is not None else stored
        print(f"Successfully updated {args.key} to {echoed}")
        if setting.key == "tool_approval_mode" and str(stored).strip().lower() == "auto":
            # Qualified on purpose (UX round 1, U4). "Successfully updated
            # tool_approval_mode to auto" reads as "my running agents are not
            # gated any more", and since #1282 that is false for every session
            # already running: a loosening is authorised only in the process
            # that holds the gate (``harness.approval.loosening_is_authorised``),
            # and this command's process holds none. The TIGHTENING direction
            # says nothing extra: it really does reach every running session,
            # and is the safe direction besides.
            print(
                "Running sessions are unchanged — a config write cannot loosen one; "
                "type /approvals auto in each session you want ungated. "
                "New sessions open at auto."
            )
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
        # The ONE entry here whose key the schema DOES carry (a READONLY row in the
        # retired section), and it is here for this table's own grammar: the three
        # keys above spell their retirement as a bracketed tag, and without one
        # `classification.notice: True` was the only row in the family a scanning
        # reader could not see was retired (design round 1, D3). Restating the
        # sentence the registry carries is deliberate — the alternative, teaching
        # this loop to synthesise a tag from `section`, would double-tag the three
        # rows above, which already carry theirs in their own text.
        "classification.notice": "[DEPRECATED] Smart hints is silent in the chat; "
        "the call and its cost are in the session log",
        "auto_save_conversation": "Whether to automatically save conversations",
        "compaction": "Compaction engine settings (enabled, strategy, thresholds); "
        "replaces conversation_length/detail_length",
        "tui": "TUI settings (theme)",
        "session": "Session settings; session.cleanup.* is the opt-in cleanup policy "
        "(off by default: nothing removes a session directory unless enabled)",
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


def config_instructions_command(args: argparse.Namespace) -> int:
    """Report which instruction files a session would actually assemble.

    Answers "what instructions am I running", which before this command was
    only recoverable by importing ``ecosystem_instructions`` by hand or by
    reading an INFO log line the TUI writes to a rotating file (#822). The
    imported ``~/.agents/AGENTS.md`` is the case that needs it: it is written
    by another tool, named in no lop-owned setting, and silently reshapes the
    system prompt of every session and every subagent.

    Deliberately PROVENANCE and not content. These files are the operator's
    standing rules and routinely run to thousands of lines; dumping them here
    would bury the one thing being asked about (which files, in what order, at
    what cost) and make the command unusable in a pipe. ``config open`` and an
    editor already show the text.

    Read-only by construction: the whole report is derived from
    ``resolve_user_instructions``, which is the same call the session factory
    makes, so this command cannot report a prompt different from the one that
    ships \u2014 the divergence #822 is about.
    """
    # ``paint`` rather than the raw ``\033[1;32m`` literals ``config_list_command``
    # still carries: this output is routinely piped into an issue or a review
    # comment, and cli_style is the module that keeps the escapes out of a
    # non-tty capture and honours NO_COLOR.
    from local_operator.cli_style import CYAN, ERROR, SUCCESS, WARNING, paint
    from local_operator.ecosystem_instructions import (
        ECOSYSTEM_INSTRUCTION_PATHS,
        ECOSYSTEM_INSTRUCTIONS_ENV,
    )
    from local_operator.session_factory import (
        MAX_USER_INSTRUCTIONS_CHARS,
        resolve_user_instructions,
    )

    # The profile prompt is resolved the way a session resolves it, so
    # ``--agent NAME`` reports what that agent would actually run rather than
    # the no-profile case. A name with no agent behind it is NOT created here:
    # this command must never write, and the interactive path's
    # create-on-missing behaviour would do exactly that.
    agent_prompt = ""
    agent_note = ""
    agent_name = getattr(args, "agent_name", None)
    if agent_name:
        from local_operator.agents import AgentRegistry

        # Guarded because ``AgentRegistry.__init__`` mkdirs both ``config_dir``
        # and ``config_dir/agents`` — so on a machine with no config root at
        # all, asking this read-only command about an agent materialised one.
        # Read-only was a stated deliverable, and there is nothing to report
        # anyway: a config dir that does not exist holds no agents.
        root = config_dir()
        if not root.exists():
            print(
                paint(f"Error: No agent found with name: {agent_name}", ERROR, stream=sys.stderr),
                file=sys.stderr,
            )
            return 1
        registry = AgentRegistry(root)
        agent = registry.get_agent_by_name(agent_name)
        if agent is None:
            print(
                paint(f"Error: No agent found with name: {agent_name}", ERROR, stream=sys.stderr),
                file=sys.stderr,
            )
            return 1
        agent_prompt = registry.get_agent_system_prompt(str(agent.id)) or ""
        agent_note = agent_name

    assembled, sources = resolve_user_instructions(agent_prompt, log_provenance=False)

    override = os.environ.get(ECOSYSTEM_INSTRUCTIONS_ENV)
    # Both counts right-aligned in one field so the ``Included`` column does not
    # move between rows: comparing ``Read`` against ``Included``, and rows
    # against each other, is the whole reason both numbers are printed, and a
    # column that shifts with the digit count defeats scanning down it. Sized
    # from the data rather than a constant so the common small-number case
    # stays tight.
    count_width = max(
        (len(f"{value:,}") for source in sources for value in (source.chars, source.included)),
        default=1,
    )
    print(paint("\n╭─ Custom instructions, in assembly order ──────", SUCCESS))
    for index, source in enumerate(sources, start=1):
        label = source.label
        if label == "agent profile" and agent_note:
            label = f"agent profile ({agent_note})"
        print(paint(f"│ {index}. {label}", SUCCESS))
        print(paint(f"│    Path: {source.path if source.path else '(agent registry)'}", SUCCESS))
        # Both numbers, always: "read 6,960 / included 0" is the collapse an
        # operator is trying to confirm, and a single figure cannot express it.
        print(
            paint(
                f"│    Read: {source.chars:>{count_width},} chars"
                f"   Included: {source.included:>{count_width},} chars",
                SUCCESS,
            )
        )
        if source.unreadable:
            # Degraded, not absent: the session started anyway with this file's
            # rules missing, which is a state the operator has to be told about
            # rather than left to read as "no such file".
            print(paint("│    Unreadable: skipped; the session ran without it", WARNING))
        elif source.chars == 0:
            # A path with no bytes behind it reads as a file that exists and is
            # being used. Saying so keeps the default install (no
            # system_prompt.md) from looking like a configured-but-broken one.
            #
            # Imported rows state the stronger fact: the loader only lists paths
            # that resolve to a real file, so an imported row with zero
            # characters is a file that EXISTS and is blank — not a path that
            # might be empty. Reporting the disjunction there would repeat, one
            # level down, the "no imported file exists" claim about a path that
            # has one.
            #
            # CYAN, not SUCCESS: every other line in the box — chrome, headers,
            # counts — is green, so a green annotation carries no signal at all,
            # and this is the row the DEFAULT install shows. The ladder is green
            # = normal contribution, cyan = present but contributed nothing,
            # yellow = degraded.
            empty_note = (
                "Empty: the file is there but holds no instructions"
                if source.label == "imported"
                else "Empty: no file at that path, or nothing in it"
            )
            print(paint(f"│    {empty_note}", CYAN))
        if source.collapsed:
            print(
                paint(
                    "│    Collapsed: identical to instructions already loaded; sent once",
                    CYAN,
                )
            )
        if source.truncated:
            print(paint("│    Truncated: hit the size cap; the tail was dropped", WARNING))
        if source.overlaps_index:
            # The superset arrangement, which the digest collapse cannot catch
            # and which two rows of non-zero "Included" cannot distinguish from
            # two genuinely distinct files. WARNING, matching ``Truncated:``:
            # this is a cost the operator is paying on every cached request and
            # can remove. The overlapping TEXT is never printed — the count and
            # the other source's row number are the whole answer.
            #
            # Kept within 80 columns in every reachable state: 79 for a
            # two-digit row number at 64,000 characters, and still 80 at three
            # digits, which a source list bounded by the override paths plus two
            # cannot reach. The box has no wrapping of its own, so a longer row
            # soft-wraps in an 80-column terminal and the overflow lands outside
            # the "│" gutter — the same reason the footer below is two lines
            # rather than one. The source is named by ROW NUMBER, not label:
            # several imported files all render as "imported", so a label could
            # not identify which file to edit.
            direction = (
                f"contains source {source.overlaps_index} verbatim"
                if source.overlap_contains
                else f"verbatim inside source {source.overlaps_index}"
            )
            print(
                paint(
                    f"│    Overlaps: {direction} "
                    f"({source.overlap_chars:,} chars); both copies are sent",
                    WARNING,
                )
            )
    if not sources:
        print(paint("│ (none — no instruction sources resolved)", SUCCESS))
    # The recurrence, not just the headroom: without it the counts read as file
    # sizes rather than as a cost paid again on every request, which is the
    # framing the guide uses and the reason the numbers are worth showing.
    print(
        paint(
            f"│ Total assembled: {len(assembled):,} of {MAX_USER_INSTRUCTIONS_CHARS:,} chars",
            SUCCESS,
        )
    )
    print(
        paint(
            "│ Re-sent with every request, in every session and subagent",
            SUCCESS,
        )
    )
    print(paint("╰──────────────────────────────────────────────", SUCCESS))

    # The imported half gets its own box because its state is the question:
    # "none" here is a real answer (the default install has no
    # ~/.agents/AGENTS.md) and must not read as an empty listing, and the
    # override has three distinct states an operator can be in by accident.
    print(paint("\n╭─ Imported user-scope instructions ────────────", SUCCESS))
    if override is None:
        default_paths = ", ".join(f"~/{relative}" for relative in ECOSYSTEM_INSTRUCTION_PATHS)
        print(paint(f"│ Source: default ({default_paths})", SUCCESS))
    elif not override.strip():
        print(
            paint(
                f"│ Source: DISABLED — {ECOSYSTEM_INSTRUCTIONS_ENV} is set and empty",
                WARNING,
            )
        )
    else:
        print(paint(f"│ Source: {ECOSYSTEM_INSTRUCTIONS_ENV}={override}", CYAN))
    imported = [source for source in sources if source.label == "imported"]
    if not imported:
        # "Nothing was imported" has two causes with opposite fixes — the
        # feature is off, or it is on and the file is simply not there — and an
        # operator who reads the wrong one goes looking in the wrong place.
        reason = (
            "the feature is disabled"
            if override is not None and not override.strip()
            else "no imported file exists at those paths"
        )
        print(paint(f"│ Files read: none — {reason}", SUCCESS))
    else:
        # "Files read" is plural and reads as a heading, so repeating it per row
        # made a two-file install read as two separate answers to one question.
        # Printed once with the entries indented beneath, matching the
        # four-space continuation the first box already uses — but only in the
        # multi-row case, since a lone "Files read: <path>" is correct as one
        # line and splitting it would cost a row for nothing.
        if len(imported) > 1:
            print(paint("│ Files read:", SUCCESS))
        for source in imported:
            if source.unreadable:
                state = "unreadable; skipped"
            elif source.collapsed:
                state = "collapsed"
            elif source.chars == 0:
                state = "empty"
            else:
                state = f"{source.included:,} chars"
            prefix = "│    " if len(imported) > 1 else "│ Files read: "
            print(paint(f"{prefix}{source.path} ({state})", SUCCESS))
    print(
        paint(
            "│ Read-only: system_prompt.md stays the only file lop writes",
            SUCCESS,
        )
    )
    print(paint("╰──────────────────────────────────────────────", SUCCESS))
    return 0


#: The extension build that first ACTS on the `role` frame (the standby card and
#: the ability to let go of one's own tabs). An install older than this keeps its
#: debugger attachments when told it is a standby, so the operator sees
#: "Local Operator is debugging this browser" bars nothing but that build can
#: release — the one rollout cost no daemon-side fix can remove. Used to gate the
#: `note:` line on the STANDBY's recorded build rather than printing it for any
#: standby at all (copy review C4).
_ROLE_AWARE_EXTENSION_VERSION = (0, 1, 13)


def _extension_version_in(label: str) -> tuple[int, ...]:
    """The version inside a pairing label (`Chrome extension 0.1.13`), or (0,).

    Labels are produced by `_browser_label` and always end in the extension
    version when the peer reported one; an entry paired before labels carried a
    version (or by a peer that sent none) parses to (0,), which compares below
    every real version — the conservative direction, since a build we cannot
    identify might be any age.
    """
    match = re.search(r"(\d+(?:\.\d+)*)\s*$", label)
    if not match:
        return (0,)
    return tuple(int(part) for part in match.group(1).split("."))


def _seen_line(entry: dict[str, Any]) -> str:
    """ "paired <when>, last seen <when>" for one identity record.

    Design §8.1 asks `pair --list` to carry both, and they are the only fields
    that still tell two installs apart when every other one collides (UX U2).
    Epoch floats from the pairing file. A half that is absent or meaningless is
    OMITTED rather than rendered as a 1970 date (review round 4, finding 2: a
    schema-1 record, whose timestamps `pairing_status` coerces to 0.0, printed
    "last seen 20709d ago"), and a record that carries neither says so in words.
    """
    parts = []
    paired = _ago(entry.get("paired_at"))
    seen = _ago(entry.get("last_seen_at"))
    if paired:
        parts.append(f"paired {paired}")
    if seen:
        parts.append(f"last seen {seen}")
    return ", ".join(parts) if parts else "no timestamps on this record"


#: A Unix timestamp below this is not a date this file could plausibly contain —
#: the project is younger than the epoch, and `pairing_status` coerces a missing
#: field to 0.0. Used to refuse the 1970 rendering rather than to be clever about
#: calendars.
_PLAUSIBLE_EPOCH_FLOOR = 1_500_000_000.0


def _ago(stamp: object) -> str:
    """A compact "how long ago" for a unix timestamp, or "" when unknowable."""
    try:
        value = float(stamp)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ""
    if value < _PLAUSIBLE_EPOCH_FLOOR:
        return ""
    seconds = max(0.0, time.time() - value)
    if seconds < 90:
        return f"{int(seconds)}s ago"
    if seconds < 5400:
        return f"{int(seconds // 60)}m ago"
    if seconds < 172800:
        return f"{int(seconds // 3600)}h ago"
    return f"{int(seconds // 86400)}d ago"


def _waiting_label(item: dict[str, Any]) -> str:
    """The `(label · short id)` suffix that identifies one waiting install.

    Both halves matter (copy review C6): the label names the browser, and the id
    prefix is the only part that stays unique when two installs run the same
    build and their labels are byte-identical. Falls back to whichever half the
    record has rather than printing an empty pair of brackets.
    """
    extension_id = str(item.get("extension_id", ""))
    label = str(item.get("label", "")) or "unnamed install"
    if not extension_id:
        return f"({label})"
    return f"({label} · {_short_extension_id(extension_id)})"


def _short_extension_id(extension_id: str) -> str:
    """The 8-character prefix of an extension id, with an ellipsis.

    Enough to tell two coexisting installs apart in a terminal line, which is
    all a human ever needs one for. The full 32 characters are still accepted by
    `--revoke` and `drive`, and so is THIS printed form: both resolvers strip a
    trailing ellipsis before matching (`normalise_target`), because copying what
    the screen shows is the most likely user action (UX round 2, U8).
    """
    return f"{extension_id[:8]}…" if len(extension_id) > 8 else extension_id


def _resolve_pairing_target(target: str, identities: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Resolve an id, an id PREFIX, or a label substring against the authorised set.

    Ambiguity resolves to None and the caller prints the candidates, because
    revoking the wrong install is not a mistake this command gets to make
    silently — and neither is handing the wheel to the wrong browser. An exact
    id always wins, so a label that happens to contain another install's id
    cannot shadow it.
    """
    # Imported here rather than at module scope: `lop` must not pay the daemon's
    # import cost to print a list, and this is the only module-level helper that
    # needs the daemon's normaliser. Same echo of `_short_extension_id`'s contract
    # as the daemon-side resolver, so a printed handle resolves on both sides.
    from local_operator.browser_bridge.daemon import normalise_target

    wanted = normalise_target(target).lower()
    if not wanted:
        return None
    for entry in identities:
        if str(entry.get("extension_id", "")).lower() == wanted:
            return entry
    prefixed = [
        entry
        for entry in identities
        if str(entry.get("extension_id", "")).lower().startswith(wanted)
    ]
    if len(prefixed) == 1:
        return prefixed[0]
    labelled = [entry for entry in identities if wanted in str(entry.get("label", "")).lower()]
    return labelled[0] if len(labelled) == 1 else None


def _print_identities(
    pairing: dict[str, Any], health: dict[str, Any] | None, *, verbose: bool = False
) -> None:
    """Print which extensions are authorised, and which one has the wheel.

    The role comes from the live daemon (`/health`) when it answers, because
    only the daemon knows which socket is driving; the FILE can only say who is
    authorised. With no daemon running the list is still printed, without roles,
    rather than implying a connection state nobody observed.

    ``verbose`` adds when each install was paired and last seen (design §8.1's
    `pair --list`), which is the only field left that can tell two installs apart
    once their labels collide — two profiles of one unpacked build, or any two
    builds at one version. `status` keeps the compact form.
    """
    identities = pairing.get("identities") or []
    if not identities:
        return
    driver = str((health or {}).get("driver_extension_id") or "")
    standby = {str(item) for item in (health or {}).get("standby_extension_ids") or []}
    count = len(identities)
    print(f"identities:          {count} authorised")
    for entry in identities:
        extension_id = str(entry.get("extension_id", ""))
        label = str(entry.get("label", "")) or "unnamed install"
        # The timestamps come from the FILE, so they print whether or not a daemon
        # answers (review round 4, finding 2): skipping them with `health is None`
        # hid the one field that distinguishes two identically-labelled installs
        # in exactly the state where a daemon is not running to name the roles.
        if health is None:
            print(f"                     - {label} ({_short_extension_id(extension_id)})")
            if verbose:
                print(f"                       {_seen_line(entry)}")
            continue
        if extension_id == driver:
            role = "driving"
        elif extension_id in standby:
            role = "standby"
        else:
            role = "paired, not connected"
        print(f"                     - {label} ({_short_extension_id(extension_id)}) {role}")
        if verbose:
            print(f"                       {_seen_line(entry)}")
    if len(identities) > 1:
        # The lever for "the wrong one is driving" (UX U7): with two installs up
        # this panel is exactly where the operator notices, and the command that
        # fixes it was discoverable only from `--help` or the docs. No
        # parenthetical about approvals any more — the line below says it
        # (copy review C13: the tip restated it word for word).
        print(
            "                     tip: 'lop browser drive <id|label>' chooses which"
            " install drives."
        )
        # The per-install nature of approvals is the one thing a handover can cost
        # that the user cannot see anywhere else (UX U6), so it belongs in this
        # panel whenever the wheel can move — not only while a standby happens to
        # be ATTACHED, which is how it was gated and why it went missing right
        # after a wheel move that left the demoted install merely authorised (UX
        # round 2's residual). Two or more authorised installs is the condition
        # the disclosure is for.
        print(
            "                     approvals:           per install; they do not follow the"
            " browser that takes over."
        )
    if standby:
        # The rollout cost, stated where it is felt (design §10 risk 2, review
        # round 1 m3): an install whose build PREDATES the role event cannot act
        # on being told `standby`, so it keeps its debugger attachments and
        # leaves "Local Operator is debugging this browser" bars on tabs only it
        # can release.
        #
        # Gated on the STANDBY's own build, from the live labels `standby_labels`
        # carries (copy review C4 / UX U7): the note used to print for any
        # standby at all — including a pair where both installs are current —
        # leaving the operator to evaluate a version condition the daemon already
        # knew. The symptom is named too, because otherwise the reader cannot
        # connect the note to the bars on their screen.
        labels = (health or {}).get("standby_labels")
        if labels is None:
            # A daemon from this branch's own earlier build lists standbys but not
            # their builds, so the condition cannot be evaluated at all: say the
            # version-unknown form rather than a pre-0.1.13 claim nobody checked.
            print(
                "note:                if a standby stops driving, close that browser's tabs or"
                " remove that build, or 'Local Operator is debugging this browser' bars"
                " stay on them"
            )
        else:
            stale = [
                str(label)
                for label in labels
                if _extension_version_in(str(label)) < _ROLE_AWARE_EXTENSION_VERSION
            ]
            if stale:
                print(
                    "note:                the standby build predates 0.1.13 and cannot release"
                    " its own tabs: if it stops driving, close that browser's tabs or remove"
                    " that build, or 'Local Operator is debugging this browser' bars stay on"
                    " them"
                )


def browser_command(args: argparse.Namespace) -> int:
    """Dispatch ``lop browser …`` without importing the daemon at CLI startup."""
    command = getattr(args, "browser_command", None)
    if command in ("tabs", "reconcile", "cleanup"):
        import asyncio
        import json
        import textwrap

        from local_operator.browser_bridge import install as browser_install
        from local_operator.browser_bridge import state as browser_state
        from local_operator.browser_bridge.resources import (
            cleanup_exact,
            read_inventory,
        )

        sessions = config_dir() / "sessions"
        if command == "cleanup":
            if not args.yes:
                print("No changes. Repeat with --yes to approve this exact session and generation.")
                return 1
            if Path(args.session_id).name != args.session_id or args.session_id in (".", ".."):
                print("Invalid session id; no action taken.")
                return 1
            try:
                result = asyncio.run(cleanup_exact(sessions / args.session_id, args.generation))
            except Exception as exc:
                # The bridge and lease errors already carry operator-grade
                # sentences naming the command that fixes them; printing the
                # class name instead threw that away and read as a truncated
                # message. Keep the class name only for genuinely unexpected types.
                detail = str(exc) or type(exc).__name__
                print(f"Cleanup blocked: {detail} Nothing was closed.")
                return 1
            if result.state == "closed":
                message = "Browser cleanup: closed. The tab was closed and its record settled."
            elif result.state == "retained":
                message = f"Browser cleanup: retained — {result.detail or 'the tab is held open'}."
            elif result.state == "pending":
                # The generation is deliberately NOT rotated by a failed attempt
                # any more, so the value they already copied stays correct.
                message = (
                    f"Browser cleanup: pending — nothing was closed ({result.detail}). "
                    "Retry the same command once the bridge answers; "
                    "the generation you copied is still current."
                )
            else:
                message = f"Browser cleanup: no action taken — {result.detail}."
            print(textwrap.fill(message, width=78, subsequent_indent="  "))
            return 0 if result.state in ("closed", "retained") else 1
        rows = read_inventory(sessions)
        # A command called `tabs` must reconcile with the browser and with
        # `status`: reporting "no records" while the bridge drives seven tabs
        # renders as its own opposite in EXACTLY the pool-exhausted state an
        # operator reaches for it in, and `{"resources": []}` reads to any
        # script as an empty browser. Unowned tabs are listed, never selectable.
        live_tabs: list[dict[str, Any]] = []
        current = browser_state.read()
        # Resolve the port exactly as the sibling `status` does. `state.read()`
        # returns None for a MISSING OR CORRUPT discovery file as much as for
        # an absent daemon, so skipping the probe on None made a healthy bridge
        # driving five tabs render as an empty browser — `status` found them at
        # the default port in the same state, which is the divergence this
        # command was added to remove.
        probe = browser_install.health(current.port if current else browser_install.DEFAULT_PORT)
        driven = (probe or {}).get("driven_tabs")
        # Read-only and never fatal, but UNKNOWN IS NOT ZERO: a probe that did
        # not answer means the live half is unknowable, and reporting that as
        # "no tabs" tells the operator the opposite of the truth in exactly the
        # wedged-bridge state that sent them here.
        live_known = isinstance(driven, list)
        if live_known:
            live_tabs = [entry for entry in driven if isinstance(entry, dict)]
        # Every durable record is redacted (no capability, no tab handle), so a
        # record cannot be matched to a live URL here. The honest framing is a
        # count of records against a count of live tabs, with the live ones
        # listed as unattributed rather than falsely claimed by a session.
        unowned = live_tabs
        if args.json:
            print(
                json.dumps(
                    {
                        "resources": rows,
                        "live_tabs": [
                            {"url": str(tab.get("url", "")), "title": str(tab.get("title", ""))}
                            for tab in unowned
                        ],
                        # Null, not 0, when the bridge did not answer: a script
                        # must not be able to read an unreachable bridge as an
                        # empty browser. The key is `live_tabs` to match the
                        # rendered heading — records are redacted, so a listed
                        # tab can only be described as live, never proven
                        # unowned.
                        "live_tabs_known": live_known,
                        "live_tab_count": len(live_tabs) if live_known else None,
                        "mode": "read-only",
                    },
                    indent=2,
                )
            )
            return 0
        if not rows and not live_tabs:
            print(
                "No browser tabs and no ownership records."
                if live_known
                else textwrap.fill(
                    "No ownership records. The bridge did not answer, so open "
                    "tabs are unknown — run 'lop browser status' to check the "
                    "daemon.",
                    width=78,
                )
            )
            return 0
        if rows:
            # Pad the state so the third column starts at one offset: at real
            # id and token widths an unpadded row loses its columns entirely,
            # and the generation is moved to its own indented line so an
            # 80-column terminal cannot wrap it mid-token — a wrapped token's
            # continuation sits in column 1, looks like a new record, and is
            # unsafe to copy, which is exactly what cleanup asks you to do.
            width = max(len(str(row.get("state", ""))) for row in rows)
            for row in rows:
                mark = "  <- cleanup candidate" if row.get("cleanup_candidate") else ""
                # rstrip so an unmarked row carries no trailing padding.
                print(f"{row['session_id']}  {str(row.get('state', '')):<{width}}{mark}".rstrip())
                print(f"  generation={row.get('generation', '')}")
                # Wrapped on the same discipline as the reason below it: a
                # retention reason is free text (it can quote a whole approval
                # URL), so an unwrapped line here ran to 141 columns beside
                # neighbours that wrap at 76.
                for line in textwrap.wrap(
                    f"terminal={row.get('terminal') or 'not established'}; "
                    f"retention={row.get('retention') or 'none recorded'}",
                    width=76,
                    initial_indent="  ",
                    subsequent_indent="    ",
                    break_long_words=False,
                ):
                    print(line)
                if row.get("ownership") == "unavailable":
                    # The record's own statement that the connected extension had
                    # no `owner_*` lifecycle, so this scope was driven
                    # capability-only. Printed because "it worked in legacy mode"
                    # and "its ownership is proven" need opposite next steps when
                    # a tab is stranded, and the two are otherwise identical on
                    # this screen. Redacted by construction: the marker names the
                    # MODE, never a capability.
                    for line in textwrap.wrap(
                        "ownership: unavailable — driven in legacy mode (the connected "
                        "browser extension has no ownership lifecycle); no tab was "
                        "reconciled or adopted",
                        width=76,
                        initial_indent="  ",
                        subsequent_indent="    ",
                        break_long_words=False,
                    ):
                        print(line)
                if not row.get("cleanup_candidate") and row.get("blocked_reason"):
                    # Wrapped: the reasons name a recovery command, and a line
                    # running past the terminal width is where that command
                    # would be broken across a wrap and mis-copied.
                    for line in textwrap.wrap(
                        f"not cleanable: {row['blocked_reason']}",
                        width=76,
                        initial_indent="  ",
                        subsequent_indent="    ",
                        break_long_words=False,
                    ):
                        print(line)
        if unowned:
            if not rows:
                print(
                    textwrap.fill(
                        f"No durable ownership records. {len(unowned)} tab(s) are open with "
                        "no proven owner — listed below; Local Operator will not close them.",
                        width=78,
                    )
                )
            print(f"\nlive tabs ({len(unowned)}, not selectable for cleanup):")
            for tab in unowned:
                print(f"  - {tab.get('url', '')}")
            print(
                "  Handles are redacted, so these cannot be attributed to a record above."
                "\n  Close an unwanted tab by hand; provenance is unproven."
            )
        elif not live_known and rows:
            # The records rendered above are only half the answer, and silence
            # here would read as "the browser is empty" beside them.
            print()
            print(
                textwrap.fill(
                    "Live tabs: unknown — the bridge did not answer, so tabs open "
                    "outside these records could not be listed. Run 'lop browser "
                    "status' to check the daemon.",
                    width=78,
                )
            )
        print("\nRead-only. PID absence, age, and localhost URLs never authorize cleanup.")
        if any(row.get("cleanup_candidate") for row in rows):
            print("Exact cleanup: lop browser cleanup SESSION --generation GENERATION --yes")
        return 0
    if command == "serve":
        from local_operator.browser_bridge.daemon import main as serve_main

        return serve_main(["--port", str(args.port)])

    from local_operator.browser_bridge import install as browser_install
    from local_operator.browser_bridge.daemon import (
        pairing_status,
        reset_pairing,
        revoke_identity,
    )

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
        if getattr(args, "repair", False):
            repair = browser_install.repair()
            for line in repair["steps"]:
                print(f"  {line}")
            if not repair["ok"]:
                print(f"\n\033[1;31m{repair['error']}\033[0m")
                return 1
            print("\nbrowser bridge state reconciled.")
            return 0
        result = browser_install.status()
        health = result.get("health") or {}
        assert isinstance(health, dict)
        print(f"installed:           {'yes' if result['installed'] else 'no'}")
        print(f"daemon healthy:      {'yes' if result['healthy'] else 'no'}")
        # A daemon that predates capability advertisement cannot be told apart
        # from an extension that advertised nothing by its own record, so the
        # harness refuses the new actions with "restart the bridge" — and this is
        # the line that makes that advice checkable from here (design §6.4 row
        # "new harness + old daemon"; review round 1, R4). `capabilities_known`
        # is the WRITER's own stamp: absent means the running bridge is older
        # than the field it is being asked about, whatever its heartbeat says.
        record = result.get("state")
        if isinstance(record, dict) and not record.get("capabilities_known"):
            print(
                "bridge:              predates the file-transfer actions — "
                "run 'lop browser restart'"
            )
        connected = bool(health.get("extension_connected"))
        unresponsive = bool(health.get("extension_unresponsive"))
        print(f"extension connected: {'yes' if connected else 'no'}")
        # The extension-update advisory, immediately under the line it is about.
        # Deliberately a NOTE and not a fault: nothing is refused for an older
        # extension any more (see MIN_SUPPORTED_PROTO), and the extension line
        # above already answers the question the reader came for. The store's
        # live version is unknowable from here, so the contingency is on the
        # store and the sentence never says "requires" or "must".
        if health.get("extension_update_available"):
            from local_operator.browser_bridge.protocol import (
                EXPECTED_EXTENSION_VERSION,
                extension_update_note,
            )

            note = extension_update_note(
                str(health.get("extension_version", "")),
                str(health.get("extension_expected_version") or EXPECTED_EXTENSION_VERSION),
            )
            print(f"                     {note} (fixes and security patches)")
        print(f"paired:              {'yes' if result['paired'] else 'no'}")
        # A paired-but-not-connected browser is the normal closed/backgrounded
        # state, not a fault; say so rather than leaving a user to guess (N2).
        # `extension_unresponsive` is the OTHER way to be paired-but-not-
        # connected — the browser is open and the extension socket is up, but
        # the worker stopped answering — and telling that user "browser not
        # currently attached, it reconnects when opened" is precisely the
        # misdiagnosis that made the wedge expensive. The two lines are
        # mutually exclusive on purpose: `extension_connected` is false in both
        # cases, so only the discriminator separates them.
        #
        # The unresponsive branch has TWO truthful spellings, chosen by
        # `link_attached`, because the daemon both observes the state and then
        # acts on it: while a mute socket is still attached the drop is still
        # ahead ("will drop and re-dial"); once it has been dropped, the link is
        # gone and re-dialling is in progress. Printing the future tense after
        # the drop would assert a severing that already happened, which is the
        # false trail D3 caught, and printing the past tense before it asserts a
        # drop nothing has performed yet. Both say the same thing the user needs:
        # this is not "open your browser".
        if result["paired"] and not connected:
            if unresponsive:
                # Payloads WITHOUT `link_attached` are a daemon from an earlier
                # head of this very change (the field is additive). There the
                # tense is unknowable, so say only what is certainly true and
                # assert no mechanism rather than guess one.
                attached_now = health.get("link_attached")
                if attached_now is True:
                    note = (
                        "browser attached but not answering; the bridge will drop and re-dial "
                        "the link — retry once in a few seconds"
                    )
                elif attached_now is False:
                    note = (
                        "browser attached but not answering; the bridge dropped the link and "
                        "is re-dialling it — retry once in a few seconds"
                    )
                else:
                    note = "browser attached but not answering; retry once in a few seconds"
            else:
                note = "browser not currently attached; it reconnects when opened"
            print(f"                     ({note})")
        # WHICH extensions are paired, and which one has the wheel. Paired
        # alone stopped answering the question the moment two installs could be
        # authorised at once: a user with both a store build and a locally
        # loaded one needs to know which of them the agent is actually driving,
        # and `drive`/`pair --revoke` are addressed by exactly these names.
        # Printed from the FILE plus live /health, so it works with the daemon
        # down as well. Above the driven-tabs block so the reader learns WHO is
        # driving before WHAT.
        _print_identities(pairing_status(), health if result["healthy"] else None)
        # Driven tabs, PLURAL and counted. `driving: <url>` implied a single
        # system-wide binding; with one tab per session that framing turned a
        # stale URL into "something is holding the bridge". Say how many tabs
        # are live, name them, and say "none" explicitly rather than printing
        # nothing (which read as "it isn't telling me").
        driven_tabs = health.get("driven_tabs")
        if connected:
            if isinstance(driven_tabs, list):
                if driven_tabs:
                    label = f"{len(driven_tabs)} tab{'s' if len(driven_tabs) != 1 else ''}"
                    print(f"driving:             {label}")
                    for entry in driven_tabs:
                        if isinstance(entry, dict):
                            print(f"                     - {entry.get('url', '')}")
                else:
                    print("driving:             no tabs driven")
            elif health.get("current_url"):
                # Older daemon still running under a newer CLI.
                print(f"driving:             {health.get('current_url')}")
        # A stale heartbeat is THE thing to surface here. status reads the live
        # /health socket while every session reads the discovery file, so when
        # the two disagree status looks authoritative and sessions silently
        # fall back to cmux. Showing the age makes that contradiction visible
        # and names the fix instead of leaving both parties to guess.
        stale_age = browser_install.stale_heartbeat_age()
        if stale_age is not None:
            print(
                f"\n\033[1;33mheartbeat:           STALE by {stale_age:.0f}s "
                "(discovery file is not being refreshed)\033[0m"
            )
            print(
                "                     sessions will fall back to cmux despite the daemon "
                "being healthy."
            )
            print("                     run 'lop browser status --repair' to reconcile.")
        print(f"port:                {result['port']}")
        print(f"log:                 {result['log']}")
        # Where `download` puts files, and how much is already there. Computed
        # HERE rather than read from /health: the user asking "where did my
        # download go" needs the real path and the real size, and both cost a
        # local stat/walk that a polled HTTP endpoint should not pay for.
        from local_operator import browser_files

        downloads = browser_files.downloads_root()
        print(f"downloads:           {downloads}")
        if downloads.is_dir():
            size = browser_files.dir_size(downloads)
            print(
                f"                     {size} bytes, one audit row per decision in "
                f"{browser_files.AUDIT_FILENAME}"
            )
        # Only when this is NOT the default install: the common case should not
        # grow a line, but an isolated run (a redirected HOME or
        # LOCAL_OPERATOR_CONFIG_DIR) is otherwise indistinguishable from the
        # real one in this output.
        supervisor = result.get("supervisor")
        if supervisor not in (browser_install.LABEL, browser_install.SYSTEMD_UNIT):
            print(f"supervisor:          {supervisor}")
            print(f"config root:         {result.get('config_root')}")
        # An inherited registration explains a daemon running under a name this
        # build would not otherwise mention, so name the file to act on.
        legacy = result.get("legacy_registration")
        if legacy:
            print(f"legacy install:      {legacy}")
            print("                     (written by an older build under the shared name;")
            print("                      'lop browser uninstall' removes it)")
        # A registration this root found but may not manage. Named explicitly,
        # because otherwise a user sees "installed: no" beside a daemon that is
        # plainly running and has nothing to act on.
        ambiguous = result.get("legacy_ambiguity")
        if ambiguous:
            print(f"\n\033[1;33munclaimed registration:\033[0m {ambiguous}")
        return 0 if result["healthy"] else 1
    if command == "pair":
        if getattr(args, "list", False):
            result = browser_install.status()
            pairing = pairing_status()
            if not pairing.get("identities"):
                print("no browser extension is paired. Run 'lop browser pair' to pair one.")
                return 0
            live_health = result.get("health")
            # /health is a plain dict when the daemon answered and None when it
            # did not; the printer takes the second case as "unknown", which is
            # what a file-only listing must say rather than inventing a driver.
            _print_identities(
                pairing, live_health if isinstance(live_health, dict) else None, verbose=True
            )
            pending = pairing.get("pending") or []
            if pending:
                # Its own block, on a line of its own (copy review C7): indented
                # under the identity rows the waiting codes read as part of the
                # `note:` above them, which is a different subject.
                print("")
                print("waiting:             these installs have asked to pair")
                for item in pending:
                    print(f"                     {item.get('code')}  {_waiting_label(item)}")
            return 0
        if getattr(args, "revoke", None):
            pairing = pairing_status()
            identities = pairing.get("identities") or []
            target = _resolve_pairing_target(args.revoke, identities)
            if target is None:
                print(
                    f"\033[1;31mno single authorised extension matches " f"'{args.revoke}'.\033[0m"
                )
                for entry in identities:
                    print(
                        f"  {_short_extension_id(str(entry.get('extension_id', '')))}"
                        f"  {entry.get('label', '') or 'unnamed install'}"
                    )
                if not identities:
                    print("  (nothing is paired)")
                return 1
            # File-level, exactly like --reset: the daemon's revocation watcher
            # then severs THIS identity's live socket within a few seconds and
            # leaves every other identity's authority untouched.
            # Keyword, not positional: this shares the daemon's
            # ``revoke_identity(root, extension_id)`` order, so a positional id
            # would be read as the CONFIG ROOT — revoking nothing at the real
            # root and writing a stray file named after the id.
            # root=None is the default config root, as every other CLI pairing
            # call uses; the id is passed BY KEYWORD because the daemon's order is
            # ``(root, extension_id)`` and a positional id would be read as the
            # root — revoking nothing and writing a stray file named after the id.
            revoke_identity(None, extension_id=target["extension_id"])
            label = str(target.get("label", "")) or _short_extension_id(
                str(target.get("extension_id", ""))
            )
            print(f"revoked {label}; any live connection for it is dropped within a few seconds.")
            return 0
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
        pending = pair.get("pending") or []
        if len(pending) > 1:
            # Two installs waiting at once cannot be told apart by a bare code:
            # the user is looking at two popups and a terminal. Name the install
            # each code belongs to, from the label the daemon recorded when the
            # code was minted (design §3.4) — and the short id as well, because
            # two installs running the SAME build share a label byte for byte
            # (copy review C6), which is precisely when "the matching popup" has
            # no referent and the id prefix is the only token that resolves.
            for item in pending:
                print(f"pairing code: {item.get('code')}   {_waiting_label(item)}")
            print("enter each code in the matching Local Operator extension popup.")
            return 0
        code = pair.get("pending_code")
        if code:
            print(f"pairing code: {code}")
            print("enter this 6-digit code in the Local Operator extension popup.")
            return 0
        if pair.get("paired"):
            # Names BOTH routes, because the old wording offered only `--reset`
            # (UX round 3, U3): a second install whose popup is showing the
            # pairing form reaches here too, and `--reset` — which revokes the
            # WORKING install as well — is not the answer the user wants. The
            # code for that install appears here once it connects, so the honest
            # line points at its popup first. "Connects" rather than "its worker
            # dials" (copy review C9): a compliance analyst has no worker.
            print(
                "a browser is already paired. To pair another, open ITS popup and enter the"
                " code that appears here (the code appears once that install's extension"
                " connects). 'lop browser pair --list' lists every authorised install;"
                " '--reset' revokes them all."
            )
            return 0
        print("no extension is waiting to pair. Open the extension popup, then retry.")
        return 1
    if command == "drive":
        result = browser_install.pin_driver(args.target)
        if not result.get("ok"):
            # Distinguish "nothing matched" from "several matched" when the
            # daemon told us how many did (copy review C8): one 404 shape used to
            # carry both readings, so an unknown id was reported with the word
            # "matches" and two unrelated installs under it. `matches` is absent
            # on a daemon predating this field, and the old sentence stands.
            matches = result.get("matches")
            message = str(result.get("error", "could not pin the driver"))
            if isinstance(matches, int):
                # Echo the target AS TYPED, not the normalised form (review round 5,
                # NIT 5): `drive 'ohcmfhja…'` used to answer "…matches 'ohcmfhja'.",
                # editing the very token the user is looking at.
                message = (
                    f"no connected extension matches '{args.target}'."
                    if matches == 0
                    else f"no single connected extension matches '{args.target}'."
                )
            print(f"\033[1;31m{message}\033[0m")
            # Candidates arrive as ids; the LABEL is what makes a list of ids
            # actionable when two installs share one (copy review C1), and it is
            # the string `status` itself printed a moment earlier. Read from the
            # pairing file, so an id with no label still lists as a bare id.
            candidates = result.get("authorized_extension_ids") or []
            if candidates:
                # A lead-in that is true whether or not anything matched, because
                # these rows are the AUTHORISED set rather than the matches (copy
                # review C8).
                print("authorised installs:")
            labels = {
                str(entry.get("extension_id", "")): str(entry.get("label", ""))
                for entry in (pairing_status().get("identities") or [])
            }
            for extension_id in candidates:
                label = labels.get(str(extension_id), "")
                suffix = f"  {label}" if label else ""
                print(f"  {_short_extension_id(str(extension_id))}{suffix}")
            return 1
        print("now driving: " f"{_short_extension_id(str(result.get('driver_extension_id', '')))}")
        return 0
    if command in ("start", "stop", "restart"):
        result = browser_install.service_action(command)
        if not result["ok"]:
            print(f"\033[1;31m{result['error']}\033[0m")
            return 1
        print(f"browser bridge {command} ok")
        return 0
    if command == "logs":
        import subprocess

        # Through logs_command() so this matches where the daemon's output
        # actually goes: systemd's default is the journal, and tailing the
        # log file there reports "cannot open" on a path nothing writes.
        command_line = browser_install.logs_command(args.lines, follow=args.follow)
        # A log file that was never created means the daemon has not run under
        # a supervisor here, which is a different thing from "it ran and said
        # nothing". Say which, instead of leaving the user with the log
        # reader's "No such file or directory" on a path they never chose.
        # Asked of the install ("does this platform's command read the log
        # file?") rather than of the argv's first token: only the installer
        # knows which platforms redirect into that file and which read the
        # journal instead, and the Windows arm's command is PowerShell's
        # Get-Content, not a tail.
        if browser_install.logs_read_the_log_file() and not browser_install.log_path().exists():
            print(
                f"no daemon log at {browser_install.log_path()}.\n"
                "The bridge has not run under a service supervisor on this machine. "
                "Run `lop browser install`, or `lop browser serve` to run it in the "
                "foreground."
            )
            return 1
        try:
            return subprocess.call(command_line)
        except FileNotFoundError:
            print(
                f"\033[1;31mcannot run `{command_line[0]}`: not installed on this "
                f"system.\033[0m\nThe daemon's output is at "
                f"{browser_install.log_location()}."
            )
            return 1
    if command == "uninstall":
        result = browser_install.uninstall(purge=args.purge)
        steps = result.get("steps", [])
        assert isinstance(steps, list)
        for step in steps:
            print(f"  {step}")
        # Print the reason too. Without this a failed uninstall exits 1 having
        # said nothing at all — the no-supervisor case produces no steps, so
        # the user got a bare non-zero exit with no explanation.
        error = result.get("error")
        if not result.get("ok") and error:
            print(f"\033[1;31m{error}\033[0m")
        # What was found and deliberately left behind, so "uninstalled" never
        # silently means "and something of yours is still registered".
        warning = result.get("warning")
        if warning:
            print(f"\033[1;33mnote:\033[0m {warning}")
        return 0 if result.get("ok") else 1
    print(
        "usage: lop browser " "{install|status|start|stop|restart|pair|drive|logs|uninstall|serve}"
    )
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


def _bind_send_positionals(
    args: argparse.Namespace,
    stdin_body: "str | None" = None,
) -> "tuple[str | None, str | None, str]":
    """Map ``lop send``'s positionals and stdin onto ``(target, message, error)``.

    ``lop send`` accepts the recipient EITHER as a positional substring or as a
    ``--pid``/``--session`` selector, so argparse alone cannot decide what a
    single positional means: it always fills ``target`` first, which made
    ``lop send --pid N "hello"`` bind "hello" to the TARGET and then fail with
    "no message given" — the exact form guides/peer-messaging documented.

    A selector fully determines the recipient, so with one present a lone
    positional normally has exactly one possible meaning: the body. With BOTH a
    positional and a selector the command names two different recipients; the
    resolver would silently prefer the selector and deliver somewhere the
    command does not appear to name, so we refuse instead of guessing.

    ``stdin_body`` is the piped body (``None`` when stdin is a tty or held no
    bytes) and it is an INPUT to the binding, not a fallback applied afterwards.
    It has to be: with a selector and one positional there are two readings —
    the positional is the body (discarding the pipe), or the positional is a
    target that conflicts with the selector (making the pipe the body). Deciding
    without knowing whether a pipe carried data is what made
    ``git log | lop send alpha --pid 10`` deliver the literal string ``alpha``
    and silently drop the commit log, at exit 0. Two candidate bodies is the
    same "the command names two things" shape as two candidate recipients, so it
    gets the same answer: refuse, rather than discard one of them silently. A
    loud error costs a retype; a silent winner costs the payload.

    Note that stdin being a pipe is NOT the discriminator — ``isatty()`` is
    False for an empty redirect (``</dev/null``) just as it is for a pipe
    carrying data, so keying off it would break ``lop send --pid N "hi"
    </dev/null``. Only actual bytes count as a piped body.

    Blank-vs-absent follows the shared core exactly (``peer_send.py``): a target
    that is empty or whitespace is ABSENCE, not a competing address, so
    ``lop send '' BODY --pid 123`` delivers. A blank ``--session`` is instead an
    error — the user explicitly asked to address by session id and supplied
    nothing, and falling back to substring matching there would silently switch
    grammars (``--session '' hello`` would start treating ``hello`` as a target).
    ``--pid`` cannot be blank; argparse's ``type=int`` rejects it first.

    Kept as a pure function (namespace + stdin in, tuple out) for the same
    reason ``resolve_peer_target`` is a core rather than parser logic: the rule
    is testable without a parser, a socket, or a registry. Expressing it in
    argparse itself was tried and rejected — ``nargs="*"`` cannot absorb words
    that follow an optional flag (``lop send peer --wake "act now"`` becomes
    "unrecognized arguments"), and ``argparse.REMAINDER`` swallows ``--wake``
    INTO the body, silently downgrading the delivery mode. See
    ``docs/design/peer-send.md`` §4.3.
    """
    # Blank/whitespace is absence, matching the core's `(target or "").strip()`.
    # An explicitly typed '' is not an address, so it must not read as one.
    target = args.target if (args.target or "").strip() else None
    message = args.message
    piped = stdin_body if (stdin_body or "").strip() else None

    if args.session is not None and not args.session.strip():
        return None, None, "empty --session (pass a session id, or drop the flag)"

    if args.pid is not None:
        selector, by = f"--pid {args.pid}", "pid"
    elif args.session:
        selector, by = f"--session {args.session}", "session id"
    else:
        # No selector: the historical grammar exactly — first positional is the
        # target, second is the body, stdin fills an absent body.
        return args.target, (message if message is not None else piped), ""

    if target is not None and message is not None:
        return (
            None,
            None,
            (
                f"ambiguous recipient: {target!r} and {selector} name different "
                f"sessions. Drop one — `lop send {selector} {shlex.quote(message)}` "
                f"to address by {by}, or `lop send {shlex.quote(target)} "
                f"{shlex.quote(message)}` to address by name"
            ),
        )
    # The sole surviving positional is the body, whichever slot argparse used:
    # a blank target is absence, so `lop send '' BODY --pid N` leaves BODY in
    # the `message` slot with nothing addressing anyone.
    typed = message if message is not None else target
    if typed is not None and piped is not None:
        # Two candidate bodies: the positional and the pipe. Refusing rather
        # than picking is the same answer two candidate RECIPIENTS get, and for
        # the same reason — silently discarding one of them is how
        # `git log | lop send alpha --pid 10` delivered the string 'alpha'.
        return (
            None,
            None,
            (
                f"ambiguous body: {typed!r} and the piped input both look like the "
                f"message. Drop one — `lop send {selector}` to send the piped input, "
                f"or `lop send {selector} {shlex.quote(typed)}` with nothing piped "
                f"to send {typed!r}"
            ),
        )
    # One positional (or none) alongside a selector: it is the body. With no
    # positional the piped input is, which is how the documented pipe form works.
    return None, (typed if typed is not None else piped), ""


def _resolve_peer_target(
    args: argparse.Namespace,
    target: "str | None",
    *,
    skipped: "list[Any] | None" = None,
) -> "tuple[Any | None, list[Any], str]":
    """Resolve a ``lop send`` target to one live SessionRecord.

    Thin adapter over the shared send-side core
    (``mobile.peer_send.resolve_peer_target``): the CLI's argparse namespace is
    mapped onto the core's keyword arguments. The resolution rules themselves —
    pid, then session id, then case-insensitive substring; only ``live`` records;
    candidates returned on ambiguity — live in the core so the in-session
    ``send`` tool resolves targets identically.

    ``target`` is passed explicitly rather than read off the namespace because
    ``_bind_send_positionals`` has already decided whether the positional was a
    target or the message body; ``args.target`` is the RAW parse and using it
    here would re-introduce the binding bug one layer down. It is required
    rather than defaulted for that reason: a caller that forgets it should fail
    loudly, not silently resolve as though no target was given.

    ``skipped`` is forwarded to the core for ``lop send`` only: the send path
    reports how many name-matches were held back for being unengaged, and no
    other caller (the stop path) has a receipt to qualify."""
    from local_operator.mobile.peer_send import resolve_peer_target

    # The flag grammar is passed in so the CLI's user-visible error keeps saying
    # `--pid` / `--session`, exactly as it did before the extraction.
    return resolve_peer_target(
        target=target,
        pid=args.pid,
        session=args.session,
        pid_hint="--pid",
        session_hint="--session",
        skipped=skipped,
    )


def send_command(args: argparse.Namespace) -> int:
    """``lop send`` — hand a message to another local lop session.

    Delivery mode maps from the flags: ``--now``/``--steer`` => steer (inject
    mid-turn), otherwise mailbox; ``--wake`` drives a turn if the mailbox
    target is idle. The body comes from the positional argument or, when
    omitted, stdin (the ergonomic path for piping a longer note).

    The positionals are rebound BEFORE anything is resolved or dialled, because
    an ambiguous recipient must cost nothing: the refusal has to happen while
    the command is still inert. stdin is read up front for the same reason —
    the binder cannot tell a lone positional's meaning without knowing whether a
    pipe also carried a body, and reading it later meant the positional won and
    the piped payload was discarded silently."""
    import asyncio

    from local_operator.mobile.peer_send import (
        candidate_lines,
        deliver_peer_message,
        skipped_clause,
        validate_peer_body,
    )

    # stdin can only change the binding when at most ONE positional was typed:
    # with both slots filled the outcome is already decided (a conflict when a
    # selector is present, the typed body otherwise), so stdin is irrelevant.
    # Restricting the read matters beyond efficiency — `slow | lop send NAME
    # "body"` must not block waiting for a producer whose output is not used,
    # and a tty read would block waiting for the user to type.
    #
    # Only real bytes count as a piped body: `isatty()` is False for an empty
    # redirect (`</dev/null`) exactly as it is for a pipe carrying data, so it
    # cannot distinguish "piped a body" from "stdin merely is not a terminal".
    # Decoded leniently off the BINARY stream rather than through `sys.stdin`'s
    # strict UTF-8 text wrapper. Reading before resolution widened this from a
    # delivery-path concern to every path: a `some-binary-producer | lop send …`
    # (a gzip or an image piped by mistake) would raise UnicodeDecodeError out
    # of send_command as a traceback, even when the target does not resolve and
    # the bytes are never used. An uncaught traceback is never an acceptable
    # user-visible failure here — the same U1 rule the delivery except-clause
    # below follows.
    #
    # `errors="replace"` DELIVERS undecodable input as U+FFFD replacement
    # characters; it does not refuse it. `validate_peer_body` rejects only an
    # empty or over-cap body, and mojibake is neither, so mis-piping a gzip
    # sends the peer a screenful of "�" at rc=0. That is deliberate: this is a
    # text-messaging command, the sender sees the recipient in the success line,
    # and no payload is lost or misdirected. Refusing on undecodable bytes
    # instead would mean inventing a new error class here and deciding what
    # fraction of replacements is "too much" — a guess, on a path where the
    # user can simply look at what they piped. The prior behaviour on this path
    # was a crash, so degraded text is strictly better.
    stdin_body = None
    if args.message is None and not sys.stdin.isatty():
        raw = sys.stdin.buffer.read() if hasattr(sys.stdin, "buffer") else sys.stdin.read()
        stdin_body = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw

    target, message, bind_error = _bind_send_positionals(args, stdin_body)
    if bind_error:
        _peer_red(bind_error)
        return 1

    # Shared with the in-session send tool: the stored fallback below must run
    # only on the live resolver's NO-MATCH form — see
    # ``peer_send.live_scan_found_nothing`` for why a bare ``record is None``
    # is not enough (it is also how a conflicting selector pair and a wedged
    # unique match come back, and neither may be converted into a stored send).
    from local_operator.mobile.peer_send import (
        live_scan_found_nothing,
        session_id_unowned,
    )

    # Matches the name/substring scan held back for being unengaged. Filled by
    # the resolver and reported on the receipt below: a sender who typed one
    # command believing it reached its needle has to learn that part of it went
    # nowhere (design round 1, D1). Always empty for the exact and stop paths.
    skipped: list[Any] = []
    record, candidates, error = _resolve_peer_target(args, target, skipped=skipped)
    if candidates:
        # "REPLACE the target with", not "add --pid": appending the flag to the
        # command the user just typed produces `NAME BODY --pid N`, which the
        # ambiguity guard now refuses. The instruction has to describe the form
        # that actually works, or it walks the user into a second error.
        print(
            f"{len(candidates)} sessions match; replace the target with one of these:",
            file=sys.stderr,
        )
        for line in candidate_lines(candidates, indent="  ", prefix="--pid"):
            print(line, file=sys.stderr)
        # The worked example must not echo the body back. For a PIPED body the
        # right recovery is to re-pipe, not to paste — printing it would dump up
        # to the whole 256 KB cap onto stderr and tell the user to retype what
        # they piped. A long typed body is elided for the same reason: the line
        # exists to show the SHAPE of the working command, not to reproduce the
        # message the user still has one line up.
        if stdin_body is not None:
            example = "<your piped input> | lop send --pid " + str(candidates[0].pid)
        elif message is not None:
            # A short body is reproduced verbatim so the line pastes and works,
            # which is what BLOCKER-2 asked for. A long one becomes a
            # PLACEHOLDER rather than a truncation: pasting `…LLL...` would
            # silently deliver a 60-character stub ending in an ellipsis, which
            # is the copy-paste trap the piped branch above already avoids.
            example = (
                f"lop send --pid {candidates[0].pid} {shlex.quote(message)}"
                if len(message) <= 60
                else f"lop send --pid {candidates[0].pid} '<your message>'"
            )
        else:
            example = ""
        if example:
            print(f"  e.g. `{example}`", file=sys.stderr)
        return 1
    cold_session_id = ""
    if record is None and session_id_unowned(error):
        # No live record OWNS this id — the scan did not know it at all, or the
        # record it found was stale (the pid is gone) — so an exact `--session`
        # may name a stored session that is simply not running. A quiet note to
        # one of those is the mailbox mode's whole purpose, so it is spooled
        # rather than refused; anything wanting attention starts a runtime.
        #
        # The predicate is what keeps a refusal about a LIVE session standing
        # (QA round 3, Q8; review round 4, MINOR-1): an unengaged or wedged
        # match is the live resolver's answer, and re-asking the store for the
        # same id would spool the note behind a process that still owns the
        # conversation while telling the sender it was merely held.
        from local_operator.mobile.peer_send import resolve_cold_session

        cold_session_id = resolve_cold_session(args.session or "") or ""
    if (
        not cold_session_id
        and not candidates
        and target
        and record is None
        and live_scan_found_nothing(error)
    ):
        # The substring found no LIVE record and named no exact id. Fall back
        # to the STORED store before refusing: a note addressed by the name a
        # session had before its terminal was closed must still deliver. The
        # resolver decides WHO; delivery still goes through the unchanged
        # cold path below, so live keeps winning over stored for the same
        # substring and the delivery mechanics are untouched.
        #
        # The ``live_scan_found_nothing(error)`` term is the whole point of
        # this guard (review round 1, BLOCKER-1/MAJOR-1): ``record is None``
        # is ALSO how a refused target+selector conflict and a wedged unique
        # match come back, and delivering those to a similarly named stored
        # session would be a send to a recipient the command never named.
        from local_operator.mobile.peer_send import (
            resolve_stored_target,
            stored_candidate_lines,
        )

        stored_id, stored_candidates, stored_error = resolve_stored_target(target)
        # ``stored_error`` is read here, unlike a plain no-match (which returns
        # "" by contract): a row that ANSWERED to the name but was withheld for
        # never having been engaged comes back as its own refusal, and printing
        # "no session matches" over it would be a false statement about a
        # session the user can see on the picker (review round 1, F-4). The
        # composed miss below still applies to a true no-match.
        if stored_candidates:
            print(
                f"{len(stored_candidates)} stored sessions match; replace the "
                "target with one of these:",
                file=sys.stderr,
            )
            for line in stored_candidate_lines(stored_candidates, indent="  ", prefix="--session"):
                print(line, file=sys.stderr)
            return 1
        if stored_id:
            cold_session_id = stored_id
        elif stored_error:
            error = stored_error
    if not cold_session_id and (error or record is None):
        if error and live_scan_found_nothing(error):
            # The stored fallback just failed too, so the message names BOTH
            # searches rather than leaving the user to discover the store on
            # their own.
            error = f"no session matches {target!r} (searched live and stored sessions)"
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
    if record is not None and record.pid == sender_pid:
        _peer_red("that target is this session; use the composer to message yourself")
        return 1

    # The binder already resolved the body from the positionals and stdin
    # together; nothing survives here means neither supplied one.
    if message is None:
        _peer_red("no message given (pass it as an argument or pipe it on stdin)")
        return 1
    text = message
    body_error = validate_peer_body(text)
    if body_error:
        _peer_red(body_error)
        return 1

    mode = "steer" if args.steer else "mailbox"
    try:
        detail = asyncio.run(
            deliver_peer_message(
                record,
                session_id=(record.session_id if record is not None else cold_session_id),
                text=text,
                mode=mode,
                wake=bool(args.wake),
                sender=sender,
            )
        )
    except TimeoutError as exc:
        # NOT "could not deliver": a read deadline expiring means no
        # ACKNOWLEDGED result, not an undelivered message — the mutation op is
        # already in the owner's socket buffer, and the receiver commits before
        # it acks (``peer_send._unanswered_dial_detail``). Saying it failed
        # invites a duplicate steer or wake. Same split, and the same words, as
        # the send TOOL's arm below it.
        _peer_red(f"no delivery confirmation: {exc}")
        return 1
    except (RuntimeError, ConnectionError, OSError, ValueError) as exc:
        # ValueError covers a read fault the frame reader could still surface
        # (e.g. an oversized non-welcome line): it must become the same soft,
        # non-zero "could not deliver" line, never an uncaught traceback (U1).
        _peer_red(f"could not deliver: {exc}")
        return 1
    if record is not None:
        name = record.conversation_name or record.session_id
        print(f"→ {name} (pid {record.pid}): {detail}{skipped_clause(skipped)}")
    else:
        print(f"→ {cold_session_id} (not running): {detail}{skipped_clause(skipped)}")
    return 0


def _non_negative_int(text: str) -> int:
    """argparse type for the cleanup limits: `/settings` enforces `minimum=0`
    on every one of them and the CLI must not be the lax door (UX U7)."""
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be 0 or more, got {value}")
    return value


def _confirm_window(text: str) -> int:
    """argparse type for ``sessions reclaim --confirm-s``: seconds, with a floor.

    A SHORTER WINDOW IS A SWEEP WITH NO CPU RUNG, not a faster sweep: the CPU
    budget is ``max(BUSY_CPU_FLOOR_S, BUSY_CPU_FRACTION * elapsed)``, so below
    ``MIN_ACTIONABLE_CONFIRM_S`` the floor dominates and no measurement can
    exceed it. ``0`` was the worst case and it was reachable — ``--confirm-s 0``
    parsed (the type was a non-negative int) and skipped the watch entirely, and
    QA round 1 (Q2) measured a process with 90.4 s of cumulative CPU being
    admitted and SIGTERMed at that window, having been correctly refused at the
    default one (6.30 s spent per 60 s against a 1.2 s budget). The floor is
    derived from those two constants rather than restated, so it moves with them.
    ``--dry-run`` remains available for looking without a window at all.
    """
    from local_operator.session.runtime.reclaim import MIN_ACTIONABLE_CONFIRM_S

    value = int(text)
    if value < MIN_ACTIONABLE_CONFIRM_S:
        raise argparse.ArgumentTypeError(
            f"the confirm window must be at least {MIN_ACTIONABLE_CONFIRM_S:.0f}s, "
            f"got {value}: a shorter window cannot measure CPU, so the sweep would "
            "act on two sightings with no separation and no CPU refusal in between"
        )
    return value


def _cleanup_row(candidate: Any, verb: str) -> str:
    """One decision, with what a user needs to judge it: name, age, size."""
    # Budgeted to 100 columns with a 12-hex id, the origin column and the
    # longest reason (`[max_inactive_days] idle over 365d`): title 16 cells,
    # one space between columns. At 110 cols a 48-char title wrapped every
    # row and a 9-row list read as 13 lines (UX round 2, U13). The reasons
    # state the LIMIT; the age and size columns state the fact.
    title = candidate.title or "(no title)"
    if len(title) > 16:
        title = title[:15] + "…"
    # One decimal under 100 d so a session just past a whole-day limit does
    # not print the same figure as the limit it exceeded.
    days = candidate.idle_days
    age = f"{days:.1f}d" if days < 100 else f"{days:.0f}d"
    # ``origin`` says whose the row is: on a store that is mostly subagent
    # runs, a list of ids and titles cannot tell the user which rows are
    # their own conversations (UX round 3, U15). 8 cells: "subagent".
    origin = getattr(candidate, "origin", "user") or "user"
    return (
        f"  {verb:<12} {candidate.session:<12} {origin:<8} {title:<16} {age:>5} "
        f"{_format_bytes(candidate.size_bytes):>6} [{candidate.policy}] {candidate.reason}"
    )


def sessions_cleanup_command(args: argparse.Namespace) -> int:
    """``lop sessions cleanup [--dry-run] [--force [--yes]]``.

    Order of operations for a real run is LIST, CONFIRM, REMOVE — never the
    reverse (UX U2). The listing is the dry run over the same policy, so what
    the user confirms is exactly what goes.

    Exit codes: 0 ran (or dry run); 1 nothing to apply (no limits); 2 refused
    (switch off without ``--force``, or confirmation declined); 3 ran with
    errors. ``--json`` always emits a JSON object on stdout, whatever the
    outcome, and carries ``enabled`` so a script can see the switch state.
    """
    import dataclasses
    import json as _json

    from local_operator.session.cleanup import (
        CLEANUP_LOG_NAME,
        CleanupResult,
        apply_cleanup,
        policy_from_config,
        run_cleanup,
    )

    root = config_dir()
    policy = policy_from_config(ConfigManager(root))
    overrides = {
        name: value
        for name, value in (
            ("max_sessions", args.max_sessions),
            ("max_inactive_days", args.max_inactive_days),
            ("max_total_bytes", args.max_total_bytes),
            ("remove_empty", args.remove_empty),
        )
        if value is not None
    }
    policy = dataclasses.replace(policy, **overrides)
    record_path = root / "sessions" / CLEANUP_LOG_NAME
    switch_hint = (
        "session.cleanup.enabled is off: turn it on in /settings > Session cleanup, "
        "or pass --force to run once"
    )

    def emit_json(result: CleanupResult, *, outcome: str, confirmed: bool | None = None) -> None:
        print(
            _json.dumps(
                {
                    "outcome": outcome,
                    "enabled": policy.enabled,
                    "forced": bool(args.force),
                    "dry_run": result.dry_run,
                    "scanned": result.scanned,
                    "removed": [dataclasses.asdict(c) for c in result.removed],
                    "protected": [
                        {"session": name, "guard": guard} for name, guard in result.protected
                    ],
                    "errors": result.errors,
                    "skipped": result.skipped,
                    "confirmed": confirmed,
                    "record": str(record_path),
                },
                indent=2,
            )
        )

    if not policy.has_any_limit:
        message = (
            "no cleanup limits configured: set session.cleanup.* in /settings or pass "
            "--max-sessions/--max-inactive-days/--max-total-bytes/--remove-empty"
        )
        if not policy.enabled:
            message += f"\n{switch_hint}"
        if args.json:
            emit_json(CleanupResult(skipped="no limits configured"), outcome="nothing-to-do")
        else:
            print(message, file=sys.stderr)
        return 1

    if not policy.enabled and not args.force and not args.dry_run:
        if args.json:
            emit_json(CleanupResult(skipped="disabled"), outcome="refused")
        else:
            print(f"refusing to remove sessions: {switch_hint}", file=sys.stderr)
            print("preview what the limits would remove with --dry-run", file=sys.stderr)
        return 2

    # The LISTING is always a dry run first, so a real run shows the user the
    # same rows before anything is removed.
    preview = run_cleanup(root, policy, dry_run=True, force=bool(args.force), actor="cli")
    policy_line = (
        f"policy: enabled={'on' if policy.enabled else 'OFF'} max_sessions={policy.max_sessions} "
        f"max_inactive_days={policy.max_inactive_days} max_total_bytes={policy.max_total_bytes} "
        f"remove_empty={policy.remove_empty}"
    )

    if args.dry_run:
        if args.json:
            emit_json(preview, outcome="dry-run")
            return 0
        print(policy_line)
        if not policy.enabled:
            print(f"note: {switch_hint}; this is a preview only")
        print(f"scanned {preview.scanned} sessions; would remove {len(preview.removed)}")
        for candidate in preview.removed:
            print(_cleanup_row(candidate, "would remove"))
        for name, guard in preview.protected:
            print(f"  kept         {name}  ({guard})")
        print(f"nothing was removed (dry run); the record of real removals is {record_path}")
        return 0

    if not args.json:
        print(policy_line)
        if not policy.enabled:
            print(f"WARNING: {switch_hint}; running because --force was given")
        print(f"scanned {preview.scanned} sessions; about to remove {len(preview.removed)}")
        for candidate in preview.removed:
            print(_cleanup_row(candidate, "will remove"))
        for name, guard in preview.protected:
            print(f"  kept         {name}  ({guard})")
    if not preview.removed:
        if args.json:
            emit_json(preview, outcome="nothing-to-do")
        else:
            print("nothing to remove")
        return 0

    confirmed: bool | None = None
    if not args.yes:
        if not sys.stdin.isatty():
            if args.json:
                emit_json(preview, outcome="refused", confirmed=False)
            else:
                print(
                    "refusing: not a terminal and --yes was not given, so nothing was removed",
                    file=sys.stderr,
                )
            return 2
        try:
            answer = input(f"remove {len(preview.removed)} session(s)? type 'yes' to confirm: ")
        except (EOFError, KeyboardInterrupt):
            answer = ""
        confirmed = answer.strip().lower() == "yes"
        if not confirmed:
            if args.json:
                emit_json(preview, outcome="refused", confirmed=False)
            else:
                print("not confirmed; nothing was removed")
            return 2

    # The removal WARNINGs go to the rotating file (`local-operator.log`),
    # not the console: the stdout listing below already carries every fact
    # and the console copy doubled each line in a terminal (UX U3, QA Q7).
    # ``file_logging`` detaches the console handlers for the block and puts
    # them back afterwards. ``apply_cleanup(preview)`` removes EXACTLY the
    # rows the user just confirmed — a second scan could rank a session
    # created meanwhile and take one shown as kept (review round 2, R2-2).
    with file_logging():
        result = apply_cleanup(root, preview, actor="cli")
    if args.json:
        emit_json(result, outcome="removed", confirmed=confirmed)
        return 3 if result.errors else 0
    print(f"removed {len(result.removed)} session(s)")
    for candidate in result.removed:
        print(_cleanup_row(candidate, "removed"))
    if result.errors:
        print(f"  {result.errors} error(s); see the log", file=sys.stderr)
    print(f"record: {record_path}")
    return 3 if result.errors else 0


def sessions_reclaim_command(args: argparse.Namespace) -> int:
    """``lop sessions reclaim [--dry-run] [--yes] [--confirm-s N]``.

    The operator's door to the external residency sweep
    (:mod:`local_operator.session.runtime.reclaim`) — the same pass the wake
    supervisor runs on its own cadence, exposed because the supervisor retires
    when nothing is fireable and because a person asking "what is still holding
    memory" should not have to wait for a wake to be due.

    Order of operations is LIST, WAIT, CONFIRM, SIGNAL. The wait is the point: the
    sweep's decision is taken from TWO sightings of the process table, so this
    command watches for ``--confirm-s`` seconds (default ``reclaim.CONFIRM_S``)
    between them and drops anything that gained a record, an attach or CPU in
    between. A runtime whose record appears while the operator is reading the
    listing is therefore never signalled.

    Exit codes: 0 looked (dry run, or nothing to reclaim) or reclaimed; 2 refused
    (confirmation declined, or no terminal and no ``--yes``); 3 signalled but at
    least one runtime had not gone within the wait.
    """
    import json as _json

    from local_operator.session.runtime.reclaim import (
        CONFIRM_S,
        EXIT_WAIT_S,
        Sightings,
        reclaim_runtimes,
    )

    root = config_dir()
    confirm_s = CONFIRM_S if args.confirm_s is None else float(args.confirm_s)
    sightings = Sightings()

    # PASS 1 — the listing. Same call, same rule, nothing signalled: what the
    # operator reads here is produced by the code that later acts, so the two
    # cannot disagree about which runtimes are candidates.
    preview = reclaim_runtimes(root, apply=False, sightings=sightings, confirm_s=confirm_s)
    candidates = preview.reclaimed + preview.pending

    def row(item: Any, verb: str) -> str:
        return (
            f"  {verb} pid {item.process.pid:<7} session {item.session_id or '<unknown>':<24} "
            f"root {item.config_root or '<unknown>':<40} "
            f"alive {item.process.age_s / 3600.0:.1f}h cpu {item.process.cpu_s:.1f}s"
        )

    if args.json:
        print(_json.dumps(preview.to_json(), indent=2))
        return 0

    print(preview.summary())
    if args.dry_run:
        for item in candidates:
            print(row(item, "would reclaim"))
        print(
            f"nothing was signalled (dry run); a real run watches {confirm_s:.0f}s before it acts, "
            "and drops anything that gains a record, an attach or CPU in that window"
        )
        return 0

    if not candidates:
        print("nothing to reclaim: every live session runtime is either recorded or refused")
        return 0

    for item in candidates:
        print(row(item, "will reclaim"))
    confirmed: bool | None = None
    if not args.yes:
        if not sys.stdin.isatty():
            print(
                "refusing: not a terminal and --yes was not given, so nothing was signalled",
                file=sys.stderr,
            )
            return 2
        try:
            answer = input(f"end {len(candidates)} unreachable runtime(s)? type 'yes' to confirm: ")
        except (EOFError, KeyboardInterrupt):
            answer = ""
        confirmed = answer.strip().lower() == "yes"
        if not confirmed:
            print("not confirmed; nothing was signalled")
            return 2

    # PASS 2 — the confirm window elapses, then the SAME memory is asked to act.
    # The decision is not re-derived from this listing: the second pass re-censuses
    # and re-reads the records, so a runtime that became reachable in between is
    # refused by the same rungs that refused the others, and the CPU delta the pass
    # measures is over the window the operator just waited out.
    if confirm_s > 0:
        print(f"watching {confirm_s:.0f}s for a record, an attach or CPU before signalling...")
        time.sleep(confirm_s)
    report = reclaim_runtimes(
        root, apply=True, sightings=sightings, confirm_s=confirm_s, wait_s=EXIT_WAIT_S
    )
    print(
        f"signalled {len(report.signalled)} runtime(s); {len(report.exited)} gone within "
        f"{EXIT_WAIT_S:.0f}s"
    )
    for item in report.exited:
        print(row(item, "gone    "))
    for item in report.signalled:
        if item in report.exited:
            continue
        # A signalled runtime that is still there is NOT a failure: its own drain is
        # bounded by SIGNAL_DRAIN_S and finishing a turn can take that long. Reported
        # as still-leaving rather than as an error, because "it did not die" is what
        # the drain is for.
        print(row(item, "leaving "))
    return 3 if len(report.exited) < len(report.signalled) else 0


def _positive_int(value: str) -> int:
    """Argparse type for counts where 0 or negative is a typo, not a request.

    `lop sessions --all --limit 0` parsed fine and listed NOTHING (`rows[:0]`),
    which reads on screen as "no sessions exist" — a lie about the store
    (review round 1, Q2). There is no zero-or-uncapped spelling to protect
    either: 0 stored rows is what plain `lop sessions` already means, and the
    no-cap case is served by the advertised DEFAULT, not by a magic number.
    """
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer") from None
    if parsed < 1:
        raise argparse.ArgumentTypeError(
            f"must be a positive integer, got {value!r} — omit --limit for the default cap"
        )
    return parsed


def sessions_command(args: argparse.Namespace) -> int:
    """``lop sessions`` — list active sessions and their resource usage.

    RSS is the always-present baseline; FOOTPRINT is the true memory number
    where the platform can report it (macOS phys footprint / Linux Pss) and —
    otherwise. HEARTBEAT_AGE surfaces wedged-ness numerically so counts can be
    eyeballed against reality."""
    import json as _json

    if getattr(args, "sessions_command", None) == "cleanup":
        return sessions_cleanup_command(args)

    if getattr(args, "sessions_command", None) == "reclaim":
        return sessions_reclaim_command(args)

    # The row shape lives in ``info.collect`` and is shared with ``/info``,
    # which needs the same "which sessions exist and what do they cost" answer
    # plus an ``is_self`` mark and roll-up counters. It was EXTRACTED rather
    # than copied because two of the details here are subtle enough that a
    # second implementation would have drifted on them: the ``getattr``
    # defaulting that keeps a record written by an OLDER runtime listing rather
    # than raising mid-table, and ``state``'s meaning (``stale`` is a record the
    # scan just deleted, not an idle session). The ``--json`` key order is part
    # of the CLI's published contract, so ``session_rows`` pins it explicitly
    # rather than deriving it from the dataclass — which would also have leaked
    # ``is_self``, a field this command never had.
    from local_operator.info.collect import session_rows

    # ``--limit`` bounds the STORED rows only — the live fleet is always listed
    # in full, and a stored cap means nothing without ``--all`` asking for them.
    # ``None`` lets ``collect`` apply its own stored cap.
    rows = session_rows(config_dir(), include_stored=args.all, stored_limit=args.limit)

    if args.json:
        print(_json.dumps(rows, indent=2))
        return 0

    if not rows:
        # `--all` asked about the STORE as well as the fleet; the live-only
        # empty line would answer a question the user did not ask (review
        # round 1, MINOR-5). The default listing keeps its established copy.
        print("no lop sessions (live or stored)" if args.all else "no active lop sessions")
        return 0

    # NEEDS is the column this release adds, and it earns its width: a parked
    # question holds a runtime resident for up to a day, so "which of these is
    # waiting on me" has to be answerable from the same place the memory is
    # visible. Blank for every session that is simply working.
    #
    # LAST_ACTIVE appears only when stored rows are present — normally under
    # ``--all``, but NOT when the flag found none to add (every stored id
    # already live, or an empty store): a stored session has no process, so
    # PID/RSS/FOOTPRINT/UPTIME/HB_AGE read "—" for it, and the one fact a
    # stored row CAN offer is when its transcript last moved. Adding the column
    # unconditionally would re-flow the live-only listing every existing
    # consumer parses, so it is appended only when the flag brought stored rows.
    show_stored = any(row["state"] == "stored" for row in rows)
    # WHY is the column THIS change adds, and it follows LAST_ACTIVE's precedent
    # for the same reason: it is appended only when some row has something to
    # say, so a healthy listing parses exactly as it did before. It is also the
    # answer to the one question the 2026-09-13 kill wave left the operator
    # unable to ask from a shell — "why did this session die, and did I ask for
    # it?" — because a killed runtime publishes nothing and its reason survives
    # only in the attention store (``completion_reason``).
    from local_operator.incidents import outcome_summary

    why = {
        row["session_id"]: (
            ""
            # A STORED OUTCOME IS NOT A LIVE ONE, and this row is the case where
            # the difference is the whole message. ``completion_kind`` is the
            # last outcome the attention store RECORDED — the previous turn's
            # end — while ``pending`` and ``busy`` come from the runtime's own
            # record and describe what it is doing NOW. A session that was
            # stopped, then resumed, and is now parked on an approval carries
            # both: an ``interrupted`` receipt from before the resume, and a
            # live parked gate.
            #
            # Printed together they contradict each other. The observed row read
            # ``NEEDS approval`` beside ``WHY the session was stopped by the
            # user`` — for a runtime that had been alive and working for six
            # hours since that stop, and was at that moment holding a turn open
            # waiting for the operator to answer a card. The WHY column is the
            # one place a shell reader learns why a session is not progressing,
            # and it was spending its width on a superseded receipt while the
            # actual answer ("it is waiting for you") was a column away.
            #
            # So the receipt is suppressed exactly when the row's own live state
            # has already overtaken it, on the precedent ``catalog.status`` set
            # for the picker: ``pending`` and the live states outrank an unread
            # completion mark there for this same reason, and the two surfaces
            # describing one session must not disagree. Nothing is invented and
            # no width is added — ``NEEDS`` already says which gate it is, and
            # ``--json`` still carries ``completion_kind``/``completion_reason``
            # verbatim for anything parsing the outcome rather than reading the
            # table.
            if _live_state_supersedes_outcome(row)
            else (
                outcome_summary(str(row.get("completion_reason") or ""))
                # A ``complete`` row has no why: its reason is a success sentence,
                # and a column that explains every healthy session says nothing.
                if row.get("completion_kind") not in ("", "complete")
                else ""
            )
        )
        for row in rows
    }
    show_why = any(why.values())
    # ONLY WHEN SOMETHING IS LEAVING, on the same rule as WHY and LAST_ACTIVE
    # above: a healthy fleet's listing must not gain a column of blanks, and the
    # table is parsed by people. Unlike WHY this says something true about a row
    # the operator may be about to act on — a signalled runtime is alive and
    # working for up to ``SIGNAL_DRAIN_S``, and a plain ``lop stop`` on it cuts
    # the turn the drain is finishing (U1/U2, PR #1141).
    leaving = {row["session_id"]: (row.get("leaving") or "") for row in rows}
    show_leaving = any(leaving.values())
    # The same rule as LEAVING and WHY above, and the same reason it must be a
    # SEPARATE column rather than part of that one: a session mid-update is alive,
    # accepting messages, and about to run them (``types.UPDATING``) — the operator
    # reading that row must not be told to re-send what is already queued.
    updating = {row["session_id"]: (row.get("updating") or "") for row in rows}
    failed = {row["session_id"]: (row.get("update_failed") or "") for row in rows}
    # THE COLUMN PRINTS WHENEVER ANY ROW HAS SOMETHING TO SAY ABOUT A MOVE, and a
    # FAILED one counts. A fleet whose only news is an abandoned update used to drop
    # the column entirely and list that session exactly as an ordinary idle one — the
    # defect design review round 1 (D1) measured against this renderer.
    show_updating = any(updating.values()) or any(failed.values())
    # THE THIRD STATE (design review round 1, D1), gated and separate for the reasons
    # the two above state: it is not a departure and not an update, it is a runtime
    # whose own bound fired, dumped and could not end it — stalled with the turn still
    # inside, and only ``lop stop`` ends it. Without this the listing rendered such a
    # session exactly as an ordinary idle one, which is the half of the operator's rule
    # (a runtime is a person's to end once it can no longer finish) that a listing
    # carries.
    held = {row["session_id"]: bool(row.get("stall_held")) for row in rows}
    show_held = any(held.values())
    header = (
        f"{'STATE':<{STATE_COLUMN_WIDTH}} {'PID':>7} {'KIND':<7} "
        f"{'NEEDS':<{NEEDS_COLUMN_WIDTH}} {'CONVERSATION':<{CONVERSATION_COLUMN_WIDTH}} "
        f"{'MODEL':<{MODEL_COLUMN_WIDTH}} {'RSS':>8} {'FOOTPRINT':>9} {'UPTIME':>8} "
        f"{'HB_AGE':>7}"
    )
    if show_stored:
        header += f" {'LAST_ACTIVE':>11}"
    if show_why:
        header += f" {'WHY':<{WHY_COLUMN_WIDTH}}"
    if show_leaving:
        header += f" {'LEAVING':<{LEAVING_COLUMN_WIDTH}}"
    if show_updating:
        header += f" {'UPDATING':<{UPDATING_COLUMN_WIDTH}}"
    if show_held:
        header += f" {'STALLED':<{HELD_COLUMN_WIDTH}}"
    print(header)
    now = time.time()
    for row in rows:
        # CELLS, not characters, for the three columns that carry text this
        # process did not author (design round 1, D2). A conversation title can
        # be CJK and a model label carries the provider's own display name, and a
        # 14-glyph title is 28 cells: a `[:24]` slice returned all of it and the
        # row ran into its neighbour, leaving the table wider than its header.
        # Measured here only for DISPLAY — `--json` above still carries the full
        # values, which is what a script should read.
        name = _fit_cell(
            row["conversation_name"] or row["session_id"] or "",
            CONVERSATION_COLUMN_WIDTH,
        )
        model = _fit_cell(row["model_label"] or "", MODEL_COLUMN_WIDTH)
        needs = _fit_cell(row.get("pending") or "", NEEDS_COLUMN_WIDTH)
        stored = row["state"] == "stored"
        state = _state_cell(row["state"])
        line = (
            f"{state:<{STATE_COLUMN_WIDTH}} "
            f"{('—' if stored else str(row['pid'])):>7} "
            f"{(row['kind'] or '—'):<7} {_pad_cell(needs, NEEDS_COLUMN_WIDTH)} "
            f"{_pad_cell(name, CONVERSATION_COLUMN_WIDTH)} "
            f"{_pad_cell(model, MODEL_COLUMN_WIDTH)} {_format_bytes(row['rss_bytes']):>8} "
            f"{_format_bytes(row['footprint_bytes']):>9} "
            f"{('—' if stored else _format_duration(row['uptime_s'])):>8} "
            f"{('—' if stored else _format_duration(row['heartbeat_age_s'])):>7}"
        )
        if show_stored:
            stamp = row["last_activity_s"]
            age = "—" if stamp is None else _format_duration(max(0.0, now - stamp))
            line += f" {age:>11}"
        if show_why:
            cell = _clamp_reason_cell(why.get(row["session_id"]) or "")
            # `:<{WHY_COLUMN_WIDTH}` would pad this by CHARACTERS and hand a
            # fitted wide cell the blanks it never needed; `_pad_cell` is the
            # same CELLS-not-characters rule the three text columns use above.
            line += f" {_pad_cell(cell, WHY_COLUMN_WIDTH)}"
        if show_leaving:
            # `_fit_cell` rather than `_clamp_reason_cell`: this column's text is
            # the harness's own phrase, so a cut only ever needs to be visible —
            # the reason clamp's marker exists for provider-authored prose.
            said = _fit_cell(leaving.get(row["session_id"]) or "", LEAVING_COLUMN_WIDTH)
            line += f" {_pad_cell(said, LEAVING_COLUMN_WIDTH)}"

        if show_updating:
            # The cell is RENDERED from the row's pair through the ONE phase reader
            # (``types.update_phase``/``update_short``), so the copy here, the info
            # panel and the phone cannot drift into three vocabularies for one state —
            # and so the FAILED phase reaches this surface at all (design review round
            # 1, D1, where the row rendered blank).
            cell = _updating_cell(
                updating.get(row["session_id"]) or "", failed.get(row["session_id"]) or ""
            )
            # MARKED, unlike the cells above: the value here is a BUILD LABEL, so a
            # silent cut hands the reader a plausible version for a session that is on
            # a different one (design review round 1, D3 — ``updating →
            # 0.59.11.dev3+g1`` cut to a real-looking ``0.59.11``). ``_clamp_reason_cell``
            # already carries that argument for WHY; this is the same mark applied to
            # the one column whose text is an identifier rather than prose.
            said = _clamp_reason_cell(cell, UPDATING_COLUMN_WIDTH)
            line += f" {_pad_cell(said, UPDATING_COLUMN_WIDTH)}"
        if show_held:
            # AFTER the updating cell, because the header appends ``STALLED`` after
            # ``UPDATING`` (agent review round 2, MAJOR-3): emitted the other way round
            # the two values sat under each other's headers on every row that carries
            # both — the held cell an updating value, and vice versa. The record's own
            # phrase is a full sentence and belongs in the notice; a list column carries
            # the fact, in the words the panel uses, so one state does not acquire two
            # vocabularies across the two surfaces a reader compares (the rule
            # ``STOP_RUNG_LABELS`` states for the stop rungs).
            said_held = HELD_CELL if held.get(row["session_id"]) else ""
            line += f" {_pad_cell(said_held, HELD_COLUMN_WIDTH)}"
        print(line)
    return 0


def _updating_cell(updating: str, failed: str = "") -> str:
    """The fleet cell for a row's update fields. ``""`` when it carries no move.

    The IMPORT IS FUNCTION-LOCAL on purpose, for the reason the column widths are
    not imported at all: this module keeps session internals out of its module
    scope so ``lop``'s CLI can start without paying for the runtime (see the
    header). One string formatter reached only on the arm that has a moving session
    is the whole cost of that here.

    THE PHASE IS READ, NOT ASSUMED. Both fields go through ``types.update_phase``, so
    a FAILED window renders its own cell instead of a blank one and the precedence
    between an open window, a failed one and an applied one lives in one place.
    """
    from local_operator.session.runtime.types import update_phase, update_short

    phase, pair = update_phase(updating, "", failed)
    return update_short(phase, pair) if phase else ""


def _clamp_reason_cell(summary: str, width: int | None = None) -> str:
    """A WHY cell inside :data:`WHY_COLUMN_WIDTH` CELLS, cut with the marker.

    A silent slice is indistinguishable from a complete sentence, and this
    column's whole purpose is to answer "why did this session die". The column
    was ALREADY clipping silently before the marker existed: the involuntary
    ``runtime-killed`` summary is 104 cells, and the old slice cut it at 47
    CHARACTERS — 47 cells as well, this sentence being ASCII — landing on a word
    boundary, so the row ended ``...exiting cleanly `` and read as a finished
    sentence that happened to stop mid-clause — that is what design round 2 saw as
    the one case that "fitted exactly", and it is why the marker is what keeps
    the next longer sentence honest rather than what fixes one string.

    CELLS, NOT CHARACTERS (design round 3, D9). The budget is a COLUMN width,
    and the strings reaching here are not all harness-authored: a FAILED turn's
    reason is the provider's own error text (``session.py``'s turn-end writer
    publishes ``outcome.error`` verbatim, ``attention`` replays it into the
    store, and ``completion_reason`` arrives in this cell). A localised provider
    error of 29 characters is 58 cells, so a ``len()`` comparison returned it
    UNCUT and the row rendered 208 cells against a 179-cell header — the reflow
    this clamp exists to prevent, in its own blind spot.

    The ellipsis is INSIDE the budget, so the table's fixed width and every
    other row's columns are unchanged; a summary that already fits is returned
    byte-for-byte, because a fitting cell must not pay for a cut that did not
    happen. The full sentence stays one flag away in ``--json``'s
    ``completion_reason``, which is what this column is a summary OF.

    ONE CELL MAY GO UNUSED, and that is the honest reading of "inside the
    budget" (review round 1, Q4 / design round 1, D3). The marker's own measured
    width is subtracted, so the text gets cells 1-47 — and with all-wide glyphs
    the longest prefix that fits 47 cells is 46, because a two-cell glyph cannot
    occupy an odd cell. The cell then measures 47 rather than 48. Nobody pads it:
    the last cell is unused, not missing, and inventing a space to fill it would
    report width the text does not have. A reason whose glyph mix reaches an odd
    boundary does land on 48.

    The unit being fitted is the GRAPHEME — not the code point, and not the
    character count the old rule used. A ZWJ family cluster is one 2-cell glyph,
    so a reason built from them fills the column instead of a third of it, and a
    VS16 sequence is 2 cells rather than the 1 its characters add up to (review
    round 2, M1/M2; see :func:`_cut_to_cells`).

    A summary that FITS is returned untouched, and that includes a joiner of its
    own at the end: the back-off in :func:`_cut_to_cells` is a property of a CUT,
    and a value nothing had to cut is not edited at all (review round 2, N2 —
    recorded as the rule, not changed, because trimming it would be a second,
    invisible edit on a cell that is already correct).

    ``width`` IS A PARAMETER because a second column needs the same mark (design
    review round 1, D3): the UPDATING cell is a BUILD LABEL, and a silent cut of
    ``updating → 0.59.11.dev3+g1`` hands the reader a real-looking ``0.59.11`` for a
    session that is on a different build. Everything above is about the WHY column,
    which is where the mark was first argued; the arithmetic is the same one, which
    is why this is a parameter rather than a second function. It defaults to
    ``WHY_COLUMN_WIDTH`` at CALL time rather than in the signature, because this
    function is defined above that constant.
    """
    if _cell_len(summary) <= (WHY_COLUMN_WIDTH if width is None else width):
        return summary
    # The marker's OWN measured width, not a hard-coded 1: the budget is
    # arithmetic, so a future marker must not be able to push the cell over.
    marker = "…"
    budget = WHY_COLUMN_WIDTH if width is None else width
    return _cut_to_cells(summary, budget - _cell_len(marker)) + marker


def _cut_to_cells(text: str, budget: int) -> str:
    """The longest prefix of ``text`` that fits ``budget`` display cells.

    A cell bound cannot be a slice: one East-Asian character is two cells, so
    ``text[:n]`` overshoots by however many wide glyphs it happens to contain.
    Neither can it be a walk over CHARACTERS, which is the defect this function
    was written with: rich measures a STRING, and two of its rules only fire on a
    whole sequence — a VS16 (``U+FE0F``) upgrades the glyph before it to two
    cells, and a ZWJ (``U+200D``) collapses the emoji it joins into one two-cell
    glyph. Summing ``cell_len(char)`` per character therefore MIS-measures both
    classes in opposite directions: ``❤️`` is 1 cell per character but 2 as a
    unit, so twenty of them (40 cells) came back UNCUT against a 24-cell budget
    — worse than the character rule this replaced, which clipped them at 24 —
    while a family cluster was charged about three times its width and left most
    of the column empty (review round 2, M1 and M2).

    So the unit measured is the GRAPHEME, via rich's own
    :func:`rich.cells.split_graphemes` — the splitter the measurement rules come
    from, so the unit measured is the unit emitted, and a combining mark or a
    joined emoji cannot be separated from what it belongs to.

    The returned prefix never ends on a ZERO-WIDTH JOINER. ``U+200D`` means "join
    with the glyph AFTER me", so a prefix ending on one emits a joiner with
    nothing to join — a stray control character immediately before the marker,
    which a terminal renders as a replacement box (review round 1, Q1). The
    back-off costs no cells, so the budget is unaffected and no other column
    moves; it is a no-op for well-formed text, because a grapheme absorbs an
    interior joiner together with the glyph it joins, and it fires only when the
    input's OWN trailing grapheme ends on one. A value that fits is not touched
    at all, trailing joiner included — see :func:`_clamp_reason_cell`.

    The prefix is the longest that FITS, not one that FILLS. With all-wide text
    the final cell can go unused (a two-cell glyph cannot occupy cell 47 of a
    47-cell budget). The invariant is that the result is bounded by ``budget``;
    occupying every cell is not a goal, and padding to reach it would report width
    the text does not have.

    ``split_graphemes`` is imported at the point of use, like every other
    third-party name here: this module's contract is that ``import
    local_operator.cli`` stays cheap (see the module docstring).
    """
    from rich.cells import split_graphemes

    spans, _total_cells = split_graphemes(text)
    used = 0
    for start, _end, width in spans:
        if used + width > budget:
            cut = text[:start]
            while cut.endswith("\u200d"):
                cut = cut[:-1]
            return cut
        used += width
    return text


# The two primitives the table's text columns are built from, kept beside the
# WHY column's own clamp so there is ONE cell-vs-character rule in this module
# rather than one per column.


def _fit_cell(text: str, width: int) -> str:
    """``text`` cut to ``width`` display CELLS, silently.

    Silent on purpose: these columns have always cut without a marker
    (``value[:24]``), a marker would change every ASCII listing that overflows,
    and none of them is the column whose PURPOSE is to answer a question. What
    changes here is only the bound — characters to cells.

    A value that already fits is returned byte-for-byte, and ``cell_len`` equals
    ``len`` for ASCII, so an all-ASCII table renders exactly as it did before.
    """
    return _cut_to_cells(text, width) if _cell_len(text) > width else text


def _pad_cell(text: str, width: int) -> str:
    """``text`` left-aligned in ``width`` display CELLS.

    ``f"{text:<{width}}"`` pads by CHARACTERS, so a fitted wide cell — 12 CJK
    glyphs are 24 cells — is handed ``width`` characters PLUS the blanks it never
    needed, leaving the row wider than its header in trailing space. Padding by
    cells is what keeps "every row is header-width" true rather than merely true
    for narrow text; for ASCII the two are identical.
    """
    return text + " " * max(0, width - _cell_len(text))


def _cell_len(text: str) -> int:
    """``len`` in terminal CELLS — what a fixed-width column is measured in.

    Imported at the point of use because this module's contract is that
    everything third-party stays out of its module-level imports so ``import
    local_operator.cli`` stays cheap (see the module docstring); ``rich`` is
    already a hard dependency and ``rich.cells`` is its width primitive.

    This measures a WHOLE string, which is the only way rich applies its
    sequence rules (VS16 upgrade, ZWJ collapse) — measuring per character is what
    :func:`_cut_to_cells` did and why it mis-counted both classes. The cut fits
    the same units this does, one grapheme at a time.
    """
    from rich.cells import cell_len

    return cell_len(text)


def _wake_create(args: argparse.Namespace) -> int:
    # THE READ IS THE WRITER'S NOW. Upstream built a ``Transcript`` here so the
    # append could reuse it rather than parse the journal twice; this command no
    # longer builds a list at all — ``arm_wake`` reads the transcript's latest
    # ``wake_schedules`` entry with the one-row reader and appends through the
    # object the write already needs — so that double parse is not reachable
    # from here, and the note it carried is answered rather than dropped.

    """``lop wake create <session> "<when>" "<message>"``.

    Persists through the TRANSCRIPT first, exactly like the in-session wake
    tool (``Session._persist_wake_schedules``), because the transcript entry
    is the source of truth: a session rebuilds its derived index entry from it
    on every open, so an index-only write is adopted as nothing and deleted
    by the next open (round 2, U4/Q9 — the wake was scheduled, the runtime
    started, and the self-prompt was gone). The index write and the
    install-on-demand hook ride after it, best-effort, as they do there.

    A session that does not exist is refused rather than created: the wake
    index keys on a session id, and an id with no transcript would produce a
    reminder the supervisor faithfully fires into nothing.
    """
    import time as _time

    from local_operator.harness.wake import parse_wake_at, parse_wake_duration
    from local_operator.paths import config_dir
    from local_operator.wakes.arm import WakeWriteError, arm_wake

    root = config_dir()
    session_id = str(args.session)
    now_ms = int(_time.time() * 1000)

    # The scheduling request, in the SHARED vocabulary the one validator reads
    # (``message``/``in``/``at``/``every``/``until``/``limit``). The flags and
    # their prepositions are translated here because that spelling is this
    # command's, not the schedule's. What the values MEAN — the 16-schedule
    # cap, the first FREE id slot, the interval floor, the bound-on-a-one-shot
    # rule — is decided once, by ``build_wake_schedule`` inside
    # ``wakes/arm.py``, which is also the writer the desktop arm route uses.
    request: dict[str, Any] = {"message": str(args.message)}
    raw = str(args.when).strip()
    # "in 2m" is the phrasing the help text advertises and the one a person
    # reaches for; the parsers below take the bare duration, so the leading
    # preposition is stripped here rather than taught to both of them.
    body = raw[3:].strip() if raw.lower().startswith("in ") else raw
    if raw.lower().startswith("at "):
        due_at = parse_wake_at(raw[3:].strip(), now_ms)
        request["at"] = raw[3:].strip()
    else:
        duration = parse_wake_duration(body)
        due_at = now_ms + duration if duration is not None else parse_wake_at(body, now_ms)
        # A bare token is a duration when one parses and an absolute clock or
        # an ISO instant otherwise — the same precedence ``parse_wake_at``
        # applies to a leading ``+``.
        request["in" if duration is not None else "at"] = body
    if due_at is None:
        print(f"could not read a time from {raw!r} (try 'in 2m' or 'at 09:30')", file=sys.stderr)
        return 1

    # WHAT THIS COMMAND CHECKS, AND WHAT IT DOES NOT. Everything here that
    # prints and returns is FLAG TRANSLATION: turning ``"in 2m"``/``"at 09:30"``
    # into the request's own vocabulary, choosing which key carries it, and
    # telling the user when a flag's value cannot be read as that flag's kind at
    # all. Every RULE — the 16-schedule cap, the first free id slot, the interval
    # floor, the bound-on-a-one-shot rule, the message length — is decided once,
    # by ``build_wake_schedule`` inside ``wakes/arm.py``, and reaches this command
    # as the SAME sentence the desktop route and the agent's tool print. This
    # command used to re-word the floor and the bound for itself, which made "one
    # rule, one place" half true; the ONLY check kept locally is that a repeat
    # interval is parseable as a duration, because "which of these two kinds is
    # this flag" is this command's question and nobody else's (review round 1,
    # R4).
    every_raw = str(getattr(args, "every", "") or "").strip()
    if every_raw and parse_wake_duration(every_raw) is None:
        print(
            f"could not read a repeat interval from {every_raw!r} "
            "(try '5m', '1h' or '8h30m'; a bare number is ambiguous)",
            file=sys.stderr,
        )
        return 1

    until_at: int | None = None
    until_raw = str(getattr(args, "until", "") or "").strip()
    if until_raw:
        # Both prepositions are stripped here rather than taught to the two
        # parsers, exactly as the `when` argument above does it — the help
        # promises "in 7d" and "at 09:30", so both have to reach a bare
        # duration/clock parser.
        lowered = until_raw.lower()
        if lowered.startswith("in "):
            body = until_raw[3:].strip()
        elif lowered.startswith("at "):
            body = until_raw[3:].strip()
        else:
            body = until_raw
        duration = parse_wake_duration(body)
        until_at = now_ms + duration if duration is not None else parse_wake_at(body, now_ms)
        if until_at is None:
            print(
                f"could not read an end time from {until_raw!r} (try 'in 7d' or 'at 09:30')",
                file=sys.stderr,
            )
            return 1
        # ``until`` is read by ``parse_wake_at`` — where a leading ``+`` marks a
        # duration — while the flag promises the ``in 7d`` spelling, so the
        # same translation ``when`` does above happens here.
        request["until"] = "+" + body if duration is not None else body

    limit = getattr(args, "limit", None)
    if every_raw:
        request["every"] = every_raw
    if limit is not None:
        request["limit"] = limit

    # ONE writer. ``arm_wake`` appends the transcript entry first (the source of
    # truth) and then rewrites the derived index CARRYING the keys it does not
    # own — ``stopped_at``, and the ``last_fired_at``/``last_attempt_at``
    # lateness stamps — which this command used to drop, so arming a wake on a
    # session the user had stopped silently un-parked the whole session. It also
    # verifies after the append that no other writer landed on top of it, since
    # nothing prevents a second process appending to the same transcript.
    import asyncio as _asyncio

    try:
        outcome = _asyncio.run(arm_wake(root, session_id, request, now_ms=now_ms))
    except WakeWriteError as error:
        # The refusal sentence comes from the shared validator, so the CLI, the
        # desktop route and the agent's tool say the same thing about the same
        # mistake.
        print(str(error), file=sys.stderr)
        return 1

    if getattr(args, "json", False):
        print(
            _json_dumps(
                {
                    "session_id": outcome.session_id,
                    "wake_id": outcome.wake_id,
                    "next_due_at": outcome.next_due_at,
                    "entry": outcome.index_path,
                    "supervisor": outcome.supervisor,
                }
            )
        )
        return 0
    from local_operator.wakes.display import format_wake_time

    # The row as PERSISTED (the message the validator stripped), not the flag
    # as typed: what the listing will show must be what was printed here.
    row = next((s for s in outcome.schedules if s.id == outcome.wake_id), None)
    assert outcome.next_due_at is not None  # a create always resolves a due time
    print(
        f"{outcome.wake_id}  {format_wake_time(outcome.next_due_at)} "
        f"({_format_due((outcome.next_due_at - now_ms) / 1000.0)})  "
        f"{row.message if row else request['message']}"
    )
    if outcome.supervisor:
        print(f"supervisor: {outcome.supervisor}")
    return 0


def _owed_age_s(record: "dict[str, Any] | None", now_ms: int) -> "float | None":
    """How long a fire has been owed, in seconds, or ``None`` when not owed.

    The ledger's ``first_attempt_ms`` is when the supervisor first failed to
    hand this occurrence to a runtime, so the difference is the age of the
    OWED fire rather than the age of the schedule. Rendered in the row tail
    (design round 1, D2): the state words alone cannot separate a wake four
    minutes late from one stuck since last week, and below 69 columns the WHEN
    column is gone, so the row had no age of any kind.
    """
    if not record:
        return None
    first = record.get("first_attempt_ms")
    if not isinstance(first, int) or isinstance(first, bool):
        return None
    return max((now_ms - first) / 1000.0, 0.0)


def _wake_rows() -> "list[dict[str, Any]]":
    """Every scheduled wake on this machine, soonest first.

    Reads the derived index rather than each transcript: the index exists
    exactly so this question can be answered without opening every session,
    and it is rewritten on every persist and every open, so a stale row
    self-heals rather than needing a repair path here.
    """
    import time as _time

    from local_operator.paths import config_dir
    from local_operator.wakes.deliveries import read_deliveries
    from local_operator.wakes.store import read_index
    from local_operator.wakes.supervisor import STALE_AFTER_S, _session_exists

    root = config_dir()
    now_ms = int(_time.time() * 1000)
    # The supervisor's ledger of fires it attempted and has not yet handed to a
    # runtime. Read once, here, so the row's state and the `status` summary are
    # derived from the same copy — the defect this whole PR is about was a wake
    # that had failed to fire being visible on NO surface, and two surfaces
    # disagreeing would be a smaller version of the same thing.
    deliveries = read_deliveries(root)
    rows: list[dict[str, Any]] = []
    for session_id, entry in read_index(root).items():
        if not isinstance(entry, dict):
            continue
        dormant = bool(entry.get("stopped_at"))
        # GHOST, asked with the supervisor's own predicate (round 2, Q4). The
        # supervisor refuses an entry whose session has no transcript and
        # retires on a ghost-only store, while this listing had no ghost
        # notion at all — so the rendered frame said "1 armed, 10m overdue"
        # about a wake the process had already gone home over. A painted frame
        # that contradicts the process is the defect class this PR exists to
        # remove, so the two surfaces share the predicate rather than deriving
        # it twice.
        ghost = not dormant and not _session_exists(root, session_id)
        record = deliveries.get(session_id)
        for raw in entry.get("schedules") or ():
            if not isinstance(raw, dict):
                continue
            due = raw.get("next_due_at")
            if isinstance(due, bool) or not isinstance(due, int):
                continue
            # A GHOST'S RECORD IS FROZEN, NOT WORK IN PROGRESS — the same
            # decision `_delivery_rows` makes for `status` (QA round 1, Q1).
            # Nothing can engage a session with no transcript, so this row must
            # not carry an owed age in its tail or its legend, or the listing
            # contradicts the process that has already gone home over it.
            delivery = (
                record
                if record is not None and not ghost and record.get("occurrence_ms") == due
                else None
            )
            rows.append(
                {
                    "session_id": session_id,
                    "cwd": entry.get("cwd") or "",
                    "wake_id": raw.get("id") or "",
                    "message": raw.get("message") or "",
                    "next_due_at": due,
                    "due_in_s": (due - now_ms) / 1000.0,
                    "dormant": dormant,
                    # RECURRENCE, so a listing can distinguish an automation
                    # from a one-shot. The store has always carried these; the
                    # lister dropped them, which meant a user who scheduled
                    # "regular disk cleaning" had no CLI way to confirm it
                    # repeats — in either the human or the --json form
                    # (round 4, R3/Q2/U14).
                    "every_ms": raw.get("every_ms"),
                    "until_at": raw.get("until_at"),
                    # Computed here beside `due_in_s`, against the SAME clock
                    # read: the renderer has no `now` of its own, and two
                    # clock reads in one listing can disagree across a second
                    # boundary.
                    "until_in_s": (
                        (int(raw["until_at"]) - now_ms) / 1000.0
                        if isinstance(raw.get("until_at"), int)
                        and not isinstance(raw.get("until_at"), bool)
                        else None
                    ),
                    "limit": raw.get("limit"),
                    "fired_count": raw.get("fired_count") or 0,
                    # RELIABILITY FIELDS. The three questions the operator
                    # could not previously answer about a wake that seemed not
                    # to fire: is it late right now, how late, and has it been
                    # late so long the supervisor has given up on it (past
                    # STALE_AFTER_S it is skipped, deliberately, and left to
                    # the session's own catch-up). `overdue` is a plain bool
                    # rather than "due_in_s < 0" recomputed by every consumer.
                    "overdue": due < now_ms,
                    "overdue_s": max((now_ms - due) / 1000.0, 0.0),
                    "stale": (now_ms - due) / 1000.0 > STALE_AFTER_S,
                    "ghost": ghost,
                    # Written by the session (the one writer of schedule
                    # state), absent on an entry that has not fired since the
                    # fields were added rather than defaulted to a lie.
                    "last_fired_at": entry.get("last_fired_at"),
                    "last_attempt_at": entry.get("last_attempt_at"),
                    # THE OWED FIRE, if the supervisor has one for this very
                    # occurrence. Matched on `occurrence_ms` rather than on the
                    # session alone: a record is about one attempt at one due
                    # time, and a recurring schedule whose next occurrence is
                    # already different must not inherit it.
                    "delivery": delivery,
                    # Its age, against the SAME clock read as `due_in_s`, for the
                    # `owed 9d` tail the listing renders (design round 1, D2).
                    "owed_age_s": _owed_age_s(delivery, now_ms),
                }
            )
    rows.sort(key=lambda row: row["next_due_at"])
    return rows


def _supervisor_parentheticals() -> tuple[str, str]:
    """The two ``supervisor:`` parentheticals in THIS host's supervisor's words.

    The two lines they belong to — "loaded but NOT running" and "not loaded" —
    were unreachable off macOS before this branch: the wake installer was the
    probe-documented ``FAIL wake.install`` ("no supervisor installer for this
    platform") on Linux and Windows, so no unit could exist to be reported on.
    Both platforms now have a real installer, which makes these the NORMAL
    states there after ``lop wake install`` — and a Linux user was being told
    "launchd has the job" and shown "a plist exists", neither of which names
    anything on their machine (design round 1, D4). ``plist_path()`` was already
    platform-shaped (it answers with the systemd unit path or the Task Scheduler
    definition file); the sentence around it is now too, because a status line
    that names another platform's supervisor reads as a bug in the install.

    THE macOS ENTRY IS BYTE-IDENTICAL to what it replaced, deliberately: this
    branch exists to make Windows and Linux work, not to rewrite the copy of the
    one platform that already worked.

    Returns ``(loaded_but_stopped, file_but_not_loaded)``. A function rather
    than a module constant so the ``supervisors`` import stays inside it — the
    CLI is the package's hottest import path and only this subcommand needs it.
    """
    from local_operator import supervisors

    kind = supervisors.supervisor()
    if kind == supervisors.SYSTEMCTL:
        return (
            "systemd has the unit; it has exited",
            "a unit file exists but systemd has not loaded it",
        )
    if kind == supervisors.SCHTASKS:
        return (
            "Task Scheduler has the task; it has exited",
            "a task definition exists but Task Scheduler has not registered it",
        )
    if kind == supervisors.LAUNCHCTL:
        return (
            "launchd has the job; it has exited",
            "a plist exists but launchd has no job",
        )
    # ``supervisor()`` answered ``None``: no supervisor exists on this host, so
    # neither line below can print (``is_supported()`` gates both). Answer in
    # nouns that name no platform rather than guessing one.
    return (
        "a supervisor has the job; it has exited",
        "a job definition exists but no supervisor has it",
    )


def wake_command(args: argparse.Namespace) -> int:
    """``lop wake status|list|serve`` — scheduled wakes and their supervisor.

    The question this answers is "will my reminder actually fire", which
    before the supervisor had an uncomfortable answer: only if a session
    happened to be open. ``status`` is therefore the important subcommand —
    it reports whether the thing that fires wakes for closed sessions exists.
    """
    from local_operator.paths import config_dir

    # BOTH branches render a wake's delivery state — `list` as the DUE-column
    # word, `status` as prose — so the two words are named once, here, rather
    # than imported into one branch and silently unbound in the other.
    from local_operator.wakes.deliveries import (
        STATE_RETRYING,
        STATE_UNDELIVERED,
        UNDELIVERED_AFTER_ATTEMPTS,
    )

    command = getattr(args, "wake_command", None) or "status"

    if command == "serve":
        # The foreground form of what the LaunchAgent runs. Useful on its own
        # for anyone who would rather run it under their own supervisor.
        import asyncio as _asyncio

        from local_operator.wakes.supervisor import serve

        return _asyncio.run(serve(config_dir(), once=bool(args.once)))

    if command == "create":
        return _wake_create(args)

    if command == "list":
        rows = _wake_rows()
        if args.json:
            print(_json_dumps(rows))
            return 0
        if not rows:
            print("no scheduled wakes")
            return 0
        import shutil

        from local_operator.harness.wake import format_duration
        from local_operator.wakes.display import format_wake_time

        # A FIXED-WIDTH TABLE, matching `lop sessions` right next door rather
        # than inventing a second listing convention. Round 1 (D4): padding a
        # pre-composed "<abs time> (<rel>)" cell never applies, because that
        # cell is already 19-31 characters wide, so the id column started at a
        # different position on every row (measured: 33/27/21/22/29) and a long
        # message produced a 155-column line that wrapped with no indent.
        # Splitting the two time facts into their own columns is what makes the
        # padding mean something.
        term_width = shutil.get_terminal_size((80, 24)).columns
        # A RENDERED TIME IS NEVER TRUNCATED (round 2, D11). `format_wake_time`
        # chooses its own form — a clock alone for today, a date for another
        # day, a year as well for another year — so its width is 11 to 24
        # characters depending on the wake, and a fixed 18 cut `Jan 01 2027
        # 9:00 AM EST` to `Jan 01 2027 9:00 A`: a half meridiem, no zone, and
        # no marker to say anything was dropped. A time that is wrong is worse
        # than a time that is absent, and unlike a message (which the reader
        # can recognise from its start) there is no recovering a mangled clock.
        #
        # So the column is sized from THE ROWS BEING RENDERED, and the
        # renderer's output is printed whole or not at all.
        when_cells = {row["next_due_at"]: format_wake_time(row["next_due_at"]) for row in rows}
        when_w = max(len(cell) for cell in when_cells.values())
        rel_w = 11  # "10m overdue"
        # SLACK for the id (round 2, R9). Ids are `uuid4().hex[:12]`, and a
        # 12-wide column truncated a 12-character id to exactly itself with no
        # room to show that anything was cut — while the 15-character `lr_` ids
        # that appear in real stores rendered as a DIFFERENT id an operator
        # cannot paste back. 13 gives a real id slack, and `_elide_id` marks
        # anything longer instead of silently shortening it.
        id_w = 13
        fixed = rel_w + id_w + 2
        message_floor = 24
        # WHEN IS THE COLUMN THAT YIELDS on a narrow terminal (round 2, D12 /
        # QA Q1), because the relative `DUE` cell answers "when does this fire"
        # in 11 characters and the absolute time is the redundant half. Dropping
        # it keeps the table aligned at 60 columns instead of wrapping every
        # row, which is the trade D12 argued for — shed the absolute time
        # rather than the alignment.
        show_when = term_width >= when_w + fixed + 1 + message_floor
        head = fixed + (when_w + 1 if show_when else 0)
        message_w = max(message_floor, term_width - head - 1)
        header = f"{'WHEN':<{when_w}} " if show_when else ""
        # WHAT A NARROW TERMINAL ACTUALLY GETS, measured rather than intended.
        # `message_w` is a budget for the message column, and the row also
        # carries a state TAIL (` · every 20m, 12/40 fired`, 25 characters on
        # the widest real row) that the budget does not include — `room` below
        # subtracts it and can go negative. So on a terminal narrower than the
        # row needs, BOTH things happen at once:
        #
        #   * the row overflows anyway — measured 50-53 columns at COLUMNS=40,
        #     53 being a row whose tail is 25 characters; and
        #   * the message degrades to one character plus an ellipsis (`r…`),
        #     because `room` is negative and the clamp floors at 1.
        #
        # 53 columns is the point at which the widest tailed row stops
        # overflowing (measured across 40/44/48/50/51/52/53/54/58/60 on the
        # design round's own fixture), and it is IRREDUCIBLE for that row:
        # 26 fixed + 1 gap + 25 tail + 1 message character = 53. A wider
        # message budget cannot help — it makes the row longer, not shorter —
        # and the only lever that would is clamping the tail, which round 5
        # (R6/U16) deliberately forbade because the tail carries the bounds a
        # user cannot be expected to remember. Narrower terminals therefore
        # wrap, and the message column really does stop saying anything; the
        # honest fix is a different layout for that width, not a budget tweak.
        print(f"{header}{'DUE':>{rel_w}} {'SESSION':<{id_w}} WAKE")
        for row in rows:
            when = when_cells[row["next_due_at"]]
            # DORMANT WINS, and the overdue/stale marks are suppressed under it
            # (round 1, Q2/R6). A dormant wake is one nothing is SUPPOSED to
            # fire, so "(OVERDUE)" — which means "should have fired and did
            # not" — contradicts it, and "no longer fired by the supervisor" is
            # true of a dormant wake for an entirely different reason
            # (reopening the session fires it; nothing revives a stale one).
            # THE STATE WORD LIVES IN THE DUE COLUMN, and the row carries no
            # prose repeating it — this is a table, like `lop sessions`, and
            # the explanation of what "stale" or "dormant" costs belongs on the
            # `status` summary that has room for a sentence. Keeping both put a
            # 155-column line in an 80-column terminal (round 1, D4) and
            # restated on every row what the reader needs told once.
            if row["dormant"]:
                state = "dormant"
            elif row["ghost"]:
                # Ghost before stale: with no session on disk, nothing can fire
                # this at all, which is a stronger statement than "the
                # supervisor stopped retrying" (round 2, Q4).
                state = "ghost"
            elif row.get("delivery"):
                # AN OWED FIRE IS WORK IN PROGRESS. The supervisor has attempted
                # this occurrence, failed, and is retrying it from that record —
                # which is also the reason a row PAST the staleness bound is
                # still being fired at all. Reading it as `stale` would state
                # the opposite of what the process is doing, and the legend
                # under the table promises `stale` wakes are not fired.
                state = (
                    "undelivered"
                    if row["delivery"].get("state") == STATE_UNDELIVERED
                    else "retrying"
                )
            elif row["stale"]:
                state = "stale"
            else:
                state = _format_due(row["due_in_s"])
            mark = ""
            # `every …` reuses the same renderer the tool listing and the wake
            # panel use, so one wake reads identically wherever it is shown.
            repeat = ""
            if row.get("every_ms"):
                repeat = f" · every {format_duration(int(row['every_ms']))}"
                if row.get("limit"):
                    repeat += f", {int(row['fired_count'])}/{int(row['limit'])} fired"
                elif row.get("fired_count"):
                    repeat += f", {int(row['fired_count'])} fired"
                # The TIME bound, shown for the same reason the count bound is:
                # a `--limit` wake advertised its bound while an `--until` one
                # was indistinguishable from an unbounded repeat, so the bound
                # a user is most likely to forget was the one not rendered
                # (round 5, R6/U16). Relative, matching the `when` column —
                # "until in 7d" answers "is this still running next week"
                # without the reader converting a timestamp.
                left = row.get("until_in_s")
                if left is not None:
                    repeat += f", until {_format_due(left)}" if left > 0 else ", expired"
            elif not row.get("every_ms"):
                # The TUI wake panel says `once` for a non-recurring schedule
                # and this listing said nothing, so the same wake read
                # differently in two places (round 1, D9).
                repeat = " · once"
            owed_age = row.get("owed_age_s")
            if owed_age is not None:
                # THE AGE GOES IN THE TAIL, where the other bounds already live
                # and where the round-5 R6/U16 rule says it is never clamped:
                # `retrying owedstale001` (9 days) and `retrying owedretry001`
                # (4 minutes) were otherwise identical rows at any width that
                # drops WHEN (below 69 columns). The message is what gives —
                # the part of the row the reader already knows — so the row's
                # total width is unchanged and only the prose shortens.
                repeat += f", owed {_format_duration(owed_age)}"
            if row.get("last_fired_at"):
                repeat += f", last fired {format_wake_time(int(row['last_fired_at']))}"
            # THE MESSAGE IS WHAT GETS CLAMPED, never the state tail. Clamping
            # the composed string instead would silently drop `until in 6d`,
            # `3/5 fired` or `(stale — …)` off the end of a long row — exactly
            # the bounds an earlier round added because a user cannot be
            # expected to remember them (round 5, R6/U16). A user-authored
            # message is the one part of the row they already know.
            tail = f"{repeat}{mark}"
            message = row["message"]
            # `room` GOES NEGATIVE on a narrow terminal, because `message_w` is
            # a budget for the whole cell and the tail is subtracted from it
            # here rather than reserved there. The `max(..., 1)` then floors
            # the message at one character plus an ellipsis (`r…`) and the row
            # overflows regardless — see the width note above the header for
            # the measured numbers and why no budget change fixes it.
            room = message_w - len(tail)
            if len(message) > room:
                message = message[: max(room - 1, 1)] + "…"
            detail = f"{message}{tail}"
            when_cell = f"{when:<{when_w}} " if show_when else ""
            print(
                f"{when_cell}{state:>{rel_w}} "
                f"{_elide_id(row['session_id'], id_w):<{id_w}} {detail}".rstrip()
            )

        if not show_when:
            # The omission is STATED, not silent: the absolute time was dropped
            # to keep the table aligned on a narrow terminal (round 2, D12), and
            # the reader is told where it went rather than left to notice.
            #
            # AND IT IS SAID IMMEDIATELY (design round 1, D6): it used to print
            # after the legend block, 14 rows below the table at 60 columns, so
            # the reader who noticed the missing column had to scroll past three
            # legends to learn why.
            print("\n(WHEN hidden — terminal too narrow)")

        # ONE legend under the table rather than the same sentence on every
        # row, and only for the states actually present: what "stale" and
        # "dormant" COST is the thing a reader needs told, but telling it per
        # row is what made a single wake occupy 155 columns.
        #
        # AN OWED FIRE IS EXCLUDED FROM `stale`, exactly as it is from that
        # count on `status`: the stale sentence promises the session's next open
        # will deliver the wake, and for a fire the supervisor is still retrying
        # that is not the plan. It gets its own words instead — `retrying` after
        # a failed attempt, `undelivered` past `UNDELIVERED_AFTER_ATTEMPTS`,
        # where the attempt count is what tells the reader whether this is a busy
        # host or a session that will never construct (`lop wake status` carries
        # both figures).
        legend_rows: list[tuple[str, str]] = []

        def _owed_state(state: str) -> bool:
            return any(
                row.get("delivery") and row["delivery"].get("state") == state for row in rows
            )

        if _owed_state(STATE_RETRYING) or _owed_state(STATE_UNDELIVERED):
            # THE OWED PAIR COMES FIRST (design round 1, D8). They are the two
            # states an operator has to act on; the three below them all mean
            # "the supervisor is not firing this at all", so a reader scanning
            # for what `retrying` means used to pass every one of those first.
            #
            # EACH WORD GETS ONE SHORT CLAUSE AND THE REST IS SAID ONCE (D5).
            # The two legends used to repeat ~100 characters of the same
            # sentence three lines apart, and the clause that actually separates
            # them — below vs at the stalled-fire threshold — was buried
            # mid-sentence in each; at 60 columns that was 21 rows of legend
            # under a 6-row table, past a standard screen.
            # Classification owns this boundary: retries 2–4 are not "one
            # failed attempt", and copy must follow future threshold changes.
            if _owed_state(STATE_RETRYING):
                legend_rows.append(
                    ("retrying", f"fewer than {UNDELIVERED_AFTER_ATTEMPTS} failed attempts.")
                )
            if _owed_state(STATE_UNDELIVERED):
                legend_rows.append(
                    ("undelivered", f"{UNDELIVERED_AFTER_ATTEMPTS}+ failed attempts.")
                )
            legend_rows.append(
                (
                    "",
                    "these are still owed and retried with a backoff; 'lop wake status' has "
                    "the attempts and the error",
                )
            )
        if any(
            row["stale"] and not row["dormant"] and not row["ghost"] and not row.get("delivery")
            for row in rows
        ):
            legend_rows.append(
                (
                    "stale",
                    "the supervisor no longer fires these; they are delivered when "
                    "their session is next opened",
                )
            )
        if any(row["ghost"] for row in rows):
            legend_rows.append(
                (
                    "ghost",
                    "no session with this id exists on disk; nothing can fire these, and "
                    "nothing clears them automatically",
                )
            )
        if any(row["dormant"] for row in rows):
            legend_rows.append(
                ("dormant", "the session was stopped; reopening it re-arms its wakes")
            )

        if legend_rows:
            # THE LEGEND COLUMN IS SIZED FROM THE WORDS ACTUALLY RENDERED, and
            # the fold follows it (round 2, D12) so a legend wraps the way the
            # rows do. Every earlier word was at most 8 characters, so a fixed
            # 9-wide column was invisible; `undelivered` is 11 and a fixed column
            # ran the label straight into its sentence ("undeliveredthe
            # supervisor has…"), which is the one line that explains a state. A
            # word-less row is the shared continuation (design round 1, D5): it
            # aligns under the labels so the shared clause reads as belonging to
            # both words above it.
            import textwrap

            legend_w = max(len(word) for word, _ in legend_rows) + 1
            for word, text in legend_rows:
                print()
                for line in textwrap.wrap(
                    f"{word:<{legend_w}}{text}",
                    width=term_width,
                    subsequent_indent=" " * legend_w,
                ):
                    print(line)

        return 0

    # status
    from local_operator.wakes.install import (
        ensure_supervisor_installed,
        is_supported,
        plist_path,
        supervisor_state,
        uninstall,
    )

    if getattr(args, "uninstall", False):
        outcome = uninstall()
        # `_wrap_status`, not a bare f-string: an unwrapped reason that runs
        # past the terminal width continues at column 0 and reads as a new
        # line of output rather than as the rest of this one, and it sits a
        # cell out of line with the wrapped status lines below it (design
        # round 2, D12). Every reason here can be long: they name a path.
        print(_wrap_status(f"supervisor: {outcome.reason}"))
        return 0

    # NOT `harness.wake.format_duration` here: the status lines use this
    # command's own single-unit ladder (`_format_duration`), and importing the
    # compound one alongside it was dead weight flake8 cannot see through a
    # function-local import (round 1, R5).
    from local_operator.wakes.supervisor import STALE_AFTER_S

    stale_after_days = STALE_AFTER_S / 86400.0

    rows = _wake_rows()
    # `install` as a subcommand and `status --install` are the same operation;
    # the hook is idempotent and now REPAIRS, so both routes reach the fix.
    wants_install = command == "install" or getattr(args, "install", False)
    if wants_install:
        outcome = ensure_supervisor_installed(config_dir())
        print(_wrap_status(f"supervisor: {outcome.reason}"))

    # RUNNING, not merely present. `plist_path().exists()` was an even weaker
    # test than the install hook's `_is_loaded()` — it reported "installed"
    # for a supervisor that had exited, which is the blind spot that let armed
    # wakes sit unfired. The file is still reported separately, because "the
    # plist is there but nothing runs" is a distinct, actionable state.
    state = supervisor_state(config_dir()) if is_supported() else None
    plist_present = is_supported() and plist_path().exists()
    # UNVERIFIABLE is a third state, distinct from stopped and from absent:
    # the store being asked about is not one launchd can supervise (an
    # isolated run, a config dir outside the real home), so the global label
    # answers about a DIFFERENT store. Nothing about it may be rendered here.
    verifiable = bool(state and state.verifiable)
    running = bool(state and state.running and state.verifiable)
    uptime_s: float | None = None
    if verifiable and state and state.pid:
        uptime_s = _process_uptime_s(state.pid)

    # FIREABLE is the classification the whole screen now hangs off (round 1,
    # D2). A wake that is dormant or stale will not be fired by the supervisor,
    # so counting it as "next" answered "when will my wake fire?" with a date
    # nine days in the past, on a row that is never coming, while the wake due
    # in three minutes was absent from the screen entirely.
    armed = [row for row in rows if not row["dormant"]]
    dormant = [row for row in rows if row["dormant"]]
    # GHOST sits beside stale as a reason the supervisor will not fire a row
    # (round 2, Q4): an index entry whose session has no transcript can never
    # be engaged, and the supervisor retires on a ghost-only store. Excluded
    # from `fireable` for exactly the same reason stale is.
    ghost = [row for row in armed if row["ghost"]]
    # AN OWED FIRE STAYS FIREABLE PAST THE STALENESS BOUND. The supervisor
    # retries it from its own record — that is the whole exception the ledger
    # introduces — so counting such a row as `stale` would put this frame back
    # in contradiction with the process: `stale` promises the wake is left to
    # the session's next open, which is not the plan for a fire something is
    # still working on.
    stale = [row for row in armed if row["stale"] and not row["ghost"] and not row.get("delivery")]
    fireable = [
        row for row in armed if (not row["stale"] or row.get("delivery")) and not row["ghost"]
    ]
    overdue = [row for row in fireable if row["overdue"]]
    upcoming = fireable  # already sorted soonest-first by `_wake_rows`

    # Probed once per session that has a wake, not per row: `wedged_runtime`
    # reads the registry, and a session with three schedules is still one
    # process. Only sessions with something armed are worth asking about — a
    # dormant session's runtime being wedged is not why its wake is not firing.
    # Each call is a `registry.scan`, which walks the run directory AND reaps
    # records for dead processes (round 2, R10) — cheap and idempotent at this
    # scale (measured 19 sessions / 28 ms), but not a free read.
    from local_operator.wakes.supervisor import wedged_runtime

    wedged: list[tuple[str, int, float]] = []
    for session_id in dict.fromkeys(row["session_id"] for row in armed):
        found = wedged_runtime(config_dir(), session_id)
        if found is not None:
            wedged.append((session_id, found[0], found[1]))

    # THE OWED FIRES, which nothing on this surface could report before: the
    # supervisor engaged a due wake, could not hand it to a runtime, and the
    # only trace was a WARNING in wake-supervisor.log (510 lifetime on this
    # machine, 128 in one day) while every other screen showed the wake as an
    # ordinary overdue row. The record behind this line is the one the
    # supervisor retries from, so the count is what the supervisor owes rather
    # than a re-derivation of it — the two cannot disagree.
    def _delivery_rows(state: str | None) -> "list[dict[str, Any]]":
        # GHOSTS ARE EXCLUDED (QA round 1, Q1). An owed record whose session has
        # no transcript on disk is frozen: `_engage_one` refuses a ghost before
        # any attempt, so nothing retries it and nothing will. Reporting it as
        # "still owed and retried with backoff" beside the `ghost:` line (which
        # says nothing can fire it) made this surface argue with itself, and
        # `--json` said `retrying: 1` with `overdue: 0`. The record is still on
        # disk and still the operator's to delete; it is simply not work in
        # progress, which is the only thing this bucket claims.
        return [
            row
            for row in armed
            if row.get("delivery")
            and not row["ghost"]
            and (state is None or row["delivery"].get("state") == state)
        ]

    owed = _delivery_rows(None)
    stalled = _delivery_rows(STATE_UNDELIVERED)
    retrying = _delivery_rows(STATE_RETRYING)

    def _attempts_label(row: dict[str, Any]) -> str:
        """``"4 attempt(s) since <time>"`` for one owed fire.

        The age is the part that matters: whether a fire has been retrying for
        a minute or for an hour is what tells the operator whether this is the
        loaded-host case or a session that will never construct. The timestamp
        renderer is imported HERE rather than hoisted: this status surface
        otherwise renders every time relatively (its own ``_format_due``
        ladder), and importing the absolute one at the top would put a name in
        scope that only this helper uses.
        """
        from local_operator.wakes.display import format_wake_time

        record = row["delivery"]
        attempts = record.get("attempts")
        count = attempts if isinstance(attempts, int) and not isinstance(attempts, bool) else 0
        first = record.get("first_attempt_ms")
        if isinstance(first, int) and not isinstance(first, bool):
            return f"{count} attempt(s) since {format_wake_time(first)}"
        return f"{count} attempt(s)"

    def _next_attempt_s(row: dict[str, Any]) -> float | None:
        nxt = row["delivery"].get("next_attempt_ms")
        if not isinstance(nxt, int) or isinstance(nxt, bool):
            return None
        return (nxt - int(time.time() * 1000)) / 1000.0

    def _retry_clause(nxt: float | None) -> str:
        """``, next attempt in 3m`` / ``, retry due now`` / ``""``.

        ``next_attempt_in_s`` is the recorded attempt minus the clock this run
        read, so it is SIGNED and design round 1 (D3) caught the old rendering:
        the clause was dropped whenever the value was not positive, which is
        exactly the state where the retry is already due — so the line that must
        answer "is this being retried?" ended at `retried with backoff` with no
        when at all, while the neighbouring `undelivered:` line printed one.
        """
        if nxt is None:
            return ""
        return f", next attempt in {_format_duration(nxt)}" if nxt > 0 else ", retry due now"

    # An ENUM plus the human sentence, not a sentence alone (round 1, D6): a
    # monitoring consumer branching on `state` had to string-match prose, and
    # the booleans do not distinguish `stopped` from `not_loaded`.
    if not is_supported():
        supervisor_state_name = "unsupported"
    elif not verifiable:
        supervisor_state_name = "unverifiable"
    elif running:
        supervisor_state_name = "running"
    elif state and state.loaded:
        supervisor_state_name = "stopped"
    elif plist_present:
        supervisor_state_name = "not_loaded"
    else:
        supervisor_state_name = "not_installed"

    payload = {
        "supported": is_supported(),
        # Kept as "a supervisor is in place" for readers that already parse
        # it, but it now means RUNNING rather than "a file exists".
        "installed": running,
        "plist": str(plist_path()) if is_supported() else "",
        "scheduled": len(rows),
        "armed": len(armed),
        "dormant": len(dormant),
        # NAMED FOR WHAT THEY MEAN (round 2, D18). `next_due_in_s` silently
        # changed meaning in round 1 — it began excluding stale rows, which is
        # what D2 asked for, under a name that still reads "the soonest due
        # wake" — and a consumer wanting the raw value had nowhere to get it.
        # Both are published: the fireable one under an explicit name, the raw
        # one under the original name so an existing consumer keeps parsing.
        "next_due_in_s": rows[0]["due_in_s"] if rows else None,
        "next_fireable_due_in_s": upcoming[0]["due_in_s"] if upcoming else None,
        "supervisor": {
            # False whenever the answer would be about another store, so a
            # monitoring caller cannot read this block as a verdict on THIS
            # one. The pid is withheld for the same reason.
            "verifiable": verifiable,
            "state": supervisor_state_name,
            "running": running,
            "loaded": bool(state and state.loaded and verifiable),
            "plist_present": plist_present,
            "pid": state.pid if (verifiable and state) else None,
            "uptime_s": uptime_s,
            "detail": state.detail if state else "",
        },
        # `overdue` counts FIREABLE rows only, which is what makes it
        # reconcilable: `scheduled` = fireable + dormant + stale + ghost, and
        # `overdue` is a subset of the fireable ones. Round 2 (D18) noted a
        # consumer had no way to tell which rows were inside it; the
        # `unfireable` block below is that breakdown.
        "overdue": len(overdue),
        "stale": len(stale),
        "ghost": len(ghost),
        # Why each non-firing row will not fire, so the counts above can be
        # reconciled without re-deriving the classification.
        "unfireable": {
            "dormant": [row["session_id"] for row in dormant],
            "stale": [row["session_id"] for row in stale],
            "ghost": [row["session_id"] for row in ghost],
        },
        "max_overdue_s": max((row["overdue_s"] for row in overdue), default=0.0),
        # The wedged case: a runtime whose process is alive but whose
        # heartbeat has gone stale holds the transcript lease without serving,
        # so its wake cannot fire and the supervisor's only trace was an
        # ordinary timeout. Reported, never repaired — see `wedged_runtime`.
        "wedged": [
            {"session_id": session_id, "pid": pid, "heartbeat_age_s": age}
            for session_id, pid, age in wedged
        ],
        # The owed fires, one entry per wake whose delivery is outstanding.
        # Additive: a monitoring consumer that predates this block keeps
        # parsing the counts above.
        "undelivered": len(stalled),
        "retrying": len(retrying),
        # AND THEY ARE A SUBSET OF `overdue`, stated as such (design round 1,
        # D4): every owed fire is overdue by construction, so `overdue` +
        # `retrying` + `undelivered` counted the same wakes twice and a consumer
        # adding the keys got 6 of 5. The flat keys above stay for the readers
        # that already parse them; this block is the one that says what they are.
        "owed": {
            "subset_of": "overdue",
            "total": len(owed),
            "retrying": len(retrying),
            "undelivered": len(stalled),
        },
        "deliveries": [
            {
                "session_id": row["session_id"],
                "message": row["message"],
                "wake_id": row["wake_id"],
                "occurrence_ms": row["delivery"].get("occurrence_ms"),
                "state": row["delivery"].get("state"),
                "attempts": row["delivery"].get("attempts"),
                "first_attempt_ms": row["delivery"].get("first_attempt_ms"),
                "last_attempt_ms": row["delivery"].get("last_attempt_ms"),
                "next_attempt_in_s": _next_attempt_s(row),
                "last_error": row["delivery"].get("last_error"),
            }
            for row in owed
        ],
    }
    if args.json:
        print(_json_dumps(payload))
        return 0

    if state is not None and not verifiable:
        # Never another store's pid. Saying "running" here would be the very
        # failure this command exists to remove, one level up: it would report
        # wakes as supervised while the running process watches a different
        # store. Observed during validation, where an isolated run printed the
        # operator's real LaunchAgent pid.
        print(_wrap_status("cannot be verified for this store", "supervisor:"))
        print(_wrap_status(f"({state.detail})"))
        print(_wrap_status("(wakes here fire only while a session is open)"))
    elif running:
        detail = f"running (pid {state.pid})" if state and state.pid else "running"
        # Suppressed under a minute: `up 0s` on a just-started supervisor is
        # noise, and the pid already says it is there (round 1, D9).
        if uptime_s is not None and uptime_s >= 60:
            detail += f", up {_format_duration(uptime_s)}"
        print(_wrap_status(detail, "supervisor:"))
    elif state and state.loaded:
        # The exact state that produced the permanent misses: the supervisor
        # knows the job, its own query returns 0, and nothing is running. The
        # parenthetical carries the SUPERVISOR'S OWN fact rather than repeating
        # the state word it was meant to disambiguate (round 1, D9), and it is
        # spelled for the supervisor answering here (round 1, D4).
        _loaded_but_stopped, _ = _supervisor_parentheticals()
        print(_wrap_status(f"loaded but NOT running ({_loaded_but_stopped})", "supervisor:"))
    elif plist_present:
        _, _file_but_unloaded = _supervisor_parentheticals()
        print(_wrap_status(f"not loaded ({_file_but_unloaded})", "supervisor:"))
    else:
        print(_wrap_status("not installed", "supervisor:"))
    # ONE remedy line, not two (round 1, Q3/D7). Both the per-state hint and
    # the ACTIONABLE branch used to fire in the not-loaded state, printing
    # `run 'lop wake install'` twice in a four-line block.
    if is_supported() and verifiable and not running:
        if rows:
            print(
                _wrap_status(
                    "(nothing will fire these while their sessions are closed — "
                    "run 'lop wake install')"
                )
            )
        else:
            print(_wrap_status("(run 'lop wake install')"))
    if not is_supported():
        # Honest rather than reassuring: on a platform with no installer the
        # wakes of a CLOSED session do not fire, and saying so is the whole
        # point of this line.
        print(
            _wrap_status(
                "(no installer for this platform — wakes fire only while a session is open)"
            )
        )

    # DORMANCY IS NAMED (round 1, D3). `scheduled: 1 (0 armed)` with no word of
    # explanation was a dead end for the operator asking "why has nothing
    # fired?", while `wake list` did state the reason.
    counts = f"{len(armed)} armed"
    if dormant:
        counts += f", {len(dormant)} dormant"
    print(f"scheduled:   {len(rows)} ({counts})")
    if dormant and not armed:
        print(_wrap_status("(dormant — their sessions were stopped; reopening one re-arms it)"))

    # ONE LINE PER DISTINCT STATE (round 1, D5; sharpened in round 2, D16).
    # `next:` names a wake in the FUTURE; an already-late one is reported by
    # `overdue:` below, because labelling something late as "next" promises a
    # future event and says twice what the line below already says (D16).
    #
    # SELECT the soonest future row rather than testing the head of the list
    # (round 3, D19). `upcoming` is sorted soonest-first, so gating on
    # `upcoming[0]` let ONE overdue row suppress `next:` for every future wake
    # — a wake a minute away went unnamed on a surface README promises reports
    # "the soonest wake that will fire".
    future = [row for row in upcoming if not row["overdue"]]
    if future:
        soonest = future[0]
        print(_wrap_status(f"{_format_due(soonest['due_in_s'])}  {soonest['message']}", "next:"))
    if overdue:
        # Counted over FIREABLE rows only, so "worst" is a wake that is
        # actually coming rather than one the supervisor has given up on. The
        # soonest overdue row is named, since with nothing in the future this
        # is the line that answers "what is the supervisor working on".
        worst = max(row["overdue_s"] for row in overdue)
        summary = f"{len(overdue)} (worst {_format_duration(worst)})  {overdue[0]['message']}"
        if owed:
            # THE SUBSET IS STATED WHERE THE COUNTS ARE (design round 1, D4).
            # Every owed fire is overdue by construction, so `overdue` already
            # contains the `retrying:`/`undelivered:` lines below and an operator
            # adding the three counted the same wakes twice (6 of 5 on the
            # designer's store). One clause, on the line a reader is already
            # doing the arithmetic against.
            summary += f" — {len(retrying)} retrying, {len(stalled)} undelivered"
        print(_wrap_status(summary, "overdue:"))
    if stalled:
        # THE DEEPEST-FAILING ONE, because a count alone cannot distinguish "a
        # wake is 20 seconds late on a busy host" from "a session has not
        # constructible for an hour", and only the second is worth attention.
        deepest = max(stalled, key=lambda row: row["delivery"].get("attempts") or 0)
        print(
            _wrap_status(
                f"{len(stalled)} — {deepest['session_id']} {deepest['message']!r} could not be "
                f"handed to a runtime: {_attempts_label(deepest)}"
                f"{_retry_clause(_next_attempt_s(deepest))} (last error: "
                f"{deepest['delivery'].get('last_error') or 'unknown'}). It is STILL OWED "
                "and retried with backoff.",
                "undelivered:",
            )
        )
    if retrying:
        soonest = min(
            retrying,
            key=lambda row: row["delivery"].get("next_attempt_ms") or 0,
        )
        print(
            _wrap_status(
                f"{len(retrying)} — {soonest['session_id']} {soonest['message']!r}: "
                f"{_attempts_label(soonest)}, retried with backoff"
                f"{_retry_clause(_next_attempt_s(soonest))}",
                "retrying:",
            )
        )
    if stale:
        print(
            _wrap_status(
                f"{len(stale)} past {int(stale_after_days)}d — delivered when their "
                "sessions are next opened",
                "stale:",
            )
        )
    if ghost:
        # The supervisor refuses these and retires on a ghost-only store
        # (round 2, Q4). Saying so here is what stops this frame contradicting
        # the process.
        #
        # THE REMEDY IS THE FILE (round 3, D20). The earlier wording said the
        # entry "is removed when that session id is next written", which no
        # reader can bring about: both callers of `store.remove_entry` need a
        # live session (a persist with no schedules left, or `cleanup` after
        # the session directory goes away), and `lop wake` exposes no cancel.
        # So a ghost row is permanent, and the only action available is to
        # delete the entry file — which is what the line now names.
        # RELATIVE, not the absolute path: an absolute config root is one
        # unbreakable token long enough to overflow a narrow terminal by
        # itself, which is the thing D21 asked to stop. `wakes/<id>.json` is
        # the form the design round suggested, and the ids it needs are on
        # screen in `lop wake list`.
        print(
            _wrap_status(
                f"{len(ghost)} with no session on disk — nothing can fire these, and "
                "nothing clears them automatically; delete its "
                "wakes/<session-id>.json entry to remove one",
                "ghost:",
            )
        )
    for session_id, pid, age in wedged:
        # Named on its own line because the remedy is a DIFFERENT command from
        # every other non-running state (round 2, D14): the surface points at
        # `lop wake install` elsewhere, and here no wake-subsystem action can
        # help at all, so it names the two commands that can.
        #
        # The remedy names the ladder's FIRST rung, with the forced rung's cost
        # beside it (design round 2, D3). A lapsed beat — which is what this
        # line describes — is exactly what `_identity_by_start_time` admits, so
        # `lop stop --pid N` is the rung that acts on this state; `--force`
        # re-reads the record only while the beat is still inside
        # `HEARTBEAT_TIMEOUT_S` (`control._identity_by_record`), so here it cannot
        # convert a refusal into a stop. It stays named for the shape it IS for —
        # an owner that is still beating but silent — with what running it
        # does. "It will not recover on its own" stays gone with it: a stale
        # beat is evidence the owner stopped reporting, not a forecast about a
        # long turn that may simply finish (``registry.classify``).
        print(
            _wrap_status(
                f"{session_id} (pid {pid}) has not sent a heartbeat in "
                f"{_format_duration(age)} and is not answering its socket; it holds "
                f"the session lease, so its wake cannot fire while it does not "
                f"answer. Nothing here can recover it — 'lop sessions' shows it; "
                f"'lop stop --pid {pid}' asks it to stop and signals it if it will "
                f"not answer, while 'lop stop --pid {pid} --force' signal-stops the "
                f"process, discarding its in-flight turn, for an owner that is "
                f"still beating but silent",
                "wedged:",
            )
        )
    return 0


def _elide_id(session_id: str, width: int) -> str:
    """A session id that fits, and that SAYS SO when it does not.

    Round 2 (R9): `session_id[:12]` in a 12-wide column rendered a
    15-character id as a different, shorter id — one an operator cannot paste
    back into `lop wake list` or `lop stop`, with nothing marking it as cut.
    Truncating an identifier is not like truncating a message: the reader can
    recognise a message from its opening words and cannot reconstruct an id.
    Ids are `uuid4().hex[:12]` so the column fits every id the product mints;
    anything longer is marked rather than silently shortened.
    """
    if len(session_id) <= width:
        return session_id
    return session_id[: max(width - 1, 1)] + "…"


#: Width of `lop sessions`' leading STATE column, in display CELLS.
#:
#: Sized for the one human PHRASE it can carry rather than for the tokens
#: around it (``wedged`` is 6 cells, ``not answering`` 13). The column used to
#: be 7 and printed the raw token, which is fine for a script and not for the
#: one person-facing place the wake surfaces send a reader to (design round 2,
#: D5; QA Q1). ``--json``'s ``state`` key is untouched: it is the wire value.
STATE_COLUMN_WIDTH = 13


def _state_cell(state: str) -> str:
    """The STATE cell for a person, from the state token a machine reads.

    ``wedged`` is the one value in this column that is a verdict rather than a
    word: ``lop wake status`` ends its wedge line with "'lop sessions' shows
    it", so this table is where an operator arrives, and it was the only
    person-facing surface where the token stood with no sentence to qualify it.
    It reads as ``not answering`` — the phrase every other surface uses, and the
    one the adjacent ``HB_AGE`` column measures — while ``--json`` keeps
    ``wedged`` for the ~15 call sites and the desktop catalogue's
    ``status.code`` that branch on it.

    ``stale`` deliberately keeps its token: it means the record's pid is GONE,
    which is a different fact from this one rather than a longer way of saying
    the same thing, and the state whose wording this change is about is the one
    where the process is still there.
    """
    return "not answering" if state == "wedged" else state


def _live_state_supersedes_outcome(row: dict[str, Any]) -> bool:
    """Has this row's CURRENT state overtaken its last stored outcome?

    The WHY column explains a session the reader cannot otherwise account for,
    and its source (``completion_kind``/``completion_reason``) is the attention
    store's record of how a turn ENDED. That is the right source for a row that
    is over — a stale record, a crash, a stop — and the wrong one for a row that
    has since been resumed and is now doing something, because the store is not
    rewritten when a session comes back: the old receipt simply stays until the
    next turn completes.

    Three live states outrank it, and they are the three ``catalog.status``
    already ranks above an unread completion mark for the picker:

    * a parked gate (``pending``) — somebody is blocked on this row right now,
      and the NEEDS column beside it already names which gate;
    * ``wedged`` — not answering NOW, which a receipt from a turn that did
      finish must not hide;
    * ``busy`` — a turn is running, so the previous turn's ending is history.

    Keeping this in step with that ranking is the point: the picker and this
    table describe the same sessions, and a user who sees "Approval needed" in
    one and "stopped by the user" in the other has to work out for themselves
    which surface is behind. Stored rows are untouched — they have no live state
    to supersede anything, and their outcome is the only thing they can say.
    """
    # STORED AND STALE ROWS HAVE NO LIVE STATE, and stale is the one that has to
    # be said out loud. A stored row obviously carries none — nothing is
    # running. A STALE row looks like it does: ``registry.scan`` classifies it
    # stale because the pid is GONE, but the record on disk is the last one the
    # dead runtime published, so the ``busy``/``pending`` flags in it are frozen
    # at whatever was true the instant before it died. Reading those as "this
    # session is working" would suppress the receipt on exactly the row the WHY
    # column was added for — a killed runtime publishes nothing further, and its
    # reason survives only in the attention store (see the column's note above,
    # and the 2026-09-13 kill wave it names).
    #
    # It is also where the ``catalog.status`` precedent this rule follows draws
    # the same line: ``catalog`` drops stale records from its live map outright
    # (``if session_id and state != "stale"``), so a stale row there has no live
    # state to outrank its completion mark and the receipt shows. Excluding it
    # here is what keeps the two surfaces agreeing rather than inverting on the
    # one row whose whole story is the outcome.
    if row.get("state") in ("stored", "stale"):
        return False
    return bool(row.get("pending")) or bool(row.get("busy")) or row.get("state") == "wedged"


#: Width of `lop sessions`' trailing WHY column, in display CELLS.
#:
#: Bounded because a reason is a SENTENCE — ``the runtime disappeared without
#: exiting cleanly while this turn was running, and no stop was asked for``
#: is 104 cells — and an unbounded column re-flows the whole table on a normal
#: terminal. The full text is one flag away in ``--json``'s
#: ``completion_reason`` and is what a script should read.
WHY_COLUMN_WIDTH = 48


#: Width of `lop sessions`' trailing LEAVING column, in display CELLS.
#:
#: A phrase, not an enum: the field's whole purpose is to say what is happening
#: in the words the operator needs (``signalled; leaving when its turn ends (up
#: to 2 min)``), so it is bounded like WHY rather than abbreviated to a token
#: nobody could read. Wide enough for the shipped phrase in full, so the common
#: case is not cut and a cut one is visibly marked (`_fit_cell`). The column
#: appears only when some row carries a value, exactly like WHY and LAST_ACTIVE
#: — a listing with no draining runtime is byte-for-byte what it was before.
#:
#: A TRAILING COLUMN RATHER THAN A TOKEN IN ``STATE``, which is the decision the
#: drain's first round recorded (design round 2, D3, kept rather than changed):
#: ``STATE`` holds seven cells that consumers branch on (``state == "stored"``),
#: so teaching it a new word to carry a display fact would spend a value the
#: machine reads to say something only a person needs.
#:
#: WIDENED FROM 40 when the phrase grew the drain's bound (UX round 2, U9): the
#: row that carries this is the one the operator reads most, and
#: ``signalled; leaving when its turn ends`` promised a boundary the 120 s bound
#: can take away. The number is the phrase's own cell width, pinned against it by
#: ``tests/unit/info/test_sessions_extraction.py`` rather than imported — this
#: module keeps session internals out of its module scope on purpose (see the
#: header) — so a reword of the phrase fails loudly there instead of silently
#: cutting the new clause off the row.
#: What the ``STALLED`` column says, and the width the header needs. The cell names
#: the state and the remedy in the register the panel uses (``HELD_STATE_WORD``),
#: because the fact is one the reader must be able to act on from a listing — ONE word
#: for it on all three surfaces (the panel, the dump and this cell), and no "stalled"
#: under a header that already says it (design review round 2, D7).
HELD_CELL = "bound held; lop stop"
HELD_COLUMN_WIDTH = len(HELD_CELL)
LEAVING_COLUMN_WIDTH = 51

#: Width of `lop sessions`' trailing UPDATING column, in display CELLS.
#:
#: A SECOND COLUMN RATHER THAN A WORD IN ``LEAVING``, and that is the feature rather
#: than a layout choice: the two fields are opposite promises. A ``leaving`` row says
#: this runtime will not take a message ("send it again once the new build is up");
#: an ``updating`` row says it ALREADY HAS it and runs it when the successor
#: boots. Folding them into one cell would make the operator re-send a message that
#: is queued — the exact harm the window exists to prevent (``types.UPDATING``).
#:
#: Sized from ``types.update_short``, whose pair is the wide part and which is why
#: the cell names only the NEW build: 26 is ``"updating → "`` (11 cells) plus the
#: longest label ``BuildStamp.label()`` can produce — ``0.59.11`` and ``@`` and the
#: 7-character ref git itself abbreviates to, so 15. The failed phase's cell is
#: shorter and carries no pair on purpose (see that function): its move did not
#: happen, so naming a build there would read as one that did.
#:
#: Like ``LEAVING_COLUMN_WIDTH`` the number is written out rather than imported
#: (this module keeps session internals out of its module scope on purpose, see the
#: header) and is pinned against the vocabulary by
#: ``tests/unit/session/runtime/test_updating_vocabulary.py``.
#:
#: Appears only when some row carries one, exactly like LEAVING and WHY: a listing
#: with no runtime mid-update is byte-for-byte what it was before.
UPDATING_COLUMN_WIDTH = 26


#: Widths of `lop sessions`' three TEXT columns, in display CELLS.
#:
#: Named rather than left as literals inside the format specs because the row
#: builder now has to MEASURE them: the header and the row have to agree on a
#: number that is used twice, and a second literal is how the two drift. These
#: three are the columns whose content this process does not author — a
#: conversation title is whatever named the conversation, and a model label is
#: the provider catalogue's own display name (``openai/gpt-5.2``, or a CJK
#: display name) — so they are the ones a wide glyph can overrun. The remaining
#: columns hold enums, pids and formatted byte counts, all ASCII and bounded.
#:
#: The values are unchanged from the literals they replace, and for ASCII text
#: ``cell_len`` equals ``len``, so every existing listing renders byte-for-byte
#: (design round 1, D2 on the WHY column's PR).
NEEDS_COLUMN_WIDTH = 8
CONVERSATION_COLUMN_WIDTH = 24
MODEL_COLUMN_WIDTH = 24


#: Width of `wake status`'s label column ("supervisor:  ", "scheduled:   ").
#: Every continuation line on that surface already indents to it, so a new
#: line that wraps at column 0 reads as a different block (round 2, D13).
_STATUS_LABEL_W = 13


def _wrap_status(text: str, label: str = "") -> str:
    """One `wake status` line, folded at the surface's own hanging indent.

    Two round-2 findings meet here. D13: the `wedged:` line was 108 columns and
    the only one whose continuation started at column 0, so the single line an
    operator must act on was the one rendered as a ragged paragraph. Q2: the
    `next:`/`overdue:` lines interpolate a user-authored wake message and were
    never clamped — a 129-character message produced a 156-column line at every
    terminal width, which is the last unclamped string on either screen.

    Wrapping rather than truncating, because unlike the `wake list` table (one
    row per wake, where a clamp keeps the columns) these lines are prose and
    the whole sentence is the payload.
    """
    import re
    import shutil
    import textwrap

    width = max(shutil.get_terminal_size((80, 24)).columns, _STATUS_LABEL_W + 24)
    indent = " " * _STATUS_LABEL_W
    first = f"{label:<{_STATUS_LABEL_W}}{text}" if label else f"{indent}{text}"

    # A QUOTED COMMAND IS ONE TOKEN. Every remedy on this surface is a command
    # the operator copies — `'lop wake install'`, `'lop stop --pid 4242'` — and
    # a wrap inside one produces a line that looks like an instruction and is
    # not runnable. `textwrap` only breaks on whitespace, so the spaces inside
    # single quotes are hidden from it and restored afterwards.
    nbsp = "\x00"
    protected = re.sub(r"'[^']*'", lambda m: m.group(0).replace(" ", nbsp), first)
    return "\n".join(
        textwrap.wrap(
            protected,
            width=width,
            subsequent_indent=indent,
            # A wake message can carry a path or a URL; breaking one makes it
            # unusable, and an over-long line is the lesser harm.
            break_long_words=False,
            break_on_hyphens=False,
        )
    ).replace(nbsp, " ")


def _json_dumps(value: Any) -> str:
    import json as _json

    return _json.dumps(value, indent=2)


def _format_due(seconds: float) -> str:
    """``in 4m`` / ``2h overdue`` — the relative form a reminder is read in."""
    overdue = seconds < 0
    seconds = abs(seconds)
    if seconds < 90:
        text = f"{int(seconds)}s"
    elif seconds < 5400:
        text = f"{int(seconds // 60)}m"
    elif seconds < 172800:
        text = f"{int(seconds // 3600)}h"
    else:
        text = f"{int(seconds // 86400)}d"
    return f"{text} overdue" if overdue else f"in {text}"


def _process_uptime_s(pid: int) -> float | None:
    """How long ``pid`` has been alive, or ``None`` when it cannot be read.

    ``ps -o etime=`` rather than a dependency: this is one line on a status
    surface, and ``psutil`` is deliberately not a dependency of this project.
    Every failure is None — an uptime is a nicety on a line whose real payload
    is the pid, and a status command must never fail because a subprocess did.
    """
    import subprocess as _subprocess

    try:
        result = _subprocess.run(  # noqa: S603 — fixed argv, no shell
            ["ps", "-p", str(pid), "-o", "etime="],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, _subprocess.SubprocessError):
        return None
    raw = result.stdout.strip()
    if result.returncode != 0 or not raw:
        return None
    # `[[dd-]hh:]mm:ss` — parsed right to left so every form falls out of the
    # same loop rather than needing a format branch per shape.
    days = 0
    if "-" in raw:
        day_part, _, raw = raw.partition("-")
        if not day_part.isdigit():
            return None
        days = int(day_part)
    parts = raw.split(":")
    if not all(part.strip().isdigit() for part in parts) or len(parts) > 3:
        return None
    seconds = 0.0
    for power, part in enumerate(reversed(parts)):
        seconds += int(part) * (60**power)
    return seconds + days * 86400


def stop_command(args: argparse.Namespace) -> int:
    """``lop stop`` — end a running session from outside it (§12).

    The CLI front end of the one kill-switch implementation
    (``session/runtime/control.py``): graceful stop op → identity-confirmed
    SIGTERM → SIGKILL, refusing to signal any target whose identity cannot
    be confirmed over its own control socket.

    Exit codes: **0** every requested stop resolved, **1** no target matched
    (nothing was running under that name), **2** partial — at least one stop
    resolved and at least one refused, so a script can tell "wrong name"
    from "one agent would not die" without parsing prose.

    Imports stay function-local (the CLI startup path must stay light — see
    ``tests/unit/test_import_graph.py``): the control module is import-light
    itself, but ``asyncio`` and the resolver are pulled only when a stop is
    actually being made.
    """
    import asyncio

    from local_operator.mobile.peer_send import candidate_lines
    from local_operator.paths import config_dir
    from local_operator.session.runtime import control

    timeout_s = args.timeout if args.timeout and args.timeout > 0 else control.DEFAULT_TIMEOUT_S

    if getattr(args, "stop_all", False):
        # Confirmation: a prompt on a TTY, a hard refusal in a pipe without
        # --yes. `lop stop --all` from a script can end a dozen agents, so a
        # pipe must never inherit a terminal's y/N affordance — it would hang
        # waiting on stdin nobody is watching, or worse, read the piped body
        # the user meant for something else.
        targets = control._stop_targets(config_dir(), own_pid=None)
        # An empty machine is a no-op in every mode (D2-3): check before
        # the TTY gate so a pipe without --yes still exits 0, not the refusal.
        if not targets:
            print("no sessions to stop")
            return 0
        if not args.yes:
            if not sys.stdin.isatty():
                _peer_red(
                    "stdin is not a terminal — pass --yes to stop every session without a prompt"
                )
                return 1
            # The listing is the confirmation, as in the TUI: consent to
            # "every session" is only informed when the user can see which.
            print(f"will stop {len(targets)} session{'s' if len(targets) != 1 else ''}:")
            for line in candidate_lines(targets, indent="  ", prefix="pid"):
                print(line)
            count = len(targets)
            # Disclose --force IN the question: consent to "stop everything"
            # is not consent to "signal everything whose socket is silent",
            # and the flag was typed once at the top of a command whose
            # listing may be long (round-3 U3-3).
            forced_part = " (--force: signal any that will not answer)" if args.force else ""
            try:
                answer = input(
                    f"stop {'all ' if count != 1 else ''}{count} lop session"
                    f"{'s' if count != 1 else ''} on this machine{forced_part}? [y/N] "
                )
            except EOFError:
                # Ctrl+D: not a yes. A traceback would exit 1 with noise and
                # nothing stopped; a clean abort says the same thing plainly.
                print("aborted")
                return 1
            if answer.strip().lower() not in ("y", "yes"):
                print("aborted")
                return 1
        outcomes = asyncio.run(
            control.stop_all(
                own_pid=None,
                timeout_s=timeout_s,
                only_pids={rec.pid for rec in targets},
                force=args.force,
                _root=config_dir(),
                _command="lop stop --all",
                on_wait=_stop_progress,
            )
        )
        return _report_stops(outcomes, args.json, summary=True)

    record, candidates, error = _resolve_stop_target(args)
    if candidates:
        print(f"{len(candidates)} sessions match; disambiguate with --pid:", file=sys.stderr)
        for line in candidate_lines(candidates, indent="  ", prefix="--pid"):
            print(line, file=sys.stderr)
        return 1
    if error or record is None:
        _peer_red(error or "no target resolved")
        return 1

    outcome = asyncio.run(
        control.stop_session(
            record,
            timeout_s=timeout_s,
            force=args.force,
            _root=config_dir(),
            # The artifact's point is naming WHO stopped it, so the CLI records
            # what the user typed rather than the function they reached.
            _command="lop stop",
            on_wait=_stop_progress,
        )
    )
    return _report_stops([outcome], args.json)


def _stop_progress(line: str) -> None:
    """Paint one progress line the ladder emits while it waits.

    STDERR, not stdout, and that is the whole reason this is a function rather
    than an inline ``print``: stdout carries the receipts — under ``--json``, a
    document a caller parses — and a progress line there would either break that
    parse or force every consumer to filter a line the final receipt supersedes
    a moment later. Progress about a wait belongs beside it, which is where the
    disambiguation listing above already goes.

    It exists because the ladder's rung-2 wait is the longest silence a `lop`
    command produces (~150 s for a wedged mid-turn target) and it used to print
    nothing at all until it resolved: an operator clearing a wedged session
    could not tell a working command from a hung one, and the natural response —
    Ctrl-C — leaves the outcome ambiguous (U5, PR #1141).
    """
    print(line, file=sys.stderr)


def _resolve_stop_target(
    args: argparse.Namespace,
) -> "tuple[Any | None, list[Any], str]":
    """Resolve a ``lop stop`` / ``lop refresh`` target through the `send` vocabulary.

    The same shared resolver `lop send` uses, so every way of addressing a
    peer — name, substring, session id, pid — behaves identically across
    `send`, `stop` and `refresh`. Only the hint strings differ (the parsers'
    own flags).

    WEDGED targets are included for both commands, for different reasons that
    happen to want the same set: they are stoppable (the ladder's signal rungs
    exist for them) and they are worth ASKING about (a rotation reports
    ``unreachable``, which is the honest answer to "why is this one still on the
    old build"). `send` keeps refusing them because nobody would read it.
    """
    from local_operator.mobile.peer_send import resolve_peer_target

    return resolve_peer_target(
        target=args.target,
        pid=args.pid,
        session=args.session,
        pid_hint="--pid",
        session_hint="--session",
        # Wedged sessions are stoppable (the ladder's signal rungs exist for
        # them); `send` keeps refusing them because nobody would read it.
        include_wedged=True,
        # A composer window is stoppable, and this is the one caller that wants
        # it resolved: the kill switch names a target in order to END it, not to
        # message it. `lop send` keeps the default True, so the fresh `/new`
        # nobody has typed in stays out of reach of delivery while remaining
        # reachable by `lop stop`.
        require_started=False,
    )


def _report_stops(outcomes: list[Any], as_json: bool, *, summary: bool = False) -> int:
    """Paint the stop outcomes and derive the exit code.

    0 clean (every outcome resolved, including "already exited"), 2 partial
    (some refused), never 1 here — 1 belongs to resolution failures above.
    A kill rung is reported as what it is; the code stays 0 because from the
    caller's side the agent IS stopped, which is the thing they asked for.
    ``summary`` (the ``--all`` path) always prints the grouped line, so an
    empty run says "no sessions to stop" instead of nothing.
    """
    if as_json:
        import json as _json

        print(
            _json.dumps(
                [
                    {
                        "pid": o.pid,
                        "session_id": o.session_id,
                        "name": o.name,
                        "method": o.method,
                        "line": o.line,
                        "wakes_dormant": o.wakes_dormant,
                    }
                    for o in outcomes
                ],
                indent=2,
            )
        )
    else:
        for outcome in outcomes:
            print(outcome.line)
        if summary:
            from local_operator.session.runtime.control import summarize

            print(summarize(outcomes))
    # Anything that did NOT end the session is partial: a refusal (identity
    # unconfirmed, nothing signalled) and a skip (a turn in flight, nothing
    # signalled) alike — in both the target is still running, which is what the
    # caller asked about. "gone" (already exited) is a clean resolution. The
    # method says which, so no receipt text is parsed here; ``ENDED_METHODS`` is
    # the one definition of "it is not running any more".
    from local_operator.session.runtime.control import ENDED_METHODS

    return 2 if any(o.method not in ENDED_METHODS for o in outcomes) else 0


def refresh_command(args: argparse.Namespace) -> int:
    """``lop refresh`` — move live sessions to the build on disk, without killing.

    The supported way to make a new build take effect on sessions that are
    WORKING. Run 1 of this command is ``lop-update``: the runtimes notice the
    moved install on their own and retire when idle, but "when idle" can be
    hours away, and the only other tool to hand was a signal sweep — which is
    what cut 32 turns off on 2026-09-14. This asks instead of telling: each
    runtime judges its own readiness, so a busy session is reported as moving
    at its next boundary rather than ended.

    Exit codes: **0** every target gave an answer (moved, busy, draining,
    already current, or its own reason), **1** no target matched, **2** partial
    — at least one runtime did not answer its control socket, so its move is not
    going to happen on its own, OR the install on disk had not settled and no
    runtime could judge it yet (``unsettled``). The second case is deliberately
    partial rather than clean (D1/M2, PR #1141): this command's own run 1 is
    ``lop-update``, so it lands inside the settle window for a whole fleet, and
    exiting 0 there would tell a rotating script the rotation is complete when
    every session on the machine is about to be retired. The receipt says to ask
    again in a few seconds, which is the honest next step.

    Imports stay function-local like every other runtime path here (the CLI
    startup path must stay light — see ``tests/unit/test_import_graph.py``).
    """
    import asyncio

    from local_operator.mobile.peer_send import candidate_lines
    from local_operator.paths import config_dir
    from local_operator.session.runtime import control

    timeout_s = (
        args.timeout if args.timeout and args.timeout > 0 else control.DEFAULT_REFRESH_TIMEOUT_S
    )

    if getattr(args, "refresh_all", False):
        # NO CONFIRMATION GATE, unlike `stop --all`: this command ends no session
        # and interrupts no turn, so "every session" is not a decision anyone has
        # to be talked through. The listing still prints, because the outcome per
        # session IS the answer the caller came for.
        targets = control._rotation_targets(config_dir(), own_pid=None)
        if not targets:
            print("no live sessions to refresh")
            return 0
        outcomes = asyncio.run(control.refresh_all(timeout_s=timeout_s, own_pid=None))
        return _report_refreshes(outcomes, args.json, summary=True)

    record, candidates, error = _resolve_stop_target(args)
    if candidates:
        print(f"{len(candidates)} sessions match; disambiguate with --pid:", file=sys.stderr)
        for line in candidate_lines(candidates, indent="  ", prefix="--pid"):
            print(line, file=sys.stderr)
        return 1
    if error or record is None:
        _peer_red(error or "no target resolved")
        return 1

    outcome = asyncio.run(control.refresh_session(record, timeout_s=timeout_s))
    return _report_refreshes([outcome], args.json)


def _report_refreshes(outcomes: list[Any], as_json: bool, *, summary: bool = False) -> int:
    """Paint the rotation outcomes and derive the exit code.

    0 when every runtime answered (however it answered — "busy" is a queued
    move, not a failure, and "draining" is that move already scheduled), 2 when
    at least one could not be asked at all, or could not yet judge the install
    (``unsettled``, see ``REFRESH_SETTLED_METHODS``). The method is the verdict,
    exactly as in ``_report_stops``: no receipt text is parsed and the two
    commands cannot disagree about what counts as partial.
    """
    if as_json:
        import json as _json

        print(
            _json.dumps(
                [
                    {
                        "pid": o.pid,
                        "session_id": o.session_id,
                        "name": o.name,
                        "method": o.method,
                        "line": o.line,
                    }
                    for o in outcomes
                ],
                indent=2,
            )
        )
    else:
        for outcome in outcomes:
            print(outcome.line)
        if summary:
            from local_operator.session.runtime.control import summarize_refresh

            print(summarize_refresh(outcomes))
    from local_operator.session.runtime.control import REFRESH_SETTLED_METHODS

    return 2 if any(o.method not in REFRESH_SETTLED_METHODS for o in outcomes) else 0


def _format_duration(seconds: float) -> str:
    """Compact duration shared with the wake panel: 45s, 12m, 3h, 2d."""
    from local_operator.wakes.display import format_age

    return format_age(seconds)


def _tail_last_lines(text: str, count: int) -> list[str]:
    """The last ``count`` lines of ``text``, newline-terminated for printing.

    Split on ``\n`` alone, not ``str.splitlines``: the latter also breaks on
    ``\v``, ``\f`` and the Unicode line separators, so it would report line
    numbers ``tail`` does not (a log line carrying a form feed would count as
    two). A trailing newline does not open a further empty line.
    """
    if count <= 0:
        return []
    parts = text.split("\n")
    if parts and parts[-1] == "":
        parts.pop()
    return [part + "\n" for part in parts[-count:]]


def _python_tail(paths: Sequence[Path], lines: int, *, follow: bool) -> int:
    """Print log tails in-process, for hosts whose userland has no ``tail``.

    Windows ships no ``tail``, and it has no ``/bin/sh`` to borrow one from, so
    this is the *whole* implementation there rather than a second behaviour
    beside ``tail`` on the platforms that have it — the caller only reaches here
    after ``subprocess.call`` has already raised ``FileNotFoundError``.

    ``follow`` is the ``-F`` shape for the reasons in the caller's comment: a
    runtime log created after the follow started must still be picked up, and
    rotation (a RENAME here, because the handler bounds the file by renaming)
    must not leave the reader watching a dead inode. Both fall out of re-stat'ing
    the PATH every poll and comparing identity, which is what ``-F`` does.
    """
    offsets: dict[Path, int] = {}
    keys: dict[Path, tuple[int, int] | None] = {}
    if not follow:
        for index, path in enumerate(paths):
            if index:
                print()
            print(f"==> {path} <==")
            for line in _tail_last_lines(_read_text(path), lines):
                sys.stdout.write(line)
        return 0
    try:
        while True:
            for path in paths:
                try:
                    stat = path.stat()
                except OSError:
                    # Not created yet, or mid-rotation and the replacement has
                    # not been renamed into place. ``-F`` retries a missing file
                    # quietly; so does this.
                    continue
                identity = (stat.st_ino, stat.st_dev)
                if keys.get(path) != identity:
                    # First sight, or the file was replaced under us: header and
                    # the tail, exactly as ``tail`` prints on opening a file.
                    keys[path] = identity
                    print(f"==> {path} <==")
                    try:
                        with path.open("r", encoding="utf-8", errors="replace") as handle:
                            text = handle.read()
                            offsets[path] = handle.tell()
                    except OSError:
                        continue
                    for line in _tail_last_lines(text, lines):
                        sys.stdout.write(line)
                    sys.stdout.flush()
                    continue
                offset = offsets.get(path, 0)
                if stat.st_size < offset:
                    # Truncated in place (a copytruncate-style rotation): what is
                    # there now is all there is.
                    offsets[path] = 0
                    continue
                if stat.st_size == offset:
                    continue
                try:
                    with path.open("r", encoding="utf-8", errors="replace") as handle:
                        handle.seek(offset)
                        chunk = handle.read()
                        offsets[path] = handle.tell()
                except OSError:
                    continue
                sys.stdout.write(chunk)
                sys.stdout.flush()
            time.sleep(0.5)
    except KeyboardInterrupt:
        return 0


def _read_text(path: Path) -> str:
    """Read a log file for the no-``tail`` path, tolerating a missing one.

    ``errors="replace"`` because a daemon's stdout and a runtime's records can
    interleave a partial multi-byte sequence; losing one line's glyphs beats
    failing the command the operator ran to read the log.
    """
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _run_tail(argv: Sequence[str], paths: Sequence[Path], lines: int, *, follow: bool) -> int:
    """Run ``tail``, or read the tails in-process on a host that has none.

    ``tail`` stays the only path taken on POSIX: the comments at the call site
    record measurements against the system ``tail`` (``-F`` follows by NAME,
    where ``-f`` does not), and reimplementing that for hosts which already have
    it would be a second behaviour to keep in step. Windows has no ``tail``, so
    ``subprocess.call`` raised ``FileNotFoundError`` straight out of ``lop mobile
    logs`` — a traceback instead of the log. Catching exactly that exception and
    falling back keeps the POSIX path byte-identical.
    """
    try:
        return subprocess.call(list(argv))
    except FileNotFoundError:
        return _python_tail(paths, lines, follow=follow)


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
        from local_operator.mobile.auth import store_description

        result = mobile_install.install()
        steps = result.get("steps", [])
        assert isinstance(steps, list)
        for step in steps:
            print(f"  {step}")
        if result.get("ok"):
            print("\nmobile daemon installed and healthy.")
            print("  open http://127.0.0.1:4098 and sign in with your portal password")
            # The store is per-platform now (Keychain, Secret Service, DPAPI), so
            # the sentence is built from the one function that answers where this
            # machine keeps it — the same answer install's own steps print.
            print(f"  the password is in {store_description()}.")
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
        # A healthy daemon with no bundle serves a 503 to every authenticated GET
        # ("mobile web bundle not built"), so `healthy: yes` on its own reads as
        # fine while the phone has no UI at all — the state generations
        # 0.61.13-0.61.16, 0.61.18 and 0.62.0 were flipped into. One line, naming
        # the remedy, rather than a redesign of this output.
        #
        # Gated on `installed`: on a machine that never set the portal up there is
        # no phone and no 503, so the line would read as a fault report where the
        # only correct advice is "you have not set this up yet" (design round 1,
        # D4). The label leads with the STATE rather than the classifier token, so
        # it is true on its own; the token stays in the parenthesis, where it is
        # the classifier's own name (D5).
        bundle = result.get("bundle")
        if result["installed"] and bundle in ("buildable", "missing-sources"):
            detail = (
                "buildable — run `lop mobile install`"
                if bundle == "buildable"
                else "no web sources to build from"
            )
            print(f"bundle:       not built ({detail})")
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
        from local_operator.paths import runtime_log_path

        log = mobile_install.log_path()
        runtime_log = runtime_log_path()
        tail = ["tail", "-n", str(args.lines)]
        if args.follow:
            # `-F`, not `-f`: follow by NAME. `-f` follows the fd it opened, so it
            # never reads a file created after it started — the normal state on a
            # machine whose daemons are up but whose runtimes have all exited — and
            # it goes blind at the first rotation of the runtimes' bounded file,
            # because bounding means renaming. Both were measured against the
            # system `tail`. Both paths go in unconditionally HERE because `-F`
            # retries a missing one quietly; a plain `tail` does not, so the
            # non-following branch below filters to what exists.
            # One limit of this argv that cannot be fixed from here, measured:
            # BSD `tail` prints its `==> file <==` header when it OPENS a file, so
            # a runtime log created after the follow started is read without a
            # header and its lines sit under the daemon's. Attribution survives —
            # every record names its logger, and a runtime names its own pid — and
            # the alternative is a multiplexer of our own, which is not worth it
            # for a header.
            tail.append("-F")
            tail.extend([str(log), str(runtime_log)])
            return _run_tail(tail, [log, runtime_log], args.lines, follow=True)
        existing = [path for path in (log, runtime_log) if path.exists()]
        if not existing:
            # `tail` with no operand reads STDIN and would hang the command on a
            # machine whose daemon has not written yet.
            print(f"no log files yet: {log} (and {runtime_log})")
            return 0
        tail.extend(str(path) for path in existing)
        return _run_tail(tail, existing, args.lines, follow=False)

    if command == "password":
        from local_operator.mobile.auth import (
            generate_password,
            load_password,
            store_description,
            store_password,
        )

        # A captured stdout (an agent tool result, a redirected log) is the
        # context window. Refuse to print the secret unless a human is at a
        # TTY. Rotation still works non-interactively via --rotate once we
        # have a TTY confirmation; without a TTY we only say where it lives.
        if not sys.stdout.isatty():
            # Built from the store question rather than written out: the password
            # is in the login Keychain on macOS and nowhere else, and naming the
            # Keychain on Linux or Windows sends the user looking for a store
            # their OS does not have.
            print(f"portal password is in {store_description()}.")
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


def _bind_serve_socket(host: str, port: int) -> socket.socket:
    """Bind the listener ``serve`` will hand to uvicorn, and return it.

    Bound HERE rather than by uvicorn for one reason: ``--port 0`` asks the
    kernel for an ephemeral port and the daemon's rendezvous record must carry
    the port ACTUALLY BOUND. Binding is the only way to know it — resolving an
    ephemeral port with a bind/close/rebind probe can lose the port to another
    process in between, and then the record and the listener disagree, silently
    and precisely on the machines the record exists for. It is the design the
    UI's own child needs, since it asks for port 0 deliberately so that two
    daemons cannot collide on a fixed one.

    Mirrors ``uvicorn.Config.bind_socket()``: the address family follows a host
    with a colon in it (an IPv6 literal), ``SO_REUSEADDR`` is set (see the
    platform note below for Windows, where a different option is required), and
    the socket is bound WITHOUT listening — ``loop.create_server`` calls ``listen``
    on the socket it is handed, which is the same sequence uvicorn uses on its
    own path. Raising is deliberate: the caller reports a bind failure through
    :func:`_refuse_serve_bind`, rather than letting it surface as a traceback.

    ``SO_REUSEADDR`` is kept because it is what lets a daemon restart
    immediately after a crash. What it does NOT do was measured rather than
    assumed, because the first version of the collision test got it wrong: it
    does not make a bind succeed over a LISTENING holder — that is refused on
    Linux and on macOS, which is why that test holds its port with a real
    listener rather than a bare bound socket. On the Linux CI runner it DOES let
    a second bind succeed over a holder that has only bound and not listened
    (measured: that is the shape that failed shard 3), while on this macOS host
    (Darwin 25.6.0, arm64) every permutation of a bound-but-not-listening holder
    was refused. So the friendly refusal below is guaranteed against a LISTENING
    holder; against a merely-bound one on Linux the collision instead surfaces
    from uvicorn's own ``listen``, as its own error.

    On Windows the option is ``SO_EXCLUSIVEADDRUSE`` instead, and that is a
    correctness fix rather than a preference: there ``SO_REUSEADDR`` does let a
    second bind succeed over a LISTENING holder (Microsoft, "Using SO_REUSEADDR
    and SO_EXCLUSIVEADDRUSE"), so the refusal above would be silent on the one
    platform where a port collision is easiest to create.
    """
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    sock = socket.socket(family=family)
    # ``getattr`` rather than ``socket.SO_EXCLUSIVEADDRUSE``: the constant exists
    # only in CPython's Windows build (``socketmodule.c`` guards it with
    # ``#ifdef SO_EXCLUSIVEADDRUSE``), so naming it directly is an attribute the
    # type checker and every POSIX run would have to be told to ignore.
    exclusive = getattr(socket, "SO_EXCLUSIVEADDRUSE", None)
    if exclusive is not None:
        # Windows only, and the choice is not cosmetic: the two options are
        # mutually EXCLUSIVE, and on Windows SO_REUSEADDR lets a second socket
        # bind an address a first, LISTENING socket already holds — the
        # documented hijack vector — so the friendly "a daemon already holds
        # 1111" refusal below would never fire and two `lop serve` processes
        # would split the port in silence. On Linux/macOS the same bind over a
        # listening holder is refused, which is why SO_REUSEADDR was (correctly)
        # kept there for fast crash restarts.
        sock.setsockopt(socket.SOL_SOCKET, exclusive, 1)
    else:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((host, port))
    except OSError:
        sock.close()
        raise
    return sock


def _refuse_serve_bind(host: str, port: int, exc: OSError) -> int:
    """Report a bind this process could not make, naming the address.

    The one refusal shape for BOTH binds ``serve_command`` makes — the listener
    it hands uvicorn and the ``--reload`` probe — because to the operator they
    are the same failure, and the address is what makes it actionable: the
    common cause is a daemon already running on the default 1111, which
    "address already in use" alone does not say.

    Exit code 1, deliberately NOT uvicorn's ``STARTUP_FAILURE`` (3, from its own
    ``Config.bind_socket``): this refusal is printed by us, with the address
    named, and the tests assert 1. Nothing in this repo branches on the number.
    """
    print(
        f"\n\033[1;31mError: cannot bind http://{host}:{port}: {exc}\033[0m",
        file=sys.stderr,
    )
    return 1


def adopt_serve_socket(fd: int, host: str) -> socket.socket:
    """Take over an already-bound listener handed across ``execve``.

    WHY THIS EXISTS AT ALL. A reload replaces this process's image and keeps the
    socket fd open across the exec, so the port a client is connected to is the
    same kernel object before and after; the successor must therefore ADOPT it
    rather than bind. A fresh ``bind`` would fail with ``EADDRINUSE`` against the
    socket this very process is still holding, and the tempting alternative —
    close it and rebind — is a window in which ``connect`` gets refused, which is
    the outage the in-place design was chosen to avoid.

    THE FAMILY COMES FROM ``host``, NOT FROM A CONSTANT (serve-reload review round 1, R1-3).
    ``socket.fromfd`` needs a family to reinterpret the descriptor under, and a
    hardcoded ``AF_INET`` reinterprets an IPv6 listener's address bytes as IPv4:
    measured in review, an adopted ``--host ::1`` daemon logged its peer as
    ``::24:b503:100:0:61963`` where an ordinary bind on the same host logs
    ``::1:62246``. The daemon still served, which is exactly why it would have
    gone unnoticed. This is the same rule `_bind_serve_socket` already follows by
    giving ``uvicorn.Config`` a host with a colon in it.

    ``socket.fromfd`` DUPLICATES the descriptor rather than wrapping it, so the
    inherited fd is closed here: leaving it open would leak one descriptor per
    reload in a process that may live for months, and would keep a second handle
    on a socket nothing is serving from.

    Deliberately does NOT re-apply ``SO_REUSEADDR`` or re-bind anything: the
    socket is already fully configured and listening, and touching it would undo
    the only property this path is for.
    """
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    listener = socket.fromfd(fd, family, socket.SOCK_STREAM)
    # The dup ``fromfd`` just made is the one we keep, and it must NOT be
    # inherited by anything this process later spawns — only a future reload of
    # this daemon should see it, and that reload publishes its own fd.
    listener.set_inheritable(False)
    try:
        os.close(fd)
    except OSError:  # pragma: no cover — an fd already reaped by the dup path
        pass
    return listener


def serve_command(host: str, port: int, reload: bool, *, listener_fd: int | None = None) -> int:
    """Start the FastAPI server using uvicorn.

    ``uvicorn`` is imported HERE, not at module scope: the HTTP facade lives
    behind the ``server`` extra, so a default install (and every non-server
    entry point) must be able to ``import local_operator.cli`` without
    fastapi/uvicorn/starlette and their dependency chain present.

    The listener is bound here and handed to uvicorn as an open socket
    (``uvicorn.Server.run(sockets=[...])``, supported by the pinned uvicorn —
    ``Server.startup`` takes the explicit-sockets branch and
    ``loop.create_server`` adopts them), so that ``--port 0`` works and the
    resolved port is known to THIS process. That port is announced to the app
    before uvicorn starts — on the app object for the path that serves it
    in-process, and through the environment only for a ``--reload`` child, which
    cannot be reached any other way (``server.registry``) — because the
    app publishes the daemon's rendezvous record at startup and that record
    has to name the port the kernel gave us rather than the ``--port``
    argument.

    ``--reload`` is the one exception: uvicorn only reloads an app given as an
    import string, and the reloader re-imports it in a CHILD process, so the
    socket cannot be handed over here. That path keeps ``uvicorn.run`` and, for
    ``--port 0``, resolves the port with the bind/close probe described above —
    the race is acceptable in a development mode that no daemon discovery
    depends on, and the identity check a discovery does (``/health``'s
    ``instance_id``, never the record's port) is what admits a candidate.
    """
    try:
        import uvicorn
    except ImportError:
        print(
            f"\n\033[1;31m{missing_extra_error('server', 'The HTTP API server')}\033[0m",
            file=sys.stderr,
        )
        return 1

    # Imported here for the same reason as uvicorn (it is the server package),
    # and it is import-light by contract: stdlib plus the session registry.
    from local_operator.server import registry as serve_registry

    listener: socket.socket | None = None
    resolved_port = port
    if listener_fd is not None and reload:
        # Both at once is a contradiction rather than a precedence rule: a
        # ``--reload`` child's port belongs to uvicorn's supervisor, which is
        # precisely why ``reload.install`` refuses to arm a reload on one. A
        # caller that asked for both has asked for neither, and refusing by name
        # is better than picking one silently.
        print(
            "--listener-fd and --reload cannot be combined: a reload child's port "
            "belongs to its supervisor, so there is no socket of its own to adopt",
            file=sys.stderr,
        )
        return 1
    if listener_fd is not None:
        # The reload path: this process is the replacement, and the socket is
        # already bound and listening in the kernel. Nothing is bound here, so a
        # failure can only be a bad descriptor — reported like a bind failure,
        # because from the client's side the consequence is the same one.
        try:
            listener = adopt_serve_socket(listener_fd, host)
        except OSError as exc:
            print(f"--listener-fd {listener_fd} could not be adopted: {exc}", file=sys.stderr)
            return 1
        resolved_port = listener.getsockname()[1]
    elif not reload:
        try:
            listener = _bind_serve_socket(host, port)
        except OSError as exc:
            return _refuse_serve_bind(host, port, exc)
        resolved_port = listener.getsockname()[1]
    elif port == 0:
        # ``--reload`` only: the port is resolved here purely so the record and
        # the banner name something, then the probe socket is released for
        # uvicorn to bind again. Its child re-imports the app, so this is the
        # one place the address cannot be handed over.
        #
        # Refused exactly like the listener above, and for the same reason: a
        # host that cannot be bound is the operator's mistake on this path too,
        # and a traceback is not a better report of it than a named address.
        try:
            probe = _bind_serve_socket(host, port)
        except OSError as exc:
            return _refuse_serve_bind(host, port, exc)
        try:
            resolved_port = probe.getsockname()[1]
        finally:
            probe.close()

    # With a socket handed down, uvicorn SKIPS its own "Uvicorn running on …"
    # line (it cannot know which of several listeners to name), so this print is
    # the one place the resolved address is reported — which is why it names
    # ``resolved_port``, not the ``--port`` argument that may have been 0.
    print(f"Starting server at http://{host}:{resolved_port}")
    if listener is not None:
        # A bound listener exists exactly when we are NOT reloading, and it is
        # the app object (not its import string) that is served in that case:
        # with no reloader there is nothing to re-import the app, so handing
        # over the object is what lets this process announce its address to the
        # very app it serves.
        from local_operator.server.app import app as asgi_app

        # Announced on the app object, NOT in the environment, and that is the
        # whole point of the split: this process may spawn children (an agent's
        # shell tool, a wrapper, another entry point booting the same app) and an
        # inherited variable would let any of them publish a record naming OUR
        # listener as its own.
        serve_registry.announce_address(asgi_app, host, resolved_port)
        # The fd a reload hands across ``execve``. Published on the app object
        # for the same reason the announcement is: the socket lives in THIS
        # frame and the reload runs in the lifespan's loop, so state is the one
        # channel that reaches both.
        from local_operator.server import reload as serve_reload

        serve_reload.bind_listener_fd(asgi_app, listener.fileno())
        config = uvicorn.Config(asgi_app, host=host, port=resolved_port)
        # ``KeyboardInterrupt`` caught here because this path replaces
        # ``uvicorn.run``, which catches it around the same call. uvicorn's own
        # signal handling turns Ctrl+C into a clean shutdown while it is
        # installed, so this only covers the windows either side of that — but
        # those windows are exactly where a traceback would otherwise reach the
        # operator instead of a plain exit.
        try:
            uvicorn.Server(config).run(sockets=[listener])
        except KeyboardInterrupt:
            pass
    else:
        # The reloader re-imports ``server.app`` in a CHILD process, so the app
        # object cannot be reached above and the inherited environment is the
        # only channel that survives into it. The value carries OUR pid, so only
        # the child this process spawns honours it, and it is read-and-cleared
        # there so nothing that child later spawns can re-publish it.
        serve_registry.announce_to_reload_child(host, resolved_port)
        uvicorn.run(
            "local_operator.server.app:app",
            host=host,
            port=resolved_port,
            reload=reload,
            reload_excludes=[".venv"],
        )
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


def _radient_hub_base_url(config_manager: ConfigManager) -> str:
    """The ONE place the CLI resolves the Radient Agent Hub API root.

    ``config.yml``'s ``values.radient_base_url`` — the NESTED key the config
    store actually holds; a flat document-root ``radient_base_url`` is dropped by
    the migration and would be silently ignored — wins when set;
    :func:`~local_operator.env.resolve_radient_api_base_url` supplies the
    ``RADIENT_API_BASE_URL``/canonical default otherwise, version segment
    included. Sharing this helper is the point: ``agents delete``, ``agents
    push`` and ``agents pull`` each carried their own literal, two of the three
    naming a route or a host that does not exist, so one configuration resolved
    three different destinations and two of them could never work.
    """
    return resolve_radient_api_base_url(config_manager.get_config_value("radient_base_url", None))


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
        from local_operator.providers.radient_credentials import (
            resolve_radient_credential_sync,
        )

        config_manager = ConfigManager(config_dir)
        base_url = _radient_hub_base_url(config_manager)
        api_key = resolve_radient_credential_sync(config_manager.config_dir, base_url)
        if not api_key:
            print("\n\033[1;31mError: RADIENT_API_KEY is required to delete from Radient\033[0m")
            return 1
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


def _build_auth_stack(config_dir: Path) -> tuple[Any, Path]:
    """``(auth_store, config_dir)`` for the login handlers.

    The second element is the config ROOT the store-first readers resolve
    under, not the ``CredentialManager`` that used to carry it: PR2b deleted
    that class and ``AuthStore``/``list_logins`` take the path directly.

    Lazy import of the providers stream's AuthStore — the CLI module top
    level must never depend on it.
    """
    from local_operator.providers.auth_store import AuthStore

    auth_store = AuthStore(config_dir=config_dir)
    return auth_store, config_dir


def login_command(args: argparse.Namespace) -> int:
    """Run the OAuth/API-key login flow for one provider."""
    try:
        from local_operator.providers.auth_cli import run_login
    except ImportError:
        print("\n\033[1;31mError: provider login support is not available in this build\033[0m")
        return 1
    auth_store, config_dir_path = _build_auth_stack(config_dir())
    try:
        return run_login(getattr(args, "provider", None), config_dir_path, auth_store)
    finally:
        auth_store.close()


def logout_command(args: argparse.Namespace) -> int:
    """Remove all stored credentials for one provider."""
    try:
        from local_operator.providers.auth_cli import run_logout
    except ImportError:
        print("\n\033[1;31mError: provider login support is not available in this build\033[0m")
        return 1
    auth_store, _config_dir = _build_auth_stack(config_dir())
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
    auth_store, config_dir_path = _build_auth_stack(config_dir())
    try:
        return list_logins(auth_store, config_dir_path)
    finally:
        auth_store.close()


#: The provider whose `/usage` window the console ticket feeds. The ticket is
#: stored under its OWN namespace (`qwencloud-console`) so it can never satisfy
#: `has_any_credential` and make local-operator believe it can CHAT on a
#: read-only console cookie -- but the usage it reports is filed under this id,
#: so this is the cache row a ticket change invalidates and the credential the
#: ticket augments rather than replaces.
_QWENCLOUD_TICKET_AUGMENTS = "alibaba-token-plan"


def _qwencloud_credential_row_exists(store: Any) -> bool:
    """Whether a Token Plan credential the ticket can augment is stored.

    `status` warns on this rather than `status` fixing it, because the fix is
    not available: `/usage` gates on `can_report_usage` -> `is_usable`, and a
    ticket row that satisfied those is exactly the blast radius the separate
    `qwencloud-console` namespace exists to prevent.

    Soft-deleted rows are NOT included, which is the point: `lop logout
    alibaba-token-plan` marks the row `logged-out`, and that is precisely the
    moment the window goes silently missing and the warning has to appear.

    An unreadable store returns True — no warning. This is a HINT, not a
    security gate, so the honest failure is silence rather than telling the
    user a credential is missing when the store could not be read at all.
    `sqlite3.ProgrammingError` still propagates, following the rule
    controller.py:268-278 records: a caller bug must not dress itself as a
    plausible degraded state.
    """
    try:
        rows = store.list_credentials(_QWENCLOUD_TICKET_AUGMENTS)
    except sqlite3.ProgrammingError:
        raise
    except Exception:  # noqa: BLE001 — an unreadable store is not a missing credential
        return True
    return bool(rows)


def qwencloud_ticket_command(args: argparse.Namespace) -> int:
    """Store / inspect / remove the QwenCloud console session cookie.

    The value is read from STDIN and never from argv: a command line is
    readable by any process running as you (`ps`) and lands in shell history.
    This mirrors `lop secret set NAME`.
    """
    from local_operator.providers.auth_store import AuthStore

    store = AuthStore()
    try:
        return _qwencloud_ticket_action(getattr(args, "qwencloud_command", None), store)
    finally:
        # Same discipline as every sibling command in this file (see
        # `login_command`, `logout_command`, `login_status_command`): the store
        # owns a SQLite connection and, lazily, the usage cache's, and a verb
        # that returns without closing leaks both.
        store.close()


def _qwencloud_ticket_action(command: str | None, store: Any) -> int:
    """One `qwencloud-ticket` verb, against an already-open store.

    Split from the command so the store's lifetime is owned in exactly one
    place, and so a test can drive a verb against a temp store it still needs
    to read assertions from afterwards.
    """
    # `_invalidate_cached_usage` is auth_cli's, reused rather than re-spelled:
    # `lop login` and `lop logout` already drop the cached usage row on a
    # credential change (auth_cli.py:400, :409) precisely so the change shows
    # at once, and the console ticket is a credential change this command's
    # own cache key cannot observe -- `_account_fingerprint` reads the
    # provider's rows, and the ticket lives in a separate namespace on
    # purpose. A second mechanism here would be the "second way of doing
    # something" this codebase treats as a defect.
    from local_operator.providers.auth_cli import _invalidate_cached_usage
    from local_operator.providers.qwencloud_console import (
        QWENCLOUD_TICKET_STALE_MS,
        TicketStoreError,
        TicketStoreLocked,
        _secret_is_present,
        delete_ticket,
        read_ticket_record,
        store_ticket,
    )

    if command == "set":
        if sys.stdin is None or sys.stdin.isatty():
            print(
                "lop qwencloud-ticket set: the value is read from stdin, never from "
                "the command line (argv is readable by any process running as you).\n"
                "  printf %s '<TICKET>' | lop qwencloud-ticket set",
                file=sys.stderr,
            )
            return 2
        value = sys.stdin.read()
        if value.endswith("\n"):
            value = value[:-1]
        try:
            store_ticket(store, value)
        except TicketStoreError as exc:
            print(f"lop qwencloud-ticket set: {exc}", file=sys.stderr)
            return 1
        # The length is read back as confirmation, but the write already
        # succeeded -- so a store that cannot be re-read afterwards must not
        # turn into "(0 characters)", which reads as "nothing was stored" on
        # the one command whose job is handling the secret. Report the length
        # written instead, and never a zero after a successful write.
        try:
            record = read_ticket_record(store)
        except TicketStoreError:
            record = None
        length = record["length"] if record else len(value.strip())
        # Same call and the same position as the login path's: after the write
        # succeeded, before the receipt. Without it a ticket swap changes
        # NOTHING the cache key observes -- the key was measured identical
        # across two different tickets -- so a latched `usage unavailable` row
        # is served for up to ~12.5 min (USAGE_UNAVAILABLE_RETRY_MS 10 min,
        # +/-25% jitter) after the user has already fixed the problem.
        _invalidate_cached_usage(_QWENCLOUD_TICKET_AUGMENTS, store)
        print(f"Stored QwenCloud console ticket ({length} characters).")
        return 0

    if command == "status":
        # Shares `rm`'s hazard in a quieter form: reporting "nothing stored"
        # for an unreadable store would tell the user the cookie is already
        # gone when it is on disk and readable by anything that can open the
        # file.
        try:
            record = read_ticket_record(store)
            # The ROW is not the whole answer, and this is the other half of the
            # fix `rm` already needed. `store_ticket` writes the VALUE first and
            # the row second, on purpose, so a crash, a kill or an older
            # `auth.db` restored from a backup leaves a SECRET ORPHAN: a live
            # value with nothing pointing at it (see `delete_ticket`'s
            # docstring for the same state from the revoking side).
            # `read_ticket_record` reads rows and reports None for it, so
            # without this probe `status` answers "no ticket stored" over a live
            # full-account cookie -- hiding the exposure AND pointing away from
            # `rm`, the only verb that revokes it.
            #
            # INSIDE this `try`, deliberately: `_secret_is_present` raises the
            # same `TicketStoreError` family (`TicketStoreLocked`,
            # `TicketStoreUnreadable`), and a store that cannot be read must
            # reach the UNKNOWN branch below rather than fall through to the
            # no-ticket receipt.
            unreferenced_value = record is None and _secret_is_present(None)
        # No separate `TicketStoreLocked` clause here, unlike `rm`. The remedy
        # ("Run `lop secret unlock`") lives in the EXCEPTION MESSAGE that Slice
        # A raises, and this clause interpolates it, so a locked store already
        # exits 1 naming the remedy. A specific clause would need a body
        # identical to this one -- measured: deleting it changed neither the
        # exit code nor a byte of stderr. `rm` earns its second clause because
        # its two bodies genuinely differ.
        except TicketStoreError as exc:
            print(
                f"lop qwencloud-ticket status: {exc}\n"
                "  Whether a ticket is stored is UNKNOWN; this is not the same "
                "as none being stored.",
                file=sys.stderr,
            )
            return 1
        if record is None:
            if unreferenced_value:
                # Exit 0, not 1: unlike the UNKNOWN branch above, the question
                # WAS answered -- something is stored, and the answer is
                # "stored with no metadata row". The metadata-orphan WARNING
                # below also rides exit 0, and 1 is this verb's dedicated code
                # for "could not be read at all".
                #
                # No length and no age: both live in the row that is missing,
                # and reading them would mean retrieving the value, which is
                # the one thing this verb promises never to do. `/usage` does
                # not read the value either -- with no row there is nothing for
                # the controller to find -- so this state is present AND
                # unusable, and the honest receipt says both.
                print(
                    "A QwenCloud console ticket VALUE is stored, but no metadata "
                    "row records it, so /usage does not read it and its age and "
                    "length are unknown."
                )
                print('  Run "lop qwencloud-ticket rm" to revoke it.')
                return 0
            print("No QwenCloud console ticket stored.")
            print("  printf %s '<TICKET>' | lop qwencloud-ticket set")
            return 0
        captured = record.get("captured_at")
        age_ms = 0
        if captured:
            age_ms = int(time.time() * 1000) - int(captured)
            days = age_ms / 86_400_000
            age = f"{days:.1f} days old"
        else:
            age = "age unknown"
        print(f"QwenCloud console ticket stored ({record['length']} characters, {age}).")
        # A METADATA ORPHAN: the row in `auth.db` says a ticket was stored but
        # its encrypted value is gone, so `/usage` has nothing to read. The
        # default is True so a pre-migration row, whose value still lives in
        # `auth.db` itself, does not trip the warning.
        if not record.get("secret_present", True):
            print(
                "  WARNING: the ticket's metadata is stored but its ENCRYPTED VALUE "
                "is missing, so /usage cannot read it. Re-run "
                "\"printf %s '<TICKET>' | lop qwencloud-ticket set\" to restore it."
            )
        if captured and age_ms > QWENCLOUD_TICKET_STALE_MS:
            print(
                "  This is older than a console session usually lasts. If /usage has "
                "stopped showing the 7 Day Credits window, capture a fresh cookie."
            )
        # A `lop /update` or `uv tool upgrade` reinstalls from PyPI and reverts
        # a locally built console fetcher while leaving this row in place --
        # a silent "nothing reads it" state. Say so instead of looking healthy.
        # Probed with getattr rather than a direct import: the symbol is absent
        # by design in a build without the console fetcher, and an import of a
        # name that may not exist is a static-analysis error rather than the
        # runtime question actually being asked ("does this build read it?").
        try:
            from local_operator.providers import usage as _usage_module

            has_fetcher = hasattr(_usage_module, "fetch_qwencloud_console_usage")
        except ImportError:
            has_fetcher = False
        if not has_fetcher:
            print(
                "  WARNING: this build has no QwenCloud console fetcher, so the stored "
                "ticket is not read by anything. Reinstall local-operator from a build "
                "that includes it."
            )
        # The precondition that actually GATES the feature, and the only one
        # `status` did not name. `/usage` asks `can_report_usage`, which
        # requires `is_usable` -- and the ticket deliberately cannot satisfy
        # that: it lives in its own namespace precisely so lop never concludes
        # it can CHAT through alibaba-token-plan on a read-only console cookie.
        # So the ticket AUGMENTS a Token Plan credential, it does not replace
        # one, and with no such row `/usage` renders nothing at all for a
        # perfectly valid ticket. Making the precondition visible is the fix
        # available here; satisfying it would be the blast radius the separate
        # namespace exists to avoid.
        if not _qwencloud_credential_row_exists(store):
            print(
                f"  WARNING: no {_QWENCLOUD_TICKET_AUGMENTS} credential is stored, so "
                "/usage will not show the 7 Day Credits window. This ticket AUGMENTS "
                "an existing Token Plan credential rather than replacing one — it "
                "reports usage but cannot authenticate the provider. Run "
                f"'lop login {_QWENCLOUD_TICKET_AUGMENTS}' (or restore the API key) "
                "to make the window visible."
            )
        return 0

    if command == "rm":
        # A revoke that cannot PROVE it worked must not report success. This is
        # the only mitigation the user has for a full-account cookie held in
        # plaintext, so "probably gone" is the one answer this command may not
        # give: it would remove the mitigation and say it had worked.
        try:
            removed = delete_ticket(store)
        except TicketStoreLocked as exc:
            # First, for the same reason as `status`: caught by the clause
            # below it, the one failure with a remedy reads as one without.
            print(
                f"lop qwencloud-ticket rm: {exc}\n"
                "  The ticket MAY STILL BE STORED. Unlock the secret store and "
                "re-run, and revoke the session in the QwenCloud console to "
                "be certain.",
                file=sys.stderr,
            )
            return 1
        except TicketStoreError as exc:
            print(
                f"lop qwencloud-ticket rm: {exc}\n"
                "  The ticket MAY STILL BE STORED, in the credential store, the "
                "encrypted secret store, or both. Re-run once they are "
                "readable, and revoke the session in the QwenCloud console to "
                "be certain.",
                file=sys.stderr,
            )
            return 1
        if not removed:
            print("No QwenCloud console ticket stored.")
            return 0
        # Symmetrical with `set`, for the reason `run_logout` drops its
        # listing: a window fetched under the credential just removed must not
        # keep rendering as though it were live.
        _invalidate_cached_usage(_QWENCLOUD_TICKET_AUGMENTS, store)
        print("Removed the stored QwenCloud console ticket.")
        # The advice belongs HERE and not only on the failure path: someone
        # revoking this credential is usually doing it because it may be
        # compromised, and success is the moment they stop worrying. Deleting
        # the row ends local use, but SQLite can keep the freed page contents
        # in the freelist until a VACUUM, and the SESSION ITSELF stays valid
        # server-side regardless -- so this command cannot be the whole answer.
        print(
            "  This ends local use of the cookie. The browser session itself is "
            "still valid until you sign it out in the QwenCloud console — do "
            "that too if the cookie may have been exposed."
        )
        return 0

    if command == "migrate":
        return _qwencloud_ticket_migrate(store)

    print(
        "usage: lop qwencloud-ticket {set,status,rm,migrate}\n"
        "  printf %s '<TICKET>' | lop qwencloud-ticket set",
        file=sys.stderr,
    )
    return 2


def _qwencloud_ticket_migrate(store: Any) -> int:
    """Move a plaintext ticket out of `auth.db` and into the encrypted store.

    VALUE-ONLY: the row is REWRITTEN, never deleted. Dropping it would lose
    `captured_at` (the clock `status`'s staleness warning measures) and
    `project_id` (what makes `_identity_key_for` upsert in place instead of
    INSERTing a duplicate on the next `set`).

    The secret is written and CONFIRMED before the plaintext is touched, so
    every interruption point leaves the value in BOTH stores rather than in
    neither, and re-running repairs it. A duplicate is recoverable; a loss is
    not.

    No redaction sink is registered here, unlike `mcp/credentials.py`. That
    path is HANDED a session and a manager to register against; a bare CLI
    invocation has neither, and there is no process-wide registry to fall back
    to -- `variables.register_redaction` is a method on a session's store. The
    guarantee this function makes instead is the stronger one, and the tests
    pin it on every branch: the value never reaches stdout, stderr, a file, or
    an exception message.
    """
    import json

    from local_operator.providers.auth_cli import _invalidate_cached_usage
    from local_operator.providers.qwencloud_console import (
        QWENCLOUD_CONSOLE_PROJECT_ID,
        QWENCLOUD_CONSOLE_PROVIDER,
        QWENCLOUD_TICKET_SECRET_NAME,
        TicketStoreError,
        TicketStoreLocked,
        _secret_is_present,
        _store_secret_value,
    )

    def _compact() -> tuple[bool, bool]:
        """`VACUUM` + a TRUNCATE checkpoint. Returns (compacted, blocked).

        One spelling for both callers -- the migrating path and the
        already-migrated no-op -- because they need the identical thing.

        `VACUUM` ALONE IS NOT ENOUGH. `AuthStore._connect` sets
        `journal_mode=WAL`, so the rebuild is itself written through the WAL
        and the freed plaintext stays readable in `auth.db` until a
        checkpoint lands. Measured on this branch: VACUUM only -> plaintext
        still found; VACUUM + checkpoint -> gone from all three files.
        """
        try:
            # `VACUUM` raises `cannot VACUUM from within a transaction` if one
            # is armed. Nothing above arms one today -- sqlite3 begins only
            # for DML and `upsert_credential` commits -- so this is a guard
            # against a later step adding DML here, at no measurable cost.
            store._conn.commit()
            store._conn.execute("VACUUM")
            store._conn.commit()
            row = store._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        except sqlite3.OperationalError as exc:
            # An `in_transaction` failure is a DEFECT IN THIS CODE, not a busy
            # database. Folding it into the "another process is reading"
            # warning would name the wrong cause in an honest-looking
            # sentence, which is the failure this module's comments already
            # warn about.
            if "within a transaction" in str(exc):
                print(
                    "lop qwencloud-ticket migrate: internal error — VACUUM ran "
                    "inside an open transaction. The value is encrypted, but the "
                    "old plaintext was NOT cleared. Please report this.",
                    file=sys.stderr,
                )
            return False, False
        except sqlite3.Error:
            return False, False
        # A checkpoint BLOCKED by another connection's read snapshot reports
        # itself in the FIRST COLUMN of this row and RAISES NOTHING: measured
        # `(1, 10, 1)` with the plaintext still readable in `auth.db-wal`.
        # Discarding the row and catching only `sqlite3.Error` is how this
        # command would claim success over a cookie that is still on disk.
        if row is not None and row[0] == 1:
            return False, True
        return True, False

    def _warn_not_compacted(blocked: bool) -> None:
        # Printed on BOTH paths, including the already-migrated no-op. On that
        # path we cannot know whether an earlier run left plaintext behind --
        # and that path exists precisely FOR the run that did. Over-warning
        # costs a line; staying silent loses the only signal the user has that
        # the credential is still readable.
        print(
            "  WARNING: the old plaintext could NOT be cleared from auth.db"
            + (" (another process is reading the database)" if blocked else "")
            + ". It remains readable on disk until a later VACUUM. Re-run "
            "`lop qwencloud-ticket migrate` when nothing else is using "
            "local-operator; re-running clears it."
        )

    def _both_copies(reason: str) -> int:
        """The one failure mode after the secret is confirmed: a duplicate."""
        print(
            f"lop qwencloud-ticket migrate: {reason}.\n"
            "  BOTH COPIES EXIST: the value is in the encrypted store AND still "
            "in auth.db, so nothing is lost. Re-run migrate.",
            file=sys.stderr,
        )
        return 1

    # `include_disabled=True` for the reason `read_ticket_record` documents: a
    # soft-deleted row is still ON DISK, and skipping it would leave plaintext
    # behind while reporting success. Clause order is load-bearing --
    # `ProgrammingError` is a caller bug and must keep propagating.
    try:
        rows = store.list_credentials(QWENCLOUD_CONSOLE_PROVIDER, include_disabled=True)
    except sqlite3.ProgrammingError:
        raise
    except (sqlite3.Error, OSError, json.JSONDecodeError) as exc:
        print(
            f"lop qwencloud-ticket migrate: the credential store could not be read "
            f"({type(exc).__name__}); nothing was changed.",
            file=sys.stderr,
        )
        return 1

    value = ""
    captured_at: Any = None
    already_migrated = False
    for row in rows:
        row_data = getattr(row, "data", None)
        if not isinstance(row_data, dict):
            continue
        if row_data.get("ticket"):
            value = str(row_data["ticket"])
            captured_at = row_data.get("captured_at")
            break
        if row_data.get("secret_name"):
            already_migrated = True

    if not value:
        if already_migrated:
            # THE NO-OP STILL COMPACTS, and that is the whole repair story.
            # `_warn_not_compacted` tells the user to re-run; the second run
            # lands HERE. An early return would make that advice a lie -- the
            # plaintext would stay readable forever while `status` reported
            # success (it sees no `ticket` key) and `rm` removed both stores
            # without ever checkpointing. Unrepairable by the tool.
            compacted, blocked = _compact()
            if not compacted:
                _warn_not_compacted(blocked)
            print("The QwenCloud console ticket is already in the encrypted secret store.")
            return 0
        # Returns WITHOUT touching `store._conn`: there is nothing to clear,
        # and this is the one path a caller with no live connection can drive.
        print("No QwenCloud console ticket stored; nothing to migrate.")
        return 0

    # Named before the write: C measured `ensure_broker` polling to
    # STARTUP_TIMEOUT_S twice when no broker is running, so a silent 10 s stall
    # here reads as a crash. Flushed so it lands before the stall, not after.
    print(
        "Encrypting the ticket (this may take a moment if the secret broker is starting)…",
        flush=True,
    )

    # THE SECRET IS WRITTEN FIRST. A crash between here and the rewrite below
    # leaves the value in both stores -- recoverable, and `/usage` keeps
    # working off the legacy row. The reverse order risks losing it entirely.
    #
    # `TicketStoreLocked` BEFORE `TicketStoreError`: it is a subclass, and a
    # broad clause above it swallows the one remedy the user can act on. A's
    # helper already puts `lop secret unlock` in the message; it is
    # interpolated, never re-spelled.
    try:
        _store_secret_value(value, None)
    except TicketStoreLocked as exc:
        print(
            f"lop qwencloud-ticket migrate: {exc}\n"
            "  NOTHING WAS CHANGED. The plaintext ticket is still in auth.db. "
            "Run `lop secret unlock`, then re-run migrate.",
            file=sys.stderr,
        )
        return 1
    except TicketStoreError as exc:
        print(
            f"lop qwencloud-ticket migrate: {exc}\n"
            "  NOTHING WAS CHANGED. The plaintext ticket is still in auth.db, so "
            "nothing is lost. If this persists, capture a fresh cookie and use "
            "\"printf %s '<TICKET>' | lop qwencloud-ticket set\" instead.",
            file=sys.stderr,
        )
        return 1

    # CONFIRMED, never inferred from the write returning -- the same rule
    # `delete_ticket` applies to the other direction, and the whole safety
    # property of this command. `describe`, not `get`, so the confirmation
    # costs no audit `get` event and does not stamp `last_used_at`.
    #
    # The raise is caught for the same reason the rest of this module catches:
    # a store that locked between the write and this check gives the SAME
    # answer (we cannot confirm, so nothing may be removed) and must report it
    # rather than surfacing a traceback carrying absolute local paths.
    try:
        confirmed = _secret_is_present(None)
    except TicketStoreError as exc:
        confirmed = False
        detail = f" ({exc})"
    else:
        detail = ""
    if not confirmed:
        print(
            f"lop qwencloud-ticket migrate: the encrypted value could not be "
            f"confirmed after writing it{detail}.\n"
            "  NOTHING WAS REMOVED. The plaintext ticket is still in auth.db, so "
            "nothing is lost. Re-run migrate.",
            file=sys.stderr,
        )
        return 1

    payload = {
        # The ORIGINAL `captured_at`, not `time.time()`: it is CAPTURE time,
        # which is what the ~7-day staleness warning measures. Re-stamping it
        # would tell the user a week-old cookie is fresh.
        "project_id": QWENCLOUD_CONSOLE_PROJECT_ID,
        "captured_at": captured_at,
        "secret_name": QWENCLOUD_TICKET_SECRET_NAME,
        "length": len(value),
    }
    # No `ticket` key -- that is the point. No `type` and no `source="login"`
    # either: each short-circuits `_identity_key_for` to None, and the next
    # `set` would INSERT a duplicate row instead of updating this one.
    try:
        store.upsert_credential(QWENCLOUD_CONSOLE_PROVIDER, payload)
    except sqlite3.ProgrammingError:
        raise
    except (sqlite3.Error, OSError) as exc:
        return _both_copies(f"the metadata row could not be rewritten ({type(exc).__name__})")

    # Confirmed by RE-READING, for the reason `delete_ticket` states: the
    # write returning is not evidence the plaintext is out of the API's view.
    try:
        remaining = store.list_credentials(QWENCLOUD_CONSOLE_PROVIDER, include_disabled=True)
    except sqlite3.ProgrammingError:
        raise
    except (sqlite3.Error, OSError, json.JSONDecodeError) as exc:
        return _both_copies(
            f"the rewritten row could not be re-read to confirm it ({type(exc).__name__})"
        )
    for row in remaining:
        row_data = getattr(row, "data", None)
        if isinstance(row_data, dict) and row_data.get("ticket"):
            return _both_copies("a row still carries a plaintext ticket after the rewrite")

    compacted, blocked = _compact()
    if not compacted:
        _warn_not_compacted(blocked)

    # Required, not belt-and-braces: C proved a cached note survives the full
    # ~5 min TTL, so a panel painted before the migration would keep rendering
    # stale state. Same call, same position as `set` and `rm` -- after the
    # write succeeded, before the receipt. Called regardless of `compacted`:
    # the value moved either way, so the cached row is stale either way.
    _invalidate_cached_usage(_QWENCLOUD_TICKET_AUGMENTS, store)
    print(
        f"Migrated the QwenCloud console ticket ({len(value)} characters) into the "
        f"encrypted secret store as {QWENCLOUD_TICKET_SECRET_NAME}."
    )
    # CONDITIONAL, and that is mandatory. This is the sentence that would
    # otherwise be false in exactly the blocked-checkpoint case: the value
    # encrypted, the plaintext still readable in `auth.db-wal`. The warning
    # above is the honest account there, and printing both would contradict
    # one with the other in a single command's output.
    if compacted:
        print("  The plaintext row in auth.db has been replaced with metadata only.")
    return 0


_MCP_INTERACTIVE_LOGIN_TIMEOUT_MS = 10 * 60_000


async def _mcp_login_server(name: str, cwd: Path) -> int:
    """Run one interactive MCP OAuth exchange and persist its token.

    The SDK's callback handler prints the authorization URL and accepts the
    final loopback redirect URL on stdin. ``McpTokenStorage`` writes the
    resulting token and client registration to ``auth.db``; a successful login
    therefore survives this short-lived manager and future Local Operator
    sessions reuse it without another browser round-trip.
    """
    from local_operator.mcp.auth import probe_oauth_capability, server_rejects_oauth
    from local_operator.mcp.config import load_all_mcp_configs
    from local_operator.mcp.manager import McpManager

    configs, _sources = load_all_mcp_configs(cwd)
    cfg = configs.get(name)
    if cfg is None:
        print(f"error: MCP server {name!r} is not configured", file=sys.stderr)
        return 1
    # NOT ``cfg.auth.type == 'oauth'`` any more — the same widening the TUI and
    # runtime grant paths already made (issue #367); for what counts as
    # evidence and why a url-only foreign import has none of the static kind,
    # see :func:`server_is_oauth_capable`. The static check refused exactly the
    # servers ``/mcp login`` handles fine — two gates for one question,
    # disagreeing.
    #
    # The split below preserves the F3 protection that motivated the strict
    # check: ``server_rejects_oauth`` is the STATIC impossibility, and it stays
    # a free, hard refusal with the wording that tells them how to add an OAuth
    # server. Only the genuinely undecidable url-only case pays for the live
    # probe, which asks the network once instead of assuming every remote URL
    # is authenticable.
    if server_rejects_oauth(cfg):
        print(
            f"error: MCP server {name!r} is not OAuth-enabled; add a remote server with --oauth",
            file=sys.stderr,
        )
        return 1
    if not await probe_oauth_capability(cfg):
        # Reachable only for a remote server that answered the probe without
        # advertising an authorization server: it may be genuinely public, or
        # discovery may be unreachable. Distinct wording, because "add a remote
        # server with --oauth" is useless advice for a server that IS one.
        #
        # The parenthetical is load-bearing, not decoration: the probe swallows
        # its transport exception and returns False for BOTH readings, so after
        # the 30 s worst case against an unroutable host a bare "does not use
        # OAuth login" tells the user a falsehood about their server when the
        # real fault is the URL or the network. The message has to admit the
        # ambiguity the code already acknowledges.
        print(
            f"error: MCP server {name!r} does not use OAuth login"
            " (no authorization server discovered — check the URL and your network)",
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

    A failed removal is NOT automatically a failed reauth, and a successful
    login is not automatically a successful reauth. Reauth's contract is "end
    up authenticated having genuinely re-granted", not "delete a row", so
    having nothing to delete is a satisfied precondition, while a delete that
    was ATTEMPTED and failed leaves a credential the coming grant would
    silently reuse. :func:`clear_for_reauth` is the single place that tells
    those apart — shared with the TUI's ``/mcp reauth`` so the two cannot
    answer one question differently, which is the defect this path exists to
    remove. A remote server that advertises no authorization server is still
    refused downstream by the live probe inside :func:`_mcp_login_server`.
    """
    from local_operator.mcp.auth import clear_for_reauth

    error = clear_for_reauth(name, cwd)
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
        from local_operator.mcp.auth import McpCredentialDeleteError, mcp_logout_server

        try:
            error = mcp_logout_server(args.name, Path.cwd())
        except McpCredentialDeleteError as exc:
            # The row was FOUND and its delete FAILED, so the credential is
            # still on disk and this server is still logged in. That is a
            # retryable condition — a sibling session holding an EXCLUSIVE
            # sqlite transaction is enough to cause it — so it is reported as
            # one error line like every other failure at this boundary.
            #
            # Caught rather than left to propagate: an escaping exception
            # reaches ``main``'s generic handler, which prints a stack trace
            # and "Please review and correct the error to continue". That
            # frames a locked store as a bug in local-operator rather than
            # something the user can act on, and buries the one sentence that
            # matters — the credential survived — under 30 lines of traceback.
            print(
                f"error: MCP logout failed: {exc}. Retry once the process holding "
                "the credential store has released it.",
                file=sys.stderr,
            )
            return 1
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
    session = await create_session(args, config_manager, agent_registry, has_ui=False)
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
        _print_first_run_quickstart()
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

    return _preflight_api_key(hosting, config_manager.config_dir, require_key=require_api_key)


def _print_first_run_quickstart() -> None:
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
    hosting: str, config_dir: Path | None, *, require_key: bool = True
) -> int | None:
    """Verify that the provider has a credential source.

    Stored OAuth and API-key rows satisfy preflight by presence, including a
    row under temporary stream-time backoff. The stream owns refresh and
    failover; doing network refresh here can turn a transient OAuth failure
    into a false "API key is required" startup error that prevents access to
    the TUI's login command. With no stored row, the AuthStore cascade still
    checks the exported environment (the legacy ``credentials.env`` file is no
    longer a rung, PR2a).

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

        auth_store = AuthStore(config_dir=config_dir)
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

    The legacy main() constructed ``SchedulerService`` (JobManager, the same
    minimal manager the server app uses), started
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
        from local_operator.types import OperatorType

        base_dir = config_dir()
        config_manager = ConfigManager(base_dir)
        from local_operator.agents import AgentRegistry  # lazy: heavy module

        agent_registry = AgentRegistry(base_dir)

        from local_operator.console import VerbosityLevel

        scheduler_service = SchedulerService(
            agent_registry=agent_registry,
            config_manager=config_manager,
            env_config=get_env_config(),
            operator_type=OperatorType.CLI,
            verbosity_level=(
                VerbosityLevel.DEBUG
                if os.environ.get("LOCAL_OPERATOR_DEBUG", "false") == "true"
                else VerbosityLevel.VERBOSE
            ),
            job_manager=JobManager(),
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


#: Subcommands whose process is LONG-LIVED and therefore worth re-execing to
#: rename. Everything absent from this set — `send`, `sessions`, `config`,
#: `credential`, `stop`, `login` — finishes in well under a second and never
#: lingers in Activity Monitor, so paying the re-exec there would be pure
#: latency for a row nobody can see. ``None`` is the bare interactive launch.
_BRANDED_SUBCOMMANDS = frozenset({None, "serve", "mobile", "exec", "browser"})


def _maybe_brand_process(args: argparse.Namespace) -> None:
    """Re-exec through the branded interpreter image when it is worth it.

    WHY THIS IS GATED, and why the gate is not "always". A re-exec restarts the
    interpreter, so the replacement process re-pays every import this one has
    already done — and `local_operator.cli` alone costs 221 ms of imports
    (pydantic/yaml, measured with `-X importtime`). Measured end to end here:
    `lop --version` went 223 ms -> 410 ms, i.e. **+186 ms**, not the ~35 ms a
    bare-interpreter re-exec suggests. That is a bad trade for a command that
    exits immediately and whose row nobody ever sees in Activity Monitor.

    So the cost is paid only by processes that live long enough for the name to
    be the point: the interactive TUI, `serve`, `mobile`, `exec`, `browser`.
    For those, ~190 ms sits against a startup already north of a second and a
    process that then runs for minutes to hours.

    Detached children (session runtimes, eval workers, exec workers) do NOT go
    through this path at all: their parent spawns them with `executable=` set to
    the branded image, so they are born branded and pay nothing.

    Never raises, and returns normally whenever branding is unavailable — in
    which case this process carries on exactly as it does today, still alive to
    have repaired the link for next time. That liveness is the whole reason the
    `lop` shebang is NOT pointed at the link; see `procname.py`.
    """
    subcommand = getattr(args, "subcommand", None)
    if subcommand not in _BRANDED_SUBCOMMANDS:
        return
    procname.reexec_branded(_process_label(args))


def _process_label(args: argparse.Namespace) -> str | None:
    """The argv[0] this process should carry, or None for the bare brand.

    Only fixed templates and machine-generated values (ports, an agent name that
    is already slugged upstream) reach this: argv is world-readable through
    `ps`, so no prompt, path, or model-produced text may ever be interpolated
    into it. See the label vocabulary in `procname.py`.
    """
    subcommand = getattr(args, "subcommand", None)
    if subcommand == "serve":
        return procname.branded_argv0(procname.LABEL_SERVE, port=int(getattr(args, "port", 0)))
    if subcommand == "mobile":
        return procname.branded_argv0(
            procname.LABEL_MOBILE, port=int(getattr(args, "port", 0) or 0)
        )
    if subcommand is None:
        # The interactive TUI. The session id is not minted yet at this point
        # (the session is built after the re-exec), so the agent name is all the
        # identity available — and it is the field an operator actually scans
        # for when several sessions are running.
        # `dest="agent_name"` — `--agent` is the flag, not the attribute.
        agent = getattr(args, "agent_name", None)
        if isinstance(agent, str) and agent:
            return procname.branded_argv0(
                procname.LABEL_SESSION_AGENT, agent=procname.safe_field(agent)
            )
    return None


def main() -> int:
    # Name this process in the OS process listing. On Linux this is a
    # ~microsecond `prctl` on the current process and nothing else happens; the
    # macOS half (which needs a re-exec) is deferred until after the parse, for
    # the cost reason documented at `_maybe_brand_process`.
    procname.set_process_name()

    # FIRST, before anything else can log. `helpers.py` used to configure the
    # root logger as an import side effect; now the entry point owns it, which
    # is what lets the TUI branch below swap the console handler for a file.
    configure_cli_logging()
    try:
        parser = build_cli_parser()
        args = parser.parse_args()

        # macOS: replace this process with a branded image so Activity Monitor
        # stops showing a wall of `python3.x`. NEVER RETURNS when it brands;
        # a no-op everywhere else. Placed after the parse so `--version` and
        # `--help` (which argparse exits from inside `parse_args`) never pay it.
        _maybe_brand_process(args)

        # Prime the login-shell PATH only on paths that actually spawn
        # subprocess work: the interactive session, exec, serve and mobile all
        # run shell commands whose PATH must match a login terminal's, but
        # `config list`, `credential`, `login`, `agents` and the like never
        # spawn a tool — yet every one of them used to pay a full login-shell
        # round-trip (`$SHELL -l -c 'echo $PATH'`) on startup. The helper is
        # NOT cached, so this membership test is the only thing keeping that
        # round-trip off the cheap subcommands — adding a name here costs
        # every invocation of it one login shell. ``None`` is the bare
        # interactive launch.
        #
        # `wake` is here for `lop wake serve`, the documented foreground form
        # of the LaunchAgent for anyone running the supervisor under their own
        # supervisor. It reaches the same `serve()` and therefore spawns
        # runtimes through `session/runtime/launch.py`, which propagates
        # `dict(os.environ)` — so without the bootstrap it hands every
        # wake-driven turn whatever PATH its own supervisor happened to have.
        _SUBPROCESS_SUBCOMMANDS = frozenset(
            {"exec", "serve", "mobile", "browser", "tunnel", "wake"}
        )
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
            # path. ``create_session(has_ui=True)`` returns a AttachedSession when
            # another process owns the transcript, so the STANDARD OperatorApp
            # renders it with no standalone attach app, exit-75 relaunch, or
            # visible mode. The shared factory still protects the sole-writer
            # invariant; exec/headless retains its refusal below.

        os.environ["LOCAL_OPERATOR_DEBUG"] = "true" if args.debug else "false"
        # (CL-12) No env_config binding here: the scheduler wrapper resolves its
        # own env config and the session factory does the same lazily — a
        # dead local would only invite drift.
        base_dir = config_dir()
        # THE ONE place config migrations run: the `lop` entry point, for the
        # config dir this command is about to use. Never from ConfigManager
        # construction — see ``local_operator.config_migrations``.
        from local_operator.config_migrations import run_startup_migrations

        run_startup_migrations(base_dir)
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
            elif args.config_command == "instructions":
                return config_instructions_command(args)
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
                from local_operator.providers.radient_credentials import (
                    resolve_radient_credential_sync,
                )

                config_manager = ConfigManager(base_dir)
                base_url = _radient_hub_base_url(config_manager)
                api_key = resolve_radient_credential_sync(config_manager.config_dir, base_url)
                if not api_key:
                    print(
                        "\n\033[1;31mError: RADIENT_API_KEY is required to push to Radient\033[0m"
                    )
                    return 1
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
                # The zip is uploaded and finished with inside this block, so
                # the context manager reclaims its temp directory on every
                # exit path. The bare export_agent() left one behind per push.
                with agent_registry.exported_agent_archive(agent.id) as (zip_path, _):
                    try:
                        agent_id = agent_registry.upload_agent_to_radient(
                            radient_client, agent_id_to_overwrite, zip_path
                        )
                        # Report the OUTCOME, never the request. The registry
                        # returns None when it really overwrote and the hub's new
                        # id when it created a listing, and branching on the flag
                        # instead printed "as overwrite" for a listing that had
                        # just been created — without ever naming it, so the user
                        # was left holding a duplicate they could not even find to
                        # delist. `--id` takes a LOCAL id, which nothing aligns
                        # with a hub listing id (import mints a fresh uuid), so
                        # that create branch is the ordinary one here, not a
                        # corner: the hub answers GET /v1/agents/{local id} with a
                        # 404.
                        if agent_id is None:
                            print(
                                f"\n\033[1;32mSuccessfully pushed agent '{agent.name}' as "
                                f"overwrite to Radient (ID: {agent_id_to_overwrite})\033[0m"
                            )
                        else:
                            print(
                                f"\n\033[1;32mSuccessfully pushed agent '{agent.name}' to "
                                f"Radient. New agent ID: {agent_id}\033[0m"
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
                base_url = _radient_hub_base_url(config_manager)
                radient_client = RadientClient(api_key=None, base_url=base_url)
                try:
                    imported_agent, renamed_from = agent_registry.download_agent_from_radient(
                        radient_client, agent_id
                    )
                    print(
                        f"\n\033[1;32mSuccessfully pulled agent '{imported_agent.name}' "
                        f"(ID: {imported_agent.id}) from Radient\033[0m"
                    )
                    if renamed_from is not None:
                        # The user asked for a name they already hold locally, so
                        # the row landed under a suffix (contract §3.6). Saying so
                        # is what stops the pull reading as "it did nothing", and
                        # the name is echoed here because the alternative is
                        # grepping the registry for where it went. "with that
                        # name", not "called X": the row the user already holds
                        # may carry a different spelling (``Coder`` arriving over
                        # a local ``coder``), and naming the wrong one sends them
                        # looking for a row that is not there.
                        print(
                            f"\033[1;33m  Renamed from '{renamed_from}': you already have an "
                            f"agent with that name.\033[0m"
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
            return serve_command(args.host, args.port, args.reload, listener_fd=args.listener_fd)
        elif args.subcommand == "mobile":
            return mobile_command(args)
        elif args.subcommand == "tunnel":
            from local_operator.tunnels.cli import main as tunnel_main

            return tunnel_main(args)
        elif args.subcommand == "secret":
            from local_operator.secrets.cli import main as secret_main

            return secret_main(args)
        elif args.subcommand == "operator":
            from local_operator.operator.cli import main as operator_main

            return operator_main(args)
        elif args.subcommand == "pair":
            from local_operator.operator.pair import main as pair_main

            return pair_main(args)
        elif args.subcommand == "qwencloud-ticket":
            return qwencloud_ticket_command(args)
        elif args.subcommand == "browser":
            return browser_command(args)
        elif args.subcommand == "send":
            return send_command(args)
        elif args.subcommand == "sessions":
            return sessions_command(args)
        elif args.subcommand == "stop":
            return stop_command(args)
        elif args.subcommand == "refresh":
            return refresh_command(args)
        elif args.subcommand == "resume-click":
            # Function-local like every other runtime import here: this module
            # is on the CLI startup path and must not pull the spawn/terminal
            # graph into every `lop` invocation.
            from local_operator.tui.resume_click import open_session

            if open_session(args.session):
                return 0
            # A CLICK THAT DOES NOTHING NEEDS A REASON, and on a real click this
            # is only half of it. The receipt is what a HAND-RUN gets, and it is
            # what makes the failure path debuggable from a terminal; the click
            # itself has no terminal (its three streams are /dev/null), so the
            # ladder raises the same sentence as an out-of-band toast before
            # returning False (UX round 2, U10).
            print(
                f"could not open a terminal for session {args.session} — "
                f"run: lop --resume {args.session}",
                file=sys.stderr,
            )
            return 1
        elif args.subcommand == "wake":
            return wake_command(args)
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

            return update_command(
                check=bool(getattr(args, "check", False)),
                refresh_daemons=bool(getattr(args, "refresh_daemons", False)),
                from_snapshot=getattr(args, "from_snapshot", None),
                services=not bool(getattr(args, "no_services", False)),
            )
        elif args.subcommand == "services":
            # Lazy for the same reason as ``update``: this pulls the serve
            # registry, and the CLI's own startup path must stay stdlib-light.
            from local_operator.services import (
                print_refreshes,
                restart_services,
                status_lines,
            )

            command = getattr(args, "services_command", None)
            if command == "status":
                for line in status_lines():
                    print(line)
                return 0
            if command == "restart":
                wait = getattr(args, "wait", None)
                refreshes = (
                    restart_services(wait_s=wait) if wait is not None else restart_services()
                )
                print_refreshes(refreshes)
                return 0
            # Mirror `install`'s dispatch instead of argparse's (design review D8):
            # `parser.error` dumped the WHOLE program's usage here — 223 columns of
            # every verb under a second `usage:` prefix — when what the reader mistyped
            # is a subcommand of this one group. It exits 2 where `install` returns 1
            # for the same situation: 2 is argparse's own usage code and the one this
            # path already exited with through `parser.error`, so nothing that scripts
            # the exit status sees a change (round 11 R11-4).
            print("usage: lop services {status, restart}", file=sys.stderr)
            return 2
        elif args.subcommand == "install":
            # Same lazy import, same reason. The generation layout's own verbs:
            # they install nothing from a network, so they never consult PyPI.
            from local_operator.update import (
                DEFAULT_KEEP_GENERATIONS,
                install_migrate_command,
                install_prune_command,
                install_status_command,
            )

            action = getattr(args, "install_command", None)
            if action == "prune":
                # The parser's default IS the policy default and its type check
                # refuses a negative, so this is only ever a non-negative int
                # (design review D5/D15; review round 6 R6-4: the ``None`` arm this
                # used to carry became unreachable when D5 landed).
                keep = int(getattr(args, "keep", DEFAULT_KEEP_GENERATIONS))
                return install_prune_command(keep=keep)
            if action == "migrate":
                return install_migrate_command()
            if action == "status":
                return install_status_command()
            print("usage: lop install {status, prune, migrate}", file=sys.stderr)
            return 1
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
                from local_operator.resume import live_runtime_pid, resolve_resume_id

                try:
                    exec_resume_id = resolve_resume_id(config_dir(), str(args.resume))
                except Exception:
                    exec_resume_id = str(args.resume)
                exec_owner = live_runtime_pid(config_dir(), exec_resume_id)
                if exec_owner is not None and exec_owner != os.getpid():
                    print(
                        f"\033[31msession {exec_resume_id} is already open in "
                        f"another process (pid {exec_owner}) — watch and steer "
                        "it there, or from the phone session list\033[0m",
                        file=sys.stderr,
                    )
                    return 1
            from local_operator.exec_mode import ExecArgs, job_status, run_exec

            if args.status:
                import json

                from local_operator.exec_startup import STARTUP_FIELDS

                run_options = (
                    *STARTUP_FIELDS,
                    "background",
                    "control",
                    "resume",
                    "agent_name",
                    "agent_id",
                    "yolo",
                    "train",
                )
                if args.command is not None or any(getattr(args, key, None) for key in run_options):
                    print(
                        "--status cannot be combined with a prompt or run options", file=sys.stderr
                    )
                    return 1
                state = job_status(args.status)
                if not state:
                    # Every other refusal in this feature names a recovery
                    # command; this one had none to name (there is no `lop
                    # exec --list`), so it names where the answer actually
                    # lives. Also carries the `exec failed: ` prefix its
                    # siblings all use, which it was alone in omitting.
                    from local_operator.exec_mode import JOBS_FILE, logs_dir

                    print(
                        f"exec failed: No exec job {args.status!r}; job IDs are printed "
                        f"by --background and recorded in {logs_dir() / JOBS_FILE}",
                        file=sys.stderr,
                    )
                    return 1
                print(json.dumps(state, ensure_ascii=False))
                return 0
            # A `lop` command an agent ran may not open a session of its own
            # UNLESS the session that ran it may delegate — an agent whose
            # inventory holds `task` is allowed to open separate top-level
            # sessions when the user asked for them (the case this relaxation
            # answers), while one that does not hold `task` is refused exactly
            # as before. Deferred to `agent_shell.py` so both entry points read
            # one rule. What such a run starts is a TOP-LEVEL conversation: the
            # operator's session list, desktop sidebar and phone history would
            # list it as a chat they opened, and it runs outside this session's
            # job manager, so nothing here can see, steer, cancel or account
            # for it — which is why an ALLOWED run is stamped `agent-shell` in
            # `session_factory`. `--status` returned above, so the read-only
            # form stays reachable for every shell, and the documented escape
            # for QA runs is `LOCAL_OPERATOR_ALLOW_NESTED_SESSION`
            # (docs/EXEC.md) — deliberately not named to the model in the text.
            refusal = exec_session_refusal()
            if refusal is not None:
                print(f"exec failed: {refusal}", file=sys.stderr)
                return 1
            exec_args = ExecArgs(
                background=args.background,
                json_mode=args.json_mode,
                team=args.team,
                profile=args.profile,
                goal=args.goal,
                clear_goal=args.clear_goal,
                loop=args.loop,
                loop_goal=args.loop_goal,
                name=args.name,
                effort=args.effort,
                agent_name=args.agent_name,
                agent_id=getattr(args, "agent_id", None),
                yolo=args.yolo,
                hosting=args.hosting,
                model=args.model,
                train=args.train,
                resume=getattr(args, "resume", None),
                # getattr, like the additive flags above it: `exec` is not the
                # only subcommand routed through this Namespace in tests, and a
                # missing attribute must read as "off", never raise.
                control=bool(getattr(args, "control", False))
                # ``--workstream`` IMPLIES ``--control``, here rather than in the
                # parser so the implication holds for both entry points that
                # build this object. A workstream is a row the operator is meant
                # to watch and steer, and the sidebar can only follow and steer a
                # run that published a live discovery record — so without the
                # socket the row would appear and then never move. Spelling it
                # as a real assignment (not a documented pairing) is what makes
                # the flag's help text true.
                or bool(getattr(args, "workstream", False)),
                tools=getattr(args, "tools", None),
                # THE SUPERVISOR'S DESCRIPTOR, forwarded here or nowhere (stage E).
                # Its absence was a real gap rather than a tidy-up: `run_session`
                # reads the field off this ExecArgs object — not off the argparse
                # Namespace — so omitting it here made `--supervisor-fd` a flag that
                # parsed, validated, and then silently did nothing: the run minted no
                # capability and wrote nothing upward, and a supervisor waited out
                # its whole timeout. Found by the e2e cell that drives a real
                # supervised run (`test_a_supervised_run_is_approved_through_the_
                # handoff`), which is why that cell exists rather than an in-process
                # probe (agent review round 6, R6-5).
                supervisor_fd=getattr(args, "supervisor_fd", None),
                # THE OPERATOR'S OWN REQUEST THAT THIS RUN BE A WORKSTREAM
                # (`lop exec --workstream`), carried to `session_factory._prepare`
                # through the narrow namespace: absent, the run is ephemeral and
                # hidden exactly as before. In `STARTUP_FIELDS`, so the detached
                # worker is told the same thing.
                workstream=bool(getattr(args, "workstream", False)),
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
                key_result = _preflight_api_key(hosting, base_dir)
                if key_result is not None:
                    return key_result
            return run_exec(args.command, exec_args)

        # The interactive path is the fall-through — every subcommand returned
        # above — so this ONE check covers `lop`, `lop --resume ID`, `--tui` and
        # every future interactive flag together, and it sits FIRST so a refused
        # run has written nothing: no config override, no registry row for an
        # autosave agent. It honours the SAME escape as the exec path (the rule
        # is `agent_shell.py`'s, in one place, on purpose), because a pty
        # harness drives this front end exactly as a bench drives exec — and it
        # deliberately does NOT honour the delegation allowance that path gained
        # (operator, 2026-09-19): an agent has no terminal, so this opens a front
        # end on the OPERATOR's screen rather than the separate session a
        # delegating agent was given permission to start. The refusal text is
        # one message for both readers (see `refusal_message`).
        # What differs between the two paths is only who DROPS the marker: the
        # places a session opens a conversation for its user — the TUI restart,
        # `/fork`'s window, and both rungs of a notification click (the terminal
        # and the desktop app) — pass `agent_shell.without_agent_shell_marker`,
        # since those are the user's gestures and not an agent's command.
        refusal = interactive_session_refusal()
        if refusal is not None:
            from local_operator.cli_style import ERROR, paint

            # The PREFIX carries the colour, the diagnostic does not (design
            # round 1, D1): the whole 646 bytes painted bold red is nine wrapped
            # lines of alarm for a message whose content is "you took the wrong
            # route", and the exec path prints the same bytes with no colour at
            # all — one sentence must not render two ways depending on which
            # entry point hit it.
            #
            # `stream=sys.stderr` because that is where this text goes (design
            # round 2, D4): `paint`'s gate reads the stream it is told about, and
            # with the default it read stdout — so `lop 2> log` with stdout on a
            # terminal wrote escapes into a file, the one shape the gate exists
            # to keep plain.
            print(paint("Error: ", ERROR, stream=sys.stderr) + refusal, file=sys.stderr)
            return 1

        config_manager = ConfigManager(base_dir)

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

            viewer_started = False

            async def viewer_factory(resume_id: "str | None"):
                """Build the TUI's session facade: a VIEWER, never an owner.

                `lop` no longer hosts the agent. It opens a viewer bound to
                nothing; the work runs in a separate runtime process that the
                TUI engages as soon as it has adopted the viewer (so the band
                is complete before the first keystroke) and that exits when it
                has nothing left to do — immediately, if the viewer leaves
                without ever using it. That is what lets a turn survive the
                terminal closing.

                Two entry states, and the choice between them is just "is
                something already running for this id":

                - a live record → ATTACH, so a second terminal joins a session
                  that is already working rather than fighting it for the lease;
                - otherwise → COLD, which costs no process and no directory.

                ``--resume`` of a session whose runtime is gone is the cold
                case (which is also what `/reload` comes back as), and so is a
                fresh launch: a new id is minted here, in the viewer, and the
                runtime materialises the directory for it on the first real
                write — an engaged-but-unused session leaves nothing behind.
                """
                nonlocal viewer_started
                import uuid as _uuid

                from local_operator.harness.types import ModelSpec
                from local_operator.mobile.attach_client import find_runtime_record
                from local_operator.session.attached import (
                    AttachedSession,
                    frontend_attach_refusal,
                )
                from local_operator.session_factory import resolve_hosting_model

                config_directory = config_manager.config_dir
                # Same expression session_factory uses for a new session's
                # directory name, so ids minted by either path are one shape.
                session_id = resume_id or _uuid.uuid4().hex[:12]
                # The CLI flags belong to startup, not every /new or picker
                # resume in this viewer. Refresh the file only at this boundary.
                birth_args = argparse.Namespace(**vars(args))
                birth_args.resume = resume_id
                if viewer_started:
                    birth_args.hosting = None
                    birth_args.model = None
                viewer_started = True
                initial_model = None
                birth_agent = current_agent

                def resolve_birth():
                    # Both the config and saved selection can require disk I/O.
                    # The captured arguments above fix WHICH conversation this
                    # read belongs to before yielding to the worker.
                    return resolve_hosting_model(
                        birth_agent, birth_args, ConfigManager(config_directory)
                    )

                try:
                    provider, model_id = await asyncio.to_thread(resolve_birth)
                    initial_model = ModelSpec(provider=provider, model_id=model_id)
                except ValueError:
                    # Setup mode must still open without a configured model.
                    pass

                async def take_over():
                    # A viewer must never win the transcript lease — the
                    # runtime owns it, and a TUI holding one would look like a
                    # runtime to the wake supervisor's live-record rule. Kept
                    # wired (AttachedSession requires a factory) and deliberately
                    # unreachable: `_can_go_cold` routes owner loss to the cold
                    # state instead of to a takeover.
                    #
                    # THE OWNER PATH IS GONE FROM `lop`. This factory only ever
                    # returns a AttachedSession — attached when a live record
                    # exists, cold otherwise — so the TUI process never builds
                    # a `Session`, never takes the lease, and never writes the
                    # transcript. That is what makes "at most one runtime per
                    # session, ever" true by construction rather than by
                    # arbitration between two kinds of writer.
                    #
                    # `session_factory.create_session` still exists and is
                    # still correct for the callers that legitimately own their
                    # session in-process: `lop exec`, the headless REPL, the
                    # server, and the mobile daemon's own attach path. Those
                    # are not the TUI and are deliberately left alone.
                    raise RuntimeError("a viewer never takes over a session")

                record = None
                if resume_id:
                    record, _owner = await asyncio.to_thread(
                        find_runtime_record, config_directory, session_id
                    )
                degraded_reason = ""
                paint_first_cwd = ""
                # PAINT FIRST, ATTACH BEHIND for a live owner whose conversation
                # is on disk: the cold facade below paints it, and the TUI's eager
                # engage binds it to that owner after the first paint (the bind
                # replays whatever the owner wrote in between). The blocking
                # `connect` held `lop --resume` on the owner's canonical sync —
                # 15 s against a busy one — before anything was drawn. A model
                # override is still sent through the live attach below, which is
                # the one path that can hand it to an existing owner; a cold
                # viewer's override is consumed by its bind.
                #
                # A STATICALLY REFUSED record (an owner on an older build) keeps
                # the dial below, mirroring the TUI's `/resume` branch: `connect`
                # raises that refusal, and it is the only thing that sets
                # `degraded_reason`, i.e. the "opened without live session state
                # — needs protocol >= N" sentence. Painting first there would
                # open an inert conversation with no word about why (review
                # round 1, F1), and the dial costs nothing: it refuses before
                # any socket is opened.
                if (
                    record is not None
                    and frontend_attach_refusal(record) is None
                    and not (initial_model is not None and (birth_args.hosting or birth_args.model))
                    and (config_directory / "sessions" / session_id / "transcript.jsonl").is_file()
                ):
                    # The owner's directory, not this terminal's: the band paints
                    # it before the bind lands, and a live conversation's cwd is
                    # the one it is working in.
                    paint_first_cwd = str(record.cwd or os.getcwd())
                    record = None
                if record is not None:
                    try:
                        attached = await AttachedSession.connect(
                            record,
                            session_id,
                            config_dir=config_directory,
                            takeover_factory=take_over,
                        )
                        if initial_model is not None and (birth_args.hosting or birth_args.model):
                            # A live resume needs the same deliberate override
                            # as a cold one; send it to the existing owner,
                            # never replace its journal from the viewer.
                            receipt = await attached.route_shared_slash(
                                "model", f"{initial_model.provider}/{initial_model.model_id}"
                            )
                            if isinstance(receipt, dict) and receipt.get("style") == "error":
                                attached.degraded_reason = str(receipt.get("text") or "")
                        return attached
                    except (ConnectionError, OSError, TimeoutError) as error:
                        # The runtime died between the scan and the dial, or is
                        # too old to attach to. Cold is the honest fallback:
                        # the conversation still opens and the next message
                        # starts a fresh runtime.
                        #
                        # But the fallback must not be SILENT. A live runtime
                        # we failed to attach to is a different situation from
                        # one that was never there, and the user cannot tell
                        # them apart from the screen: they see a session that
                        # opened without its state and no reason why. The
                        # oversized-frame case made that concrete — the failure
                        # was diagnosable in a log line nobody reads while the
                        # terminal said nothing (UX round 1, U2 / design round
                        # 1, D5). Carried onto the viewer so the TUI can put it
                        # on screen once there is a screen to put it on.
                        import logging as _logging

                        degraded_reason = str(error)
                        _logging.getLogger(__name__).warning(
                            "could not attach to the runtime for session %s (%s); "
                            "opening the conversation without live state",
                            session_id,
                            degraded_reason,
                        )
                viewer = await AttachedSession.cold(
                    session_id,
                    config_dir=config_directory,
                    cwd=paint_first_cwd or os.getcwd(),
                    takeover_factory=take_over,
                    initial_model=initial_model,
                    # Only a RESOLVED spec can be a deliberate override. Setup
                    # mode leaves `initial_model` None when the machine has no
                    # usable configuration yet, and claiming an override there
                    # would assert an intent with nothing to apply.
                    model_selection_override=(
                        initial_model is not None and bool(birth_args.hosting or birth_args.model)
                    ),
                )
                if degraded_reason:
                    viewer.degraded_reason = degraded_reason
                if paint_first_cwd:
                    # Tells the TUI that a live owner is being attached behind
                    # this paint, so it narrates and bounds that wait rather than
                    # leaving `starting…` as the only account of it (UX round 1,
                    # U1).
                    viewer.attach_behind = True
                return viewer

            async def session_factory():
                return await viewer_factory(getattr(args, "resume", None))

            # The provider controller gives the TUI the full provider/model/
            # credential/usage surface behind /model /provider /login /usage.
            # Its owning AuthStore lives for the TUI session only; the CLI
            # closes it after run_tui returns.
            from local_operator.providers.auth_store import AuthStore
            from local_operator.providers.controller import ProviderController

            tui_auth_store = AuthStore(config_dir=config_manager.config_dir)
            tui_controller = ProviderController(tui_auth_store, config_manager.config_dir)
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
                    warm_session_imports=False,
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
                    # ``None`` is meaningful, not absent: it means /new, which
                    # mints a fresh id rather than reopening one — the same
                    # distinction ``create_session``'s ``resume is not None``
                    # branch used to carry, now expressed in the viewer.
                    return await viewer_factory(resume_id)

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
    # ``sys.exit``, NOT the ``exit()`` builtin, so ``python -m local_operator.cli``
    # ends the same way the installed console scripts do. ``exit`` is
    # ``_sitebuiltins.Quitter``, which CLOSES ``sys.stdin`` before raising
    # SystemExit — and that close waits on the buffered reader's lock. The
    # interactive login reads a pasted value on a daemon thread that may still
    # be parked in that read (it is deliberately not joined; see
    # ``auth_cli.on_manual_code_input``), so the two together can deadlock a
    # cancel that has already printed its receipt: the user is told the login
    # was cancelled and the shell prompt never returns.
    #
    # ``sys.exit`` raises SystemExit without touching stdin, which is why the
    # shipped entry points have never had this shape. Site builtins are also
    # absent under ``python -S`` and in frozen builds, so this is the more
    # portable spelling regardless (QA round 1, Q1).
    sys.exit(main())
