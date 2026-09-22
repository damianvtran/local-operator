"""Backward-compatibility and additive-surface tests for the rewritten CLI.

Scope: the parser contract (every legacy flag/subcommand/dest/default
survives; new flags are additive), the help surfaces from the acceptance
criteria, the lazy-import discipline, and ``main()`` dispatch for the new
subcommands. Engine-dependent flows are NOT exercised here — they live in
``test_exec_mode.py`` with fake sessions.

The legacy engine modules these tests were once careful to avoid importing
no longer exist: the classify/plan/act operator and its executor were deleted
with the harness rewrite, and the CLI drives ``session_factory`` directly.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import socket
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from local_operator import cli
from local_operator.cli import (
    agents_create_command,
    agents_delete_command,
    agents_list_command,
    build_cli_parser,
    config_create_command,
    credential_delete_command,
    credential_update_command,
    main,
    serve_command,
)

#: The legacy --hosting choices, preserved byte-for-byte.
LEGACY_HOSTING_CHOICES = [
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
]


@pytest.fixture
def parser() -> argparse.ArgumentParser:
    return build_cli_parser()


def _fake_config_manager(*args, **kwargs) -> MagicMock:
    """ConfigManager stand-in: every get_config_value returns a falsy default."""
    return MagicMock(get_config_value=lambda *a: False)


@pytest.fixture
def tmp_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect Path.home() so no test touches the real ~/.local-operator."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return tmp_path


@pytest.fixture
def quiet_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neutralize environment repair (PATH surgery) in main() tests."""
    monkeypatch.setattr("local_operator.cli.setup_cross_platform_environment", lambda: None)


# --- Legacy parser contract -------------------------------------------------


def _fake_tui_module() -> types.ModuleType:
    """A stub for ``local_operator.tui`` that is still a PACKAGE.

    ``types.ModuleType("local_operator.tui")`` alone has no ``__path__``, so
    the moment anything imports a SUBMODULE through it Python raises
    ``ModuleNotFoundError: ... is not a package``. That is not hypothetical
    here: the CLI's viewer factory imports ``session.attached``, whose pricing
    helpers import ``local_operator.tui.costs`` (function-locally, so at call
    time rather than at import), and ``tui.costs`` is a SUBMODULE of the stub —
    so a bare stub turns a viewer-path import into an error return that these
    tests then read as "the session was never built".

    ``session.frontend_state`` used to be the importer that made this bite at
    module scope; it reads ``local_operator.model.costs`` now, which is why a
    bare stub no longer breaks a plain ``import local_operator.session.attached``
    (measured). The ``__path__`` stays: a submodule import through this stub is
    still reachable from the viewer path, and the failure mode is the one above.

    Giving the stub the real package's ``__path__`` keeps submodule imports
    resolving against the real tree while ``run_tui`` stays faked, which is
    the only part these tests mean to replace.
    """
    import local_operator.tui as real_tui

    fake = types.ModuleType("local_operator.tui")
    fake.__path__ = real_tui.__path__  # type: ignore[attr-defined]
    return fake


def test_root_defaults(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args([])
    assert args.subcommand is None
    assert args.debug is False
    assert args.agent_name is None
    assert args.train is False
    assert args.hosting is None
    assert args.model is None
    assert args.run_in is None
    # Additive root flags default off.
    assert args.yolo is False
    assert args.no_tui is False


@pytest.mark.parametrize("flag", ["--agent", "--agent-name"])
def test_agent_alias_dest(parser: argparse.ArgumentParser, flag: str) -> None:
    args = parser.parse_args([flag, "MyAgent"])
    assert args.agent_name == "MyAgent"


def test_global_flags_on_subcommands(parser: argparse.ArgumentParser) -> None:
    # parent_parser flags reach every subcommand (legacy behavior). Flags are
    # given AFTER the deepest subcommand: argparse re-applies the subparser's
    # parent defaults when the subparser takes over, so values set BEFORE it
    # are clobbered back to defaults — the exact legacy quirk, preserved.
    args = parser.parse_args(["config", "list", "--debug", "--agent", "A", "--train"])
    assert args.subcommand == "config"
    assert args.config_command == "list"
    assert args.debug is True
    assert args.agent_name == "A"
    assert args.train is True
    # Legacy quirk: a root-level --debug before the subcommand is clobbered.
    args = parser.parse_args(["--debug", "config", "list"])
    assert args.debug is False


@pytest.mark.parametrize("hosting", LEGACY_HOSTING_CHOICES)
def test_hosting_choices_preserved(parser: argparse.ArgumentParser, hosting: str) -> None:
    args = parser.parse_args(["--hosting", hosting])
    assert args.hosting == hosting


def test_hosting_rejects_unknown(parser: argparse.ArgumentParser) -> None:
    with pytest.raises(SystemExit):
        parser.parse_args(["--hosting", "nonsense"])


def test_model_and_run_in_dests(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args(["--model", "gpt-4o", "--run-in", "/tmp"])
    assert args.model == "gpt-4o"
    assert args.run_in == "/tmp"


def test_credential_subcommands(parser: argparse.ArgumentParser) -> None:
    for command in ("update", "delete"):
        args = parser.parse_args(["credential", command, "OPENAI_API_KEY"])
        assert args.subcommand == "credential"
        assert args.credential_command == command
        assert args.key == "OPENAI_API_KEY"


def test_config_subcommands(parser: argparse.ArgumentParser) -> None:
    assert parser.parse_args(["config", "create"]).config_command == "create"
    assert parser.parse_args(["config", "open"]).config_command == "open"
    assert parser.parse_args(["config", "list"]).config_command == "list"
    args = parser.parse_args(["config", "edit", "hosting", "openai"])
    assert args.config_command == "edit"
    assert args.key == "hosting"
    assert args.value == "openai"


def test_agents_list_defaults(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args(["agents", "list"])
    assert args.agents_command == "list"
    assert args.page == 1
    assert args.perpage == 10


def test_agents_create_positional(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args(["agents", "create", "NewAgent"])
    assert args.agents_command == "create"
    assert args.name == "NewAgent"


def test_teams_subcommands(parser: argparse.ArgumentParser) -> None:
    assert parser.parse_args(["teams", "list"]).teams_command == "list"
    created = parser.parse_args(
        [
            "teams",
            "create",
            "feature-release",
            "--manager",
            "manager",
            "--member",
            "coder",
            "--member",
            "reviewer:2",
        ]
    )
    assert created.teams_command == "create"
    assert created.name == "feature-release"
    assert created.manager == "manager"
    assert created.members == ["coder", "reviewer:2"]
    shown = parser.parse_args(["teams", "show", "feature-release"])
    assert shown.teams_command == "show"
    deleted = parser.parse_args(["teams", "delete", "--name", "feature-release"])
    assert deleted.teams_command == "delete"
    assert deleted.name == "feature-release"


@pytest.mark.parametrize(
    ("argv", "name", "agent_id"),
    [
        (["agents", "delete", "--name", "X"], "X", None),
        (["agents", "delete", "--id", "Y"], None, "Y"),
    ],
)
def test_agents_delete_dests(
    parser: argparse.ArgumentParser,
    argv: list[str],
    name: str | None,
    agent_id: str | None,
) -> None:
    args = parser.parse_args(argv)
    assert args.name == name
    assert args.agent_id == agent_id


def test_agents_delete_requires_exclusive_choice(
    parser: argparse.ArgumentParser,
) -> None:
    with pytest.raises(SystemExit):
        parser.parse_args(["agents", "delete"])
    with pytest.raises(SystemExit):
        parser.parse_args(["agents", "delete", "--name", "X", "--id", "Y"])


def test_agents_push_exclusive_required(parser: argparse.ArgumentParser) -> None:
    assert parser.parse_args(["agents", "push", "--name", "X"]).name == "X"
    assert parser.parse_args(["agents", "push", "--id", "Y"]).id == "Y"
    with pytest.raises(SystemExit):
        parser.parse_args(["agents", "push"])
    with pytest.raises(SystemExit):
        parser.parse_args(["agents", "push", "--name", "X", "--id", "Y"])


def test_agents_pull_id_required(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args(["agents", "pull", "--id", "abc"])
    assert args.id == "abc"
    with pytest.raises(SystemExit):
        parser.parse_args(["agents", "pull"])


def test_serve_defaults_preserved(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args(["serve"])
    assert args.subcommand == "serve"
    assert args.host == "127.0.0.1"
    assert args.port == 1111
    assert args.reload is False
    args = parser.parse_args(["serve", "--host", "localhost", "--port", "8000", "--reload"])
    assert (args.host, args.port, args.reload) == ("localhost", 8000, True)
    assert parser.parse_args(["serve", "--host", "0.0.0.0"]).host == "0.0.0.0"


def test_exec_legacy_shape(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args(["exec", "do the thing"])
    assert args.subcommand == "exec"
    assert args.command == "do the thing"
    # Additive flags default off/None.
    assert args.background is False
    assert args.json_mode is False
    assert args.agent_id is None
    # Parent flags reach exec (legacy --agent alias included).
    args = parser.parse_args(["exec", "cmd", "--agent", "A", "--train", "--debug"])
    assert args.agent_name == "A"
    assert args.train is True
    assert args.debug is True


# --- Additive surface --------------------------------------------------------


def test_root_yolo_and_no_tui(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args(["--yolo", "--no-tui"])
    assert args.yolo is True
    assert args.no_tui is True
    # --yolo before a subcommand propagates into its namespace (root flag).
    args = parser.parse_args(["--yolo", "exec", "cmd"])
    assert args.subcommand == "exec"
    assert args.yolo is True


def test_exec_additive_flags(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args(["exec", "long task", "--background", "--json", "--agent-id", "a1"])
    assert args.background is True
    assert args.json_mode is True
    assert args.agent_id == "a1"


def test_login_logout_login_status(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args(["login"])
    assert args.subcommand == "login"
    assert args.provider is None
    args = parser.parse_args(["login", "anthropic"])
    assert args.provider == "anthropic"

    args = parser.parse_args(["logout", "openai"])
    assert args.subcommand == "logout"
    assert args.provider == "openai"
    with pytest.raises(SystemExit):
        parser.parse_args(["logout"])

    args = parser.parse_args(["login-status"])
    assert args.subcommand == "login-status"


def test_mcp_subcommands(parser: argparse.ArgumentParser) -> None:
    assert parser.parse_args(["mcp", "list"]).mcp_command == "list"

    args = parser.parse_args(
        [
            "mcp",
            "add",
            "files",
            "--command",
            "npx",
            "--arg=-y",
            "--arg",
            "server",
            "--env",
            "KEY=VALUE",
            "--scope",
            "project",
        ]
    )
    assert args.mcp_command == "add"
    assert args.name == "files"
    assert args.command == "npx"
    assert args.server_args == ["-y", "server"]
    assert args.server_env == ["KEY=VALUE"]
    assert args.scope == "project"

    args = parser.parse_args(["mcp", "add", "web", "--url", "https://mcp.example.com"])
    assert args.url == "https://mcp.example.com"
    assert args.scope == "global"
    assert args.oauth is False

    oauth_args = parser.parse_args(
        ["mcp", "add", "linear", "--url", "https://mcp.linear.app/mcp", "--oauth"]
    )
    assert oauth_args.oauth is True

    login_args = parser.parse_args(["mcp", "login", "linear"])
    assert login_args.mcp_command == "login"
    logout_args = parser.parse_args(["mcp", "logout", "linear"])
    assert logout_args.mcp_command == "logout"
    assert logout_args.name == "linear"
    reauth_args = parser.parse_args(["mcp", "reauth", "linear"])
    assert reauth_args.mcp_command == "reauth"
    assert reauth_args.name == "linear"
    assert login_args.name == "linear"

    args = parser.parse_args(["mcp", "remove", "files", "--scope", "project"])
    assert args.mcp_command == "remove"
    assert args.name == "files"
    assert args.scope == "project"


def test_send_subcommand_parses(parser: argparse.ArgumentParser) -> None:
    # Default mailbox form: target + message positionals.
    args = parser.parse_args(["send", "peer-send design", "gates are green"])
    assert args.subcommand == "send"
    assert args.target == "peer-send design"
    assert args.message == "gates are green"
    assert args.steer is False
    assert args.wake is False
    assert args.pid is None
    assert args.session is None

    # Targeting flags and delivery modes.
    args = parser.parse_args(["send", "--pid", "42", "act now", "--wake"])
    assert args.pid == 42
    assert args.wake is True
    # --now and --steer are the same dest.
    assert parser.parse_args(["send", "t", "m", "--now"]).steer is True
    assert parser.parse_args(["send", "t", "m", "--steer"]).steer is True
    assert parser.parse_args(["send", "--session", "abc", "m"]).session == "abc"


def test_sessions_subcommand_parses(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args(["sessions"])
    assert args.subcommand == "sessions"
    assert args.json is False
    assert parser.parse_args(["sessions", "--json"]).json is True


@pytest.mark.parametrize("argv", [["--help"], ["exec", "--help"], ["login", "--help"]])
def test_help_surfaces_parse(argv: list[str]) -> None:
    """Acceptance: these help paths exit 0 via the module functions."""
    parser = build_cli_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(argv)
    assert excinfo.value.code == 0


# --- Lazy import discipline ---------------------------------------------------


def test_cli_import_pulls_no_engine_modules() -> None:
    """Acceptance: cli.py has no import-time dependency on textual /
    providers / session internals. Checked in a subprocess so the parent
    pytest process's module cache cannot mask a violation."""
    code = (
        "import sys, local_operator.cli\n"
        "banned = [m for m in sys.modules if m.startswith((\n"
        "  'local_operator.tui', 'local_operator.providers',\n"
        "  'local_operator.session.session', 'local_operator.session.transcript',\n"
        "  'local_operator.skills', 'local_operator.model.configure',\n"
        "  'local_operator.harness', 'textual'))]\n"
        "assert not banned, banned\n"
        "print('clean')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout


# --- Command handlers (legacy semantics survive) -------------------------------


def test_credential_update_command(tmp_home: Path) -> None:
    """The command binds a READ-ONLY manager, so it never recreates the file.

    ``CredentialManager(...)`` runs ``_ensure_config_exists`` and creates an empty
    ``credentials.env``; this command writes only a store row, so it binds through
    ``readonly`` (R5). The patch therefore targets the class and the assertion
    reads its ``readonly`` return value — asserting on a bare constructor call
    would pass against a regression to the creating path.
    """
    manager = MagicMock()
    with patch("local_operator.cli.CredentialManager") as manager_cls:
        manager_cls.readonly.return_value = manager
        args = argparse.Namespace(key="TEST_API_KEY")
        assert credential_update_command(args) == 0
        manager_cls.readonly.assert_called_once()
        manager_cls.assert_not_called()
    manager.prompt_for_credential.assert_called_once_with("TEST_API_KEY", reason="update requested")


def test_credential_delete_command(tmp_home: Path) -> None:
    with patch("local_operator.providers.registry.remove_provider_key") as remove:
        args = argparse.Namespace(key="TEST_API_KEY")
        assert credential_delete_command(args) == 0
    remove.assert_called_once_with("TEST_API_KEY")


def test_credential_update_ctrl_c_exits_130(tmp_home: Path, capsys) -> None:
    """Ctrl-C at the prompt is a cancel, not a crash: exit 130 (SIGINT
    convention), one quiet line, no stack-trace panel (item 4)."""
    manager = MagicMock()
    manager.prompt_for_credential.side_effect = KeyboardInterrupt
    with patch("local_operator.cli.CredentialManager") as manager_cls:
        manager_cls.readonly.return_value = manager
        args = argparse.Namespace(key="OPENAI_API_KEY")
        assert credential_update_command(args) == 130
    err = capsys.readouterr().err
    assert "Cancelled." in err
    assert "Traceback" not in err


def test_credential_update_empty_input_exits_1_plain(tmp_home: Path, capsys) -> None:
    """Empty/EOF input exits 1 with one plain line, ANSI stripped (item 4)."""
    manager = MagicMock()
    manager.prompt_for_credential.side_effect = ValueError(
        "\033[1;31mOPENAI_API_KEY is required for this step.\033[0m"
    )
    with patch("local_operator.cli.CredentialManager") as manager_cls:
        manager_cls.readonly.return_value = manager
        args = argparse.Namespace(key="OPENAI_API_KEY")
        assert credential_update_command(args) == 1
    err = capsys.readouterr().err
    assert "is required for this step" in err
    # The nested escape from the exception message is stripped.
    assert "\033[1;31m" not in err


def test_credential_update_unknown_key_warns(tmp_home: Path, capsys) -> None:
    """A key the registry does not know gets a difflib suggestion but is still
    stored (custom providers are legitimate) (item 8)."""
    manager = MagicMock()
    with patch("local_operator.cli.CredentialManager") as manager_cls:
        manager_cls.readonly.return_value = manager
        args = argparse.Namespace(key="OPENAI_API_KY")  # typo
        assert credential_update_command(args) == 0
    err = capsys.readouterr().err
    assert "not a known provider key" in err
    assert "OPENAI_API_KEY" in err  # the suggestion


def test_config_edit_rejects_unknown_key(tmp_home: Path, capsys) -> None:
    """`config edit` validates against the defaults and rejects a typo with a
    suggestion + exit 1, instead of silently writing junk (item 7)."""
    args = argparse.Namespace(key="hostng", value="openai")
    assert cli.config_edit_command(args) == 1
    err = capsys.readouterr().err
    assert "unknown configuration key" in err
    assert "hosting" in err  # the suggestion


def test_config_edit_accepts_known_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Against a REAL config, not a mock: the claim is that the value lands in
    the file, and a mock asserting which internal writer was called proves only
    that the writer did not change."""
    import yaml

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    args = argparse.Namespace(key="hosting", value="openai")
    assert cli.config_edit_command(args) == 0
    assert yaml.safe_load((tmp_path / "config.yml").read_text())["values"]["hosting"] == "openai"


def test_config_edit_accepts_a_dotted_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The user-facing lie this fixed: the TUI instructs users to run
    `config edit display.terminal_title false`, and the validator rejected
    every dotted key, so that command could only ever exit 1.

    Both shapes are covered because they are stored differently: `display.*` is
    a literal dotted TOP-LEVEL key, while `retry.maxRetries` is genuinely
    nested and must not lose its siblings.
    """
    import yaml

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    assert (
        cli.config_edit_command(argparse.Namespace(key="display.terminal_title", value="false"))
        == 0
    )
    assert cli.config_edit_command(argparse.Namespace(key="retry.maxRetries", value="4")) == 0

    values = yaml.safe_load((tmp_path / "config.yml").read_text())["values"]
    assert values["display.terminal_title"] is False
    assert "display" not in values, "wrote a nested mapping nothing reads"
    assert values["retry"]["maxRetries"] == 4
    assert values["retry"]["fallbackChains"] == {}, "a sibling was destroyed"


def test_config_edit_rejects_an_out_of_range_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Consumers clamp silently, so a stored 9999 the tool reads as 100 is the
    config and the behaviour disagreeing with nothing admitting it."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    assert cli.config_edit_command(argparse.Namespace(key="retry.maxRetries", value="9999")) == 1


def test_config_create_command(tmp_home: Path) -> None:
    manager = MagicMock()
    with patch("local_operator.cli.ConfigManager", return_value=manager):
        assert config_create_command() == 0
    manager._write_config.assert_called_once()


def test_config_create_command_reports_the_path_it_wrote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The message must name the file that was actually created.

    The write target is ``config_dir()``, which honours
    LOCAL_OPERATOR_CONFIG_DIR; a hardcoded ``~/.local-operator`` in the message
    sends the user to a file that does not exist and contradicts
    ``config open`` twelve lines below, which prints the resolved path. Runs
    against the real ConfigManager so the asserted path is the one on disk.
    """
    override = tmp_path / "elsewhere"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(override))
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "unused-home")

    assert config_create_command() == 0

    created = override / "config.yml"
    assert created.exists()
    assert str(created) in capsys.readouterr().out


def test_serve_command_preserves_the_address_it_was_given() -> None:
    """uvicorn is imported lazily INSIDE serve_command (so `local-operator`
    starts without the server extra), so the patch target is the uvicorn
    module itself, not a `local_operator.cli.uvicorn` attribute that no
    longer exists.

    The call SHAPE changed when the daemon gained its rendezvous record: the
    non-reload path binds the listener here and hands uvicorn the OPEN SOCKET
    (``Server.run(sockets=[sock])``), because only the process that bound it
    knows the port the kernel granted, and the record has to carry the port
    actually bound. The assertions therefore moved to the two seams that now
    carry the address — ``uvicorn.Config`` (host/port, and the ASGI app OBJECT
    rather than an import string, since there is no reloader to re-import) and
    the socket handed to ``Server.run``.

    The binder is spied rather than exercised on the requested port on
    purpose: what this test pins is that the REQUESTED host/port reach the
    binder unchanged and that the RESOLVED port reaches uvicorn and the
    announcement, and binding an ephemeral port keeps it independent of
    whether 8000 happens to be free on the machine.
    """
    from local_operator.server import registry as serve_registry
    from local_operator.server.app import app as asgi_app

    requested: list[tuple[str, int]] = []
    # Captured BEFORE the patch: a spy that looks the name up on the module at
    # call time would find itself and recurse.
    real_bind = cli._bind_serve_socket

    def spy_bind(host: str, port: int) -> socket.socket:
        requested.append((host, port))
        return real_bind(host, 0)

    with (
        patch.object(cli, "_bind_serve_socket", spy_bind),
        patch("uvicorn.Config") as mock_config,
        patch("uvicorn.Server.run") as mock_run,
    ):
        assert serve_command("localhost", 8000, False) == 0

    assert requested == [("localhost", 8000)]
    args, kwargs = mock_config.call_args
    assert args[0] is asgi_app, "an explicit socket means no reloader, so the object is served"
    sockets = mock_run.call_args.kwargs["sockets"]
    try:
        resolved_port = sockets[0].getsockname()[1]
        assert resolved_port > 0, "the socket handed over is bound, not a placeholder"
        assert kwargs == {
            "host": "localhost",
            "port": resolved_port,
        }, "uvicorn is told the bound port"
        # And the same number is what the app publishes its record with — read
        # off the app OBJECT, which is the channel this path announces on (no
        # environment: nothing this daemon spawns may inherit its address).
        assert serve_registry.advertised_address(asgi_app) == ("localhost", resolved_port)
    finally:
        for listener in sockets:
            listener.close()
        # No lifespan runs in this test to consume the announcement, and `app`
        # is a module-level singleton: clear it so a later test in the worker
        # does not boot an app still announced on a dead ephemeral port.
        state = asgi_app.state
        if serve_registry.ANNOUNCED_STATE_ATTR in state:
            del state[serve_registry.ANNOUNCED_STATE_ATTR]


def test_serve_command_still_routes_reload_through_uvicorn_run() -> None:
    """``--reload`` keeps ``uvicorn.run`` and the import string.

    uvicorn only reloads an app given as an import string, and the reloader
    re-imports it in a CHILD process, so the bound socket cannot be handed over
    on that path — the port is resolved (for the record and the banner) and
    uvicorn binds it in the child.
    """
    with patch("uvicorn.run") as mock_run:
        assert serve_command("localhost", 8000, True) == 0
    mock_run.assert_called_once_with(
        "local_operator.server.app:app",
        host="localhost",
        port=8000,
        reload=True,
        reload_excludes=[".venv"],
    )


def test_agents_list_command_empty() -> None:
    registry = MagicMock()
    registry.list_agents.return_value = []
    assert agents_list_command(argparse.Namespace(page=1, perpage=10), registry) == 0


def test_agents_list_reveals_routing_metadata_only_when_called(capsys) -> None:
    registry = MagicMock()
    agent = MagicMock()
    agent.name = "Database specialist"
    agent.id = "db-1"
    agent.created_date = "now"
    agent.version = "1.0.0"
    agent.hosting = ""
    agent.model = ""
    agent.description = "Tunes PostgreSQL queries"
    agent.tags = ["postgresql"]
    agent.categories = ["performance"]
    registry.list_agents.return_value = [agent]

    assert agents_list_command(argparse.Namespace(page=1, perpage=10), registry) == 0

    output = capsys.readouterr().out
    assert "Description: Tunes PostgreSQL queries" in output
    assert "Tags: postgresql" in output
    assert "Categories: performance" in output


def test_agents_create_command_calls_registry() -> None:
    registry = MagicMock()
    created = MagicMock(name="AgentX", id="id-1", created_date="now", version="1.0.0")
    registry.create_agent.return_value = created
    assert agents_create_command("AgentX", registry) == 0
    registry.create_agent.assert_called_once()


def test_agents_delete_command_by_name() -> None:
    registry = MagicMock()
    agent = MagicMock(name="TestAgent", id="test-id")
    agent.name = "TestAgent"
    registry.list_agents.return_value = [agent]
    args = argparse.Namespace(name="TestAgent", agent_id=None)
    assert agents_delete_command(args, registry, Path(".")) == 0
    registry.delete_agent.assert_called_once_with("test-id")


def test_agents_delete_command_not_found() -> None:
    registry = MagicMock()
    registry.list_agents.return_value = []
    args = argparse.Namespace(name="Ghost", agent_id=None)
    assert agents_delete_command(args, registry, Path(".")) == 1
    registry.delete_agent.assert_not_called()


# --- main() dispatch ------------------------------------------------------------


class _FakeHub:
    """The two hub calls ``agents push`` makes, and what the hub answered.

    ``get_agent`` returning None IS the hub's 404 — the answer a LOCAL id gets,
    because nothing aligns a local uuid with a hub listing id — so the create
    branch is the ordinary outcome for ``--id``, not a corner.
    """

    def __init__(self) -> None:
        self.existing: set[str] = set()
        self.probe_error: Exception | None = None
        self.overwrote: list[str] = []
        self.created: list[str] = []

    def get_agent(self, agent_id: str) -> dict[str, str] | None:
        if self.probe_error is not None:
            raise self.probe_error
        return {"id": agent_id} if agent_id in self.existing else None

    def overwrite_agent_in_marketplace(self, agent_id: str, zip_path: Path) -> None:
        self.overwrote.append(agent_id)

    def upload_agent_to_marketplace(self, zip_path: Path) -> str:
        listing = f"hub-listing-{len(self.created) + 1}"
        self.created.append(listing)
        return listing


def _push_fixture(tmp_home: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[_FakeHub, Any]:
    """A real local registry row and a hub that records what it was asked to do."""
    from pydantic import SecretStr

    from local_operator.agents import AgentEditFields, AgentRegistry
    from local_operator.providers import radient_credentials

    hub = _FakeHub()
    monkeypatch.setattr("local_operator.clients.radient.RadientClient", lambda **kwargs: hub)
    monkeypatch.setattr(
        radient_credentials,
        "resolve_radient_credential_sync",
        lambda *args, **kwargs: SecretStr("k"),
    )
    registry = AgentRegistry(Path.home() / ".local-operator")
    agent = registry.create_agent(AgentEditFields.model_validate({"name": "PushProbe"}))
    return hub, agent


def test_agents_push_by_id_reports_the_create_the_hub_performed(
    tmp_home: Path, quiet_env: None, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """The printed outcome follows the hub, not the flag that was typed.

    ``--id`` is a LOCAL id and nothing makes it equal a hub listing id, so this
    run creates a listing — and the branch used to test ``agent_id_to_overwrite``
    instead of what the upload returned, printing "as overwrite" for a listing it
    had just created without ever naming it. The user could not even find the
    duplicate to delist.
    """
    hub, agent = _push_fixture(tmp_home, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["program", "agents", "push", "--id", agent.id])

    assert main() == 0
    out = capsys.readouterr().out
    assert hub.created == ["hub-listing-1"]
    assert "New agent ID: hub-listing-1" in out
    assert "as overwrite" not in out


def test_agents_push_by_id_reports_an_overwrite_only_when_one_happened(
    tmp_home: Path, quiet_env: None, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """The other direction of the same branch, with the hub really holding the id."""
    hub, agent = _push_fixture(tmp_home, monkeypatch)
    hub.existing.add(agent.id)
    monkeypatch.setattr(sys, "argv", ["program", "agents", "push", "--id", agent.id])

    assert main() == 0
    out = capsys.readouterr().out
    assert hub.overwrote == [agent.id]
    assert f"as overwrite to Radient (ID: {agent.id})" in out
    assert "New agent ID" not in out


def test_agents_push_fails_rather_than_publishing_a_duplicate(
    tmp_home: Path, quiet_env: None, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """A probe that could not answer must not resolve into a second listing.

    "I could not tell whether this exists" is not "it does not exist": the run
    stops with the real error instead of creating the duplicate the old code
    created and then reported as an overwrite.
    """
    hub, agent = _push_fixture(tmp_home, monkeypatch)
    hub.probe_error = RuntimeError('HTTP 500, Response Body: {"error":"boom"}')
    monkeypatch.setattr(sys, "argv", ["program", "agents", "push", "--id", agent.id])

    assert main() == 1
    out = capsys.readouterr().out
    assert "Error pushing agent to Radient" in out
    assert "HTTP 500" in out
    assert hub.created == []


def test_main_exec_dispatch(
    tmp_home: Path, quiet_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """exec routes to exec_mode.run_exec with the parsed ExecArgs, and its
    exit code passes through (README contract). ``--hosting test`` keeps the
    CL-06 startup preflight green (the test provider needs no key)."""
    captured: dict[str, Any] = {}

    def fake_run_exec(command: str, exec_args) -> int:
        captured["command"] = command
        captured["args"] = exec_args
        return 7

    monkeypatch.setattr("local_operator.exec_mode.run_exec", fake_run_exec)
    # The CL-06 foreground preflight shares the worker's resolution path;
    # stub it green so this test stays focused on the dispatch wiring.
    monkeypatch.setattr(
        "local_operator.exec_mode.resolve_hosting_model_dry", lambda args: ("test", "m")
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "program",
            "--yolo",
            "--hosting",
            "test",
            "--model",
            "m",
            "exec",
            "do the thing",
            "--json",
            "--agent-id",
            "a1",
        ],
    )
    assert main() == 7
    assert captured["command"] == "do the thing"
    exec_args = captured["args"]
    assert exec_args.json_mode is True
    assert exec_args.agent_id == "a1"
    assert exec_args.yolo is True
    assert exec_args.background is False
    assert exec_args.agent_name is None
    assert exec_args.train is False


@pytest.mark.asyncio
async def test_mcp_login_connects_and_disconnects_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    # ``url`` is part of the shape, not decoration: an OAuth server with no URL
    # is a contradiction (a stdio transport cannot carry a bearer token), and
    # the real ``MCPHttpServerConfig`` always has one. The gate now asks
    # ``server_rejects_oauth``, which reads the transport as well as the auth
    # block, so a double omitting it describes a server that cannot exist.
    config = types.SimpleNamespace(
        auth=types.SimpleNamespace(type="oauth"), url="https://linear.example/mcp"
    )
    monkeypatch.setattr(
        "local_operator.mcp.config.load_all_mcp_configs",
        lambda _cwd: ({"linear": config}, {"linear": tmp_path / "mcp.json"}),
    )
    instances: list[Any] = []

    class FakeManager:
        def __init__(self, cwd: Path) -> None:
            self.cwd = cwd
            self.disconnected = False
            instances.append(self)

        async def connect_configured_server(
            self, name: str, *, timeout_ms: float | None = None
        ) -> Any:
            assert name == "linear"
            assert timeout_ms == 600_000
            return types.SimpleNamespace(tools=[object(), object()])

        async def disconnect_all(self) -> None:
            self.disconnected = True

    monkeypatch.setattr("local_operator.mcp.manager.McpManager", FakeManager)

    assert await cli._mcp_login_server("linear", tmp_path) == 0
    assert instances[0].cwd == tmp_path
    assert instances[0].disconnected is True
    assert "discovered 2 tools" in capsys.readouterr().out


def test_mcp_logout_command_reports_removal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """The CLI's logout is the module helper plus phrasing: success names the
    server whose credential is gone, failure carries the helper's reason."""
    monkeypatch.setattr("local_operator.mcp.auth.mcp_logout_server", lambda name, cwd: None)
    assert cli.mcp_command(argparse.Namespace(mcp_command="logout", name="linear")) == 0
    assert "'linear'" in capsys.readouterr().out

    monkeypatch.setattr(
        "local_operator.mcp.auth.mcp_logout_server",
        lambda name, cwd: "no stored credential for MCP server 'linear'",
    )
    assert cli.mcp_command(argparse.Namespace(mcp_command="logout", name="linear")) == 1
    assert "nothing" not in capsys.readouterr().out  # reason goes to stderr


def test_mcp_logout_reports_a_failed_delete_as_one_actionable_line(
    tmp_home: Path, quiet_env: None, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """A locked store must produce ONE error line, not a stack-trace banner.

    ``McpTokenStorage.clear`` RAISES when a row is found and its delete fails,
    so that a caller cannot read it as the benign "nothing was stored". The
    logout branch is the one caller with no reauth gate in front of it: left
    bare, the exception reached ``main``'s generic handler, which printed
    "Error: ...", a 30-line traceback, and "Please review and correct the
    error to continue" for a sibling process holding a sqlite write lock.

    That framing is wrong twice — it presents a retryable, user-fixable
    condition as a defect in local-operator, and it buries the sentence that
    matters (the credential is still there) under the trace.

    Driven through the real ``main()`` over a real ``AuthStore``, because the
    defect lived in the seam BETWEEN the helper and the entry point: every
    layer below was already correct and a test calling ``mcp_command``
    directly still passes the generic handler by. The store is a real one on
    ``tmp_home`` whose delete raises the real ``OperationalError`` a locked
    db raises, rather than a mock raising the app's own exception type.
    """
    import json
    import sqlite3

    from local_operator.mcp import auth as auth_mod
    from local_operator.providers.auth_store import AuthStore

    class LockedStore(AuthStore):
        """Real store, real rows — but its delete hits a held write lock."""

        deletes_attempted = 0

        def delete_credential(self, credential_id: int) -> None:
            type(self).deletes_attempted += 1
            raise sqlite3.OperationalError("database is locked")

    url = "https://locktest.example/mcp"
    (tmp_home / ".local-operator").mkdir(parents=True, exist_ok=True)
    (tmp_home / ".local-operator" / "mcp.json").write_text(
        json.dumps({"mcpServers": {"locktest": {"url": url}}})
    )
    store = LockedStore(tmp_home / ".local-operator" / "auth.db")
    monkeypatch.setattr(auth_mod, "_resolve_store", lambda given: given or store)
    # A stored grant is what makes the server OAuth-capable AND what the
    # delete then fails on — both preconditions come from this one row.
    auth_mod.McpTokenStorage(url, store)._write({"tokens": {"access_token": "SURVIVOR"}})
    auth_mod.OAUTH_CHALLENGES.clear()

    # The observation path must be LIVE before its silence means anything: a
    # missing row, or a config the loader never saw, would make the command
    # fail for an unrelated reason and this test pass without ever reaching
    # the raise. Assert the setup landed instead of assuming it.
    assert auth_mod.server_has_stored_grant(url, store) is True

    monkeypatch.setattr(sys, "argv", ["program", "mcp", "logout", "locktest"])
    assert main() == 1

    err = capsys.readouterr().err
    # The delete was really attempted — this is the failed-delete path and not
    # some earlier refusal (unknown name, not-OAuth) wearing the same rc.
    assert LockedStore.deletes_attempted == 1
    # The credential really did survive, which is what the message must say.
    assert auth_mod.server_has_stored_grant(url, store) is True
    assert "still in place" in err
    assert "Retry once" in err  # actionable: names what the user can do
    # NOT the old message, which claimed nothing was stored while the row
    # survived — the conflation this PR removed.
    assert "nothing to log out of" not in err
    # The regression itself: no traceback banner, and one line of output.
    assert "Stack Trace" not in err
    assert "Please review and correct the error" not in err
    assert len([line for line in err.splitlines() if line.strip()]) == 1


@pytest.mark.asyncio
async def test_mcp_reauth_removes_then_logs_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reauth must delete BEFORE the grant starts — a login over a surviving
    row would reuse the stored registration and never show the consent
    screen, which is the entire reason reauth exists."""
    calls: list[str] = []
    # ``store`` is part of the real signature and the reauth gate passes it, so
    # the stub accepts it too — a narrower stub raises TypeError, which the
    # gate would report as a failed removal and refuse for the wrong reason.
    monkeypatch.setattr(
        "local_operator.mcp.auth.mcp_logout_server",
        lambda name, cwd, store=None: calls.append("logout") or None,
    )

    async def fake_login(name: str, cwd: Path) -> int:
        calls.append("login")
        return 0

    monkeypatch.setattr(cli, "_mcp_login_server", fake_login)
    assert await cli._mcp_reauth_server("linear", tmp_path) == 0
    assert calls == ["logout", "login"]


@pytest.mark.asyncio
async def test_mcp_reauth_stops_when_removal_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setattr(
        "local_operator.mcp.auth.mcp_logout_server",
        lambda name, cwd, store=None: "MCP server 'linar' is not configured",
    )

    async def fake_login(name: str, cwd: Path) -> int:
        raise AssertionError("login must not start when removal failed")

    monkeypatch.setattr(cli, "_mcp_login_server", fake_login)
    assert await cli._mcp_reauth_server("linar", tmp_path) == 1
    assert "not configured" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_mcp_login_accepts_a_url_only_oauth_capable_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """A Codex-imported server has no auth block, and the CLI must not refuse it.

    The static ``cfg.auth.type == 'oauth'`` gate this replaced rejected exactly
    the servers ``/mcp login`` in the TUI handles fine — one question with two
    gates that disagreed. Capability is decided by the same live probe the rest
    of the code uses.
    """
    config = types.SimpleNamespace(auth=None, url="https://codex.example/mcp")
    monkeypatch.setattr(
        "local_operator.mcp.config.load_all_mcp_configs",
        lambda _cwd: ({"codex": config}, {"codex": tmp_path / "config.toml"}),
    )

    async def _capable(cfg, store=None):
        return True

    monkeypatch.setattr("local_operator.mcp.auth.probe_oauth_capability", _capable)

    class FakeManager:
        def __init__(self, cwd: Path) -> None:
            self.disconnected = False

        async def connect_configured_server(self, name, *, timeout_ms=None):
            return types.SimpleNamespace(tools=[object()])

        async def disconnect_all(self) -> None:
            self.disconnected = True

    monkeypatch.setattr("local_operator.mcp.manager.McpManager", FakeManager)

    assert await cli._mcp_login_server("codex", tmp_path) == 0
    assert "discovered 1 tools" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_mcp_login_still_refuses_a_server_that_cannot_take_oauth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """F3: an apikey or stdio server stays a HARD refusal, with no probe.

    An ``auth.type: apikey`` entry is the user stating how the server
    authenticates; starting an OAuth flow there answers a question they already
    answered. The refusal must also stay free — no network round trip to learn
    something the config already settles.
    """

    async def _must_not_probe(cfg, store=None):
        raise AssertionError("a statically-ineligible server must not be probed")

    monkeypatch.setattr("local_operator.mcp.auth.probe_oauth_capability", _must_not_probe)

    apikey = types.SimpleNamespace(
        auth=types.SimpleNamespace(type="apikey"), url="https://api.example/mcp"
    )
    stdio = types.SimpleNamespace(auth=None, url=None)
    monkeypatch.setattr(
        "local_operator.mcp.config.load_all_mcp_configs",
        lambda _cwd: ({"apikey": apikey, "stdio": stdio}, {}),
    )

    for name in ("apikey", "stdio"):
        assert await cli._mcp_login_server(name, tmp_path) == 1
        assert "not OAuth-enabled" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_mcp_reauth_cli_on_a_url_only_server_reconnects_authenticated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI reauth path shares the TUI's evidence-loss bug and its fix.

    ``_mcp_reauth_server`` is its own function — it does not go through
    ``run_grant`` — so it deletes the row via the same helper and then logs in.
    With the stored grant as a url-only config's ONLY capability evidence, the
    delete used to make the subsequent connect unauthenticated.
    """
    from local_operator.mcp import auth as auth_mod
    from local_operator.mcp.config import MCPHttpServerConfig
    from tests.unit.mcp.test_auth import FakeAuthStore

    url = "https://codex.example/mcp"
    cfg = MCPHttpServerConfig(url=url)
    auth_mod.OAUTH_CHALLENGES.clear()
    store = FakeAuthStore()  # never the developer's real auth.db
    auth_mod.McpTokenStorage(url, store)._write({"tokens": {"access_token": "a"}})

    monkeypatch.setattr(
        "local_operator.mcp.config.load_all_mcp_configs",
        lambda _cwd: ({"codex": cfg}, {}),
    )
    real_logout = auth_mod.mcp_logout_server
    monkeypatch.setattr(
        "local_operator.mcp.auth.mcp_logout_server",
        lambda name, cwd, _store=None: real_logout(name, cwd, store),
    )
    # The reauth gate re-reads the store to confirm nothing survived the
    # delete, and would resolve the real shared ``auth.db`` to do it.
    monkeypatch.setattr(auth_mod, "_resolve_store", lambda given: given or store)

    seen: dict[str, bool] = {}

    class FakeManager:
        def __init__(self, cwd: Path) -> None:
            pass

        async def connect_configured_server(self, name, *, timeout_ms=None):
            # The question ``_build_oauth_auth`` asks before wiring the provider.
            seen["capable"] = auth_mod.server_is_oauth_capable(cfg, store)
            if not seen["capable"]:
                raise RuntimeError(f"MCP server at {url} refused the connection (401)")
            return types.SimpleNamespace(tools=[object()])

        async def disconnect_all(self) -> None:
            pass

    monkeypatch.setattr("local_operator.mcp.manager.McpManager", FakeManager)

    assert await cli._mcp_reauth_server("codex", tmp_path) == 0
    assert auth_mod.server_has_stored_grant(url, store) is False  # really deleted
    assert seen["capable"] is True  # ...but still known to take OAuth


@pytest.mark.asyncio
async def test_mcp_reauth_with_nothing_stored_proceeds_into_the_login(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """Reauth on a url-only server holding no grant must log in, not dead-end.

    The removal fails here because ``mcp_logout_server`` gates on the STRICT
    ``server_is_oauth_capable``, and a Codex-imported config with a cold ledger
    and no stored row satisfies none of its three evidence kinds. That used to
    exit 1 saying the server "does not use OAuth login" — false for a server
    the login probe accepts, and a dead end for the user this path is for.
    Reauth's contract is "end up authenticated", so a no-op delete is a
    satisfied precondition.
    """
    from local_operator.mcp import auth as auth_mod
    from local_operator.mcp.config import MCPHttpServerConfig

    auth_mod.OAUTH_CHALLENGES.clear()
    cfg = MCPHttpServerConfig(url="https://codex.example/mcp")
    monkeypatch.setattr(
        "local_operator.mcp.config.load_all_mcp_configs",
        lambda _cwd: ({"codex": cfg}, {}),
    )

    logged_in: list[str] = []

    async def fake_login(name: str, cwd: Path) -> int:
        logged_in.append(name)
        return 0

    monkeypatch.setattr(cli, "_mcp_login_server", fake_login)

    assert await cli._mcp_reauth_server("codex", tmp_path) == 0
    assert logged_in == ["codex"]  # fell through instead of erroring
    assert capsys.readouterr().err == ""


@pytest.mark.asyncio
async def test_mcp_reauth_refuses_when_the_credential_delete_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """A FAILED delete must never look like "nothing was stored".

    The hazard the fall-through above opens if the two are conflated: both
    outcomes made ``mcp_logout_server`` return an error string, so falling
    through on "an error" also fell through when the delete had been ATTEMPTED
    and had RAISED — leaving the row on disk. The SDK then reuses the surviving
    token and ``client_info``, no consent screen comes back up, and reauth
    exits 0 for the account switch that silently did not happen.

    Reachable rather than theoretical: a sibling session holding an EXCLUSIVE
    sqlite transaction makes ``delete_credential`` raise ``database is locked``,
    which is what ``RefusingStore`` reproduces here.

    This test is the one that can SEE the distinction. Asserting only "reauth
    errors when nothing is stored" cannot: it is satisfied by a gate that
    refuses everything. So it drives the real ``_mcp_reauth_server`` against a
    real ``McpTokenStorage`` over a store whose delete raises, and asserts all
    three of rc=1, the login NOT reached, and the row still present.
    """
    from local_operator.mcp import auth as auth_mod
    from local_operator.mcp.config import MCPHttpServerConfig
    from tests.unit.mcp.test_auth import FakeAuthStore

    class RefusingStore(FakeAuthStore):
        """A store whose row cannot be deleted — `OperationalError` in the field."""

        def delete_credential(self, credential_id: int) -> None:
            raise RuntimeError("database is locked")

    url = "https://codex.example/mcp"
    store = RefusingStore()
    auth_mod.OAUTH_CHALLENGES.clear()
    auth_mod.McpTokenStorage(url, store)._write({"tokens": {"access_token": "OLD-TOKEN"}})
    monkeypatch.setattr(
        "local_operator.mcp.config.load_all_mcp_configs",
        lambda _cwd: ({"codex": MCPHttpServerConfig(url=url)}, {}),
    )
    # The CLI resolves the real shared ``auth.db``; pin it to the in-memory
    # store so the developer's own credentials are never touched.
    monkeypatch.setattr(auth_mod, "_resolve_store", lambda given: given or store)

    async def fake_login(name: str, cwd: Path) -> int:
        raise AssertionError("a surviving credential must not be reused by a fresh grant")

    monkeypatch.setattr(cli, "_mcp_login_server", fake_login)

    assert await cli._mcp_reauth_server("codex", tmp_path) == 1
    assert "still in place" in capsys.readouterr().err
    # The row really did survive — which is precisely why the login was refused.
    assert auth_mod.server_has_stored_grant(url, store) is True


@pytest.mark.asyncio
async def test_mcp_reauth_still_refuses_a_server_that_cannot_take_oauth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """Falling through on a failed removal must not weaken the static refusals.

    A stdio server and a declared ``auth.type: apikey`` server both fail the
    removal for the same "not OAuth" reason as the url-only case above, but
    they are statically ineligible — the F3 protection. They must still exit 1
    without reaching a login, or a typo could open a browser tab.
    """
    from local_operator.mcp.config import (
        MCPAuthConfig,
        MCPHttpServerConfig,
        MCPStdioServerConfig,
    )

    configs = {
        "apikey": MCPHttpServerConfig(
            url="https://api.example/mcp", auth=MCPAuthConfig(type="apikey")
        ),
        "stdio": MCPStdioServerConfig(command="run-me"),
    }
    monkeypatch.setattr(
        "local_operator.mcp.config.load_all_mcp_configs",
        lambda _cwd: (configs, {}),
    )

    async def fake_login(name: str, cwd: Path) -> int:
        raise AssertionError("an ineligible or unknown server must not reach the login")

    monkeypatch.setattr(cli, "_mcp_login_server", fake_login)

    for name in ("apikey", "stdio"):
        assert await cli._mcp_reauth_server(name, tmp_path) == 1
        assert "does not use OAuth login" in capsys.readouterr().err
    # An unknown name is still a typo the user wants told about, not a login.
    assert await cli._mcp_reauth_server("nosuch", tmp_path) == 1
    assert "not configured" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("argv", "handler"),
    [
        (["login"], "login_command"),
        (["logout", "openai"], "logout_command"),
        (["login-status"], "login_status_command"),
        (["mcp", "list"], "mcp_command"),
        (["qwencloud-ticket", "status"], "qwencloud_ticket_command"),
    ],
)
def test_main_management_command_dispatch(
    argv: list[str],
    handler: str,
    tmp_home: Path,
    quiet_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = MagicMock(return_value=23)
    monkeypatch.setattr(cli, handler, called)
    monkeypatch.setattr(sys, "argv", ["program", *argv])

    assert main() == 23
    called.assert_called_once()


def test_main_exception_banner(tmp_home: Path, quiet_env: None, capsys) -> None:
    """Red-banner handling survives: any exception -> message + exit 1."""
    with patch("local_operator.cli.ConfigManager", side_effect=Exception("Test error")):
        with patch("sys.argv", ["program"]):
            assert main() == 1
    # STDERR: main() wraps the exec dispatch, so its error presenter must not
    # write to the `exec --json` data channel. Asserting the stream is the
    # point of the test now, not incidental.
    err = capsys.readouterr().err
    assert "Error: Test error" in err
    assert "Stack Trace" in err


def test_viewer_birth_config_and_model_resolution_run_off_the_event_loop(
    tmp_home: Path,
    quiet_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guard the real nested viewer factory, not a free-standing helper."""
    import threading

    from local_operator import session_factory as factory_module

    seen: dict[str, Any] = {"active": False, "config_threads": [], "model_threads": []}
    original_resolve = factory_module.resolve_hosting_model

    def config_manager(*args, **kwargs):
        if seen["active"]:
            seen["config_threads"].append(threading.get_ident())
        return _fake_config_manager(*args, **kwargs)

    def resolve_model(*args, **kwargs):
        if seen["active"]:
            seen["model_threads"].append(threading.get_ident())
        return original_resolve(*args, **kwargs)

    async def fake_cold(*args, **kwargs):
        return object()

    async def run_tui(session_factory, session_registry=None, **kwargs):
        seen["loop_thread"] = threading.get_ident()
        seen["active"] = True
        try:
            await session_factory()
        finally:
            seen["active"] = False
        return 0

    fake_tui = _fake_tui_module()
    setattr(fake_tui, "run_tui", run_tui)
    monkeypatch.setitem(sys.modules, "local_operator.tui", fake_tui)
    monkeypatch.setattr(factory_module, "resolve_hosting_model", resolve_model)
    monkeypatch.setattr("local_operator.cli.ConfigManager", config_manager)
    monkeypatch.setattr("local_operator.cli.CredentialManager", _bare_credential_manager)
    monkeypatch.setattr("local_operator.agents.AgentRegistry", MagicMock())
    monkeypatch.setattr(
        "local_operator.session.attached.AttachedSession.cold", staticmethod(fake_cold)
    )
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    with patch("sys.argv", ["program", "--hosting", "test", "--model", "captured-a"]):
        assert main() == 0
    for key in ("config_threads", "model_threads"):
        assert seen[key], key
        assert all(thread != seen["loop_thread"] for thread in seen[key]), key


def test_setup_mode_with_a_model_flag_claims_no_override_it_cannot_apply(
    tmp_home: Path,
    quiet_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`lop --model <id>` on an unconfigured machine (review round 2, F3).

    The birth resolution raises, so setup mode reaches the viewer with no
    resolved spec. Claiming an override there left a pending intent nothing
    could satisfy, which refused every later call including the `/model` the
    refusal invited. Asserted on what `main()` really passes to `cold`.
    """
    seen: dict[str, Any] = {}

    async def fake_cold(*args, **kwargs):
        seen["initial_model"] = kwargs.get("initial_model")
        seen["override"] = kwargs.get("model_selection_override")
        return object()

    async def run_tui(session_factory, session_registry=None, **kwargs):
        await session_factory()
        return 0

    def unconfigured(*args, **kwargs):
        # The real error the factory raises, not a bare ValueError: only these
        # subclasses route into setup mode, so a stand-in would exercise the
        # fail-fast branch instead of the path under test.
        from local_operator.session_factory import HostingNotConfiguredError

        raise HostingNotConfiguredError("Hosting platform is not configured.")

    fake_tui = _fake_tui_module()
    setattr(fake_tui, "run_tui", run_tui)
    monkeypatch.setitem(sys.modules, "local_operator.tui", fake_tui)
    monkeypatch.setattr("local_operator.session_factory.resolve_hosting_model", unconfigured)
    monkeypatch.setattr("local_operator.cli.ConfigManager", _fake_config_manager)
    monkeypatch.setattr("local_operator.cli.CredentialManager", _bare_credential_manager)
    monkeypatch.setattr("local_operator.agents.AgentRegistry", MagicMock())
    monkeypatch.setattr(
        "local_operator.session.attached.AttachedSession.cold", staticmethod(fake_cold)
    )
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    with patch("sys.argv", ["program", "--model", "some-model"]):
        assert main() == 0
    assert seen["initial_model"] is None
    assert (
        seen["override"] is False
    ), "an unresolved model must not be claimed as a deliberate override"


def test_main_interactive_tty_uses_tui(
    tmp_home: Path, quiet_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """tty stdout + importable TUI -> run_tui(session_factory); the factory
    hands the TUI the wired session (TUI-003 contract), and ``values.tui.theme``
    reaches run_tui (CL-13)."""
    sentinel_session = object()
    seen: dict[str, Any] = {}

    async def fake_create_session(*args, **kwargs):
        seen["factory_called"] = True
        return sentinel_session

    # The TUI no longer builds a Session: `lop` boots a VIEWER, so the factory
    # it awaits returns a AttachedSession and `create_session` is never reached
    # from this path. Both are patched — `create_session` so a regression that
    # revived the owner path fails loudly here, and the viewer builder so the
    # test still observes the factory the CLI actually wires.
    monkeypatch.setattr("local_operator.cli.create_session", fake_create_session)

    async def fake_cold(session_id, *args, **kwargs):
        seen["factory_called"] = True
        seen["viewer_session_id"] = session_id
        return sentinel_session

    monkeypatch.setattr(
        "local_operator.session.attached.AttachedSession.cold", staticmethod(fake_cold)
    )

    fake_tui = _fake_tui_module()

    async def fake_run_tui(
        session_factory,
        theme_name: str = "dark",
        provider_controller=None,
        resume_factory=None,
        on_config_changed=None,
        warm_session_imports=True,
    ) -> int:
        assert warm_session_imports is False, "CLI viewers must skip owner imports"
        seen["theme"] = theme_name
        seen["session"] = await session_factory()
        # Recorded so the test can prove the CLI actually WIRED these. A fake
        # with `=None` defaults passes happily when nothing is passed, which is
        # how a positionally-bound controller shipped inert once already.
        seen["provider_controller"] = provider_controller
        return 0

    setattr(fake_tui, "run_tui", fake_run_tui)
    monkeypatch.setitem(sys.modules, "local_operator.tui", fake_tui)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)

    def _theme_config_manager(*args, **kwargs):
        manager = MagicMock()
        manager.get_config_value = lambda key, default=None: (
            {"theme": "light"} if key == "tui" else False
        )
        return manager

    monkeypatch.setattr("local_operator.cli.ConfigManager", _theme_config_manager)
    monkeypatch.setattr("local_operator.cli.CredentialManager", MagicMock())
    monkeypatch.setattr("local_operator.agents.AgentRegistry", MagicMock())

    with patch("sys.argv", ["program", "--hosting", "test", "--model", "m"]):
        assert main() == 0
    assert seen["factory_called"] is True
    assert seen["session"] is sentinel_session
    assert seen["theme"] == "light"
    # The whole provider surface (/provider, /accounts, /usage, /model, /login,
    # /logout) is inert without a real controller. Asserting non-None is what
    # catches a parameter that got bound positionally into the wrong slot — a
    # fake with `=None` defaults otherwise passes while the feature ships dead.
    assert seen["provider_controller"] is not None


def test_main_no_tui_flag_uses_headless_repl(
    tmp_home: Path, quiet_env: None, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """--no-tui on a tty falls back to the headless REPL."""
    fake_session = MagicMock()
    fake_session.subscribe = MagicMock(return_value=lambda: None)
    fake_session.dispose = AsyncMock()

    async def fake_create_session(*args, **kwargs):
        return fake_session

    monkeypatch.setattr("local_operator.cli.create_session", fake_create_session)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    # EOF on the first prompt exits the REPL cleanly.
    monkeypatch.setattr("builtins.input", _raise_eof)
    monkeypatch.setattr("local_operator.cli.ConfigManager", _fake_config_manager)
    monkeypatch.setattr("local_operator.cli.CredentialManager", MagicMock())
    monkeypatch.setattr("local_operator.agents.AgentRegistry", MagicMock())

    with patch("sys.argv", ["program", "--no-tui", "--hosting", "test", "--model", "m"]):
        assert main() == 0
    assert fake_session.dispose.await_count == 1


def _raise_eof(prompt: str = "") -> str:
    raise EOFError


def _bare_credential_manager(*args, **kwargs) -> MagicMock:
    """CredentialManager stand-in with no resolvable secrets: get_credential
    returns None and the credentials.env view is empty — preflight must see
    exactly the same view a keyless install has."""
    manager = MagicMock()
    manager.get_credential.return_value = None
    manager.get_credentials.return_value = {}
    return manager


# --- CL-04: --yolo reachable from subcommands ----------------------------------


def test_yolo_parses_on_exec(parser: argparse.ArgumentParser) -> None:
    """`exec "task" --yolo` parses (not exit 2) and reaches args.yolo."""
    args = parser.parse_args(["exec", "task", "--yolo"])
    assert args.subcommand == "exec"
    assert args.yolo is True
    # root-position still works (additive, both orderings documented)
    assert parser.parse_args(["--yolo", "exec", "task"]).yolo is True
    assert parser.parse_args(["exec", "task"]).yolo is False


@pytest.mark.parametrize(
    "argv",
    [
        ["serve", "--yolo"],
        ["config", "list", "--yolo"],
        ["agents", "list", "--yolo"],
        ["mcp", "add", "srv", "--yolo"],
        ["credential", "update", "K", "--yolo"],
    ],
)
def test_yolo_parses_on_every_subcommand(parser: argparse.ArgumentParser, argv: list[str]) -> None:
    assert parser.parse_args(argv).yolo is True


# --- run-shaping flags are position-independent too -----------------------------


def test_run_shaping_flags_parse_after_the_subcommand(parser: argparse.ArgumentParser) -> None:
    """`exec - --model X --hosting Y --run-in Z` parses instead of exiting 2.

    An external supervisor composes argv programmatically and naturally writes
    the flags after the subcommand. Before this, that form died with
    "unrecognized arguments" and an exit 2 that reads like a broken install.
    """
    args = parser.parse_args(
        ["exec", "-", "--hosting", "anthropic", "--model", "claude-opus-5", "--run-in", "/tmp"]
    )
    assert args.subcommand == "exec"
    assert args.hosting == "anthropic"
    assert args.model == "claude-opus-5"
    assert args.run_in == "/tmp"


def test_run_shaping_flags_still_parse_before_the_subcommand(
    parser: argparse.ArgumentParser,
) -> None:
    """The documented root-position form keeps working \u2014 this is additive."""
    args = parser.parse_args(
        ["--hosting", "anthropic", "--model", "claude-opus-5", "--run-in", "/tmp", "exec", "-"]
    )
    assert args.hosting == "anthropic"
    assert args.model == "claude-opus-5"
    assert args.run_in == "/tmp"


def test_a_subcommand_does_not_clobber_a_root_run_shaping_flag(
    parser: argparse.ArgumentParser,
) -> None:
    """The argparse re-default quirk must not swallow a value set BEFORE the
    subcommand \u2014 the exact failure ``--resume`` documents having hit."""
    args = parser.parse_args(["--model", "claude-opus-5", "exec", "task"])
    assert args.model == "claude-opus-5"


# --- CL-06: startup preflight ---------------------------------------------------


def test_main_preflight_missing_hosting_headless(
    tmp_home: Path, quiet_env: None, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """Non-tty startup with nothing configured prints the COMPLETE first-run
    quickstart (hosting + model + key + commands) and exits 1 BEFORE any turn.

    Item A1/U1: the fail-fast paths (non-tty, headless, exec) no longer die at
    the first missing field with just "Hosting platform is not configured" —
    they name everything missing at once so a scripted user fixes it in one
    pass. The interactive TUI path takes the setup state instead (see
    ``test_main_preflight_missing_hosting_tty_enters_setup``)."""
    called: dict[str, bool] = {"factory": False}

    async def fake_create_session(*args, **kwargs):
        called["factory"] = True
        return MagicMock()

    monkeypatch.setattr("local_operator.cli.create_session", fake_create_session)
    # Non-tty: no setup state is possible, so this stays fail-fast.
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
    monkeypatch.setattr("local_operator.cli.ConfigManager", _fake_config_manager)
    monkeypatch.setattr("local_operator.cli.CredentialManager", MagicMock())
    monkeypatch.setattr("local_operator.agents.AgentRegistry", MagicMock())

    with patch("sys.argv", ["program"]):
        assert main() == 1
    # stderr, matching its sibling _preflight_api_key: an error message belongs
    # on the diagnostic channel regardless of which front end asked for it.
    err = capsys.readouterr().err
    assert "not configured yet" in err
    # Every remedy named at once, not one field at a time.
    assert "login <provider>" in err
    assert "config edit hosting" in err
    assert "config edit model_name" in err
    assert "credential update" in err
    assert called["factory"] is False


def test_main_preflight_missing_hosting_tty_enters_setup(
    tmp_home: Path, quiet_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """tty + importable TUI + nothing configured -> the app OPENS (setup state)
    instead of failing at preflight (item A1/U1 headline).

    The session factory is still called: the TUI boots, the factory raises
    HostingNotConfiguredError inside it, and the app's boot-failure handler
    turns that into the guided setup state. What matters here is that main()
    reached the TUI launch (return 0) rather than returning a preflight error."""
    seen: dict[str, Any] = {}

    async def fake_create_session(*args, **kwargs):
        # The real factory would raise here; the app handles that. The point of
        # this test is that main() got PAST preflight to the launch.
        seen["factory_called"] = True
        return object()

    monkeypatch.setattr("local_operator.cli.create_session", fake_create_session)

    fake_tui = _fake_tui_module()

    async def fake_run_tui(
        session_factory,
        theme_name: str = "dark",
        provider_controller=None,
        resume_factory=None,
        on_config_changed=None,
        warm_session_imports=True,
    ) -> int:
        assert warm_session_imports is False, "CLI viewers must skip owner imports"
        # Prove the setup-state plumbing reached the app: the reconciliation
        # hook is wired so a first-run /login can take effect.
        seen["on_config_changed"] = on_config_changed
        return 0

    setattr(fake_tui, "run_tui", fake_run_tui)
    monkeypatch.setitem(sys.modules, "local_operator.tui", fake_tui)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr("local_operator.cli.ConfigManager", _fake_config_manager)
    monkeypatch.setattr("local_operator.cli.CredentialManager", MagicMock())
    monkeypatch.setattr("local_operator.agents.AgentRegistry", MagicMock())

    with patch("sys.argv", ["program"]):
        assert main() == 0
    assert seen.get("on_config_changed") is not None


def _bad_hosting_config_manager(*args, **kwargs) -> MagicMock:
    """ConfigManager stand-in whose hosting names no real provider.

    Mirrors the corrupted `config.yml` that motivated the fix: a typo'd hosting
    beside a perfectly valid model name, which is why the failure surfaced as
    "Unsupported hosting platform" rather than as a missing-model error.
    """
    values = {"hosting": "anthropicxyq", "model_name": "claude-sonnet-4-5"}
    return MagicMock(get_config_value=lambda key, *a: values.get(key, False))


def test_main_preflight_unknown_hosting_tty_enters_setup(
    tmp_home: Path, quiet_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """tty + a hosting the registry does not own -> the app OPENS (setup state).

    The hotfix: this configuration used to fail preflight/boot outright, so the
    user got a red "session failed to start" and, with no session, could not
    use `/login` or `/model` to escape it either. It is now classified exactly
    like the nothing-configured case, which is the state the in-app remedies
    work from.
    """
    seen: dict[str, Any] = {}

    async def fake_create_session(*args, **kwargs):
        seen["factory_called"] = True
        return object()

    monkeypatch.setattr("local_operator.cli.create_session", fake_create_session)

    fake_tui = _fake_tui_module()

    async def fake_run_tui(
        session_factory,
        theme_name: str = "dark",
        provider_controller=None,
        resume_factory=None,
        on_config_changed=None,
        warm_session_imports=True,
    ) -> int:
        assert warm_session_imports is False, "CLI viewers must skip owner imports"
        seen["launched"] = True
        return 0

    setattr(fake_tui, "run_tui", fake_run_tui)
    monkeypatch.setitem(sys.modules, "local_operator.tui", fake_tui)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr("local_operator.cli.ConfigManager", _bad_hosting_config_manager)
    monkeypatch.setattr("local_operator.cli.CredentialManager", MagicMock())
    monkeypatch.setattr("local_operator.agents.AgentRegistry", MagicMock())

    with patch("sys.argv", ["program"]):
        assert main() == 0
    assert seen.get("launched") is True


def test_main_preflight_unknown_hosting_non_tty_fails_fast_naming_the_value(
    tmp_home: Path, quiet_env: None, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """Non-tty keeps fail-fast, with a message naming the value AND the remedy.

    A scripted/CI run must not limp along with no usable model. It must also
    not be told "nothing is configured" (the first-run quickstart), because
    something IS configured -- just not to a real provider, and the user needs
    to know WHICH word in their config is wrong.
    """
    called: dict[str, bool] = {"factory": False}

    async def fake_create_session(*args, **kwargs):
        called["factory"] = True
        return MagicMock()

    monkeypatch.setattr("local_operator.cli.create_session", fake_create_session)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
    monkeypatch.setattr("local_operator.cli.ConfigManager", _bad_hosting_config_manager)
    monkeypatch.setattr("local_operator.cli.CredentialManager", MagicMock())
    monkeypatch.setattr("local_operator.agents.AgentRegistry", MagicMock())

    with patch("sys.argv", ["program"]):
        assert main() == 1
    err = capsys.readouterr().err
    assert "anthropicxyq" in err
    assert "not a known provider" in err
    # The remedy, not just the diagnosis.
    assert "config edit hosting" in err or "login <provider>" in err
    # NOT the first-run quickstart: it would contradict the user's own file.
    assert "not configured yet" not in err
    assert called["factory"] is False


def test_main_interactive_missing_api_key_warns_and_starts(
    tmp_home: Path,
    quiet_env: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    """A keyed provider with NO resolvable key still starts interactively.

    The fatal preflight sat between the user and the in-app `/login` remedy:
    a config whose default hosting was a keyed provider (e.g. openrouter)
    could not start at all once the key was gone. Interactive startup now
    warns on stderr and boots; the exec path keeps the fatal check (see
    test_preflight_api_key_fatal_by_default below).
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    seen: dict[str, Any] = {}

    async def fake_create_session(*args, **kwargs):
        seen["built"] = True
        return MagicMock()

    # `lop` boots a viewer, so the preflight's "the session was allowed
    # through" is now observed on the viewer builder rather than on
    # `create_session`. What the test is really asserting — that the preflight
    # did not block the launch — is unchanged.
    async def fake_cold(session_id, *args, **kwargs):
        seen["built"] = True
        return MagicMock()

    fake_tui = _fake_tui_module()

    async def fake_run_tui(
        session_factory,
        theme_name="dark",
        provider_controller=None,
        resume_factory=None,
        on_config_changed=None,
        warm_session_imports=True,
    ) -> int:
        assert warm_session_imports is False, "CLI viewers must skip owner imports"
        await session_factory()
        return 0

    setattr(fake_tui, "run_tui", fake_run_tui)
    monkeypatch.setitem(sys.modules, "local_operator.tui", fake_tui)
    monkeypatch.setattr("local_operator.cli.create_session", fake_create_session)
    monkeypatch.setattr(
        "local_operator.session.attached.AttachedSession.cold", staticmethod(fake_cold)
    )
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr("local_operator.cli.ConfigManager", _fake_config_manager)
    monkeypatch.setattr("local_operator.cli.CredentialManager", _bare_credential_manager)
    monkeypatch.setattr("local_operator.agents.AgentRegistry", MagicMock())

    with patch("sys.argv", ["program", "--hosting", "openai", "--model", "gpt-4o"]):
        assert main() == 0
    err = capsys.readouterr().err
    # The warning names the fact and the in-app remedy, and is not an Error.
    assert "Warning" in err and "openai" in err and "/login openai" in err
    assert "Error" not in err
    assert seen.get("built") is True


def test_preflight_api_key_fatal_by_default(
    tmp_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    """`_preflight_api_key` stays fatal without the interactive opt-out.

    The exec path calls it bare: a scripted one-shot run has no login prompt,
    so "start anyway and fail mid-turn" would only move the same failure
    somewhere harder to read.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert cli._preflight_api_key("openai", _bare_credential_manager()) == 1
    err = capsys.readouterr().err
    assert "OPENAI_API_KEY" in err and "Error" in err


def test_preflight_accepts_stored_oauth_under_temporary_backoff(
    tmp_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    """A transient refresh failure must not become a false missing-key error.

    Stream-time failover owns the temporary block and the next refresh. Startup
    only needs to know that the OAuth credential exists, so the user can still
    reach the TUI and `/login` if the provider ultimately rejects it.
    """
    from local_operator.providers.auth_store import AuthStore

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_OAUTH_TOKEN", raising=False)
    store = AuthStore()
    credential = store.upsert_credential(
        "anthropic",
        {
            "access": "expired-access",
            "refresh": "refresh-token",
            "expires": 0,
            "account_id": "account-id",
        },
    )
    store.block_credential(credential.id, "anthropic")
    store.close()

    assert cli._preflight_api_key("anthropic", _bare_credential_manager()) is None
    assert capsys.readouterr().err == ""


def test_main_preflight_env_key_passes(
    tmp_home: Path, quiet_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the env key present the preflight lets the session through."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    seen: dict[str, Any] = {}

    async def fake_create_session(*args, **kwargs):
        seen["built"] = True
        return MagicMock()

    # `lop` boots a viewer, so the preflight's "the session was allowed
    # through" is now observed on the viewer builder rather than on
    # `create_session`. What the test is really asserting — that the preflight
    # did not block the launch — is unchanged.
    async def fake_cold(session_id, *args, **kwargs):
        seen["built"] = True
        return MagicMock()

    fake_tui = _fake_tui_module()

    async def fake_run_tui(
        session_factory,
        theme_name="dark",
        provider_controller=None,
        resume_factory=None,
        on_config_changed=None,
        warm_session_imports=True,
    ) -> int:
        assert warm_session_imports is False, "CLI viewers must skip owner imports"
        seen.setdefault("provider_controller", provider_controller)
        await session_factory()
        return 0

    setattr(fake_tui, "run_tui", fake_run_tui)
    monkeypatch.setitem(sys.modules, "local_operator.tui", fake_tui)
    monkeypatch.setattr("local_operator.cli.create_session", fake_create_session)
    monkeypatch.setattr(
        "local_operator.session.attached.AttachedSession.cold", staticmethod(fake_cold)
    )
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr("local_operator.cli.ConfigManager", _fake_config_manager)
    monkeypatch.setattr("local_operator.cli.CredentialManager", MagicMock())
    monkeypatch.setattr("local_operator.agents.AgentRegistry", MagicMock())

    with patch("sys.argv", ["program", "--hosting", "openai", "--model", "gpt-4o"]):
        assert main() == 0
    assert seen.get("built") is True


# --- CL-13: --tui forces the TUI ------------------------------------------------


def test_tui_flag_forces_tui_on_non_tty(
    tmp_home: Path, quiet_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    async def fake_create_session(*args, **kwargs):
        return MagicMock()

    fake_tui = _fake_tui_module()

    async def fake_run_tui(
        session_factory,
        theme_name="dark",
        provider_controller=None,
        resume_factory=None,
        on_config_changed=None,
        warm_session_imports=True,
    ) -> int:
        assert warm_session_imports is False, "CLI viewers must skip owner imports"
        seen.setdefault("provider_controller", provider_controller)
        seen["ran"] = True
        return 0

    setattr(fake_tui, "run_tui", fake_run_tui)
    monkeypatch.setitem(sys.modules, "local_operator.tui", fake_tui)
    monkeypatch.setattr("local_operator.cli.create_session", fake_create_session)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)  # NOT a tty
    monkeypatch.setattr("local_operator.cli.ConfigManager", _fake_config_manager)
    monkeypatch.setattr("local_operator.cli.CredentialManager", MagicMock())
    monkeypatch.setattr("local_operator.agents.AgentRegistry", MagicMock())

    with patch("sys.argv", ["program", "--tui", "--hosting", "test", "--model", "m"]):
        assert main() == 0
    assert seen.get("ran") is True


def test_tui_flag_default_off(parser: argparse.ArgumentParser) -> None:
    assert parser.parse_args([]).tui is False
    assert parser.parse_args(["--tui"]).tui is True


# --- CL-16: deprecated config keys -----------------------------------------------


def test_config_list_marks_deprecated_keys(tmp_home: Path, capsys) -> None:
    from local_operator.cli import config_list_command

    assert config_list_command() == 0
    out = capsys.readouterr().out
    for key in ("conversation_length", "detail_length", "max_learnings_history"):
        block = out[out.index(f"│ {key}:") :]
        assert "[DEPRECATED" in block.split("╰")[0]


# --- CL-17: golden legacy parser inventory ----------------------------------------


def _walk_actions(parser: argparse.ArgumentParser) -> dict[str, dict[str, Any]]:
    out = {}
    for action in parser._actions:
        if action.option_strings and action.option_strings[0] == "-h":
            continue
        key = (
            ",".join(action.option_strings) if action.option_strings else "POS:" + str(action.dest)
        )
        out[key] = {
            "dest": action.dest,
            "default": action.default if action.default is not argparse.SUPPRESS else "<SUPPRESS>",
            "choices": sorted(action.choices) if action.choices else None,
            "required": bool(action.required),
            "nargs": action.nargs,
        }
    return out


def _inventory(parser: argparse.ArgumentParser) -> dict[str, dict[str, Any]]:
    seen: set[int] = set()
    inventory = {"$": _walk_actions(parser)}
    seen.add(id(parser))

    def record(p: argparse.ArgumentParser, path: str) -> None:
        subparsers_group = getattr(p, "_subparsers", None)
        if subparsers_group is None:
            return
        for action in subparsers_group._actions:
            if not isinstance(action, argparse._SubParsersAction):
                continue
            for name, sub in action.choices.items():
                if not isinstance(name, str) or id(sub) in seen:
                    continue
                seen.add(id(sub))
                full = (path + " " + name).strip()
                inventory[full] = _walk_actions(sub)
                record(sub, full)

    record(parser, "")
    return inventory


def test_golden_legacy_parser_surface() -> None:
    """Every legacy option (option strings, dest, default, choices, required)
    present in the ``main``-branch parser survives in the rewritten parser;
    additive options are allowed, removals or shape changes fail."""
    import json

    golden_path = Path(__file__).parent / "golden_legacy_parser.json"
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    current = _inventory(build_cli_parser())

    # The ONE deliberate relaxation of the legacy surface, recorded here rather
    # than edited into the golden data so the reason is visible to a reviewer.
    # `exec` grew loop-only and piped-stdin forms (`--loop`, `--loop-goal`, `-`,
    # omitted-with-a-pipe), and argparse cannot express "required unless one of
    # those" — so the positional is `nargs="?"` and `run_exec` enforces the real
    # rule, naming the ways to supply a prompt. Relaxing required->optional is
    # backward compatible: every legacy invocation that passed a prompt still
    # parses identically. Nothing else may change shape.
    RELAXED_TO_OPTIONAL = {("exec", "POS:command"): {"required": False, "nargs": "?"}}

    problems: list[str] = []
    for command, options in golden.items():
        if command not in current:
            problems.append(f"missing subcommand path: {command}")
            continue
        current_options = current[command]
        for key, spec in options.items():
            if key not in current_options:
                problems.append(f"{command}: removed option {key}")
                continue
            now = current_options[key]
            if spec["choices"] is not None:
                # Additive surface: NEW choices (new subcommands, new hosting
                # values) are allowed; removing a legacy choice fails.
                removed = set(spec["choices"]) - set(now["choices"] or [])
                if now["dest"] != spec["dest"] or now["required"] != spec["required"]:
                    problems.append(
                        f"{command}: {key} dest/required changed: "
                        f"{(spec['dest'], spec['required'])} -> {(now['dest'], now['required'])}"
                    )
                if removed:
                    problems.append(f"{command}: {key} lost choices: {sorted(removed)}")
                continue
            allowed = RELAXED_TO_OPTIONAL.get((command, key), {})
            for field in ("dest", "default", "required", "nargs"):
                if field in allowed and now[field] == allowed[field]:
                    continue
                if now[field] != spec[field]:
                    problems.append(
                        f"{command}: {key} {field} changed: " f"{spec[field]!r} -> {now[field]!r}"
                    )
    assert not problems, "\n".join(problems)


# --- scheduler degradation names the RIGHT cause ----------------------------


def _run_scheduler_with_import_error(monkeypatch, capsys, missing: str) -> str:
    """Drive _run_with_scheduler with a ModuleNotFoundError for `missing`."""
    import local_operator.jobs as jobs_mod

    def boom(*_args, **_kwargs):
        raise ModuleNotFoundError(f"No module named {missing!r}", name=missing)

    monkeypatch.setattr(jobs_mod, "JobManager", boom)

    async def front(*_a, **_k):
        return 0

    assert asyncio.run(cli._run_with_scheduler(front)) == 0
    return capsys.readouterr().err


def test_missing_server_extra_names_the_extra(monkeypatch, capsys) -> None:
    """A bare install has no apscheduler, and this wraps BOTH front ends, so
    it fires on the most-travelled path in the product."""
    err = _run_scheduler_with_import_error(monkeypatch, capsys, "apscheduler")
    assert 'requires the "server" extra' in err
    assert 'pip install "local-operator[server]"' in err


def test_broken_internal_import_is_not_blamed_on_the_extra(monkeypatch, capsys) -> None:
    """Catching every ModuleNotFoundError here reported a broken INTERNAL import
    as a missing extra: the user installs it, nothing changes, and the real
    defect stays invisible. Strictly less diagnostic than the raw message."""
    err = _run_scheduler_with_import_error(monkeypatch, capsys, "local_operator.scheduler_service")
    assert "server" not in err or "extra" not in err
    assert "local_operator.scheduler_service" in err


def test_resume_survives_in_front_of_the_subcommand() -> None:
    """A global option has to work where `--help` says it does.

    Routed only through `parent_parser`, argparse re-applied the parent action's
    default under the subparser and clobbered a value set BEFORE the subcommand:
    `--resume ID exec "…"` parsed as `resume=None` and started a FRESH session,
    which is verbatim the failure the field exists to prevent. Validation could
    not catch it either, because validation reads the post-clobber value.
    """
    parser = build_cli_parser()

    after = parser.parse_args(["exec", "--resume", "sess-abc123", "hi"])
    before = parser.parse_args(["--resume", "sess-abc123", "exec", "hi"])
    assert after.resume == before.resume == "sess-abc123"

    # Same for a subcommand that only reads it for validation.
    assert parser.parse_args(["--resume", "bogus", "config", "list"]).resume == "bogus"
    assert parser.parse_args(["config", "list", "--resume", "bogus"]).resume == "bogus"

    # And the bare form still means "the most recent". It has to come last: with
    # `nargs="?"` a following word IS the id, so `--resume hi` names a session
    # called `hi` and leaves exec without a prompt.
    assert parser.parse_args(["exec", "hi", "--resume"]).resume == cli.RESUME_LATEST

    # That case USED to exit 2, because `command` was a required positional.
    # `exec` now supports loop-only and piped-stdin runs, so the positional is
    # optional and argparse can no longer reject it at parse time. The ambiguity
    # is unchanged and still resolved the same way (`hi` is the session id, not
    # the prompt); only the layer that reports it moved, to `run_exec`, which
    # names how to supply a prompt instead of printing a bare usage block.
    ambiguous = parser.parse_args(["exec", "--resume", "hi"])
    assert ambiguous.resume == "hi" and ambiguous.command is None


def test_a_background_job_carries_the_session_it_was_told_to_resume() -> None:
    """`--background` is the same request run elsewhere.

    The flag was accepted by the front end and dropped at the process boundary:
    `build_worker_argv` serialized every other field, so the worker started a new
    session and reported success against the wrong history.
    """
    from local_operator.exec_mode import ExecArgs, build_worker_argv
    from local_operator.exec_worker import build_parser

    argv = build_worker_argv("hi", ExecArgs(resume="sess-abc123"))
    # `--resume=<id>` as one item: every value-carrying option uses the `=`
    # form so a value beginning with `-` cannot be read as the next option.
    assert "--resume=sess-abc123" in argv

    # And the worker on the other side accepts what was serialized — parsed from
    # the real argv minus the `python [flags] -m <module>` prefix, so the test
    # breaks if the serialization and the worker's parser ever disagree.
    # The prefix length is DERIVED rather than hardcoded: interpreter flags sit
    # between the executable and `-m` (see local_operator.interpreter), so a
    # fixed slice silently turns into a parser error when one is added.
    assert build_parser().parse_args(argv[argv.index("-m") + 2 :]).resume == "sess-abc123"

    # Nothing is emitted when nothing was asked for.
    assert not any(a.startswith("--resume") for a in build_worker_argv("hi", ExecArgs()))


def test_a_bare_resume_classifies_sessions_before_resolving_latest(tmp_path, monkeypatch) -> None:
    """``--resume`` is answered in the CLI, BEFORE any session is built.

    The session factory also backfills, but it runs when a session is
    constructed — which is after this branch has already picked a directory.
    So on the first launch after an upgrade, a bare ``--resume`` resolved
    ``@latest`` against an unclassified store and reopened whichever delegated
    run happened to finish last: the CLI spelling of the exact bug the picker
    fix is about.
    """
    import json
    import os

    from local_operator import resume as resume_mod

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    sessions = tmp_path / "sessions"

    def seed(name: str, opening: str, when: int) -> None:
        directory = sessions / name
        directory.mkdir(parents=True)
        entry = {
            "id": "e1",
            "ts": 0,
            "type": "message",
            "payload": {"kind": "message", "role": "user", "content": [{"text": opening}]},
        }
        transcript = directory / resume_mod.TRANSCRIPT_NAME
        transcript.write_text(json.dumps(entry) + "\n", encoding="utf-8")
        os.utime(transcript, (when, when))

    # The child settles AFTER the parent's last turn, which is the ordinary
    # case: a delegated review outlives the turn that launched it.
    seed("mine00000000", "fix the resume picker", 1_000_000)
    seed("child0000000", "[role: reviewer]\nreview the diff", 2_000_000)

    # Neither directory is marked, exactly like a store that predates the fix.
    assert not any((path / resume_mod.ORIGIN_NAME).exists() for path in sessions.iterdir())

    # Drive the CLI branch itself rather than re-implementing its order here:
    # the defect was entirely in WHICH function ran first, so a test that
    # calls them in the right order by hand cannot see it. `main` is stopped
    # right after the resume block by a sentinel raised from the next call it
    # makes, leaving `args.resume` holding what the branch resolved.
    resolved: list[str] = []

    class _Stop(Exception):
        pass

    monkeypatch.setattr(cli, "setup_cross_platform_environment", lambda: None)
    monkeypatch.setattr(cli.sys, "argv", ["local-operator", "--resume"])
    original = resume_mod.resolve_resume_id

    def _record(config_dir, requested):  # noqa: ANN001, ANN202
        value = original(config_dir, requested)
        resolved.append(value)
        raise _Stop

    monkeypatch.setattr(resume_mod, "resolve_resume_id", _record)
    with contextlib.suppress(_Stop, SystemExit, Exception):
        cli.main()

    assert resolved == [
        "mine00000000"
    ], f"a bare --resume reopened a subagent's transcript: {resolved}"


# --- Soft-death reaper signal scope (R1) ------------------------------------
#
# The soft-death reaper must reap on a genuine process TERMINATION but never on
# a mid-turn Ctrl-C. In the headless REPL, SIGINT is a turn abort that keeps the
# session — and its `background=true` bash jobs — ALIVE (`session.abort` spares
# them on purpose). Wiring the reaper onto SIGINT would SIGKILL those still-live
# groups while the owning process keeps running, the one case the reaper exists
# to forbid. These tests pin "SIGINT does not reap; SIGTERM does" so a future
# edit that re-adds SIGINT to the registration loop fails loudly.


def test_soft_death_reaper_installs_on_sigterm_not_sigint(monkeypatch):
    """`_install_group_reaper_soft_death` handles SIGTERM but leaves SIGINT alone."""
    import signal

    from local_operator import cli

    # Start from known signal state and restore it after, so the process this
    # suite runs in is not left with a reaper handler bound to it.
    original_term = signal.getsignal(signal.SIGTERM)
    original_int = signal.getsignal(signal.SIGINT)
    try:
        signal.signal(signal.SIGINT, signal.default_int_handler)
        int_before = signal.getsignal(signal.SIGINT)

        cli._install_group_reaper_soft_death()

        # SIGTERM was chained onto a new handler (a genuine termination reaps).
        assert callable(signal.getsignal(signal.SIGTERM))
        # SIGINT is untouched: still exactly the turn-abort default it had.
        assert signal.getsignal(signal.SIGINT) is int_before
    finally:
        signal.signal(signal.SIGTERM, original_term)
        signal.signal(signal.SIGINT, original_int)


def test_soft_death_sigterm_handler_reaps_then_chains(monkeypatch):
    """The installed SIGTERM handler reaps this process's groups, then chains."""
    import signal

    from local_operator import cli

    reaped: list[str] = []
    monkeypatch.setattr(
        "local_operator.tools.group_reaper.kill_own_groups",
        lambda: reaped.append("reaped"),
    )

    prior_calls: list[int] = []
    original_term = signal.getsignal(signal.SIGTERM)
    try:
        # A pre-existing SIGTERM handler that the reaper must chain to, not eat.
        signal.signal(signal.SIGTERM, lambda signum, frame: prior_calls.append(signum))

        cli._install_group_reaper_soft_death()
        handler = signal.getsignal(signal.SIGTERM)
        assert callable(handler)

        # Invoke the handler directly (no real signal delivery): it must reap
        # first, then chain to the handler that was installed before it.
        handler(signal.SIGTERM, None)
        assert reaped == ["reaped"]
        assert prior_calls == [signal.SIGTERM]
    finally:
        signal.signal(signal.SIGTERM, original_term)


def test_agents_list_hides_the_seed_provenance_marker(capsys) -> None:
    """F9: `seed:<name>` records that a role was installed from a packaged
    starter, so `agent op='reset'` knows it may restore it. It is bookkeeping
    this listing's reader cannot act on, and it does not belong in a
    human-facing inventory beside tags they wrote."""
    registry = MagicMock()
    agent = MagicMock()
    agent.name = "reviewer"
    agent.id = "r-1"
    agent.created_date = "now"
    agent.version = "1.0.0"
    agent.hosting = ""
    agent.model = ""
    agent.description = "Reviewing a merge request"
    agent.tags = ["role", "seed:reviewer", "tools:read,grep"]
    agent.categories = ["role"]
    registry.list_agents.return_value = [agent]

    assert agents_list_command(argparse.Namespace(page=1, perpage=10), registry) == 0

    output = capsys.readouterr().out
    assert "seed:reviewer" not in output
    assert "Tags: role, tools:read,grep" in output, "the tags a human set still show"


# --- qwencloud-ticket: the two-store surface (PR 2, slice B) -----------------
#
# The value never appears here. `FAKE_QWEN_TICKET` is the only ticket-shaped
# literal in this block, and several tests assert it is ABSENT from the output.

#: Never a real cookie.
FAKE_QWEN_TICKET = "fake-console-ticket"


class _QwenRowStore:
    """The `AuthStore` surface `_qwencloud_ticket_action` actually uses.

    `read_ticket_record`/`store_ticket`/`delete_ticket` are monkeypatched per
    test, so the only method that has to behave is `list_credentials` — which
    `_qwencloud_credential_row_exists` calls to decide whether the
    "no alibaba-token-plan credential" warning prints.
    """

    def __init__(self, augment_rows: list[Any] | None = None) -> None:
        self.augment_rows = augment_rows if augment_rows is not None else []

    def list_credentials(self, provider: str, **kwargs: Any) -> list[Any]:
        if provider == cli._QWENCLOUD_TICKET_AUGMENTS:
            return self.augment_rows
        return []


def _patch_qwen(monkeypatch: pytest.MonkeyPatch, **fns: Any) -> None:
    """Patch the qwencloud_console seam the CLI imports at point of use.

    The import inside `_qwencloud_ticket_action` is `from ... import name`, so
    the binding resolved at call time is the module attribute — patching the
    module is what the CLI sees.
    """
    from local_operator.providers import qwencloud_console

    for name, fn in fns.items():
        monkeypatch.setattr(qwencloud_console, name, fn)


def _locked_exc() -> Exception:
    from local_operator.providers.qwencloud_console import TicketStoreLocked

    return TicketStoreLocked(
        "the secret store is hardened and locked, so whether the ticket's "
        "value is present is UNKNOWN. Run `lop secret unlock`, then retry"
    )


def _qwen_audit_events(base: Path) -> dict[str, int]:
    """Audit rows by event name, read from the store's SQLite file directly.

    Read below the API on purpose: the property is what the STORE recorded, so
    asking the same API that might be skipping the record would prove nothing.
    """
    import sqlite3 as _sqlite3

    from local_operator.secrets.keys import store_path

    path = store_path(base)
    if not path.exists():
        return {}
    connection = _sqlite3.connect(path)
    try:
        rows = connection.execute("select event, count(*) from audit group by event").fetchall()
    finally:
        connection.close()
    return dict(rows)


def test_status_reports_length_and_age_without_retrieving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """B1: length and age come from the `auth.db` metadata row, never a `get`.

    Driven against a REAL secret store, because the property is about what the
    store records. The assertion is "no `get` event, and `last_used_at` stays
    None" rather than "no audit row at all": `open_store()` legitimately
    appends a broker `key`/`deny:key` row per call, so a total-delta assertion
    would fail on correct code. The second half — one real `get` moving both —
    proves the pin discriminates instead of passing vacuously.
    """
    from local_operator.providers.auth_store import AuthStore
    from local_operator.providers.qwencloud_console import (
        QWENCLOUD_TICKET_SECRET_NAME,
        store_ticket,
    )
    from local_operator.secrets import access

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    base = tmp_path / "secretbase"
    base.mkdir()
    opened = AuthStore(db_path=tmp_path / "auth.db")
    try:
        store_ticket(opened, FAKE_QWEN_TICKET, base=base)

        # `_qwencloud_ticket_action` calls `read_ticket_record(store)` with no
        # `base`, so bind this test's throwaway base onto the seam the CLI uses.
        from local_operator.providers import qwencloud_console as _qc

        real_read = _qc.read_ticket_record
        monkeypatch.setattr(
            _qc, "read_ticket_record", lambda store, **kw: real_read(store, base=base)
        )

        before = _qwen_audit_events(base)
        assert cli._qwencloud_ticket_action("status", opened) == 0
        after = _qwen_audit_events(base)

        out = capsys.readouterr()
        assert f"{len(FAKE_QWEN_TICKET)} characters" in out.out
        assert "old" in out.out
        assert FAKE_QWEN_TICKET not in out.out and FAKE_QWEN_TICKET not in out.err

        assert after.get("get", 0) == before.get("get", 0), "status retrieved the value"
        assert access.open_store(base).describe(QWENCLOUD_TICKET_SECRET_NAME).last_used_at is None

        # The pin discriminates: one real retrieval moves BOTH observables.
        access.open_store(base).get(QWENCLOUD_TICKET_SECRET_NAME)
        assert _qwen_audit_events(base).get("get", 0) > before.get("get", 0)
        assert (
            access.open_store(base).describe(QWENCLOUD_TICKET_SECRET_NAME).last_used_at is not None
        )
    finally:
        opened.close()


def test_status_names_the_repair_for_a_metadata_orphan(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """B2: metadata row present, encrypted value gone — say so and name the fix."""
    _patch_qwen(
        monkeypatch,
        read_ticket_record=lambda store, **kw: {
            "credential_id": 1,
            "captured_at": int(time.time() * 1000),
            "length": len(FAKE_QWEN_TICKET),
            "secret_present": False,
        },
    )

    assert cli._qwencloud_ticket_action("status", _QwenRowStore([object()])) == 0
    out = capsys.readouterr().out
    assert "ENCRYPTED VALUE" in out
    assert "qwencloud-ticket set" in out


def test_status_on_a_legacy_row_does_not_warn_about_a_missing_value(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """B2b: `.get(..., True)` — a pre-migration row has no `secret_present`
    key and its value lives in `auth.db`, so the orphan warning must stay
    silent. Without the default it would fire on every un-migrated install."""
    _patch_qwen(
        monkeypatch,
        read_ticket_record=lambda store, **kw: {
            "credential_id": 1,
            "captured_at": int(time.time() * 1000),
            "length": len(FAKE_QWEN_TICKET),
        },
    )

    assert cli._qwencloud_ticket_action("status", _QwenRowStore([object()])) == 0
    assert "ENCRYPTED VALUE" not in capsys.readouterr().out


def test_status_on_a_locked_store_names_the_remedy(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """B3: a locked hardened store exits non-zero and names `lop secret unlock`.

    This pins the OUTCOME, not a clause. `status` has no dedicated
    `TicketStoreLocked` clause — the remedy travels in the exception message
    Slice A raises, and the generic `TicketStoreError` clause interpolates it.
    The mutation that discriminates is therefore dropping that clause's
    `return 1` (see the report's mutation table), not reordering clauses.
    """

    def locked(store: Any, **kwargs: Any) -> Any:
        raise _locked_exc()

    _patch_qwen(monkeypatch, read_ticket_record=locked)

    assert cli._qwencloud_ticket_action("status", _QwenRowStore()) == 1
    err = capsys.readouterr().err
    assert "lop secret unlock" in err
    assert "UNKNOWN" in err


def test_status_on_a_locked_store_is_not_no_ticket_stored(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """B4: "unknown" must never be reported as "none stored".

    That collapse is the false success this PR exists to prevent: it tells the
    user their full-account cookie is gone while it sits on disk.
    """

    def locked(store: Any, **kwargs: Any) -> Any:
        raise _locked_exc()

    _patch_qwen(monkeypatch, read_ticket_record=locked)

    assert cli._qwencloud_ticket_action("status", _QwenRowStore()) == 1
    assert "No QwenCloud console ticket stored" not in capsys.readouterr().out


def test_status_reports_a_secret_orphan_rather_than_nothing_stored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """MAJOR-1: `status` answered "nothing stored" over a LIVE value.

    A SECRET ORPHAN -- value present, no metadata row -- is a state this
    feature can actually reach: `store_ticket` writes the value first and the
    row second, on purpose, so a crash, a kill or an `auth.db` restored from an
    older backup leaves the value with nothing pointing at it. `status` read
    the row and stopped, so it printed the no-ticket receipt over a live
    full-account cookie -- hiding the exposure AND pointing the user away from
    `rm`, the only verb that revokes it. `rm` was already fixed for this state
    (`test_rm_revokes_a_secret_orphan_rather_than_reporting_nothing_stored`);
    this is the same fix one verb over.

    Driven against a REAL secret store with no seam patched, because the
    property is about what is on disk: the base the CLI resolves is
    `config_dir()/secrets`, so the env var points at that base itself.

    Fails AGAINST the faithful single-site revert of the fix -- deleting the
    `_secret_is_present` probe and restoring `if record is None: print("No
    QwenCloud console ticket stored.")` -- and against nothing else. The two
    tests below assert receipts the pre-fix code either already printed or
    could not reach, so they stay green under that revert; this one is the
    only pin on the probe itself.
    """
    from local_operator.providers.auth_store import AuthStore
    from local_operator.providers.qwencloud_console import QWENCLOUD_TICKET_SECRET_NAME
    from local_operator.secrets import access
    from local_operator.secrets.keys import store_path

    base = tmp_path / "config"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(base))
    secret_store = access.open_store(base, create=True)
    secret_store.initialize()
    secret_store.set(QWENCLOUD_TICKET_SECRET_NAME, FAKE_QWEN_TICKET.encode())

    # Preconditions: the value really is on disk and no row points at it, so
    # the receipt under test is about a live credential rather than an empty
    # store -- and so a later "nothing stored" would be a lie, not a fact.
    assert store_path(base).exists()
    assert access.open_store(base).describe(QWENCLOUD_TICKET_SECRET_NAME) is not None

    opened = AuthStore(db_path=tmp_path / "auth.db")
    try:
        assert cli._qwencloud_ticket_action("status", opened) == 0
        captured = capsys.readouterr()
    finally:
        opened.close()

    assert (
        "No QwenCloud console ticket stored" not in captured.out
    ), "status reported nothing stored while the value was live in the encrypted store"
    assert "VALUE is stored" in captured.out
    assert (
        "lop qwencloud-ticket rm" in captured.out
    ), "the receipt must name the only verb that revokes the orphan"
    assert FAKE_QWEN_TICKET not in captured.out and FAKE_QWEN_TICKET not in captured.err


def test_status_on_a_store_with_no_ticket_still_says_nothing_stored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Arms-length: a store that EXISTS without this ticket is still empty.

    The ordinary host for anyone who has used `lop secret` for anything else:
    the `store_path(base).exists()` guard does not short-circuit, the store is
    opened, and the probe must still return False. Without this, a fix that
    took the orphan branch whenever any store existed would pass the test above
    while telling every such user that a cookie they never stored is sitting on
    disk.
    """
    from local_operator.providers.auth_store import AuthStore
    from local_operator.secrets import access

    base = tmp_path / "config"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(base))
    # A store with SOMETHING in it -- so the probe reaches `describe` and meets
    # `SecretNotFound` rather than returning at the existence guard.
    other = access.open_store(base, create=True)
    other.initialize()
    other.set("SOME_OTHER_KEY", b"unrelated-value")

    opened = AuthStore(db_path=tmp_path / "auth.db")
    try:
        assert cli._qwencloud_ticket_action("status", opened) == 0
        captured = capsys.readouterr()
    finally:
        opened.close()

    assert "No QwenCloud console ticket stored." in captured.out
    assert "VALUE is stored" not in captured.out


def test_status_on_a_locked_store_does_not_call_a_secret_orphan_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The probe belongs INSIDE `status`'s `TicketStoreError` clause.

    `_secret_is_present` raises `TicketStoreLocked`/`TicketStoreUnreadable` for
    a store it cannot read, and both subclass `TicketStoreError`. Called
    outside the `try`, a locked store would surface a traceback out of the verb
    instead of the UNKNOWN receipt and its exit code -- the same false success
    `test_status_on_a_locked_store_is_not_no_ticket_stored` guards, reached from
    the branch this fix adds.

    Fails against TWO shapes: the pre-fix code (no probe, so the no-ticket
    receipt and exit 0), and the probe hoisted out of the `try` (an uncaught
    `TicketStoreLocked`). The `describe` refusal is stubbed rather than hardened
    for real because a genuinely locked store needs a broker daemon and a
    passphrase; the store FILE exists either way, which is what the probe's
    existence guard reads.
    """
    from local_operator.providers.auth_store import AuthStore
    from local_operator.secrets import access

    base = tmp_path / "config"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(base))
    access.open_store(base, create=True).initialize()

    class _Denied:
        """Every operation refused the way a hardened, locked store refuses."""

        def describe(self, name: str) -> Any:
            from local_operator.secrets.client import BrokerDenied

            raise BrokerDenied("no lop session is registered with the broker")

    monkeypatch.setattr(access, "open_store", lambda *a, **k: _Denied())

    opened = AuthStore(db_path=tmp_path / "auth.db")
    try:
        assert cli._qwencloud_ticket_action("status", opened) == 1
        captured = capsys.readouterr()
    finally:
        opened.close()

    assert "UNKNOWN" in captured.err
    assert "lop secret unlock" in captured.err
    assert "No QwenCloud console ticket stored" not in captured.out


def test_rm_failure_names_both_stores(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """B5: a generic unreadable store — the ticket may be in either store."""
    from local_operator.providers.qwencloud_console import TicketStoreError

    def unreadable(store: Any, **kwargs: Any) -> Any:
        raise TicketStoreError("the credential store could not be read (OSError)")

    _patch_qwen(monkeypatch, delete_ticket=unreadable)

    assert cli._qwencloud_ticket_action("rm", _QwenRowStore()) == 1
    err = capsys.readouterr().err
    assert "MAY STILL BE STORED" in err
    assert "secret store" in err


def test_rm_on_a_locked_store_says_unlock_and_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """B6: `rm`'s locked clause DOES discriminate — its body differs from the
    generic one ("Unlock the secret store and re-run" appears nowhere else),
    which is why `rm` carries a clause `status` does not."""

    def locked(store: Any, **kwargs: Any) -> Any:
        raise _locked_exc()

    _patch_qwen(monkeypatch, delete_ticket=locked)

    assert cli._qwencloud_ticket_action("rm", _QwenRowStore()) == 1
    err = capsys.readouterr().err
    assert "Unlock the secret store" in err
    assert "MAY STILL BE STORED" in err


def test_rm_success_still_warns_about_the_browser_session(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """B7: deleting the local copy does not end the server-side session."""
    _patch_qwen(monkeypatch, delete_ticket=lambda store, **kw: True)
    monkeypatch.setattr(
        "local_operator.providers.auth_cli._invalidate_cached_usage",
        lambda *a, **k: None,
    )

    assert cli._qwencloud_ticket_action("rm", _QwenRowStore()) == 0
    out = capsys.readouterr().out
    assert "Removed the stored QwenCloud console ticket." in out
    assert "still valid until you sign it out" in out


def test_set_refuses_a_tty(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """B8: argv is readable by any process running as you, so an interactive
    `set` must refuse rather than invite the value onto a command line."""
    writes: list[str] = []

    def must_not_write(store: Any, ticket: str, **kwargs: Any) -> None:
        writes.append(ticket)

    _patch_qwen(monkeypatch, store_ticket=must_not_write)

    fake_stdin = MagicMock()
    fake_stdin.isatty.return_value = True
    monkeypatch.setattr(sys, "stdin", fake_stdin)

    assert cli._qwencloud_ticket_action("set", _QwenRowStore()) == 2
    assert "printf %s '<TICKET>' | lop qwencloud-ticket set" in capsys.readouterr().err
    assert writes == [], "a refused set must store nothing"


def test_set_on_a_locked_store_prints_the_remedy_and_exits_one(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """B9: `set` needs no locked clause of its own — `TicketStoreLocked`
    subclasses `TicketStoreError`, so the existing clause catches it and
    interpolates the message carrying the remedy."""

    def locked(store: Any, ticket: str, **kwargs: Any) -> None:
        raise _locked_exc()

    _patch_qwen(monkeypatch, store_ticket=locked)

    fake_stdin = MagicMock()
    fake_stdin.isatty.return_value = False
    fake_stdin.read.return_value = FAKE_QWEN_TICKET
    monkeypatch.setattr(sys, "stdin", fake_stdin)

    assert cli._qwencloud_ticket_action("set", _QwenRowStore()) == 1
    assert "lop secret unlock" in capsys.readouterr().err


def test_set_never_echoes_the_value(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """B10: the receipt reports a LENGTH. The value reaching stdout would put
    it in the terminal scrollback and every session transcript on disk."""
    _patch_qwen(
        monkeypatch,
        store_ticket=lambda store, ticket, **kw: None,
        read_ticket_record=lambda store, **kw: {
            "credential_id": 1,
            "captured_at": int(time.time() * 1000),
            "length": len(FAKE_QWEN_TICKET),
            "secret_present": True,
        },
    )
    monkeypatch.setattr(
        "local_operator.providers.auth_cli._invalidate_cached_usage",
        lambda *a, **k: None,
    )

    fake_stdin = MagicMock()
    fake_stdin.isatty.return_value = False
    fake_stdin.read.return_value = FAKE_QWEN_TICKET
    monkeypatch.setattr(sys, "stdin", fake_stdin)

    assert cli._qwencloud_ticket_action("set", _QwenRowStore()) == 0
    out = capsys.readouterr()
    assert FAKE_QWEN_TICKET not in out.out
    assert FAKE_QWEN_TICKET not in out.err
    assert f"({len(FAKE_QWEN_TICKET)} characters)" in out.out


def test_unknown_verb_lists_migrate(capsys: pytest.CaptureFixture[str]) -> None:
    """B11: the usage footer is the discovery surface for the new verb."""
    assert cli._qwencloud_ticket_action(None, _QwenRowStore()) == 2
    assert "{set,status,rm,migrate}" in capsys.readouterr().err


def test_migrate_is_dispatched(monkeypatch: pytest.MonkeyPatch) -> None:
    """B12: the verb reaches its handler rather than falling through to usage."""
    seen: list[Any] = []
    monkeypatch.setattr(cli, "_qwencloud_ticket_migrate", lambda store: seen.append(store) or 7)

    store = _QwenRowStore()
    assert cli._qwencloud_ticket_action("migrate", store) == 7
    assert seen == [store]


def test_migrate_with_no_ticket_is_a_quiet_no_op(capsys: pytest.CaptureFixture[str]) -> None:
    """E1: nothing stored is a SUCCESS, not the stub's honest failure.

    Replaces `test_migrate_stub_fails_honestly` (slice B), which asserted the
    stub's exit 2 / "not yet available" and which slice E breaks by design.

    `_QwenRowStore` is reused rather than widened: it defines `list_credentials`
    and nothing else — no `_conn` — so it can only drive the one path that
    returns before the VACUUM step, which is exactly this one. Every other
    migrate test uses a real `AuthStore` in `tests/unit/test_cli_migrate.py`.

    The last two assertions are what discriminate against the stub: an exit
    code alone would not, since both a no-op and a refusal can be non-zero.
    """
    assert cli._qwencloud_ticket_migrate(_QwenRowStore()) == 0
    out = capsys.readouterr()
    assert "nothing to migrate" in out.out
    assert "not yet available" not in out.err


def test_migrate_is_registered_in_the_parser() -> None:
    """B14: parser wiring, so `lop qwencloud-ticket migrate` parses rather
    than erroring out at argparse."""
    parsed = cli.build_cli_parser().parse_args(["qwencloud-ticket", "migrate"])
    assert parsed.qwencloud_command == "migrate"
