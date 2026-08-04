"""Backward-compatibility and additive-surface tests for the rewritten CLI.

Scope: the parser contract (every legacy flag/subcommand/dest/default
survives; new flags are additive), the help surfaces from the acceptance
criteria, the lazy-import discipline, and ``main()`` dispatch for the new
subcommands. Engine-dependent flows are NOT exercised here — they live in
``test_exec_mode.py`` with fake sessions.

The rewrite venv has no langchain, so these tests never import
``local_operator.operator`` / ``executor`` (the legacy test_cli.py does and
cannot run in this venv; its expectations for the removed operator flow are
obsolete by the rewrite contract).
"""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

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


@pytest.mark.parametrize(
    ("argv", "name", "agent_id"),
    [
        (["agents", "delete", "--name", "X"], "X", None),
        (["agents", "delete", "--id", "Y"], None, "Y"),
    ],
)
def test_agents_delete_dests(
    parser: argparse.ArgumentParser, argv: list[str], name: str | None, agent_id: str | None
) -> None:
    args = parser.parse_args(argv)
    assert args.name == name
    assert args.agent_id == agent_id


def test_agents_delete_requires_exclusive_choice(parser: argparse.ArgumentParser) -> None:
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
    assert args.host == "0.0.0.0"
    assert args.port == 1111
    assert args.reload is False
    args = parser.parse_args(["serve", "--host", "localhost", "--port", "8000", "--reload"])
    assert (args.host, args.port, args.reload) == ("localhost", 8000, True)


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

    args = parser.parse_args(["mcp", "remove", "files", "--scope", "project"])
    assert args.mcp_command == "remove"
    assert args.name == "files"
    assert args.scope == "project"


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
    manager = MagicMock()
    with patch("local_operator.cli.CredentialManager", return_value=manager):
        args = argparse.Namespace(key="TEST_API_KEY")
        assert credential_update_command(args) == 0
    manager.prompt_for_credential.assert_called_once_with("TEST_API_KEY", reason="update requested")


def test_credential_delete_command(tmp_home: Path) -> None:
    manager = MagicMock()
    with patch("local_operator.cli.CredentialManager", return_value=manager):
        args = argparse.Namespace(key="TEST_API_KEY")
        assert credential_delete_command(args) == 0
    manager.set_credential.assert_called_once_with("TEST_API_KEY", "")


def test_config_create_command(tmp_home: Path) -> None:
    manager = MagicMock()
    with patch("local_operator.cli.ConfigManager", return_value=manager):
        assert config_create_command() == 0
    manager._write_config.assert_called_once()


def test_serve_command_preserves_uvicorn_call() -> None:
    with patch("local_operator.cli.uvicorn.run") as mock_run:
        assert serve_command("localhost", 8000, False) == 0
    mock_run.assert_called_once_with(
        "local_operator.server.app:app", host="localhost", port=8000, reload=False
    )


def test_agents_list_command_empty() -> None:
    registry = MagicMock()
    registry.list_agents.return_value = []
    assert agents_list_command(argparse.Namespace(page=1, perpage=10), registry) == 0


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
    assert agents_delete_command(args, registry, Path(".")) == -1
    registry.delete_agent.assert_not_called()


# --- main() dispatch ------------------------------------------------------------


def test_main_exec_dispatch(
    tmp_home: Path, quiet_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """exec routes to exec_mode.run_exec with the parsed ExecArgs, and its
    exit code passes through (README contract)."""
    captured: dict = {}

    def fake_run_exec(command: str, exec_args) -> int:
        captured["command"] = command
        captured["args"] = exec_args
        return 7

    monkeypatch.setattr("local_operator.exec_mode.run_exec", fake_run_exec)
    monkeypatch.setattr(
        "sys.argv",
        ["program", "--yolo", "exec", "do the thing", "--json", "--agent-id", "a1"],
    )
    assert main() == 7
    assert captured["command"] == "do the thing"
    exec_args = captured["args"]
    assert exec_args.json_mode is True
    assert exec_args.agent_id == "a1"
    assert exec_args.yolo is True
    assert exec_args.background is False
    assert exec_args.agent_name is None


def test_main_exception_banner(tmp_home: Path, quiet_env: None, capsys) -> None:
    """Red-banner handling survives: any exception -> message + exit -1."""
    with patch("local_operator.cli.ConfigManager", side_effect=Exception("Test error")):
        with patch("sys.argv", ["program"]):
            assert main() == -1
    out = capsys.readouterr().out
    assert "Error: Test error" in out
    assert "Stack Trace" in out


def test_main_interactive_tty_uses_tui(
    tmp_home: Path, quiet_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """tty stdout + importable TUI -> asyncio.run(run_tui(session_factory));
    the factory hands the TUI the wired session (TUI-003 contract)."""
    sentinel_session = object()
    seen: dict = {}

    async def fake_create_session(*args, **kwargs):
        seen["factory_called"] = True
        return sentinel_session

    monkeypatch.setattr("local_operator.cli.create_session", fake_create_session)

    fake_tui = types.ModuleType("local_operator.tui")

    async def fake_run_tui(session_factory, theme_name: str = "dark") -> int:
        seen["theme"] = theme_name
        seen["session"] = await session_factory()
        return 0

    fake_tui.run_tui = fake_run_tui
    monkeypatch.setitem(sys.modules, "local_operator.tui", fake_tui)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr("local_operator.cli.ConfigManager", _fake_config_manager)
    monkeypatch.setattr("local_operator.cli.CredentialManager", MagicMock())
    monkeypatch.setattr("local_operator.agents.AgentRegistry", MagicMock())

    with patch("sys.argv", ["program", "--hosting", "openai", "--model", "gpt-4o"]):
        assert main() == 0
    assert seen["factory_called"] is True
    assert seen["session"] is sentinel_session
    assert seen["theme"] == "dark"


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

    with patch("sys.argv", ["program", "--no-tui", "--hosting", "openai", "--model", "m"]):
        assert main() == 0
    assert fake_session.dispose.await_count == 1


def _raise_eof(prompt: str = "") -> str:
    raise EOFError
