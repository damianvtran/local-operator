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
import subprocess
import sys
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


def test_serve_command_preserves_uvicorn_call() -> None:
    """uvicorn is imported lazily INSIDE serve_command (so `local-operator`
    starts without the server extra), so the patch target is the uvicorn
    module itself, not a `local_operator.cli.uvicorn` attribute that no
    longer exists."""
    with patch("uvicorn.run") as mock_run:
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


def test_main_exception_banner(tmp_home: Path, quiet_env: None, capsys) -> None:
    """Red-banner handling survives: any exception -> message + exit -1."""
    with patch("local_operator.cli.ConfigManager", side_effect=Exception("Test error")):
        with patch("sys.argv", ["program"]):
            assert main() == -1
    # STDERR: main() wraps the exec dispatch, so its error presenter must not
    # write to the `exec --json` data channel. Asserting the stream is the
    # point of the test now, not incidental.
    err = capsys.readouterr().err
    assert "Error: Test error" in err
    assert "Stack Trace" in err


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

    monkeypatch.setattr("local_operator.cli.create_session", fake_create_session)

    fake_tui = types.ModuleType("local_operator.tui")

    async def fake_run_tui(
        session_factory, theme_name: str = "dark", provider_controller=None
    ) -> int:
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


# --- CL-06: startup preflight ---------------------------------------------------


def test_main_preflight_missing_hosting(
    tmp_home: Path, quiet_env: None, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """Interactive startup with no hosting configured prints the legacy
    message shape and exits -1 BEFORE any turn (no session factory call)."""
    called: dict[str, bool] = {"factory": False}

    async def fake_create_session(*args, **kwargs):
        called["factory"] = True
        return MagicMock()

    monkeypatch.setattr("local_operator.cli.create_session", fake_create_session)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr("local_operator.cli.ConfigManager", _fake_config_manager)
    monkeypatch.setattr("local_operator.cli.CredentialManager", MagicMock())
    monkeypatch.setattr("local_operator.agents.AgentRegistry", MagicMock())

    with patch("sys.argv", ["program"]):
        assert main() == -1
    # stderr, matching its sibling _preflight_api_key: an error message belongs
    # on the diagnostic channel regardless of which front end asked for it.
    err = capsys.readouterr().err
    assert "Hosting platform is not configured." in err
    assert called["factory"] is False


def test_main_preflight_missing_api_key(
    tmp_home: Path,
    quiet_env: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    """A keyed provider with NO resolvable key fails preflight (-1) before the
    turn; keyless providers (test) pass through."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    called: dict[str, bool] = {"factory": False}

    async def fake_create_session(*args, **kwargs):
        called["factory"] = True
        return MagicMock()

    monkeypatch.setattr("local_operator.cli.create_session", fake_create_session)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr("local_operator.cli.ConfigManager", _fake_config_manager)
    monkeypatch.setattr("local_operator.cli.CredentialManager", _bare_credential_manager)
    monkeypatch.setattr("local_operator.agents.AgentRegistry", MagicMock())

    with patch("sys.argv", ["program", "--hosting", "openai", "--model", "gpt-4o"]):
        assert main() == -1
    # stderr: this is the most common `exec --json` failure there is (fresh
    # install, or a typo'd --hosting), so a coloured line on stdout broke the
    # consumer at exactly the moment it needed to read the error.
    err = capsys.readouterr().err
    assert "OPENAI_API_KEY" in err and "Error" in err
    assert called["factory"] is False


def test_main_preflight_env_key_passes(
    tmp_home: Path, quiet_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the env key present the preflight lets the session through."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    seen: dict[str, Any] = {}

    async def fake_create_session(*args, **kwargs):
        seen["built"] = True
        return MagicMock()

    fake_tui = types.ModuleType("local_operator.tui")

    async def fake_run_tui(session_factory, theme_name="dark", provider_controller=None) -> int:
        seen.setdefault("provider_controller", provider_controller)
        await session_factory()
        return 0

    setattr(fake_tui, "run_tui", fake_run_tui)
    monkeypatch.setitem(sys.modules, "local_operator.tui", fake_tui)
    monkeypatch.setattr("local_operator.cli.create_session", fake_create_session)
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

    fake_tui = types.ModuleType("local_operator.tui")

    async def fake_run_tui(session_factory, theme_name="dark", provider_controller=None) -> int:
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
            for field in ("dest", "default", "required", "nargs"):
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
