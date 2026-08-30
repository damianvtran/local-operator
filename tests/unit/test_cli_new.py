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


def test_credential_update_ctrl_c_exits_130(tmp_home: Path, capsys) -> None:
    """Ctrl-C at the prompt is a cancel, not a crash: exit 130 (SIGINT
    convention), one quiet line, no stack-trace panel (item 4)."""
    manager = MagicMock()
    manager.prompt_for_credential.side_effect = KeyboardInterrupt
    with patch("local_operator.cli.CredentialManager", return_value=manager):
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
    with patch("local_operator.cli.CredentialManager", return_value=manager):
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
    with patch("local_operator.cli.CredentialManager", return_value=manager):
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
    config = types.SimpleNamespace(auth=types.SimpleNamespace(type="oauth"))
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


@pytest.mark.asyncio
async def test_mcp_reauth_removes_then_logs_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reauth must delete BEFORE the grant starts — a login over a surviving
    row would reuse the stored registration and never show the consent
    screen, which is the entire reason reauth exists."""
    calls: list[str] = []
    monkeypatch.setattr(
        "local_operator.mcp.auth.mcp_logout_server",
        lambda name, cwd: calls.append("logout") or None,
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
        lambda name, cwd: "MCP server 'linar' is not configured",
    )

    async def fake_login(name: str, cwd: Path) -> int:
        raise AssertionError("login must not start when removal failed")

    monkeypatch.setattr(cli, "_mcp_login_server", fake_login)
    assert await cli._mcp_reauth_server("linar", tmp_path) == 1
    assert "not configured" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("argv", "handler"),
    [
        (["login"], "login_command"),
        (["logout", "openai"], "logout_command"),
        (["login-status"], "login_status_command"),
        (["mcp", "list"], "mcp_command"),
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
        session_factory,
        theme_name: str = "dark",
        provider_controller=None,
        resume_factory=None,
        on_config_changed=None,
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

    fake_tui = types.ModuleType("local_operator.tui")

    async def fake_run_tui(
        session_factory,
        theme_name: str = "dark",
        provider_controller=None,
        resume_factory=None,
        on_config_changed=None,
    ) -> int:
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

    fake_tui = types.ModuleType("local_operator.tui")

    async def fake_run_tui(
        session_factory,
        theme_name: str = "dark",
        provider_controller=None,
        resume_factory=None,
        on_config_changed=None,
    ) -> int:
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

    fake_tui = types.ModuleType("local_operator.tui")

    async def fake_run_tui(
        session_factory,
        theme_name="dark",
        provider_controller=None,
        resume_factory=None,
        on_config_changed=None,
    ) -> int:
        await session_factory()
        return 0

    setattr(fake_tui, "run_tui", fake_run_tui)
    monkeypatch.setitem(sys.modules, "local_operator.tui", fake_tui)
    monkeypatch.setattr("local_operator.cli.create_session", fake_create_session)
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

    fake_tui = types.ModuleType("local_operator.tui")

    async def fake_run_tui(
        session_factory,
        theme_name="dark",
        provider_controller=None,
        resume_factory=None,
        on_config_changed=None,
    ) -> int:
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

    async def fake_run_tui(
        session_factory,
        theme_name="dark",
        provider_controller=None,
        resume_factory=None,
        on_config_changed=None,
    ) -> int:
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
    # called `hi` and leaves exec without a prompt. That exits 2 with a usage
    # message rather than doing something surprising, which is the acceptable end
    # of an ambiguity argparse cannot resolve for us.
    assert parser.parse_args(["exec", "hi", "--resume"]).resume == cli.RESUME_LATEST
    with pytest.raises(SystemExit):
        parser.parse_args(["exec", "--resume", "hi"])


def test_a_background_job_carries_the_session_it_was_told_to_resume() -> None:
    """`--background` is the same request run elsewhere.

    The flag was accepted by the front end and dropped at the process boundary:
    `build_worker_argv` serialized every other field, so the worker started a new
    session and reported success against the wrong history.
    """
    from local_operator.exec_mode import ExecArgs, build_worker_argv
    from local_operator.exec_worker import build_parser

    argv = build_worker_argv("hi", ExecArgs(resume="sess-abc123"))
    assert "--resume" in argv
    assert argv[argv.index("--resume") + 1] == "sess-abc123"

    # And the worker on the other side accepts what was serialized — parsed from
    # the real argv minus the `python -m <module>` prefix, so the test breaks if
    # the serialization and the worker's parser ever disagree.
    assert build_parser().parse_args(argv[3:]).resume == "sess-abc123"

    # Nothing is emitted when nothing was asked for.
    assert "--resume" not in build_worker_argv("hi", ExecArgs())


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
