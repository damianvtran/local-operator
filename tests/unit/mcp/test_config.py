"""Config discovery, priority/dedupe, disable resolution, validation, CLI helpers."""

from __future__ import annotations

import json
import textwrap
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from local_operator.mcp.config import (
    MCPConfigWriteError,
    MCPHttpServerConfig,
    MCPServerConfig,
    MCPStdioServerConfig,
    add_server,
    list_effective_servers,
    load_all_mcp_configs,
    owned_scope_for_source,
    read_disabled_servers,
    read_enabled_servers,
    remove_server,
    validate_server_config,
)


@pytest.fixture()
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect ``Path.home()`` into the test sandbox."""
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home_dir))
    return home_dir


def _write(path: Path, doc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf-8")


def _write_toml(path: Path, body: str) -> None:
    """Write a Codex-shaped TOML config, dedented so tests can stay indented."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body), encoding="utf-8")


def _stdio(command: str = "npx", **extra: Any) -> dict[str, Any]:
    return {"type": "stdio", "command": command, **extra}


def _command(configs: Mapping[str, MCPServerConfig], name: str) -> str:
    """Narrow a loaded config to its stdio shape and return the command."""
    cfg = configs[name]
    assert isinstance(cfg, MCPStdioServerConfig)
    return cfg.command


class TestPriorityAndDedupe:
    def test_project_wins_over_user_and_imports(self, tmp_path: Path, home: Path) -> None:
        """First source to define a name wins; later sources never override."""
        cwd = tmp_path / "proj"
        _write(
            cwd / ".local-operator" / "mcp.json",
            {"mcpServers": {"srv": _stdio("proj-cmd")}},
        )
        _write(cwd / ".mcp.json", {"mcpServers": {"srv": _stdio("dot-cmd")}})
        _write(
            home / ".local-operator" / "mcp.json",
            {"mcpServers": {"srv": _stdio("user-cmd")}},
        )
        _write(
            home / ".claude.json",
            {"mcpServers": {"srv": _stdio("claude-cmd"), "claudeonly": _stdio("cc")}},
        )

        configs, sources = load_all_mcp_configs(cwd)
        assert _command(configs, "srv") == "proj-cmd"
        assert sources["srv"].endswith(".local-operator/mcp.json")
        assert _command(configs, "claudeonly") == "cc"

    def test_dot_mcp_json_beats_user(self, tmp_path: Path, home: Path) -> None:
        cwd = tmp_path / "proj"
        _write(cwd / ".mcp.json", {"mcpServers": {"srv": _stdio("dot-cmd")}})
        _write(
            home / ".local-operator" / "mcp.json",
            {"mcpServers": {"srv": _stdio("user-cmd")}},
        )

        configs, _ = load_all_mcp_configs(cwd)
        assert _command(configs, "srv") == "dot-cmd"

    def test_user_beats_foreign_imports(self, tmp_path: Path, home: Path) -> None:
        cwd = tmp_path / "proj"
        _write(
            home / ".local-operator" / "mcp.json",
            {"mcpServers": {"srv": _stdio("user-cmd")}},
        )
        _write(home / ".cursor" / "mcp.json", {"mcpServers": {"srv": _stdio("cursor-cmd")}})
        _write(
            cwd / ".vscode" / "mcp.json",
            {"mcp": {"servers": {"srv": _stdio("vscode-cmd"), "vsc": _stdio("v")}}},
        )

        configs, sources = load_all_mcp_configs(cwd)
        assert _command(configs, "srv") == "user-cmd"
        assert _command(configs, "vsc") == "v"
        assert sources["vsc"].endswith(".vscode/mcp.json")

    def test_claude_project_mcp_json_imported(self, tmp_path: Path, home: Path) -> None:
        """``.claude/.mcp.json`` (project Claude config) is a best-effort import."""
        cwd = tmp_path / "proj"
        _write(cwd / ".claude" / ".mcp.json", {"mcpServers": {"claudy": _stdio("cl")}})
        configs, sources = load_all_mcp_configs(cwd)
        assert _command(configs, "claudy") == "cl"
        assert sources["claudy"].endswith(".mcp.json")

    def test_claude_json_project_scope_imported(self, tmp_path: Path, home: Path) -> None:
        """MCP-18: ~/.claude.json projects.<cwd>.mcpServers is read too."""
        cwd = tmp_path / "proj"
        cwd.mkdir(parents=True)
        _write(
            home / ".claude.json",
            {
                "mcpServers": {"global_srv": _stdio("g-cmd")},
                "projects": {
                    str(cwd): {"mcpServers": {"proj_srv": _stdio("p-cmd")}},
                    "/some/other/path": {"mcpServers": {"other": _stdio("nope")}},
                },
            },
        )
        configs, sources = load_all_mcp_configs(cwd)
        assert _command(configs, "global_srv") == "g-cmd"
        assert _command(configs, "proj_srv") == "p-cmd"
        assert "other" not in configs  # wrong project key ignored
        assert sources["proj_srv"].endswith(".claude.json")

    def test_claude_project_scope_wins_within_file(self, tmp_path: Path, home: Path) -> None:
        """Within ~/.claude.json, project scope overrides the global key."""
        cwd = tmp_path / "proj"
        cwd.mkdir(parents=True)
        _write(
            home / ".claude.json",
            {
                "mcpServers": {"srv": _stdio("global-cmd")},
                "projects": {str(cwd): {"mcpServers": {"srv": _stdio("scoped-cmd")}}},
            },
        )
        configs, _ = load_all_mcp_configs(cwd)
        assert _command(configs, "srv") == "scoped-cmd"

    def test_claude_json_malformed_projects_degrades(self, tmp_path: Path, home: Path) -> None:
        """Best-effort: a misshaped projects key falls back to global only."""
        cwd = tmp_path / "proj"
        _write(home / ".claude.json", {"mcpServers": {"g": _stdio("g")}, "projects": 42})
        configs, _ = load_all_mcp_configs(cwd)
        assert _command(configs, "g") == "g"


class TestEnableDisable:
    def test_disabled_wins_over_enabled_and_flag(self, tmp_path: Path, home: Path) -> None:
        cwd = tmp_path / "proj"
        _write(
            cwd / ".local-operator" / "mcp.json",
            {
                "mcpServers": {
                    "a": _stdio("a"),
                    "b": _stdio("b"),
                    "c": {**_stdio("c"), "enabled": False},
                    "d": {**_stdio("d"), "enabled": False},
                },
                "disabledServers": ["a", "c"],
                "enabledServers": ["a", "d"],  # a stays dead: denylist wins
            },
        )
        configs, _ = load_all_mcp_configs(cwd)
        assert set(configs) == {
            "b",
            "d",
        }  # a: disabled wins; c: flag; d: allowlist revives

    def test_enabled_false_suppresses_and_keeps_name_owned(
        self, tmp_path: Path, home: Path
    ) -> None:
        """A disabled entry is dropped (suppressed): it must not shadow."""
        cwd = tmp_path / "proj"
        _write(
            cwd / ".local-operator" / "mcp.json",
            {"mcpServers": {"x": {**_stdio("x"), "enabled": False}}},
        )
        configs, _ = load_all_mcp_configs(cwd)
        assert "x" not in configs

    def test_disable_list_from_user_file_applies(self, tmp_path: Path, home: Path) -> None:
        cwd = tmp_path / "proj"
        _write(cwd / ".local-operator" / "mcp.json", {"mcpServers": {"a": _stdio("a")}})
        _write(home / ".local-operator" / "mcp.json", {"disabledServers": ["a"]})
        configs, _ = load_all_mcp_configs(cwd)
        assert configs == {}
        assert read_disabled_servers(cwd) == ["a"]
        assert read_enabled_servers(cwd) == []


class TestShapesAndImports:
    def test_type_inference(self, tmp_path: Path, home: Path) -> None:
        cwd = tmp_path / "proj"
        _write(
            cwd / ".local-operator" / "mcp.json",
            {
                "mcpServers": {
                    "inferred_stdio": {"command": "uvx", "args": ["pkg"]},
                    "inferred_http": {"url": "https://example.com/mcp"},
                    "remote_sse": {
                        "type": "sse",
                        "url": "https://example.com/sse",
                        "headers": {"x": "1"},
                    },
                }
            },
        )
        configs, _ = load_all_mcp_configs(cwd)
        assert isinstance(configs["inferred_stdio"], MCPStdioServerConfig)
        assert configs["inferred_stdio"].args == ["pkg"]
        assert isinstance(configs["inferred_http"], MCPHttpServerConfig)
        assert configs["remote_sse"].type == "sse"
        assert configs["remote_sse"].headers == {"x": "1"}

    def test_http_config_fields(self, tmp_path: Path, home: Path) -> None:
        cwd = tmp_path / "proj"
        _write(
            cwd / ".local-operator" / "mcp.json",
            {
                "mcpServers": {
                    "remote": {
                        "type": "http",
                        "url": "https://example.com/mcp",
                        "headers": {"Authorization": "Bearer x"},
                        "timeout": 5000,
                        "auth": {
                            "type": "oauth",
                            "token_url": "https://example.com/token",
                        },
                        "oauth": {"callback_port": 4000, "callback_path": "/cb"},
                    }
                }
            },
        )
        configs, _ = load_all_mcp_configs(cwd)
        cfg = configs["remote"]
        assert cfg.timeout == 5000
        assert cfg.auth is not None and cfg.auth.type == "oauth"
        assert cfg.oauth is not None and cfg.oauth.callback_port == 4000

    def test_malformed_file_and_entries_ignored(self, tmp_path: Path, home: Path) -> None:
        cwd = tmp_path / "proj"
        (cwd / ".local-operator").mkdir(parents=True)
        (cwd / ".local-operator" / "mcp.json").write_text("{not json", encoding="utf-8")
        _write(cwd / ".mcp.json", {"mcpServers": {"good": _stdio("g"), "bad": 42}})
        configs, _ = load_all_mcp_configs(cwd)
        assert set(configs) == {"good"}

    def test_no_configs_anywhere(self, tmp_path: Path, home: Path) -> None:
        configs, sources = load_all_mcp_configs(tmp_path)
        assert configs == {}
        assert sources == {}


class TestCodexImport:
    """``~/.codex/config.toml`` (issue #367): the one non-JSON import source."""

    def test_stdio_and_remote_servers_imported_with_provenance(
        self, tmp_path: Path, home: Path
    ) -> None:
        """Codex ``[mcp_servers.<name>]`` tables map onto both transports.

        ``command``/``args``/``env`` and ``url`` are exactly what
        ``_coerce_server_config`` already infers from, so no Codex-specific
        transport handling should be needed.
        """
        cwd = tmp_path / "proj"
        cwd.mkdir(parents=True)
        _write_toml(
            home / ".codex" / "config.toml",
            """
            [mcp_servers.node_repl]
            command = "node_repl"
            args = ["--stdio"]
            env = { NODE_PATH = "/opt/node" }

            [mcp_servers.docs]
            url = "https://developers.example.com/mcp"
            """,
        )
        configs, sources = load_all_mcp_configs(cwd)
        stdio = configs["node_repl"]
        assert isinstance(stdio, MCPStdioServerConfig)
        assert stdio.command == "node_repl"
        assert stdio.args == ["--stdio"]
        assert stdio.env == {"NODE_PATH": "/opt/node"}
        remote = configs["docs"]
        assert isinstance(remote, MCPHttpServerConfig)
        assert remote.url == "https://developers.example.com/mcp"
        assert sources["node_repl"].endswith(".codex/config.toml")
        assert sources["docs"].endswith(".codex/config.toml")

    def test_codex_never_overrides_any_earlier_source(self, tmp_path: Path, home: Path) -> None:
        """Codex is APPENDED LAST, so first-seen-wins makes it lose every tie.

        The position is the whole point of the ordering decision on #367: it
        can override neither local-operator's own files nor the other imports.
        """
        cwd = tmp_path / "proj"
        _write(
            home / ".local-operator" / "mcp.json",
            {"mcpServers": {"lop_srv": _stdio("lop-cmd")}},
        )
        _write(home / ".claude.json", {"mcpServers": {"claude_srv": _stdio("claude-cmd")}})
        _write(home / ".cursor" / "mcp.json", {"mcpServers": {"cursor_srv": _stdio("cursor-cmd")}})
        _write(
            cwd / ".vscode" / "mcp.json",
            {"mcp": {"servers": {"vscode_srv": _stdio("vscode-cmd")}}},
        )
        _write_toml(
            home / ".codex" / "config.toml",
            """
            [mcp_servers.lop_srv]
            command = "codex-cmd"

            [mcp_servers.claude_srv]
            command = "codex-cmd"

            [mcp_servers.cursor_srv]
            command = "codex-cmd"

            [mcp_servers.vscode_srv]
            command = "codex-cmd"

            [mcp_servers.codex_only]
            command = "codex-cmd"
            """,
        )
        configs, sources = load_all_mcp_configs(cwd)
        assert _command(configs, "lop_srv") == "lop-cmd"
        assert _command(configs, "claude_srv") == "claude-cmd"
        assert _command(configs, "cursor_srv") == "cursor-cmd"
        assert _command(configs, "vscode_srv") == "vscode-cmd"
        # Only the name no other tool claimed comes from Codex.
        assert _command(configs, "codex_only") == "codex-cmd"
        assert sources["codex_only"].endswith(".codex/config.toml")

    def test_malformed_toml_degrades_to_no_servers(self, tmp_path: Path, home: Path) -> None:
        """Best-effort, exactly like ``_read_json``: a broken foreign config
        must never break discovery of the files we DO own."""
        cwd = tmp_path / "proj"
        _write(
            home / ".local-operator" / "mcp.json",
            {"mcpServers": {"ours": _stdio("ours-cmd")}},
        )
        path = home / ".codex" / "config.toml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("[mcp_servers.broken\ncommand = ", encoding="utf-8")
        configs, _ = load_all_mcp_configs(cwd)
        assert set(configs) == {"ours"}

    def test_non_dict_mcp_servers_shapes_degrade(self, tmp_path: Path, home: Path) -> None:
        """The shapes a HAND-EDITED TOML actually produces: a scalar where the
        table should be, and a scalar entry inside it. Both degrade to
        contributing nothing while our own file still loads."""
        cwd = tmp_path / "proj"
        _write(
            home / ".local-operator" / "mcp.json",
            {"mcpServers": {"ours": _stdio("ours-cmd")}},
        )
        _write_toml(home / ".codex" / "config.toml", 'mcp_servers = "nope"\n')
        configs, _ = load_all_mcp_configs(cwd)
        assert set(configs) == {"ours"}

        _write_toml(home / ".codex" / "config.toml", "[mcp_servers]\nbroken = 1\n")
        configs, _ = load_all_mcp_configs(cwd)
        assert set(configs) == {"ours"}

    def test_deeply_nested_toml_degrades_instead_of_raising(
        self, tmp_path: Path, home: Path
    ) -> None:
        """``tomllib`` is a recursive-descent parser in Python, so deep inline
        nesting raises ``RecursionError`` where ``json.loads`` (C) does not.
        The reader promises every failure degrades to nothing, so this must not
        escape ``load_all_mcp_configs`` (review round 1, F2)."""
        cwd = tmp_path / "proj"
        _write(
            home / ".local-operator" / "mcp.json",
            {"mcpServers": {"ours": _stdio("ours-cmd")}},
        )
        depth = 2000
        _write_toml(
            home / ".codex" / "config.toml",
            '[mcp_servers.x]\ncommand = "c"\nnested = '
            + "{ a = " * depth
            + "1"
            + " }" * depth
            + "\n",
        )
        configs, _ = load_all_mcp_configs(cwd)
        assert set(configs) == {"ours"}

    def test_missing_codex_file_is_a_no_op(self, tmp_path: Path, home: Path) -> None:
        cwd = tmp_path / "proj"
        assert not (home / ".codex").exists()
        configs, sources = load_all_mcp_configs(cwd)
        assert configs == {}
        assert sources == {}

    def test_unmodelled_codex_keys_still_load(self, tmp_path: Path, home: Path) -> None:
        """Codex carries keys we do not model (``startup_timeout_sec``,
        ``cwd``). ``extra="allow"`` must keep the entry loadable rather than
        rejecting the server for a field that is simply not ours."""
        cwd = tmp_path / "proj"
        _write_toml(
            home / ".codex" / "config.toml",
            """
            [mcp_servers.slow]
            command = "slow-server"
            startup_timeout_sec = 120
            cwd = "/tmp/work"
            """,
        )
        configs, _ = load_all_mcp_configs(cwd)
        cfg = configs["slow"]
        assert isinstance(cfg, MCPStdioServerConfig)
        assert cfg.command == "slow-server"
        assert cfg.cwd == "/tmp/work"
        # The unmodelled key survives round-tripping rather than being dropped.
        assert cfg.model_dump()["startup_timeout_sec"] == 120

    def test_codex_enabled_false_suppresses(self, tmp_path: Path, home: Path) -> None:
        """``enabled`` IS modelled, so a server the user disabled in Codex
        stays disabled here — importing a config means importing its opinion
        about what should run."""
        cwd = tmp_path / "proj"
        _write_toml(
            home / ".codex" / "config.toml",
            """
            [mcp_servers.on]
            command = "on-cmd"

            [mcp_servers.off]
            command = "off-cmd"
            enabled = false
            """,
        )
        configs, _ = load_all_mcp_configs(cwd)
        assert set(configs) == {"on"}

    def test_local_operator_disabled_list_suppresses_a_codex_server(
        self, tmp_path: Path, home: Path
    ) -> None:
        """The enable/disable lists are format-agnostic: they come from the
        local-operator files and apply to every source, Codex included."""
        cwd = tmp_path / "proj"
        _write(home / ".local-operator" / "mcp.json", {"disabledServers": ["noisy"]})
        _write_toml(
            home / ".codex" / "config.toml",
            """
            [mcp_servers.noisy]
            command = "noisy-cmd"

            [mcp_servers.quiet]
            command = "quiet-cmd"
            """,
        )
        configs, _ = load_all_mcp_configs(cwd)
        assert set(configs) == {"quiet"}

    def test_codex_source_is_not_an_owned_write_scope(self, tmp_path: Path, home: Path) -> None:
        """``tomllib`` cannot write, so a Codex source can never resolve to a
        scope ``remove_server`` would edit. This is what makes the TUI's
        ``/mcp remove`` refusal correct rather than merely conservative."""
        cwd = tmp_path / "proj"
        _write_toml(
            home / ".codex" / "config.toml",
            """
            [mcp_servers.codexy]
            command = "codex-cmd"
            """,
        )
        _configs, sources = load_all_mcp_configs(cwd)
        assert owned_scope_for_source(sources["codexy"], cwd) is None


class TestValidation:
    def test_valid_stdio(self) -> None:
        assert validate_server_config("srv", MCPStdioServerConfig(command="npx")) == []

    def test_invalid_name(self) -> None:
        errors = validate_server_config("bad name!", MCPStdioServerConfig(command="x"))
        assert any("invalid server name" in e for e in errors)
        assert validate_server_config("x" * 101, MCPStdioServerConfig(command="x"))
        assert validate_server_config("", MCPStdioServerConfig(command="x"))
        assert validate_server_config("A.b-c:d_9", MCPStdioServerConfig(command="x")) == []

    def test_stdio_missing_command(self) -> None:
        errors = validate_server_config("srv", MCPStdioServerConfig())
        assert any("command" in e for e in errors)

    def test_http_missing_and_bad_url(self) -> None:
        assert any("url" in e for e in validate_server_config("srv", MCPHttpServerConfig()))
        errors = validate_server_config("srv", MCPHttpServerConfig(url="ftp://x"))
        assert any("http(s)" in e for e in errors)

    def test_bad_timeout_and_port(self) -> None:
        errors = validate_server_config("srv", MCPStdioServerConfig(command="x", timeout=-1))
        assert any("timeout" in e for e in errors)

    def test_non_model_config_reports_invalid(self) -> None:
        errors = validate_server_config("srv", {"nonsense": True})
        assert any("invalid config" in e for e in errors)


class TestCliHelpers:
    def test_add_remove_roundtrip_project_scope(self, tmp_path: Path, home: Path) -> None:
        cwd = tmp_path / "proj"
        cwd.mkdir()
        path = cwd / ".local-operator" / "mcp.json"
        # The writers return the PATH they wrote, so a caller can name the file
        # in its receipt rather than restating an invisible scope default.
        assert (
            add_server(
                "srv",
                command="npx",
                args=["-y", "pkg"],
                env={"K": "V"},
                scope="project",
                cwd=cwd,
            )
            == path
        )
        doc = json.loads(path.read_text())
        assert doc["mcpServers"]["srv"] == {
            "type": "stdio",
            "command": "npx",
            "args": ["-y", "pkg"],
            "env": {"K": "V"},
        }
        # Every refusal RAISES rather than printing to stderr and returning an
        # exit code: the TUI calls these same writers from inside a Textual
        # screen, where a print would corrupt the frame underneath it.
        with pytest.raises(MCPConfigWriteError):  # duplicate
            add_server("srv", command="x", scope="project", cwd=cwd)
        with pytest.raises(MCPConfigWriteError):  # both command and url
            add_server("u", command="x", url="http://x", scope="project", cwd=cwd)
        with pytest.raises(MCPConfigWriteError):  # neither
            add_server("u", scope="project", cwd=cwd)
        with pytest.raises(MCPConfigWriteError):  # invalid name
            add_server("bad name!", command="x", scope="project", cwd=cwd)
        # Remove roundtrip.
        assert remove_server("srv", scope="project", cwd=cwd) == path
        assert json.loads(path.read_text())["mcpServers"] == {}
        with pytest.raises(MCPConfigWriteError):
            remove_server("srv", scope="project", cwd=cwd)

    def test_write_error_carries_every_reason_for_the_cli_to_print(self, tmp_path: Path) -> None:
        """The CLI prints one ``error:`` line per problem, so the exception has
        to keep them as a LIST — collapsing validation output into one string
        here would silently change what a user (or a script) reads."""
        with pytest.raises(MCPConfigWriteError) as excinfo:
            add_server("bad name!", url="ftp://nope", scope="project", cwd=tmp_path)
        assert len(excinfo.value.errors) > 1
        # str() joins them for the single-line callers (the TUI notice).
        assert all(error in str(excinfo.value) for error in excinfo.value.errors)

    def test_owned_scope_only_claims_files_local_operator_writes(
        self, tmp_path: Path, home: Path
    ) -> None:
        """The gate behind ``/mcp remove``'s refusal. ``load_all_mcp_configs``
        merges eight sources but ``_scope_path`` writes exactly two, so every
        other source must come back unowned — deleting from one would either
        fail or shadow a config the user still maintains in another tool."""
        cwd = tmp_path / "owned-proj"
        cwd.mkdir()
        assert owned_scope_for_source(home / ".local-operator" / "mcp.json", cwd) == "global"
        assert owned_scope_for_source(cwd / ".local-operator" / "mcp.json", cwd) == "project"
        # Read by the loader, never written by _scope_path.
        assert owned_scope_for_source(cwd / ".mcp.json", cwd) is None
        assert owned_scope_for_source(home / ".claude.json", cwd) is None
        assert owned_scope_for_source(home / ".cursor" / "mcp.json", cwd) is None
        # tomllib is read-only, so a Codex source can never be removed in place.
        assert owned_scope_for_source(home / ".codex" / "config.toml", cwd) is None
        assert owned_scope_for_source(None, cwd) is None

    def test_owned_scope_compares_resolved_paths_not_strings(
        self, tmp_path: Path, home: Path
    ) -> None:
        """A string compare would call an owned file foreign the moment the
        path reached it by a symlink or an unnormalised prefix (macOS hands out
        both /var and /private/var for the same directory)."""
        cwd = tmp_path / "resolved-proj"
        (cwd / "sub").mkdir(parents=True)
        link = tmp_path / "link-home"
        link.symlink_to(home)
        assert owned_scope_for_source(link / ".local-operator" / "mcp.json", cwd) == "global"
        unnormalised = cwd / "sub" / ".." / ".local-operator" / "mcp.json"
        assert owned_scope_for_source(unnormalised, cwd) == "project"

    def test_add_url_server_global_scope(self, tmp_path: Path, home: Path) -> None:
        global_path = home / ".local-operator" / "mcp.json"
        assert add_server("remote", url="https://example.com/mcp", headers={"a": "b"}) == (
            global_path
        )
        doc = json.loads(global_path.read_text())
        assert doc["mcpServers"]["remote"]["type"] == "http"
        assert doc["mcpServers"]["remote"]["headers"] == {"a": "b"}
        assert remove_server("remote") == global_path

    def test_add_oauth_url_server(self, tmp_path: Path, home: Path) -> None:
        assert add_server("linear", url="https://mcp.linear.app/mcp", oauth=True) is not None
        doc = json.loads((home / ".local-operator" / "mcp.json").read_text())
        assert doc["mcpServers"]["linear"] == {
            "type": "http",
            "url": "https://mcp.linear.app/mcp",
            "auth": {"type": "oauth"},
        }
        with pytest.raises(MCPConfigWriteError):
            add_server("stdio-oauth", command="npx", oauth=True)

    def test_list_effective_servers(self, tmp_path: Path, home: Path) -> None:
        cwd = tmp_path / "proj"
        _write(
            cwd / ".local-operator" / "mcp.json",
            {
                "mcpServers": {
                    "a": _stdio("a"),
                    "off": {**_stdio("off"), "enabled": False},
                }
            },
        )
        listed = list_effective_servers(cwd)
        assert set(listed) == {"a"}
        assert listed["a"]["command"] == "a"

    def test_write_json_atomic_leaves_valid_file(self, tmp_path: Path) -> None:
        """MCP-15: the writer is tempfile + os.replace — no .tmp leftovers,
        and the target holds exactly the written doc."""
        from local_operator.mcp.config import _write_json_atomic

        path = tmp_path / "sub" / "mcp.json"
        doc = {"mcpServers": {"a": _stdio("a")}, "disabledServers": []}
        _write_json_atomic(path, doc)
        assert json.loads(path.read_text(encoding="utf-8")) == doc
        # No temp files survive in the target directory.
        leftovers = [p for p in path.parent.iterdir() if p.name.endswith(".tmp")]
        assert leftovers == []
        # Overwrite replaces content atomically (reader sees whole docs only).
        doc2 = {"mcpServers": {"b": _stdio("b")}}
        _write_json_atomic(path, doc2)
        assert json.loads(path.read_text(encoding="utf-8")) == doc2


def test_server_tool_filters_parse_aliases_for_every_transport() -> None:
    from local_operator.mcp.config import (
        MCPHttpServerConfig,
        MCPSseServerConfig,
        MCPStdioServerConfig,
    )

    payloads = [
        (MCPStdioServerConfig, {"command": "x"}),
        (MCPHttpServerConfig, {"url": "https://x.test"}),
        (MCPSseServerConfig, {"url": "https://x.test/sse"}),
    ]
    for cls, base in payloads:
        cfg = cls.model_validate(
            {
                **base,
                "enabledTools": ["search_*", "get_one"],
                "disabledTools": ["search_private"],
            }
        )
        assert cfg.enabled_tools == ["search_*", "get_one"]
        assert cfg.disabled_tools == ["search_private"]
        dumped = cfg.model_dump(by_alias=True)
        assert dumped["enabledTools"] == ["search_*", "get_one"]
        assert dumped["disabledTools"] == ["search_private"]


def test_the_user_scope_write_honours_an_isolated_config_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sandboxed config dir must redirect the WRITE, not just the reads.

    `_scope_path` resolved the user scope with a bare `Path.home()`, so an
    isolated `LOCAL_OPERATOR_CONFIG_DIR` — the mechanism every test and every
    agent probe uses to stay off the real machine — did not cover `/mcp add`
    or `/mcp remove`. During round 5 that removed a real server from a real
    developer's live `~/.local-operator/mcp.json`.

    This is the same class as the LaunchAgent write and the desktop
    notifications: a location the code reaches by rebuilding a path itself
    instead of asking the one resolver that knows about the override.
    """
    from local_operator.mcp.config import _scope_path

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "sandbox"))
    target = _scope_path(None, "user")
    assert target == tmp_path / "sandbox" / "mcp.json"
    assert Path.home() not in target.parents, "a sandboxed write must not land under the real home"


def test_the_user_scope_still_defaults_to_the_config_dir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no override the target is unchanged — the fix must not move a
    real user's servers to a new file."""
    from local_operator.mcp.config import _scope_path

    monkeypatch.delenv("LOCAL_OPERATOR_CONFIG_DIR", raising=False)
    assert _scope_path(None, "user") == Path.home() / ".local-operator" / "mcp.json"


def test_a_sandboxed_write_is_read_back_by_the_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Writer and reader must resolve the same file, or a sandboxed write is
    invisible to the read that follows it and a test silently proves nothing."""
    import json

    from local_operator.mcp.config import _local_operator_file_paths, _scope_path

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "sandbox"))
    target = _scope_path(None, "user")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"mcpServers": {"probe": {"command": "true"}}}))

    assert target in _local_operator_file_paths(str(tmp_path))


class TestCodexToolTimeoutImport:
    """Codex's per-tool-call budget must survive the import.

    Codex is the one import source local-operator does not own, and it is the
    only one that carries a timeout for the servers in it. It spells that value
    in SECONDS where local-operator spells it in MILLISECONDS, so before this
    translation it parsed into the model's ``extra`` bucket and was dropped: the
    server imported, its tools worked, and the budget the user configured for it
    silently did not apply. That is the same silent-drop shape as a server that
    fails to import at all, and it is worse to notice because nothing errors.

    THIS IS ALSO HOW A DECLARED BUDGET REACHES THIS HARNESS. Callers that need a
    specific per-tool budget (Minerva's risk-assessment server does) declare
    ``tool_timeout_sec`` in the shared per-run MCP document they hand to every
    harness, so that they bind to a value the caller owns instead of to two
    different upstream defaults. The emitting side is pinned by the codex/lop
    harness parity check in agent-runtime-svc
    (``adapters/lopcli/harness_parity_test.go``, which asserts the literal key
    spelling); the consuming side is pinned here. A rename on either side fails
    one of them.
    """

    def test_tool_timeout_sec_becomes_the_client_timeout(self, tmp_path: Path, home: Path) -> None:
        cwd = tmp_path / "proj"
        cwd.mkdir(parents=True)
        _write_toml(
            home / ".codex" / "config.toml",
            """
            [mcp_servers.slow]
            command = "slow-server"
            tool_timeout_sec = 300
            """,
        )
        configs, _ = load_all_mcp_configs(cwd)
        cfg = configs["slow"]
        assert isinstance(cfg, MCPStdioServerConfig)
        # Seconds in, milliseconds out -- and the raw Codex key is left alone so
        # the document still round-trips the way it arrived.
        assert cfg.timeout == 300_000
        assert cfg.model_dump()["tool_timeout_sec"] == 300

    def test_explicit_timeout_wins_over_tool_timeout_sec(self, tmp_path: Path, home: Path) -> None:
        """A server declaring BOTH keeps its own unit's value.

        ``timeout`` is this tool's spelling and therefore the more specific
        statement of intent, so it must not be overwritten by a translated
        foreign one.
        """
        cwd = tmp_path / "proj"
        cwd.mkdir(parents=True)
        _write_toml(
            home / ".codex" / "config.toml",
            """
            [mcp_servers.both]
            command = "srv"
            timeout = 120000
            tool_timeout_sec = 300
            """,
        )
        configs, _ = load_all_mcp_configs(cwd)
        cfg = configs["both"]
        assert isinstance(cfg, MCPStdioServerConfig)
        assert cfg.timeout == 120_000

    def test_unusable_tool_timeout_sec_leaves_the_default_in_place(
        self, tmp_path: Path, home: Path
    ) -> None:
        """A value we cannot read must not become a guessed budget.

        ``timeout`` stays None so the resolver's own default applies; a foreign
        config must never be able to break discovery, and it must not be able to
        install a nonsense timeout either. A bool is rejected explicitly: it is
        an ``int`` subclass in Python, so ``True`` would otherwise become a 1 s
        budget -- the tightest possible one -- on a quiet typo.
        """
        cwd = tmp_path / "proj"
        cwd.mkdir(parents=True)
        _write_toml(
            home / ".codex" / "config.toml",
            """
            [mcp_servers.odd]
            command = "srv"
            tool_timeout_sec = "soon"
            """,
        )
        configs, _ = load_all_mcp_configs(cwd)
        cfg = configs["odd"]
        assert isinstance(cfg, MCPStdioServerConfig)
        assert cfg.timeout is None

    def test_a_bool_tool_timeout_sec_is_not_a_budget(self, tmp_path: Path, home: Path) -> None:
        """``True`` must not become a 1 s budget.

        ``bool`` is an ``int`` subclass in Python, so a naive numeric check accepts
        ``tool_timeout_sec = true`` and installs the TIGHTEST possible timeout --
        a one-second bound on every call to that server -- from what is far more
        likely a typo than an intent. This is the subtlest constraint in the
        translation, so it is tested directly rather than left to the string case.
        """
        cwd = tmp_path / "proj"
        cwd.mkdir(parents=True)
        _write_toml(
            home / ".codex" / "config.toml",
            """
            [mcp_servers.truthy]
            command = "srv"
            tool_timeout_sec = true
            """,
        )
        configs, _ = load_all_mcp_configs(cwd)
        cfg = configs["truthy"]
        assert isinstance(cfg, MCPStdioServerConfig)
        assert cfg.timeout is None, "a bool is not a duration; the default must stand"

    @pytest.mark.parametrize("value", [0, -1, -0.5])
    def test_non_positive_tool_timeout_sec_does_not_disable_the_bound(
        self, tmp_path: Path, home: Path, value: float
    ) -> None:
        """A foreign non-positive value must not switch the bound OFF.

        ``timeout = 0`` means "no client-side bound" in this tool's own vocabulary,
        so translating a foreign ``tool_timeout_sec = 0`` into it would let
        ``~/.codex/config.toml`` silently remove the guard on a server -- the
        opposite of what a small sane default is for. Codex's schema gives a
        timeout no such "off" meaning, so the only available readings here are
        "unset" and "someone's arithmetic", and neither should disable a guard.

        Reached through the RESOLVER, not just the parsed field: leaving
        ``timeout`` unset is only useful if the default then actually applies.
        """
        from local_operator.mcp.manager import resolve_mcp_timeout_s

        cwd = tmp_path / "proj"
        cwd.mkdir(parents=True)
        _write_toml(
            home / ".codex" / "config.toml",
            f"""
            [mcp_servers.zeroed]
            command = "srv"
            tool_timeout_sec = {value}
            """,
        )
        configs, _ = load_all_mcp_configs(cwd)
        cfg = configs["zeroed"]
        assert isinstance(cfg, MCPStdioServerConfig)
        assert cfg.timeout is None
        assert (
            resolve_mcp_timeout_s(cfg) is not None
        ), "a foreign 0 must not remove the bound; our own timeout=0 still does"

    def test_preload_tools_is_read_under_its_exact_rendered_spelling(
        self, tmp_path: Path, home: Path
    ) -> None:
        """The literal ``preload_tools`` key must reach the parsed config.

        THIS IS THE LINK THAT WOULD FAIL SILENTLY. ``extra="allow"`` means an
        unrecognised key is ACCEPTED AND IGNORED, so if this spelling ever drifted
        on either side -- the emitter or the field -- the flag would parse, the
        server would load, and nothing would preload. The tools would simply go
        back to being invisible, with no error anywhere. Callers declare the flag
        by writing this exact key into a config file, so the spelling is pinned
        end-to-end from the file rather than only on the field.
        """
        cwd = tmp_path / "proj"
        cwd.mkdir(parents=True)
        _write_toml(
            home / ".codex" / "config.toml",
            """
            [mcp_servers.declared]
            command = "srv"
            preload_tools = true
            """,
        )
        configs, _ = load_all_mcp_configs(cwd)
        cfg = configs["declared"]
        assert isinstance(cfg, MCPStdioServerConfig)
        assert cfg.preload_tools is True

        # And the default is the lazy path for everything that did not declare it.
        _write_toml(
            home / ".codex" / "config.toml",
            """
            [mcp_servers.silent]
            command = "srv"
            """,
        )
        configs, _ = load_all_mcp_configs(cwd)
        silent = configs["silent"]
        assert isinstance(silent, MCPStdioServerConfig)
        assert silent.preload_tools is False


class TestHeaderBinding:
    """``bind_header_secret``/``unbind_header_secret``: the ``add_key`` config half.

    These are the CONFIG layer's own guards. The desktop host refuses a header
    write for any row the catalog would not offer ``add_key`` on (an OAuth
    server, a server that already sends a header), which means most of what
    follows is unreachable through HTTP — kept, and pinned here, as the layer
    that owns the file it writes: the alternative is a second credential header
    in someone's config, and that is not a failure worth reaching for.
    """

    HEADER = "X-Api-Key"
    KEY_ID = "ACME_KEY"

    def _isolate(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        (tmp_path / "home").mkdir(exist_ok=True)
        monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
        return tmp_path / "config" / "mcp.json"

    def _write(self, path: Path, server: dict[str, Any], *, indent: int = 4) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps({"mcpServers": {"acme": server}}, indent=indent)
        path.write_text(text)
        return text

    @pytest.mark.parametrize(
        ("servers", "header", "key_id", "code"),
        [
            # A header name is case-INSENSITIVE, so this one is already set.
            ({"headers": {"X-API-KEY": "plain"}}, "x-api-key", "ACME_KEY", "invalid_config"),
            # The transport owns it: it would fight the SDK's own header.
            ({"headers": {}}, "content-type", "ACME_KEY", "invalid_config"),
            # Not a reference name the store could ever publish.
            ({"headers": {}}, "X-Api-Key", "has-dash", "invalid_config"),
        ],
    )
    def test_a_bind_is_refused_and_writes_nothing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        servers: dict[str, Any],
        header: str,
        key_id: str,
        code: str,
    ) -> None:
        from local_operator.mcp.config import bind_header_secret

        global_file = self._isolate(tmp_path, monkeypatch)
        server = {"type": "http", "url": "https://mcp.example.invalid/rpc", **servers}
        before = self._write(global_file, server)

        with pytest.raises(MCPConfigWriteError) as caught:
            bind_header_secret("acme", header, key_id, cwd=tmp_path)

        assert caught.value.code == code, caught.value.errors
        assert global_file.read_text() == before

    def test_a_bind_refuses_a_foreign_row_and_an_unknown_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only the file that DEFINES the server may gain a header for it."""
        from local_operator.mcp.config import bind_header_secret

        global_file = self._isolate(tmp_path, monkeypatch)
        self._write(global_file, {"type": "http", "url": "https://a.invalid/rpc"})
        (tmp_path / ".mcp.json").write_text(
            json.dumps({"mcpServers": {"other": {"type": "http", "url": "https://b.invalid/rpc"}}})
        )

        with pytest.raises(MCPConfigWriteError) as unknown:
            bind_header_secret("absent", "X-Api-Key", "ACME_KEY", cwd=tmp_path)
        assert unknown.value.code == "unknown_server"

        with pytest.raises(MCPConfigWriteError) as foreign:
            bind_header_secret("other", "X-Api-Key", "ACME_KEY", cwd=tmp_path)
        assert foreign.value.code == "not_owned"

    def test_a_binding_rolls_back_to_the_files_own_bytes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """QA Q-2: the undo restores the bytes, not a re-serialisation of them.

        A hand-formatted file (4-space indent, no final newline) came back
        reindented with a newline added, so a refused ``add_key`` left a diff in
        a file nobody asked it to touch.
        """
        from local_operator.mcp.config import bind_header_secret, unbind_header_secret

        global_file = self._isolate(tmp_path, monkeypatch)
        before = self._write(global_file, {"type": "http", "url": "https://a.invalid/rpc"})

        binding = bind_header_secret("acme", self.HEADER, self.KEY_ID, cwd=tmp_path)
        assert "${ACME_KEY}" in global_file.read_text()

        unbind_header_secret("acme", self.HEADER, binding)

        assert global_file.read_text() == before
        assert "headers" not in json.loads(before)["mcpServers"]["acme"]

    def test_a_binding_a_hand_edit_replaced_loses_only_our_header(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The byte restore is conditional: a concurrent edit is never thrown away."""
        from local_operator.mcp.config import bind_header_secret, unbind_header_secret

        global_file = self._isolate(tmp_path, monkeypatch)
        self._write(global_file, {"type": "http", "url": "https://a.invalid/rpc"})
        binding = bind_header_secret("acme", self.HEADER, self.KEY_ID, cwd=tmp_path)

        edited = json.loads(global_file.read_text())
        edited["mcpServers"]["acme"]["timeout"] = 30
        global_file.write_text(json.dumps(edited, indent=2))

        unbind_header_secret("acme", self.HEADER, binding)

        after = json.loads(global_file.read_text())["mcpServers"]["acme"]
        assert after["timeout"] == 30, "the hand edit was rolled back"
        assert "headers" not in after
