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
        merges seven sources but ``_scope_path`` writes exactly two, so every
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
