"""``${NAME}`` secret references resolve before the transport sees them.

The pre-fix behaviour these tests are written against: ``env`` and ``headers``
went to the transport verbatim, so a server configured through the desktop (whose
writer demands a reference) was handed the literal ``${TOKEN}`` and could never
authenticate. The helper cases pin the reference rule; the manager cases pin the
SEAM — resolution happens before any transport is entered, the refusal names the
key, and the config that was written keeps the reference it was written with.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, cast

import pytest

from local_operator.mcp.config import MCPHttpServerConfig, MCPStdioServerConfig
from local_operator.mcp.manager import McpManager
from local_operator.mcp.secret_refs import McpSecretRefError, resolve_config_secrets

#: A synthetic value, so a failing assertion never quotes a real credential.
SENTINEL = "SENTINEL-VALUE"


def _isolate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point the config dir at a throwaway directory and return it."""
    config = tmp_path / "config"
    config.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    return config


def _store(config: Path, values: dict[str, str]) -> None:
    """Write the credential store the Settings > API credentials surface writes."""
    (config / "credentials.env").write_text(
        "".join(f"{key}={value}\n" for key, value in values.items()), encoding="utf-8"
    )


def _project(tmp_path: Path, servers: dict[str, Any]) -> Path:
    """Write ``servers`` as a project MCP config and return the project dir."""
    (tmp_path / ".local-operator").mkdir(exist_ok=True)
    (tmp_path / ".local-operator" / "mcp.json").write_text(
        json.dumps({"mcpServers": servers}), encoding="utf-8"
    )
    return tmp_path


def _stdio(env: dict[str, str]) -> MCPStdioServerConfig:
    return MCPStdioServerConfig(command="probe-cmd", env=env)


def _http(headers: dict[str, str]) -> MCPHttpServerConfig:
    return MCPHttpServerConfig(url="https://example.com/mcp", headers=headers)


class TestReferenceRule:
    def test_a_stdio_env_reference_resolves(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        _store(_isolate(monkeypatch, tmp_path), {"HUBSPOT_TOKEN": SENTINEL})

        resolved = resolve_config_secrets("hubspot", _stdio({"HUBSPOT_TOKEN": "${HUBSPOT_TOKEN}"}))

        assert resolved.env == {"HUBSPOT_TOKEN": SENTINEL}

    def test_an_embedded_header_reference_resolves_and_keeps_its_prefix(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        _store(_isolate(monkeypatch, tmp_path), {"HUBSPOT_TOKEN": SENTINEL})

        resolved = resolve_config_secrets(
            "hubspot", _http({"Authorization": "Bearer ${HUBSPOT_TOKEN}"})
        )

        assert resolved.headers == {"Authorization": f"Bearer {SENTINEL}"}

    def test_several_references_in_one_value_all_resolve(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        _store(_isolate(monkeypatch, tmp_path), {"USER": "u", "PASS": "p"})

        resolved = resolve_config_secrets(
            "basic", _http({"Authorization": "Basic ${USER}:${PASS}"})
        )

        assert resolved.headers == {"Authorization": "Basic u:p"}

    def test_an_unresolvable_reference_names_the_key_and_never_the_literal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        _isolate(monkeypatch, tmp_path)  # no store at all

        with pytest.raises(McpSecretRefError) as caught:
            resolve_config_secrets("hubspot", _stdio({"HUBSPOT_TOKEN": "${HUBSPOT_TOKEN}"}))

        message = str(caught.value)
        assert "HUBSPOT_TOKEN" in message
        assert "hubspot" in message
        # The escape is what a config that meant the reference literally has to
        # be written with, so the message names it (finding Q1).
        assert "double the $" in message
        # Never the reference text passed through as if it were a value, and
        # never the store's location for the reader to wander into.
        assert "${HUBSPOT_TOKEN}" not in message
        assert "credentials.env" not in message

    def test_an_empty_stored_value_counts_as_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        # The credentials surface lists non-empty keys only, so resolving to ""
        # would send an empty credential and hide the misconfiguration.
        _store(_isolate(monkeypatch, tmp_path), {"HUBSPOT_TOKEN": ""})

        with pytest.raises(McpSecretRefError):
            resolve_config_secrets("hubspot", _http({"Authorization": "${HUBSPOT_TOKEN}"}))

    def test_a_key_only_in_the_process_environment_does_not_resolve(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        # Deliberate: CredentialManager.get_credential falls back to os.environ,
        # and that fallback is NOT used here — a project config is untrusted
        # input, and the daemon's environment is not the credentials surface.
        _isolate(monkeypatch, tmp_path)
        monkeypatch.setenv("ONLY_IN_ENV", SENTINEL)

        with pytest.raises(McpSecretRefError) as caught:
            resolve_config_secrets("hubspot", _stdio({"ONLY_IN_ENV": "${ONLY_IN_ENV}"}))

        assert "ONLY_IN_ENV" in str(caught.value)

    @pytest.mark.parametrize(
        "value",
        [
            "plain$value",  # a bare $NAME is not a reference
            "${1BAD}",  # name does not start with a letter or underscore
            "${a b}",  # a space is not a name character
            "${}",  # empty name
            "${{x}}",  # nested braces
            "${unterminated",  # no closing brace
            "${",
            "${MISSING:-x}",  # decorated, but the candidate names nothing stored
            "cost $$5",  # no ${ anywhere, so the escape is not even a token
        ],
    )
    def test_a_value_that_is_not_a_reference_is_passed_through_untouched(
        self, value: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        config = _isolate(monkeypatch, tmp_path)

        resolved = resolve_config_secrets("handwritten", _http({"X-Literal": value}))

        assert resolved.headers == {"X-Literal": value}
        # Untouched also means the store is not CREATED: the read is a stat and
        # an early return on a missing file, never a ``CredentialManager`` (whose
        # constructor writes an empty credentials file).
        assert not (config / "credentials.env").exists()

    def test_a_mixed_value_is_refused_rather_than_half_applied(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        _store(_isolate(monkeypatch, tmp_path), {"TOKEN": SENTINEL})

        with pytest.raises(McpSecretRefError) as caught:
            resolve_config_secrets("hubspot", _http({"Authorization": "Bearer ${TOKEN}${1BAD}"}))

        message = str(caught.value)
        assert "malformed" in message
        assert "Authorization" in message
        # The value is not quoted back: it may hold literal credential text, and
        # this message reaches the log.
        assert SENTINEL not in message
        assert "${1BAD}" not in message

    def test_a_config_with_no_reference_is_returned_unchanged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        config = _isolate(monkeypatch, tmp_path)
        cfg = _stdio({"LOG_LEVEL": "debug"})

        assert resolve_config_secrets("plain", cfg) is cfg
        assert not (config / "credentials.env").exists()

    def test_the_store_mapping_stays_wrapped(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        # Finding R2: the mapping is built on every referenced connect, so it
        # keeps ``SecretStr`` and the one substituted key is what gets unwrapped.
        from pydantic import SecretStr

        from local_operator.mcp.secret_refs import _store_values

        _store(_isolate(monkeypatch, tmp_path), {"T": SENTINEL, "OTHER": f"{SENTINEL}-2"})

        store = _store_values()

        assert all(isinstance(value, SecretStr) for value in store.values())
        # A repr of the mapping is what a stray log line would print, and a
        # wrapped value redacts itself there.
        assert SENTINEL not in repr(store)


class TestEscape:
    """``$${`` is the escape (finding Q1): a literal ``${``, never a reference.

    Without it a value that means a reference literally has no expression at
    all, which is what a config written for another tool (``${HOME}``) or for a
    child that expands its own ``$VAR`` needs.
    """

    def test_a_doubled_dollar_passes_a_reference_through_literally(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        _store(_isolate(monkeypatch, tmp_path), {"T": SENTINEL})

        resolved = resolve_config_secrets("foreign", _stdio({"T": "$${T}"}))

        assert resolved.env == {"T": "${T}"}

    def test_the_escape_works_inside_a_header_value(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        _store(_isolate(monkeypatch, tmp_path), {"T": SENTINEL})

        resolved = resolve_config_secrets("foreign", _http({"Authorization": "Bearer $${T}"}))

        assert resolved.headers == {"Authorization": "Bearer ${T}"}

    def test_the_escape_and_a_real_reference_coexist(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        _store(_isolate(monkeypatch, tmp_path), {"T": SENTINEL})

        resolved = resolve_config_secrets("foreign", _stdio({"T": "${T}:$${T}"}))

        assert resolved.env == {"T": f"{SENTINEL}:${{T}}"}

    def test_a_foreign_config_reference_is_refused_then_escapable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        # The project-config case from QA cell 23: ``${HOME}/data`` in a
        # checked-in `.mcp.json`. The refusal is loud and names HOME (a key the
        # store does not hold), and ``$${HOME}`` is the documented way to keep it.
        _store(_isolate(monkeypatch, tmp_path), {"OTHER": SENTINEL})

        with pytest.raises(McpSecretRefError) as caught:
            resolve_config_secrets("project", _stdio({"PATHVAR": "${HOME}/data"}))
        assert "HOME" in str(caught.value)

        escaped = resolve_config_secrets("project", _stdio({"PATHVAR": "$${HOME}/data"}))
        assert escaped.env == {"PATHVAR": "${HOME}/data"}


class TestDecoratedFragments:
    """Finding Q3/R1: a fragment that names a stored key is refused, not handed on.

    The store accepts ANY key (``set_credential`` checks only control
    characters), so a real credential can sit at a name the reference class
    cannot spell — ``hubspot-token``, or a key behind shell/compose decoration.
    Passing those through starts the server with the literal as its credential,
    which is the failure this module exists to remove.
    """

    @pytest.mark.parametrize(
        ("store_key", "fragment"),
        [
            ("hubspot-token", "${hubspot-token}"),  # the key itself, out of class
            ("my.key", "${my.key}"),  # a dotted key
            ("TOKEN", "${TOKEN:-}"),  # compose/shell default form, no default
            ("TOKEN", "${TOKEN:-fallback}"),  # …with a default
            ("TOKEN", "${TOKEN-SUB}"),  # shell "use SUB if unset"
            ("TOKEN", "${TOKEN:?must be set}"),  # shell error form
            ("TOKEN", "${TOKEN:1}"),  # shell substring form
            ("TOKEN", "${env:TOKEN}"),  # compose "from the environment"
            ("TOKEN", "${ TOKEN }"),  # stray whitespace
            # Bash's string operators, the class round 2 found still open: each of
            # these named a stored key and was handed to the child verbatim.
            ("TOKEN", "${!TOKEN}"),  # indirection
            ("TOKEN", "${#TOKEN}"),  # length, operator in PREFIX position
            ("TOKEN", "${TOKEN#x}"),  # strip shortest prefix
            ("TOKEN", "${TOKEN##x}"),  # strip longest prefix
            ("TOKEN", "${TOKEN%suffix}"),  # strip suffix
            ("TOKEN", "${TOKEN/x/y}"),  # replace
            ("TOKEN", "${TOKEN^}"),  # upper-case
            ("TOKEN", "${TOKEN,}"),  # lower-case
            ("TOKEN", "${TOKEN^^}"),
            ("TOKEN", "${TOKEN@Q}"),  # quote
            ("TOKEN", "${TOKEN#x:-}"),  # decorated AND operator-bearing
            ("TOKEN", "${env:TOKEN#x}"),
            ("TOKEN", "${ TOKEN#x }"),
            ("TOKEN", "${ENV:TOKEN}"),  # a decoration nobody enumerated
        ],
    )
    def test_a_fragment_naming_a_stored_key_is_refused(
        self,
        store_key: str,
        fragment: str,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        _store(_isolate(monkeypatch, tmp_path), {store_key: SENTINEL})

        with pytest.raises(McpSecretRefError) as caught:
            resolve_config_secrets("handwritten", _stdio({"VALUE": fragment}))

        message = str(caught.value)
        assert store_key in message
        assert "unusable secret reference" in message
        # Names the key, never the value, and never the fragment as written.
        assert SENTINEL not in message
        assert fragment not in message

    @pytest.mark.parametrize(
        ("store_key", "fragment"),
        [
            # The store accepts ANY key (control characters only), so Settings can
            # hold a name whose punctuation no run class can spell. Round 2's run
            # shapes alone dropped the raw fragment as a candidate and handed these
            # over as literals again (review round 3, finding 1).
            ("MY KEY", "${MY KEY}"),
            ("API:KEY", "${API:KEY}"),
            ("KEY#2", "${KEY#2}"),
            ("KEY%1", "${KEY%1}"),
            ("TOKEN+2", "${TOKEN+2}"),
            ("KEY,1", "${KEY,1}"),
            ("A@B", "${A@B}"),
            ("TOKEN/2", "${TOKEN/2}"),
            # Only the STRIPPED candidate can match here, so it pins what adding
            # ``inner.strip()`` buys beyond the raw fragment.
            ("MY KEY", "${ MY KEY }"),
        ],
    )
    def test_a_stored_key_with_punctuation_outside_the_run_class_is_refused(
        self,
        store_key: str,
        fragment: str,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        _store(_isolate(monkeypatch, tmp_path), {store_key: SENTINEL})

        with pytest.raises(McpSecretRefError) as caught:
            resolve_config_secrets("handwritten", _stdio({"VALUE": fragment}))

        message = str(caught.value)
        assert store_key in message
        assert "unusable secret reference" in message
        assert SENTINEL not in message

    def test_a_fragment_naming_nothing_stored_is_still_literal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        _store(_isolate(monkeypatch, tmp_path), {"OTHER": SENTINEL})

        for fragment in (
            "${1BAD}",
            "${MISSING:-x}",
            "${}",
            "${unterminated",
            "${{x}}",
            "${a b}",
            "a$$b",
            "cost $$5",
        ):
            resolved = resolve_config_secrets("handwritten", _stdio({"VALUE": fragment}))
            assert resolved.env == {"VALUE": fragment}

    def test_a_stored_value_of_an_unexpected_type_is_not_reported_as_absent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        # The key IS there, so "add it in Settings" would send the user to an
        # entry they already have — the two failures stay apart (review nit 2).
        from local_operator.mcp import secret_refs

        _iso = _isolate(monkeypatch, tmp_path)
        _store(_iso, {"OTHER": SENTINEL})
        monkeypatch.setattr(secret_refs, "_store_values", lambda *args, **kwargs: {"T": 12345})

        with pytest.raises(McpSecretRefError) as caught:
            resolve_config_secrets("hubspot", _stdio({"T": "${T}"}))

        message = str(caught.value)
        assert "cannot read" in message
        assert "add it in Settings" not in message

    def test_an_escaped_decorated_fragment_is_literal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        # The escape wins over the candidate check, which is how a value that
        # genuinely means the text literally is written.
        _store(_isolate(monkeypatch, tmp_path), {"TOKEN": SENTINEL})

        resolved = resolve_config_secrets("handwritten", _stdio({"VALUE": "$${TOKEN:-}"}))

        assert resolved.env == {"VALUE": "${TOKEN:-}"}

    def test_the_decorated_refusal_outranks_the_mixed_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        # Both halves are wrong; the fragment that names a real credential is
        # the actionable one, so it is the one that gets named.
        _store(_isolate(monkeypatch, tmp_path), {"TOKEN": SENTINEL, "next-token": SENTINEL})

        with pytest.raises(McpSecretRefError) as caught:
            resolve_config_secrets("handwritten", _stdio({"VALUE": "${TOKEN}${next-token}"}))

        message = str(caught.value)
        assert "unusable secret reference" in message
        assert "next-token" in message


class TestManagerSeam:
    @pytest.mark.asyncio
    async def test_the_reference_the_desktop_writer_accepts_is_the_one_the_reader_resolves(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """End to end across the two halves: what `add` stores is what connect resolves.

        The defect was exactly this disagreement — the writer demanded a reference
        and the reader ignored it — so the round trip is pinned rather than assumed.
        """
        from local_operator.mcp.desktop import MCPControl, MCPDesktop

        _store(_isolate(monkeypatch, tmp_path), {"T": SENTINEL})
        manager = McpManager(str(tmp_path))
        seen: list[dict[str, str]] = []

        class _Session:
            mcp_manager = manager

        async def fake_open(*args: Any, **kwargs: Any) -> None:
            seen.append(dict(args[2].env))
            raise McpSecretRefError("stop here")

        monkeypatch.setattr(manager, "_open_transport_and_session", fake_open)

        desktop = MCPDesktop(_Session(), set(), str(tmp_path))
        await desktop.execute(
            MCPControl(
                action="add",
                name="hubspot",
                command="probe-cmd",
                scope="project",
                env={"T": "${T}"},
            )
        )

        written = json.loads(
            (tmp_path / ".local-operator" / "mcp.json").read_text(encoding="utf-8")
        )
        assert written["mcpServers"]["hubspot"]["env"] == {"T": "${T}"}
        assert seen == [{"T": SENTINEL}]

    @pytest.mark.asyncio
    async def test_the_connect_refuses_before_any_transport_is_entered(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        _isolate(monkeypatch, tmp_path)
        project = _project(
            tmp_path, {"hubspot": {"type": "stdio", "command": "probe-cmd", "env": {"T": "${T}"}}}
        )
        manager = McpManager(str(project))
        entered: list[str] = []

        async def fake_open(*args: Any, **kwargs: Any) -> None:
            entered.append("opened")
            raise AssertionError("the transport must not be entered for an unresolved reference")

        monkeypatch.setattr(manager, "_open_transport_and_session", fake_open)

        with pytest.raises(McpSecretRefError) as caught:
            await manager.connect_configured_server("hubspot", interactive=False)

        assert "T" in str(caught.value)
        assert entered == []

    @pytest.mark.asyncio
    async def test_a_resolved_stdio_env_reaches_the_transport(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        _store(_isolate(monkeypatch, tmp_path), {"T": SENTINEL})
        project = _project(
            tmp_path, {"hubspot": {"type": "stdio", "command": "probe-cmd", "env": {"T": "${T}"}}}
        )
        manager = McpManager(str(project))
        seen: list[dict[str, str]] = []

        async def fake_open(*args: Any, **kwargs: Any) -> None:
            seen.append(dict(args[2].env))
            raise McpSecretRefError("stop here")

        monkeypatch.setattr(manager, "_open_transport_and_session", fake_open)

        with pytest.raises(McpSecretRefError):
            await manager.connect_configured_server("hubspot", interactive=False)

        assert seen == [{"T": SENTINEL}]

    @pytest.mark.asyncio
    async def test_a_resolved_header_reaches_the_transport(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        _store(_isolate(monkeypatch, tmp_path), {"T": SENTINEL})
        project = _project(
            tmp_path,
            {
                "remote": {
                    "type": "http",
                    "url": "https://example.com/mcp",
                    "headers": {"Authorization": "Bearer ${T}"},
                }
            },
        )
        manager = McpManager(str(project))
        seen: list[dict[str, str]] = []

        async def fake_open(*args: Any, **kwargs: Any) -> None:
            seen.append(dict(args[2].headers))
            raise McpSecretRefError("stop here")

        monkeypatch.setattr(manager, "_open_transport_and_session", fake_open)

        with pytest.raises(McpSecretRefError):
            await manager.connect_configured_server("remote", interactive=False)

        assert seen == [{"Authorization": f"Bearer {SENTINEL}"}]

    @pytest.mark.asyncio
    async def test_the_stored_config_keeps_the_reference_it_was_written_with(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        # The tool cache is keyed on a digest of the STORED config, so a digest
        # of resolved secrets would both persist evidence of them and thrash the
        # cache on every rotation.
        _store(_isolate(monkeypatch, tmp_path), {"T": SENTINEL})
        project = _project(
            tmp_path, {"hubspot": {"type": "stdio", "command": "probe-cmd", "env": {"T": "${T}"}}}
        )
        manager = McpManager(str(project))

        async def fake_open(*args: Any, **kwargs: Any) -> None:
            raise McpSecretRefError("stop here")

        monkeypatch.setattr(manager, "_open_transport_and_session", fake_open)
        with pytest.raises(McpSecretRefError):
            await manager.connect_configured_server("hubspot", interactive=False)

        stored = manager.get_server_config("hubspot")
        assert isinstance(stored, MCPStdioServerConfig)
        assert stored.env == {"T": "${T}"}

    @pytest.mark.asyncio
    async def test_the_store_is_read_fresh_on_every_connect(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        # The credential is typically added in another process (the API server
        # handles PATCH /v1/credentials) while this one holds MCP connections,
        # so a cached store would keep a just-added credential "missing".
        config = _isolate(monkeypatch, tmp_path)
        project = _project(
            tmp_path, {"hubspot": {"type": "stdio", "command": "probe-cmd", "env": {"T": "${T}"}}}
        )
        manager = McpManager(str(project))
        resolved_env: list[dict[str, str]] = []

        async def fake_open(*args: Any, **kwargs: Any) -> None:
            resolved_env.append(dict(args[2].env))
            raise McpSecretRefError("stop here")

        monkeypatch.setattr(manager, "_open_transport_and_session", fake_open)
        with pytest.raises(McpSecretRefError):
            await manager.connect_configured_server("hubspot", interactive=False)
        _store(config, {"T": SENTINEL})
        with pytest.raises(McpSecretRefError):
            await manager.connect_configured_server("hubspot", interactive=False)

        assert resolved_env == [{"T": SENTINEL}]

    @pytest.mark.asyncio
    async def test_the_live_connection_carries_the_reference_form_not_the_value(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        # Finding R5: six registration paths install a ``ServerConnection``, which
        # is a plain dataclass whose repr prints every field, so the resolved
        # value must not outlive the transport that needed it.
        from local_operator.mcp.manager import ServerConnection

        _store(_isolate(monkeypatch, tmp_path), {"T": SENTINEL})
        manager = McpManager(str(tmp_path))
        cfg = MCPStdioServerConfig(command="probe-cmd", env={"T": "${T}"})
        conn_sent: list[Any] = []
        handed_env: list[dict[str, str]] = []

        async def fake_open(stack: Any, name: str, handed_cfg: Any, *args: Any, **_: Any) -> Any:
            conn_sent.append(handed_cfg)
            handed_env.append(dict(handed_cfg.env))
            return ServerConnection(
                name=name, config=handed_cfg, session=cast(Any, object()), stack=stack
            )

        async def no_tools(session: Any) -> list[Any]:
            return []

        monkeypatch.setattr(manager, "_open_transport_and_session", fake_open)
        monkeypatch.setattr(manager, "_list_all_tools", no_tools)

        conn = await manager._connect_server("hubspot", cfg)

        assert handed_env == [{"T": SENTINEL}]  # the transport was handed the value
        assert conn_sent[0] is not cfg
        assert conn.config is cfg  # the connection keeps the reference form
        assert SENTINEL not in repr(conn)

    def test_a_connection_built_with_a_resolved_config_does_not_print_it(self):
        """The repr guard itself (review nit 4).

        The seam test above cannot pin this: the pristine install keeps the value
        out of the repr on its own, so its `SENTINEL not in repr(conn)` passes even
        with `field(repr=False)` removed. This builds the connection the way the
        transport helper does — config carrying the RESOLVED value — which is the
        state the hardening exists for.
        """
        from local_operator.mcp.manager import ServerConnection

        resolved = MCPStdioServerConfig(command="probe-cmd", env={"T": SENTINEL})
        conn = ServerConnection(name="hubspot", config=resolved, session=cast(Any, object()))

        assert SENTINEL not in repr(conn)
        assert "config=" not in repr(conn)

    @pytest.mark.asyncio
    async def test_the_value_never_reaches_a_message_or_a_log_line(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ):
        # End to end through the real connect path with a command that cannot
        # start: the failure is reported (the project-config trust warning names
        # the command, the connect error names the child), and none of those
        # lines may carry the resolved value.
        _store(_isolate(monkeypatch, tmp_path), {"T": SENTINEL})
        project = _project(
            tmp_path,
            {
                "hubspot": {
                    "type": "stdio",
                    "command": "probe-cmd-that-does-not-exist-xyz",
                    "env": {"T": "${T}"},
                }
            },
        )
        manager = McpManager(str(project))

        with caplog.at_level(logging.DEBUG):
            with pytest.raises(Exception) as caught:
                await asyncio.wait_for(
                    manager.connect_configured_server("hubspot", interactive=False), timeout=30
                )

        assert SENTINEL not in str(caught.value)
        assert SENTINEL not in caplog.text
        await manager.disconnect_all()
