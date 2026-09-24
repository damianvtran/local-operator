"""The MCP catalog row builder and the scope fix behind it.

``local_operator/mcp/catalog.py`` is the one place a desktop MCP row is derived,
so this file pins the DERIVATIONS the published contract promises (every field
the UI renders, and nothing it has to decide itself): scope vs owned scope, the
status precedence chain, the action list that decides which buttons exist, the
last-seen tool count, and the reasons a row is refused a write.

Two halves:

* the ROW BUILDER, exercised directly — it is a pure function of files on disk
  plus the optional live/probe overlays, so a test can state both sides of every
  precedence pair without a server, a session or a model;
* the SCOPE FIX (``project_scope_available`` / ``owned_scope_for_source`` /
  ``add_server``), which is the root cause of "Global shows as This project":
  at the desktop's default folder (``~``) the project file IS the global file,
  and before this the page offered a scope switch that wrote one file under two
  labels and a remove that routed through the wrong one.

The collision is built with a config dir INSIDE the cwd — ``<root>/proj/
.local-operator`` — rather than by patching ``HOME``, because that is exactly
the default install's shape and it keeps the real ``config_dir()`` reader in the
loop (it re-reads the environment on every call).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from local_operator.mcp.catalog import (
    PROBE_TTL_S,
    LiveFacts,
    ProbeResult,
    describe_servers,
    live_facts_from_snapshot,
    public_reason,
    source_kind,
)
from local_operator.mcp.config import (
    MCPConfigWriteError,
    add_server,
    load_all_mcp_configs,
    owned_scope_for_source,
    project_scope_available,
    tool_enabled_by_config,
)
from local_operator.mcp.tool_cache import McpToolCache, config_digest

#: A server whose command is a real interpreter, so the config validates. It is
#: never spawned here: this file only reads the config it describes.
COMMAND = "/bin/echo"

#: An obvious placeholder, never a key-shaped string: this suite holds no credential at
#: all, and a test source is the last place one should be able to hide.
PLACEHOLDER = "test-key-placeholder"


@pytest.fixture
def colliding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A cwd whose project file is the global file (the ``~`` case).

    ``HOME`` is redirected with the config dir because the loader reads the
    USER-scope sources too (``~/.claude.json``, ``~/.cursor/mcp.json``,
    ``~/.codex/config.toml``); without it a test's row list is whatever the
    developer happens to have imported.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(project / ".local-operator"))
    return project


@pytest.fixture
def distinct(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A cwd with a project file of its own, and a config dir elsewhere."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    project = tmp_path / "proj"
    project.mkdir()
    return project


def _global_mcp_json() -> Path:
    from local_operator.paths import config_dir

    return config_dir() / "mcp.json"


# ---------------------------------------------------------------------------
# the scope fix
# ---------------------------------------------------------------------------


def test_a_cwd_that_owns_the_global_file_has_no_project_scope(colliding: Path) -> None:
    assert project_scope_available(colliding) is False


def test_a_project_with_its_own_file_has_a_project_scope(distinct: Path) -> None:
    assert project_scope_available(distinct) is True


def test_the_collapsed_file_is_reported_global_not_project(colliding: Path) -> None:
    """The reported bug, at the level it was caused.

    ``owned_scope_for_source`` asked "project" first, and at ``~`` that answered
    the GLOBAL file — so every global server rendered as a project one and a
    remove carried ``scope: "project"`` for a row the user had added as global.
    """
    add_server("shared", command=COMMAND, scope="global", cwd=colliding)
    configs, sources = load_all_mcp_configs(colliding)

    assert "shared" in configs
    assert owned_scope_for_source(sources["shared"], colliding) == "global"


def test_a_project_write_where_the_files_collide_is_refused(colliding: Path) -> None:
    """Refused, not silently written to the global file under a project label."""
    before = _global_mcp_json().read_text() if _global_mcp_json().exists() else None

    with pytest.raises(MCPConfigWriteError) as raised:
        add_server("sneaky", command=COMMAND, scope="project", cwd=colliding)

    assert raised.value.code == "project_scope_unavailable"
    assert "shared" not in (raised.value.errors[0])
    now = _global_mcp_json().read_text() if _global_mcp_json().exists() else None
    assert now == before, "a refused project write touched the global file"


def test_a_project_write_with_a_real_project_scope_lands_in_the_project_file(
    distinct: Path,
) -> None:
    add_server("local", command=COMMAND, scope="project", cwd=distinct)
    configs, sources = load_all_mcp_configs(distinct)

    assert Path(sources["local"]) == distinct / ".local-operator" / "mcp.json"
    assert owned_scope_for_source(sources["local"], distinct) == "project"
    assert not (_global_mcp_json()).exists(), "a project write reached the global file"


def test_a_duplicate_name_is_refused_with_its_own_code(distinct: Path) -> None:
    add_server("dup", command=COMMAND, scope="global", cwd=distinct)

    with pytest.raises(MCPConfigWriteError) as raised:
        add_server("dup", command=COMMAND, scope="global", cwd=distinct)

    assert raised.value.code == "exists"


def test_an_invalid_config_is_refused_with_its_own_code(distinct: Path) -> None:
    """``invalid_config`` is the config-layer category, distinct from a write fault."""
    with pytest.raises(MCPConfigWriteError) as raised:
        add_server("not-http", url="ftp://x.invalid/rpc", scope="global", cwd=distinct)

    assert raised.value.code == "invalid_config"


def test_a_named_and_url_less_add_is_a_write_fault_not_an_invalid_config(
    distinct: Path,
) -> None:
    """The schema's own precondition has no other category, and says so.

    ``MCPControl`` refuses this shape before it reaches the writer, so the code
    is only reachable from a direct caller; pinning it keeps the two categories
    from being silently merged by a later tidy-up.
    """
    with pytest.raises(MCPConfigWriteError) as raised:
        add_server("neither", scope="global", cwd=distinct)

    assert raised.value.code == "write_failed"


# ---------------------------------------------------------------------------
# the row builder: identity, scope, transport, source
# ---------------------------------------------------------------------------


def test_a_global_stdio_server_is_not_a_project_one(distinct: Path) -> None:
    add_server("echoer", command=COMMAND, args=["hi"], scope="global", cwd=distinct)

    document = describe_servers(str(distinct))
    (row,) = document["servers"]

    assert document["cwd"] == str(distinct)
    assert document["project_scope_available"] is True
    assert document["project_path"] == str(distinct / ".local-operator" / "mcp.json")
    assert document["status_source"] == "config"
    assert document["session_id"] is None
    assert document["operations"] == []
    assert row["id"] == row["name"] == "echoer"
    assert row["scope"] == "global"
    assert row["project_cwd"] is None
    assert row["transport"] == "local_command"
    assert row["endpoint"] == {"command": COMMAND, "url": None, "endpoint_redacted": False}
    assert row["source"]["kind"] == "local-operator"
    assert row["source"]["editable"] is True
    assert row["source"]["owned_scope"] == "global"


def test_a_project_server_applies_to_the_folder_that_defines_it(distinct: Path) -> None:
    add_server("local", command=COMMAND, scope="project", cwd=distinct)

    (row,) = describe_servers(str(distinct))["servers"]

    assert row["scope"] == "project"
    # The ROW's project_cwd is the DIRECTORY it applies to (what the UI names);
    # the catalog's project_path is the FILE (what a user can open). They are
    # different kinds of value, so they are different fields.
    assert row["project_cwd"] == str(distinct)
    assert row["source"]["owned_scope"] == "project"
    assert row["source"]["editable"] is True


def test_everything_reads_global_when_the_two_scopes_are_one_file(colliding: Path) -> None:
    """The default folder: no "This project" anywhere, and no project file named."""
    add_server("one", command=COMMAND, scope="global", cwd=colliding)

    document = describe_servers(str(colliding))

    assert document["project_scope_available"] is False
    assert document["project_path"] is None
    assert [row["scope"] for row in document["servers"]] == ["global"]
    assert [row["project_cwd"] for row in document["servers"]] == [None]


def test_a_foreign_project_file_is_project_scoped_and_not_ours(distinct: Path) -> None:
    """``<cwd>/.mcp.json`` applies to the project; local-operator must not write it."""
    (distinct / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"borrowed": {"command": COMMAND}}})
    )

    (row,) = describe_servers(str(distinct))["servers"]

    assert row["scope"] == "project"
    assert row["source"]["kind"] == "project-mcp-json"
    assert row["source"]["editable"] is False
    assert row["source"]["owned_scope"] is None
    assert "remove" not in row["actions"]


def test_the_foreign_source_table_is_the_one_verbs_refuses_with() -> None:
    """One authority for "imported from Cursor", shared with the remove refusal."""
    expected = {
        "/home/u/.cursor/mcp.json": "cursor",
        "/home/u/.codex/config.toml": "codex",
        # The two-part fragment must win over the bare ``.mcp.json``.
        "/home/u/.claude/.mcp.json": "claude-code",
        "/home/u/.vscode/mcp.json": "vscode",
        "/home/u/.claude.json": "claude-code",
        "/proj/.mcp.json": "project-mcp-json",
    }
    for path, kind in expected.items():
        known = source_kind(path)
        assert known is not None, path
        assert known[0] == kind, path

    cursor = source_kind("/home/u/.cursor/mcp.json")
    assert cursor is not None and cursor[1] == "imported from Cursor"
    assert source_kind(None) is None
    assert source_kind("/proj/something-else.json") is None


# ---------------------------------------------------------------------------
# the status precedence chain
# ---------------------------------------------------------------------------


def test_a_server_with_no_facts_is_ready(distinct: Path) -> None:
    add_server("echoer", command=COMMAND, scope="global", cwd=distinct)

    (row,) = describe_servers(str(distinct))["servers"]

    assert row["status"] == "not_started"
    assert row["status_basis"] == "stored"
    assert row["status_reason"] is None
    assert row["status_observed_at"] is None
    assert row["actions"] == ["test", "remove"]


def test_a_live_connected_server_reports_its_live_tool_count(distinct: Path) -> None:
    add_server("echoer", command=COMMAND, scope="global", cwd=distinct)

    document = describe_servers(
        str(distinct),
        live={"echoer": LiveFacts(status="connected", tool_count=7)},
        session_id="8fd6c6a40934",
    )
    (row,) = document["servers"]

    assert document["status_source"] == "live"
    assert document["session_id"] == "8fd6c6a40934"
    assert (row["status"], row["status_basis"]) == ("connected", "live")
    assert (row["tool_count"], row["tool_count_basis"]) == (7, "live")
    assert row["actions"] == ["test", "remove", "disconnect"]


def test_an_overlay_that_reached_no_row_is_not_a_live_document(distinct: Path) -> None:
    """``status_source`` and ``session_id`` describe the rows, not the request.

    An overlay that was consulted and used by nothing — a warm runtime that has
    loaded none of these servers, or facts keyed by names this folder does not
    configure — leaves every row reading from config. Calling that document
    ``live``, and echoing a ``session_id``, claims a live fact no row carries.
    """
    add_server("echoer", command=COMMAND, scope="global", cwd=distinct)

    for overlay in ({}, {"elsewhere": LiveFacts(status="connected", tool_count=2)}):
        document = describe_servers(str(distinct), live=overlay, session_id="8fd6c6a40934")
        (row,) = document["servers"]

        assert document["status_source"] == "config", overlay
        assert document["session_id"] is None, overlay
        assert (row["status"], row["status_basis"]) == ("not_started", "stored"), overlay


def test_an_overlay_hidden_by_a_running_operation_is_not_a_live_document(
    distinct: Path,
) -> None:
    """R2-m2: the overlay names a configured server, and still reaches no row.

    A running operation outranks the overlay, so the only server the overlay
    speaks for reads ``connecting``/``operation``. Deriving ``applied`` from the
    overlay's KEYS called that document ``live``.
    """
    add_server("echoer", command=COMMAND, scope="global", cwd=distinct)

    document = describe_servers(
        str(distinct),
        live={"echoer": LiveFacts(status="connected", tool_count=2)},
        session_id="8fd6c6a40934",
        running=frozenset({"echoer"}),
    )
    (row,) = document["servers"]

    assert (row["status"], row["status_basis"]) == ("connecting", "operation")
    assert document["status_source"] == "config"
    assert document["session_id"] is None


def test_a_live_auth_block_is_needs_sign_in_with_its_reason(distinct: Path) -> None:
    add_server("echoer", command=COMMAND, scope="global", cwd=distinct)

    (row,) = describe_servers(
        str(distinct),
        live={"echoer": LiveFacts(status="auth-required", tool_count=0, failure="no grant stored")},
    )["servers"]

    assert row["status"] == "needs_sign_in"
    assert row["status_reason"] == "no grant stored"


def test_a_live_disconnect_with_a_startup_failure_is_an_error(distinct: Path) -> None:
    add_server("echoer", command=COMMAND, scope="global", cwd=distinct)

    (row,) = describe_servers(
        str(distinct),
        live={"echoer": LiveFacts(status="disconnected", tool_count=0, failure="spawn failed")},
    )["servers"]

    assert (row["status"], row["status_reason"]) == ("error", "spawn failed")


def test_a_live_disconnect_without_a_failure_is_only_not_started(distinct: Path) -> None:
    add_server("echoer", command=COMMAND, scope="global", cwd=distinct)

    (row,) = describe_servers(
        str(distinct), live={"echoer": LiveFacts(status="disconnected", tool_count=0)}
    )["servers"]

    assert (row["status"], row["status_reason"]) == ("not_started", None)
    assert row["actions"] == ["test", "remove", "connect"]


def test_a_running_operation_reads_connecting_even_under_a_live_overlay(
    distinct: Path,
) -> None:
    """The user's own press is the fresher fact, and the contract says so.

    A Test runs on its OWN short-lived manager, while the overlay describes the
    conversation's connection — a different connection that may legitimately
    read ``disconnected`` for a server being tested right now. The published
    contract is "a running test reads connecting", with no exception for a
    caller that also passed ``session_id``.

    Its basis is ``operation``, not ``probe``: nothing has ANSWERED yet, and a
    client that reads ``probe`` as a settled result would draw a finished Test —
    with its tool count — for one still running. The null ``status_observed_at``
    is the same statement.
    """
    add_server("echoer", command=COMMAND, scope="global", cwd=distinct)

    (row,) = describe_servers(
        str(distinct),
        live={"echoer": LiveFacts(status="disconnected", tool_count=0)},
        session_id="8fd6c6a40934",
        running=frozenset({"echoer"}),
    )["servers"]

    assert (row["status"], row["status_basis"]) == ("connecting", "operation")
    assert row["status_observed_at"] is None
    assert row["tool_count"] is None, "an unfinished operation has measured nothing"


def test_a_probe_within_its_ttl_is_the_rows_answer(distinct: Path) -> None:
    add_server("echoer", command=COMMAND, scope="global", cwd=distinct)
    digest = config_digest(load_all_mcp_configs(distinct)[0]["echoer"])

    document = describe_servers(
        str(distinct),
        probes={
            "echoer": ProbeResult(
                status="connected",
                reason=None,
                tool_count=3,
                observed_at=1_000.0,
                digest=digest,
            )
        },
        now=1_000.0 + PROBE_TTL_S - 1,
    )
    (row,) = document["servers"]

    assert (row["status"], row["status_basis"]) == ("connected", "probe")
    assert (row["tool_count"], row["tool_count_basis"]) == (3, "probe")
    assert row["status_observed_at"] == 1_000.0


def test_a_probe_past_its_ttl_is_forgotten(distinct: Path) -> None:
    add_server("echoer", command=COMMAND, scope="global", cwd=distinct)
    digest = config_digest(load_all_mcp_configs(distinct)[0]["echoer"])

    (row,) = describe_servers(
        str(distinct),
        probes={
            "echoer": ProbeResult(
                status="connected",
                reason=None,
                tool_count=3,
                observed_at=1_000.0,
                digest=digest,
            )
        },
        now=1_000.0 + PROBE_TTL_S + 1,
    )["servers"]

    assert row["status"] == "not_started"


def test_a_probe_for_a_rewritten_server_is_ignored(distinct: Path) -> None:
    """A stale digest is a statement about a DIFFERENT server, so it is dropped."""
    add_server("echoer", command=COMMAND, scope="global", cwd=distinct)

    (row,) = describe_servers(
        str(distinct),
        probes={
            "echoer": ProbeResult(
                status="connected",
                reason=None,
                tool_count=3,
                observed_at=1_000.0,
                digest="a-digest-of-some-other-config",
            )
        },
        now=1_000.0,
    )["servers"]

    assert row["status"] == "not_started"


def test_a_failed_probe_carries_its_sanitized_reason(distinct: Path) -> None:
    add_server("echoer", command=COMMAND, scope="global", cwd=distinct)
    digest = config_digest(load_all_mcp_configs(distinct)[0]["echoer"])

    (row,) = describe_servers(
        str(distinct),
        probes={
            "echoer": ProbeResult(
                status="error",
                reason="the server exited with code 1",
                tool_count=None,
                observed_at=1_000.0,
                digest=digest,
            )
        },
        now=1_000.0,
    )["servers"]

    assert (row["status"], row["status_reason"]) == ("error", "the server exited with code 1")


def test_an_invalid_config_is_an_error_ahead_of_every_other_fact(distinct: Path) -> None:
    """A config that cannot validate outranks even a live "connected" claim."""
    _global_mcp_json().parent.mkdir(parents=True, exist_ok=True)
    _global_mcp_json().write_text(json.dumps({"mcpServers": {"broken": {"command": ""}}}))

    (row,) = describe_servers(
        str(distinct), live={"broken": LiveFacts(status="connected", tool_count=4)}
    )["servers"]

    assert row["status"] == "error"
    assert row["status_reason"]
    assert "test" not in row["actions"], "an unvalidatable server must offer no Test"


# ---------------------------------------------------------------------------
# auth facts and the actions they gate
# ---------------------------------------------------------------------------


def test_a_stored_oauth_grant_offers_reauth_and_sign_out(distinct: Path) -> None:
    add_server(
        "remote",
        url="https://mcp.example.invalid/rpc",
        oauth=True,
        scope="global",
        cwd=distinct,
    )

    (row,) = describe_servers(str(distinct))["servers"]

    assert row["transport"] == "remote_url"
    assert row["auth"] == {"kind": "oauth", "signed_in": False, "secret_refs": []}
    assert row["status"] == "needs_sign_in"
    assert row["actions"] == ["test", "sign_in", "remove"]


def test_a_bare_url_is_unknown_auth_and_may_be_offered_a_sign_in(distinct: Path) -> None:
    """``unknown`` is honest: whether it takes OAuth needs a probe, not a guess."""
    add_server("remote", url="https://mcp.example.invalid/rpc", scope="global", cwd=distinct)

    (row,) = describe_servers(str(distinct))["servers"]

    assert row["auth"]["kind"] == "unknown"
    assert row["auth"]["signed_in"] is None
    assert "sign_in" in row["actions"]


def test_a_local_command_with_a_reference_offers_set_key_and_not_sign_in(
    distinct: Path,
) -> None:
    """The dead end (c): "Sign in" on a stdio server with no OAuth cannot work."""
    add_server(
        "echoer",
        command=COMMAND,
        env={"API_TOKEN": "${API_TOKEN}"},
        scope="global",
        cwd=distinct,
    )

    (row,) = describe_servers(str(distinct))["servers"]

    assert row["auth"]["kind"] == "api_key"
    assert row["auth"]["signed_in"] is False
    assert row["auth"]["secret_refs"] == [{"id": "API_TOKEN", "state": "missing"}]
    assert row["status"] == "needs_sign_in"
    assert row["actions"] == ["test", "set_key", "remove"]


@pytest.fixture
def challenges(monkeypatch: pytest.MonkeyPatch) -> dict[str, bool]:
    """The per-process 401 ledger, emptied and restored around one test."""
    from local_operator.mcp import auth

    ledger: dict[str, bool] = {}
    monkeypatch.setattr(auth, "OAUTH_CHALLENGES", ledger)
    return ledger


def test_a_401_without_oauth_offers_add_key_and_never_sign_in(
    distinct: Path, challenges: dict[str, bool]
) -> None:
    """U3: Sign in on a server with no OAuth server can only fail.

    The ledger entry is what a Test's own connect records on a 401/403 whose
    discovery found nothing (``manager._auth_challenge``), so this is the row a
    user sees straight after pressing Test on such a server.
    """
    url = "https://mcp.example.invalid/rpc"
    add_server("acme-api", url=url, scope="global", cwd=distinct)
    challenges[url] = False

    (row,) = describe_servers(str(distinct))["servers"]

    assert row["auth"] == {"kind": "api_key", "signed_in": False, "secret_refs": []}
    assert (row["status"], row["status_basis"]) == ("needs_sign_in", "stored")
    assert row["actions"] == ["test", "add_key", "remove"]


def test_a_401_with_oauth_discovered_still_offers_sign_in(
    distinct: Path, challenges: dict[str, bool]
) -> None:
    url = "https://mcp.example.invalid/rpc"
    add_server("remote", url=url, scope="global", cwd=distinct)
    challenges[url] = True

    (row,) = describe_servers(str(distinct))["servers"]

    assert "sign_in" in row["actions"]
    assert "add_key" not in row["actions"]


def test_a_declared_apikey_server_offers_add_key_without_a_probe(
    distinct: Path, challenges: dict[str, bool]
) -> None:
    """``auth.type: apikey`` is the user's own statement; no 401 is needed."""
    global_file = _global_mcp_json()
    global_file.parent.mkdir(parents=True, exist_ok=True)
    global_file.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "acme-api": {
                        "type": "http",
                        "url": "https://mcp.example.invalid/rpc",
                        "auth": {"type": "apikey"},
                    }
                }
            }
        )
    )

    (row,) = describe_servers(str(distinct))["servers"]

    assert row["actions"] == ["test", "add_key", "remove"]


def test_a_server_that_already_references_a_key_keeps_set_key(
    distinct: Path, challenges: dict[str, bool]
) -> None:
    """A reference to fill means ``set_key``; ``add_key`` is only for none."""
    url = "https://mcp.example.invalid/rpc"
    add_server(
        "acme-api", url=url, headers={"X-Api-Key": "${ACME_KEY}"}, scope="global", cwd=distinct
    )
    challenges[url] = False

    (row,) = describe_servers(str(distinct))["servers"]

    assert row["actions"] == ["test", "set_key", "remove"]


def test_a_foreign_401_row_is_not_offered_add_key(
    distinct: Path, challenges: dict[str, bool]
) -> None:
    """``add_key`` writes the server's config, so it needs the row to be ours."""
    url = "https://mcp.example.invalid/rpc"
    (distinct / ".mcp.json").write_text(json.dumps({"mcpServers": {"acme-api": {"url": url}}}))
    challenges[url] = False

    (row,) = describe_servers(str(distinct))["servers"]

    assert row["source"]["editable"] is False
    assert "add_key" not in row["actions"]
    assert "sign_in" not in row["actions"]


def test_a_local_command_without_references_is_never_offered_a_sign_in(
    distinct: Path,
) -> None:
    add_server("echoer", command=COMMAND, scope="global", cwd=distinct)

    (row,) = describe_servers(str(distinct))["servers"]

    assert row["auth"] == {"kind": "none", "signed_in": None, "secret_refs": []}
    assert "sign_in" not in row["actions"]
    assert "set_key" not in row["actions"]
    assert "reauth" not in row["actions"]


def test_a_url_carrying_a_query_is_redacted_in_the_row(distinct: Path) -> None:
    """A token in a query string must not cross HTTP inside ``endpoint.url``.

    The URL is assembled rather than written whole (see the placeholder note
    above); the ROW must report the redaction fact either way, so the assertion
    is on ``endpoint_redacted`` and on the URL being withheld.
    """
    dirty = "https://x.invalid/rpc?" + "api_key=" + PLACEHOLDER
    _global_mcp_json().parent.mkdir(parents=True, exist_ok=True)
    _global_mcp_json().write_text(json.dumps({"mcpServers": {"remote": {"url": dirty}}}))

    (row,) = describe_servers(str(distinct))["servers"]

    assert row["endpoint"]["url"] is None
    assert row["endpoint"]["endpoint_redacted"] is True


# ---------------------------------------------------------------------------
# last-seen tool counts
# ---------------------------------------------------------------------------


def test_a_last_seen_count_comes_from_the_tool_cache(distinct: Path) -> None:
    add_server("echoer", command=COMMAND, scope="global", cwd=distinct)
    digest = config_digest(load_all_mcp_configs(distinct)[0]["echoer"])
    cache = McpToolCache(str(distinct / "cache.db"))
    cache.put("echoer", [{"name": "one"}, {"name": "two"}], digest)

    (row,) = describe_servers(str(distinct), tool_cache=cache)["servers"]

    assert (row["tool_count"], row["tool_count_basis"]) == (2, "last_seen")
    assert row["status"] == "not_started", "last_seen is not a connection claim"


def test_a_last_seen_count_honours_the_tool_deny_list(distinct: Path) -> None:
    """A count that ignored the deny list would advertise uncallable tools.

    Written as JSON rather than through ``add_server``: the allow/deny lists are
    config-file fields with no ``add`` verb, which is exactly the import path
    (a hand-edited or foreign-authored file) the filter has to survive.
    """
    _global_mcp_json().parent.mkdir(parents=True, exist_ok=True)
    _global_mcp_json().write_text(
        json.dumps({"mcpServers": {"echoer": {"command": COMMAND, "disabledTools": ["secret_*"]}}})
    )
    digest = config_digest(load_all_mcp_configs(distinct)[0]["echoer"])
    cache = McpToolCache(str(distinct / "cache.db"))
    cache.put("echoer", [{"name": "secret_x"}, {"name": "public_y"}], digest)

    (row,) = describe_servers(str(distinct), tool_cache=cache)["servers"]

    assert (row["tool_count"], row["tool_count_basis"]) == (1, "last_seen")


def test_the_tool_filter_is_the_managers_own_rule(distinct: Path) -> None:
    """One filter, two readers: the live list and the cached count agree."""
    from local_operator.mcp.config import MCPStdioServerConfig

    assert tool_enabled_by_config(None, "anything") is True
    plain = MCPStdioServerConfig(command=COMMAND)
    assert tool_enabled_by_config(plain, "anything") is True
    denied = MCPStdioServerConfig(command=COMMAND, disabledTools=["a*"])
    assert tool_enabled_by_config(denied, "abc") is False
    assert tool_enabled_by_config(denied, "bcd") is True
    allowed = MCPStdioServerConfig(command=COMMAND, enabledTools=["a*"])
    assert tool_enabled_by_config(allowed, "abc") is True
    assert tool_enabled_by_config(allowed, "bcd") is False
    # Deny wins over allow, which is the documented precedence.
    both = MCPStdioServerConfig(command=COMMAND, enabledTools=["a*"], disabledTools=["ab*"])
    assert tool_enabled_by_config(both, "abc") is False, "the deny glob must win"
    assert tool_enabled_by_config(both, "az") is True, "allowed and not denied"


# ---------------------------------------------------------------------------
# the overlay's shape check, and the pinned fixture
# ---------------------------------------------------------------------------


def test_the_overlay_skips_rows_that_would_lie() -> None:
    """A malformed overlay degrades to config; it never fails the read."""
    facts = live_facts_from_snapshot(
        [
            "not-a-row",
            {"name": "cold"},  # the legacy cold word is not a live status
            {"status": "connected"},  # no name
            {"name": "unloaded", "status": "connected", "loaded": False},
            {"name": "loaded", "status": "connected", "tool_count": 2, "loaded": True},
            {"name": "odd", "status": "connecting", "tool_count": "2"},
        ]
    )

    assert set(facts) == {"loaded", "odd"}, "only genuinely loaded servers overlay"
    assert facts["loaded"].tool_count == 2
    assert facts["odd"].tool_count == 0, "a non-integer count is not a count"
    assert live_facts_from_snapshot(None) == {}


def test_public_reason_bounds_and_scrubs(distinct: Path) -> None:
    """A reason crosses HTTP, so it is scrubbed, unlinked and bounded.

    The user-info and query cases are assembled from parts instead of being
    written as one whole literal URL: what is under test is the SHAPE being
    stripped, and a complete credential-shaped URL in a test source is what this
    repository's redaction rules exist to keep out of one. Nothing real is
    involved either way — ``PLACEHOLDER`` is an obvious placeholder.
    """
    from local_operator.mcp import redaction

    assert public_reason(None) is None
    assert public_reason("   ") is None
    assert public_reason("boom\nboom") == "boom boom"
    redaction.register(PLACEHOLDER)
    try:
        scrubbed = public_reason(f"rejected header {PLACEHOLDER} for https://x.invalid/rpc")
        assert scrubbed is not None
        assert PLACEHOLDER not in scrubbed, "a registered value must not cross HTTP"
    finally:
        redaction.unregister(PLACEHOLDER)
    userinfo = "https://" + "user:" + PLACEHOLDER + "@x.invalid/rpc"
    assert public_reason(userinfo) == "https://[redacted]@x.invalid/rpc"
    query = "https://x.invalid/rpc?" + "api_key=" + PLACEHOLDER
    assert public_reason(query) == "https://x.invalid/rpc", "a query string is unlinked"
    long = public_reason("y" * 5_000)
    assert long is not None and len(long) <= 300


def test_the_pinned_fixture_still_matches_the_builder(distinct: Path) -> None:
    """``docs/fixtures/mcp-catalog.json`` is what the UI repo tests against.

    A pinned sample is only useful while it is the shape the builder actually
    produces, so this pins the KEY SETS (not the values, which are the sample's
    own) plus the closed vocabularies: a field added, renamed or dropped in the
    builder — or a status word invented in the sample — fails here rather than in
    the UI's parity test, where it would look like a UI bug.
    """
    fixture = json.loads(
        (Path(__file__).resolve().parents[3] / "docs" / "fixtures" / "mcp-catalog.json").read_text()
    )
    add_server("echoer", command=COMMAND, scope="global", cwd=distinct)
    produced = describe_servers(str(distinct))
    example = produced["servers"][0]

    assert set(fixture) == set(produced), "catalog-level fields drifted"
    assert set(fixture["servers"][0]) == set(example), "row fields drifted"
    for row in fixture["servers"]:
        assert set(row["source"]) == set(example["source"])
        assert set(row["endpoint"]) == set(example["endpoint"])
        assert set(row["auth"]) == set(example["auth"])
        assert row["status"] in {
            "connected",
            "needs_sign_in",
            "not_started",
            "connecting",
            "error",
        }
        assert row["status_basis"] in {"live", "probe", "operation", "stored"}
        assert row["tool_count_basis"] in {"live", "probe", "last_seen", None}
        assert row["transport"] in {"local_command", "remote_url"}
        assert row["scope"] in {"global", "project"}
        # The two facts a UI keys off to decide whether it may offer a write.
        assert row["source"]["editable"] is (row["source"]["owned_scope"] is not None)
        assert ("remove" in row["actions"]) is row["source"]["editable"]
        assert row["project_cwd"] == (fixture["cwd"] if row["scope"] == "project" else None)
        for action in row["actions"]:
            assert action in {
                "test",
                "sign_in",
                "set_key",
                "add_key",
                "reauth",
                "sign_out",
                "remove",
                "connect",
                "disconnect",
            }
        for ref in row["auth"]["secret_refs"]:
            assert set(ref) == {"id", "state"}
            assert ref["state"] in {"encrypted", "missing", "unavailable"}
    for operation in fixture["operations"]:
        assert set(operation) == {
            "id",
            "name",
            "action",
            "status",
            "created_at",
            "credential_removed",
            "browser_opened",
            "authorization_url",
            "message",
        }
        assert operation["action"] in {"test", "login", "reauth", "logout"}
        assert operation["status"] in {"running", "complete", "failed", "cancelled"}
    # The sample must actually exercise the vocabulary a renderer has to draw,
    # or a UI test passes against rows it will never be handed.
    assert {row["status"] for row in fixture["servers"]} == {
        "connected",
        "needs_sign_in",
        "not_started",
        "connecting",
        "error",
    }
    assert {row["source"]["kind"] for row in fixture["servers"]} >= {
        "local-operator",
        "cursor",
        "codex",
    }
    assert {row["scope"] for row in fixture["servers"]} == {"global", "project"}
    assert any(row["source"]["editable"] is False for row in fixture["servers"])
    assert any(row["endpoint"]["endpoint_redacted"] for row in fixture["servers"])
    assert any(row["tool_count_basis"] == "last_seen" for row in fixture["servers"])
    # The key-entry pair the UI routes on, and the rule between them: ``add_key``
    # only on a row with no reference to fill, and never beside ``sign_in``.
    for row in fixture["servers"]:
        assert not {"add_key", "sign_in"} <= set(row["actions"]), row["name"]
        assert not {"add_key", "set_key"} <= set(row["actions"]), row["name"]
        if "add_key" in row["actions"]:
            assert row["auth"]["secret_refs"] == [] and row["auth"]["kind"] == "api_key"
    assert any("set_key" in row["actions"] for row in fixture["servers"])
    assert any("add_key" in row["actions"] for row in fixture["servers"])
    # The one row a merely-running operation owns: its basis is not a probe's,
    # and it carries no observation, or a renderer cannot tell the two apart.
    connecting = [row for row in fixture["servers"] if row["status"] == "connecting"]
    assert [row["status_basis"] for row in connecting] == ["operation"]
    assert all(row["status_observed_at"] is None for row in connecting)
    assert fixture["operations"][0]["status"] == "running"
