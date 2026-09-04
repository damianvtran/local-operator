"""The shared MCP grant core: who may run a grant, and how it settles.

These cover the defect that produced the module. ``/mcp reauth`` on a detached
session answered "run it from a terminal on that machine" while the user WAS on
that machine, so an expired credential could not be refreshed at all once the
session detached — the routed verb went to the owner, and the owner refused it.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from local_operator.mcp.grants import (
    GRANT_SUBCOMMANDS,
    REMOTE_GRANT_NOTICE,
    resolve_server,
    run_grant,
    start_grant,
)


class _Conn:
    def __init__(self, tools: int = 3) -> None:
        self.tools = list(range(tools))


class _Cfg:
    """The minimum a config needs to survive the static OAuth refusal.

    ``server_rejects_oauth`` refuses a stdio server (no ``url``) and one whose
    ``auth.type`` names something other than OAuth, so an http server with no
    declared auth block is the shape that reaches the live probe.
    """

    auth = None
    url = "https://mcp.example.com/mcp"


class _Manager:
    """A manager double with the accessors the grant core probes."""

    def __init__(self, *, cfg: Any = None, supports: bool = True) -> None:
        self._cfg = cfg if cfg is not None else _Cfg()
        self._supports = supports
        self.connected: list[tuple[str, float | None]] = []
        self.disconnected: list[str] = []
        self.connect_error: Exception | None = None
        self.connect_gate: asyncio.Event | None = None

    def get_server_config(self, name: str) -> Any:
        return self._cfg

    async def server_supports_oauth_login(self, cfg: Any) -> bool:
        return self._supports

    async def disconnect_server(self, name: str) -> None:
        self.disconnected.append(name)

    async def connect_configured_server(
        self, name: str, *, timeout_ms: float | None = None
    ) -> _Conn:
        if self.connect_gate is not None:
            await self.connect_gate.wait()
        self.connected.append((name, timeout_ms))
        if self.connect_error is not None:
            raise self.connect_error
        return _Conn()


class _Session:
    def __init__(self, manager: Any) -> None:
        self.mcp_manager = manager


@pytest.fixture(autouse=True)
def _no_real_credential_writes(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Never touch the developer's real ``auth.db`` from a unit test."""
    removed: list[str] = []

    def _fake_logout(name: str, cwd: str) -> str | None:
        removed.append(name)
        return None

    monkeypatch.setattr("local_operator.mcp.auth.mcp_logout_server", _fake_logout)
    return removed


def test_grant_subcommands_are_the_three_oauth_verbs() -> None:
    assert set(GRANT_SUBCOMMANDS) == {"login", "logout", "reauth"}


def test_frontend_state_aliases_the_canonical_verb_set() -> None:
    """The dispatch and the implementation must not keep separate copies.

    ``frontend_state`` decides which verbs ROUTE to the owner; this module
    decides which the owner RUNS. Two literal sets is how a fourth verb ends up
    routed by one half and refused by the other.
    """
    from local_operator.session.frontend_state import _MCP_GRANT_SUBCOMMANDS

    assert set(_MCP_GRANT_SUBCOMMANDS) == set(GRANT_SUBCOMMANDS)


@pytest.mark.asyncio
async def test_a_local_client_runs_the_grant_instead_of_being_refused() -> None:
    """The regression: a loopback client is ON the machine that stores creds."""
    manager = _Manager()
    session = _Session(manager)
    spawned: list[Any] = []

    text, kind = await start_grant(
        session,
        "reauth",
        "notion",
        browser_is_reachable=True,
        notify=lambda body, style: None,
        spawn=spawned.append,
    )

    assert "run it from a terminal on that machine" not in text
    assert "authorizing MCP server 'notion'" in text
    assert kind == "info"
    # The interactive half was actually started, not merely reported.
    assert len(spawned) == 1
    await spawned[0]
    assert manager.connected == [("notion", 600_000)]


@pytest.mark.asyncio
async def test_a_remote_client_still_gets_the_locality_refusal() -> None:
    """The refusal survives, aimed at the case it actually describes."""
    manager = _Manager()
    spawned: list[Any] = []

    text, kind = await start_grant(
        _Session(manager),
        "login",
        "notion",
        browser_is_reachable=False,
        notify=lambda body, style: None,
        spawn=spawned.append,
    )

    assert text == REMOTE_GRANT_NOTICE.format(sub="login")
    assert kind == "warning"
    # Nothing ran: no browser opened and no credential was touched.
    assert spawned == []
    assert manager.connected == []
    assert manager.disconnected == []


@pytest.mark.asyncio
async def test_start_grant_returns_before_the_browser_exchange_settles() -> None:
    """The performance contract: the receipt must not wait on a human.

    An attach client abandons a request after ``ACK_TIMEOUT_S`` (15 s) and the
    runtime's per-connection reader is serial, so a grant awaited inline would
    time out the caller AND park every other op on that connection behind a
    browser tab.
    """
    manager = _Manager()
    manager.connect_gate = asyncio.Event()  # the "human" has not acted yet
    spawned: list[Any] = []

    text, kind = await asyncio.wait_for(
        start_grant(
            _Session(manager),
            "login",
            "notion",
            browser_is_reachable=True,
            notify=lambda body, style: None,
            spawn=spawned.append,
        ),
        timeout=1.0,
    )

    assert kind == "info"
    assert "authorizing" in text
    assert manager.connected == []  # still blocked on the gate

    task = asyncio.ensure_future(spawned[0])
    manager.connect_gate.set()
    await task
    assert manager.connected == [("notion", 600_000)]


@pytest.mark.asyncio
async def test_the_settled_outcome_is_reported_through_notify() -> None:
    manager = _Manager()
    notices: list[tuple[str, str]] = []

    _text, _kind = await start_grant(
        _Session(manager),
        "login",
        "notion",
        browser_is_reachable=True,
        notify=lambda body, style: notices.append((body, style)),
        spawn=lambda coro: asyncio.ensure_future(coro),
    )
    await asyncio.sleep(0)  # let the detached task run
    await asyncio.sleep(0)

    assert notices == [("authenticated MCP server 'notion'; 3 tools available.", "success")]


@pytest.mark.asyncio
async def test_a_failed_grant_reports_rather_than_raising() -> None:
    manager = _Manager()
    manager.connect_error = RuntimeError("connection refused")
    notices: list[tuple[str, str]] = []

    await start_grant(
        _Session(manager),
        "login",
        "notion",
        browser_is_reachable=True,
        notify=lambda body, style: notices.append((body, style)),
        spawn=lambda coro: asyncio.ensure_future(coro),
    )
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert notices[0][1] == "error"
    assert "connection refused" in notices[0][0]


@pytest.mark.asyncio
async def test_logout_is_awaited_inline_because_nothing_blocks_on_a_human(
    _no_real_credential_writes: list[str],
) -> None:
    manager = _Manager()
    spawned: list[Any] = []

    text, kind = await start_grant(
        _Session(manager),
        "logout",
        "notion",
        browser_is_reachable=True,
        notify=lambda body, style: None,
        spawn=spawned.append,
    )

    assert kind == "success"
    assert "logged out of MCP server 'notion'" in text
    assert spawned == []  # no detached task: the answer is already true
    assert _no_real_credential_writes == ["notion"]


@pytest.mark.asyncio
async def test_reauth_forgets_the_old_grant_before_reconnecting(
    _no_real_credential_writes: list[str],
) -> None:
    """Order matters: the stored row goes first, then the connection.

    Reversed, the manager's auto-reconnect reads the stored credential during
    the teardown window and quietly re-authenticates the session the user just
    reset.
    """
    manager = _Manager()
    text, kind = await run_grant(manager, "reauth", "notion")

    assert _no_real_credential_writes == ["notion"]
    assert manager.disconnected == ["notion"]
    assert manager.connected == [("notion", 600_000)]
    assert kind == "success"
    assert "authenticated" in text


@pytest.mark.asyncio
async def test_an_ineligible_server_is_refused_before_anything_is_deleted() -> None:
    """A server that takes no OAuth grant must not lose its credential."""
    manager = _Manager(supports=False)
    spawned: list[Any] = []

    text, kind = await start_grant(
        _Session(manager),
        "reauth",
        "notion",
        browser_is_reachable=True,
        notify=lambda body, style: None,
        spawn=spawned.append,
    )

    assert text == "MCP server 'notion' does not use OAuth login."
    assert kind == "warning"
    assert spawned == []
    assert manager.disconnected == []


def test_resolve_server_refuses_a_session_without_mcp() -> None:
    assert resolve_server(_Session(None), "notion") == "MCP is not available in this session."


def test_resolve_server_names_an_unconfigured_server() -> None:
    class _Empty(_Manager):
        def get_server_config(self, name: str) -> Any:
            return None

    result = resolve_server(_Session(_Empty()), "notion")
    assert result == "MCP server 'notion' is not configured — see /mcp"
