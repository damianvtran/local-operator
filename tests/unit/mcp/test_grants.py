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
async def test_an_ineligible_server_never_loses_its_credential(
    _no_real_credential_writes: list[str],
) -> None:
    """A server that takes no OAuth grant must not be logged out by a reauth.

    The eligibility probe moved into the detached task (F2/Q1), so the refusal
    arrives as a notice rather than as the receipt. What must NOT move is the
    ordering: the probe still gates the credential deletion, or `/mcp reauth`
    on an api-key server would destroy a working grant and then decline to
    replace it.
    """
    manager = _Manager(supports=False)
    notices: list[tuple[str, str]] = []

    await start_grant(
        _Session(manager),
        "reauth",
        "notion",
        browser_is_reachable=True,
        notify=lambda body, style: notices.append((body, style)),
        spawn=lambda coro: asyncio.ensure_future(coro),
    )
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert notices == [("MCP server 'notion' does not use OAuth login.", "warning")]
    assert _no_real_credential_writes == [], "an ineligible server lost its credential"
    assert manager.disconnected == []
    assert manager.connected == []


def test_resolve_server_refuses_a_session_without_mcp() -> None:
    assert resolve_server(_Session(None), "notion") == "MCP is not available in this session."


def test_resolve_server_names_an_unconfigured_server() -> None:
    class _Empty(_Manager):
        def get_server_config(self, name: str) -> Any:
            return None

    result = resolve_server(_Session(_Empty()), "notion")
    assert result == "MCP server 'notion' is not configured — see /mcp"


@pytest.mark.asyncio
async def test_the_capability_probe_does_not_block_the_receipt() -> None:
    """F2/Q1: the OAuth capability probe must run INSIDE the detached task.

    The probe is up to three sequential 10 s HTTP discovery GETs with no total
    cap, on exactly the url-only config it exists for. Awaited before the spawn
    it measured 30.7 s against an unroutable host — past the invoker's 15 s
    ACK_TIMEOUT_S, and holding the runtime's serial reader the whole time.

    The original test suite could not see this: its manager answered the probe
    instantly, so only the half AFTER the spawn was ever gated.
    """
    probing = asyncio.Event()

    class _SlowProbeManager(_Manager):
        async def server_supports_oauth_login(self, cfg: Any) -> bool:
            probing.set()
            await asyncio.sleep(3600)  # the unroutable host
            return True

    manager = _SlowProbeManager()
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
        timeout=1.0,  # far inside ACK_TIMEOUT_S; fails outright if it regresses
    )

    assert kind == "info"
    assert len(spawned) == 1
    # The probe has not even started: it belongs to the task, not the request.
    assert not probing.is_set()

    task = asyncio.ensure_future(spawned[0])
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert probing.is_set(), "the probe never ran in the detached task"
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_an_ineligible_server_is_reported_through_notify() -> None:
    """The probe moved, so its refusal is now a notice rather than a receipt.

    It must still be SAID — a server that takes no OAuth grant cannot silently
    look like a grant in progress.
    """
    manager = _Manager(supports=False)
    notices: list[tuple[str, str]] = []

    text, _kind = await start_grant(
        _Session(manager),
        "login",
        "notion",
        browser_is_reachable=True,
        notify=lambda body, style: notices.append((body, style)),
        spawn=lambda coro: asyncio.ensure_future(coro),
    )
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert "authorizing" in text
    assert notices == [("MCP server 'notion' does not use OAuth login.", "warning")]
    assert manager.connected == []


@pytest.mark.asyncio
async def test_the_receipt_does_not_promise_a_browser_it_may_not_open() -> None:
    """Whether a tab opens is decided by the probe, after the receipt is sent."""
    manager = _Manager(supports=False)
    text, _ = await start_grant(
        _Session(manager),
        "login",
        "notion",
        browser_is_reachable=True,
        notify=lambda body, style: None,
        spawn=lambda coro: asyncio.ensure_future(coro),
    )
    assert "browser tab is opening" not in text


@pytest.mark.asyncio
async def test_a_reauth_cancelled_after_its_delete_says_the_grant_is_gone(
    _no_real_credential_writes: list[str],
) -> None:
    """F6: `reauth` is destructive before it is constructive.

    Cancelled between the delete and the reconnect — by a superseding grant or
    by session teardown — the server is left with no credential at all. A bare
    "cancelled" reads as "nothing changed" and sends the user back to a server
    that can no longer connect, so the notice must name the state and the
    recovery.
    """
    manager = _Manager()
    manager.connect_gate = asyncio.Event()  # never fires: cancel mid-grant
    notices: list[tuple[str, str]] = []

    await start_grant(
        _Session(manager),
        "reauth",
        "notion",
        browser_is_reachable=True,
        notify=lambda body, style: notices.append((body, style)),
        spawn=lambda coro: asyncio.ensure_future(coro),
    )
    task = next(t for t in asyncio.all_tasks() if "_settle" in str(t.get_coro()))
    for _ in range(6):  # let it reach the connect
        await asyncio.sleep(0)
    assert _no_real_credential_writes == ["notion"], "the delete should have happened"

    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    assert notices, "a cancelled grant must still report"
    body, style = notices[-1]
    assert "unauthenticated" in body, body
    assert "/mcp login notion" in body, body
    assert style == "warning"


@pytest.mark.asyncio
async def test_a_grant_cancelled_before_its_delete_says_nothing_changed() -> None:
    """The other half of F6: don't alarm a user whose credential is intact."""
    manager = _Manager()
    probing = asyncio.Event()

    class _SlowProbe(_Manager):
        async def server_supports_oauth_login(self, cfg: Any) -> bool:
            probing.set()
            await asyncio.sleep(3600)
            return True

    manager = _SlowProbe()
    notices: list[tuple[str, str]] = []

    await start_grant(
        _Session(manager),
        "reauth",
        "notion",
        browser_is_reachable=True,
        notify=lambda body, style: notices.append((body, style)),
        spawn=lambda coro: asyncio.ensure_future(coro),
    )
    task = next(t for t in asyncio.all_tasks() if "_settle" in str(t.get_coro()))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert probing.is_set()

    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    body, _style = notices[-1]
    assert "this attempt changed nothing" in body, body
    assert "unauthenticated" not in body, body
    # It speaks only about THIS grant. Claiming the server's stored credential
    # is intact would assert something this scope cannot know (QA Q2).
    assert "the stored credential is unchanged" not in body, body


@pytest.mark.asyncio
async def test_the_last_notice_never_claims_a_deleted_credential_is_intact(
    _no_real_credential_writes: list[str],
) -> None:
    """Q2: `forgotten` is scoped to ONE grant, but the user reads the server.

    Reachable in three ordinary actions: `/mcp reauth n` deletes and parks, the
    user retries with `/mcp login n` (superseding it), and that login is itself
    cancelled. The second grant deleted nothing, so a notice claiming "the
    stored credential is unchanged" would be the LAST thing the user reads
    while the server has no credential at all — defeating the point of F6.

    Neither branch may make a claim about the server's overall state.
    """
    manager = _Manager()
    manager.connect_gate = asyncio.Event()  # both grants park
    notices: list[tuple[str, str]] = []
    session = _Session(manager)

    async def _run_and_cancel(sub: str) -> None:
        await start_grant(
            session,
            sub,
            "notion",
            browser_is_reachable=True,
            notify=lambda body, style: notices.append((body, style)),
            spawn=lambda coro: asyncio.ensure_future(coro),
        )
        task = next(t for t in asyncio.all_tasks() if "_settle" in str(t.get_coro()))
        for _ in range(6):
            await asyncio.sleep(0)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    await _run_and_cancel("reauth")  # deletes, then is cancelled
    assert _no_real_credential_writes == ["notion"]
    await _run_and_cancel("login")  # deletes nothing, then is cancelled

    # The credential really is gone, and this is the last line the user sees.
    last, _style = notices[-1]
    assert "the stored credential is unchanged" not in last, last
    # Whatever it says, it must leave the user with the step that fixes it.
    assert "/mcp login notion" in last, last
