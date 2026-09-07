"""The canonical sync waits on LIVENESS, not on a wall clock.

The incident: booting `/new` on v0.51.6 printed

    ! could not start a runtime for this session: owner did not send frontend
      synchronization

from the one ``TimeoutError`` arm of ``RemoteSession._await_frontend``, which
wrapped the sync future in ``asyncio.wait_for(..., timeout=15.0)``. Two
mechanisms reach that line and neither is a broken runtime:

1. **The false timeout.** ``wait_for`` compares against a WALL clock, so a
   VIEWER whose own loop is blocked past the deadline trips it even though the
   frame already arrived on the socket. The viewer blamed the owner for its own
   stall, discarded a healthy connection, and the TUI reported a boot failure.
2. **The 15 s-vs-45 s liveness window.** ``HEARTBEAT_TIMEOUT_S`` is 45 s and the
   heartbeat is republished by the SAME loop that writes the sync, so an owner
   whose authoritative loop is busy 15-45 s (a turn in progress) still reads
   ``live``, still accepts the socket, still delivers the welcome in
   milliseconds — and cannot produce the sync inside 15 s.

Every test here is STRUCTURAL. Per ``AGENTS.md`` ("prefer a structural
invariant to a numeric one", "calibrate ceilings from CI, never from your
laptop") none of them asserts an elapsed-time bound on the success path, and no
number in this file was calibrated on a developer box.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from local_operator.session import remote as remote_module
from local_operator.session.frontend_state import FrontendSync
from local_operator.session.remote import FRONTEND_SYNC_SETTLE_TURNS, RemoteSession
from local_operator.session.runtime import registry
from local_operator.session.runtime.server import RuntimeServer
from tests.unit.session.runtime.test_server import FakeHandle


async def _record(root: Path):  # noqa: ANN202
    for _ in range(200):
        rows = registry.scan(root)
        if rows and rows[0][1] == "live":
            return rows[0][0]
        await asyncio.sleep(0.02)
    raise AssertionError("record did not publish")


async def _never():
    raise AssertionError("takeover was not expected")


def _claim(tmp_path: Path) -> None:
    """Make ``find_owner_record`` resolve this test's in-process server.

    ``find_owner_record`` starts from ``live_session_owner``, which reads the
    session directory's ``.session.pid`` marker — a real TUI writes it when it
    claims the directory. The tests below run the ``RuntimeServer`` inside the
    test process, so this process IS the owner.
    """
    import os

    session = tmp_path / "sessions" / "s1"
    session.mkdir(parents=True, exist_ok=True)
    (session / ".session.pid").write_text(str(os.getpid()), encoding="utf-8")


def _viewer(tmp_path: Path) -> RemoteSession:
    """A bare facade, constructed without dialling anything.

    ``_await_frontend`` is a pure function of the future it is handed and the
    socket's liveness, so the wait can be exercised without a server for the
    cases that are about the WAIT rather than about the protocol.
    """
    return RemoteSession(
        config_dir=tmp_path,
        session_id="s1",
        takeover_factory=_never,
    )


def _sync() -> FrontendSync:
    state = FakeHandle()._frontend.state
    return FrontendSync(epoch=state.epoch, sequence=state.sequence, snapshot=state)


# --------------------------------------------------------------------------
# 1. The false timeout (the regression this change exists for)
# --------------------------------------------------------------------------


async def _stalled_viewer_scenario(
    viewer: RemoteSession | None,
    *,
    deadline: float,
) -> tuple[str, asyncio.Future[FrontendSync]]:
    """The reported bug, over a REAL socket, in the shape that reproduces it.

    Mirrors the architect's probe exactly, because the bug lives in the
    interaction and not in any one object: an owner writes the sync frame
    PROMPTLY (well inside ``deadline``), and the viewer's loop is then blocked
    synchronously past ``deadline`` with a bare ``time.sleep`` — a Textual
    repaint, a GC pause, or CPU starvation on a loaded box all have this shape.
    The bytes land in the socket buffer during the block, so the pump has not
    run when the wall clock expires.

    ``viewer is None`` runs the PRE-FIX shape (a bare ``wait_for``) so the same
    scenario can be replayed against the code this change replaced.

    Returns the verdict and the future, so a caller can assert both what the
    wait reported AND whether the data was in fact already there.
    """
    loop = asyncio.get_running_loop()
    future: asyncio.Future[FrontendSync] = loop.create_future()
    expected = _sync()
    # The frame is due at a fifth of the deadline: the owner is healthy and
    # early by a wide margin, so nothing about this scenario is a slow owner.
    write_at = deadline / 5

    async def owner(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            await reader.readline()
            await asyncio.sleep(write_at)
            writer.write(b"frontend_sync\n")
            await writer.drain()
        finally:
            # The handler owns this transport, and from 3.12 ``wait_closed``
            # waits for every handler's connection as well as the listener.
            # Leaving it open parks the teardown in ``select`` forever.
            writer.close()

    server = await asyncio.start_server(owner, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    verdict = "no verdict"
    writer: asyncio.StreamWriter | None = None
    pumping: asyncio.Task[None] | None = None
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(b"auth\n")
        await writer.drain()

        async def pump() -> None:
            # AttachClient._pump -> RemoteSession._on_frontend_sync
            await reader.readline()
            if not future.done():
                future.set_result(expected)

        pumping = loop.create_task(pump())

        async def waiter() -> str:
            try:
                if viewer is None:
                    await asyncio.wait_for(asyncio.shield(future), timeout=deadline)
                else:
                    await viewer._await_frontend(future, timeout=deadline)
                return "synced"
            except TimeoutError:
                # The pre-fix path only. Establish, HERE — while the socket is
                # still up and the pump still alive — that the expiry was
                # FALSE: the frame resolves within the same bounded turns the
                # fixed code grants. Asserting this after the helper returns
                # would race its teardown, and a pump cancelled by teardown
                # looks identical to a frame that never came.
                turns = 0
                while not future.done() and turns < FRONTEND_SYNC_SETTLE_TURNS:
                    await asyncio.sleep(0)
                    turns += 1
                return f"TimeoutError (frame present after {turns} turns)" if future.done() else (
                    "TimeoutError (frame genuinely absent)"
                )
            except ConnectionError as error:
                return f"ConnectionError: {error}"

        waiting = loop.create_task(waiter())
        await asyncio.sleep(0.02)
        # Block the VIEWER's loop straight through the deadline. The frame is
        # already due; nothing can observe it until this returns.
        time.sleep(deadline * 2.5)
        verdict = await waiting
    finally:
        # Cancel the pump before closing the writer: it is parked in
        # ``readline`` on that transport. The owner handler closes its own
        # side (see above), so nothing here can outlive the probe.
        if pumping is not None:
            pumping.cancel()
        if writer is not None:
            writer.close()
        server.close()
    return verdict, future


@pytest.mark.asyncio
async def test_a_frame_that_arrived_during_a_viewer_stall_is_not_reported_as_a_timeout(
    tmp_path: Path,
) -> None:
    """The load-bearing regression test. Remove the settle turns and it goes red.

    Reproduces the confirmed bug over a real socket: the owner writes the sync
    on time, the viewer's own loop is blocked past the deadline, and the
    pre-fix code raised ``TimeoutError`` with ``future.done() is False`` —
    then found it done 0.05 s later. The viewer blamed the owner for its own
    stall.

    The assertion is on the RESULT, never on an elapsed time: the stall is a
    fixed multiple of the deadline, so machine load cannot make the scenario
    stop being the scenario, and no number here is calibrated on any box.
    """
    viewer = _viewer(tmp_path)
    # A live socket keeps the wait in the "owner is reachable" regime. A dead
    # client is a different (fast-fail) path, covered below.
    viewer._client = _LiveClientStub()

    verdict, future = await _stalled_viewer_scenario(viewer, deadline=0.2)

    assert verdict == "synced", (
        f"the wait reported {verdict!r}; the sync frame had already been "
        "delivered when the deadline expired, and reporting that as an owner "
        "failure is the reported bug"
    )
    assert future.done()


@pytest.mark.asyncio
async def test_the_old_wait_for_shape_fails_this_exact_scenario() -> None:
    """PROVE THE TEST CAN FAIL (``AGENTS.md``), permanently rather than by hand.

    The guard above is only believable if the scenario genuinely defeats the
    code it replaced. Rather than asking a future maintainer to re-edit
    ``_await_frontend`` and watch it go red, this replays the SAME helper
    against the pre-fix shape — ``asyncio.wait_for`` with no settle turns —
    and asserts it still trips. If the scenario ever stops reproducing the
    bug, this test fails and the one above is revealed as vacuous.
    """
    verdict, _future = await _stalled_viewer_scenario(None, deadline=0.2)

    assert verdict.startswith("TimeoutError"), (
        f"the pre-fix shape reported {verdict!r} rather than timing out; the "
        "scenario no longer reproduces the bug, so the guard above proves "
        "nothing"
    )
    # And the timeout was FALSE, not merely early: the frame is already there,
    # recoverable inside the same bounded turns the fixed wait grants. Without
    # this the test would also pass against a genuinely silent owner, which is
    # a different scenario and not the bug.
    assert "frame present after" in verdict, (
        f"the pre-fix shape reported {verdict!r}; the frame must already be on "
        "the socket for this to be the false-timeout bug rather than a real one"
    )


@pytest.mark.asyncio
async def test_settling_is_bounded_so_a_silent_owner_still_fails(tmp_path: Path) -> None:
    """The settle turns must not become an unbounded wait.

    A viewer that never gives up on a wedged owner is the ``#401`` failure
    mode: the TUI hangs with no verdict. The backstop stays finite, and the
    message says what is actually known rather than asserting something about
    the owner's behaviour the viewer cannot observe.
    """
    viewer = _viewer(tmp_path)
    future: asyncio.Future[FrontendSync] = asyncio.get_running_loop().create_future()
    viewer._client = _LiveClientStub()

    with pytest.raises(ConnectionError) as caught:
        await viewer._await_frontend(future, timeout=0.01)
    assert str(caught.value) == remote_module._SYNC_UNRESPONSIVE_REASON
    assert "did not send" not in str(caught.value), (
        "the viewer cannot observe what the owner sent; the old sentence "
        "asserted a fault it had no evidence for"
    )


# --------------------------------------------------------------------------
# 2. Fast failure must survive (the main regression risk of this change)
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_dead_socket_reports_the_pumps_reason_not_the_backstop(
    tmp_path: Path,
) -> None:
    """A connection that dies must resolve through the pump, in milliseconds.

    This is the risk the liveness wait creates: a generous backstop that also
    swallowed real failures would turn every dropped socket into a two-minute
    silence. It does not, by construction — ``_dial``'s ``on_disconnected``
    fails the future with the pump's own reason — and the assertion is on the
    REASON, never on elapsed time.

    The backstop is set absurdly high here on purpose: if the failure were
    reaching the clock instead of the pump, this test would hang rather than
    pass, which is a stronger signal than a timing bound.
    """
    future: asyncio.Future[FrontendSync] = asyncio.get_running_loop().create_future()
    viewer = _viewer(tmp_path)
    viewer._client = _LiveClientStub()

    async def kill_it() -> None:
        await asyncio.sleep(0)
        future.set_exception(ConnectionError("owner sent a frame too large to read"))

    asyncio.get_running_loop().create_task(kill_it())
    with pytest.raises(ConnectionError) as caught:
        await viewer._await_frontend(future, timeout=3600.0)
    assert "too large" in str(caught.value)


@pytest.mark.asyncio
async def test_an_unreadable_frame_over_a_real_socket_still_fails_fast(
    tmp_path: Path, monkeypatch
) -> None:
    """The same guarantee end to end, against the REAL server and socket.

    ``test_attach_frame_size`` already owns this case for the old code; it is
    repeated here against the new wait because the liveness gate is exactly the
    kind of change that converts a fast, named failure into a long silence. A
    real ``RuntimeServer`` writes a genuinely oversized frame, and the viewer
    must still come back with the pump's reason.
    """
    from local_operator.session.runtime.server import _MAX_LINE_BYTES

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions" / "s1").mkdir(parents=True)
    handle = FakeHandle()
    handle._frontend.mutate(cwd="x" * (_MAX_LINE_BYTES + 1024))
    server = RuntimeServer(handle, kind="tui")
    server.start()
    try:
        record = await _record(tmp_path)
        with pytest.raises(ConnectionError) as caught:
            await RemoteSession.connect(record, "s1", config_dir=tmp_path, takeover_factory=_never)
        assert "too large" in str(caught.value)
    finally:
        server.close()


# --------------------------------------------------------------------------
# 3. Future identity (the latent seam)
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_wait_uses_the_dials_own_future_not_the_current_attribute(
    tmp_path: Path,
) -> None:
    """A concurrent discard must not redirect or cancel an in-flight wait.

    ``_discard_rejected_client`` nulls ``_frontend_future`` and a redial
    replaces it. While ``_await_frontend`` re-read the attribute, either could
    make a healthy wait fail with "owner did not start frontend
    synchronization" (the reported sentence's sibling) or silently await a
    DIFFERENT dial's future. ``_ensure_bound`` holds ``_bind_lock`` and
    ``_recover_owner`` does not take it at all, so nothing else orders them.

    This is a fact about DATA FLOW, not about timing.
    """
    viewer = _viewer(tmp_path)
    loop = asyncio.get_running_loop()
    mine: asyncio.Future[FrontendSync] = loop.create_future()
    viewer._frontend_future = mine
    viewer._client = _LiveClientStub()
    expected = _sync()

    waiting = loop.create_task(viewer._await_frontend(mine, timeout=30.0))
    await asyncio.sleep(0)

    # A concurrent discard, exactly as _discard_rejected_client performs it.
    viewer._frontend_future = None
    await asyncio.sleep(0)
    assert not waiting.done(), "nulling the attribute must not disturb an in-flight wait"

    # And a redial replacing it must not redirect the wait either.
    other: asyncio.Future[FrontendSync] = loop.create_future()
    viewer._frontend_future = other
    other.set_result(_sync())
    await asyncio.sleep(0)
    assert not waiting.done(), "the wait must not resolve from another dial's future"

    mine.set_result(expected)
    assert await waiting is expected


# --------------------------------------------------------------------------
# 4. Bounded bind retry and connection hygiene
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_bind_that_fails_once_then_succeeds_leaves_one_connection(
    tmp_path: Path, monkeypatch
) -> None:
    """A transient bind failure becomes invisible, without leaking a socket.

    The retry's whole risk is connection hygiene: ``ATTACH_MAX_CLIENTS`` is 4,
    and a retry that leaked would evict real viewers through the runtime's LRU
    (the 272-eviction burst ``_discard_rejected_client`` documents). Counted
    directly on the server rather than inferred from timing.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _claim(tmp_path)
    handle = FakeHandle()
    server = RuntimeServer(handle, kind="tui")
    server.start()
    try:
        await _record(tmp_path)
        viewer = await RemoteSession.cold(
            "s1", config_dir=tmp_path, cwd=str(tmp_path), takeover_factory=_never
        )
        # engage_runtime must not spawn anything: the record already exists.
        monkeypatch.setattr(
            "local_operator.session.runtime.launch.engage_runtime",
            _no_engage,
        )

        real_bind = viewer._bind_to
        attempts = 0

        async def flaky_bind(record, *, sync_timeout):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                # Fail the way a busy owner does — AFTER the dial, so the
                # discard path is the one under test.
                pending = await viewer._dial(record)
                assert pending is not None
                viewer._discard_rejected_client()
                raise ConnectionError(remote_module._SYNC_UNRESPONSIVE_REASON)
            await real_bind(record, sync_timeout=sync_timeout)

        monkeypatch.setattr(viewer, "_bind_to", flaky_bind)
        await viewer._ensure_bound()

        assert attempts == 2, "the first failure must be retried, not surfaced"
        assert not viewer.is_cold, "the retry must end in a bound viewer"
        assert len(server._clients) == 1, (
            f"{len(server._clients)} connections survived a retried bind; each "
            "attempt must discard its rejected client (ATTACH_MAX_CLIENTS is 4)"
        )
        await viewer.dispose()
    finally:
        server.close()


@pytest.mark.asyncio
async def test_a_disposed_facade_stops_retrying_immediately(
    tmp_path: Path, monkeypatch
) -> None:
    """Disposal mid-retry must end the loop, never bind a socket to a dead facade.

    `/new` or `/resume` typed during a retry disposes the facade. Binding
    anyway attaches a live socket to a viewer nobody owns, which pins the old
    runtime resident and never offers it back (review round 1, MAJOR-1). The
    guard is re-checked per attempt, not once on entry, because every await in
    the loop is a point at which disposal can land.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _claim(tmp_path)
    handle = FakeHandle()
    server = RuntimeServer(handle, kind="tui")
    server.start()
    try:
        await _record(tmp_path)
        viewer = await RemoteSession.cold(
            "s1", config_dir=tmp_path, cwd=str(tmp_path), takeover_factory=_never
        )
        monkeypatch.setattr(
            "local_operator.session.runtime.launch.engage_runtime",
            _no_engage,
        )

        attempts = 0

        async def dispose_on_first(record, *, sync_timeout):
            nonlocal attempts
            attempts += 1
            viewer._disposed = True
            raise ConnectionError(remote_module._SYNC_UNRESPONSIVE_REASON)

        monkeypatch.setattr(viewer, "_bind_to", dispose_on_first)
        await viewer._ensure_bound()

        assert attempts == 1, "a disposed facade must not attempt another bind"
        assert not server._clients, "no socket may be left attached to a dead facade"
    finally:
        server.close()


@pytest.mark.asyncio
async def test_the_retry_rediscovers_the_record_each_attempt(
    tmp_path: Path, monkeypatch
) -> None:
    """A runtime that retires and respawns between attempts must be picked up.

    Reusing the first record would redial a dead pid for every remaining
    attempt and burn the whole budget on a socket that cannot answer. Asserted
    structurally: the discovery call is counted, not timed.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _claim(tmp_path)
    handle = FakeHandle()
    server = RuntimeServer(handle, kind="tui")
    server.start()
    try:
        record = await _record(tmp_path)
        viewer = await RemoteSession.cold(
            "s1", config_dir=tmp_path, cwd=str(tmp_path), takeover_factory=_never
        )
        monkeypatch.setattr(
            "local_operator.session.runtime.launch.engage_runtime",
            _no_engage,
        )

        lookups = 0

        def counting_find(config_dir, session_id):
            nonlocal lookups
            lookups += 1
            return record, None

        monkeypatch.setattr(
            "local_operator.mobile.attach_client.find_owner_record", counting_find
        )

        async def always_fail(record, *, sync_timeout):
            raise ConnectionError(remote_module._SYNC_UNRESPONSIVE_REASON)

        monkeypatch.setattr(viewer, "_bind_to", always_fail)
        with pytest.raises(ConnectionError):
            await viewer._ensure_bound()

        assert lookups == remote_module._BIND_RETRY_ATTEMPTS, (
            f"the record was read {lookups} times for "
            f"{remote_module._BIND_RETRY_ATTEMPTS} attempts; a respawned runtime "
            "publishes a new record under a new pid and must be rediscovered"
        )
        await viewer.dispose()
    finally:
        server.close()


# --------------------------------------------------------------------------
# 5. The two envelopes (budgets compose)
# --------------------------------------------------------------------------


def test_a_foreground_bind_never_inherits_the_background_backstop() -> None:
    """A user waiting on a command must not sit through the catastrophe budget.

    The budgets COMPOSE: today's worst foreground path is a 30 s engage plus a
    15 s welcome ack AHEAD of this wait. Handing the generous backstop to a
    foreground caller would make the wait this change exists to shorten longer
    instead — the architect's strongest caveat, and the one thing here that a
    reader cannot check by eye once the constants drift apart.

    Structural: an ordering fact between named constants, with no clock.
    """
    assert (
        remote_module.FRONTEND_SYNC_FOREGROUND_S < remote_module.FRONTEND_SYNC_BACKSTOP_S
    ), "the foreground envelope must stay the short one"
    assert (
        remote_module.FRONTEND_SYNC_FOREGROUND_S <= 15.0
    ), "a foreground sync wait must not exceed what it already was before this change"
    assert (
        remote_module._FOREGROUND_BIND_BUDGET_S <= remote_module.FRONTEND_SYNC_FOREGROUND_S
    ), "retries must fit INSIDE the pre-fix envelope, never extend it"
    # And the backstop stays FINITE: a wedged owner must fail, not hang (#401).
    assert 0 < remote_module.FRONTEND_SYNC_BACKSTOP_S < float("inf")


@pytest.mark.asyncio
async def test_the_foreground_default_is_the_short_envelope(
    tmp_path: Path, monkeypatch
) -> None:
    """``_ensure_bound`` defaults to the SHORT envelope.

    The default matters more than the parameter: a caller that forgets to pass
    one is far likelier to be a foreground path than a silent engage, so the
    safe default is the one that cannot hand a user a two-minute wait.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _claim(tmp_path)
    handle = FakeHandle()
    server = RuntimeServer(handle, kind="tui")
    server.start()
    try:
        await _record(tmp_path)
        viewer = await RemoteSession.cold(
            "s1", config_dir=tmp_path, cwd=str(tmp_path), takeover_factory=_never
        )
        monkeypatch.setattr(
            "local_operator.session.runtime.launch.engage_runtime",
            _no_engage,
        )
        seen: list[float] = []

        async def record_timeout(record, *, sync_timeout):
            seen.append(sync_timeout)
            raise ConnectionError("stop here")

        monkeypatch.setattr(viewer, "_bind_to", record_timeout)
        with pytest.raises(ConnectionError):
            await viewer._ensure_bound()
        # Never MORE than the foreground envelope. Attempts after the first are
        # clamped further, to whatever is left of the budget, which is what
        # keeps three attempts from composing into three envelopes.
        assert seen, "the bind must have been attempted"
        assert max(seen) <= remote_module.FRONTEND_SYNC_FOREGROUND_S
        assert seen == sorted(seen, reverse=True), (
            f"attempt envelopes {seen} must be non-increasing: each is bounded "
            "by the budget the previous attempts left"
        )

        seen.clear()
        viewer._ready_for_events = False
        with pytest.raises(ConnectionError):
            await viewer._ensure_bound(foreground=False)
        assert seen and max(seen) <= remote_module.FRONTEND_SYNC_BACKSTOP_S
        assert min(seen) > remote_module.FRONTEND_SYNC_FOREGROUND_S, (
            "a background bind must get the generous envelope, not the short one"
        )
        await viewer.dispose()
    finally:
        server.close()


class _LiveClientStub:
    """Just enough of ``AttachClient`` for the wait's liveness question."""

    connected = True

    def close(self) -> None:  # pragma: no cover - not reached by these tests
        pass


async def _no_engage(session_id, cwd, work, *, config_dir, deadline_s=30.0):
    """A record already exists; engaging must not spawn a second runtime."""
    return None
