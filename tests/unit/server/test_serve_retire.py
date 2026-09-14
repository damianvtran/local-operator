"""The serve daemon's build watch: announce, keep serving, then refuse and leave.

The daemon is the one participant on this host that used to have no rollout for
a new build — it kept serving the old one until somebody killed it. These tests
drive the three halves of the replacement separately, because they fail
differently: the POLL decides when to announce and when to leave, the PREDICATE
decides how long the daemon keeps serving, and the REFUSAL is what a client
sees once it has latched.

The poll is driven for real (with ``BUILD_CHECK_S`` shortened) rather than by
calling its internals, so the shared settle rule, the announce-then-drain order
and the exit are exercised as the lifespan starts them. ``update.installed_build``
and ``build_marker_age_s`` are faked the way
``tests/unit/session/runtime/test_process_refresh.py`` fakes them for the
runtime, so both watchers are pinned against the same rule.

THE TWO-PHASE SHAPE IS THE POINT OF THE ROUND-2 REWRITE, so most tests below
assert it in the same three lines: the announcement is readable in the record,
the daemon is NOT latched, and it has not exited. The hold that matters is the
desktop app's own relay — held through the ROUTE, not through a hand-built
``users=1`` bridge, which is exactly how the first round's tests missed it
(``test_a_desktop_relay_holds_an_announced_daemon_that_keeps_serving``).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import time
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from local_operator import buildwatch
from local_operator import update as update_mod
from local_operator.server import registry as serve_registry
from local_operator.server import retire
from local_operator.server.registry import ServeRecord
from local_operator.server.utils.desktop_sessions import (
    WATCH_TTL,
    DesktopSessionBridge,
    DesktopSessions,
)
from local_operator.server.utils.websocket_manager import WebSocketManager
from local_operator.update import BuildStamp

OLD = BuildStamp(version="0.54.30", source_ref="1111111")
NEW = BuildStamp(version="0.54.31", source_ref="2222222")

#: How long a test lets the poll run for, in seconds. The check interval is
#: shortened below; this is a multiple of it, so a test can assert on "several
#: intervals passed with no action" rather than on a single sample.
QUIET_S = 0.25


class FakeApp:
    """Exactly what the predicate and the latch touch: an object with ``state``.

    A stand-in rather than the module-level ``app`` singleton on purpose: these
    tests assert on the LATCH, and a shared app object would leak it between
    tests (the real lifespan clears it, which is itself pinned in
    ``test_serve_lifecycle``).
    """

    def __init__(self, **state: Any) -> None:
        self.state = SimpleNamespace(**state)


class FakePublisher:
    """The record plus the writes that rewrote it, in order."""

    def __init__(self, record: ServeRecord) -> None:
        self.record = record
        self.writes: list[dict[str, Any]] = []

    def heartbeat(self, **updates: object) -> None:
        for key, value in updates.items():
            setattr(self.record, key, value)
        self.writes.append(dict(updates))


def _record(**over: Any) -> ServeRecord:
    fields: dict[str, Any] = {
        "pid": 4242,
        "host": "127.0.0.1",
        "port": 53421,
        "instance_id": "instance",
        "version": OLD.version,
        "source_ref": OLD.source_ref,
        "prefix": "/tmp/prefix",
        "install_kind": "uv-tool",
        "desktop": False,
    }
    fields.update(over)
    return ServeRecord(**fields)


@pytest.fixture
def disk(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Control what the install on disk reports, and how fast the poll runs.

    Both watchers must read the same rule, so this fakes the same two
    ``update`` functions the runtime's tests fake and shortens the SHARED
    constants rather than any local copy of them.
    """
    state: dict[str, Any] = {"build": OLD, "age": 999.0}
    monkeypatch.setattr(update_mod, "installed_build", lambda *_a, **_k: state["build"])
    monkeypatch.setattr(update_mod, "build_marker_age_s", lambda *_a, **_k: state["age"])
    monkeypatch.delenv("LOP_BUILD_SETTLE_S", raising=False)
    monkeypatch.delenv("LOP_BUILD_STAGGER_S", raising=False)
    monkeypatch.delenv("LOP_BUILD_PREFIX", raising=False)
    monkeypatch.setattr(buildwatch, "BUILD_CHECK_S", 0.02)
    monkeypatch.setattr(buildwatch, "BUILD_STAGGER_S", 0.0)
    return state


async def _start(
    app: FakeApp,
    publisher: FakePublisher,
    exit_process: Any = None,
) -> tuple[asyncio.Task[None], asyncio.Event, list[bool]]:
    """Run the poll as the lifespan does, with the exit recorded instead taken."""
    stop = asyncio.Event()
    exited: list[bool] = []
    task = asyncio.create_task(
        retire.retirement_poll(
            app,  # type: ignore[arg-type] — the predicate is duck-typed by design
            publisher,  # type: ignore[arg-type]
            stop=stop,
            exit_process=exit_process or (lambda: exited.append(True)),
        )
    )
    return task, stop, exited


async def _wait_for_announcement(publisher: Any) -> None:
    """Block until the handover is readable in the record, or fail loudly.

    Bounded rather than a single sleep, and it never returns on a timeout: the
    announcement is the FIRST thing the poll does with a moved install, so a
    test that waits for it and finds nothing has found a real defect.

    Takes any publisher (the fake, or the real ``RecordPublisher`` a test drives
    over a real record file) because it only ever reads ``record.retiring_to`` —
    the reader's own question.
    """
    for _ in range(1000):
        if publisher.record.retiring_to:
            return
        await asyncio.sleep(0.005)
    raise AssertionError("the daemon never announced the handover")


def _announced_not_latched(app: FakeApp, publisher: FakePublisher, exited: list[bool]) -> None:
    """The round-2 invariant: in the record, and STILL SERVING NEW WORK.

    Three assertions rather than one because they are three different promises:
    a reader can see which build is coming, this process has NOT latched against
    new work, and it has not exited. The first version of this feature wrote the
    record only after the drain emptied, which is why both round-1 reviewers
    measured an announcement that never arrived under the app's own relay; the
    second promised the announcement while REFUSING everything, which is a
    daemon unusable for as long as its client took to react.
    """
    assert publisher.record.retiring_from == OLD.label()
    assert publisher.record.retiring_to == NEW.label()
    assert retire.retiring(cast(Any, app)) is False, "an announced daemon still admits work"
    assert exited == [], "an announced daemon is still serving"


# ---------------------------------------------------------------------------
# what the shared rule decides
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_same_build_is_never_a_retirement(disk: dict[str, Any]) -> None:
    app, publisher = FakeApp(), FakePublisher(_record())
    task, stop, exited = await _start(app, publisher)

    await asyncio.sleep(QUIET_S)
    assert publisher.writes == [], "an unchanged install is not a handover"
    assert exited == []
    assert retire.retiring(cast(Any, app)) is False

    stop.set()
    await asyncio.wait_for(task, 1.0)


@pytest.mark.asyncio
async def test_a_fresh_marker_waits_for_the_settle(
    disk: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The settle is the shared one: a half-written tree is not a new build."""
    app, publisher = FakeApp(), FakePublisher(_record())
    task, stop, exited = await _start(app, publisher)
    await asyncio.sleep(QUIET_S)  # the boot stamp is OLD; nothing has moved yet

    disk["build"], disk["age"] = NEW, 0.0  # just written: the installer may still run
    await asyncio.sleep(QUIET_S)
    assert publisher.writes == [], "a marker inside the settle window is not a move"
    assert exited == []

    stop.set()
    await asyncio.wait_for(task, 1.0)


@pytest.mark.asyncio
async def test_a_settled_new_build_announces_then_latches_then_exits(
    disk: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE SEQUENCE: announce while serving, latch once the drain is empty, leave.

    Each state is asserted at the moment it exists, because the ORDER is the
    design: a daemon that latched as soon as it announced would refuse work for
    as long as its client took to react, and one that announced only after the
    drain emptied is the circular version both round-1 reviewers reproduced by
    execution.
    """
    latched_at_exit: list[bool] = []
    exited: list[bool] = []
    monkeypatch.setattr(buildwatch, "BUILD_STAGGER_S", 0.3)  # a window to observe
    app, publisher = FakeApp(), FakePublisher(_record())

    def _exit() -> None:
        # Sampled AT the exit, which is the only moment that answers "did the
        # refusal precede the departure" without racing it.
        latched_at_exit.append(retire.retiring(cast(Any, app)))
        exited.append(True)

    task, stop, _ = await _start(app, publisher, exit_process=_exit)
    await asyncio.sleep(QUIET_S)  # several checks against the OLD stamp
    assert publisher.writes == []

    disk["build"] = NEW  # the install moves under the daemon
    await _wait_for_announcement(publisher)
    _announced_not_latched(app, publisher, exited)

    await asyncio.wait_for(task, 2.0)
    assert exited == [True], "the exit belongs to the process's own shutdown path"
    assert latched_at_exit == [True], "the refusal is latched BEFORE the exit is asked for"
    assert publisher.writes[-1] == {
        "retiring_from": OLD.label(),
        "retiring_to": NEW.label(),
    }, "a reader must see WHICH build it left for and which one is coming"
    assert retire.retiring(cast(Any, app)) is True


@pytest.mark.asyncio
async def test_a_stop_during_the_refusal_window_owns_the_exit(disk: dict[str, Any], monkeypatch):
    """A stop that lands mid-window leaves the exit to the stop path.

    Otherwise the record's last state would depend on which of two paths won a
    race: the daemon announcing a build change while uvicorn is already
    shutting down.
    """
    monkeypatch.setattr(buildwatch, "BUILD_STAGGER_S", 5.0)
    app, publisher = FakeApp(), FakePublisher(_record())
    task, stop, exited = await _start(app, publisher)
    await asyncio.sleep(QUIET_S)

    disk["build"] = NEW
    await _wait_for_announcement(publisher)
    stop.set()
    await asyncio.wait_for(task, 1.0)

    assert exited == [], "the stop path exits, not the poll"


@pytest.mark.asyncio
async def test_no_boot_stamp_means_no_watch(disk: dict[str, Any], monkeypatch) -> None:
    """An install that cannot prove a move is never retired onto one.

    An editable checkout has no marker of its own; a watcher with no baseline
    would either never fire or fire on noise.
    """

    def _unreadable(*_a: Any, **_k: Any) -> BuildStamp:
        raise RuntimeError("no dist-info")

    monkeypatch.setattr(update_mod, "installed_build", _unreadable)
    app, publisher = FakeApp(), FakePublisher(_record())
    task, stop, exited = await _start(app, publisher)

    await asyncio.sleep(QUIET_S)
    assert publisher.writes == [] and exited == []
    assert task.done(), "the poll returns instead of comparing against nothing"

    stop.set()


# ---------------------------------------------------------------------------
# the in-flight predicate: one case per term
# ---------------------------------------------------------------------------


def _live_bridge(root: Any, **over: Any) -> DesktopSessionBridge:
    """A bridge in the state ``acquire``/``watch`` leave it in."""
    bridge = DesktopSessionBridge(root, over.pop("session_id", "0123456789ab"), str(root))
    bridge.users = over.pop("users", 0)
    return bridge


@pytest.mark.asyncio
async def test_an_open_sse_stream_holds_the_daemon(disk: dict[str, Any]) -> None:
    """A turn's live feed dies with this process; the broker's replay does too."""
    broker = SimpleNamespace(stats=lambda: {"subscribers": 2})
    app, publisher = FakeApp(event_broker=broker), FakePublisher(_record())
    task, stop, exited = await _start(app, publisher)
    await asyncio.sleep(QUIET_S)

    disk["build"] = NEW  # the install moves while the turn is streaming
    await _wait_for_announcement(publisher)
    await asyncio.sleep(QUIET_S)  # several checks with the turn still streaming
    _announced_not_latched(app, publisher, exited)

    broker.stats = lambda: {"subscribers": 0}  # the turn's stream ends
    await asyncio.wait_for(task, 2.0)
    assert exited == [True]
    assert retire.retiring(cast(Any, app)) is True, "the latch follows the empty drain"


@pytest.mark.asyncio
async def test_an_open_websocket_holds_the_daemon(disk: dict[str, Any]) -> None:
    manager = WebSocketManager()
    app, publisher = FakeApp(websocket_manager=manager), FakePublisher(_record())
    task, stop, exited = await _start(app, publisher)
    await asyncio.sleep(QUIET_S)

    # The real manager's own structure, since the count reads it: one socket
    # subscribed to one message id, exactly what ``connect`` builds. A bare
    # ``object()`` stands in for the socket — nothing here touches it.
    manager.connections[next(iter(manager.connections))]["message-1"] = {cast(Any, object())}
    disk["build"] = NEW
    await _wait_for_announcement(publisher)
    await asyncio.sleep(QUIET_S)
    _announced_not_latched(app, publisher, exited)
    assert manager.live_connection_count() == 1

    manager.connections.clear()
    await asyncio.wait_for(task, 2.0)
    assert exited == [True]
    assert retire.retiring(cast(Any, app)) is True


@pytest.mark.asyncio
async def test_an_in_flight_desktop_request_holds_the_daemon(
    disk: dict[str, Any], tmp_path
) -> None:
    pool = DesktopSessions(tmp_path)
    bridge = _live_bridge(tmp_path, users=1)
    pool.bridges[bridge.session_id] = bridge
    app, publisher = FakeApp(desktop_sessions=pool), FakePublisher(_record())
    task, stop, exited = await _start(app, publisher)
    await asyncio.sleep(QUIET_S)

    disk["build"] = NEW
    await _wait_for_announcement(publisher)
    await asyncio.sleep(QUIET_S)
    _announced_not_latched(app, publisher, exited)
    assert pool.in_flight_reason() == "1 in-flight desktop request(s) on 0123456789ab"

    bridge.users = 0
    await asyncio.wait_for(task, 2.0)
    assert exited == [True]
    assert retire.retiring(cast(Any, app)) is True


@pytest.mark.asyncio
async def test_a_watching_desktop_window_holds_the_daemon(disk: dict[str, Any], tmp_path) -> None:
    """A live lease IS a viewer, and a viewer is never cut off.

    ``expires`` is written by ``watch`` (``time.monotonic() + WATCH_TTL``), so
    this sets the state that call leaves behind rather than inventing one.
    """
    pool = DesktopSessions(tmp_path)
    bridge = _live_bridge(tmp_path)
    sub = bridge.subscribe()
    sub.expires = time.monotonic() + WATCH_TTL
    pool.bridges[bridge.session_id] = bridge
    app, publisher = FakeApp(desktop_sessions=pool), FakePublisher(_record())
    task, stop, exited = await _start(app, publisher)
    await asyncio.sleep(QUIET_S)

    disk["build"] = NEW
    await _wait_for_announcement(publisher)
    await asyncio.sleep(QUIET_S)
    _announced_not_latched(app, publisher, exited)
    assert "watching session" in (pool.in_flight_reason() or "")

    sub.expires = 0.0  # the lease lapses
    await asyncio.wait_for(task, 2.0)
    assert exited == [True]
    assert retire.retiring(cast(Any, app)) is True


@pytest.mark.asyncio
async def test_a_runtime_mid_spawn_holds_the_daemon(disk: dict[str, Any], tmp_path) -> None:
    """A spawn is a handshake with a child; leaving mid-handshake strands it.

    ``warm`` answers the HTTP request before the engage settles (fire and
    forget), so ``users`` does NOT cover this.
    """
    pool = DesktopSessions(tmp_path)
    bridge = _live_bridge(tmp_path)
    bridge.warm_task = asyncio.create_task(asyncio.sleep(10))
    pool.bridges[bridge.session_id] = bridge
    app, publisher = FakeApp(desktop_sessions=pool), FakePublisher(_record())
    task, stop, exited = await _start(app, publisher)
    await asyncio.sleep(QUIET_S)

    disk["build"] = NEW
    await _wait_for_announcement(publisher)
    await asyncio.sleep(QUIET_S)
    _announced_not_latched(app, publisher, exited)
    assert "runtime being started" in (pool.in_flight_reason() or "")

    bridge.warm_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await bridge.warm_task
    await asyncio.wait_for(task, 2.0)
    assert exited == [True]
    assert retire.retiring(cast(Any, app)) is True


@pytest.mark.asyncio
async def test_an_idle_daemon_retires_once_every_term_is_clear(
    disk: dict[str, Any], tmp_path
) -> None:
    pool = DesktopSessions(tmp_path)
    pool.bridges["0123456789ab"] = _live_bridge(tmp_path)  # present but idle
    app = FakeApp(
        event_broker=SimpleNamespace(stats=lambda: {"subscribers": 0}),
        websocket_manager=WebSocketManager(),
        desktop_sessions=pool,
    )
    publisher = FakePublisher(_record())
    task, stop, exited = await _start(app, publisher)
    await asyncio.sleep(QUIET_S)

    disk["build"] = NEW
    await asyncio.wait_for(task, 2.0)
    assert exited == [True]
    assert retire.in_flight(cast(Any, app)) is None


# ---------------------------------------------------------------------------
# the refusal a client sees while retiring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_retiring_daemon_refuses_new_sessions(tmp_path) -> None:
    pool = DesktopSessions(tmp_path, retiring=lambda: True)
    with pytest.raises(retire.DaemonRetiring) as refusal:
        await pool.create(str(tmp_path))
    assert refusal.value.code == "daemon-retiring"
    assert not list((tmp_path / "sessions").glob("*")), "nothing was created"


@pytest.mark.asyncio
async def test_a_retiring_daemon_refuses_to_spawn_a_runtime(tmp_path) -> None:
    bridge = DesktopSessionBridge(tmp_path, "0123456789ab", str(tmp_path), retiring=lambda: True)
    with pytest.raises(retire.DaemonRetiring):
        await bridge.warm()


@pytest.mark.asyncio
async def test_a_retiring_daemon_still_answers_reads(tmp_path) -> None:
    """The announcement is readable until the exit removes the record."""
    pool = DesktopSessions(tmp_path, retiring=lambda: True)
    assert await pool.list(10) == []


@pytest.fixture
def restore_app_state() -> Iterator[None]:
    """Give the process the ``app`` state it had before this test.

    ``app`` is a module-level singleton: the wire tests below configure it the
    way the routes expect and the real ``lifespan`` sets a dozen attributes to
    ``None`` on its way out, which the rest of the worker's tests share. Blunt
    and total, like ``test_serve_lifecycle``'s fixture of the same name (a
    hand-written list is what goes stale as the lifespan grows), and taken
    through Starlette's mapping interface rather than ``vars(state)``.
    """
    from local_operator.server.app import app

    saved = {key: app.state[key] for key in app.state}
    state = app.state
    try:
        yield
    finally:
        for key in list(state):
            del state[key]
        for key, value in saved.items():
            state[key] = value


def test_the_create_route_refuses_with_a_503_and_no_receipt(
    tmp_path, monkeypatch, restore_app_state: None
) -> None:
    """Wire-level: a typed 503 and a code, never a 500 with a traceback.

    ``serve_retiring`` is set on the shared app the way the poll sets it; the
    refusal must arrive before the receipt is claimed, or a retry against the
    SUCCESSOR would meet the indeterminate 409.
    """
    from local_operator.server.app import app

    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", "token")
    monkeypatch.setenv("LOCAL_OPERATOR_HOME", str(tmp_path))
    app.state.config_manager = SimpleNamespace(config_dir=tmp_path)
    app.state.serve_retiring = True
    with TestClient(app) as client:
        response = client.post(
            "/v1/desktop/sessions",
            json={"request_id": "01234567-89ab-cdef-0123-456789abcdef", "cwd": str(tmp_path)},
            headers={"Authorization": "Bearer token"},
        )

    assert response.status_code == 503, response.text
    assert response.json()["detail"]["code"] == "daemon-retiring"
    assert not (tmp_path / "desktop-receipts.db").exists(), "no receipt was claimed"


def test_the_create_route_does_not_call_every_failure_a_retirement(
    tmp_path, monkeypatch, restore_app_state: None
) -> None:
    """The mirror: only the typed refusal answers 503 ``daemon-retiring``.

    Without this, a route that turned EVERY error into ``daemon-retiring`` would
    pass the test above — and the client's "rediscover the successor" response
    would fire on a genuine bug.
    """

    class Boom(DesktopSessions):
        async def create(self, cwd: str, *, target: Any = None) -> str:
            raise ValueError("something else went wrong")

    from local_operator.server.app import app

    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", "token")
    monkeypatch.setenv("LOCAL_OPERATOR_HOME", str(tmp_path))
    app.state.config_manager = SimpleNamespace(config_dir=tmp_path)
    app.state.desktop_sessions = Boom(tmp_path)
    with TestClient(app) as client:
        response = client.post(
            "/v1/desktop/sessions",
            json={"request_id": "01234567-89ab-cdef-0123-456789abcdef", "cwd": str(tmp_path)},
            headers={"Authorization": "Bearer token"},
        )

    assert response.status_code == 409, response.text
    assert "daemon-retiring" not in response.text


# =============================================================================
# The app-attached daemon: the configuration the feature exists for
# =============================================================================


@contextlib.contextmanager
def _capture(logger_name: str, level: int = logging.INFO) -> Iterator[list[logging.LogRecord]]:
    """Records emitted by ONE logger, without ``caplog``.

    ``caplog`` is unusable for anything that boots the app: importing
    ``local_operator.server.app`` runs ``configure_console_logging()``, which
    REPLACES every root handler (``logger.py``) — including the one the fixture
    installs — so such a test reads an empty ``caplog.text`` and would pass while
    asserting nothing. A handler on the logger itself is out of that function's
    reach, and the level is raised here for the same reason: the app's own
    configuration decides the root's.
    """
    records: list[logging.LogRecord] = []

    class _Sink(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    logger = logging.getLogger(logger_name)
    previous, sink = logger.level, _Sink()
    logger.addHandler(sink)
    logger.setLevel(level)
    try:
        yield records
    finally:
        logger.removeHandler(sink)
        logger.setLevel(previous)


def _drain_stream(iterator: Any, frames: list[str]) -> asyncio.Task[None]:
    """Consume a stream frame by frame, on its own task, and never to the end.

    The task is what HOLDS the stream: the route's generator parks on the next
    frame (a heartbeat every 15 s), so as long as this task is alive the bridge
    it acquired is still acquired — which is exactly how the app's relay holds a
    daemon. Cancelling it is the client disconnecting, and it runs the route's
    own ``release_once()``.
    """

    async def _read() -> None:
        async for line in iterator:
            if line.startswith("data: "):
                frames.append(line)

    return asyncio.create_task(_read())


async def _wait_for_frames(frames: list[str], count: int) -> None:
    for _ in range(1000):
        if len(frames) >= count:
            return
        await asyncio.sleep(0.005)
    raise AssertionError(f"the stream produced {len(frames)} frame(s), wanted {count}")


@pytest.mark.asyncio
async def test_a_desktop_relay_holds_an_announced_daemon_that_keeps_serving(
    disk: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    restore_app_state: None,
) -> None:
    """THE configuration the feature exists for, end to end through the route.

    Both round-1 reviewers reproduced the OPPOSITE of this by execution: with the
    desktop app's own relay held (`DesktopStreamRelay`'s
    ``GET /v1/desktop/sessions/{id}/events``), the daemon announced NOTHING —
    32 s of samples, six check intervals, the settle long past, no record change
    and no log line at the default level — because the announcement sat behind a
    drain that a standing subscription can never empty. The release valve was
    circular: the record could not say "let go" until the client had let go.

    The hold is taken through the ROUTE — ``events()`` acquires the pool's own
    bridge exactly as the app's request does — and not by hand-setting
    ``users = 1`` on a bridge, which is precisely how the first round's tests
    missed it: a hand-built term has no owner that can ever release it, so the
    test can only ever prove the drain blocks.

    What is asserted here, in order: the record ANNOUNCES while the daemon keeps
    serving; a mutating request still succeeds and the latch is still OFF; no
    exit across several check intervals; the log names the holding term; and
    when the relay goes away the daemon latches, refuses and leaves, removing its
    record.
    """
    from local_operator.server.app import app
    from local_operator.server.routes import desktop_sessions as routes

    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", "relay-token")
    app.state.config_manager = SimpleNamespace(config_dir=tmp_path)
    # A REAL record on disk under this test's root: the file is the whole channel
    # to a client that is not this process, so that is what the assertions read.
    publisher = serve_registry.publisher(_record(pid=os.getpid()), root=tmp_path)
    path = publisher.path

    exited: list[bool] = []
    latched_at_exit: list[bool] = []
    stop = asyncio.Event()

    def _exit() -> None:
        latched_at_exit.append(retire.retiring(cast(Any, app)))
        exited.append(True)
        # The step the real shutdown's lifespan performs last; without it the
        # "record removed" assertion below would be testing the driver.
        serve_registry.unpublish(os.getpid(), tmp_path)

    task = asyncio.create_task(
        retire.retirement_poll(app, publisher, stop=stop, exit_process=_exit, boot=OLD)
    )
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://localhost",
            headers={"Authorization": "Bearer relay-token"},
        ) as client:
            created = await client.post(
                "/v1/desktop/sessions",
                json={
                    "request_id": "01234567-89ab-cdef-0123-456789abcdef",
                    "cwd": str(tmp_path),
                },
            )
            assert created.status_code == 200, created.text
            session_id = created.json()["result"]["session_id"]
            pool = app.state.desktop_sessions

            # The app's own relay request, held open on a task of its own.
            relay = await routes.events(
                session_id, cast(Any, SimpleNamespace(app=app)), epoch=None, after_seq=0
            )
            frames: list[str] = []
            held = _drain_stream(relay.body_iterator, frames)
            await _wait_for_frames(frames, 1)
            open_frame = json.loads(frames[0].removeprefix("data: "))
            assert open_frame["type"] == "open", "the relay's first frame is its identity"
            subscription_id = open_frame["payload"]["subscription_id"]
            bridge = pool.bridges[session_id]
            assert bridge.users == 1, "the ROUTE's own bridge is what holds the daemon"

            with _capture("local_operator.server.retire") as records:
                disk["build"] = NEW  # the install moves under the daemon
                await _wait_for_announcement(publisher)

                # THE RECORD, as a reader sees it: announced terms, nothing else
                # changed — the record stays live and keeps heartbeating.
                announced = json.loads(path.read_text())
                assert announced["retiring_from"] == OLD.label()
                assert announced["retiring_to"] == NEW.label()

                # ... AND IT KEEPS DOING ITS JOB. A mutating desktop request
                # still succeeds and the latch is still off: this is the
                # round-2 correction, and the reason the announcement is not
                # allowed to latch. The watch lease is invisible here so no
                # speculative warm is attempted — the spawn path is refused by
                # the matrix below, not exercised by this test.
                watched = await client.post(
                    f"/v1/desktop/sessions/{session_id}/watch",
                    json={
                        "subscription_id": subscription_id,
                        "visible": False,
                        "can_notify": False,
                    },
                )
                assert watched.status_code == 200, watched.text
                assert retire.retiring(cast(Any, app)) is False, "announced != refusing"

                # NO EXIT ACROSS SEVERAL CHECK INTERVALS (0.02 s each, settle
                # long past): the standing relay alone pins it.
                await asyncio.sleep(QUIET_S)
                assert exited == []
                assert not task.done()
                assert retire.retiring(cast(Any, app)) is False
                assert json.loads(path.read_text())["retiring_to"] == NEW.label()
            assert any(
                "in-flight desktop request(s)" in record.getMessage() for record in records
            ), "the log names the holding term"

            # THE RELAY GOES AWAY — the view unmounted, or the app quit. Only now
            # can the drain empty, so only now does the daemon latch and leave.
            held.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await held
            assert bridge.users == 0, "the route released its bridge with the stream"

            await asyncio.wait_for(task, 2.0)
            assert exited == [True]
            assert latched_at_exit == [True], "the refusal is latched BEFORE the exit"
            assert not path.exists(), "a clean exit leaves no record claiming a live daemon"
    finally:
        stop.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        if getattr(app.state, "desktop_sessions", None) is not None:
            await app.state.desktop_sessions.close()


@pytest.mark.asyncio
async def test_a_bridge_asserts_admission_for_every_work_path(tmp_path: Path) -> None:
    """The bridge-level gate is the SAME question the pool's is (MAJOR-2).

    One method rather than one per caller: ``warm``, ``/messages`` and
    ``/commands`` all ask it, so a path added later reaches the same answer by
    calling the same thing. Pinned here in isolation because the route tests
    below exercise the route's error ladder rather than the bridge's own gate.
    """
    latch = {"on": False}
    pool = DesktopSessions(tmp_path, retiring=lambda: latch["on"])
    session_id = await pool.create(str(tmp_path))
    latch["on"] = True  # the daemon latches
    async with pool.session(session_id) as bridge:
        with pytest.raises(retire.DaemonRetiring) as raised:
            bridge.assert_admitting()
        assert raised.value.code == "daemon-retiring"
        assert str(raised.value) == retire.RETIRING_MESSAGE

        latch["on"] = False  # the same bridge admits again before its daemon latches
        bridge.assert_admitting()
    await pool.close()


@pytest.mark.parametrize(
    ("label", "suffix"),
    [
        ("create", ""),
        ("warm", "/warm"),
        ("messages", "/messages"),
        ("commands", "/commands"),
        ("answers", "/answers"),
    ],
)
@pytest.mark.asyncio
async def test_every_admitting_path_refuses_once_latched(
    label: str,
    suffix: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    restore_app_state: None,
) -> None:
    """The refusal matrix: EVERY path that can admit or start work answers 503.

    Round 1 measured the gap by execution: on a latched daemon
    ``POST /messages`` returned 200 and reached ``admit_prompt`` while ``/warm``
    answered 503 against the same process — and ``admit_prompt``'s
    ``_ensure_bound`` is the one call that can START a session runtime, which is
    exactly what ``warm``'s own refusal exists to prevent. ``/commands`` reaches
    it more directly still (``bind_runtime()``).

    ``/answers`` is gated for a different and stated reason: it cannot spawn
    (``answer_gate`` needs a connected client and raises otherwise), but a
    latched daemon is a process whose socket is about to close, so an answer
    delivered through it is a delivery nobody can confirm.

    Each row asserts the TYPED refusal AND that no runtime was started: the
    spawn seam (``_ensure_bound``/``warm_runtime``) is patched to fail loudly, so
    a path that reached it would fail this test rather than quietly start a
    process, and no receipt row is left behind for a retry against the successor.
    """
    from local_operator.server.app import app
    from local_operator.session.attached import AttachedSession

    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", "matrix-token")
    app.state.config_manager = SimpleNamespace(config_dir=tmp_path)
    # The route's OWN probe (the expression `host()` builds), so this test drives
    # the real wiring rather than a pool that agrees with itself. The session is
    # created BEFORE the latch, because a latched daemon cannot create one — the
    # first row of the matrix.
    app.state.serve_retiring = False
    pool = DesktopSessions(
        tmp_path,
        retiring=lambda: bool(getattr(app.state, retire.RETIRING_STATE_ATTR, False)),
    )
    app.state.desktop_sessions = pool
    session_id = await pool.create(str(tmp_path))
    app.state.serve_retiring = True  # the daemon latches

    started: list[str] = []

    async def _no_spawn(*_args: Any, **_kwargs: Any) -> None:
        started.append("spawn")
        raise AssertionError("a latched daemon reached the runtime spawn seam")

    monkeypatch.setattr(AttachedSession, "_ensure_bound", _no_spawn)
    monkeypatch.setattr(AttachedSession, "warm_runtime", _no_spawn)

    payloads: dict[str, dict[str, Any]] = {
        "": {"request_id": "01234567-89ab-cdef-0123-456789abcdef"},
        "/warm": {},
        "/messages": {
            "request_id": "01234567-89ab-cdef-0123-456789abcdef",
            "text": "hello",
        },
        "/commands": {
            "request_id": "01234567-89ab-cdef-0123-456789abcdef",
            "command": "compact",
            "args": "",
        },
        "/answers": {
            "epoch": "epoch",
            "request_id": "01234567-89ab-cdef-0123-456789abcdef",
            "approved": True,
        },
    }
    payload = payloads[suffix]
    url = "/v1/desktop/sessions"
    if suffix:
        url = f"/v1/desktop/sessions/{session_id}{suffix}"
    else:
        payload = {**payload, "cwd": str(tmp_path)}

    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://localhost",
            headers={"Authorization": "Bearer matrix-token"},
        ) as client:
            response = await client.post(url, json=payload)

        assert response.status_code == 503, f"{label}: {response.status_code} {response.text}"
        detail = response.json()["detail"]
        assert detail["code"] == "daemon-retiring", label
        assert detail["message"] == retire.RETIRING_MESSAGE, label
        assert started == [], f"{label} reached the spawn seam"
        bridge = pool.bridges.get(session_id)
        assert bridge is None or bridge.warm_task is None, f"{label} started a warm"
        assert not (
            tmp_path / "desktop-receipts.db"
        ).exists(), f"{label} claimed a receipt before refusing"
    finally:
        await pool.close()


# =============================================================================
# The announcement write, the probes, the reload child and the baseline
# =============================================================================


class _FailingPublisher(FakePublisher):
    """A publisher whose record write fails, the way a read-only root does."""

    def __init__(self, record: ServeRecord) -> None:
        super().__init__(record)
        self.fail = True

    def heartbeat(self, **updates: object) -> None:
        if self.fail:
            raise OSError("read-only record directory")
        super().heartbeat(**updates)


@pytest.mark.asyncio
async def test_a_failed_announcement_write_neither_latches_nor_kills_the_poll(
    disk: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    """The latch must FOLLOW the write, not precede it (review round 1, MINOR-3).

    With the order these rounds corrected, the latch was set and the record
    written after it, so a write that raised (a read-only or full config root —
    the failure the sibling ``heartbeat_loop`` treats as self-healing) left a
    daemon that answered 503 to everything FOREVER and never left, with nothing
    logged because the task's exception was never observed. A retirement nobody
    can recover from without killing the process is worse than no retirement.

    Here the failure is retried, the daemon keeps serving the old build, the
    latch stays off, and the poll is still alive to try again — and once the
    write succeeds, the sequence proceeds normally.
    """
    app, publisher = FakeApp(), _FailingPublisher(_record())
    task, stop, exited = await _start(app, publisher)  # type: ignore[arg-type]
    await asyncio.sleep(QUIET_S)

    with caplog.at_level("WARNING", logger="local_operator.server.retire"):
        disk["build"] = NEW
        await asyncio.sleep(QUIET_S)  # several checks, every one of them failing

    assert publisher.writes == []
    assert retire.retiring(cast(Any, app)) is False, "a failed announcement must not latch"
    assert exited == []
    assert not task.done(), "the poll must survive the failure and retry"
    assert "could not be written" in caplog.text
    assert "retrying on the next check" in caplog.text

    publisher.fail = False  # the record directory becomes writable again
    await _wait_for_announcement(publisher)
    await asyncio.wait_for(task, 2.0)
    assert exited == [True]
    assert retire.retiring(cast(Any, app)) is True


@pytest.mark.asyncio
async def test_a_dead_poll_is_logged_rather_than_silent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A build watch that dies says so instead of leaving a daemon that never rolls.

    The task is created bare by the lifespan and awaited only at teardown, so
    without the observer a poll that raised produced no line at all: a daemon
    that never retires, with no reason anywhere — the same silent no-rollout this
    change exists to remove, arrived at from the other side.
    """

    async def _boom() -> None:
        raise RuntimeError("the poll exploded")

    with caplog.at_level("ERROR", logger="local_operator.server.retire"):
        task = asyncio.create_task(_boom())
        task.add_done_callback(retire.observe_poll)
        with pytest.raises(RuntimeError, match="the poll exploded"):
            await task
        await asyncio.sleep(0)  # the done-callback runs on its own turn
    assert "will not retire" in caplog.text

    # A cancelled poll is an ordinary teardown, not a failure to report.
    caplog.clear()
    cancelled = asyncio.create_task(asyncio.sleep(60))
    cancelled.add_done_callback(retire.observe_poll)
    cancelled.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await cancelled
    await asyncio.sleep(0)
    assert caplog.text == "", "a cancelled poll is not a dead one"


@pytest.mark.parametrize("probe", ["event broker", "websocket manager"])
@pytest.mark.asyncio
async def test_an_unreadable_probe_means_stay(
    probe: str,
    disk: dict[str, Any],
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A probe this process cannot READ pins it, rather than releasing it (MINOR-4).

    The probe helpers used to fail OPEN: a raising ``stats()`` read as "no
    streams" and a raising accessor as no sockets, so a probe that broke let the
    daemon exit under a live stream — "an interruption nobody undoes", and the
    opposite of the rule this module states. Missing probes are still benign (a
    reduced app has nothing in flight either way, and reading those as "stay"
    would pin every unit-sized app forever); it is the RAISING ones that mean
    stay.

    One probe broken at a time: the predicate short-circuits, so a test that
    broke all of them would only ever exercise the first — and the message has to
    name the probe that could not be read, which is what makes the daemon's log
    explain why it is still here.
    """
    state: dict[str, Any] = {}
    if probe == "event broker":
        broker = SimpleNamespace(stats=_raiser("the broker is broken"))
        state["event_broker"] = broker
        expected = "the event broker's subscriber count"

        def heal() -> None:
            broker.stats = lambda: {"subscribers": 0}

    else:
        manager = WebSocketManager()
        broken = _raiser("the manager is broken")
        manager.live_connection_count = broken  # type: ignore[method-assign]
        state["websocket_manager"] = manager
        expected = "the websocket connection count"

        def heal() -> None:
            manager.live_connection_count = lambda: 0  # type: ignore[method-assign]

    app, publisher = FakeApp(**state), FakePublisher(_record())
    task, stop, exited = await _start(app, publisher)  # type: ignore[arg-type]
    await asyncio.sleep(QUIET_S)

    with caplog.at_level("WARNING", logger="local_operator.server.retire"):
        disk["build"] = NEW
        await _wait_for_announcement(publisher)
        await asyncio.sleep(QUIET_S)  # several checks with the probe still broken

    # Announced, still serving, and NOT latched: the unreadable probe is what
    # keeps it here, and the reason names it.
    _announced_not_latched(app, publisher, exited)
    assert not task.done()
    assert expected in caplog.text, "the log names the probe that could not be read"

    # The probe reads again — the same daemon with a healed accessor — and now
    # the drain really is empty, so the daemon leaves.
    heal()
    await asyncio.wait_for(task, 2.0)
    assert exited == [True]
    assert retire.retiring(cast(Any, app)) is True


def _raiser(message: str) -> Any:
    """A probe that raises, for the fail-closed cases."""

    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise OSError(message)

    return _boom


@pytest.mark.asyncio
async def test_an_unreadable_desktop_probe_means_stay(
    disk: dict[str, Any], tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The same rule for the term the app's own relay lives on (MINOR-4).

    Separate from the case above because the failure is INJECTED differently —
    into the pool the daemon's ``/events`` relay holds a bridge on — and because
    that is the term whose wrong direction cuts a live viewer, which is the
    failure the fail-closed rule exists to prevent.
    """
    pool = DesktopSessions(tmp_path)
    pool.in_flight_reason = _raiser("the desktop probe is broken")  # type: ignore[method-assign]
    app, publisher = FakeApp(desktop_sessions=pool), FakePublisher(_record())
    task, stop, exited = await _start(app, publisher)  # type: ignore[arg-type]
    await asyncio.sleep(QUIET_S)

    with caplog.at_level("WARNING", logger="local_operator.server.retire"):
        disk["build"] = NEW
        await _wait_for_announcement(publisher)
        await asyncio.sleep(QUIET_S)

    _announced_not_latched(app, publisher, exited)
    assert not task.done()
    assert "the desktop in-flight probe could not be read" in caplog.text

    pool.in_flight_reason = lambda: None  # type: ignore[method-assign]
    await asyncio.wait_for(task, 2.0)
    assert exited == [True]


@pytest.mark.asyncio
async def test_the_lifespan_runs_no_build_watch_on_a_reload_child(
    disk: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    restore_app_state: None,
) -> None:
    """A ``--reload`` child must not self-retire (QA round 1, Q3).

    Under ``--reload`` the PORT belongs to uvicorn's supervisor: the child serves
    through the parent's socket. A child that retired removed its own record and
    asked its own process to stop, leaving the parent alive and accepting on a
    socket with nothing behind it — measured as ``/health`` timing out with no
    record to explain it, in all three of QA's runs. A dev-mode supervisor is
    also not a production daemon: it has no successor to hand a socket to, and
    the operator is watching its console.

    The reload channel is exercised through the REAL predicate rather than by
    monkeypatching ``is_reload_child``: uvicorn spawns that child with
    ``multiprocessing``, so the announcement names our spawner's pid and is
    honoured only when that pid is the one we pretend to have.
    """
    from local_operator.server.app import app, lifespan

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(serve_registry, "_spawner_pid", lambda: 4242)
    monkeypatch.setenv(serve_registry.SERVE_ANNOUNCE_ENV, "4242 127.0.0.1 58474")
    path = serve_registry.record_path(os.getpid(), tmp_path)

    async with lifespan(app):
        assert serve_registry.is_reload_child(app) is True
        assert path.exists(), "the record is still published for a reload child"

        disk["build"] = NEW  # the install moves under it, as it would in dev
        await asyncio.sleep(QUIET_S)
        record = json.loads(path.read_text())
        assert (record["retiring_from"], record["retiring_to"]) == (
            "",
            "",
        ), "a --reload child announced a handover it cannot make"
        # ``getattr``, not ``in``: ``app`` is the module-level singleton and its
        # state is shared with every other lifespan test in this worker (the
        # restore fixture puts back the keys a previous test's lifespan created,
        # as ``None``). What matters is that no task is THERE, and what this test
        # can pin on its own is that the flip produced no announcement at all.
        assert (
            getattr(app.state, "serve_retire", None) is None
        ), "a --reload child starts no build watch at all"
    assert not path.exists()


@pytest.mark.asyncio
async def test_a_flip_right_after_the_record_appears_is_still_detected(
    disk: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    restore_app_state: None,
) -> None:
    """The baseline is sampled BEFORE the record is published (QA round 1, Q2).

    The record file is what a reader waits for, so a harness (or an updater) acts
    the instant it appears — while the poll's basline sample had not run yet.
    A baseline read after the publish adopted that marker as "the build this
    process loaded", and the daemon then never retired for it: QA measured 3 of 7
    immediate flips swallowed, silently, with no line at any log level.

    The fix is ordering in the lifespan (sample first, publish second) plus a
    logged baseline so a watcher that can never fire is diagnosable from its own
    output. This test flips the marker the moment the record exists, which is the
    window that used to swallow it.

    ``retire._request_shutdown`` is replaced with a recorder because the real one
    SIGTERMs this process: in production that is exactly right (uvicorn's clean
    shutdown, which removes the record), and in a unit test it would kill the
    test runner.
    """
    from local_operator.server.app import app, lifespan

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    serve_registry.announce_address(app, "127.0.0.1", 58474)
    path = serve_registry.record_path(os.getpid(), tmp_path)
    exits: list[bool] = []
    monkeypatch.setattr(retire, "_request_shutdown", lambda: exits.append(True))

    with _capture("local_operator.server.retire") as records:
        async with lifespan(app):
            assert path.exists(), "the record is published with the app"
            disk["build"] = NEW  # the flip lands in the gap the baseline used to swallow
            for _ in range(200):
                if json.loads(path.read_text())["retiring_to"]:
                    break
                await asyncio.sleep(0.02)
            else:
                pytest.fail("a flip made right after the record appeared was never detected")

            # ... and with nothing in flight the NEXT check latches and asks the
            # process to stop (recorded here, delivered in production).
            for _ in range(200):
                if exits:
                    break
                await asyncio.sleep(0.02)
            assert exits == [True], "an announced daemon with an empty drain leaves"
    assert any(
        "build watch baseline" in record.getMessage() for record in records
    ), "the baseline is logged, not silent"
