"""The serve daemon's build watch: announce, refuse new work, then leave.

The daemon is the one participant on this host that used to have no rollout for
a new build — it kept serving the old one until somebody killed it. These tests
drive the three halves of the replacement separately, because they fail
differently: the POLL decides when to leave, the PREDICATE decides whether
leaving is safe, and the REFUSAL is what a client sees in between.

The poll is driven for real (with ``BUILD_CHECK_S`` shortened) rather than by
calling its internals, so the shared settle rule, the announce-then-notice order
and the exit are exercised as the lifespan starts them. ``update.installed_build``
and ``build_marker_age_s`` are faked the way
``tests/unit/session/runtime/test_process_refresh.py`` fakes them for the
runtime, so both watchers are pinned against the same rule.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

from local_operator import buildwatch
from local_operator import update as update_mod
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
    app: FakeApp, publisher: FakePublisher
) -> tuple[asyncio.Task[None], asyncio.Event, list[bool]]:
    """Run the poll as the lifespan does, with the exit recorded instead taken."""
    stop = asyncio.Event()
    exited: list[bool] = []
    task = asyncio.create_task(
        retire.retirement_poll(
            app,  # type: ignore[arg-type] — the predicate is duck-typed by design
            publisher,  # type: ignore[arg-type]
            stop=stop,
            exit_process=lambda: exited.append(True),
        )
    )
    return task, stop, exited


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
async def test_a_settled_new_build_announces_then_exits(disk: dict[str, Any]) -> None:
    app, publisher = FakeApp(), FakePublisher(_record())
    task, stop, exited = await _start(app, publisher)
    await asyncio.sleep(QUIET_S)  # several checks against the OLD stamp
    assert publisher.writes == []

    disk["build"] = NEW  # the install moves under the daemon
    await asyncio.wait_for(task, 2.0)

    assert exited == [True], "the exit belongs to the process's own shutdown path"
    assert publisher.writes[-1] == {
        "retiring_from": OLD.label(),
        "retiring_to": NEW.label(),
    }, "a reader must see WHICH build it left for and which one is coming"
    assert publisher.record.retiring_from == OLD.label()
    assert publisher.record.retiring_to == NEW.label()
    assert retire.retiring(cast(Any, app)) is True, "the refusal latches with the announcement"


@pytest.mark.asyncio
async def test_a_stop_during_the_notice_owns_the_exit(disk: dict[str, Any], monkeypatch):
    """A stop that lands mid-notice leaves the exit to the stop path.

    Otherwise the record's last state would depend on which of two paths won a
    race: the daemon announcing a build change while uvicorn is already
    shutting down.
    """
    monkeypatch.setattr(buildwatch, "BUILD_STAGGER_S", 5.0)
    app, publisher = FakeApp(), FakePublisher(_record())
    task, stop, exited = await _start(app, publisher)
    await asyncio.sleep(QUIET_S)

    disk["build"] = NEW
    while not publisher.writes:
        await asyncio.sleep(0.01)
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
    await asyncio.sleep(QUIET_S)
    assert publisher.writes == [] and exited == []

    broker.stats = lambda: {"subscribers": 0}  # the turn's stream ends
    await asyncio.wait_for(task, 2.0)
    assert exited == [True]


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
    await asyncio.sleep(QUIET_S)
    assert publisher.writes == [] and exited == []
    assert manager.live_connection_count() == 1

    manager.connections.clear()
    await asyncio.wait_for(task, 2.0)
    assert exited == [True]


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
    await asyncio.sleep(QUIET_S)
    assert publisher.writes == [] and exited == []
    assert pool.in_flight_reason() == "1 in-flight desktop request(s) on 0123456789ab"

    bridge.users = 0
    await asyncio.wait_for(task, 2.0)
    assert exited == [True]


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
    await asyncio.sleep(QUIET_S)
    assert publisher.writes == [] and exited == []
    assert "watching session" in (pool.in_flight_reason() or "")

    sub.expires = 0.0  # the lease lapses
    await asyncio.wait_for(task, 2.0)
    assert exited == [True]


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
    await asyncio.sleep(QUIET_S)
    assert publisher.writes == [] and exited == []
    assert "runtime being started" in (pool.in_flight_reason() or "")

    bridge.warm_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await bridge.warm_task
    await asyncio.wait_for(task, 2.0)
    assert exited == [True]


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
