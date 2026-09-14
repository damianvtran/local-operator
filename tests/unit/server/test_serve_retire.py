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

import ast
import asyncio
import contextlib
import importlib
import json
import logging
import os
import sqlite3
import time
from collections.abc import Iterator
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient, Timeout

from local_operator import buildwatch
from local_operator import update as update_mod
from local_operator.credentials import CredentialManager
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
#: A third build, for the case where the install moves ON while a handover to
#: ``NEW`` is already announced: the record must name this one, not the first.
NEWER = BuildStamp(version="0.54.32", source_ref="3333333")

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


async def _wait_for_withdrawal(publisher: FakePublisher) -> None:
    """Block until the record no longer announces a handover, or fail loudly."""
    for _ in range(1000):
        if not publisher.record.retiring_to:
            return
        await asyncio.sleep(0.005)
    raise AssertionError("the handover was never withdrawn from the record")


@pytest.mark.asyncio
async def test_an_unreadable_stamp_is_never_a_build_to_retire_onto(disk: dict[str, Any]) -> None:
    """QA round 2's OBS-1, at the rule's own seam: a stamp nobody could read is NOT a move.

    ``update.source_ref`` maps an unreadable, empty, truncated or non-commit
    ``.lop-source`` to ``""``, so ``installed_build`` answers a VERSION-ONLY stamp
    for all of them — and a version-only stamp differs from a ``version@ref`` boot
    stamp without the build having moved anywhere. QA measured the consequence on a
    real daemon: with an aged marker ``chmod 000``-ed, the daemon announced
    ``0.54.39@1111111 → 0.54.39``, latched, exited and removed its record — i.e. it
    left for a build it could not read. The ref is the primary key precisely
    because two different builds share one version string here, so a version-only
    stamp is an absence of evidence rather than evidence of a move.
    """
    app, publisher = FakeApp(), FakePublisher(_record())
    task, stop, exited = await _start(app, publisher)
    await asyncio.sleep(QUIET_S)  # the boot stamp is OLD, the daemon is watching

    disk["build"] = BuildStamp(version=OLD.version)  # the marker cannot be read
    await asyncio.sleep(QUIET_S)

    assert publisher.writes == [], "a stamp that could not be read is not a handover"
    assert exited == [], "the daemon left for a build it could not read"
    assert retire.retiring(cast(Any, app)) is False
    assert not task.done()

    stop.set()
    await asyncio.wait_for(task, 1.0)


@pytest.mark.asyncio
async def test_a_version_move_with_no_ref_is_still_a_move(disk: dict[str, Any]) -> None:
    """The other direction of the same guard, so it cannot close too far.

    A wheel upgrade writes ``pypi <version>``: the ref is empty because there is no
    commit to record, and the version is exactly what moved. Refusing that would
    silently disable the whole feature for every PyPI install, which is why the
    guard asks about the VERSION when the ref is unreadable rather than refusing
    every version-only stamp.
    """
    app, publisher = FakeApp(), FakePublisher(_record())
    task, stop, exited = await _start(app, publisher)
    await asyncio.sleep(QUIET_S)

    disk["build"] = BuildStamp(version="0.54.40")
    await _wait_for_announcement(publisher)

    assert publisher.record.retiring_from == OLD.label()
    assert publisher.record.retiring_to == "0.54.40"
    assert exited == []

    stop.set()
    await asyncio.wait_for(task, 1.0)


@pytest.mark.asyncio
async def test_a_reverted_install_withdraws_the_announcement_and_keeps_serving(
    disk: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    """MINOR-2: an announcement that stops being true is taken back out of the record.

    The announcement used to be written once and never re-read, so an install that
    went BACK to the running build — a ``lop-update`` that failed and was rolled
    back, or one the running build superseded — still latched, exited and removed
    its record, having told every reader to hand over to a build that was no longer
    there. For an unsupervised daemon that is "the daemon is gone" with no
    successor, and for a reader of the record it is a ``retiring_to`` that names
    nothing.

    Both halves are asserted: the fields are CLEARED (a reader must not keep acting
    on a handover that is not happening) and the daemon is still serving — then, as
    the second half of the test, it announces again when the install really does
    move, because a withdrawal must not disarm the watch it belongs to.
    """
    app, publisher = FakeApp(), FakePublisher(_record())
    task, stop, exited = await _start(app, publisher)
    await asyncio.sleep(QUIET_S)

    disk["build"] = NEW
    await _wait_for_announcement(publisher)
    _announced_not_latched(app, publisher, exited)

    with caplog.at_level(logging.INFO):
        disk["build"] = OLD  # the install is back on the build this process loaded
        await _wait_for_withdrawal(publisher)

    assert publisher.writes[-1] == {"retiring_from": "", "retiring_to": ""}
    assert publisher.record.retiring_from == "" and publisher.record.retiring_to == ""
    assert retire.retiring(cast(Any, app)) is False, "a withdrawn handover must not latch"
    assert exited == [], "the daemon is the right build after all; it stays"
    assert not task.done(), "the watch goes on watching"
    assert "no longer holds" in caplog.text, "the withdrawal is visible to an operator"

    disk["build"] = NEW
    await _wait_for_announcement(publisher)
    _announced_not_latched(app, publisher, exited)

    stop.set()
    await asyncio.wait_for(task, 1.0)


@pytest.mark.asyncio
async def test_a_stamp_that_stops_reading_as_a_build_withdraws_the_announcement(
    disk: dict[str, Any],
) -> None:
    """An announced handover whose stamp can no longer be read is withdrawn too.

    The fail-closed rule of OBS-1 is not the detection's alone: an install that was
    readable when it was announced and is unreadable now is the same "never leave
    for a build you could not read", and the answer is the same one the reversion
    gets — take the handover back out of the record and keep serving rather than
    latching on a claim nothing on disk still supports.
    """
    app, publisher = FakeApp(), FakePublisher(_record())
    task, stop, exited = await _start(app, publisher)
    await asyncio.sleep(QUIET_S)

    disk["build"] = NEW
    await _wait_for_announcement(publisher)

    disk["build"] = BuildStamp(version=OLD.version, source_ref="")  # unreadable marker
    await _wait_for_withdrawal(publisher)

    assert retire.retiring(cast(Any, app)) is False
    assert exited == [], "the daemon does not leave for a stamp it cannot read"

    stop.set()
    await asyncio.wait_for(task, 1.0)


@pytest.mark.asyncio
async def test_an_install_that_moves_on_again_re_announces_onto_the_new_build(
    disk: dict[str, Any],
) -> None:
    """MINOR-2's second half: the record names the build that is there NOW.

    Between the announcement and the latch an installer can run twice (an update
    that lands while the drain is still held). The daemon is leaving either way, but
    a reader must not be sent to the build that has already been replaced: the
    target is re-read and the record rewritten onto the newest one.
    """
    app, publisher = FakeApp(), FakePublisher(_record())
    task, stop, exited = await _start(app, publisher)
    await asyncio.sleep(QUIET_S)

    disk["build"] = NEW
    await _wait_for_announcement(publisher)

    disk["build"] = NEWER
    for _ in range(1000):
        if publisher.record.retiring_to == NEWER.label():
            break
        await asyncio.sleep(0.005)
    else:
        raise AssertionError(
            f"the record still names {publisher.record.retiring_to!r} after the install moved on"
        )

    assert publisher.record.retiring_from == OLD.label()
    re_announced = [
        write for write in publisher.writes if write.get("retiring_to") == NEWER.label()
    ]
    assert re_announced, "the re-announcement must be a record write, not a field edit"
    assert retire.retiring(cast(Any, app)) is False, "re-announcing is not latching"
    assert exited == []

    stop.set()
    await asyncio.wait_for(task, 1.0)


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
    """DEFENSIVE GUARD for the lease term, not a description of a reachable state.

    ``expires`` is what ``watch`` writes (``time.monotonic() + WATCH_TTL``), and
    this sets that state directly. QA round 2's OBS-2 measured what a real daemon
    does with it: taking the lease over the app's own flow (``POST /watch`` while
    the relay was open, ``200 {"lease_seconds": 45}``) and then closing the relay
    without renewing left the daemon latching 5 s later with the log naming the
    RELAY term and no lease term at all. The lease rides the same subscription and
    does not outlive it, so ``users == 0`` with a live lease is not a state a real
    daemon reaches — the relay is the binding term (see
    ``test_an_in_flight_desktop_request_holds_the_daemon``).

    Kept, and labelled, rather than dropped: the term exists in ``in_flight`` as a
    guard for a viewer whose socket has gone while its lease has not, and a guard
    nobody exercises is a guard that rots silently. What this test pins is that the
    term still holds the daemon and still releases it when the lease lapses — not
    that the production flow can produce it.
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
    """The RECORD plane stays open to a latched daemon — session-scoped reads do not.

    ``list`` is how a reader discovers what is on this host (``scan()`` and the
    desktop app's own discovery read the record the same way), so it has to keep
    answering until the clean exit removes the record: it is what makes the
    handover OBSERVABLE instead of a daemon that vanishes.

    It is also the reason the door can be the gate for everything else. Nothing
    here takes a bridge, so nothing here is refused — and the same is true of
    ``GET /health`` and of the record file itself. Every route that DOES take a
    bridge is refused once latched, reads included (see
    ``test_every_route_that_reaches_the_door_refuses_once_latched``), because a
    session-scoped answer can only come from the build the daemon has already told
    its readers to leave.
    """
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
        async def create(self, cwd: str, *, target: Any = None, model: Any = None) -> str:
            # ``model`` because ``main``'s draft-pane route now passes the birth
            # selection through here (see ``DesktopSessions.create``): the mirror
            # is about the ERROR the adapter raises, not about that parameter.
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
    """The bridge-level gate, held across the latch — what the door cannot express.

    The door refuses a bridge that is not yet handed out; this is the question the
    callers that already HOLD one ask: ``warm``'s own speculation and the
    lease-warm loop run in-process, do not come through a route, and must not
    delegate to a route's refusal. The routes no longer call it — the enforcement is
    the pool's door (``DesktopSessions.session``), which they all come through — so
    this pins the in-process half rather than a route's error ladder.
    """
    latch = {"on": False}
    pool = DesktopSessions(tmp_path, retiring=lambda: latch["on"])
    session_id = await pool.create(str(tmp_path))
    async with pool.session(session_id) as bridge:
        assert bridge.retiring_probe is pool.retiring_probe  # one probe, one answer
        bridge.assert_admitting()  # before the latch, the same bridge admits

        latch["on"] = True  # the daemon latches while this bridge is held
        with pytest.raises(retire.DaemonRetiring) as raised:
            bridge.assert_admitting()
        assert raised.value.code == "daemon-retiring"
        assert str(raised.value) == retire.RETIRING_MESSAGE

        latch["on"] = False  # and admits again if the handover is withdrawn
        bridge.assert_admitting()
    await pool.close()


#: The ``request_id`` the matrix rows are sent with. One id, because the point of
#: the side-effect assertion below is that a REFUSED request claims nothing — a
#: claimed receipt is indeterminate for the client's retry (see ``stop``'s comment)
#: and a claimed aside id answers 409 on the retry against the successor.
REQUEST_ID = "01234567-89ab-cdef-0123-456789abcdef"


#: A well-formed ``aside_id`` (the route's own pattern), spelled once.
ASIDE_ID = "abcdef01-2345-6789-abcd-ef0123456789"


@dataclass(frozen=True)
class _DoorRoute:
    """One desktop route, as the request that reaches the door.

    Keyed by the handler function's own name rather than by the URL it happens to
    live at: the completeness test below matches this table against the ROUTER and
    against an AST walk of the router modules, and matching on a URL a test
    spelled itself is how a table comes to look complete while a route added
    under it is not covered.
    """

    endpoint: str
    method: str
    template: str
    payload: dict[str, Any] | None = None
    query: str = ""
    session_in_query: bool = False


#: EVERY route in the desktop plane that obtains a bridge, which — because the
#: pool's ``session()`` is the only thing in the process that builds one — is
#: every route that can admit or start work. Review round 2 measured the previous
#: five-row version of this list: it gated ``/messages`` and ``/commands``, and
#: left ``/mcp``, ``/credentials``, ``/fork`` (and its child admission),
#: ``/asides`` and ``/adopt`` reaching ``bind_runtime()`` on the same latched
#: daemon. The list was the mechanism then and is the TEST now.
REFUSAL_MATRIX: tuple[_DoorRoute, ...] = (
    _DoorRoute(
        "create_session",
        "POST",
        "/v1/desktop/sessions",
        {"request_id": "01234567-89ab-cdef-0123-456789abcdef", "cwd": "/"},
    ),
    _DoorRoute("skills", "GET", "/v1/desktop/skills", session_in_query=True),
    _DoorRoute("mcp_status", "GET", "/v1/desktop/sessions/{session_id}/mcp"),
    _DoorRoute(
        "mcp_control",
        "POST",
        "/v1/desktop/sessions/{session_id}/mcp",
        {"action": "list"},
    ),
    _DoorRoute(
        "credential",
        "POST",
        "/v1/desktop/sessions/{session_id}/credentials",
        {"action": "list", "key": "TEST_KEY"},
    ),
    _DoorRoute(
        "fork",
        "POST",
        "/v1/desktop/sessions/{session_id}/fork",
        {"request_id": "01234567-89ab-cdef-0123-456789abcdef", "message": ""},
    ),
    _DoorRoute(
        "stop",
        "POST",
        "/v1/desktop/stop",
        {
            "request_id": "01234567-89ab-cdef-0123-456789abcdef",
            "targets": ["SESSION_ID"],
            "confirmed": True,
        },
    ),
    _DoorRoute(
        "aside",
        "POST",
        "/v1/desktop/sessions/{session_id}/asides",
        {"request_id": "01234567-89ab-cdef-0123-456789abcdef", "text": "hello"},
    ),
    _DoorRoute(
        "adopt",
        "POST",
        "/v1/desktop/sessions/{session_id}/asides/{aside_id}/adopt",
        {"request_id": "01234567-89ab-cdef-0123-456789abcdef", "confirmed": True},
    ),
    _DoorRoute("snapshot", "GET", "/v1/desktop/sessions/{session_id}"),
    _DoorRoute("history", "GET", "/v1/desktop/sessions/{session_id}/history"),
    # The session's code memory (added upstream while this branch was in review:
    # the completeness test below is what reported it, which is the property the
    # round-2 MAJOR asked for — a route added under the door fails HERE).
    _DoorRoute("variables", "GET", "/v1/desktop/sessions/{session_id}/variables"),
    _DoorRoute(
        "create_variable",
        "POST",
        "/v1/desktop/sessions/{session_id}/variables",
        {"key": "counter", "value": "1", "type": "int"},
    ),
    _DoorRoute(
        "update_variable",
        "PATCH",
        "/v1/desktop/sessions/{session_id}/variables/{key}",
        {"value": "2", "type": "int"},
    ),
    _DoorRoute(
        "delete_variable",
        "DELETE",
        "/v1/desktop/sessions/{session_id}/variables/{key}",
    ),
    _DoorRoute("events", "GET", "/v1/desktop/sessions/{session_id}/events"),
    _DoorRoute("failovers", "GET", "/v1/desktop/sessions/{session_id}/failovers"),
    _DoorRoute(
        "entities",
        "GET",
        "/v1/desktop/sessions/{session_id}/command-entities",
        query="command=help",
    ),
    _DoorRoute(
        "prompt",
        "POST",
        "/v1/desktop/sessions/{session_id}/messages",
        {"request_id": "01234567-89ab-cdef-0123-456789abcdef", "text": "hello"},
    ),
    _DoorRoute(
        "command",
        "POST",
        "/v1/desktop/sessions/{session_id}/commands",
        {
            "request_id": "01234567-89ab-cdef-0123-456789abcdef",
            "command": "compact",
            "args": "",
        },
    ),
    _DoorRoute(
        "answer",
        "POST",
        "/v1/desktop/sessions/{session_id}/answers",
        {
            "epoch": "epoch",
            "request_id": "01234567-89ab-cdef-0123-456789abcdef",
            "approved": True,
        },
    ),
    _DoorRoute(
        "watch",
        "POST",
        "/v1/desktop/sessions/{session_id}/watch",
        {"subscription_id": "0" * 32, "visible": False, "can_notify": False},
    ),
    _DoorRoute("warm", "POST", "/v1/desktop/sessions/{session_id}/warm", {}),
)

#: Routes that refuse WITHOUT the door, because they never take a bridge: a
#: refusal there is the pool's own (``DesktopSessions.create``) and there is no
#: other entry. Named and one entry long, so the completeness test below can
#: assert the whole plane rather than a subset of it.
GATED_WITHOUT_THE_DOOR = frozenset({"create_session"})

_DESKTOP_ROUTER_MODULES = (
    "local_operator.server.routes.desktop_sessions",
    "local_operator.server.routes.desktop_lifecycle",
    "local_operator.server.routes.desktop_catalogues",
)


@dataclass(frozen=True)
class _PlaneRoute:
    """One desktop route as the ROUTER describes it: handler, method, template."""

    endpoint: str
    method: str
    template: str


@dataclass(frozen=True)
class _Module:
    """One module of the desktop plane, as the walks below see it."""

    name: str
    path: Path


def _modules(*names: str) -> list[_Module]:
    """The named modules, IMPORTED rather than globbed.

    Importing is the point: a walk over the filesystem would report a router that
    is never mounted (and so cannot serve a request), and would miss one mounted
    from somewhere the glob does not reach. These are the modules the app mounts
    (``server/app.py``'s ``include_router`` calls) plus the pool itself.
    """
    loaded: list[_Module] = []
    for name in names:
        module = importlib.import_module(name)
        loaded.append(_Module(name, Path(cast(str, module.__file__))))
    return loaded


def _router_modules() -> list[_Module]:
    """The three routers that serve the desktop plane's session routes."""
    return _modules(*_DESKTOP_ROUTER_MODULES)


def _pool_module() -> _Module:
    return _modules("local_operator.server.utils.desktop_sessions")[0]


def _plane_routes() -> dict[str, _PlaneRoute]:
    """Every route the desktop routers publish, keyed by HANDLER NAME."""
    found: dict[str, _PlaneRoute] = {}
    for name in _DESKTOP_ROUTER_MODULES:
        module = importlib.import_module(name)
        for route in module.router.routes:
            handler = route.endpoint.__name__
            methods = sorted(set(route.methods or set()) - {"HEAD", "OPTIONS"})
            found[handler] = _PlaneRoute(handler, methods[0] if methods else "", route.path)
    return found


def _enclosing(tree: ast.Module, lineno: int) -> str:
    """The innermost function containing ``lineno``, by name."""
    best: tuple[int, str] | None = None
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        end = node.end_lineno or node.lineno
        if node.lineno <= lineno <= end and (best is None or node.lineno > best[0]):
            best = (node.lineno, node.name)
    return best[1] if best else "<module>"


def _door_handlers(module: _Module) -> set[str]:
    """AST walk: this module's own handlers that obtain a session bridge.

    Module-level functions only — the router's handlers — matched on the call
    ``….session(…)``, which IS the door: ``DesktopSessions.session`` is the only
    thing in the process that builds a bridge (see :func:`_door_bypasses`), so
    "obtains a bridge" and "reaches the door" are the same set.
    """
    tree = ast.parse(module.path.read_text(encoding="utf-8"))
    reached: set[str] = set()
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for inner in ast.walk(node):
            if (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Attribute)
                and inner.func.attr == "session"
            ):
                reached.add(node.name)
                break
    return reached


def _door_bypasses(modules: list[_Module]) -> list[str]:
    """Every place in ``modules`` that obtains a bridge WITHOUT going through the door.

    THE MECHANISM GUARD, and the reason the refusal does not depend on anyone
    remembering to ask. There are exactly two ways round ``DesktopSessions.session``
    and both are reported here:

    * **building a bridge** — ``DesktopSessionBridge(...)`` anywhere but inside the
      pool's own ``session`` is a second door by definition, and a later edit that
      wanted to hand out a bridge "just this once" would be caught;
    * **reaching the pool's cache** — ``pool.bridges[session_id]`` in a handler
      skips the refusal entirely and is the way round that a route would actually
      take, so a read of ``….bridges`` from outside the pool's module is reported
      as well.

    A list of offenders rather than a boolean so the failure names the line — and
    so the falsification test below can drive this same code over a scratch copy.
    """
    pool = _pool_module().path
    offenders: list[str] = []
    for module in modules:
        tree = ast.parse(module.path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "DesktopSessionBridge"
            ):
                owner = _enclosing(tree, node.lineno)
                if module.path != pool or owner != "session":
                    offenders.append(
                        f"{module.name}:{node.lineno} builds a bridge in {owner}() "
                        "instead of obtaining one from DesktopSessions.session"
                    )
            if (
                isinstance(node, ast.Attribute)
                and node.attr == "bridges"
                and not isinstance(node.ctx, ast.Store)
                and module.path != pool
            ):
                offenders.append(
                    f"{module.name}:{node.lineno} reaches the pool's bridge cache directly "
                    f"from {_enclosing(tree, node.lineno)}()"
                )
    return offenders


def test_the_door_is_the_only_way_to_a_bridge() -> None:
    """The property that makes ``DesktopSessions.session`` the ENFORCEMENT.

    Round 2's MAJOR-1 was that the gate lived on a maintained list of routes, and
    five routes in the same plane were not on it. The fix is only a fix if the
    door is the only way in — so this asserts that for the modules that can serve a
    desktop request, and the next test shows the assertion can go red.
    """
    assert _door_bypasses([*_router_modules(), _pool_module()]) == []


def test_the_mechanism_walk_catches_a_route_that_bypasses_the_door(tmp_path: Path) -> None:
    """The guard above, driven in the direction that FAILS.

    A scratch copy of the real router module with two synthetic ungated handlers
    appended — one reaching the pool's cache, one building its own bridge, which is
    what a route added without the door could look like — must be reported. Without
    this, \"no bypasses\" would be a claim about a walk nobody has seen fail.
    """
    source = next(m for m in _router_modules() if m.name.endswith("desktop_lifecycle"))
    scratch = tmp_path / "scratch_desktop_lifecycle.py"
    scratch.write_text(
        source.path.read_text(encoding="utf-8")
        + "\n\n@router.post('/v1/desktop/sessions/{session_id}/frobnicate')\n"
        "async def frobnicate(session_id: str, request: Request):\n"
        "    bridge = host(request).bridges[session_id]\n"
        "    await bridge.remote.bind_runtime()\n"
        "    return reply({'data': {}})\n"
        "\n\n@router.post('/v1/desktop/sessions/{session_id}/widgets')\n"
        "async def widgets(session_id: str, request: Request):\n"
        "    bridge = DesktopSessionBridge(host(request).root, session_id, '/tmp')\n"
        "    await bridge.acquire()\n"
        "    return reply({'data': {}})\n",
        encoding="utf-8",
    )
    offenders = _door_bypasses([_Module("scratch.desktop_lifecycle", scratch)])
    assert any("bridge cache directly" in line for line in offenders), offenders
    assert any("builds a bridge in widgets()" in line for line in offenders), offenders


def test_the_refusal_matrix_covers_every_route_that_reaches_the_door() -> None:
    """The matrix above is complete BY CONSTRUCTION, and this is the construction.

    Walked, not remembered: the handlers come from an AST walk of the router
    modules and the paths from the routers they build, so a route that reaches the
    door and has no row in ``REFUSAL_MATRIX`` fails HERE. That is what round 2's
    MAJOR-1 asks for — the route list stops being the mechanism and becomes a test
    — and it is the difference between this and the five-row table it replaces,
    which named ``/messages``, ``/commands`` and ``/warm`` while ``/mcp``,
    ``/credentials``, ``/fork``, ``/asides`` and ``/adopt`` reached the spawn seam
    on a latched daemon unmentioned.

    ``create_session`` is the one route that refuses WITHOUT the door (the pool
    refuses it directly — it needs no bridge because it makes a session rather than
    a runtime), and it is named in ``GATED_WITHOUT_THE_DOOR`` rather than left out,
    so the two sets together are the whole plane.
    """
    plane = _plane_routes()
    reached: set[str] = set()
    for module in _router_modules():
        reached |= _door_handlers(module)
    covered = {row.endpoint for row in REFUSAL_MATRIX}
    assert reached | set(GATED_WITHOUT_THE_DOOR) == covered, (
        "REFUSAL_MATRIX must cover exactly the routes that reach the door "
        f"(walked: {sorted(reached)}, gated otherwise: {sorted(GATED_WITHOUT_THE_DOOR)}, "
        f"covered: {sorted(covered)})"
    )
    unknown = sorted(endpoint for endpoint in covered if endpoint not in plane)
    assert unknown == [], f"a row names a handler the desktop routers do not publish: {unknown}"


def _claimed_receipts(config_dir: Path) -> set[str]:
    """The request ids a receipt has been claimed for, read live from the journal.

    ``DesktopReceipts`` INSERTs the row BEFORE the operation runs, so a claimed id is
    a side effect that outlives the request — which is why "refused" has to mean "no
    row" rather than "no db": the database is created by the first claim of any
    kind, and a refused request is exactly the one that must leave no trace in it.
    """
    path = config_dir / "desktop-receipts.db"
    if not path.exists():
        return set()
    with closing(sqlite3.connect(path)) as db:
        return {row[0] for row in db.execute("SELECT id FROM receipts")}


def _fill(value: Any, session_id: str) -> Any:
    """Replace the placeholders a row uses where the rest is only known at run time."""
    if isinstance(value, dict):
        return {key: _fill(item, session_id) for key, item in value.items()}
    if isinstance(value, list):
        return [_fill(item, session_id) for item in value]
    if value == "SESSION_ID":
        return session_id
    return "counter" if value == "KEY" else value


def _request_for(route: _DoorRoute, session_id: str) -> tuple[str, str, dict[str, Any] | None]:
    """A row as ``(method, url, payload)``, with the session id substituted."""
    url = route.template.replace("{session_id}", session_id).replace("{aside_id}", ASIDE_ID)
    url = url.replace("{key}", "counter")
    if route.session_in_query:
        url += f"?session_id={session_id}"
    if route.query:
        url += ("&" if "?" in url else "?") + route.query
    payload = None if route.payload is None else _fill(route.payload, session_id)
    return route.method, url, cast(dict[str, Any] | None, payload)


@pytest.mark.parametrize("route", REFUSAL_MATRIX, ids=lambda route: route.endpoint)
@pytest.mark.asyncio
async def test_every_route_that_reaches_the_door_refuses_once_latched(
    route: _DoorRoute,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    restore_app_state: None,
) -> None:
    """The refusal matrix, one row per route that reaches the door.

    Round 1 measured the gap by execution: on a latched daemon ``POST /messages``
    returned 200 and reached ``admit_prompt`` while ``/warm`` answered 503 against
    the same process — and ``admit_prompt``'s ``_ensure_bound`` is the one call in
    the plane that can START a session runtime, which is what ``warm``'s refusal
    exists to prevent. ``/commands`` reaches it more directly still
    (``bind_runtime()``). Round 2's MAJOR-1 measured the SAME gap one layer out:
    five further routes (``/mcp``, ``/credentials``, ``/fork`` and its child
    admission, ``/asides``, ``/adopt``) reached that seam while the body, the
    README and this module's own five-row table said the refusal was complete.

    Every row asserts the TYPED refusal AND that no runtime was started: the spawn
    seam (``_ensure_bound``/``warm_runtime``) is patched to fail loudly, so a path
    that reached it would fail this test rather than quietly start a process. That
    instrument is exercised in the failing direction by
    ``test_the_spawn_seam_spy_catches_a_route_the_door_does_not_cover`` below —
    round 2's MINOR-1 was exactly that its earlier form could not distinguish
    "refused" from "never tried".

    The rows do NOT include the reads that stay open (``GET /v1/desktop/sessions``,
    ``GET /health``, the record file): those never take a bridge, which is why the
    door can be the gate for everything that does.
    """
    from local_operator.server.app import app
    from local_operator.session.attached import AttachedSession

    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", "matrix-token")
    app.state.config_manager = SimpleNamespace(config_dir=tmp_path)
    # The rows that declare `Depends(get_desktop_auth)` resolve it before the
    # handler runs — before the door, which is inside the handler — so the plane's
    # credential manager is part of the harness rather than an accident of the
    # route order (`entities` is the one row that needs it).
    app.state.credential_manager = CredentialManager(tmp_path)
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

    method, url, payload = _request_for(route, session_id)
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://localhost",
            headers={"Authorization": "Bearer matrix-token"},
            # A GET row that streams (``/events``) would hang the read if it ever
            # stopped refusing, so the budget is explicit rather than the harness'
            # default: a regression fails here instead of wedging CI.
            timeout=Timeout(15.0),
        ) as client:
            response = await client.request(method, url, json=payload)

        assert (
            response.status_code == 503
        ), f"{route.endpoint}: {response.status_code} {response.text[:200]}"
        detail = response.json()["detail"]
        assert detail["code"] == "daemon-retiring", route.endpoint
        assert detail["message"] == retire.RETIRING_MESSAGE, route.endpoint
        assert started == [], f"{route.endpoint} reached the spawn seam"
        bridge = pool.bridges.get(session_id)
        assert bridge is None or bridge.warm_task is None, f"{route.endpoint} started a warm"
        assert _claimed_receipts(tmp_path) == set(), (
            f"{route.endpoint} claimed a receipt before refusing, which leaves the "
            "client's retry indeterminate"
        )
        # `or {}` because sibling tests in this suite tear down by setting these
        # attributes to None rather than deleting them, and a refusal must be
        # judged against an EMPTY store either way.
        assert REQUEST_ID not in (
            getattr(app.state, "desktop_asides", None) or {}
        ), f"{route.endpoint} claimed aside state before refusing"
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_the_spawn_seam_spy_catches_a_route_the_door_does_not_cover(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    restore_app_state: None,
) -> None:
    """The instrument above, exercised in the direction that FAILS (MINOR-1).

    Round 2's MINOR-1 finding was that the evidence for "no runtime was started"
    could not fail: it argued from ``run/mobile`` being empty, which an isolated
    config root produces whether the route was refused or never tried at all. The
    argument is replaced by a spy on the seam (above), and THIS test is what makes
    the spy meaningful: with the door's refusal removed — the state review round 2
    measured on a real daemon — the same request, on the same latched pool, through
    the same route, reaches ``_ensure_bound`` and the spy fires.

    ``/mcp`` is the route because it is the one the reviewer measured: it reached
    ``bind_runtime()`` on a latched daemon with the daemon log naming the spawn
    attempt (``engage:``/``could not start a runtime``).
    """
    from local_operator.server.app import app
    from local_operator.session.attached import AttachedSession

    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", "matrix-token")
    app.state.config_manager = SimpleNamespace(config_dir=tmp_path)
    # The rows that declare `Depends(get_desktop_auth)` resolve it before the
    # handler runs — before the door, which is inside the handler — so the plane's
    # credential manager is part of the harness rather than an accident of the
    # route order (`entities` is the one row that needs it).
    app.state.credential_manager = CredentialManager(tmp_path)
    app.state.serve_retiring = False
    pool = DesktopSessions(
        tmp_path,
        retiring=lambda: bool(getattr(app.state, retire.RETIRING_STATE_ATTR, False)),
    )
    app.state.desktop_sessions = pool
    session_id = await pool.create(str(tmp_path))
    app.state.serve_retiring = True

    reached: list[str] = []

    async def _spy(*_args: Any, **_kwargs: Any) -> None:
        reached.append("spawn")
        raise AssertionError("the runtime spawn seam was entered")

    monkeypatch.setattr(AttachedSession, "_ensure_bound", _spy)
    # ONLY the door is removed: everything else about the daemon — the probe, the
    # latch, the route and the seam — is exactly what the matrix above drives.
    monkeypatch.setattr(DesktopSessions, "assert_admitting", lambda self: None)
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://localhost",
            headers={"Authorization": "Bearer matrix-token"},
        ) as client:
            with pytest.raises(AssertionError, match="spawn seam was entered"):
                await client.post(f"/v1/desktop/sessions/{session_id}/mcp", json={"action": "list"})
        assert reached == ["spawn"], "the spy did not see the path reach the seam"
    finally:
        await pool.close()


# =============================================================================
# The announcement write, the probes, the reload child and the baseline
# =============================================================================


@pytest.mark.asyncio
async def test_a_duplicate_aside_request_answers_409_and_claims_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    restore_app_state: None,
) -> None:
    """The aside store's duplicate check is about the REQUEST, and it answers first.

    ``/asides`` keys its off-record entries on the request id, and a repeated id is
    wrong whichever daemon receives it — so that check runs before the door, and on
    a latched daemon it answers the ``409`` the client gets in every other state.
    Pinned here rather than left as an anomaly in a transcript, and pinned in the
    form that matters: the 409 must not be the surface of a request that ADMITTED
    anything. A FRESH id on the same route is the typed 503, which
    ``test_every_route_that_reaches_the_door_refuses_once_latched`` asserts.
    """
    from local_operator.server.app import app
    from local_operator.server.routes.desktop_lifecycle import Aside
    from local_operator.session.attached import AttachedSession

    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", "matrix-token")
    app.state.config_manager = SimpleNamespace(config_dir=tmp_path)
    app.state.credential_manager = CredentialManager(tmp_path)
    # The claim the duplicate check reads, in the shape the route itself leaves: the
    # store is keyed by request id and the route only asks whether it is there.
    app.state.desktop_asides = {
        REQUEST_ID: Aside(session_id="", turns=[], created=time.monotonic())
    }
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
        raise AssertionError("a duplicate aside request reached the runtime spawn seam")

    monkeypatch.setattr(AttachedSession, "_ensure_bound", _no_spawn)
    monkeypatch.setattr(AttachedSession, "warm_runtime", _no_spawn)
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://localhost",
            headers={"Authorization": "Bearer matrix-token"},
        ) as client:
            response = await client.post(
                f"/v1/desktop/sessions/{session_id}/asides",
                json={"request_id": REQUEST_ID, "text": "hello"},
            )

        assert response.status_code == 409, response.text
        assert "already used" in response.json()["detail"]
        assert started == [], "the duplicate reached the spawn seam"
        assert _claimed_receipts(tmp_path) == set()
        assert list(app.state.desktop_asides) == [REQUEST_ID], "the claim was rewritten"
    finally:
        await pool.close()


# ---------------------------------------------------------------------------
# the announcement is re-read (MINOR-2) and a stamp nobody can read is not a move
# (OBS-1): both ends of the same rule, driven through the REAL poll
# ---------------------------------------------------------------------------


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
