"""The relay's ``heartbeat_stale`` verdict behind the phone header's status dot.

WHY THESE CASES. The dot paints YELLOW from this one boolean, so every way it
could lie is a case here: a streaming turn whose heartbeat writer is starved
(must stay fresh — busy is green), an idle runtime that sends no frames but
keeps beating (fresh), an owner quiet on BOTH clocks (stale), a never-stamped
record (no evidence of answering), and the wire shape an older client and a
disk rebuild depend on (``None`` is popped, never sent as null). The scan's
publish step is asserted to repaint only on a transition, and a live frame from
the owner to clear the verdict through the real dial loop.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from typing import Any, cast

import pytest

from local_operator.mobile import daemon as daemon_module
from local_operator.mobile.daemon import (
    SESSION_HEALTH_WINDOW_S,
    MobileDaemon,
    SessionEntry,
    _dial,
    _sync_heartbeat_stale,
    heartbeat_stale,
)
from local_operator.mobile.types import (
    HEARTBEAT_INTERVAL_S,
    HEARTBEAT_TIMEOUT_S,
    SessionProjection,
    SessionRecord,
)


def _record(*, heartbeat_at: float, port: int = 1) -> SessionRecord:
    return SessionRecord(
        pid=4242,
        kind="tui",
        session_id="5cfeedfacade",
        conversation_name="health",
        cwd="/tmp",
        model_label="test/model",
        control_port=port,
        control_key="fixture",
        heartbeat_at=heartbeat_at,
    )


def test_window_is_the_registry_timeout_three_beats() -> None:
    # ONE window, tied to the bound the catalogue already calls "not answering".
    assert SESSION_HEALTH_WINDOW_S == HEARTBEAT_TIMEOUT_S
    assert SESSION_HEALTH_WINDOW_S == 3 * HEARTBEAT_INTERVAL_S


@pytest.mark.parametrize(
    ("beat_age", "frame_age", "stale"),
    [
        # Idle runtime: beating, no frames since long ago (or ever).
        (5.0, None, False),
        (5.0, 10_000.0, False),
        # Streaming turn: beat starved past the window, frames flowing.
        (205.8, 0.1, False),
        # Both quiet past the window: the owner is not answering.
        (SESSION_HEALTH_WINDOW_S + 1, SESSION_HEALTH_WINDOW_S + 1, True),
        (SESSION_HEALTH_WINDOW_S + 1, None, True),
        # Exactly at the window is still inside it (strict ``>``, as classify).
        (SESSION_HEALTH_WINDOW_S, None, False),
    ],
)
def test_verdict_needs_both_clocks_quiet(
    beat_age: float, frame_age: float | None, stale: bool
) -> None:
    wall, mono = 1_000_000.0, 50_000.0
    entry = SessionEntry(_record(heartbeat_at=wall - beat_age))
    entry.last_frame_at = None if frame_age is None else mono - frame_age
    assert heartbeat_stale(entry, now=wall, monotonic_now=mono) is stale


def test_a_never_stamped_record_is_no_evidence_of_answering() -> None:
    entry = SessionEntry(_record(heartbeat_at=0.0))
    assert heartbeat_stale(entry, now=1_000.0, monotonic_now=1_000.0) is True
    entry.last_frame_at = 999.0
    assert heartbeat_stale(entry, now=1_000.0, monotonic_now=1_000.0) is False


def test_wire_pops_none_and_carries_a_verdict() -> None:
    projection = SessionProjection(session_id="s", pid=1, kind="tui")
    # A disk rebuild / a never-[redacted] projection: the key is ABSENT, so an
    # older client and today's durable frame bytes are unchanged.
    assert "heartbeat_stale" not in projection.to_json()
    projection.heartbeat_stale = True
    assert projection.to_json()["heartbeat_stale"] is True
    projection.heartbeat_stale = False
    assert projection.to_json()["heartbeat_stale"] is False


def test_scan_publish_repaints_only_on_a_transition(monkeypatch) -> None:
    daemon = MobileDaemon(port=0, password="pw")
    entry = SessionEntry(_record(heartbeat_at=time.time()))
    entry.projection = SessionProjection(session_id="5cfeedfacade", pid=4242, kind="tui")
    fanned: list[bool | None] = []
    monkeypatch.setattr(
        daemon_module, "_fan_out", lambda e, _d=None: fanned.append(e.projection.heartbeat_stale)
    )

    _sync_heartbeat_stale(entry, daemon)  # None -> False: a transition
    _sync_heartbeat_stale(entry, daemon)  # unchanged: no repaint
    entry.record.heartbeat_at = time.time() - SESSION_HEALTH_WINDOW_S - 5
    _sync_heartbeat_stale(entry, daemon)  # False -> True
    _sync_heartbeat_stale(entry, daemon)  # unchanged
    assert fanned == [False, True]
    assert entry.projection.heartbeat_stale is True

    # An ended entry publishes nothing: ``ended`` outranks the verdict.
    entry.ended = True
    entry.record.heartbeat_at = time.time()
    _sync_heartbeat_stale(entry, daemon)
    assert fanned == [False, True]


def test_scan_once_publishes_the_verdict_for_a_live_entry(monkeypatch) -> None:
    """Through ``_scan_once`` itself, so the call site is proved wired."""
    from local_operator.session.runtime import registry

    daemon = MobileDaemon(port=0, password="pw", dial_registrants=False)
    quiet = _record(heartbeat_at=time.time() - SESSION_HEALTH_WINDOW_S - 5)
    entry = SessionEntry(quiet)
    entry.projection = SessionProjection(session_id=quiet.session_id, pid=quiet.pid, kind="tui")
    daemon.table.entries[quiet.pid] = entry
    monkeypatch.setattr(registry, "scan", lambda: [(quiet, "wedged")])
    asyncio.run(daemon._scan_once())
    published = daemon.table.entries[quiet.pid].projection
    assert published is not None
    assert published.heartbeat_stale is True


@pytest.mark.asyncio
async def test_a_live_frame_clears_the_verdict_through_the_dial_loop() -> None:
    async def serve(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await reader.readline()  # the daemon's auth line
        frame = {
            "op": "projection",
            "data": {"session_id": "5cfeedfacade", "pid": 4242, "conversation_name": "health"},
        }
        writer.write(json.dumps(frame).encode() + b"\n")
        await writer.drain()
        writer.close()
        with contextlib.suppress(ConnectionResetError):
            await writer.wait_closed()

    server = await asyncio.start_server(serve, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    daemon = MobileDaemon(port=0, password="pw")
    # A beat long gone: only the frame can make this owner fresh.
    entry = SessionEntry(
        _record(heartbeat_at=time.time() - 10 * SESSION_HEALTH_WINDOW_S, port=port)
    )
    daemon.table.entries[entry.record.pid] = entry
    dial = asyncio.ensure_future(_dial(daemon, entry))
    try:
        for _ in range(50):
            if entry.projection is not None:
                break
            await asyncio.sleep(0.05)
        assert entry.projection is not None
        assert entry.projection.heartbeat_stale is False
        assert entry.last_frame_at is not None
        assert heartbeat_stale(entry) is False
    finally:
        dial.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await dial
        server.close()
        await server.wait_closed()


# ---------------------------------------------------------------------------
# The PHONE VIEWER link (agent review round 1, B1/M1). While a phone views a
# live session the relay holds a second connection to the owner,
# ``_connect_phone`` -> ``AttachClient``, and that client rebuilds every repaint
# against a placeholder ``pid=0`` record. Its frames are the ones the phone's
# SSE queue keeps, so they must carry the owner's pid (or the header dot reads
# a running session as "Ended") and the answering verdict (or every scan
# re-publishes a non-transition). Driven against a REAL ``RuntimeServer``.
# ---------------------------------------------------------------------------


class _ViewerHandle:
    """The smallest owner a real ``RuntimeServer`` will serve a viewer from."""

    def __init__(self, session_id: str) -> None:
        self._projection = SessionProjection(
            session_id=session_id,
            pid=0,
            kind="tui",
            conversation_name="viewed",
            cwd="/tmp",
            model_label="test/model",
        )

    @property
    def session_projection_seed(self) -> SessionProjection:
        return self._projection

    def subscribe(self, on_projection):  # noqa: ANN001, ANN202
        # Kept so the test can make the owner REPAINT: the welcome is rebuilt
        # against the real record, and only the repaints AFTER it travel
        # through ``AttachClient._pump``'s placeholder ``pid=0`` record.
        self.repaint = on_projection
        return lambda: None

    async def refresh(self) -> None:
        pass


async def _live_record() -> SessionRecord:
    from local_operator.session.runtime import registry

    deadline = asyncio.get_running_loop().time() + 5
    while asyncio.get_running_loop().time() < deadline:
        found = registry.scan()
        if found and found[0][1] == "live":
            return found[0][0]
        await asyncio.sleep(0.05)
    raise AssertionError("no live record")


@pytest.mark.asyncio
async def test_the_phone_viewer_link_serves_the_owner_pid_and_a_fresh_verdict(monkeypatch) -> None:
    from local_operator.session.runtime.server import RuntimeServer

    handle = _ViewerHandle("5cfeedfacade")
    # A REDUCED owner on purpose: the viewer path needs only the seed, a
    # subscription and refresh, and ``RuntimeServer`` probes the rest.
    registrant = RuntimeServer(cast(Any, handle), kind="tui")
    registrant.start()
    daemon = MobileDaemon(port=0, password="pw")
    try:
        record = await _live_record()
        assert record.pid > 0
        served: list[SessionProjection] = []
        real_fan_out = daemon_module._fan_out

        def spy(entry, d=None):  # noqa: ANN001, ANN202
            served.append(entry.projection)
            real_fan_out(entry, d)

        monkeypatch.setattr(daemon_module, "_fan_out", spy)
        # A phone is VIEWING (an SSE subscriber), so ``connect`` keeps the
        # viewer open past the welcome instead of releasing it at once.
        phone: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        daemon.table.session_subscribers["5cfeedfacade"] = {phone}
        await daemon._connect_phone(record)
        assert served, "the viewer link never painted its welcome"
        # The owner streams: a repaint after the welcome is the frame that
        # arrives through the pump.
        welcomes = len(served)
        handle.repaint()
        for _ in range(50):
            if len(served) > welcomes:
                break
            await asyncio.sleep(0.05)
        assert len(served) > welcomes, "the viewer link never relayed a repaint"
        entry = daemon.table.entries[record.pid]
        # B1: the phone is served the OWNER's pid, never the attach client's
        # placeholder 0 — the dot's "no running owner" rung reads it.
        assert entry.projection is not None
        assert entry.projection.pid == record.pid
        assert all(p.pid == record.pid for p in served)
        # M1: a viewer frame is the owner answering, exactly like a dial frame.
        assert entry.projection.heartbeat_stale is False
        assert entry.last_frame_at is not None

        # ...so once steady, scans publish NOTHING: 4 passes, 0 pushes.
        served.clear()
        for _ in range(4):
            daemon_module._sync_heartbeat_stale(entry, daemon)
        assert served == []
    finally:
        daemon.table.session_subscribers.pop("5cfeedfacade", None)
        daemon.release_phone_view("5cfeedfacade")
        registrant.close()


@pytest.mark.asyncio
async def test_qa_q1_replay_every_frame_the_phone_receives_keeps_the_live_pid() -> None:
    """QA round 1, Q1 (= review B1), replayed on the wire the phone reads.

    QA's sequence: a phone opens a LIVE session (its SSE queue subscribes and
    the relay's viewer link attaches), then the owner keeps painting — idle,
    busy, a pending approval, a send. 4.4 s in, every frame turned ``pid: 0``
    and ``heartbeat_stale`` fell off the wire, so the header dot read "Ended".
    Here the frames are read from the phone's own queue, serialized exactly as
    ``_fan_out`` puts them, so this asserts what the phone is SENT, not an
    attribute on a daemon object.
    """
    from local_operator.session.runtime.server import RuntimeServer

    handle = _ViewerHandle("5cfacade0001")
    registrant = RuntimeServer(cast(Any, handle), kind="tui")
    registrant.start()
    daemon = MobileDaemon(port=0, password="pw")
    phone: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    try:
        record = await _live_record()
        daemon.table.session_subscribers["5cfacade0001"] = {phone}
        await daemon._connect_phone(record)
        # The owner paints repeatedly, as a turn, an approval and a send do.
        for _ in range(3):
            handle.repaint()
            await asyncio.sleep(0.15)
        frames: list[dict[str, Any]] = []
        while not phone.empty():
            frames.append(phone.get_nowait())
        assert len(frames) >= 2, "the phone was not sent the viewer link's repaints"
        sent = [frame.get("data", frame) for frame in frames]
        # Q1's symptom first: the welcome kept the pid, every repaint lost it.
        assert [data["pid"] for data in sent] == [record.pid] * len(sent)
        # A frame from the owner IS an answer: the verdict rides every one.
        assert [data.get("heartbeat_stale") for data in sent] == [False] * len(sent)
        assert not any(data["ended"] for data in sent)
    finally:
        daemon.table.session_subscribers.pop("5cfacade0001", None)
        daemon.release_phone_view("5cfacade0001")
        registrant.close()
