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
