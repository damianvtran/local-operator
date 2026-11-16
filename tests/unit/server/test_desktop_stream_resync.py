"""An unqueueable frame RESYNCS each viewer in band instead of cutting them all.

QA Q2 on #4012: a prompt during a cold big session's warm-up led the bridge to
publish ONE ~37 MB ``history_delta``. Every subscriber queue is bounded at
``REPLAY_BYTES`` (8 MiB), so the frame could never be queued and EVERY opened
viewer was disconnected with ``subscriber overflow`` and a ``gap`` -- on the
operator's ~150 MB session, every desktop window at once.

The bridge now queues a small ``RESYNC_MARKER`` in that frame's place, and
``events`` turns it into an in-band ``gap`` plus a fresh ``snapshot`` on the
same connection (the pair the desktop renderer already handles on reconnect).
These tests pin that contract: no viewer is cut, each converges on the state a
fresh open would show, small frames still go inline, order is kept, a pending
open is not disturbed, reconnect stays honest, and one slow viewer cannot hold
the others.
"""

import asyncio
import json
from typing import Any

import pytest

from local_operator.server.utils import desktop_sessions as module
from local_operator.server.utils.desktop_sessions import DesktopSessions


@pytest.fixture(autouse=True)
def _fresh_snapshot_pool() -> Any:
    """Each test gets its own snapshot pool, joined at teardown (#4023 R1-n1).

    The resync snapshots build on the process-wide pool; without the join its
    worker thread outlives the test and the thread-leak reporter flags it --
    the same fixture ``test_desktop_snapshot_off_loop`` uses.
    """
    module.shutdown_snapshot_pool(wait=True)
    yield
    module.shutdown_snapshot_pool(wait=True)


async def _opened(bridge: Any) -> tuple[Any, Any]:
    """A subscriber past its OPEN handshake (open + snapshot consumed)."""
    sub = bridge.subscribe()
    stream = bridge.events(sub, epoch=bridge.epoch, after_seq=bridge.sequence)
    assert (await asyncio.wait_for(anext(stream), timeout=5))["type"] == "open"
    assert (await asyncio.wait_for(anext(stream), timeout=5))["type"] == "snapshot"
    return sub, stream


async def _take(stream: Any, count: int) -> list[dict[str, Any]]:
    return [await asyncio.wait_for(anext(stream), timeout=5) for _ in range(count)]


def _oversized_history_delta() -> dict[str, Any]:
    """A history delta whose frame is larger than any subscriber queue (>8 MiB)."""
    row = {"id": "", "role": "user", "content": [{"type": "text", "text": "x" * 4096}]}
    messages = [dict(row, id=f"m{n}") for n in range(2200)]
    payload = {"type": "history_delta", "messages": messages, "reset": True}
    assert len(json.dumps(payload).encode()) > module.REPLAY_BYTES
    return payload


def _comparable(snapshot: dict[str, Any]) -> dict[str, Any]:
    """A snapshot's payload minus the fields that legitimately differ per cut."""
    payload = dict(snapshot["payload"])
    payload.pop("subscription_id", None)
    return payload


@pytest.mark.asyncio
async def test_an_oversized_reset_resyncs_three_viewers_instead_of_cutting_them(tmp_path):
    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    async with pool.session(sid) as bridge:
        viewers = [await _opened(bridge) for _ in range(3)]
        bridge.publish("event", {"type": "message_end", "message_id": "before"})
        oversized_seq = bridge.sequence + 1
        bridge.publish("event", _oversized_history_delta())
        # Published before any viewer reads: at or below every resync watermark,
        # so it is superseded by the snapshot (the open path's own split).
        bridge.publish("event", {"type": "message_end", "message_id": "superseded"})

        for sub, _ in viewers:
            assert not sub.overflow, "an unqueueable frame must not cut the viewer"
            assert sub.queued_bytes < 64 * 1024, "only small frames were queued"
            assert sub.resync_pending
        assert all(size <= module.REPLAY_BYTES for _, size in bridge.replay)

        for sub, stream in viewers:
            before, gap, resynced = await _take(stream, 3)
            assert before["payload"]["message_id"] == "before"
            assert gap == {"type": "gap", "session_id": sid}
            assert resynced["type"] == "snapshot"
            assert resynced["seq"] >= oversized_seq, "the resync covers the oversized frame"
            assert not sub.resync_pending
            assert not sub.overflow
            # Converges on exactly what a fresh snapshot of the bridge shows now.
            current = await bridge.snapshot()
            assert _comparable(resynced) == _comparable(current)
        # Live delivery continues on the SAME connections after the resync.
        bridge.publish("event", {"type": "message_end", "message_id": "after"})
        for sub, stream in viewers:
            (after,) = await _take(stream, 1)
            assert after["payload"]["message_id"] == "after"
            assert sub.queue.empty() and sub.queued_bytes == 0
            await stream.aclose()


@pytest.mark.asyncio
async def test_a_small_reset_is_still_delivered_inline(tmp_path):
    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    async with pool.session(sid) as bridge:
        sub, stream = await _opened(bridge)
        payload = {"type": "history_delta", "messages": [{"id": "m1"}], "reset": True}
        bridge.publish("event", payload)
        (frame,) = await _take(stream, 1)
        assert frame["type"] == "event" and frame["payload"] == payload
        assert not sub.resync_pending and not sub.overflow
        assert bridge.replay[-1][0]["payload"] == payload, "a small frame stays replayable"
        await stream.aclose()


@pytest.mark.asyncio
async def test_frames_behind_the_marker_are_superseded_or_delivered_in_order(tmp_path):
    """Order: before-marker frames first, then gap+snapshot, then only NEWER frames.

    Two oversized frames back to back queue ONE marker (the resync covers both),
    and once it has been served the next oversized frame queues a new one.
    """
    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    async with pool.session(sid) as bridge:
        sub, stream = await _opened(bridge)
        bridge.publish("event", {"type": "message_end", "message_id": "a"})
        bridge.publish("event", _oversized_history_delta())
        bridge.publish("event", {"type": "message_end", "message_id": "b"})
        bridge.publish("event", _oversized_history_delta())
        markers = [f for f, _ in list(sub.queue._queue) if f["type"] == module.RESYNC_MARKER]
        assert len(markers) == 1, "one pending marker covers every oversized frame behind it"

        first, gap, resynced = await _take(stream, 3)
        assert first["payload"]["message_id"] == "a"
        assert gap["type"] == "gap" and resynced["type"] == "snapshot"
        assert resynced["seq"] == bridge.sequence, "the cut is taken when the marker is read"
        assert sub.queue.empty(), "frame b is at or below the watermark: superseded"

        bridge.publish("event", {"type": "message_end", "message_id": "c"})
        bridge.publish("event", _oversized_history_delta())
        c, gap2, resynced2 = await _take(stream, 3)
        bridge.publish("event", {"type": "message_end", "message_id": "d"})
        (d,) = await _take(stream, 1)
        assert c["payload"]["message_id"] == "c"
        assert gap2["type"] == "gap" and resynced2["type"] == "snapshot"
        assert d["payload"]["message_id"] == "d"
        seqs = [c["seq"], resynced2["seq"], d["seq"]]
        assert seqs == sorted(seqs) and len(set(seqs)) == 3
        await stream.aclose()


@pytest.mark.asyncio
async def test_a_slow_viewer_cannot_hold_the_others_and_reconnect_stays_honest(tmp_path):
    """One viewer that reads nothing does not delay another's resync, and an old
    cursor that missed the oversized frame is answered with a gap, not a skip."""
    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    async with pool.session(sid) as bridge:
        slow, slow_stream = await _opened(bridge)
        fast, fast_stream = await _opened(bridge)
        cursor = bridge.sequence
        bridge.publish("event", _oversized_history_delta())
        gap, resynced = await _take(fast_stream, 2)
        assert gap["type"] == "gap" and resynced["type"] == "snapshot"
        assert slow.resync_pending and not slow.overflow, "the slow viewer still waits"
        assert not bridge.replay, "the unqueueable frame is never retained for replay"

        late = bridge.subscribe()
        late_stream = bridge.events(late, epoch=bridge.epoch, after_seq=cursor)
        opened = await asyncio.wait_for(anext(late_stream), timeout=5)
        assert opened["type"] == "open" and opened["payload"]["gap"] is True
        await late_stream.aclose()
        await fast_stream.aclose()
        await slow_stream.aclose()


@pytest.mark.asyncio
async def test_a_viewer_still_opening_gets_no_marker(tmp_path):
    """Before its snapshot reads the watermark, the open snapshot supersedes the frame."""
    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    async with pool.session(sid) as bridge:
        sub = bridge.subscribe()
        bridge.publish("event", _oversized_history_delta())
        assert sub.queue.empty() and not sub.resync_pending and not sub.overflow
        stream = bridge.events(sub, epoch=bridge.epoch, after_seq=bridge.sequence)
        assert (await asyncio.wait_for(anext(stream), timeout=5))["type"] == "open"
        snap = await asyncio.wait_for(anext(stream), timeout=5)
        assert snap["type"] == "snapshot" and snap["seq"] == bridge.sequence
        await stream.aclose()
