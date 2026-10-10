"""The subscriber fold: a viewer's stream FOLDS queued delta runs under pressure.

Task 5.3 of the oscillation handoff (`~/lo-osc2/HANDOFF.md` §5.3; architect
decision `task-5.3-overflow-arm.md`). The measured shape: the desktop viewer is
episodically slow during "Thinking" bursts, and the queue's binding budget is
the FRAME COUNT (``REPLAY_COUNT = 256``), not the 8 MiB — a viewer is cut while
holding ~62-150 KB of reasoning deltas, and because an overflow is deliberately
excluded from the dwell and the grace (#1632), every cut rotates the epoch and
repaints from a cold snapshot: the reported oscillation.

The fix folds queued runs of the three streaming families (mirroring
``session/runtime/server.py::_compact_event_queue``) before the relief valve,
ONLY for an opened subscriber and ONLY when the frame would not fit. These
tests pin the fold's own contract:

* a burst folds instead of disconnecting, and the content arrives in order;
* a NON-foldable burst still cuts at 257 with ``gap`` and the WARNING;
* a merge that would exceed the per-frame cap is refused, and the valve fires;
* the merged head keeps the LATER seq (cursor safety) and replay keeps the
  ORIGINAL, unfolded frames;
* the pre-open eviction path (D1) never folds;
* every stored frame size is exact (``events`` subtracts it on dequeue).
"""

import asyncio
import json
import logging
from typing import Any

import pytest

from local_operator.server.utils import desktop_sessions as module
from local_operator.server.utils.desktop_sessions import DesktopSessions

LOGGER = "local_operator.server.utils.desktop_sessions"


async def _opened_sub(bridge: Any, *, after_seq: int = 0) -> tuple[Any, Any]:
    """A subscriber whose OPEN handshake is complete (open + snapshot consumed).

    ``sub.opened`` flips at the end of the handshake, and the fold only exists
    past it — before the handshake the queue's policy is eviction (D1), and
    folding there is forbidden by the same contract that makes eviction safe.

    ``after_seq`` defaults to 0, the cold cursor every fold test starts from.
    A later phase of a test that already published passes the bridge's CURRENT
    sequence instead, so the handshake has no replay frames in front of its
    snapshot and the same open/snapshot reads hold.
    """
    sub = bridge.subscribe()
    stream = bridge.events(sub, epoch=bridge.epoch, after_seq=after_seq)
    assert (await asyncio.wait_for(anext(stream), timeout=5))["type"] == "open"
    assert (await asyncio.wait_for(anext(stream), timeout=5))["type"] == "snapshot"
    return sub, stream


async def _drain(sub: Any, stream: Any) -> list[dict[str, Any]]:
    """Pull every frame the subscriber's queue currently holds, in order.

    Stops when the queue is empty rather than awaiting the stream's 15 s
    heartbeat timeout: these tests publish synchronously and drain afterwards,
    so the queue's length is the delivery count.
    """
    frames: list[dict[str, Any]] = []
    while not sub.queue.empty():
        frames.append(await asyncio.wait_for(anext(stream), timeout=5))
        assert len(frames) <= module.REPLAY_COUNT + 1, "the drain is bounded like the queue"
    return frames


def _peek_queue(sub: Any) -> list[tuple[dict[str, Any], int]]:
    """Read the queue's stored ``(frame, size)`` pairs without disturbing it.

    ``publish``/``events`` maintain ``queued_bytes``; a bare
    ``get_nowait``/``put_nowait`` pair does not, so the bridge is left exactly
    as it was found while a test asserts the stored sizes.
    """
    items: list[tuple[dict[str, Any], int]] = []
    while not sub.queue.empty():
        items.append(sub.queue.get_nowait())
    for item in items:
        sub.queue.put_nowait(item)
    return items


def _exact_size(frame: dict[str, Any]) -> int:
    """The module's own frame-size metric, so a stored size can be re-derived."""
    return len(json.dumps(frame, separators=(",", ":")).encode())


@pytest.mark.asyncio
async def test_a_reasoning_burst_folds_instead_of_disconnecting(tmp_path):
    """2,400 same-stream fragments with nothing reading must not cut the viewer.

    Before the fold this burst overflowed the count arm at frame 257 (~62 KB of
    an 8 MiB budget), and the cut forced a cold repaint — the reported
    oscillation. Now the queue folds runs of fragments as they accumulate, so
    the burst is delivered as a much smaller number of merged frames, in order.
    """
    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    async with pool.session(sid) as bridge:
        sub, stream = await _opened_sub(bridge)
        fragments = [f"[{n}]" for n in range(2400)]
        for fragment in fragments:
            bridge.publish(
                "event", {"type": "reasoning_delta", "message_id": "m-r", "delta": fragment}
            )

        assert not sub.overflow, "a foldable burst must never reach the relief valve"
        assert not sub.queue.full(), "and the fold must leave the queue room to breathe"
        delivered = await _drain(sub, stream)

        assert len(delivered) == 105, (
            "folds run at publish 257 and every 255 publishes after (last: 2297), "
            "so the queue holds a merged head plus 104 tail fragments"
        )
        assert "".join(frame["payload"]["delta"] for frame in delivered) == "".join(
            fragments
        ), "every fragment must arrive exactly once, in publish order"
        seqs = [frame["seq"] for frame in delivered]
        assert seqs == sorted(seqs) and len(set(seqs)) == len(
            seqs
        ), "the receipt cursor advances strictly: a merged head rides the LATER seq"
        assert seqs[-1] == bridge.sequence, "the newest frame keeps the newest seq"
        assert sub.queued_bytes == 0, "and delivery drains the byte accounting exactly"
        await stream.aclose()


@pytest.mark.asyncio
async def test_the_fold_fires_at_the_count_boundary_and_keeps_exact_sizes(tmp_path):
    """The 257th frame folds instead of cutting, and both stored sizes are exact.

    The boundary the count arm cuts at: 256 frames fill the queue, and until the
    next publish there is no pressure — so the fold runs exactly once here, and
    the queue afterwards is readable as two items: the merged run (head = the
    256th frame's seq) and the incoming 257th.
    """
    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    async with pool.session(sid) as bridge:
        sub, stream = await _opened_sub(bridge)
        fragments = [f"[{n}]" for n in range(256)]
        for fragment in fragments:
            bridge.publish(
                "event", {"type": "reasoning_delta", "message_id": "m-b", "delta": fragment}
            )
        seq_at_boundary = bridge.sequence
        assert not sub.overflow, "256 frames is the bound; nothing cuts yet"
        assert sub.queue.full(), "and the next publish is the one under pressure"

        bridge.publish("event", {"type": "reasoning_delta", "message_id": "m-b", "delta": "[256]"})
        assert not sub.overflow, "the 257th frame must fold the run instead of cutting"
        items = _peek_queue(sub)
        assert len(items) == 2, "one merged run plus the incoming frame"
        merged, merged_size = items[0]
        incoming, incoming_size = items[1]
        assert merged["payload"]["delta"] == "".join(fragments), "the whole run, in order"
        assert merged["seq"] == seq_at_boundary, "the merged head rides the LATER seq"
        assert incoming["payload"]["delta"] == "[256]"
        assert incoming["seq"] == seq_at_boundary + 1, "and the stream advances by one"
        assert merged_size == _exact_size(merged), (
            "a stored size must be the frame's exact encoded length: events() subtracts "
            "it on dequeue and publish() weighs it against REPLAY_BYTES"
        )
        assert incoming_size == _exact_size(incoming)
        assert sub.queued_bytes == merged_size + incoming_size
        await stream.aclose()


@pytest.mark.asyncio
async def test_a_fold_collapses_each_run_without_crossing_families(tmp_path):
    """Two adjacent runs of different families stay separate through the fold.

    The FAMILY is part of the merge key: a ``message_update`` run beside a
    ``reasoning_delta`` run must fold among its own frames only. Mixing them
    would pour reasoning text into the answer's accumulated delta on screen.
    """
    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    async with pool.session(sid) as bridge:
        sub, stream = await _opened_sub(bridge)
        reason = [f"r[{n}]" for n in range(256)]
        answer = [f"a[{n}]" for n in range(256)]
        for fragment in reason:
            bridge.publish(
                "event", {"type": "reasoning_delta", "message_id": "m-r", "delta": fragment}
            )
        # The 257th publish is the first message_update: it folds the reasoning
        # run first, then starts the answer's own run.
        for fragment in answer:
            bridge.publish(
                "event",
                {
                    "type": "message_update",
                    "message": {"id": "m-a", "role": "assistant"},
                    "delta": fragment,
                },
            )

        assert not sub.overflow
        delivered = await _drain(sub, stream)
        assert len(delivered) == 3, "a merged run per family, plus the tail frame"
        assert [frame["payload"]["type"] for frame in delivered] == [
            "reasoning_delta",
            "message_update",
            "message_update",
        ]
        assert delivered[0]["payload"]["delta"] == "".join(
            reason
        ), "the reasoning run is intact and did not absorb answer text"
        assert delivered[1]["payload"]["delta"] == "".join(
            answer[:-1]
        ), "the answer's run folded among its own frames only"
        assert delivered[2]["payload"]["delta"] == answer[-1]
        seqs = [frame["seq"] for frame in delivered]
        assert seqs == sorted(seqs) and seqs[-1] == bridge.sequence
        await stream.aclose()


@pytest.mark.asyncio
async def test_a_tool_execution_update_run_keeps_the_newest_snapshot(tmp_path):
    """``tool_execution_update`` is a self-replacing family: the newest frame wins.

    A frame re-sends the tool's current live view (a bounded tail for ``bash``,
    a bounded display for ``eval`` — never the whole transcript), so folding a
    run keeps only its newest frame, never a concatenation. 512 chunks must
    arrive as two frames: the view the fold kept (510) and the tail (511).
    """
    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    async with pool.session(sid) as bridge:
        sub, stream = await _opened_sub(bridge)
        count = 512
        for n in range(count):
            bridge.publish(
                "event",
                {
                    "type": "tool_execution_update",
                    "tool_call_id": "call-1",
                    "tool_name": "bash",
                    "partial_result": {"content": [{"type": "text", "text": "x" * 100 + str(n)}]},
                },
            )
        assert not sub.overflow

        delivered = await _drain(sub, stream)
        assert len(delivered) == 2, "the fold collapsed 511 snapshots into their newest, then 511"
        texts = [frame["payload"]["partial_result"]["content"][0]["text"] for frame in delivered]
        assert texts == [
            "x" * 100 + "510",
            "x" * 100 + "511",
        ], "keep-newest keeps a single frame's live view, not a concatenation"
        await stream.aclose()


@pytest.mark.asyncio
async def test_publish_to_subscription_folds_before_its_valve_and_never_merges_asides(tmp_path):
    """The aside path's fold: rescue a private frame, never merge two asides.

    ``publish_to_subscription`` is the aside sink's only delivery path, and its
    pressure branch folds FIRST, exactly as ``publish`` does, and only for an
    opened subscriber. An ``aside_delta`` frame carries no ``payload.type``, so
    it is not itself foldable — the fold helps only by reclaiming a sibling
    delta run that shares the queue. Two aside fragments must stay two frames:
    the aside's stream identity is the POST's request id on the route's side,
    so nothing in this fold may join them.
    """
    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    async with pool.session(sid) as bridge:
        sub, stream = await _opened_sub(bridge)
        fragments = [f"[{n}]" for n in range(256)]
        for fragment in fragments:
            bridge.publish(
                "event", {"type": "reasoning_delta", "message_id": "m-a", "delta": fragment}
            )
        assert sub.queue.full() and not sub.overflow, "the run alone fills the count arm"

        first = {"aside_id": "req-1", "delta": "a[0]"}
        assert bridge.publish_to_subscription("aside_delta", first, subscription_id=sub.id) is True
        assert not sub.overflow, "the fold made room for the private frame"

        second = {"aside_id": "req-1", "delta": "a[1]"}
        assert bridge.publish_to_subscription("aside_delta", second, subscription_id=sub.id) is True

        items = _peek_queue(sub)
        assert len(items) == 3, "the folded run, then each aside as its own frame"
        (merged, merged_size), (aside_one, one_size), (aside_two, two_size) = items
        assert merged["payload"]["delta"] == "".join(fragments), "the run folded in order"
        assert (
            aside_one["payload"] == first and aside_two["payload"] == second
        ), "two aside fragments never merge with each other"
        assert merged["seq"] < aside_one["seq"] < aside_two["seq"] == bridge.sequence
        for frame, size in items:
            assert size == _exact_size(frame), "stored sizes stay exact on the put path"
        assert sub.queued_bytes == merged_size + one_size + two_size

        delivered = await _drain(sub, stream)
        assert [frame["payload"] for frame in delivered] == [
            merged["payload"],
            first,
            second,
        ], "delivery order and content are exactly the queued order"
        assert all(
            frame["payload"].get("aside_id") is None for frame, _ in bridge.replay
        ), "the aside path is live-only: nothing it published entered the replay"
        await stream.aclose()

        # THE VALVE ARM: a queue of asides only has nothing the fold can
        # reclaim, so the same call still cuts. (A cold cursor would front this
        # handshake with replay frames; ``after_seq=bridge.sequence`` opens past
        # them.)
        sub2, stream2 = await _opened_sub(bridge, after_seq=bridge.sequence)
        for n in range(module.REPLAY_COUNT):
            assert bridge.publish_to_subscription(
                "aside_delta",
                {"aside_id": f"req-{n}", "delta": f"[{n}]"},
                subscription_id=sub2.id,
            )
        assert sub2.queue.full() and not sub2.overflow

        assert (
            bridge.publish_to_subscription("aside_delta", first, subscription_id=sub2.id) is False
        ), "nothing foldable in the queue: the valve must fire"
        assert sub2.overflow is True
        assert (await asyncio.wait_for(anext(stream2), timeout=5))["type"] == "gap"
        with pytest.raises(StopAsyncIteration):
            await anext(stream2)
        assert sub2.id not in bridge.subscribers


@pytest.mark.asyncio
async def test_a_genuinely_slow_reader_is_still_disconnected(tmp_path, caplog):
    """THE FALSIFICATION: non-foldable frames cut at 257, with ``gap`` + WARNING.

    The relief valve is unchanged. Frames no fold can help (here plain
    ``{"value": n}`` events) fill the count arm exactly as before the fold
    existed; the 257th publish disconnects the opened subscriber, the stream
    yields ``gap`` and stops, the subscription is popped, and the
    ``subscriber overflow`` WARNING still reaches a WARNING-level capture.
    """
    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        async with pool.session(sid) as bridge:
            sub, stream = await _opened_sub(bridge)
            for n in range(module.REPLAY_COUNT):
                bridge.publish("event", {"value": n})
            assert not sub.overflow, "256 frames is the bound; nothing cuts yet"
            assert sub.queue.full()

            bridge.publish("event", {"value": module.REPLAY_COUNT})
            assert sub.overflow is True, "a foldable nothing: the valve must fire"
            assert (
                sub.queue.qsize() == 1 and sub.queued_bytes == 0
            ), "the queue was drained to its None sentinel"
            assert (await asyncio.wait_for(anext(stream), timeout=5))["type"] == "gap"
            with pytest.raises(StopAsyncIteration):
                await anext(stream)
            assert sub.id not in bridge.subscribers, "and the relief valve revoked it"
        await asyncio.sleep(0)

    ended = [r for r in caplog.records if "desktop stream ended" in r.getMessage()]
    assert ended, "the cut must still be emitted, at the level the daemon runs"
    assert ended[-1].levelno == logging.WARNING
    assert (
        "subscriber overflow" in ended[-1].getMessage()
    ), "and it must still name the relief valve: this line is the instrument"


@pytest.mark.asyncio
async def test_a_merge_that_would_not_fit_is_refused_and_the_valve_still_fires(
    tmp_path, monkeypatch
):
    """Above the per-frame cap a merge is REFUSED; a refusal frees nothing, so the valve cuts.

    Two 600 KB reasoning frames fold to ~1.2 MB, past ``MERGE_CAP_BYTES`` (1
    MiB, mirrored from the runtime and kept well under the desktop relay's 8 MiB
    silent-drop ceiling). The refusal leaves both frames in place; the same
    pressure event then disconnects exactly as it would have without the fold.
    """
    assert module.MERGE_CAP_BYTES == 1 << 20, "mirrors the runtime's _MAX_LINE_BYTES"
    assert module.MERGE_CAP_BYTES < 8 * 1024 * 1024, "under the relay's MAX_FRAME_BYTES"
    monkeypatch.setattr(module, "REPLAY_BYTES", 1_500_000)
    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    async with pool.session(sid) as bridge:
        sub, stream = await _opened_sub(bridge)
        delta_a = "a" * 600_000
        delta_b = "b" * 600_000
        bridge.publish("event", {"type": "reasoning_delta", "message_id": "m-x", "delta": delta_a})
        bridge.publish("event", {"type": "reasoning_delta", "message_id": "m-x", "delta": delta_b})
        assert not sub.overflow, "two frames this size still fit the 1.5 MB budget"

        items = _peek_queue(sub)
        assert len(items) == 2
        escaped_a = len(json.dumps(delta_a).encode()) - 2
        assert items[1][1] + escaped_a > module.MERGE_CAP_BYTES, (
            "the candidate merge of these two frames WOULD exceed the cap, which is "
            "why the fold below must refuse it"
        )
        assert (
            bridge._compact_subscriber_queue(sub, items[1][1]) is False
        ), "the fold refuses and frees no room for another frame of that size"
        after = _peek_queue(sub)
        assert [frame["payload"]["delta"] for frame, _ in after] == [
            delta_a,
            delta_b,
        ], "a refused merge keeps both frames, unfolded, in order"

        bridge.publish(
            "event", {"type": "reasoning_delta", "message_id": "m-x", "delta": "c" * 600_000}
        )
        assert sub.overflow is True, "folding freed nothing, so the valve must fire"
        assert (await asyncio.wait_for(anext(stream), timeout=5))["type"] == "gap"
        with pytest.raises(StopAsyncIteration):
            await anext(stream)
        assert sub.id not in bridge.subscribers


@pytest.mark.asyncio
async def test_folding_preserves_the_cursor_and_replay_keeps_the_originals(tmp_path):
    """Cursor safety: merged heads ride the latest seq, and replay stays unfolded.

    A folded frame is delivered at the LATER seq — the client's receipt cursor
    advances past everything folded into it — and ``bridge.replay`` keeps the
    ORIGINAL frames, so a reconnect from a cursor below the fold is answered
    with the identical, unfolded fragments it would have received before the
    fold existed.
    """
    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    async with pool.session(sid) as bridge:
        sub, stream = await _opened_sub(bridge)
        fragments = [f"[{n}]" for n in range(300)]
        for fragment in fragments:
            bridge.publish(
                "event", {"type": "reasoning_delta", "message_id": "m-c", "delta": fragment}
            )
        assert not sub.overflow

        delivered = await _drain(sub, stream)
        assert [frame["seq"] for frame in delivered] == list(range(256, 301)), (
            "the fold at publish 257 merges fragments 1-256 under seq 256, and the "
            "tail keeps its own seqs: strictly increasing throughout"
        )
        assert delivered[0]["payload"]["delta"] == "".join(fragments[:256])
        assert sub.queued_bytes == 0
        await stream.aclose()

        retained = [(frame["seq"], frame["payload"]["delta"]) for frame, _ in bridge.replay]
        assert len(retained) == module.REPLAY_COUNT
        assert retained == [(seq, fragments[seq - 1]) for seq in range(45, 301)], (
            "replay holds the last 256 frames UNFOLDED: one fragment per frame, "
            "never a merged delta"
        )

        # A reconnect from below the fold is served the originals, frame by frame.
        sub2 = bridge.subscribe()
        stream2 = bridge.events(sub2, epoch=bridge.epoch, after_seq=44)
        assert (await asyncio.wait_for(anext(stream2), timeout=5))["type"] == "open"
        replayed = [
            await asyncio.wait_for(anext(stream2), timeout=5) for _ in range(module.REPLAY_COUNT)
        ]
        assert [(frame["seq"], frame["payload"]["delta"]) for frame in replayed] == retained
        assert (await asyncio.wait_for(anext(stream2), timeout=5))["type"] == "snapshot"
        await stream2.aclose()


@pytest.mark.asyncio
async def test_the_pre_open_eviction_path_is_untouched(tmp_path):
    """D1's boundary: before the handshake the queue EVICTS; the fold never runs.

    A foldable burst delivered pre-open must still fill the queue to
    ``REPLAY_COUNT`` by evicting one frame per arrival — every survivor an
    unmerged fragment. A fold reaching this window would collapse the run to a
    couple of frames and change what D1's eviction proof is about; the
    handshake then supersedes the survivors with the snapshot, as before.
    """
    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    async with pool.session(sid) as bridge:
        burst = module.REPLAY_COUNT + 100
        fragments = [f"[{n}]" for n in range(burst)]
        sub = bridge.subscribe()
        for fragment in fragments:
            bridge.publish(
                "event", {"type": "reasoning_delta", "message_id": "m-pre", "delta": fragment}
            )
        assert not sub.opened, "the handshake has not started"
        assert sub.queue.qsize() == module.REPLAY_COUNT, (
            "pre-open policy is eviction to the same bound, NOT folding: a fold "
            "would have left a couple of frames, not a full queue"
        )
        queued = _peek_queue(sub)
        assert [frame["payload"]["delta"] for frame, _ in queued] == fragments[100:], (
            "the oldest frames were evicted one per arrival and every survivor is "
            "an unmerged single fragment"
        )
        assert not sub.overflow

        cutoff = bridge.sequence
        stream = bridge.events(sub, epoch=bridge.epoch, after_seq=cutoff)
        assert (await asyncio.wait_for(anext(stream), timeout=5))["type"] == "open"
        snapshot = await asyncio.wait_for(anext(stream), timeout=5)
        assert snapshot["type"] == "snapshot" and snapshot["seq"] == cutoff
        assert not sub.overflow, "the handshake drains what the snapshot supersedes"
        assert sub.queue.empty()

        # STILL LIVE: the post-handshake loop answers a new frame.
        bridge.publish("event", {"after": "the handshake"})
        delivered = await asyncio.wait_for(anext(stream), timeout=5)
        assert delivered["payload"] == {"after": "the handshake"}
        await stream.aclose()


@pytest.mark.asyncio
async def test_the_incremental_merge_size_matches_a_full_dump_on_adversarial_deltas(tmp_path):
    """The fold's size arithmetic is EXACT, including for escaping-heavy fragments.

    The fold never re-dumps a merged frame while folding (it adds the incoming
    frame's size to the escaped bytes of the delta already held — exact because
    JSON escaping is per-character), but the sizes it stores must equal a full
    re-dump, because ``events`` subtracts them. Quotes, backslashes, control
    characters, emoji and U+2028 are where a wrong assumption would show.
    """
    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    async with pool.session(sid) as bridge:
        sub, stream = await _opened_sub(bridge)
        adversarial = [
            'quote " backslash \\ control \x01\x1f',
            "emoji 🚀 and U+2028 \u2028",
            "plain fragment",
        ]
        fragments = adversarial * 90  # 270 fragments: one fold, then a tail
        for fragment in fragments:
            bridge.publish(
                "event", {"type": "reasoning_delta", "message_id": "m-e", "delta": fragment}
            )
        assert not sub.overflow
        items = _peek_queue(sub)
        assert items, "the burst was published; the queue cannot be empty"
        assert sub.queued_bytes == sum(size for _, size in items)
        for frame, size in items:
            assert size == _exact_size(frame), (
                "a stored size that is not the frame's exact encoded length drifts the "
                "byte accounting every dequeue subtracts from"
            )
        assert "".join(frame["payload"]["delta"] for frame, _ in items) == "".join(fragments)
        await stream.aclose()


@pytest.mark.asyncio
async def test_a_supplement_progress_run_keeps_the_newest_beat(tmp_path) -> None:
    """§3.1 row 14 (round-1 review R8): ``supplement_progress`` keeps the NEWEST beat.

    A beat re-sends the job's current state -- memo §2.7: the family "is self-replacing
    by construction" -- and the durable copy is the journal row, so a viewer stalled
    behind a run of beats collapses them to the latest one. 512 beats arrive as two:
    the one the fold kept (512) and the tail (512).
    """
    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    async with pool.session(sid) as bridge:
        sub, stream = await _opened_sub(bridge)
        count = 512
        for n in range(count):
            bridge.publish(
                "event",
                {
                    "type": "supplement_progress",
                    "anchor": "a1b2c3d4e5f60718293a4b5c6d7e8f90",
                    "job": "3f9c1a7e5b20",
                    "version": n + 1,
                    "state": "running",
                    "stage": "generating",
                    "elapsed_s": float(n),
                },
            )
        assert not sub.overflow

        delivered = await _drain(sub, stream)
        assert len(delivered) == 2, "the fold collapsed 511 beats into their newest, then 511"
        versions = [frame["payload"]["version"] for frame in delivered]
        assert versions == [511, 512], "keep-newest keeps the latest beat, not every beat"
        await stream.aclose()
