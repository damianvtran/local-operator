"""The owner corrects already-streamed rows after durable write, without tokens."""

import asyncio
from types import SimpleNamespace

import pytest

from local_operator.harness.types import Message
from local_operator.session import transcript_index as ti
from local_operator.session.runtime.serving import ServingSessionHandle
from local_operator.session.transcript import Transcript


@pytest.mark.asyncio
async def test_quiet_durable_send_gets_late_index_annotation(tmp_path):
    ti._reset_for_tests()
    transcript = Transcript(tmp_path / "sessions" / "order-ready-fixture")
    handle = object.__new__(ServingSessionHandle)
    handle._session = SimpleNamespace(
        transcript=transcript, subscribe=lambda callback: lambda: None
    )
    ready = asyncio.Event()
    frames = []

    def received(frame):
        frames.append(frame)
        if frame.get("type") == "journal_order_ready":
            ready.set()

    unsubscribe = handle.subscribe_events(received)
    try:
        assert handle.journal_order_supported
        send = await transcript.append_message(Message.user("synthetic quiet send"))
        await asyncio.wait_for(ready.wait(), timeout=10)
        rows = [row for frame in frames for row in frame.get("rows", [])]
        assert rows == [
            {
                "id": send.id,
                "type": "message",
                "role": "user",
                "tool_call_id": None,
                "anchor": None,
                "journal_seq": 0,
            }
        ]
        domain = frames[-1]["journal_generation"]
        assert domain
        cached = ti.read_index(tmp_path, "order-ready-fixture")
        assert cached is not None and cached.journal_generation == domain
        assert cached.journal_rows[send.id] == 0
        # No further event/page/token was needed; custom rows use the same source.
        ready.clear()
        custom = await transcript.append_custom(
            "session_model_switch", {"text": "synthetic switch"}
        )
        await asyncio.wait_for(ready.wait(), timeout=10)
        assert frames[-1]["journal_generation"] == domain
        assert frames[-1]["rows"][0]["id"] == custom.id
        assert frames[-1]["rows"][0]["journal_seq"] == 1
    finally:
        unsubscribe()
        ti._reset_for_tests()


def test_read_only_handle_does_not_advertise_order_commit_support():
    handle = object.__new__(ServingSessionHandle)
    handle._session = SimpleNamespace(subscribe=lambda callback: lambda: None)
    assert not handle.journal_order_supported
    unsubscribe = handle.subscribe_events(lambda frame: None)
    unsubscribe()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["real-compaction", "missing-map"])
async def test_rejected_commit_proof_requests_one_authoritative_reconcile(
    tmp_path, monkeypatch, failure
):
    ti._reset_for_tests()
    session_id = "order-recovery-fixture"
    transcript = Transcript(tmp_path / "sessions" / session_id)
    await transcript.append_custom("subagent_roster", {"revision": 1})
    await transcript.append_custom("subagent_roster", {"revision": 2})
    initial, _ = ti.start_refresh(tmp_path, session_id)
    await initial
    entered, release, received = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original_refresh = ti.start_refresh
    appended = []

    async def build_after_gate():
        await release.wait()
        actual, _ = original_refresh(tmp_path, session_id)
        await actual
        if failure == "missing-map":
            cached = ti.fresh_resident(tmp_path, session_id)
            assert cached is not None
            # A verified reader lacking an admitted identity is also unknown;
            # recovery must not manufacture a coordinate or silently drop it.
            for entry in appended:
                cached.journal_rows.pop(entry.id, None)

    gated = asyncio.create_task(build_after_gate())

    def refresh(root, sid):
        entered.set()
        return gated, False

    monkeypatch.setattr(ti, "start_refresh", refresh)
    handle = object.__new__(ServingSessionHandle)
    handle._session = SimpleNamespace(
        transcript=transcript, subscribe=lambda callback: lambda: None
    )
    frames = []

    def on_event(frame):
        frames.append(frame)
        received.set()

    unsubscribe = handle.subscribe_events(on_event)
    try:
        old_inode = transcript.path.stat().st_ino
        appended.append(
            await transcript.append_message(Message.user("quiet original ID survives compact"))
        )
        await asyncio.wait_for(entered.wait(), timeout=2)
        if failure == "real-compaction":
            reclaimed = await transcript.compact_file(min_reclaim_bytes=0)
            assert reclaimed > 0 and transcript.path.stat().st_ino != old_inode
        else:
            appended.append(
                await transcript.append_custom(
                    "session_model_switch", {"text": "second missing coordinate"}
                )
            )
        release.set()
        await asyncio.wait_for(received.wait(), timeout=2)
        await gated
        await asyncio.sleep(0)
        assert frames == [{"type": "journal_order_ready", "reconcile": True}]
        if failure == "real-compaction":
            cached = ti.fresh_resident(tmp_path, session_id)
            assert cached is not None and appended[0].id in cached.journal_rows
            # The original ID survives in the REAL rewritten journal/index,
            # but its old writer inode cannot be labeled with the new domain.
            assert cached.sig["inode"] != old_inode
    finally:
        release.set()
        unsubscribe()
        await gated
        ti._reset_for_tests()
