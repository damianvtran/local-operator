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
