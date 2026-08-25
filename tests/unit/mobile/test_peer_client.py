"""The `lop send` sender client against a REAL in-process registrant.

The sender dials as a daemon-class connection, so it receives an unsolicited
welcome/projection push before its ack. These assert it returns the ack detail,
skips the intervening projection frames, and raises on an error frame — the
three behaviours the design's §4.1 calls out.
"""

from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest

from local_operator.mobile import registry
from local_operator.mobile.peer_client import send_peer_message
from local_operator.mobile.registrant import Registrant
from local_operator.mobile.types import TranscriptEntry
from tests.unit.mobile.test_registrant import FakeHandle, NoPeerHandle, _wait_record


@pytest.mark.asyncio
async def test_send_returns_ack_detail_and_passes_args() -> None:
    handle = FakeHandle()
    registrant = Registrant(handle, kind="tui")
    registrant.start()
    try:
        record = await _wait_record()
        detail = await send_peer_message(
            record,
            text="gates are green",
            mode="mailbox",
            wake=False,
            sender={"pid": 999, "conversation_name": "sender"},
        )
        # The FakeHandle returns the mailbox detail string; the sender surfaces
        # it verbatim after skipping the welcome/projection push.
        assert "mailbox" in detail
        name, args, kwargs = handle.calls[-1]
        assert name == "receive_peer_message"
        assert args == ("gates are green",)
        sender = cast("dict[str, Any]", kwargs["sender"])
        assert sender["pid"] == 999
    finally:
        registrant.close()


@pytest.mark.asyncio
async def test_send_survives_an_oversized_welcome_projection() -> None:
    """U1 regression: a daemon-class dial receives an unsolicited full-projection
    ``welcome`` as its first frame, and a busy target's transcript can make that
    single JSON line exceed any fixed readline limit. The sender must skip the
    oversized welcome and still return its ack cleanly, never raise
    ``ValueError``/``LimitOverrunError`` (which crashed the CLI while the message
    was already delivered).
    """
    handle = FakeHandle()
    # Seed the projection with a transcript tail whose serialized welcome line
    # exceeds 1 MiB — the old readline limit — so this test would have crashed
    # with the exact ``ValueError: Separator is not found, and chunk exceed the
    # limit`` U1 reported. 80 rows is the projection's transcript cap
    # (PROJECTION_TRANSCRIPT_LIMIT), so 20 KB per row lands ~1.6 MB on the wire.
    handle._projection.transcript = [
        TranscriptEntry(id=f"row-{i}", kind="assistant", text="x" * 20000) for i in range(80)
    ]
    registrant = Registrant(handle, kind="tui")
    registrant.start()
    try:
        record = await _wait_record()
        detail = await send_peer_message(
            record,
            text="gates are green",
            mode="mailbox",
            wake=False,
            sender={"pid": 999, "conversation_name": "sender"},
        )
        # The ack came through despite the >1 MiB welcome preceding it.
        assert "mailbox" in detail
        name, args, kwargs = handle.calls[-1]
        assert name == "receive_peer_message"
        assert args == ("gates are green",)
    finally:
        registrant.close()


@pytest.mark.asyncio
async def test_send_raises_on_error_frame() -> None:
    # A handle without the capability makes the registrant answer with an error
    # frame; the sender must raise RuntimeError, not hang or swallow it.
    handle = NoPeerHandle()
    registrant = Registrant(handle, kind="tui")
    registrant.start()
    try:
        record = await _wait_record()
        with pytest.raises(RuntimeError, match="cannot receive peer messages"):
            await send_peer_message(record, text="hi", mode="mailbox", wake=False, sender={})
    finally:
        registrant.close()


@pytest.mark.asyncio
async def test_send_times_out_on_a_silent_socket() -> None:
    # A raw listener that accepts, reads, but never acks: the sender's deadline
    # must fire rather than block forever.
    async def _handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await reader.readline()  # auth frame
        await reader.readline()  # the peer_message frame
        # Deliberately never reply.
        await asyncio.sleep(1.0)

    server = await asyncio.start_server(_handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    record = registry.SessionRecord(
        pid=1,
        kind="tui",
        session_id="s",
        conversation_name="silent",
        cwd="/tmp",
        model_label="test/model",
        control_port=port,
        control_key="k",
    )
    async with server:
        with pytest.raises(asyncio.TimeoutError):
            await send_peer_message(
                record, text="hi", mode="mailbox", wake=False, sender={}, deadline_s=0.2
            )
