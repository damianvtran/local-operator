"""Source-order declaration crosses the real dial, not only its AST allowlist."""

import asyncio
import json

import pytest

from local_operator.mobile.attach_client import AttachClient
from local_operator.network.dial import OwnerUnreachable, dial_owner
from local_operator.session.runtime.types import JOURNAL_ORDER_CAPABILITY, SessionRecord


@pytest.mark.asyncio
@pytest.mark.parametrize("capable", [False, True])
async def test_actual_attach_declares_only_capable_owner_and_dial_forwards(capable, tmp_path):
    frames = []
    finished = asyncio.Event()
    session_id = "synthetic-source-dial"

    async def owner(reader, writer):
        try:
            frame = json.loads(await reader.readline())
            frames.append(frame)
            writer.write(
                json.dumps({"op": "welcome", "data": {"session_id": session_id, "pid": 0}}).encode()
                + b"\n"
            )
            await writer.drain()
            await reader.read()
        finally:
            writer.close()
            await writer.wait_closed()
            finished.set()

    server = await asyncio.start_server(owner, "127.0.0.1", 0)
    record = SessionRecord(
        pid=1,
        kind="tui",
        session_id=session_id,
        conversation_name="",
        cwd=str(tmp_path),
        model_label="",
        control_port=server.sockets[0].getsockname()[1],
        control_key="synthetic-record-key",
        protocol=4,
        capabilities=[JOURNAL_ORDER_CAPABILITY] if capable else [],
    )
    client = AttachClient(lambda projection: None, lambda reason: None, events=True)
    try:
        # Actual production handshake construction proves old owner omission.
        await client.connect(record, session_id)
        assert frames[0].get("journal_order") is (True if capable else None)
        client.close()
        await asyncio.wait_for(finished.wait(), timeout=2)
        finished.clear()
        auth = dict(frames[0])
        # A viewer cannot replace the dialer's key/kind/locality proof.
        auth.update(key="untrusted-viewer-key", client="daemon", locality="local")
        dial, _ = await asyncio.to_thread(
            dial_owner, tmp_path, session_id, auth=auth, _record=record
        )
        try:
            assert frames[1].get("journal_order") is (True if capable else None)
            assert frames[1]["key"] == record.control_key
            assert frames[1]["client"] == "attach"
            assert frames[1]["locality"] == "remote"
        finally:
            dial.close()
        await asyncio.wait_for(finished.wait(), timeout=2)
    finally:
        client.close()
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_source_dial_cannot_bind_another_session(tmp_path):
    done = asyncio.Event()

    async def wrong_owner(reader, writer):
        try:
            await reader.readline()
            writer.write(b'{"op":"welcome","data":{"session_id":"other-source-session"}}\n')
            await writer.drain()
            await reader.read()
        finally:
            writer.close()
            await writer.wait_closed()
            done.set()

    server = await asyncio.start_server(wrong_owner, "127.0.0.1", 0)
    record = SessionRecord(
        pid=1,
        kind="tui",
        session_id="asked-source-session",
        conversation_name="",
        cwd=str(tmp_path),
        model_label="",
        control_port=server.sockets[0].getsockname()[1],
        control_key="synthetic-record-key",
        protocol=4,
    )
    try:
        with pytest.raises(OwnerUnreachable, match="other-source-session"):
            await asyncio.to_thread(
                dial_owner,
                tmp_path,
                record.session_id,
                auth={"journal_order": True},
                _record=record,
            )
        await asyncio.wait_for(done.wait(), timeout=2)
    finally:
        server.close()
        await server.wait_closed()
