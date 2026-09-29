"""The input-metadata wire strip: a new owner stays attachable to old viewers.

The hazard (review round 1, B1): ``input_mode``/``input_path`` serialize on
EVERY message — the nulls included — and a pre-carriage viewer's ``Message`` is
``extra="forbid"``, so the attach sync's first page, ``history_page`` on the
first scroll, ``frontend_sync`` on the first refresh, and the first message
event all hard-fail for the routine mixed-build pairing (a repo-venv owner
beside an older viewer build) unless the owner strips the keys on the wire.
This file holds the strip shut the way ``test_display_history_audit_capability``
holds the audit strip shut: by SPEAKING THE WIRE at a REAL runtime and
verifying what arrives the way the old viewer would. Each strip cell asserts
its NEGOTIATED twin beside it — a viewer that declares ``input_mode`` must keep
the keys, or the strip becomes a silent drop for the builds that understand
them.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import Message
from local_operator.session.history_window import (
    INPUT_WIRE_MESSAGE_FIELDS,
    DisplayHistoryWindow,
    strip_input_metadata,
)
from local_operator.session.runtime.server import (
    RuntimeServer,
    _strip_input_from_relay_frame,
)
from local_operator.session.runtime.serving import ServingSessionHandle
from tests.e2e.harness import ScriptedStream, build_session, seed_transcript, text_turn


async def _until(predicate, timeout_s: float = 10.0) -> None:
    async with asyncio.timeout(timeout_s):
        while not predicate():
            await asyncio.sleep(0.001)


async def _server(tmp_path: Path, rows: list[Message]) -> RuntimeServer:
    directory = tmp_path / "sessions" / "input-wire"
    await seed_transcript(directory, rows)
    session = build_session(directory, ScriptedStream([text_turn("unused")]), cwd=tmp_path)
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(tmp_path))
    server = RuntimeServer(handle, kind="daemon")
    await server.start_in_process()
    return server


async def _dial(
    server: RuntimeServer, *, announce_input: bool, events: bool = False
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter, dict[str, Any]]:
    """Attach over a real socket; return the connection and the pushed sync data.

    Speaks the wire directly, exactly as ``test_display_history_audit_capability``
    does: the point is to BE a viewer build that predates the carriage, which
    the current client can no longer be talked into being.
    """
    record = server._record
    reader, writer = await asyncio.open_connection("127.0.0.1", record.control_port)
    auth = {
        "key": record.control_key,
        "client": "attach",
        "locality": "local",
        "frontend_state": True,
        "display_window": True,
    }
    if announce_input:
        auth["input_mode"] = True
    if events:
        auth["events"] = True
    writer.write(json.dumps(auth).encode() + b"\n")
    await writer.drain()
    frames: list[dict[str, Any]] = []
    while True:
        line = await asyncio.wait_for(reader.readline(), timeout=10)
        if not line:
            writer.close()
            raise AssertionError(f"connection closed before the push frame: {frames}")
        frame = json.loads(line)
        frames.append(frame)
        if frame.get("op") == "frontend_sync":
            return reader, writer, frame["data"]


async def _rpc(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter, frame: dict[str, Any], *, req: int
) -> Any:
    writer.write(json.dumps({**frame, "req": req}).encode() + b"\n")
    await writer.drain()
    while True:
        line = await asyncio.wait_for(reader.readline(), timeout=10)
        if not line:
            raise AssertionError("connection closed before the result frame")
        reply = json.loads(line)
        if reply.get("op") == "result" and reply.get("req") == req:
            return reply["data"]


def _text_of(message: dict[str, Any]) -> str:
    return " ".join(
        block.get("text") or "" for block in message.get("content", []) if isinstance(block, dict)
    )


def _assert_no_input_keys(message: dict[str, Any]) -> None:
    # The absence assertions ARE the discriminator: a pre-carriage viewer's
    # ``extra="forbid"`` ``Message`` rejects exactly on these keys, so "no
    # key" is the wire property that keeps its ``model_validate`` green. The
    # re-validation below then proves the REST of the dict is a well-formed
    # message, so a cell cannot pass on a payload that fails for another
    # reason.
    for name in INPUT_WIRE_MESSAGE_FIELDS:
        assert name not in message, f"{name} leaked to a viewer that cannot accept it"
    Message.model_validate(message)


@pytest.mark.asyncio
@pytest.mark.parametrize("announce_input", [False, True])
async def test_the_attach_page_respects_the_negotiation(
    tmp_path: Path, announce_input: bool
) -> None:
    rows = [
        Message.user("dictated row", input_mode="dictated", input_path="sidecar_transcription"),
        Message.user("legacy row"),
    ]
    server = await _server(tmp_path, rows)
    reader, writer, payload = await _dial(server, announce_input=announce_input)
    try:
        window = payload["display_history"]
        assert window is not None
        dictated = next(m for m in window["messages"] if _text_of(m) == "dictated row")
        if announce_input:
            assert dictated["input_mode"] == "dictated"
            assert dictated["input_path"] == "sidecar_transcription"
            DisplayHistoryWindow.model_validate(window)
        else:
            for message in window["messages"]:
                _assert_no_input_keys(message)
    finally:
        writer.close()
        server.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("announce_input", [False, True])
async def test_the_frontend_sync_rpc_respects_the_negotiation(
    tmp_path: Path, announce_input: bool
) -> None:
    """The steady-state refresh path: ``_refresh_display_history`` calls it."""
    server = await _server(tmp_path, [Message.user("row", input_mode="dictated")])
    reader, writer, _ = await _dial(server, announce_input=announce_input)
    try:
        data = await _rpc(reader, writer, {"op": "frontend_sync"}, req=1)
    finally:
        writer.close()
        server.close()
    window = data["display_history"]
    assert window is not None
    message = window["messages"][0]
    if announce_input:
        assert message["input_mode"] == "dictated"
    else:
        _assert_no_input_keys(message)


@pytest.mark.asyncio
@pytest.mark.parametrize("announce_input", [False, True])
async def test_the_history_page_rpc_respects_the_negotiation(
    tmp_path: Path, announce_input: bool
) -> None:
    rows = [Message.user(f"row {index}", input_mode="typed") for index in range(250)]
    server = await _server(tmp_path, rows)
    reader, writer, payload = await _dial(server, announce_input=announce_input)
    try:
        token = payload["display_history"]["before_token"]
        assert token, "the 250-row fixture must page"
        page = await _rpc(
            reader, writer, {"op": "history_page", "before": token, "anchor": ""}, req=2
        )
    finally:
        writer.close()
        server.close()
    assert page["messages"], "expected an earlier page carrying messages"
    for message in page["messages"]:
        if announce_input:
            assert message["input_mode"] == "typed"
        else:
            _assert_no_input_keys(message)


@pytest.mark.asyncio
@pytest.mark.parametrize("announce_input", [False, True])
async def test_the_relayed_message_event_respects_the_negotiation(
    tmp_path: Path, announce_input: bool
) -> None:
    server = await _server(tmp_path, [Message.user("row")])
    reader, writer, _ = await _dial(server, announce_input=announce_input, events=True)
    try:
        await _until(lambda: any(conn.events_ready for conn in server._clients.values()))
        live = Message.user("live row", input_mode="dictated")
        server._relay_on_loop({"type": "message_end", "message": live.model_dump(mode="json")})
        while True:
            line = await asyncio.wait_for(reader.readline(), timeout=10)
            if not line:
                raise AssertionError("connection closed before the event frame")
            frame = json.loads(line)
            if frame.get("op") == "event" and frame["data"].get("type") == "message_end":
                break
    finally:
        writer.close()
        server.close()
    message = frame["data"]["message"]
    if announce_input:
        assert message["input_mode"] == "dictated"
    else:
        _assert_no_input_keys(message)


def test_the_sync_seed_carrier_is_walked_too() -> None:
    """``snapshot.live_events`` serializes messages the same model does."""
    sync_payload = {
        "display_history": {"messages": [Message.user("row").model_dump(mode="json")]},
        "snapshot": {
            "live_events": [
                {"type": "message_end", "message": Message.user("live").model_dump(mode="json")}
            ]
        },
    }
    strip_input_metadata(sync_payload, input_capable=False)
    assert "input_mode" not in sync_payload["display_history"]["messages"][0]
    assert "input_mode" not in sync_payload["snapshot"]["live_events"][0]["message"]
    kept = {"messages": [Message.user("row").model_dump(mode="json")]}
    strip_input_metadata(kept, input_capable=True)
    assert "input_mode" in kept["messages"][0]


def test_the_relay_frame_strip_copies_only_what_it_touches() -> None:
    message = Message.user("live", input_mode="dictated").model_dump(mode="json")
    frame = {"op": "event", "data": {"type": "message_end", "message": message}}
    stripped = _strip_input_from_relay_frame(frame)
    assert stripped is not frame
    assert "input_mode" not in stripped["data"]["message"]
    # The ORIGINAL is untouched: this frame is shared across recipients.
    assert frame["data"]["message"]["input_mode"] == "dictated"
    # Nothing to strip -> the same object back (the hot path's fast exit).
    plain = {"op": "event", "data": {"type": "agent_end"}}
    assert _strip_input_from_relay_frame(plain) is plain
    # A frontend update carries the same message shape inside changes.live_events.
    update = {
        "op": "frontend_update",
        "data": {"changes": {"live_events": [{"type": "message_end", "message": message}]}},
    }
    fresh = _strip_input_from_relay_frame(update)
    assert fresh is not update
    assert "input_mode" not in fresh["data"]["changes"]["live_events"][0]["message"]
    assert "input_mode" in update["data"]["changes"]["live_events"][0]["message"]
