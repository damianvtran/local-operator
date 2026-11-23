"""Cross-version safety for the audit fields on a display page.

The hazard this file exists for is a HARD attach break, not a degradation.
:class:`DisplayHistoryWindow` sets ``extra="forbid"``, so a viewer built before
the audit fields existed RAISES when it validates a payload carrying them, and
the viewer turns that into a failed attach — no history, no session. Mixed
builds against one sessions directory are routine on a development machine: the
global runtime is a separate uv-tool install updated on its own schedule, so a
repo checkout's venv and the global binary regularly attach to each other.

``display-history-window-v1`` cannot express this. It is advertised on the mere
PRESENCE of ``history_page`` and carries no version, so a second capability
string is the only thing that can distinguish "pages history" from "pages
pre-compaction history too".
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import Message, TextContent
from local_operator.session.history_window import (
    AUDIT_WIRE_FIELDS,
    DISPLAY_HISTORY_AUDIT_CAPABILITY,
    DisplayHistoryWindow,
    display_window,
)
from local_operator.session.runtime import server as runtime_server
from local_operator.session.runtime.server import RuntimeServer
from local_operator.session.runtime.serving import ServingSessionHandle
from local_operator.session.transcript import Transcript
from tests.e2e.harness import ScriptedStream, build_session, seed_transcript, text_turn


async def _server(tmp_path: Path, rows: list[Message]):
    directory = tmp_path / "sessions" / "audit-capability"
    await seed_transcript(directory, rows)
    session = build_session(directory, ScriptedStream([text_turn("unused")]), cwd=tmp_path)
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(tmp_path))
    server = RuntimeServer(handle, kind="daemon")
    await server.start_in_process()
    return server


async def _attach(server: RuntimeServer, *, announce_audit: bool) -> dict[str, Any]:
    """Attach as a viewer that does or does not know the audit capability.

    Speaks the wire protocol directly rather than through ``AttachClient``,
    because the point is to reproduce a viewer whose BUILD predates the field —
    which the current client can no longer be talked into being.
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
    if announce_audit:
        auth["display_history_audit"] = True
    writer.write(json.dumps(auth).encode() + b"\n")
    await writer.drain()
    frames = []
    try:
        while True:
            line = await asyncio.wait_for(reader.readline(), timeout=10)
            if not line:
                break
            frame = json.loads(line)
            frames.append(frame)
            if frame.get("op") == "frontend_sync":
                return frame["data"]
    finally:
        writer.close()
    raise AssertionError(f"no frontend_sync frame arrived: {frames}")


@pytest.mark.asyncio
async def test_owner_advertises_the_audit_capability_beside_the_window_one(tmp_path) -> None:
    server = await _server(tmp_path, [Message.user("row")])
    try:
        assert DISPLAY_HISTORY_AUDIT_CAPABILITY in server._record.capabilities
        assert "display-history-window-v1" in server._record.capabilities
    finally:
        server.close()


@pytest.mark.asyncio
async def test_a_viewer_that_did_not_negotiate_receives_no_audit_keys(tmp_path) -> None:
    """The §6 break, asserted end to end.

    An old viewer's page model forbids extra keys, so the payload it receives
    must not contain them — and the proof is that the page model validates it.
    """
    server = await _server(tmp_path, [Message.user(f"row {i}") for i in range(5)])
    try:
        payload = await _attach(server, announce_audit=False)
        window = payload["display_history"]
        assert window is not None
        for name in AUDIT_WIRE_FIELDS:
            assert name not in window, f"{name} leaked to a viewer that cannot accept it"

        # An old viewer would validate exactly this dict under extra="forbid".
        # Rebuilt from the model with the new fields removed, so the assertion
        # is about the WIRE payload rather than about a hand-written schema.
        legacy_fields = {
            name: field
            for name, field in DisplayHistoryWindow.model_fields.items()
            if name not in AUDIT_WIRE_FIELDS
        }
        assert set(window) <= set(legacy_fields)
    finally:
        server.close()


@pytest.mark.asyncio
async def test_a_viewer_that_negotiated_receives_the_audit_keys(tmp_path) -> None:
    server = await _server(tmp_path, [Message.user(f"row {i}") for i in range(5)])
    try:
        payload = await _attach(server, announce_audit=True)
        window = payload["display_history"]
        for name in AUDIT_WIRE_FIELDS:
            assert name in window
        DisplayHistoryWindow.model_validate(window)
    finally:
        server.close()


async def _rpc_frontend_sync(server: RuntimeServer, *, announce_audit: bool) -> dict[str, Any]:
    """Attach, drain the push frames, then CALL the frontend_sync RPC op.

    Deliberately NOT the pushed ``frontend_sync`` FRAME that ``_attach`` above
    reads. The frame and the op are different code paths, and covering only the
    frame is why the unstripped op shipped green through round 1 (QA Q1 /
    review R1).
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
    if announce_audit:
        auth["display_history_audit"] = True
    writer.write(json.dumps(auth).encode() + b"\n")
    await writer.drain()

    # Wait for the pushed sync so the connection is fully established, exactly
    # as a real viewer does before it ever issues a refresh.
    while True:
        line = await asyncio.wait_for(reader.readline(), timeout=10)
        if not line:
            raise AssertionError("connection closed before the push frame")
        if json.loads(line).get("op") == "frontend_sync":
            break

    # The RPC under test.
    writer.write(json.dumps({"op": "frontend_sync", "req": 1}).encode() + b"\n")
    await writer.drain()
    try:
        while True:
            line = await asyncio.wait_for(reader.readline(), timeout=10)
            if not line:
                raise AssertionError("connection closed before the result frame")
            frame = json.loads(line)
            if frame.get("op") == "result" and frame.get("req") == 1:
                return frame["data"]
    finally:
        writer.close()


@pytest.mark.asyncio
async def test_frontend_sync_rpc_strips_the_keys_for_an_old_viewer(tmp_path) -> None:
    """An old viewer's history REFRESH must not receive fields it forbids.

    The steady-state path, not a race: ``_refresh_display_history`` calls this
    op whenever the frontend's ``history_generation`` moves, so an old viewer
    attached to a new owner hits it as soon as the owner appends a row. The
    fields serialize on an UNCOMPACTED page too, so a single-row fixture is
    enough to reproduce it.
    """
    server = await _server(tmp_path, [Message.user("row")])
    try:
        data = await _rpc_frontend_sync(server, announce_audit=False)
    finally:
        server.close()
    window = data.get("display_history")
    assert isinstance(window, dict), "the RPC returned no display page"
    leaked = [field for field in AUDIT_WIRE_FIELDS if field in window]
    assert not leaked, (
        f"the frontend_sync RPC leaked {leaked} to a viewer that did not "
        f"negotiate display-history-audit-v1; its DisplayHistoryWindow sets "
        f"extra='forbid', so its refresh raises"
    )


@pytest.mark.asyncio
async def test_frontend_sync_rpc_keeps_the_keys_for_a_new_viewer(tmp_path) -> None:
    """CANARY (known-positive): the same probe must SEE the fields when the
    viewer did negotiate. Without this, a pass above could mean the fields are
    simply never emitted on this route and the test would prove nothing."""
    server = await _server(tmp_path, [Message.user("row")])
    try:
        data = await _rpc_frontend_sync(server, announce_audit=True)
    finally:
        server.close()
    window = data.get("display_history")
    assert isinstance(window, dict), "the RPC returned no display page"
    present = [field for field in AUDIT_WIRE_FIELDS if field in window]
    assert present == list(AUDIT_WIRE_FIELDS), (
        f"probe is dead: a negotiating viewer saw {present}, so the "
        f"stripped-case assertion above cannot distinguish a fix from a "
        f"route that never carries these fields at all"
    )


@pytest.mark.asyncio
async def test_display_history_routes_negotiate_inline_ui_without_changing_prose(tmp_path) -> None:
    """Attach, refresh and scroll must agree on the negotiated page shape."""
    widget = '```lop-ui\n{"type":"bar","title":"Counts","data":[["One",1]]}\n```'
    source = f"Before 🌱.\n\n{widget}\n\nAfter."
    prose = "Before 🌱.\n\n\n\nAfter."
    legacy_exclusions = {"input_mode", "input_path"}
    rows = [Message.user(f"row {index}") for index in range(300)]
    for index in (100, 298):
        rows[index] = Message.assistant(source)
        rows[index]._inline_ui_source_version = 1
        rows[index + 1] = Message.assistant(source)
    server = await _server(tmp_path, rows)
    try:
        record = server._record
        assert "inline-ui-v1" in record.capabilities
        pages = []
        pushed = []
        refreshed = []
        # The strict bool from hello, not a per-request field, grants sidecars.
        for hello in ({}, {"inline_ui": True}, {"inline_ui": "true"}):
            reader, writer = await asyncio.open_connection("127.0.0.1", record.control_port)
            try:
                writer.write(
                    json.dumps(
                        {
                            "key": record.control_key,
                            "client": "attach",
                            "locality": "local",
                            "frontend_state": True,
                            "display_window": True,
                            **hello,
                        }
                    ).encode()
                    + b"\n"
                )
                await writer.drain()
                while True:
                    line = await asyncio.wait_for(reader.readline(), timeout=10)
                    frame = json.loads(line)
                    if frame.get("op") == "frontend_sync":
                        break
                head = frame["data"]["display_history"]
                expected = rows[298].model_copy(update={"content": [TextContent(text=prose)]})
                assert head["messages"][-2] == expected.model_dump(
                    mode="json", exclude=legacy_exclusions
                )
                assert head["messages"][-1] == rows[299].model_dump(
                    mode="json", exclude=legacy_exclusions
                )
                pushed.append(head)
                writer.write(
                    json.dumps({"op": "frontend_sync", "req": "sync", "inline_ui": True}).encode()
                    + b"\n"
                )
                await writer.drain()
                while True:
                    line = await asyncio.wait_for(reader.readline(), timeout=10)
                    frame = json.loads(line)
                    if frame.get("req") == "sync":
                        assert frame.get("op") == "result", frame
                        refreshed.append(frame["data"]["display_history"])
                        break
                token = head["before_token"]
                assert token, "the fixture must be long enough to page"
                writer.write(
                    json.dumps(
                        {"op": "history_page", "req": "r1", "before": token, "inline_ui": True}
                    ).encode()
                    + b"\n"
                )
                await writer.drain()
                while True:
                    line = await asyncio.wait_for(reader.readline(), timeout=10)
                    frame = json.loads(line)
                    if frame.get("req") == "r1":
                        assert frame.get("op") == "result", frame
                        pages.append(frame["data"])
                        break
            finally:
                writer.close()
                await writer.wait_closed()

        for route, (legacy, modern, malformed), marked_index in (
            ("frontend_sync push", pushed, 298),
            ("frontend_sync RPC", refreshed, 298),
            ("history_page RPC", pages, 100),
        ):
            for name in (*AUDIT_WIRE_FIELDS, "inline_ui"):
                assert name not in legacy
            assert malformed == legacy
            assert "inline_ui" in modern, f"{route}: negotiated widget sidecar missing"
            assert modern.pop("inline_ui") == {
                "version": 1,
                "messages": {
                    rows[marked_index].id: {
                        "version": 1,
                        "text": prose,
                        "widgets": [
                            {
                                "offset": len("Before 🌱.\n\n".encode("utf-16-le")) // 2,
                                "payload": {"type": "bar", "title": "Counts", "data": [["One", 1]]},
                            }
                        ],
                    }
                },
            }
            assert modern == legacy
            print(f"{route}: exact matching page/cursors; widget only for boolean-true hello")

        carried = {message["id"]: message for message in pages[0]["messages"]}
        expected = rows[100].model_copy(update={"content": [TextContent(text=prose)]})
        assert carried[rows[100].id] == expected.model_dump(mode="json", exclude=legacy_exclusions)
        assert carried[rows[101].id] == rows[101].model_dump(mode="json", exclude=legacy_exclusions)
    finally:
        server.close()


@pytest.mark.parametrize("envelope", [{"op": "frontend_sync"}, {"op": "result", "req": "sync"}])
def test_sync_history_sidecar_yields_before_the_page(envelope, monkeypatch) -> None:
    window = DisplayHistoryWindow(
        status="ok",
        conversation_id="session",
        owner_epoch="epoch",
        history_generation=0,
        through_id=None,
    )
    window._sync_inline_ui = {"version": 1, "messages": {"message": {"widgets": ["🌱"]}}}
    page = window.model_dump(mode="json")
    assert "inline_ui" not in page
    frame = {**envelope, "data": {"display_history": page.copy()}}
    rich = {**envelope, "data": {"display_history": {**page, "inline_ui": window._sync_inline_ui}}}
    cap = len(json.dumps(rich).encode()) + 1

    monkeypatch.setattr(runtime_server, "_MAX_LINE_BYTES", cap)
    runtime_server._fit_sync_history_inline_ui(frame, window)
    assert frame == rich

    monkeypatch.setattr(runtime_server, "_MAX_LINE_BYTES", cap - 1)
    runtime_server._fit_sync_history_inline_ui(frame, window)
    assert frame["data"]["display_history"] == page


@pytest.mark.asyncio
async def test_a_new_viewer_against_an_owner_without_the_capability_is_safe(tmp_path) -> None:
    """The reverse direction, which is safe by construction and worth pinning.

    The fields are absent from the payload, Pydantic supplies ``False``, and the
    viewer therefore never asks for an audit page it cannot be served.
    """
    transcript = Transcript(tmp_path / "s")
    await transcript.append_messages([Message.user("row")])
    page = display_window(
        transcript,
        conversation_id="c",
        owner_epoch="e",
        through_id=transcript.entries()[-1].id,
    )
    payload = page.model_dump(mode="json")
    for name in AUDIT_WIRE_FIELDS:
        payload.pop(name)
    restored = DisplayHistoryWindow.model_validate(payload)
    assert restored.audit is False
    assert restored.audit_available is False
