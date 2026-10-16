"""Three real attach/RPC routes preserve strict old-page compatibility."""

from __future__ import annotations

import asyncio
import copy
import json
from typing import Any

import pytest
from pydantic import ConfigDict, create_model

from local_operator.harness.peer_cycle_display import persist_record
from local_operator.session.history_window import (
    DISPLAY_HISTORY_PEER_CYCLE_CAPABILITY,
    DisplayHistoryWindow,
    strip_peer_cycle_fields,
)
from local_operator.session.runtime.server import RuntimeServer
from local_operator.session.runtime.serving import ServingSessionHandle
from local_operator.session.transcript import Transcript
from tests.e2e.harness import ScriptedStream, build_session, text_turn
from tests.unit.session.test_peer_cycle_display_archive import make_cycle


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["attach", "history_page", "frontend_sync"])
@pytest.mark.parametrize("capable", [False, True])
async def test_optional_display_archive_on_all_three_real_routes(tmp_path, route, capable):
    transcript = Transcript(tmp_path / "sessions" / "sess")
    _, rows, record = make_cycle()
    await transcript.append_messages(rows[:-1])
    assert await persist_record(transcript, record)
    await transcript.append_message(rows[-1])
    session = build_session(
        transcript.directory, ScriptedStream([text_turn("unused")]), cwd=tmp_path
    )
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(tmp_path))
    server = RuntimeServer(handle, kind="daemon")
    await server.start_in_process()
    writer = None
    try:
        owner = server._record
        assert DISPLAY_HISTORY_PEER_CYCLE_CAPABILITY in owner.capabilities
        reader, writer = await asyncio.open_connection("127.0.0.1", owner.control_port)
        auth = {
            "key": owner.control_key,
            "client": "attach",
            "locality": "local",
            "frontend_state": True,
            "display_window": True,
        }
        if capable:
            auth["display_history_peer_cycle_records"] = True
        writer.write(json.dumps(auth).encode() + b"\n")
        await writer.drain()
        while True:
            line = await asyncio.wait_for(reader.readline(), timeout=10)
            assert line, "attach must deliver a canonical sync"
            frame = json.loads(line)
            if frame.get("op") == "frontend_sync":
                window = frame["data"]["display_history"]
                break
        if route != "attach":
            request = {"op": route, "req": 1}
            if route == "history_page":
                assert window["snapshot_token"], "real page must supply its signed cut"
                request["before"] = window["snapshot_token"]
            writer.write(json.dumps(request).encode() + b"\n")
            await writer.drain()
            while True:
                line = await asyncio.wait_for(reader.readline(), timeout=10)
                assert line, "structured history RPC must return a result"
                frame = json.loads(line)
                if frame.get("op") == "result" and frame.get("req") == 1:
                    window = (
                        frame["data"]["display_history"]
                        if route == "frontend_sync"
                        else frame["data"]
                    )
                    break
        if capable:
            envelope = window["peer_cycle_display_records"]
            assert envelope["status"] == "complete"
            assert [item["id"] for item in envelope["records"]] == [record["id"]]
        else:
            assert "peer_cycle_display_records" not in window
            # Actual old schema, not a model that silently ignores extras.
            fields: dict[str, Any] = {}
            for name, field in DisplayHistoryWindow.model_fields.items():
                if name == "peer_cycle_display_records":
                    continue
                annotation = field.annotation
                assert annotation is not None, f"old DTO field {name} must retain its actual type"
                fields[name] = (annotation, copy.deepcopy(field))
            old = create_model("OldDisplayWindow", __config__=ConfigDict(extra="forbid"), **fields)
            old.model_validate(window)
    finally:
        if writer is not None:
            writer.close()
            await writer.wait_closed()
        server.close()
        await session.dispose()


def test_no_default_null_and_strip_helper_is_in_place():
    window = DisplayHistoryWindow(
        conversation_id="sess", owner_epoch="epoch", history_generation=0, through_id=None
    )
    assert "peer_cycle_display_records" not in window.model_dump(mode="json")
    payload = {"messages": [], "peer_cycle_display_records": {"records": []}}
    assert strip_peer_cycle_fields(payload, peer_cycle_capable=True) is payload
    assert "peer_cycle_display_records" in payload
    assert strip_peer_cycle_fields(payload, peer_cycle_capable=False) is payload
    assert "peer_cycle_display_records" not in payload
