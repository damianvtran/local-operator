"""Exercise inline UI negotiation against the real runtime TCP listener."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import (
    AbortSignal,
    ChatRequest,
    Message,
    MessageStartEvent,
    MessageUpdateEvent,
    StreamEvent,
)
from local_operator.harness.wake import DueWake, WakeSchedule
from local_operator.output_contract import OutputContract
from local_operator.session import session as session_module
from local_operator.session.inline_ui import INSTRUCTIONS
from local_operator.session.inline_ui_text import INLINE_UI_V1
from local_operator.session.runtime.server import _EVENT_QUEUE_MAX, RuntimeServer
from local_operator.session.runtime.serving import ServingSessionHandle
from tests.e2e.harness import build_session, text_turn
from tests.unit.session.runtime.test_server import FakeHandle


class _InlineHandle(FakeHandle):
    def __init__(self) -> None:
        super().__init__()
        self.raw_event: Callable[[dict[str, Any]], None] | None = None
        self.sidecars_requested = False

    def subscribe_events(
        self, on_event: Callable[[dict[str, Any]], None], *, inline_ui_sidecars: bool = False
    ) -> Callable[[], None]:
        self.raw_event = on_event
        self.sidecars_requested = inline_ui_sidecars
        return lambda: None


def _event(text: str, *, delta: str = "x") -> dict[str, Any]:
    return {
        "type": "message_update",
        "message": {"id": "answer", "role": "assistant", "text": text},
        "message_id": "answer",
        "delta": delta,
        "inline_ui": {
            "version": 1,
            "messages": {
                "answer": {
                    "version": 1,
                    "widgets": [
                        {"offset": 0, "payload": {"type": "bar", "title": text, "data": [["A", 1]]}}
                    ],
                }
            },
        },
    }


async def _send(writer: asyncio.StreamWriter, frame: dict[str, Any]) -> None:
    writer.write(json.dumps(frame).encode() + b"\n")
    await writer.drain()


async def _read(reader: asyncio.StreamReader, op: str) -> dict[str, Any]:
    async with asyncio.timeout(10):
        while line := await reader.readline():
            frame = json.loads(line)
            assert frame.get("op") != "error", frame
            if frame.get("op") == op:
                return frame
    raise AssertionError(f"socket closed before {op}")


@asynccontextmanager
async def _attach(
    server: RuntimeServer, **hello: Any
) -> AsyncIterator[tuple[asyncio.StreamReader, asyncio.StreamWriter]]:
    record = server._record
    reader, writer = await asyncio.open_connection("127.0.0.1", record.control_port)
    try:
        await _send(
            writer,
            {
                "key": record.control_key,
                "client": "attach",
                "events": True,
                "frontend_state": True,
                **hello,
            },
        )
        yield reader, writer
    finally:
        writer.close()
        await writer.wait_closed()


@pytest.mark.asyncio
async def test_warm_mobile_prompt_negotiates_inline_ui_per_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Exercise a standalone wake immediately after the opted-in prompt, not a
    # wake parked until the next human message.
    monkeypatch.setattr(session_module, "WAKE_OPERATOR_QUIET_S", 0)
    requests: list[ChatRequest] = []
    helpers: list[ChatRequest] = []

    async def stream(
        request: ChatRequest, signal: AbortSignal | None = None
    ) -> AsyncIterator[StreamEvent]:
        # Naming/housekeeping calls use this transport but must not receive the
        # turn's hint. Keep them separate from the conversation request checks.
        is_turn = request.purpose == "turn"
        (requests if is_turn else helpers).append(request)
        for event in text_turn('{"ok":true}' if is_turn else "Loopback test"):
            yield event

    session = build_session(tmp_path / "sessions" / "inline-prompt", stream, cwd=tmp_path)
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(tmp_path))
    server = RuntimeServer(handle, kind="daemon")
    await server.start_in_process()
    try:
        assert INLINE_UI_V1 in server._record.capabilities

        def check_request(label: str, *, enabled: bool) -> None:
            request = requests[-1]
            assert INSTRUCTIONS not in request.system_blocks, label
            assert request.inline_ui_hint == (INSTRUCTIONS if enabled else None), label
            assert all(INSTRUCTIONS not in message.text for message in request.messages), label
            print(f"model request {label}: inline instructions={enabled}")

        async def prompt(
            socket: tuple[asyncio.StreamReader, asyncio.StreamWriter],
            text: str,
            *,
            enabled: bool = False,
            **fields: Any,
        ) -> None:
            reader, writer = socket
            before = len(requests)
            req = before + 1
            await _send(
                writer,
                {"op": "prompt", "text": text, "req": req, "wait_complete": True, **fields},
            )
            ack = await _read(reader, "ack")
            assert ack["req"] == req and ack["settled"] is True
            assert len(requests) == before + 1, text
            assert requests[-1].messages[-1].text == text
            check_request(text, enabled=enabled)

        mobile = {"client": "daemon", "origin": "mobile", "locality": "remote"}
        async with _attach(server, **mobile, inline_ui=True) as negotiated:
            await prompt(negotiated, "marked", inline_ui=True, enabled=True)
            user = next(
                entry
                for entry in session.transcript.entries()
                if entry.payload.get("role") == "user"
            )
            payload = user.payload
            assert payload["provider_payload"][session_module.REMOTE_PROMPT_KEY] is True
            assert payload["provider_payload"][session_module.INPUT_SURFACE_KEY] == "mobile"
            assert "inline_ui" not in json.dumps(payload)
            await prompt(negotiated, "unmarked")
            await prompt(negotiated, "marked before wake", inline_ui=True, enabled=True)

            before = len(requests)
            schedule = WakeSchedule(
                id="inline-control", message="check", next_due_at=0, created_at=0
            )
            async with asyncio.timeout(10):
                await session._deliver_wake(
                    DueWake(schedule=schedule, occurrence=1, planned_total=1, final=True)
                )
                while len(requests) == before or session._is_streaming:
                    await asyncio.sleep(0.001)
            assert len(requests) == before + 1
            check_request("background wake", enabled=False)

            await prompt(negotiated, "truthy command", inline_ui=1)

            contract = OutputContract(format="json")
            session.set_output_contract(contract)
            await prompt(negotiated, "structured output", inline_ui=True)
            assert contract.system_block() in requests[-1].system_blocks
            session.set_output_contract(None)

        async with _attach(server, **mobile, inline_ui="true") as malformed:
            await prompt(malformed, "truthy hello", inline_ui=True)
        async with _attach(server, client="daemon", inline_ui=True) as nonmobile:
            await prompt(nonmobile, "origin in op only", inline_ui=True, origin="mobile")
        await session.prompt("local in-process TUI default")
        check_request("local in-process TUI default", enabled=True)
        async with _attach(server, surface="desktop") as old_desktop:
            await _read(old_desktop[0], "frontend_sync")
            await prompt(old_desktop, "legacy desktop", inline_ui=True)
        async with _attach(server) as old_terminal:
            await _read(old_terminal[0], "frontend_sync")
            await prompt(old_terminal, "legacy terminal", inline_ui=True)
        async with _attach(server, inline_ui=True) as terminal:
            await _read(terminal[0], "frontend_sync")
            await prompt(terminal, "terminal opted in only at hello")
            await prompt(terminal, "negotiated terminal", inline_ui=True, enabled=True)
        async with _attach(server, surface="desktop", inline_ui=True) as desktop:
            await _read(desktop[0], "frontend_sync")
            await prompt(desktop, "negotiated desktop", inline_ui=True, enabled=True)
        server._record.capabilities.remove(INLINE_UI_V1)
        async with _attach(server, **mobile, inline_ui=True) as unsupported:
            await prompt(unsupported, "runtime capability absent", inline_ui=True)
        assert all(
            INSTRUCTIONS not in request.system_blocks and request.inline_ui_hint is None
            for request in helpers
        )
    finally:
        server.close()
        await handle.dispose()


@pytest.mark.asyncio
async def test_mobile_steer_carries_inline_ui_to_the_next_provider_call(tmp_path: Path) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    requests: list[ChatRequest] = []

    async def stream(
        request: ChatRequest, signal: AbortSignal | None = None
    ) -> AsyncIterator[StreamEvent]:
        if request.purpose == "turn":
            requests.append(request)
            if len(requests) == 1:
                entered.set()
                await release.wait()
        for event in text_turn("Complete."):
            yield event

    session = build_session(tmp_path / "sessions" / "inline-steer", stream, cwd=tmp_path)
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(tmp_path))
    server = RuntimeServer(handle, kind="daemon")
    await server.start_in_process()
    try:
        async with _attach(
            server, client="daemon", origin="mobile", locality="remote", inline_ui=True
        ) as (reader, writer):
            await _send(writer, {"op": "prompt", "text": "Start without widgets.", "req": 1})
            assert (await _read(reader, "ack"))["req"] == 1
            async with asyncio.timeout(10):
                await entered.wait()
            assert INSTRUCTIONS not in requests[0].system_blocks
            assert requests[0].inline_ui_hint is None
            await _send(
                writer, {"op": "steer", "text": "Add a table.", "inline_ui": True, "req": 2}
            )
            assert (await _read(reader, "ack"))["req"] == 2
            release.set()
            async with asyncio.timeout(10):
                while len(requests) < 2:
                    await asyncio.sleep(0.001)
            assert requests[1].messages[-1].text == "Add a table."
            assert INSTRUCTIONS not in requests[1].system_blocks
            assert requests[1].inline_ui_hint == INSTRUCTIONS
    finally:
        release.set()
        server.close()
        await handle.dispose()


@pytest.mark.asyncio
async def test_reconnect_prose_seed_is_exact_before_an_inline_fence_closes() -> None:
    handle = _InlineHandle()
    server = RuntimeServer(handle, kind="daemon")
    await server.start_in_process()
    try:
        message = Message(role="assistant")
        message._inline_ui_source_version = 1
        handle._frontend._fold_live_event(MessageStartEvent(message=message))
        for delta in ("Exact ", "prose.\n\n```lop-ui\n{broken so far"):
            event = MessageUpdateEvent(message=message, delta=delta)
            handle._frontend._fold_live_event(event)
        async with _attach(server) as legacy, _attach(server, inline_ui=True) as modern:
            for (reader, _), rich in ((legacy, False), (modern, True)):
                payload = (await _read(reader, "frontend_sync"))["data"]
                seed = payload["snapshot"]["live_events"][-1]
                assert seed["delta"] == "Exact prose.\n\n\n"
                if rich:
                    assert seed["inline_ui"] == {
                        "version": 1,
                        "messages": {
                            message.id: {
                                "version": 1,
                                "text": "Exact prose.\n\n\n",
                                "widgets": [],
                            }
                        },
                    }
                else:
                    assert "inline_ui" not in seed
    finally:
        server.close()


@pytest.mark.asyncio
async def test_reconnect_projects_widgets_from_actual_stream_fragments() -> None:
    handle = _InlineHandle()
    server = RuntimeServer(handle, kind="daemon")
    await server.start_in_process()
    try:
        message = Message(role="assistant")
        handle._frontend._fold_live_event(MessageStartEvent(message=message))
        message._inline_ui_source_version = 1
        for delta in (
            "Summary.\n\n```lop-",
            'ui\n{"type":"bar","title":"Replay","data":[["A",1]]}\n',
            "```\nEnd.",
        ):
            handle._frontend._fold_live_event(MessageUpdateEvent(message=message, delta=delta))
        assert "inline_ui" not in handle._frontend.live_events_copy()[-1]
        async with _attach(server) as legacy, _attach(server, inline_ui=True) as modern:
            for (reader, writer), rich in ((legacy, False), (modern, True)):
                frame = await _read(reader, "frontend_sync")
                await _send(writer, {"op": "frontend_sync", "req": 1})
                reply = await _read(reader, "result")
                for payload in (frame["data"], reply["data"]):
                    event = payload["snapshot"]["live_events"][-1]
                    assert event["delta"] == "Summary.\n\n\nEnd."
                    if rich:
                        widget = event["inline_ui"]["messages"][message.id]["widgets"][0]
                        assert widget == {
                            "offset": len("Summary.\n\n"),
                            "payload": {"type": "bar", "title": "Replay", "data": [["A", 1]]},
                        }
                    else:
                        assert "inline_ui" not in event
    finally:
        server.close()


def _assert_event(observed: dict[str, Any], source: dict[str, Any], *, rich: bool) -> None:
    expected = dict(source)
    if not rich:
        del expected["inline_ui"]
    assert observed == expected


@pytest.mark.asyncio
async def test_mixed_viewers_gate_raw_events_and_live_sync_over_loopback() -> None:
    handle = _InlineHandle()
    server = RuntimeServer(handle, kind="daemon")
    seed = _event("seed")
    handle._frontend.mutate(live_events=[seed])
    await server.start_in_process()
    try:
        assert INLINE_UI_V1 in server._record.capabilities
        assert handle.sidecars_requested
        async with (
            _attach(server) as legacy,
            _attach(server, inline_ui=True) as modern,
            _attach(server, inline_ui="true", locality="remote") as malformed,
        ):
            viewers = ((legacy, False), (modern, True), (malformed, False))
            for (reader, writer), rich in viewers:
                seed_frame = await _read(reader, "frontend_sync")
                _assert_event(seed_frame["data"]["snapshot"]["live_events"][0], seed, rich=rich)
                await _send(writer, {"op": "frontend_sync", "req": 1, "inline_ui": True})
                reply = await _read(reader, "result")
                assert reply["req"] == 1
                _assert_event(reply["data"]["snapshot"]["live_events"][0], seed, rich=rich)

            latest = _event("live")
            handle._frontend.mutate(live_events=[latest])
            assert handle.raw_event is not None
            handle.raw_event(latest)
            for (reader, _), rich in viewers:
                update = await _read(reader, "frontend_update")
                _assert_event(update["data"]["changes"]["live_events"][0], latest, rich=rich)
                event = await _read(reader, "event")
                _assert_event(event["data"], latest, rich=rich)

        async with _attach(server, inline_ui=True) as (reader, _):
            reconnect = await _read(reader, "frontend_sync")
            _assert_event(reconnect["data"]["snapshot"]["live_events"][0], latest, rich=True)
        print(
            "loopback: legacy, opted-in and malformed hellos; "
            "raw/update/push/RPC/reconnect verified"
        )
    finally:
        server.close()


@pytest.mark.asyncio
async def test_queue_compaction_keeps_latest_cumulative_sidecar_over_loopback() -> None:
    server = RuntimeServer(_InlineHandle(), kind="daemon")
    await server.start_in_process()
    try:
        async with _attach(server, inline_ui=True) as (reader, _):
            await _read(reader, "frontend_sync")
            # One synchronous burst fills the writer FIFO before its drain can run.
            for n in range(_EVENT_QUEUE_MAX + 1):
                server._relay_on_loop(_event(str(n)))
            folded = (await _read(reader, "event"))["data"]
            _assert_event(
                folded, _event(str(_EVENT_QUEUE_MAX - 1), delta="x" * _EVENT_QUEUE_MAX), rich=True
            )
            last = (await _read(reader, "event"))["data"]
            _assert_event(last, _event(str(_EVENT_QUEUE_MAX)), rich=True)
            print(
                f"loopback: {_EVENT_QUEUE_MAX} queued deltas folded with latest cumulative sidecar"
            )
    finally:
        server.close()
