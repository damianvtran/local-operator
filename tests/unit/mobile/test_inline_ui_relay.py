"""Exercise inline UI through the runtime listener and the daemon's real dial."""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import httpx
import pytest
from starlette.testclient import TestClient

from local_operator.harness.types import (
    AbortSignal,
    ChatRequest,
    Message,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    StreamEndEvent,
    StreamEvent,
    StreamTextDelta,
)
from local_operator.mobile.auth import COOKIE_NAME, sign_cookie
from local_operator.mobile.daemon import MobileDaemon, SessionEntry, _dial, build_app
from local_operator.mobile.projection import ProjectionFold, read_live_inline_ui
from local_operator.mobile.tui_handle import TuiSessionHandle
from local_operator.session.errors import TURN_IN_FLIGHT, TurnInFlight
from local_operator.session.inline_ui import INSTRUCTIONS
from local_operator.session.inline_ui_text import INLINE_UI_V1
from local_operator.session.runtime.inline_ui_egress import InlineUiEvents
from local_operator.session.runtime.server import RuntimeServer
from local_operator.session.runtime.serving import ServingSessionHandle
from local_operator.session.transcript import Transcript
from tests.e2e.harness import build_session, text_turn
from tests.unit.mobile.test_stream_reliability import http_scope
from tests.unit.session.runtime.test_inline_ui_socket import _InlineHandle, _read, _send


@asynccontextmanager
async def _sse_view(app, session_id: str, query: bytes = b""):
    scope = http_scope(f"/api/sessions/{session_id}/events")
    scope["query_string"] = query
    inbound: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    outbound: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    inbound.put_nowait({"type": "http.request"})
    task = asyncio.create_task(app(scope, inbound.get, outbound.put))
    try:
        start = await outbound.get()
        assert start["type"] == "http.response.start"
        assert start["status"] == 200
        assert dict(start["headers"])[b"content-type"].startswith(b"text/event-stream")
        yield outbound
    finally:
        inbound.put_nowait({"type": "http.disconnect"})
        await task


async def _next_projection(view: asyncio.Queue[dict[str, Any]]) -> tuple[bytes, dict]:
    frame = await view.get()
    assert frame["type"] == "http.response.body"
    body = frame["body"]
    prefix = b"event: projection\ndata: "
    assert body.startswith(prefix) and body.endswith(b"\n\n"), body
    return body, json.loads(body[len(prefix) :])


@pytest.mark.asyncio
async def test_live_inline_ui_reaches_daemon_without_changing_legacy_projection() -> None:
    handle = _InlineHandle()
    message = Message(role="assistant")
    message._inline_ui_source_version = 1
    prose = "Summary 📈.\n\n"
    widget = '{"type":"bar","title":"Counts","data":[["A",1]]}'
    tail = "Continued."
    source = f"{prose}```lop-ui\n{widget}\n```\n{tail}"
    # The same fold that the owner runs must never put a raw fragment in its
    # public projection, even before any negotiated sidecar reaches the relay.
    fold = ProjectionFold(handle.session_projection_seed)
    fold.fold_event(MessageStartEvent(message=message))
    for delta in (prose, "```lop-ui\n", widget, "\n```\n", tail):
        fold.fold_event(MessageUpdateEvent(message=message, delta=delta))
        visible = handle.session_projection_seed.transcript[-1].text
        assert "lop-ui" not in visible and "Counts" not in visible
    plain = prose + "\n" + tail
    assert visible == plain
    message.content = Message.assistant(source).content
    fold.fold_event(MessageEndEvent(message=message))
    assert handle.session_projection_seed.transcript[-1].text == plain
    server = RuntimeServer(handle, kind="daemon")
    await server.start_in_process()
    record = server.record
    try:
        assert INLINE_UI_V1 in record.capabilities
        assert handle.sidecars_requested and handle.raw_event is not None
        egress = InlineUiEvents(sidecars=handle.sidecars_requested)
        egress.dump(MessageStartEvent(message=message))
        event = egress.dump(MessageUpdateEvent(message=message, delta=source))
        sidecar = event["inline_ui"]["messages"][message.id]
        assert sidecar["text"] == prose + "\n" + tail
        assert sidecar["widgets"][0]["offset"] == len(prose) + 1  # The emoji uses two UTF-16 units.
        assert sidecar["widgets"][0]["payload"]["title"] == "Counts"

        # Send request-side opt-in and events=true, but no hello opt-in. The
        # daemon must get neither raw events nor sidecars before the FIFO fence.
        reader, writer = await asyncio.open_connection("127.0.0.1", record.control_port)
        try:
            await _send(writer, {"key": record.control_key, "events": True})
            legacy = await _read(reader, "projection")
            await _send(writer, {"op": "ping", "inline_ui": True, "req": 1})
            await _read(reader, "ack")
            handle.raw_event(event)
            await asyncio.sleep(0)  # Run the relay callback before queuing the fence.
            legacy_conn = next(iter(server._clients.values()))
            server._enqueue_client_frame(legacy_conn, {"op": "test_fence"})
            async with asyncio.timeout(10):
                while line := await reader.readline():
                    frame = json.loads(line)
                    if frame["op"] == "test_fence":
                        break
                    assert frame["op"] == "projection", frame["op"]
                else:
                    pytest.fail("legacy connection closed before FIFO fence")
        finally:
            writer.close()
            await writer.wait_closed()

        # Own the one dial below. The route's optional residency attach would
        # otherwise send unrelated repaints during the sidecar-only assertion.
        daemon = MobileDaemon(port=0, password="pw", dial_registrants=False)
        entry = SessionEntry(record)
        daemon.table.entries[record.pid] = entry
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        dial = asyncio.create_task(_dial(daemon, entry))
        try:
            app = build_app(daemon)
            async with asyncio.timeout(10):
                await entry.ready.wait()
                assert entry.live_inline_ui() == {}
                # An existing watcher prevents a watch transition on SSE open.
                daemon.table.session_subscribers[record.session_id] = {queue}
                # The ping fences the initial write. The runtime then refreshes
                # after the ack, even for ping, so consume that repaint too.
                assert (await daemon.request(record.pid, "ping"))["op"] == "ack"
                assert "inline_ui" not in await queue.get()
                async with (
                    _sse_view(app, record.session_id) as legacy_view,
                    _sse_view(app, record.session_id, b"inline_ui=1") as modern_view,
                    _sse_view(app, record.session_id, b"inline_ui=true") as malformed_view,
                ):
                    legacy_body, legacy_seed = await _next_projection(legacy_view)
                    assert (await _next_projection(modern_view))[0] == legacy_body
                    assert (await _next_projection(malformed_view))[0] == legacy_body
                    assert legacy_seed["transcript"] == legacy["data"]["transcript"]
                    assert legacy_seed["transcript"][0]["text"] == plain
                    assert "inline_ui" not in legacy_seed

                    # Emit only the sidecar, not a runtime repaint. All viewers
                    # already saw the same row; only modern viewers should wake.
                    handle.raw_event(event)
                    _, modern_update = await _next_projection(modern_view)
                    assert entry.live_inline_ui() == {message.id: sidecar}
                    assert modern_update.pop("inline_ui") == event["inline_ui"]
                    assert modern_update["transcript"] == legacy_seed["transcript"]
                    assert queue.empty() and legacy_view.empty() and malformed_view.empty()

                    async with _sse_view(app, record.session_id, b"inline_ui=1") as reconnect:
                        _, modern_seed = await _next_projection(reconnect)
                        assert modern_seed["inline_ui"] == event["inline_ui"]

                    await daemon.request(record.pid, "snapshot")
                    repaint = await queue.get()
                    legacy_repaint, plain_update = await _next_projection(legacy_view)
                    assert (await _next_projection(malformed_view))[0] == legacy_repaint
                    assert "inline_ui" not in plain_update and "inline_ui" not in repaint
                    assert plain_update["transcript"] == repaint["transcript"]
                    print("legacy SSE row:", repr(plain_update["transcript"][0]["text"]))
                    print("modern SSE overlay:", json.dumps(modern_seed["inline_ui"]))
                    # A marked final event can supersede a previously valid
                    # live widget. Its private tombstone is never a public row.
                    event["inline_ui"] = {"version": 1, "messages": {message.id: None}}
                    handle.raw_event(event)
                    while True:
                        _, reset = await _next_projection(modern_view)
                        if "inline_ui" not in reset:
                            break
                    assert entry.live_inline_ui() == {}
        finally:
            if entry.writer is not None:
                entry.writer.close()
                await entry.writer.wait_closed()
            dial.cancel()
            with suppress(asyncio.CancelledError):
                await dial
        assert entry.live_inline_ui() == {}
    finally:
        await server.aclose()


class _CommandHandle(_InlineHandle):
    async def prompt(
        self,
        text: str,
        images: list[dict[str, str]] | None = None,
        *,
        command_id: str | None = None,
        inline_ui: bool = False,
    ) -> str:
        if text == "busy":
            raise TurnInFlight(TURN_IN_FLIGHT)
        return await self._record("prompt", text, command_id=command_id, inline_ui=inline_ui)

    async def steer(
        self,
        text: str,
        images: list[dict[str, str]] | None = None,
        *,
        command_id: str | None = None,
        inline_ui: bool = False,
    ) -> str:
        return await self._record("steer", text, command_id=command_id, inline_ui=inline_ui)


@pytest.mark.asyncio
@pytest.mark.parametrize("capable", [True, False], ids=["v1-owner", "legacy-owner"])
async def test_http_continuation_negotiates_inline_ui_per_send(capable: bool) -> None:
    handle = _CommandHandle()
    server = RuntimeServer(handle, kind="daemon")
    await server.start_in_process()
    daemon = MobileDaemon(port=0, password="pw", dial_registrants=False)
    record = server.record
    if not capable:
        record = replace(record, capabilities=[c for c in record.capabilities if c != INLINE_UI_V1])
    entry = SessionEntry(record)
    daemon.table.entries[record.pid] = entry
    dial = asyncio.create_task(_dial(daemon, entry))
    ids = {case: str(uuid4()) for case in ("rich", "legacy", "malformed", "busy", "steer")}
    try:
        async with asyncio.timeout(10):
            await entry.ready.wait()
            transport = httpx.ASGITransport(app=build_app(daemon))
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://fixture",
                cookies={COOKIE_NAME: sign_cookie("pw")},
            ) as client:
                url = f"/api/sessions/{record.session_id}/command"
                async with httpx.AsyncClient(
                    transport=transport, base_url="http://fixture"
                ) as anon:
                    denied = await anon.post(url, json={
                        "op": "prompt", "command_id": str(uuid4()),
                        "text": "denied", "inline_ui": True,
                    })
                    assert denied.status_code == 401, denied.text
                for suffix, fields, channel in (
                    ("rich", {"inline_ui": True}, "prompt"),
                    ("legacy", {}, "prompt"),
                    ("malformed", {"inline_ui": "true"}, "prompt"),
                    ("busy", {"inline_ui": True}, "steer"),
                    ("steer", {"inline_ui": True}, "steer"),
                ):
                    reply = await client.post(url, json={
                        "op": "steer" if suffix == "steer" else "prompt",
                        "command_id": ids[suffix], "text": suffix, **fields,
                    })
                    assert reply.status_code == 200, (suffix, reply.text)
                    assert reply.json()["delivered_as"] == channel, (suffix, reply.text)
        assert [item for item in handle.calls if item[0] in {"prompt", "steer"}] == [
            ("prompt", ("rich",), {"command_id": ids["rich"], "inline_ui": capable}),
            ("prompt", ("legacy",), {"command_id": ids["legacy"], "inline_ui": False}),
            ("prompt", ("malformed",), {"command_id": ids["malformed"], "inline_ui": False}),
            ("steer", ("busy",), {"command_id": ids["busy"], "inline_ui": capable}),
            ("steer", ("steer",), {"command_id": ids["steer"], "inline_ui": capable}),
        ]
    finally:
        dial.cancel()
        with suppress(asyncio.CancelledError):
            await dial
        await server.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("capable", [True, False], ids=["v1-winner", "legacy-winner"])
async def test_cold_prompt_is_gated_by_selected_owner(
    capable: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from local_operator.session.runtime import launch

    monkeypatch.setattr("local_operator.paths.config_dir", lambda: tmp_path)
    handle = _CommandHandle()
    server = RuntimeServer(handle, kind="daemon")
    await server.start_in_process()
    record = server.record
    session_id = record.session_id
    if not capable:
        record = replace(record, capabilities=[c for c in record.capabilities if c != INLINE_UI_V1])
    journal = Transcript(tmp_path / "sessions" / session_id)
    await journal.append_message(Message.user("An existing cold conversation."))
    journal.flush()

    # Substitute process discovery only: the HTTP wake, retained command,
    # errand delivery, auth and command still use their production code.
    monkeypatch.setattr(
        "local_operator.mobile.attach_client.find_runtime_record", lambda *_: (record, None)
    )

    async def engage(
        sid: str, _cwd: str, work: launch.Errand, **_kwargs: Any
    ) -> launch.EngageOutcome:
        assert isinstance(work, launch.PromptErrand) and work.inline_ui is True
        assert work.origin == "mobile" and sid == session_id
        detail, duplicate = await launch._deliver(record, sid, work)
        return launch.EngageOutcome(sid, detail, spawned=True, duplicate=duplicate)

    monkeypatch.setattr(launch, "engage_runtime", engage)
    daemon = MobileDaemon(port=0, password="pw", dial_registrants=True)
    command_id = str(uuid4())
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=build_app(daemon)),
            base_url="http://fixture",
            cookies={COOKIE_NAME: sign_cookie("pw")},
        ) as client:
            response = await client.post(f"/api/sessions/{session_id}/command", json={
                "op": "prompt", "command_id": command_id,
                "text": "Resume with a plot", "inline_ui": True,
            })
            assert response.status_code == 200, response.text
            assert response.json()["ok"] is True
        assert [call for call in handle.calls if call[0] == "prompt"] == [
            ("prompt", ("Resume with a plot",), {"command_id": command_id, "inline_ui": capable}),
        ]
        print(f"cold POST: {response.status_code}, selected owner inline_ui={capable}")
    finally:
        await server.aclose()


def test_runtime_overlay_offsets_and_tombstones() -> None:
    payload = {"type": "bar", "title": "Counts", "data": [["A", 1]]}
    row = {"version": 1, "text": "📈 prose", "widgets": [{"offset": 2, "payload": payload}]}
    invalid = {**row, "widgets": [{"offset": 1, "payload": payload}]}
    assert read_live_inline_ui({"version": 1, "messages": {
        "valid": row, "split_surrogate": invalid, "cleared": None,
    }}) == {"valid": row, "cleared": None}
    assert not read_live_inline_ui({"version": True, "messages": {"valid": row}})


@pytest.mark.asyncio
@pytest.mark.parametrize("owner_kind", ["daemon", "tui"])
async def test_warm_mobile_send_streams_and_replays_widget(
    owner_kind: Literal["daemon", "tui"], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.analytics import recorder

    analytics = recorder.AnalyticsRecorder()
    monkeypatch.setattr(recorder, "_recorder", analytics)
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: tmp_path)
    finish_answer = asyncio.Event()
    requests: list[ChatRequest] = []
    lead = "Traffic 📈 rose.\n\n"
    payload = {"type": "bar", "title": "Requests", "data": [["June", 42]]}
    fragment = f"```lop-ui\n{json.dumps(payload)}\n```\n"

    async def stream(
        request: ChatRequest, _signal: AbortSignal | None = None
    ) -> AsyncIterator[StreamEvent]:
        if request.purpose != "turn":
            for event in text_turn("Traffic test"):
                yield event
            return
        requests.append(request)
        yield StreamTextDelta(delta=lead)
        yield StreamTextDelta(delta=fragment[:16])
        yield StreamTextDelta(delta=fragment[16:])
        await finish_answer.wait()
        yield StreamTextDelta(delta="June had 42 requests.")
        yield StreamEndEvent(stop_reason="stop")

    session = build_session(tmp_path / "sessions" / "inline-send", stream, cwd=tmp_path)
    loop = asyncio.get_running_loop()

    class Owner:
        # Mimic the Textual hop contract, not its widgets. The real session,
        # TuiSessionHandle, runtime listener and relay still own their loops.
        _session = session

        def call_from_thread(self, callback: Any) -> None:
            done = threading.Event()

            def run() -> None:
                try:
                    callback()
                finally:
                    done.set()

            loop.call_soon_threadsafe(run)
            assert done.wait(timeout=5), "the owner loop never ran the hop"

    serving = (
        ServingSessionHandle(session, loop, cwd=str(tmp_path)) if owner_kind == "daemon" else None
    )
    handle = serving if serving is not None else TuiSessionHandle(Owner())  # type: ignore[arg-type]
    server = RuntimeServer(handle, kind=owner_kind)
    await server.start_in_process()
    daemon = MobileDaemon(port=0, password="pw", dial_registrants=False)
    record = server.record
    entry = SessionEntry(record)
    daemon.table.entries[record.pid] = entry
    dial = asyncio.create_task(_dial(daemon, entry))
    app = build_app(daemon)
    history_url = f"/api/sessions/{record.session_id}/history"
    try:
        async with asyncio.timeout(15):
            await entry.ready.wait()
            async with (
                _sse_view(app, record.session_id, b"inline_ui=1") as modern,
                _sse_view(app, record.session_id) as legacy,
                httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app),
                    base_url="http://fixture",
                    cookies={COOKIE_NAME: sign_cookie("pw")},
                ) as client,
            ):
                await _next_projection(modern)
                await _next_projection(legacy)
                reply = await client.post(
                    f"/api/sessions/{record.session_id}/command",
                    json={
                        "op": "prompt",
                        "text": "Chart June traffic.",
                        "command_id": str(uuid4()),
                        "inline_ui": True,
                    },
                )
                assert reply.status_code == 200, reply.text
                assert reply.json()["delivered_as"] == "prompt"
                while True:
                    raw, frame = await _next_projection(modern)
                    assert b"```lop-ui" not in raw
                    if frame.get("inline_ui"):
                        break
                assert len(requests) == 1 and INSTRUCTIONS in requests[0].system_blocks
                assert frame["streaming"] is True and not finish_answer.is_set()
                ((row_id, overlay),) = frame["inline_ui"]["messages"].items()
                assert overlay["text"] == lead + "\n"
                assert overlay["widgets"] == [{"offset": len(lead) + 1, "payload": payload}]
                while True:
                    raw, plain = await _next_projection(legacy)
                    assert b"```lop-ui" not in raw and "inline_ui" not in plain
                    row = next((row for row in plain["transcript"] if row["id"] == row_id), None)
                    if row is not None and row["text"] == overlay["text"]:
                        break
                finish_answer.set()
                while True:
                    _, frame = await _next_projection(modern)
                    if frame["streaming"] is False:
                        break
                rich = await client.get(history_url, params={"inline_ui": "1"})
                plain = await client.get(history_url)
                assert rich.status_code == plain.status_code == 200
                rich_body = rich.json()
                saved = rich_body.pop("inline_ui")["messages"][row_id]
                assert rich_body == plain.json() and b"```lop-ui" not in plain.content
                assert saved["widgets"] == overlay["widgets"]
                assert saved["text"] == lead + "\nJune had 42 requests."
                print(f"warm {owner_kind} POST: {reply.status_code}, {reply.json()}")
                print(f"journal-backed GET: {rich.status_code}, widgets={len(saved['widgets'])}")
                # A new relay process loses all socket-side deltas. Its first
                # warm seed must recover completed rows, not need a new answer.
                entry._live_inline_ui.clear()
                async with _sse_view(app, record.session_id, b"inline_ui=1") as rejoined:
                    _, seed = await _next_projection(rejoined)
                    assert seed["inline_ui"]["messages"][row_id] == saved
    finally:
        finish_answer.set()
        dial.cancel()
        with suppress(asyncio.CancelledError):
            await dial
        await server.aclose()
        if serving is not None:
            await serving.dispose()
        else:
            await session.dispose()
        analytics.close()


def test_durable_phone_history_gates_overlay_and_preserves_literal_fences(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = tmp_path / "config"
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: cfg)
    session_id = "inline-durable"
    journal = Transcript(cfg / "sessions" / session_id)
    lead = "Trend 📈.\n\n"
    source = lead + '```lop-ui\n{"type":"bar","title":"Totals","data":[["A",5]]}\n```\nAfter.'
    marked = Message.assistant(source, id="marked")
    marked._inline_ui_source_version = 1
    literal = Message.assistant(source, id="literal")
    user = Message.user("Show both.")
    asyncio.run(journal.append_messages([user, marked, literal]))
    app = build_app(MobileDaemon(port=0, password="pw", dial_registrants=False))

    with TestClient(app) as client:
        client.post("/login", data={"password": "pw"})
        url = f"/api/sessions/{session_id}/history"
        for cursor in ({}, {"before": literal.id}, {"after": user.id}):
            plain = client.get(url, params=cursor)
            modern = client.get(url, params={**cursor, "inline_ui": "1"})
            malformed = client.get(url, params={**cursor, "inline_ui": "true"})
            assert plain.status_code == modern.status_code == malformed.status_code == 200
            assert plain.content == malformed.content
            body = modern.json()
            overlay = body.pop("inline_ui")
            assert body == plain.json()
            assert set(overlay["messages"]) == {marked.id}
            projected = overlay["messages"][marked.id]
            assert projected["widgets"][0]["offset"] == len(lead) + 1
            assert projected["text"] == lead + "\nAfter."
            rows = {row["id"]: row["text"] for row in body["entries"]}
            assert rows[marked.id] == projected["text"]
            if "before" not in cursor:
                assert rows[literal.id] == source
            assert "_inline_ui_source" not in plain.text

    async def check_cold_seed() -> None:
        async with (
            asyncio.timeout(10),
            _sse_view(app, session_id) as legacy,
            _sse_view(app, session_id, b"inline_ui=1") as modern,
        ):
            _, plain_frame = await _next_projection(legacy)
            _, rich_frame = await _next_projection(modern)
            assert "inline_ui" not in plain_frame
            overlay = rich_frame.pop("inline_ui")
            assert plain_frame["transcript"] == rich_frame["transcript"]
            assert set(overlay["messages"]) == {marked.id}
            assert overlay["messages"][marked.id]["text"] == lead + "\nAfter."

    asyncio.run(check_cold_seed())
