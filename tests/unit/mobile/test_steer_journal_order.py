"""Phone steer ORDER on the live projection vs the journal (``steer_journal_order``).

Operator report: "not all messages load on mobile and sometimes they load out
of order". One concrete out-of-order source: a phone steer is painted at SEND
time (``ProjectionFold.note_user_message``) and upgraded IN PLACE when the
drain delivers it at a later tool boundary (``absorb_user_event``). The journal
— and therefore ``/history`` (``fold_messages_to_entries``) — records it at
DELIVERY. So the live frame reads ``steer, tool, tool`` while every page, every
fresh mount and every resume reads ``tool, tool, user``.

The fix is opt-in per stream: a client that declares ``?steer_journal_order=1``
on its events stream receives the delivered steer at its journal slot (and as
``kind="user"``, the page's kind). A client that does not declare it receives
the byte-identical frame it always did — installed phones from before
local-operator-mobile #359 mis-place a held row that moves later, so the move
must never reach them.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest
from starlette.requests import Request
from starlette.testclient import TestClient

from local_operator.harness.types import (
    AgentMessage,
    Message,
    TextContent,
    ToolCall,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolResult,
)
from local_operator.mobile.auth import COOKIE_NAME, sign_cookie
from local_operator.mobile.daemon import (
    MobileDaemon,
    SessionEntry,
    _projection_frame,
    _sse,
    build_app,
)
from local_operator.mobile.projection import (
    ProjectionFold,
    fold_messages_to_entries,
    steer_journal_order_frame,
)
from local_operator.mobile.types import SessionProjection, SessionRecord

PASSWORD = "pw123"
SESSION_ID = "steer-order-session"
PID = 4242
SESSION_ROUTE = "/api/sessions/{session_id:str}/events"


def _record() -> SessionRecord:
    return SessionRecord(
        pid=PID,
        kind="tui",
        session_id=SESSION_ID,
        conversation_name="fixture",
        cwd="/synthetic",
        model_label="fixture",
        control_port=1,
        control_key="fixture",
    )


def _key(row: Any) -> str:
    """The client's row identity (``rowKey``/``rowIdentity``): a tool row is its
    call, every other row its id — the two folds mint different tool-row ids."""
    kind = row["kind"] if isinstance(row, dict) else row.kind
    if kind == "tool":
        call = row["tool_call_id"] if isinstance(row, dict) else row.tool_call_id
        return f"tool:{call}"
    ident = row["id"] if isinstance(row, dict) else row.id
    return f"{kind}:{ident}"


def _tool_round(fold: ProjectionFold, *call_ids: str) -> list[ToolCall]:
    calls = [ToolCall(id=c, name="bash", arguments={"command": "true"}) for c in call_ids]
    for call in calls:
        fold.fold_event(
            ToolExecutionStartEvent(tool_call_id=call.id, tool_name="bash", args=call.arguments)
        )
        fold.fold_event(
            ToolExecutionEndEvent(
                tool_call_id=call.id,
                tool_name="bash",
                duration_s=0.1,
                result=ToolResult(
                    tool_call_id=call.id, content=[TextContent(text="ok")], duration_s=0.1
                ),
            )
        )
    return calls


def _steered_fold() -> tuple[ProjectionFold, list[AgentMessage]]:
    """Prompt, phone steer painted at send time, two tools, then delivery.
    Returns the fold and the journal the same run writes."""
    fold = ProjectionFold(SessionProjection(session_id=SESSION_ID, pid=0))
    prompt = Message.user("do the thing", id="u1")
    fold.absorb_user_event(prompt)
    fold.note_user_message("also check X", steer=True, message_id="s1")
    calls = _tool_round(fold, "t1", "t2")
    steer = Message.user("also check X", id="s1")
    fold.absorb_user_event(steer)
    assistant = Message(role="assistant", content=[])
    assistant.tool_calls = calls
    journal: list[AgentMessage] = [
        prompt,
        assistant,
        *[Message(role="tool", tool_call_id=c.id, content=[TextContent(text="ok")]) for c in calls],
        steer,
    ]
    return fold, journal


def _content_keys(rows: list[Any]) -> list[str]:
    # The assistant row's id is the provider's on one side and the fold's on
    # the other; this test is about where the steer sits relative to the tools.
    return [_key(r) for r in rows if (r["kind"] if isinstance(r, dict) else r.kind) != "assistant"]


def test_declaring_client_live_order_equals_the_history_page_order() -> None:
    fold, journal = _steered_fold()
    frame = steer_journal_order_frame(_projection_frame(fold.projection), declared=True)
    page = fold_messages_to_entries(journal)
    assert _content_keys(frame["transcript"]) == _content_keys(page)
    assert _content_keys(frame["transcript"]) == ["user:u1", "tool:t1", "tool:t2", "user:s1"]
    # The map is a relay-internal hint, never part of what a client holds.
    assert "steer_journal_order" not in frame


def test_non_declaring_client_frame_is_the_legacy_frame() -> None:
    fold, _journal = _steered_fold()
    shared = _projection_frame(fold.projection)
    legacy = steer_journal_order_frame(shared, declared=False)
    # Send-time order and the ``steer`` kind, exactly as before the capability.
    assert _content_keys(legacy["transcript"]) == ["user:u1", "steer:s1", "tool:t1", "tool:t2"]
    assert "steer_journal_order" not in legacy
    # Byte identity: the legacy frame is the shared frame minus the hint key,
    # with every other key in its original position.
    expected = {k: v for k, v in shared.items() if k != "steer_journal_order"}
    assert json.dumps(legacy) == json.dumps(expected)
    # The shared frame (every subscriber's object) is never mutated.
    assert "steer_journal_order" in shared
    assert _content_keys(shared["transcript"]) == ["user:u1", "steer:s1", "tool:t1", "tool:t2"]


def test_a_frame_with_no_delivered_steer_is_returned_untouched() -> None:
    fold = ProjectionFold(SessionProjection(session_id=SESSION_ID, pid=0))
    fold.absorb_user_event(Message.user("hello", id="u1"))
    _tool_round(fold, "t1")
    frame = _projection_frame(fold.projection)
    assert "steer_journal_order" not in frame
    # Identity, not equality: the common case costs nothing per client.
    assert steer_journal_order_frame(frame, declared=True) is frame
    assert steer_journal_order_frame(frame, declared=False) is frame


def test_repeated_delivery_is_idempotent() -> None:
    fold, _journal = _steered_fold()
    before = _projection_frame(fold.projection)
    _tool_round(fold, "t3")
    # A redelivered event (a relayed stream, a replayed seed) for the same id.
    fold.absorb_user_event(Message.user("also check X", id="s1"))
    fold.absorb_user_event(Message.user("also check X", id="s1"))
    after = steer_journal_order_frame(_projection_frame(fold.projection), declared=True)
    # The steer keeps the slot of its FIRST delivery, never re-anchored past t3.
    assert _content_keys(after["transcript"]) == [
        "user:u1",
        "tool:t1",
        "tool:t2",
        "user:s1",
        "tool:t3",
    ]
    assert fold.projection.steer_journal_order == before["steer_journal_order"]


def test_steer_with_no_send_time_row_is_unchanged() -> None:
    """A steer the phone did not echo (another surface, or no id) is appended
    at delivery — already its journal slot — so there is nothing to move."""
    fold = ProjectionFold(SessionProjection(session_id=SESSION_ID, pid=0))
    fold.absorb_user_event(Message.user("do the thing", id="u1"))
    _tool_round(fold, "t1", "t2")
    fold.absorb_user_event(Message.user("from the tui", id="s9"))
    frame = _projection_frame(fold.projection)
    assert "steer_journal_order" not in frame
    assert steer_journal_order_frame(frame, declared=True) is frame
    assert _content_keys(frame["transcript"]) == ["user:u1", "tool:t1", "tool:t2", "user:s9"]


def test_two_steers_drained_at_one_boundary_keep_send_order() -> None:
    fold = ProjectionFold(SessionProjection(session_id=SESSION_ID, pid=0))
    fold.absorb_user_event(Message.user("go", id="u1"))
    fold.note_user_message("first", steer=True, message_id="s1")
    fold.note_user_message("second", steer=True, message_id="s2")
    _tool_round(fold, "t1")
    fold.absorb_user_event(Message.user("first", id="s1"))
    fold.absorb_user_event(Message.user("second", id="s2"))
    frame = steer_journal_order_frame(_projection_frame(fold.projection), declared=True)
    assert _content_keys(frame["transcript"]) == ["user:u1", "tool:t1", "user:s1", "user:s2"]


def test_steers_delivered_at_successive_boundaries_interleave_like_the_journal() -> None:
    fold = ProjectionFold(SessionProjection(session_id=SESSION_ID, pid=0))
    fold.absorb_user_event(Message.user("go", id="u1"))
    fold.note_user_message("first", steer=True, message_id="s1")
    _tool_round(fold, "t1")
    fold.absorb_user_event(Message.user("first", id="s1"))
    fold.note_user_message("second", steer=True, message_id="s2")
    _tool_round(fold, "t2")
    # Undelivered s2 stays where it was painted for every client.
    pending = steer_journal_order_frame(_projection_frame(fold.projection), declared=True)
    assert _content_keys(pending["transcript"]) == [
        "user:u1",
        "tool:t1",
        "user:s1",
        "steer:s2",
        "tool:t2",
    ]
    fold.absorb_user_event(Message.user("second", id="s2"))
    frame = steer_journal_order_frame(_projection_frame(fold.projection), declared=True)
    assert _content_keys(frame["transcript"]) == [
        "user:u1",
        "tool:t1",
        "user:s1",
        "tool:t2",
        "user:s2",
    ]


def test_the_hint_survives_the_runtime_to_relay_hop() -> None:
    """The fold lives in the runtime and the reorder in the relay: the map has
    to ride the socket payload and come back through ``_projection_from_json``."""
    from local_operator.mobile.types import _projection_from_json

    fold, _journal = _steered_fold()
    record = _record()
    rebuilt = _projection_from_json(json.loads(json.dumps(fold.projection.to_json())), record)
    assert rebuilt.steer_journal_order == fold.projection.steer_journal_order != {}


def test_reorder_cost_is_linear_on_a_large_transcript() -> None:
    """O(n): one pass to index, one to emit. Measured at two sizes; a quadratic
    walk would scale ~10x worse between them than a linear one."""

    def frame_of(n: int) -> dict[str, Any]:
        rows = [{"id": f"r{i}", "kind": "assistant", "text": "x"} for i in range(n)]
        moves = {}
        for i in range(0, n, 10):
            rows[i] = {"id": f"s{i}", "kind": "steer", "text": "s"}
            moves[f"s{i}"] = f"r{min(i + 5, n - 1)}"
        return {"transcript": rows, "steer_journal_order": moves}

    def best(n: int) -> float:
        frame = frame_of(n)
        samples = []
        for _ in range(5):
            start = time.perf_counter()
            steer_journal_order_frame(frame, declared=True)
            samples.append(time.perf_counter() - start)
        return min(samples)

    small, large = best(2_000), best(20_000)
    assert len(steer_journal_order_frame(frame_of(20_000), declared=True)["transcript"]) == 20_000
    assert large / small < 30, (small, large)


# --- the stream: declaration, capability, byte identity --------------------


def _events_request(query: bytes) -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": f"/api/sessions/{SESSION_ID}/events",
            "path_params": {"session_id": SESSION_ID},
            "query_string": query,
            "headers": [
                (b"host", b"fixture"),
                (b"cookie", f"{COOKIE_NAME}={sign_cookie(PASSWORD)}".encode()),
            ],
            "scheme": "http",
            "server": ("fixture", 80),
        }
    )


async def _seed(app: Any, query: bytes) -> dict[str, Any]:
    endpoint = next(r.endpoint for r in app.routes if getattr(r, "path", "") == SESSION_ROUTE)
    response = await endpoint(_events_request(query))
    try:
        event = await anext(response.body_iterator)
    finally:
        await response.body_iterator.aclose()
    data = next(line for line in event.splitlines() if line.startswith("data:"))
    return json.loads(data[len("data:") :].strip())


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cfg = tmp_path / "config"
    cfg.mkdir()
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: cfg)
    return cfg


def _live_daemon() -> MobileDaemon:
    daemon = MobileDaemon(port=0, password=PASSWORD, dial_registrants=False)
    fold, _journal = _steered_fold()
    entry = SessionEntry(_record())
    entry.projection = fold.projection
    daemon.session_projections[SESSION_ID] = fold.projection
    daemon.table.entries[PID] = entry
    return daemon


@pytest.mark.asyncio
async def test_stream_reorders_only_for_a_declaring_client(config: Path) -> None:
    daemon = _live_daemon()
    app = build_app(daemon)
    try:
        legacy = await _seed(app, b"")
        declared = await _seed(app, b"steer_journal_order=1")
        # An unrecognised value is not a declaration (a typo must not opt in).
        other = await _seed(app, b"steer_journal_order=0")
    finally:
        await daemon.close_phone_views()
    assert _content_keys(legacy["transcript"]) == ["user:u1", "steer:s1", "tool:t1", "tool:t2"]
    assert _content_keys(declared["transcript"]) == ["user:u1", "tool:t1", "tool:t2", "user:s1"]
    assert _content_keys(other["transcript"]) == _content_keys(legacy["transcript"])
    for frame in (legacy, declared, other):
        assert "steer_journal_order" not in frame


def test_legacy_seed_event_is_byte_identical_to_the_pre_capability_event() -> None:
    """The legacy SSE event equals ``_sse`` of the frame with the hint removed —
    i.e. exactly what a relay without the capability serialised."""
    fold, _journal = _steered_fold()
    shared = _projection_frame(fold.projection)
    pre = {k: v for k, v in shared.items() if k != "steer_journal_order"}
    assert _sse("projection", steer_journal_order_frame(shared, declared=False)) == _sse(
        "projection", pre
    )


def test_capability_is_advertised(config: Path) -> None:
    client = TestClient(
        build_app(MobileDaemon(port=0, password=PASSWORD, dial_registrants=False)),
        follow_redirects=False,
    )
    client.post("/login", data={"password": PASSWORD})
    capabilities = client.get("/api/sessions").json()["capabilities"]
    assert capabilities["steer_journal_order"] == {"available": True}


def test_fold_reset_drops_the_hint() -> None:
    """A resumed fold replaces the transcript from the journal (already in
    journal order); a stale map must not survive it."""
    fold, journal = _steered_fold()
    assert fold.projection.steer_journal_order
    fold.fold_history(journal)
    assert fold.projection.steer_journal_order == {}
