"""The cache keep-alive through a real ``Session`` and the real Anthropic wire.

WHY: the ping only pays if it re-reads the SAME cached prefix the next turn
sends. These tests drive real sessions through ``AnthropicClient`` (an
``httpx.MockTransport``, no network; the harness from
``test_inline_ui_head_latch``) and compare the posted JSON bodies: the ping's
``tools``, ``system`` and ``messages[:k]`` (``k`` = the last real request's
message count, the span its tail marker cached) must equal the next turn's,
with thinking on, with and without a session-state delta, and from a runtime
relaunched on the same transcript (the production case: the supervisor boots
one for the row). They also pin what the ping must NOT do — join the
transcript, emit events, move ``_last_usage`` — and the row's lifecycle through
the real wake pump: armed at turn end, replaced (never duplicated), re-armed
from the ping, retired on a provider error.
"""

from __future__ import annotations

import json
import time
from typing import Any

import httpx
import pytest

from local_operator.compaction import keepalive
from local_operator.harness.types import AgentTool, ChatRequest, TextContent, ToolResult
from local_operator.harness.wake_types import WakeSchedule
from local_operator.providers.clients import AnthropicClient
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
from local_operator.wakes.store import is_cache_keepalive_row
from tests.unit.providers.test_anthropic_thinking_replay import (
    KEY,
    _sse,
    _without_cache_control,
)
from tests.unit.session.test_inline_ui_head_latch import THINKING_SPEC
from tests.unit.session.test_session import wait_for

CONTEXT_READ = 120_000
TAIL_WRITE = 400


def _usage(read: int, write_1h: int) -> dict[str, Any]:
    return {
        "input_tokens": 9,
        "cache_read_input_tokens": read,
        "cache_creation_input_tokens": write_1h,
        "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": write_1h},
    }


def _signed(step: int, *, read: int, write_1h: int, tool: bool) -> bytes:
    """A signed-thinking step: a ``bash`` call, or a text answer that ends the run."""
    second: list[tuple[str, dict[str, Any]]]
    if tool:
        second = [
            (
                "content_block_start",
                {
                    "index": 1,
                    "content_block": {"type": "tool_use", "id": f"tu_{step}", "name": "bash"},
                },
            ),
            (
                "content_block_delta",
                {
                    "index": 1,
                    "delta": {"type": "input_json_delta", "partial_json": f'{{"cmd": "s{step}"}}'},
                },
            ),
        ]
    else:
        second = [
            ("content_block_start", {"index": 1, "content_block": {"type": "text", "text": ""}}),
            (
                "content_block_delta",
                {"index": 1, "delta": {"type": "text_delta", "text": f"done {step}."}},
            ),
        ]
    return _sse(
        [
            ("message_start", {"message": {"id": f"msg_{step}", "usage": _usage(read, write_1h)}}),
            (
                "content_block_start",
                {
                    "index": 0,
                    "content_block": {"type": "thinking", "thinking": "", "signature": ""},
                },
            ),
            (
                "content_block_delta",
                {"index": 0, "delta": {"type": "thinking_delta", "thinking": f"plan {step}"}},
            ),
            (
                "content_block_delta",
                {"index": 0, "delta": {"type": "signature_delta", "signature": f"sig-{step}"}},
            ),
            ("content_block_stop", {"index": 0}),
            *second,
            ("content_block_stop", {"index": 1}),
            (
                "message_delta",
                {
                    "delta": {"stop_reason": "tool_use" if tool else "end_turn"},
                    "usage": {"output_tokens": 5},
                },
            ),
            ("message_stop", {}),
        ]
    )


class Wire:
    """A session ``stream_fn`` over the real Anthropic client; records bodies.

    A script that is an ``int`` answers that HTTP status instead (a 429).
    """

    def __init__(self, scripts: list[bytes | int]) -> None:
        self.scripts = scripts
        self.requests: list[ChatRequest] = []
        self.bodies: list[dict[str, Any]] = []

    def __call__(self, request: ChatRequest, signal: Any = None):
        self.requests.append(request)
        script = self.scripts[len(self.requests) - 1]

        async def gen():
            def handler(http_request: httpx.Request) -> httpx.Response:
                self.bodies.append(json.loads(http_request.content))
                if isinstance(script, int):
                    return httpx.Response(
                        script,
                        json={"type": "error", "error": {"type": "rate_limit_error"}},
                    )
                return httpx.Response(
                    200, content=script, headers={"content-type": "text/event-stream"}
                )

            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
                async for event in AnthropicClient(http_client=http).stream(request, KEY):
                    yield event

        return gen()


def _bash() -> AgentTool:
    async def execute(tool_call_id, args, signal, on_update, context):
        return ToolResult(
            tool_call_id=tool_call_id,
            tool_name="bash",
            content=[TextContent(text=f"out {args.get('cmd')}")],
        )

    return AgentTool(
        name="bash",
        parameters={"type": "object", "properties": {"cmd": {"type": "string"}}},
        execute=execute,
    )


def _provider(state: dict[str, str]):
    def blocks(model_label: str = "") -> list[str]:
        return ["standing instructions", "tools", "environment", state["goal"]]

    setattr(blocks, "append_only_state", True)
    return blocks


def _session(tmp_path, wire: Wire, state: dict[str, str]) -> Session:
    return Session(
        model=THINKING_SPEC,
        stream_fn=wire,
        tools=[_bash()],
        transcript=Transcript(tmp_path / "sess"),
        system_blocks_provider=_provider(state),
    )


@pytest.fixture(autouse=True)
def _hers(monkeypatch: pytest.MonkeyPatch) -> None:
    # The default mode covers the chief of staff only; these sessions play her.
    # The ledger count is pinned too (its own tests read a real store).
    monkeypatch.setattr(Session, "_aida_is_hers", lambda self: True)
    monkeypatch.setattr(Session, "_keepalive_daily_count", lambda self: 0)


async def _drain(session: Session) -> None:
    await wait_for(lambda: all(task.done() for task in session._background_tasks), timeout=5.0)


def _keepalive_rows(session: Session) -> list[WakeSchedule]:
    return [row for row in session._wake.schedules if is_cache_keepalive_row(row)]


def _strip(body: dict[str, Any], key: str) -> Any:
    return _without_cache_control(body.get(key))


TURN = [
    _signed(0, read=0, write_1h=CONTEXT_READ, tool=True),
    _signed(1, read=CONTEXT_READ, write_1h=TAIL_WRITE, tool=False),
]
PING = _signed(2, read=CONTEXT_READ + TAIL_WRITE, write_1h=60, tool=False)
NEXT = _signed(3, read=CONTEXT_READ + TAIL_WRITE, write_1h=80, tool=False)


@pytest.mark.asyncio
@pytest.mark.parametrize("state_delta", [False, True])
@pytest.mark.parametrize("relaunch", [False, True])
async def test_the_ping_re_reads_the_prefix_the_next_turn_sends(
    tmp_path, state_delta: bool, relaunch: bool
) -> None:
    state = {"goal": "watch the deploy"}
    wire = Wire([*TURN, PING, NEXT])
    session = _session(tmp_path, wire, state)
    try:
        await session.prompt("go")
        await _drain(session)
        (row,) = _keepalive_rows(session)
        if relaunch:
            # The production path: the runtime idle-exits, and the supervisor
            # boots a fresh one on the same transcript for the overdue row.
            await session.dispose()
            session = _session(tmp_path, wire, state)
            (row,) = _keepalive_rows(session)
        if state_delta:
            # Rides the appended session-state record, after the shared history.
            state["goal"] = "watch the deploy and the alerts"
        entries_before = session._transcript.entries()
        messages_before = list(session._context.messages)
        usage_before = session._last_usage
        events: list[Any] = []
        session.subscribe(events.append)
        await session._run_cache_keepalive(row)
        await _drain(session)
        assert len(wire.bodies) == 3, "exactly one keep-alive request"
        ping_request = wire.requests[2]
        assert ping_request.purpose == "cache_keepalive"
        assert ping_request.same_route_retry is True
        assert ping_request.tool_choice == "none"
        assert ping_request.max_tokens == keepalive.KEEPALIVE_MAX_TOKENS
        # Joins nothing: no event, no conversation row, no usage adopted.
        assert events == []
        assert session._context.messages == messages_before
        assert session._last_usage is usage_before
        new_entries = session._transcript.entries()[len(entries_before) :]
        assert all(getattr(e, "type", "") == "custom" for e in new_entries)
        assert [e.payload.get("custom_type") for e in new_entries] == [
            "wake_schedules"
        ], "only the re-armed wake row is persisted"
        (rearmed,) = _keepalive_rows(session)
        assert rearmed.attempt == 1 and rearmed.id != row.id
        assert rearmed.cache_anchor_ms >= row.cache_anchor_ms

        await session.prompt("next")
    finally:
        await session.dispose()

    last_turn, ping, following = wire.bodies[1], wire.bodies[2], wire.bodies[3]
    k = len(last_turn["messages"])
    assert ping["thinking"] == following["thinking"] == last_turn["thinking"]
    assert ping.get("output_config") == following.get("output_config")
    assert _strip(ping, "tools") == _strip(following, "tools") == _strip(last_turn, "tools")
    assert _strip(ping, "system") == _strip(following, "system") == _strip(last_turn, "system")
    cached = _without_cache_control(last_turn["messages"])
    assert _without_cache_control(ping["messages"][:k]) == cached
    assert _without_cache_control(following["messages"][:k]) == cached


@pytest.mark.asyncio
async def test_turn_end_arms_one_row_and_replaces_it(tmp_path) -> None:
    state = {"goal": "g"}
    wire = Wire([*TURN, _signed(5, read=CONTEXT_READ, write_1h=TAIL_WRITE, tool=False)])
    session = _session(tmp_path, wire, state)
    try:
        await session.prompt("go")
        await _drain(session)
        (first,) = _keepalive_rows(session)
        assert first.hidden and first.kind == "cache_keepalive" and first.attempt == 0
        assert first.id == keepalive.row_id(first.cache_anchor_ms)
        assert first.next_due_at == first.cache_anchor_ms + 3_300_000
        assert first.prefix_digest == await session._keepalive_prefix_digest()
        await session.prompt("again")
        await _drain(session)
        (second,) = _keepalive_rows(session)
        assert second.id != first.id and second.cache_anchor_ms > first.cache_anchor_ms
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_the_pump_fires_pings_and_re_arms_without_losing_the_new_row(tmp_path) -> None:
    """Through the real pump: the fired row retires, the re-armed one survives."""
    wire = Wire([*TURN, PING])
    session = _session(tmp_path, wire, {"goal": "g"})
    try:
        await session.prompt("go")
        await _drain(session)
        (row,) = _keepalive_rows(session)
        fired = await session._wake.pump(now_ms=row.next_due_at)
        assert fired == 1
        await _drain(session)
        assert len(wire.bodies) == 3
        (rearmed,) = _keepalive_rows(session)
        assert rearmed.attempt == 1 and rearmed.id != row.id
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_rate_limited_ping_ends_the_chain(tmp_path) -> None:
    wire = Wire([*TURN, 429])
    session = _session(tmp_path, wire, {"goal": "g"})
    try:
        await session.prompt("go")
        await _drain(session)
        (row,) = _keepalive_rows(session)
        await session._wake.pump(now_ms=row.next_due_at)
        await _drain(session)
        assert len(wire.bodies) == 3, "one attempt, no retry or fallback"
        assert _keepalive_rows(session) == []
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_changed_prefix_sends_nothing(tmp_path) -> None:
    wire = Wire([*TURN])
    session = _session(tmp_path, wire, {"goal": "g"})
    try:
        await session.prompt("go")
        await _drain(session)
        (row,) = _keepalive_rows(session)
        await session._run_cache_keepalive(row.model_copy(update={"prefix_digest": "stale"}))
        await _drain(session)
        assert len(wire.bodies) == 2
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_late_fire_sends_nothing(tmp_path) -> None:
    wire = Wire([*TURN])
    session = _session(tmp_path, wire, {"goal": "g"})
    try:
        await session.prompt("go")
        await _drain(session)
        (row,) = _keepalive_rows(session)
        old = int(time.time() * 1000) - 3_600_000
        await session._run_cache_keepalive(row.model_copy(update={"cache_anchor_ms": old}))
        await _drain(session)
        assert len(wire.bodies) == 2
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_spent_chain_is_not_re_armed(tmp_path) -> None:
    wire = Wire([*TURN, PING])
    session = _session(tmp_path, wire, {"goal": "g"})
    try:
        await session.prompt("go")
        await _drain(session)
        (row,) = _keepalive_rows(session)
        await session._run_cache_keepalive(row.model_copy(update={"attempt": 3}))
        await _drain(session)
        assert len(wire.bodies) == 3, "the fourth ping is sent"
        assert _keepalive_rows(session) == [], "and nothing is armed after it"
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_mode_off_and_subagents_arm_nothing(tmp_path, monkeypatch) -> None:
    from local_operator.compaction.api import CompactionSettings

    wire = Wire([*TURN, *TURN])
    session = _session(tmp_path, wire, {"goal": "g"})
    try:
        session._compaction_settings = CompactionSettings(keepalive_mode="off")
        await session.prompt("go")
        await _drain(session)
        assert _keepalive_rows(session) == []
    finally:
        await session.dispose()
    child = _session(tmp_path / "child", wire, {"goal": "g"})
    try:
        child._parent_session_id = "parent0000ab"
        await child.prompt("go")
        await _drain(child)
        assert _keepalive_rows(child) == []
    finally:
        await child.dispose()


@pytest.mark.asyncio
async def test_a_5m_cache_arms_nothing(tmp_path) -> None:
    def five_minute(step: int, tool: bool) -> bytes:
        body = _signed(step, read=0, write_1h=0, tool=tool)
        return body.replace(
            b'"ephemeral_5m_input_tokens": 0', b'"ephemeral_5m_input_tokens": 500'
        ).replace(b'"cache_creation_input_tokens": 0', b'"cache_creation_input_tokens": 500')

    wire = Wire([five_minute(0, True), five_minute(1, False)])
    session = _session(tmp_path, wire, {"goal": "g"})
    try:
        await session.prompt("go")
        await _drain(session)
        assert session._idle_cache_ttl_s == 300.0
        assert _keepalive_rows(session) == []
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_economic_cold_next_counts_the_pings_a_turn_end_buys(tmp_path) -> None:
    """A gap the keep-alive bridges is not a deterministic miss."""
    from local_operator.compaction.api import CompactionSettings

    wire = Wire([*TURN])
    session = _session(tmp_path, wire, {"goal": "g"})
    try:
        await session.prompt("go")
        await _drain(session)
        two_hours = 2 * 3_600_000
        assert session._keepalive_remaining_pings() == 4
        # Expiry + 4 x 55 min > now + 2 h: warm, so no cold shortcut.
        assert session._economic_cold_next(two_hours) is False
        session._compaction_settings = CompactionSettings(keepalive_mode="off")
        assert session._keepalive_remaining_pings() == 0
        assert session._economic_cold_next(two_hours) is True
    finally:
        await session.dispose()
