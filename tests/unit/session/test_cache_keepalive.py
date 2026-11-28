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
from pathlib import Path
from typing import Any, Iterator

import httpx
import pytest

from local_operator.compaction import keepalive
from local_operator.harness.types import (
    AgentTool,
    ChatRequest,
    ModelSpec,
    TextContent,
    ToolResult,
)
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
    ``client_kw`` reaches ``AnthropicClient`` (the 1h size threshold);
    ``saver`` answers ``saver_may_reduce`` as ``SessionStreamFn`` does, whose
    real answer over a Saver-reduced turn ``tests/unit/saver/test_seam.py`` pins.
    """

    def __init__(
        self, scripts: list[bytes | int], *, client_kw: dict[str, Any] | None = None
    ) -> None:
        self.scripts = scripts
        self.client_kw = dict(client_kw or {})
        self.saver = False
        self.requests: list[ChatRequest] = []
        self.bodies: list[dict[str, Any]] = []

    def saver_may_reduce(self, model: Any) -> bool:
        return self.saver

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
                client = AnthropicClient(http_client=http, **self.client_kw)
                async for event in client.stream(request, KEY):
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


@pytest.fixture(autouse=True)
def _own_analytics_writer(tmp_path: Path) -> Iterator[None]:
    """Stop the process-wide analytics writer the provider path starts (QA Q-1).

    The same shape as ``tests/unit/tui/test_transcript_scroll_keys.py``: an
    isolated store for the module, and the ``lo-analytics-writer`` thread
    closed at teardown so the leak reporter does not pin it here.
    """
    from local_operator.analytics.recorder import reset_recorder_for_test
    from local_operator.analytics.store import AnalyticsStore

    recorder = reset_recorder_for_test(AnalyticsStore(tmp_path / "analytics-test.db"))
    yield
    recorder.close()


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
        assert (first.prefix_digest, first.prefix_message_count) == (
            await session._keepalive_prefix_digest()
        )
        assert first.prefix_message_count == len(session._context.messages)
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
@pytest.mark.parametrize("relaunch", [False, True])
async def test_a_peer_message_appended_after_the_arm_still_pings(tmp_path, relaunch: bool) -> None:
    """The regression: a quiet peer message between arm and fire is uncached tail.

    The chief of staff gets one every ~30 min without a turn; hashing the
    arm-time LAST message id refused every overnight fire as ``prefix_changed``.
    """
    wire = Wire([*TURN, PING, NEXT])
    session = _session(tmp_path, wire, {"goal": "g"})
    try:
        await session.prompt("go")
        await _drain(session)
        (row,) = _keepalive_rows(session)
        await session.receive_peer_message("status: all green", mode="mailbox")
        assert len(session._context.messages) == row.prefix_message_count + 1
        if relaunch:
            await session.dispose()
            session = _session(tmp_path, wire, {"goal": "g"})
            (row,) = _keepalive_rows(session)
            assert len(session._context.messages) == row.prefix_message_count + 1
        await session._run_cache_keepalive(row)
        await _drain(session)
        assert len(wire.bodies) == 3, "the ping was sent"
        assert wire.requests[2].purpose == "cache_keepalive"
        (rearmed,) = _keepalive_rows(session)
        assert rearmed.attempt == 1 and rearmed.id != row.id
        # The re-arm hashes the context it pinged, peer message included.
        assert rearmed.prefix_message_count == row.prefix_message_count + 1
        await session.prompt("next")
    finally:
        await session.dispose()

    last_turn, ping, following = wire.bodies[1], wire.bodies[2], wire.bodies[3]
    k = len(last_turn["messages"])
    assert _strip(ping, "tools") == _strip(following, "tools") == _strip(last_turn, "tools")
    assert _strip(ping, "system") == _strip(following, "system") == _strip(last_turn, "system")
    cached = _without_cache_control(last_turn["messages"])
    assert _without_cache_control(ping["messages"][:k]) == cached
    assert _without_cache_control(following["messages"][:k]) == cached
    # The peer message rides after the cached span in both, identically.
    assert "status: all green" in json.dumps(ping["messages"][k - 1 :])
    assert _without_cache_control(ping["messages"][: k + 1]) == _without_cache_control(
        following["messages"][: k + 1]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("rewrite", ["replace_head", "drop_middle", "truncate"])
async def test_history_rewritten_inside_the_recorded_span_sends_nothing(
    tmp_path, rewrite: str
) -> None:
    """Compaction (or anything) moving a message the arm hashed still refuses."""
    wire = Wire([*TURN])
    session = _session(tmp_path, wire, {"goal": "g"})
    try:
        await session.prompt("go")
        await _drain(session)
        (row,) = _keepalive_rows(session)
        messages = list(session._context.messages)
        assert len(messages) >= 3
        if rewrite == "replace_head":
            messages[0] = messages[0].model_copy(update={"id": "summary-epoch"})
        elif rewrite == "drop_middle":
            # Same length as the arm saw: a later message slid into the span.
            del messages[1]
            messages.append(messages[-1].model_copy(update={"id": "appended"}))
        else:
            del messages[-1]
        session._context.messages = messages
        await session._run_cache_keepalive(row)
        await _drain(session)
        assert len(wire.bodies) == 2, "no ping"
        assert _keepalive_rows(session) == [], "the chain is cleared"
    finally:
        await session.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("moved", ["system", "tools"])
async def test_a_changed_system_or_tool_array_sends_nothing(
    tmp_path, monkeypatch: pytest.MonkeyPatch, moved: str
) -> None:
    wire = Wire([*TURN])
    session = _session(tmp_path, wire, {"goal": "g"})
    try:
        await session.prompt("go")
        await _drain(session)
        (row,) = _keepalive_rows(session)
        if moved == "system":
            real = session._read_only_system_blocks

            async def moved_blocks() -> tuple[list[str], list[str]]:
                desired, sent = await real()
                return desired, ["a new epoch", *sent[1:]]

            monkeypatch.setattr(session, "_read_only_system_blocks", moved_blocks)
        else:
            tools = [*session._side_channel_tools(), _bash().model_copy(update={"name": "read"})]
            monkeypatch.setattr(session, "_side_channel_tools", lambda: tools)
        await session._run_cache_keepalive(row)
        await _drain(session)
        assert len(wire.bodies) == 2, "no ping"
        assert _keepalive_rows(session) == []
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


def _markers(node: Any) -> list[dict[str, Any]]:
    """Every ``cache_control`` dict in a posted body, in document order."""
    found: list[dict[str, Any]] = []
    if isinstance(node, dict):
        if isinstance(node.get("cache_control"), dict):
            found.append(node["cache_control"])
        for value in node.values():
            found.extend(_markers(value))
    elif isinstance(node, list):
        for value in node:
            found.extend(_markers(value))
    return found


@pytest.mark.asyncio
async def test_the_ping_writes_its_markers_at_1h_on_the_wire(tmp_path) -> None:
    """The ping's markers carry ``ttl: 1h`` in the body actually posted (review R1-2b).

    An API-key client with a 100k main-thread threshold: the ping's size hint is
    the restored conversation size (~120k), so the policy picks 1h, as it does
    for the turn request that wrote the entry.
    """
    wire = Wire([*TURN, PING], client_kw={"cache_ttl_1h_min_context_tokens": 100_000})
    session = _session(tmp_path, wire, {"goal": "g"})
    try:
        await session.prompt("go")
        await _drain(session)
        (row,) = _keepalive_rows(session)
        await session._run_cache_keepalive(row)
        await _drain(session)
    finally:
        await session.dispose()
    last_turn, ping = wire.bodies[1], wire.bodies[2]
    assert _markers(last_turn) and all(m.get("ttl") == "1h" for m in _markers(last_turn))
    ping_markers = _markers(ping)
    assert _markers(ping["system"]) and _markers(ping["messages"]), "system and history marked"
    assert ping_markers and all(m == {"type": "ephemeral", "ttl": "1h"} for m in ping_markers)


@pytest.mark.asyncio
async def test_a_ping_that_read_under_half_the_context_ends_the_chain(tmp_path) -> None:
    """Review R1-2a: a ping that did not reuse the prefix is sent, and nothing follows."""
    cold = _signed(2, read=CONTEXT_READ // 2 - 1, write_1h=CONTEXT_READ, tool=False)
    wire = Wire([*TURN, cold])
    session = _session(tmp_path, wire, {"goal": "g"})
    try:
        await session.prompt("go")
        await _drain(session)
        (row,) = _keepalive_rows(session)
        await session._wake.pump(now_ms=row.next_due_at)
        await _drain(session)
        assert len(wire.bodies) == 3, "the ping was sent"
        assert _keepalive_rows(session) == [], "and the chain ended without a re-arm"
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_radient_saver_arms_nothing_and_refuses_a_fire(tmp_path) -> None:
    """Review R1-1: while Saver reduces the turns, no row is armed and none fires."""
    wire = Wire([*TURN, *TURN])
    session = _session(tmp_path, wire, {"goal": "g"})
    try:
        await session.prompt("go")
        await _drain(session)
        (row,) = _keepalive_rows(session)
        # Fire side: Saver switched on between the arm and the fire.
        wire.saver = True
        await session._run_cache_keepalive(row)
        await _drain(session)
        assert len(wire.bodies) == 2, "no ping"
        assert _keepalive_rows(session) == [], "the refusal clears the chain"
        # Arm side: a turn that ends while Saver is on arms nothing.
        await session.prompt("again")
        await _drain(session)
        assert len(wire.bodies) == 4
        assert _keepalive_rows(session) == []
        assert session._keepalive_remaining_pings() == 0
    finally:
        await session.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("turn_row_first", [False, True])
async def test_a_turn_ending_during_the_ping_keeps_its_fresher_row(
    tmp_path, monkeypatch: pytest.MonkeyPatch, turn_row_first: bool
) -> None:
    """Review R1-3: the ping's late re-arm never replaces the row that turn armed."""
    wire = Wire([*TURN, PING, *TURN])
    session = _session(tmp_path, wire, {"goal": "g"})
    real_ping = Session._cache_keepalive_ping

    async def ping_then_a_turn(self: Session) -> Any:
        usage = await real_ping(self)
        # A message lands in the relaunched runtime while the ping streamed.
        await self.prompt("mid")
        if turn_row_first:
            await _drain(self)
        return usage

    try:
        await session.prompt("go")
        await _drain(session)
        (row,) = _keepalive_rows(session)
        monkeypatch.setattr(Session, "_cache_keepalive_ping", ping_then_a_turn)
        await session._run_cache_keepalive(row)
        await _drain(session)
        assert len(wire.bodies) == 5
        (kept,) = _keepalive_rows(session)
        assert kept.attempt == 0, "the turn's own row, not the ping's attempt=1"
        assert kept.cache_anchor_ms > row.cache_anchor_ms
        assert kept.cache_anchor_ms == session._idle_cache_anchor_ms
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_turn_whose_own_arm_is_refused_leaves_the_ping_nothing_to_re_arm(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review R2-1: the case ONLY the turn-end counter covers.

    The mid turn was served on a fallback route, so its own arm is refused
    (``route_mismatch``) and its turn-end write CLEARS the fired row instead of
    replacing it. The ping's own route does match, so its late write would
    re-arm a chain that turn up had just retired — the anchor comparison cannot
    see that, because the turn left no row to be newer than the ping's anchor.
    """
    wire = Wire([*TURN, PING, *TURN])
    session = _session(tmp_path, wire, {"goal": "g"})
    real_ping = Session._cache_keepalive_ping

    async def ping_then_a_fallback_turn(self: Session) -> Any:
        usage = await real_ping(self)
        # A message lands in the relaunched runtime while the ping streamed,
        # served by a pinned fallback (the ping's primary route then cannot see
        # that prefix, so the turn end arms nothing).
        self._active_fallback = ModelSpec(provider="openai", model_id="fallback-model")
        try:
            await self.prompt("mid")
            await _drain(self)
        finally:
            self._active_fallback = None
        return usage

    try:
        await session.prompt("go")
        await _drain(session)
        (row,) = _keepalive_rows(session)
        monkeypatch.setattr(Session, "_cache_keepalive_ping", ping_then_a_fallback_turn)
        await session._run_cache_keepalive(row)
        await _drain(session)
        assert _keepalive_rows(session) == [], "the ping does not re-arm a retired chain"
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_chain_end_clears_a_fired_row_a_stale_writer_restored(tmp_path) -> None:
    """Review R1-5: a refusal is an explicit clear, not a bare return."""
    wire = Wire([*TURN])
    session = _session(tmp_path, wire, {"goal": "g"})
    try:
        await session.prompt("go")
        await _drain(session)
        (row,) = _keepalive_rows(session)
        # The pump retired the fired row, then a full-list writer whose snapshot
        # predates that wrote it straight back.
        await session._wake.update([])
        await session._wake.update([row])
        await session._run_cache_keepalive(row.model_copy(update={"prefix_digest": "stale"}))
        await _drain(session)
        assert len(wire.bodies) == 2
        assert _keepalive_rows(session) == []
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_always_on_reads_the_sessions_own_root_once(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review R1-4: ``team_bridge.session`` from ``self._config_dir``, read once.

    Review R2-2: only a read that COMPLETED is cached. A failed one (the
    ``None`` ``read_config_values`` returns for a file it cannot read right now)
    covers nothing, and the next turn end asks again rather than keeping this
    runtime's keep-alive off for good.
    """
    import local_operator.config as config_mod
    from local_operator.compaction.api import CompactionSettings
    from local_operator.team_bridge.settings import SESSION_PATH

    wire = Wire([])
    session = _session(tmp_path, wire, {"goal": "g"})
    monkeypatch.setattr(Session, "_aida_is_hers", lambda self: False)
    own_root = tmp_path / "own-root"
    session._config_dir = own_root
    reads: list[Path] = []
    failing = {"left": 1}

    def read(root: Path) -> dict[str, Any] | None:
        reads.append(root)
        if failing["left"]:
            failing["left"] -= 1
            return None  # a torn/unreadable config.yml, config.py's own contract
        return {SESSION_PATH[0]: {SESSION_PATH[1]: session._session_id}}

    monkeypatch.setattr(config_mod, "read_config_values", read)
    try:
        session._compaction_settings = CompactionSettings(keepalive_mode="always_on")
        assert session._keepalive_settings() is None, "an unreadable read covers nothing"
        assert session._keepalive_settings() is not None, "and is retried, not cached"
        assert session._keepalive_settings() is not None
        assert reads == [own_root] * 2, "the session's own root, retried once after the failure"
        session._compaction_settings = CompactionSettings(keepalive_mode="chief_of_staff")
        assert session._keepalive_settings() is None
    finally:
        await session.dispose()
