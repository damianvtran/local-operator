"""The inline-UI head row is a property of the CONVERSATION on a thinking wire.

WHY: with thinking on, ``AnthropicClient._build_body`` pins the request-only
inline-UI discovery hint at ``messages[0]`` (PR #4582), and Sonnet/Opus/Haiku
5.5 bind every signed thinking block to ALL content before it. The live hint is
decided per run and per call — a wake run has none, a TUI steer turns it on, a
live answer from another surface turns it off, an aside never asked for one —
so following it adds or drops the head row under blocks already signed: HTTP 400
("Content that preceded this block when it was created is missing") and a miss
of the whole message-level cache. ``Session._settle_inline_ui_head_latch``
latches it instead. These tests drive real sessions through the real Anthropic
stream parser and body builder (an ``httpx.MockTransport``, no network) and
require every request's messages to extend the previous request's messages
byte-for-byte, cache markers aside.

Placing the hint at the head while still following the live value per request
breaks each request-sequence test here; the compaction tests pin the one point
where the latch may legitimately change.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from local_operator.harness.types import (
    AgentTool,
    ChatRequest,
    CustomMessage,
    Message,
    ModelSpec,
    TextContent,
    ToolResult,
)
from local_operator.harness.wake import WAKE_PROMPT_MESSAGE_TYPE
from local_operator.providers.clients import AnthropicClient
from local_operator.session.inline_ui import INSTRUCTIONS
from local_operator.session.session import Session
from local_operator.session.transcript import (
    INLINE_UI_HEAD_LATCH_CUSTOM_TYPE,
    Transcript,
)
from tests.unit.providers.test_anthropic_thinking_replay import (
    KEY,
    _signed_tool_step,
    _sse,
    _without_cache_control,
)

THINKING_SPEC = ModelSpec(
    provider="anthropic",
    model_id="claude-sonnet-5-5",
    context_window=200_000,
    reasoning_efforts=("low", "medium", "high"),
    reasoning_effort="medium",
)
HEAD_ROW = {"role": "user", "content": [{"type": "text", "text": INSTRUCTIONS}]}


def _signed_answer(step: int) -> bytes:
    """A thinking + text answer that ends the run."""
    return _sse(
        [
            ("message_start", {"message": {"id": f"msg_a{step}", "usage": {"input_tokens": 9}}}),
            (
                "content_block_start",
                {
                    "index": 0,
                    "content_block": {"type": "thinking", "thinking": "", "signature": ""},
                },
            ),
            (
                "content_block_delta",
                {"index": 0, "delta": {"type": "thinking_delta", "thinking": f"wrap {step}"}},
            ),
            (
                "content_block_delta",
                {"index": 0, "delta": {"type": "signature_delta", "signature": f"sig-a{step}"}},
            ),
            ("content_block_stop", {"index": 0}),
            ("content_block_start", {"index": 1, "content_block": {"type": "text", "text": ""}}),
            (
                "content_block_delta",
                {"index": 1, "delta": {"type": "text_delta", "text": f"done {step}."}},
            ),
            ("content_block_stop", {"index": 1}),
            (
                "message_delta",
                {"delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5}},
            ),
            ("message_stop", {}),
        ]
    )


class AnthropicWire:
    """A session ``stream_fn`` that runs the real Anthropic client per call.

    Records the JSON body actually posted, so assertions read the wire, not the
    ``ChatRequest``.
    """

    def __init__(self, scripts: list[bytes]) -> None:
        self.scripts = scripts
        self.requests: list[ChatRequest] = []
        self.bodies: list[dict[str, Any]] = []

    def __call__(self, request: ChatRequest, signal: Any = None):
        self.requests.append(request)
        script = self.scripts[len(self.requests) - 1]

        async def gen():
            def handler(http_request: httpx.Request) -> httpx.Response:
                self.bodies.append(json.loads(http_request.content))
                return httpx.Response(
                    200, content=script, headers={"content-type": "text/event-stream"}
                )

            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
                async for event in AnthropicClient(http_client=http).stream(request, KEY):
                    yield event

        return gen()


def _bash(on_run=None) -> AgentTool:
    async def execute(tool_call_id, args, signal, on_update, context):
        if on_run is not None:
            on_run()
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


def _session(tmp_path, wire: AnthropicWire, tool: AgentTool | None = None) -> Session:
    return Session(
        model=THINKING_SPEC,
        stream_fn=wire,
        tools=[tool or _bash()],
        transcript=Transcript(tmp_path / "sess"),
        system_blocks_provider=lambda: ["stable", "env"],
    )


def _wake(text: str) -> CustomMessage:
    return CustomMessage(
        custom_type=WAKE_PROMPT_MESSAGE_TYPE,
        attribution="user",
        details={"text": text, "schedule_id": "w1"},
    )


def _assert_each_body_extends_the_last(bodies: list[dict[str, Any]]) -> None:
    assert all(body["thinking"] == {"type": "adaptive"} for body in bodies)
    for index, (earlier, later) in enumerate(zip(bodies, bodies[1:])):
        before = _without_cache_control(earlier["messages"])
        after = _without_cache_control(later["messages"])
        assert after[: len(before)] == before, f"request {index + 1} rewrote earlier turns"


@pytest.mark.asyncio
async def test_hint_turned_off_mid_run_keeps_the_head_row(tmp_path):
    """On→off: a live answer from another surface must not drop the head row."""
    holder: dict[str, Session] = {}

    def answered_elsewhere() -> None:
        # The control server's live-answer path (``note_answer_surface(None, …)``)
        # while the run streams: the run's live hint is now None.
        holder["session"].note_answer_surface(None, "remote")

    wire = AnthropicWire([_signed_tool_step(0), _signed_tool_step(1), _signed_answer(2)])
    session = _session(tmp_path, wire, _bash(answered_elsewhere))
    holder["session"] = session
    try:
        await session.prompt("go")
    finally:
        await session.dispose()

    assert len(wire.bodies) == 3
    # The live answer relabelled the run, so its live hint is now None.
    assert session._run_input_surface == "remote"
    _assert_each_body_extends_the_last(wire.bodies)
    assert all(body["messages"][0] == HEAD_ROW for body in wire.bodies)


@pytest.mark.asyncio
async def test_hint_turned_on_mid_run_adds_no_head_row_under_signed_blocks(tmp_path):
    """Off→on: a TUI steer folding into a wake run must not insert a head row."""
    holder: dict[str, Session] = {}
    steered: list[bool] = []

    def person_steers() -> None:
        if not steered:
            steered.append(True)
            holder["session"].steer("also check the logs", input_surface="tui")

    wire = AnthropicWire([_signed_tool_step(0), _signed_tool_step(1), _signed_answer(2)])
    session = _session(tmp_path, wire, _bash(person_steers))
    holder["session"] = session
    try:
        await session._prompt_messages([_wake("the reminder you asked for")])
    finally:
        await session.dispose()

    assert len(wire.bodies) == 3
    # The steer DID turn the live hint on for the rest of the run, but signed
    # thinking was already in history, so the head row must stay absent.
    assert session._run_input_surface == "tui"
    _assert_each_body_extends_the_last(wire.bodies)
    assert all(body["messages"][0] != HEAD_ROW for body in wire.bodies)


@pytest.mark.asyncio
async def test_a_later_surfaceless_run_and_a_resume_keep_the_latched_head(tmp_path):
    """Across runs: a wake run (no surface) and a restarted runtime resend the row."""
    wire = AnthropicWire(
        [_signed_tool_step(0), _signed_answer(1), _signed_answer(2), _signed_answer(3)]
    )
    session = _session(tmp_path, wire)
    try:
        await session.prompt("go")
        await session._prompt_messages([_wake("the reminder you asked for")])
    finally:
        await session.dispose()

    # A fresh runtime on the same transcript: the latch is restored from disk.
    resumed = _session(tmp_path, wire)
    try:
        await resumed._prompt_messages([_wake("and again")])
    finally:
        await resumed.dispose()

    assert len(wire.bodies) == 4
    _assert_each_body_extends_the_last(wire.bodies)
    assert all(body["messages"][0] == HEAD_ROW for body in wire.bodies)


@pytest.mark.asyncio
async def test_an_aside_sends_the_turns_head_row(tmp_path):
    """Asides replay the turn's signed blocks, so they need the same head."""
    wire = AnthropicWire([_signed_tool_step(0), _signed_answer(1), _signed_answer(2)])
    session = _session(tmp_path, wire)
    try:
        await session.prompt("go")
        answer = await session.complete_aside([Message.user("why?")])
    finally:
        await session.dispose()

    assert answer == "done 2."
    turn, aside = wire.bodies[1], wire.bodies[2]
    assert aside["thinking"] == {"type": "adaptive"}
    before = _without_cache_control(turn["messages"])
    after = _without_cache_control(aside["messages"])
    assert after[0] == HEAD_ROW
    assert after[: len(before)] == before


async def _latch_then_compact(tmp_path, wire: AnthropicWire) -> tuple[Session, str]:
    """A latched conversation (signed thinking in history) plus a new compaction.

    The marker is appended directly: what is under test is the settle's
    decision at a NEW compaction boundary, not the compactor's choice of what
    to keep, which each test then states by setting the kept messages.
    """
    session = _session(tmp_path, wire)
    await session.prompt("go")
    assert session._inline_ui_head_latch == INSTRUCTIONS
    assert session._inline_ui_head_latch_compaction_id is None
    first = session._transcript.entries()[0]
    marker = await session._transcript.append_compaction(
        summary="SUMMARY", first_kept_entry_id=first.id, tokens_before=10
    )
    # A later run with no surface (a wake): the live hint is now None, so a
    # re-decision would visibly change the head row.
    session._run_input_surface = "remote"
    assert session._live_inline_ui_hint() is None
    return session, marker.id


@pytest.mark.asyncio
async def test_compaction_keeping_signed_thinking_holds_the_latch(tmp_path):
    """Turns kept verbatim still carry blocks signed under the old head row."""
    wire = AnthropicWire([_signed_tool_step(0), _signed_answer(1)])
    session, _ = await _latch_then_compact(tmp_path, wire)
    try:
        await session._settle_inline_ui_head_latch(THINKING_SPEC)
    finally:
        await session.dispose()

    assert session._inline_ui_head_latch == INSTRUCTIONS
    assert session._inline_ui_head_latch_compaction_id is None
    # Held, not rewritten: still the one row the first call latched.
    latch_rows = [
        row
        for row in Transcript(tmp_path / "sess").entries()
        if row.payload.get("custom_type") == INLINE_UI_HEAD_LATCH_CUSTOM_TYPE
    ]
    assert [row.payload["details"] for row in latch_rows] == [
        {"text": INSTRUCTIONS, "compaction_id": None}
    ]


@pytest.mark.asyncio
async def test_compaction_dropping_all_signed_thinking_redecides_the_latch(tmp_path):
    """Nothing signed survives the marker, so the live value is adopted, durably."""
    wire = AnthropicWire([_signed_tool_step(0), _signed_answer(1)])
    session, marker_id = await _latch_then_compact(tmp_path, wire)
    try:
        # The compactor kept only unsigned rows (the summary and a user turn).
        session._context.messages = [Message.user("go")]
        await session._settle_inline_ui_head_latch(THINKING_SPEC)
    finally:
        await session.dispose()

    assert session._inline_ui_head_latch is None
    assert session._inline_ui_head_latch_compaction_id == marker_id
    # Durable, so a resumed runtime sends the same (now absent) head row.
    reloaded = Transcript(tmp_path / "sess")
    assert reloaded.latest_custom(INLINE_UI_HEAD_LATCH_CUSTOM_TYPE) == {
        "text": None,
        "compaction_id": marker_id,
    }


@pytest.mark.asyncio
async def test_head_row_follows_the_requests_model_across_a_model_switch(tmp_path):
    """A ``/model`` landing after the settle must not move a thinking request's head.

    The loop snapshots the model before it settles the latch and builds the
    request with that snapshot; the hint must be placed for the same spec. Here
    the switch (to the same model, thinking off) lands while the request's
    history renders, on a wake run whose live hint is None: placed for the
    live ``self._model`` the hint is the live None, and the thinking request
    drops the head row its earlier signed blocks were bound to.
    """
    wire = AnthropicWire([_signed_tool_step(0), _signed_answer(1), _signed_answer(2)])
    session = _session(tmp_path, wire)
    try:
        await session.prompt("go")
        render = session._render_for_request

        def switch_mid_build(messages):
            session._model = THINKING_SPEC.model_copy(update={"reasoning_effort": None})
            return render(messages)

        session._render_for_request = switch_mid_build  # type: ignore[method-assign]
        await session._prompt_messages([_wake("the reminder you asked for")])
    finally:
        await session.dispose()

    assert len(wire.bodies) == 3
    # The request was still built for the thinking snapshot.
    assert wire.requests[2].model.reasoning_effort == "medium"
    _assert_each_body_extends_the_last(wire.bodies)
    assert all(body["messages"][0] == HEAD_ROW for body in wire.bodies)
