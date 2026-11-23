"""The spoken-summary turn note: who is asked for it, and that nothing stores it.

Gate (docs/design/voice-spoken-summary.md, narrowed 2026-11-23): only a prompt
that was DICTATED, in the session that HOLDS the voice slot. A typed prompt, a
prompt in another session, a wake, a monitor and a harness-injected turn get
nothing. The note is appended to the request-local copy of the history before
each provider call and is never stored, so every surface that reads a row reads
exactly what the person said.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import (
    AgentEvent,
    ChatRequest,
    Message,
    MessageStartEvent,
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
    StreamToolCallDelta,
    TextContent,
    ToolCall,
)
from local_operator.harness.wake import DueWake, WakeSchedule
from local_operator.mobile.projection import fold_messages_to_entries
from local_operator.monitors.delivery import MonitorDelivery
from local_operator.providers.clients import (
    AnthropicClient,
    GoogleClient,
    OpenAICompatClient,
)
from local_operator.session.preview import condense_entries
from local_operator.voice import slot as voice_slot
from local_operator.voice.summary_note import VOICE_SUMMARY_NOTE
from tests.unit.session.test_session import (
    ScriptedStream,
    echo_tool,
    make_session,
    wait_for,
)

WORDS = "what changed today"


def _stops(count: int = 3) -> list[list[Any]]:
    return [[StreamEndEvent(stop_reason="stop")]] * count


def _session(
    tmp_path: Path, *, holds_slot: bool, turns: list[list[Any]] | None = None, **kwargs: Any
) -> Any:
    stream = ScriptedStream(turns if turns is not None else _stops())
    session = make_session(tmp_path, stream, **kwargs)
    session._config_dir = tmp_path / "config"
    if holds_slot:
        voice_slot.enable(session._config_dir, session._session_id)
    return session


def _first_user_text(session: Any) -> str:
    return next(m.text for m in session._context.messages if m.role == "user")


def _stored_entries(tmp_path: Path) -> list[dict[str, Any]]:
    path = tmp_path / "sess" / "transcript.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _texts(request: ChatRequest) -> list[str]:
    return [m.text for m in request.messages]


def _carries_note(session: Any) -> bool:
    return any(VOICE_SUMMARY_NOTE in _texts(request) for request in session._stream_fn.requests)


@pytest.mark.asyncio
async def test_a_dictated_prompt_in_the_voice_session_sends_the_note_last(tmp_path: Path) -> None:
    session = _session(tmp_path, holds_slot=True)
    await session.prompt(WORDS, input_mode="dictated", input_path="qwen_s2s")
    request = session._stream_fn.requests[0]
    assert _texts(request)[-2:] == [WORDS, VOICE_SUMMARY_NOTE]
    assert request.messages[-1].role == "user"
    await session.dispose()


@pytest.mark.asyncio
async def test_the_note_is_never_stored_and_no_surface_shows_it(tmp_path: Path) -> None:
    """M1/M2/Q1/Q2: the row, the preview, the desktop/phone history fold and the live
    event the TUI paints all read exactly the user's words."""
    session = _session(tmp_path, holds_slot=True)
    events: list[AgentEvent] = []
    session.subscribe(events.append)
    await session.prompt(WORDS, input_mode="dictated", input_path="qwen_s2s")
    assert _carries_note(session)  # sent to the provider...

    assert _first_user_text(session) == WORDS  # ...but not stored in context,
    entries = _stored_entries(tmp_path)
    assert VOICE_SUMMARY_NOTE not in json.dumps(entries)  # not in the transcript file,
    assert [t.text for t in condense_entries(entries) if t.role == "user"] == [WORDS]  # preview,
    rows = fold_messages_to_entries(list(session._context.messages))
    assert [row.text for row in rows if row.kind == "user"] == [WORDS]  # desktop history fold,
    painted = [
        e.message.text
        for e in events
        if isinstance(e, MessageStartEvent)
        and isinstance(e.message, Message)
        and e.message.role == "user"
    ]
    assert painted == [WORDS]  # the live row the TUI paints.
    assert session._voice_note_for is None  # and the turn disarms itself
    await session.dispose()


@pytest.mark.asyncio
async def test_every_provider_call_of_the_turn_ends_with_the_note(tmp_path: Path) -> None:
    executed: list[str] = []
    turns = [
        [
            StreamToolCallDelta(index=0, id="c1", name="echo", argument_delta="{}"),
            StreamEndEvent(stop_reason="toolUse"),
        ],
        [StreamTextDelta(delta="done"), StreamEndEvent(stop_reason="stop")],
    ]
    session = _session(tmp_path, holds_slot=True, turns=turns, tools=[echo_tool(executed)])
    await session.prompt(WORDS, input_mode="dictated")
    first, second = session._stream_fn.requests
    assert _texts(first)[-1] == VOICE_SUMMARY_NOTE
    assert [m.role for m in second.messages][-3:] == ["assistant", "tool", "user"]
    assert _texts(second)[-1] == VOICE_SUMMARY_NOTE
    assert _texts(second).count(VOICE_SUMMARY_NOTE) == 1
    await session.dispose()


@pytest.mark.asyncio
async def test_a_steer_during_the_turn_drops_the_note(tmp_path: Path) -> None:
    """A newer user message displaces the dictated one, and the note with it."""
    holder: dict[str, Any] = {}
    steering = echo_tool([])
    inner = steering.execute

    async def steer_then_run(*args: Any) -> Any:
        holder["session"].steer("typed steer")
        return await inner(*args)

    steering.execute = steer_then_run
    turns = [
        [
            StreamToolCallDelta(index=0, id="c1", name="echo", argument_delta="{}"),
            StreamEndEvent(stop_reason="toolUse"),
        ],
        [StreamTextDelta(delta="done"), StreamEndEvent(stop_reason="stop")],
    ]
    session = _session(tmp_path, holds_slot=True, turns=turns, tools=[steering])
    holder["session"] = session
    await session.prompt(WORDS, input_mode="dictated")
    first, second = session._stream_fn.requests
    assert _texts(first)[-1] == VOICE_SUMMARY_NOTE
    assert "typed steer" in _texts(second)
    assert VOICE_SUMMARY_NOTE not in _texts(second)
    await session.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [None, "typed", "mixed"])
async def test_a_prompt_that_was_not_dictated_gets_no_note(tmp_path: Path, mode: Any) -> None:
    session = _session(tmp_path, holds_slot=True)
    await session.prompt("typed words", input_mode=mode)
    assert _first_user_text(session) == "typed words"
    assert not _carries_note(session)
    await session.dispose()


@pytest.mark.asyncio
async def test_a_dictated_prompt_without_the_slot_gets_no_note(tmp_path: Path) -> None:
    session = _session(tmp_path, holds_slot=False)
    await session.prompt("spoken but voice is off", input_mode="dictated")
    assert not _carries_note(session)
    await session.dispose()


@pytest.mark.asyncio
async def test_the_negative_control_a_forced_gate_does_send_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cases above only mean something if a broken gate would fail them."""
    session = _session(tmp_path, holds_slot=False)
    monkeypatch.setattr(voice_slot, "is_voice_session", lambda *_args: True)
    await session.prompt("spoken but voice is off", input_mode="dictated")
    assert _carries_note(session)
    await session.dispose()


@pytest.mark.asyncio
async def test_a_dictated_prompt_while_another_session_holds_the_slot_gets_no_note(
    tmp_path: Path,
) -> None:
    session = _session(tmp_path, holds_slot=False)
    voice_slot.enable(session._config_dir, "some-other-session")
    await session.prompt("not my slot", input_mode="dictated")
    assert not _carries_note(session)
    await session.dispose()


@pytest.mark.asyncio
async def test_a_harness_injected_dictated_turn_gets_no_note(tmp_path: Path) -> None:
    session = _session(tmp_path, holds_slot=True)
    await session.prompt("chrome", input_mode="dictated", harness_injected=True)
    assert _first_user_text(session) == "chrome"
    assert not _carries_note(session)
    await session.dispose()


@pytest.mark.asyncio
async def test_a_wake_after_a_dictated_turn_gets_no_note(tmp_path: Path) -> None:
    session = _session(tmp_path, holds_slot=True)
    await session.prompt(WORDS, input_mode="dictated")
    sent = len(session._stream_fn.requests)
    # Past the operator-quiet window, so the wake opens its own turn instead of parking.
    session._last_operator_turn_end = None
    schedule = WakeSchedule(id="w1", message="check the build", next_due_at=0, created_at=0)
    await session._deliver_wake(
        DueWake(schedule=schedule, occurrence=1, planned_total=1, final=True)
    )
    await wait_for(lambda: len(session._stream_fn.requests) > sent)
    await session.dispose()
    # The dictated message is still the newest ``Message`` user turn in context, so
    # this only holds because the dictated turn disarmed the note when it ended.
    assert all(VOICE_SUMMARY_NOTE not in _texts(r) for r in session._stream_fn.requests[sent:])


@pytest.mark.asyncio
async def test_a_monitor_delivery_after_a_dictated_turn_gets_no_note(tmp_path: Path) -> None:
    session = _session(tmp_path, holds_slot=True)
    await session.prompt(WORDS, input_mode="dictated")
    sent = len(session._stream_fn.requests)
    await session._deliver_monitor(
        MonitorDelivery(
            monitor_id="m1",
            name="watch",
            tool="bash",
            changes=1,
            checks=2,
            skipped=0,
            delta_text="+1/-0 changed lines\n+ x",
            at_ms=1_756_000_000_000,
        )
    )
    await wait_for(lambda: len(session._stream_fn.requests) > sent)
    await session.dispose()
    assert all(VOICE_SUMMARY_NOTE not in _texts(r) for r in session._stream_fn.requests[sent:])


@pytest.mark.asyncio
async def test_a_subagent_style_prompt_in_a_session_without_the_slot_gets_no_note(
    tmp_path: Path,
) -> None:
    """A child session has its own id, which never holds the slot, whatever it is told."""
    child = _session(tmp_path, holds_slot=False)
    voice_slot.enable(child._config_dir, "the-parent-session")
    await child.prompt("delegated task", input_mode="dictated", harness_injected=False)
    assert not _carries_note(child)
    await child.dispose()


@pytest.mark.asyncio
async def test_a_resumed_turn_has_no_id_and_so_no_note(tmp_path: Path) -> None:
    """The armed id is in memory only: a session resumed mid-turn asks for nothing
    and the reply simply falls back to being spoken whole."""
    first = _session(tmp_path, holds_slot=True)
    await first.prompt(WORDS, input_mode="dictated")
    await first.dispose()
    resumed = _session(tmp_path, holds_slot=True)
    assert resumed._voice_note_for is None
    history = list(first._context.messages)
    assert resumed._with_voice_note(history) is history
    await resumed.dispose()


@pytest.mark.asyncio
async def test_the_transform_never_mutates_the_context(tmp_path: Path) -> None:
    session = _session(tmp_path, holds_slot=True)
    user = Message.user(WORDS)
    context = [Message.user("earlier"), Message.assistant("reply"), user]
    snapshot = list(context)
    session._voice_note_for = user.id
    shaped = session._with_voice_note(context)
    assert context == snapshot and len(context) == 3
    assert shaped is not context and shaped[:3] == snapshot
    assert shaped[3].text == VOICE_SUMMARY_NOTE and shaped[3]._ephemeral is True
    assert shaped[3].role == "user"
    # Not armed, or a newer user message: nothing added, and the SAME list comes back.
    session._voice_note_for = None
    assert session._with_voice_note(context) is context
    session._voice_note_for = user.id
    newer = [*context, Message.user("typed after")]
    assert session._with_voice_note(newer) is newer
    # A message that merely quotes the note does not move the id.
    assert session._with_voice_note([user, Message.assistant(VOICE_SUMMARY_NOTE)])[-1].role == (
        "user"
    )
    await session.dispose()


def test_the_note_asks_for_the_label_the_server_selects() -> None:
    from local_operator.voice import speakable

    assert "`Spoken:`" in VOICE_SUMMARY_NOTE
    # A reply that follows the note's instruction is exactly what the server selects.
    reply = "Long answer.\n\nSpoken: Done. Two things changed."
    assert speakable.select_spoken(reply) == "Done. Two things changed."


# --- the wire: an ephemeral user message AFTER the tool results, per encoder ----------


def _tail_history() -> list[Message]:
    note = Message.user(VOICE_SUMMARY_NOTE)
    note._ephemeral = True
    return [
        Message.user("older"),
        Message.assistant("older reply"),
        Message.user(WORDS),
        Message(role="assistant", tool_calls=[ToolCall(id="c1", name="t", arguments={})]),
        Message(
            role="tool", tool_call_id="c1", tool_name="t", content=[TextContent(text="tool says")]
        ),
        note,
    ]


def _marked(blob: Any) -> bool:
    return "cache_control" in json.dumps(blob)


def test_anthropic_orders_the_note_after_the_tool_result_and_never_caches_it() -> None:
    body = AnthropicClient()._build_body(
        ChatRequest(
            model=ModelSpec(provider="anthropic", model_id="claude-sonnet-4-5"),
            system_blocks=["instructions", "inventory", "skills", "env"],
            messages=_tail_history(),
        )
    )
    messages = body["messages"]
    assert [m["role"] for m in messages][-3:] == ["assistant", "user", "user"]
    assert messages[-2]["content"][0]["type"] == "tool_result"
    assert messages[-1]["content"] == [{"type": "text", "text": VOICE_SUMMARY_NOTE}]
    assert _marked(messages[-2]) and not _marked(messages[-1])


def test_openai_style_orders_the_note_after_the_tool_message_and_never_caches_it() -> None:
    spec = ModelSpec(
        provider="radient", model_id="anthropic/claude-sonnet-5.5", supports_prompt_cache=True
    )
    body = OpenAICompatClient("https://api.radienthq.com/v1")._build_body(
        ChatRequest(
            model=spec, system_blocks=["instructions", "inventory"], messages=_tail_history()
        )
    )
    conversation = [m for m in body["messages"] if m["role"] != "system"]
    assert [m["role"] for m in conversation][-3:] == ["assistant", "tool", "user"]
    assert conversation[-1]["content"] == VOICE_SUMMARY_NOTE
    assert _marked(conversation[-2]) and not _marked(conversation[-1])


def test_google_orders_the_note_after_the_function_response() -> None:
    body = GoogleClient()._build_body(
        ChatRequest(
            model=ModelSpec(provider="google", model_id="gemini-2.5-pro"),
            system_blocks=["instructions"],
            messages=_tail_history(),
        )
    )
    contents = body["contents"]
    assert [c["role"] for c in contents][-3:] == ["model", "user", "user"]
    assert "functionResponse" in contents[-2]["parts"][0]
    assert contents[-1]["parts"] == [{"text": VOICE_SUMMARY_NOTE}]
