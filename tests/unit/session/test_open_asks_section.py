"""The open-asks reminder — design §16/SA-D3 (R2, PR-3).

WHY THIS FILE EXISTS
====================
The measured failure (investigation 2026-10-22, one operator window): nothing
ever mentioned an open ask again. Withdrawing was left entirely to the model's
judgment, so the user spent 18 messages in a week asking "are the asks still
relevant?" and 16 of 40 withdrawals arrived only after one of them.

The fix is a `[session-state]` tail section, `open_asks` (`<open-asks>`), built
per provider step from the queue's own fold and stamped with the latest ADMITTED
user message. The properties worth pinning, and the acceptance list behind them:

- no open ask ⇒ NO section, and the tail is byte-identical to a host without
  this feature (acceptance 1);
- one open ask and one user message ⇒ exactly one delta, and the tool steps
  inside that turn ship nothing new (acceptance 2);
- a later user message ⇒ the section re-ships with the new "as of"; two
  messages inside the same rendered minute ship nothing, because the bytes did
  not move (acceptance 3, softened by QA round 1's MINOR-1: the cadence is per
  user-MINUTE, not per message);
- settling an ask mid-turn ⇒ the next step ships the shrunk section, and `""`
  once the last one settles (acceptance 4);
- a resumed session replays it and keeps the replayed copy across a minute
  boundary instead of re-shipping it (QA round 1, Q1), a host with no ask queue
  never ships a removal at all (QA round 1, Q2), a compaction re-anchors it
  once, and a subagent never renders it (acceptance 5);
- the section arrives as a DELTA — no new `system_prefix` epoch, so the prompt
  cache prefix holds (acceptance 6).
"""

from __future__ import annotations

import inspect
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.asks import policy
from local_operator.harness.types import (
    AbortSignal,
    AgentTool,
    ChatRequest,
    CustomMessage,
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
    StreamToolCallDelta,
    TextContent,
    ToolResult,
)
from local_operator.prompts_api import CHANNEL_NONE, build_system_blocks
from local_operator.session.goal import GoalState
from local_operator.session.session import Session, _open_asks_reminder
from local_operator.session.transcript import Transcript

MODEL = ModelSpec(provider="test", model_id="m", context_window=100_000)

SKILLS = "## Guides\n\n- demo: A demo guide."
ENV = "cwd: /tmp/project\nOS: Darwin"
DATE = "2026-10-23"

ASK_TEXT = "Should the release go out tonight, or wait for the review round?"


class ScriptedStream:
    """Answers each provider call from a script, repeating a plain stop after."""

    def __init__(self, turns: list[list[Any]] | None = None) -> None:
        self.turns = list(turns or [])
        self.requests: list[ChatRequest] = []

    def __call__(self, request: ChatRequest, signal: AbortSignal | None):
        self.requests.append(request)
        index = len(self.requests) - 1
        turn = (
            self.turns[index]
            if index < len(self.turns)
            else [StreamTextDelta(delta="ok"), StreamEndEvent(stop_reason="stop")]
        )

        async def gen():
            for event in turn:
                yield event

        return gen()


class ShiftedClock:
    """One ``time`` module, moved forward by ``shift`` seconds.

    The section's "as of" is minute-granular ON PURPOSE (a reader needs "as of
    17:42", not a millisecond), so a second message inside the same minute
    renders the same bytes and correctly ships nothing. Tests that need the
    clock to have moved move it here rather than sleeping a minute.

    BOTH modules that stamp a message have to be patched together (agent review
    round 1, MAJOR-1): ``session.py`` decides the stamp, but the number itself is
    the transcript row's ``ts`` (``transcript.py``), so a shift applied only to
    ``session.time`` leaves the durable stamp where it was and nothing re-renders.
    """

    def __init__(self, shift: float) -> None:
        self.shift = shift

    def __getattr__(self, name: str) -> Any:
        return getattr(time, name)

    def time(self) -> float:
        return time.time() + self.shift


def _shift_clock(monkeypatch: pytest.MonkeyPatch, seconds: float) -> None:
    """Move every clock a user message is stamped from (see ``ShiftedClock``)."""
    clock = ShiftedClock(seconds)
    monkeypatch.setattr("local_operator.session.session.time", clock)
    monkeypatch.setattr("local_operator.session.transcript.time", clock)


def _provider(goal_state: GoalState, tools: list[AgentTool]):
    """The factory's closure in miniature: reads the holder, per render.

    ``open_asks=goal_state.open_asks() or ""`` is exactly what ``session_factory``
    does — the provider is built before the Session exists, so the section is
    read off the shared holder the facade publishes its renderer on, and the
    tri-state collapses to text for the builder (``None`` = unknown renders
    nothing, and the DELTA is what refuses to read that as a removal).
    """

    def provider(model_label: str = "") -> list[str]:
        return build_system_blocks(
            tools,
            SKILLS,
            ENV,
            DATE,
            interactive=True,
            channel=CHANNEL_NONE,
            host_has_browser=False,
            host_has_console=False,
            open_asks=goal_state.open_asks() or "",
        )

    setattr(provider, "append_only_state", True)
    setattr(provider, "host_has_browser", False)
    setattr(provider, "host_has_console", False)
    return provider


def _make_session(
    directory: Path,
    stream: ScriptedStream,
    goal_state: GoalState,
    tools: list[AgentTool] | None = None,
) -> Session:
    return Session(
        model=MODEL,
        stream_fn=stream,
        tools=tools or [],
        transcript=Transcript(directory),
        system_blocks_provider=_provider(goal_state, tools or []),
        goal_state=goal_state,
        has_ui=True,
    )


async def _noop_hook(questions: list[Any]) -> dict[str, list[str]] | None:
    """A host hook that exists: the queue gates on its presence, not its answer."""
    return None


def _install_queue(session: Session, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(policy, "NONBLOCKING_ASK", True)
    session.set_ask_handler(_noop_hook)  # type: ignore[arg-type]
    queue = session.ask_queue()
    assert queue is not None
    return queue


def _ask(queue: Any, question: str = ASK_TEXT, question_id: str = "ship") -> str:
    outcome = queue.enqueue([{"id": question_id, "question": question, "options": []}], 3600)
    assert outcome["ok"], outcome
    return str(outcome["details"]["ask_id"])


def _state_records(session: Session) -> list[CustomMessage]:
    return [
        message
        for message in session._context.messages
        if isinstance(message, CustomMessage) and message.custom_type == "session_state"
    ]


def _open_asks_sections(session: Session) -> list[str]:
    """Every ``open_asks`` section the state records carry, oldest first."""
    out: list[str] = []
    for record in _state_records(session):
        section = (record.details.get("blocks") or {}).get("3", {})
        if isinstance(section, dict) and "open_asks" in section:
            out.append(str(section["open_asks"]))
    return out


def _prefix_entries(session: Session) -> list[Any]:
    return [entry for entry in session._transcript.entries() if entry.type == "custom"]


def _request_text(request: ChatRequest) -> str:
    """Every text block a request carries, concatenated.

    The section reaches a provider as a HISTORY message (the ``[session-state]``
    record), never as a system block, so "did the model receive it" has to be
    asked of the request's messages rather than of ``system_blocks``.
    """
    parts: list[str] = []
    for message in request.messages:
        for block in message.content or ():
            text = getattr(block, "text", None)
            if text:
                parts.append(str(text))
    return "\n".join(parts)


async def _blocks(session: Session) -> list[str]:
    """The session's current system blocks, awaited if the provider is async.

    ``Session._system_blocks`` is typed ``list[str] | Awaitable[list[str]]``
    because a host's provider may be a coroutine function; the session's own
    render path awaits it, so the tests read it the same way rather than
    asserting on the union.
    """
    blocks = session._system_blocks()
    if inspect.isawaitable(blocks):
        blocks = await blocks
    return list(blocks)


# ---------------------------------------------------------------------------
# The renderer itself — the text the model reads
# ---------------------------------------------------------------------------


def test_no_rows_render_nothing_at_all() -> None:
    """Acceptance 1 at the renderer: no asks is the ABSENT section, not an
    empty heading. A section that appeared and vanished would move the tail's
    bytes on every settle and grow the start context for nothing."""
    assert _open_asks_reminder([], 1_700_000_000_000) == ""


def test_one_line_per_ask_names_the_id_the_time_and_the_first_question() -> None:
    """The id is what ``ask_withdraw`` takes, the time places the ask against
    the user's own words, and the first question says what is waiting."""
    rows = [
        {
            "ask_id": "a-b782",
            "created_at": 1_700_000_000_000,
            "status": "open",
            "questions": [
                {"id": "cap_fix", "question": "Fix the cap first?"},
                {"id": "charter_gates", "question": "And the gates?"},
            ],
        }
    ]
    text = _open_asks_reminder(rows, None)
    line = text.splitlines()[0]
    assert line.startswith("- a-b782 — asked ")
    # The ask id AND the question ids: the first is what ``ask_withdraw`` takes,
    # the second is what its ``answers`` map is keyed by (QA round 1: a live run
    # burned two refusals guessing between them).
    assert ', 2 questions (ids: cap_fix, charter_gates): "Fix the cap first?"' in line
    # No "as of" head without an admitted user message, and the settle advice
    # still reads correctly — it is the same sentence in both shapes.
    assert "As of the user's message" not in text
    assert "settle it with `ask_withdraw`" in text
    # The rule the guidance leans on, plus its tie-breaker, where it is read.
    assert "An ask the user has not addressed stays open" in text
    assert "if you are unsure, leave it open" in text


def test_a_long_or_multi_line_question_is_collapsed_and_clipped() -> None:
    """One line per ask is the contract; a pasted paragraph must not become ten
    lines of prompt, and whitespace must not survive as layout."""
    question = "Should we\n\n" + "ship the whole thing without a review " * 6
    rows = [
        {"ask_id": "a-1", "created_at": 0, "status": "open", "questions": [{"question": question}]}
    ]
    line = _open_asks_reminder(rows, None).splitlines()[0]
    assert "\n" not in line
    assert "  " not in line
    assert line.endswith('…"')
    quoted = line.split('"')[1]
    assert len(quoted) == 80


def test_the_reminder_names_the_oldest_and_counts_the_rest() -> None:
    """Mirroring the queue's own caps: a session can hold more open rows than
    any one budget (the model's 8, bridge 8, relays 8). The section names the
    oldest few — the ones a message most likely already settled — and NAMES the
    remainder by id, because a reader told about 8 of 12 would take the other
    four for settled, and an id is enough to withdraw one later."""
    rows = [
        {"ask_id": f"a-{index}", "created_at": index, "status": "open", "questions": []}
        for index in range(12)
    ]
    lines = _open_asks_reminder(rows, None).splitlines()
    assert lines[0].startswith("- a-0 —")
    assert lines[7].startswith("- a-7 —")
    assert lines[8] == "- and 4 more open, not listed here: a-8, a-9, a-10, a-11"


def test_a_relay_says_where_to_answer_it() -> None:
    """A relay holds no questions of its own by construction and an answer has
    to be written on the ORIGIN's log, so the useful fact is where it lives."""
    rows = [
        {
            "ask_id": "a-9f2c",
            "created_at": 1_700_000_000_000,
            "status": "open",
            "questions": [],
            "relay": {"session_id": "77aa1b2c", "ask_id": "a-1"},
        }
    ]
    line = _open_asks_reminder(rows, None).splitlines()[0]
    assert line.startswith("- a-9f2c — asked ")
    assert "relayed from 77aa1b2c — answer it there; only `moot` settles it here" in line


# ---------------------------------------------------------------------------
# The session wiring — what actually reaches the provider
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_open_ask_leaves_the_tail_byte_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Acceptance 1: a session with a queue but no ask pays NOTHING.

    The comparison is against the same builder called without the argument, which
    is the byte-shape every host without this feature renders — the start-context
    budget is measured on this state (``scripts/bench_context_budget.py``).
    """
    goal_state = GoalState()
    stream = ScriptedStream()
    session = _make_session(tmp_path / "sess", stream, goal_state)
    _install_queue(session, monkeypatch)
    try:
        await session.prompt("hello")
        blocks = await _blocks(session)
        assert len(blocks) == 4
        assert "<open-asks>" not in blocks[3]
        without = build_system_blocks(
            [],
            SKILLS,
            ENV,
            DATE,
            interactive=True,
            channel=CHANNEL_NONE,
            host_has_browser=False,
            host_has_console=False,
        )
        assert blocks[3] == without[3]
        assert _state_records(session) == []
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_one_open_ask_ships_one_delta_for_the_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Acceptance 2 and 6: one typed message, one delta — and the section
    arrives as a ``[session-state]`` record rather than a new prefix epoch, so
    block 0 and the frozen prefix keep the bytes they had."""
    goal_state = GoalState()
    # Two provider steps in the second turn: a tool call, then the answer.
    stream = ScriptedStream(
        [
            [StreamTextDelta(delta="ok"), StreamEndEvent(stop_reason="stop")],
            [
                StreamTextDelta(delta="checking"),
                StreamToolCallDelta(index=0, id="c1", name="echo", argument_delta="{}"),
                StreamEndEvent(stop_reason="toolUse"),
            ],
            [StreamTextDelta(delta="done"), StreamEndEvent(stop_reason="stop")],
        ]
    )
    executed: list[str] = []

    async def execute(tool_call_id, args, signal, on_update, context):
        executed.append(name)
        return ToolResult(
            tool_call_id=tool_call_id, tool_name=name, content=[TextContent(text="ok")]
        )

    name = "echo"
    tools = [AgentTool(name=name, parameters={"type": "object", "properties": {}}, execute=execute)]
    session = _make_session(tmp_path / "sess", stream, goal_state, tools)
    queue = _install_queue(session, monkeypatch)
    try:
        await session.prompt("freeze the prefix")
        prefix_before = list(session._frozen_system_blocks or [])
        assert "<open-asks>" not in prefix_before[3]

        ask_id = _ask(queue)
        await session.prompt("look at this")
        assert executed == [name], "the turn really took two provider steps"

        sections = _open_asks_sections(session)
        assert len(sections) == 1, "one user message, one delta — not one per step"
        assert ask_id in sections[0]
        assert sections[0].startswith("<open-asks>\nAs of the user's message at ")
        record = _state_records(session)[-1]
        assert str(record.details["text"]).startswith("[session-state]\n## Open asks\n")
        # ...and the model really received it: the turn's SECOND request (the one
        # after the tool call) carries the record in its history, which is the
        # only place the section can reach a provider.
        assert ask_id in _request_text(stream.requests[-1])
        assert "## Open asks" in _request_text(stream.requests[-1])
        # Acceptance 6: the prefix did not move and no new epoch was written.
        assert session._frozen_system_blocks == prefix_before
        assert (
            len(
                [
                    e
                    for e in _prefix_entries(session)
                    if e.payload.get("custom_type") == "system_prefix"
                ]
            )
            == 1
        )
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_later_message_reships_with_the_new_as_of(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Acceptance 3: the reminder re-anchors on the user's words, so a message in
    a LATER minute ships it again with the new "as of" — and a message in the
    same minute does not, because the bytes are identical and re-sending them
    would be the duplicate the delta machinery exists to avoid."""
    goal_state = GoalState()
    stream = ScriptedStream()
    session = _make_session(tmp_path / "sess", stream, goal_state)
    queue = _install_queue(session, monkeypatch)
    try:
        await session.prompt("freeze the prefix")
        _ask(queue)
        await session.prompt("first message")
        first = _open_asks_sections(session)
        assert len(first) == 1

        # Same minute: the render is byte-identical, so nothing is journalled.
        await session.prompt("second message, same minute")
        assert _open_asks_sections(session) == first

        # A minute later the stamp moves, so the section ships again.
        _shift_clock(monkeypatch, 61)
        await session.prompt("a third message, a minute on")
        sections = _open_asks_sections(session)
        assert len(sections) == 2
        assert sections[-1] != sections[0]
        assert sections[0].splitlines()[1] != sections[-1].splitlines()[1], "a new 'as of'"
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_settling_mid_turn_shrinks_the_section_then_removes_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Acceptance 4: ``ask_withdraw`` lands between two provider steps of one
    turn, so the very next request carries the shrunk section — and the record
    says ``(empty)`` once the last ask is settled, because silence would read as
    "unchanged" rather than "gone"."""
    goal_state = GoalState()
    stream = ScriptedStream()
    session = _make_session(tmp_path / "sess", stream, goal_state)
    queue = _install_queue(session, monkeypatch)
    try:
        await session.prompt("freeze the prefix")
        first_ask = _ask(queue, "First?", "q1")
        second_ask = _ask(queue, "Second?", "q2")
        await session.prompt("here is my answer to the first")
        assert len(_open_asks_sections(session)) == 1
        assert {first_ask, second_ask} <= set(_open_asks_sections(session)[0].split())

        withdrawn = session.withdraw_ask(
            first_ask, reason="answered_in_chat", answers={"q1": ["yes"]}
        )
        assert withdrawn["ok"], withdrawn
        await session._prepare_system_blocks()
        sections = _open_asks_sections(session)
        assert len(sections) == 2
        assert first_ask not in sections[-1]
        assert second_ask in sections[-1]

        # The last one settles: the section ships as "" exactly once.
        assert session.withdraw_ask(second_ask, reason="moot")["ok"]
        await session._prepare_system_blocks()
        assert _open_asks_sections(session)[-1] == ""
        assert "## Open asks\n(empty)" in str(_state_records(session)[-1].details["text"])
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_resumed_session_keeps_the_replayed_section_across_a_minute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Acceptance 5 and QA round 1 Q1: a resume replays the record, and the FIRST
    message of the new runtime does not re-ship the same asks just because the
    clock crossed a minute — that duplicate is what QA measured. The runtime's
    NEXT message does re-anchor, because the baseline is then its own render."""
    directory = tmp_path / "sess"
    goal_state = GoalState()
    stream = ScriptedStream()
    session = _make_session(directory, stream, goal_state)
    queue = _install_queue(session, monkeypatch)
    ask_id = ""
    try:
        await session.prompt("freeze the prefix")
        ask_id = _ask(queue)
        await session.prompt("what is still open?")
        shipped = _open_asks_sections(session)
        assert len(shipped) == 1
    finally:
        await session.dispose()

    # The restart crosses a minute boundary: without the replay rule this is the
    # exact shape QA measured as two copies of one ask.
    _shift_clock(monkeypatch, 120)
    resumed_stream = ScriptedStream()
    resumed = _make_session(directory, resumed_stream, GoalState())
    resumed_queue = _install_queue(resumed, monkeypatch)
    try:
        assert ask_id in str(resumed._last_system_blocks)
        replayed = _open_asks_sections(resumed)
        assert any(ask_id in section for section in replayed), "the replayed record carries it"

        await resumed.prompt("carry on")
        assert _open_asks_sections(resumed) == replayed, "no duplicate after a restart"
        assert _request_text(resumed_stream.requests[-1]).count("## Open asks") == 1

        # The next message of THIS runtime re-anchors it: the baseline is now the
        # render this runtime made, not the replayed one.
        _shift_clock(monkeypatch, 240)
        await resumed.prompt("and again")
        sections = _open_asks_sections(resumed)
        assert len(sections) == 2
        assert sections[-1].splitlines()[1] != replayed[0].splitlines()[1], "a new 'as of'"
        assert resumed_queue.open_records()
    finally:
        await resumed.dispose()


@pytest.mark.asyncio
async def test_a_resume_without_an_ask_queue_leaves_the_section_standing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """QA round 1 Q2: a host that CANNOT answer must not say the asks are gone.

    ``lop exec --resume`` builds its Session with no ask hook (#2803/#2834), so
    ``ask_queue()`` is ``None`` there and the reminder's answer is UNKNOWN, not
    empty. Before the tri-state, that ``""`` shipped as a removal and the model
    was told its still-open ask had been settled — the failure this whole feature
    exists to fix, inverted. QA's repro is the fixture: a real ask open in the
    store, a reopen without the hook, one prompt.
    """
    directory = tmp_path / "sess"
    session = _make_session(directory, ScriptedStream(), GoalState())
    queue = _install_queue(session, monkeypatch)
    ask_id = ""
    try:
        await session.prompt("freeze the prefix")
        ask_id = _ask(queue)
        await session.prompt("what is still open?")
        assert len(_open_asks_sections(session)) == 1
    finally:
        await session.dispose()

    nohook_stream = ScriptedStream()
    nohook = _make_session(directory, nohook_stream, GoalState())
    try:
        assert nohook.ask_queue() is None
        assert nohook._render_open_asks_section() is None, "unknown is not empty"
        replayed = _state_records(nohook)
        assert len(replayed) == 1 and ask_id in str(replayed[0].details["text"])

        await nohook.prompt("carry on")
        # The reopen's own inventory delta (this runtime has no ask tools) is
        # expected; what must NOT appear is a record THIS runtime writes about
        # open_asks. The replayed record stays in history and is not a write.
        written = [record for record in _state_records(nohook) if record.id != replayed[0].id]
        assert written, "the reopen journalled something, so the delta path really ran"
        assert not any(
            "open_asks" in (record.details.get("blocks") or {}).get("3", {}) for record in written
        ), "no removal was written"
        text = _request_text(nohook_stream.requests[-1])
        assert "## Open asks" in text
        assert "(empty)" not in text, "the model is not told its asks are gone"
        # The ask really is still open; only this host cannot see the queue.
        assert [row["ask_id"] for row in queue.open_records()] == [ask_id]
    finally:
        await nohook.dispose()


@pytest.mark.asyncio
async def test_a_compaction_reanchors_the_section_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Acceptance 5: after a compaction cut away the record that carried the
    section, the next step re-anchors it — once, not on every following step."""
    from local_operator.harness.types import Message

    goal_state = GoalState()
    stream = ScriptedStream()
    session = _make_session(tmp_path / "sess", stream, goal_state)
    queue = _install_queue(session, monkeypatch)
    try:
        await session.prompt("freeze the prefix")
        ask_id = _ask(queue)
        await session.prompt("what is open?")
        assert len(_open_asks_sections(session)) == 1

        kept = Message.user("retained task")
        await session._transcript.append_message(kept)
        await session._transcript.append_compaction("summary", kept.id, 100)
        session._context.messages = session._transcript.build_llm_history()
        # The cut takes the record that carried the section: what survives is the
        # prefix (frozen before the ask) and nothing else.
        assert _open_asks_sections(session) == []

        await session._prepare_system_blocks()
        sections = _open_asks_sections(session)
        assert len(sections) == 1, "the section the cut removed is re-anchored"
        assert ask_id in sections[-1]

        await session._prepare_system_blocks()
        assert _open_asks_sections(session) == sections, "and re-anchored ONCE"
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_child_holder_renders_nothing_the_child_builder_never_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Acceptance 5, the child half: the subagent prompt builder never passes
    ``open_asks``, so a delegated child cannot inherit its parent's reminder —
    the child cannot settle any of those asks (``build_ask_tool`` is gated on a
    hook it never has), so listing them would send it to a tool it lacks."""
    goal_state = GoalState()
    stream = ScriptedStream()
    session = _make_session(tmp_path / "sess", stream, goal_state)
    queue = _install_queue(session, monkeypatch)
    try:
        await session.prompt("freeze the prefix")
        _ask(queue)
        # The parent renders it...
        assert "<open-asks>" in (await _blocks(session))[3]
        # ...and the child builder's call shape (no ``open_asks`` argument)
        # renders the same tail without it. The one place to keep in step is
        # ``harness/subagent.py``'s call, and this pins the consequence.
        child_tail = build_system_blocks(
            [],
            SKILLS,
            ENV,
            DATE,
            interactive=True,
            channel=CHANNEL_NONE,
            host_has_browser=False,
            host_has_console=False,
        )[3]
        assert "<open-asks>" not in child_tail
    finally:
        await session.dispose()
