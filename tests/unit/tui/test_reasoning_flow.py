"""The reasoning phase, end to end through the app.

The block's own tests prove it renders; these prove the app WIRES it — that a
session event mounts it, that the answer's first delta REMOVES it before the
answer mounts, and that the model's private reasoning never lands in the
assistant block. The harness (app, boot, submit, frame readers) is the shipped
one from ``test_steering_approval``, reused rather than re-created so these tests
cannot pass against a friendlier app than the one users run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from local_operator import settings_io
from local_operator.config import ConfigManager
from local_operator.harness.types import (
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolResult,
)
from local_operator.tui.app import OperatorApp
from local_operator.tui.events import (
    AssistantDelta,
    AssistantMessageEnd,
    AssistantMessageStart,
    ReasoningDelta,
    ReasoningEnd,
    ToolEnded,
    ToolStarted,
    TurnEnded,
    TurnStarted,
)
from local_operator.tui.settings import settings_reload
from local_operator.tui.widgets.assistant import AssistantBlock
from local_operator.tui.widgets.reasoning import (
    REASONING_LABEL,
    REASONING_MIN_ROWS,
    REASONING_VISIBLE_ROWS,
    ReasoningBlock,
)
from local_operator.tui.widgets.transcript import UserBlock
from tests.unit.tui.test_steering_approval import (
    SteerableSession,
    _boot,
    _factory,
    _submit,
    _wait_for_row,
    rows,
)

#: The reasoning text the live-channel tests stream. Deliberately free of the
#: word ``reasoning`` itself: the assertions below look for the painted header
#: row, and a fixture that spelled the label inside its own prose would make a
#: vanished block look present.
THOUGHT = "weighing the options"

#: The header row as the frame PAINTS it: glyph, space, label. The bare label is
#: not a discriminator in this suite — the status band prints the session's cwd,
#: and a checkout named after this feature carries the word in its path, which
#: made the first version of these assertions fail against a frame with no
#: reasoning block in it at all.
REASONING_HEADER = "\u00b7 " + REASONING_LABEL


def _write_reasoning_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: bool) -> None:
    """Point the TUI at an isolated config dir and write ``display.reasoning``.

    Written through ``settings_io`` rather than by patching ``settings_get``, so
    the flat dotted key the reader actually looks up is the one exercised — a
    nested write would pass a patched test and fail a user (the pattern
    ``display.narration``'s own toggle test established). The key ships OFF, so
    every test that exercises the LIVE channel has to turn it on the way a user
    does; the caller owes a ``settings_reload()`` afterwards.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    settings_reload()
    settings_io.write_setting(
        ConfigManager(tmp_path), settings_io.BY_KEY["display.reasoning"], value
    )


def _tool_card(call_id: str, name: str) -> tuple[ToolStarted, ToolEnded]:
    """One tool call started and finished, the shape a settled frame carries."""
    return (
        ToolStarted(ToolExecutionStartEvent(tool_call_id=call_id, tool_name=name, args={})),
        ToolEnded(
            ToolExecutionEndEvent(
                tool_call_id=call_id,
                tool_name=name,
                # Empty content, the shape the peer shot scripts use: the card
                # paints its header from the start/end pair alone.
                result=ToolResult(tool_call_id=call_id, tool_name=name, content=[]),
            )
        ),
    )


@pytest.mark.asyncio
async def test_reasoning_paints_above_the_answer_and_retires_when_it_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The operator's complaint, as a frame: thinking appears, then the answer.

    Four facts, and each one is a behaviour the wiring could get wrong on its
    own: a ``reasoning_delta`` MOUNTS a block, the block sits ABOVE the answer
    it belongs to, the answer's own text is untouched by the reasoning, and the
    phase is GONE once the answer starts — removed from the transcript, not
    collapsed to a leftover header row (each model call of a turn made one, and
    they accumulated for the life of the session).

    The live channel is opt-in — ``display.reasoning`` ships OFF — so this test
    turns it on the way a user does before driving the turn.
    """
    _write_reasoning_flag(tmp_path, monkeypatch, True)
    try:
        session = SteerableSession()
        app = OperatorApp(lambda: _factory(session))
        async with app.run_test(size=(100, 30)) as pilot:
            await _boot(pilot, app)
            await _submit(pilot, app, "think about it")
            app.post_message(TurnStarted())
            app.post_message(AssistantMessageStart())
            app.post_message(ReasoningDelta(THOUGHT))
            await _wait_for_row(pilot, app, THOUGHT)

            reasoning_blocks = app.query(ReasoningBlock)
            assert len(reasoning_blocks) == 1
            phase = reasoning_blocks.first()
            assert app._reasoning_block is phase, "the live phase must be the mounted block"
            # Visible while it streams: a header row plus at least one body row.
            # A bare header is the state this block is REACHED through, so
            # asserting the live frame is past it is what tells a streaming
            # phase from a retired one.
            assert phase.size.height >= REASONING_MIN_ROWS
            assert any(THOUGHT in row for row in rows(app))
            # Ordering, read from the laid-out regions rather than from append
            # order, and read NOW: the phase is gone once the answer arrives, so
            # the comparison it used to make afterwards no longer has two sides.
            # What must hold is that the thinking sits BELOW the prompt and
            # above the turn's foot — which is where the answer then lands.
            prompt = app.query_one(UserBlock)
            assert phase.region.y > prompt.region.y
            answered_at = phase.region.y

            app.post_message(AssistantDelta("The answer is 42."))
            await _wait_for_row(pilot, app, "The answer is 42.")

            # Retired AND removed: no widget, no painted row, and nothing left
            # for the NEXT call's thinking to land in...
            assert app._reasoning_block is None
            assert len(app.query(ReasoningBlock)) == 0
            assert not any(REASONING_HEADER in row for row in rows(app))
            assert not any(THOUGHT in row for row in rows(app))

            # ...and the answer took the rows the phase vacated, rather than
            # being pushed below a residue.
            answer = app.query_one(AssistantBlock)
            assert answer.region.y <= answered_at
            assert THOUGHT not in str(answer.renderable)
    finally:
        settings_reload()


@pytest.mark.asyncio
async def test_the_reasoning_channel_is_off_by_default() -> None:
    """No config write, nothing mounted: the shipped default is OFF.

    This is a first-run frame, and the assertion is about the FRAME rather than
    about a flag being read: the reasoning channel contributes nothing to it,
    and the answer still streams normally beside it.
    """
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "think about it")
        app.post_message(TurnStarted())
        app.post_message(AssistantMessageStart())
        app.post_message(ReasoningDelta(THOUGHT))
        app.post_message(AssistantDelta("The answer is 42."))
        await _wait_for_row(pilot, app, "The answer is 42.")

        assert len(app.query(ReasoningBlock)) == 0
        assert not any(REASONING_HEADER in row for row in rows(app))
        assert not any(THOUGHT in row for row in rows(app))


@pytest.mark.asyncio
async def test_a_finished_phase_leaves_no_row_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The operator's SECOND report, as a frame: the residue must not survive.

    One turn, two model calls, both of which reason: the first hands off to a
    tool call and the second writes the answer. A retired phase used to keep its
    header row, so this frame carried one ``· reasoning`` line per call and they
    accumulated for the life of the session. The settled transcript must read
    ``user -> tools -> answer`` with no trace of the thinking.

    Asserted on three surfaces rather than one, because each can be wrong while
    the others look right: the widget tree (the reference let go but the block
    left mounted), the painted frame (a block removed whose row stayed painted),
    and the text (a phase whose rows were re-authored as a bare header). RED on
    the previous behaviour, which collapsed at retirement: at 100x30 the frame
    this test drives painted two ``· reasoning`` rows, one per model call.
    """
    _write_reasoning_flag(tmp_path, monkeypatch, True)
    try:
        session = SteerableSession()
        app = OperatorApp(lambda: _factory(session))
        async with app.run_test(size=(100, 30)) as pilot:
            await _boot(pilot, app)
            await _submit(pilot, app, "port the renderer to pygame-ce")
            app.post_message(TurnStarted())

            # Call 1: reasons, then hands off to a tool call with no prose.
            app.post_message(AssistantMessageStart())
            app.post_message(ReasoningDelta("Checking what the renderer imports."))
            await _wait_for_row(pilot, app, "Checking what the renderer imports.")
            app.post_message(ReasoningEnd())
            app.post_message(AssistantMessageEnd("", stop_reason="tool_use", has_tool_calls=True))
            started, ended = _tool_card("call-1", "bash")
            app.post_message(started)
            app.post_message(ended)

            # Call 2: reasons, then answers.
            app.post_message(AssistantMessageStart())
            app.post_message(ReasoningDelta("The arrival time is what was asked for."))
            await _wait_for_row(pilot, app, "The arrival time is what was asked for.")
            app.post_message(AssistantDelta("The answer is 42."))
            await _wait_for_row(pilot, app, "The answer is 42.")
            app.post_message(TurnEnded(aborted=False, error=None))
            for _ in range(4):
                await pilot.pause()

            assert len(app.query(ReasoningBlock)) == 0
            assert app._reasoning_block is None
            painted = rows(app)
            assert not any(REASONING_HEADER in row for row in painted)
            assert not any("Checking what the renderer imports" in row for row in painted)
            assert not any("The arrival time is what was asked for" in row for row in painted)
            # The settled transcript still carries what it should.
            assert any("port the renderer to pygame-ce" in row for row in painted)
            assert any("The answer is 42." in row for row in painted)
    finally:
        settings_reload()


@pytest.mark.asyncio
async def test_a_turn_that_ends_mid_reasoning_leaves_no_row_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The terminal paths that never reach the answer's delta (review round 1, M1).

    `_finalize_turn` is the ONE exit for every way a turn can finish, and this is
    the only assertion that the phase is retired THERE: the sibling regression
    test posts its `TurnEnded` after the answer's delta has already removed the
    block, so it passes with that call deleted — measured on the mutant, where
    deleting `_finalize_turn`'s `self._retire_reasoning_block()` left the rest of
    the file green and this test red (`assert 1 == 0`, the frame still painting
    `· reasoning`). The paths it covers are the ones with no answer to arrive:
    an abort, a turn that reasoned and then died, and a worker that returns
    without a terminal end.

    `TurnEnded(aborted=True)` rather than `TurnAbandoned`: the latter is gated on
    a matching turn epoch and is posted by the app's own worker, not by a test.
    """
    _write_reasoning_flag(tmp_path, monkeypatch, True)
    try:
        session = SteerableSession()
        app = OperatorApp(lambda: _factory(session))
        async with app.run_test(size=(100, 30)) as pilot:
            await _boot(pilot, app)
            await _submit(pilot, app, "think about it")
            app.post_message(TurnStarted())
            app.post_message(AssistantMessageStart())
            app.post_message(ReasoningDelta(THOUGHT))
            await _wait_for_row(pilot, app, THOUGHT)
            assert len(app.query(ReasoningBlock)) == 1, "the phase must be live first"

            # The turn dies with the phase still on screen: no answer delta.
            app.post_message(TurnEnded(aborted=True, error=None))
            for _ in range(4):
                await pilot.pause()

            assert len(app.query(ReasoningBlock)) == 0
            assert not any(REASONING_HEADER in row for row in rows(app))
            assert not any(THOUGHT in row for row in rows(app))
            assert app._reasoning_block is None
    finally:
        settings_reload()


@pytest.mark.asyncio
async def test_a_clear_mid_reasoning_hands_the_live_phase_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """/clear drops the phase reference with the rows it points at (round 1, n2).

    `_on_transcript_cleared` drops `_streaming_block`, `_tool_cards` and the
    working line beside it, and the reasoning block was the one reference left
    pointing at a widget `clear_blocks` had just removed — the only place
    `retire()`'s claim that "the OWNER removes the widget" was not literally
    true. The next fragment still mounts a fresh phase, because the mount is on
    demand.
    """
    _write_reasoning_flag(tmp_path, monkeypatch, True)
    try:
        session = SteerableSession()
        app = OperatorApp(lambda: _factory(session))
        async with app.run_test(size=(100, 30)) as pilot:
            await _boot(pilot, app)
            await _submit(pilot, app, "think about it")
            app.post_message(TurnStarted())
            app.post_message(AssistantMessageStart())
            app.post_message(ReasoningDelta(THOUGHT))
            await _wait_for_row(pilot, app, THOUGHT)
            assert len(app.query(ReasoningBlock)) == 1, "the phase must be live first"

            await pilot.press("slash", "c", "l", "e", "a", "r", "enter")
            for _ in range(2):
                await pilot.pause()

            assert app._reasoning_block is None
            assert len(app.query(ReasoningBlock)) == 0
            assert not any(REASONING_HEADER in row for row in rows(app))

            # A fragment after the clear mounts a fresh phase rather than
            # writing into the detached one.
            app.post_message(ReasoningDelta("a fresh thought after the clear"))
            await _wait_for_row(pilot, app, "a fresh thought after the clear")
            assert app._reasoning_block is not None
            assert len(app.query(ReasoningBlock)) == 1
    finally:
        settings_reload()


@pytest.mark.asyncio
async def test_a_squeezed_terminal_keeps_the_question_on_screen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The row budget yields to the viewport instead of expelling the question.

    At 100x14 the transcript is ~5 rows tall, and the constant six-row bound was
    taller than the whole transcript it lived in: the user's own prompt was
    pushed to y=-3 and a scrollbar appeared, which is the one place this change
    regressed against the base (design round 1, D3). The budget is a property of
    the LIVE block, so this test opts the channel on first.

    WHAT IT DOES NOT CLAIM, measured by QA round 1 and confirmed by design round
    1's D2 (deferred, not fixed here): below 100x20 the budget still does not
    count the tail-pinned working line, so the question can be scrolled out of
    the viewport while the phase streams — identically on `origin/main`, which is
    why it is not this PR's regression. The assertion below is about the block
    yielding ROWS rather than expelling the question from the layout (`region.y`
    can stay non-negative while the question is above the transcript's own clip),
    and it passed on both trees.
    """
    _write_reasoning_flag(tmp_path, monkeypatch, True)
    try:
        session = SteerableSession()
        app = OperatorApp(lambda: _factory(session))
        async with app.run_test(size=(100, 14)) as pilot:
            await _boot(pilot, app)
            await _submit(pilot, app, "work out the arrival time")
            app.post_message(TurnStarted())
            app.post_message(AssistantMessageStart())
            app.post_message(ReasoningDelta(" ".join(f"word{index}" for index in range(400))))
            for _ in range(4):
                await pilot.pause()

            block = app.query_one(ReasoningBlock)
            assert block.size.height <= REASONING_MIN_ROWS + 1
            assert block.size.height < REASONING_VISIBLE_ROWS + 1
            # The question the block belongs to is what the budget protects.
            assert app.query_one(UserBlock).region.y >= 0
    finally:
        settings_reload()


@pytest.mark.asyncio
async def test_the_reasoning_channel_can_be_switched_off(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``display.reasoning = false`` mounts nothing at all.

    Kept beside :func:`test_the_reasoning_channel_is_off_by_default` because the
    two pin different things: that one pins the shipped DEFAULT (no write), this
    one pins the WRITE — the flag a user sets in ``/settings`` reaching the
    reader. Reasoning is never durable, so OFF is also the value under which the
    live transcript and the resumed one agree exactly.
    """
    _write_reasoning_flag(tmp_path, monkeypatch, False)
    try:
        session = SteerableSession()
        app = OperatorApp(lambda: _factory(session))
        async with app.run_test(size=(100, 30)) as pilot:
            await _boot(pilot, app)
            await _submit(pilot, app, "think about it")
            app.post_message(TurnStarted())
            app.post_message(AssistantMessageStart())
            app.post_message(ReasoningDelta("weighing the options"))
            app.post_message(AssistantDelta("The answer is 42."))
            await _wait_for_row(pilot, app, "The answer is 42.")

            assert len(app.query(ReasoningBlock)) == 0
            assert not any("\u00b7 " + REASONING_LABEL in row for row in rows(app))
    finally:
        settings_reload()


@pytest.mark.asyncio
async def test_an_empty_reasoning_flush_mounts_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No fragment, no block: a header row above nothing is the hole to avoid.

    The controller's equality guard means an empty flush is rare, but a relayed
    or replayed frame can carry an empty delta, and mounting a block for it would
    leave a labelled row that then sits above the answer explaining nothing. The
    channel is turned ON here, so the assertion is about the EMPTY fragment
    rather than about the off-by-default channel, which mounts nothing anyway.
    """
    _write_reasoning_flag(tmp_path, monkeypatch, True)
    try:
        session = SteerableSession()
        app = OperatorApp(lambda: _factory(session))
        async with app.run_test(size=(100, 30)) as pilot:
            await _boot(pilot, app)
            await _submit(pilot, app, "think about it")
            app.post_message(TurnStarted())
            app.post_message(AssistantMessageStart())
            app.post_message(ReasoningDelta(""))
            app.post_message(AssistantDelta("The answer is 42."))
            await _wait_for_row(pilot, app, "The answer is 42.")

            assert len(app.query(ReasoningBlock)) == 0
            assert not any("· " + REASONING_LABEL in row for row in rows(app))
    finally:
        settings_reload()
